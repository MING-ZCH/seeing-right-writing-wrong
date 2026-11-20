"""Exp-32 task 4: re-run the Claude Opus 5.5 replies that failed to parse in the Exp-30 cue-set commit audit.

The Exp-30 e20 parser (e20_commit_judge.parse) reads only the first 6 words; 43 Opus replies put a short explanation
first and the label last, so they were counted as non-consensus. This script
  (a) re-parses the stored reply text (truncated to 200 characters at save time) with a last-label rule,
  (b) re-calls Claude Opus 5.5 with the identical prompt / image / settings (e20 prompt, max_tokens 2048, effort high)
      and parses the new reply with the original rule, falling back to the last label word,
  (c) recomputes the Exp-30 consensus rates (two-judge agreement on CORRECT / INCORRECT; rigor_e30.audit rule) with
      the re-run labels substituted for the 43 UNPARSED Opus labels.
-> outputs/e30/judge_claude-opus-5-5_commit_e20_rerun32.jsonl (full reply text), bundle/e30_reparse.json

Usage: python e32_reparse.py [--no-api]   (key from OPENAI_COMPAT_API_KEY; never printed)
"""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from e20_commit_judge import PROMPT as PROMPT_E20, parse as parse_e20
from e30_judge import img
from rigor_e28 import pct
from rigor_e30 import E30, J2, latest, rate_e20, q95

B = Path("outputs/bundle")
LAB = ("CORRECT", "INCORRECT", "NOCLAIM", "UNCLEAR")
KEY = lambda r: (r["model"], r["item_id"], r["idx"], r["group"])
OUT = E30 / "judge_claude-opus-5-5_commit_e20_rerun32.jsonl"


def parse_last(txt: str) -> str:
    w = re.sub(r"[^A-Za-z ]", " ", txt or "").strip().upper().split()
    for tok in reversed(w):
        if tok in LAB:
            return tok
    return "UNPARSED"


def rerun(todo: list[dict]) -> dict:
    import api_client
    from PIL import Image
    done = {KEY(r): r for r in map(json.loads, open(OUT))} if OUT.exists() else {}
    todo = [x for x in todo if KEY(x) not in done or done[KEY(x)].get("api_error")]

    def go(x):
        im = Image.open(img(x["base_id"])).convert("RGB")
        r = api_client.chat("claude-opus-5-5", [{"role": "user", "content": [api_client.img_part(im), {"type": "text",
                       "text": PROMPT_E20.format(c=x["context"], s=x["sentence"])}]}], max_tokens=2048, effort="high")
        t = r.get("text") or ""
        return {k: x[k] for k in ("model", "item_id", "idx", "group", "phrasing", "sentence")} | {
            "text": t, "label_orig_rule": parse_e20(t), "label_last_rule": parse_last(t), "api_error": "error" in r,
            "error": r.get("error", "")[:200] if "error" in r else ""}

    with ThreadPoolExecutor(6) as ex:
        new = list(ex.map(go, todo))
    with open(OUT, "a") as f:
        for r in new:
            f.write(json.dumps(r) + "\n")
            done[KEY(r)] = r
    return done


def rates(E: dict, keys: list, meta: dict) -> dict:
    lab = lambda j, k: E[j][k]["label"]
    c20 = {k: (lab(J2[0], k) if lab(J2[0], k) == lab(J2[1], k) and lab(J2[0], k) in ("CORRECT", "INCORRECT") else "NONCONS") for k in keys}
    rows = {}
    for m in ("qwen3vl_8b", "molmo7b", "pooled"):
        for g in ("D+C-", "D+C+rel"):
            for sn in ("all", "front", "behind"):
                ks = [k for k in keys if k[3] == g and (m == "pooled" or k[0] == m) and (sn == "all" or meta[k]["phrasing"] == sn)]
                rows[f"{m}|{g}|{sn}"] = rate_e20([c20[k] for k in ks]) | {
                    "opus_e20": pct(sum(lab(J2[0], k) == "INCORRECT" for k in ks), sum(lab(J2[0], k) in ("CORRECT", "INCORRECT") for k in ks))}
        rows_ = [(meta[k]["image_id"], k[3] == "D+C-", c20[k] == "INCORRECT") for k in keys if c20[k] != "NONCONS" and (m == "pooled" or k[0] == m)]
        rng = np.random.default_rng(302)
        cl = np.array([r[0] for r in rows_]); gp = np.array([r[1] for r in rows_]); y = np.array([r[2] for r in rows_], float)
        uc, inv = np.unique(cl, return_inverse=True)
        agg = np.zeros((len(uc), 4))
        np.add.at(agg, inv, np.stack([y * gp, gp, y * ~gp, ~gp], 1))
        st = lambda A: A[:, 0].sum() / A[:, 1].sum() - A[:, 2].sum() / A[:, 3].sum()
        bs = [st(agg[rng.integers(0, len(uc), len(uc))]) for _ in range(2000)]
        rows[f"{m}|diff"] = {"mean": round(100 * st(agg), 1), "ci": [round(100 * x, 1) for x in q95(bs, 4)]}
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-api", action="store_true")
    a = ap.parse_args()
    E = {j: latest(E30 / f"judge_{j}_commit_e20.jsonl", KEY) for j in J2}
    Q = {j: latest(E30 / f"judge_{j}_commit_ontop.jsonl", KEY) for j in J2}
    keys = sorted(set(E[J2[0]]) & set(E[J2[1]]) & set(Q[J2[0]]) & set(Q[J2[1]]))
    meta = {k: E[J2[1]][k] for k in keys}
    unp = [k for k in keys if E[J2[0]][k]["label"] == "UNPARSED"]
    print(f"[e32r] sentences {len(keys)}, Opus UNPARSED {len(unp)} {dict(Counter((k[0], k[3]) for k in unp))}", flush=True)
    res = {"n_sentences": len(keys), "n_unparsed": len(unp), "unparsed_by_cell": {f"{a_}|{b_}": v for (a_, b_), v in Counter((k[0], k[3]) for k in unp).items()},
           "cause": "e20_commit_judge.parse reads only the first 6 words; these replies give a short explanation first and the label last"}
    # (a) stored-text re-parse (stored raw is cut at 200 characters)
    stored = {k: parse_last(E[J2[0]][k]["raw"]) for k in unp}
    complete = {k: len(E[J2[0]][k]["raw"]) < 200 for k in unp}
    res["stored_reparse"] = {"labels": dict(Counter(stored.values())), "n_complete_reply": sum(complete.values())}
    # (b) API re-run
    if not a.no_api:
        todo = [E[J2[0]][k] for k in unp]
        done = rerun([{**t, "base_id": t["base_id"]} for t in todo])
    else:
        done = {KEY(r): r for r in map(json.loads, open(OUT))} if OUT.exists() else {}
    new = {}
    for k in unp:
        r = done.get(k)
        if r and not r.get("api_error"):
            new[k] = r["label_orig_rule"] if r["label_orig_rule"] != "UNPARSED" else r["label_last_rule"]
    res["rerun"] = {"n_called_ok": len(new), "api_errors": sum(1 for k in unp if done.get(k, {}).get("api_error")),
                    "labels": dict(Counter(new.values())),
                    "parsed_by_orig_rule": sum(1 for k in unp if done.get(k) and done[k]["label_orig_rule"] != "UNPARSED"),
                    "agree_with_stored_reparse": sum(1 for k in new if stored[k] == new[k]),
                    "n_stored_parsable": sum(1 for k in new if stored[k] != "UNPARSED"),
                    "agree_with_gpt": sum(1 for k in new if new[k] == E[J2[1]][k]["label"])}
    print(f"[e32r] stored re-parse {res['stored_reparse']} | rerun {res['rerun']}", flush=True)
    before = rates(E, keys, meta)
    E2 = {J2[0]: dict(E[J2[0]]), J2[1]: E[J2[1]]}
    for k, l in new.items():
        E2[J2[0]][k] = E[J2[0]][k] | {"label": l}
    after = rates(E2, keys, meta)
    E3 = {J2[0]: dict(E[J2[0]]), J2[1]: E[J2[1]]}
    for k, l in stored.items():
        if l != "UNPARSED":
            E3[J2[0]][k] = E[J2[0]][k] | {"label": l}
    after_stored = rates(E3, keys, meta)
    res["before"], res["after_rerun"], res["after_stored_reparse"] = before, after, after_stored
    f = lambda x: f"{x['k']}/{x['n']} = {x['pct']}% {x['wilson']} wc {x['worst_case']}"
    lines = []
    for row in ("qwen3vl_8b|D+C-|all", "molmo7b|D+C-|all", "qwen3vl_8b|D+C+rel|all", "molmo7b|D+C+rel|all", "pooled|D+C-|all",
                "pooled|D+C+rel|all", "molmo7b|D+C-|front", "molmo7b|D+C-|behind"):
        lines.append(f"REPARSE {row:24s} before {f(before[row])} | after re-run {f(after[row])} | after stored re-parse {f(after_stored[row])}")
    for m in ("qwen3vl_8b", "molmo7b", "pooled"):
        lines.append(f"REPARSE {m:11s} commit - comparator before {before[m + '|diff']} after {after[m + '|diff']}")
    changed = []
    for row in before:
        if "diff" in row:
            if before[row]["mean"] != after[row]["mean"]:
                changed.append(row)
        elif before[row]["pct"] != after[row]["pct"]:
            changed.append(row)
    res["rows_changed"] = changed
    res["lines"] = lines
    json.dump(res, open(B / "e30_reparse.json", "w"), indent=1)
    print("\n".join(lines))


if __name__ == "__main__":
    main()
