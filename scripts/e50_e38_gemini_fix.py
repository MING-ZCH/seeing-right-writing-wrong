"""Exp-50 side check: Exp-38 labels for Gemini-4-Argon hard-set CoT texts that were judged on the excluded no-thinking
replies (e9h/gemini-4-argon_nothink_e38.jsonl) instead of the replies kept in e9h/gemini-4-argon.jsonl.

Re-judges the current text of every mismatched key with the Exp-38 prompt (cache: outputs/e50/judge_<judge>.jsonl,
hkey(x, y, body, "pre")) and recomputes the Exp-38 Gemini hard-set cell (written relation, agreement) with the corrected
labels. Usage: python e50_e38_gemini_fix.py [--run]
"""
from __future__ import annotations

import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor

from e38_closed_judge import O, cot_items, dedup, rj
from e45_uniform_two_channel import PROMPT_PRE, hkey, parse_label

E50 = O / "e50"
JUDGES = ("claude-opus-5-5", "gpt-6-astra")


def main() -> None:
    items = {c["key"]: c for c in cot_items() if c["set"] == "e9h" and c["model"] == "gemini-4-argon"}
    old = {}
    for j in JUDGES:
        last = {}
        for r in rj(O / "e38" / f"judge_{j}.jsonl"):
            if r["key"] in items and r.get("label") is not None:
                last[r["key"]] = r
        old[j] = last
    mism = sorted(k for k in items if any(old[j][k]["body_chars"] != len(items[k]["body"]) for j in JUDGES))
    print("mismatched keys (either judge):", len(mism))
    if "--run" in sys.argv:
        import api_client
        main_key = os.environ["OPENAI_COMPAT_API_KEY"]
        for j, key in zip(JUDGES, ("main", "alt")):
            os.environ["OPENAI_COMPAT_API_KEY"] = os.environ["OPENAI_COMPAT_API_KEY_ALT"] if key == "alt" else main_key
            fn = E50 / f"judge_{j}.jsonl"
            done = {r["hkey"] for r in rj(fn) if r.get("label") is not None}
            todo = [items[k] for k in mism if hkey(items[k]["x"], items[k]["y"], items[k]["body"], "pre") not in done]

            def go(c, j=j, fn=fn):
                out = api_client.chat(j, [{"role": "user", "content": PROMPT_PRE.format(x=c["x"], y=c["y"], r=c["body"])}],
                                 max_tokens=2048, effort="high")
                raw = (out.get("text") or "").strip()
                rec = {"hkey": hkey(c["x"], c["y"], c["body"], "pre"), "judge": j, "label": parse_label(raw, c["x"], c["y"]),
                       "raw": raw[-200:], "error": out.get("error", "")[:200], "finish_reason": out.get("finish_reason"),
                       "note": "e38 gemini relabel"}
                with open(fn, "a") as f:
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")

            with ThreadPoolExecutor(8) as ex:
                list(ex.map(go, todo))
    new = {}
    for j in JUDGES:
        lab = {r["hkey"]: r["label"] for r in rj(E50 / f"judge_{j}.jsonl") if r.get("label") is not None}
        new[j] = {k: (lab.get(hkey(c["x"], c["y"], c["body"], "pre")) if k in mism else old[j][k]["label"]) for k, c in items.items()}
    rows = dedup(rj(O / "e9h" / "gemini-4-argon.jsonl"))
    cot = {iid: r for (cond, iid, ph), r in rows.items() if cond == "cot"}
    res = {}
    for tag, L in (("exp38_labels", {j: {k: old[j][k]["label"] for k in items} for j in JUDGES}), ("corrected", new)):
        n = cov = true = agree = fa = fn_ = 0
        for k, c in items.items():
            a, b = L[JUDGES[0]].get(k), L[JUDGES[1]].get(k)
            n += 1
            if a is None or a != b or a not in (c["x"], c["y"]):
                continue
            cov += 1
            true += a == c["front"]
            r = cot[c["item_id"]]
            yes = (a == c["x"]) if c["phrasing"] == "front" else (a != c["x"])
            ok = r["pred"] == ("A" if yes else "B")
            agree += ok
            if a != c["front"]:
                fa += ok
                fn_ += 1
        res[tag] = {"n": n, "coverage": cov, "written_true": true, "written_acc": 100 * true / cov, "agree": 100 * agree / cov,
                    "n_false": fn_, "agree_false": (100 * fa / fn_) if fn_ else None}
    changed = {j: sum(new[j][k] != old[j][k]["label"] for k in mism) for j in JUDGES}
    res["mismatched_keys"] = len(mism)
    res["labels_changed"] = changed
    print(json.dumps(res, indent=1))
    json.dump(res, open(E50 / "e38_gemini_relabel.json", "w"), indent=1)


if __name__ == "__main__":
    main()
