"""Exp-57 judge calls (Qwen3-VL-32B sentence-level pipeline). Key from OPENAI_COMPAT_API_KEY (never printed).

Tasks
  decomp   Exp-54 basis / step labels for the D+C- decomposition (A5): e54_judge.build / judge with the model list set to
           Qwen3-VL-32B and the task / label files in outputs/e57/ (e54tasks.jsonl, e54judge_<judge>.jsonl). Written
           relation = existing Exp-45 consensus labels (conclusion-removed body); prompts, parser, consensus = Exp-54.
  commit   Exp-30 commit audit (A2) on the Exp-57 trajectories: every D+C- commit sentence j* and relational D+C+
           comparators (Exp-30 rule: per D+C+ rationale with >= 1 relational sentence, the relational sentence nearest a
           relative position drawn, seed 30, from the commit positions). Prompts: e20 (primary) and the Exp-29
           depth/vertical prompt ('ontop', secondary: commits + 300 comparators, seed 57). Twins: commits + 200
           comparators (seed 57), e20 prompt only. -> outputs/e57/judge_<judge>_commit_<set>_<prompt>.jsonl

Usage: source configs/env.sh
       python e57_judge.py decomp-build
       python e57_judge.py decomp --judge gpt-6-astra --rpm 40
       python e57_judge.py decomp --judge claude-opus-5-5 --key alt --rpm 40
       python e57_judge.py commit --set cues --prompt e20 --judge claude-opus-5-5 --rpm 60 [--dry]
"""
from __future__ import annotations
import vtax_paths as _vtp  # noqa: E402

import argparse
import json
import os
import random
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import build_bundle as B
import e54_judge
from e20_commit_judge import PROMPT as PROMPT_E20, first_claim_idx, parse as parse_e20
from e29_judge import IMG, PROMPT_ONTOP, parse_ontop
from rigor_e17 import stype

O = Path(f"{_vtp.OUTPUTS}")
E57 = O / "e57"
MODEL = "qwen3vl_32b"


def e54_judge_paths() -> None:
    """e54_judge reads / writes E54/'tasks.jsonl' and E54/f'judge_{j}.jsonl'; keep Exp-57 files separate."""
    E57.mkdir(exist_ok=True)
    (E57 / "e54").mkdir(exist_ok=True)
    e54_judge.E54 = E57 / "e54"


def sample_commit(st: str, model: str = MODEL) -> list[dict]:
    """All D+C- commits and position-matched relational D+C+ comparators (= e30_judge.sample_commit, Exp-57 traj)."""
    rng = random.Random(30)
    T = sorted(B.rj(E57 / f"{model}_{st}_traj.jsonl"), key=lambda r: r["item_id"])
    dcm = [r for r in T if r["group"] == "D+C-" and r["commit_idx"] is not None and r["sentences"]]
    relpos = [r["commit_idx"] / max(1, len(r["sentences"]) - 1) for r in dcm]
    out = []
    for r in dcm:
        j, s = r["commit_idx"], r["sentences"]
        out.append({"model": model, "set": st, "item_id": r["item_id"], "base_id": r["base_id"], "image_id": r["image_id"],
                    "phrasing": r["phrasing"], "group": "D+C-", "idx": j, "n_sent": len(s), "commit_idx": j,
                    "first_claim_idx": first_claim_idx(s), "stype": stype(s[j]), "sentence": s[j],
                    "context": s[j - 1] if j > 0 else "(none)"})
    for r in T:
        s = r["sentences"]
        if r["group"] != "D+C+" or not any(stype(x) == "relation" for x in s):
            continue
        target = rng.choice(relpos) * (len(s) - 1)
        rel = [j for j, x in enumerate(s) if stype(x) == "relation"]
        j = min(rel, key=lambda t: abs(t - target))
        out.append({"model": model, "set": st, "item_id": r["item_id"], "base_id": r["base_id"], "image_id": r["image_id"],
                    "phrasing": r["phrasing"], "group": "D+C+rel", "idx": j, "n_sent": len(s), "commit_idx": None,
                    "first_claim_idx": first_claim_idx(s), "stype": "relation", "sentence": s[j],
                    "context": s[j - 1] if j > 0 else "(none)"})
    return out


def select(items: list[dict], st: str, prompt: str) -> list[dict]:
    """Pre-registered subsets: cue e20 = all; cue ontop = commits + 300 comparators; twins = commits + 200 comparators."""
    n_comp = None if (st == "cues" and prompt == "e20") else (300 if st == "cues" else 200)
    if n_comp is None:
        return items
    comp = [x for x in items if x["group"] == "D+C+rel"]
    keep = {x["item_id"] for x in random.Random(57).sample(comp, min(n_comp, len(comp)))}
    return [x for x in items if x["group"] == "D+C-" or x["item_id"] in keep]


def img_path(st: str, base_id: str) -> str:
    return str(IMG[st] / f"{base_id}.png")


def commit(st: str, prompt: str, jname: str, threads: int, rpm: float, dry: bool, limit: int | None) -> None:
    import api_client
    from PIL import Image
    items = select(sample_commit(st), st, prompt)
    p = E57 / f"judge_{jname}_commit_{st}_{prompt}.jsonl"
    key = lambda x: (x["model"], x["item_id"], x["idx"], x["group"])
    done = {key(r) for r in B.rj(p) if not r.get("api_error") and r.get("label", r.get("reading")) != "UNPARSED"} if p.exists() else set()
    todo = sorted((x for x in items if key(x) not in done), key=lambda x: (x["group"] != "D+C-", x["item_id"]))
    todo = todo[:limit] if limit else todo
    print(f"[e57j] {p.name}: {len(items)} items {dict(Counter(x['group'] for x in items))}; done {len(done)}, to judge {len(todo)}", flush=True)
    if dry:
        return
    lock, rl, cnt = threading.Lock(), threading.Lock(), Counter()
    gap, nxt = 60.0 / rpm, [time.time()]
    f = open(p, "a")

    def go(x):
        with rl:
            t = max(nxt[0], time.time())
            nxt[0] = t + gap
        time.sleep(max(0.0, t - time.time()))
        im = Image.open(img_path(st, x["base_id"])).convert("RGB")
        if prompt == "ontop":
            txt, parser = PROMPT_ONTOP.format(c=x["context"], s=x["sentence"]), parse_ontop
        else:
            txt, parser = PROMPT_E20.format(c=x["context"], s=x["sentence"]), lambda t: {"label": parse_e20(t)}
        r = api_client.chat(jname, [{"role": "user", "content": [api_client.img_part(im), {"type": "text", "text": txt}]}],
                       max_tokens=2048, effort="high")
        rec = x | {"judge": jname, **parser(r.get("text")), "raw": (r.get("text") or r.get("error", ""))[-300:],
                   "api_error": "error" in r}
        with lock:
            f.write(json.dumps(rec) + "\n")
            f.flush()
            cnt["err" if rec["api_error"] else "ok"] += 1
            if sum(cnt.values()) % 100 == 0:
                print(f"[e57j] {p.name}: {dict(cnt)}", flush=True)

    with ThreadPoolExecutor(threads) as ex:
        list(ex.map(go, todo))
    f.close()
    print(f"[e57j] {p.name}: finished {dict(cnt)}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("task", choices=["decomp-build", "decomp", "commit"])
    ap.add_argument("--set", default="cues", choices=["cues", "twins"])
    ap.add_argument("--prompt", default="e20", choices=["e20", "ontop"])
    ap.add_argument("--judge", choices=["claude-opus-5-5", "gpt-6-astra"])
    ap.add_argument("--threads", type=int, default=6)
    ap.add_argument("--rpm", type=float, default=40)
    ap.add_argument("--dry", action="store_true")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--key", default="main", choices=["main", "alt"])
    a = ap.parse_args()
    if a.key == "alt":
        # 2026-11-16: claude-opus-5-5 is suspended for the main key's organization; the alt key still serves it
        os.environ["OPENAI_COMPAT_API_KEY"] = os.environ["OPENAI_COMPAT_API_KEY_ALT"]
    if a.task.startswith("decomp"):
        e54_judge.M7 = [MODEL]
        e54_judge_paths()
        if a.task == "decomp-build":
            e54_judge.build()
        else:
            e54_judge.judge(a.judge, a.threads, a.rpm, "main", "", a.limit)  # key already set above
    else:
        commit(a.set, a.prompt, a.judge, a.threads, a.rpm, a.dry, a.limit)


if __name__ == "__main__":
    main()
