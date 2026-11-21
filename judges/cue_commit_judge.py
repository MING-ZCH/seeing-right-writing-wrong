"""E28 judge calls: commit-sentence audit on the cue-factorial set / twins and pre-j* relational sentences.

Same judges, prompt, label parse and call settings as judges/commit_sentence_judge.py (Claude Opus 5.5 + GPT-6-Astra, effort high,
image + previous sentence as context + target sentence). Samples (fixed seeds, drawn before any judging):
  commit   <set> in {cues, twins}: up to 40 D+C- items with a commit sentence j* per model (e28 trajectories; all if fewer),
           target = s_{j*}; relational D+C+ comparator as e20.sample_relcomp: up to 40 D+C+ rationales with >= 1
           relational sentence (srww.claim_parse.stype), target = relational sentence nearest to a relative position drawn from
           the same model's sampled D+C- commits.
  prej     every relational sentence (stype == relation) strictly before j* in the sampled D+C- rationales: Synthetic
           Depth = the e20 D+C- audit sample (e20.sample(), 40 / model, e2b trajectories, j* judged in outputs/e20);
           cues / twins = the D+C- items of the commit sample above.
-> outputs/e28/judge_<judge>_<task>_<set>.jsonl

Usage: python -m judges.cue_commit_judge --task commit --set cues --judge claude-opus-5-5 [--models qwen3vl_8b,molmo7b]
       python -m judges.cue_commit_judge --task prej --set syn --judge gpt-6-astra
       python -m judges.cue_commit_judge --task commit --set cues --dry   (print sample sizes only)
"""
from __future__ import annotations
from srww import paths as _vtp  # noqa: E402

import argparse
import json
import random
from pathlib import Path

from srww import records as B
from judges.commit_sentence_judge import judge_items, first_claim_idx
from srww.claim_parse import stype

O = Path(f"{_vtp.OUTPUTS}")
E = O / "e28"
IMG = {"cues": E / "cue_items" / "images", "twins": O.parent / "exps" / "items" / "occlusion" / "images"}
COMP = ["qwen3vl_8b", "molmo7b"]
SEED = {"cues": 28, "twins": 29}


def img_path(st: str, base_id: str) -> str:
    return str(IMG[st] / f"{base_id}.png")


def item(r: dict, st: str, group: str, j: int, **kw) -> dict:
    s = r["sentences"]
    return {"model": r["model"], "set": st, "item_id": r["item_id"], "image_id": r["image_id"], "group": group, "idx": j,
            "n_sent": len(s), "commit_idx": r.get("commit_idx"), "is_first_claim": first_claim_idx(s) == j,
            "first_claim_idx": first_claim_idx(s), "sentence": s[j], "context": s[j - 1] if j > 0 else "(none)",
            "image": img_path(st, r["base_id"])} | kw


def commit_sample(st: str, models: list[str]) -> list[dict]:
    """D+C- commit sentences and position-matched relational D+C+ sentences (fixed seed per set)."""
    rng = random.Random(SEED[st])
    out = []
    for m in models:
        T = B.rj(E / f"{m}_{st}_traj.jsonl")
        dpair = {}
        for r in T:  # Direct correct under both phrasings of the same image
            dpair.setdefault(r["base_id"], []).append(r["src_direct"] == r["gold"])
        dcm = [r for r in T if r["group"] == "D+C-" and r["commit_idx"] is not None and r["sentences"]]
        dcm = rng.sample(dcm, min(40, len(dcm)))
        relpos = []
        for r in dcm:
            j = r["commit_idx"]
            relpos.append(j / max(1, len(r["sentences"]) - 1))
            out.append(item(r, st, "D+C-", j, direct_both_phrasings=all(dpair[r["base_id"]]), stype=stype(r["sentences"][j])))
        dcp = [r for r in T if r["group"] == "D+C+" and any(stype(x) == "relation" for x in r["sentences"])]
        for r in rng.sample(dcp, min(40, len(dcp))):
            target = rng.choice(relpos) * (len(r["sentences"]) - 1) if relpos else 0
            rel = [j for j, x in enumerate(r["sentences"]) if stype(x) == "relation"]
            j = min(rel, key=lambda t: abs(t - target))
            out.append(item(r, st, "D+C+rel", j, direct_both_phrasings=all(dpair[r["base_id"]]), stype="relation"))
    return out


def prej_sample(st: str, models: list[str]) -> list[dict]:
    """All relational sentences before j* in the sampled D+C- rationales."""
    out = []
    if st == "syn":
        from judges import commit_sentence_judge as E20
        base = [b for b in E20.sample() if b["group"] == "D+C-" and b["model"] in models]
        recs = {}
        for m in models:
            for r in B.rj(O / "e2b" / f"{m}.jsonl"):
                if r["task"] == "syn_depth":
                    recs[(m, r["item_id"])] = r
        for b in base:
            r = recs[(b["model"], b["item_id"])]
            s, js = r["sentences"], b["idx"]
            for j in range(js):
                if stype(s[j]) == "relation":
                    out.append({"model": b["model"], "set": "syn", "item_id": b["item_id"], "image_id": b["item_id"], "group": "pre_jstar",
                                "idx": j, "n_sent": len(s), "commit_idx": js, "is_first_claim": first_claim_idx(s) == j,
                                "first_claim_idx": first_claim_idx(s), "sentence": s[j], "context": s[j - 1] if j > 0 else "(none)",
                                "image": b["image"]})
        return out
    T = {(r["model"], r["item_id"]): r for m in models for r in B.rj(E / f"{m}_{st}_traj.jsonl")}
    for c in commit_sample(st, models):
        if c["group"] != "D+C-":
            continue
        r = T[(c["model"], c["item_id"])]
        for j in range(c["idx"]):
            if stype(r["sentences"][j]) == "relation":
                out.append(item(r, st, "pre_jstar", j, stype="relation"))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True, choices=["commit", "prej"])
    ap.add_argument("--set", required=True, choices=["cues", "twins", "syn"])
    ap.add_argument("--judge")
    ap.add_argument("--models", default=",".join(COMP))
    ap.add_argument("--dry", action="store_true")
    a = ap.parse_args()
    models = a.models.split(",")
    items = commit_sample(a.set, models) if a.task == "commit" else prej_sample(a.set, models)
    from collections import Counter
    print(f"[e28j] {a.task} {a.set}: {len(items)} items {dict(Counter((x['model'], x['group']) for x in items))}", flush=True)
    if a.dry:
        return
    res = judge_items(a.judge, items)
    tag = "" if models == COMP else "_" + "-".join(models)
    p = E / f"judge_{a.judge}_{a.task}_{a.set}{tag}.jsonl"
    with open(p, "w") as f:
        for r in res:
            f.write(json.dumps(r) + "\n")
    print(f"[e28j] {p.name}: {len(res)}; unparsed {sum(r['label'] == 'UNPARSED' for r in res)}; "
          f"api errors {sum(r['api_error'] for r in res)}", flush=True)


if __name__ == "__main__":
    main()
