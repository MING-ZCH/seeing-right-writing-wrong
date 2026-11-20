"""E18 summary: claim and answer correctness after steering (truth / random / none), with twin-pair bootstrap CIs.

Writes outputs/bundle/steering.json (consumed by make_supp_tables.e17j_tables).
"""
from __future__ import annotations
import vtax_paths as _vtp  # noqa: E402

import glob
import json
import re
from pathlib import Path

import numpy as np

O = Path(f"{_vtp.OUTPUTS}")
B = Path("outputs/bundle")
RNG = np.random.default_rng(0)


def boot(v: np.ndarray, g: np.ndarray, n: int = 2000) -> dict:
    u = np.unique(g)
    idx = {x: np.where(g == x)[0] for x in u}
    bs = [v[np.concatenate([idx[x] for x in RNG.choice(u, len(u))])].mean() for _ in range(n)]
    return {"mean": round(100 * float(v.mean()), 1), "ci": [round(100 * float(np.quantile(bs, .025)), 1), round(100 * float(np.quantile(bs, .975)), 1)]}


def main() -> None:
    res = {}
    for f in sorted(glob.glob(str(O / "e18" / "*.jsonl"))):
        if "limit" in f:
            continue
        m = re.match(r"(.+?)_(DCm_wrong|DCp_right)_(truth|random|none)_k([\d.]+)\.jsonl", Path(f).name)
        if m is None:  # later runs in the same directory (qwen3vl_8b_mixed_*: steering_mixed / steering_dose)
            continue
        model, group, kind, k = m.group(1), m.group(2), m.group(3), float(m.group(4))
        R = [json.loads(l) for l in open(f)]
        g = np.array([r["key"].replace("_behind", "").replace("_cf", "") for r in R])
        cc = np.array([1.0 if r["claim_correct"] else 0.0 for r in R])  # no parsable claim counts as not correct
        ac = np.array([1.0 if r["answer_correct"] else 0.0 for r in R])
        parsed = np.array([r["new_named"] is not None for r in R])
        res.setdefault(model, []).append({"group": group, "kind": kind, "k": k, "n": len(R), "claim_correct": boot(cc, g), "answer_correct": boot(ac, g),
                                          "claim_parsed": round(100 * float(parsed.mean()), 1)})
    for m in res:
        res[m].sort(key=lambda r: (r["group"], r["kind"] != "none", r["kind"], r["k"]))
        for r in res[m]:
            print(m, r["group"], r["kind"], r["k"], r["n"], "claim", r["claim_correct"], "answer", r["answer_correct"], "parsed", r["claim_parsed"])
    json.dump(res, open(B / "steering.json", "w"), indent=1)


def paired() -> dict:
    """Truth minus random (same items, same k): paired twin-pair bootstrap of the difference in claim / answer correctness."""
    out = {}
    for model, ks in (("qwen3vl_8b", (32, 64, 128)), ("molmo7b", (16, 32))):
        for k in ks:
            t = {json.loads(l)["key"]: json.loads(l) for l in open(O / "e18" / f"{model}_DCm_wrong_truth_k{k}.jsonl")}
            r = {json.loads(l)["key"]: json.loads(l) for l in open(O / "e18" / f"{model}_DCm_wrong_random_k{k}.jsonl")}
            keys = sorted(set(t) & set(r))
            g = np.array([x.replace("_behind", "").replace("_cf", "") for x in keys])
            res = {}
            for f in ("claim_correct", "answer_correct"):
                d = np.array([float(bool(t[x][f])) - float(bool(r[x][f])) for x in keys])
                b = boot(d, g)
                res[f] = {"diff": b["mean"], "ci": b["ci"]}
            out[f"{model}|k{k}"] = res | {"n": len(keys)}
            print(model, k, res)
    return out


if __name__ == "__main__":
    main()
    s = json.load(open(B / "steering.json"))
    json.dump(s | {"_paired_truth_minus_random": paired()}, open(B / "steering.json", "w"), indent=1)
