"""Evaluate E6 held-out runs against the pre-registered predictions (frozen tau from the main pool).

Per model and task: Direct, CoT, 4-vote (plurality, ties split), desc, gate(CoT) and gate(vote) at the
frozen tau, random routing at the same rate (expected value), observed vs predicted 4-vote accuracy.
"""
from __future__ import annotations

import argparse
import glob
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from common import cluster_bootstrap_ci


def am(lp: dict) -> str:
    f = {k: v for k, v in lp.items() if v is not None and math.isfinite(v)}
    return max(f, key=f.get)


def probs(lp: dict) -> np.ndarray:
    v = np.array([lp[k] if lp[k] is not None else -1e4 for k in sorted(lp)], dtype=float)
    v = np.exp(v - v.max())
    return v / v.sum()


def pe(r: dict) -> str:
    return am(r.get("lp_sum") or r["top20"])


def ci(v, g) -> str:
    m, lo, hi = cluster_bootstrap_ci(np.asarray(v, float), g, 2000)
    return f"{100 * m:.1f} [{100 * lo:.1f}, {100 * hi:.1f}]"


def main() -> None:
    """CLI entry point."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--outputs", required=True)
    ap.add_argument("--models", default="qwen3b,qwen7b,internvl3_8b,phi35v,molmo7b,qwen3vl_8b,internvl35_8b")
    ap.add_argument("--out", default="outputs/e6_eval.json")
    args = ap.parse_args()
    O = Path(args.outputs)
    res = {}
    for m in args.models.split(","):
        dp, rp = O / "e6x" / f"{m}_direct.jsonl", O / "e6x" / f"{m}_rest.jsonl"
        preds = sorted(glob.glob(str(O / "e6x" / f"predictions_{m}_*.json")))
        if not (dp.exists() and rp.exists()):
            continue
        pred = json.load(open(preds[0])) if preds else {"tau": {}, "predictions": {}}
        D = {r["item_id"]: r for r in map(json.loads, open(dp))}
        R = defaultdict(dict)
        for r in map(json.loads, open(rp)):
            R[r["condition"]][r["item_id"]] = r
        keys = [k for k in D if k in R["cot"]]
        out = {}
        for task in sorted({D[k]["task"] for k in keys}) + ["pooled"]:
            ks = [k for k in keys if task == "pooled" or D[k]["task"] == task]
            g = [D[k]["image_id"] for k in ks]
            gold = [D[k]["gold"] for k in ks]
            d = np.array([pe(D[k]) == y for k, y in zip(ks, gold)], float)
            c = np.array([pe(R["cot"][k]) == y for k, y in zip(ks, gold)], float)
            v = []
            for k, y in zip(ks, gold):
                cnt = Counter(pe(o) for o in R["vote4"][k]["vote_outs"])
                top = max(cnt.values())
                win = [a for a, b in cnt.items() if b == top]
                v.append(float(y in win) / len(win))
            v = np.array(v)
            ds = np.array([pe(R["desc"][k]) == y for k, y in zip(ks, gold)], float) if R["desc"] else None
            marg = np.array([np.sort(probs(D[k].get("lp_sum") or D[k]["top20"]))[::-1][:2] @ [1, -1] for k in ks])
            row = {"n": len(ks), "direct": ci(d, g), "cot": ci(c, g), "vote4": ci(v, g),
                   "cot_minus_direct": ci(c - d, g), "vote4_minus_direct": ci(v - d, g)}
            if ds is not None:
                row["desc"] = ci(ds, g)
                row["desc_minus_direct"] = ci(ds - d, g)
            for alt, arr in (("cot", c), ("vote", v)):
                tau = pred["tau"].get(alt)
                if tau is None:
                    continue
                route = marg < tau
                gate = np.where(route, arr, d)
                rnd = d + route.mean() * (arr - d)  # expected accuracy of random routing at the same rate
                row[f"gate_{alt}"] = ci(gate, g)
                row[f"gate_{alt}_minus_direct"] = ci(gate - d, g)
                row[f"gate_{alt}_minus_random"] = ci(gate - rnd, g)
                row[f"routed_{alt}"] = round(float(route.mean()), 3)
            pv = pred["predictions"].get("__pooled__" if task == "pooled" else task, {})
            if pv:
                row["vote4_predicted"] = round(100 * pv["pred_vote4_item"], 1)
                row["vote4_pred_const_pbar"] = round(100 * pv["pred_vote4_const_pbar"], 1)
                row["vote4_observed"] = round(100 * float(v.mean()), 1)
            out[task] = row
        res[m] = out
        print(f"== {m}  (predictions file: {Path(preds[0]).name if preds else 'none'})")
        for t, row in out.items():
            print(f"  {t}: " + "; ".join(f"{k}={v}" for k, v in row.items()))
    json.dump(res, open(args.out, "w"), indent=1)


if __name__ == "__main__":
    main()
