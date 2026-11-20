"""E17e (CPU, pod): twin-refit cosine per claim stratum (round-10 Opus M4.2), layer 20, cue set.

The Tab. S46 cosine averaged probe weights over folds and over the two claim strata; within the stratum "claim = X in
front", y = 1 means the claim is right, within "claim = Y in front" it means the claim is wrong, so the two strata's
weights can point in opposite directions along a claim-correctness code and their average is not interpretable.
Here, per stratum c and position: cosine between the original probe (label y, original states) and the twin-refit
probe (label = twin relation 1 - y, twin states), both fit on all items of the stratum; and the share of twin states
the original probe assigns to the twin's relation.
Usage: python e17_cos.py --model qwen3vl_8b   (writes $W/outputs/e17/<model>_cos.json)
"""
from __future__ import annotations

import argparse
import json

import numpy as np

from e17_probe_analyze import OUT, fit, load, meta_of

CPOS = {"pre": 0, "s": 1, "rprev": 2, "r": 3, "a": 4}
OPOS = {"pre": 1, "s": 2, "rprev": 3, "r": 4, "a": 5}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    a = ap.parse_args()
    _, meta = meta_of(a.model)
    y = np.array([m["y_true"] for m in meta])
    hc = np.array([m["has_claim"] for m in meta], bool)
    claim = np.array([int(m["named"] == m["x"]) if m["has_claim"] else -1 for m in meta])
    O_, T_ = load(a.model, "orig", 20), load(a.model, "twin", 20)
    res = {}
    for p in CPOS:
        row = {}
        ws = {}
        for c in (0, 1):
            idx = np.where(hc & (claim == c))[0]
            co, mo, so = fit(O_[idx, OPOS[p]], y[idx])
            ct, mt, st = fit(T_[idx, CPOS[p]], 1 - y[idx])
            wo, wt = co.coef_[0] / so, ct.coef_[0] / st
            follow = float(np.mean(co.predict((T_[idx, CPOS[p]] - mo) / so) == 1 - y[idx]))
            row[f"stratum_claim_x_{'front' if c else 'behind'}"] = {"n": int(len(idx)), "cos": round(float(wo @ wt / (np.linalg.norm(wo) * np.linalg.norm(wt))), 3),
                                                                    "orig_probe_follows_twin": round(follow, 3)}
            ws[c] = (wo, wt)
        avg_o, avg_t = (ws[0][0] + ws[1][0]) / 2, (ws[0][1] + ws[1][1]) / 2
        row["cos_of_stratum_averaged_weights"] = round(float(avg_o @ avg_t / (np.linalg.norm(avg_o) * np.linalg.norm(avg_t))), 3)
        row["cos_between_strata_orig"] = round(float(ws[0][0] @ ws[1][0] / (np.linalg.norm(ws[0][0]) * np.linalg.norm(ws[1][0]))), 3)
        res[p] = row
        print(a.model, p, row, flush=True)
    json.dump(res, open(f"{OUT}/{a.model}_cos.json", "w"), indent=1)


if __name__ == "__main__":
    main()
