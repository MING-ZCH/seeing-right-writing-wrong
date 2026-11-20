"""Occlusion twins under both phrasings: twin flips, accuracy, and pooled d' on the judgement
"the question's subject X is in front" (truth = meta front_name == x_name), which removes yes/no polarity."""
import vtax_paths as _vtp  # noqa: E402
import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import norm

import build_bundle as B

O = Path(f'{_vtp.OUTPUTS}')


def dprime(resp, truth):
    resp, truth = np.array(resp, float), np.array(truth, bool)
    h = (resp[truth].sum() + .5) / (truth.sum() + 1)
    f = (resp[~truth].sum() + .5) / ((~truth).sum() + 1)
    return norm.ppf(h) - norm.ppf(f), -(norm.ppf(h) + norm.ppf(f)) / 2


models = sys.argv[1].split(",")
print("model cond | acc front / behind / mean | twin flip front / behind | d'(subject-front) pooled | c")
for m in models:
    files = {"front": O / "e2d" / f"{m}.jsonl", "behind": O / "e2d" / f"{m}_behind.jsonl"}
    if not all(f.exists() for f in files.values()):
        continue
    for c in ("direct", "cot", "desc"):
        accs, flips, resp, truth = {}, {}, [], []
        for ph, f in files.items():
            rs = {r["item_id"]: r for r in B.rj(f) if r["condition"] == c}
            accs[ph] = np.mean([B.pe(r) == r["gold"] for r in rs.values()])
            flips[ph] = np.mean([B.pe(rs[k]) != B.pe(rs[k + "_cf"]) for k in rs if not k.endswith("_cf")])
            for r in rs.values():
                says_front = (B.pe(r) == "A") if ph == "front" else (B.pe(r) == "B")
                resp.append(says_front)
                truth.append(r["meta"]["front_name"] == r["meta"]["x_name"])
        d, cc = dprime(resp, truth)
        print(f"{m:13s} {c:6s} | {100*accs['front']:.1f} / {100*accs['behind']:.1f} / {50*(accs['front']+accs['behind']):.1f} | "
              f"{100*flips['front']:.1f} / {100*flips['behind']:.1f} | {d:.2f} | {cc:+.2f}")
