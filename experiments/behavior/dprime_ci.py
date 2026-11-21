"""Bootstrap CI (resampling occlusion twin pairs, both phrasings kept together) for pooled d' and c,
and for the CoT - Direct difference, on the judgement "the question's subject is in front".
Writes outputs/bundle/dprime_ci.json.

Usage: python -m experiments.behavior.dprime_ci qwen3b,qwen7b,internvl3_8b
"""
from __future__ import annotations
from srww import paths as _vtp  # noqa: E402

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import norm

from srww import records as B

O = Path(f"{_vtp.OUTPUTS}")


def sdt(resp: np.ndarray, truth: np.ndarray) -> tuple[float, float]:
    h = (resp[truth].sum() + .5) / (truth.sum() + 1)
    f = (resp[~truth].sum() + .5) / ((~truth).sum() + 1)
    return norm.ppf(h) - norm.ppf(f), -(norm.ppf(h) + norm.ppf(f)) / 2


def per_pair(m: str, cond: str) -> dict[str, list[tuple[int, int]]]:
    out = defaultdict(list)
    for ph, fn in (("front", f"{m}.jsonl"), ("behind", f"{m}_behind.jsonl")):
        for r in B.rj(O / "e2d" / fn):
            if r["condition"] != cond:
                continue
            p = B.pe(r)
            says = (p == "A") if ph == "front" else (p == "B")
            out[r["image_id"]].append((int(says), int(r["meta"]["front_name"] == r["meta"]["x_name"])))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Twin-pair bootstrap CIs for pooled d' / c on the occlusion twins.")
    ap.add_argument("models", help="comma-separated model keys (outputs/e2d/<model>{,_behind}.jsonl)")
    models = ap.parse_args().models.split(",")
    rng = np.random.default_rng(0)
    res = {}
    for m in models:
        d, c, s = per_pair(m, "direct"), per_pair(m, "cot"), per_pair(m, "desc")
        pairs = sorted(set(d) & set(c))
        if not pairs:
            continue
        has_desc = bool(s) and all(len(s.get(p, [])) == len(d[p]) for p in pairs)
        conds = (d, c, s) if has_desc else (d, c)
        est, boots = {}, []
        for _ in range(2000):
            idx = rng.integers(0, len(pairs), len(pairs))
            row = []
            for P in conds:
                arr = np.array([x for i in idx for x in P[pairs[i]]])
                row.append(sdt(arr[:, 0].astype(bool), arr[:, 1].astype(bool)))
            b_row = [row[0][0], row[1][0], row[1][0] - row[0][0], row[0][1], row[1][1]]
            if has_desc:
                b_row += [row[2][0], row[2][0] - row[0][0]]
            boots.append(b_row)
        for name, P in zip(("direct", "cot", "desc"), conds):
            arr = np.array([x for p in pairs for x in P[p]])
            est[name] = sdt(arr[:, 0].astype(bool), arr[:, 1].astype(bool))
        b = np.array(boots)
        q = lambda j: [round(float(np.quantile(b[:, j], .025)), 2), round(float(np.quantile(b[:, j], .975)), 2)]
        res[m] = {"n_pairs": len(pairs), "dprime_direct": round(est["direct"][0], 2), "dprime_direct_ci": q(0),
                  "dprime_cot": round(est["cot"][0], 2), "dprime_cot_ci": q(1),
                  "dprime_cot_minus_direct": round(est["cot"][0] - est["direct"][0], 2), "diff_ci": q(2),
                  "c_direct": round(est["direct"][1], 2), "c_cot": round(est["cot"][1], 2)}
        if has_desc:
            res[m] |= {"dprime_desc": round(est["desc"][0], 2), "dprime_desc_ci": q(5),
                       "dprime_desc_minus_direct": round(est["desc"][0] - est["direct"][0], 2), "desc_diff_ci": q(6),
                       "c_desc": round(est["desc"][1], 2)}
        print(m, res[m])
    json.dump(res, open("outputs/bundle/dprime_ci.json", "w"), indent=1)


if __name__ == "__main__":
    main()
