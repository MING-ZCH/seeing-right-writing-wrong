"""E18 dose-response summary (Qwen3-VL-8B, mixed 300 cue-set claim items, layer 26).

For k in {8, 16, 32, 64, 128}: truth direction with the Direct-answer sign (and the within-claim r-probe sign where run)
vs the mean of three random directions; accuracy overall, D+C- repair, D+C+ retention, D-C+/D-C- accuracy; paired
differences truth - random mean with twin-pair bootstrap. Writes outputs/bundle/steering_dose.json.
"""
from __future__ import annotations
import vtax_paths as _vtp  # noqa: E402

import glob
import json
import re
from pathlib import Path

import numpy as np

O = Path(f"{_vtp.OUTPUTS}/e18")
B = Path("outputs/bundle")
RNG = np.random.default_rng(0)
GROUPS = ("all", "D+C-", "D+C+", "D-C-", "D-C+")


def boot(v, g, n=2000):
    if len(v) == 0:
        return {"mean": None, "ci": None, "n": 0}
    u = np.unique(g)
    idx = {x: np.where(g == x)[0] for x in u}
    bs = [v[np.concatenate([idx[x] for x in RNG.choice(u, len(u))])].mean() for _ in range(n)]
    return {"mean": round(100 * float(v.mean()), 1), "ci": [round(100 * float(np.quantile(bs, .025)), 1), round(100 * float(np.quantile(bs, .975)), 1)], "n": int(len(v))}


def main() -> None:
    runs = {}
    for f in glob.glob(str(O / "qwen3vl_8b_mixed_*_L26_*.jsonl")) + glob.glob(str(O / "qwen3vl_8b_mixed_none_k128_L20_truth.jsonl")):
        m = re.match(r"qwen3vl_8b_mixed_(truth|random|none)_k(\d+)_L(\d+)_(truth|direct_answer|direct_probe|rprobe)(?:_s(\d))?\.jsonl", Path(f).name)
        if m is None:  # *_rp26 runs use a different item sample
            continue
        kind, k, layer, sign, seed = m.group(1), int(m.group(2)), int(m.group(3)), m.group(4), m.group(5)
        runs[(kind, k, sign, seed)] = {json.loads(l)["key"]: json.loads(l) for l in open(f)}
    base = runs[("none", 128, "truth", None)]
    keys = sorted(base)
    g = np.array([k.replace("_behind", "").replace("_cf", "") for k in keys])
    grp = np.array([base[k]["item_group"] for k in keys])
    acc = lambda R: np.array([float(R[k]["answer_correct"]) for k in keys])
    sel = lambda gg: np.ones(len(keys), bool) if gg == "all" else grp == gg
    out = {"none": {gg: boot(acc(base)[sel(gg)], g[sel(gg)]) for gg in GROUPS}, "doses": []}
    for k in (8, 16, 32, 64, 128):
        rnd = [acc(v) for (kd, kk, s, sd), v in runs.items() if kd == "random" and kk == k]
        if not rnd:
            continue
        rm = np.mean(rnd, 0)
        row = {"k": k, "n_random": len(rnd), "random_mean": {gg: boot(rm[sel(gg)], g[sel(gg)]) for gg in GROUPS}}
        for sign in ("direct_answer", "direct_probe", "rprobe"):
            R = runs.get(("truth", k, sign, None))
            if R is None:
                continue
            a = acc(R)
            row[sign] = {gg: boot(a[sel(gg)], g[sel(gg)]) for gg in GROUPS}
            row[f"{sign}_minus_random"] = {gg: boot((a - rm)[sel(gg)], g[sel(gg)]) for gg in GROUPS}
            row[f"{sign}_minus_none"] = {gg: boot((a - acc(base))[sel(gg)], g[sel(gg)]) for gg in GROUPS}
        out["doses"].append(row)
        for sign in ("direct_answer", "direct_probe", "rprobe"):
            if sign in row:
                d = row[f"{sign}_minus_random"]
                print(f"k={k:3d} {sign:13s} all {row[sign]['all']['mean']} (rnd {row['random_mean']['all']['mean']}) | D+C- {row[sign]['D+C-']['mean']} (rnd {row['random_mean']['D+C-']['mean']}) "
                      f"| D+C+ {row[sign]['D+C+']['mean']} (rnd {row['random_mean']['D+C+']['mean']}) | D-C+ {row[sign]['D-C+']['mean']} | diff all {d['all']['mean']} {d['all']['ci']} "
                      f"D+C- {d['D+C-']['mean']} {d['D+C-']['ci']} D+C+ {d['D+C+']['mean']} {d['D+C+']['ci']}")
    print("none", {gg: out["none"][gg]["mean"] for gg in GROUPS})
    json.dump(out, open(B / "steering_dose.json", "w"), indent=1)


if __name__ == "__main__":
    main()
