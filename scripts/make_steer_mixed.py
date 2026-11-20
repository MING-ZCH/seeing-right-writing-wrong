"""E18 non-oracle steering summary (Qwen3-VL-8B, mixed sample of 300 cue-set claim items, k = 128).

Per configuration: CoT answer accuracy overall and by Direct/CoT group, first-claim correctness; paired differences vs
no steering and vs the mean of three random directions at the same layer (twin-pair bootstrap).
Writes outputs/bundle/steering_mixed.json.
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


def boot(v, g, n=2000):
    if len(v) == 0:
        return {"mean": float("nan"), "ci": None, "n": 0}
    u = np.unique(g)
    idx = {x: np.where(g == x)[0] for x in u}
    bs = [v[np.concatenate([idx[x] for x in RNG.choice(u, len(u))])].mean() for _ in range(n)]
    return {"mean": round(100 * float(v.mean()), 1), "ci": [round(100 * float(np.quantile(bs, .025)), 1), round(100 * float(np.quantile(bs, .975)), 1)], "n": int(len(v))}


def main() -> None:
    runs = {}
    for f in sorted(glob.glob(str(O / "qwen3vl_8b_mixed_*.jsonl"))):
        m = re.match(r"qwen3vl_8b_mixed_(truth|random|none)_k128_L(\d+)_(truth|direct_answer|direct_probe)(?:_s(\d))?\.jsonl", Path(f).name)
        if m is None:  # other doses (steering_dose) and the *_rp26 / rprobe runs on a different item sample
            continue
        kind, layer, sign, seed = m.group(1), int(m.group(2)), m.group(3), m.group(4)
        runs[(kind, layer, sign, seed)] = {json.loads(l)["key"]: json.loads(l) for l in open(f)}
    base = runs.get(("none", 20, "truth", None))
    keys = sorted(base)
    g = np.array([k.replace("_behind", "").replace("_cf", "") for k in keys])
    grp = np.array([base[k]["item_group"] for k in keys])
    acc = lambda R: np.array([float(R[k]["answer_correct"]) for k in keys])
    rows = []
    for (kind, layer, sign, seed), R in sorted(runs.items(), key=lambda t: (t[0][0] != "none", t[0][1], t[0][0], t[0][2], t[0][3] or "")):
        a = acc(R)
        row = {"kind": kind if kind != "random" else f"random (seed {seed})", "layer": layer if kind != "none" else "--",
               "sign": {"truth": "true label (oracle)", "direct_answer": "Direct answer", "direct_probe": "Direct-position probe"}[sign] if kind != "none" else "--",
               "acc_all": boot(a, g)}
        for gg, key in (("D+C-", "acc_DCm"), ("D+C+", "acc_DCp"), ("D-C-", "acc_DmCm"), ("D-C+", "acc_DmCp")):
            s = grp == gg
            row[key] = boot(a[s], g[s])
        row["vs_none"] = boot(a - acc(base), g)
        rnd = [acc(v) for (k2, l2, s2, sd), v in runs.items() if k2 == "random" and l2 == layer]
        if kind == "truth" and rnd:
            row["vs_random_mean"] = boot(a - np.mean(rnd, 0), g)
            row["vs_random_mean_DCm"] = boot((a - np.mean(rnd, 0))[grp == "D+C-"], g[grp == "D+C-"])
            row["vs_random_mean_DCp"] = boot((a - np.mean(rnd, 0))[grp == "D+C+"], g[grp == "D+C+"])
        rows.append(row)
        print(row["layer"], row["kind"], row["sign"], "all", row["acc_all"]["mean"], row["acc_all"]["ci"], "| DCm", row["acc_DCm"]["mean"], "DCp", row["acc_DCp"]["mean"],
              "DmCm", row["acc_DmCm"]["mean"], "DmCp", row["acc_DmCp"]["mean"], "| vs none", row["vs_none"]["mean"], row["vs_none"]["ci"],
              "| vs rnd", row.get("vs_random_mean", {}).get("mean"), row.get("vs_random_mean", {}).get("ci"), "DCm", row.get("vs_random_mean_DCm", {}).get("mean"),
              row.get("vs_random_mean_DCm", {}).get("ci"), "DCp", row.get("vs_random_mean_DCp", {}).get("mean"))
    groups = {gg: int((grp == gg).sum()) for gg in ("D+C-", "D+C+", "D-C-", "D-C+")}
    json.dump({"model": "qwen3vl_8b", "k": 128, "n": len(keys), "groups": groups, "rows": rows}, open(B / "steering_mixed.json", "w"), indent=1)
    print(groups)


if __name__ == "__main__":
    main()
