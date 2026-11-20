"""E17x eval (CPU, pod): cue-set-trained probes applied frozen to the original twins and Synthetic Depth (round-10 M4.3).

Within-claim probe: per claim stratum, logistic on the cue-set original-image states at r (layer 20, all claim items);
applied unchanged to the r states of the new set (cue-set standardization). Rules:
  A  written first claim != probe's decoded relation -> answer with Direct, else CoT
  B  Direct and CoT disagree -> keep the one whose implied relation matches the probe; else CoT
Also the frozen Direct-position probe (cue-set d -> new-set d) used as the answer. Twin-pair / image bootstrap.
Usage: python e17_xdist_eval.py --model qwen3vl_8b   (writes $W/outputs/e17/<model>_xdist.json)
"""
from __future__ import annotations

import argparse
import glob
import json
import math

import numpy as np

from e17_probe_analyze import OUT, fit, load, meta_of


def pe(lp):
    fin = {k: v for k, v in lp.items() if v is not None and math.isfinite(v)}
    return max(fin, key=fin.get) if fin else None


def boot(v, g, rng, n=2000):
    u = np.unique(g)
    idx = {x: np.where(g == x)[0] for x in u}
    bs = [v[np.concatenate([idx[x] for x in rng.choice(u, len(u))])].mean() for _ in range(n)]
    return {"mean": round(100 * float(v.mean()), 1), "ci": [round(100 * float(np.quantile(bs, .025)), 1), round(100 * float(np.quantile(bs, .975)), 1)], "n": int(len(v))}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    a = ap.parse_args()
    rng = np.random.default_rng(0)
    _, meta = meta_of(a.model)
    y = np.array([m["y_true"] for m in meta])
    hc = np.array([m["has_claim"] for m in meta], bool)
    claim = np.array([int(m["named"] == m["x"]) if m["has_claim"] else -1 for m in meta])
    O_ = load(a.model, "orig", 20)
    probes = {}
    for c in (0, 1):
        idx = np.where(hc & (claim == c))[0]
        probes[c] = fit(O_[idx, 4], y[idx])
    dprobe = fit(O_[:, 0], y, balanced=False)
    fs = sorted(glob.glob(f"/dev/shm/e17x/{a.model}_[0-9].npy"))
    X = np.concatenate([np.load(f).astype(np.float32) for f in fs], 0)
    M = []
    for f in fs:
        M += [json.loads(l) for l in open(f.replace(".npy", ".jsonl"))]
    res = {}
    for st in ("twins", "syn"):
        sel = np.array([m["set"] == st for m in M])
        mm = [m for m, s in zip(M, sel) if s]
        H = X[sel]
        yy = np.array([m["y_true"] for m in mm])
        g = np.array([m["pair"] for m in mm])
        has = np.array([m["has_claim"] for m in mm], bool)
        cl = np.array([int(m["named"] == m["x"]) if m["has_claim"] else -1 for m in mm])
        pr = np.full(len(mm), -1)
        for c in (0, 1):
            s = has & (cl == c)
            clf, mu, sd = probes[c]
            if s.any():
                pr[s] = clf.predict((H[s, 1] - mu) / sd)
        w, r = has & (cl != yy), has & (cl == yy)
        bal = ((pr[w] == yy[w]).mean() + (pr[r] == yy[r]).mean()) / 2
        says = lambda lp, m: int((pe(lp) == "A") == (m["phrasing"] == "front"))
        dx = np.array([says(m["lp_direct"], m) for m in mm])
        cx = np.array([says(m["lp_cot"], m) for m in mm])
        d_ok = np.array([pe(m["lp_direct"]) == m["gold"] for m in mm], float)
        c_ok = np.array([pe(m["lp_cot"]) == m["gold"] for m in mm], float)
        accA = np.where(has & (pr != cl), d_ok, c_ok)
        pickD = (dx != cx) & has & (dx == pr)
        accB = np.where(pickD, d_ok, c_ok)
        clf, mu, sd = dprobe
        dp_acc = (clf.predict((H[:, 0] - mu) / sd) == yy).astype(float)
        out = {"n": len(mm), "claim_rate": round(100 * float(has.mean()), 1), "frozen_within_claim_bal": round(float(bal), 3),
               "frozen_recall_wrong": round(float((pr[w] == yy[w]).mean()), 3), "frozen_recall_right": round(float((pr[r] == yy[r]).mean()), 3),
               "direct": boot(d_ok, g, rng), "cot": boot(c_ok, g, rng),
               "A": {"acc": boot(accA, g, rng), "vs_cot": boot(accA - c_ok, g, rng), "vs_direct": boot(accA - d_ok, g, rng), "flag_rate": round(100 * float((has & (pr != cl)).mean()), 1)},
               "B": {"acc": boot(accB, g, rng), "vs_cot": boot(accB - c_ok, g, rng), "vs_direct": boot(accB - d_ok, g, rng), "switch_rate": round(100 * float(pickD.mean()), 1)},
               "frozen_direct_probe_as_answer": boot(dp_acc, g, rng)}
        res[st] = out
        print(a.model, st, json.dumps({k: (v if not isinstance(v, dict) else {kk: vv for kk, vv in v.items() if kk in ('mean', 'ci')} if 'mean' in v else
                                            {kk: (vv['mean'], vv['ci']) if isinstance(vv, dict) else vv for kk, vv in v.items()}) for k, v in out.items()}), flush=True)
    json.dump(res, open(f"{OUT}/{a.model}_xdist.json", "w"), indent=1)


if __name__ == "__main__":
    main()
