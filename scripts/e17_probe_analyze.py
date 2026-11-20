"""E17 (CPU, pod): within-claim truth probe with image controls (round-6 Opus M1).

Cue set, first claim from claim_parse v3. Within-claim probe: among items with the same written claim, decode
y = "queried subject is the front object" of the ORIGINAL image (logistic, class-balanced, GroupKFold(5) by twin pair).
Per layer and position, reported as balanced accuracy (mean of recall on wrong-claim and right-claim items) and
recall on D+C- wrong claims, with twin-pair bootstrap CIs:
  orig        original image (positions d, pre, s, rprev, r, a)
  gray_refit  gray image, probe refit on gray states (how much truth the text alone carries in this model's states)
  gray_xfer   probe trained on orig states applied to gray states (same position)
  twin_xfer   probe trained on orig states applied to identity-swapped-twin states; follows_shown = share of predictions
              equal to the twin's truth (1 - y)
  filler      e15 filler-matched token (same offset in a same-length filler pass), v3 claim strata
Also the Direct-position probe (trained on y at d) applied at the CoT answer token a (accuracy vs truth on D+C- wrong).

Usage: python e17_probe_analyze.py --model qwen3vl_8b   (writes $W/outputs/e17/<model>_ctrl.json)
"""
from __future__ import annotations
import vtax_paths as _vtp  # noqa: E402

import argparse
import glob
import json
import math
import os

import numpy as np
from joblib import Parallel, delayed
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold

IND = "/dev/shm/e17"
OUT = os.environ.get("VTAX_ROOT", f"{_vtp.W}") + "/outputs/e17"
OPOS = {"d": 0, "pre": 1, "s": 2, "rprev": 3, "r": 4, "a": 5}
CPOS = {"pre": 0, "s": 1, "rprev": 2, "r": 3, "a": 4}


def pe(lp):
    fin = {k: v for k, v in lp.items() if v is not None and math.isfinite(v)}
    return max(fin, key=fin.get) if fin else None


def meta_of(model):
    fs = sorted(glob.glob(f"{IND}/{model}_[0-9]_orig.npy"))
    meta = []
    for f in fs:
        meta += [json.loads(l) for l in open(f.replace("_orig.npy", ".jsonl"))]
    return fs, meta


def load(model, cond, layer):
    fs = sorted(glob.glob(f"{IND}/{model}_[0-9]_{cond}.npy"))
    return np.concatenate([np.asarray(np.load(f, mmap_mode="r")[:, :, layer, :], dtype=np.float32) for f in fs], 0)


def fit(X, y, balanced=True):
    mu, sd = X.mean(0), X.std(0) + 1e-4
    clf = LogisticRegression(C=0.05, max_iter=3000, class_weight="balanced" if balanced else None).fit((X - mu) / sd, y)
    return clf, mu, sd


def within_claim(Htr, y, claim, pair, sel, Happly: dict | None = None):
    """CV predictions within claim strata on Htr; optionally apply each fold's probe to other state sets (same rows)."""
    pr = np.full(len(y), -1)
    pa = {k: np.full(len(y), -1) for k in (Happly or {})}
    for c in (0, 1):
        idx = np.where(sel & (claim == c))[0]
        if len(idx) < 50 or len(np.unique(y[idx])) < 2:
            continue
        for tr, te in GroupKFold(5).split(idx, y[idx], pair[idx]):
            if len(np.unique(y[idx[tr]])) < 2:
                continue
            clf, mu, sd = fit(Htr[idx[tr]], y[idx[tr]])
            pr[idx[te]] = clf.predict((Htr[idx[te]] - mu) / sd)
            for k, H in (Happly or {}).items():
                pa[k][idx[te]] = clf.predict((H[idx[te]] - mu) / sd)
    return pr, pa


def metrics(pr, y, claim, sel, dcm, pair, rng, n=500):
    m = sel & (pr >= 0)
    w, r = m & (claim != y), m & (claim == y)
    dw = w & dcm
    ok = (pr == y).astype(float)
    u = np.unique(pair[m])
    idx = {g: np.where((pair == g) & m)[0] for g in u}
    def stat(ii):
        ww, rr, dd = w[ii], r[ii], dw[ii]
        return [(ok[ii][ww].mean() + ok[ii][rr].mean()) / 2, ok[ii][dd].mean() if dd.any() else np.nan]
    pt = stat(np.where(m)[0])
    bs = np.array([stat(np.concatenate([idx[g] for g in rng.choice(u, len(u))])) for _ in range(n)])
    q = lambda j: [round(float(np.nanquantile(bs[:, j], .025)), 3), round(float(np.nanquantile(bs[:, j], .975)), 3)]
    return {"bal": round(float(pt[0]), 3), "bal_ci": q(0), "recall_DCm_wrong": round(float(pt[1]), 3), "recall_DCm_wrong_ci": q(1),
            "recall_wrong": round(float(ok[w].mean()), 3), "recall_right": round(float(ok[r].mean()), 3), "n_DCm_wrong": int(dw.sum())}


def job(model: str, layer: int) -> dict:
    _, meta = meta_of(model)
    y = np.array([m["y_true"] for m in meta])
    pair = np.array([m["pair"] for m in meta])
    hc = np.array([m["has_claim"] for m in meta], bool)
    claim = np.array([int(m["named"] == m["x"]) if m["has_claim"] else -1 for m in meta])
    d_ok = np.array([pe(m["lp_direct"]) == m["gold"] for m in meta])
    c_ok = np.array([pe(m["lp_cot"]) == m["gold"] for m in meta])
    dcm = hc & d_ok & ~c_ok
    rng = np.random.default_rng(layer)
    O = load(model, "orig", layer)
    G = load(model, "gray", layer)
    T = load(model, "twin", layer)
    out = {"layer": layer, "orig": {}, "gray_refit": {}, "gray_xfer": {}, "twin_xfer": {}}
    for p, i in OPOS.items():
        app = {"gray": G[:, CPOS[p]], "twin": T[:, CPOS[p]]} if p in CPOS else None
        pr, pa = within_claim(O[:, i], y, claim, pair, hc, app)
        out["orig"][p] = metrics(pr, y, claim, hc, dcm, pair, rng)
        if p in CPOS:
            out["gray_xfer"][p] = metrics(pa["gray"], y, claim, hc, dcm, pair, rng)
            tw = pa["twin"]
            m = hc & (tw >= 0)
            out["twin_xfer"][p] = metrics(tw, y, claim, hc, dcm, pair, rng) | {
                "follows_shown_image": round(float(np.mean(tw[m] == 1 - y[m])), 3),
                "follows_shown_image_DCm_wrong": round(float(np.mean(tw[m & dcm & (claim != y)] == 1 - y[m & dcm & (claim != y)])), 3)}
            prg, _ = within_claim(G[:, CPOS[p]], y, claim, pair, hc)
            out["gray_refit"][p] = metrics(prg, y, claim, hc, dcm, pair, rng)
    # filler-matched token from e15 (same items, v3 strata)
    f15 = sorted(f for f in glob.glob(f"/dev/shm/e15/{model}_[0-9].npy"))
    if f15:
        m15 = []
        for f in f15:
            m15 += [json.loads(l) for l in open(f.replace(".npy", ".jsonl"))]
        F = np.concatenate([np.asarray(np.load(f, mmap_mode="r")[:, 4, layer, :], dtype=np.float32) for f in f15], 0)
        pos15 = {m["key"]: i for i, m in enumerate(m15) if m["set"] == "cues"}
        keep = np.array([m["key"] in pos15 for m in meta])
        Ff = np.zeros_like(O[:, 0])
        Ff[keep] = F[[pos15[m["key"]] for m in meta if m["key"] in pos15]]
        prf, _ = within_claim(Ff, y, claim, pair, hc & keep)
        out["filler"] = metrics(prf, y, claim, hc & keep, dcm, pair, rng)
    # Direct probe at d applied to the CoT answer token a
    pa_ = np.full(len(y), -1)
    for tr, te in GroupKFold(5).split(O, y, pair):
        clf, mu, sd = fit(O[tr, 0], y[tr], balanced=False)
        pa_[te] = clf.predict((O[te, 5] - O[:, 5].mean(0) + O[:, 0].mean(0) - mu) / sd)
    dw = dcm & (claim != y)
    out["direct_probe_at_answer"] = {"acc_DCm_wrong": round(float((pa_[dw] == y[dw]).mean()), 3),
                                     "follows_cot_answer_DCm_wrong": round(float((pa_[dw] == claim[dw]).mean()), 3), "n": int(dw.sum())}
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--jobs", type=int, default=10)
    args = ap.parse_args()
    fs, meta = meta_of(args.model)
    L = np.load(fs[0], mmap_mode="r").shape[2]
    layers = sorted(set(list(range(0, L, 4)) + [L - 1]))
    hc = np.array([m["has_claim"] for m in meta], bool)
    print(f"[e17a] {args.model}: {len(meta)} items, {hc.sum()} claims, layers {layers}", flush=True)
    res = Parallel(n_jobs=args.jobs, verbose=5)(delayed(job)(args.model, l) for l in layers)
    os.makedirs(OUT, exist_ok=True)
    json.dump({"model": args.model, "n": len(meta), "n_claims": int(hc.sum()), "layers": res}, open(f"{OUT}/{args.model}_ctrl.json", "w"), indent=1)
    for r in res:
        o = r["orig"]
        print(f"L{r['layer']:2d} orig r {o['r']['bal']} ({o['r']['recall_DCm_wrong']}) rprev {o['rprev']['bal']} s {o['s']['bal']} pre {o['pre']['bal']} a {o['a']['bal']} ({o['a']['recall_DCm_wrong']}) "
              f"| gray refit r {r['gray_refit']['r']['bal']} xfer {r['gray_xfer']['r']['bal']} | twin follows shown {r['twin_xfer']['r']['follows_shown_image']} "
              f"| filler {r.get('filler', {}).get('bal')} | Dprobe@a DCm-wrong acc {r['direct_probe_at_answer']['acc_DCm_wrong']}", flush=True)


if __name__ == "__main__":
    main()
