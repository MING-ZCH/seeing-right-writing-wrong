"""E17d (CPU, pod): round-9 Opus M1 / oral 3, layer 20, cue set, claim_parse v3 strata.

Per model and CoT position (pre = s-1, s, rprev = r-1, r, a):
  orig_bal            within-claim probe on original states (CV), balanced accuracy
  gray_refit_bal      probe refit on gray states (original truth), balanced accuracy -> image share (orig-gray)/(orig-0.5)
  pre_swap / post_swap share of claim items whose prediction equals the twin's relation, on original states (CV) vs on
                      twin states (original-trained probe); conflict subset = right claims the probe gets right on original
  conf_*              mean |p - 0.5| on original / twin / gray states (same original-trained probe)
  twin_refit_bal      probe refit on twin states with the twin's relation as label; cos = cosine between its weight and the
                      original probe's weight (positive: twin truth encoded like original truth, i.e. image-carried)
Also writes per-item predictions at r for the detector analysis, and counts, at layer 26 r, D+C- wrong-claim items of
the steering set where the within-claim r-probe sign disagrees with the label.
Usage: python e17_m1.py --model qwen3vl_8b   (writes $W/outputs/e17/<model>_m1.json, <model>_m1_items.jsonl)
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
from sklearn.model_selection import GroupKFold

from e17_probe_analyze import OUT, fit, load, meta_of, pe

CPOS = {"pre": 0, "s": 1, "rprev": 2, "r": 3, "a": 4}
OPOS = {"pre": 1, "s": 2, "rprev": 3, "r": 4, "a": 5}


def within(Htr, y, claim, pair, sel, apply: dict | None = None):
    pr = np.full(len(y), -1)
    pp = np.full(len(y), np.nan)
    pa = {k: np.full(len(y), np.nan) for k in (apply or {})}
    W = []
    for c in (0, 1):
        idx = np.where(sel & (claim == c))[0]
        for tr, te in GroupKFold(5).split(idx, y[idx], pair[idx]):
            clf, mu, sd = fit(Htr[idx[tr]], y[idx[tr]])
            pr[idx[te]] = clf.predict((Htr[idx[te]] - mu) / sd)
            pp[idx[te]] = clf.predict_proba((Htr[idx[te]] - mu) / sd)[:, 1]
            W.append(clf.coef_[0] / sd)
            for k, H in (apply or {}).items():
                pa[k][idx[te]] = clf.predict_proba((H[idx[te]] - mu) / sd)[:, 1]
    return pr, pp, pa, np.mean(W, 0)


def bal(pr, y, claim, sel):
    m = sel & (pr >= 0)
    w, r = m & (claim != y), m & (claim == y)
    return round(float(((pr[w] == y[w]).mean() + (pr[r] == y[r]).mean()) / 2), 3)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    args = ap.parse_args()
    _, meta = meta_of(args.model)
    y = np.array([m["y_true"] for m in meta])
    pair = np.array([m["pair"] for m in meta])
    hc = np.array([m["has_claim"] for m in meta], bool)
    claim = np.array([int(m["named"] == m["x"]) if m["has_claim"] else -1 for m in meta])
    O_, G_, T_ = load(args.model, "orig", 20), load(args.model, "gray", 20), load(args.model, "twin", 20)
    res = {"layer": 20, "positions": {}}
    items = {}
    for p in CPOS:
        Ho, Hg, Ht = O_[:, OPOS[p]], G_[:, CPOS[p]], T_[:, CPOS[p]]
        pr, pp, pa, w_o = within(Ho, y, claim, pair, hc, {"twin": Ht, "gray": Hg})
        prg, _, _, _ = within(Hg, y, claim, pair, hc)
        prt, _, _, w_t = within(Ht, 1 - y, claim, pair, hc)
        ob, gb = bal(pr, y, claim, hc), bal(prg, y, claim, hc)
        twin_pred = (pa["twin"] >= 0.5).astype(int)
        conflict = hc & (claim == y) & (pr == y)
        out = {"orig_bal": ob, "gray_refit_bal": gb, "image_share_gray": round((ob - gb) / (ob - 0.5), 2),
               "pre_swap_follow_twin": round(float(np.mean(pr[hc] == 1 - y[hc])), 3),
               "post_swap_follow_twin": round(float(np.mean(twin_pred[hc] == 1 - y[hc])), 3),
               "conflict_post_swap": round(float(np.mean(twin_pred[conflict] == 1 - y[conflict])), 3), "n_conflict": int(conflict.sum()),
               "conf_orig": round(float(np.nanmean(np.abs(pp[hc] - 0.5))), 3), "conf_twin": round(float(np.nanmean(np.abs(pa["twin"][hc] - 0.5))), 3),
               "conf_gray": round(float(np.nanmean(np.abs(pa["gray"][hc] - 0.5))), 3),
               "conf_twin_flipped": round(float(np.nanmean(np.abs(pa["twin"][conflict & (twin_pred == 1 - y)] - 0.5))), 3),
               "conf_twin_kept": round(float(np.nanmean(np.abs(pa["twin"][conflict & (twin_pred == y)] - 0.5))), 3),
               "twin_refit_bal_twin_truth": bal(prt, 1 - y, claim, hc),
               "cos_twin_refit_vs_orig": round(float(w_t @ w_o / (np.linalg.norm(w_t) * np.linalg.norm(w_o))), 3)}
        res["positions"][p] = out
        if p == "r":
            for i, m in enumerate(meta):
                items[m["key"]] = {"pred_r": int(pr[i]), "p_r": None if np.isnan(pp[i]) else round(float(pp[i]), 4)}
        print(args.model, p, out, flush=True)
    # per-item data for the detector
    says = lambda lp, m: int((pe(lp) == "A") == (m["phrasing"] == "front"))
    with open(f"{OUT}/{args.model}_m1_items.jsonl", "w") as f:
        for m in meta:
            f.write(json.dumps({"key": m["key"], "pair": m["pair"], "y": m["y_true"], "claim_x": int(m["named"] == m["x"]) if m["has_claim"] else None,
                                "direct_says_x": says(m["lp_direct"], m), "cot_says_x": says(m["lp_cot"], m),
                                "direct_ok": pe(m["lp_direct"]) == m["gold"], "cot_ok": pe(m["lp_cot"]) == m["gold"], **items[m["key"]]}) + "\n")
    # steering-set disagreement count at layer 26 (r)
    if os.path.exists(f"/dev/shm/e17/{args.model}_steer_items.json"):
        O26 = load(args.model, "orig", 26)[:, 4]
        pr26, _, _, _ = within(O26, y, claim, pair, hc)
        keys = set(json.load(open(f"/dev/shm/e17/{args.model}_steer_items.json"))["DCm_wrong"])
        sel = np.array([m["key"] in keys for m in meta])
        res["steer_DCm_wrong_rprobe_disagree_L26"] = {"n": int(sel.sum()), "disagree": int((pr26[sel] != y[sel]).sum())}
        print("steer disagree", res["steer_DCm_wrong_rprobe_disagree_L26"], flush=True)
    json.dump(res, open(f"{OUT}/{args.model}_m1.json", "w"), indent=1)


if __name__ == "__main__":
    main()
