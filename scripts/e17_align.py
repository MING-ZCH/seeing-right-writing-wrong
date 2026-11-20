"""E17c (CPU, pod): what is CoT-specific given Direct-position decodability (round-7 Opus M3), layer 20, cue set.

Same claim items per model:
  decodability: plain probe at the Direct answer position d, plain probe at the relation token r, within-claim probe at r
  readout alignment: share of items where the model's answer agrees with the probe's decoded truth
      Direct: Direct answer vs probe at d;  CoT: CoT answer vs (within-claim) probe at r and plain probe at the answer token a
  twin (conditional): among D+C- wrong-claim items where the original-image within-claim probe at r is correct, the share
      whose prediction on the identity-swapped twin's states equals the twin's truth (text fixed, image swapped)
Usage: python e17_align.py --model qwen3vl_8b   (writes $W/outputs/e17/<model>_align.json)
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
from sklearn.model_selection import GroupKFold

from e17_probe_analyze import OUT, fit, load, meta_of, pe, within_claim

LAYER = 20


def boot(v, g, rng, n=1000):
    u = np.unique(g)
    idx = {x: np.where(g == x)[0] for x in u}
    bs = [v[np.concatenate([idx[x] for x in rng.choice(u, len(u))])].mean() for _ in range(n)]
    return {"mean": round(float(v.mean()), 3), "ci": [round(float(np.quantile(bs, .025)), 3), round(float(np.quantile(bs, .975)), 3)], "n": int(len(v))}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    args = ap.parse_args()
    _, meta = meta_of(args.model)
    O_ = load(args.model, "orig", LAYER)
    T_ = load(args.model, "twin", LAYER)
    y = np.array([m["y_true"] for m in meta])
    pair = np.array([m["pair"] for m in meta])
    hc = np.array([m["has_claim"] for m in meta], bool)
    claim = np.array([int(m["named"] == m["x"]) if m["has_claim"] else -1 for m in meta])
    d_ok = np.array([pe(m["lp_direct"]) == m["gold"] for m in meta])
    c_ok = np.array([pe(m["lp_cot"]) == m["gold"] for m in meta])
    says = lambda lp, m: int((pe(lp) == "A") == (m["phrasing"] == "front"))
    d_says = np.array([says(m["lp_direct"], m) for m in meta])
    c_says = np.array([says(m["lp_cot"], m) for m in meta])
    rng = np.random.default_rng(0)

    def plain(pos):
        pr = np.full(len(y), -1)
        for tr, te in GroupKFold(5).split(O_, y, pair):
            clf, mu, sd = fit(O_[tr, pos], y[tr], balanced=False)
            pr[te] = clf.predict((O_[te, pos] - mu) / sd)
        return pr

    p_d, p_r, p_a = plain(0), plain(4), plain(5)
    p_rw, pa = within_claim(O_[:, 4], y, claim, pair, hc, {"twin": T_[:, 3]})
    sel = hc
    dcm = hc & d_ok & ~c_ok
    res = {"layer": LAYER, "n_items": int(sel.sum()),
           "decodability": {"plain_d": boot((p_d == y)[sel].astype(float), pair[sel], rng),
                            "plain_r": boot((p_r == y)[sel].astype(float), pair[sel], rng),
                            "plain_a": boot((p_a == y)[sel].astype(float), pair[sel], rng),
                            "within_r": boot((p_rw == y)[sel & (p_rw >= 0)].astype(float), pair[sel & (p_rw >= 0)], rng)},
           "behaviour": {"direct_acc": round(float(d_ok[sel].mean()), 3), "cot_acc": round(float(c_ok[sel].mean()), 3)},
           "alignment": {"direct_answer_vs_probe_d": boot((d_says == p_d)[sel].astype(float), pair[sel], rng),
                         "cot_answer_vs_within_r": boot((c_says == p_rw)[sel & (p_rw >= 0)].astype(float), pair[sel & (p_rw >= 0)], rng),
                         "cot_answer_vs_plain_a": boot((c_says == p_a)[sel].astype(float), pair[sel], rng),
                         "DCm_direct_answer_vs_probe_d": boot((d_says == p_d)[dcm].astype(float), pair[dcm], rng),
                         "DCm_cot_answer_vs_within_r": boot((c_says == p_rw)[dcm & (p_rw >= 0)].astype(float), pair[dcm & (p_rw >= 0)], rng)}}
    w = dcm & (claim != y) & (p_rw == y) & (pa["twin"] >= 0)
    res["twin_conditional"] = {"DCm_wrong_orig_correct_follow_twin": boot((pa["twin"] == 1 - y)[w].astype(float), pair[w], rng)}
    w2 = hc & (claim == y) & (p_rw == y) & (pa["twin"] >= 0)
    res["twin_conditional"]["right_claims_orig_correct_follow_twin"] = boot((pa["twin"] == 1 - y)[w2].astype(float), pair[w2], rng)
    wa = dcm & (claim != y) & (pa["twin"] >= 0)
    res["twin_conditional"]["DCm_wrong_all_follow_twin"] = boot((pa["twin"] == 1 - y)[wa].astype(float), pair[wa], rng)
    os.makedirs(OUT, exist_ok=True)
    json.dump(res, open(f"{OUT}/{args.model}_align.json", "w"), indent=1)
    print(json.dumps(res))


if __name__ == "__main__":
    main()
