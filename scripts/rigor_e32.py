"""Exp-32: three-law evidence (CPU only, no new model runs).

two     L3 'two-channel' test. Per model x dataset: Direct / CoT accuracy, accuracy of the written relation (first and
        last relational claim, claim_parse v3, over items with a parsed claim) and parse coverage, share of CoT answers
        agreeing with the last written relation, D+C- / D-C+ / churn / net. Then across model x dataset points:
        x = written-relation acc - Direct acc, y = CoT acc - Direct acc (Pearson, Spearman, OLS slope with point and
        model-cluster bootstrap CIs, residuals from y = x).
        Datasets: cue-factorial set (e10 <m>_cues.jsonl, 1,600 queries, CoT text = the CoT run behind the GLM);
        occlusion twins (e2d front + behind, 800 queries; the twin d'_pair run); Synthetic Depth (e2b legacy
        rationales = Sec. 5 rationales, Tab. 1 readout from build_bundle.load_main; 431 overlapping images primary,
        all 500 as a variant). No relation parser exists for CV-Bench 2D Relation / Synthetic Relative Position.
law1    L1 per-model table (supp_law1.tex, bundle/law1.json).
law2    L2 per-model table (supp_law2.tex, bundle/law2.json).
fig     paper figure fig_two_channel.{pdf,png,data.json}.
-> bundle/two_channel.json, bundle/rigor_two_channel.txt, bundle/law{1,2}.json, paper tables/supp_law{1,2}.tex

Usage: python rigor_e32.py [--only two,law1,law2,fig]
"""
from __future__ import annotations
import vtax_paths as _vtp  # noqa: E402

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import pearsonr, spearmanr

import build_bundle as BB
from build_bundle import pe, rj
from claim_parse import first_claim_v3
from e2f_claim_edit import split_sents

O = Path(f"{_vtp.OUTPUTS}")
L = Path(f"{_vtp.LEGACY_RESULTS}")
B = Path("outputs/bundle")
PAPER = Path(f"{_vtp.PAPER}")
M7 = ["qwen3b", "qwen7b", "internvl3_8b", "phi35v", "molmo7b", "qwen3vl_8b", "internvl35_8b"]
NAMES = {"qwen3b": "Qwen2.5-VL-3B", "qwen7b": "Qwen2.5-VL-7B", "internvl3_8b": "InternVL3-8B", "phi35v": "Phi-3.5-V",
         "molmo7b": "Molmo-7B-D", "qwen3vl_8b": "Qwen3-VL-8B", "internvl35_8b": "InternVL3.5-8B",
         "qwen3vl_8b_think": "Qwen3-VL-8B-Think."}
QRX = re.compile(r"Is the (\w+ \w+) (in front of|behind) the (\w+ \w+)\?")
PREFIX = "Let's think step by step."
NB_ITEM, NB_PT = 1000, 10000


# ============================================================================= per-item records

def sent_front(s: str, front: str, back: str) -> str | None:
    c = first_claim_v3([s])
    return c[1] if c and c[1] in (front, back) else None


def claims(sents: list[str], front: str, back: str) -> tuple[str | None, str | None, int]:
    cl = [f for f in (sent_front(s, front, back) for s in sents) if f]
    return (cl[0] if cl else None), (cl[-1] if cl else None), len(cl)


def implied(front_claimed: str, x: str, phrasing: str) -> str:
    """Answer letter implied by a written 'front' object for 'Is x in front of / behind y?'."""
    yes = (front_claimed == x) if phrasing == "front" else (front_claimed != x)
    return "A" if yes else "B"


def rec(model, ds, iid, cluster, gold, d_ans, c_ans, sents, front, back, x, phrasing, extra=None) -> dict:
    f, l, k = claims(sents, front, back)
    r = {"model": model, "ds": ds, "item": iid, "cl": cluster, "gold": gold, "phrasing": phrasing,
         "d_ok": d_ans == gold, "c_ok": c_ans == gold, "c_ans": c_ans, "n_claims": k, "n_sents": len(sents),
         "first_true": None if f is None else f == front, "last_true": None if l is None else l == front,
         "last_agree": None if l is None else implied(l, x, phrasing) == c_ans,
         "first_agree": None if f is None else implied(f, x, phrasing) == c_ans}
    return r | (extra or {})


def body(text: str) -> list[str]:
    t = text[len(PREFIX):] if text.startswith(PREFIX) else text
    return split_sents(t)


def items_cues(m: str) -> list[dict]:
    R = defaultdict(dict)
    for r in rj(O / "e10" / f"{m}_cues.jsonl"):
        if r["condition"] in ("direct", "cot"):
            R[r["item_id"]][r["condition"]] = r
    out = []
    for iid, d in R.items():
        c, dr = d["cot"], d["direct"]
        mt = c["meta"]
        out.append(rec(m, "cues", iid, c["image_id"], c["gold"], pe(dr), pe(c), body(c.get("text") or ""),
                       mt["front_name"], mt["back_name"], mt["x_name"], c["phrasing"]))
    return out


def items_twins(m: str) -> list[dict]:
    out = []
    for suf, ph in (("", "front"), ("_behind", "behind")):
        R = defaultdict(dict)
        for r in rj(O / "e2d" / f"{m}{suf}.jsonl"):
            if r["condition"] in ("direct", "cot"):
                R[r["item_id"]][r["condition"]] = r
        for iid, d in R.items():
            if "cot" not in d or "direct" not in d:
                continue
            c, dr = d["cot"], d["direct"]
            mt = c["meta"]
            out.append(rec(m, "twins", f"{iid}|{ph}", iid.replace("_cf", ""), c["gold"], pe(dr), pe(c),
                           body(c.get("text") or ""), mt["front_name"], mt["back_name"], mt["x_name"], ph))
    return out


def items_syn(m: str, cat: dict) -> list[dict]:
    main = BB.load_main(O, L, m)
    if "direct" not in main or "cot" not in main:
        return []
    out = []
    for r in rj(O / "e2b" / f"{m}.jsonl"):
        if r["task"] != "syn_depth" or r["item_id"] not in main["cot"]:
            continue
        q = QRX.search(r["question"])
        x, y = q.group(1).lower(), q.group(3).lower()
        front, back = (x, y) if r["gold"] == "A" else (y, x)
        uid = r["item_id"]
        out.append(rec(m, "syn", uid, uid, r["gold"], main["direct"][uid][4], main["cot"][uid][4], r["sentences"],
                       front, back, x, "front", {"overlap": cat.get(uid) == "overlap"}))
    return out


def load_items() -> list[dict]:
    from rigor_e17i import category
    cat = {r["uid"]: category(r) for r in json.load(open(B / "syn_overlap_metrics.json"))}
    allr = []
    for m in M7:
        allr += items_cues(m) + items_twins(m) + items_syn(m, cat)
    allr += items_twins("qwen3vl_8b_think")
    return allr


# ============================================================================= cell statistics

def cell_stats(rs: list[dict], rng) -> dict:
    """All per-cell numbers with an item-cluster bootstrap for the x / y quantities."""
    cl = np.array([r["cl"] for r in rs])
    uc, inv = np.unique(cl, return_inverse=True)
    d = np.array([r["d_ok"] for r in rs], float)
    c = np.array([r["c_ok"] for r in rs], float)
    out = {"n": len(rs), "n_clusters": int(len(uc)),
           "acc_direct": 100 * d.mean(), "acc_cot": 100 * c.mean(),
           "DpCm": int(((d == 1) & (c == 0)).sum()), "DmCp": int(((d == 0) & (c == 1)).sum()),
           "DpCp": int(((d == 1) & (c == 1)).sum()), "DmCm": int(((d == 0) & (c == 0)).sum())}
    out["churn"] = 100 * (out["DpCm"] + out["DmCp"]) / len(rs)
    out["net"] = 100 * (out["DmCp"] - out["DpCm"]) / len(rs)
    for w in ("first", "last"):
        p = np.array([r[f"{w}_true"] is not None for r in rs])
        t = np.array([bool(r[f"{w}_true"]) for r in rs], float)
        ag = np.array([bool(r[f"{w}_agree"]) for r in rs], float)
        out[f"{w}_coverage"] = 100 * p.mean()
        out[f"{w}_n_parsed"] = int(p.sum())
        if p.sum():
            out[f"{w}_written_acc"] = 100 * t[p].mean()
            out[f"{w}_agree"] = 100 * ag[p].mean()
            out[f"{w}_direct_on_parsed"] = 100 * d[p].mean()
            out[f"{w}_cot_on_parsed"] = 100 * c[p].mean()
            # x / y definitions
            out[f"{w}_x"] = out[f"{w}_written_acc"] - out[f"{w}_direct_on_parsed"]          # written - Direct, parsed items
            out[f"{w}_y_l4l"] = out[f"{w}_cot_on_parsed"] - out[f"{w}_direct_on_parsed"]     # CoT - Direct, parsed items
            out[f"{w}_x_mix"] = (t[p].sum() + d[~p].sum()) / len(rs) * 100 - out["acc_direct"]  # written where parsed, else Direct
            if (~p).sum():
                out[f"{w}_unparsed_cot_minus_direct"] = 100 * (c[~p].mean() - d[~p].mean())
                out[f"{w}_n_unparsed"] = int((~p).sum())
    out["y"] = out["acc_cot"] - out["acc_direct"]
    # cluster bootstrap CIs for x, y, x_mix, y_l4l (last claim)
    p = np.array([r["last_true"] is not None for r in rs], float)
    t = np.array([bool(r["last_true"]) for r in rs], float)
    agg = np.zeros((len(uc), 6))
    np.add.at(agg, inv, np.stack([np.ones(len(rs)), d, c, p, t * p, d * p], 1))
    cp = np.zeros(len(uc))
    np.add.at(cp, inv, c * p)

    def st(A, cpp):
        s, sc = A.sum(0), cpp.sum()
        x = 100 * (s[4] - s[5]) / s[3]
        yl = 100 * (sc - s[5]) / s[3]
        y = 100 * (s[2] - s[1]) / s[0]
        xm = 100 * (s[4] + (s[1] - s[5]) - s[1]) / s[0]
        return x, y, yl, xm

    bs = []
    for _ in range(NB_ITEM):
        i = rng.integers(0, len(uc), len(uc))
        bs.append(st(agg[i], cp[i]))
    bs = np.array(bs)
    for k, nm in enumerate(("last_x", "y", "last_y_l4l", "last_x_mix")):
        out[f"{nm}_ci"] = [round(float(np.quantile(bs[:, k], .025)), 2), round(float(np.quantile(bs[:, k], .975)), 2)]
    return {k: (round(v, 2) if isinstance(v, float) else v) for k, v in out.items()}


# ============================================================================= L3 regression across points

def fit_points(pts: list[dict], xk: str, yk: str, rng) -> dict:
    x = np.array([p[xk] for p in pts], float)
    y = np.array([p[yk] for p in pts], float)
    n = len(x)
    r, pr = pearsonr(x, y)
    rho, ps = spearmanr(x, y)
    A = np.column_stack([np.ones(n), x])
    coef = np.linalg.lstsq(A, y, rcond=None)[0]
    res = y - A @ coef
    s2 = res @ res / (n - 2)
    H = A @ np.linalg.inv(A.T @ A) @ A.T
    h = np.diag(H)
    stud = res / np.sqrt(s2 * (1 - h))
    cook = res ** 2 / (2 * s2) * h / (1 - h) ** 2
    bs = []
    for _ in range(NB_PT):
        i = rng.integers(0, n, n)
        if np.ptp(x[i]) == 0:
            continue
        bs.append(np.polyfit(x[i], y[i], 1))
    bs = np.array(bs)
    models = sorted({p["model"] for p in pts})
    mi = {m: np.array([k for k, p in enumerate(pts) if p["model"] == m]) for m in models}
    bm = []
    for _ in range(NB_PT):
        i = np.concatenate([mi[models[j]] for j in rng.integers(0, len(models), len(models))])
        if np.ptp(x[i]) == 0:
            continue
        bm.append(np.polyfit(x[i], y[i], 1))
    bm = np.array(bm)
    q = lambda v: [round(float(np.quantile(v, .025)), 3), round(float(np.quantile(v, .975)), 3)]
    ident = y - x
    out = {"n_points": n, "x": xk, "y": yk, "pearson_r": round(float(r), 3), "pearson_p": float(pr),
           "spearman_rho": round(float(rho), 3), "spearman_p": float(ps),
           "ols_slope": round(float(coef[1]), 3), "ols_intercept": round(float(coef[0]), 2),
           "slope_ci_point_boot": q(bs[:, 0]), "slope_ci_model_cluster_boot": q(bm[:, 0]),
           "intercept_ci_point_boot": q(bs[:, 1]),
           "r2_ols": round(float(1 - res @ res / ((y - y.mean()) @ (y - y.mean()))), 3),
           "r2_identity": round(float(1 - ident @ ident / ((y - y.mean()) @ (y - y.mean()))), 3),
           "mean_abs_dev_from_identity": round(float(np.abs(ident).mean()), 2),
           "points": [{"model": p["model"], "ds": p["ds"], "x": round(float(x[k]), 2), "y": round(float(y[k]), 2),
                       "y_minus_x": round(float(ident[k]), 2), "ols_resid": round(float(res[k]), 2),
                       "studentized": round(float(stud[k]), 2), "cook": round(float(cook[k]), 3)} for k, p in enumerate(pts)]}
    out["points"].sort(key=lambda d: -abs(d["y_minus_x"]))
    # leave-one-model-out slope
    out["loo_model_slope"] = {m: round(float(np.polyfit(x[[k for k, p in enumerate(pts) if p["model"] != m]],
                                                        y[[k for k, p in enumerate(pts) if p["model"] != m]], 1)[0]), 3)
                              for m in models}
    keep = [k for k, p in enumerate(pts) if p["model"] not in ("molmo7b", "phi35v")]
    if len(keep) > 3:
        rr = pearsonr(x[keep], y[keep])[0]
        sl = np.polyfit(x[keep], y[keep], 1)
        ide = y[keep] - x[keep]
        out["without_molmo_phi"] = {"n": len(keep), "pearson_r": round(float(rr), 3), "ols_slope": round(float(sl[0]), 3),
                                    "ols_intercept": round(float(sl[1]), 2), "mean_abs_dev_from_identity": round(float(np.abs(ide).mean()), 2)}
    return out


DS_LABEL = {"cues": "cue set (1,600)", "twins": "occlusion twins (800)", "syn431": "Synthetic Depth 431", "syn500": "Synthetic Depth 500"}


def two_channel(lines: list[str]) -> dict:
    items = load_items()
    rng = np.random.default_rng(32)
    cells = {}
    groups = defaultdict(list)
    for r in items:
        if r["ds"] == "syn":
            groups[(r["model"], "syn500")].append(r)
            if r["overlap"]:
                groups[(r["model"], "syn431")].append(r)
        else:
            groups[(r["model"], r["ds"])].append(r)
    for (m, ds), rs in sorted(groups.items()):
        cells[f"{m}|{ds}"] = cell_stats(rs, rng) | {"model": m, "ds": ds}
        c = cells[f"{m}|{ds}"]
        lines.append(f"CELL {m:16s} {ds:6s} n={c['n']:4d} Dir {c['acc_direct']:5.1f} CoT {c['acc_cot']:5.1f} (y {c['y']:+5.1f} {c['y_ci']}) "
                     f"| cov first {c['first_coverage']:5.1f} last {c['last_coverage']:5.1f} | written acc first {c.get('first_written_acc', float('nan')):5.1f} "
                     f"last {c.get('last_written_acc', float('nan')):5.1f} | agree(last) {c.get('last_agree', float('nan')):5.1f} "
                     f"| x(last) {c.get('last_x', float('nan')):+5.1f} {c.get('last_x_ci')} y_l4l {c.get('last_y_l4l', float('nan')):+5.1f} "
                     f"x_mix {c.get('last_x_mix', float('nan')):+5.1f} | D+C- {c['DpCm']} D-C+ {c['DmCp']} churn {c['churn']:.1f} net {c['net']:+.1f}")
    fits = {}
    sets = {"primary": [k for k in cells if k.split("|")[1] in ("cues", "twins", "syn431") and k.split("|")[0] in M7],
            "with_think": [k for k in cells if k.split("|")[1] in ("cues", "twins", "syn431")],
            "syn500": [k for k in cells if k.split("|")[1] in ("cues", "twins", "syn500") and k.split("|")[0] in M7]}
    specs = {"last_main": ("last_x", "y"), "last_l4l": ("last_x", "last_y_l4l"), "last_mix": ("last_x_mix", "y"),
             "first_main": ("first_x", "y"), "first_l4l": ("first_x", "first_y_l4l"), "first_mix": ("first_x_mix", "y")}
    for sn, ks in sets.items():
        pts = [cells[k] for k in ks]
        for spn, (xk, yk) in specs.items():
            if sn != "primary" and not spn.startswith("last"):
                continue
            f = fit_points(pts, xk, yk, np.random.default_rng(320))
            fits[f"{sn}|{spn}"] = f
            lines.append(f"L3 {sn:10s} {spn:10s} n={f['n_points']} r {f['pearson_r']:+.3f} (p {f['pearson_p']:.2g}) rho {f['spearman_rho']:+.3f} "
                         f"(p {f['spearman_p']:.2g}) | OLS slope {f['ols_slope']:.3f} point-boot {f['slope_ci_point_boot']} model-boot "
                         f"{f['slope_ci_model_cluster_boot']} int {f['ols_intercept']:+.2f} | R2 ols {f['r2_ols']} identity {f['r2_identity']} "
                         f"MAD(y-x) {f['mean_abs_dev_from_identity']} | w/o Molmo+Phi {f.get('without_molmo_phi')}")
            if sn == "primary":
                lines.append("   largest |y-x|: " + "; ".join(f"{p['model']}/{p['ds']} x {p['x']:+.1f} y {p['y']:+.1f} (y-x {p['y_minus_x']:+.1f}, stud {p['studentized']:+.2f})"
                                                       for p in f["points"][:6]))
                lines.append(f"   leave-one-model-out slope {f['loo_model_slope']}")
    # per-dataset correlations (7 points each)
    per_ds = {}
    for ds in ("cues", "twins", "syn431", "syn500"):
        pts = [cells[f"{m}|{ds}"] for m in M7 if f"{m}|{ds}" in cells]
        x = np.array([p["last_x"] for p in pts]); y = np.array([p["y"] for p in pts]); yl = np.array([p["last_y_l4l"] for p in pts])
        per_ds[ds] = {"n": len(pts), "pearson_r_main": round(float(pearsonr(x, y)[0]), 3), "slope_main": round(float(np.polyfit(x, y, 1)[0]), 3),
                      "pearson_r_l4l": round(float(pearsonr(x, yl)[0]), 3), "slope_l4l": round(float(np.polyfit(x, yl, 1)[0]), 3)}
        lines.append(f"L3 per-dataset {ds:6s} {per_ds[ds]}")
    # shared-term null: x and y both subtract Direct, so r > 0 even if CoT ignored the written channel. Permute the
    # written-relation accuracy across models within each dataset (Direct / CoT fixed) and recompute r and slope.
    prim = [cells[k] for k in sets["primary"]]
    rngp = np.random.default_rng(321)
    y = np.array([p["y"] for p in prim]); yl = np.array([p["last_y_l4l"] for p in prim])
    wa = np.array([p["last_written_acc"] for p in prim]); dp = np.array([p["last_direct_on_parsed"] for p in prim])
    dsi = {ds: np.array([k for k, p in enumerate(prim) if p["ds"] == ds]) for ds in ("cues", "twins", "syn431")}
    obs_r, obs_rl = pearsonr(wa - dp, y)[0], pearsonr(wa - dp, yl)[0]
    nr, nrl, ns, nmad = [], [], [], []
    for _ in range(10000):
        w2 = wa.copy()
        for ix in dsi.values():
            w2[ix] = wa[rngp.permutation(ix)]
        xx = w2 - dp
        nr.append(pearsonr(xx, y)[0]); nrl.append(pearsonr(xx, yl)[0]); ns.append(np.polyfit(xx, y, 1)[0])
        nmad.append(np.abs(yl - xx).mean())
    nr, nrl, ns, nmad = map(np.array, (nr, nrl, ns, nmad))
    mad_obs = float(np.abs(yl - (wa - dp)).mean())
    null = {"r_main_obs": round(float(obs_r), 3), "r_main_null_mean": round(float(nr.mean()), 3), "r_main_null_95": [round(float(np.quantile(nr, .025)), 3), round(float(np.quantile(nr, .975)), 3)],
            "p_r_main": float((np.sum(nr >= obs_r) + 1) / (len(nr) + 1)),
            "r_l4l_obs": round(float(obs_rl), 3), "r_l4l_null_mean": round(float(nrl.mean()), 3), "p_r_l4l": float((np.sum(nrl >= obs_rl) + 1) / (len(nrl) + 1)),
            "slope_main_null_mean": round(float(ns.mean()), 3),
            "mad_l4l_obs": round(mad_obs, 2), "mad_l4l_null_mean": round(float(nmad.mean()), 2), "p_mad": float((np.sum(nmad <= mad_obs) + 1) / (len(nmad) + 1)),
            "level_corr_written_vs_cot_on_parsed": round(float(pearsonr(wa, np.array([p["last_cot_on_parsed"] for p in prim]))[0]), 3),
            "level_corr_direct_vs_cot_on_parsed": round(float(pearsonr(dp, np.array([p["last_cot_on_parsed"] for p in prim]))[0]), 3)}
    lines.append(f"L3 shared-Direct null (written acc permuted across models within dataset, 10,000): {null}")
    # pooled agreement and sign checks
    sign = sum(np.sign(p["last_x"]) == np.sign(p["y"]) for p in prim)
    lines.append(f"L3 sign agreement x(last) vs y over primary points: {sign}/{len(prim)}")
    disagree = [f"{p['model']}/{p['ds']}" for p in prim if np.sign(p["last_x"]) != np.sign(p["y"])]
    lines.append(f"L3 sign disagreements: {disagree}")
    return {"_doc": __doc__.split("law1")[0], "cells": cells, "fits": fits, "per_dataset": per_ds, "shared_direct_null": null,
            "sign_agree_primary": [int(sign), len(prim)], "sign_disagree": disagree}


# ============================================================================= accounting prediction (Exp-36)

def _acct_terms(W: float, aT: float, aF: float, D: float) -> float:
    """y_hat = (aT + aF - 1) W + (1 - aF) - D_all, all inputs in percent; returns points."""
    return 100 * ((aT / 100 + aF / 100 - 1) * W / 100 + (1 - aF / 100)) - D


def accounting(tc: dict, lines: list[str]) -> dict:
    """Exp-36: per-cell accounting prediction of CoT - Direct from the written channel.

    aT / aF = P(CoT answer agrees with the last written relation | relation true / false) and W = accuracy of the last
    written relation, all over items with a parsed last claim; D_all = Direct accuracy on all items of the cell.
    Seven-model cells are recomputed from the per-item records behind two_channel.json; the Qwen3-VL-32B / -8B-Thinking
    cue-set cells come from e31_cue_ladder.json (l3 + follow_cot_last, same definitions).
    """
    items = load_items()
    groups = defaultdict(list)
    for r in items:
        if r["ds"] == "syn":
            if r["overlap"]:
                groups[(r["model"], "syn431")].append(r)
        else:
            groups[(r["model"], r["ds"])].append(r)
    cells = {}
    for (m, ds), rs in sorted(groups.items()):
        p = [r for r in rs if r["last_true"] is not None]
        tr = [r for r in p if r["last_true"]]
        fa = [r for r in p if not r["last_true"]]
        W = 100 * len(tr) / len(p)
        aT = 100 * np.mean([r["last_agree"] for r in tr])
        aF = 100 * np.mean([r["last_agree"] for r in fa])
        D = 100 * np.mean([r["d_ok"] for r in rs])
        c = tc["cells"][f"{m}|{ds}"]
        cells[f"{m}|{ds}"] = {"model": m, "ds": ds, "n": len(rs), "n_parsed": len(p), "n_true": len(tr), "n_false": len(fa),
                              "W": W, "aT": aT, "aF": aF, "D_all": D, "y_hat": _acct_terms(W, aT, aF, D),
                              "y": c["y"], "x": c["last_x"], "cot_on_parsed_minus_D_all": c["last_cot_on_parsed"] - c["acc_direct"],
                              "in_fit": m in M7, "source": "per-item records (rigor_e32.load_items)"}
    e31 = json.load(open(B / "e31_cue_ladder.json"))["models"]
    for m in ("qwen3vl_32b", "qwen3vl_8b_think"):
        l3, fo = e31[m]["l3"]["cues"], e31[m]["follow_cot_last"]["all"]
        W, aT, aF, D = l3["last_written_acc"], fo["p_cons_true"]["mean"], fo["p_cons_false"]["mean"], l3["acc_direct"]
        cells[f"{m}|cues"] = {"model": m, "ds": "cues", "n": l3["n"], "n_parsed": fo["n"], "n_true": fo["p_cons_true"]["n"],
                              "n_false": fo["p_cons_false"]["n"], "W": W, "aT": aT, "aF": aF, "D_all": D,
                              "y_hat": _acct_terms(W, aT, aF, D), "y": l3["y"], "x": l3["last_x"],
                              "cot_on_parsed_minus_D_all": l3["last_cot_on_parsed"] - D, "in_fit": False,
                              "source": "e31_cue_ladder.json (Exp-31)"}
    cells = {k: {kk: (round(v, 2) if isinstance(v, float) else v) for kk, v in c.items()} for k, c in cells.items()}
    fitted = [c for c in cells.values() if c["in_fit"]]
    yh = np.array([c["y_hat"] for c in fitted]); y = np.array([c["y"] for c in fitted])
    summ = {"n_points": len(fitted), "pearson_r": round(float(pearsonr(yh, y)[0]), 3),
            "mae": round(float(np.abs(yh - y).mean()), 2), "max_abs_err": round(float(np.abs(yh - y).max()), 2),
            "mean_err_y_minus_yhat": round(float((y - yh).mean()), 2),
            "max_identity_gap_cot_on_parsed": round(float(max(abs(c["y_hat"] - c["cot_on_parsed_minus_D_all"]) for c in cells.values())), 2)}
    for c in sorted(cells.values(), key=lambda c: -abs(c["y"] - c["y_hat"])):
        lines.append(f"ACCT {c['model']:16s} {c['ds']:6s} W {c['W']:5.1f} aT {c['aT']:5.1f} aF {c['aF']:5.1f} D_all {c['D_all']:5.1f} "
                     f"| y_hat {c['y_hat']:+6.2f} y {c['y']:+6.2f} (y - y_hat {c['y'] - c['y_hat']:+5.2f}) x {c['x']:+6.2f} fit {c['in_fit']}")
    lines.append(f"ACCT summary (fitted points) {summ}")
    return {"_doc": accounting.__doc__, "summary_fitted": summ, "cells": cells}


# ============================================================================= figure

def fig_two_channel(tc: dict) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from make_figures_v3 import CANON_LABELS, MODEL_LINE_STYLE
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 7.5, "axes.labelsize": 7.5, "xtick.labelsize": 7.0,
                         "ytick.labelsize": 7.0, "legend.fontsize": 7.0, "axes.spines.top": False, "axes.spines.right": False,
                         "pdf.fonttype": 42})
    mk = {"cues": "o", "twins": "^", "syn431": "s"}
    dsn = {"cues": "Cue-factorial set", "twins": "Occlusion twins", "syn431": "Synthetic Depth (431)"}
    col_of = {m: MODEL_LINE_STYLE[m]["color"] for m in MODEL_LINE_STYLE} | {"qwen3vl_32b": "#9C755F"}
    lab_of = dict(CANON_LABELS) | {"qwen3vl_32b": "Qwen3-VL-32B"}
    cells = tc["cells"]
    fit = tc["fits"]["primary|last_main"]
    acct = json.load(open(B / "two_channel_accounting.json"))
    acc_cells, acc_sum = acct["cells"], acct["summary_fitted"]
    e31 = json.load(open(B / "e31_cue_ladder.json"))["models"]
    pts = [c for c in cells.values() if c["ds"] in mk]
    for m in ("qwen3vl_32b", "qwen3vl_8b_think"):
        pts.append(e31[m]["l3"]["cues"] | {"model": m, "ds": "cues"})
    fig, ax = plt.subplots(figsize=(3.3, 3.2))
    fig.subplots_adjust(left=0.15, right=0.98, top=0.985, bottom=0.28)
    lo, hi = -33, 13
    ax.plot([lo, hi], [lo, hi], ls="--", color="#7A828E", lw=0.8, zorder=1)
    xs = np.array([lo, hi])
    ax.plot(xs, fit["ols_intercept"] + fit["ols_slope"] * xs, color="#2F3640", lw=0.7, zorder=1, alpha=0.6)
    ax.axhline(0, color="#C9CED6", lw=0.5, zorder=0); ax.axvline(0, color="#C9CED6", lw=0.5, zorder=0)
    col_hat = "#1F242B"
    data = []
    for c in pts:
        m, ds = c["model"], c["ds"]
        in_fit = m in M7
        x, y = c["last_x"], c["y"]
        if in_fit:
            ax.scatter([x], [y], marker=mk[ds], s=26, facecolor=col_of[m], edgecolor="white", linewidth=0.5, zorder=3)
            yh = acc_cells[f"{m}|{ds}"]["y_hat"]
            ax.plot([x, x], [y, yh], color=col_hat, lw=0.5, zorder=4, solid_capstyle="butt")
            ax.scatter([x], [yh], marker="o", s=7, facecolor="none", edgecolor=col_hat, linewidth=0.6, zorder=5)
        else:
            ax.scatter([x], [y], marker=mk[ds], s=30, facecolor=col_of[m], edgecolor=col_hat, linewidth=0.9, zorder=3)
        a = acc_cells.get(f"{m}|{ds}", {})
        data.append({"model": m, "label": lab_of[m], "dataset": ds, "x_written_minus_direct": x, "y_cot_minus_direct": y,
                     "x_ci": c["last_x_ci"], "y_ci": c["y_ci"], "n": c["n"], "n_parsed": c["last_n_parsed"], "in_fit": in_fit,
                     "accounting": {k: a[k] for k in ("W", "aT", "aF", "D_all", "y_hat")} if a else None,
                     "source": "two_channel.json" if m in cells and f"{m}|{ds}" in cells else "e31_cue_ladder.json (Exp-31)"})
    for m, ds, off in (("molmo7b", "cues", (4, -9)), ("phi35v", "twins", (8, -20))):  # largest deviations from y = x
        c = cells[f"{m}|{ds}"]
        ax.annotate(CANON_LABELS[m], (c["last_x"], c["y"]), xytext=off, textcoords="offset points", fontsize=7.0,
                    color=col_of[m], ha="left" if m == "molmo7b" else "center")
    for m, txt, xy_txt, ha in (("qwen3vl_32b", "32B (not fitted)", (-6.0, -7.6), "left"),
                               ("qwen3vl_8b_think", "8B-Thinking\n(not fitted)", (-5.0, -0.2), "right")):
        c = e31[m]["l3"]["cues"]
        ax.text(*xy_txt, txt, fontsize=7.0, color=col_of[m], ha=ha, va="center", linespacing=0.9, zorder=6)
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
    ax.set_xlabel("Written relation $-$ Direct (pts)", labelpad=1.5)
    ax.set_ylabel("CoT $-$ Direct (pts)", labelpad=1.5)
    ax.grid(True, color="#E6E9EE", linewidth=0.5, zorder=0)
    ax.text(0.03, 0.975, f"OLS r = {fit['pearson_r']:.2f}, slope {fit['ols_slope']:.2f}\n"
            f"accounting $\\hat{{y}}$: r = {acc_sum['pearson_r']:.2f}",
            transform=ax.transAxes, fontsize=6.8, va="top", ha="left", color="#4A515C", linespacing=1.15)
    order = M7 + ["qwen3vl_8b_think", "qwen3vl_32b"]
    h = [Line2D([], [], marker="o", linestyle="none", markersize=4.5, markerfacecolor=col_of[m],
                markeredgecolor="white", label=lab_of[m]) for m in order]
    hd = [Line2D([], [], marker=mk[d], linestyle="none", markersize=4.5, markerfacecolor="white", markeredgecolor="black", label=dsn[d])
          for d in ("cues", "twins", "syn431")]
    hd.append(Line2D([], [], marker="o", linestyle="none", markersize=2.6, markerfacecolor="none", markeredgecolor=col_hat,
                     markeredgewidth=0.6, label="Accounting $\\hat{y}$"))
    hd.append(Line2D([], [], marker="o", linestyle="none", markersize=4.7, markerfacecolor="#C9CED6", markeredgecolor=col_hat,
                     markeredgewidth=0.9, label="Not fitted"))
    ax.legend(handles=hd, loc="lower right", frameon=False, handletextpad=0.2, labelspacing=0.22, borderaxespad=0.2)
    fig.legend(handles=h, loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.53, 0.005), handletextpad=0.15,
               columnspacing=0.6, labelspacing=0.25, fontsize=6.3)
    out = PAPER / "figures" / "fig_two_channel"
    fig.savefig(f"{out}.pdf")
    fig.savefig(f"{out}.png", dpi=300)
    json.dump({"x": "accuracy of the last written relation (claim_parse v3, items with a parsed claim) - Direct accuracy on the same items (pts)",
               "y": "CoT accuracy - Direct accuracy, all items (pts)", "fit": {k: fit[k] for k in fit if k != "points"},
               "fit_points": "7 instruct models x 3 datasets; Thinking twins and the Exp-31 cue-set points of Qwen3-VL-32B / "
                             "Qwen3-VL-8B-Thinking shown, not in the fit",
               "accounting": {"definition": "y_hat = (aT + aF - 1) W + (1 - aF) - D_all; aT / aF = P(CoT answer agrees with last "
                                            "written relation | relation true / false), W = last written-relation accuracy, all on "
                                            "items with a parsed last claim; D_all = Direct accuracy on all items. Algebraically y_hat "
                                            "= CoT accuracy on parsed items - D_all, so y - y_hat is the contribution of unparsed items.",
                              "summary_fitted": acc_sum, "source": "bundle/two_channel_accounting.json (rigor_e32.py --only acct)"},
               "points": data},
              open(f"{out}.data.json", "w"), indent=1)


# ============================================================================= L1 / L2 tables

def two_by_two(rs: list[dict], rng) -> dict:
    """P(answer agrees with last written relation | relation true / false), cluster bootstrap."""
    rs = [r for r in rs if r["last_true"] is not None]
    cl = np.array([r["cl"] for r in rs])
    t = np.array([r["last_true"] for r in rs], bool)
    a = np.array([r["last_agree"] for r in rs], bool)
    uc, inv = np.unique(cl, return_inverse=True)
    agg = np.zeros((len(uc), 4))
    np.add.at(agg, inv, np.stack([t & a, t, ~t & a, ~t], 1).astype(float))
    st = lambda A: (A[:, 0].sum() / A[:, 1].sum(), A[:, 2].sum() / A[:, 3].sum())
    bs = np.array([st(agg[rng.integers(0, len(uc), len(uc))]) for _ in range(2000)])
    e = st(agg)
    q = lambda v: [round(100 * float(np.quantile(v, .025)), 1), round(100 * float(np.quantile(v, .975)), 1)]
    return {"n": len(rs), "n_true": int(t.sum()), "n_false": int((~t).sum()), "k_true_agree": int((t & a).sum()), "k_false_agree": int((~t & a).sum()),
            "p_agree_true": round(100 * e[0], 1), "p_agree_true_ci": q(bs[:, 0]), "p_agree_false": round(100 * e[1], 1), "p_agree_false_ci": q(bs[:, 1])}


def law1(items: list[dict], lines: list[str]) -> dict:
    rng = np.random.default_rng(33)
    imgswap = json.load(open(B / "imgswap.json"))
    cu = json.load(open(B / "claim_edit_uncond.json"))
    rs_ = json.load(open(B / "relation_sensitivity.json"))
    cb = json.load(open(B / "commit_baseline.json"))
    checkers = {"internvl3_8b", "internvl35_8b", "qwen7b"}
    res = {"_sources": {
        "2x2": "per item, cue-factorial set (e10 <m>_cues.jsonl CoT text + CoT answer); last relational claim by claim_parse v3 over "
               "e2f split_sents; agree = answer implied by the written front object equals the CoT answer; denominators = cue items with a "
               "parsed last claim; twin-pair (image) cluster bootstrap 2,000. Twins / Syn. Depth 431 versions in 'twins' / 'syn431'.",
        "dvis": "imgswap.json full:all dvis (Tab. supp_recheck 'full R, all'): own full rationale, image v -> identity-swapped twin v_cf, "
                "800 twin queries per model; full:wrongR = items whose rationale has a wrong claim.",
        "edit": "claim_edit_uncond.json 'all' (Tab. supp_claim_uncond): Synthetic Depth (500 images) + occlusion twins ('in front' phrasing), "
                "items whose stored rationale has a parsable first claim, all four Direct x CoT cells; edit at the first claim, same-turn continuation.",
        "S_orig": "relation_sensitivity.json: S = P(follows appended gold conclusion) - P(follows appended wrong conclusion) under R_cut, "
                  "original image, CV-Bench 2D left/right items (Fig. 4 value) and depth items (CV-Bench 3D Depth + Synthetic Depth); "
                  "checker = rationale-body checker (InternVL3-8B, InternVL3.5-8B, Qwen2.5-VL-7B), copier = the rest.",
        "commit": "commit_baseline.json / commit_nulls.json (Tab. supp_commit_nulls): D+C- rationales of the 2,850-item pool (Sec. 5 cells, "
                  "e2b trajectories); commit share = rationales with a commit j* / all D+C-; relational share = rule-typed relational j* / committed."},
        "models": {}}
    for m in M7:
        row = {"cues": two_by_two([r for r in items if r["model"] == m and r["ds"] == "cues"], rng),
               "twins": two_by_two([r for r in items if r["model"] == m and r["ds"] == "twins"], rng),
               "syn431": two_by_two([r for r in items if r["model"] == m and r["ds"] == "syn" and r["overlap"]], rng)}
        im = imgswap[m]
        row["dvis_full_all"] = im["full:all"]["dvis"]
        row["dvis_full_wrongR"] = im["full:wrongR"]["dvis"]
        if m in cu:
            row["edit"] = {k: {"n": cu[m][k]["n"], "correct_minus_direct": cu[m][k]["correct-direct"], "neutral_minus_direct": cu[m][k]["neutral-direct"],
                               "acc_direct": cu[m][k]["direct"]["acc"]["mean"], "acc_correct": cu[m][k]["correct"]["acc"]["mean"]} for k in ("all", "syn", "occ")}
        row["type"] = "checker" if m in checkers else "copier"
        row["S_orig_LR"] = rs_[m]["LR"]["S_orig"]
        row["S_orig_depth"] = rs_[m]["depth"]["S_orig"]
        b = cb[m]
        row["commit"] = {"n_DCm": b["n_DCm"], "n_committed": b["n_committed"], "commit_share": round(100 * b["n_committed"] / b["n_DCm"], 1),
                         "relational_share_of_commits": b["commit_relational_share_of_committed"],
                         "relational_share_of_all_DCm": b["commit_relational_share_of_all_DCm"]}
        res["models"][m] = row
        c = row["cues"]
        lines.append(f"LAW1 {m:14s} cue 2x2 n={c['n']} P(agree|true) {c['p_agree_true']} {c['p_agree_true_ci']} (n={c['n_true']}) "
                     f"P(agree|false) {c['p_agree_false']} {c['p_agree_false_ci']} (n={c['n_false']}) | twins {row['twins']['p_agree_true']}/{row['twins']['p_agree_false']} "
                     f"| syn431 {row['syn431']['p_agree_true']}/{row['syn431']['p_agree_false']} | dvis full {row['dvis_full_all']['mean']:+.1f} "
                     f"| edit {('%+.1f / %+.1f' % (row['edit']['all']['correct_minus_direct']['mean'], row['edit']['all']['neutral_minus_direct']['mean'])) if 'edit' in row else '--'} "
                     f"| {row['type']} S_LR {row['S_orig_LR']['mean']:.1f} S_depth {row['S_orig_depth']['mean']:.1f} | commit {row['commit']['n_committed']}/{row['commit']['n_DCm']} "
                     f"rel {row['commit']['relational_share_of_commits']}")
    th = [r for r in items if r["model"] == "qwen3vl_8b_think"]
    res["qwen3vl_8b_think_twins_2x2"] = two_by_two(th, rng)
    res["cue_commit_share_e28_note"] = "cue-set D+C- commit share (Exp-28): Qwen3-VL-8B 247/252 = 98.0%, Molmo-7B-D 270/301 = 89.7%"
    return res


def _pm(x, ci=None, star=True, d=1) -> str:
    s = f"${x:+.{d}f}$"
    if star and ci is not None and (ci[0] > 0 or ci[1] < 0):
        s += "$^{*}$"
    return s


def law1_table(L1: dict) -> str:
    lines = [r"\begin{tabular}{lccccccc}", r"\toprule",
             r" & \multicolumn{2}{c}{Cue set: answer agrees with last written relation (\%)} & Twin swap & \multicolumn{2}{c}{Claim edit $-$ \texttt{Direct} (pts)} & Answer step & \texttt{D+C$-$} commits (\%) \\",
             r"\cmidrule(lr){2-3}\cmidrule(lr){4-4}\cmidrule(lr){5-6}\cmidrule(lr){7-7}\cmidrule(lr){8-8}",
             r"Model & relation true & relation false & image effect, full R & true claim & neutral & type, $S^{\mathrm{L/R}}_\mathrm{orig}$ & with $j^*$ / relational \\", r"\midrule"]
    for m in M7:
        r = L1["models"][m]
        c = r["cues"]
        e = r.get("edit", {}).get("all")
        ed = (f"{_pm(e['correct_minus_direct']['mean'], [e['correct_minus_direct']['lo'], e['correct_minus_direct']['hi']])} & "
              f"{_pm(e['neutral_minus_direct']['mean'], [e['neutral_minus_direct']['lo'], e['neutral_minus_direct']['hi']])}") if e else "-- & --"
        dv = r["dvis_full_all"]
        lines.append(f"{NAMES[m]} & {c['p_agree_true']:.1f} {{\\scriptsize({c['n_true']})}} & {c['p_agree_false']:.1f} {{\\scriptsize({c['n_false']})}} & "
                     f"{_pm(dv['mean'], [dv['lo'], dv['hi']])} & {ed} & {r['type']}, {r['S_orig_LR']['mean']:.1f} & "
                     f"{r['commit']['commit_share']:.1f} / {r['commit']['relational_share_of_commits']:.1f} \\\\")
    lines += [r"\bottomrule",
              r"\multicolumn{8}{p{0.97\linewidth}}{\scriptsize Agreement: all 1,600 cue-factorial queries whose CoT rationale has a parsed last relational claim "
              r"(claim\_parse v3; $n$ in parentheses); conditioning is on the written relation's truth, not on the answer's correctness. "
              r"Image effect: $P(\text{answer}=\text{gold}_v\mid v)-P(\text{answer}=\text{gold}_v\mid v_\mathrm{cf})$ with the model's own full rationale and its identity-swapped twin image, 800 twin queries (\cref{tab:supp_recheck}). "
              r"Claim edit: \texttt{Synthetic Depth} + occlusion twins (``in front'' phrasing), items with a parsable first claim, all four "
              r"\texttt{Direct}$\times$\texttt{CoT} cells (\cref{tab:supp_claim_uncond}); not run for InternVL3.5-8B. "
              r"$S^{\mathrm{L/R}}_\mathrm{orig}$: appended-conclusion check score on CV-Bench 2D left/right items (checker = follows the rationale body; copier = follows the appended conclusion). "
              r"Commits: \texttt{D+C$-$} rationales of the 2,850-item pool with a commit sentence $j^*$, and the relational share of those $j^*$ (\cref{tab:supp_commit_nulls}). "
              r"$^{*}$: 95\% CI excludes zero.} \\",
              r"\end{tabular}"]
    return "\n".join(lines) + "\n"


def law2(lines: list[str], two: dict) -> dict:
    e27 = json.load(open(B / "e27_reframe.json"))["ovl"]["commit_audit"]
    ra = json.load(open(B / "e30_reparse.json"))
    cc = json.load(open(B / "parser_v2.json"))["v3"]["cue_claims"]["cues"]
    ame = json.load(open(B / "cue_ame.json"))["models"]
    nat = json.load(open(B / "heldout_breakdown.json"))["natural_sdt"]
    cvb_unr = json.load(open(B / "e17l.json"))["cvb"]["all"]["per_model_dprime_diff"]
    pc = json.load(open(B / "probe_controls.json"))
    res = {"_sources": {
        "syn431_commit": "e27_reframe.json ovl.commit_audit (Tab. supp_syn431): e20 two-judge audit sample (40 D+C- per model, Sec. 5 cells) "
                         "restricted to the 431 overlapping images; contradicting / judgeable (two-judge consensus CORRECT/INCORRECT); comparator = "
                         "relational sentence of a D+C+ rationale (relcomp).",
        "cue_commit": "Exp-30 full audit (all cue D+C- j* and one relational sentence per D+C+ rationale), consensus after the Exp-32 re-run of "
                      "43 unparsed Opus replies (bundle/e30_reparse.json after_rerun); Qwen3-VL-8B and Molmo-7B-D only.",
        "first_claim": "parser_v2.json v3 cue_claims (Tab. supp_cue_claims): first relational claim names the true front, share of cue queries with a parsed first claim, twin-pair CI.",
        "ame": "cue_ame.json (Tab. supp_cue_ame): CoT - Direct average marginal effect (pts) of occlusion and height (lower is closer), 1,600 cue queries.",
        "natural": "heldout_breakdown.json natural_sdt (Tab. supp_natural_sdt): delta d' CoT - Direct, BLINK Relative Depth (124) and CV-Bench 3D Depth (600; unrounded e17l value).",
        "probe": "probe_controls.json (Tab. supp_probe_cot): layer-20 linear probe for the true relation, balanced accuracy at the written relation token r "
                 "on the cue set (original image), and recall on D+C- wrong claims; layer-0 (lexical) baseline at r."},
        "models": {}}
    for m in M7 + ["qwen3vl_8b_think"]:
        row = {}
        if m in e27["DCm_commit"]:
            k, n, p = e27["DCm_commit"][m]["overlap"]
            k2, n2, p2 = e27["DCp_rel"][m]["overlap"]
            row["syn431_commit"] = {"k": k, "n": n, "pct": p, "wilson": e27["DCm_commit"][m].get("wilson_overlap")}
            row["syn431_comparator"] = {"k": k2, "n": n2, "pct": p2}
        if m in ("qwen3vl_8b", "molmo7b"):
            a, b0 = ra["after_rerun"][f"{m}|D+C-|all"], ra["after_rerun"][f"{m}|D+C+rel|all"]
            row["cue_commit"] = {"k": a["k"], "n": a["n"], "pct": a["pct"], "wilson": a["wilson"]}
            row["cue_comparator"] = {"k": b0["k"], "n": b0["n"], "pct": b0["pct"]}
            row["cue_commit_exp30_before_rerun"] = ra["before"][f"{m}|D+C-|all"]["pct"]
        if m in cc:
            row["first_claim_true_front"] = cc[m]["names_front"] | {"n_parsed": cc[m]["n_parsed"]}
        if m in ame:
            row["ame_occlusion_cot_minus_direct"] = ame[m]["occlusion"]["ame"]["cot-direct"]
            row["ame_height_cot_minus_direct"] = ame[m]["vertical"]["ame"]["cot-direct"]
            row["ame_occlusion_direct"] = ame[m]["occlusion"]["ame"]["direct"]["est"]
        if m in nat:
            row["blink_ddprime"] = {"est": nat[m]["blink_depth"]["diff"], "ci": nat[m]["blink_depth"]["diff_ci"], "n": nat[m]["blink_depth"]["n"]}
            row["cvb3d_ddprime"] = {"est": cvb_unr.get(m, nat[m]["cvb_depth"]["diff"]), "ci": nat[m]["cvb_depth"]["diff_ci"], "n": nat[m]["cvb_depth"]["n"]}
        if m in pc:
            o = pc[m]["orig"]["r"]
            row["probe_r"] = {"bal": o["bal"], "ci": o["bal_ci"], "recall_DCm_wrong": o["recall_DCm_wrong"], "n_DCm_wrong": o["n_DCm_wrong"],
                              "layer0_r_bal": pc[m]["layer0_r"].get("bal") if isinstance(pc[m].get("layer0_r"), dict) else None}
        c2 = two["cells"].get(f"{m}|cues")
        if c2:
            row["cue_written_acc_last"] = c2["last_written_acc"]
            row["cue_direct_acc"] = c2["acc_direct"]
        res["models"][m] = row
        lines.append(f"LAW2 {m:16s} " + json.dumps({k: (v.get('pct', v.get('est', v.get('bal'))) if isinstance(v, dict) else v) for k, v in row.items()}))
    # summary counts
    S = res["models"]
    res["summary"] = {
        "height_ame_ci_below0": sum(S[m]["ame_height_cot_minus_direct"]["ci"][1] < 0 for m in M7),
        "occl_ame_ci_below0": [m for m in M7 if S[m]["ame_occlusion_cot_minus_direct"]["ci"][1] < 0],
        "occl_ame_ci_above0": [m for m in M7 if S[m]["ame_occlusion_cot_minus_direct"]["ci"][0] > 0],
        "blink_neg_point": sum(S[m]["blink_ddprime"]["est"] < 0 for m in S if "blink_ddprime" in S[m]),
        "blink_ci_below0": [m for m in S if "blink_ddprime" in S[m] and S[m]["blink_ddprime"]["ci"][1] < 0],
        "cvb_neg_point": sum(S[m]["cvb3d_ddprime"]["est"] < 0 for m in S if "cvb3d_ddprime" in S[m]),
        "cvb_ci_below0": [m for m in S if "cvb3d_ddprime" in S[m] and S[m]["cvb3d_ddprime"]["ci"][1] < 0],
        "n_natural_models": sum("blink_ddprime" in S[m] for m in S)}
    lines.append(f"LAW2 summary {res['summary']}")
    return res


def law2_table(L2: dict) -> str:
    lines = [r"\begin{tabular}{lccccccccc}", r"\toprule",
             r" & \multicolumn{2}{c}{Commit contradicts image (\%)} & Comparator & First claim & \multicolumn{2}{c}{\texttt{CoT}$-$\texttt{Dir.} AME (pts)} & \multicolumn{2}{c}{\texttt{CoT}$-$\texttt{Dir.} $\Delta d'$} & Probe at $r$ \\",
             r"\cmidrule(lr){2-3}\cmidrule(lr){4-4}\cmidrule(lr){5-5}\cmidrule(lr){6-7}\cmidrule(lr){8-9}\cmidrule(lr){10-10}",
             r"Model & Syn.\ Depth 431 & cue set & 431 / cue & true front (\%) & occlusion & height & BLINK & CVB 3D & bal.\ acc. \\", r"\midrule"]
    for m in M7 + ["qwen3vl_8b_think"]:
        r = L2["models"][m]
        f = lambda k: f"{r[k]['k']}/{r[k]['n']} = {r[k]['pct']:.1f}" if k in r else "--"
        comp = (f"{r['syn431_comparator']['pct']:.1f}" if "syn431_comparator" in r else "--") + " / " + (f"{r['cue_comparator']['pct']:.1f}" if "cue_comparator" in r else "--")
        fc = f"{r['first_claim_true_front']['pct']:.1f}" if "first_claim_true_front" in r else "--"
        ao = _pm(r["ame_occlusion_cot_minus_direct"]["est"], r["ame_occlusion_cot_minus_direct"]["ci"]) if "ame_occlusion_cot_minus_direct" in r else "--"
        ah = _pm(r["ame_height_cot_minus_direct"]["est"], r["ame_height_cot_minus_direct"]["ci"]) if "ame_height_cot_minus_direct" in r else "--"
        bl = _pm(r["blink_ddprime"]["est"], r["blink_ddprime"]["ci"], d=2) if "blink_ddprime" in r else "--"
        cv = _pm(r["cvb3d_ddprime"]["est"], r["cvb3d_ddprime"]["ci"], d=2) if "cvb3d_ddprime" in r else "--"
        pr = f"{r['probe_r']['bal']:.2f}" if "probe_r" in r else "--"
        lines.append(f"{NAMES[m]} & {f('syn431_commit')} & {f('cue_commit')} & {comp} & {fc} & {ao} & {ah} & {bl} & {cv} & {pr} \\\\")
    lines += [r"\bottomrule",
              r"\multicolumn{10}{p{0.97\linewidth}}{\scriptsize Commit: consensus of two judges (Claude Opus 5.5, GPT-6-Astra), contradicting / judgeable "
              r"commit sentences $j^*$ of \texttt{D+C$-$} rationales. Syn.\ Depth 431: audit sample of \cref{tab:supp_commit_judge} restricted to the 431 visibly overlapping images "
              r"(\cref{tab:supp_syn431}). Cue set: all cue-factorial \texttt{D+C$-$} commits (\cref{tab:supp_e30}), with the 43 unparsed Opus replies re-run "
              r"(Molmo-7B-D 68/179 before the re-run). Comparator: relational sentences of \texttt{D+C+} rationales, same judges, share contradicting. "
              r"First claim: share of cue queries with a parsed first relational claim whose claimed front is the true front (\cref{tab:supp_cue_claims}). "
              r"AME: \cref{tab:supp_cue_ame}. $\Delta d'$: BLINK Relative Depth ($n=124$) and CV-Bench 3D Depth ($n=600$), \cref{tab:supp_natural_sdt}. "
              r"Probe: layer-20 balanced accuracy for the true relation at the written relation token, cue set (\cref{tab:supp_probe_cot}). "
              r"$^{*}$: 95\% CI excludes zero; --: not run.} \\",
              r"\end{tabular}"]
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="two")
    a = ap.parse_args()
    only = a.only.split(",")
    if "law1" in only or "law2" in only:
        lines: list[str] = []
        if "law1" in only:
            L1 = law1(load_items(), lines)
            json.dump(L1, open(B / "law1.json", "w"), indent=1)
            (PAPER / "tables" / "supp_law1.tex").write_text(law1_table(L1))
        if "law2" in only:
            L2 = law2(lines, json.load(open(B / "two_channel.json")))
            json.dump(L2, open(B / "law2.json", "w"), indent=1)
            (PAPER / "tables" / "supp_law2.tex").write_text(law2_table(L2))
        (B / "rigor_e32_laws.txt").write_text("\n".join(lines) + "\n")
        print("\n".join(lines))
    if "two" in only:
        lines = []
        res = two_channel(lines)
        json.dump(res, open(B / "two_channel.json", "w"), indent=1)
        (B / "rigor_two_channel.txt").write_text("\n".join(lines) + "\n")
        print("\n".join(lines))
    if "acct" in only:
        lines = []
        acc = accounting(json.load(open(B / "two_channel.json")), lines)
        json.dump(acc, open(B / "two_channel_accounting.json", "w"), indent=1)
        (B / "rigor_e36_accounting.txt").write_text("\n".join(lines) + "\n")
        print("\n".join(lines))
    if "fig" in only:
        fig_two_channel(json.load(open(B / "two_channel.json")))


if __name__ == "__main__":
    main()
