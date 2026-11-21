"""Two-channel accounting on the synthetic sets (Exp-32 / Exp-36; CPU only, no new model runs).

two     Two-channel test. Per model x dataset: Direct / CoT accuracy, accuracy of the written relation (first and
        last relational claim, claim_parse v3, over items with a parsed claim) and parse coverage, share of CoT answers
        agreeing with the last written relation, D+C- / D-C+ / churn / net. Then across model x dataset points:
        x = written-relation acc - Direct acc, y = CoT acc - Direct acc (Pearson, Spearman, OLS slope with point and
        model-cluster bootstrap CIs, residuals from y = x).
        Datasets: cue-factorial set (e10 <m>_cues.jsonl, 1,600 queries, CoT text = the CoT run behind the GLM);
        occlusion twins (e2d front + behind, 800 queries; the twin d'_pair run); Synthetic Depth (e2b legacy
        rationales = Sec. 5 rationales, Tab. 1 readout from srww.records.load_main; 431 overlapping images primary,
        all 500 as a variant). No relation parser exists for CV-Bench 2D Relation / Synthetic Relative Position.
acct    Accounting prediction of CoT - Direct per cell from the written channel:
        y_hat = (aT + aF - 1) W + (1 - aF) - D_all (see accounting()).
-> bundle/two_channel.json, bundle/two_channel.txt, bundle/two_channel_accounting.{json,txt}

Also imported by the natural-image and uniform-protocol two-channel scripts (per-item records, cell statistics).
Usage: python -m experiments.two_channel.two_channel_fit [--only two,acct]
"""
from __future__ import annotations
from srww import paths as _vtp  # noqa: E402

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import pearsonr, spearmanr

from srww import records as BB
from srww.records import pe, rj
from srww.claim_parse import first_claim_v3, split_sents

O = Path(f"{_vtp.OUTPUTS}")
L = Path(f"{_vtp.LEGACY_RESULTS}")
B = Path("outputs/bundle")
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
                              "in_fit": m in M7, "source": "per-item records (two_channel_fit.load_items)"}
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


# ============================================================================= helpers

def category(r: dict, thr: float = 0.15, min_px: int = 100) -> str:
    if r["contact"] < thr:
        return "touch_or_gap"
    return "overlap" if min(r["px"]) >= min_px else "near_total"


def implied_meta(front_claimed: str, mt: dict, phrasing: str) -> str:
    yes = (front_claimed == mt["x_name"]) if phrasing == "front" else (front_claimed != mt["x_name"])
    return "A" if yes else "B"


def norm_e2d(r: dict, phrasing: str) -> dict:
    """e2d / e2d_think record -> e10 cue-record layout (item_id with _behind, base_id, phrasing, question)."""
    mt = r["meta"]
    rel = "in front of" if phrasing == "front" else "behind"
    return r | {"base_id": r["item_id"], "item_id": r["item_id"] + ("_behind" if phrasing == "behind" else ""),
                "phrasing": phrasing, "question": f"Is the {mt['x_name']} {rel} the {mt['y_name']}?\n(A) yes\n(B) no"}


def load_arms(m: str) -> dict:
    """{arm: {item_id: record}} for direct / cot (/ desc) on the cue set."""
    by: dict = defaultdict(dict)
    if m == "qwen3vl_8b_think":
        for fn, ph in (("qwen3vl_8b_think.jsonl", "front"), ("qwen3vl_8b_think_behind.jsonl", "behind")):
            for r in rj(O / "e31" / fn):
                r = norm_e2d(r, ph)
                by[r["condition"]][r["item_id"]] = r
        dp = O / "e31" / "qwen3vl_8b_think_cues_desc.jsonl"
    else:
        for r in rj(O / "e10" / f"{m}_cues.jsonl"):
            if r["condition"] in ("direct", "cot"):
                by[r["condition"]][r["item_id"]] = r
        dp = O / "e28" / f"{m}_cues_desc.jsonl"
    for r in rj(dp):
        by[r["condition"]][r["item_id"]] = r
    return dict(by)


def main() -> None:
    ap = argparse.ArgumentParser(description="Two-channel fit across model x dataset cells (two) and the accounting "
                                             "prediction of CoT - Direct (acct).")
    ap.add_argument("--only", default="two", help="comma-separated: two, acct")
    a = ap.parse_args()
    only = a.only.split(",")
    if "two" in only:
        lines = []
        res = two_channel(lines)
        json.dump(res, open(B / "two_channel.json", "w"), indent=1)
        (B / "two_channel.txt").write_text("\n".join(lines) + "\n")
        print("\n".join(lines))
    if "acct" in only:
        lines = []
        acc = accounting(json.load(open(B / "two_channel.json")), lines)
        json.dump(acc, open(B / "two_channel_accounting.json", "w"), indent=1)
        (B / "two_channel_accounting.txt").write_text("\n".join(lines) + "\n")
        print("\n".join(lines))


if __name__ == "__main__":
    main()
