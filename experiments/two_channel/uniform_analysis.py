"""Exp-45 analysis (called by `python -m experiments.two_channel.uniform_two_channel analyze`).

Per point (model x set): coverage (share of items with a consensus written relation after conclusion removal), written
accuracy W, Direct on all / covered items, CoT on all / covered items, aT / aF = P(CoT answer agrees with the relation |
relation true / false), x = W - Direct(covered), y = CoT - Direct (all), y_cov = CoT(covered) - Direct(covered),
answer-step term A = (1 - aF)(1 - W) - (1 - aT) W (= y_cov - x when every CoT answer is one of the two options),
D+C- / D-C+, item-cluster bootstrap CIs.
Fit over the 21 open synthetic points (7 models x cue set / twins / Syn. Depth 431): OLS slope / intercept / r with
point, model-cluster, item and item+model bootstrap CIs. Out of sample: natural (13 Exp-35 points) and closed
(13 Exp-38 points) from the conclusion-removed synthetic fit.
"""
from __future__ import annotations

import json
import os
from collections import Counter, defaultdict

import numpy as np
from scipy.stats import pearsonr, spearmanr

from experiments.two_channel.uniform_two_channel import (B, CLOSED, E45, JUDGES, M7, O, hkey, rj, val_sample)

DEBUG = bool(os.environ.get("E45_DEBUG"))
NB_CELL, NB_PT = (100, 500) if DEBUG else (2000, 10000)
SYN_DS = ("cues", "twins", "syn431")
COLS = ["n", "d", "c", "p", "tp", "dp", "cp", "atp", "fp", "afp"]


def implied(front_claimed: str, x: str, phrasing: str) -> str:
    yes = (front_claimed == x) if phrasing == "front" else (front_claimed != x)
    return "A" if yes else "B"


# ============================================================================= labels and records

def load_labels(suffix: str = "", extra: tuple = ()) -> dict:
    out = {}
    for j in JUDGES:
        d = {}
        for sfx in (suffix,) + tuple(extra):
            for r in rj(E45 / f"judge_{j}{sfx}.jsonl"):
                if r.get("label") is not None:
                    d.setdefault(r["hkey"], r["label"])
        out[j] = d
    return out


def shape_records(rows: list[dict], lab: dict, spec: str = "consensus") -> list[dict]:
    """Generic per-item records: has (relation), true, agree, d_ok, c_ok, cl (cluster)."""
    out = []
    for r in rows:
        if not r["body"]:
            a = b = "none"
            st = "empty"
        elif not r["has_kw"]:
            a = b = "none"
            st = "prefilter_none"
        else:
            a, b = lab[JUDGES[0]].get(r["hkey"]), lab[JUDGES[1]].get(r["hkey"])
            if DEBUG and (a is None or b is None):  # code test only: rule parse stands in for missing judge labels
                a = b = r["v3_trunc"] or "none"
            st = None
        if spec == "consensus":
            if st is None:
                st = ("missing" if a is None or b is None else "agree" if a == b and a in (r["x"], r["y"])
                      else "agree_none" if a == b else "disagree")
            claim = a if st == "agree" else None
        else:
            v = a if spec == JUDGES[0] else b
            claim = v if v in (r["x"], r["y"]) else None
            st = st or ("missing" if v is None else "ok")
        out.append({"model": r["model"], "ds": r["ds"], "group": r["group"], "cl": f"{r['ds']}|{r['cl']}", "item": r["item"],
                    "d_ok": bool(r["d_ok"]), "c_ok": bool(r["c_ok"]), "nr_ok": r.get("nr_ok"),
                    "has": claim is not None, "true": claim == r["front"] if claim else None,
                    "agree": implied(claim, r["x"], r["phrasing"]) == r["c_ans"] if claim else None,
                    "claim": claim, "status": st, "front": r["front"], "x": r["x"], "y": r["y"],
                    "v3_full": r["v3_full"], "v3_trunc": r["v3_trunc"], "hkey": r["hkey"], "key38": r.get("key38"),
                    "labA": a, "labB": b})
    return out


def v3_records(rows: list[dict], which: str) -> list[dict]:
    out = []
    for r in rows:
        claim = r[which]
        out.append({"model": r["model"], "ds": r["ds"], "group": r["group"], "cl": f"{r['ds']}|{r['cl']}",
                    "d_ok": bool(r["d_ok"]), "c_ok": bool(r["c_ok"]), "has": claim is not None,
                    "true": claim == r["front"] if claim else None,
                    "agree": implied(claim, r["x"], r["phrasing"]) == r["c_ans"] if claim else None})
    return out


def natural_records() -> list[dict]:
    from experiments.two_channel import natural_two_channel as N
    items = N.rj(N.E35 / "items.jsonl")
    N.attach(items, N.labels(), "consensus")
    out = []
    for r in items:
        c = r["claim"]
        out.append({"model": r["model"], "ds": r["ds"], "group": "natural", "cl": f"{r['ds']}|{r['cl']}", "item": r["item"],
                    "d_ok": r["d_ans"] == r["gold"], "c_ok": r["c_ans"] == r["gold"], "has": c is not None,
                    "true": c == r["gold"] if c else None, "agree": c == r["c_ans"] if c else None,
                    "status": r["lab_status"]})
    return out


# ============================================================================= cell statistics

def agg(rs: list[dict], okkey: str = "d_ok") -> np.ndarray:
    cl = [r["cl"] for r in rs]
    uc = {c: i for i, c in enumerate(sorted(set(cl)))}
    inv = np.array([uc[c] for c in cl])
    d = np.array([bool(r[okkey]) for r in rs], float)
    c = np.array([r["c_ok"] for r in rs], float)
    p = np.array([r["has"] for r in rs], float)
    t = np.array([bool(r["true"]) for r in rs], float) * p
    a = np.array([bool(r["agree"]) for r in rs], float) * p
    f = p - t
    M = np.zeros((len(uc), len(COLS)))
    np.add.at(M, inv, np.stack([np.ones(len(rs)), d, c, p, t, d * p, c * p, a * t, f, a * f], 1))
    return M


def from_sums(s: np.ndarray) -> dict:
    n, d, c, p, tp, dp, cp, atp, fp, afp = s
    o = {"coverage": 100 * p / n, "acc_direct": 100 * d / n, "acc_cot": 100 * c / n, "y": 100 * (c - d) / n}
    if p > 0:
        W = tp / p
        aT = atp / tp if tp else np.nan
        aF = afp / fp if fp else np.nan
        o |= {"W": 100 * W, "direct_cov": 100 * dp / p, "cot_cov": 100 * cp / p, "x": 100 * (W - dp / p),
              "y_cov": 100 * (cp - dp) / p, "aT": 100 * aT, "aF": 100 * aF}
        aT0 = 1.0 if np.isnan(aT) else aT
        aF0 = 1.0 if np.isnan(aF) else aF
        o["A"] = 100 * ((1 - aF0) * (1 - W) - (1 - aT0) * W)
    return o


BOOT_KEYS = ("coverage", "W", "direct_cov", "x", "y", "y_cov", "aT", "aF", "A")


def cell(rs: list[dict], rng, okkey: str = "d_ok", nb: int = NB_CELL) -> dict:
    M = agg(rs, okkey)
    out = from_sums(M.sum(0))
    out |= {"n": len(rs), "n_cov": int(sum(r["has"] for r in rs)), "n_clusters": len(M),
            "n_true": int(sum(r["has"] and r["true"] for r in rs)), "n_false": int(sum(r["has"] and not r["true"] for r in rs)),
            "DpCm": int(sum(r[okkey] and not r["c_ok"] for r in rs)), "DmCp": int(sum(not r[okkey] and r["c_ok"] for r in rs))}
    draws = {k: [] for k in BOOT_KEYS}
    for _ in range(nb):
        s = M[rng.integers(0, len(M), len(M))].sum(0)
        o = from_sums(s)
        for k in BOOT_KEYS:
            draws[k].append(o.get(k, np.nan))
    for k in BOOT_KEYS:
        v = np.array(draws[k], float)
        if k in out and np.isfinite(v).sum() > 10:
            out[f"{k}_ci"] = [float(np.nanquantile(v, .025)), float(np.nanquantile(v, .975))]
    out["_draws"] = {k: np.array(draws[k], float) for k in ("x", "y", "y_cov")}
    return out


def strip(c: dict) -> dict:
    return {k: (round(v, 3) if isinstance(v, float) else [round(u, 3) for u in v] if isinstance(v, list) else v)
            for k, v in c.items() if not k.startswith("_")}


# ============================================================================= fits

def ols(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    sl, ic = np.polyfit(x, y, 1)
    return float(sl), float(ic)


def fit(pts: list[dict], xk: str = "x", yk: str = "y", seed: int = 45) -> dict:
    rng = np.random.default_rng(seed)
    x = np.array([p[xk] for p in pts]); y = np.array([p[yk] for p in pts])
    n = len(x)
    sl, ic = ols(x, y)
    res = {"n_points": n, "slope": sl, "intercept": ic, "pearson_r": float(pearsonr(x, y)[0]), "spearman": float(spearmanr(x, y)[0]),
           "mean_abs_y_minus_x": float(np.abs(y - x).mean()), "mean_y_minus_x": float((y - x).mean()),
           "sign_agree_x_y": int(np.sum(np.sign(x) == np.sign(y)))}
    q = lambda v: [float(np.quantile(v, .025)), float(np.quantile(v, .975))]
    # point bootstrap
    bp = []
    for _ in range(NB_PT):
        i = rng.integers(0, n, n)
        if np.ptp(x[i]) > 0:
            s_, i_ = ols(x[i], y[i]); bp.append((s_, i_, pearsonr(x[i], y[i])[0] if np.ptp(y[i]) > 0 else np.nan))
    bp = np.array(bp)
    models = sorted({p["model"] for p in pts})
    mi = {m: np.array([k for k, p in enumerate(pts) if p["model"] == m]) for m in models}
    bm = []
    for _ in range(NB_PT):
        i = np.concatenate([mi[models[j]] for j in rng.integers(0, len(models), len(models))])
        if np.ptp(x[i]) > 0:
            s_, i_ = ols(x[i], y[i]); bm.append((s_, i_, pearsonr(x[i], y[i])[0]))
    bm = np.array(bm)
    for nm, bb in (("point", bp), ("model", bm)):
        res[f"slope_ci_{nm}"] = q(bb[:, 0]); res[f"intercept_ci_{nm}"] = q(bb[:, 1]); res[f"r_ci_{nm}"] = q(bb[~np.isnan(bb[:, 2]), 2])
    # item-level (cell draws) and item + model-cluster
    if all("_draws" in p for p in pts):
        dx = np.stack([p["_draws"]["x" if xk == "x" else xk] for p in pts], 1)
        dy = np.stack([p["_draws"][yk] for p in pts], 1)
        nb = dx.shape[0]
        bi, bim = [], []
        for b in range(nb):
            bi.append(ols(dx[b], dy[b]))
            i = np.concatenate([mi[models[j]] for j in rng.integers(0, len(models), len(models))])
            if np.ptp(dx[b, i]) > 0:
                bim.append(ols(dx[b, i], dy[b, i]))
        bi, bim = np.array(bi), np.array(bim)
        res["slope_ci_item"] = q(bi[:, 0]); res["intercept_ci_item"] = q(bi[:, 1])
        res["slope_ci_item_model"] = q(bim[:, 0]); res["intercept_ci_item_model"] = q(bim[:, 1])
        res["p_intercept_ge0_item_model"] = float(np.mean(bim[:, 1] >= 0))
    res["p_intercept_ge0_model"] = float(np.mean(bm[:, 1] >= 0))
    res["p_intercept_ge0_point"] = float(np.mean(bp[:, 1] >= 0))
    keep = [k for k, p in enumerate(pts) if p["model"] not in ("molmo7b", "phi35v")]
    if len(keep) > 3:
        s2, i2 = ols(x[keep], y[keep])
        res["without_molmo_phi"] = {"n": len(keep), "slope": s2, "intercept": i2, "r": float(pearsonr(x[keep], y[keep])[0])}
    res["loo_model"] = {m: dict(zip(("slope", "intercept"), ols(x[[k for k, p in enumerate(pts) if p["model"] != m]],
                                                                 y[[k for k, p in enumerate(pts) if p["model"] != m]]))) for m in models}
    return res


def oos(obs: list[dict], sl: float, ic: float, mean_y: float) -> dict:
    x = np.array([o["x"] for o in obs]); y = np.array([o["y"] for o in obs])
    yh = ic + sl * x
    tie = np.abs(y) < 1.0
    return {"n_points": len(x), "mae_fit": float(np.abs(y - yh).mean()), "bias_fit": float((y - yh).mean()),
            "mae_y_eq_x": float(np.abs(y - x).mean()), "bias_y_eq_x": float((y - x).mean()),
            "mae_mean_syn_y": float(np.abs(y - mean_y).mean()), "mae_zero": float(np.abs(y).mean()),
            "sign_agree_pred": int(np.sum(np.sign(yh) == np.sign(y))), "sign_agree_x": int(np.sum(np.sign(x) == np.sign(y))),
            "n_abs_y_ge1": int((~tie).sum()), "sign_agree_pred_abs_y_ge1": int(np.sum((np.sign(yh) == np.sign(y))[~tie])),
            "sign_agree_x_abs_y_ge1": int(np.sum((np.sign(x) == np.sign(y))[~tie])),
            "r_pred_obs": float(pearsonr(yh, y)[0]) if np.ptp(x) > 0 and np.ptp(y) > 0 else float("nan")}


def oos_boot(obs: list[dict], syn: list[dict], rng, nb: int = NB_CELL) -> dict:
    """Synthetic points resampled (point bootstrap) with their item draws; OOS points with their own item draws."""
    sx = np.stack([p["_draws"]["x"] for p in syn], 1); sy = np.stack([p["_draws"]["y"] for p in syn], 1)
    ox = np.stack([p["_draws"]["x"] for p in obs], 1); oy = np.stack([p["_draws"]["y"] for p in obs], 1)
    K = {k: [] for k in ("mae_fit", "bias_fit", "mae_y_eq_x", "mae_mean_syn_y", "fit_minus_y_eq_x", "fit_minus_mean")}
    n = sx.shape[1]
    for b in range(nb):
        i = rng.integers(0, n, n)
        if np.ptp(sx[b, i]) == 0:
            continue
        sl, ic = ols(sx[b, i], sy[b, i])
        o = oos([{"x": a, "y": c} for a, c in zip(ox[b], oy[b])], sl, ic, float(sy[b, i].mean()))
        for k in ("mae_fit", "bias_fit", "mae_y_eq_x", "mae_mean_syn_y"):
            K[k].append(o[k])
        K["fit_minus_y_eq_x"].append(o["mae_fit"] - o["mae_y_eq_x"]); K["fit_minus_mean"].append(o["mae_fit"] - o["mae_mean_syn_y"])
    return {k: [float(np.quantile(v, .025)), float(np.quantile(v, .975))] for k, v in K.items()}


def kappa(a: list, b: list) -> float:
    cats = sorted(set(a) | set(b))
    po = np.mean([u == v for u, v in zip(a, b)])
    pe_ = sum((a.count(c) / len(a)) * (b.count(c) / len(b)) for c in cats)
    return float((po - pe_) / (1 - pe_)) if pe_ < 1 else 1.0


def rel(lab, r) -> str:
    return "x" if lab == r["x"] else "y" if lab == r["y"] else (lab or "missing")


# ============================================================================= main analysis

def analyze() -> None:
    rows = rj(E45 / "items.jsonl")
    lab = load_labels()
    lines: list[str] = []
    res: dict = {"_doc": __doc__}
    rng = np.random.default_rng(45)
    recs = shape_records(rows, lab)
    nat = natural_records()
    groups = defaultdict(list)
    for r in recs + nat:
        groups[(r["group"], r["model"], r["ds"])].append(r)
    # ---- label status
    st = Counter((r["group"], r["status"]) for r in recs)
    res["label_status"] = {f"{g}|{s}": n for (g, s), n in sorted(st.items())}
    lines.append(f"LABEL STATUS (shapes): {res['label_status']}")
    # ---- cells
    cells = {}
    for (g, m, ds), rs in sorted(groups.items()):
        if g == "natural" and m == "phi35v" and ds == "blink":
            continue
        c = cell(rs, rng)
        c |= {"group": g, "model": m, "ds": ds, "status": dict(Counter(r.get("status") for r in rs))}
        if g in ("open", "closed"):
            raw = [r for r in rows if r["group"] == g and r["model"] == m and r["ds"] == ds]
            c["cov_v3_full"] = 100 * np.mean([r["v3_full"] is not None for r in raw])
            c["cov_v3_trunc"] = 100 * np.mean([r["v3_trunc"] is not None for r in raw])
            c["empty_after_trunc"] = 100 * np.mean([not r["body"] for r in raw])
        if g == "closed" and any(r["nr_ok"] is not None for r in rs):
            rr = [r for r in rs if r["nr_ok"] is not None]
            cn = cell(rr, np.random.default_rng(1), okkey="nr_ok", nb=200)
            c["direct_nr"] = {k: cn.get(k) for k in ("n", "acc_direct", "direct_cov", "x", "y", "A")}
        cells[f"{g}|{m}|{ds}"] = c
    # ---- before/after coverage for closed: Exp-38 consensus (answer line removed, conclusion kept)
    J38 = {}
    for j in JUDGES:
        d = {}
        for r in rj(O / "e38" / f"judge_{j}.jsonl"):
            if r.get("label") is not None or r["key"] not in d:
                d[r["key"]] = r.get("label")
        J38[j] = d
    for k, c in cells.items():
        if c["group"] != "closed":
            continue
        rs = groups[tuple(k.split("|"))]
        before = []
        for r in rs:
            a, b = J38[JUDGES[0]].get(r["key38"]), J38[JUDGES[1]].get(r["key38"])
            before.append(a is not None and a == b and a in (r["x"], r["y"]))
        before = np.array(before)
        after = np.array([r["has"] for r in rs])
        c["cov_e38"] = 100 * before.mean()
        c["lost_all_relations"] = 100 * float(np.mean(before & ~after))
        c["lost_all_relations_k"] = int((before & ~after).sum())
        c["gained"] = int((~before & after).sum())
        same = [r for r, bb, aa in zip(rs, before, after) if bb and aa]
        c["same_label_as_e38"] = 100 * float(np.mean([J38[JUDGES[0]][r["key38"]] == r["claim"] for r in same])) if same else None
    for k, c in cells.items():
        if c["group"] == "open":
            rs = groups[tuple(k.split("|"))]
            v3b = np.array([r["v3_full"] is not None for r in rs]); aft = np.array([r["has"] for r in rs])
            c["lost_vs_v3_full"] = 100 * float(np.mean(v3b & ~aft))
    # ---- text lines per cell
    for k, c in cells.items():
        lines.append(
            f"CELL {c['group']:7s} {c['model']:20s} {c['ds']:8s} n={c['n']:4d} cov {c['coverage']:5.1f} | W {c.get('W', np.nan):5.1f} "
            f"Dir {c['acc_direct']:5.1f} Dir|cov {c.get('direct_cov', np.nan):5.1f} CoT {c['acc_cot']:5.1f} CoT|cov {c.get('cot_cov', np.nan):5.1f} "
            f"aT/aF {c.get('aT', np.nan):5.1f}/{c.get('aF', np.nan):5.1f} (nT {c['n_true']} nF {c['n_false']}) | x {c.get('x', np.nan):+6.2f} "
            f"{[round(v, 1) for v in c.get('x_ci', [])]} y {c['y']:+6.2f} y_cov {c.get('y_cov', np.nan):+6.2f} A {c.get('A', np.nan):+6.2f} "
            f"{[round(v, 1) for v in c.get('A_ci', [])]} | D+C- {c['DpCm']} D-C+ {c['DmCp']}"
            + (f" | cov v3 full {c['cov_v3_full']:.1f} v3 trunc {c['cov_v3_trunc']:.1f} empty {c['empty_after_trunc']:.1f}" if "cov_v3_full" in c else "")
            + (f" | cov Exp-38 {c['cov_e38']:.1f} lost {c['lost_all_relations']:.1f}% ({c['lost_all_relations_k']}) same label {c['same_label_as_e38']}" if "cov_e38" in c else ""))
    # ---- synthetic fit
    syn = [cells[f"open|{m}|{ds}"] for m in M7 for ds in SYN_DS]
    syn23 = syn + [cells["open|qwen3vl_32b|cues"], cells["open|qwen3vl_8b_think|cues"]]
    F = {"uniform_21": fit(syn), "uniform_21_ycov": fit(syn, "x", "y_cov"), "uniform_23": fit(syn23)}
    for spec in JUDGES:
        rj_ = shape_records([r for r in rows if r["group"] == "open"], lab, spec)
        gj = defaultdict(list)
        for r in rj_:
            gj[(r["model"], r["ds"])].append(r)
        pts = [cell(gj[(m, ds)], np.random.default_rng(2), nb=10) | {"model": m} for m in M7 for ds in SYN_DS]
        F[f"judge_{spec}_21"] = {k: v for k, v in fit(pts).items() if not k.startswith(("slope_ci_item", "intercept_ci_item", "p_intercept_ge0_item"))}
        res.setdefault("judge_alone_cells", {})[spec] = {f"{p['model']}|{ds}": strip(p) for p, ds in zip(pts, [d for _ in M7 for d in SYN_DS])}
    # reproduction: Exp-32 (v3, conclusion kept) and Exp-35 check (v3, conclusion removed)
    for which, nm in (("v3_full", "exp32_v3_conclusion_kept"), ("v3_trunc", "exp35_v3_conclusion_removed")):
        vr = v3_records([r for r in rows if r["group"] == "open"], which)
        gv = defaultdict(list)
        for r in vr:
            gv[(r["model"], r["ds"])].append(r)
        pts = [cell(gv[(m, ds)], np.random.default_rng(3), nb=10) | {"model": m, "ds": ds} for m in M7 for ds in SYN_DS]
        F[nm] = {k: v for k, v in fit(pts).items() if not k.startswith(("slope_ci_item", "intercept_ci_item", "p_intercept_ge0_item"))}
        res.setdefault("reference_cells", {})[nm] = {f"{p['model']}|{p['ds']}": {k: round(p[k], 2) for k in ("x", "y", "coverage")} for p in pts}
    res["fits"] = {k: {kk: vv for kk, vv in v.items()} for k, v in F.items()}
    for k, f in F.items():
        lines.append(f"FIT {k:34s} n={f['n_points']} slope {f['slope']:.3f} pt {np.round(f['slope_ci_point'], 3).tolist()} model {np.round(f['slope_ci_model'], 3).tolist()}"
                     + (f" item+model {np.round(f['slope_ci_item_model'], 3).tolist()}" if "slope_ci_item_model" in f else "")
                     + f" | intercept {f['intercept']:+.2f} pt {np.round(f['intercept_ci_point'], 2).tolist()} model {np.round(f['intercept_ci_model'], 2).tolist()}"
                     + (f" item {np.round(f['intercept_ci_item'], 2).tolist()} item+model {np.round(f['intercept_ci_item_model'], 2).tolist()}" if "intercept_ci_item_model" in f else "")
                     + f" | r {f['pearson_r']:.3f} pt {np.round(f['r_ci_point'], 3).tolist()} model {np.round(f['r_ci_model'], 3).tolist()} | mean|y-x| {f['mean_abs_y_minus_x']:.2f} "
                     f"mean(y-x) {f['mean_y_minus_x']:+.2f} sign {f['sign_agree_x_y']}/{f['n_points']} | P(int>=0) model {f['p_intercept_ge0_model']:.3f}"
                     + (f" | w/o Molmo+Phi {json.dumps({kk: round(vv, 3) for kk, vv in f['without_molmo_phi'].items()})}" if "without_molmo_phi" in f else ""))
    # ---- answer-step term per model (pooled synthetic cells) and per closed / natural model
    ans = {}
    for g, dsl, models in (("open", SYN_DS, M7), ("open", ("cues",), ["qwen3vl_32b", "qwen3vl_8b_think"]),
                           ("natural", ("cvb3d", "blink"), M7), ("closed", ("e9h", "e9x", "e9c_cues"), CLOSED)):
        for m in models:
            rs = [r for ds in dsl for r in groups.get((g, m, ds), []) if not (g == "natural" and m == "phi35v" and ds == "blink")]
            if not rs:
                continue
            c = cell(rs, np.random.default_rng(4), nb=NB_CELL)
            ans[f"{g}|{m}"] = {k: c.get(k) for k in ("n", "n_cov", "coverage", "W", "aT", "aF", "A", "A_ci", "x", "y", "y_cov", "n_true", "n_false")}
            a = ans[f"{g}|{m}"]
            lines.append(f"ANSWER-STEP {g:7s} {m:20s} sets {','.join(dsl)} n {a['n']} cov {a['coverage']:.1f} W {a['W']:.1f} aT {a['aT']:.1f} aF {a['aF']:.1f} "
                         f"A {a['A']:+.2f} [{a['A_ci'][0]:+.2f}, {a['A_ci'][1]:+.2f}] (y_cov - x {a['y_cov'] - a['x']:+.2f})")
    res["answer_step_pooled"] = ans
    # ---- out of sample
    f21 = F["uniform_21"]
    mean_y = float(np.mean([p["y"] for p in syn]))
    nat_pts = [c for k, c in cells.items() if c["group"] == "natural" and c["model"] in M7 and c.get("n_cov", 0) >= 20]
    clo_pts = [c for k, c in cells.items() if c["group"] == "closed" and c.get("n_cov", 0) >= 20]
    O_ = {}
    for nm, pts in (("natural", nat_pts), ("closed", clo_pts), ("natural+closed", nat_pts + clo_pts)):
        o = oos(pts, f21["slope"], f21["intercept"], mean_y)
        o["boot"] = oos_boot(pts, syn, np.random.default_rng(46))
        o["old_fit_exp32"] = oos(pts, 0.8713, -1.4016, -4.90)
        o["points"] = [f"{c['model']}|{c['ds']}" for c in pts]
        O_[nm] = o
        lines.append(f"OOS {nm:15s} n={o['n_points']} MAE fit {o['mae_fit']:.2f} {np.round(o['boot']['mae_fit'], 2).tolist()} bias {o['bias_fit']:+.2f} "
                     f"{np.round(o['boot']['bias_fit'], 2).tolist()} | MAE y=x {o['mae_y_eq_x']:.2f} (fit - y=x {np.round(o['boot']['fit_minus_y_eq_x'], 2).tolist()}) "
                     f"| MAE mean-syn-y {o['mae_mean_syn_y']:.2f} (fit - mean {np.round(o['boot']['fit_minus_mean'], 2).tolist()}) MAE zero {o['mae_zero']:.2f} "
                     f"| sign pred {o['sign_agree_pred']}/{o['n_points']} x {o['sign_agree_x']}/{o['n_points']} (|y|>=1: pred {o['sign_agree_pred_abs_y_ge1']}/{o['n_abs_y_ge1']}, "
                     f"x {o['sign_agree_x_abs_y_ge1']}/{o['n_abs_y_ge1']}) | r {o['r_pred_obs']:.3f} | old Exp-32 fit MAE {o['old_fit_exp32']['mae_fit']:.2f} bias {o['old_fit_exp32']['bias_fit']:+.2f}")
    for c in nat_pts + clo_pts:
        c["y_hat"] = f21["intercept"] + f21["slope"] * c["x"]
    res["out_of_sample"] = O_
    # ---- strict sensitivity variant: also drop trailing sentences with a mid-sentence conclusion
    from experiments.two_channel.uniform_two_channel import RELKW, strict_body
    labS = load_labels("", ("_strict",))
    rows_s = []
    for r in rows:
        b = strict_body(r)
        rows_s.append(r | {"body": b, "hkey": hkey(r["x"], r["y"], b), "has_kw": bool(RELKW.search(b)), "changed": b != r["body"]})
    recs_s = shape_records(rows_s, labS)
    gs = defaultdict(list)
    for r in recs_s:
        gs[(r["group"], r["model"], r["ds"])].append(r)
    n_missing = sum(r["status"] == "missing" for r in recs_s)
    cs = {}
    for (g, m, ds), rs in sorted(gs.items()):
        c = cell(rs, np.random.default_rng(5))
        c["changed_pct"] = 100 * float(np.mean([r["changed"] for r in rows_s if (r["group"], r["model"], r["ds"]) == (g, m, ds)]))
        c |= {"group": g, "model": m, "ds": ds}
        cs[f"{g}|{m}|{ds}"] = c
    syn_s = [cs[f"open|{m}|{ds}"] for m in M7 for ds in SYN_DS]
    fs = fit(syn_s)
    clo_s = [c for c in cs.values() if c["group"] == "closed" and c.get("n_cov", 0) >= 20]
    nat_s = nat_pts  # natural rationales were not re-truncated (Exp-35 labels)
    os_ = {"natural": oos(nat_s, fs["slope"], fs["intercept"], float(np.mean([p["y"] for p in syn_s]))),
           "closed": oos(clo_s, fs["slope"], fs["intercept"], float(np.mean([p["y"] for p in syn_s])))}
    ans_s = {}
    for g, dsl, models in (("open", SYN_DS, M7), ("closed", ("e9h", "e9x", "e9c_cues"), CLOSED)):
        for m in models:
            rs = [r for ds in dsl for r in gs.get((g, m, ds), [])]
            c = cell(rs, np.random.default_rng(6))
            ans_s[f"{g}|{m}"] = {k: c.get(k) for k in ("n", "coverage", "W", "aT", "aF", "A", "A_ci")}
    res["strict_variant"] = {"doc": strict_body.__doc__, "n_missing_labels": n_missing, "fit_21": fs,
                             "cells": {k: strip(c) for k, c in cs.items()}, "out_of_sample": os_, "answer_step_pooled": ans_s}
    lines.append(f"\nSTRICT variant (missing labels {n_missing}): fit slope {fs['slope']:.3f} model {np.round(fs['slope_ci_model'], 3).tolist()} "
                 f"intercept {fs['intercept']:+.2f} pt {np.round(fs['intercept_ci_point'], 2).tolist()} model {np.round(fs['intercept_ci_model'], 2).tolist()} "
                 f"item+model {np.round(fs['intercept_ci_item_model'], 2).tolist()} r {fs['pearson_r']:.3f}")
    for k, c in cs.items():
        lines.append(f"STRICT CELL {k:38s} changed {c['changed_pct']:5.1f}% cov {c['coverage']:5.1f} W {c.get('W', np.nan):5.1f} aT/aF {c.get('aT', np.nan):5.1f}/"
                     f"{c.get('aF', np.nan):5.1f} x {c.get('x', np.nan):+6.2f} y {c['y']:+6.2f} A {c.get('A', np.nan):+6.2f} {[round(v, 1) for v in c.get('A_ci', [])]}")
    for k, a in ans_s.items():
        lines.append(f"STRICT ANSWER-STEP {k:28s} cov {a['coverage']:.1f} aT {a['aT']:.1f} aF {a['aF']:.1f} A {a['A']:+.2f} [{a['A_ci'][0]:+.2f}, {a['A_ci'][1]:+.2f}]")
    for k, o in os_.items():
        lines.append(f"STRICT OOS {k}: n {o['n_points']} MAE fit {o['mae_fit']:.2f} bias {o['bias_fit']:+.2f} MAE y=x {o['mae_y_eq_x']:.2f} mean {o['mae_mean_syn_y']:.2f} sign {o['sign_agree_pred']}/{o['n_points']}")
    # ---- agreement ranges (paper: 75.5-99.6% on the cue set)
    rng_ = {}
    for nm, sel in (("open7_cues", [cells[f"open|{m}|cues"] for m in M7]), ("open21", syn),
                    ("natural13", nat_pts), ("closed", clo_pts)):
        aT = [c["aT"] for c in sel if np.isfinite(c.get("aT", np.nan))]; aF = [c["aF"] for c in sel if np.isfinite(c.get("aF", np.nan))]
        rng_[nm] = {"aT_min": min(aT), "aT_max": max(aT), "aF_min": min(aF), "aF_max": max(aF), "all_min": min(aT + aF), "all_max": max(aT + aF)}
        lines.append(f"AGREEMENT {nm:12s} aT {min(aT):.1f}-{max(aT):.1f} aF {min(aF):.1f}-{max(aF):.1f}")
    res["agreement_ranges"] = rng_
    # ---- coverage drop summary
    cd = {}
    for k, c in cells.items():
        if c["group"] == "open":
            cd[k] = {"v3_full": c["cov_v3_full"], "v3_trunc": c["cov_v3_trunc"], "judge_trunc": c["coverage"], "lost_vs_v3_full": c["lost_vs_v3_full"]}
        elif c["group"] == "closed":
            cd[k] = {"e38_judge_conclusion_kept": c["cov_e38"], "judge_trunc": c["coverage"], "lost_all_relations": c["lost_all_relations"],
                     "v3_full": c["cov_v3_full"], "v3_trunc": c["cov_v3_trunc"]}
    res["coverage_drop"] = cd
    # ---- judge agreement (kappa) and validation vs claim_parse v3
    ja = {}
    for g in ("open", "closed", "all"):
        rr = [r for r in recs if (g == "all" or r["group"] == g) and r["status"] not in ("empty", "prefilter_none", "missing")]
        a = [rel(r["labA"], r) for r in rr]; b = [rel(r["labB"], r) for r in rr]
        a2 = [v if v in ("x", "y") else "no" for v in a]; b2 = [v if v in ("x", "y") else "no" for v in b]
        ja[g] = {"n": len(rr), "agree_4way": 100 * float(np.mean([u == v for u, v in zip(a, b)])), "kappa_4way": kappa(a, b),
                 "kappa_rel_or_no": kappa(a2, b2), "none_unclear_rate_opus": 100 * float(np.mean([v not in ("x", "y") for v in a])),
                 "none_unclear_rate_gpt": 100 * float(np.mean([v not in ("x", "y") for v in b])),
                 "both_shape_disagree": int(sum(u in ("x", "y") and v in ("x", "y") and u != v for u, v in zip(a, b)))}
        lines.append(f"JUDGES {g:6s} n={ja[g]['n']} agree4 {ja[g]['agree_4way']:.2f}% kappa4 {ja[g]['kappa_4way']:.3f} kappa(rel/no) {ja[g]['kappa_rel_or_no']:.3f} "
                     f"none/unclear Opus {ja[g]['none_unclear_rate_opus']:.1f}% GPT {ja[g]['none_unclear_rate_gpt']:.1f}% opposite shapes {ja[g]['both_shape_disagree']}")
    res["judge_agreement"] = ja
    # natural (Exp-35) judge agreement reused
    e35 = json.load(open(B / "e35_natural_two_channel.json"))
    res["judge_agreement_natural_exp35"] = e35["judge_agreement"]["pooled"]
    # validation subsample: random 100 open + 100 closed texts, consensus vs claim_parse v3 on the same truncated text
    vs = val_sample(rows, 100)
    byh = {(r["group"], r["model"], r["ds"], str(r["item"])): r for r in recs}
    val = {}
    for g in ("open", "closed"):
        sub = [byh[(r["group"], r["model"], r["ds"], str(r["item"]))] for r in vs if r["group"] == g]
        both = [r for r in sub if r["claim"] and r["v3_trunc"]]
        val[g] = {"n": len(sub), "judge_cov": sum(bool(r["claim"]) for r in sub), "v3_cov": sum(bool(r["v3_trunc"]) for r in sub),
                  "both": len(both), "agree_when_both": sum(r["claim"] == r["v3_trunc"] for r in both),
                  "disagree_items": [f"{r['model']}|{r['ds']}|{r['item']}: judge {r['claim']} v3 {r['v3_trunc']}" for r in both if r["claim"] != r["v3_trunc"]]}
        aa = [rel(r["labA"], r) for r in sub if r["status"] not in ("empty", "prefilter_none", "missing")]
        bb = [rel(r["labB"], r) for r in sub if r["status"] not in ("empty", "prefilter_none", "missing")]
        val[g]["kappa_4way_subsample"] = kappa(aa, bb) if aa else None
        allb = [r for r in recs if r["group"] == g and r["claim"] and r["v3_trunc"]]
        val[g]["all_texts_both"] = len(allb)
        val[g]["all_texts_agree_pct"] = 100 * float(np.mean([r["claim"] == r["v3_trunc"] for r in allb])) if allb else None
        lines.append(f"VALIDATION {g:6s} random {val[g]['n']}: judge cov {val[g]['judge_cov']} v3 cov {val[g]['v3_cov']} both {val[g]['both']} "
                     f"agree {val[g]['agree_when_both']}/{val[g]['both']} kappa4(sub) {val[g]['kappa_4way_subsample']} | all texts: agree {val[g]['all_texts_agree_pct']} "
                     f"(n {val[g]['all_texts_both']}) | disagreements {val[g]['disagree_items']}")
    # judge-level coverage before/after on the validation subsample (pre = answer line removed, conclusion kept)
    pre = load_labels("_pre")
    if pre[JUDGES[0]]:
        vp = {}
        for g in ("open", "closed"):
            sub = [r for r in vs if r["group"] == g]
            b4, af = [], []
            for r in sub:
                hk = hkey(r["x"], r["y"], r["body_pre"], "pre")
                a, b = pre[JUDGES[0]].get(hk), pre[JUDGES[1]].get(hk)
                if a is None or b is None:
                    continue
                b4.append(a == b and a in (r["x"], r["y"]))
                af.append(bool(byh[(r["group"], r["model"], r["ds"], str(r["item"]))]["claim"]))
            b4, af = np.array(b4, bool), np.array(af, bool)
            if not len(b4):
                continue
            vp[g] = {"n": int(len(b4)), "cov_conclusion_kept": 100 * float(b4.mean()), "cov_conclusion_removed": 100 * float(af.mean()),
                     "lost_all": 100 * float(np.mean(b4 & ~af))}
            lines.append(f"PRE/POST (validation subsample, judges) {g}: n {vp[g]['n']} coverage conclusion kept {vp[g]['cov_conclusion_kept']:.1f}% -> removed "
                         f"{vp[g]['cov_conclusion_removed']:.1f}% (lost all relations {vp[g]['lost_all']:.1f}%)")
        res["pre_post_validation"] = vp
    # prefilter check
    pf = load_labels("_prefilter")
    if pf[JUDGES[0]]:
        ks = set(pf[JUDGES[0]]) & set(pf[JUDGES[1]])
        nshape = sum(pf[JUDGES[0]][k] not in ("none", "unclear") and pf[JUDGES[0]][k] == pf[JUDGES[1]][k] for k in ks)
        res["prefilter_check"] = {"n": len(ks), "consensus_relation": nshape,
                                  "opus_relation": sum(pf[JUDGES[0]][k] not in ("none", "unclear") for k in ks),
                                  "gpt_relation": sum(pf[JUDGES[1]][k] not in ("none", "unclear") for k in ks)}
        lines.append(f"PREFILTER CHECK (keyword-free texts judged): {res['prefilter_check']}")
    res["cells"] = {k: strip(c) for k, c in cells.items()}
    from pathlib import Path
    ob = Path("/tmp/e45_debug") if DEBUG else B
    ob.mkdir(exist_ok=True)
    json.dump(res, open(ob / "e45_uniform_two_channel.json", "w"), indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o))
    (ob / "rigor_e45.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
