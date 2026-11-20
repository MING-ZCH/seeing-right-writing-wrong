"""Exp-20 / round-11 CPU analyses (Opus round-11 M1, M3, M4, m2, m4, m8, m9).

churn     D+-C+- cells on the 2,850 pool per model under (i) the Tab. 1 readout (build_bundle.load_main: exact full-vocab
          e1x for the V0-engine models and Molmo, top-20 e4 for the V1-engine models), (ii) the legacy original-run
          letter scores used by build_bundle.groups_from (Sec. 5 / Fig. 2b / commit and rescue audits), (iii) (ii) restricted
          to e2b records with a non-empty rationale (the e17n rescue block). Per-task cells incl. Synthetic Depth.
          -> bundle/churn.json
facts     M1 wording facts: pair-level both-correct (e17n), Spearman of twin delta-d' vs Direct twin d' / Direct vertical /
          Direct size weights, size-tercile monotone affirmation, twin area ratios, steering random directions, stale
          canonical file. -> lines in bundle/rigor_e17o.txt
mult      Benjamini-Hochberg FDR over the per-model headline tests (resampling p-values recomputed here), the competence
          slope (rho = 0.48) with a model random intercept and a cluster bootstrap over models, twin-swap image share
          normalised like the gray share. -> bundle/multiplicity.json
rescue    Rescue comparator (D-C- and D+C+ relational sentences, rescue-position-matched; both judges).
          -> bundle/rescue_comparator.json
Usage: python rigor_e17o.py --only churn,facts,mult,rescue
"""
from __future__ import annotations
import vtax_paths as _vtp  # noqa: E402

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
from scipy.stats import norm, spearmanr

import build_bundle as BB
from build_bundle import pe, rj

O = Path(f"{_vtp.OUTPUTS}")
L = Path(f"{_vtp.LEGACY_RESULTS}")
B = Path("outputs/bundle")
M7 = ["qwen3b", "qwen7b", "internvl3_8b", "phi35v", "molmo7b", "qwen3vl_8b", "internvl35_8b"]
CELLS = ("D+C-", "D-C+", "D+C+", "D-C-")
J2 = ("claude-opus-5-5", "gpt-6-astra")


def cell(d: bool, c: bool) -> str:
    return ("D+" if d else "D-") + ("C+" if c else "C-")


def summarize(cnt: Counter, n: int) -> dict:
    out = {k: int(cnt.get(k, 0)) for k in CELLS} | {"n": n}
    out["net_pts"] = round(100 * (out["D-C+"] - out["D+C-"]) / n, 2)
    out["churn_pct"] = round(100 * (out["D-C+"] + out["D+C-"]) / n, 1)
    return out


# ----------------------------------------------------------------------------- churn

def churn(lines: list[str]) -> dict:
    nb = json.load(open(B / "numbers.json"))
    res = {"_readouts": {"tab1": "build_bundle.load_main (Tab. 1): exact full-vocab lp_sum argmax from e1x for qwen3b/qwen7b/internvl3_8b/phi35v/molmo7b; "
                                 "top-20 argmax from e4 for qwen3vl_8b/qwen3vl_8b_think/internvl35_8b",
                         "legacy": "build_bundle.groups_from: argmax of the original-run letter scores (results/v2/e1 or e4); defines D+-C+- in "
                                   "Sec. 5, Fig. 2b, Tab. S23/S25/S26 and all e2b-based audits",
                         "legacy_nonempty_R": "legacy cells restricted to e2b records with >= 1 rationale sentence (e17n rescue block)"}}
    for m in BB.MODELS:
        main = BB.load_main(O, L, m)
        G = BB.groups_from(main, L, O, m)
        D, C = main["direct"], main["cot"]
        ks = [k for k in D if k in C]
        e2b = {r["item_id"]: bool(r["sentences"]) for r in rj(O / "e2b" / f"{m}.jsonl")}
        row = {}
        for t in ["all"] + BB.TASKS:
            kt = [k for k in ks if t == "all" or D[k][3] == t]
            ex = Counter(cell(bool(D[k][0]), bool(C[k][0])) for k in kt)
            lg = Counter(G[k]["group"] for k in kt)
            r = {"tab1": summarize(ex, len(kt)), "legacy": summarize(lg, len(kt)),
                 "items_cell_differs": int(sum(cell(bool(D[k][0]), bool(C[k][0])) != G[k]["group"] for k in kt))}
            if e2b:
                ne = Counter(G[k]["group"] for k in kt if e2b.get(k))
                r["legacy_nonempty_R"] = summarize(ne, sum(bool(e2b.get(k)) for k in kt))
                r["empty_R_items"] = {g: int(sum(1 for k in kt if k in e2b and not e2b[k] and G[k]["group"] == g)) for g in CELLS}
            row[t] = r
        tab1 = nb[m]["behavioral"]["cot"]["minus_direct"]
        row["tab1_cot_minus_direct"] = {"mean": tab1["mean"], "lo": tab1["lo"], "hi": tab1["hi"]}
        row["net_matches_tab1"] = abs(row["all"]["tab1"]["net_pts"] - tab1["mean"]) < 0.01
        res[m] = row
        a = row["all"]
        lines.append(f"CHURN {m:16s} Tab.1 readout D+C- {a['tab1']['D+C-']} D-C+ {a['tab1']['D-C+']} net {a['tab1']['net_pts']:+.2f} (Tab.1 {tab1['mean']:+.2f}) "
                     f"churn {a['tab1']['churn_pct']}% | legacy D+C- {a['legacy']['D+C-']} D-C+ {a['legacy']['D-C+']} net {a['legacy']['net_pts']:+.2f} "
                     f"| cells differ {a['items_cell_differs']} items"
                     + (f" | non-empty-R D+C- {a['legacy_nonempty_R']['D+C-']} D-C+ {a['legacy_nonempty_R']['D-C+']} (empty R: {a['empty_R_items']})" if "legacy_nonempty_R" in a else ""))
        s = row["syn_depth"]
        lines.append(f"CHURN-SYN {m:16s} n={s['tab1']['n']} D+C- {s['tab1']['D+C-']} D-C+ {s['tab1']['D-C+']} net {s['tab1']['net_pts']:+.1f} churn {s['tab1']['churn_pct']}% "
                     f"| legacy D+C- {s['legacy']['D+C-']} D-C+ {s['legacy']['D-C+']}")
    json.dump(res, open(B / "churn.json", "w"), indent=1)
    return res


# ----------------------------------------------------------------------------- M1 wording facts

def exact_spearman(x, y) -> tuple[float, float]:
    """Spearman rho with an exact two-sided permutation p (n <= 8)."""
    from itertools import permutations
    x, y = np.asarray(x, float), np.asarray(y, float)
    rho = spearmanr(x, y)[0]
    perms = [spearmanr(x, np.array(p))[0] for p in permutations(y)]
    return float(rho), float(np.mean(np.abs(perms) >= abs(rho) - 1e-12))


def facts(lines: list[str]) -> dict:
    res = {}
    en = json.load(open(B / "e17n.json"))
    # (a) pair-level both-correct
    pb = {m: (p["pair_both_correct_direct"], p["pair_both_correct_cot"]) for m, p in en["syn"]["pooled"].items()}
    res["pair_both_correct"] = pb
    lines.append("FACT pair-both-correct Syn. Depth Dir->CoT (e17n.json syn.pooled.pair_both_correct_*; 500 items x 2 object orders): "
                 + ", ".join(f"{m} {a}->{b}" for m, (a, b) in pb.items())
                 + f" | falls in {sum(b < a for a, b in pb.values())}/{len(pb)}")
    # (b) predictors of twin delta-d'
    tw = en["twins"]
    cu = json.load(open(B / "cues.json"))
    gl = json.load(open(B / "glm_decomp.json"))["cues"]
    rows = {m: {"twin_dd": tw[m]["diff"], "twin_direct_dprime": tw[m]["direct"],
                "twin_direct_vertical": cu[m]["twins"]["decomposition"]["vertical"]["direct"]["d"],
                "twin_direct_occlusion": cu[m]["twins"]["decomposition"]["occlusion"]["direct"]["d"],
                "cue_glm_direct_vertical": gl[m]["direct"]["vertical"]["dprime"],
                "cue_glm_direct_size": gl[m]["direct"]["size"]["dprime"],
                "cue_glm_direct_occlusion": gl[m]["direct"]["occlusion"]["dprime"]} for m in M7}
    y = [rows[m]["twin_dd"] for m in M7]
    sp = {}
    for k in ("twin_direct_dprime", "twin_direct_vertical", "cue_glm_direct_vertical", "cue_glm_direct_size", "cue_glm_direct_occlusion",
              "twin_direct_occlusion"):
        x = [rows[m][k] for m in M7]
        rho, p = exact_spearman(x, y)
        rises = [rows[m][k] for m in M7 if tw[m]["ci"][0] > 0]
        falls = [rows[m][k] for m in M7 if tw[m]["ci"][1] < 0]
        sep = (min(rises) > max(falls)) or (max(rises) < min(falls))
        sp[k] = {"rho": round(rho, 3), "p_exact": round(p, 4), "n": len(M7), "separates_sig_rises_from_sig_falls": bool(sep),
                 "rises": [round(v, 2) for v in rises], "falls": [round(v, 2) for v in falls]}
    res["twin_dd_predictors"] = {"per_model": rows, "spearman": sp}
    lines.append("FACT twin delta-d' predictors (7 models; e17n.json twins, cues.json twins.decomposition [Tab. S12], glm_decomp.json cues [Tab. S13]; "
                 "exact permutation p): " + "; ".join(f"{k} rho={v['rho']:+.2f} p={v['p_exact']:.3f} separates sig. rises/falls={v['separates_sig_rises_from_sig_falls']}"
                                                     for k, v in sp.items()))
    # (c) size-tercile monotone affirmation
    tc = {}
    for m, t in en["syn"]["tercile"].items():
        loss = [t[f"{x}|all"]["loss"] for x in ("low", "mid", "high")]
        mono = loss[0] > loss[1] > loss[2] and loss[2] < 0
        hn = t["high|no"]
        aff = hn["cot_yes"] > 50 and hn["cot_yes"] > 100 - hn["direct"]
        tc[m] = {"loss_low_mid_high": loss, "monotone_growing_loss": bool(mono), "high_no_direct_yes": round(100 - hn["direct"], 1),
                 "high_no_cot_yes": hn["cot_yes"], "cot_affirms_high_no": bool(aff), "pattern": bool(mono and aff)}
    res["tercile_pattern"] = tc
    lines.append("FACT size-tercile pattern (loss grows low->mid->high AND CoT 'yes' > 50% and > Direct 'yes' on high-ratio gold-'no'; e17n.json syn.tercile, Tab. S22): "
                 + f"{sum(v['pattern'] for k, v in tc.items() if k in M7)}/7 open models: " + ", ".join(m for m in M7 if tc[m]["pattern"])
                 + " | others: " + ", ".join(f"{m} {tc[m]['loss_low_mid_high']} yes D{tc[m]['high_no_direct_yes']:.0f}->C{tc[m]['high_no_cot_yes']:.0f}" for m in tc if not tc[m]["pattern"]))
    # (d) twin area ratio
    recs = rj(O / "e2d" / "qwen3b.jsonl")
    im = {}
    for r in recs:
        mt = r["meta"] if isinstance(r["meta"], dict) else __import__("ast").literal_eval(r["meta"])
        im[r["image_id"]] = (mt["area_ratio_front_over_back"], mt["area_front_px"] / (mt["area_back_px"] * (1 - mt["back_hidden_frac"])))
    full = np.array([v[0] for v in im.values()])
    vis = np.array([v[1] for v in im.values()])
    res["twin_area_ratio"] = {"n_scenes": len(im), "full_mean": round(float(full.mean()), 3), "full_range": [round(float(full.min()), 3), round(float(full.max()), 3)],
                              "visible_mean": round(float(vis.mean()), 3), "visible_share_gt1": round(100 * float(np.mean(vis > 1)), 1)}
    a = res["twin_area_ratio"]
    lines.append(f"FACT twin front/back area ratio (e2d meta area_ratio_front_over_back, {a['n_scenes']} scenes; e2_occlusion_set.py ratio_min 0.55): "
                 f"full-silhouette mean {a['full_mean']} range {a['full_range']}; visible-area (back x (1-hidden)) mean {a['visible_mean']}, {a['visible_share_gt1']}% > 1")
    # steering (m4)
    st = json.load(open(B / "steering.json"))
    rand_files = sorted(p.name for p in (O / "e18").glob("*_DCm_wrong_random_k*.jsonl") if "limit" not in p.name)
    pr = st["_paired_truth_minus_random"]
    g = lambda m, kind, k: next(r for r in st[m] if r["group"] == "DCm_wrong" and r["kind"] == kind and r["k"] == k)
    res["steering"] = {"random_files": rand_files, "n_random_directions_per_k": 1, "seed": 0,
                       "molmo_k32": {"truth": [g("molmo7b", "truth", 32.0)["claim_correct"]["mean"], g("molmo7b", "truth", 32.0)["answer_correct"]["mean"]],
                                     "random": [g("molmo7b", "random", 32.0)["claim_correct"]["mean"], g("molmo7b", "random", 32.0)["answer_correct"]["mean"]],
                                     "paired": pr["molmo7b|k32"]}}
    s = res["steering"]["molmo_k32"]
    lines.append(f"FACT steering random control (outputs/e18, e18_steer.py --seed default 0; steering.json): D+C- steering uses ONE random unit direction per model "
                 f"(seed 0, same direction at every k): {', '.join(rand_files)}; n = 222 claims Qwen3-VL-8B, 114 Molmo-7B-D; CIs resample items only. "
                 f"Only the mixed set (steering_mixed.json) uses 3 random seeds.")
    lines.append(f"FACT steering Molmo k=32 (steering.json): truth vs random claims {s['truth'][0]} vs {s['random'][0]} (paired {s['paired']['claim_correct']['diff']:+.1f} "
                 f"{s['paired']['claim_correct']['ci']}), answers {s['truth'][1]} vs {s['random'][1]} (paired {s['paired']['answer_correct']['diff']:+.1f} "
                 f"{s['paired']['answer_correct']['ci']}) -> truth ~ random at k=32 (both CIs include 0); Molmo truth>random only for answers at k=16 "
                 f"({pr['molmo7b|k16']['answer_correct']['diff']:+.1f} {pr['molmo7b|k16']['answer_correct']['ci']})")
    # stale canonical (m8)
    can = Counter()
    for l in open("outputs/e1x_canonical.jsonl"):
        r = json.loads(l)
        if r.get("model") == "molmo7b" and r.get("task") == "syn_depth" and r.get("condition") == "direct":
            can["n"] += 1
            can["exact"] += r["pred"] == r["gold"]
            can["top20"] += r["pred_top20"] == r["gold"]
    nb = json.load(open(B / "numbers.json"))
    res["stale_canonical_molmo_syn_direct"] = {"exact_pred": round(100 * can["exact"] / can["n"], 1), "top20_pred": round(100 * can["top20"] / can["n"], 1),
                                               "numbers_json": nb["molmo7b"]["behavioral"]["task:syn_depth"]["direct"]["mean"]}
    c = res["stale_canonical_molmo_syn_direct"]
    lines.append(f"FACT Molmo Syn. Depth Direct: paper/numbers.json {c['numbers_json']} (source {O}/e1x/molmo7b.jsonl via build_bundle.load_main; = legacy results/v2/e1) "
                 f"vs stale outputs/e1x_canonical.jsonl {c['exact_pred']} (pred) / {c['top20_pred']} (pred_top20); stale file marked README_stale.txt")
    return res


# ----------------------------------------------------------------------------- multiplicity

def boot_p(b: np.ndarray, est: float) -> float:
    """Two-sided percentile bootstrap p for H0: effect = 0."""
    nb = len(b)
    lo = (np.sum(b <= 0) + 1) / (nb + 1)
    hi = (np.sum(b >= 0) + 1) / (nb + 1)
    return float(min(1.0, 2 * min(lo, hi)))


def wald_p(est: float, se: float) -> float:
    return float(2 * norm.sf(abs(est) / se)) if se > 0 else 0.0


def dprime(resp, truth):
    h = (resp[truth].sum() + .5) / (truth.sum() + 1)
    f = (resp[~truth].sum() + .5) / ((~truth).sum() + 1)
    return float(norm.ppf(h) - norm.ppf(f))


def t_syn_pooled(m: str, rng, n_boot: int):
    """Same construction as rigor_e17n.syn (a)."""
    main = BB.load_main(O, L, m)
    keys = [k for k, v in main["direct"].items() if v[3] == "syn_depth" and k in main["cot"]]
    gold_yes = {k: (main["direct"][k][4] == "A") == bool(main["direct"][k][0]) for k in keys}
    cf = {r["item_id"]: r for r in rj(O / "e2a" / f"{m}.jsonl") if r["variant"] == "cfq" and r["task"] == "syn_depth"}
    if not cf:
        return None
    from rigor_e17n import parse_readout
    rows = []
    for k in [k for k in keys if k in cf]:
        t = gold_yes[k]
        for cond, op, cp in (("direct", main["direct"][k][4], parse_readout(cf[k]["readout_direct"])), ("cot", main["cot"][k][4], parse_readout(cf[k]["readout_cot"]))):
            rows.append((cond, op == "A", t, cf[k]["image_id"]))
            rows.append((cond, cp == "A", not t, cf[k]["image_id"]))
    arr = {c: (np.array([r[1] for r in rows if r[0] == c]), np.array([r[2] for r in rows if r[0] == c]), np.array([r[3] for r in rows if r[0] == c])) for c in ("direct", "cot")}
    est = dprime(*arr["cot"][:2]) - dprime(*arr["direct"][:2])
    cl = np.unique(arr["direct"][2])
    idx = {c: {g: np.where(arr[c][2] == g)[0] for g in cl} for c in arr}
    bs = []
    for _ in range(n_boot):
        js = rng.choice(cl, len(cl))
        s = {c: np.concatenate([idx[c][g] for g in js]) for c in arr}
        bs.append(dprime(arr["cot"][0][s["cot"]], arr["cot"][1][s["cot"]]) - dprime(arr["direct"][0][s["direct"]], arr["direct"][1][s["direct"]]))
    return est, np.array(bs)


def t_twins(m: str, rng, n_boot: int):
    """Same construction as dprime_ci.py."""
    import dprime_ci as DC
    d, c = DC.per_pair(m, "direct"), DC.per_pair(m, "cot")
    pairs = sorted(set(d) & set(c))
    if not pairs:
        return None
    A = {n: [np.array(P[p]) for p in pairs] for n, P in (("d", d), ("c", c))}
    f = lambda P, idx: dprime(np.concatenate([P[i] for i in idx])[:, 0].astype(bool), np.concatenate([P[i] for i in idx])[:, 1].astype(bool))
    full = np.arange(len(pairs))
    est = f(A["c"], full) - f(A["d"], full)
    bs = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(pairs), len(pairs))
        bs.append(f(A["c"], idx) - f(A["d"], idx))
    return est, np.array(bs)


def t_cvb(m: str, rng, n_boot: int):
    main = BB.load_main(O, L, m)
    ks = [k for k, v in main["direct"].items() if v[3] == "cvb_depth" and k in main["cot"]]
    gy = np.array([(main["direct"][k][4] == "A") == bool(main["direct"][k][0]) for k in ks])
    pd_ = np.array([main["direct"][k][4] == "A" for k in ks])
    pc = np.array([main["cot"][k][4] == "A" for k in ks])
    g = np.array([main["direct"][k][2] for k in ks])
    u = np.unique(g)
    idx = {x: np.where(g == x)[0] for x in u}
    est = dprime(pc, gy) - dprime(pd_, gy)
    bs = []
    for _ in range(n_boot):
        s = np.concatenate([idx[x] for x in rng.choice(u, len(u))])
        bs.append(dprime(pc[s], gy[s]) - dprime(pd_[s], gy[s]))
    return est, np.array(bs)


def t_glm(m: str, rng, n_boot: int):
    """CoT - Direct probit-GLM occlusion and vertical components on the cue set (same model as rigor_e17f.glm_set)."""
    import rigor_e17f as F
    R = rj(O / "e10" / f"{m}_cues.jsonl")
    data = {c: F.glm_rows([r for r in R if r["condition"] == c], True) for c in ("direct", "cot")}
    pairs = sorted(set(data["direct"]["pair"]))
    pidx = {c: {p: np.where(np.array(d["pair"]) == p)[0] for p in pairs} for c, d in data.items()}
    est = {c: F.probit_fit(d["X"], d["y"]) for c, d in data.items()}
    bs = []
    for _ in range(n_boot):
        js = rng.integers(0, len(pairs), len(pairs))
        b = {}
        for c, d in data.items():
            s = np.concatenate([pidx[c][pairs[j]] for j in js])
            b[c] = F.probit_fit(d["X"][s], d["y"][s])
        bs.append(2 * (b["cot"] - b["direct"]))
    bs = np.array(bs)
    e = 2 * (est["cot"] - est["direct"])
    return {"occlusion": (float(e[2]), bs[:, 2]), "vertical": (float(e[3]), bs[:, 3])}


def bh(p: list[float], q: float = 0.05) -> tuple[list[float], list[bool]]:
    p = np.asarray(p, float)
    m = len(p)
    o = np.argsort(p)
    adj = np.empty(m)
    run = 1.0
    for r in range(m - 1, -1, -1):
        run = min(run, p[o[r]] * m / (r + 1))
        adj[o[r]] = run
    return [float(x) for x in adj], [bool(x < q) for x in adj]


def competence_slope() -> dict:
    import pandas as pd
    import statsmodels.formula.api as smf
    rs = json.load(open(B / "relation_sensitivity.json"))
    df = pd.DataFrame([{"model": m, "rel": f, "acc": v["direct_acc"]["mean"], "dvis": v["dvis_gray"]["mean"]}
                       for m, d in rs.items() if m != "_tests" for f, v in d.items() if f != "occ"])
    out = {"n_cells": len(df), "n_models": df.model.nunique()}
    rho, p = spearmanr(df.acc, df.dvis)
    out["spearman_pooled"] = [round(float(rho), 3), round(float(p), 4)]
    ols = smf.ols("dvis ~ acc", df).fit()
    out["ols_slope_per_10pts"] = [round(10 * float(ols.params["acc"]), 2), [round(10 * float(v), 2) for v in ols.conf_int().loc["acc"]], round(float(ols.pvalues["acc"]), 4)]
    mm = smf.mixedlm("dvis ~ acc", df, groups=df["model"]).fit(reml=True, method=["lbfgs", "powell"])
    ci = mm.conf_int().loc["acc"]
    out["mixedlm_slope_per_10pts"] = [round(10 * float(mm.params["acc"]), 2), [round(10 * float(ci.iloc[0]), 2), round(10 * float(ci.iloc[1]), 2)], round(float(mm.pvalues["acc"]), 4)]
    out["mixedlm_group_var"] = round(float(mm.cov_re.iloc[0, 0]), 2)
    out["mixedlm_resid_var"] = round(float(mm.scale), 2)
    df["acc_mean"] = df.groupby("model").acc.transform("mean")
    df["acc_within"] = df.acc - df.acc_mean
    fe = smf.ols("dvis ~ acc_within + C(model)", df).fit()
    out["within_model_slope_per_10pts"] = [round(10 * float(fe.params["acc_within"]), 2), [round(10 * float(v), 2) for v in fe.conf_int().loc["acc_within"]], round(float(fe.pvalues["acc_within"]), 4)]
    mb = df.groupby("model")[["acc", "dvis"]].mean()
    rb = spearmanr(mb.acc, mb.dvis)
    out["between_model_spearman_7"] = [round(float(rb[0]), 3), round(float(rb[1]), 4)]
    rng = np.random.default_rng(7)
    ms = df.model.unique()
    bs_s, bs_r = [], []
    for _ in range(5000):
        pick = rng.choice(ms, len(ms))
        d = pd.concat([df[df.model == x] for x in pick])
        if d.acc.nunique() < 3:
            continue
        bs_s.append(10 * np.polyfit(d.acc, d.dvis, 1)[0])
        bs_r.append(spearmanr(d.acc, d.dvis)[0])
    q = lambda v: [round(float(np.nanquantile(v, .025)), 2), round(float(np.nanquantile(v, .975)), 2)]
    out["cluster_bootstrap_over_models"] = {"slope_per_10pts_ci": q(bs_s), "spearman_ci": q(bs_r), "n_boot": len(bs_s)}
    return out


def twin_share() -> dict:
    out = {}
    for m in ("qwen3vl_8b", "molmo7b", "qwen7b"):
        d = json.load(open(B / "e17_probe" / f"{m}_m1.json"))["positions"]
        out[m] = {}
        for pos in ("s", "r", "a"):
            x = d[pos]
            pre, post = x["pre_swap_follow_twin"], x["post_swap_follow_twin"]
            out[m][pos] = {"gray_share": x["image_share_gray"], "pre": pre, "post": post,
                           "twin_share_norm": round((post - pre) / (1 - 2 * pre), 2),
                           "twin_share_norm_bal_denominator": round((post - pre) / (2 * (x["orig_bal"] - 0.5)), 2)}
    return out


def mult(lines: list[str]) -> dict:
    nb = json.load(open(B / "numbers.json"))
    en = json.load(open(B / "e17n.json"))
    dp = json.load(open(B / "dprime_ci.json"))
    gl = json.load(open(B / "glm_decomp.json"))["cues"]
    rng = np.random.default_rng(20)
    NB = 2000
    tests = []

    def add(fam, m, est, lo, hi, p_resamp, p_wald, extra=None):
        tests.append({"family": fam, "model": m, "est": round(float(est), 3), "ci": [round(float(lo), 3), round(float(hi), 3)],
                      "p_resampling": float(p_resamp), "p_wald": float(p_wald)} | (extra or {}))

    for m in BB.MODELS:
        b = nb[m]["behavioral"]
        x = b["cot"]["minus_direct"]
        add("tab1_cot_minus_direct", m, x["mean"], x["lo"], x["hi"], x["p"], wald_p(x["mean"], (x["hi"] - x["lo"]) / 3.92), {"p_type": "image-cluster sign-flip (2000)"})
        x = b["filler+turn"]["minus_cot"]
        add("tab1_cot_minus_fillerturn", m, -x["mean"], -x["hi"], -x["lo"], x["p"], wald_p(x["mean"], (x["hi"] - x["lo"]) / 3.92), {"p_type": "image-cluster sign-flip (2000)"})
    for m in BB.MODELS:
        r = t_syn_pooled(m, rng, NB)
        if r is None:
            continue
        est, bs = r
        pub = en["syn"]["pooled"][m]
        add("syn_pooled_ddprime", m, est, *pub["d_dprime_ci"], boot_p(bs, est), wald_p(est, bs.std()), {"p_type": f"image-cluster bootstrap ({NB})", "boot_ci": [round(float(np.quantile(bs, .025)), 2), round(float(np.quantile(bs, .975)), 2)]})
    for m in BB.MODELS:
        r = t_twins(m, rng, NB)
        if r is None:
            continue
        est, bs = r
        add("twin_ddprime", m, est, *dp[m]["diff_ci"], boot_p(bs, est), wald_p(est, bs.std()), {"p_type": f"twin-pair bootstrap ({NB})", "boot_ci": [round(float(np.quantile(bs, .025)), 2), round(float(np.quantile(bs, .975)), 2)]})
    for m in BB.MODELS:
        est, bs = t_cvb(m, rng, NB)
        add("cvb3d_ddprime", m, est, np.quantile(bs, .025), np.quantile(bs, .975), boot_p(bs, est), wald_p(est, bs.std()), {"p_type": f"image-cluster bootstrap ({NB})"})
    for m in M7:
        r = t_glm(m, rng, 1000)
        for comp in ("occlusion", "vertical"):
            est, bs = r[comp]
            pub = gl[m]["cot-direct"][comp]
            add(f"cue_glm_{comp}", m, est, *pub["ci"], boot_p(bs, est), wald_p(est, bs.std()), {"p_type": "twin-pair bootstrap (1000)", "boot_ci": [round(float(np.quantile(bs, .025)), 2), round(float(np.quantile(bs, .975)), 2)]})
        print("glm done", m, flush=True)
    for key in ("p_resampling", "p_wald"):
        adj, sig = bh([t[key] for t in tests])
        for t, a, s in zip(tests, adj, sig):
            t[f"q_global_{key[2:]}"] = round(a, 4)
            t[f"survives_global_{key[2:]}"] = s
        for fam in sorted({t["family"] for t in tests}):
            ft = [t for t in tests if t["family"] == fam]
            adj, sig = bh([t[key] for t in ft])
            for t, a, s in zip(ft, adj, sig):
                t[f"q_family_{key[2:]}"] = round(a, 4)
                t[f"survives_family_{key[2:]}"] = s
    fams = {}
    for fam in dict.fromkeys(t["family"] for t in tests):
        ft = [t for t in tests if t["family"] == fam]
        fams[fam] = {"n": len(ft), "nominal_ci_excludes_0": sum(t["ci"][0] > 0 or t["ci"][1] < 0 for t in ft),
                     "survive_global_resampling": [t["model"] for t in ft if t["survives_global_resampling"]],
                     "survive_global_wald": [t["model"] for t in ft if t["survives_global_wald"]],
                     "fail_global_resampling_but_ci_excludes_0": [t["model"] for t in ft if not t["survives_global_resampling"] and (t["ci"][0] > 0 or t["ci"][1] < 0)],
                     "neg_survivors": [t["model"] for t in ft if t["survives_global_resampling"] and t["est"] < 0],
                     "pos_survivors": [t["model"] for t in ft if t["survives_global_resampling"] and t["est"] > 0]}
        f = fams[fam]
        lines.append(f"BH {fam:26s} n={f['n']} nominal CI!=0 {f['nominal_ci_excludes_0']} | survive q<0.05 (global, resampling p): {len(f['survive_global_resampling'])} "
                     f"(neg: {','.join(f['neg_survivors']) or '-'}; pos: {','.join(f['pos_survivors']) or '-'}) | wald: {len(f['survive_global_wald'])} "
                     f"| lost: {','.join(f['fail_global_resampling_but_ci_excludes_0']) or '-'}")
    cs = competence_slope()
    lines.append(f"SLOPE residual image use vs Direct acc ({cs['n_cells']} cells, {cs['n_models']} models; relation_sensitivity.json): pooled Spearman {cs['spearman_pooled']}; "
                 f"OLS {cs['ols_slope_per_10pts']} pts/10pts; mixedlm (random intercept per model) {cs['mixedlm_slope_per_10pts']}; within-model FE {cs['within_model_slope_per_10pts']}; "
                 f"between-model Spearman (7 means) {cs['between_model_spearman_7']}; cluster bootstrap over models slope CI {cs['cluster_bootstrap_over_models']['slope_per_10pts_ci']} "
                 f"rho CI {cs['cluster_bootstrap_over_models']['spearman_ci']}")
    ts = twin_share()
    for m, d in ts.items():
        lines.append(f"TWINSHARE {m:12s} " + " | ".join(f"{p}: gray {v['gray_share']} ; twin (post-pre)/(1-2pre) = ({v['post']}-{v['pre']})/(1-2*{v['pre']}) = {v['twin_share_norm']} "
                                                       f"[bal-denominator {v['twin_share_norm_bal_denominator']}]" for p, v in d.items()))
    res = {"tests": tests, "families": fams, "n_tests": len(tests), "q": 0.05,
           "notes": "BH over all per-model tests jointly (global) and within each family. Primary p: resampling (sign-flip for accuracy deltas; percentile "
                    "bootstrap for d' / GLM deltas, floor 1/(B+1)); secondary: Wald with bootstrap SE (CI half-width / 1.96 for Tab. 1). "
                    "Published CIs are kept in 'ci'; 'boot_ci' is this run's bootstrap.",
           "competence_slope": cs, "twin_swap_image_share": ts}
    json.dump(res, open(B / "multiplicity.json", "w"), indent=1)
    return res


# ----------------------------------------------------------------------------- rescue comparator

def wilson(k: int, n: int) -> list[float]:
    if n == 0:
        return [float("nan")] * 2
    z, p = 1.96, k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [round(100 * float(c - h), 1), round(100 * float(c + h), 1)]


def rescue(lines: list[str]) -> dict:
    from rigor_e17 import stype
    sets = {"rescue_DmCp": "_rescue", "DmCm_rel_rescuepos": "_rescue_dcm", "DpCp_rel_rescuepos": "_rescue_dcprel", "DpCp_rel_commitpos": "_relcomp"}
    res = {"sets": {}}
    exact = {}
    for m in M7:
        main = BB.load_main(O, L, m)
        exact[m] = {k: cell(bool(main["direct"][k][0]), bool(main["cot"][k][0])) for k in main["direct"] if k in main["cot"]}
    for name, suf in sets.items():
        J = {j: {(r["model"], r["item_id"]): r for r in rj(O / "e20" / f"{j}{suf}.jsonl")} for j in J2}
        if not all(J.values()):
            continue
        allk = set(J[J2[0]]) | set(J[J2[1]])
        failed = {j: sum(1 for k, r in J[j].items() if r["label"] == "UNPARSED" and (r.get("api_error") or r["raw"].startswith("HTTP"))) for j in J2}
        unparsed_text = {j: sum(1 for k, r in J[j].items() if r["label"] == "UNPARSED" and not (r.get("api_error") or r["raw"].startswith("HTTP"))) for j in J2}
        retried = {j: sum(1 for r in J[j].values() if r.get("retried")) for j in J2}
        out = {"file_suffix": suf, "failed_calls": failed, "unparsed_text_replies": unparsed_text, "retried": retried}
        for m in M7 + ["pooled"]:
            for rel_only in (False, True):
                ks = [k for k in allk if k in J[J2[0]] and k in J[J2[1]] and (m == "pooled" or k[0] == m)]
                if rel_only:
                    ks = [k for k in ks if stype(J[J2[1]][k]["sentence"]) == "relation"]
                if not ks:
                    continue
                lab = lambda j, k: J[j][k]["label"]
                cor = [k for k in ks if lab(J2[0], k) == lab(J2[1], k) == "CORRECT"]
                inc = [k for k in ks if lab(J2[0], k) == lab(J2[1], k) == "INCORRECT"]
                nj = len(cor) + len(inc)
                parsed = [k for k in ks if "UNPARSED" not in (lab(J2[0], k), lab(J2[1], k))]
                clear = [k for k in ks if {lab(J2[0], k), lab(J2[1], k)} <= {"CORRECT", "INCORRECT"}]
                a = np.array([lab(J2[0], k) == "INCORRECT" for k in clear])
                b = np.array([lab(J2[1], k) == "INCORRECT" for k in clear])
                kap = None
                if len(clear) > 5:
                    po, pe_ = np.mean(a == b), a.mean() * b.mean() + (1 - a.mean()) * (1 - b.mean())
                    kap = round(float((po - pe_) / (1 - pe_)), 3) if pe_ < 1 else None
                r = {"n": len(ks), "judgeable": nj, "consistent": round(100 * len(cor) / max(1, nj), 1), "consistent_ci": wilson(len(cor), nj),
                     "contradict": round(100 * len(inc) / max(1, nj), 1), "contradict_ci": wilson(len(inc), nj),
                     "agree_parsed": round(100 * float(np.mean([lab(J2[0], k) == lab(J2[1], k) for k in parsed])), 1) if parsed else None, "n_parsed_both": len(parsed),
                     "kappa_clear": kap,
                     "noclaim_gpt": round(100 * float(np.mean([lab(J2[1], k) == "NOCLAIM" for k in ks])), 1),
                     "relational_share": round(100 * float(np.mean([stype(J[J2[1]][k]["sentence"]) == "relation" for k in ks])), 1),
                     "mean_rel_pos": round(float(np.mean([J[J2[1]][k]["idx"] / max(1, J[J2[1]][k]["n_sent"] - 1) for k in ks])), 2)}
                if m == "pooled" and not rel_only:
                    want = {"rescue_DmCp": "D-C+", "DmCm_rel_rescuepos": "D-C-", "DpCp_rel_rescuepos": "D+C+", "DpCp_rel_commitpos": "D+C+"}[name]
                    r["same_cell_under_tab1_readout"] = round(100 * float(np.mean([exact[k[0]].get(k[1]) == want for k in ks])), 1)
                    kk = [k for k in ks if exact[k[0]].get(k[1]) == want]
                    c2 = sum(lab(J2[0], k) == lab(J2[1], k) == "CORRECT" for k in kk)
                    i2 = sum(lab(J2[0], k) == lab(J2[1], k) == "INCORRECT" for k in kk)
                    r["consistent_tab1_cell_only"] = [round(100 * c2 / max(1, c2 + i2), 1), c2 + i2]
                out[f"{m}{'|rel' if rel_only else ''}"] = r
        res["sets"][name] = out
        p, pr = out["pooled"], out.get("pooled|rel", {})
        lines.append(f"RESCUE-CMP {name:20s} n={p['n']} judgeable {p['judgeable']} consistent {p['consistent']} {p['consistent_ci']} contradict {p['contradict']} "
                     f"| relational-only {pr.get('consistent')} {pr.get('consistent_ci')} (nj {pr.get('judgeable')}) | agree {p['agree_parsed']}% (n {p['n_parsed_both']}) kappa {p['kappa_clear']} "
                     f"| failed calls {failed} unparsed text {unparsed_text} retried {retried} | rel.pos {p['mean_rel_pos']} | Tab.1-cell {p.get('same_cell_under_tab1_readout')}% -> {p.get('consistent_tab1_cell_only')}")
        lines.append(f"RESCUE-CMP {name:20s} per model consistent (nj): " + ", ".join(f"{m} {out[m]['consistent']:.0f} ({out[m]['judgeable']})" for m in M7 if m in out))
    # two-proportion contrasts (pooled, judgeable)
    S = res["sets"]

    def contrast(a, b, key="pooled"):
        x, y = S[a][key], S[b][key]
        k1, n1 = round(x["consistent"] * x["judgeable"] / 100), x["judgeable"]
        k2, n2 = round(y["consistent"] * y["judgeable"] / 100), y["judgeable"]
        from scipy.stats import fisher_exact
        return {"diff_pts": round(100 * (k1 / n1 - k2 / n2), 1), "fisher_p": float(f"{fisher_exact([[k1, n1 - k1], [k2, n2 - k2]])[1]:.3g}")}
    res["contrasts"] = {}
    for a, b in (("rescue_DmCp", "DmCm_rel_rescuepos"), ("rescue_DmCp", "DpCp_rel_rescuepos"), ("rescue_DmCp", "DpCp_rel_commitpos"),
                 ("DpCp_rel_rescuepos", "DmCm_rel_rescuepos")):
        if a in S and b in S:
            res["contrasts"][f"{a} - {b}"] = contrast(a, b)
            if "pooled|rel" in S[a] and "pooled|rel" in S[b]:
                res["contrasts"][f"{a} - {b} (relational only)"] = contrast(a, b, "pooled|rel")
    lines.append("RESCUE-CMP contrasts (consistent share, pooled judgeable; Fisher exact): " + "; ".join(f"{k} {v['diff_pts']:+.1f} p={v['fisher_p']}" for k, v in res["contrasts"].items()))
    json.dump(res, open(B / "rescue_comparator.json", "w"), indent=1)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="churn,facts,mult,rescue")
    a = ap.parse_args()
    lines = []
    fn = {"churn": churn, "facts": facts, "mult": mult, "rescue": rescue}
    pref = {"churn": ("CHURN",), "facts": ("FACT",), "mult": ("BH ", "SLOPE", "TWINSHARE"), "rescue": ("RESCUE-CMP",)}
    for k in a.only.split(","):
        fn[k](lines)
    p = B / "rigor_e17o.txt"
    old = p.read_text().splitlines() if p.exists() else []
    keep = [x for x in old if not any(x.startswith(pf) for k in a.only.split(",") for pf in pref[k])]
    p.write_text("\n".join(keep + lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
