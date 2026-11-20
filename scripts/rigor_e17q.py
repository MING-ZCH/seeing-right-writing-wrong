"""Exp-23 / round-13 CPU analyses (Opus round-13 M1 a-d, A2, m9, m10).

hetero   Common-sensitivity bound: under P(yes) = sigma(b_i +- Delta/2) with one Delta, the share of images answered
         differently under the two object orders (BC + BW) is at most sigma(D/2)^2 + sigma(-D/2)^2 (reached at b_i = 0).
         Observed share vs the bound at the observed pair log-OR, per model x arm, overall and per size tercile; image bootstrap.
strat    Size-stratified pair-matched sensitivity on Synthetic Depth (visible-area-ratio terciles / quintiles of the 486
         images with both shapes visible). (i) Mantel-Haenszel common OR over strata of matched pairs = sum BC / sum BW
         (each image is its own 1:1 matched stratum, so MH is invariant to any grouping of images and equals the pooled
         d'_pair; checked numerically). (ii) Size-standardised d'_pair: image-share-weighted mean of within-stratum
         log((BC+.5)/(BW+.5)) / 1.702, weights fixed across arms, so CoT cannot lower it by moving discordant images
         between strata. (iii) Per-stratum d'_pair. Image bootstrap (2,000), percentile CIs, two-sided bootstrap p.
glmm     Logistic GLMM with an image random intercept per arm (correlated) and an image random sensitivity slope
         (arm-specific scale, perfectly correlated across arms), fixed tercile-specific mean bias and mean sensitivity per
         arm; marginal likelihood by Gauss-Hermite quadrature over the 16 joint answer patterns; mean d' = tercile-
         averaged Delta / 1.702. Image bootstrap (NB_GLMM).
mult     Adds the Synthetic Depth Delta d'_pair family to the 53-test BH family of multiplicity.json (backup kept).
points   Accuracy at an unbiased criterion Phi(d'/2) for pooled d' (Sec. 4 old numbers), d'_pair and size-standardised d'_pair.
crit     Single-phrasing criterion shift (Tab. S23, bias_sensitivity.json): which Delta c CIs exclude 0.
twins    Original occlusion twins, pair-matched: unit = (scene, phrasing), the two identity-swapped images have opposite
         truth for "the subject is in front". Four-way split (BC / subject-front both / subject-behind both / BW),
         d'_pair, MH and cell-standardised d'_pair over the 2x2 cue cells (vertical: front lower / higher x drawn area
         ratio < / >= 0.77, e13 cells), vertical-balanced d'_pair and vertical component per arm. Scene bootstrap (2,000).
-> bundle/syn_pair_sdt_mh.json, bundle/rigor_e17q.txt, bundle/multiplicity.json (old: bundle/multiplicity_pre_e17q.json)
Usage: python rigor_e17q.py [--nb-glmm 200] [--skip-glmm]
"""
from __future__ import annotations

import argparse
import json
import shutil
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit
from scipy.stats import norm

import build_bundle as BB
from rigor_e17o import bh, boot_p, wald_p
from rigor_e17p import M7, O, B, LOGIT_PROBIT, load_pairs, outcome, ratio_terciles

NB = 2000
NAMES = {"qwen3b": "Qwen2.5-VL-3B", "qwen7b": "Qwen2.5-VL-7B", "internvl3_8b": "InternVL3-8B", "phi35v": "Phi-3.5-Vision",
         "molmo7b": "Molmo-7B-D", "qwen3vl_8b": "Qwen3-VL-8B", "qwen3vl_8b_think": "Qwen3-VL-8B-Thinking", "internvl35_8b": "InternVL3.5-8B"}
M8 = ["qwen3b", "qwen7b", "internvl3_8b", "phi35v", "molmo7b", "qwen3vl_8b", "qwen3vl_8b_think", "internvl35_8b"]


def q95(v, p: int = 3) -> list[float]:
    return [round(float(np.nanquantile(v, .025)), p), round(float(np.nanquantile(v, .975)), p)]


def lor(bc, bw):
    return np.log((np.asarray(bc, float) + .5) / (np.asarray(bw, float) + .5))


def disc_bound(lo: float) -> float:
    s = expit(lo / 2)
    return float(s * s + (1 - s) ** 2)


# ----------------------------------------------------------------------------- strata

def strata_labels(keys: list, ratio: dict, nq: int) -> tuple[np.ndarray, list[float]]:
    """Quantile stratum per image (0..nq-1) on the visible-area ratio; -1 = a shape totally occluded (no ratio)."""
    cuts = list(np.quantile(np.array(list(ratio.values())), [i / nq for i in range(1, nq)]))
    lab = np.array([int(np.searchsorted(cuts, ratio[k], side="right")) if k in ratio else -1 for k in keys])
    return lab, [round(float(c), 2) for c in cuts]


def counts(oc: np.ndarray, lab: np.ndarray, nq: int) -> np.ndarray:
    """(nq, 4) outcome counts per stratum (images with lab = -1 dropped)."""
    s = lab >= 0
    return np.bincount(lab[s] * 4 + oc[s], minlength=nq * 4).reshape(nq, 4).astype(float)


def strat_stats(oc_d, oc_c, lab3, lab5, w3, w5) -> np.ndarray:
    """Statistic vector for one (re)sample; names in STRAT_NAMES."""
    out = []
    for oc in (oc_d, oc_c):
        allc = np.bincount(oc, minlength=4).astype(float)
        c3, c5 = counts(oc, lab3, 3), counts(oc, lab5, 5)
        l3, l5 = lor(c3[:, 0], c3[:, 3]), lor(c5[:, 0], c5[:, 3])
        vis = c3.sum(0)
        out.append([lor(allc[0], allc[3]), lor(vis[0], vis[3]),
                    np.log(c3[:, 0].sum() / max(c3[:, 3].sum(), 1e-9)), np.log(c5[:, 0].sum() / max(c5[:, 3].sum(), 1e-9)),
                    np.log(vis[0] / max(vis[3], 1e-9)),
                    float(w3 @ l3), float(w5 @ l5), *l3, *l5])
    d, c = np.array(out[0]), np.array(out[1])
    return np.concatenate([d, c, c - d]) / LOGIT_PROBIT


_S = (["pooled_all", "pooled_vis", "mh_terc_raw", "mh_quint_raw", "pooled_vis_raw", "std_terc", "std_quint"]
      + [f"terc{i}" for i in range(3)] + [f"quint{i}" for i in range(5)])
STRAT_NAMES = [f"{a}_{s}" for a in ("direct", "cot", "delta") for s in _S]


def hetero_stats(oc: np.ndarray, lab3: np.ndarray) -> np.ndarray:
    """[observed BC+BW share, bound at observed log-OR, excess] overall and per tercile."""
    out = []
    for s in [np.ones(len(oc), bool)] + [lab3 == t for t in range(3)]:
        c = np.bincount(oc[s], minlength=4)
        n = c.sum()
        obs = (c[0] + c[3]) / n
        bd = disc_bound(float(lor(c[0], c[3])))
        out += [100 * obs, 100 * bd, 100 * (obs - bd)]
    return np.array(out)


def run_strat(lines: list[str], P: dict) -> dict:
    rng = np.random.default_rng(23)
    ratio, t1, t2 = ratio_terciles()
    res = {"_doc": {
        "units": "d' units (log-OR / 1.702); delta = CoT - Direct",
        "pooled_all": "log((BC+.5)/(BW+.5))/1.702 over all 500 images (= Tab. S25 d'_pair)",
        "pooled_vis": "same over the 486 images with both shapes visible (the stratified images)",
        "mh_terc_raw / mh_quint_raw": "Mantel-Haenszel common OR across strata of 1:1 matched pairs = sum_t BC_t / sum_t BW_t (no 0.5 correction); "
                                      "identical to pooled_vis_raw for any stratification (each image is its own matched stratum), checked",
        "std_terc / std_quint": "size-standardised d'_pair: sum_t w_t log((BC_t+.5)/(BW_t+.5)) / 1.702, w_t = image share of stratum t "
                                "(fixed across arms and bootstrap at the full-sample shares)",
        "terc_i / quint_i": "within-stratum d'_pair (0 = smallest visible-area ratio)",
        "hetero": "observed BC+BW share (%) vs the common-Delta maximum 100*(sigma(D/2)^2 + sigma(-D/2)^2) at the observed log-OR D; excess = observed - bound",
        "bootstrap": f"{NB} image resamples (paired across arms, strata follow the image); seed 23; p = two-sided percentile bootstrap p (floor 1/(B+1))"}}
    for m in M7:
        d = P[m]
        g = d["gold_yes"]
        oc_d, oc_c = outcome(d["direct_o"], d["direct_s"], g), outcome(d["cot_o"], d["cot_s"], g)
        lab3, cut3 = strata_labels(d["keys"], ratio, 3)
        lab5, cut5 = strata_labels(d["keys"], ratio, 5)
        w3 = np.bincount(lab3[lab3 >= 0], minlength=3) / (lab3 >= 0).sum()
        w5 = np.bincount(lab5[lab5 >= 0], minlength=5) / (lab5 >= 0).sum()
        est = strat_stats(oc_d, oc_c, lab3, lab5, w3, w5)
        he = {a: hetero_stats(oc, lab3) for a, oc in (("direct", oc_d), ("cot", oc_c))}
        n = len(g)
        bs, bh_ = [], {"direct": [], "cot": []}
        for _ in range(NB):
            i = rng.integers(0, n, n)
            bs.append(strat_stats(oc_d[i], oc_c[i], lab3[i], lab5[i], w3, w5))
            for a, oc in (("direct", oc_d), ("cot", oc_c)):
                bh_[a].append(hetero_stats(oc[i], lab3[i]))
        bs = np.array(bs)
        r = {"n_images": n, "n_visible": int((lab3 >= 0).sum()), "tercile_cuts": cut3, "quintile_cuts": cut5,
             "tercile_n": [int(x) for x in np.bincount(lab3[lab3 >= 0], minlength=3)], "quintile_n": [int(x) for x in np.bincount(lab5[lab5 >= 0], minlength=5)]}
        for j, nm in enumerate(STRAT_NAMES):
            r[nm] = round(float(est[j]), 3)
            r[nm + "_ci"] = q95(bs[:, j])
            if nm.startswith("delta_"):
                r[nm + "_p"] = round(boot_p(bs[:, j], est[j]), 5)
                r[nm + "_p_wald"] = round(wald_p(est[j], float(np.nanstd(bs[:, j]))), 5)
        r["mh_equals_pooled_check"] = bool(abs(r["direct_mh_terc_raw"] - r["direct_pooled_vis_raw"]) < 1e-9 and abs(r["cot_mh_quint_raw"] - r["cot_pooled_vis_raw"]) < 1e-9)
        # tercile cells for the record
        r["tercile_counts"] = {a: counts(oc, lab3, 3).astype(int).tolist() for a, oc in (("direct", oc_d), ("cot", oc_c))}
        hn = ["all", "low", "mid", "high"]
        r["hetero"] = {}
        for a in ("direct", "cot"):
            hb = np.array(bh_[a])
            r["hetero"][a] = {t: {"observed_disc_pct": round(float(he[a][3 * k]), 1), "bound_pct": round(float(he[a][3 * k + 1]), 1),
                                  "excess_pts": round(float(he[a][3 * k + 2]), 1), "excess_ci": q95(hb[:, 3 * k + 2], 1),
                                  "exceeds": bool(np.quantile(hb[:, 3 * k + 2], .025) > 0)} for k, t in enumerate(hn)}
        # where CoT is at chance among discordant pairs (per-stratum CoT d'_pair CI includes 0)
        r["cot_chance_terciles"] = [t for k, t in enumerate(("low", "mid", "high")) if r[f"cot_terc{k}_ci"][0] <= 0]
        r["cot_chance_quintiles"] = [k + 1 for k in range(5) if r[f"cot_quint{k}_ci"][0] <= 0]
        res[m] = r
        f = lambda k, p=2: f"{r[k]:+.{p}f} [{r[k + '_ci'][0]:+.{p}f}, {r[k + '_ci'][1]:+.{p}f}]"
        h = r["hetero"]
        lines.append(f"HETERO {m:14s} " + " | ".join(
            f"{a}: all obs {h[a]['all']['observed_disc_pct']:.1f} vs bound {h[a]['all']['bound_pct']:.1f} (excess {h[a]['all']['excess_pts']:+.1f} {h[a]['all']['excess_ci']}{' EXCEEDS' if h[a]['all']['exceeds'] else ''}); "
            + "/".join(f"{t} {h[a][t]['excess_pts']:+.1f}{'*' if h[a][t]['exceeds'] else ''}" for t in ("low", "mid", "high")) for a in ("direct", "cot")))
        lines.append(f"STRAT {m:14s} pooled d'pair (500) {r['direct_pooled_all']:.2f}->{r['cot_pooled_all']:.2f} ({f('delta_pooled_all')}) | 486 vis {f('delta_pooled_vis')} "
                     f"| MH over terciles = sum BC/sum BW (check {r['mh_equals_pooled_check']}) {r['direct_mh_terc_raw']:.2f}->{r['cot_mh_terc_raw']:.2f} "
                     f"| size-standardised terciles {r['direct_std_terc']:.2f}->{r['cot_std_terc']:.2f} ({f('delta_std_terc')}, p {r['delta_std_terc_p']}) "
                     f"| quintiles {r['direct_std_quint']:.2f}->{r['cot_std_quint']:.2f} ({f('delta_std_quint')}, p {r['delta_std_quint_p']})")
        lines.append(f"STRAT-TERC {m:14s} cuts {cut3} n {r['tercile_n']} | " + " | ".join(
            f"{t}: {r[f'direct_terc{k}']:.2f} {r[f'direct_terc{k}_ci']} -> {r[f'cot_terc{k}']:.2f} {r[f'cot_terc{k}_ci']} (delta {f(f'delta_terc{k}')})"
            for k, t in enumerate(("low", "mid", "high"))) + f" | CoT CI includes 0: {r['cot_chance_terciles']}")
        lines.append(f"STRAT-QUINT {m:14s} cuts {cut5} | " + " | ".join(
            f"Q{k + 1}: {r[f'direct_quint{k}']:.2f}->{r[f'cot_quint{k}']:.2f} ({f(f'delta_quint{k}')})" for k in range(5)) + f" | CoT CI includes 0: Q{r['cot_chance_quintiles']}")
    sig = lambda k: [m for m in M7 if res[m][k + "_ci"][1] < 0]
    pos = lambda k: [m for m in M7 if res[m][k + "_ci"][0] > 0]
    res["_summary"] = {k: {"fall_ci_lt0": sig(f"delta_{k}"), "rise_ci_gt0": pos(f"delta_{k}"),
                           "range_falls": [min((res[m][f"delta_{k}"] for m in sig(f"delta_{k}")), default=None), max((res[m][f"delta_{k}"] for m in sig(f"delta_{k}")), default=None)]}
                       for k in ("pooled_all", "pooled_vis", "std_terc", "std_quint", "terc0", "terc1", "terc2")}
    res["_summary"]["direct_exceeds_bound"] = [m for m in M7 if res[m]["hetero"]["direct"]["all"]["exceeds"]]
    res["_summary"]["cot_exceeds_bound"] = [m for m in M7 if res[m]["hetero"]["cot"]["all"]["exceeds"]]
    res["_summary"]["direct_exceeds_bound_within_tercile"] = {m: [t for t in ("low", "mid", "high") if res[m]["hetero"]["direct"][t]["exceeds"]] for m in M7}
    s = res["_summary"]
    for k in ("pooled_all", "std_terc", "std_quint", "terc0", "terc1", "terc2"):
        lines.append(f"STRAT-SUM {k:10s} fall (CI<0) {len(s[k]['fall_ci_lt0'])}/7 {s[k]['fall_ci_lt0']} range {s[k]['range_falls']} | rise {s[k]['rise_ci_gt0']}")
    lines.append(f"HETERO-SUM common-Delta bound exceeded (excess CI > 0): Direct {s['direct_exceeds_bound']} | CoT {s['cot_exceeds_bound']} | Direct within tercile {s['direct_exceeds_bound_within_tercile']}")
    return res


# ----------------------------------------------------------------------------- GLMM

GH_X, GH_W = np.polynomial.hermite_e.hermegauss(10)
GH_W = GH_W / GH_W.sum()
Z1, Z2, Z3 = (a.ravel() for a in np.meshgrid(GH_X, GH_X, GH_X, indexing="ij"))
WN = np.einsum("i,j,k->ijk", GH_W, GH_W, GH_W).ravel()
# pattern bits: (yD+, yD-, yC+, yC-), + = order whose gold is 'yes'
PAT = np.array([[(p >> 3) & 1, (p >> 2) & 1, (p >> 1) & 1, p & 1] for p in range(16)], float)


def glmm_nll(th: np.ndarray, cnt: np.ndarray) -> float:
    """cnt: (3 terciles, 16 patterns). th: mu_D[3], mu_C[3], Dl_D[3], Dl_C[3], log sD, log sC, atanh rho, log suD, log suC."""
    mD, mC, DD, DC = th[0:3], th[3:6], th[6:9], th[9:12]
    sD, sC, rho, uD, uC = np.exp(th[12]), np.exp(th[13]), np.tanh(th[14]), np.exp(th[15]), np.exp(th[16])
    nll = 0.0
    for t in range(3):
        bD = mD[t] + sD * Z1
        bC = mC[t] + sC * (rho * Z1 + np.sqrt(1 - rho * rho) * Z2)
        dD, dC = (DD[t] + uD * Z3) / 2, (DC[t] + uC * Z3) / 2
        p = np.stack([expit(bD + dD), expit(bD - dD), expit(bC + dC), expit(bC - dC)])  # (4, nodes)
        lp, lq = np.log(np.clip(p, 1e-300, 1)), np.log(np.clip(1 - p, 1e-300, 1))
        L = np.exp(PAT @ lp + (1 - PAT) @ lq) @ WN  # (16,)
        nll -= float(cnt[t] @ np.log(np.clip(L, 1e-300, None)))
    return nll


def glmm_fit(cnt: np.ndarray, x0: np.ndarray | None = None) -> np.ndarray:
    if x0 is None:
        x0 = np.concatenate([np.zeros(6), np.full(6, 2.0), [0.0, 0.0, 0.3, -0.5, -0.5]])
    r = minimize(glmm_nll, x0, args=(cnt,), method="L-BFGS-B",
                 bounds=[(-10, 10)] * 6 + [(-5, 15)] * 6 + [(-4, 2.5)] * 2 + [(-3, 3)] + [(-6, 2.5)] * 2, options={"maxiter": 3000})
    return r.x


def glmm_summ(th: np.ndarray) -> np.ndarray:
    DD, DC = th[6:9], th[9:12]
    return np.concatenate([[DD.mean(), DC.mean(), DC.mean() - DD.mean()], DD, DC, DC - DD, [np.exp(th[15]), np.exp(th[16])]]) / np.r_[np.full(12, LOGIT_PROBIT), 1.702, 1.702]


GLMM_NAMES = (["dmean_direct", "dmean_cot", "dmean_delta"] + [f"direct_terc{i}" for i in range(3)] + [f"cot_terc{i}" for i in range(3)]
              + [f"delta_terc{i}" for i in range(3)] + ["slope_sd_direct", "slope_sd_cot"])


def run_glmm(lines: list[str], P: dict, nb: int) -> dict:
    rng = np.random.default_rng(231)
    ratio, _, _ = ratio_terciles()
    res = {"_doc": "logit P(yes | image i, order s, arm a) = b_ia + s (Delta_{a,t(i)} + sigma_ua z_i)/2, (b_iD, b_iC) bivariate normal with tercile-specific means, "
                   "z_i ~ N(0,1) shared across arms (image-specific sensitivity deviation), fixed Delta per arm x tercile; 10^3-node Gauss-Hermite; "
                   f"mean d' = tercile-averaged Delta / 1.702 (486 visible images); image bootstrap ({nb}), cold-started; slope_sd in d' units. "
                   "Weakly identified (two answers per image and arm): the slope SD and bias SD/correlation can sit at their bounds, so the mean d' "
                   "depends on the normal random-effects assumption; exploratory.",
           "n_boot": nb}
    for m in M7:
        d = P[m]
        g = d["gold_yes"]
        yp = lambda a: np.where(g, d[f"{a}_o"], d[f"{a}_s"]).astype(int)
        yn = lambda a: np.where(g, d[f"{a}_s"], d[f"{a}_o"]).astype(int)
        pat = 8 * yp("direct") + 4 * yn("direct") + 2 * yp("cot") + yn("cot")
        lab, _ = strata_labels(d["keys"], ratio, 3)
        s = lab >= 0
        pat, lab = pat[s], lab[s]
        cnt = np.bincount(lab * 16 + pat, minlength=48).reshape(3, 16).astype(float)
        th = glmm_fit(cnt)
        est = glmm_summ(th)
        bs = []
        for _ in range(nb):
            i = rng.integers(0, len(pat), len(pat))
            c = np.bincount(lab[i] * 16 + pat[i], minlength=48).reshape(3, 16).astype(float)
            bs.append(glmm_summ(glmm_fit(c)))
        bs = np.array(bs)
        r = {"nll": round(glmm_nll(th, cnt), 2), "params": [round(float(x), 3) for x in th]}
        for j, nm in enumerate(GLMM_NAMES):
            r[nm] = round(float(est[j]), 3)
            r[nm + "_ci"] = q95(bs[:, j])
        r["dmean_delta_p"] = round(boot_p(bs[:, 2], est[2]), 5)
        res[m] = r
        lines.append(f"GLMM {m:14s} mean d' {r['dmean_direct']:.2f} -> {r['dmean_cot']:.2f} (delta {r['dmean_delta']:+.2f} {r['dmean_delta_ci']}, p {r['dmean_delta_p']}) "
                     f"| per tercile delta " + "/".join(f"{r[f'delta_terc{k}']:+.2f}" for k in range(3))
                     + f" | slope SD (d') Dir {r['slope_sd_direct']:.2f} {r['slope_sd_direct_ci']} CoT {r['slope_sd_cot']:.2f}")
        print("glmm", m, flush=True)
    res["_summary"] = {"fall_ci_lt0": [m for m in M7 if res[m]["dmean_delta_ci"][1] < 0], "rise_ci_gt0": [m for m in M7 if res[m]["dmean_delta_ci"][0] > 0]}
    lines.append(f"GLMM-SUM mean-d' fall (CI<0) {len(res['_summary']['fall_ci_lt0'])}/7 {res['_summary']['fall_ci_lt0']} | rise {res['_summary']['rise_ci_gt0']}")
    return res


# ----------------------------------------------------------------------------- multiplicity

FAM = "syn_dpair_ddprime"


def run_mult(lines: list[str], ST: dict, key: str) -> dict:
    path = B / "multiplicity.json"
    bak = B / "multiplicity_pre_e17q.json"
    if not bak.exists():
        shutil.copy(path, bak)
    mu = json.load(open(bak))
    tests = [t for t in mu["tests"] if t["family"] != FAM]
    old = {"n_tests": len(tests), "nominal": sum(t["ci"][0] > 0 or t["ci"][1] < 0 for t in tests),
           "survive": sum(t["survives_global_resampling"] for t in tests), "survive_wald": sum(t["survives_global_wald"] for t in tests)}
    pub = json.load(open(B / "syn_pair_sdt.json"))["pairsdt"]
    for m in M7:
        r = ST[m]
        ci = pub[m]["dpair_logit_delta_ci"] if key == "pooled_all" else r[f"delta_{key}_ci"]
        tests.append({"family": FAM, "model": m, "est": r[f"delta_{key}"], "ci": ci, "boot_ci": r[f"delta_{key}_ci"], "p_resampling": r[f"delta_{key}_p"],
                      "p_wald": r[f"delta_{key}_p_wald"], "p_type": f"image bootstrap ({NB}), statistic {key} (rigor_e17q)"})
    for kk in ("p_resampling", "p_wald"):
        adj, sg = bh([t[kk] for t in tests])
        for t, a, s in zip(tests, adj, sg):
            t[f"q_global_{kk[2:]}"] = round(a, 4)
            t[f"survives_global_{kk[2:]}"] = s
        for fam in {t["family"] for t in tests}:
            ft = [t for t in tests if t["family"] == fam]
            adj, sg = bh([t[kk] for t in ft])
            for t, a, s in zip(ft, adj, sg):
                t[f"q_family_{kk[2:]}"] = round(a, 4)
                t[f"survives_family_{kk[2:]}"] = s
    fams = {}
    for fam in dict.fromkeys(t["family"] for t in tests):
        ft = [t for t in tests if t["family"] == fam]
        fams[fam] = {"n": len(ft), "nominal_ci_excludes_0": sum(t["ci"][0] > 0 or t["ci"][1] < 0 for t in ft),
                     "survive_global_resampling": [t["model"] for t in ft if t["survives_global_resampling"]],
                     "survive_global_wald": [t["model"] for t in ft if t["survives_global_wald"]],
                     "fail_global_resampling_but_ci_excludes_0": [t["model"] for t in ft if not t["survives_global_resampling"] and (t["ci"][0] > 0 or t["ci"][1] < 0)],
                     "neg_survivors": [t["model"] for t in ft if t["survives_global_resampling"] and t["est"] < 0],
                     "pos_survivors": [t["model"] for t in ft if t["survives_global_resampling"] and t["est"] > 0]}
    mu["tests"], mu["families"], mu["n_tests"] = tests, fams, len(tests)
    mu["notes"] += (f" Exp-23 (rigor_e17q.py): family {FAM} = Synthetic Depth CoT-Direct d'_pair ({key}), image bootstrap ({NB}) percentile p; "
                    "pre-Exp-23 file: multiplicity_pre_e17q.json.")
    json.dump(mu, open(path, "w"), indent=1)
    nom = sum(t["ci"][0] > 0 or t["ci"][1] < 0 for t in tests)
    sv = sum(t["survives_global_resampling"] for t in tests)
    svw = sum(t["survives_global_wald"] for t in tests)
    lost = [(t["family"], t["model"], t["q_global_resampling"]) for t in tests if not t["survives_global_resampling"] and (t["ci"][0] > 0 or t["ci"][1] < 0)]
    lostw = [(t["family"], t["model"], t["q_global_wald"]) for t in tests if not t["survives_global_wald"] and (t["ci"][0] > 0 or t["ci"][1] < 0)]
    f = fams[FAM]
    out = {"statistic": key, "old": old, "new": {"n_tests": len(tests), "nominal": nom, "survive": sv, "survive_wald": svw},
           "lost_resampling": lost, "lost_wald": lostw,
           "dpair_family": {t["model"]: {"est": t["est"], "ci": t["ci"], "p": t["p_resampling"], "q": t["q_global_resampling"], "survives": t["survives_global_resampling"],
                                         "q_wald": t["q_global_wald"], "survives_wald": t["survives_global_wald"]} for t in tests if t["family"] == FAM}}
    lines.append(f"BH-NEW {FAM} ({key}) n=7 nominal {f['nominal_ci_excludes_0']} survive {len(f['survive_global_resampling'])} (neg {f['neg_survivors']}, pos {f['pos_survivors']}) "
                 f"| global: {old['n_tests']} -> {len(tests)} tests, nominal {old['nominal']} -> {nom}, survive (resampling) {old['survive']} -> {sv}, Wald {old['survive_wald']} -> {svw} "
                 f"| not surviving (resampling): {lost} | (Wald): {lostw}")
    lines.append("BH-NEW dpair family: " + "; ".join(f"{m} {v['est']:+.2f} {v['ci']} p {v['p']:.4f} q {v['q']:.4f}{'' if v['survives'] else ' FAILS'} (Wald q {v['q_wald']:.4f})"
                                                     for m, v in out["dpair_family"].items()))
    return out


def alt_mult(ST: dict, key: str) -> dict:
    """Survivor counts if the other d'_pair statistic were the family (no file written)."""
    mu = json.load(open(B / "multiplicity_pre_e17q.json"))
    tests = [t for t in mu["tests"] if t["family"] != FAM]
    ps = [t["p_resampling"] for t in tests] + [ST[m][f"delta_{key}_p"] for m in M7]
    cis = [t["ci"] for t in tests] + [ST[m][f"delta_{key}_ci"] for m in M7]
    _, sg = bh(ps)
    return {"statistic": key, "n_tests": len(ps), "nominal": sum(c[0] > 0 or c[1] < 0 for c in cis), "survive": int(sum(sg)),
            "dpair_survive": [m for m, s in zip(M7, sg[-7:]) if s]}


# ----------------------------------------------------------------------------- points, criterion

def run_points(lines: list[str], ST: dict) -> dict:
    en = json.load(open(B / "e17n.json"))["syn"]["pooled"]
    acc = lambda dp: 100 * norm.cdf(dp / 2)
    res = {"_doc": "accuracy at an unbiased criterion Phi(d'/2); points = 100 [Phi(d'_CoT/2) - Phi(d'_Dir/2)]"}
    for m in M7:
        r = ST[m]
        res[m] = {"pooled_dprime": round(acc(en[m]["dprime_cot"]) - acc(en[m]["dprime_direct"]), 1),
                  "pooled_dprime_e17n_loss": en[m]["loss_from_sensitivity_pooled"],
                  "dpair": round(acc(r["cot_pooled_all"]) - acc(r["direct_pooled_all"]), 1),
                  "dpair_std_terc": round(acc(r["cot_std_terc"]) - acc(r["direct_std_terc"]), 1),
                  "dpair_acc_direct_cot": [round(acc(r["direct_pooled_all"]), 1), round(acc(r["cot_pooled_all"]), 1)],
                  "dpair_std_terc_acc_direct_cot": [round(acc(r["direct_std_terc"]), 1), round(acc(r["cot_std_terc"]), 1)]}
    fall = lambda k: [m for m in M7 if ST[m][f"delta_{k}_ci"][1] < 0]
    rng_ = lambda key, ms: [min(-res[m][key] for m in ms), max(-res[m][key] for m in ms)] if ms else None
    res["_summary"] = {"pooled_dprime_loss_range": rng_("pooled_dprime", [m for m in M7 if res[m]["pooled_dprime"] < 0]),
                       "dpair_loss_range_falls": rng_("dpair", fall("pooled_all")),
                       "dpair_std_terc_loss_range_falls": rng_("dpair_std_terc", fall("std_terc"))}
    s = res["_summary"]
    lines.append(f"POINTS acc at unbiased criterion, CoT-Dir loss (pts): pooled d' {s['pooled_dprime_loss_range']} (Sec. 4 7.6-26.9) | d'pair {s['dpair_loss_range_falls']} | size-std d'pair {s['dpair_std_terc_loss_range_falls']} | per model "
                 + ", ".join(f"{m} {res[m]['pooled_dprime']:+.1f}/{res[m]['dpair']:+.1f}/{res[m]['dpair_std_terc']:+.1f}" for m in M7))
    return res


def run_crit(lines: list[str]) -> dict:
    bs = json.load(open(B / "bias_sensitivity.json"))
    res = {"_doc": "Tab. S23 (bias_sensitivity.json cot_vs_direct): c = -(z(H)+z(F))/2 on single-phrasing 'yes'; delta c < 0 = toward 'yes'"}
    for m in M8:
        v = bs[m]["cot_vs_direct"]
        ci = v["d_c_ci"]
        res[m] = {"d_c": v["d_c"], "d_c_ci": ci, "c_direct": bs[m]["direct"]["c"], "c_cot": bs[m]["cot"]["c"],
                  "significant": bool(ci[0] > 0 or ci[1] < 0), "direction": "yes" if ci[1] < 0 else "no" if ci[0] > 0 else "n.s."}
    res["_summary"] = {"toward_yes": [m for m in M8 if res[m]["direction"] == "yes"], "toward_no": [m for m in M8 if res[m]["direction"] == "no"]}
    lines.append("CRIT Tab. S23 delta c (CI excl. 0): toward yes " + str(res["_summary"]["toward_yes"]) + " | toward no " + str(res["_summary"]["toward_no"]) + " | "
                 + ", ".join(f"{m} {res[m]['d_c']:+.2f} {res[m]['d_c_ci']}" for m in M8))
    return res


# ----------------------------------------------------------------------------- twins

def twin_units(m: str, cond: str) -> dict:
    """{(scene, phrasing): {'t': [says_front, truth] for the original image, 'f': ... for the identity-swapped twin, 'above', 'ratio'}}."""
    U = defaultdict(dict)
    for ph, fn in (("front", f"{m}.jsonl"), ("behind", f"{m}_behind.jsonl")):
        for r in BB.rj(O / "e2d" / fn):
            if r["condition"] != cond:
                continue
            p = BB.pe(r)
            mt = r["meta"]
            u = U[(r["image_id"], ph)]
            says = (p == "A") if ph == "front" else (p == "B")
            u["cf" if r["item_id"].endswith("_cf") else "orig"] = (bool(says), mt["front_name"] == mt["x_name"], p == "A")
            if not r["item_id"].endswith("_cf"):
                u["above"], u["ratio"] = bool(mt["front_above"]), float(mt["area_ratio_front_over_back"])
    return U


def twin_stats(oc_d, oc_c, cell, wc, vert) -> np.ndarray:
    out = []
    for oc in (oc_d, oc_c):
        c = np.bincount(oc, minlength=4).astype(float)
        cc = np.bincount(cell * 4 + oc, minlength=16).reshape(4, 4).astype(float)
        lc = lor(cc[:, 0], cc[:, 3])
        cv = np.bincount(vert * 4 + oc, minlength=8).reshape(2, 4).astype(float)
        lv = lor(cv[:, 0], cv[:, 3])
        out.append([lor(c[0], c[3]), np.log(cc[:, 0].sum() / max(cc[:, 3].sum(), 1e-9)), float(wc @ lc), lv.mean(), (lv[0] - lv[1]) / 2, *lv, *lc])
    d, c = np.array(out[0]), np.array(out[1])
    return np.concatenate([d, c, c - d]) / LOGIT_PROBIT


_TW = ["pooled", "mh_raw", "std_cells", "vert_balanced", "vert_component", "lower", "higher", "c_lower_small", "c_lower_large", "c_higher_small", "c_higher_large"]
TW_NAMES = [f"{a}_{s}" for a in ("direct", "cot", "delta") for s in _TW]


def run_twins(lines: list[str]) -> dict:
    rng = np.random.default_rng(232)
    dp = json.load(open(B / "dprime_ci.json"))
    res = {"_doc": {
        "unit": "(scene, phrasing): the original image and its identity-swapped twin asked the same question; truth of 'subject X is in front' is opposite; 400 units / model / arm",
        "outcomes": "BC both correct; SF subject called front in both images; SB subject called behind in both; BW both wrong. "
                    "YY / NN = 'yes' / 'no' to both images (SF+front phrasing or SB+behind phrasing = YY)",
        "dpair": "log((BC+.5)/(BW+.5))/1.702 over the 400 units",
        "cells": "2x2 cue cells (e13): vertical (front drawn lower = lower-is-closer congruent / front higher) x drawn area ratio front/back < 0.77 / >= 0.77; "
                 "mh_raw = sum BC / sum BW (= pooled, matched pairs); std_cells = unit-share-weighted mean of within-cell log-ORs",
        "vert_balanced": "mean of the front-lower and front-higher log-ORs (the lower-is-closer cue enters with opposite signs and cancels under an additive logit); "
                         "vert_component = half their difference (lower-is-closer reliance)",
        "bootstrap": f"{NB} scene resamples (both phrasings and both images of a scene kept together); seed 232"}}
    for m in M8:
        Ud, Uc = twin_units(m, "direct"), twin_units(m, "cot")
        keys = sorted(k for k in Ud if k in Uc and {"orig", "cf"} <= set(Ud[k]) and {"orig", "cf"} <= set(Uc[k]))
        assert all(Ud[k]["orig"][1] != Ud[k]["cf"][1] for k in keys), m

        def oc_of(U):
            o = []
            for k in keys:
                a, b = U[k]["orig"], U[k]["cf"]
                ca, cb = a[0] == a[1], b[0] == b[1]
                o.append(0 if ca and cb else 3 if not (ca or cb) else 1 if a[0] and b[0] else 2)
            return np.array(o)

        def yy_of(U):
            o = []
            for k in keys:
                a, b = U[k]["orig"][2], U[k]["cf"][2]
                o.append(0 if a != b else 1 if a else 2)
            return np.array(o)

        oc_d, oc_c = oc_of(Ud), oc_of(Uc)
        vert = np.array([int(Ud[k]["above"]) for k in keys])  # 0 = front lower (congruent), 1 = front higher
        small = np.array([int(Ud[k]["ratio"] >= 0.77) for k in keys])
        cell = 2 * vert + small
        wc = np.bincount(cell, minlength=4) / len(cell)
        scene = np.array([k[0] for k in keys])
        us = np.unique(scene)
        sidx = {s: np.where(scene == s)[0] for s in us}
        est = twin_stats(oc_d, oc_c, cell, wc, vert)
        bs = []
        for _ in range(NB):
            i = np.concatenate([sidx[s] for s in rng.choice(us, len(us))])
            bs.append(twin_stats(oc_d[i], oc_c[i], cell[i], wc, vert[i]))
        bs = np.array(bs)
        n = len(keys)
        sh = lambda oc: [round(100 * float(np.mean(oc == j)), 1) for j in range(4)]
        yd, yc = yy_of(Ud), yy_of(Uc)
        r = {"n_units": n, "n_scenes": len(us), "cell_n": [int(x) for x in np.bincount(cell, minlength=4)],
             "split_direct": sh(oc_d), "split_cot": sh(oc_c), "split_delta": [round(b - a, 1) for a, b in zip(sh(oc_d), sh(oc_c))],
             "yy_nn_direct": [round(100 * float(np.mean(yd == 1)), 1), round(100 * float(np.mean(yd == 2)), 1)],
             "yy_nn_cot": [round(100 * float(np.mean(yc == 1)), 1), round(100 * float(np.mean(yc == 2)), 1)],
             "marginal_ddprime_tabS12": dp[m]["dprime_cot_minus_direct"], "marginal_ddprime_ci": dp[m]["diff_ci"],
             "cell_counts": {a: np.bincount(cell * 4 + oc, minlength=16).reshape(4, 4).tolist() for a, oc in (("direct", oc_d), ("cot", oc_c))}}
        for j, nm in enumerate(TW_NAMES):
            r[nm] = round(float(est[j]), 3)
            r[nm + "_ci"] = q95(bs[:, j])
            if nm.startswith("delta_"):
                r[nm + "_p"] = round(boot_p(bs[:, j], est[j]), 5)
        res[m] = r
        f = lambda k, p=2: f"{r[k]:+.{p}f} [{r[k + '_ci'][0]:+.{p}f}, {r[k + '_ci'][1]:+.{p}f}]"
        lines.append(f"TWINPAIR {m:16s} BC/SF/SB/BW Dir {'/'.join(f'{x:.1f}' for x in r['split_direct'])} CoT {'/'.join(f'{x:.1f}' for x in r['split_cot'])} "
                     f"(YY/NN Dir {r['yy_nn_direct']} CoT {r['yy_nn_cot']}) | d'pair {r['direct_pooled']:.2f}->{r['cot_pooled']:.2f} ({f('delta_pooled')}) "
                     f"| cell-std {r['direct_std_cells']:.2f}->{r['cot_std_cells']:.2f} ({f('delta_std_cells')}) | vert-balanced {r['direct_vert_balanced']:.2f}->{r['cot_vert_balanced']:.2f} ({f('delta_vert_balanced')}) "
                     f"| vertical comp {r['direct_vert_component']:.2f}->{r['cot_vert_component']:.2f} ({f('delta_vert_component')}) "
                     f"| front lower {r['direct_lower']:.2f}->{r['cot_lower']:.2f}, higher {r['direct_higher']:.2f}->{r['cot_higher']:.2f} | marginal Tab. S12 {r['marginal_ddprime_tabS12']:+.2f} {r['marginal_ddprime_ci']}")
    sig = lambda k: [m for m in M8 if res[m][k + "_ci"][1] < 0]
    pos = lambda k: [m for m in M8 if res[m][k + "_ci"][0] > 0]
    res["_summary"] = {k: {"fall": sig(f"delta_{k}"), "rise": pos(f"delta_{k}")} for k in ("pooled", "std_cells", "vert_balanced", "vert_component")}
    res["_summary"]["marginal"] = {"fall": [m for m in M8 if res[m]["marginal_ddprime_ci"][1] < 0], "rise": [m for m in M8 if res[m]["marginal_ddprime_ci"][0] > 0]}
    for k, v in res["_summary"].items():
        lines.append(f"TWINPAIR-SUM {k:15s} fall (CI<0) {v['fall']} | rise (CI>0) {v['rise']}")
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nb-glmm", type=int, default=200)
    ap.add_argument("--skip-glmm", action="store_true")
    ap.add_argument("--head", default="pooled_all", help="d'_pair statistic entered in the BH family")
    a = ap.parse_args()
    lines = []
    P = {m: load_pairs(m) for m in M7}
    out = {"strat": run_strat(lines, P)}
    out["points"] = run_points(lines, out["strat"])
    out["criterion"] = run_crit(lines)
    out["twins"] = run_twins(lines)
    out["multiplicity"] = run_mult(lines, out["strat"], a.head)
    out["multiplicity_alt"] = [alt_mult(out["strat"], k) for k in ("pooled_all", "std_terc", "std_quint") if k != a.head]
    lines.append("BH-ALT " + "; ".join(f"{x['statistic']}: {x['survive']}/{x['nominal']} of {x['n_tests']} (d'pair survivors {x['dpair_survive']})" for x in out["multiplicity_alt"]))
    if not a.skip_glmm:
        out["glmm"] = run_glmm(lines, P, a.nb_glmm)
    else:
        old = B / "syn_pair_sdt_mh.json"
        if old.exists() and "glmm" in json.load(open(old)):
            out["glmm"] = G = json.load(open(old))["glmm"]
            for m in M7:
                r = G[m]
                lines.append(f"GLMM {m:14s} (cached, n_boot {G['n_boot']}) mean d' {r['dmean_direct']:.2f} -> {r['dmean_cot']:.2f} (delta {r['dmean_delta']:+.2f} {r['dmean_delta_ci']}, p {r['dmean_delta_p']}) "
                             f"| slope SD (d') Dir {r['slope_sd_direct']:.2f} {r['slope_sd_direct_ci']} CoT {r['slope_sd_cot']:.2f}")
            lines.append(f"GLMM-SUM mean-d' fall (CI<0) {len(G['_summary']['fall_ci_lt0'])}/7 {G['_summary']['fall_ci_lt0']} | rise {G['_summary']['rise_ci_gt0']} (weakly identified; not reported)")
    # cross-check vs syn_pair_sdt.json (Tab. S25)
    sp = json.load(open(B / "syn_pair_sdt.json"))["pairsdt"]
    out["check_tabS25"] = {m: [sp[m]["dpair_logit_direct"], sp[m]["dpair_logit_cot"], sp[m]["dpair_logit_delta"], out["strat"][m]["direct_pooled_all"], out["strat"][m]["cot_pooled_all"], out["strat"][m]["delta_pooled_all"]] for m in M7}
    lines.append("CHECK vs syn_pair_sdt.json d'pair Dir/CoT/delta | recomputed: " + "; ".join(f"{m} {v[:3]} | {v[3:]}" for m, v in out["check_tabS25"].items()))
    json.dump(out, open(B / "syn_pair_sdt_mh.json", "w"), indent=1)
    (B / "rigor_e17q.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
