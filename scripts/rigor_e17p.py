"""Exp-21 / round-12 CPU analyses (Opus round-12 M1, M3).

pairsdt   Synthetic Depth pair-matched decomposition. Each of the 500 images is asked under both object orders (original
          question + e2a 'cfq' swapped question, CoT regenerated for the swapped question). Per arm the image falls into
          BC (both correct), YY (yes to both), NN (no to both) or BW (both wrong). Shares, Direct->CoT deltas, the
          pair-matched conditional log-odds log((BC+.5)/(BW+.5)) and its probit-scale conversion, the pooled delta-d'
          (same construction as rigor_e17n.syn (a)), the ratio pooled / pair-matched, an accuracy-loss decomposition and
          gross Direct->CoT transitions; YY / NN change by visible-area-ratio tercile. Pair bootstrap over images (2,000).
seeing    'Seeing right' premise: among Synthetic Depth D+C- items (all, Tab. 1 readout; and the 280 commit-audit
          items), share whose Direct answer is correct under both object orders; commit-sentence consensus
          image-contradiction on that subset vs the rest, and the relational D+C+ comparator restricted the same way.
rescuep   Exact Fisher p for rescue (D-C+) vs both D+C+ relational comparators (rescue-position- and commit-matched).
churnsyn  Synthetic Depth D+C- vs D-C+ counts per model (churn.json).
fdr       Tests whose BH survival differs between resampling and Wald p (multiplicity.json).
-> bundle/syn_pair_sdt.json, bundle/rigor_e17p.txt
Usage: python rigor_e17p.py
"""
from __future__ import annotations
import vtax_paths as _vtp  # noqa: E402

import json
from pathlib import Path

import numpy as np
from scipy.stats import fisher_exact, norm

import build_bundle as BB
from build_bundle import rj
from rigor_e17n import parse_readout

O = Path(f"{_vtp.OUTPUTS}")
L = Path(f"{_vtp.LEGACY_RESULTS}")
B = Path("outputs/bundle")
M7 = ["qwen3b", "qwen7b", "internvl3_8b", "phi35v", "molmo7b", "qwen3vl_8b", "internvl35_8b"]
J2 = ("claude-opus-5-5", "gpt-6-astra")
OUT4 = ("BC", "YY", "NN", "BW")
LOGIT_PROBIT = 1.702  # logit(p) ~= 1.702 * probit(p)
NB = 2000


def wilson(k: int, n: int) -> list[float]:
    if n == 0:
        return [float("nan")] * 2
    z, p = 1.96, k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [round(100 * float(c - h), 1), round(100 * float(c + h), 1)]


def q95(v) -> list[float]:
    return [round(float(np.nanquantile(v, .025)), 3), round(float(np.nanquantile(v, .975)), 3)]


def load_pairs(m: str) -> dict:
    """Per image: gold_yes (original question), says-yes under original / swapped question for Direct and CoT."""
    main = BB.load_main(O, L, m)
    keys = [k for k, v in main["direct"].items() if v[3] == "syn_depth" and k in main["cot"]]
    cf = {r["item_id"]: r for r in rj(O / "e2a" / f"{m}.jsonl") if r["variant"] == "cfq" and r["task"] == "syn_depth"}
    keys = [k for k in keys if k in cf]
    gold_yes = np.array([(main["direct"][k][4] == "A") == bool(main["direct"][k][0]) for k in keys])
    # sanity: the swapped question's gold is the opposite answer
    assert all((cf[k]["gold"] == "A") == (not g) for k, g in zip(keys, gold_yes)), m
    d = {"keys": keys, "gold_yes": gold_yes, "main": main, "cf": cf}
    for arm in ("direct", "cot"):
        d[f"{arm}_o"] = np.array([main[arm][k][4] == "A" for k in keys])
        d[f"{arm}_s"] = np.array([parse_readout(cf[k][f"readout_{arm}"]) == "A" for k in keys])
    return d


def outcome(yes_o: np.ndarray, yes_s: np.ndarray, gold_yes: np.ndarray) -> np.ndarray:
    """0 BC, 1 YY, 2 NN, 3 BW (golds of the two orders are opposite, so YY / NN have exactly one correct answer)."""
    c_o, c_s = yes_o == gold_yes, yes_s == ~gold_yes
    return np.where(c_o & c_s, 0, np.where(~c_o & ~c_s, 3, np.where(yes_o & yes_s, 1, 2)))


def dprime_pooled(yes_o, yes_s, gold_yes) -> float:
    """Polarity-pooled d' on 'the queried subject is in front' (rigor_e17n.syn (a))."""
    resp = np.concatenate([yes_o, yes_s])
    truth = np.concatenate([gold_yes, ~gold_yes])
    h = (resp[truth].sum() + .5) / (truth.sum() + 1)
    f = (resp[~truth].sum() + .5) / ((~truth).sum() + 1)
    return float(norm.ppf(h) - norm.ppf(f))


def stats(oc_d: np.ndarray, oc_c: np.ndarray, pd: tuple, pc: tuple) -> np.ndarray:
    """Vector of statistics for one (re)sample."""
    n = len(oc_d)
    sh = lambda oc: np.array([np.sum(oc == i) for i in range(4)], float)
    cd, cc = sh(oc_d), sh(oc_c)
    lor = lambda c: np.log((c[0] + .5) / (c[3] + .5))
    afc = lambda c: np.sqrt(2) * norm.ppf((c[0] + .5) / (c[0] + c[3] + 1))
    dd, dc = dprime_pooled(*pd), dprime_pooled(*pc)
    acc = lambda c: 100 * (c[0] + .5 * (c[1] + c[2])) / n
    return np.concatenate([100 * cd / n, 100 * cc / n, 100 * (cc - cd) / n,
                           [lor(cd), lor(cc), lor(cc) - lor(cd),
                            lor(cd) / LOGIT_PROBIT, lor(cc) / LOGIT_PROBIT, (lor(cc) - lor(cd)) / LOGIT_PROBIT,
                            afc(cd), afc(cc), afc(cc) - afc(cd),
                            dd, dc, dc - dd, acc(cc) - acc(cd)]])


SNAMES = ([f"direct_{o}" for o in OUT4] + [f"cot_{o}" for o in OUT4] + [f"delta_{o}" for o in OUT4]
          + ["logor_direct", "logor_cot", "logor_delta", "dpair_logit_direct", "dpair_logit_cot", "dpair_logit_delta",
             "dpair_2afc_direct", "dpair_2afc_cot", "dpair_2afc_delta", "dpooled_direct", "dpooled_cot", "dpooled_delta", "acc_delta"])


def ratio_terciles() -> tuple[dict, float, float]:
    geo = json.load(open(B / "syn_depth_geometry.json"))["items"]
    ratio = {}
    for k, g in geo.items():
        a = [v["px"] for v in g.values()]
        if min(a) > 0:
            ratio[k] = max(a) / min(a)
    t1, t2 = np.quantile(np.array(list(ratio.values())), [1 / 3, 2 / 3])
    return ratio, float(t1), float(t2)


def front_larger(cf: dict) -> float:
    import re
    geo = json.load(open(B / "syn_depth_geometry.json"))["items"]
    fl = []
    for k, r in cf.items():
        x, y = re.match(r"Is the (.+?) in front of the (.+?)\?", r["question"]).groups()
        front, back = (x, y) if r["gold"] == "A" else (y, x)
        if k in geo and front in geo[k] and back in geo[k]:
            fl.append(geo[k][front]["px"] > geo[k][back]["px"])
    return round(100 * float(np.mean(fl)), 1)


def pairsdt(lines: list[str], P: dict) -> dict:
    rng = np.random.default_rng(21)
    ratio, t1, t2 = ratio_terciles()
    res = {"_doc": {
        "outcomes": "BC both correct, YY yes to both orders, NN no to both, BW both wrong; % of images",
        "logor": "pair-matched conditional log-odds log((n_BC+0.5)/(n_BW+0.5)). Under a logistic model "
                 "P(yes | image i, gold) = sigma(b_i + s * Delta/2) with s = +1 for the order whose gold is 'yes' and -1 for the other, "
                 "n_BC/n_BW estimates exp(Delta) for any per-image bias b_i (conditional-logit / McNemar MLE), so log-OR = Delta = "
                 "logit(H_i) - logit(F_i), the per-image logistic sensitivity, free of global, per-image and query-affirming bias "
                 "(assumes Delta homogeneous over images and the two answers independent given b_i).",
        "dpair_logit": "log-OR / 1.702 (logit(p) ~= 1.702 probit(p)): the per-image d' = z(H_i) - z(F_i) on the probit scale; primary.",
        "dpair_2afc": "sqrt(2) * Phi^-1(BC/(BC+BW)) (2AFC relation); assumes the model compares the two orders' evidence directly; "
                      "under the yes/no model with bias it is not bias-invariant and overstates d' at small d' (e.g. d' = 1, b = 0 gives 1.37); secondary.",
        "dpooled": "polarity-pooled equal-variance d' (Tab. S24 construction, e17n.json syn.pooled)",
        "ratio_pooled_over_pair": "delta dpooled / delta dpair_logit",
        "acc_decomposition": "two-order accuracy = BC + (YY+NN)/2, so delta acc = -(dYY+dNN)/2 [bias moves] - dBW [reversals]",
        "bootstrap": f"{NB} resamples of images (paired across arms); seed 21",
        "front_larger_share": front_larger(P[M7[0]]["cf"]),
        "tercile": "visible-area-ratio terciles (rigor_e17n cut points); YY on high-ratio images = affirming that the smaller, occluded shape is in front "
                   "when asked (front shape is the larger one in front_larger_share of images); acc_orig_order_delta reproduces the Tab. S22 basis"}}
    for m in M7:
        d = P[m]
        g = d["gold_yes"]
        oc_d, oc_c = outcome(d["direct_o"], d["direct_s"], g), outcome(d["cot_o"], d["cot_s"], g)
        est = stats(oc_d, oc_c, (d["direct_o"], d["direct_s"], g), (d["cot_o"], d["cot_s"], g))
        n = len(g)
        bs = []
        for _ in range(NB):
            i = rng.integers(0, n, n)
            bs.append(stats(oc_d[i], oc_c[i], (d["direct_o"][i], d["direct_s"][i], g[i]), (d["cot_o"][i], d["cot_s"][i], g[i])))
        bs = np.array(bs)
        r = {"n_images": n}
        for j, nm in enumerate(SNAMES):
            r[nm] = round(float(est[j]), 3)
            r[nm + "_ci"] = q95(bs[:, j])
        with np.errstate(divide="ignore", invalid="ignore"):
            rb = bs[:, SNAMES.index("dpooled_delta")] / bs[:, SNAMES.index("dpair_logit_delta")]
        r["ratio_pooled_over_pair"] = round(r["dpooled_delta"] / r["dpair_logit_delta"], 2)
        r["ratio_pooled_over_pair_2afc"] = round(r["dpooled_delta"] / r["dpair_2afc_delta"], 2)
        r["ratio_pooled_over_pair_ci"] = q95(rb) if r["dpair_logit_delta_ci"][1] < 0 else None
        # accuracy decomposition (points, two-order accuracy)
        bias = -(r["delta_YY"] + r["delta_NN"]) / 2
        rev = -r["delta_BW"]
        r["acc_decomp"] = {"acc_delta": round(r["acc_delta"], 2), "from_bias_moves": round(bias, 2), "from_reversals": round(rev, 2),
                           "bias_share_of_loss": round(bias / (bias + rev), 3) if bias + rev < 0 else None}
        bc_loss = -r["delta_BC"]
        r["bc_loss_decomp"] = {"bc_loss": round(bc_loss, 1), "to_YY_NN_net": round(r["delta_YY"] + r["delta_NN"], 1), "to_BW_net": round(r["delta_BW"], 1)}
        T = np.zeros((4, 4), int)
        for a, b in zip(oc_d, oc_c):
            T[a, b] += 1
        r["transitions_direct_to_cot"] = {f"{OUT4[a]}->{OUT4[b]}": int(T[a, b]) for a in range(4) for b in range(4)}
        # gross accuracy flows (images x answers/2)
        r["gross_from_BC"] = {"BC->YY": int(T[0, 1]), "BC->NN": int(T[0, 2]), "BC->BW": int(T[0, 3]),
                              "acc_pts_lost_BC_to_YYNN": round(float(100 * 0.5 * (T[0, 1] + T[0, 2]) / n), 1), "acc_pts_lost_BC_to_BW": round(float(100 * T[0, 3] / n), 1)}
        # size dependence of the bias moves
        terc = {}
        tk = np.array([("low" if ratio[k] < t1 else "mid" if ratio[k] < t2 else "high") if k in ratio else "na" for k in d["keys"]])
        for t in ("low", "mid", "high"):
            s = tk == t
            terc[t] = {"n": int(s.sum())} | {f"{o}_{arm}": round(100 * float(np.mean(oc == i)), 1) for i, o in enumerate(OUT4) for arm, oc in (("direct", oc_d[s]), ("cot", oc_c[s]))}
            for o in OUT4:
                terc[t][f"d{o}"] = round(terc[t][f"{o}_cot"] - terc[t][f"{o}_direct"], 1)
            terc[t]["acc2_delta"] = round(-(terc[t]["dYY"] + terc[t]["dNN"]) / 2 - terc[t]["dBW"], 1)
            co = lambda arm: d[f"{arm}_o"][s] == g[s]
            terc[t]["acc_orig_order_delta"] = round(100 * float(np.mean(co("cot")) - np.mean(co("direct"))), 1)
            # 'yes' to the gold-'no' order (queried subject = the behind, smaller shape) and to the gold-'yes' order, all images
            yes_no = lambda arm: np.where(g[s], d[f"{arm}_s"][s], d[f"{arm}_o"][s])
            yes_yes = lambda arm: np.where(g[s], d[f"{arm}_o"][s], d[f"{arm}_s"][s])
            terc[t]["yes_to_behind_subject"] = [round(100 * float(np.mean(yes_no("direct"))), 1), round(100 * float(np.mean(yes_no("cot"))), 1)]
            terc[t]["yes_to_front_subject"] = [round(100 * float(np.mean(yes_yes("direct"))), 1), round(100 * float(np.mean(yes_yes("cot"))), 1)]
        hi, lo = tk == "high", tk == "low"
        bh = []
        for _ in range(NB):
            ih = rng.choice(np.where(hi)[0], hi.sum())
            il = rng.choice(np.where(lo)[0], lo.sum())
            bh.append(100 * ((np.mean(oc_c[ih] == 1) - np.mean(oc_d[ih] == 1)) - (np.mean(oc_c[il] == 1) - np.mean(oc_d[il] == 1))))
        terc["dYY_high_minus_low"] = round(terc["high"]["dYY"] - terc["low"]["dYY"], 1)
        terc["dYY_high_minus_low_ci"] = [round(float(x), 1) for x in np.quantile(bh, [.025, .975])]
        r["tercile"] = terc
        res[m] = r
        f = lambda k, p=1: f"{r[k]:+.{p}f} [{r[k + '_ci'][0]:+.{p}f}, {r[k + '_ci'][1]:+.{p}f}]"
        lines.append(f"PAIR {m:14s} Direct BC/YY/NN/BW {r['direct_BC']:.1f}/{r['direct_YY']:.1f}/{r['direct_NN']:.1f}/{r['direct_BW']:.1f} | CoT {r['cot_BC']:.1f}/{r['cot_YY']:.1f}/{r['cot_NN']:.1f}/{r['cot_BW']:.1f} "
                     f"| dBC {f('delta_BC')} dYY {f('delta_YY')} dNN {f('delta_NN')} dBW {f('delta_BW')}")
        lines.append(f"PAIR-SENS {m:14s} logOR {r['logor_direct']:.2f} -> {r['logor_cot']:.2f} ({f('logor_delta', 2)}) | d'pair(logit/1.702) {r['dpair_logit_direct']:.2f} -> {r['dpair_logit_cot']:.2f} ({f('dpair_logit_delta', 2)}) "
                     f"| d'pair(2AFC) {f('dpair_2afc_delta', 2)} | pooled d' {f('dpooled_delta', 2)} | ratio pooled/pair {r['ratio_pooled_over_pair']} {r['ratio_pooled_over_pair_ci']} (2AFC {r['ratio_pooled_over_pair_2afc']})")
        a = r["acc_decomp"]
        lines.append(f"PAIR-ACC {m:14s} two-order acc delta {a['acc_delta']:+.1f} = bias moves {a['from_bias_moves']:+.1f} + reversals {a['from_reversals']:+.1f} (bias share {a['bias_share_of_loss']}) "
                     f"| BC loss {r['bc_loss_decomp']['bc_loss']:.1f} = YY+NN {r['bc_loss_decomp']['to_YY_NN_net']:+.1f} + BW {r['bc_loss_decomp']['to_BW_net']:+.1f} | gross from BC: {r['gross_from_BC']}")
        lines.append(f"PAIR-TERC {m:14s} dYY low/mid/high {terc['low']['dYY']:+.1f}/{terc['mid']['dYY']:+.1f}/{terc['high']['dYY']:+.1f} (high-low {terc['dYY_high_minus_low']:+.1f} {terc['dYY_high_minus_low_ci']}) "
                     f"| dNN {terc['low']['dNN']:+.1f}/{terc['mid']['dNN']:+.1f}/{terc['high']['dNN']:+.1f} | dBW {terc['low']['dBW']:+.1f}/{terc['mid']['dBW']:+.1f}/{terc['high']['dBW']:+.1f} "
                     f"| two-order acc delta {terc['low']['acc2_delta']:+.1f}/{terc['mid']['acc2_delta']:+.1f}/{terc['high']['acc2_delta']:+.1f} "
                     f"| original-order acc delta (Tab. S22 basis) {terc['low']['acc_orig_order_delta']:+.1f}/{terc['mid']['acc_orig_order_delta']:+.1f}/{terc['high']['acc_orig_order_delta']:+.1f} "
                     f"| 'yes' to behind-subject order Dir->CoT " + "/".join(f"{terc[t]['yes_to_behind_subject'][0]:.0f}->{terc[t]['yes_to_behind_subject'][1]:.0f}" for t in ("low", "mid", "high"))
                     + " | 'yes' to front-subject order " + "/".join(f"{terc[t]['yes_to_front_subject'][0]:.0f}->{terc[t]['yes_to_front_subject'][1]:.0f}" for t in ("low", "mid", "high")))
    # summary counts
    sig = lambda m, k: res[m][k + "_ci"][1] < 0
    pos = lambda m, k: res[m][k + "_ci"][0] > 0
    res["_summary"] = {
        "pair_logor_drop_ci_lt0": [m for m in M7 if sig(m, "logor_delta")],
        "pair_logor_rise_ci_gt0": [m for m in M7 if pos(m, "logor_delta")],
        "YY_rise_ci_gt0": [m for m in M7 if pos(m, "delta_YY")],
        "NN_rise_ci_gt0": [m for m in M7 if pos(m, "delta_NN")],
        "BW_rise_ci_gt0": [m for m in M7 if pos(m, "delta_BW")],
        "dYY_size_dependent_ci_gt0": [m for m in M7 if res[m]["tercile"]["dYY_high_minus_low_ci"][0] > 0],
        "ratio_pooled_over_pair": {m: res[m]["ratio_pooled_over_pair"] for m in M7},
        "dpair_logit_delta_range_drops": [min(res[m]["dpair_logit_delta"] for m in M7 if sig(m, "logor_delta")), max(res[m]["dpair_logit_delta"] for m in M7 if sig(m, "logor_delta"))],
        "dpooled_delta_range_drops": [min(res[m]["dpooled_delta"] for m in M7 if res[m]["dpooled_delta"] < 0), max(res[m]["dpooled_delta"] for m in M7 if res[m]["dpooled_delta"] < 0)]}
    s = res["_summary"]
    lines.append(f"PAIR-SUM logOR drop (CI<0): {len(s['pair_logor_drop_ci_lt0'])}/7 {s['pair_logor_drop_ci_lt0']} | rise (CI>0): {s['pair_logor_rise_ci_gt0']} | YY rise CI>0: {s['YY_rise_ci_gt0']} | NN rise CI>0: {s['NN_rise_ci_gt0']} "
                 f"| BW rise CI>0: {s['BW_rise_ci_gt0']} | dYY size-dependent (high-low CI>0): {s['dYY_size_dependent_ci_gt0']} | d'pair drop range {s['dpair_logit_delta_range_drops']} vs pooled {s['dpooled_delta_range_drops']} | ratios {s['ratio_pooled_over_pair']}")
    return res


def consensus(J: dict, ks: list) -> tuple[int, int]:
    """(# both judges INCORRECT, # judgeable = both agree on CORRECT/INCORRECT)."""
    lab = lambda j, k: J[j][k]["label"]
    cons = [k for k in ks if lab(J2[0], k) == lab(J2[1], k) and lab(J2[1], k) in ("CORRECT", "INCORRECT")]
    return sum(lab(J2[1], k) == "INCORRECT" for k in cons), len(cons)


def seeing(lines: list[str], P: dict) -> dict:
    res = {"_doc": "pair-correct = Direct correct under the original question (Tab. 1 readout) AND under the swapped question (e2a cfq). "
                   "Commit-audit items are D+C- under the Sec. 5 (original-run top-20) cells; contradiction = both judges INCORRECT among items "
                   "both judges label CORRECT/INCORRECT (commit_judge.json convention); Wilson 95% CIs; Fisher exact two-sided."}
    pc_map = {}
    for m in M7:
        d = P[m]
        g = d["gold_yes"]
        pc = (d["direct_o"] == g) & (d["direct_s"] == ~g)
        pc_map[m] = dict(zip(d["keys"], pc))
        dcm = (d["direct_o"] == g) & (d["cot_o"] != g)
        dcp = (d["direct_o"] == g) & (d["cot_o"] == g)
        res[m] = {"all_DCm_tab1": {"n": int(dcm.sum()), "pair_correct": int((dcm & pc).sum()), "share": round(100 * float((dcm & pc).sum() / dcm.sum()), 1)},
                  "all_DCp_tab1": {"n": int(dcp.sum()), "share": round(100 * float((dcp & pc).sum() / dcp.sum()), 1)}}
    sets = {"DCm_commit": ("", "D+C-"), "DCp_rel_commitpos": ("_relcomp", "D+C+rel")}
    pooled = {}
    for name, (suf, grp) in sets.items():
        J = {j: {(r["model"], r["item_id"]): r for r in rj(O / "e20" / f"{j}{suf}.jsonl") if r["group"] == grp} for j in J2}
        keys = sorted(set(J[J2[0]]) & set(J[J2[1]]))
        rows = {}
        for m in M7 + ["pooled"]:
            ks = [k for k in keys if m == "pooled" or k[0] == m]
            a = [k for k in ks if pc_map[k[0]].get(k[1], False)]
            b = [k for k in ks if not pc_map[k[0]].get(k[1], False)]
            ka, na = consensus(J, a)
            kb, nb = consensus(J, b)
            kt, nt = consensus(J, ks)
            r = {"n": len(ks), "n_pair_correct": len(a), "share_pair_correct": round(100 * len(a) / max(1, len(ks)), 1),
                 "contradict_all": [round(100 * kt / max(1, nt), 1), nt, wilson(kt, nt)],
                 "contradict_pair_correct": [round(100 * ka / max(1, na), 1), na, wilson(ka, na)],
                 "contradict_rest": [round(100 * kb / max(1, nb), 1), nb, wilson(kb, nb)],
                 "counts": {"pc": [ka, na], "rest": [kb, nb], "all": [kt, nt]}}
            if na and nb:
                r["fisher_p_pc_vs_rest"] = float(f"{fisher_exact([[ka, na - ka], [kb, nb - kb]])[1]:.3g}")
            rows[m] = r
        pooled[name] = rows
    res["commit_audit"] = pooled
    a, b = pooled["DCm_commit"]["pooled"], pooled["DCp_rel_commitpos"]["pooled"]
    (k1, n1), (k2, n2) = a["counts"]["pc"], b["counts"]["pc"]
    res["contrast_pc_DCm_vs_DCp_rel"] = {"diff_pts": round(100 * (k1 / n1 - k2 / n2), 1), "fisher_p": float(f"{fisher_exact([[k1, n1 - k1], [k2, n2 - k2]])[1]:.3g}")}
    # check the audited set reproduces commit_judge.json pooled 88.5% (235/280 n, consensus_n)
    cj = json.load(open(B / "commit_judge.json"))["rows"]["pooled|D+C-"]
    res["check_commit_judge_pooled"] = {"commit_judge.json": [cj["consensus_incorrect"], cj["consensus_n"]], "recomputed": a["contradict_all"][:2]}
    lines.append("SEEING all Syn. Depth D+C- (Tab. 1 readout) Direct pair-correct share: " + ", ".join(f"{m} {res[m]['all_DCm_tab1']['share']} ({res[m]['all_DCm_tab1']['pair_correct']}/{res[m]['all_DCm_tab1']['n']})" for m in M7)
                 + " | D+C+ for reference: " + ", ".join(f"{m} {res[m]['all_DCp_tab1']['share']}" for m in M7))
    for name in sets:
        rows = pooled[name]
        lines.append(f"SEEING {name:18s} pair-correct share: " + ", ".join(f"{m} {rows[m]['share_pair_correct']} ({rows[m]['n_pair_correct']}/{rows[m]['n']})" for m in M7 + ["pooled"]))
        p = rows["pooled"]
        lines.append(f"SEEING {name:18s} consensus contradiction: all {p['contradict_all'][0]} (nj {p['contradict_all'][1]}) {p['contradict_all'][2]} | Direct pair-correct {p['contradict_pair_correct'][0]} "
                     f"(nj {p['contradict_pair_correct'][1]}) {p['contradict_pair_correct'][2]} | rest {p['contradict_rest'][0]} (nj {p['contradict_rest'][1]}) {p['contradict_rest'][2]} | Fisher p {p.get('fisher_p_pc_vs_rest')}"
                     + " | per model pc/rest: " + ", ".join(f"{m} {rows[m]['contradict_pair_correct'][0]:.0f}({rows[m]['contradict_pair_correct'][1]})/{rows[m]['contradict_rest'][0]:.0f}({rows[m]['contradict_rest'][1]})" for m in M7))
    c = res["contrast_pc_DCm_vs_DCp_rel"]
    lines.append(f"SEEING pair-correct subset: D+C- commit vs relational D+C+ contradiction {c['diff_pts']:+.1f} pts, Fisher p {c['fisher_p']} | check vs commit_judge.json {res['check_commit_judge_pooled']}")
    return res


def rescuep(lines: list[str]) -> dict:
    rc = json.load(open(B / "rescue_comparator.json"))
    S = rc["sets"]
    res = {}
    for key in ("pooled", "pooled|rel"):
        k = lambda s: (round(S[s][key]["consistent"] * S[s][key]["judgeable"] / 100), S[s][key]["judgeable"])
        for cmp_ in ("DpCp_rel_rescuepos", "DpCp_rel_commitpos", "DmCm_rel_rescuepos"):
            (k1, n1), (k2, n2) = k("rescue_DmCp"), k(cmp_)
            odds, p = fisher_exact([[k1, n1 - k1], [k2, n2 - k2]])
            _, p_less = fisher_exact([[k1, n1 - k1], [k2, n2 - k2]], alternative="less")
            res[f"rescue_DmCp - {cmp_}{' (relational only)' if key.endswith('rel') else ''}"] = {
                "rescue": [k1, n1, round(100 * k1 / n1, 1)], "comparator": [k2, n2, round(100 * k2 / n2, 1)],
                "diff_pts": round(100 * (k1 / n1 - k2 / n2), 1), "fisher_p_two_sided": float(f"{p:.4g}"), "fisher_p_one_sided_rescue_lower": float(f"{p_less:.4g}"),
                "odds_ratio": round(float(odds), 3)}
    res["_doc"] = ("DpCp_rel_rescuepos: relational D+C+ sentences at the rescue sentences' relative positions (Tab. S28, 87.0%); "
                   "DpCp_rel_commitpos: relational D+C+ sentences at the D+C- commit sentences' relative positions = the 9.7%-contradiction "
                   "comparator of the commit audit (Tab. S26/S29, e20 *_relcomp.jsonl, 90.3% consistent of 226). Consistent share among judgeable.")
    for k, v in res.items():
        if k.startswith("_"):
            continue
        lines.append(f"RESCUE-P {k:55s} {v['rescue'][0]}/{v['rescue'][1]} ({v['rescue'][2]}) vs {v['comparator'][0]}/{v['comparator'][1]} ({v['comparator'][2]}): {v['diff_pts']:+.1f} pts, "
                     f"Fisher two-sided p = {v['fisher_p_two_sided']}, one-sided p = {v['fisher_p_one_sided_rescue_lower']}, OR {v['odds_ratio']}")
    return res


def churnsyn(lines: list[str]) -> dict:
    ch = json.load(open(B / "churn.json"))
    res = {}
    for m in BB.MODELS:
        out = {}
        for ro in ("tab1", "legacy"):
            s = ch[m]["syn_depth"][ro]
            out[ro] = {"n": s["n"], "D+C-": s["D+C-"], "D-C+": s["D-C+"], "ratio": round(s["D+C-"] / max(1, s["D-C+"]), 2), "net_pts": s["net_pts"], "churn_pct": s["churn_pct"]}
        res[m] = out
    t = {m: res[m]["tab1"] for m in res}
    res["_summary"] = {"errors_outnumber_rescues_tab1": [m for m in t if t[m]["D+C-"] > t[m]["D-C+"]],
                       "reversed_tab1": [m for m in t if t[m]["D+C-"] < t[m]["D-C+"]],
                       "ratio_range_7_open_excl_think_and_ivl35": [min(t[m]["ratio"] for m in M7 if m != "internvl35_8b"), max(t[m]["ratio"] for m in M7 if m != "internvl35_8b")]}
    lines.append("CHURN-SYN-P (Tab. 1 readout, n = 500 each) " + "; ".join(f"{m} {t[m]['D+C-']}/{t[m]['D-C+']} (ratio {t[m]['ratio']}, net {t[m]['net_pts']:+.1f})" for m in t)
                 + f" | errors > rescues: {len(res['_summary']['errors_outnumber_rescues_tab1'])}/8, reversed: {res['_summary']['reversed_tab1']}")
    return res


def fdr(lines: list[str]) -> dict:
    mu = json.load(open(B / "multiplicity.json"))
    res = {"global": [], "family": []}
    for t in mu["tests"]:
        for scope in ("global", "family"):
            a, b = t[f"survives_{scope}_resampling"], t[f"survives_{scope}_wald"]
            if a != b:
                res[scope].append({"family": t["family"], "model": t["model"], "est": t["est"], "ci": t["ci"], "p_resampling": t["p_resampling"], "p_wald": round(t["p_wald"], 4),
                                   "q_resampling": t[f"q_{scope}_resampling"], "q_wald": t[f"q_{scope}_wald"], "survives_resampling": a, "survives_wald": b})
    res["n_tests"] = mu["n_tests"]
    res["n_survive_global_resampling"] = sum(t["survives_global_resampling"] for t in mu["tests"])
    res["n_survive_global_wald"] = sum(t["survives_global_wald"] for t in mu["tests"])
    res["n_nominal"] = sum(t["ci"][0] > 0 or t["ci"][1] < 0 for t in mu["tests"])
    tw = next(t for t in mu["tests"] if t["family"] == "twin_ddprime" and t["model"] == "qwen7b")
    res["qwen7b_twin"] = {"q_resampling": tw["q_global_resampling"], "q_wald": tw["q_global_wald"]}
    for scope in ("global", "family"):
        lines.append(f"FDR-DIFF {scope}: " + ("; ".join(f"{x['family']} {x['model']} est {x['est']:+.3f} {x['ci']} p_res {x['p_resampling']} q_res {x['q_resampling']} ({'survives' if x['survives_resampling'] else 'fails'}) "
                                                        f"| p_wald {x['p_wald']} q_wald {x['q_wald']} ({'survives' if x['survives_wald'] else 'fails'})" for x in res[scope]) or "none"))
    lines.append(f"FDR-SUM {res['n_tests']} tests, nominal {res['n_nominal']}, survive global resampling {res['n_survive_global_resampling']}, Wald {res['n_survive_global_wald']}; qwen7b twin q {res['qwen7b_twin']}")
    return res


def main():
    lines = []
    P = {m: load_pairs(m) for m in M7}
    out = {"pairsdt": pairsdt(lines, P), "seeing": seeing(lines, P), "rescue_p": rescuep(lines), "churn_syn": churnsyn(lines), "fdr_wald_vs_resampling": fdr(lines)}
    # cross-check against e17n (Tab. S24)
    en = json.load(open(B / "e17n.json"))["syn"]["pooled"]
    out["check_e17n"] = {m: {"both_correct": [en[m]["pair_both_correct_direct"], en[m]["pair_both_correct_cot"], out["pairsdt"][m]["direct_BC"], out["pairsdt"][m]["cot_BC"]],
                             "d_dprime": [en[m]["d_dprime"], out["pairsdt"][m]["dpooled_delta"]]} for m in M7}
    lines.append("CHECK vs e17n.json (both-correct Dir, CoT | recomputed; pooled delta-d' | recomputed): " + "; ".join(f"{m} {v['both_correct']} {v['d_dprime']}" for m, v in out["check_e17n"].items()))
    json.dump(out, open(B / "syn_pair_sdt.json", "w"), indent=1)
    (B / "rigor_e17p.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
