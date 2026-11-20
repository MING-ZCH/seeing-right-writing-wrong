"""Exp-48: reasoning-tuned open VLMs from other families — MiMo-VL-7B-RL and GLM-4.1V-9B-Thinking (CPU, stored outputs).

Runs: e48_reasoning.py (pod) -> outputs/e48/<model>_{cues,twins,nat}.jsonl with arms
  direct (empty think block, readout at "Answer: ("), think (native trace <= 4096, closed, then "Answer: ("),
  cot (empty think block + "Let's think step by step." <= 512, same-turn readout).
MiMo-VL: vLLM 0.8.4 V0, exact full-vocabulary letter scores (lp_sum, as qwen7b); GLM-4.1V: vLLM 0.11 V1, top-20.

Synthetic sets: every statistic is rigor_e31.analyze unchanged (competence rule, probit GLM + common scale, AME,
d'_pair, cells, lower-is-closer share, v3 first claims, last-claim follow 2x2, rigor_e32 x / y), called once per
(model, set, writing arm) with that arm in the "cot" slot; only the loaders are patched (rigor_e43 pattern).
Twins: the same analyze on the 800 twin queries (twin meta gets front_larger from the drawn areas) plus the original
twin d'_pair (rigor_e17q.twin_stats, seed 232) and the rigor_e32 twin point.
Natural: BLINK Relative Depth (124) and CV-Bench 3D Depth (600): accuracy (image-cluster bootstrap 2,000), paired
deltas and cells; written relation from the Exp-35 two-judge extraction if outputs/e48/judge_<judge>.jsonl exist.
-> outputs/bundle/e48_reasoning_open.json, outputs/bundle/rigor_e48.txt, shared outputs/e48/rigor_e48.txt,
   ../paper/tables/supp_e48.tex
Usage: python rigor_e48.py [--nb-glm 1000] [--table-only]
"""
from __future__ import annotations

import argparse
import json
import shutil
from collections import defaultdict
from pathlib import Path

import numpy as np

import rigor_e31 as R31
from build_bundle import pe, rj
from rigor_e31 import O, B, PAPER_TAB, s_, u_, ci_, star
from rigor_e17q import twin_stats
from rigor_e28 import cboot, q95

E48 = O / "e48"
MODELS = ["mimo_vl_7b_rl", "glm41v_9b_think"]
ARMS = ["think", "cot"]
SETS = ["cues", "twins"]
MNAME = {"mimo_vl_7b_rl": "MiMo-VL-7B-RL", "glm41v_9b_think": "GLM-4.1V-9B-Thinking"}
ANAME = {"think": "native thinking", "cot": "short CoT"}
_CACHE: dict = {}


def key(m: str, st: str, arm: str) -> str:
    return f"{m}|{st}|{arm}"


def split(k: str) -> tuple[str, str, str]:
    m, st, arm = k.split("|")
    return m, st, arm


def recs(m: str, st: str) -> list[dict]:
    p = E48 / f"{m}_{st}.jsonl"
    if p not in _CACHE:
        rs = rj(p)
        for r in rs:
            mt = r.get("meta")
            if mt is not None and "front_larger" not in mt:
                mt["front_larger"] = mt["area_front_px"] > mt["area_back_px"]
        _CACHE[p] = rs
    return _CACHE[p]


def load_arms(k: str) -> dict:
    """{direct, cot (= the writing arm)} keyed by item_id, in the e10 cue-record layout."""
    if "|" not in k:
        return _orig_load(k)
    m, st, arm = split(k)
    by: dict = defaultdict(dict)
    for r in recs(m, st):
        if r["condition"] == "direct":
            by["direct"][r["item_id"]] = r
        elif r["condition"] == arm:
            by["cot"][r["item_id"]] = r
    return dict(by)


def twins_dpair(k: str) -> dict | None:
    """Original-twin d'_pair (rigor_e31.twins_dpair definition) for the writing arm of key k."""
    m, _, arm = split(k)
    U = {c: defaultdict(dict) for c in ("direct", arm)}
    for r in recs(m, "twins"):
        if r["condition"] not in U:
            continue
        p, mt, ph = pe(r), r["meta"], r["phrasing"]
        u = U[r["condition"]][(r["image_id"], ph)]
        says = (p == "A") if ph == "front" else (p == "B")
        cf = r["base_id"].endswith("_cf")
        u["cf" if cf else "orig"] = (bool(says), mt["front_name"] == mt["x_name"], p == "A")
        if not cf:
            u["above"], u["ratio"] = bool(mt["front_above"]), float(mt["area_ratio_front_over_back"])
    Ud, Uc = U["direct"], U[arm]
    keys = sorted(q for q in Ud if q in Uc and {"orig", "cf"} <= set(Ud[q]) and {"orig", "cf"} <= set(Uc[q]))
    if not keys:
        return None

    def oc_of(UU):
        o = []
        for q in keys:
            a, b = UU[q]["orig"], UU[q]["cf"]
            ca, cb = a[0] == a[1], b[0] == b[1]
            o.append(0 if ca and cb else 3 if not (ca or cb) else 1 if a[0] and b[0] else 2)
        return np.array(o)

    rng = np.random.default_rng(232)
    oc_d, oc_c = oc_of(Ud), oc_of(Uc)
    vert = np.array([int(Ud[q]["above"]) for q in keys])
    cell = 2 * vert + np.array([int(Ud[q]["ratio"] >= 0.77) for q in keys])
    wc = np.bincount(cell, minlength=4) / len(cell)
    scene = np.array([q[0] for q in keys])
    us = np.unique(scene)
    sidx = {s: np.where(scene == s)[0] for s in us}
    est = twin_stats(oc_d, oc_c, cell, wc, vert)
    bs = np.array([twin_stats(oc_d[i], oc_c[i], cell[i], wc, vert[i])
                   for i in (np.concatenate([sidx[s] for s in rng.choice(us, len(us))]) for _ in range(R31.NB))])
    n_ = len(est) // 3
    return {"n_units": len(keys), "direct": round(float(est[0]), 3), "cot": round(float(est[n_]), 3),
            "delta": {"d": round(float(est[2 * n_]), 3), "ci": q95(bs[:, 2 * n_], 3)},
            "bw_direct": int((oc_d == 3).sum()), "bw_cot": int((oc_c == 3).sum())}


def l3_twins(k: str) -> list[dict]:
    """rigor_e32 twin items (rigor_e31.l3_twins definition) for the writing arm of key k."""
    m, st, arm = split(k)
    if st == "twins":
        return []  # analyze() already puts the twin set in its "cues" slot
    D = defaultdict(dict)
    for r in recs(m, "twins"):
        if r["condition"] in ("direct", arm):
            D[r["item_id"]]["cot" if r["condition"] == arm else "direct"] = r
    out = []
    for iid, d in D.items():
        if "cot" not in d or "direct" not in d or not d["cot"].get("text"):
            continue
        c, mt = d["cot"], d["cot"]["meta"]
        out.append(R31.r32_rec(k, "twins", f"{c['base_id']}|{c['phrasing']}", c["base_id"].replace("_cf", ""), c["gold"],
                               pe(d["direct"]), pe(c), R31.r32_body(c["text"]), mt["front_name"], mt["back_name"], mt["x_name"],
                               c["phrasing"]))
    return out


_orig_load = R31.load_arms
R31.load_arms = load_arms
R31.twins_dpair = twins_dpair
R31.l3_twins = l3_twins


# ----------------------------------------------------------------------------- readout diagnostics

def readout_diag(m: str, st: str) -> dict:
    """Letter coverage of the readout: missing letters in top-20, letter mass, exact vs top-20 agreement (V0)."""
    out = {}
    for arm in ["direct"] + ARMS:
        rs = [r for r in recs(m, st) if r["condition"] == arm]
        if not rs:
            continue
        miss = np.mean([any(v is None for v in r["top20"].values()) for r in rs])
        mass = [r.get("top20_letter_mass", np.nan) for r in rs]
        d = {"n": len(rs), "top20_missing_letter_pct": round(100 * float(miss), 2),
             "letter_mass_median": round(float(np.nanmedian(mass)), 4), "letter_mass_min": round(float(np.nanmin(mass)), 4),
             "unparsed": int(sum(pe(r) is None for r in rs))}
        if "lp_sum" in rs[0]:
            t20 = [_am(r["top20"]) for r in rs]
            d["exact_eq_top20_pct"] = round(100 * float(np.mean([a == pe(r) for a, r in zip(t20, rs)])), 2)
        if arm != "direct":
            gl = [r.get("gen_len", 0) for r in rs]
            d["gen_tokens"] = {"mean": round(float(np.mean(gl)), 1), "median": float(np.median(gl)), "max": int(max(gl)),
                               "trunc_pct": round(100 * float(np.mean([bool(r.get("trunc")) for r in rs])), 2),
                               "empty_pct": round(100 * float(np.mean([len((r.get("text") or "").replace("Let's think step by step.", "").strip()) == 0 for r in rs])), 2)}
        out[arm] = d
    return out


def _am(lp: dict) -> str | None:
    fin = {k: v for k, v in lp.items() if v is not None}
    return max(fin, key=fin.get) if fin else None


# ----------------------------------------------------------------------------- natural sets

def natural(m: str) -> dict:
    """BLINK Rel. Depth / CV-Bench 3D Depth: accuracy per arm (image-cluster bootstrap), deltas vs Direct, cells."""
    rs = recs(m, "nat")
    if not rs:
        return {}
    rng = np.random.default_rng(48)
    out = {}
    for ds in ("blink", "cvb", "pooled"):
        by = defaultdict(dict)
        for r in rs:
            if ds == "pooled" or r["set"] == ds:
                by[r["condition"]][r["item_id"]] = r
        keys = sorted(by["direct"])
        g = [by["direct"][k]["image_id"] for k in keys]
        gold = np.array([by["direct"][k]["gold"] for k in keys])
        acc = {a: np.array([float(pe(by[a][k]) == by[a][k]["gold"]) for k in keys]) for a in by}
        d = {"n": len(keys)}
        for a in acc:
            d[f"acc_{a}"] = cboot(acc[a], g, rng)
        for a in ARMS:
            if a in acc:
                d[f"acc_{a}-direct"] = cboot(acc[a] - acc["direct"], g, rng)
                P = {c: np.array([pe(by[c][k]) or "" for k in keys]) for c in ("direct", a)}
                d[f"cells_{a}"] = R31.cells(P["direct"], P[a], gold)
        out[ds] = d
    return out


# ----------------------------------------------------------------------------- table

HEAD = {("mimo_vl_7b_rl", "think"): r"\shortstack{MiMo-VL-7B-RL\\think}", ("mimo_vl_7b_rl", "cot"): r"\shortstack{MiMo-VL-7B-RL\\short CoT}",
        ("glm41v_9b_think", "think"): r"\shortstack{GLM-4.1V-9B-Th.\\think}", ("glm41v_9b_think", "cot"): r"\shortstack{GLM-4.1V-9B-Th.\\short CoT}"}


def table(res: dict) -> str:
    cols = [(m, a) for m in MODELS for a in ARMS if key(m, "cues", a) in res["synthetic"]]
    S = res["synthetic"]
    row = lambda label, cells: label + " & " + " & ".join(cells) + r" \\"
    nc = len(cols) + 1
    sec = lambda t: rf"\multicolumn{{{nc}}}{{@{{}}l@{{}}}}{{\textit{{\textbf{{{t}}}}}}} \\"
    L = [r"\begin{tabular}{@{}l" + "c" * len(cols) + "@{}}", r"\toprule",
         row(r"\textbf{Model / writing arm}", [r"\textbf{" + HEAD[c] + "}" for c in cols]), r"\midrule",
         sec(r"Who reads occlusion? Cue-factorial set, \texttt{Direct} (empty think block)")]
    c0 = lambda m, a: S[key(m, "cues", a)]
    L.append(row("Accuracy (\\%)", [u_(c0(m, a)["acc_direct_comp"]["mean"]) + ci_(*c0(m, a)["acc_direct_comp"]["ci"], nd=1, signed=False) for m, a in cols]))
    L.append(row("Occlusion weight $d'$", [u_(c0(m, a)["glm"]["direct"]["occlusion"]["d"], 2) + ci_(*c0(m, a)["glm"]["direct"]["occlusion"]["ci"], signed=False)
                                           + (r"$^\S$" if c0(m, a)["glm"]["direct"]["separated"] else "") for m, a in cols]))
    L.append(row("Follows lower-is-closer (\\%)", [u_(c0(m, a)["lower"]["direct"]["agree_lower"]["mean"]) for m, a in cols]))
    L.append(row("Occlusion-competent", [r"\textbf{yes}" if c0(m, a)["competent"] else "no" for m, a in cols]))
    L.append(r"\midrule")
    L.append(sec(r"Claim 1: what writing does to cue reliance (writing arm$-$\texttt{Direct}, cue set)"))
    L.append(row(r"Writing-arm accuracy (\%)", [u_(c0(m, a)["acc_cot"]["mean"]) for m, a in cols]))
    L.append(row(r"Occlusion weight $\Delta d'$ (primary)", [s_(c0(m, a)["glm"]["cot-direct"]["occlusion"]["d"]) + ci_(*c0(m, a)["glm"]["cot-direct"]["occlusion"]["ci"]) for m, a in cols]))
    L.append(row("Vertical weight $\\Delta d'$", [s_(c0(m, a)["glm"]["cot-direct"]["vertical"]["d"]) + star(*c0(m, a)["glm"]["cot-direct"]["vertical"]["ci"]) for m, a in cols]))
    L.append(row("Occlusion AME $\\Delta$ (pts)", [s_(c0(m, a)["ame"]["occlusion"]["ame"]["cot-direct"]["est"], 1) + star(*c0(m, a)["ame"]["occlusion"]["ame"]["cot-direct"]["ci"]) for m, a in cols]))
    L.append(row("Cue-set accuracy $\\Delta$ (pts)", [s_(c0(m, a)["acc_cot-direct"]["mean"], 1) + ci_(*c0(m, a)["acc_cot-direct"]["ci"], nd=1) for m, a in cols]))
    cells = []
    for m, a in cols:
        cp = c0(m, a)["dpair"]["cot-direct_pooled"]
        cells.append(s_(cp["d"]) + star(*cp["ci"]) + (r"$^\ddagger$" if c0(m, a)["dpair"]["split_direct"][3] == 0 else ""))
    L.append(row("Cue-set $\\Delta d'_\\mathrm{pair}$", cells))
    L.append(row("\\texttt{D+C$-$} / \\texttt{D$-$C+}; net (pts)", [f"${c0(m, a)['cells']['D+C-']}$ / ${c0(m, a)['cells']['D-C+']}$; " + s_(c0(m, a)["cells"]["net_pts"], 1) for m, a in cols]))
    L.append(r"\midrule")
    L.append(sec(r"Occlusion twins ($800$ queries)"))
    tw = lambda m, a: S.get(key(m, "twins", a))
    L.append(row(r"\texttt{Direct} / writing-arm accuracy (\%)", [u_(tw(m, a)["acc_direct"]["mean"]) + " / " + u_(tw(m, a)["acc_cot"]["mean"]) if tw(m, a) else "--" for m, a in cols]))
    L.append(row(r"Accuracy $\Delta$ (pts)", [s_(tw(m, a)["acc_cot-direct"]["mean"], 1) + ci_(*tw(m, a)["acc_cot-direct"]["ci"], nd=1) if tw(m, a) else "--" for m, a in cols]))
    cells = []
    for m, a in cols:
        t = c0(m, a).get("twins_dpair")
        cells.append(s_(t["delta"]["d"]) + star(*t["delta"]["ci"]) + (r"$^\ddagger$" if t["bw_direct"] == 0 else "") if t else "--")
    L.append(row("Twins $\\Delta d'_\\mathrm{pair}$", cells))
    L.append(row("\\texttt{D+C$-$} / \\texttt{D$-$C+}", [f"${tw(m, a)['cells']['D+C-']}$ / ${tw(m, a)['cells']['D-C+']}$" if tw(m, a) else "--" for m, a in cols]))
    L.append(r"\midrule")
    L.append(sec(r"Written relation (writing arm, parser v3)"))
    L.append(row("Last relation parsed: cues / twins (\\%)", [
        f"${c0(m, a)['l3']['cues']['last_coverage']:.1f}$ / " + (f"${tw(m, a)['l3']['cues']['last_coverage']:.1f}$" if tw(m, a) else "--") for m, a in cols]))
    L.append(row("First claim: true front / lower / larger (\\%)", [
        f"${c0(m, a)['first_claim_cot']['names_front']['pct']:.1f}$ / ${c0(m, a)['first_claim_cot']['names_lower']['pct']:.1f}$ / ${c0(m, a)['first_claim_cot']['names_larger']['pct']:.1f}$"
        if c0(m, a)["first_claim_cot"].get("n_parsed") else "--" for m, a in cols]))
    L.append(row("Answer agrees with last relation: true / false (\\%)", [
        f"${c0(m, a)['follow_cot_last']['all']['p_cons_true']['mean']:.1f}$ / ${c0(m, a)['follow_cot_last']['all']['p_cons_false']['mean']:.1f}$"
        if c0(m, a)["follow_cot_last"].get("all", {}).get("n") else "--" for m, a in cols]))
    L.append(row("Two channels, cues: $x$ / $y$ (pts)", [s_(c0(m, a)["l3"]["cues"]["last_x"], 1) + " / " + s_(c0(m, a)["l3"]["cues"]["y"], 1) for m, a in cols]))
    L.append(row("Two channels, twins: $x$ / $y$ (pts)", [s_(tw(m, a)["l3"]["cues"]["last_x"], 1) + " / " + s_(tw(m, a)["l3"]["cues"]["y"], 1) if tw(m, a) else "--" for m, a in cols]))
    J = res.get("judge_syn", {}).get("cells", {})
    jc = lambda m, a, st: J.get(f"{m}|{a}|{st}") if J.get(f"{m}|{a}|{st}", {}).get("complete") else None
    if any(jc(m, a, "cues") for m, a in cols):
        L.append(r"\midrule")
        L.append(sec(r"Written relation, two-judge extraction (last front/behind statement, conclusion kept)"))
        L.append(row("Coverage: cues / twins (\\%)", [
            (f"${jc(m, a, 'cues')['last_coverage']:.1f}$" if jc(m, a, "cues") else "--") + " / "
            + (f"${jc(m, a, 'twins')['last_coverage']:.1f}$" if jc(m, a, "twins") else "--") for m, a in cols]))
        L.append(row("Written relation accuracy, cues (\\%)", [f"${jc(m, a, 'cues')['last_written_acc']:.1f}$" if jc(m, a, "cues") else "--" for m, a in cols]))
        L.append(row("Answer agrees: true / false relation, cues (\\%)", [
            f"${jc(m, a, 'cues')['p_agree_true']:.1f}$ / ${jc(m, a, 'cues')['p_agree_false']:.1f}$" if jc(m, a, "cues") else "--" for m, a in cols]))
        for st, lab in (("cues", "cues"), ("twins", "twins")):
            L.append(row(f"Two channels, {lab}: $x$ / $y$ (pts)", [
                s_(jc(m, a, st)["last_x"], 1) + star(*jc(m, a, st)["last_x_ci"]) + " / " + s_(jc(m, a, st)["y"], 1) if jc(m, a, st) else "--" for m, a in cols]))
    N = res.get("natural", {})
    if any(N.get(m) for m, _ in cols):
        L.append(r"\midrule")
        L.append(sec(r"Held-out natural depth: \texttt{Direct} / writing arm (\%), $\Delta$ (pts)"))
        for ds, lab in (("blink", "BLINK Rel.\\ Depth ($124$)"), ("cvb", "CV-Bench 3D Depth ($600$)")):
            cells = []
            for m, a in cols:
                d = N.get(m, {}).get(ds)
                if d and f"acc_{a}" in d:
                    cells.append(u_(d["acc_direct"]["mean"]) + " / " + u_(d[f"acc_{a}"]["mean"]) + "; " + s_(d[f"acc_{a}-direct"]["mean"], 1)
                                 + star(*d[f"acc_{a}-direct"]["ci"]))
                else:
                    cells.append("--")
            L.append(row(lab, cells))
        JN = res.get("judge_nat", {}).get("cells", {})
        for ds, lab in (("blink", "BLINK"), ("cvb3d", "CV-Bench")):
            cells = []
            for m, a in cols:
                c = JN.get(f"{m}|{a}|{ds}")
                cells.append(s_(c["x"], 1) + star(*c["x_ci"]) + " / " + s_(c["y"], 1) + f" ({c['coverage']:.0f}\\%)" if c and "x" in c else "--")
            L.append(row(f"{lab}: $x$ / $y$ (pts), judges (coverage)", cells))
    L.append(r"\bottomrule")
    tk = {m: S[key(m, "cues", "think")]["cot_tokens"]["mean"] for m in MODELS if key(m, "cues", "think") in S}
    note = (r"Same cue-factorial queries ($1{,}600$), occlusion twins ($800$), image budget and statistics as \cref{M-tab:profile} and "
            r"\cref{tab:supp_e31}; one greedy run per model. Neither chat template has a thinking switch, so \texttt{Direct} = empty think block, "
            r"then ``Answer: (''. think: the model's native trace ($\le 4{,}096$ tokens"
            + (", mean " + " / ".join(f"${v:.0f}$" for v in tk.values()) + " tokens for MiMo-VL / GLM-4.1V" if tk else "")
            + r"), closed, then ``Answer: (''. short CoT: empty think block, then ``Let's think step by step.'' ($\le 512$ tokens) and the same-turn readout. "
            r"Readout: MiMo-VL exact full-vocabulary letter scores (vLLM V0), GLM-4.1V top-20 letter scores (vLLM V1). "
            r"$^\S$Probit quasi-separated ($\max|\beta|>2.5$); use the AME row. "
            r"$^\ddagger$No both-wrong unit under \texttt{Direct}, so \texttt{Direct} $d'_\mathrm{pair}$ is a lower bound and its change is not interpretable. "
            r"$^\ast$: $95\%$ CI excludes zero (twin-pair or image bootstrap). "
            r"$x$: last written relation accuracy minus \texttt{Direct} accuracy on items with a relation; $y$: writing arm$-$\texttt{Direct} accuracy on all items. "
            r"Two-judge rows: relation named by both Claude Opus 5.5 and GPT-6-Astra (text only); synthetic sets with the conclusion kept (prompt of \cref{tab:supp_e38}), "
            r"natural sets with the conclusion removed (prompt of \cref{tab:supp_e35}). Parser v3 misassigns two-clause sentences that name both shapes "
            r"(``the triangle is in front, and the circle is behind it''), which are frequent in the think traces (GLM-4.1V coverage $60$--$67\%$); "
            r"for written-relation numbers use the two-judge rows.")
    L.append(rf"\multicolumn{{{nc}}}{{p{{0.95\linewidth}}}}{{\scriptsize {note}}} \\")
    L.append(r"\end{tabular}")
    return "\n".join(L) + "\n"


# ----------------------------------------------------------------------------- main

def nat_lines(m: str, N: dict) -> list[str]:
    out = []
    fm = lambda x: f"{x['mean']:.1f} [{x['ci'][0]:.1f}, {x['ci'][1]:.1f}]"
    fa = lambda x: f"{x['mean']:+.1f} [{x['ci'][0]:+.1f}, {x['ci'][1]:+.1f}]"
    for ds, d in N.items():
        s = f"NAT {MNAME[m]} {ds} n={d['n']}: Direct {fm(d['acc_direct'])}"
        for a in ARMS:
            if f"acc_{a}" in d:
                c = d[f"cells_{a}"]
                s += f" | {a} {fm(d[f'acc_{a}'])} delta {fa(d[f'acc_{a}-direct'])} D+C- {c['D+C-']} D-C+ {c['D-C+']}"
        out.append(s)
    return out


def judge_lines(res: dict) -> list[str]:
    out = []
    J = res.get("judge_syn")
    if J:
        out.append(f"JUDGE-SYN (Exp-38 prompt, conclusion kept; consensus of Claude Opus 5.5 + GPT-6-Astra): 4-way agreement {J['judge_agreement_pct']}% on {J['n_judged_both']} texts")
        for g, c in J["cells"].items():
            if c.get("complete"):
                out.append(f"JUDGE-SYN {g}: coverage {c['last_coverage']:.1f}% written {c['last_written_acc']:.1f} vs Direct on parsed {c['last_direct_on_parsed']:.1f} "
                           f"-> x {c['last_x']:+.2f} {c['last_x_ci']} | y {c['y']:+.2f} {c['y_ci']} | agree {c['last_agree']:.1f} | "
                           f"P(agree|true) {c['p_agree_true']:.1f} {c['p_agree_true_ci']} (n {c['n_true']}) P(agree|false) {c['p_agree_false']:.1f} {c['p_agree_false_ci']} (n {c['n_false']}) "
                           f"| y - x {c['y'] - c['last_x']:+.2f} | 21-point fit predicts {R31.L3_FIT['intercept'] + R31.L3_FIT['slope'] * c['last_x']:+.2f}")
            else:
                out.append(f"JUDGE-SYN {g}: incomplete {c}")
    N = res.get("judge_nat")
    if N:
        out.append(f"JUDGE-NAT (Exp-35 prompt, conclusion removed; consensus): 4-way agreement {N['judge_agreement_pct']}% on {N['n_judged_both']} texts")
        for g, c in N["cells"].items():
            if "x" in c:
                out.append(f"JUDGE-NAT {g}: n {c['n']} coverage {c['coverage']:.1f}% written {c['written_acc']:.1f} vs Direct on parsed {c['direct_on_parsed']:.1f} "
                           f"-> x {c['x']:+.2f} {c['x_ci']} | y {c['y']:+.2f} {c['y_ci']} | agree {c['agree']:.1f} | P(agree|true) {c['p_agree_true']:.1f} "
                           f"(n {c['n_true']}) P(agree|false) {c['p_agree_false']:.1f} (n {c['n_false']}) | D+C- {c['DpCm']} D-C+ {c['DmCp']}")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--nb-glm", type=int, default=1000)
    ap.add_argument("--models", default=",".join(MODELS))
    ap.add_argument("--table-only", action="store_true")
    a = ap.parse_args()
    out_json = B / "e48_reasoning_open.json"
    if a.table_only:
        res = json.load(open(out_json))
        if "lines_base" not in res:
            res["lines_base"] = [l for l in (B / "rigor_e48.txt").read_text().splitlines() if not l.startswith("JUDGE-")]
    else:
        lines: list[str] = []
        res = {"protocol": __doc__, "nb_glm": a.nb_glm, "nb": R31.NB, "synthetic": {}, "natural": {}, "readout": {}}
        for m in a.models.split(","):
            for st in SETS + ["nat"]:
                if recs(m, st):
                    res["readout"][f"{m}|{st}"] = rd = readout_diag(m, st)
                    lines.append(f"READOUT {MNAME[m]} {st}: {json.dumps(rd)}")
            for st in SETS:
                for arm in ARMS:
                    k = key(m, st, arm)
                    R31.NAMES[k] = f"{MNAME[m]} {st} {ANAME[arm]}"
                    A = load_arms(k)
                    if not A.get("cot"):
                        lines.append(f"== {k}: no outputs yet")
                        continue
                    res["synthetic"][k] = R31.analyze(k, a.nb_glm, lines)
                    print("\n".join(lines[-40:]), flush=True)
            res["natural"][m] = natural(m)
            lines += nat_lines(m, res["natural"][m])
        res["lines_base"] = lines
    for k, fn in (("judge_syn", "e48_syn_judge.json"), ("judge_nat", "e48_nat_two_channel.json")):
        if (B / fn).exists():
            res[k] = json.load(open(B / fn))
    json.dump(res, open(out_json, "w"), indent=1, default=float)
    (B / "rigor_e48.txt").write_text("\n".join(res["lines_base"] + judge_lines(res)) + "\n")
    try:
        shutil.copy(B / "rigor_e48.txt", E48 / "rigor_e48.txt")
        shutil.copy(out_json, E48 / "e48_reasoning_open.json")
    except OSError as e:
        print("copy to shared outputs failed:", e)
    txt = table(res)
    (PAPER_TAB / "supp_e48.tex").write_text(txt)
    print(txt)


if __name__ == "__main__":
    main()
