"""Exp-51 analysis (plan S1): Direct vs CoT vs Filler(|R|)+turn on VSI-Bench MC and MindCube-Tiny (CPU only).

Inputs: shared outputs/e51/<model>_<bench>.jsonl (e51_extbench.py), items_<bench>.jsonl, sanity_<model>.json,
judge_items.jsonl + judge_<proto>_<judge>.jsonl (e51_judge.py). Pre-registration: outputs/e51/prereg.txt.

Primary (per benchmark): CoT - Direct, unweighted mean over the 6 native multi-image models, 95% percentile CI from
2,000 cluster bootstrap draws (cluster = video / image group; same draws for every model; seed 51).
Secondary: per-model CoT - Direct, CoT - Filler+turn, Filler+turn - Direct; Molmo-7B-D (composite image) row;
two-channel cells from the consensus of two text-only judges (e35.point definitions: coverage, written accuracy W,
x = W - Direct(covered), y = CoT - Direct(all), aT / aF) for protocols rm (conclusion removed, primary) and kp
(conclusion kept); judge agreement and Cohen's kappa; sanity (Direct re-read, readout parse rate, CoT truncation).
-> outputs/bundle/e51_extbench.json, outputs/bundle/rigor_e51.txt, paper tables/supp_e51.tex

Usage: python rigor_e51.py [--nb 2000]
"""
from __future__ import annotations
import vtax_paths as _vtp  # noqa: E402

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from build_bundle import pe, rj
from e35_natural_two_channel import point

O = Path(f"{_vtp.OUTPUTS}")
E51 = O / "e51"
B = Path("outputs/bundle")
PAPER = Path(f"{_vtp.PAPER}")
M6 = ["qwen3b", "qwen7b", "internvl3_8b", "phi35v", "qwen3vl_8b", "internvl35_8b"]
M7 = M6 + ["molmo7b"]
NAMES = {"qwen3b": "Qwen2.5-VL-3B", "qwen7b": "Qwen2.5-VL-7B", "internvl3_8b": "InternVL3-8B", "phi35v": "Phi-3.5-V",
         "molmo7b": "Molmo-7B-D", "qwen3vl_8b": "Qwen3-VL-8B", "internvl35_8b": "InternVL3.5-8B"}
BENCHES = ["vsi", "mc"]
BN = {"vsi": "VSI-Bench MC (600)", "mc": "MindCube-Tiny (600)"}
ARMS = ["direct", "cot", "filler_R+turn"]
COT_PREFIX = "Let's think step by step."
JUDGES = ["claude-opus-5-5", "gpt-6-astra"]
PROTOS = ["rm", "kp"]
OUT: list[str] = []


def say(s: str = "") -> None:
    OUT.append(s)
    print(s)


def ci(a) -> list[float]:
    return [round(float(np.quantile(a, 0.025)), 2), round(float(np.quantile(a, 0.975)), 2)]


def letter_ok(r: dict) -> bool:
    """Readout parse: at least one option letter has a usable score (V1: present in the top-20)."""
    lp = r.get("top20") or {}
    return any(v is not None and v > -1e3 for v in lp.values())


def load(bench: str) -> tuple[list[dict], dict[str, dict[str, dict[str, dict]]]]:
    items = rj(E51 / f"items_{bench}.jsonl")
    D = {}
    for m in M7:
        recs = rj(E51 / f"{m}_{bench}.jsonl")
        if not recs:
            continue
        d: dict = defaultdict(dict)
        for r in recs:
            d[r["condition"]][r["item_id"]] = r
        D[m] = d
    return items, D


def draws(clusters: np.ndarray, nb: int, seed: int) -> list[np.ndarray]:
    uc = np.unique(clusters)
    idx = {c: np.where(clusters == c)[0] for c in uc}
    rng = np.random.default_rng(seed)
    return [np.concatenate([idx[c] for c in rng.choice(uc, len(uc))]) for _ in range(nb)]


def kappa(a: list, b: list) -> float:
    cats = sorted(set(a) | set(b))
    n = len(a)
    po = sum(x == y for x, y in zip(a, b)) / n
    pe_ = sum((a.count(c) / n) * (b.count(c) / n) for c in cats)
    return (po - pe_) / (1 - pe_) if pe_ < 1 else float("nan")


def analyze_bench(bench: str, nb: int) -> dict:
    items, D = load(bench)
    keys = [it["key"] for it in items]
    cl = np.array([it["cluster"] for it in items])
    gold = {it["key"]: it["gold"] for it in items}
    qt = {it["key"]: it["qtype"] for it in items}
    S = draws(cl, nb, 51)
    res: dict = {"n_items": len(items), "n_clusters": int(len(np.unique(cl))), "models": {}, "qtype_n": dict(Counter(qt.values()))}
    C: dict = {}
    WR: dict = {}
    for m, d in D.items():
        if not all(len(d[a]) == len(keys) for a in ARMS):
            res["models"][m] = {"complete": False, "n": {a: len(d[a]) for a in ARMS}}
            continue
        acc = {a: np.array([pe(d[a][k]) == gold[k] for k in keys], float) for a in ARMS}
        C[m] = acc
        cot = [d["cot"][k] for k in keys]
        row = {"complete": True, "acc": {a: round(100 * acc[a].mean(), 2) for a in ARMS},
               "acc_direct_ci": ci([100 * acc["direct"][s].mean() for s in S])}
        for nm, (a, b) in {"cot-direct": ("cot", "direct"), "cot-filler": ("cot", "filler_R+turn"),
                           "filler-direct": ("filler_R+turn", "direct")}.items():
            diff = acc[a] - acc[b]
            row[nm] = round(100 * diff.mean(), 2)
            row[nm + "_ci"] = ci([100 * diff[s].mean() for s in S])
        written = np.array([bool((r.get("text") or "")[len(COT_PREFIX):].strip()) for r in cot])
        row["empty_cot_pct"] = round(100 * (1 - written.mean()), 2)
        if written.any():  # exploratory (not pre-registered): items where the model wrote a continuation
            diff = acc["cot"] - acc["direct"]
            row["cot-direct_written"] = round(100 * diff[written].mean(), 2)
            row["cot-direct_written_ci"] = ci([100 * diff[s][written[s]].mean() for s in S if written[s].any()])
            row["n_written"] = int(written.sum())
        WR[m] = written
        row["DpCm"] = int(((acc["direct"] == 1) & (acc["cot"] == 0)).sum())
        row["DmCp"] = int(((acc["direct"] == 0) & (acc["cot"] == 1)).sum())
        row["readout_parse_pct"] = {a: round(100 * float(np.mean([letter_ok(d[a][k]) for k in keys])), 2) for a in ARMS}
        row["exact_readout"] = "lp_sum" in cot[0] and d["cot"][keys[0]].get("lp_sum") != d["cot"][keys[0]].get("top20")
        row["cot_trunc_pct"] = round(100 * float(np.mean([bool(r.get("trunc")) for r in cot])), 2)
        row["cot_len_mean"] = round(float(np.mean([r["gen_len"] for r in cot])), 1)
        row["stated_parse_pct"] = round(100 * float(np.mean([r.get("stated") is not None for r in cot])), 2)
        row["stated_eq_readout_pct"] = round(100 * float(np.mean([r.get("stated") == pe(r) for r in cot if r.get("stated")])), 2) \
            if any(r.get("stated") for r in cot) else None
        row["by_qtype"] = {}
        for t in sorted(set(qt.values())):
            ix = np.array([qt[k] == t for k in keys])
            row["by_qtype"][t] = {a: round(100 * acc[a][ix].mean(), 1) for a in ARMS} | {"n": int(ix.sum())}
        if bench == "vsi":
            ix = np.array([not it.get("pruned") for it in items])
            row["nonpruned"] = {a: round(100 * acc[a][ix].mean(), 2) for a in ARMS} | {"n": int(ix.sum())}
        chance = float(np.mean([1 / len(it["choices"]) for it in items]))
        row["chance"] = round(100 * chance, 2)
        res["models"][m] = row
    # primary: mean over the 6 native multi-image models
    prim = [m for m in M6 if m in C]
    res["primary_models"] = prim
    for nm, (a, b) in {"cot-direct": ("cot", "direct"), "cot-filler": ("cot", "filler_R+turn"),
                       "filler-direct": ("filler_R+turn", "direct")}.items():
        if not prim:
            continue
        dm = np.mean([C[m][a] - C[m][b] for m in prim], axis=0)
        res["pooled_" + nm] = round(100 * dm.mean(), 2)
        res["pooled_" + nm + "_ci"] = ci([100 * dm[s].mean() for s in S])
        if bench == "vsi":
            ix = np.array([not it.get("pruned") for it in items])
            res["pooled_nonpruned_" + nm] = round(100 * dm[ix].mean(), 2)
    if prim:
        dm = [C[m]["cot"] - C[m]["direct"] for m in prim]
        res["pooled_cot-direct_written"] = round(100 * float(np.mean([d_[WR[m]].mean() for d_, m in zip(dm, prim)])), 2)
        res["pooled_cot-direct_written_ci"] = ci([100 * float(np.mean([d_[s][WR[m][s]].mean() for d_, m in zip(dm, prim)]))
                                                   for s in S])
        res["pooled_acc"] = {a: round(100 * float(np.mean([C[m][a].mean() for m in prim])), 2) for a in ARMS}
    res["two_channel"] = two_channel(bench, D, items, C, nb)
    return res


def two_channel(bench: str, D: dict, items: list[dict], C: dict, nb: int) -> dict:
    rows = [r for r in rj(E51 / "judge_items.jsonl") if r["bench"] == bench]
    if not rows:
        return {}
    out: dict = {}
    for proto in PROTOS:
        lab = {j: {r["hk"]: r["option"] for r in rj(E51 / f"judge_{proto}_{j}.jsonl") if not r.get("api_error")} for j in JUDGES}
        if not all(lab.values()):
            continue
        pa, pb = [], []
        cells: dict = {}
        for m in M7:
            rs = [r for r in rows if r["model"] == m]
            if not rs:
                continue
            missing = sum(1 for r in rs if r[f"hk_{proto}"] and any(r[f"hk_{proto}"] not in lab[j] for j in JUDGES))
            for r in rs:
                h = r[f"hk_{proto}"]
                if not h:
                    r["claim"] = None
                    continue
                a, b = lab[JUDGES[0]].get(h), lab[JUDGES[1]].get(h)
                if a is not None and b is not None:
                    pa.append(a)
                    pb.append(b)
                r["claim"] = a if (a == b and a not in (None, "NONE", "UNPARSED")) else None
            A = {"d": np.array([r["d_ans"] == r["gold"] for r in rs], float), "c": np.array([r["c_ans"] == r["gold"] for r in rs], float),
                 "p": np.array([r["claim"] is not None for r in rs], float), "t": np.array([r["claim"] == r["gold"] for r in rs], float),
                 "a": np.array([r["claim"] is not None and r["claim"] == r["c_ans"] for r in rs], float)}
            pt = point(A)
            cl = np.array([r["cl"] for r in rs])
            bx, byy = [], []
            for s in draws(cl, nb, 52):
                q = point({k: v[s] for k, v in A.items()})
                byy.append(q["y"])
                if "x" in q:
                    bx.append(q["x"])
            pt["x_ci"] = ci(bx) if bx else None
            pt["y_ci"] = ci(byy)
            pt["missing_labels"] = missing
            pt["empty_text_pct"] = round(100 * float(np.mean([not r[f"hk_{proto}"] for r in rs])), 1)
            cells[m] = {k: (round(v, 3) if isinstance(v, float) else v) for k, v in pt.items()}
        n = len(pa)
        out[proto] = {"cells": cells, "judge_n_both": n,
                      "judge_agree_pct": round(100 * float(np.mean([a == b for a, b in zip(pa, pb)])), 2) if n else None,
                      "judge_kappa": round(kappa(pa, pb), 3) if n else None,
                      "judge_none_pct": {JUDGES[0]: round(100 * pa.count("NONE") / n, 1) if n else None,
                                         JUDGES[1]: round(100 * pb.count("NONE") / n, 1) if n else None}}
    return out


def tc_summary(R: dict) -> dict:
    """Pre-registered H2-H4 summaries over the 12 primary model x benchmark two-channel cells."""
    out = {}
    for proto in PROTOS:
        cells = []
        for b in BENCHES:
            tc = R[b].get("two_channel", {}).get(proto, {}).get("cells", {})
            for m in M6:
                c, row = tc.get(m), R[b]["models"].get(m, {})
                if c and "x" in c:
                    cells.append((b, m, c, row))
        if not cells:
            continue
        x = np.array([c["x"] for _, _, c, _ in cells])
        y = np.array([c["y"] for _, _, c, _ in cells])
        yc = np.array([c["y_l4l"] for _, _, c, _ in cells])
        sl, ic = np.polyfit(x, y, 1)
        aF = [c["p_agree_false"] for _, _, c, _ in cells]
        h4 = [(b, m) for b, m, c, row in cells if row["acc_direct_ci"][0] > row["chance"]]
        out[proto] = {"n_cells": len(cells),
                      "mean_aT": {b: round(float(np.mean([c["p_agree_true"] for bb, _, c, _ in cells if bb == b])), 1) for b in BENCHES},
                      "mean_aF": {b: round(float(np.mean([c["p_agree_false"] for bb, _, c, _ in cells if bb == b])), 1) for b in BENCHES},
                      "aF_range": [round(float(min(aF)), 1), round(float(max(aF)), 1)],
                      "slope": round(float(sl), 3), "intercept": round(float(ic), 2), "r": round(float(np.corrcoef(x, y)[0, 1]), 3),
                      "mae_y_x": round(float(np.mean(np.abs(y - x))), 2), "mae_ycov_x": round(float(np.mean(np.abs(yc - x))), 2),
                      "n_within3": int((np.abs(y - x) <= 3).sum()), "n_sign": int((np.sign(x) == np.sign(y)).sum()),
                      "h4_cells": [f"{b}:{m}" for b, m in h4],
                      "h4_x_neg": int(sum(1 for b, m, c, _ in cells if (b, m) in h4 and c["x"] < 0)),
                      "cells": {f"{b}:{m}": {"x": c["x"], "y": c["y"], "y_cov": c["y_l4l"], "coverage": c["coverage"]} for b, m, c, _ in cells}}
    return out


def fmt(v, ci_=None, sign=True) -> str:
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "--"
    s = f"{v:+.1f}" if sign else f"{v:.1f}"
    return s + (f" [{ci_[0]:+.1f}, {ci_[1]:+.1f}]" if ci_ else "")


def report(R: dict) -> None:
    say("Exp-51 (plan S1): Direct vs CoT vs Filler(|R|)+turn on external multi-image spatial benchmarks")
    say("Pre-registration: outputs/e51/prereg.txt. Primary = CoT - Direct, mean over 6 native multi-image models.")
    say("")
    say("Sanity: Direct re-read on 70 frozen held-out items (default single-image engine) vs stored e6x Direct")
    for m, s in R["sanity"].items():
        say(f"  {NAMES[m]:16s} same prediction {s['n_same_pred']}/{s['n_compared']}  max |d logprob| {s['max_abs_dlogprob']:.4f}")
    for b in BENCHES:
        r = R[b]
        say("")
        say(f"=== {BN[b]}: {r['n_items']} items, {r['n_clusters']} clusters; types {r['qtype_n']}")
        if "pooled_cot-direct" in r:
            say(f"PRIMARY CoT - Direct (mean of {len(r['primary_models'])}): {fmt(r['pooled_cot-direct'], r['pooled_cot-direct_ci'])}")
            say(f"  pooled acc Direct / CoT / Filler+turn: {r['pooled_acc']['direct']:.1f} / {r['pooled_acc']['cot']:.1f} / "
                f"{r['pooled_acc']['filler_R+turn']:.1f}")
            say(f"  CoT - Filler+turn {fmt(r['pooled_cot-filler'], r['pooled_cot-filler_ci'])}; Filler+turn - Direct "
                f"{fmt(r['pooled_filler-direct'], r['pooled_filler-direct_ci'])}")
            if b == "vsi":
                say(f"  non-pruned (debiased) subset CoT - Direct {fmt(r['pooled_nonpruned_cot-direct'])}")
            say(f"  exploratory: CoT - Direct on items with a non-empty CoT continuation (per-model mean) "
                f"{fmt(r['pooled_cot-direct_written'], r['pooled_cot-direct_written_ci'])}")
            say("  empty continuation % / CoT - Direct on written items: " + "; ".join(
                f"{NAMES[m]} {r['models'][m]['empty_cot_pct']:.1f} / {fmt(r['models'][m].get('cot-direct_written'), r['models'][m].get('cot-direct_written_ci'))}"
                for m in M7 if r["models"].get(m, {}).get("complete")))
        say(f"{'model':16s} {'chance':>6s} {'Dir':>6s} {'CoT':>6s} {'Fill':>6s} {'CoT-Dir [CI]':>22s} {'CoT-Fill [CI]':>22s} "
            f"{'D+C-':>5s} {'D-C+':>5s} {'trunc':>6s} {'len':>6s} {'parse D/C/F':>18s} {'stated':>7s}")
        for m in M7:
            x = r["models"].get(m)
            if not x or not x.get("complete"):
                say(f"{NAMES[m]:16s} incomplete {x}")
                continue
            say(f"{NAMES[m] + ('*' if m == 'molmo7b' else ''):16s} {x['chance']:6.1f} {x['acc']['direct']:6.1f} {x['acc']['cot']:6.1f} "
                f"{x['acc']['filler_R+turn']:6.1f} {fmt(x['cot-direct'], x['cot-direct_ci']):>22s} "
                f"{fmt(x['cot-filler'], x['cot-filler_ci']):>22s} {x['DpCm']:5d} {x['DmCp']:5d} {x['cot_trunc_pct']:6.1f} "
                f"{x['cot_len_mean']:6.0f} {'/'.join(f'{v:.0f}' for v in x['readout_parse_pct'].values()):>18s} {x['stated_parse_pct']:7.1f}")
        say("  (* composite image, not in the primary mean)")
        say("  per question type, Direct / CoT (mean of primary models):")
        types = sorted(r["qtype_n"])
        for t in types:
            vals = [(r["models"][m]["by_qtype"][t]["direct"], r["models"][m]["by_qtype"][t]["cot"]) for m in r["primary_models"]]
            say(f"    {t:30s} n {r['qtype_n'][t]:3d}  Dir {np.mean([v[0] for v in vals]):5.1f}  CoT {np.mean([v[1] for v in vals]):5.1f}  "
                f"CoT-Dir {np.mean([v[1] - v[0] for v in vals]):+5.1f}  per model "
                + " ".join(f"{v[1] - v[0]:+.0f}" for v in vals))
        for proto, tc in r.get("two_channel", {}).items():
            say(f"  two-channel ({'conclusion removed, primary' if proto == 'rm' else 'conclusion kept'}): judges agree "
                f"{tc['judge_agree_pct']}% kappa {tc['judge_kappa']} (n {tc['judge_n_both']}); NONE {tc['judge_none_pct']}")
            for m, c in tc["cells"].items():
                say(f"    {NAMES[m]:16s} cov {c['coverage']:5.1f}  W {c.get('written_acc', float('nan')):5.1f}  Dir(cov) "
                    f"{c.get('direct_on_parsed', float('nan')):5.1f}  x {fmt(c.get('x'), c.get('x_ci')):>20s}  y {fmt(c['y'], c['y_ci']):>20s}  "
                    f"y_cov {fmt(c.get('y_l4l'))}  aT/aF {c.get('p_agree_true', float('nan')):5.1f}/{c.get('p_agree_false', float('nan')):5.1f} "
                    f"(nF {c.get('n_false')})  missing {c['missing_labels']}")


def table(R: dict) -> None:
    L = [r"% Exp-51 (rigor_e51.py): Direct vs CoT vs Filler(|R|)+turn on two external multi-image spatial benchmarks",
         r"% (VSI-Bench MC: 600 items, 8 uniform frames; MindCube-Tiny: 600 items, 2-4 views). Not \input; bare tabular.",
         r"% Suggested caption: \caption{\textbf{External multi-image spatial benchmarks (pre-registered, frozen held-out",
         r"% protocol).} Accuracy (\%) and differences (points, 95\% cluster-bootstrap CI; clusters = videos / image groups).",
         r"% $x$ = written-claim accuracy $-$ \texttt{Direct} on covered items, $y$ = \texttt{CoT}$-$\texttt{Direct};",
         r"% written claim = last answer-relevant claim with the conclusion removed, consensus of two text-only judges;",
         r"% $a_T$ / $a_F$ = answer agrees with the claim when it is true / false. Mean = primary endpoint over the six",
         r"% native multi-image models; Molmo-7B-D reads a composite image and is not in the mean.}",
         r"% Wrap in \resizebox{\textwidth}{!}{...}.",
         r"\begin{tabular}{@{}lcccccccc@{}}", r"\toprule",
         r"\textbf{Model} & \texttt{Direct} & \texttt{CoT} & \texttt{Fill.+turn} & \texttt{CoT}$-$\texttt{Direct} & "
         r"\texttt{CoT}$-$\texttt{Fill.} & Cov. & $x$ / $y$ & $a_T$ / $a_F$ \\"]

    def c2(v, c):
        s = lambda z: f"{z:+.1f}".replace("-", "$-$")
        return f"${v:+.1f}$" + r"{\tiny[" + s(c[0]) + "," + s(c[1]) + "]}"

    for b in BENCHES:
        r = R[b]
        tc = r.get("two_channel", {}).get("rm", {}).get("cells", {})
        L += [r"\midrule", rf"\multicolumn{{9}}{{@{{}}l@{{}}}}{{\textit{{\textbf{{{BN[b]}}}}}}} \\"]
        for m in M7:
            x = r["models"].get(m)
            if not x or not x.get("complete"):
                continue
            t = tc.get(m, {})
            xy = f"${t['x']:+.1f}$ / ${t['y']:+.1f}$" if "x" in t else "--"
            ag = (f"${t['p_agree_true']:.0f}$ / ${t['p_agree_false']:.0f}$" if "p_agree_true" in t and not math.isnan(t["p_agree_false"]) else "--")
            cov = f"${t['coverage']:.0f}$" if t else "--"
            dag = r"$^\dagger$" if m == "molmo7b" else ""
            L.append(f"{NAMES[m]}{dag} & ${x['acc']['direct']:.1f}$ & ${x['acc']['cot']:.1f}$ & "
                     f"${x['acc']['filler_R+turn']:.1f}$ & {c2(x['cot-direct'], x['cot-direct_ci'])} & "
                     f"{c2(x['cot-filler'], x['cot-filler_ci'])} & {cov} & {xy} & {ag} \\\\")
        if "pooled_cot-direct" in r:
            pa = r["pooled_acc"]
            L.append(rf"\textbf{{Mean of {len(r['primary_models'])}}} & ${pa['direct']:.1f}$ & ${pa['cot']:.1f}$ & ${pa['filler_R+turn']:.1f}$ & "
                     rf"\textbf{{{c2(r['pooled_cot-direct'], r['pooled_cot-direct_ci'])}}} & {c2(r['pooled_cot-filler'], r['pooled_cot-filler_ci'])} & & & \\")
    L += [r"\bottomrule", r"\multicolumn{9}{@{}l@{}}{\footnotesize $^\dagger$ composite image (one-image model); not in the mean.} \\",
          r"\end{tabular}"]
    (PAPER / "tables" / "supp_e51.tex").write_text("\n".join(L) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--nb", type=int, default=2000)
    a = ap.parse_args()
    R: dict = {"sanity": {}}
    for m in M7:
        p = E51 / f"sanity_{m}.json"
        if p.exists():
            R["sanity"][m] = json.load(open(p))
    for b in BENCHES:
        R[b] = analyze_bench(b, a.nb)
    pts = [(R[b]["models"][m]["acc"]["direct"] - R[b]["models"][m]["chance"], R[b]["models"][m]["cot-direct"])
           for b in BENCHES for m in M6 if R[b]["models"].get(m, {}).get("complete")]
    if len(pts) > 2:
        xa, ya = np.array(pts).T
        R["exploratory_direct_above_chance_vs_cot_minus_direct"] = {
            "n_cells": len(pts), "pearson_r": round(float(np.corrcoef(xa, ya)[0, 1]), 3),
            "ols_slope": round(float(np.polyfit(xa, ya, 1)[0]), 3), "cells": [list(map(float, p)) for p in pts]}
    R["two_channel_summary"] = tc_summary(R)
    report(R)
    say("")
    for proto, t in R["two_channel_summary"].items():
        say(f"Two-channel summary ({proto}; 6 primary models x 2 benchmarks = {t['n_cells']} cells): mean aT / aF "
            f"VSI {t['mean_aT']['vsi']} / {t['mean_aF']['vsi']}, MindCube {t['mean_aT']['mc']} / {t['mean_aF']['mc']}; "
            f"aF range {t['aF_range']}; y on x slope {t['slope']} intercept {t['intercept']} r {t['r']}; "
            f"mean |y - x| {t['mae_y_x']} (covered-item y: {t['mae_ycov_x']}); |y - x| <= 3 in {t['n_within3']}/{t['n_cells']}; "
            f"sign(x) = sign(y) {t['n_sign']}/{t['n_cells']}")
        say(f"  H4 cells (Direct CI lower bound > chance): {t['h4_cells']}; x < 0 in {t['h4_x_neg']}/{len(t['h4_cells'])}")
    if "exploratory_direct_above_chance_vs_cot_minus_direct" in R:
        e = R["exploratory_direct_above_chance_vs_cot_minus_direct"]
        say("")
        say(f"Exploratory (not pre-registered): across {e['n_cells']} primary model x benchmark cells, CoT - Direct vs "
            f"Direct - chance: r = {e['pearson_r']}, slope {e['ols_slope']}")
    B.mkdir(parents=True, exist_ok=True)
    json.dump(R, open(B / "e51_extbench.json", "w"), indent=1, default=float)
    (B / "rigor_e51.txt").write_text("\n".join(OUT) + "\n")
    table(R)


if __name__ == "__main__":
    main()
