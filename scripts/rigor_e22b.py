"""Exp-22c analysis: is the commit-sentence LOCATION causal? Gate vs location- / count- / criterion-matched controls.

Inputs: outputs/e22/<model>_<set>.jsonl (direct, cot, gate, rand, tgate) and outputs/e22b/<model>_<set>.jsonl
(locfree = A, randfree = B, randsel = C; see e22b_gate.py). All arms share the identical plain rationale and Direct answer.
Key contrasts (paired cluster bootstrap over twin pairs / Synthetic Depth images, 2,000 resamples):
  A - B       location effect without the Direct filter (one unchecked sample per event, equal event counts)
  Gate - C    location effect with the filter (best-of-4 toward Direct, equal event counts)
  C - B       selection effect at random locations
  Gate - A    selection effect at flagged locations
on accuracy, agreement with the Direct answer, pooled d', Synthetic Depth d'_pair.
CPU analyses on e22 data (review round 13, M2 / m7 / Q1-4):
  1 first-sample non-flip at flagged positions (gate cand_p[0], locfree) vs readout change of one sample at random positions
  2 replaced-sentence claim truth split by Direct-correct / Direct-wrong items
  3 gate gain on items with resample-only events vs items with >= 1 drop
  4 turn-closed criterion readout vs in-turn outcome readout on the final prefix (disagreement)
  5 gate events at j* (gold-defined last crossing of the plain trajectory) on D+C- items
-> outputs/bundle/e22b_gate.json, rigor_e22b.txt, supp_gate_loc.tex
Usage: python rigor_e22b.py [--nboot 2000]
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

import rigor_e22 as R22
from rigor_e22 import LABEL, LOGIT_PROBIT, MODELS, O, SETS, dprime, pred, q95, sent_claim

B = Path("outputs/bundle")
ARMS = ["direct", "cot", "gate", "rand", "locfree", "randfree", "randsel"]
CONTRASTS = [("locfree", "randfree"), ("gate", "randsel"), ("randsel", "randfree"), ("gate", "locfree"),
             ("locfree", "cot"), ("randfree", "cot"), ("randsel", "cot"), ("gate", "cot"), ("gate", "rand"),
             ("locfree", "direct"), ("randfree", "direct"), ("randsel", "direct"), ("gate", "direct")]
PRIORITY = [("molmo7b", "cues"), ("molmo7b", "twins"), ("molmo7b", "syn"), ("qwen3vl_8b", "syn"), ("qwen7b", "syn"),
            ("internvl3_8b", "syn")]


def load(m: str, s: str) -> dict | None:
    p22, p22b = O / "e22" / f"{m}_{s}.jsonl", O / "e22b" / f"{m}_{s}.jsonl"
    if not p22.exists():
        return None
    by: dict[str, dict[str, dict]] = {}
    for p in (p22, p22b):
        if p.exists():
            for line in open(p):
                r = json.loads(line)
                if r["arm"] != "tgate":
                    by.setdefault(r["arm"], {})[r["item_id"]] = r
    arms = [a for a in ARMS if a in by]
    keys = sorted(by["direct"])
    clusters = sorted({by["direct"][k]["image_id"] for k in keys})
    cid = {c: i for i, c in enumerate(clusters)}
    d = {"by": by, "arms": arms, "keys": keys, "cl": np.array([cid[by["direct"][k]["image_id"]] for k in keys]),
         "ncl": len(clusters)}
    r0 = [by["direct"][k] for k in keys]
    d["tru"] = (np.array([r["gold"] == "A" for r in r0]) if s == "syn"
                else np.array([r["meta"]["front_name"].lower() == r["x"] for r in r0]))
    d["behind"] = np.array([r["phrasing"] == "behind" for r in r0])
    dans = [by["cot"][k]["direct_ans"] for k in keys]
    for a in arms:
        p = [pred(by[a][k]) for k in keys]
        d[f"corr_{a}"] = np.array([pp == by["direct"][k]["gold"] for pp, k in zip(p, keys)])
        d[f"says_{a}"] = np.array([(pp == "A") != b for pp, b in zip(p, d["behind"])])
        d[f"agr_{a}"] = np.array([pp == x for pp, x in zip(p, dans)])
    if s == "syn":
        o_idx, s_idx = {}, {}
        for i, k in enumerate(keys):
            (s_idx if k.endswith("_cfq") else o_idx)[by["direct"][k]["image_id"]] = i
        imgs = sorted(set(o_idx) & set(s_idx))
        d["pair_o"], d["pair_s"] = np.array([o_idx[g] for g in imgs]), np.array([s_idx[g] for g in imgs])
        d["pair_cl"] = np.array([cid[g] for g in imgs])
    return d


def stats(d: dict, s: str, idx: np.ndarray, pidx: np.ndarray | None) -> dict[str, float]:
    out = {}
    for a in d["arms"]:
        out[f"acc_{a}"] = 100 * d[f"corr_{a}"][idx].mean()
        out[f"agr_{a}"] = 100 * d[f"agr_{a}"][idx].mean()
        out[f"dp_{a}"] = dprime(d[f"says_{a}"][idx], d["tru"][idx])
        if s == "syn" and pidx is not None:
            co, cs = d[f"corr_{a}"][d["pair_o"][pidx]], d[f"corr_{a}"][d["pair_s"][pidx]]
            out[f"dpair_{a}"] = math.log(((co & cs).sum() + .5) / ((~co & ~cs).sum() + .5)) / LOGIT_PROBIT
    for a, b in CONTRASTS:
        if a in d["arms"] and b in d["arms"]:
            for mtr in ("acc", "agr", "dp", "dpair"):
                if f"{mtr}_{a}" in out:
                    out[f"d{mtr}_{a}-{b}"] = out[f"{mtr}_{a}"] - out[f"{mtr}_{b}"]
    # location x selection interaction: (Gate - A) - (C - B) = (Gate - C) - (A - B)
    for mtr in ("acc", "agr", "dp", "dpair"):
        if f"d{mtr}_gate-locfree" in out and f"d{mtr}_randsel-randfree" in out:
            out[f"i{mtr}"] = out[f"d{mtr}_gate-locfree"] - out[f"d{mtr}_randsel-randfree"]
    return out


def boot(d: dict, s: str, nb: int, seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    members = [np.where(d["cl"] == c)[0] for c in range(d["ncl"])]
    pmembers = [np.where(d["pair_cl"] == c)[0] for c in range(d["ncl"])] if s == "syn" else None
    full = stats(d, s, np.arange(len(d["keys"])), np.arange(len(d["pair_o"])) if s == "syn" else None)
    bs = {k: [] for k in full}
    for _ in range(nb):
        cs = rng.integers(0, d["ncl"], d["ncl"])
        idx = np.concatenate([members[c] for c in cs])
        pidx = np.concatenate([pmembers[c] for c in cs]) if s == "syn" else None
        for k, v in stats(d, s, idx, pidx).items():
            bs[k].append(v)
    return {k: {"est": round(float(v), 3), "ci": [round(x, 3) for x in q95(bs[k])]} for k, v in full.items()}


def sub_boot(d: dict, mask: np.ndarray, a: str, b: str, nb: int = 2000, seed: int = 1) -> dict | None:
    """Accuracy difference a - b on a subset of items (cluster bootstrap restricted to the subset)."""
    idx = np.where(mask)[0]
    if len(idx) == 0:
        return None
    cl = d["cl"][idx]
    u = np.unique(cl)
    members = [idx[cl == c] for c in u]
    diff = d[f"corr_{a}"].astype(float) - d[f"corr_{b}"].astype(float)
    rng = np.random.default_rng(seed)
    bs = [diff[np.concatenate([members[c] for c in rng.integers(0, len(u), len(u))])].mean() for _ in range(nb)]
    return {"n": int(len(idx)), "est": round(100 * float(diff[idx].mean()), 2), "ci": [round(100 * x, 2) for x in q95(bs)]}


def jstar(traj: list[float]) -> int | None:
    """1-based sentence index j* (gold readout; = Direct readout on D+C- items): last crossing, never recovers."""
    return next((j for j in range(1, len(traj)) if traj[j - 1] >= .5 and all(v < .5 for v in traj[j:])), None)


def pct(x) -> float | None:
    x = list(x)
    return round(100 * float(np.mean(x)), 1) if x else None


def cpu_analyses(d: dict) -> dict:
    by, keys = d["by"], d["keys"]
    cD, cC, cG = d["corr_direct"], d["corr_cot"], d["corr_gate"]
    gate = [by["gate"][k] for k in keys]
    cot = [by["cot"][k] for k in keys]
    out = {}
    # 1) first-sample non-flip rate and readout change
    ge = [(e, i) for i, r in enumerate(gate) for e in r["events"]]
    g1 = [r["events"][0] for r in gate if r["events"]]
    a1 = {"gate_all_events_first_sample_keeps_pct": pct(e["cand_p"][0] >= .5 for e, _ in ge),
          "gate_first_event_first_sample_keeps_pct": pct(e["cand_p"][0] >= .5 for e in g1),
          "gate_any_of_K_keeps_pct": pct(any(p >= .5 for p in e["cand_p"]) for e, _ in ge),
          "gate_first_sample_dP_vs_greedy": round(float(np.mean([e["cand_p"][0] - e["p_greedy"] for e, _ in ge])), 3) if ge else None,
          "n_gate_events": len(ge)}
    if "locfree" in d["arms"]:
        le = [e for k in keys for e in by["locfree"][k]["events"]]
        a1 |= {"locfree_sample_keeps_pct": pct(e["cand_p"][0] >= .5 for e in le),
               "locfree_dP_vs_greedy": round(float(np.mean([e["cand_p"][0] - e["p_greedy"] for e in le])), 3) if le else None,
               "n_locfree_events": len(le)}
    if "randfree" in d["arms"]:
        re_ = [e for k in keys for e in by["randfree"][k]["events"]]
        hi = [e for e in re_ if e["p_before"] >= .5]
        fl = [e for e in hi if e["p_greedy"] < .5]
        a1 |= {"randfree_pbefore_hi_pct": pct(e["p_before"] >= .5 for e in re_),
               "randfree_sample_keeps_given_hi_pct": pct(e["cand_p"][0] >= .5 for e in hi),
               "randfree_greedy_was_flip_given_hi_pct": pct(e["p_greedy"] < .5 for e in hi),
               "randfree_sample_keeps_given_flip_pct": pct(e["cand_p"][0] >= .5 for e in fl),
               "randfree_dP_vs_greedy": round(float(np.mean([e["cand_p"][0] - e["p_greedy"] for e in re_])), 3) if re_ else None,
               "randfree_abs_dP_vs_greedy": round(float(np.mean([abs(e["cand_p"][0] - e["p_greedy"]) for e in re_])), 3) if re_ else None,
               "n_randfree_events": len(re_), "n_randfree_flip_events": len(fl)}
    if "locfree" in d["arms"]:
        le = [e for k in keys for e in by["locfree"][k]["events"]]
        a1["locfree_abs_dP_vs_greedy"] = round(float(np.mean([abs(e["cand_p"][0] - e["p_greedy"]) for e in le])), 3) if le else None
    out["first_sample"] = a1
    # 2) replaced-sentence claim truth by Direct correctness
    a2 = {}
    for lab, want in (("direct_correct", True), ("direct_wrong", False)):
        t = [sent_claim(e["greedy"], r) == r["gold"] for r, c in zip(gate, cD) if c == want for e in r["events"]
             if sent_claim(e["greedy"], r) is not None]
        kept = [sent_claim(e["cands"][e["tries"] - 1], r) == r["gold"] for r, c in zip(gate, cD) if c == want
                for e in r["events"] if e["type"] == "resample" and sent_claim(e["cands"][e["tries"] - 1], r) is not None]
        a2[lab] = {"n_items": int((cD == want).sum()), "n_events": sum(len(r["events"]) for r, c in zip(gate, cD) if c == want),
                   "n_replaced_claims": len(t), "replaced_claim_true_pct": pct(t),
                   "n_kept_claims": len(kept), "kept_claim_true_pct": pct(kept)}
    out["replaced_claim_truth"] = a2
    # 3) gate gain: resample-only vs >= 1 drop
    ev_types = [[e["type"] for e in r["events"]] for r in gate]
    res_only = np.array([bool(t) and all(x == "resample" for x in t) for t in ev_types])
    any_drop = np.array(["drop" in t for t in ev_types])
    out["gain_by_drop"] = {lab: {"gate-cot": sub_boot(d, m_, "gate", "cot"), "gate-direct": sub_boot(d, m_, "gate", "direct"),
                                 "cot-direct": sub_boot(d, m_, "cot", "direct")}
                           for lab, m_ in (("resample_only", res_only), ("any_drop", any_drop))}
    # 4) closed-turn criterion readout vs in-turn outcome readout on the final prefix
    a4 = {}
    for a in ("cot", "gate", "locfree"):
        if a not in d["arms"]:
            continue
        rs = [by[a][k] for k in keys]
        closed = np.array([r["traj"][-1] >= .5 for r in rs])
        inturn = np.array([pred(r) == r["direct_ans"] for r in rs])
        a4[a] = {"disagree_pct": pct(closed != inturn), "closed_d_inturn_not_pct": pct(closed & ~inturn),
                 "closed_not_inturn_d_pct": pct(~closed & inturn)}
    out["readout_mismatch"] = a4
    # 5) gate events at j* on D+C- items (D+C- from e22's Direct / plain CoT)
    dcm = np.where(cD & ~cC)[0]
    js = [jstar(cot[i]["traj"]) for i in dcm]
    first = [gate[i]["events"][0]["pos"] if gate[i]["events"] else None for i in dcm]
    ncross = [sum(t0 >= .5 > t1 for t0, t1 in zip(cot[i]["traj"][:-1], cot[i]["traj"][1:])) for i in dcm]
    has = [j is not None for j in js]
    out["jstar"] = {"n_DpCm": int(len(dcm)), "jstar_defined": int(sum(has)),
                    "gated": int(sum(f is not None for f in first)),
                    "first_event_at_jstar": int(sum(f is not None and f == j for f, j in zip(first, js))),
                    "first_event_before_jstar": int(sum(f is not None and j is not None and f < j for f, j in zip(first, js))),
                    "jstar_but_not_gated": int(sum(f is None and j is not None for f, j in zip(first, js))),
                    "gated_no_jstar": int(sum(f is not None and j is None for f, j in zip(first, js))),
                    "multi_crossing_items": int(sum(c > 1 for c in ncross)),
                    "gate_events_on_DpCm": int(sum(len(gate[i]["events"]) for i in dcm)),
                    "gate_events_total": int(sum(len(r["events"]) for r in gate)),
                    "repaired_DpCm_first_at_jstar": int(sum(f is not None and f == j and cG[i] for f, j, i in zip(first, js, dcm))),
                    "repaired_DpCm": int(sum(cG[i] for i in dcm))}
    return out


def match_stats(d: dict) -> dict:
    by, keys = d["by"], d["keys"]
    out = {}
    nev = lambda a: np.array([len(by[a][k]["events"]) for k in keys]) if a in d["arms"] else None
    g, a_, b_, c_, r_ = nev("gate"), nev("locfree"), nev("randfree"), nev("randsel"), nev("rand")
    out["events_total"] = {a: int(x.sum()) for a, x in (("gate", g), ("rand", r_), ("locfree", a_), ("randfree", b_),
                                                          ("randsel", c_)) if x is not None}
    if a_ is not None and b_ is not None:
        out["randfree_eq_locfree_items_pct"] = pct(a_ == b_)
        out["randfree_fallback_events"] = int(sum(by["randfree"][k]["n_fallback"] for k in keys))
    if c_ is not None:
        out["randsel_eq_gate_items_pct"] = pct(c_ == g)
        out["randsel_fallback_events"] = int(sum(by["randsel"][k]["n_fallback"] for k in keys))
        t = [e["type"] for k in keys for e in by["randsel"][k]["events"]]
        out["randsel_drops"] = t.count("drop")
        out["gate_drops"] = sum(e["type"] == "drop" for k in keys for e in by["gate"][k]["events"])
    if a_ is not None:
        out["locfree_gated_pct"] = pct(a_ > 0)
        # first locfree event sits where the gate's first event sits (same plain prefix); same sample when seeds coincide
        same_pos = [by["locfree"][k]["events"][0]["pos"] == by["gate"][k]["events"][0]["pos"]
                    for k in keys if by["locfree"][k]["events"] and by["gate"][k]["events"]]
        same_smp = [by["locfree"][k]["events"][0]["cands"][0] == by["gate"][k]["events"][0]["cands"][0]
                    for k in keys if by["locfree"][k]["events"] and by["gate"][k]["events"]]
        out["first_event_same_pos_as_gate_pct"] = pct(same_pos)
        out["first_event_same_sample_as_gate_pct"] = pct(same_smp)
        gated = a_ > 0
        out["locfree_gated_vs_gate_gated"] = {"both": int((gated & (g > 0)).sum()), "gate_only": int((~gated & (g > 0)).sum()),
                                              "locfree_only": int((gated & (g == 0)).sum())}
    # zero-event items should reproduce the plain CoT answer
    for a in ("randfree", "randsel"):
        if a in d["arms"]:
            z = [k for k in keys if not by[a][k]["events"]]
            out[f"{a}_zero_event_same_pred_as_cot_pct"] = pct(pred(by[a][k]) == pred(by["cot"][k]) for k in z)
    return out


def fmt(x: dict, nd: int = 1) -> str:
    return f"{x['est']:+.{nd}f} [{x['ci'][0]:+.{nd}f}, {x['ci'][1]:+.{nd}f}]"


def sig(x: dict) -> bool:
    return x["ci"][0] > 0 or x["ci"][1] < 0


def latex(res: dict) -> str:
    SL = {"cues": "Cues", "twins": "Twins", "syn": "Syn.\\ Depth"}
    cell = lambda b, k, nd=1: ((r"\textbf{" if sig(b[k]) else "{") + fmt(b[k], nd) + "}").replace("-", "$-$") if k in b else "--"
    L = [r"\begin{table*}[t]", r"\centering", r"\fontsize{6.2}{7.4}\selectfont", r"\setlength{\tabcolsep}{2.4pt}",
         r"\caption{\textbf{Is the gate's sentence location causal?} All arms start from the same plain greedy rationale and "
         r"Direct answer as \cref{tab:supp_gate} (re-run in that experiment; see its caption). \emph{Gate}: readout-flipping sentences resampled toward Direct (best of "
         r"up to $K{=}4$ by $P(\text{Direct})\ge0.5$, else dropped). \emph{A (loc.)}: fires where the gate's criterion fires "
         r"on the evolving trajectory, but keeps \emph{one} unchecked $T{=}0.7$ sample. \emph{B (rand.)}: exactly as many "
         r"single unchecked samples per item as A, at random sentences of the evolving text. \emph{C (rand.+sel.)}: exactly as "
         r"many events per item as the Gate, at random sentences, with the Gate's best-of-4 acceptance and drop rule. "
         r"A$-$B: location without the Direct filter; Gate$-$C: location with the filter; C$-$B: selection toward Direct at "
         r"random locations. Differences in points with $95\%$ paired cluster-bootstrap CIs ($2{,}000$ resamples); bold: CI "
         r"excludes 0. $^\dagger$cells where CoT loses significantly to Direct.}",
         r"\label{tab:supp_gate_loc}",
         r"\begin{tabular}{@{}llcccccc@{\hspace{5pt}}cccc@{}}", r"\toprule",
         r"\multicolumn{12}{@{}l}{\textbf{(a) Accuracy (\%)}} \\",
         r"Model & Set & Direct & CoT & Gate & A & B & C & A$-$B & Gate$-$C & C$-$B & Gate$-$A \\", r"\midrule"]
    for m in MODELS:
        for s in SETS:
            r = res.get(m, {}).get(s)
            if not r or "acc_locfree" not in r["boot"]:
                continue
            b = r["boot"]
            dag = r"$^\dagger$" if (m, s) in PRIORITY else ""
            L.append(f"{LABEL[m] if s == 'cues' else ''} & {SL[s]}{dag} & "
                     + " & ".join(f"{b[f'acc_{a}']['est']:.1f}" for a in ("direct", "cot", "gate", "locfree", "randfree", "randsel"))
                     + f" & {cell(b, 'dacc_locfree-randfree')} & {cell(b, 'dacc_gate-randsel')} & {cell(b, 'dacc_randsel-randfree')}"
                       f" & {cell(b, 'dacc_gate-locfree')} \\\\")
        L.append(r"\addlinespace[1pt]")
    L[-1] = r"\bottomrule"
    L += [r"\end{tabular}", r"\par\vspace{4pt}",
          r"\begin{tabular}{@{}llcccccc@{}}", r"\toprule",
          r"\multicolumn{8}{@{}l}{\textbf{(b) Agreement with Direct, sensitivity, events}} \\",
          r"Model & Set & \shortstack{Agree w/ D (\%)\\Gate / A / B / C} & "
          r"\shortstack{$d'$\\A$-$B} & \shortstack{$d'$\\Gate$-$C} & "
          r"\shortstack{$d'$\\C$-$B} & \shortstack{Events\\A = B} & \shortstack{Events\\Gate = C} \\", r"\midrule"]
    for m in MODELS:
        for s in SETS:
            r = res.get(m, {}).get(s)
            if not r or "acc_locfree" not in r["boot"]:
                continue
            b, ms = r["boot"], r["match"]
            dag = r"$^\dagger$" if (m, s) in PRIORITY else ""
            ag = " / ".join(f"{b[f'agr_{a}']['est']:.1f}" for a in ("gate", "locfree", "randfree", "randsel"))
            dk = "dpair" if s == "syn" else "dp"
            ev = ms["events_total"]
            L.append(f"{LABEL[m] if s == 'cues' else ''} & {SL[s]}{dag} & {ag} & "
                     f"{cell(b, f'd{dk}_locfree-randfree', 2)} & {cell(b, f'd{dk}_gate-randsel', 2)} & {cell(b, f'd{dk}_randsel-randfree', 2)} & "
                     f"{ev['locfree']} / {ev['randfree']} & {ev['gate']} / {ev['randsel']} \\\\")
        L.append(r"\addlinespace[1pt]")
    L[-1] = r"\bottomrule"
    L += [r"\end{tabular}",
          r"\par\vspace{2pt}\raggedright\footnotesize Agree w/ D: final answer equals the model's own Direct answer. $d'$ "
          r"pooled over both phrasings (Cues, Twins); on Syn.\ Depth the pair-matched $d'_{\text{pair}}=\ln[(\mathrm{BC}+.5)/"
          r"(\mathrm{BW}+.5)]/1.702$. Events: total resample/drop events (A, B: single unchecked samples; Gate, C: up to 4 "
          r"samples, drops included); B and C match A and the Gate item by item. A's first event is at the Gate's first event "
          r"(same plain prefix); later events of A and the Gate differ because their texts diverge.",
          r"\end{table*}"]
    return "\n".join(L) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--nboot", type=int, default=2000)
    ap.add_argument("--out", default=str(B))
    ap.add_argument("--from-json", action="store_true", help="only rewrite supp_gate_loc.tex from an existing e22b_gate.json")
    args = ap.parse_args()
    out = Path(args.out)
    if args.from_json:
        (out / "supp_gate_loc.tex").write_text(latex(json.load(open(out / "e22b_gate.json"))))
        return
    res, lines = {}, []
    for m in MODELS:
        for s in SETS:
            d = load(m, s)
            if d is None:
                lines.append(f"MISSING {m} {s}")
                continue
            r = {"n": len(d["keys"]), "n_clusters": d["ncl"], "arms": d["arms"], "boot": boot(d, s, args.nboot),
                 "cpu": cpu_analyses(d), "match": match_stats(d), "desc": R22.descriptives(d)}
            res.setdefault(m, {})[s] = r
            b = r["boot"]
            lines.append(f"== {m} {s} n={r['n']} clusters={r['n_clusters']} arms={','.join(d['arms'])}"
                         + ("  [CoT loses significantly]" if (m, s) in PRIORITY else ""))
            lines.append("  acc  " + "  ".join(f"{a} {b[f'acc_{a}']['est']:.1f}" for a in d["arms"]))
            lines.append("  agrD " + "  ".join(f"{a} {b[f'agr_{a}']['est']:.1f}" for a in d["arms"]))
            for a, c in CONTRASTS:
                if f"dacc_{a}-{c}" in b:
                    ln = f"  {a}-{c}: acc {fmt(b[f'dacc_{a}-{c}'])}  agrD {fmt(b[f'dagr_{a}-{c}'])}  d' {fmt(b[f'ddp_{a}-{c}'], 2)}"
                    if f"ddpair_{a}-{c}" in b:
                        ln += f"  d'pair {fmt(b[f'ddpair_{a}-{c}'], 2)}"
                    lines.append(ln)
            if "iacc" in b:
                lines.append(f"  interaction (Gate-A)-(C-B): acc {fmt(b['iacc'])}  agrD {fmt(b['iagr'])}  d' {fmt(b['idp'], 2)}"
                             + (f"  d'pair {fmt(b['idpair'], 2)}" if "idpair" in b else ""))
            lines.append("  match " + json.dumps(r["match"]))
            for k, v in r["cpu"].items():
                lines.append(f"  cpu.{k} " + json.dumps(v))
            for a in ("gate", "locfree", "randfree", "randsel"):
                if a in r["desc"]:
                    g = r["desc"][a]
                    keep = ("agree_direct_pct", "repaired", "cot_DpCm", "rescues_lost", "cot_DmCp", "consistency_pct",
                            "gen_tok_ratio_vs_cot", "readouts_mean", "events_mean", "resamples", "drops", "tries_mean",
                            "gated_pct", "body_tok_mean")
                    lines.append(f"  desc {a:8s} " + json.dumps({k: g.get(k) for k in keep}))
    out.mkdir(parents=True, exist_ok=True)
    json.dump(res, open(out / "e22b_gate.json", "w"), indent=1)
    (out / "rigor_e22b.txt").write_text("\n".join(lines) + "\n")
    (out / "supp_gate_loc.tex").write_text(latex(res))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
