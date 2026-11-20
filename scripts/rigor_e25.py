"""Exp-25 analysis: answer-first-then-explain (AFE) vs plain CoT vs the commit-sentence gate, on the same items.

Inputs: outputs/e22/<model>_<set>.jsonl (arms direct / cot / gate) and outputs/e25/<model>_<set>.jsonl (arm afe).
Claims: every sentence of the rationale body (e2b split, as rigor_e22) is parsed with claim_parse v3 (rigor_e22.sent_claim,
the parser behind kept_sample_claim_true_pct / replaced_sent_claim_true_pct); a parsed two-object depth-order claim is
true iff the answer it implies equals the gold answer. CoT / gate body = text after "Let's think step by step.";
AFE body = text after "Answer: (d)\nExplanation: Let's think step by step." (the prefix itself is not parsed).
Per arm (answer = same-turn readout for CoT / gate, the stated Direct answer d for AFE):
  claim truth (% true of all parsed claims; also on Direct-correct / Direct-wrong items; cues / twins also by vertical
  congruence: front object lower = lower-is-closer congruent);
  items with >= 1 false claim; consistency (answer = last parsed claim, items with a claim); any claim contradicting
  the answer;
  rationalization on Direct-wrong items: >= 1 claim implying d (false, since d is wrong); last claim implies d;
  dissent: >= 1 true claim (contradicts d);
  tokens: generated tokens per item (gate: incl. discarded samples), kept rationale tokens.
Exp-26 (round-14 M1 / m9), answer-matched and claim-only variants:
  rat_mcot_{cot,afe}: rationalization on CoT's D-C- items (Direct wrong and CoT ends at d), CoT vs AFE on the same items;
  rat_mgate_{gate,afe}: same on the gate's D-G- items; rat_rcot_{cot,afe}: on CoT's D-C+ rescues;
  *_xc: sentences matching rigor_e22.CONCL (conclusion / restatement regex) excluded before parsing
    (claim_true_xc, rat_any_xc, ratxc_mcot_*, ratxc_mgate_*);
  post_acc_afe: accuracy of the forced readout after the AFE explanation (stated answer held at d), vs acc_direct.
Paired cluster bootstrap (twin pair / Synthetic Depth image; 2,000) per cell; pooled = unweighted mean over the 12
cells with independent per-cell resampling.
-> outputs/bundle/e25_afe.json, outputs/bundle/rigor_e25.txt, outputs/bundle/supp_afe.tex
Usage: python rigor_e25.py [--nboot 2000] [--suffix _limit10]
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np

from rigor_e22 import CONCL, COT_PREFIX, LABEL, MODELS, O, SETS, pred, q95, sent_claim, split_sentences

B = Path("outputs/bundle")
ARMS = ["cot", "gate", "afe"]
ALABEL = {"cot": "CoT", "gate": "Gate", "afe": "AFE"}
DIFFS = [("afe", "cot"), ("afe", "gate"), ("gate", "cot")]
MAIN = ["claim_true", "claim_true_dc", "claim_true_dw", "any_false", "any_false_claimed", "consistency", "contra",
        "rat_any", "rat_last", "dissent", "tok", "body_tok", "claimed", "claims_per_item", "claim_true_vinc", "claim_true_vcon",
        "vgap", "claim_true_xc", "rat_any_xc"]
# answer-matched populations: (key, arm whose end answer defines the population, end at d?)
MATCHED = [("mcot", "cot", True), ("mgate", "gate", True), ("rcot", "cot", False)]
MDIFFS = [("rat_mcot", "cot"), ("rat_mgate", "gate"), ("rat_rcot", "cot"), ("ratxc_mcot", "cot"), ("ratxc_mgate", "gate")]
SFX = ""
QUOTED = re.compile(r"[\"\u201c\u201d]|\bstatement\b", re.I)
EXCL_QUOTED = False


def body_of(r: dict) -> str:
    return r["body"] if r["arm"] == "afe" else r["text"][len(COT_PREFIX):].strip()


def sents_of(r: dict) -> list[str]:
    """Rationale sentences; with --exclude-quoted, sentences quoting / restating the question are skipped."""
    ss = split_sentences(body_of(r))
    return [x for x in ss if not QUOTED.search(x)] if EXCL_QUOTED else ss


def load(m: str, s: str) -> dict | None:
    p22, p25 = O / "e22" / f"{m}_{s}.jsonl", O / "e25" / f"{m}_{s}{SFX}.jsonl"
    if not p25.exists():
        return None
    by: dict[str, dict] = {}
    for l in open(p22):
        r = json.loads(l)
        if r["arm"] in ("direct", "cot", "gate"):
            by.setdefault(r["arm"], {})[r["item_id"]] = r
    for l in open(p25):
        r = json.loads(l)
        by.setdefault("afe", {})[r["item_id"]] = r
    keys = sorted(by["afe"])
    clusters = sorted({by["direct"][k]["image_id"] for k in keys})
    cid = {c: i for i, c in enumerate(clusters)}
    d = {"keys": keys, "cl": np.array([cid[by["direct"][k]["image_id"]] for k in keys]), "ncl": len(clusters)}
    dr = [by["direct"][k] for k in keys]
    d_ans = [pred(r) for r in dr]
    d["cD"] = np.array([a == r["gold"] for a, r in zip(d_ans, dr)])
    d["afe_is_direct"] = all(by["afe"][k]["ans"] == a for k, a in zip(keys, d_ans))
    d["post_agree"] = 100 * float(np.mean([by["afe"][k]["post_pred"] == a for k, a in zip(keys, d_ans)]))
    d["post_acc"] = 100 * float(np.mean([by["afe"][k]["post_pred"] == r["gold"] for k, r in zip(keys, dr)]))
    d["post_corr"] = np.array([by["afe"][k]["post_pred"] == r["gold"] for k, r in zip(keys, dr)])
    d["eos_pct"] = 100 * float(np.mean([by["afe"][k]["eos"] for k in keys]))
    meta = [r.get("meta") or {} for r in dr]
    d["vinc"] = np.array([bool(mm.get("front_above")) for mm in meta])          # front object higher: lower-is-closer misleads
    d["has_vert"] = s != "syn"
    for a in ARMS:
        rs = [by[a][k] for k in keys]
        ans = [r["ans"] if a == "afe" else pred(r) for r in rs]
        cas = [[c for c in (sent_claim(x, r) for x in sents_of(r)) if c is not None] for r in rs]
        d[f"{a}_nc"] = np.array([len(c) for c in cas], float)
        d[f"{a}_nt"] = np.array([sum(x == r["gold"] for x in c) for c, r in zip(cas, rs)], float)
        d[f"{a}_has"] = d[f"{a}_nc"] > 0
        d[f"{a}_anyF"] = d[f"{a}_nt"] < d[f"{a}_nc"]
        d[f"{a}_cons"] = np.array([bool(c) and c[-1] == p for c, p in zip(cas, ans)])
        d[f"{a}_contra"] = np.array([any(x != p for x in c) for c, p in zip(cas, ans)])
        d[f"{a}_ratA"] = np.array([any(x == da for x in c) for c, da in zip(cas, d_ans)])
        d[f"{a}_ratL"] = np.array([bool(c) and c[-1] == da for c, da in zip(cas, d_ans)])
        d[f"{a}_endD"] = np.array([p == da for p, da in zip(ans, d_ans)])
        cxs = [[c for c in (sent_claim(x, r) for x in sents_of(r) if not CONCL.search(x)) if c is not None] for r in rs]
        d[f"{a}_ncX"] = np.array([len(c) for c in cxs], float)
        d[f"{a}_ntX"] = np.array([sum(x == r["gold"] for x in c) for c, r in zip(cxs, rs)], float)
        d[f"{a}_ratAX"] = np.array([any(x == da for x in c) for c, da in zip(cxs, d_ans)])
        d[f"{a}_tok"] = np.array([r["gen_tok"] for r in rs], float)
        d[f"{a}_btok"] = np.array([r["body_tok"] for r in rs], float)
        d[f"{a}_acc"] = 100 * float(np.mean([p == r["gold"] for p, r in zip(ans, dr)]))
    return d


def _pct(num: float, den: float) -> float:
    return 100 * num / den if den > 0 else np.nan


def stats(d: dict, idx: np.ndarray) -> dict[str, float]:
    out = {}
    cD = d["cD"][idx]
    for a in ARMS:
        nc, nt, has = d[f"{a}_nc"][idx], d[f"{a}_nt"][idx], d[f"{a}_has"][idx]
        out[f"claim_true_{a}"] = _pct(nt.sum(), nc.sum())
        out[f"claim_true_dc_{a}"] = _pct(nt[cD].sum(), nc[cD].sum())
        out[f"claim_true_dw_{a}"] = _pct(nt[~cD].sum(), nc[~cD].sum())
        out[f"any_false_{a}"] = 100 * d[f"{a}_anyF"][idx].mean()
        out[f"any_false_claimed_{a}"] = _pct(d[f"{a}_anyF"][idx][has].sum(), has.sum())
        out[f"consistency_{a}"] = _pct(d[f"{a}_cons"][idx][has].sum(), has.sum())
        out[f"contra_{a}"] = 100 * d[f"{a}_contra"][idx].mean()
        out[f"rat_any_{a}"] = _pct(d[f"{a}_ratA"][idx][~cD].sum(), (~cD).sum())
        out[f"rat_last_{a}"] = _pct(d[f"{a}_ratL"][idx][~cD].sum(), (~cD).sum())
        out[f"dissent_{a}"] = _pct((nt[~cD] > 0).sum(), (~cD).sum())
        out[f"tok_{a}"] = d[f"{a}_tok"][idx].mean()
        out[f"body_tok_{a}"] = d[f"{a}_btok"][idx].mean()
        out[f"claimed_{a}"] = 100 * has.mean()
        out[f"claims_per_item_{a}"] = nc.mean()
        if d["has_vert"]:
            vi = d["vinc"][idx]
            out[f"claim_true_vinc_{a}"] = _pct(nt[vi].sum(), nc[vi].sum())
            out[f"claim_true_vcon_{a}"] = _pct(nt[~vi].sum(), nc[~vi].sum())
            out[f"vgap_{a}"] = out[f"claim_true_vcon_{a}"] - out[f"claim_true_vinc_{a}"]
        out[f"claim_true_xc_{a}"] = _pct(d[f"{a}_ntX"][idx].sum(), d[f"{a}_ncX"][idx].sum())
        out[f"rat_any_xc_{a}"] = _pct(d[f"{a}_ratAX"][idx][~cD].sum(), (~cD).sum())
    for key, ref, at_d in MATCHED:
        pop = ~cD & (d[f"{ref}_endD"][idx] == at_d)
        out[f"n_{key}"] = float(pop.sum())
        for a in (ref, "afe"):
            out[f"rat_{key}_{a}"] = _pct(d[f"{a}_ratA"][idx][pop].sum(), pop.sum())
            out[f"ratxc_{key}_{a}"] = _pct(d[f"{a}_ratAX"][idx][pop].sum(), pop.sum())
    for k, ref in MDIFFS:
        out[f"d_{k}_afe-{ref}"] = out[f"{k}_afe"] - out[f"{k}_{ref}"]
    out["acc_direct"] = 100 * cD.mean()
    out["post_acc_afe"] = 100 * d["post_corr"][idx].mean()
    out["d_post_acc_afe-direct"] = out["post_acc_afe"] - out["acc_direct"]
    for x, y in DIFFS:
        for k in MAIN:
            if f"{k}_{x}" in out:
                out[f"d_{k}_{x}-{y}"] = out[f"{k}_{x}"] - out[f"{k}_{y}"]
    out["tok_ratio_afe/cot"] = d["afe_tok"][idx].sum() / d["cot_tok"][idx].sum()
    out["tok_ratio_afe/gate"] = d["afe_tok"][idx].sum() / d["gate_tok"][idx].sum()
    out["tok_ratio_gate/cot"] = d["gate_tok"][idx].sum() / d["cot_tok"][idx].sum()
    return out


def boot_draws(d: dict, nb: int, seed: int = 0) -> tuple[dict, dict]:
    rng = np.random.default_rng(seed)
    members = [np.where(d["cl"] == c)[0] for c in range(d["ncl"])]
    full = stats(d, np.arange(len(d["keys"])))
    bs = {k: [] for k in full}
    for _ in range(nb):
        idx = np.concatenate([members[c] for c in rng.integers(0, d["ncl"], d["ncl"])])
        for k, v in stats(d, idx).items():
            bs[k].append(v)
    return full, {k: np.array(v, float) for k, v in bs.items()}


def summarize(full: dict, bs: dict) -> dict:
    return {k: {"est": round(float(v), 3), "ci": [round(x, 3) for x in q95(bs[k])]} for k, v in full.items()}


def fmt(x: dict, nd: int = 1, bold: bool = False) -> str:
    s = f"{x['est']:+.{nd}f} [{x['ci'][0]:+.{nd}f}, {x['ci'][1]:+.{nd}f}]"
    sig = x["ci"][0] > 0 or x["ci"][1] < 0
    return s if not bold else (f"\\textbf{{{s}}}" if sig else s)


def tex(s: str) -> str:
    return s.replace("-", "$-$")


def matched_lines(b: dict) -> list[str]:
    """Exp-26 text lines: answer-matched / claim-only rationalization and the post-explanation readout."""
    L = []
    for key, ref, _ in MATCHED:
        L.append(f"  {key:6s} n {b[f'n_{key}']['est']:.1f}  rat {ref} {b[f'rat_{key}_{ref}']['est']:.2f} afe {b[f'rat_{key}_afe']['est']:.2f}"
                 + (f"  afe-{ref} {fmt(b[f'd_rat_{key}_afe-{ref}'], 2)}" if f"d_rat_{key}_afe-{ref}" in b else "")
                 + f"  | excl. concl. {ref} {b[f'ratxc_{key}_{ref}']['est']:.2f} afe {b[f'ratxc_{key}_afe']['est']:.2f}"
                 + (f"  afe-{ref} {fmt(b[f'd_ratxc_{key}_afe-{ref}'], 2)}" if f"d_ratxc_{key}_afe-{ref}" in b else ""))
    L.append(f"  post-explanation readout acc: Direct {b['acc_direct']['est']:.2f} -> AFE post {b['post_acc_afe']['est']:.2f}  "
             f"{fmt(b['d_post_acc_afe-direct'], 2)}")
    return L


def latex(res: dict) -> str:
    SL = {"cues": "Cues", "twins": "Twins", "syn": "Syn.\\ Depth"}
    L = [r"\begin{table*}[t]", r"\centering", r"\fontsize{6.2}{7.4}\selectfont", r"\setlength{\tabcolsep}{2.6pt}",
         r"\caption{\textbf{Answer first, then explain (AFE).} The assistant turn starts with the model's own Direct answer "
         r"(\texttt{Answer: (X)}) followed by \texttt{Explanation: Let's think step by step.}; the explanation is decoded greedily "
         r"($\le512$ tokens) in the same single user turn as the CoT arm, so AFE accuracy equals Direct accuracy (the re-run Direct of \cref{tab:supp_gate}; see its caption). CoT and Gate "
         r"are the plain-CoT and commit-sentence-gate rationales of \cref{tab:supp_gate} on the same items. Claims: every "
         r"rationale sentence is parsed with the rule-based depth-order parser (v3); a claim is true if the order it states "
         r"matches the image. True claims (\%) pooled over all parsed claims of the cell; D$+$ / D$-$: items Direct answers "
         r"correctly / wrongly. Front higher: Cues / Twins items whose front object is the higher one in the image, where the "
         r"lower-is-closer prior points the wrong way. Differences in points with $95\%$ paired cluster-bootstrap CIs "
         r"(twin pairs / images; $2{,}000$ resamples); bold: CI excludes 0. Pooled: unweighted mean of the $12$ cells "
         r"(front higher: the $8$ Cues / Twins cells). Panels (c)--(d), in the continuation: answer-matched and claim-only "
         r"rationalization, and the answer re-read after the explanation.}",
         r"\label{tab:supp_afe}"]
    L += [r"\begin{tabular}{@{}llccccccc@{}}", r"\toprule",
          r"\multicolumn{9}{@{}l}{\textbf{(a) Truth of parsed depth-order claims}} \\",
          r"Model & Set & \shortstack{Claims/item\\CoT / Gate / AFE} & \shortstack{True (\%)\\CoT / Gate / AFE} & AFE$-$CoT & AFE$-$Gate & "
          r"\shortstack{True on D$+$ (\%)\\CoT / Gate / AFE} & \shortstack{True on D$-$ (\%)\\CoT / Gate / AFE} & "
          r"\shortstack{True, front higher (\%)\\CoT / Gate / AFE} \\", r"\midrule"]
    rows = [(m, s) for m in MODELS for s in SETS if res.get(m, {}).get(s)]
    for j, (m, s) in enumerate(rows):
        b = res[m][s]["boot"]
        tri = lambda k, nd=1: " / ".join(f"{b[f'{k}_{a}']['est']:.{nd}f}" for a in ARMS)
        vin = tri("claim_true_vinc") if "claim_true_vinc_afe" in b else "--"
        L.append(f"{LABEL[m] if s == 'cues' else ''} & {SL[s]} & {tri('claims_per_item', 2)} & {tri('claim_true')} & "
                 f"{tex(fmt(b['d_claim_true_afe-cot'], bold=True))} & {tex(fmt(b['d_claim_true_afe-gate'], bold=True))} & "
                 f"{tri('claim_true_dc')} & {tri('claim_true_dw')} & {vin} \\\\")
        if s == "syn" and j < len(rows) - 1:
            L.append(r"\addlinespace[1pt]")
    if "pooled" in res:
        b = res["pooled"]
        tri = lambda k, nd=1: " / ".join(f"{b[f'{k}_{a}']['est']:.{nd}f}" for a in ARMS)
        vin = tri("claim_true_vinc") if "claim_true_vinc_afe" in b else "--"
        L += [r"\midrule", f"Pooled & 12 cells & {tri('claims_per_item', 2)} & {tri('claim_true')} & "
              f"{tex(fmt(b['d_claim_true_afe-cot'], bold=True))} & {tex(fmt(b['d_claim_true_afe-gate'], bold=True))} & "
              f"{tri('claim_true_dc')} & {tri('claim_true_dw')} & {vin} \\\\"]
    L += [r"\bottomrule", r"\end{tabular}", r"\par\vspace{4pt}",
          r"\begin{tabular}{@{}llccccccc@{}}", r"\toprule",
          r"\multicolumn{9}{@{}l}{\textbf{(b) False claims, consistency, rationalization of Direct errors, cost}} \\",
          r"Model & Set & \shortstack{$\ge1$ false claim (\%)\\CoT / Gate / AFE} & AFE$-$CoT & \shortstack{Consist. (\%)\\CoT / Gate / AFE} & "
          r"\shortstack{Contradicts ans. (\%)\\CoT / Gate / AFE} & \shortstack{Rationalizes D$-$ (\%)\\CoT / Gate / AFE} & AFE$-$CoT & "
          r"\shortstack{Tokens\\AFE/CoT / AFE/Gate} \\", r"\midrule"]
    for j, (m, s) in enumerate(rows):
        b = res[m][s]["boot"]
        tri = lambda k: " / ".join(f"{b[f'{k}_{a}']['est']:.1f}" for a in ARMS)
        L.append(f"{LABEL[m] if s == 'cues' else ''} & {SL[s]} & {tri('any_false')} & {tex(fmt(b['d_any_false_afe-cot'], bold=True))} & "
                 f"{tri('consistency')} & {tri('contra')} & {tri('rat_any')} & {tex(fmt(b['d_rat_any_afe-cot'], bold=True))} & "
                 f"{b['tok_ratio_afe/cot']['est']:.2f} / {b['tok_ratio_afe/gate']['est']:.2f} \\\\")
        if s == "syn" and j < len(rows) - 1:
            L.append(r"\addlinespace[1pt]")
    if "pooled" in res:
        b = res["pooled"]
        tri = lambda k: " / ".join(f"{b[f'{k}_{a}']['est']:.1f}" for a in ARMS)
        L += [r"\midrule", f"Pooled & 12 cells & {tri('any_false')} & {tex(fmt(b['d_any_false_afe-cot'], bold=True))} & "
              f"{tri('consistency')} & {tri('contra')} & {tri('rat_any')} & {tex(fmt(b['d_rat_any_afe-cot'], bold=True))} & "
              f"{b['tok_ratio_afe/cot']['est']:.2f} / {b['tok_ratio_afe/gate']['est']:.2f} \\\\"]
    L += [r"\bottomrule", r"\end{tabular}",
          r"\par\vspace{2pt}\raggedright\footnotesize $\ge1$ false claim: share of all items whose rationale contains at least one "
          r"false parsed claim. Consist.: the answer (CoT, Gate: same-turn readout; AFE: the stated Direct answer) equals the answer "
          r"implied by the last parsed claim (items with a claim). Contradicts ans.: at least one parsed claim implies the other "
          r"option. Rationalizes D$-$: on items Direct answers wrongly, the rationale contains at least one claim that implies the "
          r"wrong Direct answer (always a false claim). Tokens: generated tokens (Gate: including discarded samples) relative to "
          r"plain CoT and to the Gate; Pooled: mean of the cell ratios."]
    L += [r"\end{table*}"]
    L += latex_matched(res, rows)
    return "\n".join(L) + "\n"


def latex_matched(res: dict, rows: list) -> list[str]:
    """Panels (c)-(d): answer-matched and claim-only rationalization; claim truth without conclusion sentences and the
    re-read answer after the explanation."""
    SL = {"cues": "Cues", "twins": "Twins", "syn": "Syn.\\ Depth"}
    L = [r"\begin{table*}[t]", r"\centering", r"\fontsize{6.2}{7.4}\selectfont", r"\setlength{\tabcolsep}{2.2pt}",
         r"{\normalsize\textbf{Table~\ref{tab:supp_afe} (continued).}}\par\vspace{3pt}",
         r"\begin{tabular}{@{}llcccccc@{}}", r"\toprule",
         r"\multicolumn{8}{@{}l}{\textbf{(c) Rationalization of Direct errors on answer-matched items and without conclusion sentences}} \\",
         r"Model & Set & \shortstack{Rat., CoT's D$-$C$-$ (\%)\\CoT / AFE} & AFE$-$CoT & "
         r"\shortstack{Rat., Gate's D$-$G$-$ (\%)\\Gate / AFE} & AFE$-$Gate & \shortstack{Rat., all D$-$, no concl.\ (\%)\\CoT / Gate / AFE} & AFE$-$CoT \\",
         r"\midrule"]

    def row(b: dict, name: str, sl: str, nd_n: int = 0) -> str:
        return (f"{name} & {sl} & "
                f"{b['rat_mcot_cot']['est']:.1f} / {b['rat_mcot_afe']['est']:.1f} & {tex(fmt(b['d_rat_mcot_afe-cot'], bold=True))} & "
                f"{b['rat_mgate_gate']['est']:.1f} / {b['rat_mgate_afe']['est']:.1f} & {tex(fmt(b['d_rat_mgate_afe-gate'], bold=True))} & "
                + " / ".join(f"{b[f'rat_any_xc_{a}']['est']:.1f}" for a in ARMS) + f" & {tex(fmt(b['d_rat_any_xc_afe-cot'], bold=True))} \\\\")

    for j, (m, s) in enumerate(rows):
        L.append(row(res[m][s]["boot"], LABEL[m] if s == "cues" else "", SL[s]))
        if s == "syn" and j < len(rows) - 1:
            L.append(r"\addlinespace[1pt]")
    if "pooled" in res:
        L += [r"\midrule", row(res["pooled"], "Pooled", "12 cells", 1)]
    L += [r"\bottomrule", r"\end{tabular}", r"\par\vspace{4pt}"]
    # (d) model x set grid
    ms = [m for m in MODELS if m in res]
    L += [r"\begin{tabular}{@{}lcccccc@{}}", r"\toprule",
          r"\multicolumn{7}{@{}l}{\textbf{(d) Claim truth without conclusion sentences; re-read answer after the AFE explanation}} \\",
          r" & \multicolumn{3}{c}{True claims, no concl.\ (\%), CoT / Gate / AFE} & \multicolumn{3}{c}{AFE re-read acc.\ (\%) ($\Delta$ vs.\ Direct [95\% CI])} \\",
          r"\cmidrule(lr){2-4}\cmidrule(lr){5-7}",
          "Model & " + " & ".join(SL[s] for s in SETS) + " & " + " & ".join(SL[s] for s in SETS) + r" \\", r"\midrule"]
    for m in ms:
        bs = [res[m][s]["boot"] for s in SETS]
        L.append(f"{LABEL[m]} & " + " & ".join(" / ".join(f"{b[f'claim_true_xc_{a}']['est']:.1f}" for a in ARMS) for b in bs) + " & "
                 + " & ".join(f"{b['post_acc_afe']['est']:.1f} ({tex(fmt(b['d_post_acc_afe-direct'], bold=True))})" for b in bs) + r" \\")
    L += [r"\bottomrule", r"\end{tabular}"]
    nn = (f"{res['pooled']['n_mcot']['est']:.0f} / {res['pooled']['n_mgate']['est']:.0f}" if "pooled" in res else "--")
    if "pooled" in res:
        b = res["pooled"]
        pooled = (f"Pooled over the 12 cells: true claims without conclusion sentences " + " / ".join(f"{b[f'claim_true_xc_{a}']['est']:.1f}" for a in ARMS)
                  + f" (AFE$-$CoT {tex(fmt(b['d_claim_true_xc_afe-cot']))}, AFE$-$Gate {tex(fmt(b['d_claim_true_xc_afe-gate']))}); "
                  f"re-read {b['acc_direct']['est']:.1f}$\\to${b['post_acc_afe']['est']:.1f} ({tex(fmt(b['d_post_acc_afe-direct'], nd=2))}). ")
    else:
        pooled = ""
    L += [r"\par\vspace{2pt}\raggedright\footnotesize Rat.: the rationale contains at least one parsed claim implying the wrong "
          r"\texttt{Direct} answer $d$, on answer-matched Direct-wrong items: \texttt{CoT}'s D$-$C$-$ items (\texttt{CoT} also ends at $d$) "
          r"or the Gate's D$-$G$-$ items (the Gate ends at $d$), comparing that arm's rationale with the AFE explanation of the same items "
          r"(AFE always ends at $d$; mean items per cell " + nn + "). No concl.: sentences matching the conclusion pattern (\emph{answer is}, \emph{answer:}, "
          r"\emph{therefore}, \emph{thus}, \emph{hence}, \emph{so the}, \emph{in conclusion}, \emph{the correct option/answer}, or an option letter) "
          r"are removed before parsing, so restatements of the answer do not count; Rat.\ D$-$ is then over all Direct-wrong items as in (b). "
          r"Re-read acc.: forced readout at \texttt{Answer: (} after the AFE explanation, with the stated answer $d$ in context; $\Delta$ vs.\ \texttt{Direct}, paired cluster bootstrap. "
          + pooled + r"All claim statistics: rule-based parser only.", r"\end{table*}"]
    return L


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--nboot", type=int, default=2000)
    ap.add_argument("--suffix", default="")
    ap.add_argument("--out", default=str(B))
    ap.add_argument("--exclude-quoted", action="store_true",
                    help="robustness: skip sentences with quotation marks or 'statement' (restated questions)")
    ap.add_argument("--from-json", action="store_true", help="only rewrite supp_afe<tag>.tex from an existing e25_afe<tag>.json")
    args = ap.parse_args()
    global SFX, EXCL_QUOTED
    SFX = args.suffix
    EXCL_QUOTED = args.exclude_quoted
    out = Path(args.out)
    if args.from_json:
        tag = (SFX or "") + ("_strict" if EXCL_QUOTED else "")
        (out / f"supp_afe{tag}.tex").write_text(latex(json.load(open(out / f"e25_afe{tag}.json"))))
        return
    res: dict = {}
    lines: list[str] = []
    pool_full: dict[str, list] = {}
    pool_bs: dict[str, list] = {}
    for m in MODELS:
        for s in SETS:
            d = load(m, s)
            if d is None:
                lines.append(f"MISSING {m} {s}")
                continue
            full, bs = boot_draws(d, args.nboot)
            for k, v in full.items():
                pool_full.setdefault(k, []).append(v)
                pool_bs.setdefault(k, []).append(bs[k])
            b = summarize(full, bs)
            r = {"n": len(d["keys"]), "n_clusters": d["ncl"], "n_direct_wrong": int((~d["cD"]).sum()),
                 "acc": {a: round(d[f"{a}_acc"], 2) for a in ARMS} | {"direct": round(100 * float(d["cD"].mean()), 2)},
                 "afe_answer_is_direct": d["afe_is_direct"], "afe_post_readout_agree_direct_pct": round(d["post_agree"], 1),
                 "afe_post_readout_acc": round(d["post_acc"], 1), "afe_eos_pct": round(d["eos_pct"], 1), "boot": b}
            res.setdefault(m, {})[s] = r
            lines.append(f"== {m} {s} n={r['n']} clusters={r['n_clusters']} Direct-wrong={r['n_direct_wrong']}")
            lines.append(f"  acc {r['acc']}  AFE answer == Direct: {d['afe_is_direct']}  post-explanation readout = Direct "
                         f"{r['afe_post_readout_agree_direct_pct']}% (acc {r['afe_post_readout_acc']})  AFE eos {r['afe_eos_pct']}%")
            for k in MAIN:
                if f"{k}_afe" not in b:
                    continue
                lines.append(f"  {k:18s} " + "  ".join(f"{a} {b[f'{k}_{a}']['est']:.2f}" for a in ARMS)
                             + "  | " + "  ".join(f"{x}-{y} {fmt(b[f'd_{k}_{x}-{y}'], 2)}" for x, y in DIFFS))
            lines.append("  tokens ratio " + "  ".join(f"{k[10:]} {fmt(b[k], 2)}" for k in b if k.startswith("tok_ratio")))
            lines += matched_lines(b)
    if pool_full:
        # unweighted mean over the cells that define the statistic (vertical-congruence keys: cues + twins only)
        pf = {k: float(np.nanmean(v)) for k, v in pool_full.items()}
        pb = {k: np.nanmean(np.vstack(pool_bs[k]), axis=0) for k in pool_full}
        res["pooled"] = summarize(pf, pb)
        res["pooled_n_cells"] = {k: len(v) for k, v in pool_full.items()}
        b = res["pooled"]
        lines.append(f"== POOLED (unweighted mean over cells; {len(pool_full['claim_true_afe'])} cells, "
                     f"vertical keys {len(pool_full.get('vgap_afe', []))} cue/twin cells)")
        for k in MAIN:
            if f"{k}_afe" not in b:
                continue
            lines.append(f"  {k:18s} " + "  ".join(f"{a} {b[f'{k}_{a}']['est']:.2f}" for a in ARMS)
                         + "  | " + "  ".join(f"{x}-{y} {fmt(b[f'd_{k}_{x}-{y}'], 2)}" for x, y in DIFFS))
        lines.append("  tokens ratio " + "  ".join(f"{k[10:]} {fmt(b[k], 2)}" for k in b if k.startswith("tok_ratio")))
        lines += matched_lines(b)
    out.mkdir(parents=True, exist_ok=True)
    tag = (SFX or "") + ("_strict" if EXCL_QUOTED else "")
    json.dump(res, open(out / f"e25_afe{tag}.json", "w"), indent=1)
    (out / f"rigor_e25{tag}.txt").write_text("\n".join(lines) + "\n")
    (out / f"supp_afe{tag}.tex").write_text(latex(res))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
