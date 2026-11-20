"""Generate main-paper Table 3 (consequences) from the numbers bundle and gate CV files.

CV gate sources: exact E1x readouts (Qwen2.5-VL-3B/7B, InternVL3-8B, Phi) -> outputs/e0x/gate_cv.csv;
Molmo -> legacy V1 top-20 readouts, outputs/e0/gate_cv.csv; Qwen3-VL-8B -> outputs/e0x_qwen3vl/gate_cv.csv;
InternVL3.5-8B -> <W>/outputs/e0x_ivl35/gate_cv.csv (all with the CoT alternative).
"""
from __future__ import annotations
import vtax_paths as _vtp  # noqa: E402

import csv
import json
from pathlib import Path

B = Path("outputs/bundle")
W = Path(f"{_vtp.OUTPUTS}")
OUT = Path(f"{_vtp.TABLES}/tab3_faithfulness_and_api.tex")
MODELS = [("qwen3b", "Qwen2.5-VL-3B"), ("qwen7b", "Qwen2.5-VL-7B"), ("internvl3_8b", "InternVL3-8B"),
          ("phi35v", "Phi-3.5-Vision"), ("molmo7b", "Molmo-7B-D"), ("qwen3vl_8b", "Qwen3-VL-8B"),
          ("internvl35_8b", "InternVL3.5-8B")]
GATE_SRC = {"qwen3b": Path("outputs/e0x/gate_cv.csv"), "qwen7b": Path("outputs/e0x/gate_cv.csv"),
            "internvl3_8b": Path("outputs/e0x/gate_cv.csv"), "phi35v": Path("outputs/e0x/gate_cv.csv"),
            "molmo7b": Path("outputs/e0/gate_cv.csv"), "qwen3vl_8b": Path("outputs/e0x_qwen3vl/gate_cv.csv"),
            "internvl35_8b": W / "e0x_ivl35/gate_cv.csv"}


def ci(d: dict | None, sign: bool = False) -> str:
    if not d or not d.get("n"):
        return "--"
    f = "{:+.1f}" if sign else "{:.1f}"
    return f"${f.format(d['mean'])}\\,[{f.format(d['lo'])},{f.format(d['hi'])}]$"


def acc_delta(d: dict | None, dd: dict | None) -> str:
    if not d or not d.get("n"):
        return "--"
    return f"${d['mean']:.1f}$ (${dd['mean']:+.1f}$)" if dd else f"${d['mean']:.1f}$"


def gate(m: str) -> str:
    p = GATE_SRC[m]
    if not p.exists():
        return "--"
    for r in csv.DictReader(open(p)):
        if r["model"] == m and r["alternative"] == "cot":
            g, lo, hi = (100 * float(r[k]) for k in ("gate_minus_direct", "gate_minus_direct_lo", "gate_minus_direct_hi"))
            return f"${g:+.1f}\\,[{lo:+.1f},{hi:+.1f}]$"
    return "--"


def parse(s: str) -> tuple[float, float, float]:
    m, rest = s.split(" ", 1)
    lo, hi = rest.strip("[]").split(", ")
    return float(m), float(lo), float(hi)


def main() -> None:
    nb = json.load(open(B / "numbers.json"))
    ev = json.load(open(B / "e6_eval.json"))
    rows_a, rows_b = [], []
    A = {m: nb[m]["fix_main_pool"] for m, _ in MODELS}
    H = {m: nb[m]["fix_heldout"] for m, _ in MODELS}
    E = {m: ev.get(m, {}) for m, _ in MODELS}

    def row(label: str, cells: list[str], color: str = "") -> str:
        return f"{color}{label} & " + " & ".join(cells) + r" \\"

    rows_a.append(row(r"\texttt{Direct}", [ci(A[m].get("direct")) for m, _ in MODELS]))
    rows_a.append(row(r"\texttt{Short CoT} ($\Delta$ Dir.)", [acc_delta(A[m].get("cot"), A[m].get("cot_minus_direct")) for m, _ in MODELS]))
    rows_a.append(row(r"\texttt{Sym-Direct} ($\Delta$ Dir.)", [acc_delta(A[m].get("sym_direct"), A[m].get("sym_direct_minus_direct")) for m, _ in MODELS],
                      r"\rowcolor{tabsage}"))
    rows_a.append(row(r"\texttt{Sym-CoT} ($\Delta$ Dir.)", [acc_delta(A[m].get("sym_cot"), A[m].get("sym_cot_minus_direct")) for m, _ in MODELS]))
    rows_a.append(row(r"\texttt{Consist.-else-Dir.} ($\Delta$)", [acc_delta(A[m].get("cot_consistent_else_direct"),
                                                                           A[m].get("cot_consistent_else_direct_minus_direct")) for m, _ in MODELS]))
    rows_a.append(row(r"\texttt{Dir./CoT consist.(\%)}", [
        f"${A[m]['direct_consistent_rate']['mean']:.1f}$ / ${A[m]['cot_consistent_rate']['mean']:.1f}$" if A[m].get("direct_consistent_rate") else "--"
        for m, _ in MODELS]))
    rows_a.append(row(r"\texttt{CV Gate $-$ Direct}", [gate(m) + (r"$^\dagger$" if m == "molmo7b" else "") for m, _ in MODELS]))

    def ev_cell(m: str, task: str, key: str, sign: bool = True) -> str:
        r = E[m].get(task)
        if not r or key not in r:
            return "--"
        a, lo, hi = parse(r[key])
        f = "{:+.1f}" if sign else "{:.1f}"
        return f"${f.format(a)}\\,[{f.format(lo)},{f.format(hi)}]$"

    def dc(m: str, task: str) -> str:
        r = E[m].get(task)
        if not r:
            return "--"
        d, c = parse(r["direct"])[0], parse(r["cot"])[0]
        return f"${d:.1f}/{c:.1f}$ (${parse(r['cot_minus_direct'])[0]:+.1f}$)"

    rows_b.append(row(r"\texttt{Held-out Direct}", [ev_cell(m, "pooled", "direct", sign=False) for m, _ in MODELS]))
    rows_b.append(row(r"\texttt{Held-out CoT} ($\Delta$ Dir.)", [
        f"${parse(E[m]['pooled']['cot'])[0]:.1f}$ (${parse(E[m]['pooled']['cot_minus_direct'])[0]:+.1f}$)" if E[m].get("pooled") else "--"
        for m, _ in MODELS]))
    rows_b.append(row(r"\texttt{BLINK Depth} ($124$, $\Delta$)", [dc(m, "blink_depth") for m, _ in MODELS], r"\rowcolor{tabred}"))
    rows_b.append(row(r"\texttt{BLINK Rel.} ($143$, $\Delta$)", [dc(m, "blink_relation") for m, _ in MODELS]))
    rows_b.append(row(r"\texttt{Sym-Dir./Sym-CoT} ($\Delta$)", [
        f"${H[m]['sym_direct_minus_direct']['mean']:+.1f}$ / ${H[m]['sym_cot_minus_direct']['mean']:+.1f}$" if H[m].get("sym_direct_minus_direct") else "--"
        for m, _ in MODELS]))
    hb = json.load(open(B / "heldout_breakdown.json"))

    def hb_cell(m: str, key: str, fmt: bool = False) -> str:
        x = hb.get(m, {}).get("depth324", {})
        d = x.get("format", {}).get(key) if fmt else x.get(key)
        return f"${d['mean']:+.1f}\\,[{d['lo']:+.1f},{d['hi']:+.1f}]$" if d else "--"

    rows_b.append(row(r"\texttt{Depth $324$}: CoT$-$Dir.", [hb_cell(m, "cot_minus_direct") for m, _ in MODELS],
                      r"\rowcolor{tabred}"))
    rows_b.append(row(r"\texttt{Depth $324$}: CoT$-$F+t", [hb_cell(m, "cot_minus_filler_turn", True) for m, _ in MODELS]))
    extra = [row(r"\texttt{Consist.-else-Dir.} ($\Delta$)", [acc_delta(H[m].get("cot_consistent_else_direct"),
                                                                    H[m].get("cot_consistent_else_direct_minus_direct")) for m, _ in MODELS])]
    rows_b.append(row(r"\texttt{Frozen Gate $-$ Dir.}", [ev_cell(m, "pooled", "gate_cot_minus_direct") for m, _ in MODELS]))
    extra.append(row(r"\texttt{Frozen Gate $-$ Rnd.}", [ev_cell(m, "pooled", "gate_cot_minus_random") for m, _ in MODELS]))
    vote_rows = [f"{n} & {E[m]['pooled']['vote4_predicted']:.1f} & {E[m]['pooled']['vote4_observed']:.1f} & "
                 f"{E[m]['pooled']['vote4_pred_const_pbar']:.1f} \\\\" for m, n in MODELS
                 if E[m].get("pooled", {}).get("vote4_predicted") is not None]
    extra.append(row(r"\texttt{Held-out Desc} ($\Delta$ Dir.)", [
        f"${parse(E[m]['pooled']['desc'])[0]:.1f}$ (${parse(E[m]['pooled']['desc_minus_direct'])[0]:+.1f}$)" if E[m].get("pooled", {}).get("desc") else "--"
        for m, _ in MODELS]))

    head = " & ".join(rf"\textbf{{{n}}}" for _, n in MODELS)
    tex = rf"""\begin{{table*}}[t]
\centering
\caption{{\textbf{{Order-swap symmetrization, consistency routing, and frozen-protocol held-out evaluation.}} Accuracy (\%) and paired $\Delta$ (percentage points vs.\ \texttt{{Direct}}) with $95\%$ image-cluster bootstrap CIs ($2,000$ resamples). \textbf{{Top:}} Main pool ($n=2,850$ binary items; option order swapped on \texttt{{CV-Bench}}, object order swapped on synthetic sets). \textbf{{Bottom:}} Held-out pool ($n=767$ binary items: \texttt{{BLINK}} Spatial Relation $143$, Relative Depth $124$, and five \texttt{{3DSRBench}} COCO categories $\times 100$) under the frozen protocol (predictions and the gate threshold $\tau$ fixed on the main pool and time-stamped before the held-out CoT runs).}}
\label{{tab:consequences}}
\fontsize{{6.5}}{{7.5}}\selectfont
\setlength{{\tabcolsep}}{{1.45pt}}
\begin{{tabular}}{{@{{}}lccccccc@{{}}}}
\toprule
\textbf{{Protocol / Metric}} & {head} \\
\midrule
\rowcolor{{tabgray}}\multicolumn{{8}}{{@{{}}l@{{}}}}{{\textit{{\textbf{{A. Main Pool ($n=2,850$ binary items): Order-Swap Consistency, Symmetrization, and 5-Fold CV Margin Gate}}}}}} \\
""" + "\n".join(rows_a) + r"""
\midrule
\rowcolor{tabgray}\multicolumn{8}{@{}l@{}}{\textit{\textbf{B. Held-Out Pool ($n=767$ binary items: BLINK + 3DSRBench): Frozen-Protocol Generalization}}} \\
""" + "\n".join(rows_b) + r"""
\bottomrule
\end{tabular}
\vspace{1pt}
\footnotesize{\raggedright $^\dagger$\texttt{Molmo-7B-D}: gate CV and frozen $\tau$ use its V1-engine top-$20$ readouts from the original runs (\cref{sec:supp_engine}). V1-engine models (\texttt{Molmo-7B-D}, \texttt{Qwen3-VL-8B}, \texttt{InternVL3.5-8B}) use top-$20$ readouts throughout. Depth $324$: depth-valid held-out items (BLINK Relative Depth + \texttt{3DSRBench} closer-to-camera + in-front-of); F+t: length-matched filler with the turn closed. Equal entries of \texttt{Qwen2.5-VL-3B}/\texttt{Molmo-7B-D} (\texttt{BLINK Depth}) and \texttt{Qwen3-VL-8B}/\texttt{InternVL3.5-8B} (Depth $324$) are coincidences of different items (\cref{tab:supp_heldout_breakdown}). Per-category held-out results are in \cref{tab:supp_heldout}.\par}
\end{table*}
"""
    OUT.write_text(tex)
    # pre-registered 4-vote prediction (moved out of the main table to keep it unscaled at 6.5 pt)
    vote = "\n".join([r"\begin{tabular}{lccc}", r"\toprule",
                      r"Model & Predicted & Observed & Constant-$\bar{p}$ baseline \\", r"\midrule", *vote_rows,
                      r"\bottomrule", r"\end{tabular}"]) + "\n"
    (OUT.parent / "supp_vote.tex").write_text(vote)
    head_s = " & ".join(n for _, n in MODELS)
    (OUT.parent / "supp_heldout_extra.tex").write_text("\n".join(
        [r"\begin{tabular}{@{}lccccccc@{}}", r"\toprule", rf"Held-out ($n=767$) & {head_s} \\", r"\midrule", *extra,
         r"\bottomrule", r"\end{tabular}"]) + "\n")
    print("wrote", OUT)


if __name__ == "__main__":
    main()
