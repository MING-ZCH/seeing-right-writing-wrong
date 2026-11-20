"""Build main Tab. 2 and the Round-15b (reframe) supplement tables from bundle files.

Outputs (all under paper/tables/):
    tab2_model_profile.tex  main-paper Tab. 2 (per-model profile grouped by the three findings)
    supp_heuristics.tex     heuristic baselines per set and per-model rule agreement
    supp_syn431.tex         Synthetic Depth: all 500 vs. 431 visibly overlapping items

Every number is read from vtax_exps/outputs/bundle (e27_reframe.json,
common_scale.json, e6_eval.json, heldout_split.json, law1.json, law2.json,
commit_nulls.json, imgswap.json, relation_sensitivity.json); nothing is
typed by hand. The Tab. 2 Direct twin image effect is derived as 2 * twin Direct accuracy - 100.
"""
import vtax_paths as _vtp  # noqa: E402
import json
import os
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BUNDLE = Path(_vtp.OUTPUTS) / "bundle"
TABLES = Path(os.environ.get("TABLES_OUT", ROOT / f"{_vtp.PAPER}" / "tables"))

MODELS = ["qwen3b", "qwen7b", "internvl3_8b", "phi35v", "molmo7b", "qwen3vl_8b", "internvl35_8b"]
HEAD = {
    "qwen3b": r"\shortstack{Qwen2.5-VL\\3B}",
    "qwen7b": r"\shortstack{Qwen2.5-VL\\7B}",
    "internvl3_8b": r"\shortstack{InternVL3\\8B}",
    "phi35v": r"\shortstack{Phi-3.5\\Vision}",
    "molmo7b": r"\shortstack{Molmo\\7B-D}",
    "qwen3vl_8b": r"\shortstack{Qwen3-VL\\8B}",
    "internvl35_8b": r"\shortstack{InternVL3.5\\8B}",
}
NAME = {
    "qwen3b": "Qwen2.5-VL-3B", "qwen7b": "Qwen2.5-VL-7B", "internvl3_8b": "InternVL3-8B",
    "phi35v": "Phi-3.5-V", "molmo7b": "Molmo-7B-D", "qwen3vl_8b": "Qwen3-VL-8B",
    "internvl35_8b": "InternVL3.5-8B",
}


def s(x: float, nd: int = 2) -> str:
    """Signed number in math mode."""
    v = round(float(x), nd)
    if v == 0:
        v = 0.0
    return f"${v:+.{nd}f}$"


def u(x: float, nd: int = 1) -> str:
    """Unsigned number in math mode."""
    return f"${float(x):.{nd}f}$"


def ci(lo: float, hi: float, nd: int = 2, signed: bool = True) -> str:
    """Compact CI in tiny font."""
    f = (lambda v: f"{float(v):+.{nd}f}") if signed else (lambda v: f"{float(v):.{nd}f}")
    txt = f"[{f(lo)},{f(hi)}]".replace("-", "$-$")
    return r"{\tiny" + txt + "}"


def star(lo: float, hi: float) -> str:
    return r"$^\ast$" if (lo > 0 or hi < 0) else ""


def parse_str_ci(txt: str):
    """'-0.1 [-4.3, 3.9]' -> (-0.1, -4.3, 3.9)."""
    a, b = txt.split("[")
    lo, hi = b.rstrip("]").split(",")
    return float(a), float(lo), float(hi)


def row(label: str, cells) -> str:
    return label + " & " + " & ".join(cells) + r" \\"


def hu(x: float, nd: int = 1, signed: bool = True) -> str:
    """Decimal half-up rounding of an exact value (57.25 -> 57.3, -2.25 -> -2.3), as in supp_common_scale."""
    v = Decimal(repr(float(x))).quantize(Decimal(1).scaleb(-nd), rounding=ROUND_HALF_UP)
    return f"${v:+.{nd}f}$" if signed else f"${v:.{nd}f}$"


def main_profile(e27, cs, e6, law1, law2, cnull, imgswap, relsens) -> str:
    """Main Tab. 2, grouped by the three findings (Exp-33 / Exp-41 layout)."""
    comp = e27["comp"]["models"]
    occ = e27["heur"]["occ_model"]
    L1, L2 = law1["models"], law2["models"]
    red = r"\cellcolor{tabred}"
    head = lambda t: r"\rowcolor{tabgray}\multicolumn{8}{@{}l@{}}{\textit{\textbf{" + t + r"}}} \\"
    lines = [
        r"\begin{tabular}{@{}l" + "c" * len(MODELS) + "@{}}",
        r"\toprule",
        row(r"\textbf{Model}", [r"\textbf{" + HEAD[m] + "}" for m in MODELS]),
        r"\midrule",
        head(r"What \texttt{Direct} relies on: cue-factorial set (rule: accuracy CI low $>70\%$ and occlusion weight CI low $>0$)"),
    ]
    lines.append(row("Accuracy (\\%)", [
        u(comp[m]["cue_direct_acc"]["mean"]) + ci(*comp[m]["cue_direct_acc"]["ci"], nd=1, signed=False) for m in MODELS]))
    lines.append(row("Follows lower-is-closer (\\%)", [u(occ[m]["cues"]["direct"]["agree_lower"]["mean"]) for m in MODELS]))
    lines.append(row("Reads occlusion", [r"\textbf{yes}" if comp[m]["competent"] else "no" for m in MODELS]))
    lines += [r"\midrule", head("1. The written relation decides the answer")]
    lines.append(row(red + "Commit at a relational sentence (\\%)", [
        red + u(L1[m]["commit"]["relational_share_of_commits"]) for m in MODELS]))
    lines.append(row(r"\quad excess over position-matched null", [s(cnull[m]["position_matched"]["excess"], 1) for m in MODELS]))
    lines.append(row(r"Answer agrees with written relation: true / false (\%)", [
        u(L1[m]["cues"]["p_agree_true"]) + " / " + u(L1[m]["cues"]["p_agree_false"]) for m in MODELS]))
    cells = []
    for m in MODELS:
        e = L1[m].get("edit", {}).get("all", {}).get("correct_minus_direct")
        cells.append(s(e["mean"], 1) + star(e["lo"], e["hi"]) if e else "--")
    lines.append(row(r"True first claim written $-$ \texttt{Direct} (pts)", cells))
    cells = []
    for m in MODELS:
        # Direct twin image effect P(ans=gold_v|v) - P(ans=gold_v|v_cf) = 2 acc - 100 (every twin query has its pair)
        twin_direct = 2 * relsens[m]["occ"]["direct_acc"]["mean"] - 100
        d = imgswap[m]["cut:all"]["dvis"]
        cells.append(hu(twin_direct, signed=False) + " / " + s(d["mean"], 1) + star(d["lo"], d["hi"]))
    lines.append(row(r"Twin image effect: \texttt{Direct} / own $R_\mathrm{cut}$ (pts)", cells))
    lines.append(row("Answer step", [L1[m]["type"] for m in MODELS]))
    lines += [r"\midrule", head("2. The written relation is often wrong (commit sentences contradicting the image, two-judge consensus)")]
    lines.append(row(red + r"\texttt{Syn.\ Depth} 431: commit vs.\ matched", [
        red + f"${L2[m]['syn431_commit']['k']}/{L2[m]['syn431_commit']['n']}$ vs.\\ "
        f"${L2[m]['syn431_comparator']['k']}/{L2[m]['syn431_comparator']['n']}$" for m in MODELS]))
    lines.append(row(r"Cue set: first claim names true front (\%)", [u(L2[m]["first_claim_true_front"]["pct"]) for m in MODELS]))
    lines += [r"\midrule", head(r"3. Net effect (\texttt{CoT}$-$\texttt{Direct})")]
    lines.append(row("Cue set: height effect (AME, pts)", [
        s(L2[m]["ame_height_cot_minus_direct"]["est"], 1) + star(*L2[m]["ame_height_cot_minus_direct"]["ci"]) for m in MODELS]))
    lines.append(row("Cue set: accuracy (pts)", [
        hu(cs[m]["acc_cot_minus_direct"]["est"]) + star(*cs[m]["acc_cot_minus_direct"]["ci"]) for m in MODELS]))
    lines.append(row(r"\texttt{Syn.\ Depth} 431: accuracy (pts)", [
        s(e27["ovl"]["models"][m]["single_order"]["overlap"]["acc_diff"], 1) + star(*e27["ovl"]["models"][m]["single_order"]["overlap"]["acc_diff_ci"])
        for m in MODELS]))
    cells = []
    for m in MODELS:
        v, lo, hi = parse_str_ci(e6[m]["pooled"]["cot_minus_direct"])
        cells.append(red + s(v, 1) + ci(lo, hi, nd=1))
    lines.append(row(red + "Held-out $767$: accuracy (pts)", cells))
    hs = json.loads((BUNDLE / "heldout_split.json").read_text())
    cells = []
    for m in MODELS:
        x = hs[m]["depth324"]["cot_minus_direct"]
        # Phi-3.5-Vision writes empty rationales on BLINK, so its depth cell is a format effect.
        mark = r"$^\ddagger$" if m == "phi35v" else ""
        cells.append(s(x["mean"], 1) + mark + ci(x["lo"], x["hi"], nd=1))
    lines.append(row(r"\quad depth items ($324$)", cells))
    lines.append(row(r"\quad \texttt{BLINK} Rel.\ Depth $\Delta d'$", [
        s(L2[m]["blink_ddprime"]["est"]) + star(*L2[m]["blink_ddprime"]["ci"]) for m in MODELS]))
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines) + "\n"


def supp_heuristics(e27) -> str:
    h = e27["heur"]
    sd = h["syn_depth"]
    lines = [
        r"\begin{tabular}{lcccc}",
        r"\toprule",
        r"Rule (answer without reading occlusion) & \texttt{Syn.\ Depth} 500 / 431 & Twins (400 img.) & Cue set (1{,}600) & Hard set (288) \\",
        r"\midrule",
    ]
    lines.append(row("Larger drawn area is in front", ["--", u(h["twins"]["size_drawn"]), u(h["cues"]["size_drawn"]), u(h["hard"]["size_drawn"])]))
    lines.append(row("Larger visible area is in front", [
        u(sd["all"]["size_visible"]) + " / " + u(sd["overlap431"]["size_visible"]),
        u(h["twins"]["size_visible"]), u(h["cues"]["size_visible"]), u(h["hard"]["size_visible"])]))
    lines.append(row("Lower in the image is closer", [
        u(sd["all"]["lower"]) + " / " + u(sd["overlap431"]["lower"]),
        u(h["twins"]["lower"]), u(h["cues"]["lower"]), u(h["hard"]["lower"])]))
    lines.append(row("Logistic on visible ratio + height (5-fold)", [
        u(sd["all"]["logistic_cv_acc"]) + " / " + u(sd["overlap431"]["logistic_cv_acc"]),
        u(h["twins"]["visible_and_lower"]["logistic_cv_acc"]),
        u(h["cues"]["visible_and_lower"]["logistic_cv_acc"]),
        u(h["hard"]["visible_and_lower"]["logistic_cv_acc"])]))
    lines += [r"\midrule", r"\multicolumn{5}{l}{\textit{Per model, \texttt{Direct} (two object orders on \texttt{Syn.\ Depth}; \%)}} \\",
              row("Model", ["\\texttt{Syn.\\ Depth} acc.\\ / agrees with size rule", "Twins acc.\\ / follows lower", "Cue acc.\\ / follows lower (\\texttt{CoT})", "Competent"]),
              r"\midrule"]
    for m in MODELS:
        sm = h["syn_model"][m]["direct"]
        oc = h["occ_model"][m]
        comp = e27["comp"]["models"][m]
        lines.append(row(NAME[m], [
            u(sm["acc_two_order"]["mean"]) + " / " + u(sm["agree_size_rule"]["mean"]),
            u(oc["twins"]["direct"]["acc"]["mean"]) + " / " + u(oc["twins"]["direct"]["agree_lower"]["mean"]),
            u(oc["cues"]["direct"]["acc"]["mean"]) + " / " + u(oc["cues"]["direct"]["agree_lower"]["mean"])
            + " (" + u(oc["cues"]["cot"]["agree_lower"]["mean"]) + ")",
            "yes" if comp["competent"] else "no"]))
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines) + "\n"


def supp_syn431(e27) -> str:
    ovl = e27["ovl"]["models"]
    ca = e27["ovl"]["commit_audit"]
    lines = [
        r"\begin{tabular}{lcccccc}",
        r"\toprule",
        r" & \multicolumn{2}{c}{$\Delta d'_\mathrm{pair}$ [95\% CI]} & \multicolumn{2}{c}{Pooled $\Delta d'$ [95\% CI]} & \multicolumn{2}{c}{Commit contradicts / relational \texttt{D+C$+$}} \\",
        r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(lr){6-7}",
        r"Model & all 500 & overlapping 431 & all 500 & overlapping 431 & all 500 & overlapping 431 \\",
        r"\midrule",
    ]

    def cell(d, key):
        lo, hi = d[key + "_ci"]
        return s(d[key]) + " " + ci(lo, hi)

    for m in MODELS:
        a, o = ovl[m]["all"], ovl[m]["overlap"]
        cm, rc = ca["DCm_commit"][m], ca["DCp_rel"][m]
        lines.append(row(NAME[m], [
            cell(a, "dpair_delta"), cell(o, "dpair_delta"),
            cell(a, "dpooled_delta"), cell(o, "dpooled_delta"),
            f"${cm['all'][0]}/{cm['all'][1]}$ / ${rc['all'][0]}/{rc['all'][1]}$",
            f"${cm['overlap'][0]}/{cm['overlap'][1]}$ / ${rc['overlap'][0]}/{rc['overlap'][1]}$"]))
    cm, rc = ca["DCm_commit"]["pooled"], ca["DCp_rel"]["pooled"]
    lines += [r"\midrule", row("Pooled", ["", "", "", "",
                                          f"${cm['all'][0]}/{cm['all'][1]}$ ({cm['all'][2]:.1f}\\%) / ${rc['all'][0]}/{rc['all'][1]}$ ({rc['all'][2]:.1f}\\%)",
                                          f"${cm['overlap'][0]}/{cm['overlap'][1]}$ ({cm['overlap'][2]:.1f}\\%) / ${rc['overlap'][0]}/{rc['overlap'][1]}$ ({rc['overlap'][2]:.1f}\\%)"])]
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines) + "\n"


def main() -> None:
    e27 = json.loads((BUNDLE / "e27_reframe.json").read_text())
    cs = json.loads((BUNDLE / "common_scale.json").read_text())
    e6 = json.loads((BUNDLE / "e6_eval.json").read_text())
    ld = lambda f: json.loads((BUNDLE / f).read_text())
    out = {
        "tab2_model_profile.tex": main_profile(e27, cs, e6, ld("law1.json"), ld("law2.json"), ld("commit_nulls.json"),
                                               ld("imgswap.json"), ld("relation_sensitivity.json")),
        "supp_heuristics.tex": supp_heuristics(e27),
        "supp_syn431.tex": supp_syn431(e27),
    }
    for name, txt in out.items():
        (TABLES / name).write_text(txt)
        assert txt.count(r"\\") > 5, name
        print(name, len(txt.splitlines()), "lines")


if __name__ == "__main__":
    main()
