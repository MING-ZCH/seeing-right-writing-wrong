"""Exp-58 (R23): label-free BodyRel readout (Part A, e58_bodyrel.py) and leverage checks (Part B, e58_leverage.py).

Outputs: outputs/bundle/e58_bodyrel.json, e58_leverage.json, rigor_e58.txt; paper tables supp_e58a.tex (A), supp_e58b.tex (B).
Pre-registration: shared outputs/e58/prereg.txt (copy in outputs/e58/prereg.txt).
Usage: python rigor_e58.py [--only a,b]
"""
from __future__ import annotations
import vtax_paths as _vtp  # noqa: E402

import argparse
import json
import re
from pathlib import Path

import numpy as np

B = Path("outputs/bundle")
PAPER = Path(f"{_vtp.PAPER}")
NAMES = {"qwen3b": "Qwen2.5-VL-3B", "qwen7b": "Qwen2.5-VL-7B", "internvl3_8b": "InternVL3-8B", "phi35v": "Phi-3.5-V",
         "molmo7b": "Molmo-7B-D", "qwen3vl_8b": "Qwen3-VL-8B", "internvl35_8b": "InternVL3.5-8B",
         "qwen3vl_32b": "Qwen3-VL-32B", "qwen3vl_8b_think": "Qwen3-VL-8B-Think."}
M7 = ["qwen3b", "qwen7b", "internvl3_8b", "phi35v", "molmo7b", "qwen3vl_8b", "internvl35_8b"]
EXTRA = ["qwen3vl_32b", "qwen3vl_8b_think"]


def jdefault(o):
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, np.bool_):
        return bool(o)
    return str(o)


def neg(s: str) -> str:
    return re.sub(r"(?<![\w.])-(?=\d)", "$-$", s)


def sg(v: float, d: int = 1) -> str:
    return neg(f"{v:+.{d}f}")


def ci(v: list, d: int = 1) -> str:
    return neg(f"[{v[0]:+.{d}f}, {v[1]:+.{d}f}]")


def cip(v: list, d: int = 2) -> str:
    return neg(f"[{v[0]:.{d}f}, {v[1]:.{d}f}]")


# ============================================================================= table A

def table_a(res: dict) -> str:
    S = res["sets"]
    blocks = [("cues", "Cue set (in-sample: the idea was formed here)"), ("twins", "Occlusion twins"),
              ("syn431", "Synthetic Depth 431"), ("cvb3d", "CV-Bench 3D Depth (held-out)"),
              ("blink", "BLINK Relative Depth (held-out; Phi-3.5-V excluded: empty rationales)"),
              ("natural", "Natural depth pooled, CVB 3D + BLINK (held-out, primary)")]
    p1, p2 = res["primary"]["P1_cues_in_sample"], res["primary"]["P2_natural_held_out"]
    L = [r"% Exp-58 Part A (rigor_e58.py --only a). Suggested caption: Label-free BodyRel readout. For each item, if the",
         r"% conclusion-removed rationale body has a written relation (last relation; two-judge consensus, Tab. supp_e45 /",
         r"% supp_e35), BodyRel answers the option that relation implies for the asked question (converse and other-shape",
         r"% questions mapped deterministically); otherwise it falls back to the CoT answer (BR$_C$) or the Direct answer",
         r"% (BR$_D$). Cov.: share of items with a relation. Differences in points with paired cluster-bootstrap 95% CIs",
         r"% (2,000 draws shared by all models; clusters: cue twin pair, twin pair, Syn. Depth image, natural image). Mean:",
         r"% unweighted over the seven instruct models (BLINK: six). Pre-registered primary: BR$_C-$Dir., 7-model mean, cue set",
         f"% (in-sample) {p1['est']:+.1f} [{p1['ci'][0]:+.1f}, {p1['ci'][1]:+.1f}] and natural depth pooled (held-out) "
         f"{p2['est']:+.1f} [{p2['ci'][0]:+.1f}, {p2['ci'][1]:+.1f}].",
         r"% $^\dagger$Not in the mean. Natural pooled with Phi-3.5-V BLINK included (BodyRel = fallback there): see bundle e58_bodyrel.json.",
         r"\begin{tabular}{lrrrrrrrrr}", r"\toprule",
         r"Model & $n$ & Cov. & \texttt{Dir.} & \texttt{CoT} & BR$_C$ & BR$_C-$\texttt{Dir.} & BR$_C-$\texttt{CoT} & BR$_D$ & BR$_D-$\texttt{Dir.} \\",
         r"\midrule"]
    for ds, title in blocks:
        st = S[ds]
        L.append(f"\\multicolumn{{10}}{{l}}{{\\textit{{{title}}}}} \\\\")
        for m in M7 + EXTRA:
            if m not in st:
                continue
            o = st[m]
            mk = "$^\\dagger$" if m in EXTRA else ""
            L.append(f"{NAMES[m]}{mk} & {o['n']} & {o['coverage']:.1f} & {o['acc_direct']:.1f} & {o['acc_cot']:.1f} & {o['acc_brc']:.1f} & "
                     f"{sg(o['brc_dir'])} {ci(o['brc_dir_ci'])} & {sg(o['brc_cot'])} {ci(o['brc_cot_ci'])} & {o['acc_brd']:.1f} & "
                     f"{sg(o['brd_dir'])} {ci(o['brd_dir_ci'])} \\\\")
        mm = st["mean7"]
        k = len(mm["models"])
        L.append(f"\\textbf{{Mean ({k})}} & & {mm['coverage']:.1f} & {mm['acc_direct']:.1f} & {mm['acc_cot']:.1f} & {mm['acc_brc']:.1f} & "
                 f"\\textbf{{{sg(mm['brc_dir'])}}} {ci(mm['brc_dir_ci'])} & {sg(mm['brc_cot'])} {ci(mm['brc_cot_ci'])} & {mm['acc_brd']:.1f} & "
                 f"{sg(mm['brd_dir'])} {ci(mm['brd_dir_ci'])} \\\\")
        L.append(f"\\quad CI $>0$ / $<0$ & & & & & & {mm['brc_dir_n_pos']} / {mm['brc_dir_n_neg']} & {mm['brc_cot_n_pos']} / {mm['brc_cot_n_neg']} & & "
                 f"{mm['brd_dir_n_pos']} / {mm['brd_dir_n_neg']} \\\\")
        if ds != "natural":
            L.append(r"\midrule")
    L += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(L) + "\n"


# ============================================================================= table B

def table_b(res: dict) -> str:
    mp, F, A = res["main_pool"], res["fits"], res["agreement"]
    M8 = ["qwen3b", "qwen7b", "internvl3_8b", "phi35v", "molmo7b", "qwen3vl_8b", "qwen3vl_8b_think", "internvl35_8b"]
    L = [r"% Exp-58 Part B (rigor_e58.py --only b). Suggested caption: Leverage of Synthetic Depth. (a) CoT$-$Direct on",
         r"% the main pool (Tab. 1): full 2,850-item pool (item-weighted, as in Tab. 1), Synthetic Depth alone, and the pool",
         r"% without Synthetic Depth (2,350 items; task-weighted mean of CVB 2D Relation, CVB 3D Depth, CVB 3D Distance,",
         r"% Syn. RelPos, and item-weighted); 95% image-cluster bootstrap CIs (2,000 draws, shared across models for the means).",
         r"% (b) Two-channel fits $y = a + b\,x$ over the open instruct points (7 models x cue set / twins / Syn. Depth 431)",
         r"% with and without the seven Syn. Depth 431 points: Fig. 5 / Tab. supp_two_channel (conclusion kept, rule parser)",
         r"% and Tab. supp_e45 (conclusion removed, two judges). CIs: point bootstrap / model-cluster bootstrap (10,000).",
         r"% (c) Agreement of the CoT answer with the last relation of the conclusion-removed body when that relation is",
         r"% true / false ($a_T$ / $a_F$, %; Exp-45), with the number of true / false relations.",
         r"\begin{tabular}{@{}c@{}}",
         r"\textbf{(a)} Main pool: \texttt{CoT}$-$\texttt{Direct} (points) \\[2pt]",
         r"\begin{tabular}{lrrrr}", r"\toprule",
         r"Model & Full pool (2,850) & Syn.\ Depth (500) & w/o Syn.\ Depth, task-weighted & w/o Syn.\ Depth, item-weighted \\", r"\midrule"]
    for m in M8:
        o = mp[m]
        f = o["full_pool_item_weighted"]
        L.append(f"{NAMES[m]} & {sg(f['est'])} {ci(f['ci'])} & {sg(o['per_task']['syn_depth'])} {ci(o['per_task_ci']['syn_depth'])} & "
                 f"{sg(o['task_weighted_no_syn_depth'])} {ci(o['task_weighted_no_syn_depth_ci'])} & "
                 f"{sg(o['item_weighted_no_syn_depth'])} {ci(o['item_weighted_no_syn_depth_ci'])} \\\\")
    L.append(r"\midrule")
    for nm, lab in (("mean8", "Mean (8)"), ("mean7_instruct", "Mean (7 instruct)")):
        o = mp[nm]
        L.append(f"{lab} & {sg(o['full_pool'])} & {sg(o['syn_depth'])} {ci(o['syn_depth_ci'])} & "
                 f"{sg(o['task_weighted_no_syn_depth'])} {ci(o['task_weighted_no_syn_depth_ci'])} & "
                 f"{sg(o['item_weighted_no_syn_depth'])} {ci(o['item_weighted_no_syn_depth_ci'])} \\\\")
    L += [r"\bottomrule", r"\end{tabular} \\[6pt]",
          r"\textbf{(b)} Two-channel fits with and without Synthetic Depth 431 \\[2pt]",
          r"\begin{tabular}{llrrrr}", r"\toprule",
          r"Protocol & Points & $r$ & Slope $b$ & Intercept $a$ & sign$(x)=$sign$(y)$ \\", r"\midrule"]
    for key, lab in (("fig5_conclusion_kept_v3", "Fig.~5: conclusion kept"), ("s13_uniform_removed", "Tab.~S13: conclusion removed")):
        for sub, pl in (("21", "21 (all)"), ("14_no_syn431", "14 (w/o Syn.\\ Depth)")):
            f = F[f"{key}|{sub}"]
            L.append(f"{lab if sub == '21' else ''} & {pl} & {f['pearson_r']:.2f} {cip(f['r_ci_point'])} / {cip(f['r_ci_model'])} & "
                     f"{f['slope']:.2f} {cip(f['slope_ci_point'])} / {cip(f['slope_ci_model'])} & "
                     f"{sg(f['intercept'], 2)} {ci(f['intercept_ci_point'], 2)} / {ci(f['intercept_ci_model'], 2)} & "
                     f"{f['sign_agree_x_y']}/{f['n_points']} \\\\")
    L += [r"\bottomrule", r"\end{tabular} \\[6pt]",
          r"\textbf{(c)} Agreement with the last relation of the conclusion-removed body, $a_T$ / $a_F$ (\%) \\[2pt]",
          r"\begin{tabular}{lrrr}", r"\toprule", r"Model & Cue set & Twins & Syn.\ Depth 431 \\", r"\midrule"]
    for m in M7:
        cells = []
        for ds in ("cues", "twins", "syn431"):
            a = A[f"{m}|{ds}"]
            cells.append(f"{a['aT']:.1f} / {a['aF']:.1f} {{\\scriptsize ({a['n_true']}/{a['n_false']})}}")
        L.append(f"{NAMES[m]} & " + " & ".join(cells) + r" \\")
    aT = [v["aT"] for v in A.values()]
    aF = [v["aF"] for v in A.values()]
    L.append(r"\midrule")
    L.append(f"Range (21 cells) & \\multicolumn{{3}}{{c}}{{$a_T$ {min(aT):.1f}--{max(aT):.1f}, $a_F$ {min(aF):.1f}--{max(aF):.1f}}} \\\\")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{tabular}"]
    return "\n".join(L) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="a,b")
    a = ap.parse_args()
    parts = a.only.split(",")
    txt = B / "rigor_e58.txt"
    old = txt.read_text().splitlines() if txt.exists() else []
    lines: list[str] = []
    if "a" in parts:
        import e58_bodyrel
        la: list[str] = []
        ra = e58_bodyrel.run(la)
        json.dump(ra, open(B / "e58_bodyrel.json", "w"), indent=1, default=jdefault)
        (PAPER / "tables" / "supp_e58a.tex").write_text(table_a(ra))
        lines += ["# Part A: BodyRel"] + la
    if "b" in parts:
        import e58_leverage
        lb: list[str] = []
        rb = e58_leverage.run(lb)
        for k in list(rb["fits"]):
            if isinstance(rb["fits"][k], dict):
                rb["fits"][k] = {kk: vv for kk, vv in rb["fits"][k].items() if not kk.startswith("_")}
        json.dump(rb, open(B / "e58_leverage.json", "w"), indent=1, default=jdefault)
        (PAPER / "tables" / "supp_e58b.tex").write_text(table_b(rb))
        lines += ["# Part B: leverage"] + lb
    keep, cur = [], None
    for x in old:  # keep the lines of a part that was not re-run
        if x.startswith("# Part"):
            cur = "a" if x.startswith("# Part A") else "b"
        if cur not in parts:
            keep.append(x)
    txt.write_text("\n".join(keep + lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
