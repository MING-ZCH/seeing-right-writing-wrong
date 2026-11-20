"""Exp-63a/b: draft supplement snippet (two compact tables + one paragraph each) from r28_written_ame.json and
r28_jstar_condition.json. Standalone file, not \\input anywhere.
-> outputs/bundle/r28_supp_draft.tex
Usage: python r28_supp_draft.py
"""
from __future__ import annotations

import json
import re
from pathlib import Path

B = Path("outputs/bundle")
A_MODELS = ["qwen3b", "qwen7b", "internvl3_8b", "phi35v", "molmo7b", "qwen3vl_8b", "internvl35_8b", "qwen3vl_32b",
            "qwen3vl_8b_think"]
B_MODELS = ["internvl35_38b", "qwen32b", "mimo_vl_7b_rl|cot", "glm41v_9b_think|cot"]
TEXN = {"qwen3b": "Qwen2.5-VL-3B", "qwen7b": "Qwen2.5-VL-7B", "internvl3_8b": "InternVL3-8B", "phi35v": "Phi-3.5-V",
        "molmo7b": "Molmo-7B-D", "qwen3vl_8b": "Qwen3-VL-8B", "internvl35_8b": "InternVL3.5-8B", "qwen3vl_32b": "Qwen3-VL-32B",
        "qwen3vl_8b_think": "Qwen3-VL-8B-Think.", "internvl35_38b": "InternVL3.5-38B$^\\dagger$",
        "qwen32b": "Qwen2.5-VL-32B$^\\dagger$", "mimo_vl_7b_rl|cot": "MiMo-VL-7B-RL$^\\dagger$",
        "glm41v_9b_think|cot": "GLM-4.1V-9B-Think.$^\\dagger$"}


def m_(s: str) -> str:
    return re.sub(r"(?<![\w.])-(?=\d)", "$-$", s)


def d(x: dict) -> str:
    lo, hi = (m_(f"{v:+.1f}") for v in x["ci"])
    return m_(f"{x['est']:+.1f}") + f" {{\\tiny[{lo}, {hi}]}}"


def v1(x: dict) -> str:
    return m_(f"{x['est']:.1f}")


def table_a(R: dict) -> str:
    L = [r"\begin{table*}[t]", r"\centering", r"\scriptsize", r"\setlength{\tabcolsep}{1.5pt}",
         r"\caption{\textbf{Cue effects of the written relation} (cue-factorial set). Average marginal effect (AME, points) of the "
         r"height (lower is closer) and occlusion cues from the probit of \cref{tab:supp_cue_ame}, fitted to the \texttt{Direct} answer (D), "
         r"to the last relation written in the conclusion-removed rationale $R_\mathrm{cut}$ (W; two text-only judges, consensus) and to "
         r"the \texttt{CoT} answer (C), all on the items with a written relation (Cov.). W$-$D: change made by writing; C$-$W: change "
         r"at the answer step. Twin-pair bootstrap $95\%$ CIs ($1{,}000$ draws, same draws and seed as \cref{tab:supp_cue_ame}). "
         r"$^\dagger$No $R_\mathrm{cut}$ judge labels: rule parser on $R_\mathrm{cut}$; it reproduces the judges' height changes "
         r"but not their occlusion changes (e.g.\ Molmo-7B-D W$-$D occlusion $-6.0$ vs.\ $+11.1$), so only the height columns are "
         r"interpretable for these rows.}",
         r"\label{tab:supp_written_ame}",
         r"\begin{tabular}{lr rrrcc rrrcc}", r"\toprule",
         r" & & \multicolumn{5}{c}{Height (lower is closer)} & \multicolumn{5}{c}{Occlusion} \\",
         r"\cmidrule(lr){3-7}\cmidrule(lr){8-12}",
         r"Model & Cov. & D & W & C & W$-$D & C$-$W & D & W & C & W$-$D & C$-$W \\", r"\midrule"]
    for m in A_MODELS + B_MODELS:
        if m == B_MODELS[0]:
            L.append(r"\midrule")
        r = R[m]
        v = r["variants"]["consensus"] if "consensus" in r["variants"] else r["variants"]["v3_trunc"]
        h, o = v["vertical"]["ame"], v["occlusion"]["ame"]
        L.append(f"{TEXN[m]} & {v['coverage_pct']:.0f} & {v1(h['D'])} & {v1(h['W'])} & {v1(h['C'])} & {d(h['W-D'])} & {d(h['C-W'])} & "
                 f"{v1(o['D'])} & {v1(o['W'])} & {v1(o['C'])} & {d(o['W-D'])} & {d(o['C-W'])} \\\\")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table*}"]
    return "\n".join(L)


def pc(x: dict | None, ci: bool = True) -> str:
    if not x:
        return "--"
    s = f"{x['mean']:.1f}"
    return s + (f" {{\\tiny[{x['ci'][0]:.1f}, {x['ci'][1]:.1f}]}}" if ci else "")


def p3(x: dict | None) -> str:
    return "--" if not x else m_(f"{100 * x['mean']:+.1f}")


def table_b(J: dict) -> tuple[str, dict]:
    rows = [("qwen3vl_8b|cues", "judge", "Qwen3-VL-8B", "cue set", "judges"),
            ("qwen3vl_8b|cues", "parser", "Qwen3-VL-8B", "cue set", "parser"),
            ("qwen3vl_32b|cues", "parser", "Qwen3-VL-32B", "cue set", "parser"),
            ("qwen3vl_8b|twins", "judge", "Qwen3-VL-8B", "twins", "judges"),
            ("molmo7b|cues", "judge", "Molmo-7B-D", "cue set", "judges"),
            ("qwen3vl_8b|syn", "judge", "Qwen3-VL-8B", "Syn.\\ Depth", "judges"),
            ("molmo7b|syn", "judge", "Molmo-7B-D", "Syn.\\ Depth", "judges")]
    L = [r"\begin{table*}[t]", r"\centering", r"\scriptsize", r"\setlength{\tabcolsep}{2.8pt}",
         r"\caption{\textbf{False relational sentences before the commit sentence $j^\ast$} (\texttt{D+C$-$} rationales with a "
         r"commit). $k$: earliest sentence before $j^\ast$ that contradicts the image (judges: two image-conditioned judges, "
         r"consensus, on every relational sentence before $j^\ast$ of a sample; parser: rule-parsed depth claim naming the occluded "
         r"shape as front, all rationales). $P$: forced-readout $P(\text{gold})$ after the sentence; $\Delta P$: its change (points) at "
         r"$k$ and at $j^\ast$ (same rationales). $P\ge.5$ just before $j^\ast$ holds by definition, so $P$ after $k$ is only "
         r"partly informative; the twin-image effect (readout under the own image minus under the identity-swapped twin, "
         r"\cref{tab:supp_e57}) is not constrained by the definition. Depth: share of sentences with an explicit depth word "
         r"(in front, behind, closer, farther). Last columns: $j^\ast$ judged false when an earlier false sentence exists vs.\ not. "
         r"Image-cluster bootstrap $95\%$ CIs ($2{,}000$ draws).}",
         r"\label{tab:supp_jstar_condition}",
         r"\begin{tabular}{lll r c c rr ccc cc cc}", r"\toprule",
         r" & & & & & & \multicolumn{2}{c}{$\Delta P$} & \multicolumn{3}{c}{Twin-image effect} & \multicolumn{2}{c}{Depth word} & "
         r"\multicolumn{2}{c}{$j^\ast$ false} \\",
         r"\cmidrule(lr){7-8}\cmidrule(lr){9-11}\cmidrule(lr){12-13}\cmidrule(lr){14-15}",
         r"Model & Set & Labels & $n$ & Earlier false $k$ & $P$ after $k$ & at $k$ & at $j^\ast$ & before $k$ & after $k$ & after $j^\ast$ & "
         r"$k$ & $j^\ast$ & earlier & none \\", r"\midrule"]
    for key, src, mn, sn, lab in rows:
        b = J["units"][key][src]
        mk, mj = b["move_earliest_false"], b["move_jstar_with_earlier"]
        fc = b["form_earliest_vs_jstar"]
        ie = (pc(mk.get("image_effect_before"), False), pc(mk.get("image_effect_after"), False), pc(mj.get("image_effect_after"), False))
        L.append(f"{mn} & {sn} & {lab} & {b['n_rationales']} & {pc(b['earlier_false'])} & {mk['P_after']['mean']:.2f} & "
                 f"{p3(mk['dP'])} & {p3(mj['dP'])} & {ie[0]} & {ie[1]} & {m_(ie[2])} & {pc(fc.get('k_depth_word'), False)} & "
                 f"{pc(fc.get('jstar_depth_word'), False)} & {pc(b['jstar_false_given_earlier'], False)} & {pc(b['jstar_false_given_none'], False)} \\\\")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table*}"]
    return "\n".join(L), J


def paragraphs(R: dict, J: dict) -> tuple[str, str]:
    def f(x: dict) -> str:
        lo, hi = (m_(f"{v:+.1f}") for v in x["ci"])
        return m_(f"{x['est']:+.1f}") + f" [{lo}, {hi}]"

    def sg(v: float, fmt_: str = "+.1f") -> str:
        return m_(format(v, fmt_))

    va = {m: R[m]["variants"]["consensus"] for m in A_MODELS}
    hw = [va[m]["vertical"]["ame"]["W-D"]["est"] for m in A_MODELS]
    cw = [va[m]["vertical"]["ame"]["C-W"]["est"] for m in A_MODELS]
    mo, q32 = va["molmo7b"]["occlusion"]["ame"], va["qwen3vl_32b"]["occlusion"]["ame"]
    cov = [va[m]["coverage_pct"] for m in A_MODELS]
    pa = (r"\paragraph{Cue effects of the written relation.} \Cref{tab:supp_written_ame} refits the probit of "
          r"\cref{tab:supp_cue_ame} to the last relation written before the conclusion (W, two-judge consensus) and refits "
          rf"\texttt{{Direct}} (D) and \texttt{{CoT}} (C) on the same items (coverage {min(cov):.0f}--{max(cov):.0f}\%). Height falls "
          rf"already in W, in all nine judged models (W$-$D {sg(min(hw), '.1f')} to {sg(max(hw), '.1f')} points, all CIs below 0) "
          rf"and in the four parser-only models; the answer step partly restores it (C$-$W up to {sg(max(cw))}). The occlusion loss "
          rf"of \texttt{{Qwen3-VL-32B}} is in W ({f(q32['W-D'])}; C$-$W {sg(q32['C-W']['est'])}). "
          rf"\texttt{{Molmo-7B-D}}'s is not: W uses occlusion more than D ({f(mo['W-D'])}) and the loss arises at the answer step "
          rf"({f(mo['C-W'])}). The occlusion gains of \texttt{{InternVL3-8B}} and \texttt{{InternVL3.5-8B}} are larger in W "
          rf"({sg(va['internvl3_8b']['occlusion']['ame']['W-D']['est'])}, {sg(va['internvl35_8b']['occlusion']['ame']['W-D']['est'])}) "
          r"than in C.")
    q = J["units"]["qwen3vl_8b|cues"]
    jb, pb = q["judge"], q["parser"]
    mk, mj = jb["move_earliest_false"], jb["move_jstar_with_earlier"]
    fc = jb["form_earliest_vs_jstar"]
    sf = fc["same_front_if_both"]
    n32 = J["units"]["qwen3vl_32b|cues"]["parser"]["n_rationales"]
    ef, ex = jb["earlier_false"], pb["move_false_explicit_before_jstar"]
    dpk, dpj = 100 * mk["dP"]["mean"], 100 * mj["dP"]["mean"]
    iek, iej = mk["image_effect_after"]["mean"], mj["image_effect_after"]["mean"]
    pbk = (r"\paragraph{False sentences before $j^\ast$.} "
           rf"Of {jb['n_rationales']} judged \texttt{{Qwen3-VL-8B}} cue-set \texttt{{D+C$-$}} rationales, {ef['mean']:.1f}\% "
           rf"[{ef['ci'][0]:.1f}, {ef['ci'][1]:.1f}] contain an image-contradicting relational sentence $k$ before $j^\ast$ "
           rf"(\cref{{tab:supp_jstar_condition}}). Writing $k$ leaves the readout in place: $P(\text{{gold}})$ changes by {sg(dpk)} "
           rf"points at $k$ vs.\ {sg(dpj)} at $j^\ast$, and a twin image still moves it by {iek:.1f} points after $k$ "
           rf"but by {sg(iej, '.1f')} after $j^\ast$. $k$ names the same front shape as $j^\ast$ ({round(sf['mean'] * sf['n'] / 100)}/{sf['n']} "
           rf"where both parse), mostly as overlap (depth word {fc['k_depth_word']['mean']:.0f}\% vs.\ {fc['jstar_depth_word']['mean']:.0f}\%; "
           rf"inference marker {fc['k_infer_marker']['mean']:.0f}\% vs.\ {fc['jstar_infer_marker']['mean']:.0f}\%). Wording alone does not "
           rf"commit: {ex['n']} parsed pre-$j^\ast$ false sentences with an explicit front/behind predicate leave $P(\text{{gold}})$ at "
           rf"{ex['P_after']['mean']:.2f}. Rule-parsed full sets ({pb['n_rationales']}; \texttt{{Qwen3-VL-32B}} {n32}) agree; "
           rf"in \texttt{{Molmo-7B-D}} earlier false sentences are rare ({J['units']['molmo7b|cues']['judge']['earlier_false']['mean']:.0f}\%).")
    return pa, pbk


def main() -> None:
    R = json.load(open(B / "r28_written_ame.json"))["models"]
    J = json.load(open(B / "r28_jstar_condition.json"))
    pa, pb = paragraphs(R, J)
    ta = table_a(R)
    tb, _ = table_b(J)
    def wc(s: str) -> int:
        x = re.sub(r"\\paragraph\{[^}]*\}", "", s.replace("$-$", "-"))
        x = re.sub(r"\\(texttt|cref|Cref|text)\{([^}]*)\}", r"\2", x)
        return len(re.sub(r"[\\$]", "", x).split())
    txt = ("% Exp-63a/b draft supplement snippet (r28_supp_draft.py). NOT \\input anywhere; numbers generated from\n"
           "% outputs/bundle/r28_written_ame.json and r28_jstar_condition.json. Needs booktabs + cleveref; \\tiny CIs.\n"
           f"% paragraph word counts (approx.): A {wc(pa)}, B {wc(pb)}\n\n"
           + pa + "\n\n" + ta + "\n\n" + pb + "\n\n" + tb + "\n")
    (B / "r28_supp_draft.tex").write_text(txt)
    print(txt)


if __name__ == "__main__":
    main()
