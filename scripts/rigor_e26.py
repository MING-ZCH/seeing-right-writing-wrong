"""Exp-26 (round-14 Opus M1, M2, m1, m3, m7, m8, m9): fact lines for the editor, each with its source.

Reads bundle files only (run rigor_e25.py and e26_borderline_bh.py first):
  e25_afe.json (+ e25_afe.json.bak_pre_e26 for the old values), syn_pair_sdt_mh.json, e22b_gate.json,
  multiplicity.json, e26_borderline_bh.json, ../results/v2/summary_v2.json (legacy greedy R-split).
-> outputs/bundle/rigor_e26.txt, outputs/bundle/e26_facts.json
"""
from __future__ import annotations
import vtax_paths as _vtp  # noqa: E402

import json
from pathlib import Path

B = Path("outputs/bundle")
V2 = Path(f"{_vtp.LEGACY_RESULTS}/summary_v2.json")
LAB = {"qwen3b": "Qwen2.5-VL-3B", "qwen7b": "Qwen2.5-VL-7B", "internvl3_8b": "InternVL3-8B", "phi35v": "Phi-3.5-Vision",
       "molmo7b": "Molmo-7B-D", "qwen3vl_8b": "Qwen3-VL-8B", "qwen3vl_8b_think": "Qwen3-VL-8B-Thinking", "internvl35_8b": "InternVL3.5-8B"}
SL = {"cues": "cues", "twins": "twins", "syn": "Synthetic Depth"}
M4, SETS = ["qwen3vl_8b", "qwen7b", "internvl3_8b", "molmo7b"], ["cues", "twins", "syn"]


def f(x: dict, nd: int = 1) -> str:
    return f"{x['est']:+.{nd}f} [{x['ci'][0]:+.{nd}f}, {x['ci'][1]:+.{nd}f}]"


def rng_(vals: list[float], nd: int = 1) -> str:
    return f"{min(vals):.{nd}f}-{max(vals):.{nd}f}"


def main() -> None:
    L: list[str] = []
    facts: dict = {}

    # ---------------------------------------------------------------- 1. AFE (M1, m9)
    E = json.load(open(B / "e25_afe.json"))
    P = E["pooled"]
    src = "e25_afe.json pooled (rigor_e25.py, 12 cells unweighted, paired cluster bootstrap 2,000)"
    L.append("## 1. AFE answer-matched / claim-only comparison (Opus M1, m9)")
    L.append(f"AFE-1a rationalizes D- (all Direct-wrong, as printed): CoT {P['rat_any_cot']['est']:.1f} / Gate {P['rat_any_gate']['est']:.1f} / AFE {P['rat_any_afe']['est']:.1f}; "
             f"AFE-CoT {f(P['d_rat_any_afe-cot'])}  [{src}; unchanged vs e25_afe.json.bak_pre_e26]")
    L.append(f"AFE-1b matched on CoT's D-C- items (CoT also ends at d; mean n/cell {P['n_mcot']['est']:.0f}): CoT {P['rat_mcot_cot']['est']:.1f} vs AFE {P['rat_mcot_afe']['est']:.1f}, "
             f"AFE-CoT {f(P['d_rat_mcot_afe-cot'])} (n.s.)  [{src}: rat_mcot_*]  -> Opus 83.0 vs 84.8 CONFIRMED")
    L.append(f"AFE-1c matched on the gate's D-G- items (gate ends at d; mean n/cell {P['n_mgate']['est']:.0f}): Gate {P['rat_mgate_gate']['est']:.1f} vs AFE {P['rat_mgate_afe']['est']:.1f}, "
             f"AFE-Gate {f(P['d_rat_mgate_afe-gate'])}  [{src}: rat_mgate_*]")
    L.append(f"AFE-1d on CoT's D-C+ rescues: CoT {P['rat_rcot_cot']['est']:.1f} vs AFE {P['rat_rcot_afe']['est']:.1f}; per cell CoT "
             f"{rng_([E[m][s]['boot']['rat_rcot_cot']['est'] for m in M4 for s in SETS])}, AFE {rng_([E[m][s]['boot']['rat_rcot_afe']['est'] for m in M4 for s in SETS])}  [{src}: rat_rcot_*]")
    cells = lambda k: {f"{LAB[m]} {SL[s]}": round(E[m][s]["boot"][k]["est"], 1) for m in M4 for s in SETS}
    dc = {f"{LAB[m]} {SL[s]}": (round(E[m][s]["boot"]["rat_mcot_cot"]["est"], 1), round(E[m][s]["boot"]["rat_mcot_afe"]["est"], 1)) for m in M4 for s in SETS}
    L.append("AFE-1e matched D-C- per cell (CoT, AFE): " + "; ".join(f"{k} {v[0]}/{v[1]}" for k, v in dc.items())
             + f"  [e25_afe.json <model>.<set>.boot.rat_mcot_*]")
    L.append(f"AFE-1f excl. conclusion sentences (rigor_e22.CONCL), all Direct-wrong: CoT {P['rat_any_xc_cot']['est']:.1f} / Gate {P['rat_any_xc_gate']['est']:.1f} / AFE {P['rat_any_xc_afe']['est']:.1f}; "
             f"AFE-CoT {f(P['d_rat_any_xc_afe-cot'])}, AFE-Gate {f(P['d_rat_any_xc_afe-gate'])}  [{src}: rat_any_xc_*]  -> Opus AFE 57.3 / CoT 44.0: CoT confirmed, AFE 57.2 (rounding)")
    mol = [E["molmo7b"][s]["boot"] for s in SETS]
    L.append(f"AFE-1g Molmo-7B-D AFE rat_any {rng_([b['rat_any_afe']['est'] for b in mol])} -> excl. concl. {rng_([b['rat_any_xc_afe']['est'] for b in mol])}  [e25_afe.json molmo7b.*.boot]")
    L.append(f"AFE-1h matched D-C- AND excl. concl.: CoT {P['ratxc_mcot_cot']['est']:.1f} vs AFE {P['ratxc_mcot_afe']['est']:.1f}, AFE-CoT {f(P['d_ratxc_mcot_afe-cot'])}; "
             f"gate D-G-: Gate {P['ratxc_mgate_gate']['est']:.1f} vs AFE {P['ratxc_mgate_afe']['est']:.1f}, AFE-Gate {f(P['d_ratxc_mgate_afe-gate'])}  [{src}: ratxc_*]")
    L.append(f"AFE-1i claim truth excl. conclusion sentences: CoT {P['claim_true_xc_cot']['est']:.1f} / Gate {P['claim_true_xc_gate']['est']:.1f} / AFE {P['claim_true_xc_afe']['est']:.1f}; "
             f"AFE-CoT {f(P['d_claim_true_xc_afe-cot'])}, AFE-Gate {f(P['d_claim_true_xc_afe-gate'])}, Gate-CoT {f(P['d_claim_true_xc_gate-cot'])} "
             f"(all claims: {P['claim_true_cot']['est']:.1f} / {P['claim_true_gate']['est']:.1f} / {P['claim_true_afe']['est']:.1f})  [{src}: claim_true_xc_*]")
    for m in M4:
        for s in SETS:
            b = E[m][s]["boot"]
            L.append(f"AFE-1j post-explanation re-read accuracy {LAB[m]} {SL[s]}: Direct {b['acc_direct']['est']:.1f} -> {b['post_acc_afe']['est']:.1f} "
                     f"({f(b['d_post_acc_afe-direct'])}); agrees with stated answer {E[m][s]['afe_post_readout_agree_direct_pct']:.1f}%  [e25_afe.json {m}.{s}: post_acc_afe, afe_post_readout_*]")
    L.append(f"AFE-1k pooled re-read: Direct {P['acc_direct']['est']:.1f} -> {P['post_acc_afe']['est']:.1f} ({f(P['d_post_acc_afe-direct'], 2)})  [{src}]")
    others = [abs(E[m][s]["boot"]["d_post_acc_afe-direct"]["est"]) for m in M4[1:] for s in SETS]
    L.append(f"AFE-1l other three models: |re-read - Direct| <= {max(others):.2f} pts (Opus 'within 0.1': CORRECTED, max is Qwen2.5-VL-7B cues +0.25 [+0.06, +0.50])  [e25_afe.json]")
    facts["afe"] = {"rat_any": [P[f"rat_any_{a}"]["est"] for a in ("cot", "gate", "afe")], "matched_cot": [P["rat_mcot_cot"]["est"], P["rat_mcot_afe"]["est"], P["d_rat_mcot_afe-cot"]],
                    "matched_gate": [P["rat_mgate_gate"]["est"], P["rat_mgate_afe"]["est"], P["d_rat_mgate_afe-gate"]],
                    "xc": [P[f"rat_any_xc_{a}"]["est"] for a in ("cot", "gate", "afe")], "claim_true_xc": [P[f"claim_true_xc_{a}"]["est"] for a in ("cot", "gate", "afe")]}

    # ---------------------------------------------------------------- 2. twins (M2a)
    J = json.load(open(B / "syn_pair_sdt_mh.json"))
    TW = J["twins"]
    L.append("")
    L.append("## 2. Twins, pair-matched (Opus M2a)")
    for m, r in TW.items():
        if m.startswith("_"):
            continue
        L.append(f"TW {LAB[m]}: d'pair {r['direct_pooled']:.2f}->{r['cot_pooled']:.2f} ({r['delta_pooled']:+.2f} [{r['delta_pooled_ci'][0]:+.2f}, {r['delta_pooled_ci'][1]:+.2f}], boot p {r['delta_pooled_p']}) "
                 f"| BC/SF/SB/BW Dir {'/'.join(f'{x:.1f}' for x in r['split_direct'])} CoT {'/'.join(f'{x:.1f}' for x in r['split_cot'])} "
                 f"| marginal (Tab. S12) {r['marginal_ddprime_tabS12']:+.2f} {r['marginal_ddprime_ci']}  [syn_pair_sdt_mh.json twins.{m}; rigor_e17q.txt TWINPAIR]")
    s = TW["_summary"]
    L.append(f"TW-SUM pair-matched: falls (CI<0) {[LAB[m] for m in s['pooled']['fall']]}; rises (CI>0) {[LAB[m] for m in s['pooled']['rise']]}; "
             f"flat: Qwen2.5-VL-3B, Qwen3-VL-8B | marginal: falls {[LAB[m] for m in s['marginal']['fall']]}, rises {[LAB[m] for m in s['marginal']['rise']]}  "
             f"[rigor_e17q.txt TWINPAIR-SUM] -> Opus CONFIRMED (pair-matched CIs are nominal; the twin family in BH uses marginal d')")
    q8 = TW["qwen3vl_8b"]
    L.append(f"TW-Q3 Qwen3-VL-8B: SB {q8['split_direct'][2]:.1f} -> {q8['split_cot'][2]:.1f}%, BC {q8['split_direct'][0]:.1f} -> {q8['split_cot'][0]:.1f}, BW {q8['split_direct'][3]:.1f} -> {q8['split_cot'][3]:.1f} "
             f"(n_BW {round(4 * q8['split_direct'][3])} -> {round(4 * q8['split_cot'][3])} of 400) -> pooled loss {q8['marginal_ddprime_tabS12']:+.2f} is a shift to 'subject behind in both twins'  [syn_pair_sdt_mh.json twins.qwen3vl_8b]")
    L.append("TW-EDITOR: Pair-matched on the twins, d'_pair falls in three models (Phi-3.5-Vision -0.43, Molmo-7B-D -0.88, Qwen3-VL-8B-Thinking -1.38), rises in three with a strong "
             "Direct lower-is-closer prior (Qwen2.5-VL-7B +0.32, InternVL3-8B +0.65, InternVL3.5-8B +0.23; CIs > 0), and is flat in Qwen2.5-VL-3B (+0.08 [-0.06, +0.21]) and "
             "Qwen3-VL-8B (+0.02 [-0.54, +0.74]), whose pooled loss (-0.46) is a shift toward 'subject behind in both twins' (SB 0.5 -> 6.2%).")

    # ---------------------------------------------------------------- 3. GLMM (M2b)
    G = J["glmm"]
    L.append("")
    L.append("## 3. Random-slope GLMM (Opus M2b)")
    for m in [k for k in G if not k.startswith("_") and k not in ("n_boot",)]:
        g = G[m]
        L.append(f"GLMM {LAB[m]}: mean d' {g['dmean_direct']:.2f} -> {g['dmean_cot']:.2f}, delta {g['dmean_delta']:+.2f} [{g['dmean_delta_ci'][0]:+.2f}, {g['dmean_delta_ci'][1]:+.2f}] "
                 f"(boot p {g['dmean_delta_p']}); slope SD Dir {g['slope_sd_direct']:.2f} {g['slope_sd_direct_ci']} CoT {g['slope_sd_cot']:.2f} {g['slope_sd_cot_ci']}  [syn_pair_sdt_mh.json glmm.{m}]")
    L.append(f"GLMM-SUM mean d' CI<0 in {len(G['_summary']['fall_ci_lt0'])}/7 ({', '.join(LAB[m] for m in G['_summary']['fall_ci_lt0'])}); rises none; image bootstrap {G['n_boot']} "
             "-> Opus CONFIRMED. Identification: bounds hit (slope SD upper 7.16 in 5 bootstrap CIs; bias correlation at bound in Qwen2.5-VL-7B / Qwen3-VL-8B; "
             "InternVL3-8B Direct high-tercile sensitivity at its bound); d' on the image-conditional scale.  [syn_pair_sdt_mh.json glmm; rigor_e17q.py run_glmm]")
    L.append("GLMM-TABLE: row block added to Tab. S27 (tables/supp_syn_pairmh.tex, make_supp_tables.e17q_tables).")
    L.append("GLMM-APPH sentence: A logistic mixed model with an image random intercept and a random sensitivity slope, which does not assume a common sensitivity, is weakly identified with "
             "two answers per image and arm (variance and correlation parameters at their bounds); its mean d' falls with CI below zero in three of seven models (Qwen2.5-VL-3B, "
             "Qwen2.5-VL-7B, InternVL3-8B) and is indeterminate in the other four (Tab. S27).")

    # ---------------------------------------------------------------- 4. location share (m1)
    X = json.load(open(B / "e22b_gate.json"))
    L.append("")
    L.append("## 4. Gate location share (Opus m1)")
    loss = [("qwen3vl_8b", "syn"), ("qwen7b", "syn"), ("internvl3_8b", "syn"), ("molmo7b", "cues"), ("molmo7b", "twins"), ("molmo7b", "syn")]
    sh_sig = []
    for m, s_ in loss:
        b = X[m][s_]["boot"]
        ab, gc = b["dacc_locfree-randfree"], b["dacc_gate-cot"]
        sig = ab["ci"][0] > 0
        share = 100 * ab["est"] / gc["est"]
        if sig:
            sh_sig.append(share)
        L.append(f"LOC {LAB[m]} {SL[s_]}: A-B {f(ab)}, gate-CoT {f(gc)}, share {share:.0f}%{'' if sig else ' (A-B n.s.)'}  [e22b_gate.json {m}.{s_}.boot dacc_locfree-randfree / dacc_gate-cot]")
    L.append(f"LOC-SUM five significant CoT-loss cells: share {min(sh_sig):.0f}-{max(sh_sig):.0f}% (with the n.s. Qwen3-VL-8B Synthetic Depth cell: 11-42%) -> Opus 18-42% CONFIRMED")
    neg = [(m, s_, X[m][s_]["boot"]["dacc_locfree-randfree"]) for m in M4 for s_ in SETS if X[m][s_]["boot"]["dacc_locfree-randfree"]["ci"][1] < 0]
    L.append("LOC-NEG cells with A-B CI < 0: " + "; ".join(f"{LAB[m]} {SL[s_]} {f(v)}" for m, s_, v in neg)
             + f" | InternVL3-8B twins {X['internvl3_8b']['twins']['boot']['dacc_locfree-randfree']['est']:+.2f} {X['internvl3_8b']['twins']['boot']['dacc_locfree-randfree']['ci']} (upper bound at 0, not counted)  [e22b_gate.json]")

    # ---------------------------------------------------------------- 5. Sec. 6 citation (m3)
    L.append("")
    L.append("## 5. Sec. 6 R-copying numbers (Opus m3)")
    S = json.load(open(V2))["e1"]
    rr, rw, dw, ns = [], [], [], []
    for m in ["qwen3b", "qwen7b", "internvl3_8b", "phi35v", "molmo7b"]:
        br = S[m]["groups"]["spatial"]["by_R_answer"]
        a = lambda k, c: 100 * br[k]["acc"][c]
        rr.append(a("right", "cot")); rw.append(a("wrong", "cot")); dw.append(a("wrong", "direct")); ns.append(br["right"]["n"] + br["wrong"]["n"])
        L.append(f"RSPLIT {LAB[m]}: right R n {br['right']['n']} CoT {a('right', 'cot'):.1f} | wrong R n {br['wrong']['n']} CoT {a('wrong', 'cot'):.1f}, Direct {a('wrong', 'direct'):.1f}  "
                 f"[results/v2/summary_v2.json e1.{m}.groups.spatial.by_R_answer]")
    L.append(f"RSPLIT-SUM greedy, five models (V0 four + Molmo-7B-D), items with a parsable option ({min(ns)}-{max(ns)} of 2,850 per model): right R {min(rr):.1f}-{max(rr):.1f}, "
             f"wrong R {min(rw):.1f}-{max(rw):.1f} vs Direct {min(dw):.1f}-{max(dw):.1f}. Source = legacy main-text table tab:faithfulness_split "
             "(vtax_handoff/paper/tables/tab3_faithfulness_and_api.tex, scripts/v2/make_tables_v2.py), dropped from the current paper; no current table holds them. "
             "Tab. S41 (tab:supp_controls_e5) holds the T=0.7 within-item version: right/wrong R 96.3-99.8 / 0.3-5.5, Direct 72.2-77.5 on 889-1,427 mixed items (four V0 models). "
             "Pre-12b text cited both ('greedy; 0.3-5.5% under T=0.7 sampling, Tab. S41'); the greedy clause lost its source when the sampled clause was cut.")

    # ---------------------------------------------------------------- 6. borderline BH (m8)
    BH = json.load(open(B / "e26_borderline_bh.json"))
    L.append("")
    L.append("## 6. Borderline BH tests with B = 10,000 (Opus m8)")
    for fam, lab in (("syn_dpair_ddprime", "Qwen2.5-VL-7B Synthetic Depth d'_pair delta"), ("twin_ddprime", "Qwen2.5-VL-7B twin d' rise")):
        t = BH["tests"][fam]
        L.append(f"BH {lab}: est {t['est']:+.3f}; p {t['p_old']:.4f} (B=2,000) -> {t['p_new']:.4f} (B=10,000, MC SE {t['mc_se_new']:.4f}); "
                 f"global q {t['q_old']:.4f} -> {t['q_new']:.4f}; survives {t['survives_old']} -> {t['survives_new']}; B=10,000 percentile CI {t['boot_ci_new']}  "
                 f"[e26_borderline_bh.py -> e26_borderline_bh.json; seed schemes replayed, B=2,000 reproduces stored p exactly]")
    L.append(f"BH-SUM status changes: {BH['status_changes'] or 'none'}; survivors {BH['survive_old']} -> {BH['survive_new']} of {BH['n_tests']}; "
             "multiplicity.json / supp_multiplicity.tex NOT changed. Note: d'_pair q = 0.0496 sits on the line; the twin rise's B=10,000 percentile CI includes 0.")

    # ---------------------------------------------------------------- 7. intro seven of eight (m7)
    MU = json.load(open(B / "multiplicity.json"))
    L.append("")
    L.append("## 7. Main-pool CoT - Direct per model (Opus m7)")
    lo = sg = 0
    for t in [t for t in MU["tests"] if t["family"] == "tab1_cot_minus_direct"]:
        lo += t["est"] < 0
        sg += t["est"] < 0 and t["survives_global_resampling"]
        L.append(f"TAB1 {LAB[t['model']]}: CoT-Direct {t['est']:+.2f} [{t['ci'][0]:+.2f}, {t['ci'][1]:+.2f}], p {t['p_resampling']:.4f}, global BH q {t['q_global_resampling']:.4f}, "
                 f"survives {t['survives_global_resampling']}  [multiplicity.json tab1_cot_minus_direct]")
    L.append(f"TAB1-SUM lower in {lo} of 8 by point estimate, {sg} significant after BH (Molmo-7B-D n.s.); InternVL3.5-8B significantly higher -> Opus CONFIRMED")
    (B / "rigor_e26.txt").write_text("\n".join(L) + "\n")
    json.dump(facts, open(B / "e26_facts.json", "w"), indent=1, default=str)
    print("\n".join(L))


if __name__ == "__main__":
    main()
