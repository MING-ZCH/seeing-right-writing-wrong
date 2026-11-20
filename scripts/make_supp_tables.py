"""Generate supplementary LaTeX tables from the numbers bundle (no hand-copied numbers).

Writes into paper/tables/: supp_judge.tex, supp_twins.tex, supp_heldout.tex, supp_cotpbar.tex,
supp_closed.tex. Re-run after the bundle is rebuilt.
"""
from __future__ import annotations
import vtax_paths as _vtp  # noqa: E402

import json
import math
import re
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path

B = Path("outputs/bundle")
T = Path(f"{_vtp.TABLES}")


def tex_minus(s: str) -> str:
    """Typeset text-mode negative numbers (e.g. CI bounds) as $-$; math mode and ranges like 8--36 are left unchanged."""
    out, math = [], False
    for i, ch in enumerate(s):
        if ch == "$" and (i == 0 or s[i - 1] != "\\"):
            math = not math
        elif (not math and ch == "-" and s[i + 1:i + 2].isdigit()
              and (i == 0 or not (s[i - 1].isalnum() or s[i - 1] in "-{"))):
            out.append("$-$")
            continue
        out.append(ch)
    return "".join(out)


def wt(path: Path, s: str) -> None:
    path.write_text(tex_minus(s))
O = Path(f"{_vtp.OUTPUTS}")
NAMES = {"qwen3b": "Qwen2.5-VL-3B", "qwen7b": "Qwen2.5-VL-7B", "internvl3_8b": "InternVL3-8B", "phi35v": "Phi-3.5-V",
         "molmo7b": "Molmo-7B-D", "qwen3vl_8b": "Qwen3-VL-8B", "qwen3vl_8b_think": "Qwen3-VL-8B-Think.",
         "internvl35_8b": "InternVL3.5-8B"}
TASKS = [("cvb_relation", "2D Rel."), ("cvb_depth", "3D Depth"), ("cvb_distance", "3D Dist."),
         ("syn_relpos", "Syn. 2D"), ("syn_depth", "Syn. Depth")]


def wilson(k: int, n: int) -> tuple[float, float]:
    if n == 0:
        return float("nan"), float("nan")
    p, z = k / n, 1.96
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


def fmt_ci(d: dict | None) -> str:
    if not d or not d.get("n"):
        return "--"
    return f"${d['mean']:+.1f}$\\,{{\\scriptsize[{d['lo']:+.1f},{d['hi']:+.1f}]}}"


def pct0(d: dict | None) -> str:
    if not d or not d.get("n"):
        return "--"
    return f"${d['mean']:.0f}$\\,{{\\scriptsize[{d['lo']:.0f},{d['hi']:.0f}]}}"


def judge_table() -> None:
    rows = []
    for fn, default in (("e2_judge.jsonl", "claude-opus-5-5"), ("e2_judge2.jsonl", None)):
        p = O / fn
        if p.exists():
            rows += [json.loads(l) | {"jn": json.loads(l).get("judge_name") or default} for l in open(p)]
    groups = [g for g in ("D+C-", "D+C+", "D-C+") if any(r["group"] == g for r in rows)]
    judges = [j for j in ("claude-opus-5-5", "gpt-6-astra") if any(r["jn"] == j for r in rows)]
    jl = {"claude-opus-5-5": "Opus", "gpt-6-astra": "GPT"}
    gl = {"D+C-": "D+C$-$", "D+C+": "D+C$+$", "D-C+": "D$-$C$+$"}
    cs = json.load(open(B / "claim_strat.json")) if (B / "claim_strat.json").exists() else {}
    lines = [r"\begin{tabular}{ll" + "c" * len(groups) + "}", r"\toprule",
             "Model & Labeler & " + " & ".join(gl[g] for g in groups) + r" \\", r"\midrule"]
    for m in NAMES:
        if not any(r["model"] == m for r in rows):
            if m in cs:  # no judge sample: rule-based parse on all parsed items only
                cells = [pct0(cs[m].get(f"{g}|all", {}).get("rule_claim_wrong")) for g in groups]
                ns = [str(cs[m].get(f"{g}|all", {}).get("rule_claim_wrong", {}).get("n", "--")) for g in groups]
                lines.append(f"{NAMES[m]} & Rule (all) & " + " & ".join(cells) + r" \\")
                lines.append(r" & $n$ (all) & " + " & ".join(ns) + r" \\")
                lines.append(r"\addlinespace[2pt]")
            continue
        for k_j, j in enumerate(judges):
            cells = []
            for g in groups:
                s = [r for r in rows if r["model"] == m and r["group"] == g and r["jn"] == j and r["judge"] != "unclear"]
                if not s:
                    cells.append("--")
                    continue
                k = sum(r["judge"] == "incorrect" for r in s)
                lo, hi = wilson(k, len(s))
                cells.append(f"${100 * k / len(s):.0f}$\\,{{\\scriptsize[{100 * lo:.0f},{100 * hi:.0f}]}}")
            lines.append(f"{NAMES[m] if k_j == 0 else ''} & {jl[j]} & " + " & ".join(cells) + r" \\")
        j2 = json.load(open(B / "judge2.json"))["rates"] if (B / "judge2.json").exists() else {}
        cc = []
        for g in groups:
            r = j2.get(f"{m}|{g}")
            cc.append(f"${r['consensus_incorrect']:.0f}$\\,{{\\scriptsize[{r['consensus_ci'][0]:.0f},{r['consensus_ci'][1]:.0f}]}}"
                      if r and r.get("consensus_incorrect") is not None else "--")
        lines.append(r" & Consensus & " + " & ".join(cc) + r" \\")
        ns = [str(len({r["item_id"] for r in rows if r["model"] == m and r["group"] == g})) for g in groups]
        lines.append(r" & $n$ (judged) & " + " & ".join(ns) + r" \\")
        if m in cs:
            lines.append(r" & Rule (all) & " + " & ".join(pct0(cs[m].get(f"{g}|all", {}).get("rule_claim_wrong"))
                                                       for g in groups) + r" \\")
            lines.append(r" & $n$ (all) & " + " & ".join(str(cs[m].get(f"{g}|all", {}).get("rule_claim_wrong", {}).get("n", "--"))
                                                      for g in groups) + r" \\")
        lines.append(r"\addlinespace[2pt]")
    lines += [r"\bottomrule", r"\end{tabular}"]
    wt(T / "supp_judge.tex", "\n".join(lines) + "\n")


def twins_table() -> None:
    dp = json.load(open(B / "dprime_ci.json"))
    txt = (B / "polar_summary.txt").read_text().splitlines()
    acc = {}
    for line in txt:
        m = re.match(r"(\S+)\s+(direct|cot|desc)\s+\|\s+([\d.nan]+) / ([\d.nan]+) / ([\d.nan]+) \| ([\d.nan]+) / ([\d.nan]+)", line)
        if m:
            acc[(m.group(1), m.group(2))] = m.groups()[2:]
    lines = [r"\begin{tabular}{llccccc}", r"\toprule",
             r"Model & Cond. & Acc. front / behind & Twin flip front / behind & $d'$ [95\% CI] & $c$ & $\Delta d'$ (vs.\ Dir.) \\",
             r"\midrule"]
    for m in NAMES:
        if m not in dp:
            continue
        for cond in ("direct", "cot", "desc"):
            a = acc.get((m, cond))
            d = dp[m]
            if not a or "nan" in a or f"dprime_{cond}" not in d:
                continue
            dd = f"${d['dprime_' + cond]:.2f}$\\,{{\\scriptsize[{d['dprime_' + cond + '_ci'][0]:.2f},{d['dprime_' + cond + '_ci'][1]:.2f}]}}"
            dkey, ckey = {"cot": ("dprime_cot_minus_direct", "diff_ci"), "desc": ("dprime_desc_minus_direct", "desc_diff_ci")}.get(cond, (None, None))
            diff = f"${d[dkey]:+.2f}$\\,{{\\scriptsize[{d[ckey][0]:+.2f},{d[ckey][1]:+.2f}]}}" if dkey else ""
            name = NAMES[m] if cond == "direct" else ""
            label = {"direct": "Direct", "cot": "CoT", "desc": "Desc."}[cond]
            lines.append(f"{name} & {label} & {a[0]} / {a[1]} & {a[3]} / {a[4]} & {dd} & "
                         f"${d['c_' + cond]:+.2f}$ & {diff} \\\\")
        lines.append(r"\addlinespace[2pt]")
    lines += [r"\bottomrule", r"\end{tabular}"]
    wt(T / "supp_twins.tex", "\n".join(lines) + "\n")


def heldout_table() -> None:
    ev = json.load(open(B / "e6_eval.json"))
    tasks = ["blink_relation", "blink_depth", "3dsr_location_closer_to_camera", "3dsr_location_above",
             "3dsr_height_higher", "3dsr_orientation_in_front_of", "3dsr_orientation_on_the_left", "pooled"]
    short = ["BLINK Rel.", "BLINK Depth", "3DSR Closer", "3DSR Above", "3DSR Higher", "3DSR Front", "3DSR Left", "Pooled"]
    lines = [r"\begin{tabular}{l" + "c" * len(tasks) + "}", r"\toprule", "Model & " + " & ".join(short) + r" \\", r"\midrule"]
    for m in NAMES:
        if m not in ev:
            continue
        cells = []
        for t in tasks:
            r = ev[m].get(t)
            if not r:
                cells.append("--")
                continue
            d = float(r["direct"].split()[0])
            c = float(r["cot"].split()[0])
            cells.append(f"{d:.1f}/{c:.1f}")
        lines.append(f"{NAMES[m]} & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    wt(T / "supp_heldout.tex", "\n".join(lines) + "\n")


def cotpbar_table() -> None:
    nb = json.load(open(B / "numbers.json"))
    lines = [r"\begin{tabular}{l" + "c" * len(TASKS) + "}", r"\toprule", "Model & " + " & ".join(t for _, t in TASKS) + r" \\",
             r"\midrule"]
    for m in NAMES:
        if m == "molmo7b" or m not in nb:  # Molmo excluded from probability-based claims
            continue
        beh = nb[m].get("behavioral", {})
        cells = [fmt_ci(beh.get(f"task:{t}", {}).get("cot_minus_pbar")) for t, _ in TASKS]
        if all(c == "--" for c in cells):
            continue
        lines.append(f"{NAMES[m]} & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    wt(T / "supp_cotpbar.tex", "\n".join(lines) + "\n")


def closed_table() -> None:
    p = B / "e9x_summary.json"
    if not p.exists():
        return
    ev = json.load(open(p))
    disp = {"claude-opus-5-5": "Claude Opus 5.5", "gpt-6-astra": "GPT-6-Astra", "gemini-4-argon": "Gemini-4-Argon",
            "qwen3.8-max-0902": "Qwen3.8-Max", "deepseek-v4.1-flash": "DeepSeek-V4.1-Flash", "hy4": "HY4"}
    disp = {m: n for m, n in disp.items() if m != "hy4" or m in ev}  # HY4 row only once its runs are summarized
    cols = [("occ_both", "Occl. twins"), ("syn_depth", "Syn. Depth"), ("blink_depth", "BLINK Depth"),
            ("blink_relation", "BLINK Rel.")]
    lines = [r"\begin{tabular}{l" + "c" * (len(cols) + 1) + "}", r"\toprule",
             "Model & " + " & ".join(c for _, c in cols) + r" & $d'$ Dir./CoT \\", r"\midrule"]
    for m, name in disp.items():
        r = ev.get(m)
        if not r:
            lines.append(f"{name} & " + " & ".join(["--"] * (len(cols) + 1)) + r" \\")
            continue
        cells = []
        for k, _ in cols:
            v = r.get(k)
            if not v:
                cells.append("--")
            elif isinstance(v["direct"], dict):  # schema of analyze_e9x.py since 2026-11-13
                nr = f" [{v['direct_nr']['acc'][0]:.1f}]" if "direct_nr" in v else ""
                cells.append(f"{v['direct']['acc'][0]:.1f}{nr}/{v['cot']['acc'][0]:.1f} ($n$={v['direct']['n']})")
            else:
                cells.append(f"{v['direct'].split()[0]}/{v['cot'].split()[0]} ($n$={v['n']})")
        if "occ_sdt" in r:
            s = r["occ_sdt"]
            dfmt = lambda v: "ceiling" if v >= 6.0 else f"{v:.2f}"
            nr = f" [{dfmt(s['direct_nr']['dprime'])}]" if "direct_nr" in s else ""
            cells.append(f"{dfmt(s['direct']['dprime'])}{nr}/{dfmt(s['cot']['dprime'])}")
        else:
            sd, sc = r.get("occ_sdt_direct"), r.get("occ_sdt_cot")
            cells.append(f"{sd['dprime']:.2f}/{sc['dprime']:.2f}" if sd and sc else "--")
        lines.append(f"{name} & " + " & ".join(cells) + r" \\")
    lines.append(r"\bottomrule")
    if any("direct_nr" in (r.get("occ_sdt") or {}) for r in ev.values()):
        lines.append(rf"\multicolumn{{{len(cols) + 2}}}{{l}}{{\scriptsize Brackets: Direct at each API's lowest reasoning setting: reasoning\_effort=none for DeepSeek-V4.1-Flash and Qwen3.8-Max (0\% of calls reason),}} \\")
        lines.append(rf"\multicolumn{{{len(cols) + 2}}}{{l}}{{\scriptsize minimal for Claude Opus 5.5 (22.8\% of calls still produce reasoning tokens), low for GPT-6-Astra (0.3\%).}} \\")
    lines.append(r"\end{tabular}")
    wt(T / "supp_closed.tex", "\n".join(lines) + "\n")


def pct_ci(d: dict | None, sign: bool = False) -> str:
    if not d or not d.get("n"):
        return "--"
    f = "{:+.1f}" if sign else "{:.1f}"
    return f"${f.format(d['mean'])}$\\,{{\\scriptsize[{f.format(d['lo'])},{f.format(d['hi'])}]}}"


def flag_table() -> None:
    p = B / "flag_pr.json"
    if not p.exists():
        return
    fl = json.load(open(p))
    lines = [r"\begin{tabular}{llcccccc}", r"\toprule",
             r"Model & Pool & $n$ & CoT error & Flag rate & Precision & Recall & $P$(err$\mid$consistent) \\", r"\midrule"]
    for m in NAMES:
        for pool, lab in (("main", "Main"), ("heldout", "Held-out")):
            r = fl.get(f"{m}|{pool}")
            if not r:
                continue
            lines.append(f"{NAMES[m] if pool == 'main' else ''} & {lab} & {r['n']} & {pct_ci(r['cot_error_rate'])} & "
                         f"{pct_ci(r['flag_rate'])} & {pct_ci(r['precision'])} & {pct_ci(r['recall'])} & "
                         f"{pct_ci(r['p_err_given_consistent'])} \\\\")
        lines.append(r"\addlinespace[2pt]")
    lines += [r"\bottomrule", r"\end{tabular}"]
    wt(T / "supp_flag.tex", "\n".join(lines) + "\n")


def claim_edit_table() -> None:
    p = B / "claim_edit.json"
    if not p.exists():
        return
    ce = json.load(open(p))
    lines = [r"\begin{tabular}{llccccccc}", r"\toprule",
             r"Model & Group & $n$ & Original & Correct claim & Wrong claim & Neutral & Correct$-$Neutral & Wrong$-$Neutral \\",
             r"\midrule"]
    for m in NAMES:
        for grp in ("D+C-", "D+C+"):
            for mode in ("same", "closed"):
                r = ce.get(f"{m}|{mode}|all|{grp}|all")
                if not r:
                    continue
                g = grp.replace("-", "$-$").replace("+", "$+$") + ("" if mode == "same" else " (closed)")
                lines.append(f"{NAMES[m] if (grp, mode) == ('D+C-', 'same') else ''} & {g} & {r['n_items']} & "
                             f"{pct_ci(r['acc:orig'])} & {pct_ci(r['acc:correct'])} & {pct_ci(r['acc:wrong'])} & "
                             f"{pct_ci(r['acc:neutral'])} & {pct_ci(r['diff:correct-neutral'], True)} & "
                             f"{pct_ci(r['diff:wrong-neutral'], True)} \\\\")
        lines.append(r"\addlinespace[2pt]")
    lines += [r"\bottomrule", r"\end{tabular}"]
    wt(T / "supp_claim_edit.tex", "\n".join(lines) + "\n")


def commit_table() -> None:
    p = B / "commit_baseline.json"
    if not p.exists():
        return
    cb = json.load(open(p))
    lines = [r"\begin{tabular}{lccccc}", r"\toprule",
             r"Model & $n$ committed & Commit at rel.\ sent. & Rel.\ share of all sent. & Random-sentence null [95\%] & Obs.$-$null (perm.\ $p$) \\",
             r"\midrule"]
    for m in NAMES:
        r = cb.get(m)
        if not r:
            continue
        exc = r["commit_relational_share_of_committed"] - r["null_uniform_mean"]  # same rounded inputs as the null table
        pv = "$<10^{-4}$" if r["perm_p_uniform"] < 2e-4 else f"{r['perm_p_uniform']:.4f}"
        lines.append(f"{NAMES[m]} & {r['n_committed']}/{r['n_DCm']} & {r['commit_relational_share_of_committed']:.1f} & "
                     f"{r['sentence_relational_share_all_sentences']:.1f} & {r['null_uniform_mean']:.1f} "
                     f"{{\\scriptsize[{r['null_uniform_95'][0]:.1f},{r['null_uniform_95'][1]:.1f}]}} & ${exc:+.1f}$ ({pv}) \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    wt(T / "supp_commit.tex", "\n".join(lines) + "\n")


def bias_table() -> None:
    p = B / "bias_sensitivity.json"
    if not p.exists():
        return
    bs = json.load(open(p))
    lines = [r"\begin{tabular}{lcccccccc}", r"\toprule",
             r"Model & Acc. Dir./CoT & Hit Dir./CoT & FA Dir./CoT & $d'$ Dir./CoT & $c$ Dir./CoT & $\Delta d'$ [95\% CI] & $\Delta c$ [95\% CI] & \shortstack{Bal.-acc.\ loss:\\sens./crit.} \\",
             r"\midrule"]
    for m in NAMES:
        r = bs.get(m)
        if not r or "cot" not in r or "cot_vs_direct" not in r:
            continue
        d, c, v = r["direct"], r["cot"], r["cot_vs_direct"]
        lines.append(f"{NAMES[m]} & {d['acc']:.1f}/{c['acc']:.1f} & {d['hit']:.2f}/{c['hit']:.2f} & {d['fa']:.2f}/{c['fa']:.2f} & "
                     f"{d['dprime']:.2f}/{c['dprime']:.2f} & ${d['c']:+.2f}$/${c['c']:+.2f}$ & "
                     f"${v['d_dprime']:+.2f}$\\,{{\\scriptsize[{v['d_dprime_ci'][0]:+.2f},{v['d_dprime_ci'][1]:+.2f}]}} & "
                     f"${v['d_c']:+.2f}$\\,{{\\scriptsize[{v['d_c_ci'][0]:+.2f},{v['d_c_ci'][1]:+.2f}]}} & "
                     f"{v['loss_from_sensitivity']:.1f}/{v['loss_from_criterion']:.1f} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    wt(T / "supp_bias.tex", "\n".join(lines) + "\n")


def state_table() -> None:
    p = B / "state.json"
    if not p.exists():
        return
    st = json.load(open(p))
    dci = lambda d: f"${d['mean']:+.2f}$\\,{{\\scriptsize[{d['ci'][0]:+.2f},{d['ci'][1]:+.2f}]}}"
    lines = [r"\begin{tabular}{lcccccccc}", r"\toprule",
             r" & \multicolumn{4}{c}{Occlusion twins (pooled $d'$, both phrasings)} & \multicolumn{3}{c}{Synthetic Depth (acc.)} & State fields \\",
             r"\cmidrule(lr){2-5}\cmidrule(lr){6-8}\cmidrule(lr){9-9}",
             r"Model & Dir./CoT/State/Fill$_S$ & State$-$Dir. & State$-$CoT & Acc.\ Dir./CoT/State & Dir./CoT/State & State$-$Dir. & State$-$CoT & JSON / front true (tw./syn) \\",
             r"\midrule"]
    for m in NAMES:
        r = st.get(m)
        if not r:
            continue
        t, s, ps = r["twins"], r["syn_depth"], r["state_parse"]
        lines.append(
            f"{NAMES[m]} & {t['direct']['dprime']:.2f}/{t['cot']['dprime']:.2f}/{t['state']['dprime']:.2f}/{t['filler_S+turn']['dprime']:.2f} & "
            f"{dci(t['d_dprime:state-direct'])} & {dci(t['d_dprime:state-cot'])} & "
            f"{t['direct']['acc']:.1f}/{t['cot']['acc']:.1f}/{t['state']['acc']:.1f} & "
            f"{s['acc:direct']['mean']:.1f}/{s['acc:cot']['mean']:.1f}/{s['acc:state']['mean']:.1f} & "
            f"{pct_ci(s['diff:state-direct'], True)} & {pct_ci(s['diff:state-cot'], True)} & "
            f"{ps['all']['json_ok']:.0f} / {ps['twins']['front_true'] if ps['twins']['front_true'] is not None else '--'}/"
            f"{ps['syn']['front_true'] if ps['syn']['front_true'] is not None else '--'} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    wt(T / "supp_state.tex", "\n".join(lines) + "\n")


def natural_audit_table() -> None:
    p = B / "natural_audit.json"
    if not p.exists():
        return
    na = json.load(open(p))["rates"]
    cols = [("blink", "D+C-"), ("blink", "D+C+"), ("cvb", "D+C-"), ("cvb", "D+C+")]
    lines = [r"\begin{tabular}{llcccc}", r"\toprule",
             r" & & \multicolumn{2}{c}{BLINK Relative Depth} & \multicolumn{2}{c}{CV-Bench 3D Depth} \\",
             r"\cmidrule(lr){3-4}\cmidrule(lr){5-6}",
             r"Model & Labeler & D+C$-$ & D+C$+$ & D+C$-$ & D+C$+$ \\", r"\midrule"]
    cell = lambda r, j: (f"${r[j + '_incorrect']:.0f}$\\,{{\\scriptsize[{r[j + '_ci'][0]:.0f},{r[j + '_ci'][1]:.0f}]}}"
                         if r and r.get(j + "_incorrect") is not None else "--")
    for m in NAMES:
        rs = [na.get(f"{m}|{s}|{g}") for s, g in cols]
        if not any(rs):
            continue
        lines.append(f"{NAMES[m]} & Opus & " + " & ".join(cell(r, "opus") for r in rs) + r" \\")
        lines.append(r" & GPT & " + " & ".join(cell(r, "gpt") for r in rs) + r" \\")
        lines.append(r" & Consensus & " + " & ".join(cell(r, "consensus") for r in rs) + r" \\")
        lines.append(r" & $n$ & " + " & ".join(str(r["n"]) if r else "0" for r in rs) + r" \\")
        lines.append(r"\addlinespace[2pt]")
    lines += [r"\bottomrule", r"\end{tabular}"]
    wt(T / "supp_natural_audit.tex", "\n".join(lines) + "\n")


def cues_tables() -> None:
    p = B / "cues.json"
    if not p.exists():
        return
    cu = json.load(open(p))
    cells = ["front_larger|front_lower", "front_larger|front_higher", "front_smaller|front_lower", "front_smaller|front_higher"]
    dci = lambda d: f"${d['d']:+.2f}$\\,{{\\scriptsize[{d['ci'][0]:+.2f},{d['ci'][1]:+.2f}]}}"
    names = dict(NAMES) | {"api:deepseek-v4.1-flash": "DeepSeek-V4.1-Flash$^\\dagger$"}
    lines = [r"\begin{tabular}{lcccccc}", r"\toprule",
             r"Model & Direct $d'$ (LL/LH/SL/SH) & CoT$-$Dir.\ $d'$ (LL/LH/SL/SH) & Direct occl./vert./size & "
             r"CoT$-$Dir.\ occlusion & CoT$-$Dir.\ vertical & CoT$-$Dir.\ size \\", r"\midrule"]
    for m, name in names.items():
        r = cu.get(m, {}).get("cues")
        if not r or not r.get("decomposition") or not all(c in r["cells"] for c in cells):
            continue
        c, dc = r["cells"], r["decomposition"]
        lines.append(f"{name} & " + "/".join(f"{c[k]['direct']['dprime']:.2f}" for k in cells) + " & "
                     + "/".join(f"{c[k]['cot-direct']['dprime']:+.2f}" for k in cells) + " & "
                     + "/".join(f"{dc[x]['direct']['d']:+.2f}" for x in ("occlusion", "vertical", "size")) + " & "
                     + " & ".join(dci(dc[x]["cot-direct"]) for x in ("occlusion", "vertical", "size")) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    wt(T / "supp_cues.tex", "\n".join(lines) + "\n")
    # original twins + Synthetic Depth: vertical split and claim heuristics
    lines = [r"\begin{tabular}{llcccccc}", r"\toprule",
             r"Model & Set & Direct occl./vert. & CoT$-$Dir.\ occlusion & Desc.$-$Dir.\ occlusion & CoT$-$Dir.\ vertical & CoT claim: true/lower & State front: true/lower \\",
             r"\midrule"]
    for m in NAMES:
        for s, lab in (("twins", "Twins"), ("syn", "Syn. Depth")):
            r = cu.get(m, {}).get(s)
            if not r or not r.get("decomposition"):
                continue
            dc, cl = r["decomposition"], r.get("claims", {})
            ccl, scl = cl.get("cot", {}), cl.get("state", {})
            desc = dci(dc["occlusion"]["desc-direct"]) if "desc-direct" in dc["occlusion"] else "--"
            lines.append(f"{NAMES[m] if s == 'twins' else ''} & {lab} & {dc['occlusion']['direct']['d']:+.2f}/{dc['vertical']['direct']['d']:+.2f} & "
                         f"{dci(dc['occlusion']['cot-direct'])} & {desc} & {dci(dc['vertical']['cot-direct'])} & "
                         f"{ccl.get('names_true_front', '--')}/{ccl.get('names_lower', '--')} & "
                         f"{scl.get('names_true_front', '--')}/{scl.get('names_lower', '--')} \\\\")
        lines.append(r"\addlinespace[2pt]")
    lines += [r"\bottomrule", r"\end{tabular}"]
    wt(T / "supp_cues_twins.tex", "\n".join(lines) + "\n")


def _d(x: dict) -> str:
    return f"${x['mean']:+.1f}$\\,{{\\scriptsize[{x['lo']:+.1f},{x['hi']:+.1f}]}}" if isinstance(x, dict) and "mean" in x else "--"


def heldout_breakdown_tables() -> None:
    p = B / "heldout_breakdown.json"
    if not p.exists():
        return
    hb = json.load(open(p))
    names = dict(NAMES) | {"qwen3vl_8b_think": "Qwen3-VL-8B-Think$^\\ddagger$"}
    subs = [("pooled667", "667 (excl.\\ 3DSR left)"), ("depth324", "Depth-valid 324"), ("blink_depth", "BLINK Rel.\\ Depth 124")]
    lines = [r"\begin{tabular}{llcccccc}", r"\toprule",
             r"Model & Subset & Direct & CoT$-$Dir. & Sym$-$Dir. & Cons.$-$Dir. & Gate$-$Dir. & Flag P / R / base err. \\",
             r"\midrule"]
    for m, name in names.items():
        if m not in hb:
            continue
        for k, (s, lab) in enumerate(subs):
            x = hb[m][s]
            fl = x.get("flag")
            flag = f"{fl['precision']['mean']:.1f} / {fl['recall']['mean']:.1f} / {fl['base_error']['mean']:.1f}" if fl else "--"
            lines.append(f"{name if k == 0 else ''} & {lab} & {x['direct']['mean']:.1f} & {_d(x['cot_minus_direct'])} & "
                         f"{_d(x.get('sym_minus_direct'))} & {_d(x.get('cons_else_direct_minus_direct'))} & "
                         f"{_d(x.get('gate_minus_direct'))} & {flag} \\\\")
        lines.append(r"\addlinespace[2pt]")
    lines.append(r"\bottomrule")
    dup = json.load(open(B / "e27_reframe.json")).get("cons", {}).get("depth_valid_dup") if (B / "e27_reframe.json").exists() else None
    if dup:
        lines.append(rf"\multicolumn{{8}}{{p{{0.97\linewidth}}}}{{\scriptsize Qwen3-VL-8B and InternVL3.5-8B share the depth-valid CoT$-$Dir.\ and Sym$-$Dir.\ entries by coincidence: "
                     rf"equal net item counts (${dup['cot_minus_direct_net_items'][0]}$ and ${dup['sym_net_items'][0]}$ of ${dup['n']}$) with per-item differences on "
                     rf"{dup['items_with_different_cot_delta']} and {dup['items_with_different_sym_delta']} items; the seed-0 cluster bootstrap over the same {dup['n']} items "
                     rf"then gives intervals that agree to 0.1 point (unrounded upper bounds {dup['stored_unrounded']['qwen3vl_8b']['cot_minus_direct']['hi']:+.2f} vs.\ "
                     rf"{dup['stored_unrounded']['internvl35_8b']['cot_minus_direct']['hi']:+.2f}).}} \\")
    lines.append(r"\end{tabular}")
    wt(T / "supp_heldout_breakdown.tex", "\n".join(lines) + "\n")
    nat = hb.get("natural_sdt", {})
    # Unrounded CV-Bench 3D Delta d' (difference of unrounded d'), so this table matches Tabs. S2 and S5.
    el_ = json.load(open(B / "e17l.json")) if (B / "e17l.json").exists() else {}
    cvb_unr = el_.get("cvb", {}).get("all", {}).get("per_model_dprime_diff", {})
    lines = [r"\begin{tabular}{llcccc}", r"\toprule",
             r"Model & Benchmark & $n$ & $d'$ Direct$\to$CoT & $\Delta d'$ [95\% CI] & $c$ Direct$\to$CoT \\", r"\midrule"]
    for m, name in names.items():
        for k, (t, lab) in enumerate((("cvb_depth", "CV-Bench 3D Depth"), ("blink_depth", "BLINK Rel.\\ Depth"))):
            x = nat.get(m, {}).get(t)
            if not x:
                continue
            dd = cvb_unr.get(m, x["diff"]) if t == "cvb_depth" else x["diff"]
            lines.append(f"{name if k == 0 else ''} & {lab} & {x['n']} & {x['direct']['dprime']:.2f}$\\to${x['cot']['dprime']:.2f} & "
                         f"${dd:+.2f}$ [{x['diff_ci'][0]:+.2f}, {x['diff_ci'][1]:+.2f}] & "
                         f"${x['direct']['c']:+.2f}\\to{x['cot']['c']:+.2f}$ \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    wt(T / "supp_natural_sdt.tex", "\n".join(lines) + "\n")


def recheck_tables() -> None:
    ss, im = B / "swap_split.json", B / "imgswap.json"
    if not ss.exists() or not im.exists():
        return
    sw, ix = json.load(open(ss)), json.load(open(im))
    cells = ["D+C+", "D+C-", "D-C+", "D-C-"]
    lines = [r"\begin{tabular}{lccccccc}", r"\toprule",
             r" & \multicolumn{4}{c}{$S$ (R$_\mathrm{cut}$, original image) by source cell} & \multicolumn{3}{c}{Own R, image $v\to v_\mathrm{cf}$: $\Delta_\mathrm{vis}$} \\",
             r"\cmidrule(lr){2-5}\cmidrule(lr){6-8}",
             r"Model & D+C+ & D+C$-$ & D$-$C+ & D$-$C$-$ & full R, wrong claim & R$_\mathrm{cut}$, wrong claim & full R, all \\", r"\midrule"]
    for m, name in NAMES.items():
        if m not in sw and m not in ix:
            continue
        a = [_d(sw.get(m, {}).get(f"rcut:original:{c}", {}).get("S")) for c in cells]
        b = [_d(ix.get(m, {}).get(k, {}).get("dvis")) for k in ("full:wrongR", "cut:wrongR", "full:all")]
        lines.append(f"{name} & " + " & ".join(a + b) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    wt(T / "supp_recheck.tex", "\n".join(lines) + "\n")


def claim_uncond_table() -> None:
    p = B / "claim_edit_uncond.json"
    if not p.exists():
        return
    cu = json.load(open(p))
    lines = [r"\begin{tabular}{lcccccccc}", r"\toprule",
             r"Model & $n$ (D+C+/D+C$-$/D$-$C+/D$-$C$-$) & Direct & CoT & Original & Neutral & Correct & Neutral$-$Dir. & Correct$-$Dir. \\",
             r" & & \multicolumn{5}{c}{accuracy (\%) / $d'$} & \multicolumn{2}{c}{pts [95\% CI]} \\", r"\midrule"]
    for m, name in NAMES.items():
        r = cu.get(m, {}).get("all")
        if not r:
            continue
        c = r["cells"]
        cell = "/".join(str(c[k]) for k in ("D+C+", "D+C-", "D-C+", "D-C-"))
        vals = " & ".join(f"{r[k]['acc']['mean']:.1f} / {r[k]['dprime']:.2f}" for k in ("direct", "cot", "orig", "neutral", "correct"))
        star = "" if cu[m].get("complete_cells") else "$^\\ast$"
        lines.append(f"{name}{star} & {cell} & {vals} & {_d(r['neutral-direct'])} & {_d(r['correct-direct'])} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    wt(T / "supp_claim_uncond.tex", "\n".join(lines) + "\n")


def e17f_tables() -> None:
    """Round-4 tables: answer-step image sensitivity, cue-set claims, GLM, commit nulls, held-out format controls."""
    rs = json.load(open(B / "relation_sensitivity.json")) if (B / "relation_sensitivity.json").exists() else {}
    if rs:
        lines = [r"\begin{tabular}{lccccc}", r"\toprule",
                 r"Model & 2D left/right & 2D above/below & Depth & 3D distance & Occlusion twins \\",
                 r" & \multicolumn{4}{c}{Direct acc.\ / $S_\mathrm{orig}$ / $S_\mathrm{orig}-S_\mathrm{gray}$ (L/R: mirror $\Delta S$)} & Direct acc.\ / own cut-$R$ swap $\Delta_\mathrm{vis}$ \\", r"\midrule"]
        for m, name in NAMES.items():
            if m not in rs:
                continue
            cells = []
            for f in ("LR", "UD", "depth", "distance"):
                v = rs[m].get(f)
                if not v:
                    cells.append("--")
                    continue
                c = f"{v['direct_acc']['mean']:.1f} / {v['S_orig']['mean']:.1f} / ${v['dvis_gray']['mean']:+.1f}$"
                if f == "LR":
                    c += f" (${v['dS_mirror']['mean']:+.1f}$)"
                cells.append(c)
            o = rs[m]["occ"]
            cells.append(f"{o['direct_acc']['mean']:.1f} / {_d(o['dvis_cut_swap'])}")
            lines.append(f"{name} & " + " & ".join(cells) + r" \\")
        t = rs.get("_tests", {})
        mu_ = json.load(open(B / "multiplicity.json")) if (B / "multiplicity.json").exists() else {}
        cs = mu_.get("competence_slope")
        slope = (rf" Mixed-effects check (\cref{{tab:supp_multiplicity}}): model-random-intercept slope ${cs['mixedlm_slope_per_10pts'][0]:+.1f}$ pts per 10 accuracy points "
                 rf"[{cs['mixedlm_slope_per_10pts'][1][0]:+.1f}, {cs['mixedlm_slope_per_10pts'][1][1]:+.1f}] ($p = {cs['mixedlm_slope_per_10pts'][2]:.3f}$); "
                 rf"within-model fixed effects ${cs['within_model_slope_per_10pts'][0]:+.1f}$ [{cs['within_model_slope_per_10pts'][1][0]:+.1f}, {cs['within_model_slope_per_10pts'][1][1]:+.1f}].") if cs else ""
        lines += [r"\bottomrule",
                  rf"\multicolumn{{6}}{{p{{0.97\linewidth}}}}{{\scriptsize Spearman $\rho$(Direct acc., gray $\Delta$): {t['spearman_direct_acc_vs_gray_dvis_all'][0]:.2f} (28 model$\times$relation cells); twins $\rho$(Direct acc., cut-$R$ swap) = {t['spearman_twins_direct_acc_vs_cut_swap'][0]:.2f} (7 models).{slope}}} \\",
                  r"\end{tabular}"]
        wt(T / "supp_answer_step.tex", "\n".join(lines) + "\n")
    cc = json.load(open(B / "cue_claims.json")) if (B / "cue_claims.json").exists() else {}
    if (B / "parser_v2.json").exists():  # round-5 parser: symmetric verbs excluded, negation-aware (claim_parse v3)
        cc = json.load(open(B / "parser_v2.json"))["v3"]["cue_claims"]
    if cc:
        pc = lambda x: f"{x['pct']:.1f} [{x['ci'][0]:.1f}, {x['ci'][1]:.1f}]"
        bc = lambda x: f"${x['b']:+.2f}$ [{x['ci'][0]:+.2f}, {x['ci'][1]:+.2f}]"
        lines = [r"\begin{tabular}{lccccccc}", r"\toprule",
                 r" & \multicolumn{3}{c}{First claim names (\%, twin-pair 95\% CI)} & \multicolumn{3}{c}{Logit: claims subject in front} & ``on top'' \\",
                 r"\cmidrule(lr){2-4}\cmidrule(lr){5-7}",
                 r"Model & true front & lower & larger & subject front & subject lower & subject larger & share / names upper \\", r"\midrule"]
        for m, name in NAMES.items():
            r = cc.get("cues", {}).get(m)
            if not r:
                continue
            lg, ot = r["logit"], r["on_top"]
            lines.append(f"{name} & {pc(r['names_front'])} & {pc(r['names_lower'])} & {pc(r['names_larger'])} & {bc(lg['x_front'])} & "
                         f"{bc(lg['x_lower'])} & {bc(lg['x_larger'])} & {ot['share_of_claims']:.1f} / {ot['names_upper_if_on_top']:.1f} \\\\")
        lines += [r"\bottomrule", r"\end{tabular}"]
        wt(T / "supp_cue_claims.tex", "\n".join(lines) + "\n")
    gl = json.load(open(B / "glm_decomp.json")) if (B / "glm_decomp.json").exists() else {}
    if gl:
        dc = lambda x: f"${x['dprime']:+.2f}$ [{x['ci'][0]:+.2f}, {x['ci'][1]:+.2f}]"
        lines = [r"\begin{tabular}{lcccccc}", r"\toprule",
                 r"Model & Direct occl./vert./size & CoT$-$Dir.\ occlusion & CoT$-$Dir.\ vertical & CoT$-$Dir.\ size & CoT vertical & AUC Dir.$\to$CoT \\", r"\midrule"]
        for m, name in NAMES.items():
            g = gl.get("cues", {}).get(m)
            if not g:
                continue
            d0, dd = g["direct"], g["cot-direct"]
            lines.append(f"{name} & {d0['occlusion']['dprime']:+.2f}/{d0['vertical']['dprime']:+.2f}/{d0['size']['dprime']:+.2f} & "
                         f"{dc(dd['occlusion'])} & {dc(dd['vertical'])} & {dc(dd['size'])} & {dc(g['cot']['vertical'])} & "
                         f"{d0['auc']['auc']:.3f}$\\to${g['cot']['auc']['auc']:.3f} \\\\")
        if (B / "deepseek_glm.json").exists():
            ds = json.load(open(B / "deepseek_glm.json"))
            d0, dd = ds["direct"], ds["cot-direct"]
            lines.append(f"DeepSeek-V4.1-Flash$^\\dagger$ & {d0['occlusion']['dprime']:+.2f}/{d0['vertical']['dprime']:+.2f}/{d0['size']['dprime']:+.2f} & "
                         f"{dc(dd['occlusion'])} & {dc(dd['vertical'])} & {dc(dd['size'])} & {dc(ds['cot']['vertical'])} & -- \\\\")
        lines += [r"\bottomrule", r"\end{tabular}"]
        wt(T / "supp_glm.tex", "\n".join(lines) + "\n")
    cn = json.load(open(B / "commit_nulls.json")) if (B / "commit_nulls.json").exists() else {}
    if cn:
        lines = [r"\begin{tabular}{lcccc}", r"\toprule",
                 r"Model & Observed relational (\%) & Uniform null: excess & Non-concl.\ obs./null: excess & Position-matched null: excess \\", r"\midrule"]
        for m, name in NAMES.items():
            r = cn.get(m)
            if not r:
                continue
            lines.append(f"{name} & {r['obs_committed']:.1f} & {r['uniform']['null']:.1f}: ${r['uniform']['excess']:+.1f}$ & "
                         f"{r['nonconclusion']['obs']:.1f}/{r['nonconclusion']['null']:.1f}: ${r['nonconclusion']['excess']:+.1f}$ & "
                         f"{r['position_matched']['null']:.1f}: ${r['position_matched']['excess']:+.1f}$ \\\\")
        lines += [r"\bottomrule", r"\end{tabular}"]
        wt(T / "supp_commit_nulls.tex", "\n".join(lines) + "\n")
    hb = json.load(open(B / "heldout_breakdown.json")) if (B / "heldout_breakdown.json").exists() else {}
    if hb:
        lines = [r"\begin{tabular}{llcccccc}", r"\toprule",
                 r"Model & Subset & CoT$-$Dir. & CoT$-$F+t & F+t$-$Dir. & Prefix$-$Dir. & CoT$-$Prefix & Empty $R$ (\%) \\", r"\midrule"]
        for m, name in NAMES.items():
            for k, (sub, lab) in enumerate((("blink_depth", "BLINK Rel.\\ Depth"), ("depth324", "Depth-valid 324"))):
                x = hb.get(m, {}).get(sub, {})
                fm = x.get("format")
                if not fm:
                    continue
                lines.append(f"{name if k == 0 else ''} & {lab} & {_d(x['cot_minus_direct'])} & {_d(fm['cot_minus_filler_turn'])} & "
                             f"{_d(fm['filler_turn_minus_direct'])} & {_d(fm['prefix_only_minus_direct'])} & {_d(fm['cot_minus_prefix_only'])} & "
                             f"{fm['empty_cot_share']:.0f} \\\\")
        lines += [r"\bottomrule", r"\end{tabular}"]
        wt(T / "supp_heldout_format.tex", "\n".join(lines) + "\n")


def e17g_tables() -> None:
    """Gemini round-4 tables: queried-subject affirmation, Qwen3-VL ladder halves, held-out cross-model means."""
    af = json.load(open(B / "affirmation.json")) if (B / "affirmation.json").exists() else {}
    if (B / "parser_v2.json").exists():  # round-5 parser (claim_parse v3)
        af = json.load(open(B / "parser_v2.json"))["v3"]["affirmation"]
    if af:
        pc = lambda x: f"{x['mean']:.1f} [{x['ci'][0]:.1f}, {x['ci'][1]:.1f}]" if x else "--"
        lines = [r"\begin{tabular}{lccccccccc}", r"\toprule",
                 r" & \multicolumn{4}{c}{Cue-factorial set} & \multicolumn{3}{c}{Original twins} & \multicolumn{2}{c}{Synthetic Depth} \\",
                 r"\cmidrule(lr){2-5}\cmidrule(lr){6-8}\cmidrule(lr){9-10}",
                 r"Model & affirm (all) & affirm (D+C$-$) & strict & $\beta_\mathrm{affirm}$ / $\beta_\mathrm{truth}$ / $|\beta_\mathrm{lower}|$ & affirm (all) & affirm (D+C$-$) & yes D$\to$CoT & affirm (all) / strict & judge D+C$-$ \\",
                 r"\midrule"]
        for m, name in NAMES.items():
            c, t, y = af.get("cues", {}).get(m), af.get("twins", {}).get(m), af.get("syn", {}).get(m)
            if not c:
                continue
            j = (y or {}).get("judge", {}).get("e2_judge.jsonl", {}).get("D+C-")
            lines.append(f"{name} & {pc(c['all']['affirm'])} & {pc(c.get('DCm', {}).get('affirm'))} & {c['strict']['mean']:.1f} & "
                         f"${c['logit']['affirmation']['b']:+.2f}$ / ${c['logit']['x_front']['b']:+.2f}$ / {abs(c['logit']['x_lower']['b']):.2f} & "
                         f"{pc(t['all']['affirm'])} & {pc(t.get('DCm', {}).get('affirm'))} & {t['yes_rate']['direct']:.1f}$\\to${t['yes_rate']['cot']:.1f} & "
                         f"{y['all']['affirm']['mean']:.1f} / {y['strict']['mean']:.1f} & {f'{j[0]:.1f} ({j[1]})' if j else '--'} \\\\")
        lines += [r"\bottomrule", r"\end{tabular}"]
        wt(T / "supp_affirmation.tex", "\n".join(lines) + "\n")
    ld = json.load(open(B / "ladder_cues.json")) if (B / "ladder_cues.json").exists() else {}
    lu = json.load(open(B / "ladder_unified.json")) if (B / "ladder_unified.json").exists() else {}
    if ld and lu:
        dc = lambda x: f"${x['d']:+.2f}$\\,{{\\scriptsize[{x['ci'][0]:+.2f}, {x['ci'][1]:+.2f}]}}"
        de = lambda x, nd=2: f"${x['est']:+.{nd}f}$\\,{{\\scriptsize[{x['ci'][0]:+.{nd}f}, {x['ci'][1]:+.{nd}f}]}}"
        lines = [r"\begin{tabular}{lccccccc}", r"\toprule",
                 r"Qwen3-VL & Acc.\ D$\to$CoT & Acc.\ front-lower / higher, D$\to$CoT & Per-half occl.$-$size $\Delta d'$ & "
                 r"GLM occl.\ $\Delta d'$ & AUC D$\to$CoT ($\Delta$) & Pooled $d'$ D$\to$CoT ($\Delta$) & $d'_\mathrm{pair}$ D$\to$CoT ($\Delta$) \\",
                 r"\midrule"]
        runs = {"2b": "2B", "4b": "4B", "8b": "8B re-run$^a$", "8b_main_e2d": "8B main$^b$", "32b": "32B"}
        for sz in ("2b", "4b", "8b", "8b_main_e2d", "32b"):
            u = lu.get(sz)
            if not u:
                continue
            d, c, dd = u["direct"], u["cot"], u["cot-direct"]
            pooled = f"{d['dprime']:.2f}$\\to${c['dprime']:.2f} ({de(dd['dprime'])})"
            cap = "$^c$" if d["BW"] == 0 else ""
            pair = f"{d['dpair']:.2f}{cap}$\\to${c['dpair']:.2f} ({de(dd['dpair'])})"
            if sz in ld:
                h, g = ld[sz]["halves"], ld[sz]["glm"]
                halves = (f"{h['direct']['front_lower']['acc']:.1f}$\\to${h['cot']['front_lower']['acc']:.1f} / "
                          f"{h['direct']['front_higher']['acc']:.1f}$\\to${h['cot']['front_higher']['acc']:.1f}")
                glm = "separated" if g["direct"].get("separated") else f"${g['cot-direct']['occlusion']['dprime']:+.2f}$"
                a = g["cot-direct"]["auc"]
                auc = (f"{g['direct']['auc']['auc']:.3f}$\\to${g['cot']['auc']['auc']:.3f} "
                       f"(${a['diff']:+.3f}$\\,{{\\scriptsize[{a['ci'][0]:+.3f}, {a['ci'][1]:+.3f}]}})")
                half = dc(h["cot-direct"]["occ_minus_size"])
            else:
                halves = glm = auc = half = "--"
            lines.append(f"{runs[sz]} & {d['acc']:.1f}$\\to${c['acc']:.1f} & {halves} & {half} & {glm} & "
                         f"{auc} & {pooled} & {pair} \\\\")
        lines += [r"\bottomrule",
                  r"\multicolumn{8}{p{0.97\linewidth}}{\scriptsize Original twins ($800$ prompts per condition; both phrasings). Every row uses one run throughout. "
                  r"$^a$\texttt{Qwen3-VL-8B} re-run used for the ladder halves and AUC (and the gate inputs); "
                  r"$^b$main twin run of \cref{tab:supp_twins,tab:supp_twins_pair} (no per-half or AUC rows). "
                  r"Per-half $d'$: front-lower and front-higher twins scored separately; occl.$-$size = mean of the halves (occlusion and size are not separable on these twins). "
                  r"GLM: probit occlusion weight; the 32B \texttt{Direct} fit is quasi-separated. Pooled $d'$: subject-in-front over both phrasings. "
                  r"$d'_\mathrm{pair}$: $\ln[(n_\mathrm{BC}+0.5)/(n_\mathrm{BW}+0.5)]/1.702$ over $400$ scene--phrasing units; "
                  r"$^c$no both-wrong unit under \texttt{Direct}, so the value is a lower bound and its change is not interpretable. "
                  r"Brackets: $95\%$ bootstrap CIs (twin pairs for per-half and AUC; scenes for pooled $d'$ and $d'_\mathrm{pair}$).} \\",
                  r"\end{tabular}"]
        wt(T / "supp_ladder_cues.tex", "\n".join(lines) + "\n")
    hm = json.load(open(B / "heldout_mean.json")) if (B / "heldout_mean.json").exists() else {}
    if hm:
        lab = {"cot-direct|8 open": "CoT$-$Direct, 8 open models", "cot-direct|7 (no Phi)": "CoT$-$Direct, without Phi-3.5-Vision",
               "cot-direct|7 instruct": "CoT$-$Direct, 7 Instruct models", "cot-filler_turn|7": "CoT$-$Filler+turn, 7 models",
               "cot-filler_turn|6 (no Phi)": "CoT$-$Filler+turn, without Phi-3.5-Vision"}
        ex = json.load(open(B / "heldout_extra_means.json")) if (B / "heldout_extra_means.json").exists() else {}
        if (B / "e17j_natural.json").exists():  # one bootstrap (seed 0, 4,000 image-cluster resamples) shared with supp_natural_tasks
            nat = json.load(open(B / "e17j_natural.json"))
            hm = {"blink_depth": nat["blink_depth"], "depth324": nat["depth324"]}
            ex = {"blink_relation": nat["blink_relation"], "3dsr_depth": nat["3dsr_depth"]}
        cell = lambda a: f"${a['mean']:+.1f}$ [{a['ci'][0]:+.1f}, {a['ci'][1]:+.1f}]" if a else "--"
        lines = [r"\begin{tabular}{lcccc}", r"\toprule",
                 r"Cross-model mean paired $\Delta$ (pts) & BLINK Rel.\ Depth ($n=124$) & Depth-valid ($n=324$) & BLINK Spatial Rel.\ ($n=143$) & 3DSR depth ($n=200$) \\",
                 r"\midrule"]
        for k, l in lab.items():
            lines.append(f"{l} & {cell(hm['blink_depth'].get(k))} & {cell(hm['depth324'].get(k))} & "
                         f"{cell(ex.get('blink_relation', {}).get(k))} & {cell(ex.get('3dsr_depth', {}).get(k))} \\\\")
        lines += [r"\bottomrule", r"\end{tabular}"]
        wt(T / "supp_heldout_mean.tex", "\n".join(lines) + "\n")


def e17i_tables() -> None:
    """Round-5 tables: common-scale GLM, Synthetic Depth overlap reanalysis, cue-set judge parse."""
    cs = json.load(open(B / "common_scale.json")) if (B / "common_scale.json").exists() else {}
    if cs:
        def hu(x: float, f: str) -> str:
            """Decimal half-up rounding of the stored accuracy (4 decimals; e.g. 57.25 -> 57.3, -2.25 -> -2.3)."""
            nd = int(f[-3])
            v = Decimal(repr(x)).quantize(Decimal(1).scaleb(-nd), rounding=ROUND_HALF_UP)
            return f.format(v)

        c = lambda x, f="{:+.2f}": f"${f.format(x['est'])}$ [{f.format(x['ci'][0])}, {f.format(x['ci'][1])}]"
        lines = [r"\begin{tabular}{lccccccc}", r"\toprule",
                 r"Model & $s$ & $\delta_\mathrm{occl}$ & $\delta_\mathrm{vert}$ & $\delta_\mathrm{size}$ & retained occl./vert. & Acc.\ Dir.$\to$CoT & CoT$-$Dir.\ acc. \\",
                 r"\midrule"]
        for m, name in NAMES.items():
            r = cs.get(m)
            if not r:
                continue
            a = r["acc_cot_minus_direct"]
            lines.append(f"{name} & {c(r['s'], '{:.2f}')} & {c(r['delta_occlusion'])} & {c(r['delta_vertical'])} & {c(r['delta_size'])} & "
                         f"{r['retain_occlusion']['est']:.2f}/{r['retain_vertical']['est']:.2f} & {hu(r['acc_direct']['est'], '{:.1f}')}$\\to${hu(r['acc_cot']['est'], '{:.1f}')} & "
                         f"${hu(a['est'], '{:+.1f}')}$ [{a['ci'][0]:+.1f}, {a['ci'][1]:+.1f}] \\\\")
        lines += [r"\bottomrule", r"\end{tabular}"]
        wt(T / "supp_common_scale.tex", "\n".join(lines) + "\n")
    so = json.load(open(B / "syn_overlap.json")) if (B / "syn_overlap.json").exists() else {}
    if so:
        lines = [r"\begin{tabular}{lccccccc}", r"\toprule",
                 r" & \multicolumn{3}{c}{All items ($n=500$)} & \multicolumn{4}{c}{Visibly overlapping items ($n=" + str(so["counts"]["overlap"]) + r"$)} \\",
                 r"\cmidrule(lr){2-4}\cmidrule(lr){5-8}",
                 r"Model & Acc.\ Dir.$\to$CoT & $d'$ Dir.$\to$CoT & $\Delta d'$ & Acc.\ Dir.$\to$CoT & CoT$-$Dir.\ [95\% CI] & $d'$ Dir.$\to$CoT & $\Delta d'$ [95\% CI] \\",
                 r"\midrule"]
        for m, name in NAMES.items():
            r = so["models"].get(m)
            if not r:
                continue
            a, o = r["all"], r["overlap"]
            lines.append(f"{name} & {a['acc_direct']:.1f}$\\to${a['acc_cot']:.1f} & {a['dprime_direct']:.2f}$\\to${a['dprime_cot']:.2f} & ${a['dprime_diff']:+.2f}$ & "
                         f"{o['acc_direct']:.1f}$\\to${o['acc_cot']:.1f} & ${o['acc_diff']:+.1f}$ [{o['acc_diff_ci'][0]:+.1f}, {o['acc_diff_ci'][1]:+.1f}] & "
                         f"{o['dprime_direct']:.2f}$\\to${o['dprime_cot']:.2f} & ${o['dprime_diff']:+.2f}$ [{o['dprime_diff_ci'][0]:+.2f}, {o['dprime_diff_ci'][1]:+.2f}] \\\\")
        jr = so["judge_range_all"], so["judge_range_overlap"]
        lines += [r"\midrule",
                  rf"\multicolumn{{8}}{{l}}{{\scriptsize Two-judge consensus share of first claims judged incorrect: all items D+C$-$ {jr[0]['DCm'][0]:.1f}--{jr[0]['DCm'][1]:.1f}\% vs.\ D+C$+$ {jr[0]['DCp'][0]:.1f}--{jr[0]['DCp'][1]:.1f}\%; overlapping items {jr[1]['DCm'][0]:.1f}--{jr[1]['DCm'][1]:.1f}\% vs.\ {jr[1]['DCp'][0]:.1f}--{jr[1]['DCp'][1]:.1f}\%.}} \\",
                  rf"\multicolumn{{8}}{{l}}{{\scriptsize Image analysis: {so['counts']['overlap']} overlapping, {so['counts']['touch_or_gap']} touching or separated (gold follows size), {so['counts']['near_total']} near-total occlusion (one shape $<100$ visible px).}} \\",
                  r"\bottomrule", r"\end{tabular}"]
        wt(T / "supp_syn_overlap.tex", "\n".join(lines) + "\n")
    cj = json.load(open(B / "cue_judge.json")) if (B / "cue_judge.json").exists() else {}
    if cj:
        lines = [r"\begin{tabular}{lcccccc}", r"\toprule",
                 r"Labeler (cue set, $n=" + str(cj["n_consensus"]) + r"$ consensus) & agree (3-way) & $\kappa$ & agree when both claim & true front & lower & affirm \\",
                 r"\midrule"]
        js = cj["judge_shares_pooled"]
        lines.append(f"Judges (Opus vs.\\ GPT) & {cj['judge_vs_judge']['agree']:.3f} & {cj['judge_vs_judge']['kappa']:.2f} & -- & "
                     f"{js['true_front']:.1f} & {js['lower']:.1f} & {js['affirm']:.1f} \\\\")
        for v, lab in (("v1", "Rule v1 (overlap = front)"), ("v2", "Rule v2 (symmetric verbs excluded)"), ("v3", "Rule v3 (v2 + negation)")):
            r, sh = cj[f"rule_{v}_vs_consensus"], cj["rule_shares_same_sample"][v]
            lines.append(f"{lab} & {r['agree_3way']:.3f} & {r['kappa_3way']:.2f} & {r['agree_when_both_claim']:.3f} & {sh['true_front']:.1f} & {sh['lower']:.1f} & {sh['affirm']:.1f} \\\\")
        lines += [r"\midrule", r"Judge consensus by model & \multicolumn{6}{l}{true front / lower / larger / affirm (affirm on CoT-wrong items)} \\"]
        for m, name in NAMES.items():
            r = cj["judge_shares_by_model"].get(m)
            if r:
                lines.append(f"{name} ($n={r['n']}$) & \\multicolumn{{6}}{{l}}{{{r['true_front']:.1f} / {r['lower']:.1f} / {r['larger']:.1f} / {r['affirm']:.1f} ({r['affirm_cot_wrong']:.1f})}} \\\\")
        lines += [r"\bottomrule", r"\end{tabular}"]
        wt(T / "supp_cue_judge.tex", "\n".join(lines) + "\n")


def e17j_tables() -> None:
    """Round-6 tables: two-stage decomposition, judge-based affirmation, 3DSRBench categories, BLINK binding, steering."""
    ts = json.load(open(B / "e17j_twostage.json")) if (B / "e17j_twostage.json").exists() else {}
    if ts:
        g = lambda r, k, f="{:+.2f}": f"${f.format(r[k]['est'])}$" if k in r else "--"
        gc = lambda r, k: f"${r[k]['est']:+.2f}$ [{r[k]['ci'][0]:+.2f}, {r[k]['ci'][1]:+.2f}]"
        lines = [r"\begin{tabular}{lccccccccc}", r"\toprule",
                 r" & Direct & \multicolumn{3}{c}{CoT answer: occlusion $d'$ weight} & \multicolumn{2}{c}{Claim formation} & \multicolumn{3}{c}{Claim $\to$ answer} \\",
                 r"\cmidrule(lr){3-5}\cmidrule(lr){6-7}\cmidrule(lr){8-10}",
                 r"Model & occl./vert. & no text terms & + phrasing & + first-named & occl.\ / vert. & phrasing & follows claim & occl.\ beyond claim & vert.\ beyond claim \\", r"\midrule"]
        for m, name in NAMES.items():
            r = ts.get(m)
            if not r:
                continue
            lines.append(f"{name} & {r['direct.occ']['est']:+.2f}/{r['direct.vert']['est']:+.2f} & {g(r, 'cot_no_text.occ')} & {g(r, 'cot_phr.occ')} & {g(r, 'cot_phr_first.occ')} & "
                         f"{r['claim.occ']['est']:+.2f}/{r['claim.vert']['est']:+.2f} & {gc(r, 'claim.ph')} & {100 * r['follow']['est']:.1f}\\% & {gc(r, 'answer_given_claim.occ')} & {gc(r, 'answer_given_claim.vert')} \\\\")
        lines += [r"\midrule", r"Model & \multicolumn{2}{c}{Leave-one-cue-out $\delta$ ($d'$): occl.\ / vert.} & \multicolumn{3}{c}{Retained: occl.\ / vert.\ / ratio [95\% CI]} & \multicolumn{4}{c}{} \\", r"\midrule"]
        for m, name in NAMES.items():
            r = ts.get(m)
            if not r:
                continue
            lines.append(f"{name} & \\multicolumn{{2}}{{c}}{{{gc(r, 'joint_delta_occ')} / {gc(r, 'joint_delta_vert')}}} & \\multicolumn{{3}}{{c}}{{{r['retain_occ']['est']:.2f} / {r['retain_vert']['est']:.2f} / "
                         f"{r['retain_ratio']['est']:.2f} [{r['retain_ratio']['ci'][0]:.2f}, {r['retain_ratio']['ci'][1]:.2f}]}} & \\multicolumn{{4}}{{c}}{{}} \\\\")
        lines += [r"\bottomrule",
                  r"\multicolumn{10}{p{0.97\linewidth}}{\scriptsize Upper part: probit weights in $d'$ units ($2\beta$). ``No text terms'' omits the phrasing regressor; ``+ phrasing'' is the specification of \cref{tab:supp_glm}, so those values match it. "
                  r"Lower part: each $\delta$ comes from a separate joint fit of Direct and CoT answers in which CoT weights equal a common scale $s$ times the Direct weights and one cue (occlusion or vertical) is freed ($d'$ units). "
                  r"In Qwen3-VL-8B the vertical weight changes sign (Direct $+1.49$, CoT $-0.64$), which pulls $s$ toward zero when only occlusion is freed, so $\delta_\mathrm{occl}$ absorbs most of the CoT occlusion weight ($+2.83$).} \\",
                  r"\end{tabular}"]
        wt(T / "supp_twostage.tex", "\n".join(lines) + "\n")
    af = json.load(open(B / "e17j_affirm.json")) if (B / "e17j_affirm.json").exists() else {}
    if af:
        lines = [r"\begin{tabular}{lcccccc}", r"\toprule",
                 r"Model ($n$ consensus of 80) & $\kappa$ judges & $\kappa$ rule v2 / v3 & $\beta_\mathrm{affirm}$ judges & $\beta_\mathrm{affirm}$ rule v2 & $\beta_\mathrm{affirm}$ rule v3 & robust \\", r"\midrule"]
        bc = lambda x: f"${x['b_affirm']:+.2f}$ [{x['b_affirm_ci'][0]:+.2f}, {x['b_affirm_ci'][1]:+.2f}]" if x else "--"
        for m, name in NAMES.items():
            r = af.get(m)
            if not r:
                continue
            lines.append(f"{name} ({r['n_consensus']}) & {r['kappa_judges']:.2f} & {r['kappa_rule_v2']:.2f} / {r['kappa_rule_v3']:.2f} & {bc(r.get('judge'))} & {bc(r.get('v2'))} & "
                         f"{bc(r.get('v3'))} & {'yes' if r['robust_positive'] else 'no'} \\\\")
        lines += [r"\bottomrule", r"\end{tabular}"]
        wt(T / "supp_affirm_judge.tex", "\n".join(lines) + "\n")
    na = json.load(open(B / "e17j_natural.json")) if (B / "e17j_natural.json").exists() else {}
    if na:
        lab = {"blink_depth": "BLINK Relative Depth", "blink_relation": "BLINK Spatial Relation", "3dsr_location_closer_to_camera": "3DSR closer",
               "3dsr_orientation_in_front_of": "3DSR in front of", "3dsr_height_higher": "3DSR height", "3dsr_location_above": "3DSR above",
               "3dsr_orientation_on_the_left": "3DSR left"}
        cell = lambda a: f"${a['mean']:+.1f}$ [{a['ci'][0]:+.1f}, {a['ci'][1]:+.1f}]" if a else "--"
        lines = [r"\begin{tabular}{lcccc}", r"\toprule",
                 r"Held-out task (cross-model mean, pts) & CoT$-$Dir.\ (8 open) & CoT$-$Dir.\ (no Phi) & CoT$-$F+t (7) & CoT$-$F+t (no Phi) \\", r"\midrule"]
        for t, l in lab.items():
            r = na.get(t, {})
            lines.append(f"{l} ($n={r.get('cot-direct|8 open', {}).get('n_items', '--')}$) & {cell(r.get('cot-direct|8 open'))} & {cell(r.get('cot-direct|7 (no Phi)'))} & "
                         f"{cell(r.get('cot-filler_turn|7'))} & {cell(r.get('cot-filler_turn|6 (no Phi)'))} \\\\")
        lines += [r"\bottomrule", r"\end{tabular}"]
        wt(T / "supp_natural_tasks.tex", "\n".join(lines) + "\n")
    bb = json.load(open(B / "blink_binding.json")) if (B / "blink_binding.json").exists() else {}
    if bb:
        lines = [r"\begin{tabular}{llcccccc}", r"\toprule", r"Group & Labeler & $n$ & Correct & Swapped & Misnamed & None & Binding error (both judges) \\", r"\midrule"]
        for grp, gl in (("cot_wrong", "CoT wrong"), ("cot_right", "CoT right")):
            for j, jl in (("claude-opus-5-5", "Opus"), ("gpt-6-astra", "GPT"), ("consensus", "Consensus")):
                r = bb[f"{grp}|{j}"]
                be = f"{bb[f'{grp}|binding_error_both_judges']:.1f}" if j == "consensus" else ""
                lines.append(f"{gl if j == 'claude-opus-5-5' else ''} & {jl} & {r['n']} & {r['CORRECT']:.1f} & {r['SWAPPED']:.1f} & {r['MISNAMED']:.1f} & {r['NONE']:.1f} & {be} \\\\")
        lines += [r"\bottomrule",
                  rf"\multicolumn{{8}}{{p{{0.97\linewidth}}}}{{\scriptsize Consensus ($n = {bb['cot_wrong|consensus']['n']}$, {bb['cot_right|consensus']['n']}) requires both judges to assign the same 4-way label; "
                  rf"Binding error (both judges) is the share of all $n = {bb['cot_wrong|gpt-6-astra']['n']}$ / {bb['cot_right|gpt-6-astra']['n']} items where both judges flag Swapped or Misnamed; Fisher $p = 0.166$.}} \\",
                  r"\end{tabular}"]
        wt(T / "supp_blink_binding.tex", "\n".join(lines) + "\n")
    st = json.load(open(B / "steering.json")) if (B / "steering.json").exists() else {}
    if st:
        lines = [r"\begin{tabular}{lllccc}", r"\toprule", r"Model & Items & Direction, $k$ ($\alpha = k\sigma$) & $n$ & First claim correct [95\% CI] & Answer correct [95\% CI] \\", r"\midrule"]
        for m, rows in st.items():
            if m.startswith("_"):
                continue
            for r in rows:
                cc = lambda d: f"{d['mean']:.1f} [{d['ci'][0]:.1f}, {d['ci'][1]:.1f}]"
                grp = {"DCm_wrong": "D+C$-$ wrong claims", "DCp_right": "D+C$+$ right claims"}[r["group"]]
                kind = {"truth": "truth direction", "random": "random direction", "none": "no steering"}[r["kind"]]
                lay = {"qwen3vl_8b": 26, "molmo7b": 20}.get(m)
                lines.append(f"{NAMES.get(m, m)}, L{lay} & {grp} & {kind}, {r['k']:g} & {r['n']} & {cc(r['claim_correct'])} & {cc(r['answer_correct'])} \\\\")
        lines += [r"\bottomrule",
                  r"\multicolumn{6}{p{0.97\linewidth}}{\scriptsize Steering layer: Qwen3-VL-8B L26 (D+C$-$ wrong claims $n = 222$), Molmo-7B-D L20 ($n = 114$). Truth and random directions use the oracle (ground-truth) sign; "
                  r"the non-oracle variant is in \cref{tab:supp_steering_nonoracle}.} \\",
                  r"\end{tabular}"]
        wt(T / "supp_steering.tex", "\n".join(lines) + "\n")


def e17k_tables() -> None:
    """Round-7 tables: commit-sentence audit, loss locus, probe readout alignment, non-oracle steering."""
    cj = json.load(open(B / "commit_judge.json")) if (B / "commit_judge.json").exists() else {}
    if cj:
        rows = cj["rows"]
        cell = lambda r, k: f"{r[k + '_incorrect']:.0f}" if r.get(k + "_incorrect") is not None else "--"
        lines = [r"\begin{tabular}{lcccccccc}", r"\toprule",
                 r" & \multicolumn{4}{c}{D+C$-$ commit sentences $j^\ast$} & \multicolumn{3}{c}{D+C$+$ sentences, matched position} & \\",
                 r"\cmidrule(lr){2-5}\cmidrule(lr){6-8}",
                 r"Model & Opus / GPT & Consensus [95\% CI] & $j^\ast$ = first claim & cons.: first claim / later & Opus / GPT & Consensus [95\% CI] & no claim (GPT) & \\", r"\midrule"]
        for m, name in list(NAMES.items()) + [("pooled", "Pooled")]:
            a, b = rows.get(f"{m}|D+C-"), rows.get(f"{m}|D+C+")
            if not a or not b:
                continue
            fl = lambda v: "--" if v is None else f"{v:.0f}"
            lines.append(f"{name} & {cell(a, 'opus')} / {cell(a, 'gpt')} & {a['consensus_incorrect']:.0f} [{a['consensus_ci'][0]:.0f}, {a['consensus_ci'][1]:.0f}] & "
                         f"{a['is_first_claim']:.0f}\\% & {fl(a['commit_is_first_claim_consensus_incorrect'])} / {fl(a['commit_later_or_other_consensus_incorrect'])} & "
                         f"{cell(b, 'opus')} / {cell(b, 'gpt')} & {b['consensus_incorrect']:.0f} [{b['consensus_ci'][0]:.0f}, {b['consensus_ci'][1]:.0f}] & {b['gpt_noclaim']:.0f} & \\\\")
        lines += [r"\bottomrule", rf"\multicolumn{{9}}{{l}}{{\scriptsize Share of sentences judged to contradict the image (\%, among correct/incorrect labels); judge agreement {100 * cj['agree']:.1f}\%, $\kappa$ = {cj['kappa_clear']:.2f}.}} \\", r"\end{tabular}"]
        wt(T / "supp_commit_judge.tex", "\n".join(lines) + "\n")
    ek = json.load(open(B / "e17k.json")) if (B / "e17k.json").exists() else {}
    if ek:
        lines = [r"\begin{tabular}{llccccccccl}", r"\toprule",
                 r"Set & Model & Parse rate & Direct & First claim correct & Ans.\ $\mid$ claim right & Ans.\ $\mid$ claim wrong & CoT & claim-stage loss & post-claim loss & Locus \\", r"\midrule"]
        short = {"Direct near chance: no loss to localize": "Direct near chance", "little loss on this set": "little loss",
                 "loss after the first claim (later text or readout)": "after first claim", "loss at claim formation": "claim formation",
                 "loss both at the first claim and after it": "both"}
        for which, lab in (("cues", "Cue set"), ("twins", "Twins")):
            for m, name in NAMES.items():
                r = ek["locus"].get(which, {}).get(m)
                if not r:
                    continue
                lines.append(f"{lab if m == 'qwen3b' else ''} & {name} & {r['claim_rate']:.1f} & {r['direct_acc_claim_items']['mean']:.1f} & {r['p_claim_correct']['mean']:.1f} & {r['p_ans_given_claim_correct']['mean']:.1f} & "
                             f"{r['p_ans_given_claim_wrong']['mean']:.1f} & {r['cot_acc_claim_items']['mean']:.1f} & ${r['claim_stage_loss']:+.1f}$ & ${r['post_claim_loss']:+.1f}$ & {short[r['classification']]} \\\\")
            lines.append(r"\midrule")
        lines[-1] = r"\bottomrule"
        lines += [r"\multicolumn{11}{p{0.97\linewidth}}{\scriptsize Accuracies in \%, all computed on the items whose rationale contains a parsable first claim (claim\_parse v3; parse rate = share of items). Claim-stage loss = Direct $-$ first claim correct; post-claim loss = first claim correct $-$ CoT; they sum to Direct $-$ CoT.} \\", r"\end{tabular}"]
        wt(T / "supp_locus.tex", "\n".join(lines) + "\n")
    E17 = Path(f"{_vtp.OUTPUTS}/e17")
    al = {m: json.load(open(E17 / f"{m}_align.json")) for m in ("qwen3vl_8b", "molmo7b", "qwen7b") if (E17 / f"{m}_align.json").exists()}
    if al:
        c = lambda x: f"{x['mean']:.2f} [{x['ci'][0]:.2f}, {x['ci'][1]:.2f}]"
        lines = [r"\begin{tabular}{l" + "c" * len(al) + "}", r"\toprule", "Layer 20, cue set & " + " & ".join(NAMES[m] for m in al) + r" \\", r"\midrule",
                 r"\multicolumn{" + str(len(al) + 1) + r"}{l}{\textit{Decodability of the true relation (accuracy; within-claim: balanced)}} \\"]
        for k, lab in (("plain_d", "Probe at the Direct answer position"), ("plain_r", "Probe at the relation token $r$"),
                       ("within_r", "Within-claim probe at $r$"), ("plain_a", "Probe at the CoT answer token")):
            lines.append(lab + " & " + " & ".join(c(al[m]["decodability"][k]) for m in al) + r" \\")
        lines.append(r"\multicolumn{" + str(len(al) + 1) + r"}{l}{\textit{Behaviour and readout alignment (answer agrees with the probe's decoded relation)}} \\")
        lines.append("Direct / CoT accuracy & " + " & ".join(f"{al[m]['behaviour']['direct_acc']:.2f} / {al[m]['behaviour']['cot_acc']:.2f}" for m in al) + r" \\")
        for k, lab in (("direct_answer_vs_probe_d", "Direct answer vs probe at Direct position"), ("cot_answer_vs_plain_a", "CoT answer vs probe at CoT answer token")):
            lines.append(lab + " & " + " & ".join(c(al[m]["alignment"][k]) for m in al) + r" \\")
        lines.append(r"\multicolumn{" + str(len(al) + 1) + r"}{l}{\textit{Identity-swapped twin image, original-image probe at $r$ (share predicting the twin's relation)}} \\")
        for k, lab in (("DCm_wrong_all_follow_twin", "D+C$-$ wrong claims (twin relation = written claim)"),
                       ("right_claims_orig_correct_follow_twin", "Right claims, probe correct on original (twin contradicts text)")):
            lines.append(lab + " & " + " & ".join(c(al[m]["twin_conditional"][k]) for m in al) + r" \\")
        lines += [r"\bottomrule", r"\end{tabular}"]
        wt(T / "supp_probe_align.tex", "\n".join(lines) + "\n")
    sm = json.load(open(B / "steering_mixed.json")) if (B / "steering_mixed.json").exists() else {}
    if sm:
        lines = [r"\begin{tabular}{lllcccc}", r"\toprule",
                 r"Layer & Direction & Sign & CoT acc.\ (all, $n=300$) & D+C$-$ repaired & D+C$+$ kept & D$-$C$-$ / D$-$C$+$ acc. \\", r"\midrule"]
        cc = lambda d: f"{d['mean']:.1f} [{d['ci'][0]:.1f}, {d['ci'][1]:.1f}]" if d and d.get("ci") else "--"
        for r in sm["rows"]:
            lines.append(f"{r['layer']} & {r['kind']} & {r['sign']} & {cc(r['acc_all'])} & {cc(r['acc_DCm'])} & {cc(r['acc_DCp'])} & "
                         f"{r['acc_DmCm']['mean']:.1f} / {r['acc_DmCp']['mean']:.1f} \\\\")
        lines += [r"\bottomrule", r"\end{tabular}"]
        wt(T / "supp_steering_nonoracle.tex", "\n".join(lines) + "\n")


def e17l_tables() -> None:
    """Round-8 tables (overwrite the round-7 versions): commit audit with denominators/bounds, probe image share,
    steering dose-response, CV-Bench 3D Depth."""
    el = json.load(open(B / "e17l.json")) if (B / "e17l.json").exists() else {}
    cm = el.get("commit")
    cj = json.load(open(B / "commit_judge.json"))["rows"] if (B / "commit_judge.json").exists() else {}
    if cm:
        def wil(k, n):
            if n == 0:
                return [float("nan")] * 2
            z, p = 1.96, k / n
            d = 1 + z * z / n
            c = (p + z * z / (2 * n)) / d
            h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
            return [100 * (c - h), 100 * (c + h)]

        def contra(r, dec):
            k = round(r["consensus_incorrect"] / 100 * r["judgeable"])
            lo, hi = wil(k, r["judgeable"])
            f = f"{{:.{dec}f}}"
            return (f"{f.format(r['consensus_incorrect'])} [{f.format(lo)}, {f.format(hi)}] "
                    f"({f.format(r['lower_bound_unjudgeable_consistent'])}--{f.format(r['upper_bound_unjudgeable_contradicting'])})")
        lines = [r"\begin{tabular}{lcccccc}", r"\toprule",
                 r" & \multicolumn{4}{c}{D+C$-$ commit sentences $j^\ast$} & D+C$+$, matched position & D+C$+$, relational only \\",
                 r"\cmidrule(lr){2-5}\cmidrule(lr){6-6}\cmidrule(lr){7-7}",
                 r"Model & contradict [95\% CI] (bounds) & $j^\ast$ = first claim & first claim / later & no claim (\%, GPT) & contradict [95\% CI] (bounds) & contradict [95\% CI] (bounds) \\",
                 r"\midrule"]
        for m, name in list(NAMES.items()) + [("pooled", "Pooled")]:
            row = cm.get(m)
            if not row or "D+C-" not in row:
                continue
            dec = 1 if m == "pooled" else 0
            f = f"{{:.{dec}f}}"
            a_, b_, c_ = row["D+C-"], row.get("D+C+"), row.get("D+C+rel")
            k = cj.get(f"{m}|D+C-", {})
            fl = lambda v: "--" if v is None else f.format(v)
            nj = (lambda r: f"($n_\\mathrm{{j}}$ = {r['judgeable']}/{r['n']}) ") if m == "pooled" else (lambda r: f"{{\\scriptsize {r['judgeable']}/{r['n']}:}} ")
            if m == "pooled":
                lines.append(r"\midrule")
            lines.append(f"{name} ($n_\\mathrm{{j}}$ = {a_['judgeable']}/{a_['n']}) & {contra(a_, dec)} & {fl(k.get('is_first_claim'))} & "
                         f"{fl(k.get('commit_is_first_claim_consensus_incorrect'))} / {fl(k.get('commit_later_or_other_consensus_incorrect'))} & {f.format(a_['noclaim_gpt'])} & "
                         f"{nj(b_)}{contra(b_, dec)} & {(nj(c_) + contra(c_, dec)) if c_ else '--'} \\\\")
        lines += [r"\bottomrule",
                  r"\multicolumn{7}{p{0.97\linewidth}}{\scriptsize Contradict: share of judgeable sentences ($n_\mathrm{j}$: both judges CORRECT or both INCORRECT) that both judges mark INCORRECT, Wilson 95\% CI; bounds count unjudgeable sentences as consistent (lower) or contradicting (upper). "
                  r"$j^\ast$ = first claim: share of commits that are the audited first two-object claim; first claim / later: contradict share split by that. "
                  r"No claim (\%, GPT): share of sentences GPT-6-Astra labels NOCLAIM (no checkable spatial claim about the shapes). Relational only: D+C$+$ comparator restricted to relational sentences nearest the matched position.} \\",
                  r"\end{tabular}"]
        wt(T / "supp_commit_judge.tex", "\n".join(lines) + "\n")
    pi = el.get("probe_image")
    if pi:
        c = lambda x: f"{x['bal']:.2f} [{x['ci'][0]:.2f}, {x['ci'][1]:.2f}]"
        ms = [m for m in ("qwen3vl_8b", "molmo7b", "qwen7b") if m in pi]
        lines = [r"\begin{tabular}{l" + "c" * len(ms) + "}", r"\toprule", "Within-claim probe at $r$, layer 20 & " + " & ".join(NAMES[m] for m in ms) + r" \\", r"\midrule",
                 "Original image (balanced acc.) & " + " & ".join(c(pi[m]["orig_r"]) for m in ms) + r" \\",
                 "Gray image, probe refit & " + " & ".join(c(pi[m]["gray_refit_r"]) for m in ms) + r" \\",
                 "CoT text up to the relation word (TF-IDF) & " + " & ".join(c(pi[m]["text_tfidf"]) for m in ms) + r" \\",
                 "Conflict: right claim + twin image, follows image & " + " & ".join(f"{pi[m]['conflict_right_claim_twin_follow_image']['mean']:.2f} [{pi[m]['conflict_right_claim_twin_follow_image']['ci'][0]:.2f}, {pi[m]['conflict_right_claim_twin_follow_image']['ci'][1]:.2f}]" for m in ms) + r" \\",
                 "Image-carried share of above-chance decoding & " + " & ".join(f"{pi[m]['image_share_of_above_chance']:.2f}" for m in ms) + r" \\",
                 r"\midrule",
                 "Direct accuracy (cue set) & " + " & ".join(f"{pi[m]['direct_acc']:.2f}" for m in ms) + r" \\",
                 r"\bottomrule", r"\multicolumn{" + str(len(ms) + 1) + r"}{p{0.97\linewidth}}{\scriptsize Image-carried share = (original $-$ gray refit) / (original $-$ 0.5). Readout alignment (answer agrees with the Direct-position probe) equals Direct accuracy within 0.002 and is not reported.} \\",
                 r"\end{tabular}"]
        wt(T / "supp_probe_align.tex", "\n".join(lines) + "\n")
    sd = json.load(open(B / "steering_dose.json")) if (B / "steering_dose.json").exists() else {}
    if sd:
        cc = lambda d: f"{d['mean']:.1f}" if d and d.get("mean") is not None else "--"
        dc = lambda d: f"${d['mean']:+.1f}$ [{d['ci'][0]:+.1f}, {d['ci'][1]:+.1f}]" if d and d.get("ci") else "--"
        lines = [r"\begin{tabular}{llccccc}", r"\toprule",
                 r"$k$ & Sign & All (rnd) & D+C$-$ repaired (rnd) & D+C$+$ kept (rnd) & D$-$C$+$ & D+C$-$: steered $-$ random \\", r"\midrule"]
        n0 = sd["none"]
        lines.append(f"0 & no steering & {cc(n0['all'])} & {cc(n0['D+C-'])} & {cc(n0['D+C+'])} & {cc(n0['D-C+'])} & -- \\\\")
        lab = {"direct_answer": "Direct answer", "direct_probe": "Direct probe", "rprobe": "$r$ probe"}
        for row in sd["doses"]:
            for s in ("direct_answer", "direct_probe", "rprobe"):
                if s not in row:
                    continue
                t, r = row[s], row["random_mean"]
                lines.append(f"{row['k']} & {lab[s]} & {cc(t['all'])} ({cc(r['all'])}) & {cc(t['D+C-'])} ({cc(r['D+C-'])}) & {cc(t['D+C+'])} ({cc(r['D+C+'])}) & {cc(t['D-C+'])} & {dc(row[s + '_minus_random']['D+C-'])} \\\\")
        sf = el.get("steer_facts", {})
        da, dp, rp = sf.get("direct_answer_y", {}), sf.get("direct_probe_y", {}), sf.get("rprobe_y", {})
        lines += [r"\bottomrule",
                  rf"\multicolumn{{7}}{{p{{0.97\linewidth}}}}{{\scriptsize Qwen3-VL-8B, layer 26 ($\sigma = 1.06$, residual norm $\approx 259$; $k = 128$ is about half the norm), mixed 300 cue-set claim items "
                  rf"(D+C$-$ {sd['none']['D+C-']['n']}, D+C$+$ {sd['none']['D+C+']['n']}, D$-$C$+$ {sd['none']['D-C+']['n']}); random: mean of three random directions with the same sign. "
                  rf"Signs differ from the true label on {da.get('all', {}).get('differ', '--')}/300 items for the Direct answer (all Direct-wrong), {dp.get('all', {}).get('differ', '--')}/300 for the Direct-position probe, "
                  rf"and {rp.get('all', {}).get('differ', '--')}/300 for the within-claim $r$ probe; on D+C$-$/D+C$+$ the first two equal the oracle sign.}} \\\\",
                  r"\end{tabular}"]
        wt(T / "supp_steering_nonoracle.tex", "\n".join(lines) + "\n")
    cv = el.get("cvb")
    if cv:
        lines = [r"\begin{tabular}{lcccc}", r"\toprule", r"Model & CoT$-$Dir.\ acc.\ [95\% CI] & $\Delta d'$ & & \\", r"\midrule"]
        lines = [r"\begin{tabular}{lcc}", r"\toprule", r"CV-Bench 3D Depth ($n=" + str(cv["n_items"]) + r"$) & CoT$-$Dir.\ acc.\ [95\% CI] & $\Delta d'$ \\", r"\midrule"]
        for m, name in NAMES.items():
            r = cv["per_model"].get(m)
            if not r:
                continue
            star = "$^\\ast$" if r["sig_neg"] or r["sig_pos"] else ""
            lines.append(f"{name} & ${r['diff']:+.1f}$ [{r['ci'][0]:+.1f}, {r['ci'][1]:+.1f}]{star} & ${cv['all']['per_model_dprime_diff'].get(m, float('nan')):+.2f}$ \\\\")
        lines.append(r"\midrule")
        for k, lab in (("all", "Cross-model mean, 8 models"), ("no_phi", "Cross-model mean, without Phi-3.5-Vision")):
            r = cv[k]
            lines.append(f"{lab} & ${r['acc_diff']:+.1f}$ [{r['acc_ci'][0]:+.1f}, {r['acc_ci'][1]:+.1f}] & ${r['dprime_diff']:+.2f}$ [{r['dprime_ci'][0]:+.2f}, {r['dprime_ci'][1]:+.2f}] \\\\")
        lines += [r"\bottomrule", r"\multicolumn{3}{l}{\scriptsize $^\ast$ per-model 95\% image-cluster CI excludes 0. Cross-model means: image-cluster bootstrap with models fixed.} \\", r"\end{tabular}"]
        wt(T / "supp_cvb_depth.tex", "\n".join(lines) + "\n")


def e17m_tables() -> None:
    """Round-9 tables: probe image dependence by position (replaces the round-8 supp_probe_align), steering dose with
    D-C- and the anti-oracle r-probe test, and the probe-arbitrated fallback detector."""
    E17 = Path(f"{_vtp.OUTPUTS}/e17")
    m1 = {m: json.load(open(E17 / f"{m}_m1.json")) for m in ("qwen3vl_8b", "molmo7b", "qwen7b") if (E17 / f"{m}_m1.json").exists()}
    if m1:
        pos = [("pre", "$s{-}1$"), ("s", "$s$"), ("rprev", "$r{-}1$"), ("r", "$r$"), ("a", "answer")]
        lines = [r"\begin{tabular}{llccccccc}", r"\toprule",
                 r"Model & Position & Original bal. & Gray refit (image share) & Follows twin: pre / post swap & Conflict subset & $|p-0.5|$ conflict: flipped / kept & $|p-0.5|$ orig.\ / twin / gray & Twin refit bal. \\", r"\midrule"]
        for m, d in m1.items():
            for k, (p, lab) in enumerate(pos):
                v = d["positions"][p]
                lines.append(f"{NAMES[m] if k == 0 else ''} & {lab} & {v['orig_bal']:.2f} & {v['gray_refit_bal']:.2f} ({v['image_share_gray']:.2f}) & "
                             f"{v['pre_swap_follow_twin']:.2f} / {v['post_swap_follow_twin']:.2f} & {v['conflict_post_swap']:.2f} & {v['conf_twin_flipped']:.2f} / {v['conf_twin_kept']:.2f} & "
                             f"{v['conf_orig']:.2f} / {v['conf_twin']:.2f} / {v['conf_gray']:.2f} & {v['twin_refit_bal_twin_truth']:.2f} \\\\")
            lines.append(r"\addlinespace[2pt]")
        mu_ = json.load(open(B / "multiplicity.json")) if (B / "multiplicity.json").exists() else {}
        ts = mu_.get("twin_swap_image_share", {})
        tw = ("Normalized twin-swap image share at $r$ / answer, (post $-$ pre) / (1 $-$ 2 pre), on the same scale as the gray share: "
              + "; ".join(f"{NAMES[m]} {ts[m]['r']['twin_share_norm']:.2f} / {ts[m]['a']['twin_share_norm']:.2f}" for m in ts) + r" (\cref{tab:supp_multiplicity}). ") if ts else ""
        lines += [r"\bottomrule",
                  r"\multicolumn{9}{p{0.97\linewidth}}{\scriptsize Layer 20, cue set, within-claim probe (claim\_parse v3 strata, twin-pair CV). Image share = (original $-$ gray refit) / (original $-$ 0.5). "
                  r"Follows twin: share of claim items predicted as the twin's relation, by the original-trained probe on original states (pre swap; = 1 $-$ plain accuracy) and on twin states (post swap); a text-only probe would not move. "
                  + tw +
                  r"Conflict subset: right claims the probe gets right on the original image (text-only null $\approx 0$). Twin refit: probe refit on twin states with the twin's relation as label.} \\",
                  r"\end{tabular}"]
        wt(T / "supp_probe_align.tex", "\n".join(lines) + "\n")
    em = json.load(open(B / "e17m.json")) if (B / "e17m.json").exists() else {}
    det = em.get("detector")
    if det:
        c = lambda d: f"{d['mean']:.1f}" if d else "--"
        sg = lambda v: f"{v:+.1f}".replace("-", "$-$")
        dc = lambda d: f"${d['mean']:+.1f}$ [{sg(d['ci'][0])}, {sg(d['ci'][1])}]"
        lines = [r"\begin{tabular}{llccccc}", r"\toprule", r"Model & Rule & Accuracy & vs CoT & vs Direct & Flagged (\%) & flagged D+C$-$ / D$-$C$+$ \\", r"\midrule"]
        rl = {"A_claim_vs_probe": "A: claim $\\neq$ probe $\\to$ Direct", "B_disagree_plus_probe": "B: D$\\neq$C, probe arbitrates", "C_A_conf>0.25": "A with $|p-0.5|>0.25$"}
        for m, d in det.items():
            lines.append(f"{NAMES[m]} & Direct / CoT & {c(d['direct'])} / {c(d['cot'])} & & & & \\\\")
            for k, lab in rl.items():
                v = d[k]
                lines.append(f" & {lab} & {c(v['acc'])} & {dc(v['vs_cot'])} & {dc(v['vs_direct'])} & {v['flag_rate']:.1f} & {v['by_cell']['D+C-']['flagged']} / {v['by_cell']['D-C+']['flagged']} \\\\")
            lines.append(r"\addlinespace[2pt]")
        lines += [r"\bottomrule",
                  r"\multicolumn{7}{p{0.97\linewidth}}{\scriptsize Cue set, all prompts (items without a parsable claim keep the CoT answer); twin-pair bootstrap CIs. The probe is a supervised linear readout trained on labelled cue-set items (twin-pair CV); "
                  r"a probe at the Direct position used directly as the answer reaches 0.99 in-distribution, so these gains show that the decodable signal can be read out by a supervised probe, not a label-free method.} \\",
                  r"\end{tabular}"]
        wt(T / "supp_detector.tex", "\n".join(lines) + "\n")
    sd = json.load(open(B / "steering_dose.json")) if (B / "steering_dose.json").exists() else {}
    if sd:
        cc = lambda d: f"{d['mean']:.1f}" if d and d.get("mean") is not None else "--"
        sg = lambda v: f"{v:+.1f}".replace("-", "$-$")
        dc = lambda d: f"${d['mean']:+.1f}$ [{sg(d['ci'][0])}, {sg(d['ci'][1])}]" if d and d.get("ci") else "--"
        lines = [r"\begin{tabular}{llcccccc}", r"\toprule",
                 r"$k$ & Sign & All (rnd) & D+C$-$ (rnd) & D+C$+$ (rnd) & D$-$C$+$ (rnd) & D$-$C$-$ (rnd) & D+C$-$: steered $-$ random \\", r"\midrule"]
        n0 = sd["none"]
        lines.append(f"0 & no steering & {cc(n0['all'])} & {cc(n0['D+C-'])} & {cc(n0['D+C+'])} & {cc(n0['D-C+'])} & {cc(n0['D-C-'])} & -- \\\\")
        lab = {"direct_answer": "Direct answer", "direct_probe": "Direct probe", "rprobe": "$r$ probe"}
        for row in sd["doses"]:
            for s in ("direct_answer", "direct_probe", "rprobe"):
                if s not in row:
                    continue
                t, r = row[s], row["random_mean"]
                lines.append(f"{row['k']} & {lab[s]} & {cc(t['all'])} ({cc(r['all'])}) & {cc(t['D+C-'])} ({cc(r['D+C-'])}) & {cc(t['D+C+'])} ({cc(r['D+C+'])}) & "
                             f"{cc(t['D-C+'])} ({cc(r['D-C+'])}) & {cc(t['D-C-'])} ({cc(r['D-C-'])}) & {dc(row[s + '_minus_random']['D+C-'])} \\\\")
        rp = em.get("rp26")
        if rp:
            lines.append(r"\midrule")
            lines.append(rf"\multicolumn{{8}}{{l}}{{\textit{{Anti-oracle test: {rp['n']} D+C$-$ items of the 222-item set where the $r$ probe disagrees with the label (answer correct, \%)}}}} \\")
            for k in ("k16", "k128"):
                a = rp[k]["answer_correct"]
                lines.append(f"{k[1:]} & oracle / $r$ probe / random & \\multicolumn{{3}}{{c}}{{{cc(a['oracle'])} / {cc(a['rprobe'])} / {cc(a['random_mean'])} (none {cc(rp['none']['answer_correct'])})}} & "
                             f"\\multicolumn{{3}}{{c}}{{oracle $-$ $r$ probe {dc(a['oracle_minus_rprobe'])}}} \\\\")
        sf = json.load(open(B / "e17l.json")).get("steer_facts", {}) if (B / "e17l.json").exists() else {}
        da, dp, rpf = sf.get("direct_answer_y", {}), sf.get("direct_probe_y", {}), sf.get("rprobe_y", {})
        lines += [r"\bottomrule",
                  rf"\multicolumn{{8}}{{p{{0.97\linewidth}}}}{{\scriptsize Qwen3-VL-8B, layer 26 ($\sigma = 1.06$, residual norm $\approx 259$), mixed 300 cue-set claim items; random: mean of three random directions. "
                  rf"Signs differ from the true label on {da.get('all', {}).get('differ', '--')}/300 items for the Direct answer (all Direct-wrong), {dp.get('all', {}).get('differ', '--')}/300 for the Direct-position probe, "
                  rf"and {rpf.get('all', {}).get('differ', '--')}/300 for the within-claim $r$ probe.}} \\",
                  r"\end{tabular}"]
        wt(T / "supp_steering_nonoracle.tex", "\n".join(lines) + "\n")


def e17n_tables() -> None:
    """Round-10 tables: Synthetic Depth confound, cross-distribution arbitration, D-C+ rescue sentences."""
    en = json.load(open(B / "e17n.json")) if (B / "e17n.json").exists() else {}
    sy = en.get("syn")
    if sy:
        bsj = json.load(open(B / "bias_sensitivity.json")) if (B / "bias_sensitivity.json").exists() else {}
        for m, p in sy["pooled"].items():
            single = bsj.get(m, {}).get("cot_vs_direct", {}).get("loss_from_sensitivity")
            if single is not None:
                p["loss_from_sensitivity_single_phrasing"] = single
        sg = lambda v: f"{v:+.2f}".replace("-", "$-$")
        lines = [r"\begin{tabular}{lccccccc}", r"\toprule",
                 r" & \multicolumn{3}{c}{Both object orders pooled} & \multicolumn{3}{c}{CoT$-$Dir.\ acc.\ by visible-area-ratio tercile} & Ratio $\le$ 1.8 \\",
                 r"\cmidrule(lr){2-4}\cmidrule(lr){5-7}",
                 r"Model & $d'$ Dir.$\to$CoT ($\Delta$ [95\% CI]) & sens.\ loss pooled / single & both orders correct (\%) & low / mid / high & high-ratio ``no'': Dir.$\to$CoT (CoT ``yes'') & CoT$-$Dir.\ [95\% CI] \\", r"\midrule"]
        for m, name in NAMES.items():
            t = sy["tercile"].get(m)
            if not t:
                continue
            p = sy["pooled"].get(m)
            pc = (f"{p['dprime_direct']:.2f}$\\to${p['dprime_cot']:.2f} (${p['d_dprime']:+.2f}$ [{sg(p['d_dprime_ci'][0])}, {sg(p['d_dprime_ci'][1])}])" if p else "--")
            sl = f"{p['loss_from_sensitivity_pooled']:.1f} / {p['loss_from_sensitivity_single_phrasing']:.1f}".replace("-", "$-$") if p and p.get("loss_from_sensitivity_single_phrasing") is not None else "--"
            hn = t["high|no"]
            s_ = sy["small_ratio"][m]
            pb = f"{p['pair_both_correct_direct']:.1f}$\\to${p['pair_both_correct_cot']:.1f}" if p else "--"
            lines.append(f"{name} & {pc} & {sl} & {pb} & ${t['low|all']['loss']:+.1f}$ / ${t['mid|all']['loss']:+.1f}$ / ${t['high|all']['loss']:+.1f}$ & "
                         f"{hn['direct']:.0f}$\\to${hn['cot']:.0f} ({hn['cot_yes']:.0f}\\%) & ${s_['loss']:+.1f}$ [{s_['ci'][0]:+.1f}, {s_['ci'][1]:+.1f}] \\\\".replace("[-", "[$-$").replace(", -", ", $-$"))
        lines += [r"\bottomrule",
                  rf"\multicolumn{{8}}{{p{{0.97\linewidth}}}}{{\scriptsize Polarity-pooled $d'$: each item asked with the original and the object-order-swapped question (CoT regenerated for the swapped question); response ``yes'' = the queried subject is in front; image-cluster bootstrap (Qwen3-VL-8B-Thinking has no swapped run). "
                  rf"Sensitivity loss: $100[\Phi(d'_D/2)-\Phi(d'_C/2)]$, pooled vs single phrasing. Both orders correct: share of the $500$ items answered correctly under both object orders (a fixed yes/no bias cannot raise it, but a shift toward either answer lowers it). Terciles of the visible-area ratio of the two shapes (cut points {sy['ratio_terciles'][0]:.1f}, {sy['ratio_terciles'][1]:.1f}; {sy['share_ratio_gt_1.8']:.0f}\% of items exceed 1.8, the cue-set maximum); ratio $\le$ 1.8: $n = 55$.}} \\",
                  r"\end{tabular}"]
        wt(T / "supp_syndepth_confound.tex", "\n".join(lines) + "\n")
    E17 = B / "e17_probe"
    xd = {m: json.load(open(E17 / f"{m}_xdist.json")) for m in ("qwen3vl_8b", "molmo7b", "qwen7b") if (E17 / f"{m}_xdist.json").exists()}
    if xd:
        sg = lambda v: f"{v:+.1f}".replace("-", "$-$")
        dc = lambda d: f"${d['mean']:+.1f}$ [{sg(d['ci'][0])}, {sg(d['ci'][1])}]"
        lines = [r"\begin{tabular}{llcccccc}", r"\toprule",
                 r"Model & Set (frozen cue-set probe) & within-claim bal. & Direct / CoT & Rule B acc. & B vs CoT & B vs Direct & Direct-position probe as answer \\", r"\midrule"]
        for m, d in xd.items():
            for k, (st, lab) in enumerate((("twins", "Original twins"), ("syn", "Synthetic Depth"))):
                r = d[st]
                lines.append(f"{NAMES[m] if k == 0 else ''} & {lab} & {r['frozen_within_claim_bal']:.2f} & {r['direct']['mean']:.1f} / {r['cot']['mean']:.1f} & {r['B']['acc']['mean']:.1f} & "
                             f"{dc(r['B']['vs_cot'])} & {dc(r['B']['vs_direct'])} & {r['frozen_direct_probe_as_answer']['mean']:.1f} \\\\")
        lines += [r"\bottomrule",
                  r"\multicolumn{8}{p{0.97\linewidth}}{\scriptsize Probes trained on all cue-set claim items (layer 20; within-claim probe at $r$ per claim stratum; Direct-position probe at $d$) and applied unchanged. Rule B: when Direct and CoT disagree, keep the answer whose implied relation matches the probe. CV-Bench 3D Depth and the hard set have no comparable two-shape claims or stored CoT and are not tested.} \\",
                  r"\end{tabular}"]
        wt(T / "supp_detector_xdist.tex", "\n".join(lines) + "\n")
    rs, rj_ = en.get("rescue"), en.get("rescue_judge")
    if rs:
        lines = [r"\begin{tabular}{lcccccc}", r"\toprule",
                 r" & \multicolumn{3}{c}{D$-$C$+$: sentence where $P$(gold) rises to $\ge 0.5$} & \multicolumn{2}{c}{Judged against the image (Synthetic Depth)} & D+C$-$ commits \\",
                 r"\cmidrule(lr){2-4}\cmidrule(lr){5-6}",
                 r"Model & $n$ / committed & relational & conclusion & consistent / contradict ($n_\mathrm{j}$) & no claim & relational / contradict \\", r"\midrule"]
        for m, name in list(NAMES.items()) + [("pooled", "Pooled")]:
            a = rs.get(m, {}).get("D-C+") if m != "pooled" else None
            j = (rj_ or {}).get(m)
            if not a and not j:
                continue
            col1 = f"{a['n']} / {a['n_commit']} & {a['type_relation']:.0f} & {a['type_conclusion']:.0f}" if a else "-- & -- & --"
            col2 = f"{j['consistent_with_image']:.0f} / {j['contradict']:.0f} ({j['judgeable']}) & {j['noclaim_gpt']:.0f}" if j else "-- & --"
            dcm = j.get("DCm_commit_contradict") if j else None
            col3 = (f"{rs[m]['D+C-']['type_relation']:.0f}" if m != "pooled" else "--") + (f" / {dcm:.0f}" if dcm is not None else " / --")
            lines.append(f"{name} & {col1} & {col2} & {col3} \\\\")
        rc = json.load(open(B / "rescue_comparator.json"))["sets"] if (B / "rescue_comparator.json").exists() else {}
        if "rescue_DmCp" in rc:
            p = rc["rescue_DmCp"]["pooled"]
            nf = rc["rescue_DmCp"]["failed_calls"]
            ci_ = lambda x: f"${x['consistent']:.1f}\\%$ $[{x['consistent_ci'][0]:.1f}, {x['consistent_ci'][1]:.1f}]$"
            judge_txt = (f"pooled consistent {ci_(p)}; judge agreement ${p['agree_parsed']:.1f}\\%$ on the ${p['n_parsed_both']}$ sentences both judges parsed; "
                         f"{sum(rc['rescue_DmCp']['retried'].values())} Opus calls that failed with server errors were re-run, {sum(nf.values())} still fail, "
                         f"{sum(rc['rescue_DmCp']['unparsed_text_replies'].values())} replies have no label")
            cmp_txt = ""
            if "DmCm_rel_rescuepos" in rc and "DpCp_rel_rescuepos" in rc:
                a, b = rc["DmCm_rel_rescuepos"]["pooled"], rc["DpCp_rel_rescuepos"]["pooled"]
                pr = rc["rescue_DmCp"].get("pooled|rel", p)
                cmp_txt = (f" Comparators (same judges and prompt; relational sentences at the rescue sentences' relative positions, same $n$ per model): "
                           f"\\texttt{{D$-$C$-$}} {ci_(a)} consistent ($n_\\mathrm{{j}} = {a['judgeable']}$), \\texttt{{D+C+}} {ci_(b)} ($n_\\mathrm{{j}} = {b['judgeable']}$); "
                           f"relational rescue sentences only: {ci_(pr)} ($n_\\mathrm{{j}} = {pr['judgeable']}$).")
        else:
            judge_txt = r"pooled consistent $81.8\%$ $[75.0, 87.0]$; judge agreement $86.6\%$ on the $201$ sentences both judges parsed; $20$ Opus calls failed after retries"
            cmp_txt = ""
        lines += [r"\bottomrule",
                  r"\multicolumn{7}{p{0.97\linewidth}}{\scriptsize Mirror of the commit analysis for D$-$C$+$ items (all tasks): the rescue sentence is where the prefix readout of the gold option rises to $\ge 0.5$ and stays. Sentence types by rule (\%). "
                  rf"Image check: both judges, same protocol as the D+C$-$ commit audit, Synthetic Depth only (up to 40 per model, $221$ sentences); consistent / contradict = both judges agree, share of the $n_\mathrm{{j}}$ judgeable sentences ({judge_txt}).{cmp_txt} "
                  r"No claim: GPT judge. Last column: \texttt{D+C$-$} commits, relational share and consensus image-contradiction share (\cref{tab:supp_commit_judge}).} \\",
                  r"\end{tabular}"]
        wt(T / "supp_rescue.tex", "\n".join(lines) + "\n")


def wald_diff_txt(mu: dict, fams: list) -> str:
    """Name every test whose global BH survival differs between resampling and Wald p."""
    lab = dict(fams)
    diff = [t for t in mu["tests"] if t["survives_global_resampling"] != t["survives_global_wald"]]
    if not diff:
        return "identical"
    one = lambda t: (f"{NAMES[t['model']]} {lab.get(t['family'], t['family'])} survives only with resampling $p$ ($q = {t['q_global_resampling']:.3f}$ vs.\\ Wald $q = {t['q_global_wald']:.3f}$)"
                     if t["survives_global_resampling"] else
                     f"{NAMES[t['model']]} {lab.get(t['family'], t['family'])} survives only with Wald $p$ ($q = {t['q_global_wald']:.3f}$ vs.\\ resampling $q = {t['q_global_resampling']:.3f}$)")
    return "identical except: " + "; ".join(one(t) for t in diff)


def e17p_tables() -> None:
    """Round-12 table (new file, not \\input): Synthetic Depth pair-matched decomposition."""
    if not (B / "syn_pair_sdt.json").exists():
        return
    sp = json.load(open(B / "syn_pair_sdt.json"))
    P = sp["pairsdt"]
    sg = lambda v, p=1: f"{v:+.{p}f}"
    ci = lambda r, k, p=1: f"${sg(r[k], p)}$ [{sg(r[k + '_ci'][0], p)}, {sg(r[k + '_ci'][1], p)}]"
    sh = lambda r, arm: "/".join(f"{r[f'{arm}_{o}']:.1f}" for o in ("BC", "YY", "NN", "BW"))
    lines = [r"\begin{tabular}{lccccccccc}", r"\toprule",
             r" & \multicolumn{2}{c}{BC / YY / NN / BW (\%)} & \multicolumn{3}{c}{CoT$-$Dir.\ (pts, [95\% CI])} & \multicolumn{3}{c}{Sensitivity} & Two-order acc.\ $\Delta$ \\",
             r"\cmidrule(lr){2-3}\cmidrule(lr){4-6}\cmidrule(lr){7-9}",
             r"Model & Dir. & CoT & YY & NN & BW & $d'_\mathrm{pair}$ Dir.$\to$CoT ($\Delta$ [95\% CI]) & pooled $\Delta d'$ & ratio & = bias + reversal \\", r"\midrule"]
    for m, name in NAMES.items():
        r = P.get(m)
        if not r:
            continue
        a = r["acc_decomp"]
        rc = r["ratio_pooled_over_pair_ci"]
        ratio = f"{r['ratio_pooled_over_pair']:.2f}" + (f" [{rc[0]:.2f}, {rc[1]:.2f}]" if rc and rc[1] < 50 else (f" [{rc[0]:.2f}, $>$50]" if rc else ""))
        lines.append(f"{name} & {sh(r, 'direct')} & {sh(r, 'cot')} & {ci(r, 'delta_YY')} & {ci(r, 'delta_NN')} & {ci(r, 'delta_BW')} & "
                     f"{r['dpair_logit_direct']:.2f}$\\to${r['dpair_logit_cot']:.2f} ({ci(r, 'dpair_logit_delta', 2)}) & ${r['dpooled_delta']:+.2f}$ & {ratio} & "
                     f"${a['acc_delta']:+.1f}$ = ${a['from_bias_moves']:+.1f}$ ${a['from_reversals']:+.1f}$ \\\\")
    s = P["_summary"]
    terc = "; ".join(f"{NAMES[m]} {P[m]['tercile']['low']['dYY']:+.0f}/{P[m]['tercile']['mid']['dYY']:+.0f}/{P[m]['tercile']['high']['dYY']:+.0f} (BW {P[m]['tercile']['low']['dBW']:+.0f}/{P[m]['tercile']['mid']['dBW']:+.0f}/{P[m]['tercile']['high']['dBW']:+.0f})"
                     for m in ("qwen3b", "internvl3_8b", "qwen3vl_8b", "molmo7b"))
    mh = json.load(open(B / "syn_pair_sdt_mh.json"))["strat"] if (B / "syn_pair_sdt_mh.json").exists() else {}
    if mh:
        terc += (r". $d'_\mathrm{pair}$ by tercile, Dir.$\to$CoT: "
                 + "; ".join(f"{NAMES[m]} " + "/".join(f"{mh[m][f'direct_terc{k}']:.2f}$\\to${mh[m][f'cot_terc{k}']:.2f}" for k in range(3)) for m in mh if not m.startswith("_"))
                 + r" (Mantel--Haenszel pooling over size strata equals the pooled $d'_\mathrm{pair}$ for matched pairs; size-standardised estimates and CIs: \texttt{syn\_pair\_sdt\_mh.json})")
    lines += [r"\bottomrule",
              rf"\multicolumn{{10}}{{p{{0.97\linewidth}}}}{{\scriptsize Each of the $500$ \texttt{{Synthetic Depth}} images is asked under both object orders (\cref{{tab:supp_syndepth_confound}}); the two golds are opposite. BC: both orders correct; YY: ``yes'' to both (affirms both queried relations); NN: ``no'' to both; BW: both wrong (a consistent reversed percept). "
              rf"$d'_\mathrm{{pair}} = \ln[(n_\mathrm{{BC}}+0.5)/(n_\mathrm{{BW}}+0.5)]/1.702$: the pair-matched (conditional-logit) sensitivity; with $P(\text{{yes}}) = \sigma(b_i \pm \Delta/2)$ for image $i$, $n_\mathrm{{BC}}/n_\mathrm{{BW}}$ estimates $e^{{\Delta}}$ for any per-image bias $b_i$, including a bias toward affirming the queried relation; $1.702$ converts logit to probit units "
              rf"(assumes a common $\Delta$ across images and conditionally independent answers; the 2AFC conversion $\sqrt{{2}}\,\Phi^{{-1}}[n_\mathrm{{BC}}/(n_\mathrm{{BC}}+n_\mathrm{{BW}})]$ gives the same signs, \texttt{{syn\_pair\_sdt.json}}). Pooled $\Delta d'$: \cref{{tab:supp_syndepth_confound}}; ratio = pooled $\Delta d'$ / $\Delta d'_\mathrm{{pair}}$. "
              rf"Two-order accuracy $=$ BC $+$ (YY $+$ NN)/2, so its change splits exactly into bias moves $-(\Delta\mathrm{{YY}}+\Delta\mathrm{{NN}})/2$ and reversals $-\Delta\mathrm{{BW}}$ (descriptive; a sensitivity loss alone also moves BC into YY/NN). "
              rf"Pair bootstrap over images ($2{{,}}000$). $\Delta$YY by visible-area-ratio tercile (low/mid/high): {terc}.}} \\",
              r"\end{tabular}"]
    wt(T / "supp_syn_pairsdt.tex", "\n".join(lines) + "\n")


def e17o_tables() -> None:
    """Round-11 tables (new files, not \\input): churn, rescue comparators, multiplicity."""
    ch = json.load(open(B / "churn.json")) if (B / "churn.json").exists() else {}
    if ch:
        lines = [r"\begin{tabular}{lccccccccc}", r"\toprule",
                 r" & \multicolumn{7}{c}{2,850-item pool, \cref{M-tab:behavioral_main} readout} & \texttt{Synthetic Depth} ($n=500$) & Sec.~5 cells \\",
                 r"\cmidrule(lr){2-8}\cmidrule(lr){9-9}\cmidrule(lr){10-10}",
                 r"Model & D+C$-$ & D$-$C+ & D+C+ & D$-$C$-$ & net (pts) & Tab.~1 $\Delta$ & churn (\%) & D+C$-$ / D$-$C+ (churn \%, net) & D+C$-$ / D$-$C+ \\", r"\midrule"]
        for m, name in NAMES.items():
            r = ch.get(m)
            if not r:
                continue
            a, s, lg = r["all"]["tab1"], r["syn_depth"]["tab1"], r["all"]["legacy"]
            lines.append(f"{name} & {a['D+C-']} & {a['D-C+']} & {a['D+C+']} & {a['D-C-']} & ${a['net_pts']:+.1f}$ & ${r['tab1_cot_minus_direct']['mean']:+.1f}$ & {a['churn_pct']:.1f} & "
                         f"{s['D+C-']} / {s['D-C+']} ({s['churn_pct']:.1f}, ${s['net_pts']:+.1f}$) & {lg['D+C-']} / {lg['D-C+']} \\\\")
        diff = {m: ch[m]["all"]["items_cell_differs"] for m in NAMES if m in ch}
        ph = ch.get("phi35v", {}).get("all", {})
        lines += [r"\bottomrule",
                  r"\multicolumn{10}{p{0.97\linewidth}}{\scriptsize Cells cross \texttt{Direct} and \texttt{CoT} correctness on the same item. "
                  r"Tab.~1 readout: exact full-vocabulary option probabilities (vLLM V0 engine: Qwen2.5-VL-3B/7B, InternVL3-8B, Phi-3.5-Vision) or top-20 log-probabilities (V1 engine: Molmo-7B-D, Qwen3-VL-8B, -Thinking, InternVL3.5-8B); "
                  r"net = 100 (D$-$C+ $-$ D+C$-$) / 2,850 reproduces the \texttt{CoT}$-$\texttt{Direct} row of \cref{M-tab:behavioral_main} exactly; churn = 100 (D+C$-$ + D$-$C+) / 2,850. "
                  rf"Sec.~5 cells: the original runs' top-20 letter scores, which define the D$\pm$C$\pm$ groups of the error-onset, commit, rescue and steering analyses; they differ from the Tab.~1 cells on "
                  rf"{diff.get('qwen3b', 0)} / {diff.get('qwen7b', 0)} / {diff.get('internvl3_8b', 0)} / {diff.get('phi35v', 0)} items (Qwen2.5-VL-3B / 7B / InternVL3-8B / Phi-3.5-Vision; near-tie items) and on none for the other models. "
                  rf"Phi-3.5-Vision has {sum(ph.get('empty_R_items', {}).values())} items with an empty rationale ({ph.get('empty_R_items', {}).get('D+C-', 0)} D+C$-$, {ph.get('empty_R_items', {}).get('D-C+', 0)} D$-$C+), which the sentence-level rescue analysis drops (\cref{{tab:supp_rescue}}: $323$ D$-$C+).}} \\",
                  r"\end{tabular}"]
        wt(T / "supp_churn.tex", "\n".join(lines) + "\n")
    rc = json.load(open(B / "rescue_comparator.json")) if (B / "rescue_comparator.json").exists() else {}
    S = rc.get("sets", {})
    cols = [("rescue_DmCp", ""), ("rescue_DmCp", "|rel"), ("DmCm_rel_rescuepos", ""), ("DpCp_rel_rescuepos", ""), ("DpCp_rel_commitpos", "")]
    if all(c in S for c, _ in cols):
        cell_ = lambda x, ci=False: "--" if not x else (f"{x['consistent']:.1f} [{x['consistent_ci'][0]:.1f}, {x['consistent_ci'][1]:.1f}] ({x['judgeable']})" if ci else f"{x['consistent']:.0f} ({x['judgeable']})")
        lines = [r"\begin{tabular}{lccccc}", r"\toprule",
                 r" & \multicolumn{2}{c}{D$-$C+ rescue sentences} & \multicolumn{2}{c}{Relational, rescue-matched position} & D+C+ relational, \\",
                 r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}",
                 r"Model & all & relational & D$-$C$-$ & D+C+ & commit-matched (\cref{tab:supp_commit_judge}) \\", r"\midrule"]
        for m, name in list(NAMES.items()) + [("pooled", "Pooled")]:
            if m not in S["rescue_DmCp"]:
                continue
            lines.append(f"{name} & " + " & ".join(cell_(S[c].get(m + r), m == "pooled") for c, r in cols) + r" \\")
        ag = lambda c: f"{S[c]['pooled']['agree_parsed']:.1f}\\% ($\\kappa$ {S[c]['pooled']['kappa_clear']:.2f})"
        ct = rc.get("contrasts", {})
        fp = lambda p: f"p < 10^{{-4}}" if p < 1e-4 else f"p = {p:.3f}"
        fx = lambda k: f"${ct[k]['diff_pts']:+.1f}$ pts (${fp(ct[k]['fisher_p'])}$)" if k in ct else "--"
        lines += [r"\bottomrule",
                  r"\multicolumn{6}{p{0.97\linewidth}}{\scriptsize Share (\%) of judgeable sentences (both judges agree on CORRECT or INCORRECT; $n_\mathrm{j}$ in parentheses) that are consistent with the image; Wilson $95\%$ CIs for the pooled row. "
                  r"All sets are \texttt{Synthetic Depth}, same seven models, judges (Claude Opus 5.5, GPT-6-Astra), prompt and context window as \cref{tab:supp_commit_judge}. Rescue-matched sets draw, per model, the relative position of a random rescue sentence of that model and take the nearest relational sentence (rule-based type) of a random \texttt{D$-$C$-$} or \texttt{D+C+} rationale; $n$ per model equals the rescue sample where available "
                  rf"(\texttt{{D$-$C$-$}}: {S['DmCm_rel_rescuepos']['pooled']['n']} sentences, InternVL3-8B and Molmo-7B-D have only {S['DmCm_rel_rescuepos'].get('internvl3_8b', {}).get('n', '--')} and {S['DmCm_rel_rescuepos'].get('molmo7b', {}).get('n', '--')} eligible items; \texttt{{D+C+}}: {S['DpCp_rel_rescuepos']['pooled']['n']}). "
                  rf"Judge agreement on sentences both parsed: rescue {ag('rescue_DmCp')}, \texttt{{D$-$C$-$}} {ag('DmCm_rel_rescuepos')}, \texttt{{D+C+}} {ag('DpCp_rel_rescuepos')}. "
                  rf"Rescue $-$ \texttt{{D$-$C$-$}}: {fx('rescue_DmCp - DmCm_rel_rescuepos')} (relational only {fx('rescue_DmCp - DmCm_rel_rescuepos (relational only)')}); rescue $-$ \texttt{{D+C+}}: {fx('rescue_DmCp - DpCp_rel_rescuepos')} (relational only {fx('rescue_DmCp - DpCp_rel_rescuepos (relational only)')}); Fisher exact on pooled counts.}} \\",
                  r"\end{tabular}"]
        wt(T / "supp_rescue_cmp.tex", "\n".join(lines) + "\n")
    mu = json.load(open(B / "multiplicity.json")) if (B / "multiplicity.json").exists() else {}
    if mu:
        fams = [("tab1_cot_minus_direct", r"CoT$-$Dir.\ acc."), ("tab1_cot_minus_fillerturn", r"CoT$-$F$+$t acc."), ("syn_pooled_ddprime", r"Syn.\ Depth pooled $\Delta d'$"),
                ("syn_dpair_ddprime", r"Syn.\ Depth $\Delta d'_\mathrm{pair}$"),
                ("twin_ddprime", r"Twin $\Delta d'$"), ("cvb3d_ddprime", r"CV-B 3D $\Delta d'$"), ("cue_glm_occlusion", r"Cue occl.\ $\Delta$"), ("cue_glm_vertical", r"Cue vert.\ $\Delta$")]
        fams = [x for x in fams if x[0] in mu["families"]]
        T_ = {(t["family"], t["model"]): t for t in mu["tests"]}
        disp = {}
        gd = json.load(open(B / "glm_decomp.json")).get("cues", {}) if (B / "glm_decomp.json").exists() else {}
        for m_, g_ in gd.items():
            for fam_, comp_ in (("cue_glm_occlusion", "occlusion"), ("cue_glm_vertical", "vertical")):
                if comp_ in g_.get("cot-direct", {}):
                    disp[(fam_, m_)] = g_["cot-direct"][comp_]["dprime"]
        el_ = json.load(open(B / "e17l.json")) if (B / "e17l.json").exists() else {}
        for m_, v_ in el_.get("cvb", {}).get("all", {}).get("per_model_dprime_diff", {}).items():
            disp[("cvb3d_ddprime", m_)] = v_

        def c(t):
            if not t:
                return "--"
            # Display the value printed in the source table (computed from unrounded estimates), not the 3-decimal test estimate.
            src = disp.get((t["family"], t["model"]))
            v = f"{src:+.2f}" if src is not None else (f"{t['est']:+.1f}" if t["family"].startswith("tab1") else f"{t['est']:+.2f}")
            nom = t["ci"][0] > 0 or t["ci"][1] < 0
            if t["survives_global_resampling"]:
                return f"$\\mathbf{{{v}}}$"
            return f"${v}^{{\\circ}}$" if nom else f"${v}$"
        lines = [r"\begin{tabular}{l" + "c" * len(fams) + "}", r"\toprule", "Model & " + " & ".join(h for _, h in fams) + r" \\", r"\midrule"]
        for m, name in NAMES.items():
            lines.append(f"{name} & " + " & ".join(c(T_.get((f, m))) for f, _ in fams) + r" \\")
        lines.append(r"\midrule")
        F = mu["families"]
        lines.append(r"Survive BH / CI $\neq 0$ & " + " & ".join(f"{len(F[f]['survive_global_resampling'])} / {F[f]['nominal_ci_excludes_0']}" for f, _ in fams) + r" \\")
        cs = mu["competence_slope"]
        ts = mu["twin_swap_image_share"]
        tw = "; ".join(f"{NAMES[m]} {ts[m]['r']['twin_share_norm']:.2f} / {ts[m]['a']['twin_share_norm']:.2f} (gray {ts[m]['r']['gray_share']:.2f} / {ts[m]['a']['gray_share']:.2f})" for m in ts)
        lines += [r"\bottomrule",
                  rf"\multicolumn{{{len(fams) + 1}}}{{p{{0.97\linewidth}}}}{{\scriptsize Bold: survives Benjamini--Hochberg FDR $q < 0.05$ over all {mu['n_tests']} per-model tests jointly; $^\circ$: nominal $95\%$ CI excludes zero but does not survive; plain: neither. "
                  r"$p$-values: image-cluster sign-flip permutation ($2{,}000$) for the accuracy differences of \cref{M-tab:behavioral_main}; two-sided percentile bootstrap ($2{,}000$; GLM $1{,}000$) with the same resampling unit as the published CIs (image clusters; twin pairs) for $\Delta d'$ and the CoT$-$Direct probit-GLM components; survivor sets with Wald $p$ (bootstrap SE) are "
                  + wald_diff_txt(mu, fams) + r". "
                  rf"Competence slope (residual answer-step image use vs.\ \texttt{{Direct}} accuracy, {cs['n_cells']} model$\times$relation cells): pooled OLS ${cs['ols_slope_per_10pts'][0]:+.2f}$ pts per 10 accuracy points "
                  rf"[{cs['ols_slope_per_10pts'][1][0]:+.2f}, {cs['ols_slope_per_10pts'][1][1]:+.2f}]; model random intercept ${cs['mixedlm_slope_per_10pts'][0]:+.2f}$ [{cs['mixedlm_slope_per_10pts'][1][0]:+.2f}, {cs['mixedlm_slope_per_10pts'][1][1]:+.2f}] ($p = {cs['mixedlm_slope_per_10pts'][2]:.3f}$); "
                  rf"cluster bootstrap over models [{cs['cluster_bootstrap_over_models']['slope_per_10pts_ci'][0]:+.2f}, {cs['cluster_bootstrap_over_models']['slope_per_10pts_ci'][1]:+.2f}] (Spearman $\rho$ [{cs['cluster_bootstrap_over_models']['spearman_ci'][0]:+.2f}, {cs['cluster_bootstrap_over_models']['spearman_ci'][1]:+.2f}]). "
                  rf"Twin-swap image share at $r$ / answer, normalised like the gray share, (post $-$ pre) / (1 $-$ 2 pre): {tw}.}} \\",
                  r"\end{tabular}"]
        wt(T / "supp_multiplicity.tex", "\n".join(lines) + "\n")


def e17q_tables() -> None:
    """Round-13 tables (new files, not \\input): size-stratified d'_pair on Synthetic Depth; pair-matched twins."""
    if not (B / "syn_pair_sdt_mh.json").exists():
        return
    J = json.load(open(B / "syn_pair_sdt_mh.json"))
    S, TW = J["strat"], J["twins"]
    G = {}  # the random-slope GLMM is weakly identified (see legend); kept in the JSON only
    glmm_note = (r"A logistic GLMM with an image random intercept and random sensitivity slope is weakly identified here (two answers per image and arm; slope SD at its bound, bootstrap CIs spanning several $d'$ units) and is not reported (\texttt{syn\_pair\_sdt\_mh.json}). "
                 if "glmm" in J else "")
    ci = lambda r, k, p=2: f"${r[k]:+.{p}f}$ [{r[k + '_ci'][0]:+.{p}f}, {r[k + '_ci'][1]:+.{p}f}]"
    ar = lambda r, k: f"{r['direct_' + k]:.2f}$\\to${r['cot_' + k]:.2f}"
    hb = lambda h: f"{h['observed_disc_pct']:.1f}/{h['bound_pct']:.1f}" + ("$^{*}$" if h["exceeds"] else "")
    lines = [r"\begin{tabular}{lcccccccc" + ("c" if G else "") + "}", r"\toprule",
             r" & \multicolumn{2}{c}{BC$+$BW / bound (\%)} & \multicolumn{3}{c}{$\Delta d'_\mathrm{pair}$ [95\% CI]} & \multicolumn{3}{c}{$d'_\mathrm{pair}$ by size tercile, Dir.$\to$CoT ($\Delta$ [95\% CI])}"
             + (r" & GLMM \\" if G else r" \\"),
             r"\cmidrule(lr){2-3}\cmidrule(lr){4-6}\cmidrule(lr){7-9}",
             r"Model & Dir. & CoT & pooled $=$ MH & size-std.\ (terciles) & size-std.\ (quintiles) & low ($<3.2\times$) & mid & high ($\ge 8.4\times$)" + (r" & mean $\Delta d'$ \\" if G else r" \\"), r"\midrule"]
    for m, name in NAMES.items():
        r = S.get(m)
        if not r:
            continue
        h = r["hetero"]
        row = (f"{name} & {hb(h['direct']['all'])} & {hb(h['cot']['all'])} & {ci(r, 'delta_pooled_all')} & {ci(r, 'delta_std_terc')} & {ci(r, 'delta_std_quint')} & "
               + " & ".join(f"{ar(r, f'terc{k}')} ({ci(r, f'delta_terc{k}')})" for k in range(3)))
        if G:
            g = G.get(m)
            row += f" & {ci(g, 'dmean_delta')}" if g else " & --"
        lines.append(row + r" \\")
    GM = J.get("glmm", {})
    if GM and not G:
        lines += [r"\midrule",
                  r"\multicolumn{9}{@{}l}{\emph{Random-slope GLMM (exploratory, weakly identified): mean $d'$ Dir.\ / CoT; mean $\Delta d'$ [95\% CI]; image slope SD Dir.\ / CoT; "
                  r"$d'$ by size tercile, Dir.$\to$CoT ($\Delta$ [95\% CI])}} \\"]
        for m, name in NAMES.items():
            g = GM.get(m)
            if not g:
                continue
            lines.append(f"{name} & {g['dmean_direct']:.2f} & {g['dmean_cot']:.2f} & {ci(g, 'dmean_delta')} & "
                         rf"\multicolumn{{2}}{{c}}{{{g['slope_sd_direct']:.2f} / {g['slope_sd_cot']:.2f}}} & "
                         + " & ".join(f"{g[f'direct_terc{k}']:.2f}$\\to${g[f'cot_terc{k}']:.2f} ({ci(g, f'delta_terc{k}')})" for k in range(3)) + r" \\")
        fl = GM["_summary"]["fall_ci_lt0"]
        glmm_note = (rf"GLMM rows: logistic model with an image random intercept per arm and an image random sensitivity slope (one image effect, arm-specific SD), tercile-specific fixed bias and sensitivity per arm, "
                     rf"mean $d'$ $=$ tercile-averaged sensitivity$/1.702$; image bootstrap (${GM['n_boot']}$, cold-started). Its mean $\Delta d'$ has CI $<0$ in {['none', 'one', 'two', 'three', 'four', 'five', 'six', 'seven'][len(fl)]} of seven models ({', '.join(NAMES[m] for m in fl)}); no CI is $>0$. "
                     r"It is weakly identified with two answers per image and arm: the slope SD, the bias SD and the bias correlation reach their bounds in several fits or bootstrap CIs, the InternVL3-8B \texttt{Direct} high-tercile sensitivity sits at its bound, and $d'$ is on the image-conditional scale, so magnitudes are not comparable with $d'_\mathrm{pair}$ and depend on the normal random-effects assumption. "
                     r"It is a heterogeneity-robust check of sign only and is not in the multiplicity family. ")
    ss = S["_summary"]
    ncol = 10 if G else 9
    hi_chance = "; ".join(f"{NAMES[m]} {S[m]['cot_terc2']:.2f} [{S[m]['cot_terc2_ci'][0]:.2f}, {S[m]['cot_terc2_ci'][1]:.2f}] (top quintile {S[m]['cot_quint4']:.2f} [{S[m]['cot_quint4_ci'][0]:.2f}, {S[m]['cot_quint4_ci'][1]:.2f}])"
                          for m in ("qwen3b", "internvl3_8b", "qwen3vl_8b"))
    lines += [r"\bottomrule",
              rf"\multicolumn{{{ncol}}}{{p{{0.97\linewidth}}}}{{\scriptsize \texttt{{Synthetic Depth}}, both object orders per image (\cref{{tab:supp_syn_pairsdt}}). "
              r"BC$+$BW / bound: observed share of images answered differently under the two orders vs.\ its maximum under a common sensitivity, $100[\sigma(D/2)^2+\sigma(-D/2)^2]$ at the observed pair log-odds $D$; $^{*}$: excess CI $>0$ (image bootstrap), i.e.\ sensitivity varies across images. "
              r"Pooled $=$ MH: $d'_\mathrm{pair}$ of \cref{tab:supp_syn_pairsdt}; the Mantel--Haenszel common odds ratio over any grouping of matched pairs is $\sum n_\mathrm{BC}/\sum n_\mathrm{BW}$, so size-stratified MH equals the pooled estimate. "
              rf"Size-std.: image-share-weighted mean of within-stratum $\ln[(n_\mathrm{{BC}}+0.5)/(n_\mathrm{{BW}}+0.5)]/1.702$ over visible-area-ratio terciles or quintiles of the {S['qwen3b']['n_visible']} images with both shapes visible, weights fixed across arms, so \texttt{{CoT}} cannot lower it by moving discordant images between size strata. "
              + "Direct high-tercile values with $n_\\mathrm{BW}=0$ are the $+0.5$ cap (a lower bound): "
              + ", ".join(f"{NAMES[m]} {S[m]['direct_terc2']:.2f}" for m in NAMES if m in S and S[m]["tercile_counts"]["direct"][2][3] == 0) + ". "
              + (r"GLMM: logistic, image random intercept per arm and image random sensitivity slope, tercile-specific fixed effects; weakly identified (two answers per image and arm), exploratory. " if G else glmm_note)
              + rf"Falls with CI $<0$: pooled {len(ss['pooled_all']['fall_ci_lt0'])}/7, size-std.\ terciles {len(ss['std_terc']['fall_ci_lt0'])}/7, quintiles {len(ss['std_quint']['fall_ci_lt0'])}/7; by tercile {len(ss['terc0']['fall_ci_lt0'])}/{len(ss['terc1']['fall_ci_lt0'])}/{len(ss['terc2']['fall_ci_lt0'])}. "
              rf"\texttt{{CoT}} $d'_\mathrm{{pair}}$ in the high tercile: {hi_chance}. Image bootstrap ($2{{,}}000$).}} \\",
              r"\end{tabular}"]
    wt(T / "supp_syn_pairmh.tex", "\n".join(lines) + "\n")
    sh = lambda r, a: "/".join(f"{x:.1f}" for x in r[f"split_{a}"])
    lines = [r"\begin{tabular}{lcccccccc}", r"\toprule",
             r" & \multicolumn{2}{c}{BC / SF / SB / BW (\%)} & \multicolumn{2}{c}{$d'_\mathrm{pair}$} & \multicolumn{2}{c}{$d'_\mathrm{pair}$ Dir.$\to$CoT by vertical cell} & \multicolumn{2}{c}{CoT$-$Dir.\ [95\% CI]} \\",
             r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(lr){6-7}\cmidrule(lr){8-9}",
             r"Model & Dir. & CoT & Dir.$\to$CoT ($\Delta$ [95\% CI]) & cue-cell std.\ $\Delta$ & front lower & front higher & vertical-balanced & vertical comp. \\", r"\midrule"]
    for m, name in NAMES.items():
        r = TW.get(m)
        if not r:
            continue
        lines.append(f"{name} & {sh(r, 'direct')} & {sh(r, 'cot')} & {ar(r, 'pooled')} ({ci(r, 'delta_pooled')}) & {ci(r, 'delta_std_cells')} & "
                     f"{r['direct_lower']:.2f}$\\to${r['cot_lower']:.2f} & ${r['direct_higher']:+.2f}$$\\to$${r['cot_higher']:+.2f}$ & {ci(r, 'delta_vert_balanced')} & {ci(r, 'delta_vert_component')} \\\\")
    ts = TW["_summary"]
    nm = lambda ms: ", ".join(NAMES[m] for m in ms) or "none"
    lines += [r"\bottomrule",
              r"\multicolumn{9}{p{0.97\linewidth}}{\scriptsize Original occlusion twins (\cref{tab:supp_twins}). Unit: a scene and one phrasing (``in front of'' / ``behind''), asked on the original image and its identity-swapped twin, whose truth for ``the subject is in front'' is opposite ($400$ units per model and arm). "
              r"BC: both images correct; SF / SB: subject called front / behind in both (no discrimination between the twins); BW: both wrong. $d'_\mathrm{pair} = \ln[(n_\mathrm{BC}+0.5)/(n_\mathrm{BW}+0.5)]/1.702$; the Mantel--Haenszel estimate over the $2\times2$ cue cells (vertical: front drawn lower / higher; drawn area ratio $<$ / $\ge 0.77$) equals it, and cue-cell std.\ weights within-cell log-odds by cell share. "
              r"Front lower: the lower-is-closer cue agrees with occlusion in both twins; front higher: it predicts the reverse in both. Vertical-balanced $=$ mean of the two vertical cells (the vertical cue cancels under an additive logit); vertical comp.\ $=$ half their difference. "
              r"In Qwen2.5-VL-7B and InternVL3.5-8B the front-lower cells have $n_\mathrm{BW}=0$ under both arms, so their $d'_\mathrm{pair}$ sits at the $+0.5$ cap and the vertical-balanced change reflects only fewer reversals in front-higher cells. "
              rf"Twin-pair bootstrap ($2{{,}}000$). CI $<0$ / $>0$: pooled {nm(ts['pooled']['fall'])} / {nm(ts['pooled']['rise'])}; vertical-balanced {nm(ts['vert_balanced']['fall'])} / {nm(ts['vert_balanced']['rise'])}; marginal $\Delta d'$ (\cref{{tab:supp_twins}}) {nm(ts['marginal']['fall'])} / {nm(ts['marginal']['rise'])}.}} \\",
              r"\end{tabular}"]
    wt(T / "supp_twins_pair.tex", "\n".join(lines) + "\n")


if __name__ == "__main__":
    judge_table()
    twins_table()
    heldout_table()
    cotpbar_table()
    closed_table()
    flag_table()
    claim_edit_table()
    bias_table()
    commit_table()
    state_table()
    natural_audit_table()
    cues_tables()
    heldout_breakdown_tables()
    recheck_tables()
    claim_uncond_table()
    e17f_tables()
    e17g_tables()
    e17i_tables()
    e17j_tables()
    e17k_tables()
    e17l_tables()
    e17m_tables()
    e17n_tables()
    e17o_tables()
    e17p_tables()
    e17q_tables()
    print("wrote supp tables to", T)
