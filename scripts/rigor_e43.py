"""Exp-43: >= 30B non-Qwen3 open VLMs on the cue-factorial set (CPU, stored outputs).

Models: InternVL3.5-38B (internvl35_38b; vLLM 0.11 V1, top-20 readout, max_dynamic_patch 6, exactly as internvl35_8b)
and Qwen2.5-VL-32B-Instruct (qwen32b; vLLM 0.8.4 V0, exact full-vocabulary letter scores, max_pixels 1280*28*28,
exactly as qwen3b / qwen7b). Both: e10_state.py --set cues (assistant prefix "Let's think step by step.", greedy,
<= 512 tokens, forced readout at "Answer: ("), VX_TP=2, VX_GPU_UTIL=0.6, then e28_run.py --parts desc.

Every statistic is rigor_e31.analyze unchanged (competence rule, probit GLM + common scale, AME, d'_pair, cells,
lower-is-closer share, v3 first claims, last-claim follow 2x2, rigor_e32 two-channel x / y); only the model list,
the twin-file lookup (no twin runs for the new models) and the table header / note differ. Reference rows
(internvl35_8b, qwen7b) are re-analysed as a reproduction check against Tab. 2 / S18-S20 / cue_ame.json.
-> outputs/bundle/e43_big_open.json, outputs/bundle/rigor_e43.txt, ../paper/tables/supp_e43.tex
Usage: python rigor_e43.py [--nb-glm 1000] [--models ...] [--table-only]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import rigor_e31 as R31
from rigor_e31 import O, B, PAPER_TAB, s_, u_, ci_, star

NEW = ["internvl35_38b", "qwen32b"]
REF = ["internvl35_8b", "qwen7b"]
R31.NAMES.update({"internvl35_38b": "InternVL3.5-38B", "qwen32b": "Qwen2.5-VL-32B",
                  "internvl35_8b": "InternVL3.5-8B", "qwen7b": "Qwen2.5-VL-7B"})
_twins_orig, _l3_orig = R31.twins_dpair, R31.l3_twins
E31_MODELS = ("qwen3vl_32b", "qwen3vl_8b_think", "qwen3vl_8b")


def twins_dpair(m: str) -> dict | None:
    """rigor_e31.twins_dpair with the source generalised to outputs/e2d/<m>{,_behind}.jsonl (None if no twin run)."""
    if m in E31_MODELS:
        return _twins_orig(m)
    if not (O / "e2d" / f"{m}.jsonl").exists():
        return None
    import numpy as np
    from collections import defaultdict
    from build_bundle import pe, rj
    from rigor_e17q import twin_stats
    from rigor_e28 import q95
    U = {c: defaultdict(dict) for c in ("direct", "cot")}
    for ph, fn_ in (("front", f"{m}.jsonl"), ("behind", f"{m}_behind.jsonl")):
        for r in rj(O / "e2d" / fn_):
            if r["condition"] not in U:
                continue
            p, mt = pe(r), r["meta"]
            u = U[r["condition"]][(r["image_id"], ph)]
            says = (p == "A") if ph == "front" else (p == "B")
            u["cf" if r["item_id"].endswith("_cf") else "orig"] = (bool(says), mt["front_name"] == mt["x_name"], p == "A")
            if not r["item_id"].endswith("_cf"):
                u["above"], u["ratio"] = bool(mt["front_above"]), float(mt["area_ratio_front_over_back"])
    Ud, Uc = U["direct"], U["cot"]
    keys = sorted(k for k in Ud if k in Uc and {"orig", "cf"} <= set(Ud[k]) and {"orig", "cf"} <= set(Uc[k]))
    if not keys:
        return None

    def oc_of(UU):
        o = []
        for k in keys:
            a, b = UU[k]["orig"], UU[k]["cf"]
            ca, cb = a[0] == a[1], b[0] == b[1]
            o.append(0 if ca and cb else 3 if not (ca or cb) else 1 if a[0] and b[0] else 2)
        return np.array(o)

    rng = np.random.default_rng(232)
    oc_d, oc_c = oc_of(Ud), oc_of(Uc)
    vert = np.array([int(Ud[k]["above"]) for k in keys])
    cell = 2 * vert + np.array([int(Ud[k]["ratio"] >= 0.77) for k in keys])
    wc = np.bincount(cell, minlength=4) / len(cell)
    scene = np.array([k[0] for k in keys])
    us = np.unique(scene)
    sidx = {s: np.where(scene == s)[0] for s in us}
    est = twin_stats(oc_d, oc_c, cell, wc, vert)
    bs = np.array([twin_stats(oc_d[i], oc_c[i], cell[i], wc, vert[i])
                   for i in (np.concatenate([sidx[s] for s in rng.choice(us, len(us))]) for _ in range(R31.NB))])
    n_ = len(est) // 3
    return {"n_units": len(keys), "direct": round(float(est[0]), 3), "cot": round(float(est[n_]), 3),
            "delta": {"d": round(float(est[2 * n_]), 3), "ci": q95(bs[:, 2 * n_], 3)},
            "bw_direct": int((oc_d == 3).sum()), "bw_cot": int((oc_c == 3).sum())}


def l3_twins(m: str) -> list[dict]:
    """rigor_e32 twin items where an e2d twin run exists (none for the Exp-43 models)."""
    if m in E31_MODELS:
        return _l3_orig(m)
    if not (O / "e2d" / f"{m}.jsonl").exists():
        return []
    from collections import defaultdict
    from build_bundle import pe, rj
    out = []
    for suf, ph in (("", "front"), ("_behind", "behind")):
        Rr = defaultdict(dict)
        for r in rj(O / "e2d" / f"{m}{suf}.jsonl"):
            if r["condition"] in ("direct", "cot"):
                Rr[r["item_id"]][r["condition"]] = r
        for iid, d in Rr.items():
            if "cot" not in d or "direct" not in d or not d["cot"].get("text"):
                continue
            c, mt = d["cot"], d["cot"]["meta"]
            out.append(R31.r32_rec(m, "twins", f"{iid}|{ph}", iid.replace("_cf", ""), c["gold"], pe(d["direct"]), pe(c),
                                   R31.r32_body(c["text"]), mt["front_name"], mt["back_name"], mt["x_name"], ph))
    return out


R31.twins_dpair = twins_dpair
R31.l3_twins = l3_twins

HEAD = {"internvl35_38b": r"\shortstack{InternVL3.5\\38B}", "qwen32b": r"\shortstack{Qwen2.5-VL\\32B}",
        "internvl35_8b": r"\shortstack{InternVL3.5\\8B}", "qwen7b": r"\shortstack{Qwen2.5-VL\\7B}"}


def table(res: dict, ms: list[str]) -> str:
    """supp_e31 layout (rigor_e31.table) for the Exp-43 models."""
    ms = [m for m in ms if m in res]
    row = lambda label, cells: label + " & " + " & ".join(cells) + r" \\"
    nc = len(ms) + 1
    L = [r"\begin{tabular}{@{}l" + "c" * len(ms) + "@{}}", r"\toprule",
         row(r"\textbf{Model}", [r"\textbf{" + HEAD[m] + "}" for m in ms]), r"\midrule",
         rf"\multicolumn{{{nc}}}{{@{{}}l@{{}}}}{{\textit{{\textbf{{Who reads occlusion? Cue-factorial set, \texttt{{Direct}}}}}}}} \\"]
    L.append(row("Accuracy (\\%)", [u_(res[m]["acc_direct_comp"]["mean"]) + ci_(*res[m]["acc_direct_comp"]["ci"], nd=1, signed=False) for m in ms]))
    L.append(row("Occlusion weight $d'$", [u_(res[m]["glm"]["direct"]["occlusion"]["d"], 2) + ci_(*res[m]["glm"]["direct"]["occlusion"]["ci"], signed=False)
                                           + (r"$^\S$" if res[m]["glm"]["direct"]["separated"] else "") for m in ms]))
    L.append(row("Follows lower-is-closer (\\%)", [u_(res[m]["lower"]["direct"]["agree_lower"]["mean"]) for m in ms]))
    L.append(row("Occlusion-competent", [r"\textbf{yes}" if res[m]["competent"] else "no" for m in ms]))
    L.append(r"\midrule")
    L.append(rf"\multicolumn{{{nc}}}{{@{{}}l@{{}}}}{{\textit{{\textbf{{Claim 1: what writing does to cue reliance (\texttt{{CoT}}$-$\texttt{{Direct}})}}}}}} \\")
    L.append(row(r"Occlusion weight $\Delta d'$ (primary)", [s_(res[m]["glm"]["cot-direct"]["occlusion"]["d"]) + ci_(*res[m]["glm"]["cot-direct"]["occlusion"]["ci"]) for m in ms]))
    L.append(row("Vertical weight $\\Delta d'$", [s_(res[m]["glm"]["cot-direct"]["vertical"]["d"]) + star(*res[m]["glm"]["cot-direct"]["vertical"]["ci"]) for m in ms]))
    L.append(row("Occlusion AME $\\Delta$ (pts)", [s_(res[m]["ame"]["occlusion"]["ame"]["cot-direct"]["est"], 1) + star(*res[m]["ame"]["occlusion"]["ame"]["cot-direct"]["ci"]) for m in ms]))
    L.append(row("Cue-set accuracy $\\Delta$ (pts)", [s_(res[m]["acc_cot-direct"]["mean"], 1) + star(*res[m]["acc_cot-direct"]["ci"]) for m in ms]))
    cells = []
    for m in ms:
        if "desc-direct" in res[m]["glm"]:
            o = res[m]["glm"]["desc-direct"]["occlusion"]
            cells.append(s_(o["d"]) + star(*o["ci"]) + " / " + s_(res[m]["acc_desc-direct"]["mean"], 1) + star(*res[m]["acc_desc-direct"]["ci"]))
        else:
            cells.append("--")
    L.append(row("Describe$-$\\texttt{Direct}: occl.\\ $\\Delta d'$ / acc.\\ $\\Delta$", cells))
    cells = []
    for m in ms:
        cp = res[m]["dpair"]["cot-direct_pooled"]
        txt = s_(cp["d"]) + star(*cp["ci"]) + (r"$^\ddagger$" if res[m]["dpair"]["split_direct"][3] == 0 else "")
        if "desc-direct_pooled" in res[m]["dpair"]:
            dp = res[m]["dpair"]["desc-direct_pooled"]
            txt += " / " + s_(dp["d"]) + star(*dp["ci"])
        else:
            txt += " / --"
        cells.append(txt)
    L.append(row("Cue-set $\\Delta d'_\\mathrm{pair}$: \\texttt{CoT} / Describe", cells))
    L.append(row("\\texttt{D+C$-$} / \\texttt{D$-$C+}; net (pts)", [f"${res[m]['cells']['D+C-']}$ / ${res[m]['cells']['D-C+']}$; " + s_(res[m]["cells"]["net_pts"], 1) for m in ms]))
    L.append(r"\midrule")
    L.append(rf"\multicolumn{{{nc}}}{{@{{}}l@{{}}}}{{\textit{{\textbf{{Written relation (\texttt{{CoT}}, parser v3)}}}}}} \\")
    L.append(row("First claim: true front / lower / larger (\\%)", [
        f"${res[m]['first_claim_cot']['names_front']['pct']:.1f}$ / ${res[m]['first_claim_cot']['names_lower']['pct']:.1f}$ / ${res[m]['first_claim_cot']['names_larger']['pct']:.1f}$"
        if res[m]["first_claim_cot"].get("n_parsed") else "--" for m in ms]))
    L.append(row("Answer agrees with last relation: true / false (\\%)", [
        f"${res[m]['follow_cot_last']['all']['p_cons_true']['mean']:.1f}$ / ${res[m]['follow_cot_last']['all']['p_cons_false']['mean']:.1f}$"
        if res[m]["follow_cot_last"].get("all", {}).get("n") else "--" for m in ms]))
    L.append(row("Two channels: $x$ / $y$ (pts)", [
        s_(res[m]["l3"]["cues"]["last_x"], 1) + " / " + s_(res[m]["l3"]["cues"]["y"], 1) for m in ms]))
    L.append(r"\bottomrule")
    note = (r"Same $1{,}600$ cue-factorial queries, prompts, greedy decoding ($\le 512$ tokens), image budget and statistics as "
            r"\cref{M-tab:profile}; one run per column, tensor parallel 2. Readout as each model's 8B sibling: InternVL3.5-38B top-20 letter scores (vLLM V1), "
            r"Qwen2.5-VL-32B exact full-vocabulary letter scores (vLLM V0). "
            r"$^\S$Probit quasi-separated ($\max|\beta|>2.5$); use the AME row. "
            r"$^\ddagger$No both-wrong unit under \texttt{Direct}, so \texttt{Direct} $d'_\mathrm{pair}$ is a lower bound and its change is not interpretable. "
            r"$^\ast$: $95\%$ CI excludes zero (twin-pair bootstrap). Describe: question-blind description, then the question. "
            r"$x$: last written relation (parser v3) accuracy minus \texttt{Direct} accuracy on parsed items; $y$: \texttt{CoT}$-$\texttt{Direct} accuracy on all items.")
    L.append(rf"\multicolumn{{{nc}}}{{p{{0.95\linewidth}}}}{{\scriptsize {note}}} \\")
    L.append(r"\end{tabular}")
    return "\n".join(L) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--nb-glm", type=int, default=1000)
    ap.add_argument("--models", default=",".join(NEW + REF))
    ap.add_argument("--table-only", action="store_true")
    ap.add_argument("--out-tag", default="e43_big_open")
    ap.add_argument("--merge", action="store_true", help="update an existing bundle instead of overwriting it")
    a = ap.parse_args()
    out_json = B / f"{a.out_tag}.json"
    out_txt = B / ("rigor_e43.txt" if a.out_tag == "e43_big_open" else f"rigor_{a.out_tag}.txt")
    if a.table_only:
        res = json.load(open(out_json))["models"]
    else:
        lines: list[str] = []
        res = {}
        if a.merge and out_json.exists():  # keep models analysed in an earlier call
            res = json.load(open(out_json))["models"]
            old_txt = out_txt.read_text().splitlines() if out_txt.exists() else []
            keep = {R31.NAMES[m] for m in res if m not in a.models.split(",")}
            blk, cur = [], None
            for ln in old_txt:
                if ln.startswith("== "):
                    cur = ln[3:].split(" (")[0]
                if cur in keep:
                    blk.append(ln)
            lines.extend(blk)
        for m in a.models.split(","):
            if not R31.load_arms(m).get("cot"):
                lines.append(f"== {m}: no outputs yet")
                continue
            res[m] = R31.analyze(m, a.nb_glm, lines)
            print("\n".join(lines[-40:]), flush=True)
        doc = {"protocol": __doc__, "nb_glm": a.nb_glm, "nb": R31.NB, "models": res}
        json.dump(doc, open(out_json, "w"), indent=1, default=float)
        out_txt.write_text("\n".join(lines) + "\n")
    if a.out_tag == "e43_big_open" and any(m in res for m in NEW):
        txt = table(res, NEW)
        (PAPER_TAB / "supp_e43.tex").write_text(txt)
        print(txt)


if __name__ == "__main__":
    main()
