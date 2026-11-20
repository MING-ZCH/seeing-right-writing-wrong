"""Exp-62 analysis: natural-object occlusion twins (COCO composites); prereg outputs/e62/prereg.txt.

Per model: Direct accuracy, twin-pair both-correct rate, CoT - Direct, image effect
E_c = 100 * mean[P_c(gold_v | v) - P_c(gold_v | v_cf)] for Direct and fixed R_cut (+ strict / full R, indicator
version), written relation of R_cut (two text-only judges, fixed sample) vs Direct and answer agreement.
CIs: 2,000 twin-pair cluster bootstrap draws (seed 62), shared by all models / arms.

-> outputs/bundle/e62_natural_twins.json, outputs/bundle/rigor_e62.txt, paper/tables/supp_e62.tex
   (json / txt also copied to shared outputs/e62/)

Usage: python rigor_e62.py [--no-table]
"""
from __future__ import annotations
import vtax_paths as _vtp  # noqa: E402

import argparse
import json
import math
import re
import shutil
from collections import Counter
from pathlib import Path

import numpy as np

from rigor_e60 import Boot as _Boot, ci, h, pe, u
import rigor_e60

REPO = Path(__file__).resolve().parent
SH = Path(f"{_vtp.OUTPUTS}/e62")
BUNDLE = Path(_vtp.OUTPUTS) / "bundle"
TABLE = REPO.parent / f"{_vtp.PAPER}" / "tables" / "supp_e62.tex"
M7 = ["qwen3b", "qwen7b", "internvl3_8b", "phi35v", "molmo7b", "qwen3vl_8b", "internvl35_8b"]
EXTRA = ["qwen3vl_32b"]
NAMES = {"qwen3b": "Qwen2.5-VL-3B", "qwen7b": "Qwen2.5-VL-7B", "internvl3_8b": "InternVL3-8B", "phi35v": "Phi-3.5-Vision",
         "molmo7b": "Molmo-7B-D", "qwen3vl_8b": "Qwen3-VL-8B", "internvl35_8b": "InternVL3.5-8B",
         "qwen3vl_32b": "Qwen3-VL-32B"}
JUDGES = ("claude-opus-5-5", "gpt-6-astra")
PHR = ("closer", "farther", "which")
ANS_LEAK = re.compile(r"\([AB]\)|answer is|\boption [AB]\b", re.I)
rigor_e60.SEED = 62


class Boot(_Boot):
    """Twin-pair cluster bootstrap (rigor_e60.Boot with seed 62)."""


def rj(p: Path) -> list[dict]:
    return [json.loads(l) for l in open(p)] if p.exists() else []


def p2(lp: dict | None, g: str) -> float:
    """Two-option renormalised probability of g (missing -> 0.5), as rigor_e60.p2."""
    fin = {k: v for k, v in (lp or {}).items() if v is not None and math.isfinite(v) and v > -1e3}
    if not fin:
        return 0.5
    m = max(fin.values())
    z = sum(math.exp(v - m) for v in fin.values())
    return math.exp(fin[g] - m) / z if g in fin else 0.0


def letter_mass(r: dict) -> float:
    return sum(math.exp(v) for v in r["top20"].values() if v is not None)


def rel_consensus() -> tuple[dict, dict]:
    """Consensus written relation per task id (FIRST / SECOND / NONE / unresolved / missing) and judge agreement."""
    lab = {j: {r["tid"]: r["label"] for r in rj(SH / f"judge_rel_{j}.jsonl") if r.get("label")} for j in JUDGES}
    out = {}
    for t in rj(SH / "rel_tasks.jsonl"):
        a, b = (lab[j].get(t["tid"]) for j in JUDGES)
        out[t["tid"]] = "missing" if a is None or b is None else (a if a == b else "unresolved")
    ks = sorted(set(lab[JUDGES[0]]) & set(lab[JUDGES[1]]))
    agr = {}
    if ks:
        a = [lab[JUDGES[0]][k] for k in ks]
        b = [lab[JUDGES[1]][k] for k in ks]
        po = float(np.mean([x == y for x, y in zip(a, b)]))
        ca, cb = Counter(a), Counter(b)
        pe_ = sum(ca[c] * cb[c] for c in set(a) | set(b)) / len(ks) ** 2
        agr = {"n": len(ks), "agree": round(100 * po, 2), "kappa": round((po - pe_) / (1 - pe_), 3),
               "dist": {j: dict(Counter(lab[j].values())) for j in JUDGES}}
    return out, agr


def implied_letter(it: dict, closer_name: str) -> str:
    """Answer letter implied by a written 'closer_name is closer' relation for this item's question."""
    m = it["meta"]
    if it["phrasing"] == "closer":
        return "A" if closer_name == m["x_name"] else "B"
    if it["phrasing"] == "farther":
        return "A" if closer_name == m["y_name"] else "B"
    return "A" if closer_name == m["which_opts"][0] else "B"


def model_arrays(m: str, items: list[dict], rel: dict, rel_rows: dict) -> dict | None:
    recs = {(r["uid"], r["arm"], r["img"]): r for r in rj(SH / f"{m}.jsonl")}
    if not recs:
        return None
    A: dict = {k: [] for k in ("acc_d", "acc_cot", "acc_rcut", "stated_ok", "Ep_direct", "Ep_rcut", "Ep_strict",
                               "Ep_full", "Ei_direct", "Ei_rcut", "Ei_strict", "Ei_full", "mass_d", "trunc", "leak",
                               "prefix_only", "strict_diff", "gen_len", "written", "wtrue", "follow_cot", "follow_rcut",
                               "follow_rcut_cf", "in_sample")}
    for it in items:
        u_, g, tw = it["uid"], it["gold"], it["twin_item"]
        d, dcf = recs[(u_, "direct", "v")], recs[(tw, "direct", "v")]
        cot = recs[(u_, "cot", "v")]
        A["acc_d"].append(float(pe(d) == g))
        A["acc_cot"].append(float(pe(cot) == g))
        A["acc_rcut"].append(float(pe(recs[(u_, "rcut", "v")]) == g))
        A["stated_ok"].append(float(cot.get("stated") == g))
        A["Ep_direct"].append(p2(d["lp_sum"], g) - p2(dcf["lp_sum"], g))
        A["Ei_direct"].append(float(pe(d) == g) - float(pe(dcf) == g))
        for arm in ("rcut", "strict"):
            rv, rc = recs[(u_, arm, "v")], recs[(u_, arm, "vcf")]
            A[f"Ep_{arm}"].append(p2(rv["lp_sum"], g) - p2(rc["lp_sum"], g))
            A[f"Ei_{arm}"].append(float(pe(rv) == g) - float(pe(rc) == g))
        fc = recs[(u_, "full", "vcf")]
        A["Ep_full"].append(p2(cot["lp_sum"], g) - p2(fc["lp_sum"], g))
        A["Ei_full"].append(float(pe(cot) == g) - float(pe(fc) == g))
        A["mass_d"].append(letter_mass(d))
        A["trunc"].append(float(cot["trunc"]))
        A["gen_len"].append(cot["gen_len"])
        A["leak"].append(float(bool(ANS_LEAK.search(cot["rcut"]))))
        A["prefix_only"].append(float(cot["rcut"].strip() == "Let's think step by step."))
        A["strict_diff"].append(float(cot["strict"].strip() != cot["rcut"].strip()))
        rr = rel_rows.get((m, u_))
        if rr is None:
            A["in_sample"].append(False)
            A["written"].append("n/a")
            for k in ("wtrue", "follow_cot", "follow_rcut", "follow_rcut_cf"):
                A[k].append(np.nan)
            continue
        A["in_sample"].append(True)
        lab = "NONE" if rr["prefilter_none"] else rel.get(rr["tid"], "missing")
        A["written"].append(lab)
        if lab in ("FIRST", "SECOND"):
            closer = rr["first"] if lab == "FIRST" else rr["second"]
            wl = implied_letter(it, closer)
            A["wtrue"].append(float(closer == it["front_name"]))
            A["follow_cot"].append(float(pe(cot) == wl))
            A["follow_rcut"].append(float(pe(recs[(u_, "rcut", "v")]) == wl))
            A["follow_rcut_cf"].append(float(pe(recs[(u_, "rcut", "vcf")]) == wl))
        else:
            for k in ("wtrue", "follow_cot", "follow_rcut", "follow_rcut_cf"):
                A[k].append(np.nan)
    return {k: np.array(v) for k, v in A.items()}


def pair_both(items: list[dict], acc: np.ndarray) -> tuple[np.ndarray, list[str]]:
    """Both-correct indicator per (pair, phrasing) and its cluster (pair) ids."""
    idx = {it["uid"]: i for i, it in enumerate(items)}
    vals, groups = [], []
    for it in items:
        if it["uid"] < it["twin_item"]:
            vals.append(acc[idx[it["uid"]]] * acc[idx[it["twin_item"]]])
            groups.append(it["pair"])
    return np.array(vals), groups


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-table", action="store_true")
    a = ap.parse_args()
    lines: list[str] = []
    P = lambda s: (lines.append(s), print(s))  # noqa: E731
    items = rj(SH / "stimuli/items_final.jsonl")
    man = json.load(open(SH / "stimuli/manifest.json"))
    P(f"E62 items: {len(items)} ({man['n_pairs']} pairs, {man['n_images']} images); dropped pairs {man['dropped_pairs']}")
    P(f"Balance: {json.dumps({k: v for k, v in man['report'].items() if k not in ('category_counts', 'tag')})}")
    rel, agr = rel_consensus()
    rr = json.load(open(SH / "rel_rows.json")) if (SH / "rel_rows.json").exists() else {"rows": []}
    rel_rows = {(r["model"], r["uid"]): r for r in rr["rows"]}
    P(f"Relation judges: {agr}")
    boot = Boot([it["pair"] for it in items])
    groups_pairs = pair_both(items, np.zeros(len(items)))[1]
    boot_p = Boot(groups_pairs)
    ph = np.array([it["phrasing"] for it in items])
    front_large = np.array([it["meta"]["front_is_large"] for it in items])
    res: dict = {"n_items": len(items), "n_pairs": man["n_pairs"], "judges_rel": agr, "models": {}}
    md: dict = {}
    for m in M7 + EXTRA:
        A = model_arrays(m, items, rel, rel_rows)
        if A is None:
            P(f"{m}: no records")
            continue
        r: dict = {}
        for k in ("acc_d", "acc_cot", "acc_rcut", "stated_ok"):
            r[k] = ci(100 * A[k].mean(), 100 * boot.draws(A[k]))
        dlt = A["acc_cot"] - A["acc_d"]
        r["cot_minus_direct"] = ci(100 * dlt.mean(), 100 * boot.draws(dlt))
        both, _ = pair_both(items, A["acc_d"])
        r["both_d"] = ci(100 * both.mean(), 100 * boot_p.draws(both))
        both_c, _ = pair_both(items, A["acc_cot"])
        r["both_cot"] = ci(100 * both_c.mean(), 100 * boot_p.draws(both_c))
        for k in ("Ep_direct", "Ep_rcut", "Ep_strict", "Ep_full", "Ei_direct", "Ei_rcut", "Ei_strict", "Ei_full"):
            dr = 100 * boot.draws(A[k])
            r[k] = ci(100 * A[k].mean(), dr)
            if m in M7:
                md.setdefault(k, []).append(dr)
        dd = A["Ep_rcut"] - A["Ep_direct"]
        r["D_rcut_minus_direct"] = ci(100 * dd.mean(), 100 * boot.draws(dd))
        if m in M7:
            md.setdefault("D", []).append(100 * boot.draws(dd))
            md.setdefault("acc_d", []).append(100 * boot.draws(A["acc_d"]))
            md.setdefault("cmd", []).append(100 * boot.draws(dlt))
        ed, er = r["Ep_direct"]["mean"], r["Ep_rcut"]["mean"]
        r["ratio"] = round(er / ed, 3) if ed > 5 else None
        r["reads_occlusion"] = bool(r["Ep_direct"]["lo"] > 0 and r["both_d"]["lo"] > 25)
        r["much_smaller"] = bool(r["D_rcut_minus_direct"]["hi"] < 0 and er <= 0.5 * ed) if r["reads_occlusion"] else None
        r["by_phrasing"] = {p: {k: round(float(100 * A[k][ph == p].mean()), 2)
                                for k in ("acc_d", "acc_cot", "Ep_direct", "Ep_rcut", "Ep_strict")} for p in PHR}
        r["by_visible_size"] = {"front_visibly_larger": round(float(100 * A["acc_d"][front_large].mean()), 2),
                                "near_equal_visible": round(float(100 * A["acc_d"][~front_large].mean()), 2)}
        r["sanity"] = {"direct_letter_mass_median": round(float(np.median(A["mass_d"])), 4),
                       "direct_letter_mass_lt_0.5": round(100 * float(np.mean(A["mass_d"] < 0.5)), 2),
                       "cot_trunc": round(float(100 * A["trunc"].mean()), 2), "gen_len_median": float(np.median(A["gen_len"])),
                       "rcut_answer_leak": round(float(100 * A["leak"].mean()), 2),
                       "rcut_prefix_only": round(float(100 * A["prefix_only"].mean()), 2),
                       "strict_differs": round(float(100 * A["strict_diff"].mean()), 2)}
        s = A["in_sample"]
        if s.any():
            w = A["written"][s]
            cons = np.isin(w, ["FIRST", "SECOND"])
            wt = A["wtrue"][s][cons]
            mask_w = s & np.isin(A["written"], ["FIRST", "SECOND"])
            mask_n = s & (A["written"] == "NONE")
            r["written"] = {"n_sample": int(s.sum()), "counts": dict(Counter(w.tolist())),
                            "coverage": round(float(100 * cons.mean()), 2),
                            "acc_written": ci(float(100 * np.nanmean(A["wtrue"][mask_w])), 100 * boot.draws(A["wtrue"], mask_w))
                            if mask_w.sum() >= 10 else None,
                            "acc_direct_same_items": round(float(100 * A["acc_d"][mask_w].mean()), 2) if mask_w.any() else None,
                            "acc_direct_sample": round(float(100 * A["acc_d"][s].mean()), 2),
                            "follow_cot": round(float(100 * np.nanmean(A["follow_cot"][mask_w])), 2) if mask_w.any() else None,
                            "follow_rcut_v": round(float(100 * np.nanmean(A["follow_rcut"][mask_w])), 2) if mask_w.any() else None,
                            "follow_rcut_vcf": round(float(100 * np.nanmean(A["follow_rcut_cf"][mask_w])), 2) if mask_w.any() else None,
                            "Ep_rcut_written": round(float(100 * A["Ep_rcut"][mask_w].mean()), 2) if mask_w.any() else None,
                            "Ep_rcut_none": round(float(100 * A["Ep_rcut"][mask_n].mean()), 2) if mask_n.any() else None,
                            "Ep_direct_written": round(float(100 * A["Ep_direct"][mask_w].mean()), 2) if mask_w.any() else None,
                            "Ep_direct_none": round(float(100 * A["Ep_direct"][mask_n].mean()), 2) if mask_n.any() else None,
                            "n_written": int(mask_w.sum()), "n_none": int(mask_n.sum())}
            if mask_w.sum() >= 10:
                dw = np.where(mask_w, np.nan_to_num(A["wtrue"]) - A["acc_d"], 0.0)
                r["written"]["written_minus_direct"] = ci(float(100 * dw[mask_w].mean()), 100 * boot.draws(dw, mask_w))
                r["written"]["Ep_rcut_written_ci"] = ci(float(100 * A["Ep_rcut"][mask_w].mean()),
                                                        100 * boot.draws(A["Ep_rcut"], mask_w))
            if mask_n.sum() >= 10:
                r["written"]["Ep_rcut_none_ci"] = ci(float(100 * A["Ep_rcut"][mask_n].mean()),
                                                     100 * boot.draws(A["Ep_rcut"], mask_n))
        res["models"][m] = r
        W = r.get("written") or {}
        P(f"{m:14s} Direct {u(r['acc_d']['mean'])} [{u(r['acc_d']['lo'])}, {u(r['acc_d']['hi'])}] | both-correct "
          f"{u(r['both_d']['mean'])} [{u(r['both_d']['lo'])}, {u(r['both_d']['hi'])}] | CoT {u(r['acc_cot']['mean'])} | "
          f"CoT-Direct {h(r['cot_minus_direct']['mean'])} [{h(r['cot_minus_direct']['lo'])}, {h(r['cot_minus_direct']['hi'])}] | "
          f"reads occlusion {r['reads_occlusion']}")
        P(f"{'':14s} E_direct {h(ed)} [{h(r['Ep_direct']['lo'])}, {h(r['Ep_direct']['hi'])}] | E_rcut {h(er)} "
          f"[{h(r['Ep_rcut']['lo'])}, {h(r['Ep_rcut']['hi'])}] | D {h(r['D_rcut_minus_direct']['mean'])} "
          f"[{h(r['D_rcut_minus_direct']['lo'])}, {h(r['D_rcut_minus_direct']['hi'])}] | ratio {r['ratio']} | much smaller "
          f"{r['much_smaller']} | E_strict {h(r['Ep_strict']['mean'])} | E_full {h(r['Ep_full']['mean'])} | indicator "
          f"E_direct {h(r['Ei_direct']['mean'])} E_rcut {h(r['Ei_rcut']['mean'])}")
        P(f"{'':14s} by phrasing {r['by_phrasing']} | Direct acc by visible size {r['by_visible_size']}")
        P(f"{'':14s} SANITY {r['sanity']}")
        if W:
            P(f"{'':14s} WRITTEN n={W['n_sample']} {W['counts']} coverage {W['coverage']} | acc written "
              f"{W['acc_written']} vs Direct same items {W['acc_direct_same_items']} (written-Direct "
              f"{W.get('written_minus_direct')}) | answer = written: CoT readout "
              f"{W['follow_cot']}, R_cut v {W['follow_rcut_v']}, R_cut v_cf {W['follow_rcut_vcf']} | E_rcut written "
              f"{W['Ep_rcut_written']} / NONE {W['Ep_rcut_none']} (n {W['n_written']} / {W['n_none']}; NONE CI "
              f"{W.get('Ep_rcut_none_ci')}) (E_direct {W['Ep_direct_written']} / {W['Ep_direct_none']})")
    have7 = [m for m in M7 if m in res["models"]]
    if len(have7) == len(M7):
        mean = {}
        for k, key in (("Ep_direct", "Ep_direct"), ("Ep_rcut", "Ep_rcut"), ("Ep_strict", "Ep_strict"), ("Ep_full", "Ep_full"),
                       ("D", "D_rcut_minus_direct"), ("acc_d", "acc_d"), ("cmd", "cot_minus_direct")):
            mean[key] = ci(float(np.mean([res["models"][m][key]["mean"] for m in M7])), np.vstack(md[k]).mean(0))
        res["mean7"] = mean
        P("MEAN7 " + " | ".join(f"{k} {h(v['mean'])} [{h(v['lo'])}, {h(v['hi'])}]" for k, v in mean.items()))
    readers = [m for m in res["models"] if res["models"][m]["reads_occlusion"]]
    smaller = [m for m in readers if res["models"][m]["much_smaller"]]
    res["decision"] = {"readers": readers, "much_smaller": smaller,
                       "verdict": ("SUPPORTED" if readers and len(smaller) == len(readers) else
                                   "PARTLY" if smaller else "NOT SUPPORTED" if readers else "UNINFORMATIVE")}
    P(f"DECISION {res['decision']}")
    BUNDLE.mkdir(parents=True, exist_ok=True)
    json.dump(res, open(BUNDLE / "e62_natural_twins.json", "w"), indent=1)
    (BUNDLE / "rigor_e62.txt").write_text("\n".join(lines) + "\n")
    shutil.copy(BUNDLE / "e62_natural_twins.json", SH / "e62_natural_twins.json")
    shutil.copy(BUNDLE / "rigor_e62.txt", SH / "rigor_e62.txt")
    if not a.no_table:
        write_table(res)


def cell(c: dict, sign: bool = True) -> str:
    f = h if sign else u
    return f"${f(c['mean'])}$\\,{{\\tiny$[{f(c['lo'])}, {f(c['hi'])}]$}}"


def write_table(res: dict) -> None:
    """Bare tabular; suggested caption in the header comment."""
    L = ["% Tab. supp_e62 (Exp-62, rigor_e62.py). Suggested caption:",
         "% \\textbf{Natural-object occlusion twins} (Exp-62). Two segmented COCO objects pasted on a natural background",
         f"% with 15-40\\% overlap; the twin image swaps only the paste order ({res['n_pairs']} pairs, {res['n_items']} prompts:",
         "% closer / farther yes-no and which-is-closer, asked in camera terms). Direct = forced readout; both-correct =",
         "% both twins answered correctly for the same question (image-blind answering gives 0, coin flipping 25);",
         "% CoT = greedy rationale, same-turn readout. Image effect E = P(answer = g_v | v) - P(answer = g_v | v_cf) in points",
         "% with the context fixed: Direct, or the model's own rationale on v with its concluding sentence removed (R_cut).",
         "% Written = relation stated by R_cut (two text-only judges, consensus; fixed 360-item sample): its accuracy / Direct",
         "% accuracy on the same items (coverage = share of R_cut with a stated relation), and share of R_cut readouts under",
         "% v_cf that follow it. Qwen3-VL-32B is not in the mean. 95\\% twin-pair bootstrap CIs (2,000). Compiles under \\resizebox.",
         "\\begin{tabular}{lcccccccc}", "\\toprule",
         " & \\multicolumn{3}{c}{Accuracy (\\%)} & \\multicolumn{3}{c}{Image effect $E$ (points)} & "
         "\\multicolumn{2}{c}{Written relation of R$_\\mathrm{cut}$} \\\\",
         "\\cmidrule(lr){2-4}\\cmidrule(lr){5-7}\\cmidrule(lr){8-9}",
         "Model & Direct & both-correct & CoT$-$Direct & Direct & R$_\\mathrm{cut}$ fixed & ratio & acc.\\ / Direct (cov.) & "
         "follows, $v_\\mathrm{cf}$ \\\\", "\\midrule"]
    for m in M7 + EXTRA:
        if m not in res["models"]:
            continue
        if m in EXTRA:
            L.append("\\midrule")
        r = res["models"][m]
        W = r.get("written") or {}
        wa = (f"{u(W['acc_written']['mean'])} / {u(W['acc_direct_same_items'])} ({u(W['coverage'], 0)})"
              if W.get("acc_written") else "--")
        fo = u(W["follow_rcut_vcf"]) if W.get("follow_rcut_vcf") is not None else "--"
        ratio = u(r["ratio"], 2) if r["ratio"] is not None else "n/a"
        L.append(f"{NAMES[m]} & {cell(r['acc_d'], False)} & {cell(r['both_d'], False)} & {cell(r['cot_minus_direct'])} & "
                 f"{cell(r['Ep_direct'])} & {cell(r['Ep_rcut'])} & {ratio} & {wa} & {fo} \\\\")
    if "mean7" in res:
        mn = res["mean7"]
        L += ["\\midrule", f"Mean of 7 & {cell(mn['acc_d'], False)} & & {cell(mn['cot_minus_direct'])} & "
              f"{cell(mn['Ep_direct'])} & {cell(mn['Ep_rcut'])} & & & \\\\"]
    L += ["\\bottomrule", "\\end{tabular}"]
    TABLE.write_text("\n".join(L) + "\n")
    print(f"wrote {TABLE}")


if __name__ == "__main__":
    main()
