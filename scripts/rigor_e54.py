"""Exp-54: where does CoT lose items, and is finding 2 more than uniform attenuation?

Part A (--only a): decomposition of every D+C- item (Direct right, CoT wrong) and, mirrored, every D-C+ item, for the
seven open instruct models on the cue-factorial set, the occlusion twins and Synthetic Depth 431.
  Written relation = last relation of the conclusion-removed body, Exp-45 two-judge consensus (Claude Opus 5.5 +
  GPT-6-Astra; no consensus = no relation). Robustness: claim_parse v3 on the same truncated body and on the full text.
  D+C-  (a) last relation FALSE ("writing wrong"); subtypes (exclusive, in this order) from the Exp-54 basis judges:
            affirmation (no visual reason given AND the relation affirms the queried subject/predicate), height
            (vertical-position reason incl. 'on top' = higher), size (larger = closer), occlusion misread (overlap
            reason, wrong shape), other false (other reason, QUESTION basis that denies, judges disagree).
        (b) last relation TRUE, answer wrong ("answering wrong"); subtypes from the Exp-54 step judges: converse /
            other-shape mapping (the question is not the written proposition in its own words: other shape, converse
            or contrary predicate), conclusion reversal (question restates the written proposition, the removed
            conclusion states the opposite answer), other (incl. unresolved form).
        (c) no relation.
  D-C+  (a') TRUE relation ("writing right"; subtype = basis: occlusion / height / size / affirmation / other),
        (b') FALSE relation, answer right (mapping / conclusion reversal / other), (c') no relation.
  CIs: item-cluster bootstrap (clusters = Exp-45 'cl': cue twin pair, twin pair, Syn. Depth image), 2,000 draws;
  pooled shares resample clusters within each model x set cell.
Part B (--only b): uniform-attenuation null for finding 2 on the cue set. Probit cue model of rigor_e17f.glm_rows
  (says-subject-front ~ 1 + phrasing + occlusion + vertical + size, ridge 0.1) fitted to Direct, CoT and the answer
  implied by the last written relation (Exp-45 consensus, primary; v3 with the conclusion kept = Exp-32 / paper x,
  robustness). Null: written cue weights = s * Direct cue weights with s = s_cot (paper's common scale), s_proj
  (projection of the written weights) or s_ML (constrained ML on the written relation, intercept and phrasing free);
  x_null = expected written accuracy on the covered items under the null. LR test free vs s_ML (df 2).
  delta_k = b_written,k - s * b_Direct,k. Twin-pair (image_id) bootstrap, 1,000 draws.

Outputs: outputs/bundle/e54_decomp.json, e54_attenuation.json, rigor_e54.txt; paper tables supp_e54a.tex, supp_e54b.tex.
Usage: python rigor_e54.py [--only a,b] [--nb 2000]
"""
from __future__ import annotations
import vtax_paths as _vtp  # noqa: E402

import argparse
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from scipy.stats import chi2, norm

from e45_analyze import implied, load_labels, shape_records
from rigor_e32 import sent_front
from e45_uniform_two_channel import E45, JUDGES, M7, NAMES, O, rj

B = Path("outputs/bundle")
PAPER = Path(f"{_vtp.PAPER}")
E54 = O / "e54"
DS3 = ("cues", "twins", "syn431")
DSN = {"cues": "Cue set", "twins": "Twins", "syn431": "Syn.\\ Depth 431"}
ONTOP = re.compile(r"\bon (the )?top\b|\batop\b|\bover\b|\babove\b", re.I)
q95 = lambda v: [float(np.nanquantile(v, .025)), float(np.nanquantile(v, .975))]

CAT_DM = ["a_affirm", "a_height", "a_size", "a_occ", "a_other", "b_map", "b_rev", "b_other", "c_none"]
CAT_DP = ["a_occ", "a_height", "a_size", "a_affirm", "a_other", "b_map", "b_rev", "b_other", "c_none"]


# ============================================================================= Part A: labels

def judge_labels() -> tuple[dict, dict]:
    """tid -> {judge: label}; task meta by tid."""
    tasks = {t["tid"]: t for t in rj(E54 / "tasks.jsonl")}
    lab = defaultdict(dict)
    for j in JUDGES:
        for r in rj(E54 / f"judge_{j}.jsonl"):
            if r.get("label") is not None:
                lab[r["tid"]].setdefault(j, r["label"])
    return lab, tasks


def tid_of(kind: str, row: dict, claim: str) -> str:
    import hashlib
    from e54_judge import PROMPT_BASIS, PROMPT_STEP, question, removed_sents
    conc = " ".join(removed_sents(row["sents"], row["trunc"]))
    p = (PROMPT_BASIS if kind == "basis" else PROMPT_STEP).format(x=row["x"], y=row["y"], q=question(row), r=row["body"],
                                                                 f=claim, c=conc)
    return hashlib.sha1(f"{kind}\x00{p}".encode()).hexdigest()[:20]


def geometry(rows: list[dict]) -> dict:
    """(model-free) key (ds, item) -> {'lower': shape, 'larger': shape} for the cue set, twins and Syn. Depth."""
    from rigor_e17f import geom_meta
    g = {}
    for r in rj(O / "e10" / "qwen7b_cues.jsonl"):
        if r["condition"] == "direct":
            gm = geom_meta(r["meta"], r["meta"]["x_name"].lower())
            g[("cues", r["item_id"])] = {"lower": gm["lower"], "larger": gm["larger"]}
    for suf, ph in (("", "front"), ("_behind", "behind")):
        for r in rj(O / "e2d" / f"qwen7b{suf}.jsonl"):
            if r["condition"] == "direct":
                gm = geom_meta(r["meta"], r["meta"]["x_name"].lower())
                g[("twins", f"{r['item_id']}|{ph}")] = {"lower": gm["lower"], "larger": gm["larger"]}
    sg = json.load(open(B / "syn_depth_geometry.json"))["items"]
    for uid, d in sg.items():
        (a, da), (b, db) = list(d.items())
        lower = None if da["cy"] is None or db["cy"] is None else (a if da["cy"] > db["cy"] else b)
        g[("syn431", uid)] = {"lower": lower, "larger": a if (da["px"] or 0) > (db["px"] or 0) else b}
    return g


def records() -> list[dict]:
    rows = [r for r in rj(E45 / "items.jsonl") if r["group"] == "open" and r["model"] in M7 and r["ds"] in DS3]
    recs = shape_records(rows, load_labels())
    lab, tasks = judge_labels()
    geo = geometry(rows)
    out = []
    for row, r in zip(rows, recs):
        dm = r["d_ok"] and not r["c_ok"]
        dp = (not r["d_ok"]) and r["c_ok"]
        o = {"model": r["model"], "ds": r["ds"], "item": r["item"], "cl": r["cl"], "dm": dm, "dp": dp,
             "dd": r["d_ok"] and r["c_ok"], "has": r["has"], "true": r["true"], "claim": r["claim"],
             "x": row["x"], "y": row["y"], "front": row["front"], "phrasing": row["phrasing"],
             "v3_trunc": row["v3_trunc"], "v3_full": row["v3_full"], "c_ans": row["c_ans"], "gold": row["gold"]}
        gm = geo.get((r["ds"], r["item"]))
        o["lower"], o["larger"] = (gm["lower"], gm["larger"]) if gm else (None, None)
        if r["has"]:
            o["affirms"] = implied(r["claim"], row["x"], row["phrasing"]) == "A"
            lrs = [t for t in row["trunc"] if sent_front(t, row["front"], row["back"]) == r["claim"]]
            o["lrs"] = lrs[-1] if lrs else None
            o["ontop_upper"] = bool(o["lrs"] and ONTOP.search(o["lrs"]) and o["lower"] is not None and r["claim"] != o["lower"])
            kind = "basis" if (dm and not r["true"]) or (dp and r["true"]) else "step"
            if dm or dp or (o["dd"] and r["true"]):
                t = tid_of(kind, row, r["claim"])
                if t in tasks:
                    o["kind"], o["tid"], o["jl"] = kind, t, lab.get(t, {})
                    o["ref"] = bool(tasks[t].get("ref")) and o["dd"] and tasks[t]["first"] == f"{r['model']}|{r['ds']}|{r['item']}"
        out.append(o)
    return out


def basis_cons(jl: dict) -> str | None:
    a, b = (jl.get(JUDGES[0]) or {}).get("basis"), (jl.get(JUDGES[1]) or {}).get("basis")
    return a if a is not None and a == b else None


def step_cons(jl: dict) -> tuple[dict | None, str | None]:
    a, b = jl.get(JUDGES[0]), jl.get(JUDGES[1])
    if not a or not b:
        return None, None
    form = None
    if (a["subj"], a["pred"], a["neg"]) == (b["subj"], b["pred"], b["neg"]):
        form = {"subj": a["subj"], "pred": a["pred"], "neg": a["neg"]}
    conc = a["conc"] if a["conc"] == b["conc"] else None
    return form, conc


def q_form(o: dict, form: dict) -> str:
    """Question relative to the written proposition: identical / converse / swap (other shape, same predicate) /
    contrary (same shape, opposite predicate). Negation folded into the predicate."""
    pred = form["pred"]
    if form["neg"]:
        pred = "BEHIND" if pred == "FRONT" else "FRONT"
    qp = "FRONT" if o["phrasing"] == "front" else "BEHIND"
    same_s, same_p = form["subj"] == o["x"], pred == qp
    return {(True, True): "identical", (False, False): "converse", (False, True): "swap", (True, False): "contrary"}[(same_s, same_p)]


def classify(o: dict, order: str = "primary") -> str | None:
    if not (o["dm"] or o["dp"]):
        return None
    if not o["has"]:
        return "c_none"
    writing = (o["dm"] and not o["true"]) or (o["dp"] and o["true"])
    if writing:
        bc = basis_cons(o.get("jl", {}))
        if order == "affirm_first":
            if o["dm"] and o["affirms"]:
                return "a_affirm"
        if order == "ontop" and o["dm"]:
            if bc == "QUESTION" and o["affirms"]:
                return "a_affirm"
            if bc == "VERTICAL" or o.get("ontop_upper"):
                return "a_height"
            return {"SIZE": "a_size", "OCCLUSION": "a_occ"}.get(bc, "a_other")
        if o["dp"]:
            return {"OCCLUSION": "a_occ", "VERTICAL": "a_height", "SIZE": "a_size"}.get(bc) or (
                "a_affirm" if bc == "QUESTION" and o["affirms"] else "a_other")
        if bc == "QUESTION" and o["affirms"]:
            return "a_affirm"
        return {"VERTICAL": "a_height", "SIZE": "a_size", "OCCLUSION": "a_occ"}.get(bc, "a_other")
    form, conc = step_cons(o.get("jl", {}))
    o["qform"] = q_form(o, form) if form else None
    rel_ans = "YES" if o["affirms"] else "NO"
    o["conc_rev"] = conc in ("YES", "NO") and conc != rel_ans
    if o["qform"] not in (None, "identical"):
        return "b_map"
    if o["conc_rev"]:
        return "b_rev"
    return "b_other"


def classify_geo(o: dict, which: str) -> str | None:
    """Rule-only variant: written relation from claim_parse v3 (which = v3_trunc / v3_full); (a) subtypes by geometry,
    affirm first: affirms -> lower named front -> larger named front -> other; (b) not subdivided."""
    if not (o["dm"] or o["dp"]):
        return None
    c = o[which]
    if c is None:
        return "c_none"
    true = c == o["front"]
    if (o["dm"] and not true) or (o["dp"] and true):
        if o["dm"]:
            if implied(c, o["x"], o["phrasing"]) == "A":
                return "a_affirm"
            if o["lower"] is not None and c == o["lower"]:
                return "a_height"
            if o["larger"] is not None and c == o["larger"]:
                return "a_size"
        return "a_other"
    return "b_other"


# ============================================================================= Part A: shares and bootstrap

def share_boot(recs: list[dict], key: str, cats: list[str], nb: int, rng, strata: str | None = None) -> dict:
    """Shares of categories among recs (already filtered) with cluster bootstrap (stratified by `strata`)."""
    groups = defaultdict(lambda: defaultdict(list))
    for r in recs:
        groups[r[strata] if strata else 0][r["cl"]].append(r[key])
    mats = []
    for g, cl in groups.items():
        M = np.array([[sum(v == c for v in vs) for c in cats] for vs in cl.values()], float)
        mats.append(M)
    tot = sum(M.sum(0) for M in mats)
    n = tot.sum()
    est = tot / n if n else np.full(len(cats), np.nan)
    draws = []
    for _ in range(nb):
        s = sum(M[rng.integers(0, len(M), len(M))].sum(0) for M in mats)
        draws.append(s / s.sum() if s.sum() else np.full(len(cats), np.nan))
    draws = np.array(draws)
    out = {"n": int(n), "counts": {c: int(v) for c, v in zip(cats, tot)}}
    for i, c in enumerate(cats):
        out[c] = {"pct": 100 * float(est[i]), "ci": [100 * v for v in q95(draws[:, i])]}
    # top-level groups
    for g in ("a", "b", "c"):
        idx = [i for i, c in enumerate(cats) if c.startswith(g + "_")]
        out[g] = {"pct": 100 * float(est[idx].sum()), "ci": [100 * v for v in q95(draws[:, idx].sum(1))],
                  "k": int(tot[idx].sum())}
    ia = [i for i, c in enumerate(cats) if c.startswith("a_")]
    ib = [i for i, c in enumerate(cats) if c.startswith("b_")]
    ab = draws[:, ia].sum(1) / np.maximum(draws[:, ia].sum(1) + draws[:, ib].sum(1), 1e-12)
    den = tot[ia].sum() + tot[ib].sum()
    out["a_of_ab"] = {"pct": 100 * float(tot[ia].sum() / den) if den else None, "ci": [100 * v for v in q95(ab)]}
    return out


def part_a(lines: list[str], nb: int) -> dict:
    rng = np.random.default_rng(54)
    R = records()
    for o in R:
        o["cat"] = classify(o)
        o["cat_af"] = classify(o, "affirm_first") if o["cat"] else None
        o["cat_ot"] = classify(o, "ontop") if o["cat"] else None
        o["cat_v3t"] = classify_geo(o, "v3_trunc")
        o["cat_v3f"] = classify_geo(o, "v3_full")
    res = {"_doc": __doc__.split("Part B")[0].strip(), "cells": {}, "models": {}, "pooled": {}}
    DM = [o for o in R if o["dm"]]
    DP = [o for o in R if o["dp"]]
    # coverage of new labels
    need = [o for o in DM + DP if o["has"]]
    miss = [o for o in need if len(o.get("jl", {})) < 2]
    res["label_coverage"] = {"need": len(need), "both_judges": len(need) - len(miss), "missing": len(miss)}
    lines.append(f"LABELS need {len(need)} both judges {len(need) - len(miss)} missing {len(miss)}")
    # judge agreement
    ag = {}
    for kind in ("basis", "step"):
        pairs = [o["jl"] for o in R if o.get("kind") == kind and len(o.get("jl", {})) == 2]
        if kind == "basis":
            a = [p[JUDGES[0]]["basis"] for p in pairs]
            b = [p[JUDGES[1]]["basis"] for p in pairs]
            ag["basis"] = {"n": len(a), "agree": 100 * np.mean([x == y for x, y in zip(a, b)]), "kappa": kappa(a, b),
                           "dist_opus": dict(Counter(a)), "dist_gpt": dict(Counter(b))}
        else:
            fa = [(p[JUDGES[0]]["subj"], p[JUDGES[0]]["pred"], p[JUDGES[0]]["neg"]) for p in pairs]
            fb = [(p[JUDGES[1]]["subj"], p[JUDGES[1]]["pred"], p[JUDGES[1]]["neg"]) for p in pairs]
            ca = [p[JUDGES[0]]["conc"] for p in pairs]
            cb = [p[JUDGES[1]]["conc"] for p in pairs]
            ag["step"] = {"n": len(fa), "form_agree": 100 * np.mean([x == y for x, y in zip(fa, fb)]),
                          "form_kappa": kappa([str(x) for x in fa], [str(y) for y in fb]),
                          "conc_agree": 100 * np.mean([x == y for x, y in zip(ca, cb)]), "conc_kappa": kappa(ca, cb)}
    res["judge_agreement"] = ag
    lines.append(f"JUDGE agreement {json.dumps({k: {kk: (round(vv, 3) if isinstance(vv, float) else vv) for kk, vv in v.items()} for k, v in ag.items()})}")
    for o in R:
        o["mds"] = f"{o['model']}|{o['ds']}"
    # per cell / per model / pooled
    for m in M7:
        for ds in DS3:
            dm = [o for o in DM if o["model"] == m and o["ds"] == ds]
            dp = [o for o in DP if o["model"] == m and o["ds"] == ds]
            res["cells"][f"{m}|{ds}"] = {"DpCm": share_boot(dm, "cat", CAT_DM, nb, rng),
                                         "DmCp": share_boot(dp, "cat", CAT_DP, nb, rng),
                                         "DpCm_v3_trunc": share_boot(dm, "cat_v3t", CAT_DM, nb // 4, rng),
                                         "DpCm_v3_full": share_boot(dm, "cat_v3f", CAT_DM, nb // 4, rng)}
        dm = [o for o in DM if o["model"] == m]
        dp = [o for o in DP if o["model"] == m]
        res["models"][m] = {"DpCm": share_boot(dm, "cat", CAT_DM, nb, rng, "mds"),
                            "DmCp": share_boot(dp, "cat", CAT_DP, nb, rng, "mds"),
                            "DpCm_affirm_first": share_boot(dm, "cat_af", CAT_DM, nb // 4, rng, "mds"),
                            "DpCm_ontop": share_boot(dm, "cat_ot", CAT_DM, nb // 4, rng, "mds"),
                            "DpCm_v3_trunc": share_boot(dm, "cat_v3t", CAT_DM, nb // 4, rng, "mds"),
                            "DpCm_v3_full": share_boot(dm, "cat_v3f", CAT_DM, nb // 4, rng, "mds")}
    for ds in DS3 + ("all",):
        dm = [o for o in DM if ds == "all" or o["ds"] == ds]
        dp = [o for o in DP if ds == "all" or o["ds"] == ds]
        res["pooled"][ds] = {"DpCm": share_boot(dm, "cat", CAT_DM, nb, rng, "mds"),
                             "DmCp": share_boot(dp, "cat", CAT_DP, nb, rng, "mds"),
                             "DpCm_affirm_first": share_boot(dm, "cat_af", CAT_DM, nb // 4, rng, "mds"),
                             "DpCm_ontop": share_boot(dm, "cat_ot", CAT_DM, nb // 4, rng, "mds"),
                             "DpCm_v3_trunc": share_boot(dm, "cat_v3t", CAT_DM, nb // 4, rng, "mds"),
                             "DpCm_v3_full": share_boot(dm, "cat_v3f", CAT_DM, nb // 4, rng, "mds")}
    # (a) cross-tab: geometric properties of the false relation (overlapping) by basis
    xt = {}
    for m in M7 + ["all"]:
        A = [o for o in DM if o["has"] and not o["true"] and (m == "all" or o["model"] == m)]
        bc = Counter(basis_cons(o.get("jl", {})) or "disagree" for o in A)
        xt[m] = {"n": len(A), "basis": dict(bc),
                 "affirms_pct": 100 * np.mean([o["affirms"] for o in A]) if A else None,
                 "names_lower_pct": 100 * np.mean([o["claim"] == o["lower"] for o in A]) if A else None,
                 "names_larger_pct": 100 * np.mean([o["claim"] == o["larger"] for o in A]) if A else None,
                 "ontop_wording_pct": 100 * np.mean([bool(o.get("lrs") and ONTOP.search(o["lrs"])) for o in A]) if A else None,
                 "ontop_upper_pct": 100 * np.mean([o.get("ontop_upper", False) for o in A]) if A else None,
                 "lrs_found_pct": 100 * np.mean([o.get("lrs") is not None for o in A]) if A else None,
                 "affirms_by_basis": {k: 100 * np.mean([o["affirms"] for o in A if (basis_cons(o.get("jl", {})) or "disagree") == k])
                                      for k in bc}}
    res["false_relation_profile"] = xt
    # (b) question forms: D+C- true-relation items vs the D+C+ reference sample
    qf = {}
    for m in M7 + ["all"]:
        Bm = [o for o in DM if o["has"] and o["true"] and (m == "all" or o["model"] == m)]
        Rf = [o for o in R if o.get("ref") and (m == "all" or o["model"] == m)]
        for o in Rf:
            form, conc = step_cons(o.get("jl", {}))
            o["qform"] = q_form(o, form) if form else None
            o["conc_rev"] = conc in ("YES", "NO") and conc != ("YES" if o["affirms"] else "NO")
        qf[m] = {"DpCm_true": dict(Counter(o.get("qform") or "unresolved" for o in Bm)),
                 "DpCm_true_conc_rev": int(sum(bool(o.get("conc_rev")) for o in Bm)),
                 "ref_DpCp_true": dict(Counter(o.get("qform") or "unresolved" for o in Rf)),
                 "ref_conc_rev": int(sum(bool(o.get("conc_rev")) for o in Rf)), "n_ref": len(Rf)}
        # failure rate by form: wrong = D+C- true (all), right = D+C+ true (all) x reference form shares
        n_right = sum(1 for o in R if o["dd"] and o["has"] and o["true"] and (m == "all" or o["model"] == m))
        refres = [o.get("qform") for o in Rf if o.get("qform")]
        rates = {}
        for f in ("identical", "converse", "swap", "contrary"):
            w = qf[m]["DpCm_true"].get(f, 0)
            p = (sum(1 for v in refres if v == f) / len(refres)) if refres else float("nan")
            rights = n_right * p
            rates[f] = {"wrong": w, "right_est": round(rights, 1),
                        "p_wrong_pct": round(100 * w / (w + rights), 2) if (w + rights) > 0 else None}
        qf[m]["p_wrong_given_true_by_form"] = rates
        qf[m]["n_right_true"] = n_right
    res["question_forms"] = qf
    # print
    for m in M7:
        for ds in DS3:
            c = res["cells"][f"{m}|{ds}"]
            lines.append(fmt_cell(f"{m}|{ds}", c))
        c = res["models"][m]
        lines.append(fmt_cell(f"{m}|POOLED", c) + variants(c))
    for ds in DS3 + ("all",):
        lines.append(fmt_cell(f"ALL|{ds}", res["pooled"][ds]) + variants(res["pooled"][ds]))
    for m in M7 + ["all"]:
        x = xt[m]
        lines.append(f"FALSEREL {m:14s} n {x['n']} basis {x['basis']} affirms {x['affirms_pct']:.1f} lower {x['names_lower_pct']:.1f} "
                     f"larger {x['names_larger_pct']:.1f} | on-top wording {x['ontop_wording_pct']:.1f} on-top+upper {x['ontop_upper_pct']:.1f} (sentence found {x['lrs_found_pct']:.0f}%) | affirms by basis {{{', '.join(f'{k}: {v:.0f}' for k, v in x['affirms_by_basis'].items())}}}")
    for m in M7 + ["all"]:
        x = qf[m]
        lines.append(f"QFORM {m:14s} D+C- true {x['DpCm_true']} (conc rev {x['DpCm_true_conc_rev']}) | ref D+C+ {x['ref_DpCp_true']} "
                     f"(n {x['n_ref']}, conc rev {x['ref_conc_rev']}) | P(wrong|true,form) "
                     + ", ".join(f"{f} {v['p_wrong_pct']}" for f, v in x["p_wrong_given_true_by_form"].items()))
    # sanity vs Exp-45 cells
    e45 = json.load(open(B / "e45_uniform_two_channel.json"))["cells"]
    bad = []
    for m in M7:
        for ds in DS3:
            ref = e45[f"open|{m}|{ds}"]
            n_dm = sum(o["dm"] for o in R if o["model"] == m and o["ds"] == ds)
            n_dp = sum(o["dp"] for o in R if o["model"] == m and o["ds"] == ds)
            if (n_dm, n_dp) != (ref["DpCm"], ref["DmCp"]):
                bad.append((m, ds, n_dm, ref["DpCm"], n_dp, ref["DmCp"]))
    res["sanity_vs_e45"] = {"mismatch": bad}
    lines.append(f"SANITY D+C- / D-C+ counts vs Exp-45 cells: {len(bad)} mismatches {bad}")
    res["_records"] = [{k: o.get(k) for k in ("model", "ds", "item", "cl", "dm", "dp", "has", "true", "claim", "affirms",
                                              "cat", "cat_af", "cat_ot", "cat_v3t", "cat_v3f", "qform", "conc_rev", "ontop_upper")}
                       for o in R if o["dm"] or o["dp"]]
    return res


def variants(c: dict) -> str:
    af, ot = c["DpCm_affirm_first"], c["DpCm_ontop"]
    return (f" | affirm-first: aff {af['a_affirm']['pct']:.1f} ht {af['a_height']['pct']:.1f} sz {af['a_size']['pct']:.1f} "
            f"occ {af['a_occ']['pct']:.1f} oth {af['a_other']['pct']:.1f} | on-top=height: aff {ot['a_affirm']['pct']:.1f} "
            f"ht {ot['a_height']['pct']:.1f} [{ot['a_height']['ci'][0]:.1f}, {ot['a_height']['ci'][1]:.1f}] sz {ot['a_size']['pct']:.1f} "
            f"occ {ot['a_occ']['pct']:.1f} oth {ot['a_other']['pct']:.1f}")


def kappa(a: list, b: list) -> float:
    cats = sorted(set(a) | set(b), key=str)
    n = len(a)
    if not n:
        return float("nan")
    po = np.mean([x == y for x, y in zip(a, b)])
    pe = sum((a.count(c) / n) * (b.count(c) / n) for c in cats)
    return float((po - pe) / (1 - pe)) if pe < 1 else 1.0


def fmt_cell(name: str, c: dict) -> str:
    d, p = c["DpCm"], c["DmCp"]
    f = lambda s, k: f"{s[k]['pct']:.1f}"
    fc = lambda s, k: f"{s[k]['pct']:.1f} [{s[k]['ci'][0]:.1f}, {s[k]['ci'][1]:.1f}]"
    s = (f"DECOMP {name:22s} D+C- n {d['n']:4d}: a {fc(d, 'a')} (aff {f(d, 'a_affirm')} ht {f(d, 'a_height')} sz {f(d, 'a_size')} occ {f(d, 'a_occ')} oth {f(d, 'a_other')}) "
         f"b {fc(d, 'b')} (map {f(d, 'b_map')} rev {f(d, 'b_rev')} oth {f(d, 'b_other')}) c {f(d, 'c_none')} a/(a+b) {d['a_of_ab']['pct']:.1f} "
         f"| v3 trunc a {f(c['DpCm_v3_trunc'], 'a')} b {f(c['DpCm_v3_trunc'], 'b')} c {f(c['DpCm_v3_trunc'], 'c')}; "
         f"v3 full a {f(c['DpCm_v3_full'], 'a')} b {f(c['DpCm_v3_full'], 'b')} c {f(c['DpCm_v3_full'], 'c')} "
         f"|| D-C+ n {p['n']:4d}: a' {fc(p, 'a')} (occ {f(p, 'a_occ')} ht {f(p, 'a_height')} sz {f(p, 'a_size')} aff {f(p, 'a_affirm')} oth {f(p, 'a_other')}) "
         f"b' {fc(p, 'b')} (map {f(p, 'b_map')} rev {f(p, 'b_rev')} oth {f(p, 'b_other')}) c' {f(p, 'c_none')}")
    return s


# ============================================================================= Part B: attenuation null

def nll(X: np.ndarray, y: np.ndarray, b: np.ndarray) -> float:
    z = X @ b
    return float(-(y * norm.logcdf(z) + (1 - y) * norm.logcdf(-z)).sum())


def probit_ridge(X: np.ndarray, y: np.ndarray, ridge: float = 0.1, pen: np.ndarray | None = None) -> np.ndarray:
    """Probit MLE with ridge on columns flagged in pen (default: all but the first) = rigor_e17f.probit_fit."""
    pen = np.r_[0.0, np.ones(X.shape[1] - 1)] if pen is None else pen

    def f(b):
        z = X @ b
        return -(y * norm.logcdf(z) + (1 - y) * norm.logcdf(-z)).sum() + 0.5 * ridge * (pen * b ** 2).sum()

    def g(b):
        z = X @ b
        lam1 = np.exp(norm.logpdf(z) - norm.logcdf(z))
        lam0 = np.exp(norm.logpdf(z) - norm.logcdf(-z))
        return -(X.T @ (y * lam1 - (1 - y) * lam0)) + ridge * pen * b
    return minimize(f, np.zeros(X.shape[1]), jac=g, method="BFGS", options={"gtol": 1e-7, "maxiter": 500}).x


def constrained(X: np.ndarray, y: np.ndarray, bdir: np.ndarray, s_fixed: float | None) -> tuple[np.ndarray, float]:
    """Fit z = c0 + c1 * phrasing + s * (cues @ bdir_cues); s free (ML) or fixed. Returns (full b, s)."""
    lin = X[:, 2:5] @ bdir[2:5]
    if s_fixed is None:
        Z = np.column_stack([X[:, 0], X[:, 1], lin])
        c = probit_ridge(Z, y, pen=np.array([0.0, 1.0, 0.0]))
        s = float(c[2])
    else:
        Z = np.column_stack([X[:, 0], X[:, 1]])
        off = s_fixed * lin
        c = _probit_offset(Z, y, off)
        s = s_fixed
    return np.r_[c[0], c[1], s * bdir[2:5]], s


def _probit_offset(Z: np.ndarray, y: np.ndarray, off: np.ndarray, ridge: float = 0.1) -> np.ndarray:
    pen = np.array([0.0, 1.0])

    def f(b):
        z = Z @ b + off
        return -(y * norm.logcdf(z) + (1 - y) * norm.logcdf(-z)).sum() + 0.5 * ridge * (pen * b ** 2).sum()
    return minimize(f, np.zeros(Z.shape[1]), method="BFGS", options={"gtol": 1e-7}).x


def exp_acc(X: np.ndarray, truth: np.ndarray, b: np.ndarray) -> float:
    p = norm.cdf(X @ b)
    return float(np.mean(np.where(truth, p, 1 - p)))


def cue_data(m: str) -> dict:
    """Design, Direct / CoT responses for the cue set; written relation (consensus / v3 kept) aligned by item."""
    from rigor_e17f import glm_rows
    if m == "qwen3vl_8b_think":
        from rigor_e31 import load_arms
        A = load_arms(m)
        by = {c: A[c] for c in ("direct", "cot")}
    else:
        R = rj(O / "e10" / f"{m}_cues.jsonl")
        by = {c: {r["item_id"]: r for r in R if r["condition"] == c} for c in ("direct", "cot")}
    keys = sorted(set(by["direct"]) & set(by["cot"]))
    D = glm_rows([by["direct"][k] for k in keys], use_size=True)
    C = glm_rows([by["cot"][k] for k in keys], use_size=True)
    assert np.array_equal(D["X"], C["X"])
    rows = {r["item"]: r for r in rj(E45 / "items.jsonl") if r["group"] == "open" and r["model"] == m and r["ds"] == "cues"}
    recs = {r["item"]: r for r in shape_records(list(rows.values()), load_labels())}
    wc, wv = [], []
    for k in keys:
        r, row = recs[k], rows[k]
        wc.append(np.nan if not r["has"] else float(r["claim"] == row["x"]))
        wv.append(np.nan if row["v3_full"] is None else float(row["v3_full"] == row["x"]))
    # Direct's vertical reading: does the Direct answer name the lower shape as front?
    return {"keys": keys, "X": D["X"], "truth": D["truth"], "y_dir": D["y"], "y_cot": C["y"], "pair": np.array(D["pair"]),
            "y_w": {"consensus": np.array(wc), "v3_kept": np.array(wv)}}


def att_stats(d: dict, idx: np.ndarray, src: str, s_cot_ref: float | None = None) -> dict:
    X, tr = d["X"][idx], d["truth"][idx]
    bd = probit_ridge(X, d["y_dir"][idx])
    bc = probit_ridge(X, d["y_cot"][idx])
    yw_all = d["y_w"][src][idx]
    cov = ~np.isnan(yw_all)
    Xw, yw, trw = X[cov], yw_all[cov], tr[cov]
    bw = probit_ridge(Xw, yw)
    cd, cc, cw = bd[2:5], bc[2:5], bw[2:5]
    s_cot = float(cc @ cd / (cd @ cd))
    s_proj = float(cw @ cd / (cd @ cd))
    b_ml, s_ml = constrained(Xw, yw, bd, None)
    b_scot, _ = constrained(Xw, yw, bd, s_cot)
    b_sproj, _ = constrained(Xw, yw, bd, s_proj)
    # fully uniform: z_w = s * z_dir (all coefficients incl. intercept and phrasing scaled), s by ML
    zd = Xw @ bd
    sf = minimize(lambda v: nll(zd[:, None], yw, np.array([v[0]])), [0.5], method="Nelder-Mead").x[0]
    out = {"s_cot": s_cot, "s_proj": s_proj, "s_ml": s_ml, "s_full": float(sf),
           "b_dir_occ": cd[0], "b_dir_vert": cd[1], "b_dir_size": cd[2],
           "b_w_occ": cw[0], "b_w_vert": cw[1], "b_w_size": cw[2], "b_cot_occ": cc[0], "b_cot_vert": cc[1],
           "delta_occ": cw[0] - s_proj * cd[0], "delta_vert": cw[1] - s_proj * cd[1], "delta_size": cw[2] - s_proj * cd[2],
           "delta_occ_scot": cw[0] - s_cot * cd[0], "delta_vert_scot": cw[1] - s_cot * cd[1],
           "delta_occ_sml": cw[0] - s_ml * cd[0], "delta_vert_sml": cw[1] - s_ml * cd[1],
           "unif_occ_loss_pred": (1 - s_proj) * cd[0], "occ_loss_obs": cd[0] - cw[0],
           "lr": 2 * (nll(Xw, yw, b_ml) - nll(Xw, yw, bw)),
           "coverage": 100 * cov.mean(), "acc_dir_all": 100 * float(np.mean(d["y_dir"][idx] == tr)),
           "acc_dir_cov": 100 * float(np.mean(d["y_dir"][idx][cov] == trw)),
           "acc_cot_all": 100 * float(np.mean(d["y_cot"][idx] == tr)),
           "x_obs": 100 * float(np.mean(yw == trw)), "x_fit": 100 * exp_acc(Xw, trw, bw),
           "x_null_ml": 100 * exp_acc(Xw, trw, b_ml), "x_null_scot": 100 * exp_acc(Xw, trw, b_scot),
           "x_null_sproj": 100 * exp_acc(Xw, trw, b_sproj),
           "x_null_full": 100 * float(np.mean(np.where(trw, norm.cdf(sf * zd), 1 - norm.cdf(sf * zd))))}
    # Direct-fit expected accuracy on the covered items (model-check reference for x_null)
    out["dir_fit_cov"] = 100 * exp_acc(Xw, trw, bd)
    for k in ("ml", "scot", "sproj", "full"):
        out[f"gap_{k}"] = out["x_obs"] - out[f"x_null_{k}"]
    out["x"] = out["x_obs"] - out["acc_dir_cov"]
    out["x_null_minus_dir_ml"] = out["x_null_ml"] - out["acc_dir_cov"]
    # by vertical congruence (front shape lower = height cue valid / front higher = height cue misleading)
    xl = Xw[:, 3] > 0  # subject lower
    fl = (xl == trw)  # front shape is the lower one
    for nm, sel in (("front_lower", fl), ("front_higher", ~fl)):
        out[f"{nm}_n"] = int(sel.sum())
        out[f"{nm}_dir"] = 100 * float(np.mean(d["y_dir"][idx][cov][sel] == trw[sel]))
        out[f"{nm}_obs"] = 100 * float(np.mean(yw[sel] == trw[sel]))
        out[f"{nm}_null_ml"] = 100 * exp_acc(Xw[sel], trw[sel], b_ml)
        out[f"{nm}_null_scot"] = 100 * exp_acc(Xw[sel], trw[sel], b_scot)
        out[f"{nm}_gap_ml"] = out[f"{nm}_obs"] - out[f"{nm}_null_ml"]
    return out


BKEYS = None


def part_b(lines: list[str], nb: int) -> dict:
    rng = np.random.default_rng(5454)
    cs = json.load(open(B / "common_scale.json"))
    res = {"_doc": "Part B" + __doc__.split("Part B")[1].split("Outputs:")[0], "models": {}}
    models = M7 + ["qwen3vl_32b", "qwen3vl_8b_think"]
    for m in models:
        d = cue_data(m)
        pairs = np.unique(d["pair"])
        pidx = {p: np.where(d["pair"] == p)[0] for p in pairs}
        res["models"][m] = {}
        for src in ("consensus", "v3_kept"):
            est = att_stats(d, np.arange(len(d["keys"])), src)
            draws = defaultdict(list)
            for _ in range(nb):
                idx = np.concatenate([pidx[p] for p in rng.choice(pairs, len(pairs))])
                st = att_stats(d, idx, src)
                for k, v in st.items():
                    draws[k].append(v)
            o = {k: {"est": float(v), "ci": q95(np.array(draws[k], float))} for k, v in est.items()}
            o["lr_p"] = float(chi2.sf(est["lr"], 2))
            o["p_gap_ml_ge0"] = float(np.mean(np.array(draws["gap_ml"]) >= 0))
            res["models"][m][src] = o
            r = o
            ref = cs.get(m, {}).get("s", {}).get("est")
            lines.append(
                f"ATT {m:16s} {src:9s} cov {r['coverage']['est']:.1f} | s_cot {r['s_cot']['est']:.3f} (paper {ref}) s_proj {r['s_proj']['est']:.3f} "
                f"{fci(r['s_proj'], 3)} s_ML {r['s_ml']['est']:.3f} {fci(r['s_ml'], 3)} | b_dir occ/vert/size {r['b_dir_occ']['est']:.2f}/{r['b_dir_vert']['est']:.2f}/{r['b_dir_size']['est']:.2f} "
                f"b_w {r['b_w_occ']['est']:.2f}/{r['b_w_vert']['est']:.2f}/{r['b_w_size']['est']:.2f} | d_occ {r['delta_occ']['est']:+.2f} {fci(r['delta_occ'])} "
                f"d_vert {r['delta_vert']['est']:+.2f} {fci(r['delta_vert'])} (s_cot: {r['delta_occ_scot']['est']:+.2f} {fci(r['delta_occ_scot'])} / "
                f"{r['delta_vert_scot']['est']:+.2f} {fci(r['delta_vert_scot'])}) | occ loss obs {r['occ_loss_obs']['est']:.2f} unif {r['unif_occ_loss_pred']['est']:.2f} "
                f"| LR {r['lr']['est']:.1f} p {o['lr_p']:.2g} | Dir(cov) {r['acc_dir_cov']['est']:.1f} x_obs {r['x_obs']['est']:.1f} x_fit {r['x_fit']['est']:.1f} "
                f"x_null ML {r['x_null_ml']['est']:.1f} s_cot {r['x_null_scot']['est']:.1f} s_proj {r['x_null_sproj']['est']:.1f} full {r['x_null_full']['est']:.1f} "
                f"| gap ML {r['gap_ml']['est']:+.1f} {fci(r['gap_ml'], 1)} s_cot {r['gap_scot']['est']:+.1f} {fci(r['gap_scot'], 1)} "
                f"full {r['gap_full']['est']:+.1f} {fci(r['gap_full'], 1)} | x {r['x']['est']:+.1f} {fci(r['x'], 1)} null x {r['x_null_minus_dir_ml']['est']:+.1f} "
                f"| front-lower n {r['front_lower_n']['est']:.0f} dir {r['front_lower_dir']['est']:.1f} obs {r['front_lower_obs']['est']:.1f} null {r['front_lower_null_ml']['est']:.1f} "
                f"gap {r['front_lower_gap_ml']['est']:+.1f} {fci(r['front_lower_gap_ml'], 1)} | front-higher n {r['front_higher_n']['est']:.0f} dir {r['front_higher_dir']['est']:.1f} "
                f"obs {r['front_higher_obs']['est']:.1f} null {r['front_higher_null_ml']['est']:.1f} gap {r['front_higher_gap_ml']['est']:+.1f} {fci(r['front_higher_gap_ml'], 1)}")
        print(lines[-2], "\n", lines[-1], flush=True)
    # sanity: paper common scale (CoT vs Direct, point estimates)
    san = {m: {"s_cot_here": res["models"][m]["consensus"]["s_cot"]["est"], "s_paper": cs[m]["s"]["est"]} for m in M7}
    res["sanity_common_scale"] = san
    lines.append("SANITY s_cot vs common_scale.json: " + ", ".join(f"{m} {v['s_cot_here']:.3f}/{v['s_paper']:.3f}" for m, v in san.items()))
    return res


def fci(r: dict, nd: int = 2) -> str:
    return f"[{r['ci'][0]:.{nd}f}, {r['ci'][1]:.{nd}f}]"


# ============================================================================= tables

SHORT = {"qwen3b": "Qwen2.5-VL-3B", "qwen7b": "Qwen2.5-VL-7B", "internvl3_8b": "InternVL3-8B", "phi35v": "Phi-3.5-V",
         "molmo7b": "Molmo-7B-D", "qwen3vl_8b": "Qwen3-VL-8B", "internvl35_8b": "InternVL3.5-8B",
         "qwen3vl_32b": "Qwen3-VL-32B", "qwen3vl_8b_think": "Qwen3-VL-8B-Think."}


def table_a(res: dict) -> str:
    def p(s, k):
        return f"{s[k]['pct']:.0f}"

    def pc(s, k):
        return f"{s[k]['pct']:.0f} [{s[k]['ci'][0]:.0f}, {s[k]['ci'][1]:.0f}]"

    L = [r"% Exp-54 (rigor_e54.py --only a). Suggested caption: Decomposition of CoT losses (D+C$-$: Direct right, CoT",
         r"% wrong) and rescues (D$-$C+) by the last written relation of the conclusion-removed rationale (Exp-45 two-judge",
         r"% consensus). (a) false relation (writing wrong), split by the reason the text gives (two-judge consensus):",
         r"% affirmation = no visual reason and the relation affirms the queried subject; height = vertical position",
         r"% (incl. 'on top' = higher); size = larger is closer; occl. = misread overlap; other = other reason or",
         r"% judges disagree. (b) true relation, wrong answer (answering wrong): mapping = the question is not the written",
         r"% proposition in its own words (other shape, converse or contrary predicate); reversal = question restated,",
         r"% the removed conclusion states the opposite answer; other. (c) no relation. Shares in % of D+C$-$ with",
         r"% item-cluster bootstrap 95% CIs (2,000 draws). Mirror: D$-$C+ with a true relation (a'), a false relation",
         r"% but a right answer (b') and no relation (c'). Pooled rows resample clusters within each model x set.",
         r"% Counting 'on top / over / above' wording that names the upper shape as height (rule) raises height to 10% of",
         r"% all D+C$-$ (Qwen3-VL-8B 11%); counting the conclusion as written relation (v3) moves Molmo-7B-D to 61% (a).",
         r"\begin{tabular}{llr rrrrrr rrrr r rrrr}", r"\toprule",
         r" & & & \multicolumn{6}{c}{(a) false relation} & \multicolumn{4}{c}{(b) true relation, wrong answer} & (c) & "
         r"\multicolumn{4}{c}{D$-$C+ mirror} \\",
         r"\cmidrule(lr){4-9}\cmidrule(lr){10-13}\cmidrule(lr){15-18}",
         r"Model & Set & $n$ & all & aff. & height & size & occl. & other & all & map. & rev. & other & none & $n$ & (a') & (b') & (c') \\",
         r"\midrule"]
    for m in M7:
        for ds in DS3 + ("pooled",):
            c = res["models"][m] if ds == "pooled" else res["cells"][f"{m}|{ds}"]
            d, q = c["DpCm"], c["DmCp"]
            name = SHORT[m] if ds == "cues" else ""
            dsn = r"\textit{pooled}" if ds == "pooled" else DSN[ds]
            L.append(f"{name} & {dsn} & {d['n']} & {pc(d, 'a')} & {p(d, 'a_affirm')} & {p(d, 'a_height')} & {p(d, 'a_size')} & "
                     f"{p(d, 'a_occ')} & {p(d, 'a_other')} & {pc(d, 'b')} & {p(d, 'b_map')} & {p(d, 'b_rev')} & {p(d, 'b_other')} & {p(d, 'c_none')} & "
                     f"{q['n']} & {p(q, 'a')} & {p(q, 'b')} & {p(q, 'c')} \\\\")
        L.append(r"\midrule")
    d, q = res["pooled"]["all"]["DpCm"], res["pooled"]["all"]["DmCp"]
    L.append(f"All seven & all sets & {d['n']} & {pc(d, 'a')} & {p(d, 'a_affirm')} & {p(d, 'a_height')} & {p(d, 'a_size')} & "
             f"{p(d, 'a_occ')} & {p(d, 'a_other')} & {pc(d, 'b')} & {p(d, 'b_map')} & {p(d, 'b_rev')} & {p(d, 'b_other')} & {p(d, 'c_none')} & "
             f"{q['n']} & {p(q, 'a')} & {p(q, 'b')} & {p(q, 'c')} \\\\")
    L += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(L) + "\n"


def table_b(res: dict) -> str:
    L = [r"% Exp-54 (rigor_e54.py --only b). Suggested caption: Uniform-attenuation null for the written relation on the",
         r"% cue-factorial set. Probit cue model (phrasing, occlusion, vertical, size; Tab. supp_common_scale) fitted to Direct",
         r"% answers and to the answer implied by the last written relation (conclusion removed, two-judge consensus).",
         r"% $s_{\mathrm{CoT}}$: paper's common scale (CoT vs Direct); $s_{\mathrm{ML}}$: written cue weights constrained to",
         r"% $s\,b_{\mathrm{Dir}}$ (intercept, phrasing free), LR test vs free weights (df 2). $\delta_k = b_{\mathrm{W},k} -",
         r"% s_{\mathrm{proj}} b_{\mathrm{Dir},k}$. Dir.: Direct accuracy on the covered items; $x_{\mathrm{obs}}$: written-relation",
         r"% accuracy; $x_{\mathrm{null}}$: expected written accuracy under $s_{\mathrm{ML}} b_{\mathrm{Dir}}$; gap $= x_{\mathrm{obs}} -",
         r"% x_{\mathrm{null}}$ (and with $s_{\mathrm{CoT}}$). v3: gap with the rule-parsed last relation, conclusion kept",
         r"% (Exp-32 definition). Twin-pair bootstrap 95% CIs (1,000 draws). Lower block: Exp-31 models (not in the paper's seven).",
         r"\begin{tabular}{l rrrr rr rrr rr rrr}", r"\toprule",
         r"Model & $s_{\mathrm{CoT}}$ & $s_{\mathrm{ML}}$ & LR $\chi^2$ & $p$ & $\delta_{\mathrm{occ}}$ & $\delta_{\mathrm{vert}}$ & "
         r"Dir. & $x_{\mathrm{obs}}$ & $x_{\mathrm{null}}$ & $x_{\mathrm{obs}}-$Dir. & $x_{\mathrm{null}}-$Dir. & "
         r"gap ($s_{\mathrm{ML}}$) & gap ($s_{\mathrm{CoT}}$) & gap v3 \\", r"\midrule"]
    for m in M7 + ["qwen3vl_32b", "qwen3vl_8b_think"]:
        if m == "qwen3vl_32b":
            L.append(r"\midrule")
        r = res["models"][m]["consensus"]
        v = res["models"][m]["v3_kept"]
        f2 = lambda k: f"{r[k]['est']:+.2f} [{r[k]['ci'][0]:+.2f}, {r[k]['ci'][1]:+.2f}]"
        f1 = lambda s, k: f"{s[k]['est']:+.1f} [{s[k]['ci'][0]:+.1f}, {s[k]['ci'][1]:+.1f}]"
        pv = r["lr_p"]
        ps = "$<$0.001" if pv < 1e-3 else f"{pv:.3f}"
        row = (f" & {r['s_cot']['est']:.2f} & {r['s_ml']['est']:.2f} & {r['lr']['est']:.1f} & {ps} & {f2('delta_occ')} & "
               f"{f2('delta_vert')} & {r['acc_dir_cov']['est']:.1f} & {r['x_obs']['est']:.1f} & {r['x_null_ml']['est']:.1f} & "
               f"{f1(r, 'x')} & {r['x_null_minus_dir_ml']['est']:+.1f} & {f1(r, 'gap_ml')} & {f1(r, 'gap_scot')} & {f1(v, 'gap_ml')} \\\\")
        L.append(SHORT[m] + re.sub(r"(?<![\w.])-(?=\d)", "$-$", row))
    L += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(L) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="a,b")
    ap.add_argument("--nb", type=int, default=2000)
    ap.add_argument("--nb-b", type=int, default=1000)
    a = ap.parse_args()
    txt = B / "rigor_e54.txt"
    old = txt.read_text().splitlines() if txt.exists() else []
    lines: list[str] = []
    parts = a.only.split(",")
    if "a" in parts:
        ra = part_a(lines, a.nb)
        json.dump(ra, open(B / "e54_decomp.json", "w"), indent=1, default=float)
        (PAPER / "tables" / "supp_e54a.tex").write_text(table_a(ra))
    if "btable" in parts:
        (PAPER / "tables" / "supp_e54b.tex").write_text(table_b(json.load(open(B / "e54_attenuation.json"))))
        parts.remove("btable")
    if "b" in parts:
        rb = part_b(lines, a.nb_b)
        json.dump(rb, open(B / "e54_attenuation.json", "w"), indent=1, default=float)
        (PAPER / "tables" / "supp_e54b.tex").write_text(table_b(rb))
    pref = {"a": ("LABELS", "JUDGE", "DECOMP", "FALSEREL", "QFORM", "SANITY D+C-"), "b": ("ATT", "SANITY s_cot")}
    keep = [x for x in old if not any(x.startswith(pref[p]) for p in parts)]
    txt.write_text("\n".join(keep + lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
