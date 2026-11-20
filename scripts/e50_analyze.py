"""Exp-50 analysis: closed models under real / gray / twin images (answers + judge-extracted written relation).

-> outputs/bundle/e50_closed_image_blind.json, outputs/bundle/rigor_e50.txt, paper/tables/supp_e50.tex
"""
from __future__ import annotations
import vtax_paths as _vtp  # noqa: E402

import json
import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from e50_closed_image_blind import CLOSED, CONDS, E50, JUDGES, O, load_gen, rj
from e50_judge import caches, label_of, text_records

B = Path("outputs/bundle")
PAPER = Path(f"{_vtp.PAPER}")
NAMES = {"claude-opus-5-5": "Claude Opus 5.5", "gpt-6-astra": "GPT-6-Astra", "gemini-4-argon": "Gemini-4-Argon",
         "qwen3.8-max-0902": "Qwen3.8-Max", "hy4": "HY4", "deepseek-v4.1-flash": "DeepSeek-V4.1-Flash"}
SETN = {"hard": "Hard occl.", "cues": "Cue set"}
PROTOS = ("kept", "removed")
NBOOT = 2000
CANNOT = re.compile(r"cannot (?:be )?(?:determine|tell|see|identify|verify|confirm|assess|answer)|can(?:'|’)t (?:determine|tell|see|verify)|"
                    r"unable to|not possible to (?:determine|tell)|impossible to (?:determine|tell)|no way to (?:determine|tell)|"
                    r"insufficient|not enough information|no (?:visible )?(?:shapes?|objects?)\b|not visible|"
                    r"(?:does|do) not (?:show|contain|depict)|isn(?:'|’)t (?:any|visible)|nothing (?:is )?visible", re.I)
BLANK = re.compile(r"\b(?:uniform(?:ly)?|solid|plain|blank|empty|featureless)\b[^.\n]{0,40}\b(?:gr[ae]y|image|background|field|canvas)|"
                   r"\bgr[ae]y (?:image|square|background|field|canvas|rectangle)\b|\bentirely gr[ae]y\b|\bonly gr[ae]y\b", re.I)


def implied_letter(rel: str | None, x: str, y: str, phrasing: str) -> str | None:
    """Answer letter implied by a written relation naming `rel` in front (A = yes)."""
    if rel not in (x, y):
        return None
    yes = (rel == x) if phrasing == "front" else (rel == y)
    return "A" if yes else "B"


def boot_ci(groups: list, fn, n: int = NBOOT, seed: int = 50) -> list[float]:
    """Percentile CI of fn(list of cluster ids) over a cluster bootstrap."""
    rng = np.random.default_rng(seed)
    u = sorted(set(groups))
    vals = []
    for _ in range(n):
        pick = rng.choice(len(u), len(u), replace=True)
        v = fn([u[i] for i in pick])
        if v is not None and not np.isnan(v):
            vals.append(v)
    return [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))] if vals else [float("nan")] * 2


def kappa(a: list, b: list) -> float:
    cats = sorted(set(a) | set(b))
    n = len(a)
    if not n:
        return float("nan")
    po = sum(x == y for x, y in zip(a, b)) / n
    ca, cb = Counter(a), Counter(b)
    pe = sum(ca[c] * cb[c] for c in cats) / n / n
    return (po - pe) / (1 - pe) if pe < 1 else 1.0


def analyze() -> None:
    S = json.load(open(E50 / "sample.json"))
    P = {(p["set"], p["uid"]): p for v in S.values() for p in v}
    G = {(r["model"], r["set"], r["cond"], r["uid"]): r for r in load_gen()}
    T = {(r["model"], r["set"], r["cond"], r["uid"]): r for r in text_records()}
    L = {j: caches(j) for j in JUDGES}
    lab = defaultdict(dict)   # (model, set, cond, uid) -> {(judge, proto): (label, src)}
    for k, r in T.items():
        for j in JUDGES:
            for pr in PROTOS:
                lab[k][(j, pr)] = label_of(r, pr, *L[j])

    from e45_uniform_two_channel import hkey
    PRES = {j: {r["hkey"]: r["label"] for r in rj(E50 / f"presence_{j}.jsonl") if r.get("label")} for j in JUDGES}

    def consensus(k, pr):
        a, b = lab[k][(JUDGES[0], pr)][0], lab[k][(JUDGES[1], pr)][0]
        return a if (a is not None and a == b) else None

    # per prompt rows
    rows = []
    for m in CLOSED:
        for (st, uid), p in P.items():
            for cond in CONDS:
                k = (m, st, cond, uid)
                g = G.get(k)
                x, front = p["meta"]["x_name"].lower(), p["meta"]["front_name"].lower()
                back = p["meta"]["back_name"].lower()
                y = back if x == front else front
                img_front = p["twin_meta"]["front_name"].lower() if cond == "twin" else front
                row = {"model": m, "set": st, "cond": cond, "uid": uid, "group": p["group"], "phrasing": p["phrasing"],
                       "gold": p["gold"], "x": x, "y": y, "real_front": front, "img_front": img_front,
                       "missing": g is None, "api_error": bool(g and "error" in g),
                       "error_msg": (g or {}).get("error", "")[:120] if g and "error" in g else "",
                       "empty": bool(g and "error" not in g and not (g.get("text") or "").strip()),
                       "finish_reason": (g or {}).get("finish_reason"), "source": (g or {}).get("source"),
                       "pred": (g or {}).get("pred") if g and "error" not in g else None,
                       "reasoning_tokens": (g or {}).get("reasoning_tokens"),
                       "text_chars": len((g or {}).get("text") or "")}
                txt = (g or {}).get("text") or ""
                body = re.split(r"Answer:", txt)[0]
                row["cannot"] = bool(CANNOT.search(body))
                row["blank"] = bool(BLANK.search(body))
                hp = hkey(x, y, txt, "presence")
                row["pres"] = [PRES[j].get(hp) for j in JUDGES] if txt.strip() else [None, None]
                row["pres_cons"] = row["pres"][0] if (row["pres"][0] and row["pres"][0] == row["pres"][1]) else None
                for pr in PROTOS:
                    if k in T:
                        la = lab[k][(JUDGES[0], pr)][0]
                        lb = lab[k][(JUDGES[1], pr)][0]
                        rel = consensus(k, pr)
                        rel = rel if rel in (x, y) else None
                        row[f"lab_{pr}"] = [la, lb]
                        row[f"src_{pr}"] = [lab[k][(JUDGES[0], pr)][1], lab[k][(JUDGES[1], pr)][1]]
                    else:
                        la = lb = rel = None
                        row[f"lab_{pr}"] = [None, None]
                        row[f"src_{pr}"] = ["", ""]
                    row[f"rel_{pr}"] = rel
                    row[f"either_{pr}"] = any(v in (x, y) for v in (la, lb))
                    row[f"unlab_{pr}"] = la is None or lb is None
                    row[f"impl_{pr}"] = implied_letter(rel, x, y, p["phrasing"])
                rows.append(row)

    def cell(m, st, cond):
        return [r for r in rows if r["model"] == m and r["set"] == st and r["cond"] == cond]

    def pct(v):
        return 100.0 * float(np.mean(v)) if len(v) else float("nan")

    res = {"cells": {}, "image_effect": {}, "judges": {}, "failures": {}}
    for m in CLOSED:
        for st in ("hard", "cues"):
            byc = {c: {r["uid"]: r for r in cell(m, st, c)} for c in CONDS}
            for cond in CONDS:
                rs = list(byc[cond].values())
                n = len(rs)
                valid = [r for r in rs if r["pred"] is not None]
                d = {"n": n, "valid": len(valid), "api_error": sum(r["api_error"] for r in rs),
                     "missing": sum(r["missing"] for r in rs), "empty": sum(r["empty"] for r in rs),
                     "content_filter": sum(r["finish_reason"] == "content_filter" for r in rs),
                     "no_letter": sum((not r["api_error"]) and (not r["missing"]) and r["pred"] is None for r in rs),
                     "acc_all": pct([r["pred"] == r["gold"] for r in rs]),
                     "acc_valid": pct([r["pred"] == r["gold"] for r in valid]),
                     "share_A": pct([r["pred"] == "A" for r in rs]), "share_B": pct([r["pred"] == "B" for r in rs]),
                     "share_none": pct([r["pred"] is None for r in rs]),
                     "cannot": pct([r["cannot"] for r in rs if not r["api_error"]]),
                     "cannot_no_letter": sum(r["cannot"] and r["pred"] is None and not r["api_error"] for r in rs),
                     "blank_mention": pct([r["blank"] for r in rs if not r["api_error"]]),
                     "median_reasoning_tokens": float(np.median([r["reasoning_tokens"] or 0 for r in rs if not r["api_error"]])) if rs else None,
                     "sources": dict(Counter(r["source"] for r in rs))}
                grp = [r["group"] for r in rs]
                byg = defaultdict(list)
                for r in rs:
                    byg[r["group"]].append(r)
                d["acc_all_ci"] = boot_ci(grp, lambda gs: pct([r["pred"] == r["gold"] for g in gs for r in byg[g]]))
                pj = [r for r in rs if r["pres"][0] and r["pres"][1]]
                d["presence"] = {"n_judged": len(pj), "n_consensus": sum(r["pres_cons"] is not None for r in pj),
                                 **{c: pct([r["pres_cons"] == c for r in pj]) for c in ("VISIBLE_RELATION", "NOT_VISIBLE", "OTHER")},
                                 "either_visible_relation": pct([("VISIBLE_RELATION" in r["pres"]) for r in pj]),
                                 "rel_kept_and_visible": sum(r["rel_kept"] is not None and r["pres_cons"] == "VISIBLE_RELATION" for r in pj),
                                 "rel_kept_and_not_visible": sum(r["rel_kept"] is not None and r["pres_cons"] == "NOT_VISIBLE" for r in pj)}
                # gray answer bias: yes-share by phrasing
                for ph in ("front", "behind"):
                    d[f"share_A_{ph}"] = pct([r["pred"] == "A" for r in rs if r["phrasing"] == ph])
                for pr in PROTOS:
                    cov = [r for r in rs if r[f"rel_{pr}"] is not None]
                    d[pr] = {"coverage": pct([r[f"rel_{pr}"] is not None for r in rs]),
                             "n_rel": len(cov),
                             "either_judge": pct([r[f"either_{pr}"] for r in rs]),
                             "unlabelled": sum(r[f"unlab_{pr}"] for r in rs),
                             "written_acc_img": pct([r[f"rel_{pr}"] == r["img_front"] for r in cov]),
                             "written_acc_real": pct([r[f"rel_{pr}"] == r["real_front"] for r in cov]),
                             "rel_names_x": pct([r[f"rel_{pr}"] == r["x"] for r in cov]),
                             "implied_eq_gold": pct([r[f"impl_{pr}"] == r["gold"] for r in cov]),
                             "agree": pct([r[f"impl_{pr}"] == r["pred"] for r in cov if r["pred"] is not None]),
                             "agree_true": pct([r[f"impl_{pr}"] == r["pred"] for r in cov
                                                if r["pred"] is not None and r[f"rel_{pr}"] == r["img_front"]]),
                             "agree_false": pct([r[f"impl_{pr}"] == r["pred"] for r in cov
                                                 if r["pred"] is not None and r[f"rel_{pr}"] != r["img_front"]]),
                             "n_false": sum(r[f"rel_{pr}"] != r["img_front"] for r in cov),
                             "n_agree_den": sum(r["pred"] is not None for r in cov)}
                res["cells"][f"{m}|{st}|{cond}"] = d
            # image effects (paired by prompt)
            uids = sorted(set(byc["real"]) & set(byc["twin"]) & set(byc["gray"]))
            grp = {u: byc["real"][u]["group"] for u in uids}
            by_g = defaultdict(list)
            for u in uids:
                by_g[grp[u]].append(u)

            def eff(gs, a="real", b="twin", key="pred"):
                us = [u for g in gs for u in by_g[g]]
                return pct([byc[a][u][key] == byc[a][u]["gold"] for u in us]) - pct([byc[b][u][key] == byc[b][u]["gold"] for u in us])

            e = {"n": len(uids), "answer_twin": eff(list(by_g)), "answer_twin_ci": boot_ci(list(by_g), eff),
                 "answer_gray": eff(list(by_g), "real", "gray"),
                 "answer_gray_ci": boot_ci(list(by_g), lambda gs: eff(gs, "real", "gray"))}
            for pr in PROTOS:
                kk = f"impl_{pr}"
                e[f"written_twin_{pr}"] = eff(list(by_g), key=kk)
                e[f"written_twin_{pr}_ci"] = boot_ci(list(by_g), lambda gs, kk=kk: eff(gs, key=kk))
                both = [u for u in uids if byc["real"][u][kk] and byc["twin"][u][kk]]
                e[f"written_twin_{pr}_both_n"] = len(both)
                e[f"written_twin_{pr}_both"] = (pct([byc["real"][u][kk] == byc["real"][u]["gold"] for u in both])
                                                 - pct([byc["twin"][u][kk] == byc["twin"][u]["gold"] for u in both]))
            # valid-only answer image effect
            vb = [u for u in uids if byc["real"][u]["pred"] and byc["twin"][u]["pred"]]
            e["answer_twin_valid_n"] = len(vb)
            e["answer_twin_valid"] = (pct([byc["real"][u]["pred"] == byc["real"][u]["gold"] for u in vb])
                                      - pct([byc["twin"][u]["pred"] == byc["twin"][u]["gold"] for u in vb]))
            if st == "hard":
                # Direct (letter only, reasoning high) on the same prompts and their twins, e9h replies
                dr = {}
                for r in rj(O / "e9h" / f"{m}.jsonl"):
                    if r["condition"] == "direct" and "error" not in r and (r.get("pred") or r["item_id"] not in dr):
                        dr[r["item_id"]] = r
                tw = {p["uid"]: p["twin_uid"] for p in S["hard"]}

                def eff_d(gs):
                    us = [u for g in gs for u in by_g[g]]
                    a = pct([(dr.get(u) or {}).get("pred") == byc["real"][u]["gold"] for u in us])
                    b = pct([(dr.get(tw[u]) or {}).get("pred") == byc["real"][u]["gold"] for u in us])
                    return a - b

                e["direct_twin"] = eff_d(list(by_g))
                e["direct_twin_ci"] = boot_ci(list(by_g), eff_d)
                e["direct_acc_real"] = pct([(dr.get(u) or {}).get("pred") == byc["real"][u]["gold"] for u in uids])
                e["cot_minus_direct_twin"] = e["answer_twin"] - e["direct_twin"]
                e["cot_minus_direct_twin_ci"] = boot_ci(list(by_g), lambda gs: eff(gs) - eff_d(gs))
            res["image_effect"][f"{m}|{st}"] = e

    # judges: 4-way agreement on called / cached labels (prefilter excluded)
    def lab4(v, r):
        return "x" if v == r["x"] else "y" if v == r["y"] else v

    for pr in PROTOS:
        for scope in ("all",) + CONDS:
            rs = [r for r in rows if (scope == "all" or r["cond"] == scope) and None not in r[f"lab_{pr}"]
                  and "prefilter" not in r[f"src_{pr}"]]
            a = [lab4(r[f"lab_{pr}"][0], r) for r in rs]
            b = [lab4(r[f"lab_{pr}"][1], r) for r in rs]
            res["judges"][f"{pr}|{scope}"] = {"n": len(rs), "agree": pct([u == v for u, v in zip(a, b)]), "kappa": kappa(a, b),
                                              "opposite_shape": sum({u, v} == {"x", "y"} for u, v in zip(a, b)),
                                              "shape_vs_none": sum((u in "xy") != (v in "xy") for u, v in zip(a, b)),
                                              "src": dict(Counter("/".join(r[f"src_{pr}"]) for r in rs).most_common(6))}
        res["judges"][f"{pr}|prefiltered_texts"] = sum("prefilter" in r[f"src_{pr}"] for r in rows)
        res["judges"][f"{pr}|unlabelled_texts"] = sum(r[f"unlab_{pr}"] and not r["api_error"] and not r["missing"] for r in rows)

    for scope in CONDS:
        rs = [r for r in rows if r["cond"] == scope and r["pres"][0] and r["pres"][1]]
        a, b = [r["pres"][0] for r in rs], [r["pres"][1] for r in rs]
        res["judges"][f"presence|{scope}"] = {"n": len(rs), "agree": pct([u == v for u, v in zip(a, b)]), "kappa": kappa(a, b),
                                              "pairs": dict(Counter(f"{u}/{v}" for u, v in zip(a, b)))}
    gray = [r for r in rows if r["cond"] == "gray"]
    for pr in PROTOS:
        for ji, j in enumerate(JUDGES):
            res["judges"][f"{pr}|gray_shape_rate|{j}"] = {
                m: pct([r[f"lab_{pr}"][ji] in (r["x"], r["y"]) for r in gray if r["model"] == m]) for m in CLOSED}
    gv = [r for r in gray if r["pres_cons"] == "VISIBLE_RELATION"]
    res["gray_visible_relation"] = {
        "n": len(gv), "by_model_set": dict(Counter(f"{r['model']}|{r['set']}" for r in gv)),
        "with_consensus_relation": sum(r["rel_kept"] is not None for r in gv),
        "relation_eq_real_front": sum(r["rel_kept"] == r["real_front"] for r in gv if r["rel_kept"]),
        "relation_names_queried_x": sum(r["rel_kept"] == r["x"] for r in gv if r["rel_kept"]),
        "answer_agrees_relation": sum(r["impl_kept"] == r["pred"] for r in gv if r["rel_kept"]),
        "answer_yes": sum(r["pred"] == "A" for r in gv), "answer_eq_real_gold": sum(r["pred"] == r["gold"] for r in gv),
        "gray_yes_answers_total": dict(Counter(r["model"] for r in gray if r["pred"] == "A")),
        "gray_yes_with_visible_relation": dict(Counter(r["model"] for r in gv if r["pred"] == "A"))}
    att = Counter()
    for m in CLOSED:
        for r in rj(E50 / f"gen_{m}.jsonl"):
            if "error" in r:
                att[f"{m}: {r['error'][:70]}"] += 1
    res["gen_error_rows_all_attempts"] = dict(att)
    # failures
    err = Counter((r["model"], r["error_msg"][:60]) for r in rows if r["api_error"])
    res["failures"] = {"api_errors": {f"{k[0]}: {k[1]}": v for k, v in err.items()},
                       "content_filter": dict(Counter(r["model"] for r in rows if r["finish_reason"] == "content_filter")),
                       "empty": dict(Counter(f"{r['model']}|{r['set']}|{r['cond']}" for r in rows if r["empty"])),
                       "no_letter": dict(Counter(f"{r['model']}|{r['set']}|{r['cond']}" for r in rows
                                                 if r["pred"] is None and not r["api_error"] and not r["missing"])),
                       "finish_reasons": dict(Counter(f"{r['model']}:{r['finish_reason']}" for r in rows if r["source"] == "e50"))}
    # gray examples per model (first 2 texts per set)
    ex = {}
    for m in CLOSED:
        for st in ("hard", "cues"):
            ks = sorted(k for k in G if k[0] == m and k[1] == st and k[2] == "gray" and "error" not in G[k])[:2]
            ex[f"{m}|{st}"] = [(G[k].get("text") or "")[:300] for k in ks]
    res["gray_examples"] = ex
    res["open_model_reference"] = {
        "image_effect_direct_2acc_minus_1": "3 to 76 pts (76 Qwen3-VL-8B, 62 Molmo-7B-D); faithfulness.tex",
        "image_effect_fixed_rationale_conclusion_removed": "2.0 to 8.9 pts (Molmo 8.9, Qwen3-VL 8.4); Tab. supp_answer_step",
        "image_effect_fixed_full_rationale": "<= 0.4 pts in five models, 6.8 / 8.5 in InternVL3 / InternVL3.5; Tab. supp_recheck",
        "gray_prefix_readout_e30e": "gray j* = real j*: Qwen3-VL-8B 59/103 (57.3%), Molmo 63/77 (81.8%); within one sentence 83.5 / 93.5%;"
                                    " P(gold) before j* real/gray 90.6/59.9 (Qwen3-VL), 72.5/54.5 (Molmo); Tab. supp_e30 (e)"}
    res["rows"] = rows
    B.mkdir(exist_ok=True)
    json.dump(res, open(B / "e50_closed_image_blind.json", "w"), indent=1, default=float)
    write_txt(res)
    write_tex(res)


def f1(v, sign=False):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "--"
    return f"{v:+.1f}" if sign else f"{v:.1f}"


def write_txt(res: dict) -> None:
    L = ["Exp-50: closed models, real / gray / twin image (CoT, reasoning high); 150 hard-set + 150 cue-set prompts (seed 0)", ""]
    L.append("Answers (acc vs the real image's gold, failed replies wrong; [95% pair-cluster CI]); gray: yes / no / no-letter %, "
             "'cannot tell' %, 'blank/gray image' mention %")
    for m in CLOSED:
        for st in ("hard", "cues"):
            c = {cd: res["cells"][f"{m}|{st}|{cd}"] for cd in CONDS}
            e = res["image_effect"][f"{m}|{st}"]
            g = c["gray"]
            L.append(f"{NAMES[m]:20s} {st:5s} real {f1(c['real']['acc_all'])} [{f1(c['real']['acc_all_ci'][0])}, {f1(c['real']['acc_all_ci'][1])}] "
                     f"(valid {c['real']['valid']}/{c['real']['n']}) | gray {f1(g['acc_all'])} yes/no/none {f1(g['share_A'])}/{f1(g['share_B'])}/{f1(g['share_none'])} "
                     f"(yes front {f1(g['share_A_front'])}, behind {f1(g['share_A_behind'])}) cannot {f1(g['cannot'])} blank {f1(g['blank_mention'])} "
                     f"presence vis-rel/not-vis/other {f1(g['presence']['VISIBLE_RELATION'])}/{f1(g['presence']['NOT_VISIBLE'])}/{f1(g['presence']['OTHER'])} "
                     f"(n {g['presence']['n_judged']}, either vis-rel {f1(g['presence']['either_visible_relation'])}; real vis-rel "
                     f"{f1(c['real']['presence']['VISIBLE_RELATION'])} n {c['real']['presence']['n_judged']}) | "
                     f"twin acc vs real gold {f1(c['twin']['acc_all'])} | image effect {f1(e['answer_twin'], 1)} "
                     f"[{f1(e['answer_twin_ci'][0], 1)}, {f1(e['answer_twin_ci'][1], 1)}] (valid-both {f1(e['answer_twin_valid'], 1)}, n {e['answer_twin_valid_n']}); "
                     f"real - gray {f1(e['answer_gray'], 1)}"
                     + (f" | Direct image effect {f1(e['direct_twin'], 1)} [{f1(e['direct_twin_ci'][0], 1)}, {f1(e['direct_twin_ci'][1], 1)}] "
                        f"(Direct acc {f1(e['direct_acc_real'])}), CoT - Direct {f1(e['cot_minus_direct_twin'], 1)} "
                        f"[{f1(e['cot_minus_direct_twin_ci'][0], 1)}, {f1(e['cot_minus_direct_twin_ci'][1], 1)}]" if "direct_twin" in e else ""))
    L.append("")
    L.append(f"Gray texts that describe both shapes as visible with a depth relation (presence consensus): {res['gray_visible_relation']}")
    L.append(f"API error rows over all attempts (each retried until success): {res['gen_error_rows_all_attempts']}")
    L.append("")
    for pr in PROTOS:
        L.append(f"Written relation, protocol '{pr}' (consensus of two judges): coverage %, written acc vs the shown image (n false), "
                 "agree all / true / false, written image effect (twin) [CI]; gray: coverage, either-judge, names queried subject x %, implied = real gold %")
        for m in CLOSED:
            for st in ("hard", "cues"):
                c = {cd: res["cells"][f"{m}|{st}|{cd}"][pr] for cd in CONDS}
                e = res["image_effect"][f"{m}|{st}"]
                L.append(f"{NAMES[m]:20s} {st:5s} real cov {f1(c['real']['coverage'])} acc {f1(c['real']['written_acc_img'])} ({c['real']['n_false']}) "
                         f"agree {f1(c['real']['agree'])}/{f1(c['real']['agree_true'])}/{f1(c['real']['agree_false'])} | "
                         f"twin cov {f1(c['twin']['coverage'])} acc {f1(c['twin']['written_acc_img'])} ({c['twin']['n_false']}) "
                         f"agree {f1(c['twin']['agree'])}/{f1(c['twin']['agree_true'])}/{f1(c['twin']['agree_false'])} | "
                         f"written effect {f1(e[f'written_twin_{pr}'], 1)} [{f1(e[f'written_twin_{pr}_ci'][0], 1)}, {f1(e[f'written_twin_{pr}_ci'][1], 1)}] "
                         f"(both-covered {f1(e[f'written_twin_{pr}_both'], 1)}, n {e[f'written_twin_{pr}_both_n']}) | "
                         f"gray cov {f1(c['gray']['coverage'])} either {f1(c['gray']['either_judge'])} n_rel {c['gray']['n_rel']} "
                         f"x {f1(c['gray']['rel_names_x'])} =gold {f1(c['gray']['implied_eq_gold'])} agree {f1(c['gray']['agree'])} unlab {c['gray']['unlabelled']}")
        L.append("")
    L.append("Judges (4-way: shape x / shape y / none / unclear; prefiltered texts excluded):")
    for k, v in res["judges"].items():
        L.append(f"  {k}: {v}")
    L.append("")
    L.append("Failures:")
    for k, v in res["failures"].items():
        L.append(f"  {k}: {v}")
    L.append("")
    L.append("Per-cell reply accounting (n / valid / api_error / empty / content_filter / no_letter / cannot&no-letter / median reasoning tokens / sources):")
    for k, c in res["cells"].items():
        L.append(f"  {k}: {c['n']} / {c['valid']} / {c['api_error']} / {c['empty']} / {c['content_filter']} / {c['no_letter']} / "
                 f"{c['cannot_no_letter']} / {c['median_reasoning_tokens']} / {c['sources']}")
    L.append("")
    L.append("Open-model reference numbers (paper):")
    for k, v in res["open_model_reference"].items():
        L.append(f"  {k}: {v}")
    L.append("")
    L.append("Gray-image examples:")
    for k, v in res["gray_examples"].items():
        for t in v:
            L.append(f"  [{k}] " + t.replace("\n", " | ")[:300])
    (B / "rigor_e50.txt").write_text("\n".join(L) + "\n")
    print("\n".join(L[:60]))


def write_tex(res: dict) -> None:
    H = [r"% Exp-50 (research_log): closed models under the real image, a uniform gray image and the identity-swapped twin.",
         r"% Bare tabular (not \input); needs \resizebox. Caption text is in the footer row.",
         r"\begin{tabular}{llcccccccccccc}", r"\toprule",
         r"Model & Set & \multicolumn{3}{c}{\texttt{CoT} acc.\ vs.\ real gold} & \multicolumn{3}{c}{Gray image (\texttt{CoT})} & "
         r"\multicolumn{2}{c}{Image effect (twin)} & \multicolumn{2}{c}{Written rel.\ acc.} & Written effect & Agree \\",
         r"\cmidrule(lr){3-5}\cmidrule(lr){6-8}\cmidrule(lr){9-10}\cmidrule(lr){11-12}",
         r" & & real & gray & twin & yes / no / none & absent & visible rel. & \texttt{CoT} & \texttt{Dir.} & real & twin & kept / rem. & kept / rem. \\",
         r"\midrule"]
    for st in ("hard", "cues"):
        for m in CLOSED:
            c = {cd: res["cells"][f"{m}|{st}|{cd}"] for cd in CONDS}
            e = res["image_effect"][f"{m}|{st}"]
            g = c["gray"]
            ag = []
            for pr in PROTOS:
                sel = [r for r in res["rows"] if r["model"] == m and r["set"] == st and r["cond"] in ("real", "twin")
                       and r["rel_" + pr] and r["pred"]]
                ag.append(f1(100 * sum(r["impl_" + pr] == r["pred"] for r in sel) / len(sel)) if sel else "--")
            dtw = (f"${f1(e['direct_twin'], 1)}$" if "direct_twin" in e else "--")
            H.append(f"{NAMES[m]} & {SETN[st]} & {f1(c['real']['acc_all'])} & {f1(g['acc_all'])} & {f1(c['twin']['acc_all'])} & "
                     f"{f1(g['share_A'])} / {f1(g['share_B'])} / {f1(g['share_none'])} & {f1(g['presence']['NOT_VISIBLE'])} & "
                     f"{f1(g['presence']['VISIBLE_RELATION'])} & "
                     f"${f1(e['answer_twin'], 1)}$ {{\\scriptsize[{f1(e['answer_twin_ci'][0], 1)}, {f1(e['answer_twin_ci'][1], 1)}]}} & {dtw} & "
                     f"{f1(c['real']['kept']['written_acc_img'])} & {f1(c['twin']['kept']['written_acc_img'])} & "
                     f"${f1(e['written_twin_kept'], 1)}$ / ${f1(e['written_twin_removed'], 1)}$ & {ag[0]} / {ag[1]} \\\\")
        if st == "hard":
            H.append(r"\midrule")
    jr, jg = res["judges"]["kept|real"], res["judges"]["kept|gray"]
    jp = res["judges"]["presence|gray"]
    H += [r"\bottomrule",
          r"\multicolumn{14}{p{0.98\linewidth}}{\scriptsize 150 prompts per set (seed 0; 75 ``in front of'', 75 ``behind''), "
          r"\texttt{CoT} prompt of the closed-model audit, reasoning high. Real: item image (hard set: the replies of Tab.~\ref{tab:supp_e38} reused); "
          r"gray: uniform gray image of the same size; twin: identity-swapped twin with the same question. Accuracies in \% against the real image's "
          r"gold; failed or letter-free replies count as wrong. Gray: share of yes / no / no letter; absent / visible rel.: share of full replies "
          r"that both judges classify as stating that the shapes are absent or the image is blank, or as describing both shapes as visible with "
          r"one in front. Image effect: $P(\text{answer}=\text{gold}_v\mid v)-P(\text{answer}=\text{gold}_v\mid v_\mathrm{cf})$, "
          r"with the rationale regenerated under each image (pair-cluster bootstrap CI); \texttt{Dir.}: the same for the letter-only reply "
          r"(hard set only). Written relation: last depth claim, two-judge consensus, accuracy against the image shown (conclusion kept). Written "
          r"effect: image effect of the answer implied by the written relation (a missing relation counts as not $\text{gold}_v$), conclusion "
          r"kept / removed. Agree: answers that agree with the written relation, real and twin images. Relation judges (Claude Opus 5.5, "
          f"GPT-6-Astra, text only): 4-way agreement {jr['agree']:.1f}\\% on real-image texts ($\\kappa={jr['kappa']:.2f}$) but "
          f"{jg['agree']:.1f}\\% on gray-image texts ($\\kappa={jg['kappa']:.2f}$), where GPT-6-Astra reads premise denials (``the X cannot "
          r"be behind the Y'') as relations; the gray columns therefore use the presence classification "
          f"(agreement {jp['agree']:.1f}\\%, $\\kappa={jp['kappa']:.2f}$, $n={jp['n']}$).}} \\\\",
          r"\end{tabular}"]
    (PAPER / "tables" / "supp_e50.tex").write_text("\n".join(H) + "\n")
