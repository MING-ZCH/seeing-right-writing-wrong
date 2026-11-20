"""Exp-38 (part 2): closed / API models on the two-channel axes, with a judge-extracted last written relation.

Per model x set (e9h hard occlusion set, e9x occlusion twins, e9c_cues DeepSeek cue set), on items with a valid
Direct and CoT reply (last row with a non-empty pred per (condition, item_id, phrasing)):
coverage of an extracted last written relation (two-judge consensus, e38_closed_judge.py), Direct accuracy (all items /
items with a relation), CoT accuracy, written-relation accuracy, P(agree | true), P(agree | false), x = written -
Direct (relation items), y = CoT - Direct (all items), D+C- / D-C+ -- all via rigor_e32.cell_stats (Exp-32 definitions,
pair-cluster bootstrap). Same with the lowest-reasoning Direct (direct_nr) where it exists, and with each judge alone.
Out of sample: y_pred = slope * x + intercept from the 21-point open-model fit (bundle/two_channel.json, not refitted).
Also: judge agreement / kappa, self-judging check, rule-parse (claim_parse v3) comparison, and the valid-n audit for
Tab. S closed (supp_closed.tex).
-> bundle/e38_closed_two_channel.json, bundle/rigor_e38.txt, paper tables/supp_e38.tex

Usage: python rigor_e38.py
"""
from __future__ import annotations
import vtax_paths as _vtp  # noqa: E402

import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import pearsonr

from e2f_claim_edit import split_sents
from e38_closed_judge import CLOSED, JUDGES, OUT as JDIR, SETS, dedup, rj, strip_answer
from rigor_e32 import _acct_terms, cell_stats, claims, implied

O = Path(f"{_vtp.OUTPUTS}")
B = Path("outputs/bundle")
PAPER = Path(f"{_vtp.PAPER}")
NAMES = {"claude-opus-5-5": "Claude Opus 5.5", "gpt-6-astra": "GPT-6-Astra", "gemini-4-argon": "Gemini-4-Argon",
         "qwen3.8-max-0902": "Qwen3.8-Max", "hy4": "HY4", "deepseek-v4.1-flash": "DeepSeek-V4.1-Flash"}
SETN = {"e9h": "Hard occl.", "e9x": "Occl. twins", "e9c_cues": "Cue set"}
SHAPE = lambda lab, r: lab not in (None, "none", "unclear")


def kappa(a: list, b: list) -> float:
    cats = sorted(set(a) | set(b))
    po = np.mean([x == y for x, y in zip(a, b)])
    pe_ = sum((a.count(c) / len(a)) * (b.count(c) / len(b)) for c in cats)
    return float((po - pe_) / (1 - pe_)) if pe_ < 1 else 1.0


def rel3(lab: str | None, x: str) -> str:
    """Judge label as x / y / none / unclear (None = unparsed judge reply)."""
    if lab in ("none", "unclear") or lab is None:
        return lab or "unparsed"
    return "x" if lab == x else "y"


def load_judges(first: bool = False) -> dict:
    J = {}
    for j in JUDGES:
        d = {}
        for r in rj(JDIR / f"judge_{j}{'_first' if first else ''}.jsonl"):
            if r.get("label") is not None or r["key"] not in d:
                d[r["key"]] = r
        J[j] = d
    return J


def load_records(J: dict, JF: dict) -> list[dict]:
    """Per-item records (Direct + CoT valid) with judge labels, consensus and rule parse."""
    out = []
    for st, models in SETS.items():
        for m in models:
            R = defaultdict(dict)
            for (cond, iid, ph), r in dedup(rj(O / st / f"{m}.jsonl")).items():
                if r["task"].startswith("occ"):
                    R[iid][cond] = r
            for iid, d in R.items():
                if "cot" not in d or "direct" not in d:
                    continue
                c, dr = d["cot"], d["direct"]
                mt = c["meta"]
                x, front, back = mt["x_name"].lower(), mt["front_name"].lower(), mt["back_name"].lower()
                key = f"{st}|{m}|{iid}"
                labs = {j: J[j].get(key, {}).get("label") for j in JUDGES}
                a, b = labs[JUDGES[0]], labs[JUDGES[1]]
                cons = a if (a is not None and a == b) else None
                fa, fb = (JF[j].get(key, {}).get("label") for j in JUDGES)
                cons_first = fa if (fa is not None and fa == fb) else None
                labs_first = {JUDGES[0]: fa, JUDGES[1]: fb}
                bod, _ = strip_answer(c.get("text") or "")
                _, rule_last, _ = claims(split_sents(bod), front, back)
                nr = d.get("direct_nr")
                out.append({"model": m, "ds": st, "item": iid, "cl": c["image_id"], "gold": c["gold"], "phrasing": c["phrasing"],
                            "x": x, "front": front, "level": (mt or {}).get("level"),
                            "d_ok": dr["pred"] == c["gold"], "c_ok": c["pred"] == c["gold"], "c_ans": c["pred"],
                            "nr_ok": None if nr is None else nr["pred"] == c["gold"],
                            "lab": labs, "cons": cons, "cons_first": cons_first, "lab_first": labs_first, "first_judged": fa is not None and fb is not None, "judged_both": a is not None and b is not None, "rule_last": rule_last})
    return out


def with_label(rs: list[dict], how: str, direct: str = "direct") -> list[dict]:
    """Attach first/last fields as rigor_e32.cell_stats expects; ``how`` = consensus / judge name / rule."""
    out = []
    for r in rs:
        if direct == "direct_nr" and r["nr_ok"] is None:
            continue
        lab = r["cons"] if how == "consensus" else (r["rule_last"] if how == "rule" else r["lab"][how])
        ok = SHAPE(lab, r)
        fl = r["cons_first"] if how == "consensus" else None
        fok = SHAPE(fl, r)
        out.append(r | {"d_ok": r["nr_ok"] if direct == "direct_nr" else r["d_ok"],
                        "first_true": (fl == r["front"]) if fok else None,
                        "first_agree": (implied(fl, r["x"], r["phrasing"]) == r["c_ans"]) if fok else None,
                        "first_lab": fl if fok else None, "last_lab": lab if ok else None,
                        "last_true": (lab == r["front"]) if ok else None,
                        "last_agree": (implied(lab, r["x"], r["phrasing"]) == r["c_ans"]) if ok else None})
    return out


def follow(rs: list[dict]) -> dict:
    """Finding 1 (agreement by truth of the written relation) and finding 2 (false relation where Direct is right)."""
    p = [r for r in rs if r["last_true"] is not None]
    tr, fa = [r for r in p if r["last_true"]], [r for r in p if not r["last_true"]]
    pc = lambda k, n: round(100 * k / n, 1) if n else None
    dpcm = [r for r in p if r["d_ok"] and not r["c_ok"]]
    dpcp = [r for r in p if r["d_ok"] and r["c_ok"]]
    dp = [r for r in p if r["d_ok"]]
    res = {"n_true": len(tr), "n_false": len(fa),
           "p_agree_true": pc(sum(r["last_agree"] for r in tr), len(tr)), "p_agree_false": pc(sum(r["last_agree"] for r in fa), len(fa)),
           "agree_true_k": int(sum(r["last_agree"] for r in tr)), "agree_false_k": int(sum(r["last_agree"] for r in fa)),
           "n_DpCm_rel": len(dpcm), "false_in_DpCm": pc(sum(not r["last_true"] for r in dpcm), len(dpcm)),
           "false_in_DpCm_k": int(sum(not r["last_true"] for r in dpcm)),
           "n_DpCp_rel": len(dpcp), "false_in_DpCp": pc(sum(not r["last_true"] for r in dpcp), len(dpcp)),
           "false_in_DpCp_k": int(sum(not r["last_true"] for r in dpcp)),
           "n_Dp_rel": len(dp), "false_in_Dp": pc(sum(not r["last_true"] for r in dp), len(dp)),
           "false_in_Dp_k": int(sum(not r["last_true"] for r in dp))}
    if p:
        W = 100 * len(tr) / len(p)
        aT = res["p_agree_true"] if tr else 0.0
        aF = res["p_agree_false"] if fa else 0.0
        res["acct_y_hat"] = round(_acct_terms(W, aT, aF, 100 * np.mean([r["d_ok"] for r in rs])), 2)
    return res


def follow_first(rs: list[dict]) -> dict:
    """First vs last written relation: agreement of the answer with the first relation, revisions, self-correction."""
    p = [r for r in rs if r["first_true"] is not None]
    if not p:
        return {}
    tr, fa = [r for r in p if r["first_true"]], [r for r in p if not r["first_true"]]
    pc = lambda k, n: round(100 * k / n, 1) if n else None
    both = [r for r in p if r["last_lab"] is not None]
    rev = [r for r in both if r["first_lab"] != r["last_lab"]]
    return {"first_n_true": len(tr), "first_n_false": len(fa),
            "first_p_agree_true": pc(sum(r["first_agree"] for r in tr), len(tr)),
            "first_p_agree_false": pc(sum(r["first_agree"] for r in fa), len(fa)),
            "n_first_and_last": len(both), "first_ne_last": pc(len(rev), len(both)), "first_ne_last_k": len(rev),
            "revised_false_to_true": sum(not r["first_true"] and r["last_lab"] == r["front"] for r in rev),
            "revised_true_to_false": sum(r["first_true"] and r["last_lab"] != r["front"] for r in rev),
            "false_first_cot_right": pc(sum(r["c_ok"] for r in fa), len(fa)),
            "false_first_given_Dp": pc(sum(not r["first_true"] for r in p if r["d_ok"]), sum(r["d_ok"] for r in p)),
            "false_first_given_DpCp": pc(sum(not r["first_true"] for r in p if r["d_ok"] and r["c_ok"]), sum(r["d_ok"] and r["c_ok"] for r in p)),
            "false_first_given_DpCm": pc(sum(not r["first_true"] for r in p if r["d_ok"] and not r["c_ok"]), sum(r["d_ok"] and not r["c_ok"] for r in p))}


def cell(rs: list[dict], how: str, direct: str, fit: dict, seed: int, fit_first: dict | None = None) -> dict | None:
    rr = with_label(rs, how, direct)
    if not rr:
        return None
    c = cell_stats(rr, np.random.default_rng(seed))
    c |= follow(rr) | follow_first(rr)
    if fit_first and "first_x" in c:
        c["first_y_pred"] = round(fit_first["ols_slope"] * c["first_x"] + fit_first["ols_intercept"], 2)
        c["first_y_minus_pred"] = round(c["y"] - c["first_y_pred"], 2)
    if "last_x" in c:
        c["y_pred"] = round(fit["ols_slope"] * c["last_x"] + fit["ols_intercept"], 2)
        c["y_minus_pred"] = round(c["y"] - c["y_pred"], 2)
    return c


def judge_agreement(rs: list[dict]) -> dict:
    both = [r for r in rs if r["judged_both"]]
    a = [rel3(r["lab"][JUDGES[0]], r["x"]) for r in both]
    b = [rel3(r["lab"][JUDGES[1]], r["x"]) for r in both]
    a2 = [v if v in ("x", "y") else "no" for v in a]
    b2 = [v if v in ("x", "y") else "no" for v in b]
    sh = [(u, v) for u, v in zip(a, b) if u in ("x", "y") and v in ("x", "y")]
    return {"n": len(rs), "n_both_labelled": len(both),
            "agree_4way": round(100 * float(np.mean([u == v for u, v in zip(a, b)])), 1) if both else None,
            "kappa_4way": round(kappa(a, b), 3) if both else None,
            "agree_2way_rel_vs_no": round(100 * float(np.mean([u == v for u, v in zip(a2, b2)])), 1) if both else None,
            "kappa_shape_or_no": round(kappa(a2, b2), 3) if both else None,
            "n_disagree": int(sum(u != v for u, v in zip(a, b))), "n_both_shape": len(sh), "agree_when_both_shape": round(100 * float(np.mean([u == v for u, v in sh])), 1) if sh else None,
            "labels_" + JUDGES[0]: dict(Counter(a)), "labels_" + JUDGES[1]: dict(Counter(b))}


def self_judging(recs: list[dict]) -> dict:
    """For each judge: P(other judge disagrees | self extracted a relation) on its own texts vs the other models' texts."""
    out = {}
    for j, o in (JUDGES, JUDGES[::-1]):
        for st in ("e9h", "e9x"):
            rs = [r for r in recs if r["ds"] == st and r["judged_both"] and SHAPE(r["lab"][j], r)]
            own = [r for r in rs if r["model"] == j]
            oth = [r for r in rs if r["model"] != j]
            dis = lambda L: round(100 * float(np.mean([r["lab"][o] != r["lab"][j] for r in L])), 1) if L else None
            sw = lambda L: round(100 * float(np.mean([SHAPE(r["lab"][o], r) and r["lab"][o] != r["lab"][j] for r in L])), 1) if L else None
            out[f"{j}|{st}"] = {"n_own": len(own), "other_disagrees_own": dis(own), "other_names_other_shape_own": sw(own),
                                "n_others": len(oth), "other_disagrees_others": dis(oth), "other_names_other_shape_others": sw(oth)}
    return out


LASTLINE = __import__("re").compile(r"^\W*\(?([A-F])\)?\W*$")


def last_line_letter(text: str, letters: list[str]) -> str | None:
    """Letter alone on the last non-empty line (e9x_api.parse_letter anchors '^' at the text start only)."""
    ln = [l for l in (text or "").strip().splitlines() if l.strip()]
    m = LASTLINE.match(ln[-1]) if ln else None
    return m.group(1) if m and m.group(1) in letters else None


def valid_n_audit() -> dict:
    """Valid replies per model x task x condition on e9x / e9h / e9c (for Tab. S closed)."""
    res = {}
    for st, models in (("e9x", CLOSED), ("e9h", CLOSED), ("e9c_cues", ["deepseek-v4.1-flash"])):
        for m in models:
            rows = rj(O / st / f"{m}.jsonl")
            items = defaultdict(list)
            for r in rows:
                items[(r["condition"], r["item_id"], r.get("phrasing"))].append(r)
            agg = defaultdict(lambda: Counter())
            accs = defaultdict(lambda: [0, 0, 0])
            for (cond, iid, ph), rr in items.items():
                task = "occ" if rr[0]["task"].startswith("occ") else rr[0]["task"]
                v = [r for r in rr if r.get("pred")]
                txt = [r for r in rr if "error" not in r]
                k = f"{task}|{cond}"
                agg[k]["n_items"] += 1
                if v:
                    agg[k]["valid"] += 1
                    accs[k][0] += v[-1]["pred"] == v[-1]["gold"]
                elif txt:
                    agg[k]["unparsed_reply"] += 1
                    t = txt[-1].get("text") or ""
                    if not t.strip():
                        agg[k]["unparsed_empty"] += 1
                    ll = last_line_letter(t, txt[-1].get("letters") or ["A", "B"])
                    if ll:
                        agg[k]["recoverable_last_line"] += 1
                        accs[k][2] += ll == txt[-1]["gold"]
                else:
                    agg[k]["api_error_only"] += 1
            for k, cnt in agg.items():
                n, nv = cnt["n_items"], cnt["valid"]
                res[f"{st}|{m}|{k}"] = dict(cnt) | {"acc_valid_only": round(100 * accs[k][0] / nv, 1) if nv else None,
                                                    "acc_missing_as_wrong": round(100 * accs[k][0] / n, 1) if n else None,
                                                    "acc_with_last_line_recovery_missing_as_wrong": round(100 * (accs[k][0] + accs[k][2]) / n, 1) if n else None}
    return res


def main() -> None:
    fits = json.load(open(B / "two_channel.json"))["fits"]
    fit, fit_first = fits["primary|last_main"], fits["primary|first_main"]
    J, JF = load_judges(), load_judges(first=True)
    recs = load_records(J, JF)
    lines = [f"Exp-38 closed-model two-channel audit; open-model fit (21 pts, not refitted): y = {fit['ols_slope']} x {fit['ols_intercept']:+.2f}"]
    groups = defaultdict(list)
    for r in recs:
        groups[(r["model"], r["ds"])].append(r)
    cells, seed = {}, 380
    for (m, st), rs in sorted(groups.items(), key=lambda kv: (list(SETS).index(kv[0][1]), CLOSED.index(kv[0][0]))):
        seed += 1
        e = {"model": m, "ds": st, "n": len(rs), "n_judged_both": sum(r["judged_both"] for r in rs)}
        e["consensus"] = cell(rs, "consensus", "direct", fit, seed, fit_first)
        if any(r["nr_ok"] is not None for r in rs):
            e["consensus_direct_nr"] = cell(rs, "consensus", "direct_nr", fit, seed, fit_first)
        if any(r["first_judged"] for r in rs):
            fj = [r for r in rs if r["first_judged"]]
            fa_ = [rel3(r["cons_first"], r["x"]) for r in fj]
            e["first_judges"] = judge_agreement([r | {"lab": r["lab_first"], "judged_both": r["first_judged"]} for r in rs])
        for j in JUDGES:
            e[f"judge_{j}"] = cell(rs, j, "direct", fit, seed)
        e["rule_v3"] = cell(rs, "rule", "direct", fit, seed)
        e["judges"] = judge_agreement(rs)
        both_shape = [r for r in rs if r["cons"] and SHAPE(r["cons"], r) and SHAPE(r["rule_last"], r)]
        e["rule_vs_consensus"] = {"rule_coverage": round(100 * np.mean([SHAPE(r["rule_last"], r) for r in rs]), 1),
                                  "n_both": len(both_shape),
                                  "agree_when_both": round(100 * np.mean([r["rule_last"] == r["cons"] for r in both_shape]), 1) if both_shape else None}
        cells[f"{m}|{st}"] = e
        c = e["consensus"]
        lines.append(
            f"CELL {m:20s} {st:8s} n={c['n']:4d} cov {c['last_coverage']:5.1f} ({c['last_n_parsed']}) | Dir {c['acc_direct']:5.1f} "
            f"Dir(rel) {c.get('last_direct_on_parsed', float('nan')):5.1f} CoT {c['acc_cot']:5.1f} | written {c.get('last_written_acc', float('nan')):5.1f} "
            f"agree|T {c['p_agree_true']} ({c['agree_true_k']}/{c['n_true']}) agree|F {c['p_agree_false']} ({c['agree_false_k']}/{c['n_false']}) "
            f"| x {c.get('last_x', float('nan')):+6.2f} {c.get('last_x_ci')} y {c['y']:+6.2f} {c.get('y_ci')} y_pred {c.get('y_pred', float('nan')):+6.2f} "
            f"(y-pred {c.get('y_minus_pred', float('nan')):+6.2f}) acct y_hat {c.get('acct_y_hat')} | D+C- {c['DpCm']} D-C+ {c['DmCp']} "
            f"| false|D+C- {c['false_in_DpCm']} ({c['false_in_DpCm_k']}/{c['n_DpCm_rel']}) false|D+C+ {c['false_in_DpCp']} "
            f"({c['false_in_DpCp_k']}/{c['n_DpCp_rel']}) false|D+ {c['false_in_Dp']}")
        if "consensus_direct_nr" in e and e["consensus_direct_nr"]:
            n_ = e["consensus_direct_nr"]
            lines.append(f"  direct_nr: n={n_['n']} Dir {n_['acc_direct']:5.1f} Dir(rel) {n_.get('last_direct_on_parsed', float('nan')):5.1f} "
                         f"x {n_.get('last_x', float('nan')):+6.2f} y {n_['y']:+6.2f} {n_.get('y_ci')} y_pred {n_.get('y_pred', float('nan')):+6.2f} "
                         f"D+C- {n_['DpCm']} D-C+ {n_['DmCp']}")
        if c.get("first_n_parsed"):
            fj = e["first_judges"]
            lines.append(f"  FIRST claim: judged both {fj['n_both_labelled']} agree4 {fj['agree_4way']} kappa4 {fj['kappa_4way']} | cov {c['first_coverage']:5.1f} "
                         f"written {c.get('first_written_acc', float('nan')):5.1f} agree|T {c['first_p_agree_true']} ({c['first_n_true']}) agree|F {c['first_p_agree_false']} "
                         f"({c['first_n_false']}) | first!=last {c['first_ne_last']} ({c['first_ne_last_k']}/{c['n_first_and_last']}; F->T {c['revised_false_to_true']}, "
                         f"T->F {c['revised_true_to_false']}) | CoT right when first false {c['false_first_cot_right']} | false first | D+ {c['false_first_given_Dp']} "
                         f"D+C+ {c['false_first_given_DpCp']} D+C- {c['false_first_given_DpCm']} | x_first {c.get('first_x', float('nan')):+6.2f} "
                         f"y_pred(first fit) {c.get('first_y_pred', float('nan')):+6.2f}")
        for j in JUDGES:
            q = e[f"judge_{j}"]
            lines.append(f"  judge {j:16s} cov {q['last_coverage']:5.1f} written {q.get('last_written_acc', float('nan')):5.1f} "
                         f"agree|T {q['p_agree_true']} agree|F {q['p_agree_false']} x {q.get('last_x', float('nan')):+6.2f} y_pred {q.get('y_pred', float('nan')):+6.2f}")
        g = e["judges"]
        lines.append(f"  judges: both labelled {g['n_both_labelled']}/{g['n']} agree4 {g['agree_4way']} kappa4 {g['kappa_4way']} "
                     f"agree(rel/no) {g['agree_2way_rel_vs_no']} kappa(rel/no) {g['kappa_shape_or_no']} agree|both shape {g['agree_when_both_shape']} ({g['n_both_shape']})"
                     f" | rule v3 coverage {e['rule_vs_consensus']['rule_coverage']} agree w/ consensus {e['rule_vs_consensus']['agree_when_both']} ({e['rule_vs_consensus']['n_both']})")
    # pooled judge agreement
    pooled = {st: judge_agreement([r for r in recs if r["ds"] == st]) for st in SETS} | {"all": judge_agreement(recs)}
    sj = self_judging(recs)
    lines.append(f"JUDGES pooled: " + "; ".join(f"{k}: agree4 {v['agree_4way']} kappa4 {v['kappa_4way']} kappa(rel/no) {v['kappa_shape_or_no']} "
                                                 f"agree|both shape {v['agree_when_both_shape']} (n {v['n_both_labelled']})" for k, v in pooled.items()))
    for k, v in sj.items():
        lines.append(f"SELF-JUDGE {k}: other judge disagrees on own texts {v['other_disagrees_own']}% (names other shape {v['other_names_other_shape_own']}%, n {v['n_own']}) "
                     f"vs other models' texts {v['other_disagrees_others']}% ({v['other_names_other_shape_others']}%, n {v['n_others']})")
    # out-of-sample placement
    oos = {}
    for variant in ("consensus", "consensus_direct_nr", f"judge_{JUDGES[0]}", f"judge_{JUDGES[1]}"):
        pts = [(k, e[variant]) for k, e in cells.items() if e.get(variant) and "last_x" in e[variant]]
        pts_ok = [(k, c) for k, c in pts if not k.startswith("gemini-4-argon|e9h") or cells[k]["n"] >= 400]
        if len(pts_ok) < 2:
            continue
        x = np.array([c["last_x"] for _, c in pts_ok]); y = np.array([c["y"] for _, c in pts_ok]); yp = np.array([c["y_pred"] for _, c in pts_ok])
        tie = np.abs(y) < 1.0
        sgn = lambda a, b: int(np.sum(np.sign(a) == np.sign(b)))
        offc = np.array([not k.endswith("|e9x") for k, _ in pts_ok])
        oos[variant] = {"n_points": len(pts_ok), "points": [k for k, _ in pts_ok],
                        "mae_fit": round(float(np.abs(y - yp).mean()), 2), "max_abs_err_fit": round(float(np.abs(y - yp).max()), 2),
                        "mean_err_y_minus_pred": round(float((y - yp).mean()), 2),
                        "mae_identity": round(float(np.abs(y - x).mean()), 2),
                        "sign_agree_x_y": [sgn(x, y), len(x)], "sign_agree_pred_y": [sgn(yp, y), len(x)],
                        "n_ties_abs_y_lt1": int(tie.sum()),
                        "sign_agree_x_y_excl_ties": [sgn(x[~tie], y[~tie]), int((~tie).sum())],
                        "mae_fit_off_ceiling": round(float(np.abs(y - yp)[offc].mean()), 2) if offc.any() else None,
                        "pearson_x_y": round(float(pearsonr(x, y)[0]), 3) if np.ptp(x) > 0 and np.ptp(y) > 0 else None,
                        "x_range": [round(float(x.min()), 2), round(float(x.max()), 2)], "y_range": [round(float(y.min()), 2), round(float(y.max()), 2)]}
        o = oos[variant]
        lines.append(f"OOS {variant:28s} n={o['n_points']} MAE(fit) {o['mae_fit']} (max {o['max_abs_err_fit']}, mean y-pred {o['mean_err_y_minus_pred']:+.2f}; "
                     f"off-ceiling sets {o['mae_fit_off_ceiling']}) MAE(y=x) {o['mae_identity']} | sign x~y {o['sign_agree_x_y']} pred~y {o['sign_agree_pred_y']} "
                     f"excl |y|<1 {o['sign_agree_x_y_excl_ties']} | r(x,y) {o['pearson_x_y']} x {o['x_range']} y {o['y_range']}")
    # Gemini-4-Argon hard set, answer level (same summary as outputs/e9h/summary.json; that file is not rewritten)
    import analyze_e9x as AE
    gs = AE.summarize(AE.load(O / "e9h" / "gemini-4-argon.jsonl"))
    gem = {"occ_both": gs.get("occ_both"), "occ_sdt": gs.get("occ_sdt"), "occ_sdt_diff": gs.get("occ_sdt_diff"),
           "by_level": {k: v for k, v in gs.items() if k.startswith("level_") or k.startswith("occ_level")}}
    ob = gs["occ_both"]
    lines.append(f"GEMINI e9h answer level: Direct n {ob['direct']['n']} acc {ob['direct']['acc']} unparsed {ob['direct']['unparsed']} | CoT n {ob['cot']['n']} "
                 f"acc {ob['cot']['acc']} | CoT-Direct {ob.get('cot_minus_direct')} | d' Direct {gs['occ_sdt']['direct']} CoT {gs['occ_sdt']['cot']} "
                 f"| d' CoT-Direct {gs['occ_sdt_diff'].get('cot_minus_direct')}")
    va = valid_n_audit()
    for k, v in sorted(va.items()):
        if v.get("valid", 0) < v["n_items"]:
            lines.append(f"VALID-N {k:52s} valid {v.get('valid', 0)}/{v['n_items']} (unparsed reply {v.get('unparsed_reply', 0)}, API error only "
                         f"{v.get('api_error_only', 0)}; empty text {v.get('unparsed_empty', 0)}, letter alone on last line {v.get('recoverable_last_line', 0)}) "
                         f"acc valid-only {v['acc_valid_only']} missing-as-wrong {v['acc_missing_as_wrong']} with last-line recovery {v['acc_with_last_line_recovery_missing_as_wrong']}")
    res = {"_doc": __doc__, "fit_used": {k: fit[k] for k in ("n_points", "ols_slope", "ols_intercept", "pearson_r")},
           "fit_used_first": {k: fit_first[k] for k in ("n_points", "ols_slope", "ols_intercept", "pearson_r")},
           "cells": cells, "judge_agreement_pooled": pooled, "self_judging": sj, "out_of_sample": oos, "valid_n_closed": va,
           "gemini_hard_set_answer_level": gem}
    json.dump(res, open(B / "e38_closed_two_channel.json", "w"), indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o))
    open(B / "rigor_e38.txt", "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))
    write_tex(res)


def f1(v, sign=False):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "--"
    v = 0.0 if abs(v) < 0.05 else v
    return f"{v:+.1f}" if sign else f"{v:.1f}"


def write_tex(res: dict) -> None:
    cells, oos, pooled = res["cells"], res["out_of_sample"], res["judge_agreement_pooled"]
    rows = []
    for k, e in cells.items():
        c = e["consensus"]
        nr = e.get("consensus_direct_nr")
        star = "$^\\dagger$" if e["model"] == "gemini-4-argon" and e["ds"] == "e9h" and e["n"] < 400 else ""
        rows.append(f"{NAMES[e['model']]}{star} & {SETN[e['ds']]} & {c['n']} & {c['last_n_parsed']} ({c['last_coverage']:.1f}) & "
                    f"{f1(c['acc_direct'])} & {f1(c.get('last_direct_on_parsed'))} & {f1(c['acc_cot'])} & {f1(c.get('last_written_acc'))} & "
                    f"{f1(c['p_agree_true'])} / {f1(c['p_agree_false'])} ({c['n_false']}) & {f1(c.get('last_x'), True)} & {f1(c['y'], True)} & "
                    f"{f1(c.get('y_pred'), True)} & {c['DpCm']} / {c['DmCp']} & "
                    f"{(f1(nr['y'], True) + ' / ' + f1(nr.get('y_pred'), True)) if nr else '--'} \\\\")
    o, on = oos.get("consensus", {}), oos.get("consensus_direct_nr", {})
    pa = pooled["all"]
    sj = res["self_judging"]
    foot = (f"Accuracies in \\%. Written relation: last depth/occlusion claim in the CoT text with the answer line removed, extracted by two text-only "
            f"judges (Claude Opus 5.5, GPT-6-Astra; shape names given, no image); a relation counts only when both judges name the same shape. "
            f"Agree: share of CoT answers that agree with the written relation when it is true / false (number of false relations in brackets). "
            f"$x$ = written $-$ \\texttt{{Dir.}} on items with a relation; $y$ = \\texttt{{CoT}} $-$ \\texttt{{Dir.}} on all items; "
            f"$\\hat y$ = {res['fit_used']['ols_slope']}$x${res['fit_used']['ols_intercept']:+.2f} from the 21 open-model points (Tab.~\\ref{{tab:supp_two_channel}}, not refitted). "
            f"Last column: $y$ / $\\hat y$ with the lowest-reasoning \\texttt{{Direct}}. "
            f"$n$: items with a valid \\texttt{{Direct}} and \\texttt{{CoT}} reply. "
            f"Judges: 4-way labels (shape $x$ / shape $y$ / none / unclear) differ on {pa['n_disagree']} of {pa['n_both_labelled']:,} texts "
            f"($\\kappa={pa['kappa_4way']:.3f}$); no text is labelled none or unclear by both. "
            f"Self-judging: the other judge disagrees with Claude on Claude's hard-set texts in {sj['claude-opus-5-5|e9h']['other_disagrees_own']}\\% "
            f"(other models {sj['claude-opus-5-5|e9h']['other_disagrees_others']}\\%) and with GPT on GPT's in {sj['gpt-6-astra|e9h']['other_disagrees_own']}\\% "
            f"({sj['gpt-6-astra|e9h']['other_disagrees_others']}\\%). ")
    if o:
        foot += (f"Because every \\texttt{{CoT}} answer agrees with its written relation and a relation is extracted from $\\ge 99.9\\%$ of texts, $x=y$ in every row "
                 f"(the closed points test the fit's intercept, not its slope). Out of sample ({o['n_points']} points): $y-\\hat y$ = +{min(c['consensus']['y_minus_pred'] for c in cells.values()):.1f} to "
                 f"+{max(c['consensus']['y_minus_pred'] for c in cells.values()):.1f} (MAE {o['mae_fit']} pts) against MAE {o['mae_identity']} for $y=x$; "
                 f"sign($x$) = sign($y$) in {o['sign_agree_x_y_excl_ties'][0]}/{o['sign_agree_x_y_excl_ties'][1]} rows with $|y|\\ge 1$, sign($\\hat y$) = sign($y$) in "
                 f"{o['sign_agree_pred_y'][0]}/{o['sign_agree_pred_y'][1]}. ")
        fr = [c["consensus"] for c in cells.values() if c["consensus"].get("n_first_and_last")]
        if fr:
            foot += (f"First written relation (random 100 texts per model on the hard set and cue set; {sum(c['n_first_and_last'] for c in fr)} texts): "
                     f"identical to the last in {sum(c['n_first_and_last'] - c['first_ne_last_k'] for c in fr)}. ")
    if on:
        foot += f"Lowest-reasoning \\texttt{{Direct}}: MAE {on['mae_fit']} over {on['n_points']} points. "
    if any("$^\\dagger$" in r for r in rows):
        foot += "$^\\dagger$Partial (quota-limited) run; not in the out-of-sample statistics."
    tex = ("\\begin{tabular}{llcccccccccccc}\n\\toprule\n"
           "Model & Set & $n$ & relation ($\\%$) & \\texttt{Dir.} & \\texttt{Dir.}\\,(rel.) & \\texttt{CoT} & written & agree T / F & $x$ & $y$ & $\\hat y$ & "
           "\\texttt{D+C$-$} / \\texttt{D$-$C+} & low-reas. $y$ / $\\hat y$ \\\\\n\\midrule\n" + "\n".join(rows) +
           "\n\\bottomrule\n\\multicolumn{14}{p{0.98\\linewidth}}{\\scriptsize " + foot + "} \\\\\n\\end{tabular}\n")
    (PAPER / "tables" / "supp_e38.tex").write_text(tex)


if __name__ == "__main__":
    main()
