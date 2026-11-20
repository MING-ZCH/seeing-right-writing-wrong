"""Exp-30 analyses (round-18 external reviews, section B), competent models Qwen3-VL-8B and Molmo-7B-D, cue set.

follow   Task 2 (Opus M4): non-circular 2x2 tables over ALL items: written relation true/false x final answer
         consistent/inconsistent with it. CoT arm (e10 CoT answer, rationale sentences of the Exp-28 trajectories):
         relation = claim_parse v3 claimed-front of (a) the last relational claim, (b) the last claim outside
         therefore/thus/hence/answer sentences (rigor_e22.CONCL), (c) the first claim, (d) the commit sentence j* when it
         parses, else (a); plus a judge-truth version on Direct-right items (j* consensus for D+C-, comparator consensus
         for D+C+, from Task 1). Description arm (Exp-28 occlusion-prompted descriptions, desc-arm answer): described
         front = two-judge consensus FRONT_CLAIMED over all 800 descriptions (Task 1 descall), and the parser version.
         P(consistent | true), P(consistent | false), difference; cluster bootstrap over scenes (twin pair = 4 items).
logit    Task 3a (Opus M2): logistic models of accuracy on the Exp-29 template edits (F0-F2, B0-B2; ECHO separately),
         terms predicate match, subject match, sentence predicate, gold polarity, question predicate, wording family.
audit    Task 1: two-judge audit of all cue-set D+C- commits and all relational D+C+ comparators, original (e20) and
         depth/vertical (Exp-29) prompts; Wilson CIs and worst-case bounds.
argedit  Task 3b: argument-order edit (outputs/e30/<m>_<set>_argedit_<part>.jsonl).
ndesc    Task 4: occlusion-neutral description + description-length fillers (outputs/e30/<m>_cues_ndesc.jsonl).
gray     Task 5: gray-image prefix readout (outputs/e30/<m>_cues_gray.jsonl).
-> bundle/e30_<section>.json, bundle/rigor_e30.txt, bundle/supp_e30.tex (copied to paper tables/, not \\input)
Usage: python rigor_e30.py [--only follow,logit,audit,argedit,ndesc,gray]
"""
from __future__ import annotations
import vtax_paths as _vtp  # noqa: E402

import argparse
import json
import re
import shutil
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from build_bundle import pe, rj
from claim_parse import first_claim_v3
from rigor_e17 import stype
from rigor_e28 import J2, NAMES, cboot, pct, q95, wilson
from rigor_e29 import claimed_role, cons2, depth_contra, implied_answer, own_contra

O = Path(f"{_vtp.OUTPUTS}")
E28, E29, E30 = O / "e28", O / "e29", O / "e30"
B = Path("outputs/bundle")
PAPER_TAB = Path(f"{_vtp.TABLES}")
COMP = ["qwen3vl_8b", "molmo7b"]
CONCL22 = re.compile(r"\b(answer is|answer:|therefore|thus|hence|so the|in conclusion|the correct (option|answer)|\([AB]\))", re.I)
NB = 2000
fa = lambda x: f"{x['mean']:+.1f} [{x['ci'][0]:+.1f}, {x['ci'][1]:+.1f}]"
fm = lambda x: f"{x['mean']:.1f} [{x['ci'][0]:.1f}, {x['ci'][1]:.1f}]"
# e2b.split_sentences (copied: importing e2b pulls vx, which needs the pod's legacy common)
_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z(\"'*\d])|\n+")
_ENUM = re.compile(r"^\s*(\d+|[a-z])[.)]\s*$|^\s*[-*]+\s*$")


def split_sentences(text: str) -> list[str]:
    raw = [s.strip() for s in _SPLIT.split(text) if s and s.strip()]
    out: list[str] = []
    carry = ""
    for s in raw:
        if _ENUM.match(s):
            carry = (carry + " " + s).strip()
            continue
        out.append((carry + " " + s).strip() if carry else s)
        carry = ""
    return out


def latest(path: Path, key) -> dict:
    """{key(r): r} keeping the last non-error record per key (resumed judge files append)."""
    out = {}
    for r in rj(path):
        k = key(r)
        if r.get("api_error") and k in out:
            continue
        out[k] = r
    return out


# ============================================================================= Task 2: non-circular 2x2

def sent_front(s: str, mt: dict) -> str | None:
    c = first_claim_v3([s])
    return c[1] if c and c[1] in (mt["front_name"], mt["back_name"]) else None


def implied(front_claimed: str, mt: dict, phrasing: str) -> str:
    yes = (front_claimed == mt["x_name"]) if phrasing == "front" else (front_claimed != mt["x_name"])
    return "A" if yes else "B"


def two_by_two(rows: list[tuple], rng) -> dict:
    """rows: (cluster, true(bool), consistent(bool)). P(cons|true), P(cons|false), diff with cluster bootstrap."""
    if not rows:
        return {"n": 0}
    cl = np.array([r[0] for r in rows])
    t = np.array([r[1] for r in rows], bool)
    c = np.array([r[2] for r in rows], bool)
    uc, inv = np.unique(cl, return_inverse=True)
    agg = np.zeros((len(uc), 4))
    np.add.at(agg, inv, np.stack([t & c, t, ~t & c, ~t], 1).astype(float))

    def stat(a):
        s = a.sum(0)
        pt = s[0] / s[1] if s[1] else np.nan
        pf = s[2] / s[3] if s[3] else np.nan
        return pt, pf, pt - pf

    est = stat(agg)
    bs = np.array([stat(agg[rng.integers(0, len(uc), len(uc))]) for _ in range(NB)])
    n_tc, n_t, n_fc, n_f = (int(x) for x in agg.sum(0))
    return {"n": len(rows), "cells": {"true_cons": n_tc, "true_incons": n_t - n_tc, "false_cons": n_fc, "false_incons": n_f - n_fc},
            "p_cons_true": {"mean": round(100 * est[0], 1), "ci": [round(100 * x, 1) for x in q95(bs[:, 0], 4)], "wilson": wilson(n_tc, n_t), "n": n_t},
            "p_cons_false": {"mean": round(100 * est[1], 1), "ci": [round(100 * x, 1) for x in q95(bs[:, 1], 4)], "wilson": wilson(n_fc, n_f), "n": n_f},
            "diff": {"mean": round(100 * est[2], 1), "ci": [round(100 * x, 1) for x in q95(bs[:, 2], 4)]},
            "share_true": round(100 * n_t / len(rows), 1)}


def fmt22(r: dict) -> str:
    if not r.get("n"):
        return "n=0"
    t, f, d = r["p_cons_true"], r["p_cons_false"], r["diff"]
    return (f"n={r['n']} (true {t['n']}, false {f['n']}) P(cons|true) {t['mean']} [{t['ci'][0]}, {t['ci'][1]}] "
            f"P(cons|false) {f['mean']} [{f['ci'][0]}, {f['ci'][1]}] diff {d['mean']:+.1f} [{d['ci'][0]:+.1f}, {d['ci'][1]:+.1f}] cells {r['cells']}")


def judge_truth_cot() -> dict:
    """{(model, item_id): (idx, 'CORRECT'|'INCORRECT')} two-judge consensus (e20 prompt) for Task-1 sentences."""
    J = {j: latest(E30 / f"judge_{j}_commit_e20.jsonl", lambda r: (r["model"], r["item_id"], r["idx"], r["group"])) for j in J2}
    out = {}
    for k, a in J[J2[0]].items():
        b = J[J2[1]].get(k)
        if b and a["label"] == b["label"] and a["label"] in ("CORRECT", "INCORRECT"):
            out[(k[0], k[1])] = (k[2], a["label"], k[3])
    return out


def desc_roles() -> dict:
    """{(model, base_id): consensus described-front role} over all 800 descriptions per model (Task-1 descall)."""
    J = {j: latest(E30 / f"judge_{j}_descall.jsonl", lambda r: (r["model"], r["item_id"])) for j in J2}
    out = {}
    for k, a in J[J2[0]].items():
        b = J[J2[1]].get(k)
        if not b:
            continue
        ra = claimed_role(a["front_claimed"], a["front_name"], a["back_name"])
        rb = claimed_role(b["front_claimed"], b["front_name"], b["back_name"])
        out[k] = ra if ra == rb else "NC"
    return out


def follow(lines: list[str]) -> dict:
    res = {}
    jt = judge_truth_cot()
    dr = desc_roles()
    for m in COMP:
        rng = np.random.default_rng(300)
        T = rj(E28 / f"{m}_cues_traj.jsonl")
        R = {(r["item_id"], r["condition"]): r for r in rj(O / "e10" / f"{m}_cues.jsonl") if r["condition"] in ("direct", "cot")}
        defs = defaultdict(list)
        cover = Counter()
        for r in T:
            mt, ph, iid = r["meta"], r["phrasing"], r["item_id"]
            ans = pe(R[(iid, "cot")])
            d_ok = pe(R[(iid, "direct")]) == r["gold"]
            c_ok = ans == r["gold"]
            cl = [(j, sent_front(s, mt)) for j, s in enumerate(r["sentences"])]
            cl = [(j, f) for j, f in cl if f]
            body = [(j, f) for j, f in cl if not CONCL22.search(r["sentences"][j])]
            pick = {"last": cl[-1] if cl else None, "last_body": body[-1] if body else None, "first": cl[0] if cl else None}
            cj = [x for x in cl if x[0] == r["commit_idx"]] if r["commit_idx"] is not None else []
            pick["commit_or_last"] = cj[0] if cj else pick["last"]
            cover["n"] += 1
            for k, v in pick.items():
                if v is None:
                    continue
                cover[k] += 1
                tr = v[1] == mt["front_name"]
                cons = implied(v[1], mt, ph) == ans
                for sub in ("all", "D+" if d_ok else "D-", f"C{'+' if c_ok else '-'}", ph, f"{'D+' if d_ok else 'D-'}|{ph}"):
                    defs[(k, sub)].append((r["image_id"], tr, cons))
            # judge truth on Direct-right items (Task-1 sentences): implied answer = gold if true else the other letter
            if d_ok and (m, iid) in jt:
                idx, lab, grp = jt[(m, iid)]
                tr = lab == "CORRECT"
                imp = r["gold"] if tr else ("B" if r["gold"] == "A" else "A")
                for sub in ("D+", f"D+|{ph}"):
                    defs[("judge_commit_or_comparator", sub)].append((r["image_id"], tr, imp == ans))
                cover["judge"] += 1
        out = {"coverage": dict(cover)}
        # validation: parser truth vs two-judge consensus on the Task-1 sentences that the parser parses
        Tm = {r["item_id"]: r for r in T}
        pv = []
        for (mm, iid), (idx, lab, grp) in jt.items():
            if mm != m:
                continue
            f_ = sent_front(Tm[iid]["sentences"][idx], Tm[iid]["meta"])
            if f_:
                pv.append(((f_ == Tm[iid]["meta"]["front_name"]) == (lab == "CORRECT"), grp))
        out["parser_vs_judge_cot"] = {"n_judged_consensus": sum(1 for k in jt if k[0] == m), "n_parsed": len(pv),
                                      "agree": pct(sum(a for a, _ in pv), len(pv)),
                                      "agree_commits": pct(sum(a for a, g_ in pv if g_ == "D+C-"), sum(g_ == "D+C-" for _, g_ in pv))}
        for (k, sub), rows in sorted(defs.items()):
            out[f"cot|{k}|{sub}"] = two_by_two(rows, rng)
        # description arm
        D = {r["item_id"]: r for r in rj(E28 / f"{m}_cues_desc.jsonl") if r["condition"] == "desc"}
        ddefs = defaultdict(list)
        dcover = Counter()
        for iid, r in D.items():
            mt, ph = r["meta"], r["phrasing"]
            ans = pe(r)
            d_ok = pe(R[(iid, "direct")]) == r["gold"]
            role = dr.get((m, r["base_id"]))
            dcover["n"] += 1
            dcover[f"judge_{role}"] += 1
            if role in ("front", "back"):
                tr = role == "front"
                cons = implied_answer(role, mt, ph) == ans
                for sub in ("all", "D+" if d_ok else "D-", ph, f"{'D+' if d_ok else 'D-'}|{ph}", "desc+" if ans == r["gold"] else "desc-"):
                    ddefs[("judge", sub)].append((r["image_id"], tr, cons))
            sents = split_sentences(r["text"])
            cl = [f for f in (sent_front(s, mt) for s in sents) if f]
            for k, v in (("parser_last", cl[-1] if cl else None), ("parser_first", cl[0] if cl else None)):
                if v is None:
                    continue
                dcover[k] += 1
                tr = v == mt["front_name"]
                cons = implied(v, mt, ph) == ans
                for sub in ("all", "D+" if d_ok else "D-", ph):
                    ddefs[(k, sub)].append((r["image_id"], tr, cons))
        out["desc_coverage"] = dict(dcover)
        for (k, sub), rows in sorted(ddefs.items()):
            out[f"desc|{k}|{sub}"] = two_by_two(rows, rng)
        # parser vs judge agreement on the described front (validation of the parser numbers)
        agree = [(dr.get((m, r["base_id"])), ("front" if (cl := [f for f in (sent_front(s, r["meta"]) for s in split_sentences(r["text"])) if f])
                                              and cl[-1] == r["meta"]["front_name"] else "back" if cl else None))
                 for r in D.values() if r["phrasing"] == "front"]
        ok = [(a, b) for a, b in agree if a in ("front", "back") and b]
        out["desc_parser_vs_judge"] = {"n": len(ok), "agree_pct": round(100 * np.mean([a == b for a, b in ok]), 1) if ok else None}
        res[m] = out
        cov = out["coverage"]
        lines.append(f"FOLLOW {m} CoT coverage (of {cov['n']} items): last {cov.get('last')} last_body {cov.get('last_body')} first {cov.get('first')} "
                     f"commit_or_last {cov.get('commit_or_last')} judge(D+ only) {cov.get('judge')} | parser-vs-judge truth on judged sentences "
                     f"{out['parser_vs_judge_cot']['agree']['k']}/{out['parser_vs_judge_cot']['agree']['n']} = {out['parser_vs_judge_cot']['agree']['pct']}% "
                     f"(commits {out['parser_vs_judge_cot']['agree_commits']['pct']}%)")
        for k in ("last", "last_body", "first", "commit_or_last"):
            for sub in ("all", "D+", "D-", "C+", "C-", "front", "behind", "D+|front", "D+|behind"):
                if f"cot|{k}|{sub}" in out:
                    lines.append(f"  FOLLOW {m} CoT {k:15s} {sub:9s} {fmt22(out[f'cot|{k}|{sub}'])}")
        for sub in ("D+", "D+|front", "D+|behind"):
            if f"cot|judge_commit_or_comparator|{sub}" in out:
                lines.append(f"  FOLLOW {m} CoT judge(j*/comparator) {sub:9s} {fmt22(out[f'cot|judge_commit_or_comparator|{sub}'])}")
        lines.append(f"FOLLOW {m} Desc coverage {out['desc_coverage']} | parser-vs-judge described front agree {out['desc_parser_vs_judge']}")
        for k in ("judge", "parser_last", "parser_first"):
            for sub in ("all", "D+", "D-", "front", "behind", "D+|front", "D+|behind", "desc+", "desc-"):
                if f"desc|{k}|{sub}" in out:
                    lines.append(f"  FOLLOW {m} Desc {k:12s} {sub:9s} {fmt22(out[f'desc|{k}|{sub}'])}")
    return res


# ============================================================================= Task 3a: logistic model on Exp-29 edits

def logit_fit(X: np.ndarray, y: np.ndarray, ridge: float = 1e-6) -> tuple[np.ndarray, float]:
    """Logistic regression by IRLS; returns (beta, log-likelihood)."""
    b = np.zeros(X.shape[1])
    for _ in range(100):
        eta = np.clip(X @ b, -30, 30)
        p = 1 / (1 + np.exp(-eta))
        W = p * (1 - p) + 1e-9
        H = X.T @ (X * W[:, None]) + ridge * np.eye(X.shape[1])
        step = np.linalg.solve(H, X.T @ (y - p) - ridge * b)
        b += step
        if np.abs(step).max() < 1e-8:
            break
    eta = np.clip(X @ b, -30, 30)
    ll = float(np.sum(y * eta - np.log1p(np.exp(eta))))
    return b, ll


def design(recs: list[dict], terms: list[str]) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    cols, names = [], []
    f = {k: np.array([r[k] for r in recs], float) for k in ("pm", "sm", "sp_front", "gold_yes", "qp_front", "w1", "w2", "neg")}
    f["sp_x_sm"] = f["sp_front"] * f["sm"]
    f["sp_x_pm"] = f["sp_front"] * f["pm"]
    f["sp_x_gold"] = f["sp_front"] * f["gold_yes"]
    for t in terms:
        cols.append(f[t])
        names.append(t)
    X = np.column_stack([np.ones(len(recs))] + cols)
    y = np.array([r["acc"] for r in recs], float)
    g = np.array([r["image_id"] for r in recs])
    return X, y, g, ["const"] + names


def fit_report(recs: list[dict], terms: list[str], rng, nb: int = 1000) -> dict:
    X, y, g, names = design(recs, terms)
    b, ll = logit_fit(X, y)
    ug = np.unique(g)
    gi = {u: np.where(g == u)[0] for u in ug}
    bs = []
    for _ in range(nb):
        s = np.concatenate([gi[u] for u in rng.choice(ug, len(ug))])
        bs.append(logit_fit(X[s], y[s])[0])
    bs = np.array(bs)
    ll0 = logit_fit(X[:, :1], y)[1]
    out = {"n": len(y), "ll": round(ll, 2), "ll_null": round(ll0, 2), "mcfadden_r2": round(1 - ll / ll0, 3), "aic": round(2 * len(b) - 2 * ll, 1),
           "coef": {nm: {"b": round(float(b[i]), 3), "ci": q95(bs[:, i], 3)} for i, nm in enumerate(names)}, "drop_one": {}}
    for i, nm in enumerate(names[1:], 1):
        keep = [c for c in range(X.shape[1]) if c != i]
        ll_d = logit_fit(X[:, keep], y)[1]
        out["drop_one"][nm] = {"lr_chi2": round(2 * (ll - ll_d), 2), "delta_aic": round(2 * (ll - ll_d) - 2, 1),
                               "r2_loss": round((ll - ll_d) / -ll0, 3)}
    return out


def edit_rows(m: str, st: str, part: str) -> list[dict]:
    rows = []
    for r in rj(E29 / f"{m}_{st}_edit_{part}.jsonl"):
        a = r["arm"]
        if a[0] not in "FB" and a != "ECHO":
            continue
        mt, ph = r["meta"], r["phrasing"]
        if a == "ECHO":
            sp, subj, neg = ph, mt["x_name"], r["gold"] == "B"
            w = 3
        else:
            sp = "front" if a[0] == "F" else "behind"
            subj = mt["front_name"] if a[0] == "F" else mt["back_name"]
            neg, w = False, int(a[1])
        rows.append({"arm": a, "image_id": r["image_id"], "phrasing": ph, "acc": float(pe(r) == r["gold"]), "yes": float(pe(r) == "A"),
                     "pm": float(sp == ph), "sm": float(subj == mt["x_name"]), "sp_front": float(sp == "front"),
                     "gold_yes": float(r["gold"] == "A"), "qp_front": float(ph == "front"), "w1": float(w == 1), "w2": float(w == 2),
                     "neg": float(neg), "wording": w})
    return rows


def cell_table(rows: list[dict], rng) -> dict:
    out = {}
    for sp in (1.0, 0.0):
        for qp in (1.0, 0.0):
            for gy in (1.0, 0.0):
                s = [r for r in rows if r["sp_front"] == sp and r["qp_front"] == qp and r["gold_yes"] == gy]
                if not s:
                    continue
                k = f"sent_{'front' if sp else 'behind'}|q_{'front' if qp else 'behind'}|gold_{'yes' if gy else 'no'}"
                out[k] = {"pm": int(s[0]["pm"]), "sm": int(s[0]["sm"]), "acc": cboot([r["acc"] for r in s], [r["image_id"] for r in s], rng, 1000),
                          "yes": round(100 * float(np.mean([r["yes"] for r in s])), 1)}
    return out


def logit(lines: list[str]) -> dict:
    res = {}
    M_MAIN = ["pm", "sm", "sp_front", "gold_yes", "qp_front", "w1", "w2"]
    M_INT = M_MAIN + ["sp_x_sm"]
    for m in COMP:
        for st in ("cues", "twins"):
            for part in ("jstar", "dcp"):
                rows = edit_rows(m, st, part)
                if not rows:
                    continue
                rng = np.random.default_rng(301)
                fb = [r for r in rows if r["arm"] != "ECHO"]
                r_ = {"n": len(fb), "cells": cell_table(fb, rng), "main": fit_report(fb, M_MAIN, rng), "inter": fit_report(fb, M_INT, rng)}
                # single-term models (which one term explains most)
                r_["single"] = {t: fit_report(fb, [t], rng, nb=200)["mcfadden_r2"] for t in ("pm", "sm", "sp_front", "gold_yes", "qp_front")}
                r_["single"]["sp_front+sp_x_sm"] = fit_report(fb, ["sp_front", "sp_x_sm"], rng, nb=200)["mcfadden_r2"]
                r_["single"]["sp_front+sp_x_pm"] = fit_report(fb, ["sp_front", "sp_x_pm"], rng, nb=200)["mcfadden_r2"]
                ec = [r for r in rows if r["arm"] == "ECHO"]
                if ec:
                    r_["echo"] = {g: cboot([r["acc"] for r in ec if r["gold_yes"] == g], [r["image_id"] for r in ec if r["gold_yes"] == g], rng, 1000)
                                  for g in (1.0, 0.0)}
                res[f"{m}|{st}|{part}"] = r_
                c, f_ = r_["main"]["coef"], r_["inter"]["coef"]
                lines.append(f"LOGIT {m:11s} {st:5s} {part:5s} n={len(fb)} | main R2 {r_['main']['mcfadden_r2']} AIC {r_['main']['aic']} | "
                             + " ".join(f"{t} {c[t]['b']:+.2f} {c[t]['ci']}" for t in M_MAIN[:5])
                             + f" | +sp_x_sm R2 {r_['inter']['mcfadden_r2']} AIC {r_['inter']['aic']}: "
                             + " ".join(f"{t} {f_[t]['b']:+.2f} {f_[t]['ci']}" for t in ("pm", "sm", "sp_front", "sp_x_sm", "gold_yes")))
                lines.append(f"   drop-one LR chi2 (main): " + ", ".join(f"{t} {v['lr_chi2']}" for t, v in r_["main"]["drop_one"].items())
                             + f" | single-term R2 {r_['single']}")
                for k, v in r_["cells"].items():
                    lines.append(f"   cell {k:34s} pm={v['pm']} sm={v['sm']} acc {fm(v['acc'])} (n={v['acc']['n']}) yes {v['yes']}%")
                if ec:
                    lines.append(f"   ECHO gold yes {fm(r_['echo'][1.0])} gold no (negated) {fm(r_['echo'][0.0])}")
    return res


# ============================================================================= Task 1: full two-judge audit

def rate_e20(labs: list[str]) -> dict:
    """labs: per-sentence consensus in {INCORRECT, CORRECT, NONCONS}; Wilson CI on judgeable, worst-case bounds on all."""
    k, c, n = labs.count("INCORRECT"), labs.count("CORRECT"), len(labs)
    return pct(k, k + c) | {"n_items": n, "worst_case": [round(100 * k / n, 1) if n else None, round(100 * (n - c) / n, 1) if n else None]}


def audit(lines: list[str]) -> dict:
    key = lambda r: (r["model"], r["item_id"], r["idx"], r["group"])
    E = {j: latest(E30 / f"judge_{j}_commit_e20.jsonl", key) for j in J2}
    Q = {j: latest(E30 / f"judge_{j}_commit_ontop.jsonl", key) for j in J2}
    keys = sorted(set(E[J2[0]]) & set(E[J2[1]]) & set(Q[J2[0]]) & set(Q[J2[1]]))
    meta = {k: E[J2[1]][k] for k in keys}
    lab = lambda J, j, k: J[j][k]["label"] if not J[j][k].get("api_error") else "APIERR"
    c20 = {k: (lab(E, J2[0], k) if lab(E, J2[0], k) == lab(E, J2[1], k) and lab(E, J2[0], k) in ("CORRECT", "INCORRECT") else "NONCONS") for k in keys}
    own = {k: {"YES": "INCORRECT", "NO": "CORRECT"}.get(cons2(own_contra(Q[J2[0]][k]), own_contra(Q[J2[1]][k])), "NONCONS") for k in keys}
    dep = {k: {"YES": "INCORRECT", "NO": "CORRECT"}.get(cons2(depth_contra(Q[J2[0]][k]), depth_contra(Q[J2[1]][k])), "NONCONS") for k in keys}
    both = [k for k in keys if {lab(E, J2[0], k), lab(E, J2[1], k)} <= {"CORRECT", "INCORRECT"}]
    a = np.array([lab(E, J2[0], k) == "INCORRECT" for k in both])
    b = np.array([lab(E, J2[1], k) == "INCORRECT" for k in both])
    po = float(np.mean(a == b))
    pe_ = a.mean() * b.mean() + (1 - a.mean()) * (1 - b.mean())
    res = {"n_sentences": len(keys), "expected": {"D+C-": 517, "D+C+rel": 2120},
           "kappa_e20_clear": round((po - pe_) / (1 - pe_), 3), "agree_e20_all": round(100 * float(np.mean([lab(E, J2[0], k) == lab(E, J2[1], k) for k in keys])), 1),
           "agree_reading": round(100 * float(np.mean([Q[J2[0]][k]["reading"] == Q[J2[1]][k]["reading"] for k in keys])), 1),
           "api_errors_last": {j: sum(E[j][k].get("api_error", False) for k in keys) + sum(Q[j][k].get("api_error", False) for k in keys) for j in J2},
           "unparsed": {j: sum(E[j][k]["label"] == "UNPARSED" for k in keys) + sum("UNPARSED" in (Q[j][k]["reading"], Q[j][k]["depth_true"]) for k in keys) for j in J2},
           "rows": {}}
    e28 = {(r["model"], r["item_id"], r["idx"]) for r in rj(E28 / "judge_gpt-6-astra_commit_cues.jsonl")}
    p120 = {r["item_id"] for r in rj(E29 / "judge_gpt-6-astra_prej2.jsonl")}
    earliest = json.load(open(E29 / "earliest_qwen3vl_8b_cues.json"))
    subsets = {"all": lambda k: True, "front": lambda k: meta[k]["phrasing"] == "front", "behind": lambda k: meta[k]["phrasing"] == "behind",
               "jstar_relation": lambda k: meta[k]["stype"] == "relation", "jstar_conclusion": lambda k: meta[k]["stype"] == "conclusion",
               "jstar_nonrel": lambda k: meta[k]["stype"] == "non_relational", "in_e28_sample40": lambda k: k[:3] in e28,
               "not_in_e28_sample": lambda k: k[:3] not in e28,
               "prej120": lambda k: k[0] == "qwen3vl_8b" and k[1] in p120,
               "prej120_contra_before": lambda k: k[0] == "qwen3vl_8b" and k[1] in p120 and k[1] in earliest and earliest[k[1]] < meta[k]["commit_idx"]}
    for m in COMP + ["pooled"]:
        for g in ("D+C-", "D+C+rel"):
            for sn, sf in subsets.items():
                ks = [k for k in keys if k[3] == g and (m == "pooled" or k[0] == m) and sf(k)]
                if not ks or (g == "D+C+rel" and sn not in ("all", "front", "behind", "in_e28_sample40")):
                    continue
                row = {"n": len(ks), "e20": rate_e20([c20[k] for k in ks]), "own": rate_e20([own[k] for k in ks]),
                       "depth": rate_e20([dep[k] for k in ks]),
                       "opus_e20": pct(sum(lab(E, J2[0], k) == "INCORRECT" for k in ks), sum(lab(E, J2[0], k) in ("CORRECT", "INCORRECT") for k in ks)),
                       "gpt_e20": pct(sum(lab(E, J2[1], k) == "INCORRECT" for k in ks), sum(lab(E, J2[1], k) in ("CORRECT", "INCORRECT") for k in ks)),
                       "noclaim_either": sum("NOCLAIM" in (lab(E, J2[0], k), lab(E, J2[1], k)) for k in ks),
                       "reading": dict(Counter(Q[J2[0]][k]["reading"] if Q[J2[0]][k]["reading"] == Q[J2[1]][k]["reading"] else "NC" for k in ks))}
                res["rows"][f"{m}|{g}|{sn}"] = row
                f = lambda x: f"{x['k']}/{x['n']} = {x['pct']}% {x['wilson']} wc [{x['worst_case'][0]}, {x['worst_case'][1]}]"
                lines.append(f"AUDIT {m:11s} {g:8s} {sn:22s} n={len(ks):4d} | e20 {f(row['e20'])} | own-reading {f(row['own'])} | depth {f(row['depth'])} "
                             f"| Opus {row['opus_e20']['pct']} GPT {row['gpt_e20']['pct']} | NOCLAIM(either) {row['noclaim_either']}")
    # contrast commit - comparator with a cluster bootstrap over scenes (consensus, judgeable only)
    for m in COMP + ["pooled"]:
        for nm, C_ in (("e20", c20), ("own", own)):
            rows = [(meta[k]["image_id"], k[3] == "D+C-", C_[k] == "INCORRECT") for k in keys if C_[k] != "NONCONS" and (m == "pooled" or k[0] == m)]
            rng = np.random.default_rng(302)
            cl = np.array([r[0] for r in rows])
            gp = np.array([r[1] for r in rows])
            y = np.array([r[2] for r in rows], float)
            uc, inv = np.unique(cl, return_inverse=True)
            agg = np.zeros((len(uc), 4))
            np.add.at(agg, inv, np.stack([y * gp, gp, y * ~gp, ~gp], 1))
            st = lambda A: A[:, 0].sum() / A[:, 1].sum() - A[:, 2].sum() / A[:, 3].sum()
            bs = [st(agg[rng.integers(0, len(uc), len(uc))]) for _ in range(NB)]
            res["rows"][f"{m}|diff|{nm}"] = {"mean": round(100 * st(agg), 1), "ci": [round(100 * x, 1) for x in q95(bs, 4)]}
            lines.append(f"AUDIT {m:11s} commit - comparator ({nm}) {res['rows'][f'{m}|diff|{nm}']['mean']:+.1f} {res['rows'][f'{m}|diff|{nm}']['ci']}")
    lines.append(f"AUDIT sentences {len(keys)} (expected 517 commits + 2120 comparators) | e20 kappa(clear) {res['kappa_e20_clear']} agree {res['agree_e20_all']}% "
                 f"| reading agree {res['agree_reading']}% | api errors (last record) {res['api_errors_last']} unparsed {res['unparsed']}")
    return res


# ============================================================================= Task 3b: argument-order edit

AE = ["FS", "FO", "FN", "BS", "BO", "BN"]
# semantic content of each true template: which proposition it asserts ('front' = "F is in front of K",
# 'behind' = "K is behind F"), its surface subject (F / K) and surface predicate.
SEM = {"FS": ("front", "F", "front", False), "FO": ("front", "K", "front", False), "BN": ("front", "F", "behind", True),
       "BS": ("behind", "K", "behind", False), "BO": ("behind", "F", "behind", False), "FN": ("behind", "K", "front", True),
       "F0": ("front", "F", "front", False), "F1": ("front", "F", "front", False), "F2": ("front", "F", "front", False),
       "B0": ("behind", "K", "behind", False), "B1": ("behind", "K", "behind", False), "B2": ("behind", "K", "behind", False)}
QTYPES = ["identical", "negation", "converse", "swap"]


def qtype(assert_pred: str, phrasing: str, gold: str) -> str:
    """Relation of the question to the asserted proposition.

    identical: same subject and predicate (gold yes); negation: same subject, opposite predicate (gold no);
    converse: other subject, converse predicate (gold yes); swap: other subject, same predicate (gold no).
    """
    same_pred = assert_pred == phrasing
    if gold == "A":
        return "identical" if same_pred else "converse"
    return "swap" if same_pred else "negation"


def sem_row(r: dict, arm: str) -> dict:
    ap, ss, sp, neg = SEM[arm]
    mt = r["meta"]
    subj = mt["front_name"] if ss == "F" else mt["back_name"]
    qt = qtype(ap, r["phrasing"], r["gold"])
    # literal: the question repeats the sentence's surface subject and predicate (with or without its negation)
    literal = subj == mt["x_name"] and sp == r["phrasing"]
    return {"arm": arm, "image_id": r["image_id"], "acc": float(pe(r) == r["gold"]), "yes": float(pe(r) == "A"), "qt": qt,
            "assert_front": float(ap == "front"), "surf_sm": float(subj == mt["x_name"]), "surf_pm": float(sp == r["phrasing"]),
            "neg": float(neg), "literal": float(literal)}


def rows_sem(R: list[dict]) -> list[dict]:
    return [sem_row(r, r["arm"]) for r in R if r["arm"] in SEM]


def edit_rows_sem(m: str, st: str, part: str) -> list[dict]:
    return [sem_row(r, r["arm"]) for r in rj(E29 / f"{m}_{st}_edit_{part}.jsonl") if r["arm"] in SEM]


def semantic(rows: list[dict], rng) -> dict:
    out = {"cells": {}}
    for ap in (1.0, 0.0):
        for qt in QTYPES:
            s = [r for r in rows if r["assert_front"] == ap and r["qt"] == qt]
            if s:
                out["cells"][f"{'front' if ap else 'behind'}|{qt}"] = cboot([r["acc"] for r in s], [r["image_id"] for r in s], rng, 1000)
        for qt in QTYPES:
            for arm in sorted({r["arm"] for r in rows if r["assert_front"] == ap}):
                s = [r for r in rows if r["arm"] == arm and r["qt"] == qt]
                if s:
                    out["cells"][f"arm_{arm}|{qt}"] = round(100 * float(np.mean([r["acc"] for r in s])), 1)
    X = []
    for r in rows:
        X.append({**r, "pm": float(r["qt"] in ("identical", "swap")), "sm": r["surf_sm"], "sp_front": r["assert_front"],
                  "gold_yes": float(r["qt"] in ("identical", "converse")), "qp_front": 0.0, "w1": float(r["qt"] == "converse"),
                  "w2": float(r["qt"] == "swap")})
    # query type as dummies (w1 = converse, w2 = swap, pm-gold coding gives negation), surface subject match, negation
    # model A: query type (identical / negation / converse / swap) + asserted predicate + surface subject match + negated
    rowsA = [{**x, "qp_front": float(x["qt"] == "negation")} for x in X]
    out["logit_qtype"] = fit_report(rowsA, ["qp_front", "w1", "w2", "sp_front", "sm", "neg"], rng, 300)
    out["logit_qtype"]["names"] = {"qp_front": "negation", "w1": "converse", "w2": "swap", "sp_front": "asserts_front", "sm": "surface_subject_match", "neg": "negated"}
    return out


def semantic_lines(tag: str, S: dict) -> list[str]:
    c = S["cells"]
    L = [f"{tag} acc by asserted proposition x query type: " + " | ".join(
        f"{ap}: " + ", ".join(f"{qt} {c[f'{ap}|{qt}']['mean']} [{c[f'{ap}|{qt}']['ci'][0]}, {c[f'{ap}|{qt}']['ci'][1]}] (n={c[f'{ap}|{qt}']['n']})"
                              for qt in QTYPES if f"{ap}|{qt}" in c) for ap in ("front", "behind"))]
    L.append(f"{tag} per arm: " + " | ".join(f"{k[4:]} {v}" for k, v in c.items() if k.startswith("arm_")))
    lg = S["logit_qtype"]
    L.append(f"{tag} logit (ref = identical): R2 {lg['mcfadden_r2']} | " + " ".join(
        f"{lg['names'][t]} {v['b']:+.2f} {v['ci']} (chi2 {lg['drop_one'][t]['lr_chi2']})" for t, v in lg["coef"].items() if t != "const"))
    return L


def argedit(lines: list[str]) -> dict:
    res = {}
    for m in COMP:
        for st in ("cues", "twins"):
            for part in ("jstar", "dcp"):
                R = rj(E30 / f"{m}_{st}_argedit_{part}.jsonl")
                if not R:
                    continue
                rng = np.random.default_rng(303)
                idx = {(r["item_id"], r["arm"]): r for r in R}
                keys = sorted({r["item_id"] for r in R})
                g = [idx[(k, "orig")]["image_id"] for k in keys]
                out = {"n": len(keys), "acc": {}, "cells": {}}
                for a in ["orig", "neutral"] + AE:
                    out["acc"][a] = cboot([float(pe(idx[(k, a)]) == idx[(k, a)]["gold"]) for k in keys], g, rng, 1000)
                for a in AE:
                    out["acc"][f"{a}-neutral"] = cboot([float(pe(idx[(k, a)]) == idx[(k, a)]["gold"]) - float(pe(idx[(k, 'neutral')]) == idx[(k, 'neutral')]["gold"]) for k in keys], g, rng, 1000)
                # cells: arm x question predicate x gold -> accuracy, P(yes), subject match
                for a in AE:
                    for ph in ("front", "behind"):
                        for gy in ("A", "B"):
                            ks = [k for k in keys if idx[(k, a)]["phrasing"] == ph and idx[(k, a)]["gold"] == gy]
                            if not ks:
                                continue
                            rr = [idx[(k, a)] for k in ks]
                            out["cells"][f"{a}|q_{ph}|gold_{'yes' if gy == 'A' else 'no'}"] = {
                                "n": len(ks), "sm": int(rr[0]["subject_match"]), "pm": int(rr[0]["pred_match"]),
                                "acc": cboot([float(pe(r) == r["gold"]) for r in rr], [r["image_id"] for r in rr], rng, 1000),
                                "yes": round(100 * float(np.mean([pe(r) == "A" for r in rr])), 1)}
                # logistic model on the six arms
                rows = []
                for k in keys:
                    for a in AE:
                        r = idx[(k, a)]
                        rows.append({"arm": a, "image_id": r["image_id"], "acc": float(pe(r) == r["gold"]), "pm": float(r["pred_match"]),
                                     "sm": float(r["subject_match"]), "sp_front": float(r["sent_pred"] == "front"), "gold_yes": float(r["gold"] == "A"),
                                     "qp_front": float(r["phrasing"] == "front"), "neg": float(r["negated"]), "w1": 0.0, "w2": 0.0})
                aff = [r for r in rows if not r["neg"]]
                out["logit_affirmative"] = fit_report(aff, ["pm", "sm", "sp_front", "gold_yes", "qp_front"], rng, 500)
                out["logit_affirmative_int"] = fit_report(aff, ["pm", "sm", "sp_front", "gold_yes", "qp_front", "sp_x_sm", "sp_x_pm"], rng, 500)
                out["logit_all"] = fit_report(rows, ["pm", "sm", "sp_front", "gold_yes", "qp_front", "neg"], rng, 500)
                out["semantic"] = semantic(rows_sem(R), rng)
                res[f"{m}|{st}|{part}"] = out
                A = out["acc"]
                lines += semantic_lines(f"ARGEDIT-SEM {m:11s} {st:5s} {part:5s}", out["semantic"])
                e29r = edit_rows_sem(m, st, part)
                if e29r:
                    out["semantic_e29"] = semantic(e29r, rng)
                    lines += semantic_lines(f"E29-SEM     {m:11s} {st:5s} {part:5s}", out["semantic_e29"])
                lines.append(f"ARGEDIT {m:11s} {st:5s} {part:5s} n={len(keys)} | " + " | ".join(f"{a} {fm(A[a])}" for a in ["orig", "neutral"] + AE))
                lines.append("   vs neutral: " + " | ".join(f"{a} {fa(A[f'{a}-neutral'])}" for a in AE))
                for a in AE:
                    lines.append(f"   {a}: " + " | ".join(f"q_{ph}/gold_{gy}: acc {out['cells'][f'{a}|q_{ph}|gold_{gy}']['acc']['mean']} (n={out['cells'][f'{a}|q_{ph}|gold_{gy}']['n']}, "
                                                     f"sm={out['cells'][f'{a}|q_{ph}|gold_{gy}']['sm']}, yes {out['cells'][f'{a}|q_{ph}|gold_{gy}']['yes']}%)"
                                                     for ph in ("front", "behind") for gy in ("yes", "no") if f"{a}|q_{ph}|gold_{gy}" in out["cells"]))
                for nm in ("logit_affirmative", "logit_affirmative_int", "logit_all"):
                    L_ = out[nm]
                    lines.append(f"   {nm}: R2 {L_['mcfadden_r2']} | " + " ".join(f"{t} {v['b']:+.2f} {v['ci']}" for t, v in L_["coef"].items() if t != "const")
                                 + " | drop-one chi2 " + ", ".join(f"{t} {v['lr_chi2']}" for t, v in L_["drop_one"].items()))
    return res


# ============================================================================= Task 4: occlusion-neutral description

DEPTHW = re.compile(r"\b(overlap\w*|cover\w*|hid(e|es|den|ing)|obscur\w*|occlu\w*|behind|in front|on top|underneath|beneath|"
                    r"partially visible|partly visible|foreground|background shape|layer\w*)\b", re.I)


def glm_multi(recs_by: dict, rng, nb: int) -> dict:
    from rigor_e17f import glm_rows, probit_fit
    data = {c: glm_rows(rs, use_size=True) for c, rs in recs_by.items()}
    pairs = sorted(set(data["direct"]["pair"]))
    pidx = {c: {p: np.where(np.array(d["pair"]) == p)[0] for p in pairs} for c, d in data.items()}
    est = {c: probit_fit(d["X"], d["y"]) for c, d in data.items()}
    bs = {c: [] for c in data}
    for _ in range(nb):
        js = rng.integers(0, len(pairs), len(pairs))
        for c, d in data.items():
            s = np.concatenate([pidx[c][pairs[j]] for j in js])
            bs[c].append(probit_fit(d["X"][s], d["y"][s]))
    bs = {c: np.array(v) for c, v in bs.items()}
    names = ["occlusion", "vertical", "size"]
    out = {}
    for c in data:
        out[c] = {k: {"d": round(2 * float(est[c][2 + i]), 2), "ci": q95(2 * bs[c][:, 2 + i])} for i, k in enumerate(names)}
        if c != "direct":
            out[f"{c}-direct"] = {k: {"d": round(2 * float(est[c][2 + i] - est["direct"][2 + i]), 2),
                                      "ci": q95(2 * (bs[c][:, 2 + i] - bs["direct"][:, 2 + i]))} for i, k in enumerate(names)}
            b, b0 = est[c][2:5], est["direct"][2:5]
            s = float(b @ b0 / (b0 @ b0))
            sb = [float(bs[c][i][2:5] @ bs["direct"][i][2:5] / (bs["direct"][i][2:5] @ bs["direct"][i][2:5])) for i in range(nb)]
            out[f"{c}-direct"]["common_scale_s"] = {"est": round(s, 3), "ci": q95(sb, 3)}
    for a1, a0 in (("ndesc", "desc"), ("ndesc", "filler_desc"), ("desc", "filler_desc"), ("ndesc", "cot")):
        if a1 in est and a0 in est:
            out[f"{a1}-{a0}"] = {"occlusion": {"d": round(2 * float(est[a1][2] - est[a0][2]), 2), "ci": q95(2 * (bs[a1][:, 2] - bs[a0][:, 2]))}}
    return out


def ndesc(lines: list[str], nb_glm: int = 1000) -> dict:
    from rigor_e28 import cue_units, pair_sdt
    res = {}
    for m in COMP:
        N = rj(E30 / f"{m}_cues_ndesc.jsonl")
        if not N:
            continue
        rng = np.random.default_rng(304)
        by = {c: {r["item_id"]: r for r in N if r["condition"] == c} for c in ("ndesc", "filler_desc", "filler_ndesc", "filler_turn", "direct_reread")}
        R = rj(O / "e10" / f"{m}_cues.jsonl")
        for c in ("direct", "cot"):
            by[c] = {r["item_id"]: r for r in R if r["condition"] == c}
        by["desc"] = {r["item_id"]: r for r in rj(E28 / f"{m}_cues_desc.jsonl") if r["condition"] == "desc"}
        keys = sorted(by["ndesc"])
        assert all(set(keys) <= set(by[c]) for c in by), m
        arms = ["direct", "cot", "desc", "ndesc", "filler_desc", "filler_ndesc", "filler_turn"]
        g = [by["direct"][k]["image_id"] for k in keys]
        acc = {c: np.array([float(pe(by[c][k]) == by[c][k]["gold"]) for k in keys]) for c in arms}
        texts = {by["ndesc"][k]["base_id"]: by["ndesc"][k] for k in keys}
        dtexts = {by["desc"][k]["base_id"]: by["desc"][k] for k in keys}
        nd_depth = np.mean([bool(DEPTHW.search(t["text"])) for t in texts.values()])
        od_depth = np.mean([bool(DEPTHW.search(t["text"])) for t in dtexts.values()])
        nd_claim = np.mean([any(sent_front(s, t["meta"]) for s in split_sentences(t["text"])) for t in texts.values()])
        od_claim = np.mean([any(sent_front(s, t["meta"]) for s in split_sentences(t["text"])) for t in dtexts.values()])
        r = {"n": len(keys), "direct_reread_same": round(100 * float(np.mean([pe(by["direct_reread"][k]) == pe(by["direct"][k]) for k in keys])), 2),
             "ndesc_tokens": round(float(np.mean([t["gen_len"] for t in texts.values()])), 1),
             "ndesc_trunc": round(100 * float(np.mean([t["trunc"] for t in texts.values()])), 1),
             "desc_tokens": round(float(np.mean([int(t["gen_len"]) for t in dtexts.values()])), 1),
             "filler_tokens": round(float(np.mean([by["filler_desc"][k]["filler_len"] for k in keys])), 1),
             "filler_target": round(float(np.mean([by["filler_desc"][k]["target_len"] for k in keys])), 1),
             "filler_len_exact": round(100 * float(np.mean([by["filler_desc"][k]["filler_len"] == by["filler_desc"][k]["target_len"] for k in keys])), 1),
             "ndesc_depth_wording_pct": round(100 * float(nd_depth), 1), "desc_depth_wording_pct": round(100 * float(od_depth), 1),
             "ndesc_parsed_claim_pct": round(100 * float(nd_claim), 1), "desc_parsed_claim_pct": round(100 * float(od_claim), 1)}
        for c in arms:
            r[f"acc_{c}"] = cboot(acc[c], g, rng)
            if c != "direct":
                r[f"acc_{c}-direct"] = cboot(acc[c] - acc["direct"], g, rng)
        for a1, a0 in (("ndesc", "desc"), ("ndesc", "filler_desc"), ("desc", "filler_desc"), ("ndesc", "filler_ndesc")):
            r[f"acc_{a1}-{a0}"] = cboot(acc[a1] - acc[a0], g, rng)
        # accuracy within the neutral arm split by whether the description uses depth wording
        dw = np.array([bool(DEPTHW.search(texts[by["ndesc"][k]["base_id"]]["text"])) for k in keys])
        for nm, s in (("with_depth_wording", dw), ("without_depth_wording", ~dw)):
            gg = [x for x, y in zip(g, s) if y]
            if s.sum():
                r[f"ndesc_{nm}"] = {"n": int(s.sum()), "ndesc": cboot(acc["ndesc"][s], gg, rng), "direct": cboot(acc["direct"][s], gg, rng),
                                    "ndesc-direct": cboot(acc["ndesc"][s] - acc["direct"][s], gg, rng)}
        r["glm"] = glm_multi({c: [by[c][k] for k in keys] for c in arms}, rng, nb_glm)
        Us = {c: cue_units([by[c][k] for k in keys]) for c in arms}
        r["dpair"] = pair_sdt(Us, lambda u: 2 * int(u["above"]) + int(u["larger"]), lambda u: int(u["above"]), rng,
                              [(c, "direct") for c in arms if c != "direct"] + [("ndesc", "desc"), ("ndesc", "filler_desc")])
        # GLM restricted to images whose neutral description has no depth wording (does the cost need occlusion talk?)
        for c in arms:
            for ph in ("front", "behind"):
                s = np.array([by["direct"][k]["phrasing"] == ph for k in keys])
                r[f"acc_{c}|{ph}"] = round(100 * float(acc[c][s].mean()), 1)
                r[f"yes_{c}|{ph}"] = round(100 * float(np.mean([pe(by[c][k]) == "A" for k, y in zip(keys, s) if y])), 1)
        keep_imgs = {u for u in texts if not DEPTHW.search(texts[u]["text"])}
        k2 = [k for k in keys if by["ndesc"][k]["base_id"] in keep_imgs]
        if len(k2) > 200:
            r["glm_no_depth_wording"] = glm_multi({c: [by[c][k] for k in k2] for c in ("direct", "ndesc", "filler_desc")}, rng, max(200, nb_glm // 4))
            r["glm_no_depth_wording"]["n"] = len(k2)
        res[m] = r
        G, P = r["glm"], r["dpair"]
        f2 = lambda x: f"{x['d']:+.2f} [{x['ci'][0]:+.2f}, {x['ci'][1]:+.2f}]"
        lines.append(f"NDESC {m:11s} n={r['n']} | tokens ndesc {r['ndesc_tokens']} (trunc {r['ndesc_trunc']}%) desc {r['desc_tokens']} filler {r['filler_tokens']} "
                     f"(target {r['filler_target']}, exact {r['filler_len_exact']}%) | depth wording ndesc {r['ndesc_depth_wording_pct']}% desc {r['desc_depth_wording_pct']}% "
                     f"| parsed two-object depth claim ndesc {r['ndesc_parsed_claim_pct']}% desc {r['desc_parsed_claim_pct']}% | Direct re-read same {r['direct_reread_same']}%")
        for c in arms:
            lines.append(f"   {c:12s} acc {fm(r[f'acc_{c}'])}" + (f" ({fa(r[f'acc_{c}-direct'])} vs Direct)" if c != "direct" else "")
                         + f" | occl {G[c]['occlusion']['d']:.2f} {G[c]['occlusion']['ci']}" + (f" Δ {f2(G[f'{c}-direct']['occlusion'])} s {G[f'{c}-direct']['common_scale_s']['est']}" if c != "direct" else "")
                         + f" | vert {G[c]['vertical']['d']:+.2f} | d'pair {P[f'{c}_pooled']:.2f}" + (f" Δ {f2(P[f'{c}-direct_pooled'])}" if c != "direct" else ""))
        lines.append(f"   ndesc-desc acc {fa(r['acc_ndesc-desc'])} occl {f2(G['ndesc-desc']['occlusion'])} d'pair {f2(P['ndesc-desc_pooled'])} | "
                     f"ndesc-filler_desc acc {fa(r['acc_ndesc-filler_desc'])} occl {f2(G['ndesc-filler_desc']['occlusion'])} d'pair {f2(P['ndesc-filler_desc_pooled'])} | "
                     f"desc-filler_desc acc {fa(r['acc_desc-filler_desc'])} occl {f2(G['desc-filler_desc']['occlusion'])}")
        lines.append("   acc front/behind (yes%): " + " | ".join(f"{c} {r[f'acc_{c}|front']}/{r[f'acc_{c}|behind']} ({r[f'yes_{c}|front']}/{r[f'yes_{c}|behind']})" for c in arms))
        for nm in ("with_depth_wording", "without_depth_wording"):
            if f"ndesc_{nm}" in r:
                x = r[f"ndesc_{nm}"]
                lines.append(f"   ndesc items {nm}: n={x['n']} ndesc {fm(x['ndesc'])} direct {fm(x['direct'])} diff {fa(x['ndesc-direct'])}")
        if "glm_no_depth_wording" in r:
            x = r["glm_no_depth_wording"]
            lines.append(f"   GLM on images whose neutral description has no depth wording (n={x['n']}): occl Direct {x['direct']['occlusion']['d']:.2f} "
                         f"ndesc {x['ndesc']['occlusion']['d']:.2f} Δ {f2(x['ndesc-direct']['occlusion'])} | filler_desc Δ {f2(x['filler_desc-direct']['occlusion'])}")
    return res


# ============================================================================= Task 5: gray-image prefix readout

def lo(p: float) -> float:
    p = min(max(p, 1e-6), 1 - 1e-6)
    return float(np.log(p / (1 - p)))


def gray(lines: list[str]) -> dict:
    res = {}
    for m in COMP:
        R = rj(E30 / f"{m}_cues_gray.jsonl")
        if not R:
            continue
        rng = np.random.default_rng(305)
        n = len(R)
        repro = [max(abs(a - b) for a, b in zip(r["p_gold_real"], r["p_gold_e28"])) for r in R]
        same_real = [r["commit_idx_real"] == r["commit_idx_e28"] for r in R]
        jr = [r["commit_idx_e28"] for r in R]
        jg = [r["commit_idx_gray"] for r in R]
        defined = [i for i in range(n) if jg[i] is not None]
        eq = [i for i in defined if jg[i] == jr[i]]
        # gray trajectory never at/above .5 before the end -> no gold-defined commit possible
        always_below = sum(all(p < .5 for p in r["p_gold_gray"]) for r in R)
        start_below = sum(r["p_gold_gray"][0] < .5 for r in R)
        # largest log-odds drop sentence (gold-free definition of the sentence that moves the readout most)
        md_r, md_g, corr, rank_g, d_r, d_g = [], [], [], [], [], []
        for r in R:
            lr = [lo(p) for p in r["p_gold_real"]]
            lg = [lo(p) for p in r["p_gold_gray"]]
            dr_ = np.diff(lr)
            dg_ = np.diff(lg)
            md_r.append(int(np.argmin(dr_)))
            md_g.append(int(np.argmin(dg_)))
            if len(dr_) >= 3 and np.std(dr_) > 0 and np.std(dg_) > 0:
                corr.append(float(np.corrcoef(dr_, dg_)[0, 1]))
            j = r["commit_idx_e28"]
            d_r.append(float(dr_[j]))
            d_g.append(float(dg_[j]))
            rank_g.append(int((dg_ < dg_[j]).sum()) + 1)
        js = np.array(jr)
        p = lambda key, off: [R[i][key][js[i] + off] for i in range(n)]
        out = {"n": n, "max_abs_real_minus_e28": round(float(np.max(repro)), 3), "real_jstar_reproduces_e28": pct(sum(same_real), n),
               "gray_jstar_defined": pct(len(defined), n), "gray_jstar_equals_real": pct(len(eq), len(defined)),
               "gray_jstar_equals_real_of_all": pct(len(eq), n),
               "gray_jstar_within1": pct(sum(abs(jg[i] - jr[i]) <= 1 for i in defined), len(defined)), "gray_always_below_half": pct(always_below, n), "gray_start_below_half": pct(start_below, n),
               "gray_jstar_offset": dict(sorted(Counter(jg[i] - jr[i] for i in defined).items())),
               "maxdrop_real_is_jstar": pct(sum(a == b for a, b in zip(md_r, jr)), n),
               "maxdrop_gray_is_jstar": pct(sum(a == b for a, b in zip(md_g, jr)), n),
               "maxdrop_gray_equals_real": pct(sum(a == b for a, b in zip(md_g, md_r)), n),
               "gray_drop_rank_at_jstar_le1": pct(sum(x == 1 for x in rank_g), n), "gray_drop_rank_at_jstar_le2": pct(sum(x <= 2 for x in rank_g), n),
               "corr_sentence_dlogit_median": round(float(np.median(corr)), 3) if corr else None, "n_corr": len(corr),
               "dlogit_at_jstar_real": round(float(np.mean(d_r)), 2), "dlogit_at_jstar_gray": round(float(np.mean(d_g)), 2),
               "dlogit_at_jstar_gray_neg": pct(sum(x < 0 for x in d_g), n)}
        for v in ("real", "gray"):
            for off, nm in ((0, "before"), (1, "after")):
                vals = p(f"p_gold_{v}", off)
                out[f"p_{v}_{nm}_jstar"] = {"mean": round(100 * float(np.mean(vals)), 1), "ge_half": pct(sum(x >= .5 for x in vals), n)}
            out[f"p_{v}_final"] = round(100 * float(np.mean([R[i][f"p_gold_{v}"][-1] for i in range(n)])), 1)
            out[f"p_{v}_start"] = round(100 * float(np.mean([R[i][f"p_gold_{v}"][0] for i in range(n)])), 1)
        res[m] = out
        lines.append(f"GRAY {m:11s} n={n} | real re-read vs e28 max|dP| {out['max_abs_real_minus_e28']}, j* reproduced {out['real_jstar_reproduces_e28']['pct']}% | "
                     f"gray j* defined {out['gray_jstar_defined']['k']}/{n} (gray start <.5: {start_below}, always <.5: {always_below}); "
                     f"gray j* = real j* {out['gray_jstar_equals_real']['k']}/{out['gray_jstar_equals_real']['n']} = {out['gray_jstar_equals_real']['pct']}% "
                     f"{out['gray_jstar_equals_real']['wilson']} (of all {out['gray_jstar_equals_real_of_all']['pct']}%), within 1 sentence "
                     f"{out['gray_jstar_within1']['pct']}% {out['gray_jstar_within1']['wilson']}, offsets {out['gray_jstar_offset']}")
        lines.append(f"   largest log-odds drop: real at j* {out['maxdrop_real_is_jstar']['pct']}% | gray at j* {out['maxdrop_gray_is_jstar']['pct']}% "
                     f"{out['maxdrop_gray_is_jstar']['wilson']} | gray = real {out['maxdrop_gray_equals_real']['pct']}% | gray drop at j* ranks 1st {out['gray_drop_rank_at_jstar_le1']['pct']}%, "
                     f"top-2 {out['gray_drop_rank_at_jstar_le2']['pct']}% | median per-rationale corr of sentence Δlogit real vs gray {out['corr_sentence_dlogit_median']} (n={out['n_corr']}) | "
                     f"mean Δlogit at j* real {out['dlogit_at_jstar_real']} gray {out['dlogit_at_jstar_gray']} (gray <0 in {out['dlogit_at_jstar_gray_neg']['pct']}%)")
        lines.append(f"   P(gold) before / after j*: real {out['p_real_before_jstar']['mean']} / {out['p_real_after_jstar']['mean']} | gray {out['p_gray_before_jstar']['mean']} / "
                     f"{out['p_gray_after_jstar']['mean']} (gray >= .5 before j*: {out['p_gray_before_jstar']['ge_half']['pct']}%, after: {out['p_gray_after_jstar']['ge_half']['pct']}%) "
                     f"| start real {out['p_real_start']} gray {out['p_gray_start']} | final real {out['p_real_final']} gray {out['p_gray_final']}")
    return res


# ============================================================================= table

def table() -> str:
    J = {k: json.load(open(B / f"e30_{k}.json")) for k in ("audit", "follow", "argedit", "ndesc", "gray")}
    pc = lambda c: f"{c['k']}/{c['n']} ({c['pct']:.1f}) {{\\scriptsize[{c['wilson'][0]:.1f}, {c['wilson'][1]:.1f}]}}"
    ci = lambda x: f"{x['mean']:.1f} {{\\scriptsize[{x['ci'][0]:.1f}, {x['ci'][1]:.1f}]}}"
    cd = lambda x: f"${x['mean']:+.1f}$ {{\\scriptsize$[{x['ci'][0]:+.1f}, {x['ci'][1]:+.1f}]$}}"
    c2 = lambda x: f"${x['d']:+.2f}$ {{\\scriptsize$[{x['ci'][0]:+.2f}, {x['ci'][1]:+.2f}]$}}"
    L = [r"\begin{table*}[t]", r"\centering", r"\scriptsize", r"\setlength{\tabcolsep}{3pt}",
         r"\caption{\textbf{Full-sample audits and mechanism tests on the cue-factorial set} (occlusion-competent models). "
         r"(a) Every \texttt{D+C$-$} commit sentence $j^\ast$ and, as comparator, one relational sentence from every \texttt{D+C$+$} rationale "
         r"(nearest a relative position drawn from the same model's commit positions), each judged by two image-conditioned judges with the original prompt "
         r"and with the depth/vertical prompt (contradiction under the reading the sentence uses); consensus contradicting / judgeable with Wilson CIs, "
         r"worst-case bounds count every non-consensus sentence as not contradicting or as contradicting. "
         r"(b) Non-circular consistency over all items: written relation (last relational claim of the CoT rationale, rule parser; front shape named by the "
         r"question-blind description, two-judge consensus) true or false, and whether the final answer agrees with it; scene-cluster bootstrap CIs. "
         r"(c) True-sentence edits classified by how the question relates to the written proposition (identical: same subject and predicate; negation: same subject, "
         r"opposite predicate; converse: other subject, converse predicate; swap: other subject, same predicate), pooled over three surface forms per asserted proposition "
         r"(FS ``The F is in front of the K.'', FO ``The K has the F in front of it.'', BN ``The F is not behind the K.'' assert F in front; "
         r"BS ``The K is behind the F.'', BO ``The F has the K behind it.'', FN ``The K is not in front of the F.'' assert K behind); $j^\ast$ rows edit every committed "
         r"\texttt{D+C$-$} rationale ($n = 270$), \texttt{D+C$+$} rows the first claim of $200$ correct rationales; last column: coefficient of ``grammatical subject names the queried shape'' "
         r"in a logistic fit of correctness with question type (reference: identical), asserted predicate, and negation, image-cluster bootstrap CI; \texttt{Qwen3-VL-8B} row: accuracy per form ($n = 247$). "
         r"(d) Occlusion-neutral description (shape, color, 2D position) and a turn-closed filler matched per image to the length of the occlusion-prompted description, "
         r"changes vs.\ \texttt{Direct} (occlusion: probit weight in $d'$ units). (e) Prefix readout of $120$ \texttt{D+C$-$} rationales per model recomputed with a uniform gray image; "
         r"gray $j^\ast$ is undefined where the gray readout never reaches $0.5$ before the collapse; ``largest drop same sentence'': largest per-sentence log-odds drop at the same sentence under both images (\%).}",
         r"\label{tab:supp_e30}", r"\resizebox{\textwidth}{!}{%", r"\begin{tabular}{lcccccc}", r"\toprule",
         r"(a) Commit audit & \texttt{D+C$-$} $j^\ast$, orig.\ prompt & worst case & depth/vertical prompt & Rel.\ \texttt{D+C$+$}, orig. & Rel.\ \texttt{D+C$+$}, depth/vert. & Commit$-$comparator \\", r"\midrule"]
    A = J["audit"]["rows"]
    if (B / "e30_reparse.json").exists():  # Exp-32: 43 UNPARSED Opus e20 replies re-judged; panel (a) uses the post-rerun consensus
        R = json.load(open(B / "e30_reparse.json"))["after_rerun"]
        A = {k: dict(v) for k, v in A.items()}
        for k, v in R.items():
            if k.endswith("|diff"):
                A[k + "|e20"] = v
            elif k in A:
                A[k]["e20"] = {kk: vv for kk, vv in v.items() if kk != "opus_e20"}
    for m in COMP:
        a, b = A[f"{m}|D+C-|all"], A[f"{m}|D+C+rel|all"]
        L.append(f"{NAMES[m]} & {pc(a['e20'])} & {a['e20']['worst_case'][0]:.1f}--{a['e20']['worst_case'][1]:.1f} & {pc(a['own'])} & {pc(b['e20'])} & {pc(b['own'])} & {cd(A[f'{m}|diff|e20'])} \\\\")
    L += [r"\midrule", r"(b) Answer agrees with relation & CoT: P(agree $\mid$ true) & CoT: P(agree $\mid$ false) & Desc.: P(agree $\mid$ true) & Desc.: P(agree $\mid$ false) & CoT $n$ true/false & Desc.\ $n$ true/false \\", r"\midrule"]
    F = J["follow"]
    for m in COMP:
        c, d = F[m]["cot|last|all"], F[m].get("desc|judge|all", {"n": 0})
        dd = (ci(d["p_cons_true"]), ci(d["p_cons_false"]), f"{d['p_cons_true']['n']}/{d['p_cons_false']['n']}") if d.get("n") else ("--", "--", "--")
        L.append(f"{NAMES[m]} & {ci(c['p_cons_true'])} & {ci(c['p_cons_false'])} & {dd[0]} & {dd[1]} & {c['p_cons_true']['n']}/{c['p_cons_false']['n']} & {dd[2]} \\\\")
    L += [r"\midrule", r"(c) Accuracy after a true sentence & identical & negation & converse & swap & \multicolumn{2}{c}{surface subject = queried shape: logit coef.} \\", r"\midrule"]
    G = J["argedit"]
    for m, part, nm in (("molmo7b", "jstar", r"Molmo-7B-D, $j^\ast$"), ("molmo7b", "dcp", r"Molmo-7B-D, \texttt{D+C$+$} first claim")):
        S = G[f"{m}|cues|{part}"]["semantic"]
        lg = S["logit_qtype"]["coef"]["sm"]
        for ap, an in (("front", "asserts F in front"), ("behind", "asserts K behind")):
            L.append(f"{nm}, {an} & " + " & ".join(ci(S["cells"][f"{ap}|{qt}"]) for qt in QTYPES)
                     + (f" & \\multicolumn{{2}}{{c}}{{${lg['b']:+.2f}$ [{lg['ci'][0]:+.2f}, {lg['ci'][1]:+.2f}]}}" if ap == "front" else " & \\multicolumn{2}{c}{}") + r" \\")
    q = G["qwen3vl_8b|cues|jstar"]["acc"]
    L.append(r"Qwen3-VL-8B, $j^\ast$, accuracy per form & \multicolumn{6}{l}{" + ", ".join(f"{a} {q[a]['mean']:.1f}" for a in ["neutral"] + AE)
             + r" (FS/BS canonical; FO/BO/FN/BN other subject or negated)} \\")
    L += [r"\midrule", r"(d) Description arms vs \texttt{Direct} & $\Delta$acc.\ occl.-prompted desc. & $\Delta$acc.\ neutral desc. & $\Delta$occl.\ occl.-prompted & $\Delta$occl.\ neutral & $\Delta$occl.\ filler & $\Delta d'_\mathrm{pair}$ neutral \\", r"\midrule"]
    N = J["ndesc"]
    for m in COMP:
        r = N[m]
        L.append(f"{NAMES[m]} & {cd(r['acc_desc-direct'])} & {cd(r['acc_ndesc-direct'])} & {c2(r['glm']['desc-direct']['occlusion'])} & {c2(r['glm']['ndesc-direct']['occlusion'])} & "
                 f"{c2(r['glm']['filler_desc-direct']['occlusion'])} & {c2(r['dpair']['ndesc-direct_pooled'])} \\\\")
    L += [r"\midrule", r"(e) Gray-image readout & gray $j^\ast$ defined & gray $j^\ast$ = real $j^\ast$ & within one sentence & largest drop same sentence & $\Delta$logit at $j^\ast$ real / gray & P(gold) before $j^\ast$ real / gray \\", r"\midrule"]
    for m in COMP:
        g = J["gray"][m]
        L.append(f"{NAMES[m]} & {g['gray_jstar_defined']['k']}/{g['gray_jstar_defined']['n']} & {pc(g['gray_jstar_equals_real'])} & {g['gray_jstar_within1']['pct']:.1f} & "
                 f"{g['maxdrop_gray_equals_real']['pct']:.1f} & ${g['dlogit_at_jstar_real']:.1f}$ / ${g['dlogit_at_jstar_gray']:.1f}$ & {g['p_real_before_jstar']['mean']:.1f} / {g['p_gray_before_jstar']['mean']:.1f} \\\\")
    L += [r"\bottomrule", r"\end{tabular}}", r"\end{table*}"]
    return "\n".join(L).replace("Qwen3-VL-8B", r"\texttt{Qwen3-VL-8B}").replace("Molmo-7B-D", r"\texttt{Molmo-7B-D}") + "\n"


# ============================================================================= main

SECTIONS = {"follow": follow, "logit": logit, "audit": audit, "argedit": argedit, "ndesc": ndesc, "gray": gray}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="audit,follow,logit,argedit,ndesc,gray")
    ap.add_argument("--table", action="store_true", help="rebuild supp_e30.tex from bundle/e30_*.json only")
    a = ap.parse_args()
    if a.table:
        tex = table()
        (B / "supp_e30.tex").write_text(tex)
        shutil.copy(B / "supp_e30.tex", PAPER_TAB / "supp_e30.tex")
        print(tex)
        return
    lines: list[str] = []
    for k in a.only.split(","):
        sec: list[str] = []
        out = SECTIONS[k](sec)
        json.dump(out, open(B / f"e30_{k}.json", "w"), indent=1)
        (E30 / "rigor_sections").mkdir(parents=True, exist_ok=True)
        (E30 / "rigor_sections" / f"{k}.txt").write_text("\n".join(sec) + "\n")
        lines += sec
        print("\n".join(sec), flush=True)
    # rigor_e30.txt = concatenation of all section files present
    allsec = [p.read_text() for p in sorted((E30 / "rigor_sections").glob("*.txt"))]
    (B / "rigor_e30.txt").write_text("".join(allsec))


if __name__ == "__main__":
    main()
