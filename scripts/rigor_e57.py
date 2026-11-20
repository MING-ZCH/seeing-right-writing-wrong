"""Exp-57 analyses: the sentence-level pipeline on Qwen3-VL-32B (R22 Opus Q3), with Qwen3-VL-8B on the same sets.

traj     A1 commit trajectories (outputs/e57/<m>_<set>_traj.jsonl; 8B: Exp-28 outputs/e28): D+C- commit share (Wilson),
         D+C+ commit share (all, and a seed-57 sample of 150), j* sentence type (rigor_e17.stype), j* = first claim,
         uniform null (rigor_e17.commit_baseline algorithm, 10,000 draws) and position-matched null (rigor_e17f
         commit_nulls algorithm, 4,000 draws). Sanity: the same null code on the Qwen3-VL-8B main pool (e2b) vs
         commit_nulls.json / commit_baseline.json.
audit    A2 two-judge commit audit (outputs/e57/judge_<judge>_commit_<set>_<prompt>.jsonl): consensus contradiction rate
         (Wilson, worst-case bounds), comparator rate, commit - comparator (scene bootstrap, seed 302), agreement, kappa.
uncond   A3a unconditional first-claim edit (outputs/e57/<m>_<set>_uncond.jsonl) over all four Direct x CoT cells:
         accuracy of Direct / CoT / orig / correct / neutral / wrong and differences (build_bundle.ci, twin-pair
         cluster bootstrap, 2,000), by phrasing and by cell; continuation restates a false relation (claim_parse v3).
argedit  A3b six-form edit at j* / D+C+ control (rigor_e30 argedit statistics on outputs/e57/<m>_cues_argedit_*.jsonl).
imgswap  A4 own-R image swap (outputs/e57/<m>_<set>_imgswap.jsonl): image effect P(gold_v | v) - P(gold_v | v_cf) for R
         full / R_cut, gray-image effect for R_cut, Direct 100 (2 acc - 1) on the same queries, twin-pair bootstrap.
decomp   A5 D+C- decomposition (rigor_e54 Part A code with the model list = Qwen3-VL-32B, labels in outputs/e57/e54).
sanity   Direct re-read, R(full) re-read, closed P_J vs same-turn CoT, edit 'orig' regeneration.
-> outputs/bundle/e57_sentence32b.json, outputs/bundle/rigor_e57.txt, paper/tables/supp_e57.tex
Usage: python rigor_e57.py [--only traj,audit,uncond,argedit,imgswap,decomp,sanity] [--table-only]
"""
from __future__ import annotations
import vtax_paths as _vtp  # noqa: E402

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

import build_bundle as BB
from build_bundle import ci, pe, rj
from claim_parse import first_claim_v3
from e20_commit_judge import first_claim_idx
from rigor_e17 import stype
from rigor_e28 import J2, cboot, pct, q95, wilson

O = Path(f"{_vtp.OUTPUTS}")
E57, E28 = O / "e57", O / "e28"
B = Path("outputs/bundle")
L = Path(f"{_vtp.LEGACY_RESULTS}")
PAPER_TAB = Path(f"{_vtp.TABLES}")
M32, M8 = "qwen3vl_32b", "qwen3vl_8b"
NAMES = {M32: "Qwen3-VL-32B", M8: "Qwen3-VL-8B"}
NB = 2000
fa = lambda x: f"{x['mean']:+.1f} [{x['lo']:+.1f}, {x['hi']:+.1f}]" if x and "mean" in x else "--"
fm = lambda x: f"{x['mean']:.1f} [{x['lo']:.1f}, {x['hi']:.1f}]" if x and "mean" in x else "--"


def pair_of(base_id: str) -> str:
    return base_id.replace("_cf", "")


# ============================================================================= A1 commit trajectories

def commit_items(T: list[dict]) -> list[dict]:
    """Committed D+C- trajectories: sentence types, j* (0-based), image cluster."""
    out = []
    for r in T:
        if r["group"] != "D+C-" or r["commit_idx"] is None or not r["sentences"]:
            continue
        out.append({"types": [stype(s) for s in r["sentences"]], "j": r["commit_idx"], "img": pair_of(r.get("image_id") or r["base_id"])})
    return out


def nulls(items: list[dict], rng_u, rng_p, n_u: int = 10000, n_p: int = 4000) -> dict:
    """Uniform (rigor_e17.commit_baseline) and position-matched (rigor_e17f.commit_nulls) relational-commit nulls."""
    if not items:
        return {}
    isrel = [np.array([t == "relation" for t in it["types"]]) for it in items]
    obs = np.array([it["types"][it["j"]] == "relation" for it in items], float)
    exp = np.array([x.mean() for x in isrel])
    nu = np.zeros(n_u)
    for x in isrel:
        nu += x[rng_u.integers(0, len(x), n_u)]
    nu /= len(items)
    nc = [(it, np.array([t == "relation" for t in it["types"] if t != "conclusion"])) for it in items]
    nc = [(it, x) for it, x in nc if len(x) and it["types"][it["j"]] != "conclusion"]
    obs_nc = float(np.mean([it["types"][it["j"]] == "relation" for it, _ in nc])) if nc else float("nan")
    nn = np.zeros(n_u)
    for _, x in nc:
        nn += x[rng_u.integers(0, len(x), n_u)]
    nn /= max(1, len(nc))
    relpos = np.array([it["j"] / max(1, len(it["types"]) - 1) for it in items])
    lens = np.array([len(x) for x in isrel])
    npm = np.zeros(n_p)
    for k in range(len(items)):
        others = rng_p.integers(0, len(items), n_p)
        pos = np.rint(relpos[k] * (lens[others] - 1)).astype(int)
        npm += np.array([isrel[o][p] for o, p in zip(others, pos)])
    npm /= len(items)
    om = obs.mean()
    return {"n_committed": len(items), "obs": round(100 * om, 1),
            "uniform": {"null": round(100 * nu.mean(), 1), "null_95": [round(100 * v, 1) for v in q95(nu, 4)],
                        "excess": round(100 * (om - nu.mean()), 1), "perm_p": float((np.sum(nu >= om) + 1) / (n_u + 1)),
                        "obs_minus_item_expected": ci(obs - exp, [it["img"] for it in items])},
            "nonconclusion": {"obs": round(100 * obs_nc, 1), "null": round(100 * nn.mean(), 1),
                              "excess": round(100 * (obs_nc - nn.mean()), 1), "perm_p": float((np.sum(nn >= obs_nc) + 1) / (n_u + 1))},
            "position_matched": {"null": round(100 * npm.mean(), 1), "null_95": [round(100 * v, 1) for v in q95(npm, 4)],
                                 "excess": round(100 * (om - npm.mean()), 1), "perm_p": float((np.sum(npm >= om) + 1) / (n_p + 1))}}


def traj_stats(T: list[dict]) -> dict:
    cnt = Counter(r["group"] for r in T)
    out = {"n": len(T), "cells": dict(cnt)}
    for g in ("D+C-", "D+C+", "D-C+", "D-C-"):
        rs = [x for x in T if x["group"] == g]
        if not rs:
            continue
        cm = [x for x in rs if x["commit_idx"] is not None]
        row = {"n": len(rs), "commit": pct(len(cm), len(rs)),
               "closed_final_correct": round(100 * float(np.mean([x["p_gold"][-1] >= .5 for x in rs])), 1),
               "mean_sentences": round(float(np.mean([len(x["sentences"]) for x in rs])), 2)}
        if g == "D+C+":
            s = random.Random(57).sample(rs, min(150, len(rs)))
            row["commit_sample150"] = pct(sum(x["commit_idx"] is not None for x in s), len(s))
        if g == "D+C-" and cm:
            fc = [first_claim_idx(x["sentences"]) for x in cm]
            row["jstar_type"] = {k: round(100 * v / len(cm), 1) for k, v in Counter(stype(x["sentences"][x["commit_idx"]]) for x in cm).items()}
            row["jstar_relational"] = pct(sum(stype(x["sentences"][x["commit_idx"]]) == "relation" for x in cm), len(cm))
            row["jstar_relpos_mean"] = round(float(np.mean([x["commit_idx"] / max(1, len(x["sentences"]) - 1) for x in cm])), 3)
            row["jstar_is_first_claim"] = round(100 * float(np.mean([f == x["commit_idx"] for f, x in zip(fc, cm)])), 1)
            row["first_claim_before_jstar"] = round(100 * float(np.mean([f is not None and f < x["commit_idx"] for f, x in zip(fc, cm)])), 1)
            row["p0_ge_half"] = round(100 * float(np.mean([x["p_gold"][0] >= .5 for x in rs])), 1)
            row["p_gold_before_jstar"] = round(float(np.mean([x["p_gold"][x["commit_idx"]] for x in cm])), 3)
            last = [x for x in cm if all(v >= .5 for v in x["p_gold"][:x["commit_idx"] + 1])]
            row["jstar_first_crossing"] = round(100 * len(last) / len(cm), 1)
        out[g] = row
    agree = [(("A" if x["p_gold"][-1] >= .5 else "B") if x["gold"] == "A" else ("B" if x["p_gold"][-1] >= .5 else "A")) == x["src_cot"] for x in T]
    out["closed_PJ_agrees_sameturn_cot"] = round(100 * float(np.mean(agree)), 2)
    out["nulls"] = nulls(commit_items(T), np.random.default_rng(0), np.random.default_rng(1))
    return out


def null_sanity() -> dict:
    """Exp-57 null code on the Qwen3-VL-8B main pool (Synthetic Depth D+C-, e2b) vs the stored Tab. S42 numbers."""
    G = BB.groups_from({}, L, O, M8)
    T = []
    for r in rj(O / "e2b" / f"{M8}.jsonl"):
        if G.get(r["item_id"], {}).get("group") != "D+C-":
            continue
        pg = [BB.probs(t.get("lp_sum") or t["lp_bare"])[r["gold"]] for t in sorted(r["traj"], key=lambda x: x["j"])]
        js = next((j for j in range(1, len(pg)) if pg[j - 1] >= 0.5 and all(v < 0.5 for v in pg[j:])), None)
        T.append({"group": "D+C-", "commit_idx": None if js is None else js - 1, "sentences": r["sentences"], "base_id": r["image_id"]})
    mine = nulls(commit_items(T), np.random.default_rng(0), np.random.default_rng(1))
    st = json.load(open(B / "commit_nulls.json"))[M8]
    sb = json.load(open(B / "commit_baseline.json"))[M8]
    return {"exp57_code": {"obs": mine["obs"], "uniform_null": mine["uniform"]["null"], "position_null": mine["position_matched"]["null"],
                           "position_excess": mine["position_matched"]["excess"], "n_committed": mine["n_committed"]},
            "stored": {"obs": st["obs_committed"], "uniform_null": sb["null_uniform_mean"], "position_null": st["position_matched"]["null"],
                       "position_excess": st["position_matched"]["excess"], "n_committed": sb["n_committed"]}}


def traj(lines: list[str]) -> dict:
    res = {"null_sanity_8b_mainpool": null_sanity()}
    s = res["null_sanity_8b_mainpool"]
    lines.append(f"NULL-SANITY 8B main pool: Exp-57 code {s['exp57_code']} | stored (Tab. S42) {s['stored']}")
    for m, d in ((M32, E57), (M8, E28)):
        for st in ("cues", "twins"):
            T = rj(d / f"{m}_{st}_traj.jsonl")
            if not T:
                continue
            r = traj_stats(T)
            res[f"{m}|{st}"] = r
            dm, nl = r.get("D+C-", {}), r["nulls"]
            lines.append(f"TRAJ {NAMES[m]:12s} {st:5s} cells {dict(sorted(r['cells'].items()))} | D+C- commit {dm['commit']['k']}/{dm['commit']['n']} = "
                         f"{dm['commit']['pct']}% {dm['commit']['wilson']} | j* relational {dm['jstar_relational']['pct']}% {dm['jstar_relational']['wilson']} "
                         f"types {dm['jstar_type']} | null uniform {nl['uniform']['null']} (+{nl['uniform']['excess']}, p {nl['uniform']['perm_p']:.4f}) "
                         f"position-matched {nl['position_matched']['null']} {nl['position_matched']['null_95']} (excess {nl['position_matched']['excess']:+.1f}, "
                         f"p {nl['position_matched']['perm_p']:.4f}) | j*=first claim {dm['jstar_is_first_claim']}% first claim before j* {dm['first_claim_before_jstar']}% "
                         f"first crossing {dm['jstar_first_crossing']}% P(gold) at j*-1 {dm['p_gold_before_jstar']} | D+C+ commit {r['D+C+']['commit']['k']}/{r['D+C+']['commit']['n']} "
                         f"(sample150 {r['D+C+']['commit_sample150']['k']}/150) | closed P_J = same-turn CoT {r['closed_PJ_agrees_sameturn_cot']}%")
    return res


# ============================================================================= A2 commit audit

LABELS = ("CORRECT", "INCORRECT", "NOCLAIM", "UNCLEAR")
REPARSE = {"on": True}


def reparse(r: dict) -> dict:
    """Exp-32 fix: the e20 parser reads only the first six words; replies that put the label last are re-read from the
    stored reply tail (last label word)."""
    if r.get("label") != "UNPARSED" or not REPARSE["on"]:
        return r
    import re
    w = [t for t in re.sub(r"[^A-Za-z ]", " ", r.get("raw") or "").upper().split() if t in LABELS]
    return r | {"label": w[-1], "reparsed": True} if w else r


def latest(path: Path) -> dict:
    out = {}
    for r in rj(path):
        k = (r["model"], r["item_id"], r["idx"], r["group"])
        if r.get("api_error") and k in out:
            continue
        out[k] = reparse(r) if "label" in r else r
    return out


def rate_e20(labs: list[str]) -> dict:
    k, c, n = labs.count("INCORRECT"), labs.count("CORRECT"), len(labs)
    return pct(k, k + c) | {"n_items": n, "worst_case": [round(100 * k / n, 1) if n else None, round(100 * (n - c) / n, 1) if n else None]}


def audit_one(st: str, prompt: str, lines: list[str]) -> dict:
    from rigor_e29 import cons2, depth_contra, own_contra
    J = {j: latest(E57 / f"judge_{j}_commit_{st}_{prompt}.jsonl") for j in J2}
    keys = sorted(set(J[J2[0]]) & set(J[J2[1]]))
    if not keys:
        return {}
    meta = {k: J[J2[1]][k] for k in keys}
    if prompt == "e20":
        lab = lambda j, k: J[j][k]["label"] if not J[j][k].get("api_error") else "APIERR"
        con = {k: (lab(J2[0], k) if lab(J2[0], k) == lab(J2[1], k) and lab(J2[0], k) in ("CORRECT", "INCORRECT") else "NONCONS") for k in keys}
        both = [k for k in keys if {lab(J2[0], k), lab(J2[1], k)} <= {"CORRECT", "INCORRECT"}]
        a = np.array([lab(J2[0], k) == "INCORRECT" for k in both])
        b = np.array([lab(J2[1], k) == "INCORRECT" for k in both])
        agree_all = float(np.mean([lab(J2[0], k) == lab(J2[1], k) for k in keys]))
        unparsed = {j: sum(lab(j, k) in ("UNPARSED", "APIERR") for k in keys) for j in J2}
    else:
        con = {k: {"YES": "INCORRECT", "NO": "CORRECT"}.get(cons2(own_contra(J[J2[0]][k]), own_contra(J[J2[1]][k])), "NONCONS") for k in keys}
        dep = {k: {"YES": "INCORRECT", "NO": "CORRECT"}.get(cons2(depth_contra(J[J2[0]][k]), depth_contra(J[J2[1]][k])), "NONCONS") for k in keys}
        oa = [own_contra(J[J2[0]][k]) for k in keys]
        ob = [own_contra(J[J2[1]][k]) for k in keys]
        both = [i for i, k in enumerate(keys) if {oa[i], ob[i]} <= {"YES", "NO"}]
        a = np.array([oa[i] == "YES" for i in both])
        b = np.array([ob[i] == "YES" for i in both])
        agree_all = float(np.mean([J[J2[0]][k]["reading"] == J[J2[1]][k]["reading"] for k in keys]))
        unparsed = {j: sum("UNPARSED" in (J[j][k]["reading"], J[j][k]["depth_true"]) for k in keys) for j in J2}
    po = float(np.mean(a == b))
    pe_ = a.mean() * b.mean() + (1 - a.mean()) * (1 - b.mean())
    res = {"n_sentences": len(keys), "agree_all": round(100 * agree_all, 1), "kappa_clear": round((po - pe_) / (1 - pe_), 3) if pe_ < 1 else None,
           "n_clear_both": len(both), "unparsed_or_error": unparsed, "rows": {}}
    subsets = {"all": lambda k: True, "front": lambda k: meta[k]["phrasing"] == "front", "behind": lambda k: meta[k]["phrasing"] == "behind",
               "jstar_relation": lambda k: meta[k]["stype"] == "relation", "jstar_conclusion": lambda k: meta[k]["stype"] == "conclusion",
               "jstar_nonrel": lambda k: meta[k]["stype"] == "non_relational",
               "jstar_first_claim": lambda k: meta[k]["first_claim_idx"] == meta[k]["commit_idx"],
               "jstar_later_claim": lambda k: None not in (meta[k]["first_claim_idx"], meta[k]["commit_idx"]) and meta[k]["first_claim_idx"] < meta[k]["commit_idx"]}
    for g in ("D+C-", "D+C+rel"):
        for sn, sf in subsets.items():
            ks = [k for k in keys if k[3] == g and sf(k)]
            if not ks or (g == "D+C+rel" and sn not in ("all", "front", "behind")):
                continue
            row = {"n": len(ks), "cons": rate_e20([con[k] for k in ks])}
            if prompt == "e20":
                row["opus"] = pct(sum(lab(J2[0], k) == "INCORRECT" for k in ks), sum(lab(J2[0], k) in ("CORRECT", "INCORRECT") for k in ks))
                row["gpt"] = pct(sum(lab(J2[1], k) == "INCORRECT" for k in ks), sum(lab(J2[1], k) in ("CORRECT", "INCORRECT") for k in ks))
                row["noclaim_either"] = sum("NOCLAIM" in (lab(J2[0], k), lab(J2[1], k)) for k in ks)
            else:
                row["depth"] = rate_e20([dep[k] for k in ks])
                row["reading"] = dict(Counter(J[J2[0]][k]["reading"] if J[J2[0]][k]["reading"] == J[J2[1]][k]["reading"] else "NC" for k in ks))
            res["rows"][f"{g}|{sn}"] = row
            c = row["cons"]
            lines.append(f"AUDIT {st:5s} {prompt:5s} {g:8s} {sn:18s} n={len(ks):4d} consensus {c['k']}/{c['n']} = {c['pct']}% {c['wilson']} "
                         f"wc [{c['worst_case'][0]}, {c['worst_case'][1]}]" + (f" | Opus {row['opus']['pct']} GPT {row['gpt']['pct']} NOCLAIM(either) {row['noclaim_either']}" if prompt == "e20" else
                                                                               f" | depth-only {row['depth']['pct']}% reading {row['reading']}"))
    rows = [(pair_of(meta[k]["base_id"]), k[3] == "D+C-", con[k] == "INCORRECT") for k in keys if con[k] != "NONCONS"]
    rng = np.random.default_rng(302)
    cl = np.array([r[0] for r in rows])
    gp = np.array([r[1] for r in rows])
    y = np.array([r[2] for r in rows], float)
    uc, inv = np.unique(cl, return_inverse=True)
    agg = np.zeros((len(uc), 4))
    np.add.at(agg, inv, np.stack([y * gp, gp, y * ~gp, ~gp], 1))
    stat = lambda A: A[:, 0].sum() / A[:, 1].sum() - A[:, 2].sum() / A[:, 3].sum()
    bs = [stat(agg[rng.integers(0, len(uc), len(uc))]) for _ in range(NB)]
    res["diff"] = {"mean": round(100 * stat(agg), 1), "lo": round(100 * q95(bs, 4)[0], 1), "hi": round(100 * q95(bs, 4)[1], 1)}
    lines.append(f"AUDIT {st:5s} {prompt:5s} commit - comparator {fa(res['diff'])} | agreement {res['agree_all']}% kappa(clear) {res['kappa_clear']} "
                 f"(n {len(keys)}, clear both {len(both)}) unparsed/error {unparsed}")
    return res


def audit(lines: list[str]) -> dict:
    res = {}
    for st, prompt in (("cues", "e20"), ("cues", "ontop"), ("twins", "e20")):
        r = audit_one(st, prompt, lines)
        if r:
            res[f"{st}|{prompt}"] = r
    REPARSE["on"] = False
    sub: list[str] = []
    r = audit_one("cues", "e20", sub)
    REPARSE["on"] = True
    if r:
        res["cues|e20|no_reparse"] = r
        lines += [f"(no re-parse) {x}" for x in sub if "D+C-     all" in x or "D+C+rel  all" in x or "commit - comparator" in x]
    J = rj(E57 / "judge_claude-opus-5-5_commit_cues_e20.jsonl") + rj(E57 / "judge_claude-opus-5-5_commit_twins_e20.jsonl")
    res["reparsed_counts"] = dict(Counter(reparse(x)["label"] for x in J if x.get("label") == "UNPARSED"))
    lines.append(f"AUDIT Opus replies with the label last (Exp-32 re-parse from the stored tail): {res['reparsed_counts']}")
    rr = json.load(open(B / "e30_reparse.json"))["after_rerun"]  # Exp-30 audit after the Exp-32 re-run of 43 replies
    res["ref_8b_cues_e20"] = {"commit": rr[f"{M8}|D+C-|all"], "comparator": rr[f"{M8}|D+C+rel|all"], "diff": rr[f"{M8}|diff"]}
    return res


# ============================================================================= A3a unconditional edit

UNC = ("orig", "correct", "wrong", "neutral")


def uncond_one(R: list[dict]) -> dict:
    items = defaultdict(dict)
    for r in R:
        items[r["item_id"]][r["cond"]] = r
    keys = sorted(k for k, v in items.items() if all(c in v for c in UNC))
    out = {}
    for sub, f in (("all", lambda o: True), ("front", lambda o: o["phrasing"] == "front"), ("behind", lambda o: o["phrasing"] == "behind"),
                   ("D+C-", lambda o: o["group"] == "D+C-"), ("D+C+", lambda o: o["group"] == "D+C+"), ("D-C+", lambda o: o["group"] == "D-C+"),
                   ("D+C-_claimwrong", lambda o: o["group"] == "D+C-" and o["orig_claim_wrong"] == 1)):
        ks = [k for k in keys if f(items[k]["orig"])]
        if len(ks) < 5:
            continue
        o0 = [items[k]["orig"] for k in ks]
        gold = [o["gold"] for o in o0]
        g = [pair_of(o["base_id"]) for o in o0]
        pred = {"direct": [o["src_direct"] for o in o0], "cot": [o["src_cot"] for o in o0]}
        for c in UNC:
            pred[c] = [pe(items[k][c]) for k in ks]
        acc = {c: [float(a == y) for a, y in zip(p, gold)] for c, p in pred.items()}
        row = {"n": len(ks), "cells": dict(Counter(o["group"] for o in o0)), "orig_claim_wrong": round(100 * float(np.mean([o["orig_claim_wrong"] for o in o0])), 1)}
        for c in acc:
            row[c] = ci(acc[c], g)
        for a, b in (("correct", "direct"), ("neutral", "direct"), ("orig", "direct"), ("wrong", "direct"), ("correct", "cot"),
                     ("neutral", "cot"), ("orig", "cot"), ("wrong", "neutral"), ("correct", "neutral")):
            row[f"{a}-{b}"] = ci(np.array(acc[a]) - np.array(acc[b]), g)
        # continuation restates a false relation (first v3 claim of the continuation names the back shape as front);
        # diagnostic only: v3 misreads 32B's bare-noun sentences ("the pentagon is obscured by the circle")
        for c in UNC:
            fl = []
            for k in ks:
                x = items[k][c]
                fc = first_claim_v3([x["continuation"]])
                if fc and fc[1] in (x["meta"]["front_name"], x["meta"]["back_name"]):
                    fl.append(float(fc[1] == x["meta"]["back_name"]))
            row[f"cont_false_claim:{c}"] = {"pct": round(100 * float(np.mean(fl)), 1) if fl else None, "n": len(fl)}
        out[sub] = row
    return out


def uncond(lines: list[str]) -> dict:
    res = {}
    for m in (M32, M8):
        for st in ("cues", "twins"):
            R = rj(E57 / f"{m}_{st}_uncond.jsonl")
            if not R:
                continue
            r = uncond_one(R)
            res[f"{m}|{st}"] = r
            for sub in ("all", "front", "behind", "D+C-", "D+C+"):
                if sub not in r:
                    continue
                x = r[sub]
                lines.append(f"UNCOND {NAMES[m]:12s} {st:5s} {sub:6s} n={x['n']} | acc Direct {x['direct']['mean']:.1f} CoT {x['cot']['mean']:.1f} orig {x['orig']['mean']:.1f} "
                             f"correct {x['correct']['mean']:.1f} neutral {x['neutral']['mean']:.1f} wrong {x['wrong']['mean']:.1f} | correct-D {fa(x['correct-direct'])} "
                             f"neutral-D {fa(x['neutral-direct'])} neutral-CoT {fa(x['neutral-cot'])} wrong-neutral {fa(x['wrong-neutral'])}")
    return res


# ============================================================================= A3b six-form edit

def argedit(lines: list[str]) -> dict:
    import rigor_e30 as R30
    res = {}
    for part in ("jstar", "dcp"):
        R = rj(E57 / f"{M32}_cues_argedit_{part}.jsonl")
        if not R:
            continue
        rng = np.random.default_rng(303)
        idx = {(r["item_id"], r["arm"]): r for r in R}
        keys = sorted({r["item_id"] for r in R})
        g = [idx[(k, "orig")]["image_id"] for k in keys]
        out = {"n": len(keys), "acc": {}, "restates_false": {}}
        for a in ["orig", "neutral"] + R30.AE:
            out["acc"][a] = R30.cboot([float(pe(idx[(k, a)]) == idx[(k, a)]["gold"]) for k in keys], g, rng, 1000)
            fl = []
            for k in keys:
                x = idx[(k, a)]
                fc = first_claim_v3([x["continuation"]])
                if fc and fc[1] in (x["meta"]["front_name"], x["meta"]["back_name"]):
                    fl.append(float(fc[1] == x["meta"]["back_name"]))
            out["restates_false"][a] = {"pct": round(100 * float(np.mean(fl)), 1) if fl else None, "n": len(fl), "n_items": len(keys)}
        for a in R30.AE:
            out["acc"][f"{a}-neutral"] = R30.cboot([float(pe(idx[(k, a)]) == idx[(k, a)]["gold"]) - float(pe(idx[(k, 'neutral')]) == idx[(k, 'neutral')]["gold"]) for k in keys], g, rng, 1000)
        out["semantic"] = R30.semantic(R30.rows_sem(R), rng)
        res[part] = out
        A = out["acc"]
        lines.append(f"ARGEDIT {NAMES[M32]} cues {part:5s} n={len(keys)} | " + " | ".join(f"{a} {A[a]['mean']} [{A[a]['ci'][0]}, {A[a]['ci'][1]}]" for a in ["orig", "neutral"] + R30.AE))
        lines += R30.semantic_lines(f"ARGEDIT-SEM {NAMES[M32]} cues {part}", out["semantic"])
    ref = json.load(open(B / "e30_argedit.json")).get(f"{M8}|cues|jstar", {})
    if ref:
        res["ref_8b_jstar"] = {a: ref["acc"][a] for a in ["orig", "neutral"] + R30.AE}
    return res


# ============================================================================= A4 image swap

def imgswap_one(R: list[dict]) -> dict:
    by = {(r["item_id"], r["ctx"], r["img"]): r for r in R}
    keys = sorted({r["item_id"] for r in R})
    rec = {r["item_id"]: r for r in R if r["ctx"] == "full" and r["img"] == "own"}
    out = {}
    # first-claim truth of the rationale (e17e.load_cot_ok rule, on the R the readout sees)
    for sub in ("all", "cot_wrong", "cot_right", "D+C-", "front", "behind"):
        rows = {}
        for ctx in ("full", "cut"):
            dv, dg, own, tw, dd, g = [], [], [], [], [], []
            for k in keys:
                r0 = rec[k]
                if sub == "cot_wrong" and r0["group"][2:] != "C-":
                    continue
                if sub == "cot_right" and r0["group"][2:] != "C+":
                    continue
                if sub == "D+C-" and r0["group"] != "D+C-":
                    continue
                if sub in ("front", "behind") and r0["phrasing"] != sub:
                    continue
                a, b = by.get((k, ctx, "own")), by.get((k, ctx, "twin"))
                if a is None or b is None:
                    continue
                own.append(float(pe(a) == a["gold"]))
                tw.append(float(pe(b) == a["gold"]))
                dv.append(own[-1] - tw[-1])
                gr = by.get((k, ctx, "gray"))
                if gr is not None:
                    dg.append(own[-1] - float(pe(gr) == a["gold"]))
                g.append(pair_of(r0["base_id"]))
            if len(dv) < 10:
                continue
            rows[ctx] = {"acc_own": ci(own, g), "acc_goldv_under_twin": ci(tw, g), "image_effect": ci(dv, g)}
            if dg:
                rows[ctx]["gray_effect"] = ci(dg, g)
        # Direct image effect on the same queries: c(v) + c(v_cf) - 1 per query (paired yes/no twins)
        dd, g = [], []
        for k in keys:
            r0 = rec[k]
            t = rec.get(r0["twin_id"])
            if t is None:
                continue
            if sub == "cot_wrong" and r0["group"][2:] != "C-" or sub == "cot_right" and r0["group"][2:] != "C+" or sub == "D+C-" and r0["group"] != "D+C-":
                continue
            if sub in ("front", "behind") and r0["phrasing"] != sub:
                continue
            dd.append(float(r0["src_direct"] == r0["gold"]) + float(t["src_direct"] == t["gold"]) - 1)
            g.append(pair_of(r0["base_id"]))
        if len(dd) >= 10:
            rows["direct_effect"] = ci(dd, g)
        out[sub] = rows
    return out


def imgswap(lines: list[str]) -> dict:
    res = {}
    for st in ("twins", "cues"):
        R = rj(E57 / f"{M32}_{st}_imgswap.jsonl")
        if not R:
            continue
        r = imgswap_one(R)
        res[st] = r
        for sub in ("all", "cot_wrong", "D+C-", "front", "behind"):
            x = r.get(sub, {})
            if "cut" not in x:
                continue
            lines.append(f"IMGSWAP {NAMES[M32]} {st:5s} {sub:6s} | Direct effect {fm(x.get('direct_effect'))} | R full {fm(x['full']['image_effect'])} | "
                         f"R_cut {fm(x['cut']['image_effect'])} (own acc {x['cut']['acc_own']['mean']:.1f}, gold_v under twin {x['cut']['acc_goldv_under_twin']['mean']:.1f}) "
                         f"gray effect {fm(x['cut'].get('gray_effect'))} | n {x['cut']['image_effect']['n']}")
    ref = json.load(open(B / "imgswap.json")).get(M8, {})
    res["ref_8b_twins"] = {"cut_all": ref.get("cut:all", {}).get("dvis"), "full_all": ref.get("full:all", {}).get("dvis")}
    return res


# ============================================================================= exploratory (prereg addendum 2)

def expl(lines: list[str]) -> dict:
    res = {}
    for m in (M32, M8):
        R = rj(E57 / f"{m}_cues_imgswap_trunc.jsonl")
        if R:
            by = {(r["item_id"], r["img"]): r for r in R}
            ks = sorted({r["item_id"] for r in R})
            row = {}
            for sub, f in (("all", lambda r: True), ("D+C-", lambda r: r["group"] == "D+C-")):
                kk = [k for k in ks if (k, "twin") in by and f(by[(k, "own")])]
                own = [float(pe(by[(k, "own")]) == by[(k, "own")]["gold"]) for k in kk]
                tw = [float(pe(by[(k, "twin")]) == by[(k, "own")]["gold"]) for k in kk]
                g = [pair_of(by[(k, "own")]["base_id"]) for k in kk]
                row[sub] = {"acc_own": ci(own, g), "image_effect": ci(np.array(own) - np.array(tw), g),
                            "own_equals_cot": round(100 * float(np.mean([pe(by[(k, 'own')]) == by[(k, 'own')]["src_cot"] for k in kk])), 1)}
            res[f"{m}|trunc"] = row
            lines.append(f"EXPL R_trunc {NAMES[m]:12s} cues | image effect all {fm(row['all']['image_effect'])} (own acc {row['all']['acc_own']['mean']:.1f}, "
                         f"= CoT answer {row['all']['own_equals_cot']}%) | D+C- {fm(row['D+C-']['image_effect'])}")
        R = rj(E57 / f"{m}_cues_trajtwin.jsonl")
        if R:
            out = {}
            for sub in ("all", "D+C-", "D+C+"):
                rs = [r for r in R if sub == "all" or r["group"] == sub]
                g = [pair_of(r["base_id"]) for r in rs]
                eff = lambda j_of: ci([r["p_gold_own"][j_of(r)] - r["p_gold_twin"][j_of(r)] for r in rs], g)
                row = {"n": len(rs), "j0": eff(lambda r: 0), "J": eff(lambda r: -1)}
                cm = [r for r in rs if r["commit_idx"] is not None]
                if cm:
                    gc = [pair_of(r["base_id"]) for r in cm]
                    # p_gold index j = prefix with j sentences; j* (0-based sentence) -> prefix index j*+1, j*-1 prefix = commit_idx
                    row["before_jstar"] = ci([r["p_gold_own"][r["commit_idx"]] - r["p_gold_twin"][r["commit_idx"]] for r in cm], gc)
                    row["after_jstar"] = ci([r["p_gold_own"][r["commit_idx"] + 1] - r["p_gold_twin"][r["commit_idx"] + 1] for r in cm], gc)
                    row["twin_pgold_before_jstar"] = round(float(np.mean([r["p_gold_twin"][r["commit_idx"]] for r in cm])), 3)
                    row["own_pgold_before_jstar"] = round(float(np.mean([r["p_gold_own"][r["commit_idx"]] for r in cm])), 3)
                # by relative position (deciles of j / J)
                prof = defaultdict(list)
                for r in rs:
                    J = len(r["p_gold_own"]) - 1
                    for j in range(J + 1):
                        prof[min(9, int(10 * j / max(1, J)))].append(r["p_gold_own"][j] - r["p_gold_twin"][j])
                row["profile_decile"] = {k: round(100 * float(np.mean(v)), 1) for k, v in sorted(prof.items())}
                out[sub] = row
            res[f"{m}|trajtwin"] = out
            a, d = out["all"], out["D+C-"]
            lines.append(f"EXPL prefix image effect {NAMES[m]:12s} cues | all: j=0 {fm(a['j0'])} J {fm(a['J'])} | D+C- (n {d['n']}): j=0 {fm(d['j0'])} "
                         f"before j* {fm(d.get('before_jstar'))} after j* {fm(d.get('after_jstar'))} J {fm(d['J'])} | P(gold) before j* own {d.get('own_pgold_before_jstar')} "
                         f"twin {d.get('twin_pgold_before_jstar')} | all-item profile by decile {a['profile_decile']}")
    return res


# ============================================================================= A6 probe (prereg addendum 1)

def probe(lines: list[str]) -> dict:
    res = {}
    for m, p, layers in ((M32, E57 / f"{M32}_ctrl.json", (36, 20)), (M8, O / "e17" / f"{M8}_ctrl.json", (20,))):
        if not p.exists():
            continue
        d = json.load(open(p))
        byl = {x["layer"]: x for x in d["layers"]}
        res[m] = {"n": d["n"], "n_claims": d["n_claims"], "layers": {}}
        for L_ in layers:
            x = byl[L_]
            row = {"r": x["orig"]["r"], "rprev": x["orig"]["rprev"], "a": x["orig"]["a"], "gray_refit_r": x["gray_refit"]["r"],
                   "twin_follows_shown_r": x["twin_xfer"]["r"]["follows_shown_image"], "direct_probe_at_answer": x["direct_probe_at_answer"]}
            res[m]["layers"][L_] = row
            lines.append(f"PROBE {NAMES[m]:12s} L{L_} | r bal {row['r']['bal']} {row['r']['bal_ci']} D+C- wrong-claim recall {row['r']['recall_DCm_wrong']} "
                         f"{row['r']['recall_DCm_wrong_ci']} (n {row['r']['n_DCm_wrong']}) | rprev {row['rprev']['bal']} | a {row['a']['bal']} | gray refit r "
                         f"{row['gray_refit_r']['bal']} {row['gray_refit_r']['bal_ci']} | twin xfer follows shown {row['twin_follows_shown_r']} | "
                         f"Direct probe at a (D+C- wrong): acc {row['direct_probe_at_answer']['acc_DCm_wrong']} follows CoT {row['direct_probe_at_answer']['follows_cot_answer_DCm_wrong']}")
        res[m]["profile_r"] = {x["layer"]: x["orig"]["r"]["bal"] for x in d["layers"]}
        res[m]["profile_gray_r"] = {x["layer"]: x["gray_refit"]["r"]["bal"] for x in d["layers"]}
        lines.append(f"PROBE {NAMES[m]:12s} profile r bal by layer {res[m]['profile_r']} | gray refit {res[m]['profile_gray_r']}")
    return res


# ============================================================================= A5 decomposition

def decomp(lines: list[str]) -> dict:
    import rigor_e54 as R54
    R54.M7 = [M32]
    R54.E54 = E57 / "e54"
    R54.DS3 = ("cues",)
    rng = np.random.default_rng(54)
    R = R54.records()
    for o in R:
        o["cat"] = R54.classify(o)
        o["cat_af"] = R54.classify(o, "affirm_first") if o["cat"] else None
        o["cat_v3t"] = R54.classify_geo(o, "v3_trunc")
        o["cat_v3f"] = R54.classify_geo(o, "v3_full")
    DM = [o for o in R if o["dm"]]
    DP = [o for o in R if o["dp"]]
    need = [o for o in DM + DP if o["has"]]
    miss = [o for o in need if len(o.get("jl", {})) < 2]
    res = {"label_coverage": {"need": len(need), "both_judges": len(need) - len(miss)},
           "DpCm": R54.share_boot(DM, "cat", R54.CAT_DM, NB, rng), "DmCp": R54.share_boot(DP, "cat", R54.CAT_DP, NB, rng),
           "DpCm_affirm_first": R54.share_boot(DM, "cat_af", R54.CAT_DM, NB, rng),
           "DpCm_v3_trunc": R54.share_boot(DM, "cat_v3t", R54.CAT_DM, NB, rng), "DpCm_v3_full": R54.share_boot(DM, "cat_v3f", R54.CAT_DM, NB, rng)}
    d0 = res["DpCm"]
    res["a_of_ab_wilson"] = pct(d0["a"]["k"], d0["a"]["k"] + d0["b"]["k"])
    pairs = [o["jl"] for o in R if o.get("kind") == "basis" and len(o.get("jl", {})) == 2]
    a = [p[R54.JUDGES[0]]["basis"] for p in pairs]
    b = [p[R54.JUDGES[1]]["basis"] for p in pairs]
    res["basis_agreement"] = {"n": len(a), "agree": round(100 * float(np.mean([x == y for x, y in zip(a, b)])), 1) if a else None,
                              "kappa": round(R54.kappa(a, b), 3) if a else None}
    fl = [o for o in DM if o["has"] and not o["true"]]
    res["false_relation_profile"] = {"n": len(fl), "affirms_queried": round(100 * float(np.mean([o["affirms"] for o in fl])), 1) if fl else None,
                                     "names_lower": round(100 * float(np.mean([o["claim"] == o["lower"] for o in fl if o["lower"]])), 1) if fl else None,
                                     "names_larger": round(100 * float(np.mean([o["claim"] == o["larger"] for o in fl if o["larger"]])), 1) if fl else None,
                                     "basis": dict(Counter(R54.basis_cons(o.get("jl", {})) for o in fl))}
    tr = [o for o in DM if o["has"] and o["true"]]
    res["true_relation_items"] = [{"item": o["item"], "cat": o["cat"], "qform": o.get("qform"), "conc_rev": o.get("conc_rev")} for o in tr]
    ref = json.load(open(B / "e54_decomp.json"))["cells"].get(f"{M8}|cues", {}).get("DpCm")
    res["ref_8b_cues"] = {k: ref[k] for k in ("n", "a", "b", "c", "a_of_ab")} if ref else None
    d = res["DpCm"]
    f3 = lambda x: f"{x['pct']:.1f} [{x['ci'][0]:.1f}, {x['ci'][1]:.1f}]"
    lines.append(f"DECOMP {NAMES[M32]} cues D+C- n={d['n']} | (a) false relation {f3(d['a'])} (b) true relation, wrong answer {f3(d['b'])} (c) no relation {f3(d['c'])} "
                 f"| a/(a+b) {f3(d['a_of_ab'])} (Wilson {res['a_of_ab_wilson']['k']}/{res['a_of_ab_wilson']['n']} {res['a_of_ab_wilson']['wilson']}) | subtypes {d['counts']} | labels {res['label_coverage']} basis agree {res['basis_agreement']}")
    lines.append(f"DECOMP robustness: v3 truncated a/(a+b) {f3(res['DpCm_v3_trunc']['a_of_ab'])}, v3 conclusion kept {f3(res['DpCm_v3_full']['a_of_ab'])} | "
                 f"false-relation profile {res['false_relation_profile']} | D-C+ (a') {f3(res['DmCp']['a'])} (b') {f3(res['DmCp']['b'])} (c') {f3(res['DmCp']['c'])}")
    if ref:
        lines.append(f"DECOMP ref Qwen3-VL-8B cues (Exp-54): n {ref['n']} a {ref['a']['pct']:.1f} b {ref['b']['pct']:.1f} c {ref['c']['pct']:.1f} a/(a+b) {ref['a_of_ab']['pct']:.1f}")
    return res


# ============================================================================= sanity

def sanity(lines: list[str]) -> dict:
    res = {}
    for m in (M32, M8):
        for st in ("cues", "twins"):
            row = {}
            R = rj(E57 / f"{m}_{st}_reread.jsonl")
            if R:
                row["direct_reread_agree"] = pct(sum(pe(r) == r["src_direct"] for r in R), len(R))
                row["direct_acc_stored"] = round(100 * float(np.mean([r["src_direct"] == r["gold"] for r in R])), 2)
                row["cot_acc_stored"] = round(100 * float(np.mean([r["src_cot"] == r["gold"] for r in R])), 2)
            S = [r for r in rj(E57 / f"{m}_{st}_imgswap.jsonl") if r["ctx"] == "full" and r["img"] == "own"]
            if S:
                row["cot_full_reread_agree"] = pct(sum(pe(r) == r["src_cot"] for r in S), len(S))
            U = [r for r in rj(E57 / f"{m}_{st}_uncond.jsonl") if r["cond"] == "orig"]
            if U:
                row["uncond_orig_equals_stored_cot"] = pct(sum(pe(r) == r["src_cot"] for r in U), len(U))
                row["uncond_trunc"] = {c: round(100 * float(np.mean([r["trunc"] for r in rj(E57 / f"{m}_{st}_uncond.jsonl") if r["cond"] == c])), 1) for c in UNC}
            if row:
                res[f"{m}|{st}"] = row
                lines.append(f"SANITY {NAMES[m]:12s} {st:5s} {json.dumps(row)}")
    return res


# ============================================================================= table

def table(R: dict) -> str:
    """Bare tabular (Qwen3-VL-32B vs Qwen3-VL-8B, cue set unless stated)."""
    def c(x, nd=1):
        return "--" if not x or "mean" not in x else f"{x['mean']:.{nd}f} [{x['lo']:.{nd}f}, {x['hi']:.{nd}f}]"

    def d(x, nd=1):
        return "--" if not x or "mean" not in x else f"${x['mean']:+.{nd}f}$ [{x['lo']:+.{nd}f}, {x['hi']:+.{nd}f}]"

    def w(p):
        return "--" if not p or p.get("n") in (0, None) else f"{p['k']}/{p['n']} = {p['pct']:.1f} [{max(0.0, p['wilson'][0]):.1f}, {p['wilson'][1]:.1f}]"

    T = R.get("traj", {})
    rows = []
    for st in ("cues", "twins"):
        t32, t8 = T.get(f"{M32}|{st}", {}), T.get(f"{M8}|{st}", {})
        lab = "cue set" if st == "cues" else "twins"
        rows.append((f"A1 D+C$-$ commit share ({lab})", w(t32.get("D+C-", {}).get("commit")), w(t8.get("D+C-", {}).get("commit"))))
        rows.append((f"A1 $j^\\ast$ relational / position-matched null ({lab})",
                     *(f"{x['D+C-']['jstar_relational']['pct']:.1f} / {x['nulls']['position_matched']['null']:.1f} (${x['nulls']['position_matched']['excess']:+.1f}$, $p={x['nulls']['position_matched']['perm_p']:.4f}$)"
                       if x.get("D+C-") else "--" for x in (t32, t8))))
        rows.append((f"A1 D+C$+$ commit share ({lab})", w(t32.get("D+C+", {}).get("commit")), w(t8.get("D+C+", {}).get("commit"))))
    A = R.get("audit", {})
    a32 = A.get("cues|e20", {}).get("rows", {})
    ref = A.get("ref_8b_cues_e20", {})
    rows.append(("A2 commits judged false (cue set, consensus)", w(a32.get("D+C-|all", {}).get("cons")), w(ref.get("commit"))))
    rows.append(("A2 relational D+C$+$ comparators judged false", w(a32.get("D+C+rel|all", {}).get("cons")), w(ref.get("comparator"))))
    dd = A.get("cues|e20", {}).get("diff")
    rd = ref.get("diff")
    rows.append(("A2 commit $-$ comparator", d(dd), "--" if not rd else f"${rd['mean']:+.1f}$ [{rd['ci'][0]:+.1f}, {rd['ci'][1]:+.1f}]"))
    U = R.get("uncond", {})
    for st in ("cues", "twins"):
        u32, u8 = U.get(f"{M32}|{st}", {}).get("all", {}), U.get(f"{M8}|{st}", {}).get("all", {})
        lab = "cue set" if st == "cues" else "twins"
        rows.append((f"A3 true first claim $-$ \\texttt{{Direct}} ({lab})", d(u32.get("correct-direct")), d(u8.get("correct-direct"))))
        rows.append((f"A3 neutral sentence $-$ \\texttt{{Direct}} ({lab})", d(u32.get("neutral-direct")), d(u8.get("neutral-direct"))))
        rows.append((f"A3 accuracy after a wrong first claim ({lab})", c(u32.get("wrong")), c(u8.get("wrong"))))
    G = R.get("argedit", {})
    j = G.get("jstar", {}).get("acc", {})
    r8 = G.get("ref_8b_jstar", {})
    fmt_arm = lambda x: "--" if not x else f"{x['mean']:.1f}"
    rows.append(("A3 six forms at $j^\\ast$: FS / BS / FO / BO / FN / BN (neutral)",
                 " / ".join(fmt_arm(j.get(a)) for a in ("FS", "BS", "FO", "BO", "FN", "BN")) + f" ({fmt_arm(j.get('neutral'))})",
                 " / ".join(fmt_arm(r8.get(a)) for a in ("FS", "BS", "FO", "BO", "FN", "BN")) + f" ({fmt_arm(r8.get('neutral'))})"))
    I = R.get("imgswap", {})
    tw = I.get("twins", {}).get("all", {})
    r8i = I.get("ref_8b_twins", {})
    rows.append(("A4 twin image effect, \\texttt{Direct} (twins)", c(tw.get("direct_effect")), "62.3"))
    rows.append(("A4 twin image effect, own $R_\\mathrm{cut}$ (twins)", c(tw.get("cut", {}).get("image_effect")),
                 "--" if not r8i.get("cut_all") else f"{r8i['cut_all']['mean']:.1f} [{r8i['cut_all']['lo']:.1f}, {r8i['cut_all']['hi']:.1f}]"))
    cu = I.get("cues", {}).get("all", {})
    rows.append(("A4 twin image effect, \\texttt{Direct} / own $R_\\mathrm{cut}$ (cue set)",
                 f"{c(cu.get('direct_effect'))} / {c(cu.get('cut', {}).get('image_effect'))}", "--"))
    X = R.get("expl", {})
    rows.append(("A4$^\\dagger$ own conclusion-removed body (Exp-45 rule): image effect, all / D+C$-$",
                 *(f"{c(X.get(f'{m}|trunc', {}).get('all', {}).get('image_effect'))} / {c(X.get(f'{m}|trunc', {}).get('D+C-', {}).get('image_effect'))}" for m in (M32, M8))))
    rows.append(("A4$^\\dagger$ prefix image effect on D+C$-$: before / after $j^\\ast$",
                 *(f"{c(X.get(f'{m}|trajtwin', {}).get('D+C-', {}).get('before_jstar'))} / {c(X.get(f'{m}|trajtwin', {}).get('D+C-', {}).get('after_jstar'))}" for m in (M32, M8))))
    D = R.get("decomp", {})
    p3 = lambda x: "--" if not x else f"{x['pct']:.1f} [{x['ci'][0]:.1f}, {x['ci'][1]:.1f}]"
    dm, r8d = D.get("DpCm", {}), D.get("ref_8b_cues") or {}
    rows.append(("A5 D+C$-$: false written relation (a)", p3(dm.get("a")), p3(r8d.get("a"))))
    rows.append(("A5 D+C$-$: true relation, wrong answer (b)", p3(dm.get("b")), p3(r8d.get("b"))))
    aw = D.get("a_of_ab_wilson")
    a32 = p3(dm.get("a_of_ab")) if not aw else f"{aw['k']}/{aw['n']} [{aw['wilson'][0]:.1f}, {aw['wilson'][1]:.1f}]"
    rows.append(("A5 D+C$-$: no relation (c) / $a/(a+b)$", f"{p3(dm.get('c'))} / {a32}", f"{p3(r8d.get('c'))} / {p3(r8d.get('a_of_ab'))}"))
    P = R.get("probe", {})
    pr = lambda m, L_: P.get(m, {}).get("layers", {}).get(L_) or P.get(m, {}).get("layers", {}).get(str(L_))
    f_p = lambda x: "--" if not x else f"{x['r']['bal']:.2f} [{x['r']['bal_ci'][0]:.2f}, {x['r']['bal_ci'][1]:.2f}] / {x['gray_refit_r']['bal']:.2f}"
    rows.append(("A6 within-claim truth probe at the relation token: original / gray refit (32B L36, 8B L20)", f_p(pr(M32, 36)), f_p(pr(M8, 20))))
    head = ["% Suggested caption: Sentence-level pipeline on Qwen3-VL-32B (Exp-57) next to Qwen3-VL-8B on the same sets and protocols.",
            "% A1 commit trajectories (forced readout after each sentence; j* = commit sentence), position-matched null as Tab. S42;",
            "% A2 image-conditioned two-judge audit (Claude Opus 5.5 + GPT-6-Astra, Exp-30 prompt, consensus, Wilson CIs);",
            "% A3 first claim replaced (true / neutral / wrong) and the rest regenerated, all four Direct x CoT cells; six true forms at j*;",
            "% A4 image effect P(gold_v|v) - P(gold_v|v_cf) with the own rationale fixed and its conclusion removed (twin-pair bootstrap);",
            "% A5 D+C- decomposition with the conclusion removed (Exp-45 / Exp-54 judges); A6 within-claim probe (Exp-17j protocol, layer at the same relative depth).",
            "% dagger: exploratory (prereg addendum 2). 8B: Exp-17/28/30/45/54 numbers except A3 and the dagger rows (rerun here with the same code).",
            "% Bare tabular; compiles standalone under \\resizebox{\\linewidth}{!}{...}. Generated by vtax_exps/rigor_e57.py."]
    body = ["\\begin{tabular}{lll}", "\\toprule", "Quantity & Qwen3-VL-32B & Qwen3-VL-8B \\\\", "\\midrule"]
    body += [f"{a} & {b} & {c_} \\\\" for a, b, c_ in rows]
    body += ["\\bottomrule", "\\end{tabular}"]
    return "\n".join(head + body) + "\n"


SECTIONS = {"traj": traj, "audit": audit, "uncond": uncond, "argedit": argedit, "imgswap": imgswap, "expl": expl, "probe": probe, "decomp": decomp, "sanity": sanity}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default=",".join(SECTIONS))
    ap.add_argument("--table-only", action="store_true")
    a = ap.parse_args()
    jp = B / "e57_sentence32b.json"
    R = json.load(open(jp)) if jp.exists() else {}
    lines: list[str] = []
    if not a.table_only:
        for s in a.only.split(","):
            sec: list[str] = []
            R[s] = SECTIONS[s](sec)
            R[f"_lines_{s}"] = sec
            print("\n".join(sec), flush=True)
        json.dump(R, open(jp, "w"), indent=1, default=float)
    for s in SECTIONS:
        lines += R.get(f"_lines_{s}", [])
    (B / "rigor_e57.txt").write_text("\n".join(lines) + "\n")
    (PAPER_TAB / "supp_e57.tex").write_text(table(R))


if __name__ == "__main__":
    main()
