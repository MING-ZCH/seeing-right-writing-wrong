"""Exp-28 analyses (round-16 external reviews): cue-factorial evidence for the occlusion-competent models.

desc     Question-blind description arm on the cue-factorial set (outputs/e28/<m>_cues_desc.jsonl) vs Direct / CoT of
         e10 (the runs behind Tab. S18/S19): accuracy (twin-pair cluster bootstrap), probit-GLM cue weights
         (rigor_e17f.glm_rows / probit_fit, says-subject-front ~ phrasing + occlusion + vertical + size, d' = 2 beta),
         Desc - Direct / Desc - CoT / CoT - Direct with a joint twin-pair bootstrap (NB_GLM), common-scale factor
         s = b_arm . b_dir / |b_dir|^2 and the occlusion deviation (rigor_e17i.common_scale), d'_pair over
         (scene, phrasing) units (rigor_e17q.twin_stats: pooled = log((BC+.5)/(BW+.5))/1.702, vertical-balanced, vertical
         component; cells = front lower/higher x front drawn larger/smaller). Twins: the same d'_pair statistics for the
         existing e2d description arm (all seven models).
traj     Commit trajectories (outputs/e28/<m>_<set>_traj.jsonl): D+/-C+/- cells from the stored readouts, commit share
         (e20 rule), j* position / sentence type / first claim, agreement of the closed-turn P_J with the same-turn CoT.
audit    Two-judge commit audit (outputs/e28/judge_*_commit_<set>.jsonl): consensus contradiction rate with Wilson CI and
         worst-case bounds (non-consensus / NOCLAIM / UNCLEAR counted as not contradicting or as contradicting).
prej     Pre-j* relational sentences (judge_*_prej_<set>.jsonl; j* labels from outputs/e20 for Synthetic Depth and from
         the commit audit for cues / twins): share of D+C- rationales with a consensus image-contradicting relational
         sentence before j*, and (index of the first contradicting sentence - j*).
edit     Claim edit at j* (outputs/e28/<m>_<set>_edit.jsonl): accuracy orig / correct / neutral, paired twin-pair bootstrap.
-> bundle/e28_cues.json, bundle/rigor_e28.txt, bundle/supp_e28.tex (copied to paper tables/, not \\input)
Usage: python rigor_e28.py [--nb-glm 1000] | python rigor_e28.py --from-json  (table only)
"""
from __future__ import annotations
import vtax_paths as _vtp  # noqa: E402

import argparse
import json
import shutil
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

import build_bundle as BB
from build_bundle import pe, rj
from rigor_e17 import stype
from rigor_e17f import glm_rows, probit_fit
from rigor_e17q import LOGIT_PROBIT, twin_stats, twin_units

O = Path(f"{_vtp.OUTPUTS}")
E = O / "e28"
B = Path("outputs/bundle")
PAPER_TAB = Path(f"{_vtp.TABLES}")
COMP = ["qwen3vl_8b", "molmo7b"]
CUE_MODELS = ["qwen3vl_8b", "molmo7b", "internvl3_8b"]
M7 = ["qwen3b", "qwen7b", "internvl3_8b", "phi35v", "molmo7b", "qwen3vl_8b", "internvl35_8b"]
NAMES = {"qwen3b": "Qwen2.5-VL-3B", "qwen7b": "Qwen2.5-VL-7B", "internvl3_8b": "InternVL3-8B", "phi35v": "Phi-3.5-V",
         "molmo7b": "Molmo-7B-D", "qwen3vl_8b": "Qwen3-VL-8B", "internvl35_8b": "InternVL3.5-8B"}
J2 = ("claude-opus-5-5", "gpt-6-astra")
NB = 2000
CUE_NAMES = ["occlusion", "vertical", "size"]


def q95(v, p: int = 2) -> list[float]:
    return [round(float(np.nanquantile(v, .025)), p), round(float(np.nanquantile(v, .975)), p)]


def wilson(k: int, n: int) -> list[float]:
    if n == 0:
        return [float("nan")] * 2
    z, p = 1.96, k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [round(float(100 * (c - h)), 1), round(float(100 * (c + h)), 1)]


def pct(k: int, n: int) -> dict:
    return {"k": int(k), "n": int(n), "pct": round(100 * k / n, 1) if n else None, "wilson": wilson(k, n)}


def cboot(vals, groups, rng, nb: int = NB) -> dict:
    """Mean (x100) with a cluster bootstrap percentile CI."""
    v = np.asarray(vals, float)
    g = np.asarray(groups)
    ug = np.unique(g)
    idx = {u: np.where(g == u)[0] for u in ug}
    bs = []
    for _ in range(nb):
        s = np.concatenate([idx[u] for u in rng.choice(ug, len(ug))])
        bs.append(v[s].mean())
    return {"mean": round(100 * float(v.mean()), 1), "ci": [round(100 * x, 1) for x in q95(bs, 4)], "n": int(len(v)),
            "n_clusters": int(len(ug))}


# ============================================================================= desc arm

def cue_units(recs: list[dict]) -> dict:
    """{(scene, phrasing): {'orig'|'cf': (says_front, x_is_front, says_yes), 'above', 'larger'}} for cue-set records."""
    U = defaultdict(dict)
    for r in recs:
        p = pe(r)
        mt = r["meta"]
        ph = r["phrasing"]
        u = U[(r["image_id"], ph)]
        says = (p == "A") if ph == "front" else (p == "B")
        u["cf" if r["base_id"].endswith("_cf") else "orig"] = (bool(says), mt["front_name"] == mt["x_name"], p == "A")
        if not r["base_id"].endswith("_cf"):
            u["above"], u["larger"] = bool(mt["front_above"]), bool(mt["front_larger"])
    return U


def outcomes(U: dict, keys: list) -> np.ndarray:
    o = []
    for k in keys:
        a, b = U[k]["orig"], U[k]["cf"]
        ca, cb = a[0] == a[1], b[0] == b[1]
        o.append(0 if ca and cb else 3 if not (ca or cb) else 1 if a[0] and b[0] else 2)
    return np.array(o)


_TW = ["pooled", "mh_raw", "std_cells", "vert_balanced", "vert_component", "lower", "higher"]


def pair_sdt(Us: dict, cell_of, vert_of, rng, contrasts) -> dict:
    """d'_pair statistics per arm and contrasts (twin_stats layout), scene bootstrap."""
    arms = list(Us)
    keys = sorted(k for k in Us[arms[0]] if all({"orig", "cf"} <= set(Us[a].get(k, {})) for a in arms))
    for a in arms:
        assert all(Us[a][k]["orig"][1] != Us[a][k]["cf"][1] for k in keys)
    oc = {a: outcomes(Us[a], keys) for a in arms}
    cell = np.array([cell_of(Us[arms[0]][k]) for k in keys])
    vert = np.array([vert_of(Us[arms[0]][k]) for k in keys])
    wc = np.bincount(cell, minlength=4) / len(cell)
    scene = np.array([k[0] for k in keys])
    us = np.unique(scene)
    sidx = {s: np.where(scene == s)[0] for s in us}
    out = {"n_units": len(keys), "n_scenes": len(us)}
    for a in arms:
        out[f"split_{a}"] = [round(100 * float(np.mean(oc[a] == j)), 1) for j in range(4)]
    for a1, a0 in contrasts:
        est = twin_stats(oc[a0], oc[a1], cell, wc, vert)
        bs = []
        for _ in range(NB):
            i = np.concatenate([sidx[s] for s in rng.choice(us, len(us))])
            bs.append(twin_stats(oc[a0][i], oc[a1][i], cell[i], wc, vert[i]))
        bs = np.array(bs)
        n_ = len(est) // 3
        for j, nm in enumerate(_TW):
            out.setdefault(f"{a0}_{nm}", round(float(est[j]), 3))
            out.setdefault(f"{a1}_{nm}", round(float(est[n_ + j]), 3))
            out[f"{a1}-{a0}_{nm}"] = {"d": round(float(est[2 * n_ + j]), 3), "ci": q95(bs[:, 2 * n_ + j], 3)}
            out.setdefault(f"{a0}_{nm}_ci", q95(bs[:, j], 3))
            out.setdefault(f"{a1}_{nm}_ci", q95(bs[:, n_ + j], 3))
    return out


def glm_arms(recs_by: dict, rng, nb: int) -> dict:
    data = {c: glm_rows(rs, use_size=True) for c, rs in recs_by.items()}
    pairs = sorted(set(data["direct"]["pair"]))
    pidx = {c: {p: np.where(np.array(d["pair"]) == p)[0] for p in pairs} for c, d in data.items()}
    est = {c: probit_fit(d["X"], d["y"]) for c, d in data.items()}

    def scale(b, b0):
        b, b0 = b[2:5], b0[2:5]
        s = float(b @ b0 / (b0 @ b0))
        return [s, *(b - s * b0)]

    bs = {c: [] for c in data}
    for _ in range(nb):
        js = rng.integers(0, len(pairs), len(pairs))
        for c, d in data.items():
            s = np.concatenate([pidx[c][pairs[j]] for j in js])
            bs[c].append(probit_fit(d["X"][s], d["y"][s]))
    bs = {c: np.array(v) for c, v in bs.items()}
    out = {}
    for c in data:
        out[c] = {k: {"d": round(2 * float(est[c][2 + i]), 2), "ci": q95(2 * bs[c][:, 2 + i])} for i, k in enumerate(CUE_NAMES)}
        out[c]["acc"] = round(100 * float(np.mean(data[c]["y"] == data[c]["truth"])), 1)
    for a1, a0 in (("cot", "direct"), ("desc", "direct"), ("desc", "cot")):
        out[f"{a1}-{a0}"] = {k: {"d": round(2 * float(est[a1][2 + i] - est[a0][2 + i]), 2),
                                 "ci": q95(2 * (bs[a1][:, 2 + i] - bs[a0][:, 2 + i]))} for i, k in enumerate(CUE_NAMES)}
        pt = scale(est[a1], est[a0])
        bb = np.array([scale(bs[a1][i], bs[a0][i]) for i in range(nb)])
        out[f"{a1}-{a0}"]["common_scale"] = {nm: {"est": round(pt[i], 3), "ci": q95(bb[:, i], 3)}
                                            for i, nm in enumerate(["s", "delta_occlusion", "delta_vertical", "delta_size"])}
    return out


def desc(lines: list[str], nb_glm: int) -> dict:
    res = {"cues": {}, "twins": {}}
    for m in CUE_MODELS:
        p = E / f"{m}_cues_desc.jsonl"
        if not p.exists():
            lines.append(f"DESC-CUES {m}: missing {p.name}")
            continue
        rng = np.random.default_rng(28)
        D = rj(p)
        R = rj(O / "e10" / f"{m}_cues.jsonl")
        by = {c: {r["item_id"]: r for r in R if r["condition"] == c} for c in ("direct", "cot")}
        by["desc"] = {r["item_id"]: r for r in D if r["condition"] == "desc"}
        rr = {r["item_id"]: r for r in D if r["condition"] == "direct_reread"}
        keys = sorted(by["desc"])
        assert set(keys) == set(by["direct"]), m
        same = np.mean([pe(rr[k]) == pe(by["direct"][k]) for k in keys])
        g = [by["direct"][k]["image_id"] for k in keys]
        acc = {c: np.array([float(pe(by[c][k]) == by[c][k]["gold"]) for k in keys]) for c in ("direct", "cot", "desc")}
        r = {"n": len(keys), "direct_reread_same_pred": round(100 * float(same), 2),
             "desc_tokens_mean": round(float(np.mean([by["desc"][k]["gen_len"] for k in keys])), 1),
             "desc_trunc": round(100 * float(np.mean([by["desc"][k]["trunc"] for k in keys])), 1),
             "cot_tokens_mean": round(float(np.mean([by["cot"][k].get("gen_len", 0) for k in keys])), 1)}
        for c in acc:
            r[f"acc_{c}"] = cboot(acc[c], g, rng)
        for a1, a0 in (("cot", "direct"), ("desc", "direct"), ("desc", "cot")):
            r[f"acc_{a1}-{a0}"] = cboot(acc[a1] - acc[a0], g, rng)
        r["glm"] = glm_arms({c: [by[c][k] for k in keys] for c in ("direct", "cot", "desc")}, rng, nb_glm)
        Us = {c: cue_units([by[c][k] for k in keys]) for c in ("direct", "cot", "desc")}
        r["dpair"] = pair_sdt(Us, lambda u: 2 * int(u["above"]) + int(u["larger"]), lambda u: int(u["above"]), rng,
                              [("cot", "direct"), ("desc", "direct"), ("desc", "cot")])
        res["cues"][m] = r
        G, P = r["glm"], r["dpair"]
        f = lambda x: f"{x['d']:+.2f} [{x['ci'][0]:+.2f}, {x['ci'][1]:+.2f}]"
        fa = lambda x: f"{x['mean']:+.1f} [{x['ci'][0]:+.1f}, {x['ci'][1]:+.1f}]"
        lines.append(f"DESC-CUES {m:13s} acc Dir {r['acc_direct']['mean']} CoT {r['acc_cot']['mean']} Desc {r['acc_desc']['mean']} | "
                     f"Desc-Dir {fa(r['acc_desc-direct'])} Desc-CoT {fa(r['acc_desc-cot'])} CoT-Dir {fa(r['acc_cot-direct'])} | "
                     f"GLM occl Dir {G['direct']['occlusion']['d']:.2f} CoT {G['cot']['occlusion']['d']:.2f} Desc {G['desc']['occlusion']['d']:.2f}; "
                     f"dOccl Desc-Dir {f(G['desc-direct']['occlusion'])} Desc-CoT {f(G['desc-cot']['occlusion'])} CoT-Dir {f(G['cot-direct']['occlusion'])}; "
                     f"dVert Desc-Dir {f(G['desc-direct']['vertical'])} CoT-Dir {f(G['cot-direct']['vertical'])}; "
                     f"s Desc {G['desc-direct']['common_scale']['s']['est']} {G['desc-direct']['common_scale']['s']['ci']} "
                     f"delta_occ {G['desc-direct']['common_scale']['delta_occlusion']['est']:+.3f} {G['desc-direct']['common_scale']['delta_occlusion']['ci']} "
                     f"(CoT s {G['cot-direct']['common_scale']['s']['est']}) | d'pair Dir {P['direct_pooled']:.2f} CoT {P['cot_pooled']:.2f} Desc {P['desc_pooled']:.2f}; "
                     f"Desc-Dir {f(P['desc-direct_pooled'])} Desc-CoT {f(P['desc-cot_pooled'])} CoT-Dir {f(P['cot-direct_pooled'])}; "
                     f"vert-bal Desc-Dir {f(P['desc-direct_vert_balanced'])} CoT-Dir {f(P['cot-direct_vert_balanced'])}; "
                     f"vert comp Dir {P['direct_vert_component']:.2f} CoT {P['cot_vert_component']:.2f} Desc {P['desc_vert_component']:.2f} | "
                     f"sanity Direct re-read same pred {r['direct_reread_same_pred']}% | desc tokens {r['desc_tokens_mean']} (trunc {r['desc_trunc']}%) vs CoT {r['cot_tokens_mean']}")
    for m in M7:
        rng = np.random.default_rng(29)
        Us = {c: twin_units(m, c) for c in ("direct", "cot", "desc")}
        if not Us["desc"]:
            continue
        P = pair_sdt(Us, lambda u: 2 * int(u["above"]) + int(u["ratio"] >= 0.77), lambda u: int(u["above"]), rng,
                     [("cot", "direct"), ("desc", "direct"), ("desc", "cot")])
        res["twins"][m] = P
        f = lambda x: f"{x['d']:+.2f} [{x['ci'][0]:+.2f}, {x['ci'][1]:+.2f}]"
        lines.append(f"DESC-TWINS {m:13s} d'pair Dir {P['direct_pooled']:.2f} CoT {P['cot_pooled']:.2f} Desc {P['desc_pooled']:.2f} | "
                     f"Desc-Dir {f(P['desc-direct_pooled'])} Desc-CoT {f(P['desc-cot_pooled'])} CoT-Dir {f(P['cot-direct_pooled'])} | "
                     f"vert-bal Dir {P['direct_vert_balanced']:.2f} CoT {P['cot_vert_balanced']:.2f} Desc {P['desc_vert_balanced']:.2f}; "
                     f"Desc-Dir {f(P['desc-direct_vert_balanced'])} CoT-Dir {f(P['cot-direct_vert_balanced'])} | "
                     f"vert comp Dir {P['direct_vert_component']:.2f} CoT {P['cot_vert_component']:.2f} Desc {P['desc_vert_component']:.2f} "
                     f"(Desc-Dir {f(P['desc-direct_vert_component'])}) | split BC/SF/SB/BW Dir {P['split_direct']} Desc {P['split_desc']}")
    return res


# ============================================================================= trajectories

def traj(lines: list[str]) -> dict:
    res = {}
    for st in ("cues", "twins"):
        for m in CUE_MODELS:
            T = rj(E / f"{m}_{st}_traj.jsonl")
            if not T:
                continue
            cnt = Counter(r["group"] for r in T)
            r = {"n": len(T), "cells": dict(cnt)}
            for g in ("D+C-", "D+C+", "D-C+", "D-C-"):
                rs = [x for x in T if x["group"] == g]
                if not rs:
                    continue
                cm = [x for x in rs if x["commit_idx"] is not None]
                r[g] = {"n": len(rs), "commit": pct(len(cm), len(rs)),
                        "closed_final_correct": round(100 * float(np.mean([x["p_gold"][-1] >= .5 for x in rs])), 1),
                        "mean_sentences": round(float(np.mean([len(x["sentences"]) for x in rs])), 1)}
                if g == "D+C-" and cm:
                    from e20_commit_judge import first_claim_idx
                    ty = Counter(stype(x["sentences"][x["commit_idx"]]) for x in cm)
                    r[g]["jstar_type"] = {k: round(100 * v / len(cm), 1) for k, v in ty.items()}
                    r[g]["jstar_relpos_mean"] = round(float(np.mean([x["commit_idx"] / max(1, len(x["sentences"]) - 1) for x in cm])), 3)
                    r[g]["jstar_is_first_claim"] = round(100 * float(np.mean([first_claim_idx(x["sentences"]) == x["commit_idx"] for x in cm])), 1)
                    r[g]["first_claim_before_jstar"] = round(100 * float(np.mean([(first_claim_idx(x["sentences"]) is not None and first_claim_idx(x["sentences"]) < x["commit_idx"]) for x in cm])), 1)
                    r[g]["p0_ge_half"] = round(100 * float(np.mean([x["p_gold"][0] >= .5 for x in rs])), 1)
            # closed-turn P_J vs stored same-turn CoT answer
            agree = [(("A" if x["p_gold"][-1] >= .5 else "B") if x["gold"] == "A" else ("B" if x["p_gold"][-1] >= .5 else "A")) == x["src_cot"] for x in T]
            r["closed_PJ_agrees_sameturn_cot"] = round(100 * float(np.mean(agree)), 1)
            res[f"{m}|{st}"] = r
            d = r.get("D+C-", {})
            lines.append(f"TRAJ {st:5s} {m:13s} cells {dict(sorted(cnt.items()))} | D+C- commit {d.get('commit')} j* type {d.get('jstar_type')} "
                         f"j*=first claim {d.get('jstar_is_first_claim')}% first claim before j* {d.get('first_claim_before_jstar')}% relpos {d.get('jstar_relpos_mean')} | "
                         f"D+C+ commit {r.get('D+C+', {}).get('commit')} | closed P_J = same-turn CoT {r['closed_PJ_agrees_sameturn_cot']}%")
    return res


# ============================================================================= judges

def labels(task: str, st: str) -> dict:
    J = {}
    for j in J2:
        p = E / f"judge_{j}_{task}_{st}.jsonl"
        J[j] = {(r["model"], r["item_id"], r["group"], r["idx"]): r for r in rj(p)}
    return J


def consensus(J: dict, k) -> str:
    a, b = J[J2[0]].get(k, {}).get("label"), J[J2[1]].get(k, {}).get("label")
    if a == b and a in ("CORRECT", "INCORRECT"):
        return a
    return "NONCONS"


def rate(J: dict, ks: list) -> dict:
    lab = [consensus(J, k) for k in ks]
    kk, nc = lab.count("INCORRECT"), lab.count("CORRECT")
    n = len(ks)
    r = {"n_items": n, "consensus": pct(kk, kk + nc), "worst_case": [round(100 * kk / n, 1) if n else None,
                                                                       round(100 * (n - nc) / n, 1) if n else None]}
    for j, jl in zip(J2, ("opus", "gpt")):
        L = [J[j][k]["label"] for k in ks if k in J[j]]
        cl = [x for x in L if x in ("CORRECT", "INCORRECT")]
        r[jl] = pct(cl.count("INCORRECT"), len(cl)) | {"noclaim": L.count("NOCLAIM"), "unclear": L.count("UNCLEAR"),
                                                        "unparsed": L.count("UNPARSED")}
    return r


def audit(lines: list[str]) -> dict:
    res = {}
    for st in ("cues", "twins"):
        J = labels("commit", st)
        if not J[J2[0]] or not J[J2[1]]:
            lines.append(f"AUDIT {st}: judge files missing")
            continue
        keys = sorted(set(J[J2[0]]) & set(J[J2[1]]))
        meta = {k: J[J2[1]][k] for k in keys}
        both = [k for k in keys if {J[J2[0]][k]["label"], J[J2[1]][k]["label"]} <= {"CORRECT", "INCORRECT"}]
        a = np.array([J[J2[0]][k]["label"] == "INCORRECT" for k in both])
        b = np.array([J[J2[1]][k]["label"] == "INCORRECT" for k in both])
        po = float(np.mean(a == b))
        pe_ = a.mean() * b.mean() + (1 - a.mean()) * (1 - b.mean())
        r = {"n": len(keys), "agree_all": round(100 * float(np.mean([J[J2[0]][k]["label"] == J[J2[1]][k]["label"] for k in keys])), 1),
             "kappa_clear": round((po - pe_) / (1 - pe_), 3) if pe_ < 1 else None}
        for m in COMP + ["pooled"]:
            for g in ("D+C-", "D+C+rel"):
                ks = [k for k in keys if k[2] == g and (m == "pooled" or k[0] == m)]
                if not ks:
                    continue
                row = rate(J, ks)
                if g == "D+C-":
                    row["direct_both_phrasings"] = rate(J, [k for k in ks if meta[k].get("direct_both_phrasings")])
                    for t in ("relation", "conclusion", "non_relational"):
                        sub = [k for k in ks if meta[k].get("stype") == t]
                        if sub:
                            row[f"jstar_{t}"] = rate(J, sub)
                    row["is_first_claim"] = round(100 * float(np.mean([meta[k]["is_first_claim"] for k in ks])), 1)
                    for p_ in ("front", "behind"):
                        row[f"phrasing_{p_}"] = rate(J, [k for k in ks if k[1].endswith("_behind") == (p_ == "behind")])
                r[f"{m}|{g}"] = row
                c = row["consensus"]
                lines.append(f"AUDIT {st:5s} {m:13s} {g:8s} n={row['n_items']:3d} consensus {c['k']}/{c['n']} = {c['pct']}% {c['wilson']} "
                             f"worst-case [{row['worst_case'][0]}, {row['worst_case'][1]}] | Opus {row['opus']['pct']} GPT {row['gpt']['pct']}"
                             + (f" | Direct right both phrasings {row['direct_both_phrasings']['consensus']['k']}/{row['direct_both_phrasings']['consensus']['n']}"
                                f" | j* relational {row.get('jstar_relation', {}).get('consensus')} conclusion {row.get('jstar_conclusion', {}).get('consensus')}"
                                f" | j* = first claim {row['is_first_claim']}%"
                                f" | front {row['phrasing_front']['consensus']['k']}/{row['phrasing_front']['consensus']['n']} {row['phrasing_front']['consensus']['wilson']}"
                                f" behind {row['phrasing_behind']['consensus']['k']}/{row['phrasing_behind']['consensus']['n']} {row['phrasing_behind']['consensus']['wilson']}"
                                if g == "D+C-" else ""))
        lines.append(f"AUDIT {st:5s} judge agreement {r['agree_all']}% kappa(clear) {r['kappa_clear']} (n={r['n']})")
        res[st] = r
    return res


def prej(lines: list[str]) -> dict:
    res = {}
    for st in ("syn", "cues", "twins"):
        J = labels("prej", st)
        if not J[J2[0]] and not J[J2[1]]:
            continue
        # j* labels and the sampled D+C- rationales
        if st == "syn":
            import e20_commit_judge as E20
            base = [(b["model"], b["item_id"], b["idx"]) for b in E20.sample() if b["group"] == "D+C-" and b["model"] in COMP]
            JJ = {j: {(r["model"], r["item_id"]): r["label"] for r in rj(O / "e20" / f"{j}.jsonl") if r["group"] == "D+C-"} for j in J2}
            nsent = {}
            for m in COMP:
                for r in rj(O / "e2b" / f"{m}.jsonl"):
                    if r["task"] == "syn_depth":
                        nsent[(m, r["item_id"])] = r["sentences"]
        else:
            JC = labels("commit", st)
            base = sorted({(k[0], k[1], k[3]) for k in JC[J2[0]] if k[2] == "D+C-"})
            JJ = {j: {(k[0], k[1]): v["label"] for k, v in JC[j].items() if k[2] == "D+C-"} for j in J2}
            nsent = {}
            for m in COMP:
                for r in rj(E / f"{m}_{st}_traj.jsonl"):
                    nsent[(m, r["item_id"])] = r["sentences"]
        pre = defaultdict(list)
        for j in J2:
            for k in J[j]:
                pre[(k[0], k[1])].append(k)
        r = {}
        for m in COMP + ["pooled"]:
            bb = [b for b in base if m == "pooled" or b[0] == m]
            n_any_rel, n_contra, n_contra_hi, diffs, n_jstar_contra, n_jstar_cons = 0, 0, 0, [], 0, 0
            n_pre_sent, n_pre_contra_sent, n_pre_correct_sent = 0, 0, 0
            first_claim_pre = 0
            for (mm, iid, js) in bb:
                ks = sorted(set(pre.get((mm, iid), [])), key=lambda k: k[3])
                lab = {k[3]: consensus(J, k) for k in ks}
                n_pre_sent += len(ks)
                n_pre_contra_sent += sum(v == "INCORRECT" for v in lab.values())
                n_pre_correct_sent += sum(v == "CORRECT" for v in lab.values())
                if ks:
                    n_any_rel += 1
                contra = [i for i, v in lab.items() if v == "INCORRECT"]
                if contra:
                    n_contra += 1
                if any(v != "CORRECT" for v in lab.values()):
                    n_contra_hi += 1
                a, b2 = JJ[J2[0]].get((mm, iid)), JJ[J2[1]].get((mm, iid))
                jl = a if (a == b2 and a in ("CORRECT", "INCORRECT")) else "NONCONS"
                n_jstar_cons += jl != "NONCONS"
                n_jstar_contra += jl == "INCORRECT"
                if contra:
                    diffs.append(min(contra) - js)
                elif jl == "INCORRECT":
                    diffs.append(0)
                else:
                    diffs.append(None)
            n = len(bb)
            dd = [x for x in diffs if x is not None]
            row = {"n_rationales": n, "with_pre_relational": pct(n_any_rel, n), "pre_contradicting": pct(n_contra, n),
                   "pre_contradicting_worst_case_hi": round(100 * n_contra_hi / n, 1) if n else None,
                   "pre_sentences": n_pre_sent, "pre_sentences_consensus_incorrect": n_pre_contra_sent,
                   "pre_sentences_consensus_correct": n_pre_correct_sent,
                   "pre_sentence_contra_rate": pct(n_pre_contra_sent, n_pre_contra_sent + n_pre_correct_sent),
                   "jstar_contradicting": pct(n_jstar_contra, n_jstar_cons),
                   "first_contra_minus_jstar": {"n_defined": len(dd), "none": diffs.count(None),
                                                "zero": dd.count(0), "neg": sum(x < 0 for x in dd),
                                                "hist": dict(sorted(Counter(dd).items())),
                                                "median": float(np.median(dd)) if dd else None}}
            r[m] = row
            h = row["first_contra_minus_jstar"]
            lines.append(f"PREJ {st:5s} {m:13s} D+C- rationales {n} | with >=1 pre-j* relational {row['with_pre_relational']['k']} | "
                         f"with a consensus image-contradicting pre-j* relational sentence {row['pre_contradicting']['k']}/{n} = {row['pre_contradicting']['pct']}% "
                         f"{row['pre_contradicting']['wilson']} (upper bound if every non-consensus counts {row['pre_contradicting_worst_case_hi']}%) | "
                         f"pre-j* sentences {n_pre_sent}, consensus incorrect {n_pre_contra_sent} / correct {n_pre_correct_sent} | "
                         f"j* contradicting {row['jstar_contradicting']['k']}/{row['jstar_contradicting']['n']} | "
                         f"first contradicting - j*: <0 {h['neg']}, =0 {h['zero']}, none {h['none']} hist {h['hist']}")
        res[st] = r
    return res


# ============================================================================= claim edit at j*

def edit(lines: list[str]) -> dict:
    res = {}
    for st in ("cues", "twins"):
        for m in CUE_MODELS:
            R = rj(E / f"{m}_{st}_edit.jsonl")
            if not R:
                continue
            rng = np.random.default_rng(30)
            idx = {(r["item_id"], r["cond"]): r for r in R}
            keys = sorted({r["item_id"] for r in R})
            g = [idx[(k, "orig")]["image_id"] for k in keys]
            acc = {c: np.array([float(pe(idx[(k, c)]) == idx[(k, c)]["gold"]) for k in keys]) for c in ("orig", "correct", "neutral")}
            T = {r["item_id"]: r for r in rj(E / f"{m}_{st}_traj.jsonl")}
            ty = np.array([stype(T[k]["sentences"][T[k]["commit_idx"]]) for k in keys])
            r = {"n": len(keys), "n_scenes": len(set(g))}
            for c in acc:
                r[f"acc_{c}"] = cboot(acc[c], g, rng)
            for a1, a0 in (("correct", "neutral"), ("correct", "orig"), ("neutral", "orig")):
                r[f"{a1}-{a0}"] = cboot(acc[a1] - acc[a0], g, rng)
            for t in ("relation", "conclusion", "non_relational"):
                s = ty == t
                if s.sum() >= 5:
                    gg = [x for x, y in zip(g, s) if y]
                    r[f"by_type_{t}"] = {"n": int(s.sum()), "correct": cboot(acc["correct"][s], gg, rng),
                                         "neutral": cboot(acc["neutral"][s], gg, rng), "orig": cboot(acc["orig"][s], gg, rng),
                                         "correct-neutral": cboot(acc["correct"][s] - acc["neutral"][s], gg, rng)}
            ph = np.array([idx[(k, "orig")]["phrasing"] for k in keys])
            for p_ in ("front", "behind"):
                s = ph == p_
                gg = [x for x, y in zip(g, s) if y]
                r[f"by_phrasing_{p_}"] = {"n": int(s.sum()), **{c: cboot(acc[c][s], gg, rng) for c in ("orig", "correct", "neutral")},
                                          "correct-neutral": cboot(acc["correct"][s] - acc["neutral"][s], gg, rng)}
            res[f"{m}|{st}"] = r
            fa = lambda x: f"{x['mean']:+.1f} [{x['ci'][0]:+.1f}, {x['ci'][1]:+.1f}]"
            lines.append(f"EDIT {st:5s} {m:13s} D+C- with j* n={r['n']} | acc orig {r['acc_orig']['mean']} correct {r['acc_correct']['mean']} {r['acc_correct']['ci']} "
                         f"neutral {r['acc_neutral']['mean']} {r['acc_neutral']['ci']} | correct-neutral {fa(r['correct-neutral'])} correct-orig {fa(r['correct-orig'])} "
                         f"neutral-orig {fa(r['neutral-orig'])} | by j* type " + "; ".join(
                             f"{t} n={r[f'by_type_{t}']['n']} corr {r[f'by_type_{t}']['correct']['mean']} neu {r[f'by_type_{t}']['neutral']['mean']}"
                             for t in ("relation", "conclusion", "non_relational") if f"by_type_{t}" in r)
                         + " | by phrasing " + "; ".join(
                             f"{p_} n={r[f'by_phrasing_{p_}']['n']} corr {r[f'by_phrasing_{p_}']['correct']['mean']} neu {r[f'by_phrasing_{p_}']['neutral']['mean']} "
                             f"corr-neu {fa(r[f'by_phrasing_{p_}']['correct-neutral'])}" for p_ in ("front", "behind")))
    return res


# ============================================================================= compute / table

def compute(lines: list[str]) -> dict:
    tot, cells = 0.0, {}
    for p in sorted(E.glob("*.timing.json")):
        if "limit" in p.name:
            continue
        t = json.load(open(p))["total"]
        cells[p.name] = round(t / 60, 1)
        tot += t
    calls = Counter()
    for p in E.glob("judge_*.jsonl"):
        for r in rj(p):
            calls[r["judge"]] += 1
    lines.append(f"COMPUTE GPU {tot / 3600:.2f} GPU-h on H20 (one GPU per job, incl. model load) {cells} | judge calls {dict(calls)}")
    return {"gpu_h": round(tot / 3600, 2), "minutes": cells, "judge_calls": dict(calls)}


def table(R: dict) -> str:
    f2 = lambda x: f"${x['d']:+.2f}$ {{\\scriptsize$[{x['ci'][0]:+.2f}, {x['ci'][1]:+.2f}]$}}"
    fa = lambda x: f"${x['mean']:+.1f}$ {{\\scriptsize$[{x['ci'][0]:+.1f}, {x['ci'][1]:+.1f}]$}}"
    fp = lambda c: f"{c['k']}/{c['n']} ({c['pct']:.1f})" if c["n"] else "--"
    L = [r"\begin{table*}[t]", r"\centering", r"\scriptsize", r"\setlength{\tabcolsep}{3pt}",
         r"\caption{\textbf{Cue-factorial evidence for the commit and description analyses} (occlusion-competent models; \texttt{InternVL3-8B} as a non-competent contrast in (a)). "
         r"(a) Question-blind description arm on the cue-factorial set ($1{,}600$ prompts; description generated once per image, question in a new user turn, forced readout) "
         r"against \texttt{Direct} and \texttt{CoT} of \cref{tab:supp_glm}: accuracy (\%) and its change (points), change in the probit occlusion weight ($d'$ units), "
         r"and change in the pair-matched $d'_\mathrm{pair}$; $95\%$ twin-pair bootstrap CIs ($1{,}000$ draws for the weights, so \texttt{CoT}$-$\texttt{Direct} CIs can differ from \cref{tab:supp_glm} in the second decimal). "
         r"(b) Commit sentences $j^\ast$ of \texttt{D+C$-$} rationales on the cue-factorial set and the original twins ($40$ per model and set): two-judge consensus contradicting the image / judgeable, "
         r"relational \texttt{D+C$+$} sentences nearest the same relative position, and \texttt{D+C$-$} rationales with a consensus image-contradicting relational sentence before $j^\ast$ (Wilson CIs in \cref{sec:supp_commit}; \texttt{Qwen3-VL-8B} cue set: $120$ rationales, the $40$ audited ones and $80$ new, with the $40$-rationale value after the semicolon); "
         r"\texttt{Synthetic Depth} rows re-use the audit sample of \cref{tab:supp_commit_judge}. "
         r"Edit columns: accuracy (\%) on all \texttt{D+C$-$} items with a commit after $j^\ast$ is kept (Orig.), replaced by the true relation (Correct), or replaced by a neutral sentence (Neutral), "
         r"the text after it removed, and the rationale regenerated; last column: Correct$-$Neutral (points) on ``in front of'' / ``behind'' questions. "
         r"The edit columns come from the first edit run; the main text reports the seven-template re-run of \cref{tab:supp_e29}a, and \cref{M-tab:profile} the first-claim edit of \cref{tab:supp_claim_uncond}.}",
         r"\label{tab:supp_e28}"]
    D = R["desc"]["cues"]
    L += [r"\resizebox{\textwidth}{!}{%", r"\begin{tabular}{lccccccc}", r"\toprule",
          r"(a) & Acc.\ Dir./CoT/Desc. & $\Delta$acc.\ Desc.$-$Dir. & $\Delta$acc.\ Desc.$-$CoT & $\Delta$occl.\ CoT$-$Dir. & $\Delta$occl.\ Desc.$-$Dir. & $\Delta d'_\mathrm{pair}$ CoT$-$Dir. & $\Delta d'_\mathrm{pair}$ Desc.$-$Dir. \\", r"\midrule"]
    for m in CUE_MODELS:
        if m not in D:
            continue
        r = D[m]
        L.append(f"{NAMES[m]} & {r['acc_direct']['mean']:.1f}/{r['acc_cot']['mean']:.1f}/{r['acc_desc']['mean']:.1f} & {fa(r['acc_desc-direct'])} & {fa(r['acc_desc-cot'])} & "
                 f"{f2(r['glm']['cot-direct']['occlusion'])} & {f2(r['glm']['desc-direct']['occlusion'])} & {f2(r['dpair']['cot-direct_pooled'])} & {f2(r['dpair']['desc-direct_pooled'])} \\\\")
    L += [r"\bottomrule", r"\end{tabular}}", r"\\[4pt]", r"\resizebox{\textwidth}{!}{%", r"\begin{tabular}{llccccccc}", r"\toprule",
          r"(b) & Set & \texttt{D+C$-$} $j^\ast$ contradicts & Rel.\ \texttt{D+C$+$} contradicts & Pre-$j^\ast$ contradicting & Edit Orig. & Edit Correct & Edit Neutral & Corr.$-$Neu.\ front / behind \\", r"\midrule"]
    for m in COMP:
        for st in ("cues", "twins", "syn"):
            a = R["audit"].get(st, {})
            p = R["prej"].get(st, {}).get(m)
            e = R["edit"].get(f"{m}|{st}")
            c1 = fp(a[f"{m}|D+C-"]["consensus"]) if f"{m}|D+C-" in a else fp(p["jstar_contradicting"]) if p else "--"
            c2 = fp(a[f"{m}|D+C+rel"]["consensus"]) if f"{m}|D+C+rel" in a else "--"
            c3 = fp(p["pre_contradicting"]) if p else "--"
            if p and (m, st) == ("qwen3vl_8b", "cues") and (B / "e29_early.json").exists():
                # Exp-29 extended the pre-j* audit to 120 rationales (the 40 above + 80 new)
                ex = json.load(open(B / "e29_early.json"))["earliest"]["before_jstar_pct"]
                c3 = f"{ex['k']}/{ex['n']} ({ex['pct']:.1f}); {c3}"
            ed = [f"{e[f'acc_{c}']['mean']:.1f}" for c in ("orig", "correct", "neutral")] if e else ["--"] * 3
            ph = (f"${e['by_phrasing_front']['correct-neutral']['mean']:+.1f}$ / ${e['by_phrasing_behind']['correct-neutral']['mean']:+.1f}$"
                  if e else "--")
            if c1 == c2 == c3 == "--" and ed[0] == "--":
                continue
            sname = {"cues": "Cue-fact.", "twins": "Twins", "syn": r"Syn.\ Depth"}[st]
            L.append(f"{NAMES[m]} & {sname} & {c1} & {c2} & {c3} & {ed[0]} & {ed[1]} & {ed[2]} & {ph} \\\\")
    L += [r"\bottomrule", r"\end{tabular}}", r"\end{table*}"]
    return "\n".join(L) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--nb-glm", type=int, default=1000)
    ap.add_argument("--only", default="desc,traj,audit,prej,edit")
    ap.add_argument("--from-json", action="store_true", help="rebuild supp_e28.tex from bundle/e28_cues.json only")
    a = ap.parse_args()
    if a.from_json:
        tex = table(json.load(open(B / "e28_cues.json")))
        (B / "supp_e28.tex").write_text(tex)
        shutil.copy(B / "supp_e28.tex", PAPER_TAB / "supp_e28.tex")
        print(tex)
        return
    lines: list[str] = []
    out = {}
    fns = {"desc": lambda: desc(lines, a.nb_glm), "traj": lambda: traj(lines), "audit": lambda: audit(lines),
           "prej": lambda: prej(lines), "edit": lambda: edit(lines)}
    for k in a.only.split(","):
        out[k] = fns[k]()
    out["compute"] = compute(lines)
    json.dump(out, open(B / "e28_cues.json", "w"), indent=1)
    (B / "rigor_e28.txt").write_text("\n".join(lines) + "\n")
    if all(k in out for k in ("desc", "audit", "prej", "edit")):
        tex = table(out)
        (B / "supp_e28.tex").write_text(tex)
        shutil.copy(B / "supp_e28.tex", PAPER_TAB / "supp_e28.tex")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
