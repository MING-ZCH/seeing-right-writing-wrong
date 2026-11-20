"""Exp-31: Qwen3-VL-32B-Instruct and Qwen3-VL-8B-Thinking on the cue-factorial set (CPU, stored outputs).

Every statistic reuses the code behind Tab. 2 / Tabs. S18-S20 so the numbers are directly comparable with the seven
open models:
  accuracy      Direct / CoT / Desc; competence-rule CI as rigor_e27.comp (common.cluster_bootstrap_ci, clusters =
                (scene, phrasing), 2,000); paired deltas as rigor_e28 (cboot, twin-pair clusters = 4 prompts, 2,000).
  GLM           rigor_e17f.glm_rows / probit_fit (says-subject-front ~ phrasing + occlusion + vertical + size,
                d' = 2 beta), joint twin-pair bootstrap (1,000), common scale s and per-cue deviation (rigor_e28).
  AME           r18_cue_ame.stats (probability-scale average marginal effect and nonparametric cue contrast).
  d'_pair       rigor_e28.pair_sdt over (scene, phrasing) units (rigor_e17q.twin_stats), 2,000 scene resamples.
  cells         D+C-, D-C+, D+C+, D-C-, churn = (D+C- + D-C+) / n, net = (D-C+ - D+C-) / n (churn.json).
  shortcut      share of answers whose implied front object is the lower one (rigor_e27.heur occ_agree).
  first claim   claim_parse.first_claim_v3 on rigor_e17f.claims_for sentences: names the true front / lower / larger.
  follow 2x2    rigor_e30.follow: last relational claim (v3 parser, all items) true / false x CoT answer agrees.
  twins         original occlusion twins Delta d'_pair (rigor_e17q.run_twins; 32B from outputs/scale).
Thinking: Direct = empty think block, CoT = native think trace (e2d_think.py layout), Desc = e31_think_desc.py.
-> outputs/bundle/e31_cue_ladder.json, outputs/bundle/rigor_e31.txt,
   ../paper/tables/supp_e31.tex
Usage: python rigor_e31.py [--nb-glm 1000] [--table-only]
"""
from __future__ import annotations
import vtax_paths as _vtp  # noqa: E402

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np

from build_bundle import pe, rj
from claim_parse import first_claim_v3
from common import cluster_bootstrap_ci
from e2f_claim_edit import split_sents
from r18_cue_ame import CUES, stats as ame_stats
from rigor_e17f import geom_meta, glm_rows, probit_fit
from rigor_e17q import LOGIT_PROBIT, twin_stats
from rigor_e27 import occ_meta_rules
from rigor_e28 import cboot, cue_units, pair_sdt, q95
from rigor_e30 import implied, sent_front, split_sentences, two_by_two
from rigor_e32 import body as r32_body, cell_stats, rec as r32_rec

O = Path(f"{_vtp.OUTPUTS}")
B = Path("outputs/bundle")
PAPER_TAB = Path(f"{_vtp.TABLES}")
COT_PREFIX = "Let's think step by step."
NEW = ["qwen3vl_32b", "qwen3vl_8b_think"]
REF = ["qwen3vl_8b"]  # reproduction check against the stored bundle numbers
NAMES = {"qwen3vl_32b": "Qwen3-VL-32B", "qwen3vl_8b_think": "Qwen3-VL-8B-Thinking", "qwen3vl_8b": "Qwen3-VL-8B"}
QRX = re.compile(r"Is the (\w+ \w+) (in front of|behind) the (\w+ \w+)\?")
NB = 2000


# ----------------------------------------------------------------------------- loading

def norm_e2d(r: dict, phrasing: str) -> dict:
    """e2d / e2d_think record -> e10 cue-record layout (item_id with _behind, base_id, phrasing, question)."""
    mt = r["meta"]
    rel = "in front of" if phrasing == "front" else "behind"
    return r | {"base_id": r["item_id"], "item_id": r["item_id"] + ("_behind" if phrasing == "behind" else ""),
                "phrasing": phrasing, "question": f"Is the {mt['x_name']} {rel} the {mt['y_name']}?\n(A) yes\n(B) no"}


def load_arms(m: str) -> dict:
    """{arm: {item_id: record}} for direct / cot (/ desc) on the cue set."""
    by: dict = defaultdict(dict)
    if m == "qwen3vl_8b_think":
        for fn, ph in (("qwen3vl_8b_think.jsonl", "front"), ("qwen3vl_8b_think_behind.jsonl", "behind")):
            for r in rj(O / "e31" / fn):
                r = norm_e2d(r, ph)
                by[r["condition"]][r["item_id"]] = r
        dp = O / "e31" / "qwen3vl_8b_think_cues_desc.jsonl"
    else:
        for r in rj(O / "e10" / f"{m}_cues.jsonl"):
            if r["condition"] in ("direct", "cot"):
                by[r["condition"]][r["item_id"]] = r
        dp = O / "e28" / f"{m}_cues_desc.jsonl"
    for r in rj(dp):
        by[r["condition"]][r["item_id"]] = r
    return dict(by)


def body(r: dict) -> str:
    t = r.get("text") or ""
    return t[len(COT_PREFIX):].strip() if t.startswith(COT_PREFIX) else t.strip()


# ----------------------------------------------------------------------------- statistics

def glm_arms(recs_by: dict, contrasts: list, rng, nb: int) -> dict:
    """Probit cue weights per arm and arm contrasts with a joint twin-pair bootstrap (rigor_e28.glm_arms, any arms)."""
    names = ["occlusion", "vertical", "size"]
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
        out[c] = {k: {"d": round(2 * float(est[c][2 + i]), 2), "ci": q95(2 * bs[c][:, 2 + i])} for i, k in enumerate(names)}
        out[c]["max_abs_coef"] = round(float(np.abs(est[c][2:]).max()), 2)
        out[c]["separated"] = bool(np.abs(est[c][2:]).max() > 2.5)
    for a1, a0 in contrasts:
        out[f"{a1}-{a0}"] = {k: {"d": round(2 * float(est[a1][2 + i] - est[a0][2 + i]), 2),
                                 "ci": q95(2 * (bs[a1][:, 2 + i] - bs[a0][:, 2 + i]))} for i, k in enumerate(names)}
        pt = scale(est[a1], est[a0])
        bb = np.array([scale(bs[a1][i], bs[a0][i]) for i in range(nb)])
        out[f"{a1}-{a0}"]["common_scale"] = {nm: {"est": round(pt[i], 3), "ci": q95(bb[:, i], 3)}
                                            for i, nm in enumerate(["s", "delta_occlusion", "delta_vertical", "delta_size"])}
    return out


def ame_arms(recs_by: dict, contrasts: list, nb: int) -> dict:
    """r18_cue_ame on any arms (twin-pair bootstrap, arms resampled jointly, seed 18)."""
    data = {c: glm_rows(rs, use_size=True) for c, rs in recs_by.items()}
    pair = np.array(data["direct"]["pair"])
    pairs = sorted(set(pair))
    pidx = {p: np.where(pair == p)[0] for p in pairs}
    allidx = np.arange(len(pair))
    est = {c: ame_stats(data[c], allidx) for c in data}
    rng = np.random.default_rng(18)
    bs = {c: {cue: {k: [] for k in ("ame", "diff")} for cue in CUES} for c in data}
    for _ in range(nb):
        s = np.concatenate([pidx[pairs[j]] for j in rng.integers(0, len(pairs), len(pairs))])
        for c in data:
            st = ame_stats(data[c], s)
            for cue in CUES:
                for k in ("ame", "diff"):
                    bs[c][cue][k].append(st[cue][k])
    q1 = lambda v: [round(float(np.quantile(v, .025)), 1), round(float(np.quantile(v, .975)), 1)]
    r = {}
    for cue in CUES:
        r[cue] = {}
        for k in ("ame", "diff"):
            r[cue][k] = {c: {"est": round(est[c][cue][k], 1), "ci": q1(bs[c][cue][k])} for c in data}
            for a1, a0 in contrasts:
                v1, v0 = np.array(bs[a1][cue][k]), np.array(bs[a0][cue][k])
                r[cue][k][f"{a1}-{a0}"] = {"est": round(est[a1][cue][k] - est[a0][cue][k], 1), "ci": q1(v1 - v0)}
    return r


def cells(pd: np.ndarray, pc: np.ndarray, gold: np.ndarray) -> dict:
    d, c = pd == gold, pc == gold
    n = len(gold)
    k = {"D+C-": int((d & ~c).sum()), "D-C+": int((~d & c).sum()), "D+C+": int((d & c).sum()), "D-C-": int((~d & ~c).sum())}
    return k | {"n": n, "churn_pct": round(100 * (k["D+C-"] + k["D-C+"]) / n, 1), "net_pts": round(100 * (k["D-C+"] - k["D+C-"]) / n, 2)}


def lower_share(recs: list[dict]) -> dict:
    """rigor_e27.heur occ_agree: implied front = lower object (and = true front, = visibly larger)."""
    rows = []
    for r in recs:
        mt = r["meta"]
        says_yes = pe(r) == "A"
        x, y = mt["x_name"], mt["y_name"]
        front_pred = (x if says_yes else y) if r["phrasing"] == "front" else (y if says_yes else x)
        lower = mt["front_name"] if not mt["front_above"] else mt["back_name"]
        vis_larger = mt["front_name"] if occ_meta_rules(mt)["size_visible"] == 1.0 else mt["back_name"]
        rows.append((front_pred == mt["front_name"], front_pred == lower, front_pred == vis_larger, r["image_id"].replace("_cf", "")))
    a = np.array([x[:3] for x in rows], float)
    g = [x[3] for x in rows]
    f = lambda v: {"mean": round(100 * cluster_bootstrap_ci(v, g, NB)[0], 3),
                   "ci": [round(100 * x, 1) for x in cluster_bootstrap_ci(v, g, NB)[1:]]}
    return {"acc": f(a[:, 0]), "agree_lower": f(a[:, 1]), "agree_visible_larger": f(a[:, 2]), "n": len(rows)}


def first_claims(recs: list[dict], rng) -> dict:
    """rigor_e17f.cue_claims with claim_parse v3 (rigor_e17i.parser): share of parsed first claims naming front / lower / larger."""
    cl = []
    for r in recs:
        x = QRX.search(r["question"]).group(1).lower()
        sents = split_sents(body(r))
        fc = first_claim_v3(sents)
        if fc:
            cl.append({"named": fc[1], "pair": r["image_id"], **geom_meta(r["meta"], x)})
    out = {"n": len(recs), "n_parsed": len(cl), "parsed_pct": round(100 * len(cl) / len(recs), 1)}
    if not cl:
        return out
    pairs = sorted({c["pair"] for c in cl})
    pidx = {p: [i for i, c in enumerate(cl) if c["pair"] == p] for p in pairs}
    arr = {k: np.array([float(c["named"] == c[k]) for c in cl]) for k in ("front", "lower", "larger")}
    boots = {k: [] for k in arr}
    for _ in range(NB):
        s = np.concatenate([pidx[pairs[j]] for j in rng.integers(0, len(pairs), len(pairs))])
        for k in arr:
            boots[k].append(arr[k][s].mean())
    for k in arr:
        out[f"names_{k}"] = {"pct": round(100 * arr[k].mean(), 1), "ci": [round(100 * v, 1) for v in q95(boots[k], 4)]}
    return out


def follow_2x2(direct: dict, cot: dict, keys: list, rng) -> dict:
    """rigor_e30.follow 'last' definition over all items: last v3-parsed relational claim true/false x CoT answer agrees."""
    defs = defaultdict(list)
    n_cov = 0
    for k in keys:
        r = cot[k]
        mt, ph = r["meta"], r["phrasing"]
        ans = pe(r)
        d_ok = pe(direct[k]) == r["gold"]
        cl = [f for f in (sent_front(s, mt) for s in split_sentences(body(r))) if f]
        if not cl:
            continue
        n_cov += 1
        tr = cl[-1] == mt["front_name"]
        cons = implied(cl[-1], mt, ph) == ans
        for sub in ("all", "D+" if d_ok else "D-", ph):
            defs[sub].append((r["image_id"], tr, cons))
    out = {"coverage": n_cov, "n": len(keys)}
    for sub, rows in defs.items():
        out[sub] = two_by_two(rows, rng)
    return out


def twins_dpair(m: str) -> dict | None:
    """Original-twin d'_pair (rigor_e17q.run_twins layout; seed 232, 2,000 scene resamples)."""
    src = {"qwen3vl_32b": (O / "scale", "qwen3vl_32b"), "qwen3vl_8b_think": (O / "e2d", "qwen3vl_8b_think"),
           "qwen3vl_8b": (O / "e2d", "qwen3vl_8b")}[m]
    U = {c: defaultdict(dict) for c in ("direct", "cot")}
    for ph, fn in (("front", f"{src[1]}.jsonl"), ("behind", f"{src[1]}_behind.jsonl")):
        for r in rj(src[0] / fn):
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
                   for i in (np.concatenate([sidx[s] for s in rng.choice(us, len(us))]) for _ in range(NB))])
    n_ = len(est) // 3
    return {"n_units": len(keys), "direct": round(float(est[0]), 3), "cot": round(float(est[n_]), 3),
            "delta": {"d": round(float(est[2 * n_]), 3), "ci": q95(bs[:, 2 * n_], 3)},
            "bw_direct": int((oc_d == 3).sum()), "bw_cot": int((oc_c == 3).sum())}


# ----------------------------------------------------------------------------- per model

def analyze(m: str, nb_glm: int, lines: list[str]) -> dict:
    A = load_arms(m)
    keys = sorted(set(A["direct"]) & set(A["cot"]))
    arms = ["direct", "cot"] + (["desc"] if "desc" in A and set(keys) <= set(A["desc"]) else [])
    contrasts = [("cot", "direct")] + ([("desc", "direct"), ("desc", "cot")] if "desc" in arms else [])
    rng = np.random.default_rng(31)
    R = {"n": len(keys), "arms": arms}
    ref = {r["item_id"]: r for r in rj(O / "e10" / "qwen3vl_8b_cues.jsonl") if r["condition"] == "direct"}
    same = [k in ref and ref[k]["meta"] == A["direct"][k]["meta"] and ref[k]["gold"] == A["direct"][k]["gold"]
            and ref[k]["question"].split("\n")[0] == A["direct"][k]["question"].split("\n")[0] for k in keys]
    R["same_items_as_seven_model_runs_pct"] = round(100 * float(np.mean(same)), 2)
    gold = np.array([A["direct"][k]["gold"] for k in keys])
    P = {a: np.array([pe(A[a][k]) or "" for k in keys]) for a in arms}
    acc = {a: (P[a] == gold).astype(float) for a in arms}
    # competence-rule accuracy CI exactly as rigor_e27.comp (clusters = (scene, phrasing))
    g_comp = [k.replace("_cf", "") for k in keys]
    for a in arms:
        mu, lo, hi = cluster_bootstrap_ci(acc[a], g_comp, NB)
        R[f"acc_{a}_comp"] = {"mean": round(100 * mu, 3), "ci": [round(100 * lo, 1), round(100 * hi, 1)]}
    g = [A["direct"][k]["image_id"] for k in keys]
    for a in arms:
        R[f"acc_{a}"] = cboot(acc[a], g, rng)
    for a1, a0 in contrasts:
        R[f"acc_{a1}-{a0}"] = cboot(acc[a1] - acc[a0], g, rng)
    R["unparsed"] = {a: int((P[a] == "").sum()) for a in arms}
    # GLM, AME, d'_pair
    recs_by = {a: [A[a][k] for k in keys] for a in arms}
    R["glm"] = G = glm_arms(recs_by, contrasts, rng, nb_glm)
    R["ame"] = ame_arms(recs_by, contrasts, nb_glm)
    Us = {a: cue_units(recs_by[a]) for a in arms}
    R["dpair"] = pair_sdt(Us, lambda u: 2 * int(u["above"]) + int(u["larger"]), lambda u: int(u["above"]), rng, contrasts)
    # competence rule (rigor_e27.comp)
    ok1, ok2 = R["acc_direct_comp"]["ci"][0] > 70, G["direct"]["occlusion"]["ci"][0] > 0
    R["competent"] = bool(ok1 and ok2)
    R["rule_parts"] = {"cue_acc_ci_lo_gt70": ok1, "glm_occl_ci_lo_gt0": ok2}
    # cells, churn, net; shortcut; claims; follow 2x2
    R["cells"] = cells(P["direct"], P["cot"], gold)
    if "desc" in arms:
        R["cells_desc"] = cells(P["direct"], P["desc"], gold)
    R["lower"] = {a: lower_share(recs_by[a]) for a in arms}
    R["first_claim_cot"] = first_claims(recs_by["cot"], rng)
    if "desc" in arms:
        R["first_claim_desc"] = first_claims(recs_by["desc"], rng)
    R["follow_cot_last"] = follow_2x2(A["direct"], A["cot"], keys, rng)
    if "desc" in arms:
        R["follow_desc_last"] = follow_2x2(A["direct"], A["desc"], keys, rng)
    # generation lengths
    gl = [A["cot"][k].get("gen_len", 0) for k in keys]
    R["cot_tokens"] = {"mean": round(float(np.mean(gl)), 1), "median": float(np.median(gl)),
                       "max": int(max(gl)), "trunc_pct": round(100 * float(np.mean([x >= (4096 if m.endswith("think") else 512) for x in gl])), 2)}
    if "desc" in arms:
        dk = {A["desc"][k]["base_id"]: A["desc"][k] for k in keys}
        R["desc_tokens"] = {"mean": round(float(np.mean([r["gen_len"] for r in dk.values()])), 1),
                            "trunc_pct": round(100 * float(np.mean([bool(r.get("trunc")) for r in dk.values()])), 2)}
        if m.endswith("think"):
            R["desc_tokens"]["think_mean"] = round(float(np.mean([r["think_len"] for r in dk.values()])), 1)
            R["desc_tokens"]["think_trunc_pct"] = round(100 * float(np.mean([r["think_trunc"] for r in dk.values()])), 2)
        rr = A.get("direct_reread", {})
        if rr:
            R["desc_direct_reread_same_pct"] = round(100 * float(np.mean([pe(rr[k]) == pe(A["direct"][k]) for k in keys if k in rr])), 2)
    R["twins_dpair"] = twins_dpair(m)
    # L3 two-channel point (rigor_e32.cell_stats: x = last written relation acc - Direct acc on parsed items, y = CoT - Direct)
    l3 = [r32_rec(m, "cues", k, A["cot"][k]["image_id"], A["cot"][k]["gold"], pe(A["direct"][k]), pe(A["cot"][k]),
                  r32_body(A["cot"][k].get("text") or ""), A["cot"][k]["meta"]["front_name"], A["cot"][k]["meta"]["back_name"],
                  A["cot"][k]["meta"]["x_name"], A["cot"][k]["phrasing"]) for k in keys]
    R["l3"] = {"cues": cell_stats(l3, np.random.default_rng(32))}
    tw = l3_twins(m)
    if tw:
        R["l3"]["twins"] = cell_stats(tw, np.random.default_rng(32))
    # ---- text lines
    fa = lambda x: f"{x['mean']:+.1f} [{x['ci'][0]:+.1f}, {x['ci'][1]:+.1f}]"
    fm = lambda x: f"{x['mean']:.1f} [{x['ci'][0]:.1f}, {x['ci'][1]:.1f}]"
    fd = lambda x: f"{x['d']:+.2f} [{x['ci'][0]:+.2f}, {x['ci'][1]:+.2f}]"
    fe = lambda x: f"{x['est']:+.1f} [{x['ci'][0]:+.1f}, {x['ci'][1]:+.1f}]"
    nm = NAMES[m]
    lines.append(f"== {nm} ({m}) n={len(keys)} arms {arms} unparsed {R['unparsed']} | items identical to the e10 cue runs "
                 f"(meta, gold, question): {R['same_items_as_seven_model_runs_pct']}%")
    lines.append(f"ACC {nm}: Direct {fm(R['acc_direct_comp'])} (rule CI, (scene, phrasing) clusters) / {fm(R['acc_direct'])} (twin-pair clusters) | "
                 f"CoT {fm(R['acc_cot'])} | CoT-Dir {fa(R['acc_cot-direct'])}"
                 + (f" | Desc {fm(R['acc_desc'])} Desc-Dir {fa(R['acc_desc-direct'])} Desc-CoT {fa(R['acc_desc-cot'])}" if "desc" in arms else ""))
    lines.append(f"RULE {nm}: competent={R['competent']} {R['rule_parts']} (Direct acc CI lo {R['acc_direct_comp']['ci'][0]}, occlusion d' CI lo {G['direct']['occlusion']['ci'][0]})")
    for a in arms:
        lines.append(f"GLM {nm} {a:6s}: occl {fd(G[a]['occlusion'])} height {fd(G[a]['vertical'])} size {fd(G[a]['size'])} (max|b| {G[a]['max_abs_coef']}, separated {G[a]['separated']})")
    for a1, a0 in contrasts:
        c = G[f"{a1}-{a0}"]
        cs = c["common_scale"]
        lines.append(f"GLM {nm} {a1}-{a0}: occl {fd(c['occlusion'])} height {fd(c['vertical'])} size {fd(c['size'])} | common scale s {cs['s']['est']} {cs['s']['ci']} "
                     f"delta_occ {cs['delta_occlusion']['est']:+.3f} {cs['delta_occlusion']['ci']}")
    for cue in CUES:
        a_ = R["ame"][cue]["ame"]
        lines.append(f"AME {nm} {cue:9s}: " + " ".join(f"{a} {fe(a_[a])}" for a in arms) + " | "
                     + " ".join(f"{a1}-{a0} {fe(a_[f'{a1}-{a0}'])}" for a1, a0 in contrasts))
    D = R["dpair"]
    lines.append(f"DPAIR {nm}: " + " ".join(f"{a} {D[f'{a}_pooled']:.2f} {D[f'{a}_pooled_ci']}" for a in arms) + " | "
                 + " ".join(f"{a1}-{a0} {fd(D[f'{a1}-{a0}_pooled'])}" for a1, a0 in contrasts)
                 + f" | splits BC/SF/SB/BW " + " ".join(f"{a} {D[f'split_{a}']}" for a in arms))
    c = R["cells"]
    lines.append(f"CELLS {nm} (Direct vs CoT): D+C- {c['D+C-']} D-C+ {c['D-C+']} D+C+ {c['D+C+']} D-C- {c['D-C-']} | churn {c['churn_pct']}% net {c['net_pts']:+.2f} pts"
                 + (f" | Desc: D+X- {R['cells_desc']['D+C-']} D-X+ {R['cells_desc']['D-C+']} churn {R['cells_desc']['churn_pct']}% net {R['cells_desc']['net_pts']:+.2f}" if "desc" in arms else ""))
    lines.append(f"LOWER {nm}: follows lower-is-closer " + " ".join(f"{a} {fm(R['lower'][a]['agree_lower'])}" for a in arms))
    for key in ("first_claim_cot", "first_claim_desc"):
        if key in R and R[key].get("n_parsed"):
            fc = R[key]
            lines.append(f"CLAIM {nm} {key}: parsed {fc['n_parsed']}/{fc['n']} ({fc['parsed_pct']}%) names front {fc['names_front']['pct']} {fc['names_front']['ci']} "
                         f"lower {fc['names_lower']['pct']} {fc['names_lower']['ci']} larger {fc['names_larger']['pct']} {fc['names_larger']['ci']}")
    for key in ("follow_cot_last", "follow_desc_last"):
        if key in R:
            f2 = R[key]
            for sub in ("all", "D+", "D-"):
                if sub in f2 and f2[sub].get("n"):
                    x = f2[sub]
                    lines.append(f"FOLLOW {nm} {key} {sub}: coverage {f2['coverage']}/{f2['n']} | P(agree|true) {x['p_cons_true']['mean']} {x['p_cons_true']['ci']} (n={x['p_cons_true']['n']}) "
                                 f"P(agree|false) {x['p_cons_false']['mean']} {x['p_cons_false']['ci']} (n={x['p_cons_false']['n']}) cells {x['cells']}")
    lines.append(f"TOKENS {nm}: CoT {R['cot_tokens']}" + (f" | Desc {R['desc_tokens']} | Direct re-read same {R.get('desc_direct_reread_same_pct')}%" if "desc" in arms else ""))
    t = R["twins_dpair"]
    if t:
        lines.append(f"TWINS {nm}: d'pair Direct {t['direct']:.2f} -> CoT {t['cot']:.2f} delta {fd(t['delta'])} (BW units Direct {t['bw_direct']}, CoT {t['bw_cot']})")
    for ds, c in R["l3"].items():
        pred = L3_FIT["intercept"] + L3_FIT["slope"] * c["last_x"]
        lines.append(f"L3 {nm} {ds}: last-claim coverage {c['last_coverage']:.1f}% written acc {c['last_written_acc']:.1f} vs Direct on parsed {c['last_direct_on_parsed']:.1f} "
                     f"-> x {c['last_x']:+.2f} {c['last_x_ci']} | y (CoT-Direct, all) {c['y']:+.2f} {c['y_ci']} | like-for-like y {c['last_y_l4l']:+.2f} | "
                     f"agree(last) {c['last_agree']:.1f} | 21-point fit predicts y {pred:+.2f} (y - x {c['y'] - c['last_x']:+.2f})")
    return R


L3_FIT = {"slope": 0.871, "intercept": -1.4}  # Exp-32 21-point OLS (7 models x cues / twins / Syn. 431, last claim)


def l3_twins(m: str) -> list[dict]:
    """rigor_e32.items_twins for a model (32B from outputs/scale)."""
    d0 = {"qwen3vl_32b": O / "scale", "qwen3vl_8b_think": O / "e2d", "qwen3vl_8b": O / "e2d"}[m]
    out = []
    for suf, ph in (("", "front"), ("_behind", "behind")):
        R = defaultdict(dict)
        for r in rj(d0 / f"{m}{suf}.jsonl"):
            if r["condition"] in ("direct", "cot"):
                R[r["item_id"]][r["condition"]] = r
        for iid, d in R.items():
            if "cot" not in d or "direct" not in d or not d["cot"].get("text"):
                continue
            c, mt = d["cot"], d["cot"]["meta"]
            out.append(r32_rec(m, "twins", f"{iid}|{ph}", iid.replace("_cf", ""), c["gold"], pe(d["direct"]), pe(c),
                               r32_body(c["text"]), mt["front_name"], mt["back_name"], mt["x_name"], ph))
    return out


# ----------------------------------------------------------------------------- LaTeX

def s_(x: float, nd: int = 2) -> str:
    v = round(float(x), nd)
    return f"${(0.0 if v == 0 else v):+.{nd}f}$"


def u_(x: float, nd: int = 1) -> str:
    return f"${float(x):.{nd}f}$"


def ci_(lo: float, hi: float, nd: int = 2, signed: bool = True) -> str:
    f = (lambda v: f"{float(v):+.{nd}f}") if signed else (lambda v: f"{float(v):.{nd}f}")
    return r"{\tiny" + f"[{f(lo)},{f(hi)}]".replace("-", "$-$") + "}"


def star(lo: float, hi: float) -> str:
    return r"$^\ast$" if (lo > 0 or hi < 0) else ""


def table(res: dict) -> str:
    ms = [m for m in NEW if m in res]
    head = {"qwen3vl_32b": r"\shortstack{Qwen3-VL\\32B}", "qwen3vl_8b_think": r"\shortstack{Qwen3-VL\\8B-Think.$^\dagger$}"}
    row = lambda label, cells: label + " & " + " & ".join(cells) + r" \\"
    nc = len(ms) + 1
    L = [r"\begin{tabular}{@{}l" + "c" * len(ms) + "@{}}", r"\toprule",
         row(r"\textbf{Model}", [r"\textbf{" + head[m] + "}" for m in ms]), r"\midrule",
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
        if "glm" in res[m] and "desc-direct" in res[m]["glm"]:
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
    cells = []
    for m in ms:
        t = res[m].get("twins_dpair")
        cells.append(s_(t["delta"]["d"]) + star(*t["delta"]["ci"]) + (r"$^\ddagger$" if t["bw_direct"] == 0 else "") if t else "--")
    L.append(row("Twins: \\texttt{CoT} $\\Delta d'_\\mathrm{pair}$", cells))
    L.append(row("\\texttt{D+C$-$} / \\texttt{D$-$C+}; net (pts)", [f"${res[m]['cells']['D+C-']}$ / ${res[m]['cells']['D-C+']}$; " + s_(res[m]["cells"]["net_pts"], 1) for m in ms]))
    L.append(r"\midrule")
    L.append(rf"\multicolumn{{{nc}}}{{@{{}}l@{{}}}}{{\textit{{\textbf{{Written relation (\texttt{{CoT}}, parser v3)}}}}}} \\")
    L.append(row("First claim: true front / lower / larger (\\%)", [
        f"${res[m]['first_claim_cot']['names_front']['pct']:.1f}$ / ${res[m]['first_claim_cot']['names_lower']['pct']:.1f}$ / ${res[m]['first_claim_cot']['names_larger']['pct']:.1f}$"
        if res[m]["first_claim_cot"].get("n_parsed") else "--" for m in ms]))
    L.append(row("Answer agrees with last relation: true / false (\\%)", [
        f"${res[m]['follow_cot_last']['all']['p_cons_true']['mean']:.1f}$ / ${res[m]['follow_cot_last']['all']['p_cons_false']['mean']:.1f}$"
        if res[m]["follow_cot_last"].get("all", {}).get("n") else "--" for m in ms]))
    L.append(r"\bottomrule")
    th = res.get("qwen3vl_8b_think", {}).get("cot_tokens", {}).get("mean")
    note = (r"Same $1{,}600$ cue-factorial queries, readout (top-20 letter scores, vLLM V1), image budget and statistics as "
            r"\cref{M-tab:profile}; one run per column. $^\dagger$\texttt{CoT} = native thinking trace"
            + (f" (mean ${th:.0f}$ tokens)" if th else "")
            + r", \texttt{Direct} = empty think block. $^\S$Probit quasi-separated ($\max|\beta|>2.5$); use the AME row. "
            r"$^\ddagger$No both-wrong unit under \texttt{Direct}, so \texttt{Direct} $d'_\mathrm{pair}$ is a lower bound and its change is not interpretable. "
            r"$^\ast$: $95\%$ CI excludes zero (twin-pair bootstrap). Describe: question-blind description, then the question.")
    L.append(rf"\multicolumn{{{nc}}}{{p{{0.95\linewidth}}}}{{\scriptsize {note}}} \\")
    L.append(r"\end{tabular}")
    return "\n".join(L) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--nb-glm", type=int, default=1000)
    ap.add_argument("--models", default=",".join(NEW + REF))
    ap.add_argument("--table-only", action="store_true")
    a = ap.parse_args()
    out_json = B / "e31_cue_ladder.json"
    if a.table_only:
        res = json.load(open(out_json))["models"]
    else:
        lines: list[str] = []
        res = {}
        for m in a.models.split(","):
            if not load_arms(m).get("cot"):
                lines.append(f"== {m}: no outputs yet")
                continue
            res[m] = analyze(m, a.nb_glm, lines)
            print("\n".join(lines[-40:]), flush=True)
        doc = {"protocol": __doc__, "nb_glm": a.nb_glm, "nb": NB, "models": res}
        json.dump(doc, open(out_json, "w"), indent=1, default=float)
        (B / "rigor_e31.txt").write_text("\n".join(lines) + "\n")
    txt = table(res)
    (PAPER_TAB / "supp_e31.tex").write_text(txt)
    print(txt)


if __name__ == "__main__":
    main()
