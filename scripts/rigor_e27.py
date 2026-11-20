"""Exp-27 (round-15 external reviews, plan A reframe): CPU analyses from stored outputs, no GPU / API.

heur      heuristic baselines ('larger visible area is in front', 'lower in the image is closer', drawn-size, their
          combinations) on Synthetic Depth (both orders), original twins, cue-factorial set and hard occlusion set;
          per-model agreement of Direct (and CoT) with the rules and Direct accuracy where the size rule is wrong.
comp      competence table: cue-set Direct accuracy / GLM occlusion weight / twins / Synthetic Depth / commit audit /
          claim edit / loss locus per model; pre-stated 'occlusion-competent' rule and the competent-model summary.
ovl       Synthetic Depth restricted to the 431 visibly overlapping items: accuracy, pooled d', d'_pair, commit audit.
cons      internal-consistency checks (re-show deltas, Sym-Direct rounding, depth-valid duplicate check, Tab. S24,
          Fig. 2b / Fig. 4 values, probe-control ranges, App. M count, first-claim = j* shares, twin counts).
desc      question-blind description control (twins, held-out) and the cue-set equivalent (none run).
closed    closed-model reasoning settings per model x set x arm from the stored API records.
readout   readout pipelines (V0 exact vs top-20) and greedy re-run drift.
-> outputs/bundle/e27_reframe.json, outputs/bundle/rigor_e27.txt
Usage: python rigor_e27.py [--only heur,comp,...]
"""
from __future__ import annotations
import vtax_paths as _vtp  # noqa: E402

import argparse
import glob
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import norm

import build_bundle as BB
from build_bundle import pe, rj
from common import cluster_bootstrap_ci

O = Path(f"{_vtp.OUTPUTS}")
L = Path(f"{_vtp.LEGACY_RESULTS}")
B = Path("outputs/bundle")
M7 = ["qwen3b", "qwen7b", "internvl3_8b", "phi35v", "molmo7b", "qwen3vl_8b", "internvl35_8b"]
M8 = M7[:6] + ["qwen3vl_8b_think", "internvl35_8b"]
NAMES = BB.NAMES
QRX = re.compile(r"Is the (\w+ \w+) (in front of|behind) the (\w+ \w+)\?")
J2 = ("claude-opus-5-5", "gpt-6-astra")
NB = 2000


def r1(x, nd=1):
    return None if x is None else round(float(x), nd)


def wilson(k: int, n: int) -> list:
    if n == 0:
        return [None, None]
    z, p = 1.96, k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [r1(100 * (c - h)), r1(100 * (c + h))]


def cb(v, g, scale=100.0, nd=1) -> dict:
    v = np.asarray(v, float)
    if len(v) == 0:
        return {"n": 0}
    m, lo, hi = cluster_bootstrap_ci(v, list(g), NB)
    return {"mean": round(scale * m, nd + 2), "ci": [round(scale * lo, nd), round(scale * hi, nd)], "n": int(len(v))}


def fmt(d: dict, sign=True, nd=1) -> str:
    if not d or not d.get("n"):
        return "--"
    f = f"{{:{'+' if sign else ''}.{nd}f}}"
    return f"{f.format(d['mean'])} [{f.format(d['ci'][0])}, {f.format(d['ci'][1])}]"


# ----------------------------------------------------------------------------- geometry

def syn_items() -> dict:
    """Per Synthetic Depth image: front / back names, visible px, vertical centre, category, original-question subject."""
    from rigor_e17i import category, syn_overlap_metrics
    geo = json.load(open(B / "syn_depth_geometry.json"))["items"]
    cat = {r["uid"]: category(r) for r in syn_overlap_metrics()}
    cf = {r["item_id"]: r for r in rj(O / "e2a" / "qwen3b.jsonl") if r["variant"] == "cfq" and r["task"] == "syn_depth"}
    out = {}
    for k, r in cf.items():
        x, y = re.match(r"Is the (.+?) in front of the (.+?)\?", r["question"]).groups()
        front, back = (x, y) if r["gold"] == "A" else (y, x)
        g = geo.get(k, {})
        if front not in g or back not in g:
            continue
        out[k] = {"front": front, "back": back, "px_f": g[front]["px"], "px_b": g[back]["px"],
                  "cy_f": g[front]["cy"], "cy_b": g[back]["cy"], "cat": cat.get(k), "subj_orig": y, "subj_swap": x}
    return out


def rule_syn(it: dict) -> dict:
    """1 / 0 / 0.5 (undefined) credit for each rule on one Synthetic Depth image."""
    pf, pb = it["px_f"], it["px_b"]
    size = 0.5 if pf == pb else float(pf > pb)
    if it["cy_f"] is None or it["cy_b"] is None or it["cy_f"] == it["cy_b"]:
        low = 0.5
    else:
        low = float(it["cy_f"] > it["cy_b"])  # image y grows downward: larger cy = lower = closer
    return {"size_visible": size, "lower": low}


def occ_meta_rules(meta: dict) -> dict:
    """Rules for twin / cue / hard images (front is drawn on top, so its visible area is its drawn area)."""
    af, ab = meta["area_front_px"], meta["area_back_px"]
    vis_b = ab * (1 - meta["back_hidden_frac"])
    return {"size_drawn": float(af > ab) if af != ab else 0.5, "size_visible": float(af > vis_b) if af != vis_b else 0.5,
            "lower": 0.0 if meta["front_above"] else 1.0, "vis_ratio": af / max(1.0, vis_b), "drawn_ratio": af / ab}


def combos(size: np.ndarray, low: np.ndarray, feat: np.ndarray | None = None, grp=None) -> dict:
    """Agree-only rule (coverage, accuracy) and the best fixed tie-break; optional 5-fold logistic on continuous cues."""
    agree = (size != 0.5) & (low != 0.5) & (size == low)
    res = {"agree_coverage": r1(100 * agree.mean()), "agree_acc": r1(100 * size[agree].mean()) if agree.any() else None,
           "size_else_lower": r1(100 * np.where(size != 0.5, size, low).mean()),
           "agree_else_coin": r1(100 * np.where(agree, size, 0.5).mean())}
    if feat is not None:
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import GroupKFold
        # predict 'shape 1 is in front' with random shape order, so the label is balanced
        rng = np.random.default_rng(27)
        flip = rng.random(len(feat)) < 0.5
        X = np.where(flip[:, None], -feat, feat)
        y = (~flip).astype(int)
        g = np.arange(len(y)) if grp is None else np.asarray(grp)
        pred = np.zeros(len(y))
        for tr, te in GroupKFold(5).split(X, y, g):
            clf = LogisticRegression(C=1.0).fit(X[tr], y[tr])
            pred[te] = clf.predict(X[te])
        res["logistic_cv_acc"] = r1(100 * (pred == y).mean())
    return res


# ----------------------------------------------------------------------------- 1. heuristics

def heur(lines: list) -> dict:
    res = {"_doc": "rule accuracy = share of images whose rule-chosen front shape is the true front shape (0.5 credit if the "
                   "rule is undefined). Rules answer by relation, not by phrasing, so accuracy is identical under both object "
                   "orders / both phrasings of the same image. size_visible: larger visible area is in front; size_drawn: "
                   "larger drawn (unoccluded) area; lower: lower in the image is closer. Synthetic Depth geometry: "
                   "syn_depth_geometry.json (px = visible pixels, cy = vertical centre); twins: e2d meta; cue set: e10 *_cues "
                   "meta; hard set: e2h meta."}
    S = syn_items()
    ks = sorted(S, key=lambda k: int(k.split("_")[-1]))
    R = {k: rule_syn(S[k]) for k in ks}
    ratio = {k: max(S[k]["px_f"], S[k]["px_b"]) / max(1, min(S[k]["px_f"], S[k]["px_b"])) for k in ks}
    out = {}
    for sub, sel in (("all", ks), ("overlap431", [k for k in ks if S[k]["cat"] == "overlap"]),
                     ("touch_or_gap", [k for k in ks if S[k]["cat"] == "touch_or_gap"]),
                     ("near_total", [k for k in ks if S[k]["cat"] == "near_total"])):
        size = np.array([R[k]["size_visible"] for k in sel])
        low = np.array([R[k]["lower"] for k in sel])
        feat = np.array([[np.log((S[k]["px_f"] + 1) / (S[k]["px_b"] + 1)),
                          0.0 if None in (S[k]["cy_f"], S[k]["cy_b"]) else (S[k]["cy_f"] - S[k]["cy_b"]) / 100.0] for k in sel])
        out[sub] = {"n": len(sel), "size_visible": r1(100 * size.mean()), "size_visible_ci": wilson(int(size.sum()), len(sel)),
                    "lower": r1(100 * low.mean()), "lower_ci": wilson(int(low.sum()), len(sel)),
                    "n_size_wrong": int((size == 0).sum()), "n_lower_wrong": int((low == 0).sum())} | combos(size, low, feat)
    rv = np.array([ratio[k] for k in ks if min(S[k]["px_f"], S[k]["px_b"]) > 0])
    t1, t2 = np.quantile(rv, [1 / 3, 2 / 3])
    for t, lo, hi in (("ratio_low", 0, t1), ("ratio_mid", t1, t2), ("ratio_high", t2, 1e12)):
        sel = [k for k in ks if lo <= ratio[k] < hi and min(S[k]["px_f"], S[k]["px_b"]) > 0]
        out[t] = {"n": len(sel), "size_visible": r1(100 * np.mean([R[k]["size_visible"] for k in sel])),
                  "lower": r1(100 * np.mean([R[k]["lower"] for k in sel]))}
    out["ratio_terciles"] = [r1(t1, 2), r1(t2, 2)]
    out["median_visible_ratio"] = r1(float(np.median(rv)), 2)
    res["syn_depth"] = out
    lines.append(f"HEUR syn_depth all n={out['all']['n']}: size(visible) {out['all']['size_visible']}% {out['all']['size_visible_ci']} "
                 f"(wrong on {out['all']['n_size_wrong']}), lower {out['all']['lower']}% {out['all']['lower_ci']}, agree-only coverage "
                 f"{out['all']['agree_coverage']}% acc {out['all']['agree_acc']}%, size-else-lower {out['all']['size_else_lower']}%, "
                 f"logistic(size,vertical) 5-fold {out['all']['logistic_cv_acc']}% | overlap431 size {out['overlap431']['size_visible']}% "
                 f"lower {out['overlap431']['lower']}% | touch/gap(40) size {out['touch_or_gap']['size_visible']}% | near-total(29) size "
                 f"{out['near_total']['size_visible']}% | by ratio tercile size {out['ratio_low']['size_visible']}/{out['ratio_mid']['size_visible']}/"
                 f"{out['ratio_high']['size_visible']} lower {out['ratio_low']['lower']}/{out['ratio_mid']['lower']}/{out['ratio_high']['lower']} "
                 f"(cuts {out['ratio_terciles']}, median ratio {out['median_visible_ratio']})")

    # twins / cues / hard: one row per image
    def imgs_from(recs):
        seen = {}
        for r in recs:
            if r["condition"] == "direct" and r["item_id"] not in seen and r.get("meta"):
                seen[r["item_id"]] = r["meta"] if isinstance(r["meta"], dict) else json.loads(r["meta"])
        return seen

    sets = {"twins": imgs_from(rj(O / "e2d" / "qwen3b.jsonl")),
            "cues": imgs_from(rj(O / "e10" / "qwen3b_cues.jsonl")),
            "hard": imgs_from(rj(O / "e2h" / "qwen3vl_8b.jsonl"))}
    for name, im in sets.items():
        rows = [occ_meta_rules(m) for m in im.values()]
        sd = np.array([x["size_drawn"] for x in rows])
        sv = np.array([x["size_visible"] for x in rows])
        lw = np.array([x["lower"] for x in rows])
        feat = np.array([[np.log(x["vis_ratio"]), 1.0 if x["lower"] else -1.0] for x in rows])
        grp = [k.replace("_cf", "") for k in im]
        o = {"n_images": len(rows), "size_drawn": r1(100 * sd.mean()), "size_visible": r1(100 * sv.mean()),
             "size_visible_ci": wilson(int(sv.sum()), len(sv)), "smaller_visible_in_front": r1(100 * (1 - sv).mean()), "lower": r1(100 * lw.mean()),
             "mean_visible_ratio_front_over_back": r1(float(np.mean([x["vis_ratio"] for x in rows])), 2),
             "mean_drawn_ratio_front_over_back": r1(float(np.mean([x["drawn_ratio"] for x in rows])), 2),
             "drawn_and_lower": combos(sd, lw), "visible_and_lower": combos(sv, lw, feat, grp)}
        if name == "hard":
            lv = Counter(m.get("level") for m in im.values())
            o["levels"] = dict(lv)
            o["by_level"] = {l: {"size_visible": r1(100 * np.mean([occ_meta_rules(m)["size_visible"] for m in im.values() if m.get("level") == l])),
                                 "lower": r1(100 * np.mean([occ_meta_rules(m)["lower"] for m in im.values() if m.get("level") == l]))} for l in lv}
        res[name] = o
        lines.append(f"HEUR {name:5s} n_img={o['n_images']}: size(drawn) {o['size_drawn']}%, size(visible) {o['size_visible']}% {o['size_visible_ci']} "
                     f"(mean visible ratio front/back {o['mean_visible_ratio_front_over_back']}, drawn {o['mean_drawn_ratio_front_over_back']}), "
                     f"lower {o['lower']}% | drawn&lower agree cov {o['drawn_and_lower']['agree_coverage']}% acc {o['drawn_and_lower']['agree_acc']} | "
                     f"visible&lower agree cov {o['visible_and_lower']['agree_coverage']}% acc {o['visible_and_lower']['agree_acc']}, "
                     f"size-else-lower {o['visible_and_lower']['size_else_lower']}%, logistic 5-fold {o['visible_and_lower']['logistic_cv_acc']}%"
                     + (f" | by level {o['by_level']}" if name == "hard" else ""))

    # per-model agreement with the rules on Synthetic Depth (both orders) and on twins / cues (lower rule)
    from rigor_e17p import load_pairs
    per = {}
    for m in M7:
        d = load_pairs(m)
        keys = d["keys"]
        S_ = [S[k] for k in keys]
        sz = np.array([R[k]["size_visible"] for k in keys])
        lw = np.array([R[k]["lower"] for k in keys])
        ovl = np.array([S[k]["cat"] == "overlap" for k in keys])
        row = {}
        for arm in ("direct", "cot"):
            # implied front: original order asks about subj_orig, swapped order about subj_swap
            fo = np.array([s["subj_orig"] if y else (s["subj_swap"]) for s, y in zip(S_, d[f"{arm}_o"])])
            fs = np.array([s["subj_swap"] if y else (s["subj_orig"]) for s, y in zip(S_, d[f"{arm}_s"])])
            truth = np.array([s["front"] for s in S_])
            larger = np.array([s["front"] if r_ == 1.0 else s["back"] for s, r_ in zip(S_, sz)])
            lower = np.array([s["front"] if r_ == 1.0 else s["back"] for s, r_ in zip(S_, lw)])
            corr = np.concatenate([fo == truth, fs == truth]).astype(float)
            agree_size = np.concatenate([fo == larger, fs == larger]).astype(float)
            agree_low = np.concatenate([fo == lower, fs == lower]).astype(float)
            szw = np.concatenate([sz == 0, sz == 0])
            lww = np.concatenate([lw == 0, lw == 0])
            ov2 = np.concatenate([ovl, ovl])
            g = np.concatenate([np.arange(len(keys))] * 2)
            row[arm] = {"acc_two_order": cb(corr, g), "agree_size_rule": cb(agree_size, g), "agree_lower_rule": cb(agree_low, g),
                        "acc_where_size_wrong": {"k": int(corr[szw].sum()), "n": int(szw.sum()),
                                                 "pct": r1(100 * corr[szw].mean()) if szw.any() else None,
                                                 "wilson": wilson(int(corr[szw].sum()), int(szw.sum()))},
                        "acc_where_lower_wrong": cb(corr[lww], g[lww]), "acc_where_lower_right": cb(corr[~lww], g[~lww]),
                        "share_correct_also_size_right": r1(100 * np.mean(agree_size[corr == 1] == 1)),
                        "acc_overlap431_two_order": cb(corr[ov2], g[ov2]),
                        "acc_where_size_wrong_overlap": {"k": int(corr[szw & ov2].sum()), "n": int((szw & ov2).sum())}}
        per[m] = row
        x = row["direct"]
        lines.append(f"HEUR-MODEL syn {m:14s} Direct two-order acc {fmt(x['acc_two_order'], False)} | agrees with size rule {fmt(x['agree_size_rule'], False)} "
                     f"(CoT {fmt(row['cot']['agree_size_rule'], False)}) | with lower rule {fmt(x['agree_lower_rule'], False)} | acc where size wrong "
                     f"{x['acc_where_size_wrong']['k']}/{x['acc_where_size_wrong']['n']} | acc where lower wrong {fmt(x['acc_where_lower_wrong'], False)} vs right "
                     f"{fmt(x['acc_where_lower_right'], False)} | share of Direct-correct answers also size-correct {x['share_correct_also_size_right']}%")
    res["syn_model"] = per

    # twins / cue set: Direct agreement with the lower rule, the visible-size rule and with occlusion (accuracy)
    def occ_agree(recs, cond):
        rows = []
        for r in recs:
            if r["condition"] != cond:
                continue
            meta = r["meta"] if isinstance(r["meta"], dict) else json.loads(r["meta"])
            q = r.get("question") or ""
            ph = "behind" if (" behind " in q or r.get("phrasing") == "behind") else "front"
            says_yes = pe(r) == "A"
            x = meta["x_name"]
            y = meta["y_name"]
            front_pred = (x if says_yes else y) if ph == "front" else (y if says_yes else x)
            lower = meta["front_name"] if not meta["front_above"] else meta["back_name"]
            rr = occ_meta_rules(meta)
            vis_larger = meta["front_name"] if rr["size_visible"] == 1.0 else meta["back_name"]
            rows.append((front_pred == meta["front_name"], front_pred == lower, front_pred == vis_larger, r["image_id"].replace("_cf", "")))
        a = np.array([x[:3] for x in rows], float)
        g = [x[3] for x in rows]
        return {"acc": cb(a[:, 0], g), "agree_lower": cb(a[:, 1], g), "agree_visible_larger": cb(a[:, 2], g), "n": len(rows)}

    occ = {}
    for m in M8:
        recs = []
        for fn, ph in ((f"{m}.jsonl", "front"), (f"{m}_behind.jsonl", "behind")):
            for r in rj(O / "e2d" / fn):
                r["phrasing"] = ph
                recs.append(r)
        o = {"twins": {c: occ_agree(recs, c) for c in ("direct", "cot") if any(r["condition"] == c for r in recs)}}
        cr = rj(O / "e10" / f"{m}_cues.jsonl")
        if cr:
            o["cues"] = {c: occ_agree(cr, c) for c in ("direct", "cot")}
        occ[m] = o
        t = o["twins"]["direct"]
        s = f"HEUR-MODEL occ {m:16s} twins Direct acc {fmt(t['acc'], False)} agrees lower {fmt(t['agree_lower'], False)} visible-larger {fmt(t['agree_visible_larger'], False)}"
        if "cues" in o:
            c = o["cues"]["direct"]
            s += f" | cues Direct acc {fmt(c['acc'], False)} agrees lower {fmt(c['agree_lower'], False)} visible-larger {fmt(c['agree_visible_larger'], False)}"
            s += f" | cues CoT agrees lower {fmt(o['cues']['cot']['agree_lower'], False)}"
        lines.append(s)
    res["occ_model"] = occ
    return res


# ----------------------------------------------------------------------------- 2. competence

def comp(lines: list, H: dict | None) -> dict:
    gl = json.load(open(B / "glm_decomp.json"))["cues"]
    mh = json.load(open(B / "syn_pair_sdt_mh.json"))["twins"]
    sp = json.load(open(B / "syn_pair_sdt.json"))
    so = json.load(open(B / "syn_overlap.json"))["models"]
    dp = json.load(open(B / "dprime_ci.json"))
    cj = json.load(open(B / "commit_judge.json"))["rows"]
    ce = json.load(open(B / "claim_edit.json"))
    ek = json.load(open(B / "e17k.json"))["locus"]
    hs = json.load(open(O / "e2h" / "summary.json")) if (O / "e2h" / "summary.json").exists() else {}
    rule = ("occlusion-competent := (i) Direct accuracy on the size- and vertical-balanced cue-factorial set has 95% CI lower "
            "bound > 70% AND (ii) the Direct probit-GLM occlusion weight on that set has CI lower bound > 0. Models without a "
            "cue-set run (Qwen3-VL-8B-Thinking) are 'not evaluable' under the rule. Secondary, reported only: Direct twin "
            "pair-matched d' CI lower bound > 1 and twin accuracy CI lower bound > 70%.")
    res = {"rule": rule, "models": {}}
    for m in M8:
        r = {}
        # cue set: Direct / CoT accuracy (both phrasings, base-scene clusters)
        cr = [x for x in rj(O / "e10" / f"{m}_cues.jsonl") if x["condition"] in ("direct", "cot")]
        if cr:
            D = {(x["item_id"], x["phrasing"]): x for x in cr if x["condition"] == "direct"}
            C = {(x["item_id"], x["phrasing"]): x for x in cr if x["condition"] == "cot"}
            ks = sorted(set(D) & set(C))
            g = [k[0].replace("_cf", "") for k in ks]
            dacc = np.array([pe(D[k]) == D[k]["gold"] for k in ks], float)
            cacc = np.array([pe(C[k]) == C[k]["gold"] for k in ks], float)
            r["cue_direct_acc"] = cb(dacc, g)
            r["cue_cot_acc"] = cb(cacc, g)
            r["cue_cot_minus_direct"] = cb(cacc - dacc, g)
            G = gl[m]
            r["cue_glm_occl_direct"] = {"d": G["direct"]["occlusion"]["dprime"], "ci": G["direct"]["occlusion"]["ci"]}
            r["cue_glm_vert_direct"] = {"d": G["direct"]["vertical"]["dprime"], "ci": G["direct"]["vertical"]["ci"]}
            r["cue_glm_occl_delta"] = {"d": G["cot-direct"]["occlusion"]["dprime"], "ci": G["cot-direct"]["occlusion"]["ci"]}
            r["cue_glm_vert_delta"] = {"d": G["cot-direct"]["vertical"]["dprime"], "ci": G["cot-direct"]["vertical"]["ci"]}
            r["cue_auc_delta"] = G["cot-direct"].get("auc")
        # twins
        T = mh.get(m)
        if T:
            r["twin_dpair_direct"] = {"d": r1(T["direct_pooled"], 2), "ci": [r1(x, 2) for x in T["direct_pooled_ci"]]}
            r["twin_dpair_delta"] = {"d": r1(T["delta_pooled"], 2), "ci": [r1(x, 2) for x in T["delta_pooled_ci"]]}
        if m in dp and isinstance(dp[m], dict):
            r["twin_pooled_d_direct"] = dp[m]["dprime_direct"]
            r["twin_pooled_d_delta"] = {"d": dp[m]["dprime_cot_minus_direct"], "ci": dp[m]["diff_ci"]}
        tw = []
        for fn in (f"{m}.jsonl", f"{m}_behind.jsonl"):
            tw += [x for x in rj(O / "e2d" / fn) if x["condition"] in ("direct", "cot")]
        accd = [float(pe(x) == x["gold"]) for x in tw if x["condition"] == "direct"]
        accc = [float(pe(x) == x["gold"]) for x in tw if x["condition"] == "cot"]
        gd = [x["image_id"].replace("_cf", "") for x in tw if x["condition"] == "direct"]
        r["twin_direct_acc"] = cb(accd, gd)
        if len(accc) == len(accd):
            r["twin_cot_minus_direct"] = cb(np.array(accc) - np.array(accd), gd)
        # Synthetic Depth
        if m in sp["pairsdt"]:
            P = sp["pairsdt"][m]
            r["syn_dpair_direct"] = P["dpair_logit_direct"]
            r["syn_dpair_delta"] = {"d": P["dpair_logit_delta"], "ci": P["dpair_logit_delta_ci"]}
            r["syn_dpooled_delta"] = {"d": P["dpooled_delta"], "ci": P["dpooled_delta_ci"]}
        if m in so:
            r["syn_acc_delta"] = {"d": so[m]["all"]["acc_diff"], "ci": so[m]["all"]["acc_diff_ci"], "direct": so[m]["all"]["acc_direct"]}
        # commit audit (Tab. S30 sample, two-judge consensus among judgeable)
        for grp in ("D+C-", "D+C+"):
            c = cj.get(f"{m}|{grp}")
            if c:
                k = int(round(c["consensus_incorrect"] * c["consensus_n"] / 100))
                r[f"commit_{grp}"] = {"k": k, "n_judgeable": c["consensus_n"], "n": c["n"], "pct": c["consensus_incorrect"]}
        seeing = sp["seeing"]["commit_audit"]
        for name in ("DCm_commit", "DCp_rel_commitpos"):
            s = seeing[name].get(m)
            if s:
                r[f"{name}_relcomp" if name.startswith("DCp") else name] = {"all": s["contradict_all"][:2], "direct_pair_correct": s["contradict_pair_correct"][:2],
                                                                             "counts_all": s["counts"]["all"], "counts_pc": s["counts"]["pc"]}
        # claim edit (Tab. S33), same-turn regeneration, by set
        for sub in ("all", "syn", "occ"):
            x = ce.get(f"{m}|same|{sub}|D+C-|all")
            y = ce.get(f"{m}|same|{sub}|D+C+|all")
            if x:
                r[f"claim_edit_{sub}"] = {"n_DCm": x["n_items"], "DCm_correct": x["acc:correct"]["mean"], "DCm_neutral": x["acc:neutral"]["mean"],
                                          "DCm_correct_minus_neutral": [x["diff:correct-neutral"][k] for k in ("mean", "lo", "hi")],
                                          "n_DCp": y["n_items"] if y else None, "DCp_wrong": y["acc:wrong"]["mean"] if y else None,
                                          "DCp_neutral": y["acc:neutral"]["mean"] if y else None,
                                          "DCp_wrong_minus_neutral": [y["diff:wrong-neutral"][k] for k in ("mean", "lo", "hi")] if y else None}
        # loss locus (Tab. S19)
        for sset in ("cues", "twins"):
            x = ek.get(sset, {}).get(m)
            if x:
                r[f"locus_{sset}"] = {"direct": x["direct_acc"]["mean"], "first_claim_correct": x["p_claim_correct"]["mean"],
                                      "cot": x["cot_acc"]["mean"], "claim_stage_loss": x["claim_stage_loss"], "post_claim_loss": x["post_claim_loss"],
                                      "classification": x["classification"]}
        # hard occlusion set (open models run there)
        if m in hs and "pooled/direct" in hs[m]:
            h = hs[m]
            r["hard"] = {"direct_acc": h["pooled/direct"]["acc_mean"], "direct_acc_ci": h["pooled/direct"]["acc_ci"], "cot_acc": h["pooled/cot"]["acc_mean"],
                         "direct_d": h["pooled/direct"]["dprime"], "direct_d_ci": h["pooled/direct"]["ci"], "d_delta": h["pooled/cot_minus_direct_dprime"]}
        # rule
        if "cue_direct_acc" in r:
            ok1 = r["cue_direct_acc"]["ci"][0] > 70
            ok2 = r["cue_glm_occl_direct"]["ci"][0] > 0
            r["competent"] = bool(ok1 and ok2)
            r["rule_parts"] = {"cue_acc_ci_lo_gt70": ok1, "glm_occl_ci_lo_gt0": ok2}
        else:
            r["competent"] = None
        if "twin_dpair_direct" in r:
            r["secondary"] = {"twin_dpair_ci_lo_gt1": r["twin_dpair_direct"]["ci"][0] > 1, "twin_acc_ci_lo_gt70": r["twin_direct_acc"]["ci"][0] > 70}
        res["models"][m] = r
        f2 = lambda d: f"{d['d']:+.2f} [{d['ci'][0]:+.2f}, {d['ci'][1]:+.2f}]" if d else "--"
        lines.append(f"COMP {m:16s} competent={r['competent']} | cue Direct {fmt(r.get('cue_direct_acc'), False)} CoT-Dir {fmt(r.get('cue_cot_minus_direct'))} "
                     f"| GLM occl Dir {r.get('cue_glm_occl_direct')} dOccl {f2(r.get('cue_glm_occl_delta'))} vert Dir {r.get('cue_glm_vert_direct', {}).get('d')} dVert {f2(r.get('cue_glm_vert_delta'))} "
                     f"| twins Direct acc {fmt(r.get('twin_direct_acc'), False)} CoT-Dir {fmt(r.get('twin_cot_minus_direct'))} d'pair Dir {r.get('twin_dpair_direct')} d {f2(r.get('twin_dpair_delta'))} pooled d {f2(r.get('twin_pooled_d_delta'))} "
                     f"| syn acc {f2(r.get('syn_acc_delta'))} d'pair {r.get('syn_dpair_direct')} d {f2(r.get('syn_dpair_delta'))} pooled {f2(r.get('syn_dpooled_delta'))}")
        if "commit_D+C-" in r:
            lines.append(f"COMP-AUDIT {m:16s} commit D+C- {r['commit_D+C-']['k']}/{r['commit_D+C-']['n_judgeable']} ({r['commit_D+C-']['pct']}%) vs D+C+ (pos.-matched any) "
                         f"{r['commit_D+C+']['k']}/{r['commit_D+C+']['n_judgeable']} | relational comparator {r.get('DCp_rel_commitpos_relcomp', {}).get('all')} "
                         f"| D+C- on Direct pair-correct items {r.get('DCm_commit', {}).get('direct_pair_correct')} "
                         f"| claim edit all/syn/occ D+C- correct {[r.get(f'claim_edit_{s}', {}).get('DCm_correct') for s in ('all', 'syn', 'occ')]} "
                         f"neutral {[r.get(f'claim_edit_{s}', {}).get('DCm_neutral') for s in ('all', 'syn', 'occ')]} "
                         f"| locus cues {r.get('locus_cues')} | locus twins {r.get('locus_twins')} | hard {r.get('hard')}")
    comp_m = [m for m in M8 if res["models"][m]["competent"]]
    res["competent"] = comp_m
    res["not_competent"] = [m for m in M8 if res["models"][m]["competent"] is False]
    res["not_evaluable"] = [m for m in M8 if res["models"][m]["competent"] is None]
    # pooled commit audit over competent models vs the rest
    def pool(ms, key, sub="counts_all"):
        k = sum(res["models"][m][key][sub][0] for m in ms if key in res["models"][m])
        n = sum(res["models"][m][key][sub][1] for m in ms if key in res["models"][m])
        return {"k": k, "n": n, "pct": r1(100 * k / n) if n else None, "wilson": wilson(k, n)}
    rest = [m for m in M7 if m not in comp_m]
    res["pooled_audit"] = {
        "competent": {"DCm": pool(comp_m, "DCm_commit"), "DCp_rel": pool(comp_m, "DCp_rel_commitpos_relcomp"),
                      "DCm_direct_pair_correct": pool(comp_m, "DCm_commit", "counts_pc")},
        "rest": {"DCm": pool(rest, "DCm_commit"), "DCp_rel": pool(rest, "DCp_rel_commitpos_relcomp"),
                 "DCm_direct_pair_correct": pool(rest, "DCm_commit", "counts_pc")},
        "all7": {"DCm": pool(M7, "DCm_commit"), "DCp_rel": pool(M7, "DCp_rel_commitpos_relcomp")}}
    pa = res["pooled_audit"]
    lines.append(f"COMP-RULE {rule}")
    lines.append(f"COMP-RESULT competent {comp_m} | not competent {res['not_competent']} | not evaluable {res['not_evaluable']}")
    lines.append(f"COMP-POOLED commit audit (Syn. Depth, two-judge consensus among judgeable): competent D+C- {pa['competent']['DCm']} vs relational D+C+ {pa['competent']['DCp_rel']} "
                 f"(D+C- restricted to Direct pair-correct images {pa['competent']['DCm_direct_pair_correct']}) | rest D+C- {pa['rest']['DCm']} vs {pa['rest']['DCp_rel']} | all7 {pa['all7']}")
    return res


# ----------------------------------------------------------------------------- 3. 431 overlap filter

def ovl(lines: list) -> dict:
    from rigor_e17i import category, syn_overlap_metrics
    from rigor_e17p import SNAMES, consensus, load_pairs, outcome, stats
    cat = {r["uid"]: category(r) for r in syn_overlap_metrics()}
    so = json.load(open(B / "syn_overlap.json"))["models"]
    res = {"_doc": "431 = syn_overlap 'overlap' category (contact >= 0.15 and both shapes >= 100 visible px). Two-order statistics as "
                   "rigor_e17p.stats (image bootstrap 2,000, seed 27); single-order accuracy / d' from syn_overlap.json (Tab. S11). "
                   "Commit audit = e20 judge records (Sec. 5 cells) restricted to overlapping item ids; consensus as commit_judge.json.",
           "models": {}}
    rng = np.random.default_rng(27)
    idx = {n: i for i, n in enumerate(SNAMES)}
    for m in M7:
        d = load_pairs(m)
        g = d["gold_yes"]
        out = {}
        for sub in ("all", "overlap"):
            sel = np.array([sub == "all" or cat.get(k) == "overlap" for k in d["keys"]])
            yd, sd_, yc, sc = d["direct_o"][sel], d["direct_s"][sel], d["cot_o"][sel], d["cot_s"][sel]
            gg = g[sel]
            oc_d, oc_c = outcome(yd, sd_, gg), outcome(yc, sc, gg)
            est = stats(oc_d, oc_c, (yd, sd_, gg), (yc, sc, gg))
            n = int(sel.sum())
            bs = []
            for _ in range(NB):
                i = rng.integers(0, n, n)
                bs.append(stats(oc_d[i], oc_c[i], (yd[i], sd_[i], gg[i]), (yc[i], sc[i], gg[i])))
            bs = np.array(bs)
            q = lambda nm: [r1(np.quantile(bs[:, idx[nm]], .025), 2), r1(np.quantile(bs[:, idx[nm]], .975), 2)]
            out[sub] = {"n": n, "dpair_direct": r1(est[idx["dpair_logit_direct"]], 2), "dpair_cot": r1(est[idx["dpair_logit_cot"]], 2),
                        "dpair_delta": r1(est[idx["dpair_logit_delta"]], 3), "dpair_delta_ci": q("dpair_logit_delta"),
                        "dpooled_direct": r1(est[idx["dpooled_direct"]], 2), "dpooled_cot": r1(est[idx["dpooled_cot"]], 2),
                        "dpooled_delta": r1(est[idx["dpooled_delta"]], 3), "dpooled_delta_ci": q("dpooled_delta"),
                        "acc2_delta": r1(est[idx["acc_delta"]], 2), "acc2_delta_ci": q("acc_delta")}
        out["single_order"] = {k: so[m][k] for k in ("all", "overlap")}
        res["models"][m] = out
        a, o = out["all"], out["overlap"]
        lines.append(f"OVL {m:14s} single-order acc Dir->CoT all {so[m]['all']['acc_direct']}->{so[m]['all']['acc_cot']} ({so[m]['all']['acc_diff']:+.1f}) "
                     f"| 431 {so[m]['overlap']['acc_direct']}->{so[m]['overlap']['acc_cot']} ({so[m]['overlap']['acc_diff']:+.1f} {so[m]['overlap']['acc_diff_ci']}) "
                     f"| d'pair all {a['dpair_direct']}->{a['dpair_cot']} ({a['dpair_delta']:+.2f} {a['dpair_delta_ci']}) 431 {o['dpair_direct']}->{o['dpair_cot']} ({o['dpair_delta']:+.2f} {o['dpair_delta_ci']}) "
                     f"| pooled d' all {a['dpooled_delta']:+.2f} {a['dpooled_delta_ci']} 431 {o['dpooled_direct']}->{o['dpooled_cot']} ({o['dpooled_delta']:+.2f} {o['dpooled_delta_ci']}) "
                     f"| two-order acc delta all {a['acc2_delta']:+.1f} 431 {o['acc2_delta']:+.1f} {o['acc2_delta_ci']}")
    th = so.get("qwen3vl_8b_think")
    if th:
        res["models"]["qwen3vl_8b_think"] = {"single_order": {k: th[k] for k in ("all", "overlap")}}
        lines.append(f"OVL qwen3vl_8b_think single-order acc all {th['all']['acc_diff']:+.1f} 431 {th['overlap']['acc_diff']:+.1f} {th['overlap']['acc_diff_ci']} (no swapped-order run)")
    # commit audit restricted to overlapping items
    aud = {}
    for name, suf, grp in (("DCm_commit", "", "D+C-"), ("DCp_rel", "_relcomp", "D+C+rel")):
        J = {j: {(r["model"], r["item_id"]): r for r in rj(O / "e20" / f"{j}{suf}.jsonl") if r["group"] == grp} for j in J2}
        keys = sorted(set(J[J2[0]]) & set(J[J2[1]]))
        rows = {}
        for m in M7 + ["pooled"]:
            ks = [k for k in keys if m == "pooled" or k[0] == m]
            ko = [k for k in ks if cat.get(k[1]) == "overlap"]
            k_all, n_all = consensus(J, ks)
            k_o, n_o = consensus(J, ko)
            rows[m] = {"all": [k_all, n_all, r1(100 * k_all / max(1, n_all))], "overlap": [k_o, n_o, r1(100 * k_o / max(1, n_o))],
                       "n_items_all": len(ks), "n_items_overlap": len(ko), "wilson_overlap": wilson(k_o, n_o)}
        aud[name] = rows
    res["commit_audit"] = aud
    lines.append("OVL-AUDIT D+C- commit contradiction all -> 431: " + "; ".join(f"{m} {aud['DCm_commit'][m]['all'][0]}/{aud['DCm_commit'][m]['all'][1]} ({aud['DCm_commit'][m]['all'][2]}) -> "
                                                                           f"{aud['DCm_commit'][m]['overlap'][0]}/{aud['DCm_commit'][m]['overlap'][1]} ({aud['DCm_commit'][m]['overlap'][2]})" for m in M7 + ["pooled"]))
    lines.append("OVL-AUDIT relational D+C+ comparator all -> 431: " + "; ".join(f"{m} {aud['DCp_rel'][m]['all'][2]} -> {aud['DCp_rel'][m]['overlap'][2]} (nj {aud['DCp_rel'][m]['overlap'][1]})" for m in M7 + ["pooled"]))
    # summary of headline changes
    s = res["models"]
    res["summary"] = {
        "single_order_acc_delta_range_all": [min(so[m]["all"]["acc_diff"] for m in M7 if so[m]["all"]["acc_diff"] < 0), max(so[m]["all"]["acc_diff"] for m in M7 if so[m]["all"]["acc_diff"] < 0)],
        "single_order_acc_delta_range_431": [min(so[m]["overlap"]["acc_diff"] for m in M7 if so[m]["overlap"]["acc_diff"] < 0), max(so[m]["overlap"]["acc_diff"] for m in M7 if so[m]["overlap"]["acc_diff"] < 0)],
        "dpair_fall_ci_lt0_all": [m for m in M7 if s[m]["all"]["dpair_delta_ci"][1] < 0],
        "dpair_fall_ci_lt0_431": [m for m in M7 if s[m]["overlap"]["dpair_delta_ci"][1] < 0],
        "dpair_range_falls_all": [min(s[m]["all"]["dpair_delta"] for m in M7 if s[m]["all"]["dpair_delta_ci"][1] < 0), max(s[m]["all"]["dpair_delta"] for m in M7 if s[m]["all"]["dpair_delta_ci"][1] < 0)],
        "dpair_range_falls_431": [min(s[m]["overlap"]["dpair_delta"] for m in M7 if s[m]["overlap"]["dpair_delta_ci"][1] < 0), max(s[m]["overlap"]["dpair_delta"] for m in M7 if s[m]["overlap"]["dpair_delta_ci"][1] < 0)],
        "dpooled_fall_ci_lt0_431": [m for m in M7 if s[m]["overlap"]["dpooled_delta_ci"][1] < 0],
        "dpooled_range_falls_431": [min(s[m]["overlap"]["dpooled_delta"] for m in M7 if s[m]["overlap"]["dpooled_delta"] < 0), max(s[m]["overlap"]["dpooled_delta"] for m in M7 if s[m]["overlap"]["dpooled_delta"] < 0)],
        "commit_pooled_all": aud["DCm_commit"]["pooled"]["all"], "commit_pooled_431": aud["DCm_commit"]["pooled"]["overlap"],
        "relcomp_pooled_all": aud["DCp_rel"]["pooled"]["all"], "relcomp_pooled_431": aud["DCp_rel"]["pooled"]["overlap"]}
    lines.append(f"OVL-SUM {res['summary']}")
    return res


# ----------------------------------------------------------------------------- 4. internal consistency

def cons(lines: list) -> dict:
    res = {}
    # (c) re-show deltas (Tab. 1), unrounded
    rs = {}
    for m in BB.MODELS:
        main = BB.load_main(O, L, m)
        if "cot" not in main:
            continue
        ks = [k for k in main["direct"] if k in main["cot"]]
        row = {"conds": sorted(main.keys())}
        for c in ("cot+turn", "cot+restate", "cot+reimage", "cot+reimage_restate"):
            if c in main and all(k in main[c] for k in ks):
                row[c] = round(100 * float(np.mean([main[c][k][0] - main["cot"][k][0] for k in ks])), 3)
                row[c + "_counts"] = [int(sum(main[c][k][0] for k in ks)), int(sum(main["cot"][k][0] for k in ks)), len(ks)]
        rs[m] = row
    vals = [(m, c, v) for m, r in rs.items() for c, v in r.items() if c in ("cot+turn", "cot+restate", "cot+reimage", "cot+reimage_restate")]
    post = [(m, c, v) for m, c, v in vals if c != "cot+turn"]
    res["reshow"] = {"per_model": rs, "post_cot_resupply_range": [min(v for *_, v in post), max(v for *_, v in post)],
                     "argmax": max(post, key=lambda x: x[2]), "argmin": min(post, key=lambda x: x[2]),
                     "incl_cot_turn_range": [min(v for *_, v in vals), max(v for *_, v in vals)]}
    lines.append("CONS-c re-show minus CoT (pts, unrounded, 2,850 items): " + "; ".join(
        f"{m} " + " ".join(f"{c.replace('cot+', '')} {r[c]:+.3f}" for c in ("cot+turn", "cot+restate", "cot+reimage", "cot+reimage_restate") if c in r) for m, r in rs.items())
                 + f" | post-CoT re-supply (restate/reimage/both) range {res['reshow']['post_cot_resupply_range']} max {res['reshow']['argmax']}")
    # (b) Sym-Direct, unrounded (numbers.json fix_main_pool is stored at 2 dp -> table double-rounds)
    nb = json.load(open(B / "numbers.json"))
    sym = {}
    for m in M7:
        main = BB.load_main(O, L, m)
        x = BB.sym_and_gate(O, main, m)
        sym[m] = {k: x[k]["mean"] for k in ("direct", "sym_direct", "sym_direct_minus_direct", "sym_cot_minus_direct", "cot_consistent_else_direct_minus_direct", "cot_minus_direct")}
        sym[m]["stored_numbers_json"] = {k: nb[m]["fix_main_pool"][k]["mean"] for k in ("sym_direct_minus_direct", "sym_cot_minus_direct")}
        sym[m]["printed_old"] = f"{nb[m]['fix_main_pool']['sym_direct_minus_direct']['mean']:+.1f}"
        sym[m]["printed_correct"] = f"{x['sym_direct_minus_direct']['mean']:+.1f}"
    res["sym_direct"] = sym
    lines.append("CONS-b Sym-Direct minus Direct (unrounded 4 dp / stored 2 dp / printed old -> correct): " + "; ".join(
        f"{m} {v['sym_direct_minus_direct']} / {v['stored_numbers_json']['sym_direct_minus_direct']} / {v['printed_old']} -> {v['printed_correct']} (Sym {v['sym_direct']}, Dir {v['direct']})" for m, v in sym.items()))
    # (a) depth-valid duplicate check
    from heldout_breakdown import DEPTH, model_rows
    Rr = {m: {r["item"]: r for r in model_rows(m) if r["task"] in DEPTH} for m in ("qwen3vl_8b", "internvl35_8b")}
    a, b = Rr["qwen3vl_8b"], Rr["internvl35_8b"]
    ks = sorted(set(a) & set(b))
    dv = lambda R_, key: np.array([float(R_[k][key] == R_[k]["gold"]) - float(R_[k]["pd"] == R_[k]["gold"]) for k in ks])
    hb = json.load(open(B / "heldout_breakdown.json"))
    res["depth_valid_dup"] = {
        "n": len(ks), "same_direct_pred": int(sum(a[k]["pd"] == b[k]["pd"] for k in ks)), "same_cot_pred": int(sum(a[k]["pc"] == b[k]["pc"] for k in ks)),
        "cot_minus_direct_net_items": [int(dv(a, "pc").sum()), int(dv(b, "pc").sum())], "items_with_different_cot_delta": int((dv(a, "pc") != dv(b, "pc")).sum()),
        "sym_net_items": [int(dv(a, "sym").sum()), int(dv(b, "sym").sum())], "items_with_different_sym_delta": int((dv(a, "sym") != dv(b, "sym")).sum()),
        "direct_acc": [r1(100 * np.mean([a[k]["pd"] == a[k]["gold"] for k in ks]), 2), r1(100 * np.mean([b[k]["pd"] == b[k]["gold"] for k in ks]), 2)],
        "stored_unrounded": {m: {"cot_minus_direct": hb[m]["depth324"]["cot_minus_direct"], "sym_minus_direct": hb[m]["depth324"]["sym_minus_direct"]} for m in ("qwen3vl_8b", "internvl35_8b")}}
    x = res["depth_valid_dup"]
    lines.append(f"CONS-a depth-valid 324 Qwen3-VL-8B vs InternVL3.5-8B: not a copy. Direct preds identical on {x['same_direct_pred']}/324, CoT on {x['same_cot_pred']}/324; "
                 f"CoT-Direct per-item deltas differ on {x['items_with_different_cot_delta']} items but both net {x['cot_minus_direct_net_items']} (= -4.63 pts); "
                 f"Sym-Direct deltas differ on {x['items_with_different_sym_delta']} items, both net {x['sym_net_items']} (= -1.23 pts); Direct acc {x['direct_acc']}; "
                 f"CI identical to 0.1 because the bootstrap uses the same seed-0 cluster draws on the same 324 items; unrounded {x['stored_unrounded']}")
    # (d) / (j) Tab. S24
    from rigor_e17 import bias_sensitivity
    bs_new = bias_sensitivity()
    bs_old = json.load(open(B / "bias_sensitivity.json"))
    res["tab_s24"] = {m: {"d_dprime_old": bs_old[m]["cot_vs_direct"]["d_dprime"], "d_dprime_new": bs_new[m]["cot_vs_direct"]["d_dprime"],
                          "acc_loss": bs_new[m]["cot_vs_direct"]["acc_loss"], "balanced_acc_loss": bs_new[m]["cot_vs_direct"]["balanced_acc_loss"],
                          "sens_plus_crit": r1(bs_new[m]["cot_vs_direct"]["loss_from_sensitivity"] + bs_new[m]["cot_vs_direct"]["loss_from_criterion"]),
                          "sens_old": bs_old[m]["cot_vs_direct"]["loss_from_sensitivity"], "sens_new": bs_new[m]["cot_vs_direct"]["loss_from_sensitivity"],
                          "crit_old": bs_old[m]["cot_vs_direct"]["loss_from_criterion"], "crit_new": bs_new[m]["cot_vs_direct"]["loss_from_criterion"]}
                      for m in bs_new if "cot_vs_direct" in bs_new[m]}
    res["_bias_new"] = bs_new
    lines.append("CONS-d/j Tab. S24 (single-phrasing Syn. Depth): d_dprime old (diff of rounded d') -> new (unrounded); sens+crit = balanced-acc loss vs acc loss: " + "; ".join(
        f"{m} {v['d_dprime_old']:+.2f}->{v['d_dprime_new']:+.2f}, sens/crit {v['sens_old']}/{v['crit_old']}->{v['sens_new']}/{v['crit_new']} sum {v['sens_plus_crit']} = bal {v['balanced_acc_loss']} (acc {v['acc_loss']})"
        for m, v in res["tab_s24"].items()))
    # (e) Fig. 2b normalization
    cbj = json.load(open(B / "commit_baseline.json"))
    f2 = {m: {"rel_of_committed": cbj[m]["commit_relational_share_of_committed"], "n_committed": cbj[m]["n_committed"], "n_DCm": cbj[m]["n_DCm"],
              "rel_of_all_DCm": r1(cbj[m]["commit_relational_share_of_committed"] * cbj[m]["n_committed"] / cbj[m]["n_DCm"]),
              "fig2b_bar_label": nb[m]["error_onset"]["D+C-"].get("type_relation")} for m in M7 if m in cbj}
    res["fig2b"] = {"per_model": f2, "range_committed": [min(v["rel_of_committed"] for v in f2.values()), max(v["rel_of_committed"] for v in f2.values())],
                    "range_all_DCm": [min(v["fig2b_bar_label"] for v in f2.values()), max(v["fig2b_bar_label"] for v in f2.values())]}
    lines.append(f"CONS-e Fig. 2b: bars/labels = relational commits as % of all D+C- items {res['fig2b']['range_all_DCm']}; text/Tab. S28 = % of committed D+C- {res['fig2b']['range_committed']}; "
                 + "; ".join(f"{m} {v['fig2b_bar_label']} (all, n={v['n_DCm']}) vs {v['rel_of_committed']} (committed, n={v['n_committed']})" for m, v in f2.items()))
    # (f) Fig. 4 L/R subset S_orig
    es = json.load(open(B / "e3x_same_subset.json"))
    res["fig4_lr"] = {m: {"S_orig_LR": es[m]["S_original_same_subset"]["mean"], "S_gray_LR": es[m]["S_gray_same_subset"]["mean"],
                          "S_mirror_LR": es[m]["S_mirror_same_subset"]["mean"], "S_orig_full": es[m]["S_original_full"]["mean"], "n_LR": es[m]["n_same_subset"]}
                      for m in M7 if m in es}
    lines.append("CONS-f Fig. 4 L/R subset: " + "; ".join(f"{m} S_orig L/R {v['S_orig_LR']} (full {v['S_orig_full']}), S_gray L/R {v['S_gray_LR']}, S_mirror {v['S_mirror_LR']}, n {v['n_LR']}" for m, v in res["fig4_lr"].items()))
    # (g) probe controls
    pc = json.load(open(B / "probe_controls.json"))
    ctrl = {}
    for m, v in pc.items():
        c = {"filler": v["filler"].get("r") or v["filler"], "layer0_r": v["layer0_r"], "text": v["baselines"].get("text"), "cue": v["baselines"].get("cue")}
        ctrl[m] = {}
        for k, x in c.items():
            if isinstance(x, dict) and "bal" not in x:
                x = x.get("r") or next((y for y in x.values() if isinstance(y, dict) and "bal" in y), None)
            if isinstance(x, dict) and "bal" in x:
                ctrl[m][k] = {"bal": x["bal"], "recall_DCm_wrong": x.get("recall_DCm_wrong")}
    bal = [x["bal"] for v in ctrl.values() for x in v.values()]
    rec = [x["recall_DCm_wrong"] for v in ctrl.values() for x in v.values() if x.get("recall_DCm_wrong") is not None]
    res["probe_controls"] = {"per_model": ctrl, "bal_range": [min(bal), max(bal)], "recall_range": [min(rec), max(rec)] if rec else None}
    lines.append(f"CONS-g probe controls (filler / layer-0 r / TF-IDF text / cue covariates): balanced acc {res['probe_controls']['bal_range']}, wrong-claim recall {res['probe_controls']['recall_range']} | {ctrl}")
    # (h) App. M CV-Bench 3D Depth d' (Tab. S40)
    nat = hb["natural_sdt"]
    cvd = {m: {"diff": v["cvb_depth"]["diff"], "ci": v["cvb_depth"]["diff_ci"]} for m, v in nat.items() if "cvb_depth" in v}
    res["appM"] = {"per_model": cvd, "point_lt0": [m for m, v in cvd.items() if v["diff"] < 0], "ci_lt0": [m for m, v in cvd.items() if v["ci"][1] < 0],
                   "point_le_m025": [m for m, v in cvd.items() if v["diff"] <= -0.25]}
    lines.append(f"CONS-h CV-Bench 3D Depth delta d' (Tab. S40): point <0 in {len(res['appM']['point_lt0'])}/8 {res['appM']['point_lt0']}, <= -0.25 in {len(res['appM']['point_le_m025'])}, CI<0 in {len(res['appM']['ci_lt0'])}/8 {res['appM']['ci_lt0']} | {cvd}")
    # (i) first claim = j*: full Syn. Depth commit set vs Tab. S30 sample
    from e20_commit_judge import commit_idx, first_claim_idx
    cj = json.load(open(B / "commit_judge.json"))["rows"]
    fc = {}
    for m in M7:
        G = BB.groups_from({}, L, O, m)
        recs = [r for r in rj(O / "e2b" / f"{m}.jsonl") if r["task"] == "syn_depth" and G.get(r["item_id"], {}).get("group") == "D+C-" and r["sentences"]]
        com = [r for r in recs if commit_idx(r) is not None]
        same = [first_claim_idx(r["sentences"]) == commit_idx(r) for r in com]
        fc[m] = {"n_committed": len(com), "first_claim_is_jstar_full": r1(100 * np.mean(same)) if com else None, "sample_tabS30": cj[f"{m}|D+C-"]["is_first_claim"]}
    res["first_claim_jstar"] = {"per_model": fc, "range_full": [min(v["first_claim_is_jstar_full"] for v in fc.values()), max(v["first_claim_is_jstar_full"] for v in fc.values())],
                                "range_sample": [min(v["sample_tabS30"] for v in fc.values()), max(v["sample_tabS30"] for v in fc.values())],
                                "pooled_sample": cj["pooled|D+C-"]["is_first_claim"]}
    lines.append(f"CONS-i first claim = j*: full Syn. Depth committed D+C- {res['first_claim_jstar']['range_full']} vs Tab. S30 sample {res['first_claim_jstar']['range_sample']} (pooled {cj['pooled|D+C-']['is_first_claim']}) | {fc}")
    # (k) twins pair-matched counts
    tw = json.load(open(B / "syn_pair_sdt_mh.json"))["twins"]
    cls = {m: ("fall" if tw[m]["delta_pooled_ci"][1] < 0 else "rise" if tw[m]["delta_pooled_ci"][0] > 0 else "flat") for m in M8 if m in tw}
    res["twins_counts"] = {"per_model": {m: {"delta": r1(tw[m]["delta_pooled"], 2), "ci": [r1(x, 2) for x in tw[m]["delta_pooled_ci"]], "class": cls[m]} for m in cls},
                           "non_thinking7": dict(Counter(cls[m] for m in M7)), "all8": dict(Counter(cls.values()))}
    lines.append(f"CONS-k twins d'_pair (Tab. S13): 7 non-Thinking {res['twins_counts']['non_thinking7']}, all 8 {res['twins_counts']['all8']} | {res['twins_counts']['per_model']}")
    # (l) other review items checked from stored files
    oth = {}
    e9 = json.load(open(B / "e9x_summary.json"))
    q = e9.get("qwen3.8-max-0902", {}).get("blink_depth") or {}
    oth["closed_qwen38max_blink"] = q.get("cot_minus_direct")
    st = json.load(open(B / "state.json"))
    oth["state_keys"] = list(st.keys())[:5]
    res["other"] = oth
    return res


# ----------------------------------------------------------------------------- 6. description control

def desc(lines: list) -> dict:
    dp = json.load(open(B / "dprime_ci.json"))
    cu = json.load(open(B / "cues.json"))
    ev = json.load(open(B / "e6_eval.json"))
    res = {"twins": {}, "heldout": {}, "cues": "no question-blind description arm was run on the cue-factorial set (e10 cue conditions: direct, cot, state, "
                                                "filler_S+turn, filler_R+turn); the structured JSON state arm is question-aware and is the closest analogue"}
    for m in M8:
        r = dp.get(m)
        if not isinstance(r, dict) or "dprime_desc" not in r:
            continue
        dec = cu.get(m, {}).get("twins", {}).get("decomposition", {})
        res["twins"][m] = {"d_direct": r["dprime_direct"], "d_cot": r["dprime_cot"], "d_desc": r["dprime_desc"],
                           "cot_minus_direct": [r["dprime_cot_minus_direct"], r["diff_ci"]], "desc_minus_direct": [r["dprime_desc_minus_direct"], r["desc_diff_ci"]],
                           "occl_component": {k: dec.get("occlusion", {}).get(k) for k in ("cot-direct", "desc-direct")},
                           "vert_component": {k: dec.get("vertical", {}).get(k) for k in ("cot-direct", "desc-direct")}}
        x = res["twins"][m]
        lines.append(f"DESC twins {m:16s} pooled d' Direct {x['d_direct']} CoT {x['d_cot']} Desc {x['d_desc']} | CoT-Dir {x['cot_minus_direct']} Desc-Dir {x['desc_minus_direct']} "
                     f"| occlusion component CoT-Dir {x['occl_component']['cot-direct']} Desc-Dir {x['occl_component']['desc-direct']}")
    for m, v in ev.items():
        p = v.get("pooled") if isinstance(v, dict) else None
        if p and "desc_minus_direct" in p:
            res["heldout"][m] = {"desc_minus_direct": p["desc_minus_direct"], "cot_minus_direct": p.get("cot_minus_direct")}
    lines.append("DESC held-out 767 Desc-Direct vs CoT-Direct: " + "; ".join(f"{m} {v['desc_minus_direct']} vs {v['cot_minus_direct']}" for m, v in res["heldout"].items()))
    lines.append("DESC cues: " + res["cues"])
    return res


# ----------------------------------------------------------------------------- 5. closed models

def closed(lines: list) -> dict:
    rows = defaultdict(lambda: defaultdict(list))
    for f in sorted(glob.glob(str(O / "e9x" / "*.jsonl")) + glob.glob(str(O / "e9h" / "*.jsonl")) + glob.glob(str(O / "e9c_cues" / "*.jsonl"))):
        if "limit" in f:
            continue
        src = Path(f).parent.name
        for r in rj(Path(f)):
            if "error" in r:
                continue
            t = r["task"]
            s = "hard" if src == "e9h" else "cues" if src == "e9c_cues" else ("twins" if t.startswith("occ_") else t)
            k = (r["model"], s, r["condition"])
            rows[k]["eff"].append(r.get("effort"))
            rows[k]["rt"].append(r.get("reasoning_tokens") or 0)
            rows[k]["ct"].append(r.get("completion_tokens") or 0)
            rows[k]["src"].append(src)
    out = {}
    for (m, s, c), v in sorted(rows.items()):
        rt = np.array(v["rt"])
        eff = Counter(v["eff"])
        stated = {"direct": "high", "cot": "high"}.get(c)
        if c == "direct_nr":
            stated = "none" if m.startswith(("deepseek", "qwen")) else "minimal" if m.startswith("claude") else "low" if m.startswith("gpt") else "?"
        if s == "cues":
            stated = "none"
        out.setdefault(m, {}).setdefault(s, {})[c] = {
            "n": int(len(rt)), "effort_recorded": {str(k): n for k, n in eff.items()},
            "effort_used": (list(eff)[0] if len(eff) == 1 and None not in eff else f"{stated} (script default / launch arg; field not recorded)" if None in eff else str(dict(eff))),
            "frac_with_reasoning_tokens": r1(100 * float((rt > 0).mean())), "mean_reasoning_tokens": r1(float(rt.mean())), "source": Counter(v["src"]).most_common(1)[0][0]}
    res = {"protocol": {
        "direct_prompt": "user turn: question + options + \"Answer with the option's letter only.\"; answer parsed from the returned text",
        "cot_prompt": "user turn: question + options + \"Let's think step by step: first describe what you see that matters for the question, then reason, "
                      "and end with 'Answer: (X)' where X is the option letter.\"; answer parsed from 'Answer: (X)' in the returned text",
        "open_model_cot": "open models: CoT is an assistant-turn prefix 'Let's think step by step.' continued greedily (<= 512 tokens), answer read as the forced "
                          "next-token distribution after 'Answer: (' (same turn); Direct = forced readout right after the question",
        "effort_mechanism": "api_client.chat sends reasoning_effort=<effort> (Gemini since 2026-11-13: thinking_config.thinking_level); temperature not set "
                            "(provider default); max_completion_tokens 16,384 (Direct/CoT) or 512 (direct_nr)",
        "e9x_default_effort": "e9x_api.py --effort default 'high' for both Direct and CoT; research_log Exp-9x: 'effort=high'; records written before the "
                              "'effort' field was added carry None",
        "direct_nr": "only Direct was re-run without reasoning ('direct_nr': none for DeepSeek / Qwen3.8-Max; Claude 'minimal'; GPT 'low'); no CoT arm at "
                     "reasoning none/low exists except DeepSeek-V4.1-Flash on the cue set (both arms 'none')"},
        "per_model": out}
    for m, v in out.items():
        for s, cc in v.items():
            lines.append(f"CLOSED {m:20s} {s:14s} " + " | ".join(f"{c}: n={x['n']} effort {x['effort_used']} reasoning>0 {x['frac_with_reasoning_tokens']}% mean {x['mean_reasoning_tokens']}" for c, x in cc.items()))
    return res


# ----------------------------------------------------------------------------- 7. readout and drift

def readout(lines: list) -> dict:
    ch = json.load(open(B / "churn.json"))
    res = {"engines": {"V0_exact_full_vocab": ["qwen3b", "qwen7b", "internvl3_8b", "phi35v"],
                       "V1_top20": ["molmo7b", "qwen3vl_8b", "qwen3vl_8b_think", "internvl35_8b"]},
           "usage": {
               "Tab. 1 / Tab. S3 / Fig. 1c / Tab. S4 left / Tab. 2 main pool / Tab. S11 / S24-S27": "e1x exact readout for V0 models (legacy greedy rationales re-read), top-20 for V1 models",
               "Sec. 5 D+-C+- cells (Fig. 2a,b, Tab. S28-S32 commit / rescue / judge samples, steering, claim edits)": "original (legacy) runs' top-20 letter scores for all models (results/v2 e1, e4)",
               "Tab. S38 (state), Tab. S62 (probe x-dist), Tab. S16/S17 cue set, twins via e10": "e10 re-run: CoT regenerated greedily in a new run, same engine as Direct",
               "Tab. S35/S36 (gate), Tab. S37 (AFE)": "e22 / e25 re-run: plain CoT = one-shot greedy generation in that run"}}
    diff = {}
    for m in BB.MODELS:
        t, l_ = ch[m]["syn_depth"]["tab1"], ch[m]["syn_depth"]["legacy"]
        main = BB.load_main(O, L, m)
        G = BB.groups_from({}, L, O, m)
        ks = [k for k in main.get("direct", {}) if k in G and k in main.get("cot", {})]
        grp_t = {k: ("D+" if main["direct"][k][0] else "D-") + ("C+" if main["cot"][k][0] else "C-") for k in ks}
        nd = sum(grp_t[k] != G[k]["group"] for k in ks)
        diff[m] = {"items_cell_differs_2850": nd, "syn_tab1": [t["D+C-"], t["D-C+"]], "syn_legacy": [l_["D+C-"], l_["D-C+"]]}
    res["cells"] = diff
    lines.append("READOUT D+-C+- cells differing between Tab. 1 readout and Sec. 5 (legacy top-20) cells: " + "; ".join(f"{m} {v['items_cell_differs_2850']}" for m, v in diff.items()))
    # greedy re-run drift: e1x / legacy CoT vs e10 CoT on Synthetic Depth
    drift = {}
    for m in M7:
        main = BB.load_main(O, L, m)
        leg = json.load(open(L / "e1" / f"{m}.json")) if (L / "e1" / f"{m}.json").exists() else json.load(open(O / "e4" / f"{m}.json"))
        legR = {it["uid"]: it.get("cot") for it in leg["items"] if it["task"] == "syn_depth"}
        e10 = {r["item_id"]: r for r in rj(O / "e10" / f"{m}.jsonl") if r["task"] == "syn_depth" and r["condition"] in ("cot",)}
        e10d = {r["item_id"]: r for r in rj(O / "e10" / f"{m}.jsonl") if r["task"] == "syn_depth" and r["condition"] == "direct"}
        ks = [k for k in e10 if k in main["cot"]]
        a_main = np.mean([main["cot"][k][0] for k in ks])
        a_e10 = np.mean([pe(e10[k]) == e10[k]["gold"] for k in ks])
        same_text = np.mean([(legR.get(k) or "").strip() == (e10[k].get("text") or "").strip() for k in ks])
        same_pred = np.mean([main["cot"][k][4] == pe(e10[k]) for k in ks])
        dsame = np.mean([main["direct"][k][4] == pe(e10d[k]) for k in ks if k in e10d])
        drift[m] = {"n": len(ks), "cot_acc_tab1": r1(100 * a_main), "cot_acc_e10": r1(100 * a_e10), "delta": r1(100 * (a_e10 - a_main)),
                    "identical_rationale_text": r1(100 * same_text), "same_cot_pred": r1(100 * same_pred), "same_direct_pred": r1(100 * dsame)}
    res["drift_syn_depth"] = drift
    lines.append("DRIFT Syn. Depth CoT acc Tab. 1 (legacy rationale) vs e10 re-run (Tab. S38/S62): " + "; ".join(
        f"{m} {v['cot_acc_tab1']} vs {v['cot_acc_e10']} ({v['delta']:+.1f}; identical rationale text {v['identical_rationale_text']}%, same CoT answer {v['same_cot_pred']}%, same Direct answer {v['same_direct_pred']}%)" for m, v in drift.items()))
    res["disclosure"] = ("All open-model runs decode greedily (T = 0, seed 0) with vLLM; batch composition and engine version change floating-point "
                         "reductions, so a rationale regenerated in a later run can differ from the stored one: Tab. 1 re-reads the original "
                         "rationales, while the state, probe, gate and answer-first experiments regenerate them, which moves Synthetic Depth CoT "
                         "accuracy by up to 3.6 points (Direct answers are unchanged); the D+/-C+/- cells of Sec. 5 use the original runs' top-20 "
                         "readout and differ from Tab. 1 on 22-83 near-tie items in the four exact-readout models.")
    return res


def fixrefresh(lines: list) -> dict:
    """Rewrite numbers.json fix_main_pool / fix_heldout at 4 dp and bias_sensitivity.json from unrounded estimates."""
    nb = json.load(open(B / "numbers.json"))
    ref = json.load(open(B / "bak_pre_e27" / "numbers.json")) if (B / "bak_pre_e27" / "numbers.json").exists() else nb
    chg = {}
    for m in BB.MODELS:
        if m not in nb or "fix_main_pool" not in nb[m]:
            continue
        main = BB.load_main(O, L, m)
        for key, ho in (("fix_main_pool", False), ("fix_heldout", True)):
            new = BB.sym_and_gate(O, main, m, heldout=ho)
            old = ref[m][key]
            for k, v in new.items():
                if isinstance(v, dict) and "mean" in v and k in old and isinstance(old[k], dict) and "mean" in old[k]:
                    if abs(round(v["mean"], 2) - old[k]["mean"]) > 0.0051:
                        chg.setdefault(m, {})[f"{key}:{k}"] = [old[k]["mean"], v["mean"]]
                    if f"{old[k]['mean']:+.1f}" != f"{v['mean']:+.1f}":
                        chg.setdefault(m, {})[f"{key}:{k}:printed"] = [f"{old[k]['mean']:+.1f}", f"{v['mean']:+.1f}"]
            nb[m][key] = new
    json.dump(nb, open(B / "numbers.json", "w"), indent=1)
    from rigor_e17 import bias_sensitivity
    bs = bias_sensitivity()
    json.dump(bs, open(B / "bias_sensitivity.json", "w"), indent=1)
    lines.append(f"FIXREFRESH numbers.json fix_main_pool / fix_heldout rewritten at 4 dp (value changes beyond 2-dp rounding: "
                 f"{ {m: {k: v for k, v in d.items() if not k.endswith(':printed')} for m, d in chg.items()} }); printed-digit changes: "
                 f"{ {m: {k: v for k, v in d.items() if k.endswith(':printed')} for m, d in chg.items()} }; bias_sensitivity.json rewritten from unrounded estimates")
    return {"changes": chg}


def facts(lines: list, E: dict) -> dict:
    """Human-readable fact list (value + source) and the consistency-fix table, built from e27_reframe.json."""
    H, C, V, K, D, R = E["heur"], E["comp"], E["ovl"], E["cons"], E["desc"], E["readout"]
    cm = C["models"]
    F = []

    def add(tag, text, src):
        F.append({"tag": tag, "fact": text, "source": src})

    f2 = lambda d: f"{d['d']:+.2f} [{d['ci'][0]:+.2f}, {d['ci'][1]:+.2f}]"
    sd, tw, cu, hd = H["syn_depth"], H["twins"], H["cues"], H["hard"]
    add("H1", f"Synthetic Depth (500 images): 'larger visible area is in front' is right on {sd['all']['size_visible']}% {sd['all']['size_visible_ci']} "
              f"(wrong on {sd['all']['n_size_wrong']} images); on the 431 overlapping images {sd['overlap431']['size_visible']}%; 'lower is closer' {sd['all']['lower']}% "
              f"{sd['all']['lower_ci']}; both agree on {sd['all']['agree_coverage']}% of images (accuracy {sd['all']['agree_acc']}%); size-else-lower {sd['all']['size_else_lower']}%; "
              f"5-fold logistic on log visible ratio + vertical offset {sd['all']['logistic_cv_acc']}%. Median visible ratio {sd['median_visible_ratio']}x. "
              "Rules answer by relation, so accuracy is identical under both object orders.",
        "e27_reframe.json heur.syn_depth (syn_depth_geometry.json px/cy; front from e2a cfq question+gold; category from rigor_e17i)")
    add("H2", f"Original twins (400 images): drawn-size rule {tw['size_drawn']}% (front never drawn larger; mean drawn ratio {tw['mean_drawn_ratio_front_over_back']}), "
              f"visible-size rule {tw['size_visible']}% {tw['size_visible_ci']} (front fully visible, back partly hidden; mean visible ratio {tw['mean_visible_ratio_front_over_back']}), "
              f"lower rule {tw['lower']}%, visible-size-else-lower {tw['visible_and_lower']['size_else_lower']}% (logistic {tw['visible_and_lower']['logistic_cv_acc']}%).",
        "e27_reframe.json heur.twins (e2d meta: area_front_px, area_back_px, back_hidden_frac, front_above)")
    add("H3", f"Cue-factorial set (1,600 images): drawn-size {cu['size_drawn']}%, lower {cu['lower']}% (balanced by design), visible-size {cu['size_visible']}% {cu['size_visible_ci']} "
              f"(occlusion hides part of the back shape, so the front is visibly larger even when drawn smaller), visible-size-else-lower {cu['visible_and_lower']['size_else_lower']}% "
              f"(logistic {cu['visible_and_lower']['logistic_cv_acc']}%).", "e27_reframe.json heur.cues (e10 *_cues meta)")
    add("H4", f"Hard occlusion set (288 images): drawn-size {hd['size_drawn']}%, visible-size {hd['size_visible']}% (i.e. 'smaller visible area is in front' {hd['smaller_visible_in_front']}%), "
              f"lower {hd['lower']}%; a fitted logistic on visible ratio reaches {hd['visible_and_lower']['logistic_cv_acc']}% by learning the reversed size rule. By level visible-size {hd['by_level']}.",
        "e27_reframe.json heur.hard (e2h meta)")
    for m in M7:
        x = H["syn_model"][m]["direct"]
        o = H["occ_model"][m]
        add(f"H5-{m}", f"{NAMES[m]}: Synthetic Depth Direct two-order accuracy {fmt(x['acc_two_order'], False)}, agreement with the size rule {fmt(x['agree_size_rule'], False)} "
                       f"(CoT {fmt(H['syn_model'][m]['cot']['agree_size_rule'], False)}); {x['share_correct_also_size_right']}% of Direct-correct answers are also size-rule answers; "
                       f"accuracy on the 7 size-rule-wrong images (both orders) {x['acc_where_size_wrong']['k']}/{x['acc_where_size_wrong']['n']}. "
                       f"Direct agreement with 'lower is closer': twins {fmt(o['twins']['direct']['agree_lower'], False)}, cue set {fmt(o['cues']['direct']['agree_lower'], False)} "
                       f"(CoT {fmt(o['cues']['cot']['agree_lower'], False)}); with the visible-size rule on the cue set {fmt(o['cues']['direct']['agree_visible_larger'], False)}.",
            "e27_reframe.json heur.syn_model / heur.occ_model (Tab. 1 readout + e2a cfq; e2d; e10 cues)")
    add("H6", "On Synthetic Depth size and occlusion are collinear (size rule wrong on 7/500 images), so the share of Direct accuracy 'explained' by size is not "
              "identifiable there: Direct agreement with the size rule equals Direct accuracy within 0.8 points in every model. The size-controlled sets separate them.",
        "e27_reframe.json heur.syn_model")
    add("C0", C["rule"], "pre-stated in rigor_e27.comp before reading the per-model table")
    add("C1", f"Competent: {[NAMES[m] for m in C['competent']]}; not competent: {[NAMES[m] for m in C['not_competent']]}; not evaluable (no cue-set run): "
              f"{[NAMES[m] for m in C['not_evaluable']]}. InternVL3-8B fails (i) only: cue Direct {fmt(cm['internvl3_8b']['cue_direct_acc'], False)}, occlusion weight "
              f"{cm['internvl3_8b']['cue_glm_occl_direct']}. Thinking: twins Direct {fmt(cm['qwen3vl_8b_think']['twin_direct_acc'], False)}, d'_pair {cm['qwen3vl_8b_think']['twin_dpair_direct']} "
              "(BW = 0 cap, lower bound) meets the secondary twin criteria.", "e27_reframe.json comp.models (e10 cues, glm_decomp.json, syn_pair_sdt_mh.json twins, e2d)")
    for m in C["competent"]:
        r = cm[m]
        ov = V["models"][m]["overlap"]
        add(f"C2-{m}", f"{NAMES[m]}: cue set Direct {fmt(r['cue_direct_acc'], False)}, CoT-Direct {fmt(r['cue_cot_minus_direct'])}; GLM occlusion {r['cue_glm_occl_direct']['d']} -> "
                       f"change {f2(r['cue_glm_occl_delta'])}, vertical {r['cue_glm_vert_direct']['d']} -> change {f2(r['cue_glm_vert_delta'])}; AUC change {r['cue_auc_delta']}. "
                       f"Twins Direct {fmt(r['twin_direct_acc'], False)}, CoT-Direct {fmt(r['twin_cot_minus_direct'])}; d'_pair {r['twin_dpair_direct']['d']} change {f2(r['twin_dpair_delta'])} "
                       f"(pooled d' change {f2(r['twin_pooled_d_delta'])}). Synthetic Depth accuracy change {r['syn_acc_delta']['d']:+.1f} {r['syn_acc_delta']['ci']}, d'_pair {r['syn_dpair_direct']} "
                       f"change {f2(r['syn_dpair_delta'])}; on 431 overlapping {ov['dpair_delta']:+.2f} {ov['dpair_delta_ci']}. Hard set Direct {r['hard']['direct_acc']}% "
                       f"{r['hard']['direct_acc_ci']}, d' {r['hard']['direct_d']} {r['hard']['direct_d_ci']}, CoT d' change {r['hard']['d_delta']}.",
            "e27_reframe.json comp.models + ovl.models (Tab. S12-S17, S26, e2h/summary.json)")
        add(f"C3-{m}", f"{NAMES[m]}: commit audit (Tab. S30 sample, Synthetic Depth) D+C- {r['commit_D+C-']['k']}/{r['commit_D+C-']['n_judgeable']} judgeable contradict the image "
                       f"({r['commit_D+C-']['pct']}%); relational D+C+ comparator {r['DCp_rel_commitpos_relcomp']['all']}; D+C- on images Direct gets right under both orders "
                       f"{r['DCm_commit']['direct_pair_correct']}. Claim edit (same-turn regeneration, Tab. S33): D+C- correct claim {r['claim_edit_all']['DCm_correct']}% vs neutral "
                       f"{r['claim_edit_all']['DCm_neutral']}% (diff {r['claim_edit_all']['DCm_correct_minus_neutral']}); D+C+ wrong claim {r['claim_edit_all']['DCp_wrong']}% vs neutral "
                       f"{r['claim_edit_all']['DCp_neutral']}%; twins only: D+C- correct {r['claim_edit_occ']['DCm_correct']}% vs {r['claim_edit_occ']['DCm_neutral']}% (n={r['claim_edit_occ']['n_DCm']}), "
                       f"D+C+ wrong {r['claim_edit_occ']['DCp_wrong']}% vs {r['claim_edit_occ']['DCp_neutral']}%. Loss locus (Tab. S19) cues {r['locus_cues']}; twins {r['locus_twins']}.",
            "commit_judge.json, syn_pair_sdt.json seeing, claim_edit.json, e17k.json locus")
    pa = C["pooled_audit"]
    add("C4", f"Pooled over the two competent models: commit sentences contradict the image in {pa['competent']['DCm']['k']}/{pa['competent']['DCm']['n']} = {pa['competent']['DCm']['pct']}% "
              f"{pa['competent']['DCm']['wilson']} vs {pa['competent']['DCp_rel']['k']}/{pa['competent']['DCp_rel']['n']} = {pa['competent']['DCp_rel']['pct']}% {pa['competent']['DCp_rel']['wilson']} "
              f"for matched relational D+C+ sentences; restricted to images Direct gets right under both orders {pa['competent']['DCm_direct_pair_correct']['k']}/"
              f"{pa['competent']['DCm_direct_pair_correct']['n']} = {pa['competent']['DCm_direct_pair_correct']['pct']}%. Other five models {pa['rest']['DCm']['pct']}% vs {pa['rest']['DCp_rel']['pct']}%.",
        "e27_reframe.json comp.pooled_audit (e20 judge records via syn_pair_sdt.json seeing)")
    s = V["summary"]
    add("O1", f"431 overlapping items: single-order CoT-Direct accuracy drops range {s['single_order_acc_delta_range_431']} (all 500: {s['single_order_acc_delta_range_all']}); "
              f"Qwen2.5-VL-3B -27.4 -> {V['models']['qwen3b']['single_order']['overlap']['acc_diff']}; pooled d' falls CI<0 in {len(s['dpooled_fall_ci_lt0_431'])}/7 "
              f"({s['dpooled_range_falls_431']}); d'_pair falls CI<0 in {len(s['dpair_fall_ci_lt0_431'])}/7 {[NAMES[m] for m in s['dpair_fall_ci_lt0_431']]} "
              f"(all 500: {len(s['dpair_fall_ci_lt0_all'])}/7), range {s['dpair_range_falls_431']}; Qwen2.5-VL-7B {V['models']['qwen7b']['overlap']['dpair_delta']:+.2f} "
              f"{V['models']['qwen7b']['overlap']['dpair_delta_ci']}, Phi-3.5-V {V['models']['phi35v']['overlap']['dpair_delta']:+.2f} {V['models']['phi35v']['overlap']['dpair_delta_ci']}; "
              f"InternVL3.5-8B still rises. Commit audit pooled {s['commit_pooled_all']} -> {s['commit_pooled_431']}; relational comparator {s['relcomp_pooled_all']} -> {s['relcomp_pooled_431']}.",
        "e27_reframe.json ovl (syn_overlap.json categories; rigor_e17p.stats; e20 judge records)")
    fx = []
    sy = K["sym_direct"]
    fx.append({"id": "a", "item": "Tab. 2 / Tab. S46 depth-valid rows Qwen3-VL-8B = InternVL3.5-8B", "old": "-4.6 [-9.6, +0.3]; Sym -1.2 [-4.0, +1.5] (both)",
               "new": f"values unchanged, not a copy: per-item CoT-Direct deltas differ on {K['depth_valid_dup']['items_with_different_cot_delta']} items and Sym-Direct on "
                      f"{K['depth_valid_dup']['items_with_different_sym_delta']}, both nets equal (-15/324, -4/324); unrounded CoT-Direct upper bounds +0.31 vs +0.32",
               "source": "heldout_breakdown.model_rows (e6x, e5c)", "changed": "footnote in Tab. 2 (make_tab3.py) and note row in Tab. S46 (make_supp_tables.py)"})
    fx.append({"id": "b", "item": "Tab. 2 Sym-Direct", "old": "Qwen2.5-VL-3B +2.0, Qwen3-VL-8B +3.0, InternVL3-8B +0.3",
               "new": f"Qwen2.5-VL-3B +2.0 (correct: {sy['qwen3b']['sym_direct_minus_direct']}; Sym = {sy['qwen3b']['sym_direct']}, the earlier +1.9 used the rounded 82.2), "
                      f"Qwen3-VL-8B +2.9 ({sy['qwen3vl_8b']['sym_direct_minus_direct']}), InternVL3-8B +0.4 ({sy['internvl3_8b']['sym_direct_minus_direct']}); the same double rounding also fixed "
                      "InternVL3-8B Direct 89.7 -> 89.6 and CoT consistency 85.0/87.0/86.3 -> 85.1/87.1/86.4 (Qwen2.5-VL-7B / InternVL3-8B / InternVL3.5-8B)",
               "source": "build_bundle.sym_and_gate at 4 dp; numbers.json had 2-dp means that make_tab3 re-rounded", "changed": "numbers.json fix_main_pool/fix_heldout (4 dp), Tab. 2 regenerated"})
    rs = K["reshow"]
    fx.append({"id": "c", "item": "Tab. 1 re-show +1.5 vs text +1.4", "old": "Tab. 1 marginals implied +1.5 (Qwen2.5-VL-3B, InternVL3-8B)",
               "new": f"unrounded max post-CoT re-supply gain +{rs['post_cot_resupply_range'][1]:.2f} (41/2,850 items) in both models; range {rs['post_cot_resupply_range']} "
                      f"(with CoT+turn {rs['incl_cot_turn_range'][0]:+.2f}); Tab. 1 also double-rounded 4 cells: InternVL3-8B Direct 89.7 -> 89.6, InternVL3-8B CoT+restate 85.0 -> 84.9, "
                      "Qwen2.5-VL-3B CoT+both 73.0 -> 72.9, Qwen3-VL-8B CoT+reimage 87.7 -> 87.6",
               "source": "build_bundle.load_main (e1x exact readout / legacy)", "changed": "tables/tab2_behavioral_and_tasks.tex (hand-maintained, 4 cells)"})
    t24 = K["tab_s24"]
    fx.append({"id": "d", "item": "Tab. S24 'Loss: sens./crit.'", "old": "labelled accuracy loss; e.g. Qwen2.5-VL-3B 29.2/-1.2",
               "new": "relabelled 'Bal.-acc. loss: sens./crit.' (sums to the balanced-accuracy change by construction); values from unrounded estimates: "
                      + "; ".join(f"{NAMES[m]} {v['sens_new']}/{v['crit_new']} (bal {v['balanced_acc_loss']}, acc {v['acc_loss']})" for m, v in t24.items())
                      + ". Appendix caption still says 'accuracy loss' (editor).",
               "source": "rigor_e17.bias_sensitivity (now unrounded differences)", "changed": "bias_sensitivity.json, Tab. S24 header+values (also Delta c last digit), Tab. S25 'single' column (now read from bias_sensitivity.json)"})
    f2b = K["fig2b"]
    fx.append({"id": "e", "item": "Fig. 2b normalization", "old": "bars/labels 63.9-87.0 (all D+C-) vs text 69.3-88.5 (committed)",
               "new": f"both correct, different denominators: of all D+C- {f2b['range_all_DCm']}, of committed D+C- {f2b['range_committed']}; caption should say 'share of all D+C- items "
                      "(incl. no commit)', text should cite 'of committed items' with Tab. S28", "source": "commit_baseline.json, numbers.json error_onset", "changed": "none (caption/text for editor)"})
    lr = K["fig4_lr"]
    fx.append({"id": "f", "item": "Fig. 4 mirrored S vs full-pool S_orig", "old": "S_orig 68.2/61.1/36.3 (full pool) next to S_mirror on L/R items",
               "new": f"S_orig on the same L/R items: InternVL3.5-8B {lr['internvl35_8b']['S_orig_LR']}, InternVL3-8B {lr['internvl3_8b']['S_orig_LR']}, Qwen2.5-VL-7B {lr['qwen7b']['S_orig_LR']} "
                      f"(n {lr['internvl35_8b']['n_LR']}/{lr['internvl3_8b']['n_LR']}/{lr['qwen7b']['n_LR']}); now drawn as a '|' marker on each Mirrored row",
               "source": "e3x_same_subset.json", "changed": "make_figures_v3.build_fig3_swaps; fig3_swaps.pdf/png regenerated (same 3.28x3.58 in)"})
    pc = K["probe_controls"]
    fx.append({"id": "g", "item": "probe control range 0.53-0.73", "old": "stated 'under the same metrics'",
               "new": f"balanced accuracy {pc['bal_range']}; wrong-claim recall {pc['recall_range']}", "source": "probe_controls.json (filler r, layer0_r, baselines.text, baselines.cue)", "changed": "none (text)"})
    am = K["appM"]
    fx.append({"id": "h", "item": "App. M 'five of eight' (CV-Bench 3D Depth d')", "old": "lowers d' by 0.25-1.21 in five of eight",
               "new": f"point estimate <= -0.24 in five (Tab. S40: -0.25 to -1.21), CI<0 in four ({[NAMES[m] for m in am['ci_lt0']]}); Qwen2.5-VL-7B -0.25 [-0.52, +0.02] n.s.",
               "source": "Tab. S40 (e17l.json cvb), heldout_breakdown.json natural_sdt", "changed": "none (text)"})
    fc = K["first_claim_jstar"]
    fx.append({"id": "i", "item": "Sec. 5 'first claim is j* in 30-89%'", "old": "30-89%",
               "new": f"Tab. S30 audited sample {fc['range_sample']} (pooled {fc['pooled_sample']}); recomputed on all committed Synthetic Depth D+C- items with the same parser "
                      f"{fc['range_full']} (7 models); the 89% upper end is not reproduced by the current parser", "source": "e2b trajectories + e20_commit_judge.first_claim_idx/commit_idx; commit_judge.json",
               "changed": "none (text)"})
    fx.append({"id": "j", "item": "Qwen2.5-VL-3B single-phrasing delta d'", "old": "-1.93 (Tab. S11) vs -1.94 (Tab. S24)",
               "new": "-1.93 in both (S24 took the difference of rounded d'); Tab. S11 should be labelled single-phrasing (main text uses the polarity-pooled -1.77)",
               "source": "bias_sensitivity.json (unrounded), syn_overlap.json", "changed": "Tab. S24"})
    tc = K["twins_counts"]
    fx.append({"id": "k", "item": "twin d'_pair counts", "old": "abstract 'falls in three, rises in three'",
               "new": f"7 non-Thinking: fall {tc['non_thinking7']['fall']} (Phi-3.5-V, Molmo-7B-D), flat {tc['non_thinking7']['flat']} (Qwen2.5-VL-3B, Qwen3-VL-8B), rise {tc['non_thinking7']['rise']} "
                      f"(Qwen2.5-VL-7B, InternVL3-8B, InternVL3.5-8B); with Thinking: fall {tc['all8']['fall']}, flat {tc['all8']['flat']}, rise {tc['all8']['rise']}",
               "source": "syn_pair_sdt_mh.json twins (Tab. S13)", "changed": "none (text)"})
    dr = R["drift_syn_depth"]
    fx.append({"id": "l1", "item": "greedy re-run drift (Tab. S3 vs S38/S62; S26 vs S35)", "old": "Qwen2.5-VL-7B 70.0 vs 66.4; InternVL3-8B 66.0 vs 63.4",
               "new": "; ".join(f"{NAMES[m]} {v['cot_acc_tab1']} vs {v['cot_acc_e10']} (identical text {v['identical_rationale_text']}%, same CoT answer {v['same_cot_pred']}%)" for m, v in dr.items())
                      + ". Cause: Tab. 1 re-reads the original (legacy) greedy rationales; e10/e22/e25 regenerate them in a later vLLM run; Direct answers identical (100%).",
               "source": "e1x / results/v2 e1 vs e10 *.jsonl", "changed": "none (disclosure sentence)"})
    fx.append({"id": "l2", "item": "Tab. 1 / Sec. 5 D+-C+- cells", "old": "83/27/22/40 items differ (Tab. S4)",
               "new": f"confirmed {[R['cells'][m]['items_cell_differs_2850'] for m in ('qwen3b', 'qwen7b', 'internvl3_8b', 'phi35v')]} (V0 models), 0 for V1 models",
               "source": "churn.json, build_bundle.groups_from", "changed": "none"})
    fx.append({"id": "l3", "item": "Qwen3.8-Max BLINK +4.0 vs 90.3 -> 94.4", "old": "+4.0", "new": "+4.0 correct (117-112 = 5/124 = 4.03); 94.4 - 90.3 = 4.1 is rounded marginals",
               "source": "e9x_summary.json", "changed": "none"})
    fx.append({"id": "l4", "item": "Phi-3.5-V BLINK CoT empty", "old": "-", "new": "100% empty rationales on BLINK Relative Depth (format effect; exclude from CoT means or mark)",
               "source": "heldout_breakdown.json blink_depth.format.empty_cot_share", "changed": "none"})
    add("D1", "Description control (question-blind, twins, pooled d' change Desc-Direct vs CoT-Direct): " + "; ".join(
        f"{NAMES[m]} {v['desc_minus_direct'][0]:+.2f} {v['desc_minus_direct'][1]} vs {v['cot_minus_direct'][0]:+.2f}" for m, v in D["twins"].items())
        + ". Occlusion component (cue-cell decomposition) Desc-Direct vs CoT-Direct: " + "; ".join(
        f"{NAMES[m]} {v['occl_component']['desc-direct']['d']:+.2f} vs {v['occl_component']['cot-direct']['d']:+.2f}" for m, v in D["twins"].items() if v["occl_component"].get("desc-direct"))
        + ". Held-out (767) Desc-Direct accuracy: " + "; ".join(f"{NAMES[m]} {v['desc_minus_direct']}" for m, v in D["heldout"].items()) + ". " + D["cues"] + ".",
        "dprime_ci.json, cues.json twins.decomposition, e6_eval.json")
    add("Q1", "Closed models: Direct and CoT both ran with reasoning_effort='high' (Gemini: thinking_level high) on twins, Synthetic Depth (250) and BLINK (e9x) and on the hard set "
              "(e9h); hidden reasoning was present in both arms (Direct calls with reasoning tokens: Claude Opus 5.5 77.6-100%, GPT-6-Astra 51.7-99.1%, Gemini/DeepSeek/Qwen3.8-Max/HY4 100%). "
              "CoT was a user-turn instruction ('Let's think step by step: first describe ..., end with Answer: (X)') with the answer parsed from text; open models get the CoT as an "
              "assistant prefix and a forced single-token readout. A reasoning-off Direct ('direct_nr') exists for twins / Synthetic Depth / BLINK Depth (DeepSeek, Qwen3.8-Max: none, "
              "0% reasoning; Claude: minimal, 22.8% of twin calls still reason; GPT: low, 0.2%) and on the hard set (Claude minimal 89.6%, GPT low 29.3%, DeepSeek/Qwen3.8-Max none 0%). "
              "No CoT arm at reasoning none exists except DeepSeek-V4.1-Flash on the cue set (both arms none, 0% reasoning). Temperature not set (provider default); one call per item.",
        "e9x_api.py, api_client.py, outputs/e9x|e9h|e9c_cues records (effort, reasoning_tokens); research_log Exp-9x")
    add("Q2", "Gemini-4-Argon hard set is partial (79 Direct / 82 CoT of 576); Qwen3.8-Max hard set has 854/852 records for 576 prompts (duplicate rows from a re-run).",
        "outputs/e9h/*.jsonl counts")
    add("R1", "Readouts: V0 exact full-vocabulary (Qwen2.5-VL-3B/7B, InternVL3-8B, Phi-3.5-V) for Tab. 1/S3/S4/S11/S24-S27/Tab. 2 main pool; V1 top-20 (Molmo-7B-D, Qwen3-VL-8B, "
              "-Thinking, InternVL3.5-8B) throughout; Sec. 5 groups (Fig. 2a,b, Tab. S28-S34, steering) use the original runs' top-20 letter scores for all models.",
        "build_bundle.py docstring, Tab. S4 note, e27_reframe.json readout.usage")
    add("R2", R["disclosure"], "e27_reframe.json readout.drift_syn_depth, readout.cells")
    for f_ in F:
        lines.append(f"FACT {f_['tag']}: {f_['fact']} [source: {f_['source']}]")
    for x in fx:
        lines.append(f"FIX ({x['id']}) {x['item']}: old {x['old']} -> new {x['new']} [source: {x['source']}; changed: {x['changed']}]")
    return {"facts": F, "fixes": fx}


# ----------------------------------------------------------------------------- main

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="heur,comp,ovl,cons,desc,closed,readout")
    a = ap.parse_args()
    out = json.load(open(B / "e27_reframe.json")) if (B / "e27_reframe.json").exists() else {}
    old = (B / "rigor_e27.txt").read_text().splitlines() if (B / "rigor_e27.txt").exists() else []
    lines: list[str] = []
    pref = {"heur": "HEUR", "comp": "COMP", "ovl": "OVL", "cons": "CONS", "desc": "DESC", "closed": "CLOSED", "readout": ("READOUT", "DRIFT"), "fixrefresh": "FIXREFRESH", "facts": ("FACT", "FIX ")}
    for k in a.only.split(","):
        if k == "comp":
            out[k] = comp(lines, out.get("heur"))
        elif k == "facts":
            out[k] = facts(lines, out)
        else:
            out[k] = {"heur": heur, "ovl": ovl, "cons": cons, "desc": desc, "closed": closed, "readout": readout, "fixrefresh": fixrefresh}[k](lines)
        print("\n".join(x for x in lines if x.startswith(pref[k] if isinstance(pref[k], str) else pref[k])), flush=True)
    keep = [x for x in old if not any(x.startswith(pref[k] if isinstance(pref[k], str) else pref[k]) for k in a.only.split(","))]
    bias_new = out.get("cons", {}).pop("_bias_new", None)
    if bias_new is not None:
        out["_bias_sensitivity_unrounded"] = bias_new
    json.dump(out, open(B / "e27_reframe.json", "w"), indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o))
    (B / "rigor_e27.txt").write_text("\n".join(keep + lines) + "\n")


if __name__ == "__main__":
    main()
