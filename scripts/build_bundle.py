"""Build the numbers bundle for the paper from per-item outputs (re-runnable as experiments finish).

Writes outputs/bundle/numbers.json (nested dict of every reported number with 95% image-cluster
bootstrap CIs) and outputs/bundle/tables.md (human-readable tables). Sections that lack inputs are
skipped. Readouts: exact full-vocab (sum over letter variants) for V0 models; top-20 for V1 models
(Molmo, Qwen3-VL, InternVL3.5), whose top-20 option mass is >= 0.99.

Usage: python build_bundle.py --outputs <W>/outputs --legacy ../results/v2
"""
from __future__ import annotations
import vtax_paths as _vtp  # noqa: E402

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import norm

from common import cluster_bootstrap_ci, cluster_signflip_p, plurality_win_prob

MODELS = ["qwen3b", "qwen7b", "internvl3_8b", "phi35v", "molmo7b", "qwen3vl_8b", "qwen3vl_8b_think", "internvl35_8b"]
NAMES = {"qwen3b": "Qwen2.5-VL-3B", "qwen7b": "Qwen2.5-VL-7B", "internvl3_8b": "InternVL3-8B",
         "phi35v": "Phi-3.5-V", "molmo7b": "Molmo-7B-D", "qwen3vl_8b": "Qwen3-VL-8B",
         "internvl35_8b": "InternVL3.5-8B", "qwen3vl_8b_think": "Qwen3-VL-8B-Thinking"}
TASKS = ["cvb_relation", "cvb_depth", "cvb_distance", "syn_relpos", "syn_depth"]


def rj(p: Path) -> list[dict]:
    return [json.loads(l) for l in open(p)] if p.exists() else []


def am(lp: dict) -> str | None:
    fin = {k: v for k, v in lp.items() if v is not None and math.isfinite(v)}
    return max(fin, key=fin.get) if fin else None


def pe(r: dict) -> str | None:
    """Prediction of a readout record: exact (lp_sum) if present else top-20."""
    if "lp_sum" in r:
        return am(r["lp_sum"])
    return am(r["top20"])


def probs(lp: dict) -> dict:
    fin = {k: v for k, v in lp.items() if v is not None and math.isfinite(v)}
    if not fin:
        return {k: 1 / len(lp) for k in lp}
    m = max(fin.values())
    z = sum(math.exp(v - m) for v in fin.values())
    return {k: (math.exp(lp[k] - m) / z if k in fin else 0.0) for k in lp}


def ci(vals, groups, scale=100.0) -> dict:
    v = np.asarray(vals, dtype=float)
    if len(v) == 0:
        return {"n": 0}
    m, lo, hi = cluster_bootstrap_ci(v, groups, 2000)
    return {"mean": round(scale * m, 4), "lo": round(scale * lo, 4), "hi": round(scale * hi, 4), "n": int(len(v))}


def fmt(d: dict) -> str:
    if not d or d.get("n", 0) == 0:
        return "--"
    return f"{d['mean']:.1f} [{d['lo']:.1f}, {d['hi']:.1f}]"


# ----------------------------------------------------------------------------- behavioral

def load_main(O: Path, L: Path, m: str) -> dict:
    """{condition: {uid: (correct, pgold, image_id, task, pred)}} on the 2,850 pool."""
    out = defaultdict(dict)
    recs = rj(O / "e1x" / f"{m}.jsonl")
    if recs:
        for r in recs:
            p = pe(r)
            pg = probs(r["lp_sum"] if "lp_sum" in r else r["top20"])[r["gold"]]
            out[r["condition"]][r["item_id"]] = (float(p == r["gold"]), pg, r["image_id"], r["task"], p)
        return out
    src = L / "e1" / f"{m}.json"
    if not src.exists():
        src = O / "e4" / f"{m}.json"
    if not src.exists():
        return out
    for it in json.load(open(src))["items"]:
        if it["task"] not in TASKS:
            continue
        for c, sc in it["scores"].items():
            p = am(sc)
            c = "cot" if c == "think" else c  # Thinking model: native think trace plays the CoT role
            out[c][it["uid"]] = (float(p == it["gold"]), probs(sc)[it["gold"]], it["group"], it["task"], p)
    return out


def behavioral(main: dict) -> dict:
    res = {}
    if "direct" not in main:
        return res
    D = main["direct"]
    for c, rows in main.items():
        keys = [k for k in D if k in rows]
        g = [D[k][2] for k in keys]
        res[c] = {"acc": ci([rows[k][0] for k in keys], g)}
        for ref in ("direct", "cot", "cot+turn", "filler"):
            if ref in main and c != ref:
                d = [rows[k][0] - main[ref][k][0] for k in keys if k in main[ref]]
                res[c][f"minus_{ref}"] = ci(d, g) | {"p": round(cluster_signflip_p(d, g, 2000), 4)}
    # per task
    for t in TASKS:
        keys = [k for k in D if D[k][3] == t]
        g = [D[k][2] for k in keys]
        row = {}
        for c in ("direct", "cot", "filler+turn", "cot+turn", "filler"):
            if c in main:
                row[c] = ci([main[c][k][0] for k in keys], g)
        if "cot" in main:
            row["cot_minus_direct"] = ci([main["cot"][k][0] - D[k][0] for k in keys], g)
            row["cot_minus_pbar"] = ci([main["cot"][k][0] - D[k][1] for k in keys], g)
            if "filler+turn" in main:
                row["cot_minus_fillerturn"] = ci([main["cot"][k][0] - main["filler+turn"][k][0] for k in keys], g)
        res[f"task:{t}"] = row
    return res


# ----------------------------------------------------------------------------- image contingency

def sdt(yes_given_yes: list, yes_given_no: list) -> dict:
    eps = 0.5
    h = (sum(yes_given_yes) + eps) / (len(yes_given_yes) + 2 * eps)
    f = (sum(yes_given_no) + eps) / (len(yes_given_no) + 2 * eps)
    return {"dprime": round(norm.ppf(h) - norm.ppf(f), 3), "c": round(-(norm.ppf(h) + norm.ppf(f)) / 2, 3),
            "hit": round(h, 3), "fa": round(f, 3)}


def occlusion(O: Path, m: str) -> dict:
    res = {}
    for ph, fn in (("front", f"{m}.jsonl"), ("behind", f"{m}_behind.jsonl")):
        recs = rj(O / "e2d" / fn)
        if not recs:
            continue
        for c in ("direct", "cot", "desc"):
            rs = {r["item_id"]: r for r in recs if r["condition"] == c}
            if not rs:
                continue
            acc = [float(pe(r) == r["gold"]) for r in rs.values()]
            g = [r["image_id"] for r in rs.values()]
            both, flip, pg = [], [], []
            for k, r in rs.items():
                if k.endswith("_cf") or k + "_cf" not in rs:
                    continue
                t = rs[k + "_cf"]
                both.append(float(pe(r) == r["gold"] and pe(t) == t["gold"]))
                flip.append(float(pe(r) != pe(t)))
                pg.append(k)
            yy = [float(pe(r) == "A") for r in rs.values() if r["gold"] == "A"]
            yn = [float(pe(r) == "A") for r in rs.values() if r["gold"] == "B"]
            res[f"{ph}:{c}"] = {"acc": ci(acc, g), "pair_both_correct": ci(both, pg), "twin_flip": ci(flip, pg),
                                "yes_rate": round(float(np.mean([pe(r) == "A" for r in rs.values()])), 3),
                                "sdt": sdt(yy, yn)}
    # pooled front+behind: SDT on the judgement "the question's subject X is in front"
    # (truth = meta front_name == x_name), which removes yes/no polarity from the response variable
    for c in ("direct", "cot", "desc"):
        if f"front:{c}" in res and f"behind:{c}" in res:
            hit, fa = [], []
            for ph, fn in (("front", f"{m}.jsonl"), ("behind", f"{m}_behind.jsonl")):
                for r in rj(O / "e2d" / fn):
                    if r["condition"] != c:
                        continue
                    says = (pe(r) == "A") if ph == "front" else (pe(r) == "B")
                    (hit if r["meta"]["front_name"] == r["meta"]["x_name"] else fa).append(float(says))
            res[f"pooled:{c}"] = {"sdt": sdt(hit, fa),
                                  "acc_mean_two_phrasings": round((res[f"front:{c}"]["acc"]["mean"] +
                                                                   res[f"behind:{c}"]["acc"]["mean"]) / 2, 2)}
    return res


def question_cf(O: Path, L: Path, m: str, G: dict) -> dict:
    """E2a counterfactual-question / polarity / mirror flips and gray/none summaries."""
    res = {}
    recs = rj(O / "e2a" / f"{m}.jsonl") + rj(O / "e2a" / f"{m}_polar.jsonl")
    for var in ("cfq", "polar", "mirror"):
        for task in ("syn_depth", "syn_relpos", "cvb_relation", "all"):
            sel = [r for r in recs if r["variant"] == var and (task == "all" or r["task"] == task)
                   and r["item_id"] in G and r["stated"] is not None and G[r["item_id"]]["cot_free"] is not None]
            if len(sel) < 20:
                continue
            g = [r["image_id"] for r in sel]
            res[f"{var}:{task}"] = {
                "R_flip": ci([float(r["stated"] != G[r["item_id"]]["cot_free"]) for r in sel], g),
                "Direct_flip": ci([float(pe(r["readout_direct"]) != G[r["item_id"]]["direct"]) for r in sel], g),
                "R_pair_both_correct": ci([float(r["stated"] == r["gold"] and G[r["item_id"]]["cot_free"] == r["gold_orig"])
                                           for r in sel], g),
                "Direct_pair_both_correct": ci([float(pe(r["readout_direct"]) == r["gold"] and
                                                      G[r["item_id"]]["direct"] == r["gold_orig"]) for r in sel], g)}
    for var in ("gray", "none"):
        for task in TASKS:
            sel = [r for r in recs if r["variant"] == var and r["task"] == task and r["item_id"] in G]
            if not sel:
                continue
            res[f"{var}:{task}"] = {"R_optionA_rate": round(float(np.mean([r["stated"] == "A" for r in sel])), 3),
                                    "R_orig_optionA_rate": round(float(np.mean([G[r["item_id"]]["cot_free"] == "A"
                                                                                for r in sel])), 3),
                                    "acc_realimg+R_var": ci([float(pe(r["readout_cot"]) == r["gold"]) for r in sel],
                                                            [r["image_id"] for r in sel])}
    return res


def groups_from(main: dict, L: Path, O: Path, m: str) -> dict:
    src = L / "e1" / f"{m}.json"
    if not src.exists():
        src = O / "e4" / f"{m}.json"
    G = {}
    if not src.exists():
        return G
    for it in json.load(open(src))["items"]:
        if it["task"] not in TASKS:
            continue
        d, c = am(it["scores"]["direct"]), am(it["scores"].get("cot") or it["scores"]["think"])
        G[it["uid"]] = {"direct": d, "cot": c, "cot_free": it.get("cot_free"),
                        "group": ("D+" if d == it["gold"] else "D-") + ("C+" if c == it["gold"] else "C-")}
    return G


# ----------------------------------------------------------------------------- error onset, swaps

def error_onset(O: Path, m: str, G: dict) -> dict:
    import re
    rel = re.compile(r"\b(in front|behind|closer|farther|further|nearer|left|right|above|below|on top|under|beneath|"
                     r"foreground|background|occlud\w*|overlap\w*|hidden|blocks?|covers?)\b", re.I)
    concl = re.compile(r"\b(answer|correct option|conclusion)\b|\([A-F]\)", re.I)
    res = {}
    recs = rj(O / "e2b" / f"{m}.jsonl")
    if not recs:
        return res
    for grp in ("D+C-", "D+C+"):
        sel = [r for r in recs if G.get(r["item_id"], {}).get("group") == grp]
        types, p0low, pos = Counter(), [], []
        for r in sel:
            pg = [probs(t.get("lp_sum") or t["lp_bare"])[r["gold"]] for t in sorted(r["traj"], key=lambda x: x["j"])]
            p0low.append(pg[0] < 0.5)
            js = next((j for j in range(1, len(pg)) if pg[j - 1] >= 0.5 and all(v < 0.5 for v in pg[j:])), None)
            if js is None:
                types["no_commit"] += 1
                continue
            s = r["sentences"][js - 1]
            types["conclusion" if concl.search(s) else ("relation" if rel.search(s) else "non_relational")] += 1
            pos.append(js / max(1, len(r["sentences"])))
        n = max(1, len(sel))
        res[grp] = {"n": len(sel), "P0_below_half": round(100 * float(np.mean(p0low)), 1) if sel else None,
                    "commit_rel_position": round(float(np.mean(pos)), 2) if pos else None,
                    **{f"type_{k}": round(100 * v / n, 1) for k, v in types.items()}}
    return res


def claim_truth(O: Path, m: str, G: dict) -> dict:
    """Syn. Depth: does the rationale's first two-object occlusion/front claim contradict the image?

    Heuristic parse: the first sentence naming both objects with an occlusion/front verb; the first-named
    object is the claimed front object for active verbs, the other one for passive ("covered by", "behind").
    """
    import re
    obj = re.compile(r"\b(red|blue|green|yellow|purple|orange|cyan|gray|grey|pink|brown|black|white)\s+"
                     r"(circle|square|triangle|diamond|hexagon|pentagon|star|rectangle)s?\b", re.I)
    act = re.compile(r"\b(overlap\w*|cover\w*|on top of|obscur\w*|block\w*|in front of)\b", re.I)
    pas = re.compile(r"\b(covered|hidden|obscured|blocked|overlapped|occluded)\s+by\b|\bbehind\b|\bunderneath\b|"
                     r"\bbeneath\b", re.I)
    qrx = re.compile(r"Is the (\w+ \w+) in front of the (\w+ \w+)\?")
    res = {}
    recs = [r for r in rj(O / "e2b" / f"{m}.jsonl") if r["task"] == "syn_depth"]
    for grp in ("D+C-", "D+C+"):
        wrong, parsed, tot, g = [], 0, 0, []
        for r in recs:
            if G.get(r["item_id"], {}).get("group") != grp:
                continue
            q = qrx.search(r["question"])
            if not q:
                continue
            tot += 1
            x, y = q.group(1).lower(), q.group(2).lower()
            front = x if r["gold"] == "A" else y
            for s in r["sentences"]:
                objs = [(o.group(1) + " " + o.group(2)).lower().replace("grey", "gray") for o in obj.finditer(s)]
                if len(set(objs)) < 2 or not (act.search(s) or pas.search(s)):
                    continue
                if re.search(r"\?|whether|if the|determine|asks", s, re.I):
                    continue
                first = objs[0]
                claimed = first if not pas.search(s) else [o for o in objs if o != first][0]
                wrong.append(float(claimed != front))
                g.append(r["item_id"])
                parsed += 1
                break
        if tot:
            res[grp] = {"items": tot, "parsed_share": round(100 * parsed / tot, 1),
                        "claim_contradicts_image": ci(wrong, g)}
    return res


def midtrace(O: Path, m: str) -> dict:
    res = {}
    recs = rj(O / "e2c" / f"{m}.jsonl")
    for p in ("start", "rel", "commit"):
        for mode in ("reimage", "text"):
            neg = [r for r in recs if r["point"] == p and r["mode"] == mode and r["group"] == "D+C-"]
            pos = [r for r in recs if r["point"] == p and r["mode"] == mode and r["group"] == "D+C+"]
            if neg:
                res[f"{p}:{mode}"] = {"recovered_DCm": ci([float(pe(r) == r["gold"]) for r in neg], [r["image_id"] for r in neg]),
                                      "kept_DCp": ci([float(pe(r) == r["gold"]) for r in pos], [r["image_id"] for r in pos])}
    return res


def swaps(O: Path, m: str) -> dict:
    res = {}
    recs = rj(O / "e3x" / f"{m}.jsonl")
    if not recs:
        return res
    idx = defaultdict(dict)
    meta = {}
    for r in recs:
        idx[(r["item_id"], r["image"], r["context"])][r["Y"]] = r
        meta[r["item_id"]] = r
    for ctx in ("rcut", "concl"):
        S = {}
        for img in ("original", "gray", "mirror"):
            vals, fg_, fw_, g = [], [], [], []
            for uid, r0 in meta.items():
                d = idx.get((uid, img, ctx))
                if not d:
                    continue
                gold = r0["gold"]
                w = [c for c in r0["letters"] if c != gold][0]
                if gold in d and w in d:
                    fg, fw = float(pe(d[gold]) == gold), float(pe(d[w]) == w)
                    vals.append(fg - fw); fg_.append(fg); fw_.append(fw); g.append(r0["image_id"])
                    S.setdefault(img, {})[uid] = fg - fw
            if vals:
                res[f"{ctx}:{img}"] = {"F_gold": ci(fg_, g), "F_wrong": ci(fw_, g), "S": ci(vals, g)}
        if "original" in S and "gray" in S:
            ks = [k for k in S["original"] if k in S["gray"]]
            res[f"{ctx}:dvis"] = ci([S["original"][k] - S["gray"][k] for k in ks], [meta[k]["image_id"] for k in ks])
    return res


# ----------------------------------------------------------------------------- fixes

def sym_and_gate(O: Path, main: dict, m: str, heldout: bool = False) -> dict:
    """Symmetrized Direct / CoT over the order-swapped twin; CoT-if-consistent-else-Direct."""
    res = {}
    if heldout:
        drecs = {r["item_id"]: r for r in rj(O / "e6x" / f"{m}_direct.jsonl")}
        rest = rj(O / "e6x" / f"{m}_rest.jsonl")
        crecs = {r["item_id"]: r for r in rest if r["condition"] == "cot"}
        vrecs = {r["item_id"]: r for r in rest if r["condition"] == "vote4"}
        desc = {r["item_id"]: r for r in rest if r["condition"] == "desc"}
        swp = rj(O / "e5c" / f"{m}_heldout.jsonl")
    else:
        drecs, crecs, vrecs, desc = {}, {}, {}, {}
        swp = rj(O / "e5c" / f"{m}.jsonl")
    sd = {r["item_id"]: r for r in swp if r["condition"] == "direct_swap"}
    sc = {r["item_id"]: r for r in swp if r["condition"] == "cot_swap"}
    rows = defaultdict(list)
    g = []
    flag_rows: list[tuple[float, float, str]] = []  # (CoT order-inconsistent, CoT wrong, image)
    for uid in (sd if sd else drecs):
        if heldout:
            if uid not in drecs or uid not in crecs:
                continue
            d0, c0 = drecs[uid], crecs[uid]
            gold, letters, img = d0["gold"], d0["letters"], d0["image_id"]
            pd0 = probs(d0.get("lp_sum") or d0["top20"]); pc0 = probs(c0.get("lp_sum") or c0["top20"])
        else:
            if "direct" not in main or uid not in main["direct"]:
                continue
            rec_d = main["direct"][uid]; rec_c = main["cot"][uid]
            gold = sd[uid]["gold_orig"]; letters = sd[uid]["letters"]; img = rec_d[2]
            pd0 = {gold: rec_d[1], [l for l in letters if l != gold][0]: 1 - rec_d[1]}
            pc0 = {gold: rec_c[1], [l for l in letters if l != gold][0]: 1 - rec_c[1]}
        if uid not in sd or uid not in sc:
            continue
        other = [l for l in letters if l != gold][0]
        # in the twin the gold letter is flipped: content of original letter X sits at the other letter
        flipmap = {gold: other, other: gold}
        pd1 = probs(sd[uid].get("lp_sum") or sd[uid]["top20"]); pc1 = probs(sc[uid].get("lp_sum") or sc[uid]["top20"])
        sym_d = {x: pd0[x] + pd1[flipmap[x]] for x in letters}
        sym_c = {x: pc0[x] + pc1[flipmap[x]] for x in letters}
        if heldout:
            p_d0, p_c0 = max(pd0, key=pd0.get), max(pc0, key=pc0.get)
        else:  # recorded predictions; argmax of the two-letter reconstruction would send ties to gold
            p_d0, p_c0 = rec_d[4], rec_c[4]
        p_c1 = flipmap[max(pc1, key=pc1.get)]
        rows["direct"].append(float(p_d0 == gold))
        rows["cot"].append(float(p_c0 == gold))
        rows["sym_direct"].append(float(max(sym_d, key=sym_d.get) == gold))
        rows["sym_cot"].append(float(max(sym_c, key=sym_c.get) == gold))
        rows["cot_consistent_else_direct"].append(float((p_c0 if p_c0 == p_c1 else p_d0) == gold))
        rows["cot_consistent_rate"].append(float(p_c0 == p_c1))
        rows["direct_consistent_rate"].append(float(p_d0 == flipmap[max(pd1, key=pd1.get)]))
        flag_rows.append((float(p_c0 != p_c1), float(p_c0 != gold), img))
        if heldout and uid in vrecs:
            preds = [pe(o) for o in vrecs[uid]["vote_outs"]]
            cnt = Counter(preds)
            top = max(cnt.values())
            win = [k for k, v in cnt.items() if v == top]
            rows["vote4"].append(float(gold in win) / len(win))
        if heldout and uid in desc:
            rows["desc"].append(float(pe(desc[uid]) == gold))
        g.append(img)
    for k, v in rows.items():
        gg = g if len(v) == len(g) else g[: len(v)]
        res[k] = ci(v, gg)
        if k not in ("direct", "cot_consistent_rate", "direct_consistent_rate") and len(v) == len(rows["direct"]):
            res[f"{k}_minus_direct"] = ci(np.array(v) - np.array(rows["direct"]), gg)
    if flag_rows:
        fl, er, im = (list(x) for x in zip(*flag_rows))
        res["flag"] = {
            "flag_rate": ci(fl, im), "cot_error_rate": ci(er, im),
            "precision": ci([e for f, e in zip(fl, er) if f], [i for f, i in zip(fl, im) if f]),
            "recall": ci([f for f, e in zip(fl, er) if e], [i for e, i in zip(er, im) if e]),
            "p_err_given_consistent": ci([e for f, e in zip(fl, er) if not f], [i for f, i in zip(fl, im) if not f]),
            "n": len(fl), "n_flagged": int(sum(fl)), "n_cot_errors": int(sum(er))}
    return res


def gate_e0(path: Path) -> dict:
    out = {}
    if not path.exists():
        return out
    import csv
    for r in csv.DictReader(open(path)):
        out[f"{r['model']}:{r['alternative']}"] = {k: r[k] for k in ("n", "direct_acc", "alt_acc", "gate_cv_acc",
                                                                      "gate_minus_direct", "gate_minus_direct_lo",
                                                                      "gate_minus_direct_hi", "routed_frac_cv", "tau_frozen")}
    return out


def main_() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--outputs", required=True)
    ap.add_argument("--legacy", default=f"{_vtp.LEGACY_RESULTS}")
    ap.add_argument("--out-dir", default="outputs/bundle")
    args = ap.parse_args()
    O, L, out = Path(args.outputs), Path(args.legacy), Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    gate = gate_e0(Path("outputs/e0x/gate_cv.csv"))  # exact readouts (V0 models)
    for p, ms in ((Path("outputs/e0/gate_cv.csv"), ("molmo7b",)), (Path("outputs/e0x_qwen3vl/gate_cv.csv"), ("qwen3vl_8b",)),
                  (O / "e0x_ivl35" / "gate_cv.csv", ("internvl35_8b",))):
        gate |= {k: v for k, v in gate_e0(p).items() if k.split(":")[0] in ms}
    B = {"gate_main_pool_cv": gate}
    md = []
    for m in MODELS:
        main = load_main(O, L, m)
        G = groups_from(main, L, O, m)
        b = {"behavioral": behavioral(main), "occlusion_twins": occlusion(O, m), "question_cf": question_cf(O, L, m, G),
             "error_onset": error_onset(O, m, G), "claim_truth_syn_depth": claim_truth(O, m, G),
             "midtrace_reimage": midtrace(O, m), "swaps": swaps(O, m),
             "fix_main_pool": sym_and_gate(O, main, m), "fix_heldout": sym_and_gate(O, main, m, heldout=True)}
        B[m] = b
        md.append(f"## {NAMES[m]}\n")
        for sec, d in b.items():
            if not d:
                continue
            md.append(f"### {sec}\n")
            for k, v in d.items():
                if isinstance(v, dict) and "mean" in v:
                    md.append(f"- {k}: {fmt(v)}")
                elif isinstance(v, dict):
                    md.append(f"- {k}: " + "; ".join(f"{kk}={fmt(vv) if isinstance(vv, dict) and 'mean' in vv else vv}"
                                                     for kk, vv in v.items()))
            md.append("")
    json.dump(B, open(out / "numbers.json", "w"), indent=1)
    (out / "tables.md").write_text("\n".join(md))
    print(f"wrote {out}/numbers.json and tables.md")


if __name__ == "__main__":
    main_()
