"""Exp-58 Part B: leverage checks.

(1) Main pool (Tab. 1, 8 open models): CoT - Direct reproduced on the full 2,850 pool (item-weighted, image-cluster
    bootstrap as build_bundle.behavioral); then without Synthetic Depth: task-weighted mean over CVB 2D Relation,
    CVB 3D Depth, CVB 3D Distance, Syn. RelPos (and item-weighted, 2,350 items), image-cluster bootstrap 2,000 with
    draws shared across models (global image clusters).
(2) Two-channel fits with and without the 7 Synthetic Depth 431 points: Fig. 5 / Tab. S7 (conclusion kept, claim_parse
    v3, 21 points) and Tab. S13 (uniform conclusion-removed protocol, Exp-45 two-judge consensus, 21 points);
    e45_analyze.fit (point, model-cluster, item and item + model bootstrap) plus r on the item / item + model draws.
(3) Conclusion-removed agreement aT / aF (Exp-45 cells) for 7 models x 3 synthetic sets.
"""
from __future__ import annotations
import vtax_paths as _vtp  # noqa: E402

import json
from pathlib import Path

import numpy as np
from scipy.stats import pearsonr

import build_bundle as BB
from common import cluster_bootstrap_ci
from e45_analyze import cell, fit, load_labels, shape_records, v3_records
from e45_uniform_two_channel import B, E45, M7, O, rj

L = Path(f"{_vtp.LEGACY_RESULTS}")
M8 = BB.MODELS
TASKS4 = ["cvb_relation", "cvb_depth", "cvb_distance", "syn_relpos"]
SYN_DS = ("cues", "twins", "syn431")
NB = 2000
q95 = lambda v: [float(np.nanquantile(v, .025)), float(np.nanquantile(v, .975))]


# ============================================================================= (1) main pool

def main_pool(lines: list[str]) -> dict:
    nums = json.load(open(B / "numbers.json"))
    data = {}
    for m in M8:
        main = BB.load_main(O, L, m)
        D, C = main["direct"], main["cot"]
        keys = [k for k in D if k in C]
        data[m] = [(k, D[k][2], D[k][3], C[k][0] - D[k][0], D[k][0], C[k][0]) for k in keys]
    # reproduction (same estimator / seed as build_bundle.behavioral)
    rep, bad = {}, []
    for m in M8:
        d = [r[3] for r in data[m]]
        g = [r[1] for r in data[m]]
        mu, lo, hi = cluster_bootstrap_ci(d, g, 2000)
        ref = nums[m]["behavioral"]["cot"]["minus_direct"]
        rep[m] = {"n": len(d), "est": 100 * mu, "ci": [100 * lo, 100 * hi], "paper_numbers_json": [ref["mean"], ref["lo"], ref["hi"]],
                  "acc_direct": 100 * np.mean([r[4] for r in data[m]]), "acc_cot": 100 * np.mean([r[5] for r in data[m]])}
        if abs(100 * mu - ref["mean"]) > 1e-3 or abs(100 * lo - ref["lo"]) > 1e-3 or abs(100 * hi - ref["hi"]) > 1e-3:
            bad.append(m)
    lines.append(f"SANITY Tab. 1 CoT - Direct (2,850, item-weighted, image-cluster CI) reproduced: mismatches {bad}; "
                 + ", ".join(f"{m} {v['est']:+.1f} [{v['ci'][0]:+.1f}, {v['ci'][1]:+.1f}]" for m, v in rep.items()))
    # shared global image clusters
    allc = sorted({r[1] for m in M8 for r in data[m]}, key=str)
    cidx = {c: i for i, c in enumerate(allc)}
    rng = np.random.default_rng(58)
    draws = rng.integers(0, len(allc), (NB, len(allc)))
    W = np.stack([np.bincount(dr, minlength=len(allc)) for dr in draws]).astype(float)
    tasks5 = TASKS4 + ["syn_depth"]
    shared = {}
    for t in tasks5:
        imgs = {r[1] for m in M8 for r in data[m] if r[2] == t}
        shared[t] = imgs
    overlap = {f"{a}&{b}": len(shared[a] & shared[b]) for i, a in enumerate(tasks5) for b in tasks5[i + 1:] if shared[a] & shared[b]}
    out, bd = {}, {}
    for m in M8:
        S = np.zeros((len(allc), len(tasks5)))
        n = np.zeros((len(allc), len(tasks5)))
        for k, c, t, d, _, _ in data[m]:
            S[cidx[c], tasks5.index(t)] += d
            n[cidx[c], tasks5.index(t)] += 1
        per_task = 100 * (W @ S) / (W @ n)  # (NB, 5)
        est_t = 100 * S.sum(0) / n.sum(0)
        i4 = [tasks5.index(t) for t in TASKS4]
        tw4, tw5 = per_task[:, i4].mean(1), per_task.mean(1)
        iw4 = 100 * (W @ S[:, i4].sum(1)) / (W @ n[:, i4].sum(1))
        o = {"per_task": {t: float(est_t[j]) for j, t in enumerate(tasks5)},
             "per_task_ci": {t: q95(per_task[:, j]) for j, t in enumerate(tasks5)},
             "task_weighted_no_syn_depth": float(est_t[i4].mean()), "task_weighted_no_syn_depth_ci": q95(tw4),
             "item_weighted_no_syn_depth": float(100 * S[:, i4].sum() / n[:, i4].sum()), "item_weighted_no_syn_depth_ci": q95(iw4),
             "n_no_syn_depth": int(n[:, i4].sum()),
             "task_weighted_all5": float(est_t.mean()), "task_weighted_all5_ci": q95(tw5),
             "full_pool_item_weighted": rep[m]}
        out[m] = o
        bd[m] = {"tw4": tw4, "iw4": iw4, "tw5": tw5, "syn": per_task[:, tasks5.index("syn_depth")]}
    for nm, ms in (("mean8", M8), ("mean7_instruct", [m for m in M8 if m != "qwen3vl_8b_think"])):
        out[nm] = {k: float(np.mean([out[m][f"{k}"] for m in ms])) for k in ("task_weighted_no_syn_depth", "item_weighted_no_syn_depth", "task_weighted_all5")}
        out[nm] |= {f"{k}_ci": q95(np.mean([bd[m][b] for m in ms], 0)) for k, b in (("task_weighted_no_syn_depth", "tw4"), ("item_weighted_no_syn_depth", "iw4"), ("task_weighted_all5", "tw5"))}
        out[nm]["syn_depth"] = float(np.mean([out[m]["per_task"]["syn_depth"] for m in ms]))
        out[nm]["syn_depth_ci"] = q95(np.mean([bd[m]["syn"] for m in ms], 0))
        out[nm]["full_pool"] = float(np.mean([rep[m]["est"] for m in ms]))
    out["_image_overlap_between_tasks"] = overlap
    out["_n_models_ci_below0_no_syn"] = int(sum(out[m]["task_weighted_no_syn_depth_ci"][1] < 0 for m in M8))
    out["_n_models_ci_above0_no_syn"] = int(sum(out[m]["task_weighted_no_syn_depth_ci"][0] > 0 for m in M8))
    for m in M8 + ["mean8", "mean7_instruct"]:
        o = out[m]
        full = o["full_pool_item_weighted"]["est"] if m in M8 else o["full_pool"]
        lines.append(f"MAINPOOL {m:16s} full {full:+.2f} | w/o Syn. Depth task-weighted {o['task_weighted_no_syn_depth']:+.2f} "
                     f"[{o['task_weighted_no_syn_depth_ci'][0]:+.2f}, {o['task_weighted_no_syn_depth_ci'][1]:+.2f}] item-weighted "
                     f"{o['item_weighted_no_syn_depth']:+.2f} [{o['item_weighted_no_syn_depth_ci'][0]:+.2f}, {o['item_weighted_no_syn_depth_ci'][1]:+.2f}] "
                     f"| all 5 task-weighted {o['task_weighted_all5']:+.2f} | Syn. Depth {o['per_task']['syn_depth'] if m in M8 else o['syn_depth']:+.2f}"
                     + (" | per task " + ", ".join(f"{t} {v:+.1f}" for t, v in o["per_task"].items()) if m in M8 else ""))
    lines.append(f"MAINPOOL image overlap between tasks: {overlap}; models with CI < 0 / > 0 (w/o Syn. Depth, task-weighted): "
                 f"{out['_n_models_ci_below0_no_syn']} / {out['_n_models_ci_above0_no_syn']}")
    return out


# ============================================================================= (2) fits

def fit_plus(pts: list[dict]) -> dict:
    f = fit(pts)
    dx = np.stack([p["_draws"]["x"] for p in pts], 1)
    dy = np.stack([p["_draws"]["y"] for p in pts], 1)
    models = sorted({p["model"] for p in pts})
    mi = {m: np.array([k for k, p in enumerate(pts) if p["model"] == m]) for m in models}
    rng = np.random.default_rng(5858)
    ri, rim = [], []
    for b in range(dx.shape[0]):
        ri.append(pearsonr(dx[b], dy[b])[0])
        i = np.concatenate([mi[models[j]] for j in rng.integers(0, len(models), len(models))])
        if np.ptp(dx[b, i]) > 0 and np.ptp(dy[b, i]) > 0:
            rim.append(pearsonr(dx[b, i], dy[b, i])[0])
    f["r_ci_item"] = q95(np.array(ri))
    f["r_ci_item_model"] = q95(np.array(rim))
    f["points"] = [{"model": p["model"], "ds": p["ds"], "x": p["x"], "y": p["y"]} for p in pts]
    return f


def fits(lines: list[str]) -> dict:
    rows = [r for r in rj(E45 / "items.jsonl") if r["group"] == "open" and r["model"] in M7 and r["ds"] in SYN_DS]
    srcs = {"fig5_conclusion_kept_v3": v3_records(rows, "v3_full"), "s13_uniform_removed": shape_records(rows, load_labels())}
    res = {}
    for nm, recs in srcs.items():
        g = {}
        for r in recs:
            g.setdefault((r["model"], r["ds"]), []).append(r)
        rng = np.random.default_rng(45)
        pts = [cell(g[(m, ds)], rng, nb=NB) | {"model": m, "ds": ds} for m in M7 for ds in SYN_DS]
        for sub, sel in (("21", pts), ("14_no_syn431", [p for p in pts if p["ds"] != "syn431"])):
            f = fit_plus(sel)
            res[f"{nm}|{sub}"] = f
            lines.append(f"FIT {nm:26s} {sub:13s} n {f['n_points']} r {f['pearson_r']:.3f} pt {np.round(f['r_ci_point'], 3).tolist()} "
                         f"model {np.round(f['r_ci_model'], 3).tolist()} item+model {np.round(f['r_ci_item_model'], 3).tolist()} | slope {f['slope']:.3f} "
                         f"pt {np.round(f['slope_ci_point'], 3).tolist()} model {np.round(f['slope_ci_model'], 3).tolist()} item+model "
                         f"{np.round(f['slope_ci_item_model'], 3).tolist()} | intercept {f['intercept']:+.2f} pt {np.round(f['intercept_ci_point'], 2).tolist()} "
                         f"model {np.round(f['intercept_ci_model'], 2).tolist()} item+model {np.round(f['intercept_ci_item_model'], 2).tolist()} "
                         f"| spearman {f['spearman']:.3f} | sign {f['sign_agree_x_y']}/{f['n_points']} | mean(y-x) {f['mean_y_minus_x']:+.2f}")
    k, u = res["fig5_conclusion_kept_v3|21"], res["s13_uniform_removed|21"]
    ok = (round(k["pearson_r"], 3) == 0.931 and round(k["slope"], 3) == 0.871 and round(u["slope"], 3) == 0.745
          and round(u["intercept"], 2) == -4.55)
    lines.append(f"SANITY fits: Fig. 5 r {k['pearson_r']:.3f} (0.931) slope {k['slope']:.3f} (0.871) intercept {k['intercept']:+.2f} (-1.40); "
                 f"Tab. S13 slope {u['slope']:.3f} (0.745) intercept {u['intercept']:+.2f} (-4.55) r {u['pearson_r']:.3f} (0.885) -> {'OK' if ok else 'MISMATCH'}")
    res["_sanity_ok"] = ok
    return res


# ============================================================================= (3) agreement

def agreement(lines: list[str]) -> dict:
    cells = json.load(open(B / "e45_uniform_two_channel.json"))["cells"]
    out = {}
    for m in M7:
        for ds in SYN_DS:
            c = cells[f"open|{m}|{ds}"]
            out[f"{m}|{ds}"] = {k: c.get(k) for k in ("aT", "aF", "aT_ci", "aF_ci", "n_true", "n_false", "coverage", "W", "A")}
    aT = [v["aT"] for v in out.values()]
    aF = [v["aF"] for v in out.values()]
    lines.append(f"AGREEMENT (conclusion removed, 21 cells): aT {min(aT):.1f}-{max(aT):.1f}, aF {min(aF):.1f}-{max(aF):.1f}; "
                 + "; ".join(f"{k} {v['aT']:.1f}/{v['aF']:.1f}" for k, v in out.items()))
    return out


def run(lines: list[str]) -> dict:
    return {"_doc": __doc__, "main_pool": main_pool(lines), "fits": fits(lines), "agreement": agreement(lines)}
