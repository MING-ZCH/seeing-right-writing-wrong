"""Exp-17i analyses (round-5 queue).

1. common_scale: per model on the cue set, probit cue weights b (occlusion, vertical, size) for Direct and CoT;
   common scale s = <b_cot, b_dir> / <b_dir, b_dir> (best uniform attenuation) and per-cue deviation
   delta_k = b_cot,k - s * b_dir,k (delta_occ < 0: occlusion lost beyond uniform attenuation); twin-pair bootstrap.
   Also cue-set accuracy Direct vs CoT.
2. syn_overlap: Synthetic Depth items classified by image analysis (contact fraction of a shape's boundary lying
   within 3 px of the other shape's fill, and visible area): overlap / touch-or-gap / near-total occlusion;
   Direct/CoT accuracy and d' and the judge first-claim contrast on visibly overlapping items only.
3. parser: cue-claim and affirmation shares re-run with claim_parse.first_claim_v2 / v3.
4. heldout_extra_means: cross-model mean paired differences on BLINK Spatial Relation and 3DSR depth items.
5. mean_r: mean |R| tokens on the main pool for Qwen3-VL-8B and InternVL3.5-8B.

Usage: python rigor_e17i.py [--only common,syn,parser,heldout,meanr]
"""
from __future__ import annotations
import vtax_paths as _vtp  # noqa: E402

import argparse
import json
import re
from pathlib import Path

import numpy as np
from scipy.stats import norm

import build_bundle as BB
from build_bundle import rj

O = Path(f"{_vtp.OUTPUTS}")
SH = f"{_vtp.DATA_ROOT}"
L = Path(f"{_vtp.LEGACY_RESULTS}")
B = Path("outputs/bundle")
M7 = ["qwen3b", "qwen7b", "internvl3_8b", "phi35v", "molmo7b", "qwen3vl_8b", "internvl35_8b"]
RNG = np.random.default_rng(0)
q = lambda v: [round(float(np.quantile(v, .025)), 3), round(float(np.quantile(v, .975)), 3)]


# ============================================================================= 1. common scale

def common_scale(lines: list[str]) -> dict:
    from rigor_e17f import glm_rows, probit_fit
    res = {}
    for m in M7:
        R = rj(O / "e10" / f"{m}_cues.jsonl")
        data = {c: glm_rows([r for r in R if r["condition"] == c], use_size=True) for c in ("direct", "cot")}
        pairs = sorted(set(data["direct"]["pair"]))
        pidx = {c: {p: np.where(np.array(d["pair"]) == p)[0] for p in pairs} for c, d in data.items()}

        def stats(sel):
            bd = probit_fit(data["direct"]["X"][sel["direct"]], data["direct"]["y"][sel["direct"]])[2:5]
            bc = probit_fit(data["cot"]["X"][sel["cot"]], data["cot"]["y"][sel["cot"]])[2:5]
            s = float(bc @ bd / (bd @ bd))
            dl = bc - s * bd
            acc = {c: float(np.mean(data[c]["y"][sel[c]] == data[c]["truth"][sel[c]])) for c in data}
            return [s, *dl, bc[0] / bd[0], bc[1] / bd[1], acc["cot"] - acc["direct"], acc["direct"], acc["cot"]]

        full = {c: np.arange(len(d["y"])) for c, d in data.items()}
        est = stats(full)
        bs = []
        for _ in range(300):
            js = RNG.integers(0, len(pairs), len(pairs))
            bs.append(stats({c: np.concatenate([pidx[c][pairs[j]] for j in js]) for c in data}))
        bs = np.array(bs)
        names = ["s", "delta_occlusion", "delta_vertical", "delta_size", "retain_occlusion", "retain_vertical", "acc_cot_minus_direct",
                 "acc_direct", "acc_cot"]
        res[m] = {n: {"est": round(e, 3) if "acc" not in n else round(100 * e, 4),
                      "ci": q(bs[:, i]) if "acc" not in n else [round(100 * v, 1) for v in q(bs[:, i])]} for i, (n, e) in enumerate(zip(names, est))}
        r = res[m]
        lines.append(f"COMMON {m:14s} s {r['s']['est']:.2f} {r['s']['ci']} | d_occ {r['delta_occlusion']['est']:+.2f} {r['delta_occlusion']['ci']} "
                     f"d_vert {r['delta_vertical']['est']:+.2f} {r['delta_vertical']['ci']} d_size {r['delta_size']['est']:+.2f} {r['delta_size']['ci']} "
                     f"| retain occ {r['retain_occlusion']['est']:.2f} vert {r['retain_vertical']['est']:.2f} | acc {r['acc_direct']['est']}->{r['acc_cot']['est']} "
                     f"({r['acc_cot_minus_direct']['est']:+.1f} {r['acc_cot_minus_direct']['ci']})")
    n_neg = sum(res[m]["delta_occlusion"]["ci"][1] < 0 for m in res)
    n_pos = sum(res[m]["delta_occlusion"]["ci"][0] > 0 for m in res)
    res["_summary"] = {"delta_occ_ci_below0": n_neg, "delta_occ_ci_above0": n_pos,
                       "s_range": [min(res[m]["s"]["est"] for m in M7), max(res[m]["s"]["est"] for m in M7)]}
    lines.append(f"COMMON summary {res['_summary']}")
    return res


# ============================================================================= 2. Synthetic Depth overlap

def syn_overlap_metrics() -> list[dict]:
    cache = B / "syn_overlap_metrics.json"
    if cache.exists():
        return json.load(open(cache))
    from PIL import Image
    from scipy.ndimage import binary_dilation, binary_erosion
    d = json.load(open(f"{SH}/datasets/spatial_probing/depth_ordering.json"))
    pal = {k: np.array(v) for k, v in json.load(open(B / "syn_depth_geometry.json"))["palette"].items()}
    qrx = re.compile(r"Is the (\w+) (\w+) in front of the (\w+) (\w+)\?")
    out = []
    for i, ex in enumerate(d):
        im = np.asarray(Image.open(ex["image_path"].replace("<DATA_ROOT>", SH)).convert("RGB")).astype(int)
        qm = qrx.search(ex["question"])
        ms = [np.abs(im - pal[c]).sum(-1) <= 30 for c in (qm.group(1), qm.group(3))]
        fr = []
        for a, b in ((0, 1), (1, 0)):
            bd = ms[a] & ~binary_erosion(ms[a], np.ones((3, 3), bool))
            fr.append(float((bd & binary_dilation(ms[b], np.ones((7, 7), bool))).sum() / max(1, bd.sum())))
        out.append({"uid": f"syn_depth_{i}", "contact": round(max(fr), 3), "px": [int(ms[0].sum()), int(ms[1].sum())], "answer": ex["answer"]})
    json.dump(out, open(cache, "w"))
    return out


def category(r: dict, thr: float = 0.15, min_px: int = 100) -> str:
    if r["contact"] < thr:
        return "touch_or_gap"
    return "overlap" if min(r["px"]) >= min_px else "near_total"


def sdt(py: np.ndarray, gy: np.ndarray) -> float:
    h = (py[gy].sum() + .5) / (gy.sum() + 1)
    f = (py[~gy].sum() + .5) / ((~gy).sum() + 1)
    return float(norm.ppf(h) - norm.ppf(f))


def syn_overlap(lines: list[str]) -> dict:
    met = syn_overlap_metrics()
    cat = {r["uid"]: category(r) for r in met}
    counts = {c: sum(v == c for v in cat.values()) for c in ("overlap", "touch_or_gap", "near_total")}
    flagged = {u: (cat[u], next(r for r in met if r["uid"] == u)["contact"]) for u in
               ("syn_depth_471", "syn_depth_16", "syn_depth_107", "syn_depth_58", "syn_depth_96")}
    res = {"counts": counts, "share_overlap": round(100 * counts["overlap"] / len(met), 1), "flagged": flagged,
           "rule": "contact = share of a shape's boundary pixels within 3 px of the other shape's fill (max over the two shapes); "
                   "overlap: contact >= 0.15 and both shapes >= 100 visible px; near_total: contact >= 0.15 and one shape < 100 px",
           "models": {}}
    lines.append(f"SYN-OVERLAP counts {counts} share {res['share_overlap']}% flagged {flagged}")
    for m in BB.MODELS:
        main = BB.load_main(O, L, m)
        if "direct" not in main or "cot" not in main:
            continue
        keys = sorted(k for k, v in main["direct"].items() if v[3] == "syn_depth" and k in main["cot"])
        row = {}
        for sub, ks in (("all", keys), ("overlap", [k for k in keys if cat.get(k) == "overlap"]),
                        ("touch_or_gap", [k for k in keys if cat.get(k) == "touch_or_gap"])):
            d = np.array([main["direct"][k][0] for k in ks])
            c = np.array([main["cot"][k][0] for k in ks])
            pyd = np.array([main["direct"][k][4] == "A" for k in ks])
            pyc = np.array([main["cot"][k][4] == "A" for k in ks])
            gy = np.array([(main["direct"][k][4] == "A") == bool(main["direct"][k][0]) for k in ks])
            bs = []
            for _ in range(1000):
                i = RNG.integers(0, len(ks), len(ks))
                bs.append((c[i].mean() - d[i].mean(), sdt(pyc[i], gy[i]) - sdt(pyd[i], gy[i])))
            bs = np.array(bs)
            row[sub] = {"n": len(ks), "acc_direct": round(100 * d.mean(), 1), "acc_cot": round(100 * c.mean(), 1),
                        "acc_diff": round(100 * (c.mean() - d.mean()), 1), "acc_diff_ci": [round(100 * v, 1) for v in q(bs[:, 0])],
                        "dprime_direct": round(sdt(pyd, gy), 2), "dprime_cot": round(sdt(pyc, gy), 2),
                        "dprime_diff": round(sdt(pyc, gy) - sdt(pyd, gy), 2), "dprime_diff_ci": [round(v, 2) for v in q(bs[:, 1])]}
        res["models"][m] = row
        o, a = row["overlap"], row["all"]
        lines.append(f"SYN-OVERLAP {m:16s} all n={a['n']} acc {a['acc_direct']}->{a['acc_cot']} ({a['acc_diff']:+.1f}) d' {a['dprime_direct']}->{a['dprime_cot']} "
                     f"| overlap n={o['n']} acc {o['acc_direct']}->{o['acc_cot']} ({o['acc_diff']:+.1f} {o['acc_diff_ci']}) d' {o['dprime_direct']}->{o['dprime_cot']} "
                     f"({o['dprime_diff']:+.2f} {o['dprime_diff_ci']}) | touch n={row['touch_or_gap']['n']} acc {row['touch_or_gap']['acc_direct']}->{row['touch_or_gap']['acc_cot']}")
    # judge first-claim contrast on overlapping items
    jr = []
    for fn in ("e2_judge.jsonl", "e2_judge2.jsonl"):
        for r in map(json.loads, open(O / fn)):
            jr.append(r | {"jn": r.get("judge_name") or "claude-opus-5-5"})
    by = {}
    for r in jr:
        if r["item_id"].startswith("syn_depth"):
            by.setdefault((r["model"], r["item_id"], r["group"]), {})[r["jn"]] = r["judge"]
    jt = {}
    for sub in ("all", "overlap"):
        for m in M7:
            for g in ("D+C-", "D+C+"):
                its = [(k, v) for k, v in by.items() if k[0] == m and k[2] == g and (sub == "all" or cat.get(k[1]) == "overlap")]
                cons = [v for _, v in its if v.get("claude-opus-5-5") in ("correct", "incorrect") and v.get("gpt-6-astra") == v.get("claude-opus-5-5")]
                opus = [v["claude-opus-5-5"] for _, v in its if v.get("claude-opus-5-5") in ("correct", "incorrect")]
                jt[f"{sub}|{m}|{g}"] = {"n": len(its), "opus_incorrect": round(100 * np.mean([x == "incorrect" for x in opus]), 1) if opus else None,
                                        "consensus_incorrect": round(100 * np.mean([v["claude-opus-5-5"] == "incorrect" for v in cons]), 1) if cons else None,
                                        "n_consensus": len(cons)}
    res["judge"] = jt
    for sub in ("all", "overlap"):
        dm = [jt[f"{sub}|{m}|D+C-"]["consensus_incorrect"] for m in M7 if jt[f"{sub}|{m}|D+C-"]["consensus_incorrect"] is not None]
        dp = [jt[f"{sub}|{m}|D+C+"]["consensus_incorrect"] for m in M7 if jt[f"{sub}|{m}|D+C+"]["consensus_incorrect"] is not None]
        res[f"judge_range_{sub}"] = {"DCm": [min(dm), max(dm)], "DCp": [min(dp), max(dp)]}
        lines.append(f"SYN-OVERLAP judge consensus {sub}: D+C- {min(dm)}-{max(dm)} vs D+C+ {min(dp)}-{max(dp)} | n D+C- "
                     + ",".join(str(jt[f'{sub}|{m}|D+C-']['n_consensus']) for m in M7))
    return res


# ============================================================================= 3. parser re-run

def parser(lines: list[str]) -> dict:
    import rigor_e17f
    import rigor_e17g
    from claim_parse import first_claim_v2, first_claim_v3
    out = {}
    for name, f in (("v2", first_claim_v2), ("v3", first_claim_v3)):
        rigor_e17f.first_claim = f
        rigor_e17g.first_claim = f
        ll: list[str] = []
        cc = rigor_e17f.cue_claims(ll)
        af = rigor_e17g.affirmation(ll)
        out[name] = {"cue_claims": {k: v for k, v in cc.items() if k in ("cues", "twins")}, "affirmation": af}
        lines += [f"[{name}] " + x for x in ll if x.startswith(("CLAIMS cues", "AFFIRM cues", "AFFIRM syn:"))]
    return out


# ============================================================================= 4. held-out extra means

def heldout_extra(lines: list[str]) -> dict:
    import heldout_breakdown as HB
    from rigor_e17g import cluster_boot
    rows = {m: {r["item"]: r for r in HB.model_rows(m)} for m in M7}
    rows["qwen3vl_8b_think"] = {r["item"]: r for r in HB.think_rows()}
    dep = {"3dsr_location_closer_to_camera", "3dsr_orientation_in_front_of"}
    res = {}
    for sub, f in (("blink_relation", lambda t: t == "blink_relation"), ("3dsr_depth", lambda t: t in dep)):
        for label, models, key in (("cot-direct|8 open", M7 + ["qwen3vl_8b_think"], "pd"),
                                   ("cot-direct|7 (no Phi)", [m for m in M7 if m != "phi35v"] + ["qwen3vl_8b_think"], "pd"),
                                   ("cot-filler_turn|7", M7, "pf"), ("cot-filler_turn|6 (no Phi)", [m for m in M7 if m != "phi35v"], "pf")):
            items = sorted(k for k, r in rows[M7[0]].items() if f(r["task"]) and all(k in rows[m] and key in rows[m][k] for m in models))
            if not items:
                continue
            diff = np.array([np.mean([float(rows[m][k]["pc"] == rows[m][k]["gold"]) - float(rows[m][k][key] == rows[m][k]["gold"])
                                      for m in models]) for k in items])
            cb = cluster_boot({"d": diff}, [rows[M7[0]][k]["img"] for k in items], n=4000)["d"]
            res.setdefault(sub, {})[label] = cb | {"n_items": len(items), "n_models": len(models)}
            lines.append(f"HELDOUT-MEAN {sub:14s} {label:28s} n={len(items)} {cb['mean']:+.1f} {cb['ci']}")
    return res


# ============================================================================= 5. mean |R|

def mean_r(lines: list[str]) -> dict:
    res = {}
    for m in ("qwen3vl_8b", "internvl35_8b"):
        it = json.load(open(O / "e4" / f"{m}.json"))["items"]
        L_ = np.array([x["cot_len"] for x in it])
        res[m] = {"n": len(L_), "mean": round(float(L_.mean()), 1), "median": float(np.median(L_)),
                  "truncated_pct": round(100 * float(np.mean([bool(x.get("cot_truncated")) for x in it])), 1)}
        lines.append(f"MEAN-R {m} {res[m]}")
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="common,syn,parser,heldout,meanr")
    a = ap.parse_args()
    lines: list[str] = []
    fns = {"common": ("common_scale", common_scale), "syn": ("syn_overlap", syn_overlap), "parser": ("parser_v2", parser),
           "heldout": ("heldout_extra_means", heldout_extra), "meanr": ("mean_r", mean_r)}
    for k in a.only.split(","):
        name, f = fns[k]
        json.dump(f(lines), open(B / f"{name}.json", "w"), indent=1)
    old = (B / "rigor_e17i.txt").read_text().splitlines() if (B / "rigor_e17i.txt").exists() else []
    pref = {"common": "COMMON", "syn": "SYN-OVERLAP", "parser": "[v", "heldout": "HELDOUT-MEAN", "meanr": "MEAN-R"}
    keep = [x for x in old if not any(x.startswith(pref[k]) for k in a.only.split(","))]
    (B / "rigor_e17i.txt").write_text("\n".join(keep + lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
