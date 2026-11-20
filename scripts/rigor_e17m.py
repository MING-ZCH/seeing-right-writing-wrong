"""Exp-17m CPU analyses (round-9 Opus).

steer_dc  D-C+ and D-C- accuracy by sign source vs random at each dose (steering_dose.json): is there sign specificity on
          Direct-wrong items?
detector  cue set, three probed models: fall back to Direct when the within-claim probe at r (layer 20, CV) decodes a
          relation that disagrees with the written first claim (A), or when Direct and CoT disagree, pick the answer
          consistent with the probe (B); net accuracy vs CoT and vs Direct over all items (all four D x C cells),
          twin-pair bootstrap. Also the Direct-position probe used as the answer (an in-distribution supervised ceiling).
minors    m1 P0 >= 0.5 shares and denominators; m6 ladder accuracies from one estimator; m14 BLINK depth item overlap
          Qwen2.5-VL-3B vs Molmo-7B-D.
Usage: python rigor_e17m.py --only steer_dc,detector,minors
"""
from __future__ import annotations
import vtax_paths as _vtp  # noqa: E402

import argparse
import json
from pathlib import Path

import numpy as np

from build_bundle import pe, rj

O = Path(f"{_vtp.OUTPUTS}")
B = Path("outputs/bundle")
RNG = np.random.default_rng(0)


def boot(v, g, n=2000):
    u = np.unique(g)
    idx = {x: np.where(g == x)[0] for x in u}
    bs = [v[np.concatenate([idx[x] for x in RNG.choice(u, len(u))])].mean() for _ in range(n)]
    return {"mean": round(100 * float(v.mean()), 1), "ci": [round(100 * float(np.quantile(bs, .025)), 1), round(100 * float(np.quantile(bs, .975)), 1)], "n": int(len(v))}


def steer_dc(lines):
    sd = json.load(open(B / "steering_dose.json"))
    res = []
    for row in sd["doses"]:
        for s in ("direct_answer", "direct_probe", "rprobe"):
            if s not in row:
                continue
            r = {"k": row["k"], "sign": s}
            for g in ("D-C+", "D-C-", "D+C-", "D+C+"):
                r[g] = {"sign": row[s][g]["mean"], "random": row["random_mean"][g]["mean"], "diff": row[f"{s}_minus_random"][g]["mean"],
                        "ci": row[f"{s}_minus_random"][g]["ci"], "n": row[s][g]["n"]}
            res.append(r)
            lines.append(f"STEER-DC k={row['k']:3d} {s:13s} " + " | ".join(f"{g} {r[g]['sign']} vs rnd {r[g]['random']} ({r[g]['diff']:+.1f} {r[g]['ci']})" for g in ("D-C+", "D-C-")))
    return res


def detector(lines):
    res = {}
    for m in ("qwen3vl_8b", "molmo7b", "qwen7b"):
        p = O / "e17" / f"{m}_m1_items.jsonl"
        if not p.exists():
            continue
        R = rj(p)
        g = np.array([r["pair"] for r in R])
        d_ok = np.array([r["direct_ok"] for r in R], float)
        c_ok = np.array([r["cot_ok"] for r in R], float)
        y = np.array([r["y"] for r in R])
        dx = np.array([r["direct_says_x"] for r in R])
        cx = np.array([r["cot_says_x"] for r in R])
        has = np.array([r["claim_x"] is not None and r["pred_r"] >= 0 for r in R])
        claim = np.array([r["claim_x"] if r["claim_x"] is not None else -1 for r in R])
        pr = np.array([r["pred_r"] for r in R])
        conf = np.array([abs(r["p_r"] - 0.5) if r["p_r"] is not None else 0 for r in R])
        flagA = has & (pr != claim)
        accA = np.where(flagA, d_ok, c_ok)
        # B: when Direct and CoT disagree, take the one whose implied relation matches the probe
        disagree = dx != cx
        pick_direct = disagree & has & (dx == pr)
        accB = np.where(pick_direct, d_ok, c_ok)
        # C: A with confidence gate (flag only if |p - 0.5| > 0.25)
        flagC = flagA & (conf > 0.25)
        accC = np.where(flagC, d_ok, c_ok)
        grp = np.where(d_ok > 0, np.where(c_ok > 0, "D+C+", "D+C-"), np.where(c_ok > 0, "D-C+", "D-C-"))
        out = {"n": len(R), "direct": boot(d_ok, g), "cot": boot(c_ok, g)}
        for name, acc, flag in (("A_claim_vs_probe", accA, flagA), ("B_disagree_plus_probe", accB, pick_direct), ("C_A_conf>0.25", accC, flagC)):
            out[name] = {"acc": boot(acc, g), "vs_cot": boot(acc - c_ok, g), "vs_direct": boot(acc - d_ok, g),
                         "flag_rate": round(100 * float(flag.mean()), 1),
                         "by_cell": {cl: {"flagged": int(flag[grp == cl].sum()), "n": int((grp == cl).sum()),
                                          "acc_after": round(100 * float(acc[grp == cl].mean()), 1)} for cl in ("D+C-", "D+C+", "D-C+", "D-C-")}}
        res[m] = out
        lines.append(f"DETECT {m:12s} Direct {out['direct']['mean']} CoT {out['cot']['mean']} | " + " | ".join(
            f"{k}: {v['acc']['mean']} (vs CoT {v['vs_cot']['mean']:+.1f} {v['vs_cot']['ci']}, vs Direct {v['vs_direct']['mean']:+.1f} {v['vs_direct']['ci']}; flag {v['flag_rate']}%)"
            for k, v in out.items() if k.startswith(("A_", "B_", "C_"))))
    return res


def minors(lines):
    nb = json.load(open(B / "numbers.json"))
    m1 = {m: {"share_P0_ge_half": round(100 - nb[m]["error_onset"]["D+C-"]["P0_below_half"], 1), "n_DCm": nb[m]["error_onset"]["D+C-"]["n"]}
          for m in nb if nb[m].get("error_onset", {}).get("D+C-")}
    vals = [v["share_P0_ge_half"] for v in m1.values()]
    lines.append(f"M1 P0>=0.5 range {min(vals)}-{max(vals)}: " + ", ".join(f"{m} {v['share_P0_ge_half']} (n={v['n_DCm']})" for m, v in m1.items()))
    # m6: ladder accuracies, one estimator (argmax over A/B of the stored log-probs vs gold, both phrasings)
    m6 = {}
    for s in ("2b", "4b", "32b"):
        acc = {}
        for c in ("direct", "cot"):
            R = [r for f in (f"qwen3vl_{s}.jsonl", f"qwen3vl_{s}_behind.jsonl") for r in rj(O / "scale" / f) if r["condition"] == c]
            acc[c] = round(100 * float(np.mean([pe(r) == r["gold"] for r in R])), 1)
        m6[s] = acc
    R8 = [r for r in rj(O / "e10" / "qwen3vl_8b.jsonl") if r["task"].startswith("occ")]
    m6["8b"] = {c: round(100 * float(np.mean([pe(r) == r["gold"] for r in R8 if r["condition"] == c])), 1) for c in ("direct", "cot")}
    lines.append(f"M6 ladder acc (argmax vs gold, both phrasings) {m6}")
    # m14: BLINK depth item overlap
    import heldout_breakdown as HB
    rows = {m: {r["item"]: r for r in HB.model_rows(m) if r["task"] == "blink_depth"} for m in ("qwen3b", "molmo7b")}
    ks = sorted(set(rows["qwen3b"]) & set(rows["molmo7b"]))
    m14 = {}
    for c, key in (("direct", "pd"), ("cot", "pc")):
        a = np.array([rows["qwen3b"][k][key] == rows["qwen3b"][k]["gold"] for k in ks])
        b = np.array([rows["molmo7b"][k][key] == rows["molmo7b"][k]["gold"] for k in ks])
        same_pred = np.mean([rows["qwen3b"][k][key] == rows["molmo7b"][k][key] for k in ks])
        m14[c] = {"acc_qwen3b": round(100 * a.mean(), 1), "acc_molmo": round(100 * b.mean(), 1), "both_correct": int((a & b).sum()), "only_qwen3b": int((a & ~b).sum()),
                  "only_molmo": int((~a & b).sum()), "both_wrong": int((~a & ~b).sum()), "same_prediction_pct": round(100 * float(same_pred), 1), "n": len(ks)}
    lines.append(f"M14 BLINK depth overlap {m14}")
    return {"m1": m1, "m6": m6, "m14": m14}


def rp26(lines):
    """r-probe (anti-oracle on these items) vs oracle sign vs random, on the D+C- items where the L26 r-probe disagrees."""
    import glob
    import re
    runs = {}
    for f in glob.glob(str(O / "e18" / "qwen3vl_8b_mixed_*_rp26.jsonl")):
        m = re.match(r"qwen3vl_8b_mixed_(truth|random|none)_k(\d+)_L26_(\w+?)(?:_s(\d))?_rp26\.jsonl", Path(f).name)
        runs[(m.group(1), int(m.group(2)), m.group(3), m.group(4))] = {r["key"]: r for r in rj(Path(f))}
    base = runs[("none", 0, "truth", None)]
    keys = sorted(base)
    g = np.array([k.replace("_behind", "").replace("_cf", "") for k in keys])
    val = lambda R, f: np.array([float(bool(R[k][f])) for k in keys])
    res = {"n": len(keys), "none": {f: boot(val(base, f), g) for f in ("claim_correct", "answer_correct")}}
    for k in (16, 128):
        rnd = {f: np.mean([val(v, f) for (kd, kk, s_, sd), v in runs.items() if kd == "random" and kk == k], 0) for f in ("claim_correct", "answer_correct")}
        o, rp = runs[("truth", k, "truth", None)], runs[("truth", k, "rprobe", None)]
        row = {}
        for f in ("claim_correct", "answer_correct"):
            row[f] = {"oracle": boot(val(o, f), g), "rprobe": boot(val(rp, f), g), "random_mean": boot(rnd[f], g),
                      "oracle_minus_rprobe": boot(val(o, f) - val(rp, f), g), "oracle_minus_random": boot(val(o, f) - rnd[f], g),
                      "rprobe_minus_random": boot(val(rp, f) - rnd[f], g)}
        res[f"k{k}"] = row
        a = row["answer_correct"]
        lines.append(f"RP26 k={k} n={len(keys)} answers: oracle {a['oracle']['mean']} rprobe {a['rprobe']['mean']} random {a['random_mean']['mean']} none {res['none']['answer_correct']['mean']} | "
                     f"oracle-rprobe {a['oracle_minus_rprobe']['mean']:+.1f} {a['oracle_minus_rprobe']['ci']} | claims: oracle {row['claim_correct']['oracle']['mean']} rprobe {row['claim_correct']['rprobe']['mean']} "
                     f"random {row['claim_correct']['random_mean']['mean']} oracle-rprobe {row['claim_correct']['oracle_minus_rprobe']['mean']:+.1f} {row['claim_correct']['oracle_minus_rprobe']['ci']}")
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="steer_dc,detector,minors")
    a = ap.parse_args()
    lines = []
    fn = {"steer_dc": steer_dc, "detector": detector, "minors": minors, "rp26": rp26}
    out = json.load(open(B / "e17m.json")) if (B / "e17m.json").exists() else {}
    for k in a.only.split(","):
        out[k] = fn[k](lines)
    json.dump(out, open(B / "e17m.json", "w"), indent=1)
    old = (B / "rigor_e17m.txt").read_text().splitlines() if (B / "rigor_e17m.txt").exists() else []
    pref = {"steer_dc": "STEER-DC", "detector": "DETECT", "minors": ("M1 ", "M6 ", "M14 "), "rp26": "RP26"}
    keep = [x for x in old if not any(x.startswith(pref[k]) for k in a.only.split(","))]
    (B / "rigor_e17m.txt").write_text("\n".join(keep + lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
