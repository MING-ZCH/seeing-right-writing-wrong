"""Exp-17l CPU analyses (round-8 Opus).

probe_image  Qwen3-VL-8B / Molmo-7B-D at r (layer 20): original within-claim balanced accuracy, gray refit, TF-IDF text,
             conflict test (right claims + twin image), image-carried share = (orig - gray) / (orig - 0.5);
             readout alignment vs accuracy (to show it is redundant).
commit       commit-sentence audit denominators, no-claim shares, worst-case bounds, commit sentence types, and the
             relational-only D+C+ comparator.
cvb          CV-Bench 3D Depth: per-model CoT - Direct with CIs, cross-model mean (item bootstrap) with / without Phi,
             and the same for d' (A = signal).
steer_facts  sign disagreement with the true label for each non-oracle sign; sigma and residual norm per layer.

Usage: python rigor_e17l.py --only probe_image,commit,cvb,steer_facts
"""
from __future__ import annotations
import vtax_paths as _vtp  # noqa: E402

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.stats import norm

import build_bundle as BB
from build_bundle import rj

O = Path(f"{_vtp.OUTPUTS}")
L = Path(f"{_vtp.LEGACY_RESULTS}")
B = Path("outputs/bundle")
M7 = ["qwen3b", "qwen7b", "internvl3_8b", "phi35v", "molmo7b", "qwen3vl_8b", "internvl35_8b"]
RNG = np.random.default_rng(0)


def probe_image(lines):
    pc = json.load(open(B / "probe_controls.json"))
    res = {}
    for m in ("qwen3vl_8b", "molmo7b", "qwen7b"):
        al = json.load(open(O / "e17" / f"{m}_align.json"))
        o, g, t = pc[m]["orig"]["r"], pc[m]["gray_refit"]["r"], pc[m]["baselines"]["text"]
        tw = al["twin_conditional"]["right_claims_orig_correct_follow_twin"]
        share = (o["bal"] - g["bal"]) / (o["bal"] - 0.5)
        res[m] = {"orig_r": {"bal": o["bal"], "ci": o["bal_ci"]}, "gray_refit_r": {"bal": g["bal"], "ci": g["bal_ci"]},
                  "text_tfidf": {"bal": t["bal"], "ci": t["bal_ci"]}, "conflict_right_claim_twin_follow_image": tw,
                  "image_share_of_above_chance": round(float(share), 2),
                  "direct_acc": al["behaviour"]["direct_acc"], "direct_alignment": al["alignment"]["direct_answer_vs_probe_d"]["mean"],
                  "cot_acc": al["behaviour"]["cot_acc"], "cot_alignment_a": al["alignment"]["cot_answer_vs_plain_a"]["mean"]}
        r = res[m]
        lines.append(f"PROBE-IMG {m:12s} orig {o['bal']} {o['bal_ci']} | gray {g['bal']} {g['bal_ci']} | text {t['bal']} {t['bal_ci']} | conflict {tw['mean']} {tw['ci']} "
                     f"| image share {r['image_share_of_above_chance']} | Direct acc {r['direct_acc']} vs alignment {r['direct_alignment']}")
    return res


def commit(lines):
    from rigor_e17 import stype
    J = {j: {(r["model"], r["item_id"], r["group"]): r for r in rj(O / "e20" / f"{j}.jsonl")} for j in ("claude-opus-5-5", "gpt-6-astra")}
    JR = {j: {(r["model"], r["item_id"], r["group"]): r for r in rj(O / "e20" / f"{j}_relcomp.jsonl")} for j in ("claude-opus-5-5", "gpt-6-astra")
          if (O / "e20" / f"{j}_relcomp.jsonl").exists()}
    res = {}
    for m in M7 + ["pooled"]:
        row = {}
        for g, src in (("D+C-", J), ("D+C+", J), ("D+C+rel", JR)):
            if len(src) < 2:
                continue
            ks = [k for k in set(src["claude-opus-5-5"]) & set(src["gpt-6-astra"]) if k[2] == g and (m == "pooled" or k[0] == m)]
            if not ks:
                continue
            lab = lambda j, k: src[j][k]["label"]
            cons_inc = [k for k in ks if lab("claude-opus-5-5", k) == lab("gpt-6-astra", k) == "INCORRECT"]
            cons_cor = [k for k in ks if lab("claude-opus-5-5", k) == lab("gpt-6-astra", k) == "CORRECT"]
            judg = len(cons_inc) + len(cons_cor)
            types = [stype(src["gpt-6-astra"][k]["sentence"]) for k in ks]
            r = {"n": len(ks), "judgeable": judg, "consensus_incorrect": round(100 * len(cons_inc) / max(1, judg), 1),
                 "noclaim_gpt": round(100 * np.mean([lab("gpt-6-astra", k) == "NOCLAIM" for k in ks]), 1),
                 "noclaim_opus": round(100 * np.mean([lab("claude-opus-5-5", k) == "NOCLAIM" for k in ks]), 1),
                 "lower_bound_unjudgeable_consistent": round(100 * len(cons_inc) / len(ks), 1),
                 "upper_bound_unjudgeable_contradicting": round(100 * (len(ks) - len(cons_cor)) / len(ks), 1),
                 "type_relation": round(100 * np.mean([t == "relation" for t in types]), 1),
                 "type_conclusion": round(100 * np.mean([t == "conclusion" for t in types]), 1)}
            rel = [k for k, t in zip(ks, types) if t == "relation"]
            ri = [k for k in rel if lab("claude-opus-5-5", k) == lab("gpt-6-astra", k) == "INCORRECT"]
            rc = [k for k in rel if lab("claude-opus-5-5", k) == lab("gpt-6-astra", k) == "CORRECT"]
            r["relational_only_consensus_incorrect"] = round(100 * len(ri) / max(1, len(ri) + len(rc)), 1)
            r["relational_only_judgeable"] = len(ri) + len(rc)
            row[g] = r
        res[m] = row
        lines.append(f"COMMIT {m:14s} " + " | ".join(f"{g}: {r['judgeable']}/{r['n']} judgeable, cons {r['consensus_incorrect']} (bounds {r['lower_bound_unjudgeable_consistent']}"
                                                       f"-{r['upper_bound_unjudgeable_contradicting']}), noclaim GPT {r['noclaim_gpt']} Opus {r['noclaim_opus']}, "
                                                       f"relational {r['type_relation']}% concl {r['type_conclusion']}%, rel-only {r['relational_only_consensus_incorrect']} ({r['relational_only_judgeable']})"
                                                       for g, r in row.items()))
    return res


def dprime(py, gy):
    h = (py[gy].sum() + .5) / (gy.sum() + 1)
    f = (py[~gy].sum() + .5) / ((~gy).sum() + 1)
    return norm.ppf(h) - norm.ppf(f)


def cvb(lines):
    nb = json.load(open(B / "numbers.json"))
    models = [m for m in BB.MODELS if m in nb]
    per = {}
    data = {}
    for m in models:
        e = nb[m].get("behavioral", {}).get("task:cvb_depth", {}).get("cot_minus_direct")
        if not e:
            continue
        per[m] = {"diff": e["mean"], "ci": [e["lo"], e["hi"]], "sig_neg": e["hi"] < 0, "sig_pos": e["lo"] > 0}
        main = BB.load_main(O, L, m)
        data[m] = {k: (main["direct"][k], main["cot"][k]) for k, v in main["direct"].items() if v[3] == "cvb_depth" and k in main.get("cot", {})}
    common = sorted(set.intersection(*[set(v) for v in data.values()]))
    grp = np.array([data[models[0]][k][0][2] for k in common])
    res = {"per_model": per, "n_items": len(common), "n_sig_neg": sum(v["sig_neg"] for v in per.values()), "n_models": len(per)}
    for label, ms in (("all", list(data)), ("no_phi", [m for m in data if m != "phi35v"])):
        acc = np.array([[data[m][k][1][0] - data[m][k][0][0] for k in common] for m in ms])  # models x items
        py_d = {m: np.array([data[m][k][0][4] == "A" for k in common]) for m in ms}
        py_c = {m: np.array([data[m][k][1][4] == "A" for k in common]) for m in ms}
        gy = {m: np.array([(data[m][k][0][4] == "A") == bool(data[m][k][0][0]) for k in common]) for m in ms}
        u = np.unique(grp)
        idx = {g: np.where(grp == g)[0] for g in u}
        dd = lambda s: np.mean([dprime(py_c[m][s], gy[m][s]) - dprime(py_d[m][s], gy[m][s]) for m in ms])
        bs_a, bs_d = [], []
        for _ in range(2000):
            s = np.concatenate([idx[g] for g in RNG.choice(u, len(u))])
            bs_a.append(acc[:, s].mean())
            bs_d.append(dd(s))
        full = np.arange(len(common))
        res[label] = {"models": ms, "acc_diff": round(100 * float(acc.mean()), 1), "acc_ci": [round(100 * float(np.quantile(bs_a, .025)), 1), round(100 * float(np.quantile(bs_a, .975)), 1)],
                      "dprime_diff": round(float(dd(full)), 2), "dprime_ci": [round(float(np.quantile(bs_d, .025)), 2), round(float(np.quantile(bs_d, .975)), 2)],
                      "per_model_dprime_diff": {m: round(float(dprime(py_c[m], gy[m]) - dprime(py_d[m], gy[m])), 2) for m in ms}}
        lines.append(f"CVB {label:7s} n={len(common)} models={len(ms)} acc {res[label]['acc_diff']:+.1f} {res[label]['acc_ci']} | d' {res[label]['dprime_diff']:+.2f} {res[label]['dprime_ci']}")
    lines.append(f"CVB per-model sig. negative {res['n_sig_neg']}/{res['n_models']}: " + ", ".join(f"{m} {v['diff']:+.1f} {v['ci']}" for m, v in per.items()))
    hb = json.load(open(B / "heldout_breakdown.json"))
    res["natural_sdt_cvb"] = {m: hb[m].get("natural_sdt", {}).get("cvb_depth") for m in hb if hb[m].get("natural_sdt")}
    return res


def steer_facts(lines):
    mixed = json.load(open(O / "e17" / "meta" / "qwen3vl_8b_mixed_items.json"))
    res = {}
    for sign in ("direct_answer_y", "direct_probe_y", "rprobe_y"):
        row = {}
        for g in ("all", "D+C-", "D+C+", "D-C-", "D-C+"):
            it = [v for v in mixed.values() if g == "all" or v["group"] == g]
            row[g] = {"differ": int(sum(v[sign] != v["y"] for v in it)), "n": len(it)}
        res[sign] = row
        lines.append(f"SIGN {sign:16s} " + " | ".join(f"{g} {r['differ']}/{r['n']}" for g, r in row.items()))
    res["scale"] = {"L20": {"sigma": 0.43, "resid_norm_median": 97.6, "k128_alpha_over_norm": round(128 * 0.43 / 97.6, 2)},
                    "L26": {"sigma": 1.06, "resid_norm_median": 258.8, "k128_alpha_over_norm": round(128 * 1.06 / 258.8, 2)}}
    lines.append(f"SCALE {res['scale']}")
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="probe_image,commit,cvb,steer_facts")
    a = ap.parse_args()
    lines = []
    fn = {"probe_image": probe_image, "commit": commit, "cvb": cvb, "steer_facts": steer_facts}
    out = json.load(open(B / "e17l.json")) if (B / "e17l.json").exists() else {}
    for k in a.only.split(","):
        out[k] = fn[k](lines)
    json.dump(out, open(B / "e17l.json", "w"), indent=1)
    old = (B / "rigor_e17l.txt").read_text().splitlines() if (B / "rigor_e17l.txt").exists() else []
    pref = {"probe_image": "PROBE-IMG", "commit": "COMMIT", "cvb": "CVB", "steer_facts": ("SIGN", "SCALE")}
    keep = [x for x in old if not any(x.startswith(pref[k]) for k in a.only.split(","))]
    (B / "rigor_e17l.txt").write_text("\n".join(keep + lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
