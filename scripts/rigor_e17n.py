"""Exp-17n CPU analyses (round-10 Opus M1, Q2, m5, m9).

syn       Synthetic Depth confound.
          (a) polarity-pooled d' / c: each item is asked with the original and the object-order-swapped question (e2a cfq,
              CoT regenerated for the swapped question); response "yes" = the queried subject is in front; both prompts pooled
              (one subject-front and one subject-behind prompt per item); image-cluster bootstrap; 7 models with cfq runs.
              Also the accuracy-equivalent sensitivity loss 100 * (Phi(d'_D / 2) - Phi(d'_C / 2)) vs the single-phrasing value.
          (b) CoT - Direct accuracy by visible-area-ratio tercile x gold answer (syn_depth_geometry.json), 8 models; share of
              CoT "yes" on gold-"no" high-ratio items.
          (c) items with ratio <= 1.8 (the cue-set maximum).
twins     twin-set pooled d' change per model (dprime_ci.json).
rescue    D-C+ commit-to-gold sentences (P(gold) < 0.5 before, >= 0.5 from it on) from e2b, sentence types vs D+C- commits.
Usage: python rigor_e17n.py --only syn,twins,rescue
"""
from __future__ import annotations
import vtax_paths as _vtp  # noqa: E402

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.stats import norm

import build_bundle as BB
from build_bundle import pe, rj

O = Path(f"{_vtp.OUTPUTS}")
L = Path(f"{_vtp.LEGACY_RESULTS}")
B = Path("outputs/bundle")
M7 = ["qwen3b", "qwen7b", "internvl3_8b", "phi35v", "molmo7b", "qwen3vl_8b", "internvl35_8b"]
RNG = np.random.default_rng(0)


def dprime(resp, truth):
    h = (resp[truth].sum() + .5) / (truth.sum() + 1)
    f = (resp[~truth].sum() + .5) / ((~truth).sum() + 1)
    return float(norm.ppf(h) - norm.ppf(f)), float(-(norm.ppf(h) + norm.ppf(f)) / 2)


def parse_readout(v):
    if isinstance(v, str):
        import ast
        v = ast.literal_eval(v)
    return pe(v)


def syn(lines):
    geo = json.load(open(B / "syn_depth_geometry.json"))["items"]
    ratio = {}
    for k, g in geo.items():
        a = [v["px"] for v in g.values()]
        if min(a) > 0:
            ratio[k] = max(a) / min(a)
    rv = np.array(list(ratio.values()))
    t1, t2 = np.quantile(rv, [1 / 3, 2 / 3])
    bs = json.load(open(B / "bias_sensitivity.json"))
    res = {"ratio_terciles": [round(float(t1), 2), round(float(t2), 2)], "share_ratio_gt_1.8": round(100 * float(np.mean(rv > 1.8)), 1), "pooled": {}, "tercile": {}, "small_ratio": {}}
    for m in BB.MODELS:
        main = BB.load_main(O, L, m)
        if "direct" not in main or "cot" not in main:
            continue
        keys = [k for k, v in main["direct"].items() if v[3] == "syn_depth" and k in main["cot"]]
        gold_yes = {k: (main["direct"][k][4] == "A") == bool(main["direct"][k][0]) for k in keys}
        # (a) polarity-pooled
        cf = {r["item_id"]: r for r in rj(O / "e2a" / f"{m}.jsonl") if r["variant"] == "cfq" and r["task"] == "syn_depth"}
        if cf:
            ks = [k for k in keys if k in cf]
            rows = []  # (cond, says_yes, subject_front, cluster)
            for k in ks:
                g = cf[k]["image_id"]
                t = gold_yes[k]
                for cond, orig_pred, cf_pred in (("direct", main["direct"][k][4], parse_readout(cf[k]["readout_direct"])),
                                                 ("cot", main["cot"][k][4], parse_readout(cf[k]["readout_cot"]))):
                    rows.append((cond, orig_pred == "A", t, g))
                    rows.append((cond, cf_pred == "A", not t, g))
            out = {}
            arr = {c: (np.array([r[1] for r in rows if r[0] == c]), np.array([r[2] for r in rows if r[0] == c]), np.array([r[3] for r in rows if r[0] == c])) for c in ("direct", "cot")}
            est = {c: dprime(a[0], a[1]) for c, a in arr.items()}
            cl = np.unique(arr["direct"][2])
            idx = {c: {g: np.where(arr[c][2] == g)[0] for g in cl} for c in arr}
            bsd = []
            for _ in range(1000):
                js = RNG.choice(cl, len(cl))
                d = {c: dprime(arr[c][0][np.concatenate([idx[c][g] for g in js])], arr[c][1][np.concatenate([idx[c][g] for g in js])]) for c in arr}
                bsd.append([d["direct"][0], d["cot"][0], d["cot"][0] - d["direct"][0], d["cot"][1] - d["direct"][1]])
            bsd = np.array(bsd)
            q = lambda j: [round(float(np.quantile(bsd[:, j], .025)), 2), round(float(np.quantile(bsd[:, j], .975)), 2)]
            sens = lambda d: 100 * norm.cdf(d / 2)
            out = {"n_items": len(ks), "dprime_direct": round(est["direct"][0], 2), "dprime_direct_ci": q(0), "dprime_cot": round(est["cot"][0], 2),
                   "dprime_cot_ci": q(1), "d_dprime": round(est["cot"][0] - est["direct"][0], 2), "d_dprime_ci": q(2),
                   "c_direct": round(est["direct"][1], 2), "c_cot": round(est["cot"][1], 2), "d_c_ci": q(3),
                   "loss_from_sensitivity_pooled": round(sens(est["direct"][0]) - sens(est["cot"][0]), 1),
                   "loss_from_sensitivity_single_phrasing": bs.get(m, {}).get("cot_vs_direct", {}).get("loss_from_sensitivity"),
                   "pair_both_correct_direct": round(100 * float(np.mean([(main["direct"][k][4] == "A") == gold_yes[k] and (parse_readout(cf[k]["readout_direct"]) == "A") == (not gold_yes[k]) for k in ks])), 1),
                   "pair_both_correct_cot": round(100 * float(np.mean([(main["cot"][k][4] == "A") == gold_yes[k] and (parse_readout(cf[k]["readout_cot"]) == "A") == (not gold_yes[k]) for k in ks])), 1)}
            res["pooled"][m] = out
            lines.append(f"SYN-POOLED {m:16s} d' {out['dprime_direct']} -> {out['dprime_cot']} ({out['d_dprime']:+.2f} {out['d_dprime_ci']}) c {out['c_direct']:+.2f} -> {out['c_cot']:+.2f} "
                         f"| sens. loss pooled {out['loss_from_sensitivity_pooled']} vs single {out['loss_from_sensitivity_single_phrasing']} | pair both-correct {out['pair_both_correct_direct']} -> {out['pair_both_correct_cot']}")
        # (b) tercile x gold
        tb = {}
        for tname, lo, hi in (("low", 0, t1), ("mid", t1, t2), ("high", t2, 1e9)):
            for gname, gv in (("no", False), ("yes", True)):
                ks = [k for k in keys if k in ratio and lo <= ratio[k] < hi and gold_yes[k] == gv]
                d = np.array([main["direct"][k][0] for k in ks])
                c = np.array([main["cot"][k][0] for k in ks])
                tb[f"{tname}|{gname}"] = {"n": len(ks), "direct": round(100 * float(d.mean()), 1), "cot": round(100 * float(c.mean()), 1),
                                          "cot_yes": round(100 * float(np.mean([main["cot"][k][4] == "A" for k in ks])), 1)}
            ks = [k for k in keys if k in ratio and lo <= ratio[k] < hi]
            tb[f"{tname}|all"] = {"n": len(ks), "loss": round(100 * float(np.mean([main["cot"][k][0] - main["direct"][k][0] for k in ks])), 1)}
        res["tercile"][m] = tb
        lines.append(f"SYN-TERC {m:16s} " + " | ".join(f"{t}: no {tb[t + '|no']['direct']}->{tb[t + '|no']['cot']} yes {tb[t + '|yes']['direct']}->{tb[t + '|yes']['cot']} loss {tb[t + '|all']['loss']:+.1f}"
                                                     for t in ("low", "mid", "high")) + f" | CoT yes on high-no {tb['high|no']['cot_yes']}")
        # (c) small ratio
        ks = [k for k in keys if k in ratio and ratio[k] <= 1.8]
        d = np.array([main["direct"][k][0] for k in ks])
        c = np.array([main["cot"][k][0] for k in ks])
        g = np.array([main["direct"][k][2] for k in ks])
        u = np.unique(g)
        idx = {x: np.where(g == x)[0] for x in u}
        bsl = [(c - d)[np.concatenate([idx[x] for x in RNG.choice(u, len(u))])].mean() for _ in range(2000)]
        res["small_ratio"][m] = {"n": len(ks), "direct": round(100 * float(d.mean()), 1), "cot": round(100 * float(c.mean()), 1),
                                 "loss": round(100 * float((c - d).mean()), 1), "ci": [round(100 * float(np.quantile(bsl, .025)), 1), round(100 * float(np.quantile(bsl, .975)), 1)]}
        r = res["small_ratio"][m]
        lines.append(f"SYN-SMALL {m:16s} ratio<=1.8 n={r['n']} acc {r['direct']} -> {r['cot']} ({r['loss']:+.1f} {r['ci']})")
    return res


def twins(lines):
    dp = json.load(open(B / "dprime_ci.json"))
    res = {}
    for m, r in dp.items():
        if not isinstance(r, dict) or "dprime_cot_minus_direct" not in r:
            continue
        lo, hi = r["diff_ci"]
        res[m] = {"direct": r["dprime_direct"], "cot": r["dprime_cot"], "diff": r["dprime_cot_minus_direct"], "ci": [lo, hi],
                  "direction": "falls (CI<0)" if hi < 0 else "rises (CI>0)" if lo > 0 else ("falls (n.s.)" if r["dprime_cot_minus_direct"] < 0 else "rises (n.s.)")}
    lines.append("TWINS " + " | ".join(f"{m} {v['direct']:.2f}->{v['cot']:.2f} ({v['diff']:+.2f} [{v['ci'][0]:+.2f}, {v['ci'][1]:+.2f}]) {v['direction']}" for m, v in res.items()))
    return res


def rescue(lines):
    from rigor_e17 import stype
    res = {}
    for m in M7:
        G = BB.groups_from({}, L, O, m)
        R = rj(O / "e2b" / f"{m}.jsonl")
        out = {}
        for grp in ("D-C+", "D+C-"):
            items = []
            for r in R:
                if G.get(r["item_id"], {}).get("group") != grp or not r["sentences"]:
                    continue
                pg = [BB.probs(t.get("lp_sum") or t["lp_bare"])[r["gold"]] for t in sorted(r["traj"], key=lambda x: x["j"])]
                if grp == "D-C+":
                    js = next((j for j in range(1, len(pg)) if pg[j - 1] < 0.5 and all(v >= 0.5 for v in pg[j:])), None)
                else:
                    js = next((j for j in range(1, len(pg)) if pg[j - 1] >= 0.5 and all(v < 0.5 for v in pg[j:])), None)
                items.append((r, js, pg))
            com = [(r, js) for r, js, _ in items if js is not None]
            types = [stype(r["sentences"][js - 1]) for r, js in com]
            out[grp] = {"n": len(items), "n_commit": len(com), "p0_cross": round(100 * float(np.mean([pg[0] < 0.5 if grp == "D-C+" else pg[0] >= 0.5 for _, _, pg in items])), 1) if items else None,
                        "type_relation": round(100 * float(np.mean([t == "relation" for t in types])), 1) if types else None,
                        "type_conclusion": round(100 * float(np.mean([t == "conclusion" for t in types])), 1) if types else None,
                        "tasks": {t: sum(r["task"] == t for r, _ in com) for t in sorted({r["task"] for r, _ in com})}}
        res[m] = out
        a, b = out["D-C+"], out["D+C-"]
        lines.append(f"RESCUE {m:14s} D-C+ n={a['n']} committed {a['n_commit']} (P0<0.5 {a['p0_cross']}%) relational {a['type_relation']}% concl {a['type_conclusion']}% | D+C- relational {b['type_relation']}% concl {b['type_conclusion']}%")
    return res


def rescue_judge(lines):
    """Image check of D-C+ commit-to-gold sentences (Synthetic Depth; both judges, e20 protocol)."""
    J = {j: {(r["model"], r["item_id"]): r for r in rj(O / "e20" / f"{j}_rescue.jsonl")} for j in ("claude-opus-5-5", "gpt-6-astra")}
    cj = json.load(open(B / "commit_judge.json"))["rows"] if (B / "commit_judge.json").exists() else {}
    res = {}
    for m in M7 + ["pooled"]:
        ks = [k for k in set(J["claude-opus-5-5"]) & set(J["gpt-6-astra"]) if m == "pooled" or k[0] == m]
        if not ks:
            continue
        lab = lambda j, k: J[j][k]["label"]
        inc = [k for k in ks if lab("claude-opus-5-5", k) == lab("gpt-6-astra", k) == "INCORRECT"]
        cor = [k for k in ks if lab("claude-opus-5-5", k) == lab("gpt-6-astra", k) == "CORRECT"]
        jd = len(inc) + len(cor)
        res[m] = {"n": len(ks), "judgeable": jd, "consistent_with_image": round(100 * len(cor) / max(1, jd), 1),
                  "contradict": round(100 * len(inc) / max(1, jd), 1), "noclaim_gpt": round(100 * np.mean([lab("gpt-6-astra", k) == "NOCLAIM" for k in ks]), 1),
                  "DCm_commit_contradict": cj.get(f"{m}|D+C-", {}).get("consensus_incorrect")}
        r = res[m]
        lines.append(f"RESCUE-JUDGE {m:14s} n={r['n']} judgeable {r['judgeable']} consistent {r['consistent_with_image']} contradict {r['contradict']} noclaim {r['noclaim_gpt']} | D+C- commits contradict {r['DCm_commit_contradict']}")
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="syn,twins,rescue")
    a = ap.parse_args()
    lines = []
    fn = {"syn": syn, "twins": twins, "rescue": rescue, "rescue_judge": rescue_judge}
    out = json.load(open(B / "e17n.json")) if (B / "e17n.json").exists() else {}
    for k in a.only.split(","):
        out[k] = fn[k](lines)
    json.dump(out, open(B / "e17n.json", "w"), indent=1)
    old = (B / "rigor_e17n.txt").read_text().splitlines() if (B / "rigor_e17n.txt").exists() else []
    pref = {"syn": "SYN-", "twins": "TWINS", "rescue": "RESCUE ", "rescue_judge": "RESCUE-JUDGE"}
    keep = [x for x in old if not any(x.startswith(pref[k]) for k in a.only.split(","))]
    (B / "rigor_e17n.txt").write_text("\n".join(keep + lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
