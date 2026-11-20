"""Exp-17 rigor analyses from existing outputs (CPU). Run from vtax_exps/ after build_bundle.py.

Writes into outputs/bundle/:
  claim_strat.json      claim truth (rule-based + judges) for D+C-, D+C+, D-C+, D-C- on Synthetic Depth, and
                        stratification by Direct P(gold) > 0.9 (legacy readouts that define the groups)
  commit_baseline.json  share of relational sentences among all rationale sentences of D+C- items and a
                        random-sentence permutation test for the commit-sentence type
  bias_sensitivity.json Synthetic Depth yes/no SDT per model: hit, false alarm, d', c (Direct vs CoT and controls)
  flag_pr.json          order-swap inconsistency flag: precision / recall of CoT errors (main pool and held-out)
  e3x_same_subset.json  conclusion-swap selectivity S restricted to items that have original, gray and mirror readouts
"""
from __future__ import annotations

import json
import math
import re
from collections import Counter
from pathlib import Path

import numpy as np
from scipy.stats import norm

import build_bundle as B
from common import cluster_bootstrap_ci
from e2_judge import L, O
from e2f_claim_edit import QRX, first_claim

OUT = Path("outputs/bundle")
M6 = ["qwen3b", "qwen7b", "internvl3_8b", "phi35v", "molmo7b", "qwen3vl_8b"]
M7 = M6 + ["internvl35_8b"]
REL = re.compile(r"\b(in front|behind|closer|farther|further|nearer|left|right|above|below|on top|under|beneath|"
                 r"foreground|background|occlud\w*|overlap\w*|hidden|blocks?|covers?)\b", re.I)
CONCL = re.compile(r"\b(answer|correct option|conclusion)\b|\([A-F]\)", re.I)


def ci(v, g, scale=100.0, nd=1) -> dict:
    v = np.asarray(v, float)
    if len(v) == 0:
        return {"n": 0}
    m, lo, hi = cluster_bootstrap_ci(v, list(g), 2000)
    return {"mean": round(scale * m, nd), "lo": round(scale * lo, nd), "hi": round(scale * hi, nd), "n": int(len(v))}


def legacy_items(m: str) -> dict:
    src = L / "e1" / f"{m}.json"
    if not src.exists():
        src = O / "e4" / f"{m}.json"
    return {it["uid"]: it for it in json.load(open(src))["items"] if it["task"] in B.TASKS}


# ----------------------------------------------------------------------------- 2. claim truth + confidence

def judge_labels() -> dict:
    lab = {}
    for fn, default in (("e2_judge.jsonl", "claude-opus-5-5"), ("e2_judge2.jsonl", None)):
        p = O / fn
        if not p.exists():
            continue
        for l in open(p):
            r = json.loads(l)
            lab[(r["model"], r["item_id"], r.get("judge_name") or default)] = r["judge"]
    return lab


def claim_strat() -> dict:
    lab = judge_labels()
    res = {}
    for m in M7:
        leg = legacy_items(m)
        G = B.groups_from({}, L, O, m)
        e2b = {r["item_id"]: r for r in B.rj(O / "e2b" / f"{m}.jsonl") if r["task"] == "syn_depth"}
        rows = []
        for uid, r in e2b.items():
            g = G.get(uid, {}).get("group")
            q = QRX.search(r["question"])
            if not g or not q:
                continue
            x, y = q.group(1).lower(), q.group(2).lower()
            front = x if r["gold"] == "A" else y
            pg = B.probs(leg[uid]["scores"]["direct"])[r["gold"]]
            fc = first_claim(r["sentences"])
            rows.append({"uid": uid, "group": g, "pg": pg, "parsed": fc is not None,
                         "wrong": None if fc is None else int(fc[1] != front),
                         "claude": lab.get((m, uid, "claude-opus-5-5")), "gpt": lab.get((m, uid, "gpt-6-astra"))})
        mr = {}
        for g in ("D+C-", "D+C+", "D-C+", "D-C-"):
            for strat in ("all", "pg>0.9", "pg<=0.9"):
                sel = [r for r in rows if r["group"] == g and (strat == "all" or (r["pg"] > 0.9) == (strat == "pg>0.9"))]
                if not sel:
                    continue
                par = [r for r in sel if r["parsed"]]
                d = {"n_items": len(sel), "parsed_share": round(100 * len(par) / len(sel), 1),
                     "rule_claim_wrong": ci([r["wrong"] for r in par], [r["uid"] for r in par])}
                for jn in ("claude", "gpt"):
                    jl = [r for r in par if r[jn] in ("correct", "incorrect")]
                    d[f"{jn}_claim_wrong"] = ci([r[jn] == "incorrect" for r in jl], [r["uid"] for r in jl])
                mr[f"{g}|{strat}"] = d
        dcm = [r for r in rows if r["group"] == "D+C-"]
        mr["DCm_high_conf_count"] = {"n_DCm": len(dcm), "n_pg_gt_0.9": sum(r["pg"] > 0.9 for r in dcm),
                                     "share": round(100 * float(np.mean([r["pg"] > 0.9 for r in dcm])), 1) if dcm else None,
                                     "median_pg": round(float(np.median([r["pg"] for r in dcm])), 3) if dcm else None}
        # pool-wide (all 5 spatial tasks)
        allg = [(G[u]["group"], B.probs(it["scores"]["direct"])[it["gold"]]) for u, it in leg.items() if u in G]
        dcm_all = [p for g, p in allg if g == "D+C-"]
        mr["DCm_high_conf_count_pool"] = {"n_DCm": len(dcm_all), "n_pg_gt_0.9": int(sum(p > 0.9 for p in dcm_all)),
                                          "share": round(100 * float(np.mean([p > 0.9 for p in dcm_all])), 1)}
        res[m] = mr
    return res


# ----------------------------------------------------------------------------- 3. commit-sentence baseline

def stype(s: str) -> str:
    return "conclusion" if CONCL.search(s) else ("relation" if REL.search(s) else "non_relational")


def commit_baseline(n_perm: int = 10000) -> dict:
    rng = np.random.default_rng(0)
    res = {}
    pooled_obs, pooled_null = [], []
    for m in M7:
        G = B.groups_from({}, L, O, m)
        recs = [r for r in B.rj(O / "e2b" / f"{m}.jsonl") if G.get(r["item_id"], {}).get("group") == "D+C-"]
        types_all, items = Counter(), []
        for r in recs:
            ts = [stype(s) for s in r["sentences"]]
            types_all.update(ts)
            pg = [B.probs(t.get("lp_sum") or t["lp_bare"])[r["gold"]] for t in sorted(r["traj"], key=lambda x: x["j"])]
            js = next((j for j in range(1, len(pg)) if pg[j - 1] >= 0.5 and all(v < 0.5 for v in pg[j:])), None)
            if js is None or not ts:
                continue
            items.append({"types": ts, "commit": ts[js - 1], "img": r["image_id"], "task": r["task"]})
        if not items:
            continue
        isrel = [np.array([t == "relation" for t in it["types"]]) for it in items]
        obs = np.array([it["commit"] == "relation" for it in items], float)
        exp = np.array([x.mean() for x in isrel])
        # uniform random sentence per item
        null = np.zeros(n_perm)
        for k, x in enumerate(isrel):
            null += x[rng.integers(0, len(x), n_perm)]
        null /= len(items)
        # restricted to non-conclusion sentences (observed share among non-conclusion commits)
        nc = [(it, np.array([t == "relation" for t in it["types"] if t != "conclusion"])) for it in items]
        nc = [(it, x) for it, x in nc if len(x) and it["commit"] != "conclusion"]
        obs_nc = float(np.mean([it["commit"] == "relation" for it, _ in nc])) if nc else float("nan")
        null_nc = np.zeros(n_perm)
        for it, x in nc:
            null_nc += x[rng.integers(0, len(x), n_perm)]
        null_nc /= max(1, len(nc))
        tot = sum(types_all.values())
        res[m] = {
            "n_DCm": len(recs), "n_committed": len(items),
            "commit_relational_share_of_committed": round(100 * obs.mean(), 1),
            "commit_relational_share_of_all_DCm": round(100 * obs.sum() / len(recs), 1),
            "sentence_relational_share_all_sentences": round(100 * types_all["relation"] / tot, 1),
            "sentence_type_shares": {k: round(100 * v / tot, 1) for k, v in types_all.items()},
            "null_uniform_mean": round(100 * null.mean(), 1),
            "null_uniform_95": [round(100 * float(np.quantile(null, .025)), 1), round(100 * float(np.quantile(null, .975)), 1)],
            "perm_p_uniform": float((np.sum(null >= obs.mean()) + 1) / (n_perm + 1)),
            "obs_minus_item_expected": ci(obs - exp, [it["img"] for it in items]),
            "nonconcl_obs": round(100 * obs_nc, 1), "nonconcl_null_mean": round(100 * null_nc.mean(), 1),
            "perm_p_nonconcl": float((np.sum(null_nc >= obs_nc) + 1) / (n_perm + 1)),
            "commit_type_shares_of_committed": {k: round(100 * v / len(items), 1)
                                                for k, v in Counter(it["commit"] for it in items).items()}}
        pooled_obs.append(obs.mean())
        pooled_null.append(null.mean())
    return res


# ----------------------------------------------------------------------------- 5. bias vs sensitivity (Syn. Depth)

def sdt_arr(pred_yes: np.ndarray, gold_yes: np.ndarray) -> tuple[float, float, float, float]:
    h = (pred_yes[gold_yes].sum() + .5) / (gold_yes.sum() + 1)
    f = (pred_yes[~gold_yes].sum() + .5) / ((~gold_yes).sum() + 1)
    return norm.ppf(h) - norm.ppf(f), -(norm.ppf(h) + norm.ppf(f)) / 2, h, f


def bias_sensitivity() -> dict:
    rng = np.random.default_rng(0)
    res = {}
    for m in B.MODELS:
        main = B.load_main(O, L, m)
        if "direct" not in main:
            continue
        D = main["direct"]
        keys = sorted(k for k, v in D.items() if v[3] == "syn_depth")
        gold = {}
        for it in json.load(open(L / "e1" / f"{m}.json" if (L / "e1" / f"{m}.json").exists() else O / "e4" / f"{m}.json"))["items"]:
            if it["task"] == "syn_depth":
                gold[it["uid"]] = it["gold"]
        keys = [k for k in keys if k in gold]
        gy = np.array([gold[k] == "A" for k in keys])
        mr = {"n": len(keys), "n_gold_yes": int(gy.sum())}
        arrs, raw = {}, {}
        for c in ("direct", "cot", "cot+turn", "filler+turn"):
            if c not in main or not all(k in main[c] for k in keys):
                continue
            py = np.array([main[c][k][4] == "A" for k in keys])
            acc = np.array([main[c][k][0] for k in keys])
            d, cc, h, f = sdt_arr(py, gy)
            arrs[c] = py
            ba = 0.5 * (py[gy].mean() + (1 - py[~gy]).mean())
            raw[c] = {"acc": 100 * acc.mean(), "ba": 100 * ba, "d": d, "c": cc, "unb": 100 * norm.cdf(d / 2)}
            mr[c] = {"acc": round(100 * acc.mean(), 1), "balanced_acc": round(100 * ba, 1), "hit": round(h, 3),
                     "fa": round(f, 3), "yes_rate": round(float(py.mean()), 3), "dprime": round(d, 2), "c": round(cc, 2),
                     "acc_unbiased_at_dprime": round(100 * norm.cdf(d / 2), 1)}
        if "direct" in arrs:
            for c in [x for x in arrs if x != "direct"]:
                boots = []
                for _ in range(2000):
                    i = rng.integers(0, len(keys), len(keys))
                    d0, c0, _, _ = sdt_arr(arrs["direct"][i], gy[i])
                    d1, c1, _, _ = sdt_arr(arrs[c][i], gy[i])
                    boots.append((d1 - d0, c1 - c0, arrs["direct"][i][~gy[i]].mean(), arrs[c][i][~gy[i]].mean()))
                b = np.array(boots)
                q = lambda j, s=1.0: [round(s * float(np.quantile(b[:, j], .025)), 2), round(s * float(np.quantile(b[:, j], .975)), 2)]
                tot = raw["direct"]["ba"] - raw[c]["ba"]
                sens = raw["direct"]["unb"] - raw[c]["unb"]
                mr[f"{c}_vs_direct"] = {
                    "d_dprime": round(raw[c]["d"] - raw["direct"]["d"], 2), "d_dprime_ci": q(0),
                    "d_c": round(raw[c]["c"] - raw["direct"]["c"], 2), "d_c_ci": q(1),
                    "d_fa": round(mr[c]["fa"] - mr["direct"]["fa"], 3),
                    "acc_loss": round(raw["direct"]["acc"] - raw[c]["acc"], 1),
                    "balanced_acc_loss": round(tot, 1),
                    "loss_from_sensitivity": round(sens, 1), "loss_from_criterion": round(tot - sens, 1)}
        res[m] = mr
    return res


# ----------------------------------------------------------------------------- 6. flag precision / recall

def flag_pr(nb: dict) -> dict:
    res = {}
    for m in M7:
        for pool, key in (("main", "fix_main_pool"), ("heldout", "fix_heldout")):
            f = nb.get(m, {}).get(key, {}).get("flag")
            if f:
                res[f"{m}|{pool}"] = f
    return res


# ----------------------------------------------------------------------------- grok: E3x same-subset S

def e3x_same_subset() -> dict:
    res = {}
    for m in M7:
        recs = B.rj(O / "e3x" / f"{m}.jsonl")
        if not recs:
            continue
        idx, meta = {}, {}
        for r in recs:
            idx.setdefault((r["item_id"], r["image"], r["context"]), {})[r["Y"]] = r
            meta[r["item_id"]] = r
        S = {}
        for img in ("original", "gray", "mirror"):
            for uid, r0 in meta.items():
                d = idx.get((uid, img, "rcut"))
                if not d:
                    continue
                gold = r0["gold"]
                w = [c for c in r0["letters"] if c != gold][0]
                if gold in d and w in d:
                    S.setdefault(img, {})[uid] = float(B.pe(d[gold]) == gold) - float(B.pe(d[w]) == w)
        if "mirror" not in S:
            continue
        ks = [k for k in S["mirror"] if k in S.get("original", {}) and k in S.get("gray", {})]
        g = [meta[k]["image_id"] for k in ks]
        res[m] = {"n_same_subset": len(ks),
                  **{f"S_{img}_same_subset": ci([S[img][k] for k in ks], g) for img in ("original", "gray", "mirror")},
                  "S_original_full": ci(list(S["original"].values()), [meta[k]["image_id"] for k in S["original"]]),
                  "S_gray_full": ci(list(S["gray"].values()), [meta[k]["image_id"] for k in S["gray"]]),
                  "tasks_same_subset": dict(Counter(meta[k].get("task") for k in ks))}
    return res


def main() -> None:
    nb = json.load(open(OUT / "numbers.json"))
    for name, fn in (("flag_pr", lambda: flag_pr(nb)), ("e3x_same_subset", e3x_same_subset),
                     ("bias_sensitivity", bias_sensitivity), ("commit_baseline", commit_baseline),
                     ("claim_strat", claim_strat)):
        r = fn()
        json.dump(r, open(OUT / f"{name}.json", "w"), indent=1)
        print(f"== {name}")
        print(json.dumps(r, indent=None)[:6000])


if __name__ == "__main__":
    main()
