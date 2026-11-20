"""Exp-17g analyses for the Gemini round-4 reviews.

1. affirmation: CoT first claims that affirm the question (name the queried subject X as front for "Is X in front of
   Y?", as behind for "Is X behind Y?") on the cue set, twins and Synthetic Depth; overall and on D+C- items; logit
   with subject-first (intercept), affirmation (phrasing), true-front, lower and larger terms; answer-level yes rate;
   relation to the Synthetic Depth false-alarm rise.
2. heldout_mean: cross-model mean paired difference (CoT - Direct, CoT - Filler(|R|)+turn) on BLINK Relative Depth and
   the depth-valid 324 subset with an image-cluster bootstrap, with and without Phi-3.5-Vision.
3. ladder: Qwen3-VL 2B/4B/8B/32B on the original twins: front-lower vs front-higher halves (accuracy, d', AUC),
   identifiable components (occlusion-minus-size and vertical), probit GLM.
4. deepseek_glm: clamp-free GLM components for DeepSeek-V4.1-Flash on the cue set (reasoning off, both arms).

Usage: python rigor_e17g.py  (writes outputs/bundle/{affirmation,heldout_mean,ladder_cues,deepseek_glm}.json, rigor_e17g.txt)
"""
from __future__ import annotations
import vtax_paths as _vtp  # noqa: E402

import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import binomtest, norm, spearmanr

import heldout_breakdown as HB
from build_bundle import pe, rj
from e13_cue_analysis import geom, load, syn_geometry
from e2f_claim_edit import first_claim, split_sents
from rigor_e17f import QRX, auc, geom_meta, glm_set, logit_fit, probit_fit, q, resp_prob

O = Path(f"{_vtp.OUTPUTS}")
B = Path("outputs/bundle")
M7 = ["qwen3b", "qwen7b", "internvl3_8b", "phi35v", "molmo7b", "qwen3vl_8b", "internvl35_8b"]
RNG = np.random.default_rng(0)
STRICT = re.compile(r"\b(cover\w*|in front of|behind|on top of|obscur\w*|block\w*|occlud\w*|hidden|underneath|beneath)\b", re.I)
RESTATE = re.compile(r"question|need to|check|we are|let's|let us|to answer|asked", re.I)
NEG = re.compile(r"\bnot\b|n't\b|\bno\b|\bneither\b|\bnor\b", re.I)
COT_PREFIX = "Let's think step by step."


def cluster_boot(vals: dict[str, np.ndarray], clusters: list, n: int = 2000) -> dict:
    """Mean and 95% CI (cluster bootstrap) for each array in vals (same row order as clusters)."""
    cl = sorted(set(clusters))
    idx = {c: np.where(np.array(clusters) == c)[0] for c in cl}
    bs = {k: [] for k in vals}
    for _ in range(n):
        s = np.concatenate([idx[cl[j]] for j in RNG.integers(0, len(cl), len(cl))])
        for k, v in vals.items():
            bs[k].append(v[s].mean())
    return {k: {"mean": round(100 * float(v.mean()), 1), "ci": [round(100 * x, 1) for x in q(bs[k])]} for k, v in vals.items()}


# ============================================================================= 1. affirmation

def affirm_rows(m: str, which: str, sg: dict) -> list[dict]:
    R = load(m, "cues" if which == "cues" else "twins")
    sel = (lambda r: r["task"] == "syn_depth") if which == "syn" else (lambda r: r["task"].startswith("occ"))
    by = defaultdict(dict)
    for r in R:
        if sel(r) and r["condition"] in ("direct", "cot"):
            by[r["item_id"]][r["condition"]] = r
    out = []
    for k, d in by.items():
        if "direct" not in d or "cot" not in d or not d["cot"].get("text"):
            continue
        r = d["cot"]
        g = geom(r, sg)
        if g is None:
            continue
        front_ph = QRX.search(r["question"]).group(2) == "in front of"
        ss = split_sents(r["text"][len(COT_PREFIX):])
        fc = first_claim(ss)
        sent = ss[fc[0]] if fc else ""
        out.append({"named": fc[1] if fc else None, "front_ph": front_ph, "pair": r["image_id"], **g, "item": k,
                    "strict": bool(fc) and bool(STRICT.search(sent)) and not RESTATE.search(sent) and not NEG.search(sent),
                    "d_ok": pe(d["direct"]) == r["gold"], "c_ok": pe(r) == r["gold"],
                    "d_yes": pe(d["direct"]) == "A", "c_yes": pe(r) == "A"})
    return out


def judge_affirm(m: str, items: dict) -> dict:
    """Synthetic Depth: the image-conditioned judges' verdict on the first claim fixes the claimed-front object
    (correct -> true front, incorrect -> back object); affirmation = claims the queried subject is in front."""
    out = {}
    for fn, keys in (("e2_judge.jsonl", ("judge",)), ("e2_judge2.jsonl", None)):
        for r in map(json.loads, open(O / fn)):
            if r["model"] != m or r["item_id"] not in items:
                continue
            v = r.get("judge") if keys else (r.get("judge") or r.get("judge_gpt") or r.get("verdict"))
            if v not in ("correct", "incorrect"):
                continue
            c = items[r["item_id"]]
            claimed = c["front"] if v == "correct" else c["back"]
            out.setdefault(fn, {}).setdefault(r["group"], []).append(float(claimed == c["x"]))
    return {fn: {g: [round(100 * float(np.mean(v)), 1), len(v)] for g, v in d.items()} for fn, d in out.items()}


def affirmation(lines: list[str]) -> dict:
    sg = syn_geometry()
    bs = json.load(open(B / "bias_sensitivity.json"))
    res = {}
    for which in ("cues", "twins", "syn"):
        for m in M7:
            rows = affirm_rows(m, which, sg)
            cl = [r for r in rows if r["named"]]
            if not cl:
                continue
            aff = np.array([float((c["named"] == c["x"]) == c["front_ph"]) for c in cl])
            dcm = np.array([c["d_ok"] and not c["c_ok"] for c in cl])
            wrong = np.array([c["named"] != c["front"] for c in cl])
            vals = {"affirm": aff, "true_front": np.array([float(c["named"] == c["front"]) for c in cl]),
                    "lower": np.array([float(c["named"] == c["lower"]) for c in cl])}
            row = {"n_parsed": len(cl), "n_DCm": int(dcm.sum())}
            row["all"] = cluster_boot(vals, [c["pair"] for c in cl])
            row["all"]["affirm"]["binom_p_vs_50"] = float(binomtest(int(aff.sum()), len(aff), .5).pvalue)
            st = np.array([c["strict"] for c in cl])
            row["strict"] = cluster_boot({"affirm": aff[st]}, [c["pair"] for c, f in zip(cl, st) if f])["affirm"] | {"n": int(st.sum())}
            sd = st & dcm
            if sd.sum() >= 20:
                row["strict_DCm"] = cluster_boot({"affirm": aff[sd]}, [c["pair"] for c, f in zip(cl, sd) if f])["affirm"] | {"n": int(sd.sum())}
            if which == "syn":
                row["judge"] = judge_affirm(m, {c["item"]: c for c in cl})
            sub = np.where(dcm)[0]
            if len(sub) >= 20:
                v2 = {k: v[sub] for k, v in vals.items()}
                if which == "cues":
                    v2["larger"] = np.array([float(cl[i]["named"] == cl[i]["larger"]) for i in sub])
                row["DCm"] = cluster_boot(v2, [cl[i]["pair"] for i in sub])
                row["DCm"]["affirm"]["binom_p_vs_50"] = float(binomtest(int(aff[sub].sum()), len(sub), .5).pvalue)
            # wrong first claims (any item): which rule do they follow?
            sw = np.where(wrong)[0]
            if len(sw) >= 20:
                v3 = {"affirm": aff[sw], "lower": vals["lower"][sw]}
                if which == "cues":
                    v3["larger"] = np.array([float(cl[i]["named"] == cl[i]["larger"]) for i in sw])
                row["wrong_claims"] = cluster_boot(v3, [cl[i]["pair"] for i in sw]) | {"n": len(sw)}
            # logit: claims X front ~ subject-first (intercept) + affirmation (phrasing) + X front + X lower (+ X larger)
            y = np.array([float(c["named"] == c["x"]) for c in cl])
            cols = ["subject_first", "affirmation", "x_front", "x_lower"] + (["x_larger"] if which == "cues" else [])
            if which == "syn":
                cols.remove("affirmation")
                cols[0] = "subject_first=affirmation"
            feats = {"affirmation": lambda c: 1.0 if c["front_ph"] else -1.0, "x_front": lambda c: 1.0 if c["front"] == c["x"] else -1.0,
                     "x_lower": lambda c: 1.0 if c["lower"] == c["x"] else -1.0, "x_larger": lambda c: 1.0 if c["larger"] == c["x"] else -1.0}
            X = np.column_stack([np.ones(len(cl))] + [np.array([feats[k](c) for c in cl]) for k in cols[1:]])
            b = logit_fit(X, y)
            pairs = sorted({c["pair"] for c in cl})
            pidx = {p: [i for i, c in enumerate(cl) if c["pair"] == p] for p in pairs}
            bb = np.array([logit_fit(X[s], y[s]) for s in
                           (np.concatenate([pidx[pairs[j]] for j in RNG.integers(0, len(pairs), len(pairs))]) for _ in range(600))])
            row["logit"] = {k: {"b": round(float(b[i]), 3), "ci": [round(v, 3) for v in q(bb[:, i])]} for i, k in enumerate(cols)}
            # answer-level yes rate
            row["yes_rate"] = {"direct": round(100 * np.mean([r["d_yes"] for r in rows]), 1), "cot": round(100 * np.mean([r["c_yes"] for r in rows]), 1)}
            if which == "syn":
                row["syn_fa"] = {"direct": bs[m]["direct"]["fa"], "cot": bs[m]["cot"]["fa"]}
            res.setdefault(which, {})[m] = row
            a = row["all"]["affirm"]
            lg = " ".join(f"{k}={v['b']:+.2f}{v['ci']}" for k, v in row["logit"].items())
            dc = row.get("DCm", {}).get("affirm")
            wc = row.get("wrong_claims", {})
            lines.append(f"AFFIRM {which:5s} {m:14s} n={len(cl)} affirm {a['mean']} {a['ci']} strict {row['strict']['mean']} (n={row['strict']['n']}) "
                         f"strictD+C- {row.get('strict_DCm', {}).get('mean')} judge {row.get('judge')} | D+C- (n={row['n_DCm']}) "
                         f"{dc['mean'] if dc else '--'} {dc['ci'] if dc else ''} | wrong claims: affirm {wc.get('affirm', {}).get('mean')} "
                         f"lower {wc.get('lower', {}).get('mean')} larger {wc.get('larger', {}).get('mean', '--')} | yes {row['yes_rate']} | {lg}")
    syn = res.get("syn", {})
    ms = [m for m in M7 if m in syn]
    rho, p = spearmanr([syn[m]["all"]["affirm"]["mean"] for m in ms], [syn[m]["syn_fa"]["cot"] - syn[m]["syn_fa"]["direct"] for m in ms])
    res["_syn_affirm_vs_fa_rise"] = {"spearman": round(float(rho), 3), "p": float(p), "n": len(ms)}
    lines.append(f"AFFIRM syn: Spearman(affirm share, FA rise) = {rho:.2f} p={p:.3f}")
    return res


# ============================================================================= 2. held-out cross-model mean

def heldout_mean(lines: list[str]) -> dict:
    rows = {m: {r["item"]: r for r in HB.model_rows(m)} for m in M7}
    rows["qwen3vl_8b_think"] = {r["item"]: r for r in HB.think_rows()}
    subsets = {"blink_depth": lambda t: t == "blink_depth", "depth324": lambda t: t in HB.DEPTH}
    res = {}
    for sub, f in subsets.items():
        for label, models, key in (("cot-direct|8 open", M7 + ["qwen3vl_8b_think"], "pd"),
                                   ("cot-direct|7 (no Phi)", [m for m in M7 if m != "phi35v"] + ["qwen3vl_8b_think"], "pd"),
                                   ("cot-direct|7 instruct", M7, "pd"),
                                   ("cot-filler_turn|7", M7, "pf"),
                                   ("cot-filler_turn|6 (no Phi)", [m for m in M7 if m != "phi35v"], "pf")):
            items = sorted(k for k, r in rows[M7[0]].items() if f(r["task"]) and all(k in rows[m] and key in rows[m][k] for m in models))
            diff = np.array([np.mean([float(rows[m][k]["pc"] == rows[m][k]["gold"]) - float(rows[m][k][key] == rows[m][k]["gold"])
                                      for m in models]) for k in items])
            cb = cluster_boot({"d": diff}, [rows[M7[0]][k]["img"] for k in items], n=4000)["d"]
            per = {m: round(100 * float(np.mean([float(rows[m][k]["pc"] == rows[m][k]["gold"]) - float(rows[m][k][key] == rows[m][k]["gold"])
                                                 for k in items])), 1) for m in models}
            res.setdefault(sub, {})[label] = cb | {"n_items": len(items), "n_models": len(models), "per_model": per}
            lines.append(f"HELDOUT-MEAN {sub:11s} {label:28s} n={len(items)} mean {cb['mean']:+.1f} {cb['ci']}")
    return res


# ============================================================================= 3. ladder

def ladder_records(size: str) -> list[dict]:
    if size == "8b":
        return [r for r in rj(O / "e10" / "qwen3vl_8b.jsonl") if r["task"].startswith("occ") and r["condition"] in ("direct", "cot")]
    recs = []
    for ph, fn in (("front", f"qwen3vl_{size}.jsonl"), ("behind", f"qwen3vl_{size}_behind.jsonl")):
        for r in rj(O / "scale" / fn):
            mt = r["meta"]
            recs.append(r | {"question": f"Is the {mt['x_name']} {'in front of' if ph == 'front' else 'behind'} the {mt['y_name']}?"})
    return recs


def half_stats(recs: list[dict]) -> dict:
    """Per condition: acc, d' and AUC in front-lower / front-higher halves; components with pair bootstrap."""
    out = {}
    rows = []
    for r in recs:
        qm = QRX.search(r["question"])
        x, behind = qm.group(1).lower(), qm.group(2) == "behind"
        g = geom_meta(r["meta"], x)
        pa = resp_prob(r)
        rows.append({"cond": r["condition"], "says": (pe(r) == "A") != behind, "truth": g["x_front"],
                     "score": pa if not behind else 1 - pa, "half": "front_lower" if g["lower"] == g["front"] else "front_higher",
                     "pair": r["image_id"], "ok": pe(r) == r["gold"]})
    pairs = sorted({r["pair"] for r in rows})
    for c in ("direct", "cot"):
        rs = [r for r in rows if r["cond"] == c]
        arr = {h: [r for r in rs if r["half"] == h] for h in ("front_lower", "front_higher")}

        def stat(sel):
            s = np.array([r["says"] for r in sel], bool)
            t = np.array([r["truth"] for r in sel], bool)
            hh = (s[t].sum() + .5) / (t.sum() + 1)
            ff = (s[~t].sum() + .5) / ((~t).sum() + 1)
            return float(norm.ppf(hh) - norm.ppf(ff)), 100 * float(np.mean([r["ok"] for r in sel])), \
                auc(np.array([r["score"] for r in sel]), t)

        est = {h: stat(v) for h, v in arr.items()}
        pidx = {h: defaultdict(list) for h in arr}
        for h, v in arr.items():
            for i, r in enumerate(v):
                pidx[h][r["pair"]].append(i)
        boots = []
        for _ in range(1000):
            js = [pairs[j] for j in RNG.integers(0, len(pairs), len(pairs))]
            st = {h: stat([arr[h][i] for p in js for i in pidx[h].get(p, [])]) for h in arr}
            boots.append([(st["front_lower"][0] + st["front_higher"][0]) / 2, (st["front_lower"][0] - st["front_higher"][0]) / 2])
        boots = np.array(boots)
        out[c] = {h: {"dprime": round(e[0], 2), "acc": round(e[1], 1), "auc": round(e[2], 3)} for h, e in est.items()}
        out[c]["occ_minus_size"] = {"d": round((est["front_lower"][0] + est["front_higher"][0]) / 2, 2), "ci": [round(v, 2) for v in q(boots[:, 0])]}
        out[c]["vertical"] = {"d": round((est["front_lower"][0] - est["front_higher"][0]) / 2, 2), "ci": [round(v, 2) for v in q(boots[:, 1])]}
        out[c]["_boots"] = boots
    b0, b1 = out["direct"].pop("_boots"), out["cot"].pop("_boots")
    out["cot-direct"] = {k: {"d": round(out["cot"][k]["d"] - out["direct"][k]["d"], 2), "ci": [round(v, 2) for v in q(b1[:, i] - b0[:, i])]}
                         for i, k in enumerate(("occ_minus_size", "vertical"))}
    return out


def ladder(lines: list[str]) -> dict:
    res = {}
    for s in ("2b", "4b", "8b", "32b"):
        recs = ladder_records(s)
        h = half_stats(recs)
        g = glm_set({c: [r for r in recs if r["condition"] == c] for c in ("direct", "cot")}, use_size=False, n_boot=300)
        res[s] = {"halves": h, "glm": g}
        d, c, dd = h["direct"], h["cot"], h["cot-direct"]
        lines.append(f"LADDER {s:4s} Direct acc low/high {d['front_lower']['acc']}/{d['front_higher']['acc']} d' {d['front_lower']['dprime']:+.2f}/{d['front_higher']['dprime']:+.2f} "
                     f"AUC {d['front_lower']['auc']}/{d['front_higher']['auc']} | CoT acc {c['front_lower']['acc']}/{c['front_higher']['acc']} "
                     f"d' {c['front_lower']['dprime']:+.2f}/{c['front_higher']['dprime']:+.2f} AUC {c['front_lower']['auc']}/{c['front_higher']['auc']} "
                     f"| comps D occ-size {d['occ_minus_size']['d']:+.2f} vert {d['vertical']['d']:+.2f}; CoT-D occ-size {dd['occ_minus_size']['d']:+.2f} {dd['occ_minus_size']['ci']} "
                     f"vert {dd['vertical']['d']:+.2f} {dd['vertical']['ci']} | GLM sep {g['direct']['separated']} CoT-D occ {g['cot-direct']['occlusion']['dprime']:+.2f} "
                     f"vert {g['cot-direct']['vertical']['dprime']:+.2f} {g['cot-direct']['vertical']['ci']} AUC {g['direct']['auc']['auc']}->{g['cot']['auc']['auc']}")
    return res


# ============================================================================= 4. DeepSeek GLM on the cue set

def deepseek_glm(lines: list[str]) -> dict:
    seen, rows = set(), {"direct": [], "cot": []}
    for r in rj(O / "e9c_cues" / "deepseek-v4.1-flash.jsonl"):
        k = (r["item_id"], r["condition"])
        if "error" in r or k in seen or r.get("pred") is None or r["condition"] not in rows:
            continue
        seen.add(k)
        mt = r["meta"]
        x = mt["x_name"].lower()
        xf = mt["front_name"].lower() == x
        xl = xf == (not mt["front_above"])
        xg = xf == bool(mt["front_larger"])
        behind = r["phrasing"] == "behind"
        rows[r["condition"]].append((float((r["pred"] == "A") != behind), -1.0 if behind else 1.0, 1.0 if xf else -1.0,
                                     1.0 if xl else -1.0, 1.0 if xg else -1.0, r["image_id"]))
    data = {c: (np.array([v[:5] for v in rs]), [v[5] for v in rs]) for c, rs in rows.items()}
    pairs = sorted(set(data["direct"][1]))
    X = {c: np.column_stack([np.ones(len(a)), a[:, 1:5]]) for c, (a, _) in data.items()}
    y = {c: a[:, 0] for c, (a, _) in data.items()}
    est = {c: probit_fit(X[c], y[c]) for c in X}
    pidx = {c: {p: np.where(np.array(data[c][1]) == p)[0] for p in pairs} for c in X}
    bb = {c: [] for c in X}
    for _ in range(300):
        js = RNG.integers(0, len(pairs), len(pairs))
        for c in X:
            s = np.concatenate([pidx[c][pairs[j]] for j in js])
            bb[c].append(probit_fit(X[c][s], y[c][s]))
    names = {2: "occlusion", 3: "vertical", 4: "size"}
    res = {}
    for c in X:
        b = np.array(bb[c])
        res[c] = {n: {"dprime": round(2 * float(est[c][i]), 2), "ci": [round(2 * v, 2) for v in q(b[:, i])]} for i, n in names.items()}
    d0, d1 = np.array(bb["direct"]), np.array(bb["cot"])
    res["cot-direct"] = {n: {"dprime": round(2 * float(est["cot"][i] - est["direct"][i]), 2), "ci": [round(2 * v, 2) for v in q(d1[:, i] - d0[:, i])]}
                         for i, n in names.items()}
    lines.append(f"DEEPSEEK-GLM {json.dumps(res['cot-direct'])} | Direct {json.dumps(res['direct'])}")
    return res


def main() -> None:
    lines: list[str] = []
    import sys
    if "--affirm-only" not in sys.argv:
        json.dump(deepseek_glm(lines), open(B / "deepseek_glm.json", "w"), indent=1)
        json.dump(heldout_mean(lines), open(B / "heldout_mean.json", "w"), indent=1)
        json.dump(ladder(lines), open(B / "ladder_cues.json", "w"), indent=1)
    else:
        lines = [x for x in (B / "rigor_e17g.txt").read_text().splitlines() if not x.startswith("AFFIRM")]
    json.dump(affirmation(lines), open(B / "affirmation.json", "w"), indent=1)
    (B / "rigor_e17g.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
