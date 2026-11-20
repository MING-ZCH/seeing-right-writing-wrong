"""Exp-52 analysis: attention-edge knockout at the forced answer position (S5) + Direct-probe transport (S6).

Reads shared outputs/e52 (e52_knockout.py), writes
  outputs/bundle/e52_knockout.json, outputs/bundle/e52_transport.json, outputs/bundle/rigor_e52.txt,
  paper/tables/supp_e52.tex, paper/figures/supp_e52_knockout.{pdf,png}.
Estimates are population-weighted (N_cell / n_cell within each set; pooled = mean of the cue-set and twin
estimates); CIs: 2,000 twin-pair cluster bootstrap draws within each set, shared by all conditions (paired).

Usage: python rigor_e52.py [--models qwen3vl_8b qwen7b molmo7b] [--no-probe]
"""
from __future__ import annotations
import vtax_paths as _vtp  # noqa: E402

import argparse
import glob
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

O = Path(f"{_vtp.OUTPUTS}/e52")
B = Path("outputs/bundle")
PAPER = Path(f"{_vtp.PAPER}")
NAMES = {"qwen3vl_8b": "Qwen3-VL-8B", "qwen7b": "Qwen2.5-VL-7B", "molmo7b": "Molmo-7B-D"}
SETS = ("cues", "twins")
NB = 2000
COT_X = ("a", "b", "c", "ac", "a_all", "a_full", "c_afull", "d", "r1", "r2", "a_lo", "a_hi", "c_lo", "c_hi")
DIR_X = ("a", "a_full", "d", "r1", "r2")
NEED_REL = {"b", "c", "ac", "c_lo", "c_hi", "c_afull"}
LINES: list[str] = []
EXTRA_CHECK: dict = {}


def say(s: str = "") -> None:
    print(s)
    LINES.append(s)


def pg(ab: list[float], gold: str) -> float:
    m = max(ab)
    e = [math.exp(v - m) for v in ab]
    return e["AB".index(gold)] / sum(e)


def load(model: str) -> tuple[list[dict], dict, dict]:
    recs = []
    for f in sorted(glob.glob(str(O / f"{model}_[0-9].jsonl"))):
        for l in open(f):
            try:
                recs.append(json.loads(l))
            except json.JSONDecodeError:
                print(f"[warn] unreadable (partial) line in {f}")
    extra = {}
    for f in sorted(glob.glob(str(O / f"{model}_x_[0-9].jsonl"))):
        for l in open(f):
            x = json.loads(l)
            extra[x["key"]] = x
    EXTRA_CHECK[model] = []
    for r in recs:
        if r["key"] in extra and "c_afull" not in r["cot"]:
            r["cot"]["c_afull"] = extra[r["key"]]["c_afull"]
            EXTRA_CHECK[model].append(abs(pg(extra[r["key"]]["none"], r["gold"]) - pg(r["cot"]["none"], r["gold"])))
    smp = json.load(open(O / f"sample_{model}.json"))
    hd = json.load(open(O / f"heads_{model}.json"))
    return recs, smp, hd


# ============================================================================= weighted bootstrap

class WB:
    """Population-weighted means with a paired twin-pair cluster bootstrap within each set."""

    def __init__(self, recs: list[dict], smp: dict, seed: int = 52):
        rng = np.random.default_rng(seed)
        self.n = len(recs)
        self.set = np.array([r["set"] for r in recs])
        self.w = np.array([smp["N_cell"][r["set"]][r["cell"]] / smp["n_cell"][r["set"]][r["cell"]] for r in recs], float)
        self.mult = np.zeros((NB, self.n), np.float32)
        for s in SETS:
            ix = np.where(self.set == s)[0]
            if not len(ix):
                continue
            pairs = sorted({recs[i]["pair"] for i in ix})
            pid = {p: k for k, p in enumerate(pairs)}
            cl = np.array([pid[recs[i]["pair"]] for i in ix])
            cnt = rng.multinomial(len(pairs), np.full(len(pairs), 1 / len(pairs)), size=NB)
            self.mult[:, ix] = cnt[:, cl]

    def est(self, x: np.ndarray, mask: np.ndarray | None = None, weighted: bool = True) -> dict:
        mask = np.isfinite(x) if mask is None else (mask & np.isfinite(x))
        xx = np.where(mask, x, 0.0)
        w = (self.w if weighted else np.ones(self.n)) * mask
        pts, bs = [], []
        for s in SETS:
            m = self.set == s
            if w[m].sum() == 0:
                continue
            pts.append((w[m] * xx[m]).sum() / w[m].sum())
            mw = self.mult[:, m] * w[m]
            bs.append((mw * xx[m]).sum(1) / np.maximum(mw.sum(1), 1e-12))
        p = float(np.mean(pts))
        b = np.mean(bs, 0)
        return {"mean": round(100 * p, 2), "lo": round(100 * float(np.quantile(b, .025)), 2),
                "hi": round(100 * float(np.quantile(b, .975)), 2), "n": int(mask.sum()),
                "p_le0": round(float((b <= 0).mean()), 4)}

    def by_set(self, x: np.ndarray, mask: np.ndarray | None = None) -> dict:
        out = {}
        for s in SETS:
            m = (self.set == s) if mask is None else (mask & (self.set == s))
            out[s] = self.est(x, m)
        return out


def f1(d: dict, ci: bool = True, sign: bool = True) -> str:
    v = f"{d['mean']:+.1f}" if sign else f"{d['mean']:.1f}"
    return f"{v} [{d['lo']:+.1f}, {d['hi']:+.1f}]" if ci else v


# ============================================================================= S5 knockout analysis

def analyze_model(model: str) -> dict:
    recs, smp, hd = load(model)
    W = WB(recs, smp)
    n = len(recs)
    rel = np.array([r["has_rel"] for r in recs])
    gold = [r["gold"] for r in recs]
    written = [r["written"] for r in recs]
    cells = np.array([r["cell"] for r in recs])
    res = {"model": model, "n": n, "n_rel": int(rel.sum()), "n_cell": smp["n_cell"], "N_cell": smp["N_cell"],
           "visual_heads": {"k": hd["k"], "of": hd["n_heads_total"], "per_layer": hd["per_layer"],
                            "score_visual_mean": hd["score_visual_mean"], "score_all_mean": hd["score_all_mean"]}}
    say(f"\n=== {NAMES[model]} ({model}): n = {n} prompts (cues {int((W.set == 'cues').sum())}, twins {int((W.set == 'twins').sum())}),"
        f" parsed last relation {int(rel.sum())}; visual heads {hd['k']}/{hd['n_heads_total']} (mean last-row image mass"
        f" {hd['score_visual_mean']:.3f} vs all heads {hd['score_all_mean']:.4f})")

    def arr(arm: str, cond: str, fn) -> np.ndarray:
        return np.array([fn(r[arm][cond], i) if cond in r[arm] else np.nan for i, r in enumerate(recs)], float)

    P = {arm: {c: arr(arm, c, lambda ab, i: pg(ab, gold[i])) for c in ("none",) + xs}
         for arm, xs in (("cot", COT_X), ("direct", DIR_X))}
    P["filler"] = {"none": arr("filler", "none", lambda ab, i: pg(ab, gold[i]))}
    pred = {arm: {c: np.where(np.isnan(v), np.nan, (v > 0.5).astype(float)) for c, v in d.items()} for arm, d in P.items()}
    # agreement with the written relation (letter implied by the last relational sentence)
    def agree(cond: str) -> np.ndarray:
        out = np.full(n, np.nan)
        for i, r in enumerate(recs):
            if r["has_rel"] and cond in r["cot"]:
                ab = r["cot"][cond]
                out[i] = float(("A" if ab[0] >= ab[1] else "B") == written[i])
        return out
    AG = {c: agree(c) for c in ("none",) + COT_X}

    # ---------------- sanity
    san = {}
    for arm, key in (("direct", "p_direct_stored"), ("cot", "p_cot_stored"), ("filler", "p_filler_stored")):
        st = np.array([r[key] if r[key] is not None else np.nan for r in recs], float)
        d = np.abs(P[arm]["none"] - st)
        ok = np.isfinite(d)
        san[arm] = {"max_abs_dP": round(float(np.nanmax(d)), 4), "mean_abs_dP": round(float(np.nanmean(d)), 4),
                    "argmax_agree": round(float(((P[arm]["none"][ok] > .5) == (st[ok] > .5)).mean()), 4), "n": int(ok.sum())}
    cm = hd["meta"]
    dcal = np.array([abs(m["p_direct_hf"] - m["p_direct_stored"]) for m in cm])
    san["direct_calib"] = {"max_abs_dP": round(float(dcal.max()), 4), "mean_abs_dP": round(float(dcal.mean()), 4),
                           "argmax_agree": round(float(np.mean([(m["p_direct_hf"] > .5) == (m["p_direct_stored"] > .5) for m in cm])), 4),
                           "n": len(cm)}
    s2 = [r for r in recs if "s2_identity" in r]
    san["s2_identity_max_abs_dP"] = float(max(abs(pg(r["s2_identity"], r["gold"]) - pg(r["cot"]["none"], r["gold"])) for r in s2))
    san["s2_vs_sdpa_max_abs_dP"] = round(float(max(abs(pg(r["s2_sdpa"], r["gold"]) - pg(r["cot"]["none"], r["gold"])) for r in s2)), 4)
    san["s2_vs_sdpa_mean_abs_dP"] = round(float(np.mean([abs(pg(r["s2_sdpa"], r["gold"]) - pg(r["cot"]["none"], r["gold"])) for r in s2])), 4)
    d3 = [abs(pg(r["s3_bos_ko"], r["gold"]) - pg(r["s3_reduced"], r["gold"])) for r in s2]
    gap = [abs((r["s3_bos_ko"][0] - r["s3_bos_ko"][1]) - (r["s3_reduced"][0] - r["s3_reduced"][1])) for r in s2]
    san["s3_text_free"] = {"n": len(s2), "max_abs_dP": round(float(max(d3)), 4), "mean_abs_dP": round(float(np.mean(d3)), 4),
                           "max_abs_dlogit_gap": round(float(max(gap)), 4),
                           "mean_P_gold_text_free": round(float(np.mean([pg(r["s3_bos_ko"], r["gold"]) for r in s2])), 3),
                           "share_pred_A_text_free": round(float(np.mean([r["s3_bos_ko"][0] > r["s3_bos_ko"][1] for r in s2])), 3)}
    res["sanity"] = san
    say(f"Sanity S1 (HF none vs stored e10): Direct max|dP| {san['direct']['max_abs_dP']} mean {san['direct']['mean_abs_dP']}"
        f" agree {san['direct']['argmax_agree']:.3f}; CoT {san['cot']['max_abs_dP']} / {san['cot']['mean_abs_dP']} / {san['cot']['argmax_agree']:.3f};"
        f" filler_R+turn {san['filler']['max_abs_dP']} / {san['filler']['mean_abs_dP']} / {san['filler']['argmax_agree']:.3f};"
        f" calibration Direct {san['direct_calib']['max_abs_dP']} / {san['direct_calib']['mean_abs_dP']} / {san['direct_calib']['argmax_agree']:.3f}")
    say(f"Sanity S2: empty knockout = none, max|dP| {san['s2_identity_max_abs_dP']:.2e}; patched fp32-score path vs stock SDPA max|dP|"
        f" {san['s2_vs_sdpa_max_abs_dP']} mean {san['s2_vs_sdpa_mean_abs_dP']} (n {len(s2)})")
    s3 = san["s3_text_free"]
    if EXTRA_CHECK.get(model):
        san["extra_pass_none_max_abs_dP"] = round(float(max(EXTRA_CHECK[model])), 6)
        say(f"Sanity: separate c_afull pass recomputes CoT none with max|dP| {san['extra_pass_none_max_abs_dP']} (n {len(EXTRA_CHECK[model])})")
    f32 = sorted(glob.glob(str(O / "proc_test" / f"{model}_fp32" / f"{model}_0_limit*.jsonl")))
    if f32:
        rr = [json.loads(l) for l in open(f32[-1])]
        rr = [r for r in rr if "s3_bos_ko" in r]
        d = [abs(pg(r["s3_bos_ko"], r["gold"]) - pg(r["s3_reduced"], r["gold"])) for r in rr]
        dg = [abs((r["s3_bos_ko"][0] - r["s3_bos_ko"][1]) - (r["s3_reduced"][0] - r["s3_reduced"][1])) for r in rr]
        ds = [abs(pg(r["s2_sdpa"], r["gold"]) - pg(r["cot"]["none"], r["gold"])) for r in rr]
        san["s3_fp32"] = {"n": len(rr), "max_abs_dP": float(max(d)), "max_abs_dlogit_gap": float(max(dg)), "s2_sdpa_max_abs_dP": float(max(ds))}
        say(f"Sanity S3 in fp32 (n {len(rr)}): max|dP| {max(d):.2e}, max |d logit gap| {max(dg):.2e}; patched vs SDPA max|dP| {max(ds):.2e}")
    say(f"Sanity S3: all keys but token 0 blocked at the answer rows vs separate [token 0 + answer rows] pass: max|dP| {s3['max_abs_dP']}"
        f" mean {s3['mean_abs_dP']}, max |d(logit A - logit B)| {s3['max_abs_dlogit_gap']}; text-free P(gold) {s3['mean_P_gold_text_free']}")

    # ---------------- accuracy, flips, agreement
    out = {"cot": {}, "direct": {}, "filler": {}}
    for arm, xs in (("cot", COT_X), ("direct", DIR_X)):
        acc0 = pred[arm]["none"]
        out[arm]["none"] = {"acc": W.est(acc0), "acc_rel": W.est(acc0, rel) if arm == "cot" else None,
                            "acc_by_set": W.by_set(acc0)}
        for c in xs:
            v = pred[arm][c]
            mask = rel if c in NEED_REL else None
            e = {"dacc": W.est(v - acc0, mask), "flip": W.est((v != acc0).astype(float) * np.where(np.isnan(v), np.nan, 1), mask),
                 "dP": W.est(P[arm][c] - P[arm]["none"], mask), "absdP": W.est(np.abs(P[arm][c] - P[arm]["none"]), mask),
                 "dacc_by_set": W.by_set(v - acc0, mask)}
            if arm == "cot":
                e["dacc_rel"] = W.est(v - acc0, rel)
                e["flip_rel"] = W.est((v != acc0).astype(float) * np.where(np.isnan(v), np.nan, 1), rel)
                e["dagree"] = W.est(AG[c] - AG["none"], rel)
                e["agree"] = W.est(AG[c], rel)
                e["cells"] = {cl: {"dacc": W.est(v - acc0, (cells == cl) & (rel if c in NEED_REL else True), weighted=False),
                                   "dagree": W.est(AG[c] - AG["none"], (cells == cl) & rel, weighted=False)}
                              for cl in ("D+C+", "D+C-", "D-C+", "D-C-")}
            out[arm][c] = e
    out["cot"]["none"]["agree"] = W.est(AG["none"], rel)
    out["cot"]["none"]["agree_cells"] = {cl: W.est(AG["none"], (cells == cl) & rel, weighted=False) for cl in ("D+C+", "D+C-", "D-C+", "D-C-")}
    out["filler"]["none"] = {"acc": W.est(pred["filler"]["none"])}
    # paired contrasts
    fl = lambda c: np.where(np.isnan(pred["cot"][c]), np.nan, (pred["cot"][c] != pred["cot"]["none"]).astype(float))
    dpred = pred["direct"]["none"]
    agD = lambda c: np.where(np.isnan(pred["cot"][c]), np.nan, (pred["cot"][c] == dpred).astype(float))
    con = {
        "primary_dagree_a_minus_c": W.est((AG["a"] - AG["none"]) - (AG["c"] - AG["none"]), rel),
        "primary_flip_c_minus_a": W.est(fl("c") - fl("a"), rel),
        "dacc_c_minus_a_rel": W.est(pred["cot"]["c"] - pred["cot"]["a"], rel),
        "absdP_c_minus_a_rel": W.est(np.abs(P["cot"]["c"] - P["cot"]["none"]) - np.abs(P["cot"]["a"] - P["cot"]["none"]), rel),
        "flip_ac_minus_c": W.est(fl("ac") - fl("c"), rel),
        "flip_cafull_minus_c": W.est(fl("c_afull") - fl("c"), rel),
        "dacc_cafull_minus_c": W.est(pred["cot"]["c_afull"] - pred["cot"]["c"], rel),
        "absdP_cafull_vs_c": W.est(np.abs(P["cot"]["c_afull"] - P["cot"]["c"]), rel),
        "absdP_afull_vs_none_direct": W.est(np.abs(P["direct"]["a_full"] - P["direct"]["none"])),
        "cot_flip_d_minus_rand": W.est(fl("d") - (fl("r1") + fl("r2")) / 2),
        "cot_dacc_d_minus_rand": W.est(pred["cot"]["d"] - (pred["cot"]["r1"] + pred["cot"]["r2"]) / 2),
        "direct_dacc_d_minus_rand": W.est(pred["direct"]["d"] - (pred["direct"]["r1"] + pred["direct"]["r2"]) / 2),
        "direct_flip_d_minus_rand": W.est((pred["direct"]["d"] != pred["direct"]["none"]).astype(float)
                                          - ((pred["direct"]["r1"] != pred["direct"]["none"]).astype(float)
                                             + (pred["direct"]["r2"] != pred["direct"]["none"]).astype(float)) / 2),
        "afull_flip_direct_minus_cot": W.est((pred["direct"]["a_full"] != pred["direct"]["none"]).astype(float) - fl("a_full")),
        "afull_absdP_direct_minus_cot": W.est(np.abs(P["direct"]["a_full"] - P["direct"]["none"]) - np.abs(P["cot"]["a_full"] - P["cot"]["none"])),
        "direct_vs_cot_flip_a": W.est((pred["direct"]["a"] != pred["direct"]["none"]).astype(float) - fl("a")),
    }
    con["agree_direct_none"] = W.est(agD("none"), rel)
    for c in ("c", "ac", "c_afull", "c_hi"):
        con[f"agree_direct_{c}"] = W.est(agD(c), rel)
        con[f"dagree_direct_{c}"] = W.est(agD(c) - agD("none"), rel)
    con["agree_direct_cells"] = {cl: {c: W.est(agD(c), (cells == cl) & rel, weighted=False) for c in ("none", "c", "c_afull")}
                                 for cl in ("D+C+", "D+C-", "D-C+", "D-C-")}
    out["contrasts"] = con
    res["results"] = out
    ok = con["primary_dagree_a_minus_c"]["lo"] > 0 and con["primary_flip_c_minus_a"]["lo"] > 0
    res["primary_met"] = bool(ok)

    c0 = out["cot"]
    say(f"CoT none: acc {f1(c0['none']['acc'], sign=False)}; on parsed-relation prompts {f1(c0['none']['acc_rel'], sign=False)};"
        f" agreement with written relation {f1(c0['none']['agree'], sign=False)}")
    say(f"Direct none: acc {f1(out['direct']['none']['acc'], sign=False)}; filler_R+turn acc {f1(out['filler']['none']['acc'], sign=False)}")
    say("PRIMARY (CoT, parsed-relation prompts, population-weighted, pooled over sets):")
    for c in ("a", "c"):
        e = c0[c]
        say(f"  ({c}) dAcc {f1(e['dacc_rel'])} | dAgree {f1(e['dagree'])} | flip {f1(e['flip_rel'], sign=False)} | mean|dP| {f1(e['absdP'], sign=False)}")
    say(f"  dAgree(a) - dAgree(c) = {f1(con['primary_dagree_a_minus_c'])}; flip(c) - flip(a) = {f1(con['primary_flip_c_minus_a'])}"
        f" -> prediction {'SUPPORTED' if ok else 'NOT supported'} in this model")
    say("CoT, all knockouts (dAcc | flip | dAgree | mean|dP|):")
    for c in COT_X:
        e = c0[c]
        say(f"  {c:6s} dAcc {f1(e['dacc'])} | flip {f1(e['flip'], sign=False)} | dAgree {f1(e['dagree'])} | |dP| {f1(e['absdP'], sign=False)}"
            f" | cues {f1(e['dacc_by_set']['cues'], ci=False)} twins {f1(e['dacc_by_set']['twins'], ci=False)}")
    say("CoT per cell (unweighted): dAcc / dAgree")
    for c in ("a", "b", "c", "ac", "a_full", "d"):
        say("  " + c.ljust(6) + " ".join(f"{cl} {f1(c0[c]['cells'][cl]['dacc'], ci=False)}/{f1(c0[c]['cells'][cl]['dagree'], ci=False)}"
                                        for cl in ("D+C+", "D+C-", "D-C+", "D-C-")))
    say("  agreement (none) by cell: " + " ".join(f"{cl} {c0['none']['agree_cells'][cl]['mean']:.1f}" for cl in ("D+C+", "D+C-", "D-C+", "D-C-")))
    say("Direct knockouts (dAcc | flip | mean|dP|):")
    for c in DIR_X:
        e = out["direct"][c]
        say(f"  {c:6s} dAcc {f1(e['dacc'])} | flip {f1(e['flip'], sign=False)} | |dP| {f1(e['absdP'], sign=False)}")
    for k, v in con.items():
        if k != "agree_direct_cells":
            say(f"  contrast {k}: {f1(v)}")
    say("  answer = Direct answer, by cell (none / c / c_afull): " + " ".join(
        f"{cl} {con['agree_direct_cells'][cl]['none']['mean']:.1f}/{con['agree_direct_cells'][cl]['c']['mean']:.1f}/"
        f"{con['agree_direct_cells'][cl]['c_afull']['mean']:.1f}" for cl in ("D+C+", "D+C-", "D-C+", "D-C-")))

    # ---------------- attention mass at the last answer row (no knockout)
    att = {}
    shards = sorted(glob.glob(str(O / f"att_{model}_[0-9].npz")))
    if shards:
        A = {k: np.concatenate([np.load(f)[k] for f in shards]).astype(np.float32) for k in ("direct", "cot", "filler")}
        L = A["direct"].shape[1]
        for k, v in A.items():
            att[k] = {"all_layers": round(float(v.mean()), 4), "upper_half": round(float(v[:, L // 2:].mean()), 4),
                      "per_layer": [round(float(x), 4) for x in v.mean((0, 2))]}
        body = np.array([np.mean(r["att_cot"]["body"]) for r in recs if "body" in r.get("att_cot", {})])
        conc = np.array([np.mean(r["att_cot"]["conc"]) for r in recs if "conc" in r.get("att_cot", {})])
        fill = np.array([np.mean(r["att_fill"]) for r in recs])
        att["cot_body_mass"], att["cot_conc_mass"], att["filler_text_mass"] = (round(float(body.mean()), 4),
                                                                              round(float(conc.mean()), 4), round(float(fill.mean()), 4))
        say(f"Last-row image attention mass (mean over heads; all layers / upper half): Direct {att['direct']['all_layers']} /"
            f" {att['direct']['upper_half']}, CoT {att['cot']['all_layers']} / {att['cot']['upper_half']}, filler {att['filler']['all_layers']} /"
            f" {att['filler']['upper_half']}; CoT body {att['cot_body_mass']}, conclusion {att['cot_conc_mass']}; filler text {att['filler_text_mass']}"
            " (unweighted sample means)")
    res["attention"] = att
    return res


# ============================================================================= S6 transport

def probe_layer(l: int, H: dict, cal: np.ndarray, ycal: np.ndarray, y: np.ndarray, folds: list) -> dict:
    """Direct-trained probe at one layer, applied to Direct / CoT / filler held-out states; plus per-condition refits."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import balanced_accuracy_score, roc_auc_score
    from sklearn.preprocessing import StandardScaler
    oof = {k: np.zeros(len(y)) for k in ("direct", "cot", "filler")}
    oof_re = {k: np.zeros(len(y)) for k in ("direct", "cot", "filler")}
    for tr, te in folds:
        Xtr = np.concatenate([cal, H["direct"][tr]]).astype(np.float32)
        ytr = np.concatenate([ycal, y[tr]])
        sc = StandardScaler().fit(Xtr)
        clf = LogisticRegression(C=0.1, max_iter=3000).fit(sc.transform(Xtr), ytr)
        for k in oof:
            oof[k][te] = clf.decision_function(sc.transform(H[k][te].astype(np.float32)))
            sc2 = StandardScaler().fit(H[k][tr].astype(np.float32))
            c2 = LogisticRegression(C=0.1, max_iter=3000).fit(sc2.transform(H[k][tr].astype(np.float32)), y[tr])
            oof_re[k][te] = c2.decision_function(sc2.transform(H[k][te].astype(np.float32)))
    sgn = 2 * y - 1
    m_dir = float((oof["direct"] * sgn).mean())
    r = {"layer": l}
    for k in oof:
        r[k] = {"bacc": float(balanced_accuracy_score(y, oof[k] > 0)), "auc": float(roc_auc_score(y, oof[k])),
                "margin": float((oof[k] * sgn).mean()), "margin_rel": float((oof[k] * sgn).mean() / m_dir),
                "margin_centered_rel": float(((oof[k] - oof[k].mean()) * sgn).mean() / m_dir),
                "bacc_retrained": float(balanced_accuracy_score(y, oof_re[k] > 0))}
    r["_oof"] = {k: oof[k].tolist() for k in oof}
    return r


def transport(model: str) -> dict:
    from joblib import Parallel, delayed
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import balanced_accuracy_score, roc_auc_score
    from sklearn.model_selection import GroupKFold
    from sklearn.preprocessing import StandardScaler

    recs, smp, hd = load(model)
    shards = sorted(glob.glob(str(O / f"h_{model}_[0-9].npz")))
    H = {k: np.concatenate([np.load(f)[k] for f in shards]) for k in ("direct", "cot", "filler")}
    keys = np.concatenate([np.load(f)["keys"] for f in shards])
    assert list(keys) == [r["key"] for r in recs]
    cal = np.load(O / f"calib_{model}.npz")["h"]
    ycal = np.array([m["y_true"] for m in hd["meta"]])
    y = np.array([r["y_true"] for r in recs])
    groups = np.array([r["set"] + ":" + r["pair"] for r in recs])
    folds = list(GroupKFold(5).split(np.zeros(len(y)), y, groups))
    L1 = H["direct"].shape[1]

    rows = Parallel(n_jobs=12)(delayed(probe_layer)(l, {k: np.ascontiguousarray(H[k][:, l]) for k in H}, np.ascontiguousarray(cal[:, l]),
                                                    ycal, y, folds) for l in range(L1))
    rows.sort(key=lambda r: r["layer"])
    lstar = next((r["layer"] for r in rows if r["direct"]["bacc"] >= 0.9), None)
    # cluster bootstrap of balanced accuracy at l*, L/2 and L
    rng = np.random.default_rng(52)
    ug = sorted(set(groups))
    gid = np.array([ug.index(g) for g in groups])
    cnt = rng.multinomial(len(ug), np.full(len(ug), 1 / len(ug)), size=NB)
    mult = cnt[:, gid]

    def bacc_ci(score: np.ndarray) -> list[float]:
        p = score > 0
        out = []
        for b in range(NB):
            w = mult[b]
            tpr = (w * (p & (y == 1))).sum() / max((w * (y == 1)).sum(), 1)
            tnr = (w * (~p & (y == 0))).sum() / max((w * (y == 0)).sum(), 1)
            out.append((tpr + tnr) / 2)
        return [round(float(np.quantile(out, .025)), 3), round(float(np.quantile(out, .975)), 3)]

    key_layers = sorted({l for l in (lstar, (L1 - 1) // 2, L1 - 1) if l is not None})
    cis = {str(l): {k: bacc_ci(np.array(rows[l]["_oof"][k])) for k in ("direct", "cot", "filler")} for l in key_layers}
    for r in rows:
        r.pop("_oof")
    res = {"model": model, "n_eval": len(y), "n_calib": len(ycal), "n_layers_plus_emb": L1, "l_star": lstar, "rows": rows,
           "bacc_ci": cis, "note": "unweighted over the stratified sample; Direct probe trained on Direct states (calibration + "
                                   "train folds), applied unchanged to CoT / filler states of held-out folds"}
    say(f"\nS6 {NAMES[model]}: Direct probe (answer-row residual, target = true relation), l* = {lstar} (first layer with held-out"
        f" Direct bacc >= 0.9) of {L1 - 1}")
    for l in key_layers:
        r = rows[l]
        say(f"  layer {l}: bacc Direct {r['direct']['bacc']:.3f} {cis[str(l)]['direct']} | CoT {r['cot']['bacc']:.3f} {cis[str(l)]['cot']}"
            f" (AUC {r['cot']['auc']:.3f}, margin {r['cot']['margin_rel']:+.2f}, centered {r['cot']['margin_centered_rel']:+.2f})"
            f" | filler {r['filler']['bacc']:.3f} {cis[str(l)]['filler']} (AUC {r['filler']['auc']:.3f}, margin {r['filler']['margin_rel']:+.2f},"
            f" centered {r['filler']['margin_centered_rel']:+.2f}) | retrained D/C/F {r['direct']['bacc_retrained']:.3f}/"
            f"{r['cot']['bacc_retrained']:.3f}/{r['filler']['bacc_retrained']:.3f}")
    mins = {k: min(rows[l][k]["bacc"] for l in range(1, L1)) for k in ("cot", "filler")}
    say(f"  min transfer bacc over layers 1..L: CoT {mins['cot']:.3f}, filler {mins['filler']:.3f}")
    res["min_transfer_bacc"] = mins
    return res


# ============================================================================= table / figure

def cell(d: dict, sign: bool = True) -> str:
    v = f"{d['mean']:+.1f}" if sign else f"{d['mean']:.1f}"
    v = v.replace("-", "$-$")
    lo, hi = (f"{d['lo']:+.1f}".replace("-", "$-$"), f"{d['hi']:+.1f}".replace("-", "$-$"))
    return f"{v}\\,{{\\tiny[{lo},{hi}]}}"


def table(all_res: dict) -> str:
    hdr = [
        "% Suggested caption: \\textbf{Attention-edge knockout at the forced answer position} (Exp-52). Stored \\texttt{CoT}",
        "% rationales (cue-factorial set + occlusion twins, 400 + 400 prompts per model, stratified by Direct/CoT cell,",
        "% population-weighted). Knockouts edit only the answer-prefix rows (``\\textbackslash nAnswer: ('') in all layers:",
        "% image = rows$\\to$image tokens blocked; body / concl. = rows$\\to$rationale before / from the last relational sentence",
        "% blocked; image (all text) = image blocked from every position after it; vis.\\ heads = top-5\\% answer$\\to$image heads",
        "% (calibration split) zeroed at the rows; rand.\\ = layer-matched random heads (mean of 2 sets, CI = widest of the two).",
        "% $\\Delta$Acc and flips (answers that change) in points; concl.\\ columns on prompts with a parsed relation (92-98\\%);",
        "% agree = answer equals the letter implied by the last written relation. 95\\% twin-pair bootstrap CIs.",
        "% Bare tabular; compiles standalone under \\resizebox.",
    ]
    lines = hdr + ["\\begin{tabular}{llcccccccccccc}", "\\toprule",
                   " & & & \\multicolumn{5}{c}{$\\Delta$Acc under knockout} & \\multicolumn{4}{c}{Answer flips} & \\multicolumn{2}{c}{Agree with written} \\\\",
                   "\\cmidrule(lr){4-8}\\cmidrule(lr){9-12}\\cmidrule(lr){13-14}",
                   "Model & Arm & Acc & image & concl. & image (all text) & vis.\\ heads & rand.\\ heads & image & image (all text) & concl."
                   " & concl.\\,+\\,image (all text) & none & $\\Delta$ concl. \\\\",
                   "\\midrule"]
    for m, r in all_res.items():
        c, d = r["results"]["cot"], r["results"]["direct"]
        rnd = lambda arm: {"mean": (arm["r1"]["dacc"]["mean"] + arm["r2"]["dacc"]["mean"]) / 2,
                           "lo": min(arm["r1"]["dacc"]["lo"], arm["r2"]["dacc"]["lo"]), "hi": max(arm["r1"]["dacc"]["hi"], arm["r2"]["dacc"]["hi"])}
        lines.append(f"{NAMES[m]} & CoT & {c['none']['acc']['mean']:.1f} & {cell(c['a']['dacc'])} & {cell(c['c']['dacc'])}"
                     f" & {cell(c['a_full']['dacc'])} & {cell(c['d']['dacc'])} & {cell(rnd(c))} & {cell(c['a']['flip'], False)}"
                     f" & {cell(c['a_full']['flip'], False)} & {cell(c['c']['flip'], False)} & {cell(c['c_afull']['flip'], False)}"
                     f" & {c['none']['agree']['mean']:.1f} & {cell(c['c']['dagree'])} \\\\")
        lines.append(f" & Direct & {d['none']['acc']['mean']:.1f} & {cell(d['a']['dacc'])} & -- & {cell(d['a_full']['dacc'])}"
                     f" & {cell(d['d']['dacc'])} & {cell(rnd(d))} & {cell(d['a']['flip'], False)} & {cell(d['a_full']['flip'], False)}"
                     f" & -- & -- & -- & -- \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    return "\n".join(lines) + "\n"


def figure(all_res: dict, tr: dict) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 7, "font.family": "DejaVu Sans"})
    ms = list(all_res)
    ncol = 1 + (1 if tr else 0)
    fig, axes = plt.subplots(1, ncol, figsize=(7.0 if tr else 4.2, 2.4), gridspec_kw={"width_ratios": [1.4, 1] if tr else [1]})
    axes = np.atleast_1d(axes)
    ax = axes[0]
    conds = [("a", "image"), ("b", "body"), ("c", "concl."), ("a_full", "image\n(all text)"), ("c_afull", "concl. +\nimage (all)"),
             ("d", "vis.\nheads"), ("r", "rand.\nheads")]
    x = np.arange(len(conds))
    wd = 0.8 / (2 * len(ms))
    cols = ["#1f77b4", "#d62728", "#2ca02c"]
    for k, m in enumerate(ms):
        for j, arm in enumerate(("cot", "direct")):
            vals, los, his = [], [], []
            for c, _ in conds:
                R = all_res[m]["results"][arm]
                if c == "r":
                    e = {"mean": (R["r1"]["flip"]["mean"] + R["r2"]["flip"]["mean"]) / 2, "lo": min(R["r1"]["flip"]["lo"], R["r2"]["flip"]["lo"]),
                         "hi": max(R["r1"]["flip"]["hi"], R["r2"]["flip"]["hi"])}
                elif c in R:
                    e = R[c]["flip"]
                else:
                    e = None
                vals.append(e["mean"] if e else np.nan)
                los.append(e["mean"] - e["lo"] if e else 0)
                his.append(e["hi"] - e["mean"] if e else 0)
            pos = x + (2 * k + j - len(ms) + 0.5) * wd
            ax.bar(pos, vals, wd, color=cols[k], alpha=1.0 if arm == "cot" else 0.4, hatch=None if arm == "cot" else "///",
                   edgecolor="white" if arm == "cot" else cols[k], linewidth=0.3,
                   label=f"{NAMES[m]} {'CoT' if arm == 'cot' else 'Direct'}")
            ax.errorbar(pos, vals, yerr=[los, his], fmt="none", ecolor="k", elinewidth=0.5, capsize=1)
    ax.set_xticks(x)
    ax.set_xticklabels([n for _, n in conds], fontsize=6)
    ax.set_ylabel("answers flipped (% of prompts)")
    ax.set_title("(a) Knockout from the answer rows", loc="left", fontsize=7.5, fontweight="bold")
    ax.legend(fontsize=5.2, frameon=False, ncol=1, loc="upper left")
    ax.spines[["top", "right"]].set_visible(False)
    if tr:
        ax = axes[1]
        for k, m in enumerate(ms):
            if m not in tr:
                continue
            rows = tr[m]["rows"]
            L = len(rows) - 1
            xs = [r["layer"] / L for r in rows]
            for arm, ls in (("direct", "-"), ("cot", "--"), ("filler", ":")):
                ax.plot(xs, [r[arm]["auc"] for r in rows], ls, color=cols[k], lw=1)
        from matplotlib.lines import Line2D
        ax.legend([Line2D([], [], color="k", ls=ls, lw=1) for ls in ("-", "--", ":")],
                  ["Direct (held-out)", "CoT (transfer)", "filler + turn (transfer)"], fontsize=5.5, frameon=False, loc="upper left")
        ax.axhline(0.5, color="gray", lw=0.5)
        ax.set_xlabel("layer / L")
        ax.set_ylabel("Direct-trained probe AUC (held-out)")
        ax.set_title("(b) Direct-probe transport", loc="left", fontsize=7.5, fontweight="bold")
        ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(PAPER / "figures" / "supp_e52_knockout.pdf")
    fig.savefig(PAPER / "figures" / "supp_e52_knockout.png", dpi=300)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["qwen3vl_8b", "qwen7b"])
    ap.add_argument("--no-probe", action="store_true", help="reuse bundle/e52_transport.json")
    ap.add_argument("--probe-models", nargs="+", default=None)
    args = ap.parse_args()
    say("Exp-52 rigor (attention-edge knockout at the forced answer position; prereg outputs/e52/prereg.txt)")
    all_res = {m: analyze_model(m) for m in args.models}
    json.dump(all_res, open(B / "e52_knockout.json", "w"), indent=1)
    tr = json.load(open(B / "e52_transport.json")) if (B / "e52_transport.json").exists() else {}
    if not args.no_probe:
        for m in args.probe_models or args.models:
            if glob.glob(str(O / f"h_{m}_[0-9].npz")):
                tr[m] = transport(m)
        json.dump(tr, open(B / "e52_transport.json", "w"), indent=1)
    else:
        for m, t in tr.items():
            say(f"\nS6 {NAMES[m]} (from bundle/e52_transport.json): l* = {t['l_star']}")
            for l, ci in t["bacc_ci"].items():
                r = t["rows"][int(l)]
                say(f"  layer {l}: bacc Direct {r['direct']['bacc']:.3f} {ci['direct']} | CoT {r['cot']['bacc']:.3f} {ci['cot']}"
                    f" (AUC {r['cot']['auc']:.3f}, margin {r['cot']['margin_rel']:+.2f}, centered {r['cot']['margin_centered_rel']:+.2f})"
                    f" | filler {r['filler']['bacc']:.3f} {ci['filler']} (AUC {r['filler']['auc']:.3f}, margin {r['filler']['margin_rel']:+.2f},"
                    f" centered {r['filler']['margin_centered_rel']:+.2f}) | retrained D/C/F {r['direct']['bacc_retrained']:.3f}/"
                    f"{r['cot']['bacc_retrained']:.3f}/{r['filler']['bacc_retrained']:.3f}")
            say(f"  min transfer bacc over layers 1..L: CoT {t['min_transfer_bacc']['cot']:.3f}, filler {t['min_transfer_bacc']['filler']:.3f}")
    tr = {m: t for m, t in tr.items() if m in all_res}
    (PAPER / "tables" / "supp_e52.tex").write_text(table(all_res))
    figure(all_res, tr)
    (B / "rigor_e52.txt").write_text("\n".join(LINES) + "\n")


if __name__ == "__main__":
    main()
