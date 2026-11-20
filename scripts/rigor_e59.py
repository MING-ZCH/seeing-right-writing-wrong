"""Exp-59 analysis: Exp-52 knockouts on R_cut (conclusion removed) for seven open models + 2D left/right.

Reads shared outputs/e59 (e59_knockout.py) and outputs/e52 (reproduction check), writes
  outputs/bundle/e59_knockout.json, outputs/bundle/rigor_e59.txt, paper/tables/supp_e59.tex.
Estimates are population-weighted (N_cell / n_cell within each set; occlusion pooled = mean of the cue-set and twin
estimates); CIs: 2,000 cluster bootstrap draws within each set (twin pairs / image groups), shared by all
conditions of a model (paired contrasts and ratios computed within draws).

Usage: python rigor_e59.py [--models ...]
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

SH = Path(f"{_vtp.OUTPUTS}")
O, O52 = SH / "e59", SH / "e52"
B = Path("outputs/bundle")
PAPER = Path(f"{_vtp.PAPER}")
M7 = ["qwen3b", "qwen7b", "internvl3_8b", "phi35v", "molmo7b", "qwen3vl_8b", "internvl35_8b"]
NAMES = {"qwen3b": "Qwen2.5-VL-3B", "qwen7b": "Qwen2.5-VL-7B", "internvl3_8b": "InternVL3-8B", "phi35v": "Phi-3.5-V",
         "molmo7b": "Molmo-7B-D", "qwen3vl_8b": "Qwen3-VL-8B", "internvl35_8b": "InternVL3.5-8B"}
LR_MODELS = ["internvl3_8b", "internvl35_8b", "qwen7b"]
E52_MODELS = ["qwen3vl_8b", "qwen7b", "molmo7b"]
NB = 2000
CELLS = ("D+C+", "D+C-", "D-C+", "D-C-")
LINES: list[str] = []


def say(s: str = "") -> None:
    print(s)
    LINES.append(s)


def pg(ab, gold: str) -> float:
    if ab is None:
        return np.nan
    m = max(ab)
    e = [math.exp(v - m) for v in ab]
    return e["AB".index(gold)] / sum(e)


def pg_lp(lp: dict | None, gold: str) -> float:
    """P(gold) from a stored letter -> logprob dict (2-letter renormalization)."""
    if not lp:
        return np.nan
    fin = {k: v for k, v in lp.items() if v is not None and math.isfinite(v)}
    if gold not in fin:
        return 0.0
    m = max(fin.values())
    return math.exp(fin[gold] - m) / sum(math.exp(v - m) for v in fin.values())


def load(model: str, which: str) -> tuple[list[dict], dict]:
    recs = []
    for f in sorted(glob.glob(str(O / f"{model}_{which}_[0-9].jsonl"))):
        for l in open(f):
            try:
                recs.append(json.loads(l))
            except json.JSONDecodeError:
                print(f"[warn] partial line in {f}")
    smp = json.load(open(O / f"{'sample' if which == 'occ' else 'lr'}_{model}.json"))
    return recs, smp


class WB:
    """Population-weighted means with a paired cluster bootstrap within each set."""

    def __init__(self, recs: list[dict], smp: dict, sets: tuple[str, ...], seed: int = 59):
        rng = np.random.default_rng(seed)
        self.sets = sets
        self.n = len(recs)
        self.set = np.array([r["set"] for r in recs])
        self.w = np.array([smp["N_cell"][r["set"]][r["cell"]] / smp["n_cell"][r["set"]][r["cell"]] for r in recs], float)
        self.mult = np.zeros((NB, self.n), np.float32)
        for s in sets:
            ix = np.where(self.set == s)[0]
            if not len(ix):
                continue
            pairs = sorted({recs[i]["pair"] for i in ix})
            pid = {p: k for k, p in enumerate(pairs)}
            cl = np.array([pid[recs[i]["pair"]] for i in ix])
            cnt = rng.multinomial(len(pairs), np.full(len(pairs), 1 / len(pairs)), size=NB)
            self.mult[:, ix] = cnt[:, cl]

    def draws(self, x: np.ndarray, mask: np.ndarray | None = None, sets: tuple[str, ...] | None = None,
              weighted: bool = True) -> tuple[float, np.ndarray, int]:
        mask = np.isfinite(x) if mask is None else (mask & np.isfinite(x))
        xx = np.where(mask, x, 0.0)
        w = (self.w if weighted else np.ones(self.n)) * mask
        pts, bs = [], []
        for s in sets or self.sets:
            m = self.set == s
            if w[m].sum() == 0:
                continue
            pts.append((w[m] * xx[m]).sum() / w[m].sum())
            mw = self.mult[:, m] * w[m]
            bs.append((mw * xx[m]).sum(1) / np.maximum(mw.sum(1), 1e-12))
        return float(np.mean(pts)), np.mean(bs, 0), int(mask.sum())

    def est(self, x, mask=None, sets=None, weighted=True) -> dict:
        p, b, n = self.draws(x, mask, sets, weighted)
        return summ(p, b, n)


def summ(p: float, b: np.ndarray, n: int, scale: float = 100) -> dict:
    return {"mean": round(scale * p, 2), "lo": round(scale * float(np.quantile(b, .025)), 2),
            "hi": round(scale * float(np.quantile(b, .975)), 2), "n": n}


def f1(d: dict, sign: bool = False, ci: bool = True) -> str:
    v = f"{d['mean']:+.1f}" if sign else f"{d['mean']:.1f}"
    return f"{v} [{d['lo']:.1f}, {d['hi']:.1f}]" if ci else v


def arrays(recs: list[dict], arms: dict) -> tuple[dict, dict]:
    gold = [r["gold"] for r in recs]
    P = {arm: {c: np.array([pg(r[arm].get(c), gold[i]) for i, r in enumerate(recs)]) for c in cs} for arm, cs in arms.items()}
    pred = {arm: {c: np.where(np.isnan(v), np.nan, (v > .5).astype(float)) for c, v in d.items()} for arm, d in P.items()}
    return P, pred


def knock(W: WB, P: dict, pred: dict, arm: str, c: str, sets=None) -> dict:
    fl = np.where(np.isnan(pred[arm][c]), np.nan, (pred[arm][c] != pred[arm]["none"]).astype(float))
    ad = np.abs(P[arm][c] - P[arm]["none"])
    return {"flip": W.draws(fl, sets=sets), "absdP": W.draws(ad, sets=sets),
            "dacc": W.draws(pred[arm][c] - pred[arm]["none"], sets=sets)}


def pack(k: dict) -> dict:
    return {m: summ(*v) for m, v in k.items()}


def contrast(a: tuple, b: tuple) -> dict:
    """a - b from paired draws (point, draws, n)."""
    return summ(a[0] - b[0], a[1] - b[1], min(a[2], b[2]))


def ratio(a: tuple, b: tuple) -> dict:
    """a / b (absolute ratio, not in points)."""
    return summ(a[0] / b[0], a[1] / np.maximum(b[1], 1e-12), min(a[2], b[2]), scale=1)


# ============================================================================= sanity

def sanity_common(recs: list[dict], arms: tuple[str, ...]) -> dict:
    out = {}
    for arm in arms:
        rr = [r for r in recs if arm in r and "s2_identity" in r[arm]]
        if not rr:
            continue
        idn = max(abs(pg(r[arm]["s2_identity"], r["gold"]) - pg(r[arm]["none"], r["gold"])) for r in rr)
        ref = [abs(pg(r[arm]["s2_ref"], r["gold"]) - pg(r[arm]["none"], r["gold"])) for r in rr]
        out[arm] = {"n": len(rr), "identity_max_abs_dP": float(idn), "ref_max_abs_dP": round(float(max(ref)), 4),
                    "ref_mean_abs_dP": round(float(np.mean(ref)), 4),
                    "ref_argmax_agree": round(float(np.mean([(pg(r[arm]["s2_ref"], r["gold"]) > .5) == (pg(r[arm]["none"], r["gold"]) > .5) for r in rr])), 4)}
    lead = [r[a]["_n"]["lead_mismatch"] for r in recs for a in arms if a in r]
    out["tail_lead_mismatch_max"] = int(max(lead)) if lead else None
    return out


def agree_stored(mine: np.ndarray, stored: np.ndarray) -> dict:
    ok = np.isfinite(mine) & np.isfinite(stored)
    d = np.abs(mine[ok] - stored[ok])
    dis = (mine[ok] > .5) != (stored[ok] > .5)
    near = (np.abs(mine[ok][dis] - .5) < .15) & (np.abs(stored[ok][dis] - .5) < .15)
    return {"n": int(ok.sum()), "n_disagree": int(dis.sum()), "n_disagree_both_within_0.15_of_half": int(near.sum()),
            "argmax_agree": round(float(((mine[ok] > .5) == (stored[ok] > .5)).mean()), 4) if ok.any() else None,
            "max_abs_dP": round(float(d.max()), 4) if ok.any() else None, "mean_abs_dP": round(float(d.mean()), 4) if ok.any() else None}


def e52_repro(model: str, recs: list[dict]) -> dict:
    """Per-prompt comparison with the stored Exp-52 records (same sample)."""
    old = {}
    for f in sorted(glob.glob(str(O52 / f"{model}_[0-9].jsonl"))):
        for l in open(f):
            r = json.loads(l)
            old[r["key"]] = r
    for f in sorted(glob.glob(str(O52 / f"{model}_x_[0-9].jsonl"))):
        for l in open(f):
            x = json.loads(l)
            if x["key"] in old and "c_afull" not in old[x["key"]]["cot"]:
                old[x["key"]]["cot"]["c_afull"] = x["c_afull"]
    out = {"n_matched": sum(r["key"] in old for r in recs), "n_e52": len(old)}
    for arm_new, arm_old, conds in (("direct", "direct", ("none", "a_full")), ("full", "cot", ("none", "a_full", "c", "c_afull"))):
        for c in conds:
            a, b = [], []
            for r in recs:
                o = old.get(r["key"])
                if o is None or c not in r[arm_new] or c not in o[arm_old]:
                    continue
                a.append(pg(r[arm_new][c], r["gold"]))
                b.append(pg(o[arm_old][c], r["gold"]))
            a, b = np.array(a), np.array(b)
            out[f"{arm_new}:{c}"] = {"n": len(a), "max_abs_dP": round(float(np.abs(a - b).max()), 4),
                                     "mean_abs_dP": round(float(np.abs(a - b).mean()), 5),
                                     "argmax_agree": round(float(((a > .5) == (b > .5)).mean()), 4)}
    return out


# ============================================================================= occlusion

def analyze_occ(model: str) -> dict:
    recs, smp = load(model, "occ")
    W = WB(recs, smp, ("cues", "twins"))
    arms = {"direct": ("none", "a_full"), "cut": ("none", "a_full", "a", "rel"), "full": ("none", "a_full", "c", "c_afull")}
    P, pred = arrays(recs, arms)
    relc = np.array([r["has_rel_cut"] for r in recs])
    relf = np.array([r["has_rel"] for r in recs])
    res = {"model": model, "n": len(recs), "n_cell": smp["n_cell"], "N_cell": smp["N_cell"],
           "n_rel_cut": int(relc.sum()), "n_rel_full": int(relf.sum())}
    say(f"\n=== {NAMES[model]} ({model}) occlusion: n = {len(recs)} (cues {int((W.set == 'cues').sum())}, twins "
        f"{int((W.set == 'twins').sum())}); last relation parsed in R_cut {int(relc.sum())}, in full R {int(relf.sum())}")
    # ---------------- sanity
    san = sanity_common(recs, ("direct", "cut", "full"))
    tw = np.array([r["set"] == "twins" for r in recs])
    st14 = np.array([pg_lp(r["e14_cut_lp"], r["gold"]) for r in recs])
    san["S1_cut_vs_e14_twins"] = agree_stored(np.where(tw, P["cut"]["none"], np.nan), st14)
    san["S1_full_vs_e10"] = agree_stored(P["full"]["none"], np.array([r["p_cot_stored"] for r in recs], float))
    san["S1_direct_vs_e10"] = agree_stored(P["direct"]["none"], np.array([r["p_direct_stored"] for r in recs], float))
    if model in E52_MODELS:
        san["S3_e52"] = e52_repro(model, recs)
    res["sanity"] = san
    s1 = san["S1_cut_vs_e14_twins"]
    if s1["n"]:
        say(f"Sanity S1: R_cut none vs stored e14 (twins, n {s1['n']}): argmax agreement {100 * s1['argmax_agree']:.1f}%, "
            f"max |dP| {s1['max_abs_dP']}, mean {s1['mean_abs_dP']} -> {'PASS' if s1['argmax_agree'] >= .99 else 'FAIL'} (>= 99%);"
            f" disagreements {s1['n_disagree']}, of which both P within 0.15 of 0.5: {s1['n_disagree_both_within_0.15_of_half']}")
    for k in ("S1_full_vs_e10", "S1_direct_vs_e10"):
        v = san[k]
        say(f"  {k}: agreement {100 * v['argmax_agree']:.1f}%, max |dP| {v['max_abs_dP']}, mean {v['mean_abs_dP']} (n {v['n']})")
    say("Sanity S2 (first 20 prompts per worker): " + "; ".join(
        f"{a}: identity max|dP| {san[a]['identity_max_abs_dP']:.1e}, stock-attention ref max|dP| {san[a]['ref_max_abs_dP']} "
        f"mean {san[a]['ref_mean_abs_dP']} agree {100 * san[a]['ref_argmax_agree']:.0f}%" for a in ("direct", "cut", "full") if a in san)
        + f"; tail alignment leading mismatches max {san['tail_lead_mismatch_max']}")
    if "S3_e52" in san:
        s3 = san["S3_e52"]
        say(f"Sanity S3 (vs stored Exp-52, {s3['n_matched']}/{s3['n_e52']} prompts): " + "; ".join(
            f"{k} max|dP| {v['max_abs_dP']} mean {v['mean_abs_dP']} agree {100 * v['argmax_agree']:.1f}%"
            for k, v in s3.items() if isinstance(v, dict)))
    # ---------------- knockouts
    K = {}
    for arm, cs in arms.items():
        for c in cs[1:]:
            K[f"{arm}:{c}"] = knock(W, P, pred, arm, c)
    out = {k: pack(v) for k, v in K.items()}
    acc = {arm: summ(*W.draws(pred[arm]["none"])) for arm in arms}
    # primary contrasts
    dA, cA = K["direct:a_full"], K["cut:a_full"]
    con = {"flip_direct_minus_cut": contrast(dA["flip"], cA["flip"]),
           "absdP_direct_minus_cut": contrast(dA["absdP"], cA["absdP"]),
           "ratio_absdP_cut_over_direct": ratio(cA["absdP"], dA["absdP"]),
           "ratio_flip_cut_over_direct": ratio(cA["flip"], dA["flip"]),
           "absdP_cut_minus_full_afull": contrast(cA["absdP"], K["full:a_full"]["absdP"]),
           "flip_cut_minus_full_afull": contrast(cA["flip"], K["full:a_full"]["flip"])}
    h1 = (con["flip_direct_minus_cut"]["lo"] > 0 and con["absdP_direct_minus_cut"]["lo"] > 0
          and con["ratio_absdP_cut_over_direct"]["mean"] <= 0.5)
    res["H1_supported"] = bool(h1)
    # agreement with the written relation of R_cut
    wc = [r["written_cut"] for r in recs]
    gl = [r["gold"] for r in recs]
    letter = lambda p, g: g if p > .5 else ("B" if g == "A" else "A")
    ag = {}
    for c in ("none", "a_full", "rel"):
        v = np.array([float(letter(P["cut"][c][i], gl[i]) == wc[i]) if (wc[i] and np.isfinite(P["cut"][c][i])) else np.nan
                      for i in range(len(recs))])
        ag[c] = W.draws(v, relc)
    out["cut_agree_written"] = {c: summ(*v) for c, v in ag.items()}
    out["cut_dagree_written"] = {c: contrast(ag[c], ag["none"]) for c in ("a_full", "rel")}
    # by set
    out["by_set"] = {s: {k: pack(knock(W, P, pred, *k.split(":"), sets=(s,))) for k in ("direct:a_full", "cut:a_full", "full:a_full")}
                     for s in ("cues", "twins")}
    # per cell (unweighted)
    cells = np.array([r["cell"] for r in recs])
    out["cells_flip"] = {cl: {k: summ(*W.draws(np.where(np.isnan(pred[a][c]), np.nan, (pred[a][c] != pred[a]["none"]).astype(float)),
                                               cells == cl, weighted=False))
                              for k, (a, c) in (("direct:a_full", ("direct", "a_full")), ("cut:a_full", ("cut", "a_full")))}
                         for cl in CELLS}
    res.update({"acc": acc, "knock": out, "contrasts": con})
    # ---------------- print
    say(f"Accuracy (none): Direct {f1(acc['direct'])}, R_cut {f1(acc['cut'])}, full R {f1(acc['full'])}")
    say("PRIMARY image blocked from all text (a_full), pooled cues + twins:")
    say(f"  Direct: flip {f1(out['direct:a_full']['flip'])}% | mean|dP| {f1(out['direct:a_full']['absdP'])} | dAcc {f1(out['direct:a_full']['dacc'], True)}")
    say(f"  R_cut : flip {f1(out['cut:a_full']['flip'])}% | mean|dP| {f1(out['cut:a_full']['absdP'])} | dAcc {f1(out['cut:a_full']['dacc'], True)}")
    say(f"  Direct - R_cut: flip {f1(con['flip_direct_minus_cut'], True)}, |dP| {f1(con['absdP_direct_minus_cut'], True)};"
        f" ratio |dP| R_cut/Direct {con['ratio_absdP_cut_over_direct']['mean']:.2f} [{con['ratio_absdP_cut_over_direct']['lo']:.2f},"
        f" {con['ratio_absdP_cut_over_direct']['hi']:.2f}] -> H1 {'SUPPORTED' if h1 else 'NOT supported'}")
    say(f"  full R: flip {f1(out['full:a_full']['flip'])}% | |dP| {f1(out['full:a_full']['absdP'])};"
        f" R_cut - full R |dP| {f1(con['absdP_cut_minus_full_afull'], True)}")
    say(f"Secondary R_cut: a (answer->image) flip {f1(out['cut:a']['flip'])}% |dP| {f1(out['cut:a']['absdP'])}; rel (answer->last"
        f" relation) flip {f1(out['cut:rel']['flip'])}% |dP| {f1(out['cut:rel']['absdP'])}; agreement with written R_cut relation"
        f" none {f1(out['cut_agree_written']['none'])}, a_full {f1(out['cut_dagree_written']['a_full'], True)}, rel"
        f" {f1(out['cut_dagree_written']['rel'], True)}")
    say(f"Full R: c flip {f1(out['full:c']['flip'])}%, c_afull flip {f1(out['full:c_afull']['flip'])}%")
    for s in ("cues", "twins"):
        b = out["by_set"][s]
        say(f"  {s}: a_full flip Direct {f1(b['direct:a_full']['flip'], ci=False)} / R_cut {f1(b['cut:a_full']['flip'], ci=False)}"
            f" / full {f1(b['full:a_full']['flip'], ci=False)}; |dP| {f1(b['direct:a_full']['absdP'], ci=False)} /"
            f" {f1(b['cut:a_full']['absdP'], ci=False)} / {f1(b['full:a_full']['absdP'], ci=False)}")
    say("  per cell (unweighted) a_full flip Direct / R_cut: " + " ".join(
        f"{cl} {out['cells_flip'][cl]['direct:a_full']['mean']:.1f}/{out['cells_flip'][cl]['cut:a_full']['mean']:.1f}" for cl in CELLS))
    return res


# ============================================================================= 2D left/right

def analyze_lr(model: str) -> dict:
    recs, smp = load(model, "lr")
    W = WB(recs, smp, ("lr",))
    arms = {"direct": ("none", "a_full"), "cut": ("none", "a_full", "a", "rel")}
    P, pred = arrays(recs, arms)
    rel = np.array([r["has_rel_cut"] for r in recs])
    res = {"model": model, "n": len(recs), "n_cell": smp["n_cell"], "N_cell": smp["N_cell"], "n_rel_cut": int(rel.sum())}
    say(f"\n=== {NAMES[model]} 2D left/right (main pool): n = {len(recs)}, image groups {len({r['pair'] for r in recs})},"
        f" relation sentence in R_cut {int(rel.sum())}")
    san = sanity_common(recs, ("direct", "cut"))
    mine, stored = [], []
    for r in recs:
        for y, ab in r["e3x"].items():
            st = r["e3x_lp"].get(y) if r["e3x_lp"] else None
            if st is None:
                continue
            mine.append(pg(ab, r["gold"]))
            stored.append(pg_lp(st, r["gold"]))
    san["S1_e3x"] = agree_stored(np.array(mine), np.array(stored))
    san["S1_direct_vs_main"] = agree_stored(P["direct"]["none"], np.array([r["p_direct_stored"] for r in recs], float))
    san["S1_cot_full_main_note"] = "full-R arm not run on the main pool"
    res["sanity"] = san
    s1 = san["S1_e3x"]
    say(f"Sanity S1: R_cut + appended conclusion vs stored e3x (Y = gold / wrong, n {s1['n']}): agreement "
        f"{100 * s1['argmax_agree']:.1f}% -> {'PASS' if s1['argmax_agree'] >= .99 else 'FAIL'} (max |dP| {s1['max_abs_dP']}, mean"
        f" {s1['mean_abs_dP']}; disagreements {s1['n_disagree']}, both P within 0.15 of 0.5: {s1['n_disagree_both_within_0.15_of_half']});"
        f" Direct vs Tab. 1 records {100 * san['S1_direct_vs_main']['argmax_agree']:.1f}% (max |dP| {san['S1_direct_vs_main']['max_abs_dP']})")
    say("Sanity S2: " + "; ".join(f"{a}: identity {san[a]['identity_max_abs_dP']:.1e}, ref max|dP| {san[a]['ref_max_abs_dP']}"
                                  for a in ("direct", "cut") if a in san))
    K = {f"{arm}:{c}": knock(W, P, pred, arm, c) for arm, cs in arms.items() for c in cs[1:]}
    out = {k: pack(v) for k, v in K.items()}
    acc = {arm: summ(*W.draws(pred[arm]["none"])) for arm in arms}
    dA, cA = K["direct:a_full"], K["cut:a_full"]
    con = {"flip_direct_minus_cut": contrast(dA["flip"], cA["flip"]), "absdP_direct_minus_cut": contrast(dA["absdP"], cA["absdP"]),
           "ratio_absdP_cut_over_direct": ratio(cA["absdP"], dA["absdP"]), "ratio_flip_cut_over_direct": ratio(cA["flip"], dA["flip"])}
    h2 = out["cut:a_full"]["flip"]["lo"] > 5 and out["cut:a_full"]["absdP"]["lo"] > 5 and con["ratio_absdP_cut_over_direct"]["mean"] >= .25
    res.update({"acc": acc, "knock": out, "contrasts": con, "H2_image_dependent": bool(h2)})
    say(f"Accuracy (none): Direct {f1(acc['direct'])}, R_cut {f1(acc['cut'])}")
    say(f"  a_full Direct: flip {f1(out['direct:a_full']['flip'])}% | |dP| {f1(out['direct:a_full']['absdP'])} | dAcc {f1(out['direct:a_full']['dacc'], True)}")
    say(f"  a_full R_cut : flip {f1(out['cut:a_full']['flip'])}% | |dP| {f1(out['cut:a_full']['absdP'])} | dAcc {f1(out['cut:a_full']['dacc'], True)}")
    say(f"  ratio |dP| R_cut/Direct {con['ratio_absdP_cut_over_direct']['mean']:.2f} [{con['ratio_absdP_cut_over_direct']['lo']:.2f},"
        f" {con['ratio_absdP_cut_over_direct']['hi']:.2f}]; Direct - R_cut flip {f1(con['flip_direct_minus_cut'], True)}"
        + (f" -> H2 (still image-dependent) {'SUPPORTED' if h2 else 'NOT supported'}" if model != "qwen7b" else " (descriptive)"))
    say(f"  R_cut a flip {f1(out['cut:a']['flip'])}% |dP| {f1(out['cut:a']['absdP'])}; rel flip {f1(out['cut:rel']['flip'])}% |dP| {f1(out['cut:rel']['absdP'])}")
    return res


def analyze_conflict(model: str) -> dict | None:
    """Prereg addendum 1: R_cut + appended gold / wrong conclusion, none / a_full / a (2D left/right)."""
    recs = []
    for f in sorted(glob.glob(str(O / f"{model}_conflict_[0-9].jsonl"))):
        recs += [json.loads(l) for l in open(f)]
    if not recs:
        return None
    smp = json.load(open(O / f"lr_{model}.json"))
    W = WB(recs, smp, ("lr",))
    res = {"n": len(recs)}
    D = {}
    for c in ("none", "a_full", "a"):
        fg = np.array([float(pg(r["conf"][r["gold"]][c], r["gold"]) > .5) for r in recs])
        fw = np.array([float(pg(r["conf"][r["wrong"]][c], r["gold"]) <= .5) for r in recs])
        D[c] = {"F_gold": W.draws(fg), "F_wrong": W.draws(fw), "S": W.draws(fg - fw), "reject_wrong": W.draws(1 - fw)}
        res[c] = {k: summ(*v) for k, v in D[c].items()}
    res["dS_none_minus_afull"] = contrast(D["none"]["S"], D["a_full"]["S"])
    res["dS_none_minus_a"] = contrast(D["none"]["S"], D["a"]["S"])
    say(f"\n--- {NAMES[model]} 2D left/right, R_cut + appended conclusion (prereg addendum 1, exploratory; n {len(recs)}):")
    for c in ("none", "a_full", "a"):
        say(f"  {c:6s}: F(gold concl.) {f1(res[c]['F_gold'])} | F(wrong concl.) {f1(res[c]['F_wrong'])} | S {f1(res[c]['S'], True)}"
            f" | answers gold despite wrong conclusion {f1(res[c]['reject_wrong'])}")
    say(f"  dS = S(none) - S(image from all text) {f1(res['dS_none_minus_afull'], True)}; S(none) - S(answer->image) {f1(res['dS_none_minus_a'], True)}")
    return res


# ============================================================================= table

def c_(d: dict, pct: bool = True) -> str:
    v = f"{d['mean']:.1f}"
    return f"{v}\\,{{\\tiny[{d['lo']:.1f},{d['hi']:.1f}]}}"


def cs_(d: dict) -> str:
    m = lambda x: f"{x:+.1f}".replace("-", "$-$")
    return f"{m(d['mean'])}\\,{{\\tiny[{m(d['lo'])},{m(d['hi'])}]}}"


def table(occ: dict, lr: dict) -> str:
    hdr = [
        "% Suggested caption: \\textbf{Image knockout after writing, with the conclusion removed} (Exp-59). Teacher-forced",
        "% readout at ``Answer: ('' for Direct and for the stored rationale with its concluding sentence removed (R$_\\mathrm{cut}$,",
        "% the truncation of the fixed-rationale twin swap). \\emph{Image (all text)} = image tokens blocked from every",
        "% text position after the image (question, rationale, answer prefix); flips = answers that change, $|\\Delta P|$ =",
        "% mean absolute change of P(gold), both in points; ratio = $|\\Delta P|$ under R$_\\mathrm{cut}$ / under Direct.",
        "% Occlusion = cue-factorial set + occlusion twins (400 + 400 prompts per model, stratified by Direct/CoT cell,",
        "% population-weighted); 2D left/right = CV-Bench 2D Relation + Synthetic Relative Position (400 items per model).",
        "% Full R = Exp-52 protocol (rationale with its conclusion). Last relation = answer prefix $\\to$ last remaining",
        "% relational sentence of R$_\\mathrm{cut}$ blocked. Last two columns (2D left/right, exploratory, prereg addendum):",
        "% R$_\\mathrm{cut}$ + appended conclusion ``Therefore, the answer is (Y)'', $S$ = P(answer = Y $|$ Y gold) $-$",
        "% P(answer = Y $|$ Y wrong) (as in Tab.~supp\\_answer\\_step), without knockout and with the image blocked from all text.",
        "% 95\\% cluster bootstrap CIs (twin pairs / image groups). Bare tabular; compiles standalone under \\resizebox.",
    ]
    lines = hdr + ["\\begin{tabular}{llcccccccccc}", "\\toprule",
                   " & & \\multicolumn{2}{c}{Direct, image (all text)} & \\multicolumn{2}{c}{R$_\\mathrm{cut}$, image (all text)}"
                   " & & Full R, image (all text) & \\multicolumn{2}{c}{R$_\\mathrm{cut}$, last relation}"
                   " & \\multicolumn{2}{c}{R$_\\mathrm{cut}$ + concl.: $S$} \\\\",
                   "\\cmidrule(lr){3-4}\\cmidrule(lr){5-6}\\cmidrule(lr){8-8}\\cmidrule(lr){9-10}\\cmidrule(lr){11-12}",
                   "Items & Model & flips & $|\\Delta P|$ & flips & $|\\Delta P|$ & ratio & flips & flips & $|\\Delta P|$ & none & image (all text) \\\\",
                   "\\midrule"]
    for blk, D in (("Occlusion", occ), ("2D left/right", lr)):
        first = True
        for m in M7:
            if m not in D:
                continue
            k, con = D[m]["knock"], D[m]["contrasts"]
            ra = con["ratio_absdP_cut_over_direct"]
            full = c_(k["full:a_full"]["flip"]) if "full:a_full" in k else "--"
            cf = D[m].get("conflict")
            sc = f"{cs_(cf['none']['S'])} & {cs_(cf['a_full']['S'])}" if cf else "-- & --"
            lines.append(f"{blk if first else ''} & {NAMES[m]} & {c_(k['direct:a_full']['flip'])} & {c_(k['direct:a_full']['absdP'])}"
                         f" & {c_(k['cut:a_full']['flip'])} & {c_(k['cut:a_full']['absdP'])} & {ra['mean']:.2f}\\,{{\\tiny[{ra['lo']:.2f},{ra['hi']:.2f}]}}"
                         f" & {full} & {c_(k['cut:rel']['flip'])} & {c_(k['cut:rel']['absdP'])} & {sc} \\\\")
            first = False
        if blk == "Occlusion":
            lines.append("\\midrule")
    lines += ["\\bottomrule", "\\end{tabular}"]
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=M7)
    ap.add_argument("--no-paper", action="store_true")
    args = ap.parse_args()
    say("Exp-59 rigor (knockouts on R_cut for seven open models + 2D left/right; prereg outputs/e59/prereg.txt)")
    occ = {m: analyze_occ(m) for m in args.models if glob.glob(str(O / f"{m}_occ_[0-9].jsonl"))}
    lr = {m: analyze_lr(m) for m in LR_MODELS if m in args.models and glob.glob(str(O / f"{m}_lr_[0-9].jsonl"))}
    for m in lr:
        cf = analyze_conflict(m)
        if cf:
            lr[m]["conflict"] = cf
    say("\n=== SUMMARY")
    say("Occlusion, image from all text: model | Direct flip / |dP| | R_cut flip / |dP| | ratio | full-R flip | H1")
    for m, r in occ.items():
        k, c = r["knock"], r["contrasts"]
        say(f"  {NAMES[m]:15s} {f1(k['direct:a_full']['flip'], ci=False)} / {f1(k['direct:a_full']['absdP'], ci=False)} | "
            f"{f1(k['cut:a_full']['flip'])} / {f1(k['cut:a_full']['absdP'])} | {c['ratio_absdP_cut_over_direct']['mean']:.2f} | "
            f"{f1(k['full:a_full']['flip'], ci=False)} | {'yes' if r['H1_supported'] else 'NO'} | S1 e14 "
            f"{r['sanity']['S1_cut_vs_e14_twins']['argmax_agree']}")
    h1_all = len(occ) == 7 and all(r["H1_supported"] for r in occ.values())
    say(f"H1 (all seven models): {'SUPPORTED' if h1_all else 'NOT supported'} ({sum(r['H1_supported'] for r in occ.values())}/{len(occ)})")
    for m, r in lr.items():
        k, c = r["knock"], r["contrasts"]
        say(f"  L/R {NAMES[m]:15s} Direct {f1(k['direct:a_full']['flip'], ci=False)} / {f1(k['direct:a_full']['absdP'], ci=False)} | R_cut "
            f"{f1(k['cut:a_full']['flip'])} / {f1(k['cut:a_full']['absdP'])} | ratio {c['ratio_absdP_cut_over_direct']['mean']:.2f}"
            f" | H2 {'yes' if r['H2_image_dependent'] else 'no'} | S1 e3x {100 * r['sanity']['S1_e3x']['argmax_agree']:.1f}%")
    B.mkdir(parents=True, exist_ok=True)
    json.dump({"occlusion": occ, "left_right": lr, "H1_all_seven": h1_all}, open(B / "e59_knockout.json", "w"), indent=1)
    (B / "rigor_e59.txt").write_text("\n".join(LINES) + "\n")
    if not args.no_paper:
        (PAPER / "tables" / "supp_e59.tex").write_text(table(occ, lr))


if __name__ == "__main__":
    main()
