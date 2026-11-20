"""Exp-60 analysis: image effect of a horizontal mirror under a fixed rationale on natural left/right items.

E_c = 100 * mean_i(1[pred_c(v_i) = g_i] - 1[pred_c(mirror(v_i)) = g_i]) for c in {direct, rcut, full}; primary =
E_rcut vs E_direct per model and 7-model mean on the filtered CV-Bench 2D Relation left/right set (prereg
outputs/e60/prereg.txt). CIs: 2,000 item-cluster bootstrap draws (seed 60), shared by all models and contexts.

-> outputs/bundle/e60_mirror.json, outputs/bundle/rigor_e60.txt, paper/tables/supp_e60.tex
   (copies of the json / txt in shared outputs/e60/)

Usage: python rigor_e60.py [--no-table]
"""
from __future__ import annotations
import vtax_paths as _vtp  # noqa: E402

import argparse
import json
import math
import os
import shutil
from collections import Counter
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path

import numpy as np

os.environ.setdefault("VTAX_DATA_ROOT", f"{_vtp.DATA_ROOT}")
import vx  # noqa: E402
from e60_items import E60, JUDGES, M7, r_cut, stored_r  # noqa: E402
from e60_judge import has_lr, rel_key  # noqa: E402

REPO = Path(__file__).resolve().parent
BUNDLE = Path(_vtp.OUTPUTS) / "bundle"
TABLE = REPO.parent / f"{_vtp.PAPER}" / "tables" / "supp_e60.tex"
NAMES = {"qwen3b": "Qwen2.5-VL-3B", "qwen7b": "Qwen2.5-VL-7B", "internvl3_8b": "InternVL3-8B", "phi35v": "Phi-3.5-Vision",
         "molmo7b": "Molmo-7B-D", "qwen3vl_8b": "Qwen3-VL-8B", "internvl35_8b": "InternVL3.5-8B"}
CTXS = ("direct", "rcut", "full")
NB, SEED = 2000, 60


def h(x: float, nd: int = 1) -> str:
    """Half-up rounding with sign."""
    q = Decimal(str(round(x, 6))).quantize(Decimal(10) ** -nd, rounding=ROUND_HALF_UP)
    return f"{q:+.{nd}f}"


def u(x: float, nd: int = 1) -> str:
    """Half-up rounding without forced sign."""
    q = Decimal(str(round(x, 6))).quantize(Decimal(10) ** -nd, rounding=ROUND_HALF_UP)
    return f"{q:.{nd}f}"


def pe(r: dict) -> str | None:
    """Prediction of a readout record: argmax of lp_sum (exact) if present else top-20 (= build_bundle.pe)."""
    sc = r.get("lp_sum") or r["top20"]
    fin = {k: v for k, v in sc.items() if v is not None and math.isfinite(v)}
    return max(fin, key=fin.get) if fin else None


def p2(lp: dict, g: str) -> float:
    """Two-letter renormalized probability of g from lp_sum."""
    pr = vx.probs_from_lp(lp)
    return pr[g] if pr else 0.5


class Boot:
    """Shared item-cluster bootstrap over a fixed item list."""

    def __init__(self, groups: list[str]):
        keys = sorted(set(groups))
        pos = {k: i for i, k in enumerate(keys)}
        self.inv = np.array([pos[g] for g in groups])
        self.c = len(keys)
        self.idx = np.random.default_rng(SEED).integers(0, self.c, size=(NB, self.c))
        self.cnt = np.bincount(self.inv, minlength=self.c).astype(float)

    def draws(self, v: np.ndarray, mask: np.ndarray | None = None) -> np.ndarray:
        """Bootstrap means of v (optionally over a subset mask); NaN where a draw has no item."""
        m = np.ones(len(v), bool) if mask is None else mask
        s = np.bincount(self.inv[m], weights=v[m], minlength=self.c)
        n = np.bincount(self.inv[m], minlength=self.c).astype(float)
        with np.errstate(invalid="ignore", divide="ignore"):
            return s[self.idx].sum(1) / n[self.idx].sum(1)


def ci(point: float, d: np.ndarray) -> dict:
    """Point + 95% percentile CI (x100 already applied by caller)."""
    d = d[np.isfinite(d)]
    return {"mean": round(float(point), 4), "lo": round(float(np.percentile(d, 2.5)), 4),
            "hi": round(float(np.percentile(d, 97.5)), 4)}


def rel_labels() -> dict[str, str]:
    """Consensus R_cut relation per task id: LEFT / RIGHT / NONE / unresolved (judged tasks only)."""
    lab = {j: {r["tid"]: r["label"] for r in vx.read_jsonl(f"{E60}/judge_rel_{j}.jsonl") if r.get("label")} for j in JUDGES}
    out = {}
    for t in vx.read_jsonl(f"{E60}/rel_tasks.jsonl"):
        a, b = (lab[j].get(t["tid"]) for j in JUDGES)
        if a is None or b is None:
            out[t["tid"]] = "missing"
        else:
            out[t["tid"]] = a if a == b else "unresolved"
    return out


def judge_agreement() -> dict:
    """Two-judge agreement and Cohen's kappa on the relation task."""
    lab = {j: {r["tid"]: r["label"] for r in vx.read_jsonl(f"{E60}/judge_rel_{j}.jsonl") if r.get("label")} for j in JUDGES}
    ks = sorted(set(lab[JUDGES[0]]) & set(lab[JUDGES[1]]))
    if not ks:
        return {}
    a = [lab[JUDGES[0]][k] for k in ks]
    b = [lab[JUDGES[1]][k] for k in ks]
    po = np.mean([x == y for x, y in zip(a, b)])
    ca, cb = Counter(a), Counter(b)
    pe_ = sum(ca[c] * cb[c] for c in set(a) | set(b)) / len(ks) ** 2
    return {"n": len(ks), "agree": round(100 * po, 2), "kappa": round((po - pe_) / (1 - pe_), 3),
            "dist": {j: dict(Counter(lab[j].values())) for j in JUDGES}}


def model_arrays(m: str, items: list[dict], rel: dict) -> dict:
    """Per-item arrays for one model on a fixed item list."""
    recs = {(r["item_id"], r["img"], r["ctx"]): r for r in vx.read_jsonl(f"{E60}/{m}.jsonl")}
    rs = stored_r(m)
    A: dict = {f"{k}_{c}": [] for k in ("d", "p", "flip", "acc_o", "acc_m", "follow_o", "follow_m") for c in CTXS}
    A["rel"] = []
    for it in items:
        g, gm = it["gold"], it["gold_mirror"]
        ow = vx.option_words(it)
        body = r_cut(rs[it["uid"]])
        if it["set"] == "cvb":
            if not has_lr(body):
                lab = "NONE"
            else:
                lab = rel.get(rel_key(it["x"], it["y"], body), "unjudged")
        else:
            lab = "n/a"
        word_letter = {v.lower(): k for k, v in ow.items()}
        wl = word_letter.get(lab.lower()) if lab in ("LEFT", "RIGHT") else None
        A["rel"].append("TRUE" if wl == g else ("FALSE" if wl else lab))
        for c in CTXS:
            ro, rm = recs[(it["uid"], "orig", c)], recs[(it["uid"], "mirror", c)]
            po, pm = pe(ro), pe(rm)
            A[f"d_{c}"].append(float(po == g) - float(pm == g))
            A[f"p_{c}"].append(p2(ro["lp_sum"], g) - p2(rm["lp_sum"], g))
            A[f"flip_{c}"].append(float(po != pm))
            A[f"acc_o_{c}"].append(float(po == g))
            A[f"acc_m_{c}"].append(float(pm == gm))
            A[f"follow_o_{c}"].append(float(po == wl) if wl else np.nan)
            A[f"follow_m_{c}"].append(float(pm == wl) if wl else np.nan)
    return {k: (np.array(v) if k != "rel" else np.array(v)) for k, v in A.items()}


def sanity(items_all: list[dict]) -> dict:
    """Re-reads vs stored readouts (argmax agreement; mean |dP(gold)| where stored logprobs exist)."""
    out = {}
    cvb = {it["uid"]: it for it in items_all if it["set"] == "cvb"}
    blink = {it["uid"]: it for it in items_all if it["set"] == "blink"}
    for m in M7:
        recs = {(r["item_id"], r["img"], r["ctx"]): r for r in vx.read_jsonl(f"{E60}/{m}.jsonl")}
        s: dict = {}
        e3x = {(r["item_id"], r["image"]): r for r in vx.read_jsonl(f"{vx.OUT}/e3x/{m}.jsonl") if r["context"] == "direct"}
        for img, e3img in (("orig", "original"), ("mirror", "mirror")):
            ag, dp = [], []
            for uid, it in cvb.items():
                st = e3x.get((uid, e3img))
                if st is None or (uid, img, "direct") not in recs:
                    continue
                r = recs[(uid, img, "direct")]
                ag.append(pe(r) == pe(st))
                dp.append(abs(p2(r["lp_sum"], it["gold"]) - p2(st["lp_sum"], it["gold"])))
            s[f"direct_{img}_vs_e3x"] = {"n": len(ag), "agree": round(100 * np.mean(ag), 2), "mean_abs_dP": round(float(np.mean(dp)), 4),
                                         "max_abs_dP": round(float(np.max(dp)), 4)}
        leg = vx.legacy_e1(m)
        for ctx, key in (("full", "cot"), ("direct", "direct")):
            ag = []
            for uid in cvb:
                sc = leg.get(uid, {}).get("scores", {}).get(key)
                if not sc or (uid, "orig", ctx) not in recs:
                    continue
                fin = {k: v for k, v in sc.items() if v is not None and math.isfinite(v)}
                if not fin:
                    continue
                ag.append(recs[(uid, "orig", ctx)]["pred_top20"] == max(fin, key=fin.get))
            s[f"{ctx}_orig_vs_e1_{key}"] = {"n": len(ag), "agree": round(100 * np.mean(ag), 2) if ag else None}
        p5 = f"{_vtp.LEGACY_RESULTS}/e5/{m}.json"
        if os.path.exists(p5):
            e5 = json.load(open(p5))["items"]
            ag = []
            for uid in cvb:
                sc = e5.get(uid, {}).get("scores", {}).get("cot_noans")
                if not sc or (uid, "orig", "rcut") not in recs:
                    continue
                fin = {k: v for k, v in sc.items() if v is not None and math.isfinite(v)}
                ag.append(recs[(uid, "orig", "rcut")]["pred_top20"] == max(fin, key=fin.get))
            s["rcut_orig_vs_e5_cot_noans"] = {"n": len(ag), "agree": round(100 * np.mean(ag), 2)}
        e6 = {(r["item_id"], r["condition"]): r for f in ("direct", "rest")
              for r in vx.read_jsonl(f"{vx.OUT}/e6x/{m}_{f}.jsonl") if r["task"] == "blink_relation"}
        for ctx, cond in (("direct", "direct"), ("full", "cot")):
            ag = [pe(recs[(uid, "orig", ctx)]) == pe(e6[(uid, cond)]) for uid in blink
                  if (uid, cond) in e6 and (uid, "orig", ctx) in recs]
            s[f"blink_{ctx}_orig_vs_e6x"] = {"n": len(ag), "agree": round(100 * np.mean(ag), 2) if ag else None}
        rs = stored_r(m)
        s["rcut_prefix_only"] = sum(r_cut(rs[uid]).strip() == vx.C.COT_PREFIX for uid in cvb)
        out[m] = s
    return out


def main() -> None:
    """Run the analysis and write bundle, text summary and table."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-table", action="store_true")
    a = ap.parse_args()
    lines: list[str] = []
    P = lambda s: (lines.append(s), print(s))  # noqa: E731
    items_all = vx.read_jsonl(f"{E60}/items.jsonl")
    excl = Counter(r for it in items_all if it["set"] == "cvb" for r in it["exclude"])
    items = [it for it in items_all if it["keep"] and it["set"] == "cvb"]
    blink = [it for it in items_all if it["keep"] and it["set"] == "blink"]
    P(f"E60 items: CV-Bench L/R candidates {sum(it['set'] == 'cvb' for it in items_all)}, kept {len(items)} "
      f"(excluded {dict(excl)}); BLINK kept {len(blink)}")
    rel = rel_labels()
    res: dict = {"n_items": len(items), "exclusions": dict(excl), "judges_rel": judge_agreement(), "models": {}}
    P(f"Relation judges: {res['judges_rel']}")
    res["sanity"] = sanity(items_all)
    for m, s in res["sanity"].items():
        P(f"SANITY {m:14s} " + " | ".join(f"{k}: {v}" for k, v in s.items()))
    boot = Boot([it["group"] for it in items])
    mean_draws: dict = {}
    for m in M7:
        A = model_arrays(m, items, rel)
        r: dict = {"n": len(items)}
        for c in CTXS:
            d = A[f"d_{c}"]
            dr = 100 * boot.draws(d)
            mean_draws.setdefault(f"E_{c}", []).append(dr)
            r[f"E_{c}"] = ci(100 * d.mean(), dr)
            r[f"Ep_{c}"] = ci(100 * A[f"p_{c}"].mean(), 100 * boot.draws(A[f"p_{c}"]))
            r[f"flip_{c}"] = round(100 * A[f"flip_{c}"].mean(), 2)
            r[f"acc_orig_{c}"] = round(100 * A[f"acc_o_{c}"].mean(), 2)
            r[f"acc_mirror_{c}"] = round(100 * A[f"acc_m_{c}"].mean(), 2)
        dd = 100 * boot.draws(A["d_direct"] - A["d_rcut"])
        mean_draws.setdefault("D", []).append(dd)
        r["D_direct_minus_rcut"] = ci(100 * (A["d_direct"] - A["d_rcut"]).mean(), dd)
        ed, er = r["E_direct"]["mean"], r["E_rcut"]["mean"]
        r["ratio_rcut_over_direct"] = round(er / ed, 4) if ed >= 20 else None
        r["ratio_informative"] = ed >= 20
        rc = Counter(A["rel"].tolist())
        r["rel_counts"] = dict(rc)
        r["by_rel"] = {}
        for lab in ("TRUE", "FALSE", "NONE"):
            mask = A["rel"] == lab
            if mask.sum() == 0:
                continue
            sub = {"n": int(mask.sum())}
            for c in CTXS:
                dr = 100 * boot.draws(A[f"d_{c}"], mask)
                sub[f"E_{c}"] = ci(100 * A[f"d_{c}"][mask].mean(), dr) if mask.sum() >= 10 else \
                    {"mean": round(100 * A[f"d_{c}"][mask].mean(), 4), "lo": None, "hi": None}
            if lab in ("TRUE", "FALSE"):
                sub["follow_rcut_orig"] = round(100 * np.nanmean(A["follow_o_rcut"][mask]), 2)
                sub["follow_rcut_mirror"] = round(100 * np.nanmean(A["follow_m_rcut"][mask]), 2)
            r["by_rel"][lab] = sub
        res["models"][m] = r
        P(f"{m:14s} n={len(items)} E_direct {h(ed)} [{h(r['E_direct']['lo'])}, {h(r['E_direct']['hi'])}] | "
          f"E_rcut {h(er)} [{h(r['E_rcut']['lo'])}, {h(r['E_rcut']['hi'])}] | E_full {h(r['E_full']['mean'])} | "
          f"D {h(r['D_direct_minus_rcut']['mean'])} [{h(r['D_direct_minus_rcut']['lo'])}, {h(r['D_direct_minus_rcut']['hi'])}] | "
          f"ratio {r['ratio_rcut_over_direct']} | Direct acc o/m {r['acc_orig_direct']}/{r['acc_mirror_direct']} | "
          f"Rcut acc o/m {r['acc_orig_rcut']}/{r['acc_mirror_rcut']} | flips d/rc/f {r['flip_direct']}/{r['flip_rcut']}/{r['flip_full']}")
        P(f"{'':14s} probability version Ep_direct {h(r['Ep_direct']['mean'])}, Ep_rcut {h(r['Ep_rcut']['mean'])} "
          f"[{h(r['Ep_rcut']['lo'])}, {h(r['Ep_rcut']['hi'])}], Ep_full {h(r['Ep_full']['mean'])}; rel {dict(rc)}")
        for lab, sub in r["by_rel"].items():
            P(f"{'':14s} {lab:5s} n={sub['n']:3d} E_direct {h(sub['E_direct']['mean'])} E_rcut {h(sub['E_rcut']['mean'])}"
              + (f" [{h(sub['E_rcut']['lo'])}, {h(sub['E_rcut']['hi'])}]" if sub['E_rcut']['lo'] is not None else "")
              + f" E_full {h(sub['E_full']['mean'])}"
              + (f" | R_cut answer = written rel. orig {sub['follow_rcut_orig']} / mirror {sub['follow_rcut_mirror']}"
                 if "follow_rcut_orig" in sub else ""))
    mean: dict = {}
    for k, v in mean_draws.items():
        arr = np.vstack(v).mean(0)
        pt = {"E_direct": "E_direct", "E_rcut": "E_rcut", "E_full": "E_full", "D": "D_direct_minus_rcut"}[k]
        mean[pt] = ci(float(np.mean([res["models"][m][pt]["mean"] for m in M7])), arr)
    for lab in ("TRUE", "FALSE"):
        vals = [res["models"][m]["by_rel"].get(lab, {}).get("E_rcut", {}).get("mean") for m in M7]
        vals = [v for v in vals if v is not None]
        mean[f"E_rcut_{lab}"] = round(float(np.mean(vals)), 4) if vals else None
        vals = [res["models"][m]["by_rel"].get(lab, {}).get("E_direct", {}).get("mean") for m in M7]
        mean[f"E_direct_{lab}"] = round(float(np.mean(vals)), 4) if vals else None
    res["mean7"] = mean
    inf = [m for m in M7 if res["models"][m]["ratio_informative"]]
    small = [m for m in inf if res["models"][m]["E_rcut"]["mean"] <= res["models"][m]["E_direct"]["mean"] / 3]
    a_ok = mean["D_direct_minus_rcut"]["lo"] > 0
    verdict = "SUPPORTED" if a_ok and len(small) >= 4 else ("PARTLY" if a_ok else "NOT SUPPORTED")
    res["decision"] = {"a_meanD_ci_gt0": a_ok, "informative_models": inf, "ratio_le_third": small, "verdict": verdict}
    P(f"MEAN7 E_direct {h(mean['E_direct']['mean'])} [{h(mean['E_direct']['lo'])}, {h(mean['E_direct']['hi'])}] | "
      f"E_rcut {h(mean['E_rcut']['mean'])} [{h(mean['E_rcut']['lo'])}, {h(mean['E_rcut']['hi'])}] | "
      f"E_full {h(mean['E_full']['mean'])} [{h(mean['E_full']['lo'])}, {h(mean['E_full']['hi'])}] | "
      f"D {h(mean['D_direct_minus_rcut']['mean'])} [{h(mean['D_direct_minus_rcut']['lo'])}, {h(mean['D_direct_minus_rcut']['hi'])}] | "
      f"E_rcut TRUE {mean['E_rcut_TRUE']} FALSE {mean['E_rcut_FALSE']}")
    P(f"DECISION {verdict}: mean D CI>0 {a_ok}; ratio <= 1/3 in {len(small)}/{len(inf)} informative models {small}")
    # Exploratory (not pre-registered): looser text filters.
    res["sensitivity"] = {}
    for name, sel in (("all_383_no_text_filter", lambda it: True),
                      ("claude_text_NO", lambda it: it["text_labels"].get("claude-opus-5-5") == "NO")):
        its = [it for it in items_all if it["set"] == "cvb" and sel(it)]
        b2 = Boot([it["group"] for it in its])
        sens: dict = {"n": len(its), "models": {}}
        md: dict = {}
        for m in M7:
            A = model_arrays(m, its, rel)
            sens["models"][m] = {}
            for c in CTXS:
                dr = 100 * b2.draws(A[f"d_{c}"])
                md.setdefault(c, []).append(dr)
                sens["models"][m][f"E_{c}"] = ci(100 * A[f"d_{c}"].mean(), dr)
        sens["mean7"] = {f"E_{c}": ci(float(np.mean([sens["models"][m][f"E_{c}"]["mean"] for m in M7])), np.vstack(md[c]).mean(0))
                         for c in CTXS}
        res["sensitivity"][name] = sens
        P(f"SENS {name} n={len(its)}: " + " | ".join(
            f"{m} D {h(sens['models'][m]['E_direct']['mean'])} Rc {h(sens['models'][m]['E_rcut']['mean'])}" for m in M7)
          + " | MEAN7 " + " ".join(f"E_{c} {h(sens['mean7'][f'E_{c}']['mean'])} [{h(sens['mean7'][f'E_{c}']['lo'])}, "
                                    f"{h(sens['mean7'][f'E_{c}']['hi'])}]" for c in CTXS))
    # BLINK descriptive
    res["blink"] = {}
    if blink:
        for m in M7:
            A = model_arrays(m, blink, rel)
            res["blink"][m] = {"n": len(blink), **{f"E_{c}": round(100 * A[f'd_{c}'].mean(), 2) for c in CTXS}}
            P(f"BLINK {m:14s} n={len(blink)} " + " ".join(f"E_{c} {res['blink'][m][f'E_{c}']:+.1f}" for c in CTXS))
    BUNDLE.mkdir(parents=True, exist_ok=True)
    json.dump(res, open(BUNDLE / "e60_mirror.json", "w"), indent=1)
    (BUNDLE / "rigor_e60.txt").write_text("\n".join(lines) + "\n")
    shutil.copy(BUNDLE / "e60_mirror.json", f"{E60}/e60_mirror.json")
    shutil.copy(BUNDLE / "rigor_e60.txt", f"{E60}/rigor_e60.txt")
    if not a.no_table:
        write_table(res)


def cell(c: dict) -> str:
    """'+x.x {\\scriptsize[+lo, +hi]}' cell."""
    return f"${h(c['mean'])}$ {{\\scriptsize$[{h(c['lo'])}, {h(c['hi'])}]$}}"


def write_table(res: dict) -> None:
    """Bare tabular; suggested caption in the header comment."""
    n = res["n_items"]
    L = ["% Tab. supp_e60 (Exp-60, rigor_e60.py). Suggested caption:",
         f"% Fixed rationale, mirrored natural image. CV-Bench 2D Relation left/right items without legible text (n = {n});",
         "% a horizontal mirror swaps the correct option while the scene stays natural. Image effect",
         "% E = P(answer = g_v | v) - P(answer = g_v | mirror(v)) in points (100 = follows the image, 0 = ignores it),",
         "% with the context fixed: Direct (Answer: only), R_cut (the model's own stored rationale with its concluding",
         "% answer sentence removed, as in Tab. supp_answer_step) or the full rationale R. 95% item bootstrap CIs (2,000).",
         "% True / false: R_cut's last written left/right relation (two text-only judges, consensus) is true / false for v.",
         "\\begin{tabular}{lccccccc}", "\\toprule",
         " & Direct acc. & \\multicolumn{3}{c}{Image effect $E$ (points)} & $E_{R_\\mathrm{cut}}/E_\\mathrm{Dir}$ "
         "& \\multicolumn{2}{c}{$E_{R_\\mathrm{cut}}$ by written relation} \\\\",
         "\\cmidrule(lr){3-5} \\cmidrule(lr){7-8}",
         "Model & ($v$ / mirror) & Direct & $R_\\mathrm{cut}$ & full $R$ & & true ($n$) & false ($n$) \\\\", "\\midrule"]
    for m in M7:
        r = res["models"][m]
        br = r["by_rel"]
        f = lambda lab: (f"${h(br[lab]['E_rcut']['mean'])}$ ({br[lab]['n']})" if lab in br else "--")  # noqa: E731
        ratio = u(r["ratio_rcut_over_direct"], 2) if r["ratio_rcut_over_direct"] is not None else "n/a"
        L.append(f"{NAMES[m]} & {u(r['acc_orig_direct'])} / {u(r['acc_mirror_direct'])} & {cell(r['E_direct'])} & "
                 f"{cell(r['E_rcut'])} & ${h(r['E_full']['mean'])}$ & {ratio} & {f('TRUE')} & {f('FALSE')} \\\\")
    mn = res["mean7"]
    L += ["\\midrule",
          f"Mean of 7 & & {cell(mn['E_direct'])} & {cell(mn['E_rcut'])} & {cell(mn['E_full'])} & & "
          f"${h(mn['E_rcut_TRUE'])}$ & ${h(mn['E_rcut_FALSE'])}$ \\\\",
          "\\bottomrule", "\\end{tabular}"]
    TABLE.write_text("\n".join(L) + "\n")
    print(f"wrote {TABLE}")


if __name__ == "__main__":
    main()
