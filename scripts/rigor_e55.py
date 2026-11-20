"""Exp-55 analysis: perception-first instruction in the USER turn (CPU only).

Inputs: outputs/e55/<m>_{cues,syn,nat}[_shard*].jsonl (U1, U2, 64-item Direct re-read); comparators
outputs/e10/<m>_cues.jsonl, outputs/e10/<m>.jsonl (syn_depth) for Direct / CoT on the cue set and Synthetic Depth,
outputs/e44/<m>_nat*.jsonl (fresh Direct / CoT) for BLINK + CV-Bench, and Exp-44 P1 (assistant-prefix version) on all
sets for the U1 - P1 comparison.
Pre-registration: outputs/e55/prereg.txt (primary = U1 - Direct, unweighted mean over the 7 models, on (a) the cue set
and (b) BLINK Rel. Depth + CV-Bench 3D Depth; 2,000 paired cluster draws, seed 55; met iff both CI lower bounds > 0).

Bootstrap, AME and written-relation code are those of rigor_e44 (acc_table extended with the E55 contrasts).
-> outputs/bundle/e55_user_prompt.json, outputs/bundle/rigor_e55.txt, paper tables/supp_e55.tex,
   shared outputs/e55/summary.json, rigor_e55.txt and items_<set>.jsonl (per-item predictions).

Usage: python rigor_e55.py [--nb 2000] [--nb-ame 1000] [--no-ame]
"""
from __future__ import annotations
import vtax_paths as _vtp  # noqa: E402

import argparse
import json
import os
import re
import shutil
from decimal import ROUND_HALF_UP, Decimal
from collections import defaultdict
from multiprocessing import Pool
from pathlib import Path

import numpy as np

import rigor_e44 as R
from build_bundle import pe, rj

O = Path(os.environ.get("E55_O", f"{_vtp.OUTPUTS}"))
B = Path(os.environ.get("E55_B", "outputs/bundle"))
PAPER = Path(os.environ.get("E55_PAPER", f"{_vtp.PAPER}"))
M7, NAMES = R.M7, R.NAMES
ARMS = ["direct", "cot", "u1", "u2", "p1"]
PROMPTS = ["u1", "u2", "p1"]
CONTRASTS = [("u1", "direct"), ("u1", "cot"), ("u2", "direct"), ("u2", "cot"), ("u2", "u1"), ("cot", "direct"),
             ("p1", "direct"), ("u1", "p1"), ("u2", "p1")]
SETS = ["cues", "cues_front", "cues_behind", "syn", "blink", "cvb", "nat"]
SET_NAMES = R.SET_NAMES | {"cues_front": "Cue set, 'in front of' (800)", "cues_behind": "Cue set, 'behind' (800)"}
COT_PREFIX = R.COT_PREFIX
# rigor_e44's AME / written-relation helpers read these module globals
R.ARMS, R.PROMPTS = ARMS, PROMPTS


# ============================================================================= loading

def files(d: str, m: str, s: str) -> list[Path]:
    return [p for p in [O / d / f"{m}_{s}.jsonl"] + sorted((O / d).glob(f"{m}_{s}_shard*of*.jsonl")) if p.exists()]


def load_model(m: str) -> dict:
    """set -> arm -> item_id -> record (only items present in every arm); Direct re-reads; Exp-44 P1 records."""
    D: dict[str, dict[str, dict[str, dict]]] = {s: defaultdict(dict) for s in ("cues", "syn", "blink", "cvb")}
    for r in rj(O / "e10" / f"{m}_cues.jsonl"):
        if r["condition"] in ("direct", "cot"):
            D["cues"][r["condition"]][r["item_id"]] = r | {"set": "cues", "arm": r["condition"]}
    for r in rj(O / "e10" / f"{m}.jsonl"):
        if r["task"] == "syn_depth" and r["condition"] in ("direct", "cot"):
            D["syn"][r["condition"]][r["item_id"]] = r | {"set": "syn", "arm": r["condition"], "phrasing": "front"}
    for s in ("cues", "syn", "nat"):
        for r in (x for f in files("e44", m, s) for x in rj(f)):
            if r["arm"] == "p1" or (s == "nat" and r["arm"] in ("direct", "cot")):
                D[r["set"]][r["arm"]][r["item_id"]] = r
    san: dict[str, list[dict]] = {}
    for s in ("cues", "syn", "nat"):
        for r in (x for f in files("e55", m, s) for x in rj(f)):
            if r["arm"] == "direct":
                san.setdefault(s, []).append(r)
            else:
                D[r["set"]][r["arm"]][r["item_id"]] = r
    out = {}
    for s, A in D.items():
        if not all(a in A for a in ARMS):
            continue
        keys = sorted(set.intersection(*[set(A[a]) for a in ARMS]))
        out[s] = {a: {k: A[a][k] for k in keys} for a in ARMS}
    return {"data": out, "sanity_direct": san, "raw": D}


def correct(r: dict) -> float:
    return float(pe(r) == r["gold"])


# ============================================================================= bootstrap

def acc_table(datas: dict[str, dict], s: str, nb: int, rng) -> dict:
    """rigor_e44.acc_table with the E55 arms / contrasts (shared cluster draws across models and arms)."""
    models = [m for m in M7 if s in datas[m]]
    if not models:
        return {}
    keys = sorted(set.intersection(*[set(datas[m][s]["direct"]) for m in models]))
    uc, inv = R.cluster_arrays(datas[models[0]][s]["direct"], keys)
    C = len(uc)
    S = np.zeros((len(models), len(ARMS), C))
    N = np.zeros(C)
    np.add.at(N, inv, 1.0)
    for i, m in enumerate(models):
        for j, a in enumerate(ARMS):
            np.add.at(S[i, j], inv, np.array([correct(datas[m][s][a][k]) for k in keys]))
    est = 100 * S.sum(2) / N.sum()
    W = rng.multinomial(C, np.full(C, 1.0 / C), size=nb)
    bs = 100 * np.einsum("bc,mac->bma", W, S) / (W @ N)[:, None, None]
    q = lambda v: [round(float(np.quantile(v, .025)), 1), round(float(np.quantile(v, .975)), 1)]
    ia = {a: j for j, a in enumerate(ARMS)}
    out = {"n_items": len(keys), "n_clusters": int(C), "models": models, "per_model": {}, "pooled": {}}
    for i, m in enumerate(models):
        r = {f"acc:{a}": round(float(est[i, ia[a]]), 4) for a in ARMS}
        for a, b in CONTRASTS:
            dv = bs[:, i, ia[a]] - bs[:, i, ia[b]]
            r[f"{a}-{b}"] = {"est": round(float(est[i, ia[a]] - est[i, ia[b]]), 4), "ci": q(dv),
                             "p_le0": round(float((dv <= 0).mean()), 4)}
        out["per_model"][m] = r
    pe_, pb = est.mean(0), bs.mean(1)
    out["pooled"] = {f"acc:{a}": round(float(pe_[ia[a]]), 4) for a in ARMS}
    for a, b in CONTRASTS:
        dv = pb[:, ia[a]] - pb[:, ia[b]]
        cis = [out["per_model"][m][f"{a}-{b}"]["ci"] for m in models]
        out["pooled"][f"{a}-{b}"] = {"est": round(float(pe_[ia[a]] - pe_[ia[b]]), 4), "ci": q(dv),
                                     "p_le0": round(float((dv <= 0).mean()), 4),
                                     "n_ci_gt0": sum(c[0] > 0 for c in cis), "n_ci_lt0": sum(c[1] < 0 for c in cis)}
    return out


# ============================================================================= format / sanity

NAT_RX = re.compile(r"\b(closer|nearer|farther|further away)\b", re.I)


def gen_stats(recs: dict[str, dict], prefix_default: str) -> dict:
    gl = [r.get("gen_len") for r in recs.values() if r.get("gen_len") is not None]
    out = {"n": len(recs)}
    if gl:
        out["mean_tok"] = round(float(np.mean(gl)), 1)
        out["trunc_pct"] = round(100 * float(np.mean([bool(r.get("trunc")) or (r.get("gen_len") or 0) >= 512
                                                       for r in recs.values()])), 2)
    bodies = []
    for r in recs.values():
        t, p = r.get("text") or "", r.get("prefix") or prefix_default
        bodies.append(t[len(p):] if t.startswith(p) else t)
    out["empty_pct"] = round(100 * float(np.mean([not b.strip() for b in bodies])), 2)
    out["n_empty"] = int(sum(not b.strip() for b in bodies))
    out["states_closer_pct"] = round(100 * float(np.mean([bool(NAT_RX.search(b)) for b in bodies])), 1)
    return out


def sanity(m: str, L: dict) -> dict:
    out: dict = {}
    for s, rs in L["sanity_direct"].items():
        sets = ("blink", "cvb") if s == "nat" else (s,)
        ref = {k: v for x in sets for k, v in L["raw"][x]["direct"].items()}
        ok = [pe(r) == pe(ref[r["item_id"]]) for r in rs if r["item_id"] in ref]
        out[f"direct_reread_{s}"] = {"n": len(ok), "n_agree": int(sum(ok)),
                                     "pass": len(ok) > 0 and sum(ok) >= len(ok) - 1}
    for s, A in L["data"].items():
        for a in ("cot", "u1", "u2", "p1"):
            out.setdefault("gen", {}).setdefault(s, {})[a] = gen_stats(A[a], COT_PREFIX)
    return out


def phrasing_table(datas: dict[str, dict]) -> dict:
    """Cue set: accuracy and yes-rate (pred A) by phrasing per model x arm."""
    out = {}
    for m in M7:
        if "cues" not in datas[m]:
            continue
        d = datas[m]["cues"]
        for ph in ("front", "behind"):
            ks = [k for k, r in d["direct"].items() if r.get("phrasing") == ph]
            out.setdefault(m, {})[ph] = {a: {"acc": round(100 * float(np.mean([correct(d[a][k]) for k in ks])), 1),
                                             "yes": round(100 * float(np.mean([pe(d[a][k]) == "A" for k in ks])), 1)}
                                         for a in ARMS}
    return out


# ============================================================================= report

def hu(x: float, d: int = 1, sign: bool = False) -> str:
    """Half-up rounding of a stored (4-decimal) value, as in the other supp tables."""
    v = Decimal(str(x)).quantize(Decimal(1).scaleb(-d), rounding=ROUND_HALF_UP)
    if v == 0:
        v = abs(v)
    return f"{v:+}" if sign else f"{v}"


def f_ci(x: dict, d: int = 1) -> str:
    return f"{hu(x['est'], d, True)} [{x['ci'][0]:+.1f}, {x['ci'][1]:+.1f}]"


def report(res: dict) -> list[str]:
    L = ["Exp-55 perception-first instruction in the USER turn (pre-registered: outputs/e55/prereg.txt)", ""]
    pr = res["primary"]
    g = lambda d, k: f_ci(d[k], 2) if k in d else "n/a"
    L.append(f"PRIMARY (U1 - Direct, mean over 7 models): cue set {g(pr, 'cues')}; BLINK+CV-Bench depth {g(pr, 'nat')}"
             f" -> met: {pr['met']} (complete: {pr['complete_7_models']})")
    L.append(f"  weaker 'removes the CoT loss' criterion: {pr['removes_cot_loss']}")
    L.append("")
    for s in SETS:
        t = res["acc"].get(s)
        if not t:
            continue
        L.append(f"== {SET_NAMES[s]}  (n = {t['n_items']}, clusters = {t['n_clusters']})")
        for m in t["models"] + ["pooled"]:
            r = t["per_model"][m] if m != "pooled" else t["pooled"]
            L.append(f"  {NAMES.get(m, 'MEAN (7)'):15s} Dir {r['acc:direct']:5.1f} CoT {r['acc:cot']:5.1f} U1 {r['acc:u1']:5.1f} "
                     f"U2 {r['acc:u2']:5.1f} P1(E44) {r['acc:p1']:5.1f} | " +
                     " ".join(f"{a.upper()}-{b.upper()[:3]} {f_ci(r[f'{a}-{b}'])}" for a, b in CONTRASTS))
        p = t["pooled"]
        L.append("  per-model CI>0 / CI<0 counts: " + "; ".join(f"{a}-{b} {p[f'{a}-{b}']['n_ci_gt0']}/{p[f'{a}-{b}']['n_ci_lt0']}"
                                                              for a, b in CONTRASTS))
        L.append("")
    L.append("== Cue set by phrasing (accuracy / yes-rate)")
    for m, ph in res["phrasing"].items():
        L.append(f"  {NAMES[m]:15s} " + " | ".join(f"{p}: " + " ".join(f"{a} {ph[p][a]['acc']:.1f}/{ph[p][a]['yes']:.0f}" for a in ARMS)
                                                   for p in ("front", "behind")))
    L.append("")
    if res.get("ame"):
        L.append("== Cue-set AMEs (points; probit GLM, twin-pair bootstrap)")
        for m in M7:
            if m not in res["ame"]:
                continue
            for c in ("occlusion", "vertical"):
                a = res["ame"][m][c]
                L.append(f"  {NAMES[m]:15s} {c:9s} Dir {a['direct']['est']:+.1f} CoT {a['cot']['est']:+.1f} U1 {a['u1']['est']:+.1f} "
                         f"U2 {a['u2']['est']:+.1f} P1 {a['p1']['est']:+.1f} | U1-D {f_ci(a['u1-direct'])} U1-CoT {f_ci(a['u1-cot'])} "
                         f"U2-D {f_ci(a['u2-direct'])} CoT-D {f_ci(a['cot-direct'])}")
        L.append("")
    L.append("== Written relation (claim_parse v3; last claim; x = written acc - Direct on parsed items)")
    for m in M7:
        for s in ("cues", "syn"):
            w = res["written"].get(m, {}).get(s)
            if not w:
                continue
            fx = lambda v, f: (f.format(v) if v is not None else "nan")
            L.append(f"  {NAMES[m]:15s} {s:4s} " + " | ".join(
                f"{a}: cov {w[a]['last_coverage']:.0f} wr {fx(w[a]['last_written_acc'], '{:.1f}')} x {fx(w[a]['last_x'], '{:+.1f}')}"
                f" agree T/F {w[a]['agree_if_true']}/{w[a]['agree_if_false']}" for a in ["cot"] + PROMPTS))
    L.append("")
    L.append("== Sanity / format (gen: per set x arm, empty %, mean tokens, truncation %)")
    for m in M7:
        sa = res["sanity"].get(m, {})
        L.append(f"  {NAMES[m]:15s} reread " + " ".join(f"{k[14:]} {v['n_agree']}/{v['n']}" for k, v in sa.items() if k.startswith("direct_reread")))
        for s, A in sa.get("gen", {}).items():
            L.append(f"  {'':15s} {s:5s} " + " | ".join(f"{a}: empty {v['empty_pct']:.1f}% tok {v.get('mean_tok', float('nan')):.0f} "
                                                      f"trunc {v.get('trunc_pct', float('nan')):.1f}%" for a, v in A.items()))
    return L


def latex(res: dict) -> str:
    def c(x: dict) -> str:
        star = x["ci"][0] > 0 or x["ci"][1] < 0
        v = hu(x["est"], 1, True)
        return (f"$\\mathbf{{{v}}}$" if star else f"${v}$") + f"{{\\scriptsize$[{x['ci'][0]:+.1f},{x['ci'][1]:+.1f}]$}}"
    Lx = [r"% Exp-55 (rigor_e55.py): perception-first instruction in the user turn. Not \input; bare tabular like the other",
          r"% supp tables. Suggested caption: \textbf{Perception-first instruction in the user turn (Exp-55, pre-registered).}",
          r"% U1 appends to the user turn, after the question, ``Before answering, look at where the two shapes overlap and state",
          r"% which shape's outline is complete there; that shape is in front. Then give your answer.'' (synthetic sets) or",
          r"% ``Before answering, look for depth cues for the two marked points/objects (which one occludes the other, which rests",
          r"% lower on the ground plane or is nearer the camera) and state which is closer. Then give your answer.'' (BLINK /",
          r"% CV-Bench); the assistant turn starts with ``Let's think step by step.'' (greedy, <=512 tokens, same-turn readout).",
          r"% U2 adds a converse reminder (``When you answer, check whether the question asks about the same object or the other",
          r"% one, and whether it asks `in front of' or `behind'.''). P1 = Exp-44 assistant-prefix version. Accuracy (%) of",
          r"% Direct, CoT and U1, differences in points with 95% paired cluster-bootstrap CIs (twin pairs / images; bold: CI",
          r"% excludes 0); Empty = % of U1 continuations that are empty. Mean: unweighted over the seven models, same resampled",
          r"% clusters for all models. Pre-registered primary: U1-Direct on the cue set and on BLINK + CV-Bench depth, both CI",
          r"% lower bounds > 0. Wrap in \resizebox{\textwidth}{!}{...} (11 columns).",
          r"\begin{tabular}{@{}llccccccccc@{}}", r"\toprule",
          r"Set & Model & \texttt{Dir.} & \texttt{CoT} & U1 & U1$-$\texttt{Dir.} & U1$-$\texttt{CoT} & U2$-$\texttt{Dir.} & "
          r"U2$-$U1 & U1$-$P1 & Empty \\", r"\midrule"]
    short = {"cues": "Cue set", "syn": "Syn. Depth", "nat": "BLINK+CVB"}
    for s in ("cues", "syn", "nat"):
        t = res["acc"].get(s)
        if not t:
            continue
        rows = t["models"] + ["pooled"]
        for k, m in enumerate(rows):
            r = t["per_model"][m] if m != "pooled" else t["pooled"]
            lab = f"\\multirow{{{len(rows)}}}{{*}}{{{short[s]}}}" if k == 0 else ""
            nm = NAMES.get(m, r"\textit{Mean}")
            if m == "pooled":
                em = np.mean([res["sanity"][x]["gen"][s]["u1"]["empty_pct"] for x in t["models"]])
            else:
                em = res["sanity"][m]["gen"][s]["u1"]["empty_pct"]
            Lx.append(f"{lab} & {nm} & {hu(r['acc:direct'])} & {hu(r['acc:cot'])} & {hu(r['acc:u1'])} & " +
                      " & ".join(c(r[x]) for x in ("u1-direct", "u1-cot", "u2-direct", "u2-u1", "u1-p1")) + f" & {em:.1f} \\\\")
        Lx.append(r"\midrule" if s != "nat" else r"\bottomrule")
    Lx += [r"\end{tabular}"]
    return "\n".join(Lx) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--nb", type=int, default=2000)
    ap.add_argument("--nb-ame", type=int, default=1000)
    ap.add_argument("--no-ame", action="store_true")
    a = ap.parse_args()
    loaded = {m: load_model(m) for m in M7}
    datas = {m: loaded[m]["data"] for m in M7}
    for m in M7:
        if "blink" in datas[m] and "cvb" in datas[m]:
            datas[m]["nat"] = {x: datas[m]["blink"][x] | datas[m]["cvb"][x] for x in ARMS}
        if "cues" in datas[m]:
            for ph in ("front", "behind"):
                ks = [k for k, r in datas[m]["cues"]["direct"].items() if r.get("phrasing") == ph]
                datas[m][f"cues_{ph}"] = {x: {k: datas[m]["cues"][x][k] for k in ks} for x in ARMS}
    rng = np.random.default_rng(55)
    res: dict = {"prereg": str(O / "e55" / "prereg.txt"), "nb": a.nb, "acc": {}}
    for s in SETS:
        res["acc"][s] = acc_table(datas, s, a.nb, rng)
    pr = {s: res["acc"][s]["pooled"]["u1-direct"] for s in ("cues", "nat") if res["acc"].get(s)}
    complete = all(len(res["acc"].get(s, {}).get("models", [])) == 7 for s in ("cues", "nat"))
    met = complete and all(pr[s]["ci"][0] > 0 for s in ("cues", "nat"))
    rem = complete and all(pr[s]["ci"][1] >= 0 for s in ("cues", "nat")) and \
        any(res["acc"][s]["pooled"]["u1-cot"]["ci"][0] > 0 for s in ("cues", "nat"))
    res["primary"] = pr | {"met": met, "removes_cot_loss": rem, "complete_7_models": complete}
    res["phrasing"] = phrasing_table(datas)
    if not a.no_ame:
        with Pool(7) as pool:
            out = pool.map(R.ame_model, [(m, datas[m]["cues"], a.nb_ame) for m in M7 if "cues" in datas[m]])
        res["ame"] = dict(out)
    res["written"] = R.written(datas, np.random.default_rng(32))
    res["sanity"] = {m: sanity(m, loaded[m]) for m in M7}
    B.mkdir(parents=True, exist_ok=True)
    json.dump(res, open(B / "e55_user_prompt.json", "w"), indent=1)
    txt = "\n".join(report(res)) + "\n"
    (B / "rigor_e55.txt").write_text(txt)
    (PAPER / "tables" / "supp_e55.tex").write_text(latex(res))
    shutil.copy(B / "e55_user_prompt.json", O / "e55" / "summary.json")
    shutil.copy(B / "rigor_e55.txt", O / "e55" / "rigor_e55.txt")
    for s in ("cues", "syn", "blink", "cvb"):
        with open(O / "e55" / f"items_{s}.jsonl", "w") as f:
            for m in M7:
                if s not in datas[m]:
                    continue
                for k, r in datas[m][s]["direct"].items():
                    f.write(json.dumps({"model": m, "set": s, "item_id": k, "image_id": r["image_id"], "gold": r["gold"],
                                        "phrasing": r.get("phrasing"), **{x: pe(datas[m][s][x][k]) for x in ARMS}}) + "\n")
    print(txt)


if __name__ == "__main__":
    main()
