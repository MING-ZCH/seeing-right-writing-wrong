"""Exp-44 analysis: label-free prompt variants of the CoT prefix (CPU only).

Inputs: outputs/e44/<m>_{cues,syn,nat}.jsonl (P1-P3; nat also fresh Direct / CoT), stored comparators
outputs/e10/<m>_cues.jsonl and outputs/e10/<m>.jsonl (syn_depth) for Direct / CoT on the cue set and Synthetic Depth.
Pre-registration: outputs/e44/prereg.txt (primary = P1 - Direct pooled over the 7 models on (a) the cue set and
(b) BLINK Relative Depth + CV-Bench 3D Depth; met iff both CI lower bounds > 0).

Per model x set x arm: accuracy; Prompt - Direct, Prompt - CoT (paired cluster bootstrap, 2,000, seed 44; cue set
cluster = twin pair, BLINK = item, CV-Bench = image group, Syn. Depth = image). Pooled = unweighted mean over models,
same resampled clusters for every model. Cue set: occlusion / height AMEs per arm (r18_cue_ame definition, 1,000
twin-pair draws). Written relation (claim_parse v3, rigor_e32.rec / cell_stats definitions) on cue set + Syn. Depth.
-> outputs/bundle/e44_prompt_fix.json, outputs/bundle/rigor_e44.txt, paper tables/supp_e44.tex,
   shared outputs/e44/summary.json (copy of the bundle) and outputs/e44/items_<set>.jsonl (per-item correctness).

Usage: python rigor_e44.py [--nb 2000] [--nb-ame 1000]
"""
from __future__ import annotations
import vtax_paths as _vtp  # noqa: E402

import argparse
import json
import os
import shutil
from collections import defaultdict
from multiprocessing import Pool
from pathlib import Path

import numpy as np

from build_bundle import pe, rj
from e2f_claim_edit import split_sents

O = Path(os.environ.get("E44_O", f"{_vtp.OUTPUTS}"))
B = Path(os.environ.get("E44_B", "outputs/bundle"))
PAPER = Path(os.environ.get("E44_PAPER", f"{_vtp.PAPER}"))
M7 = ["qwen3b", "qwen7b", "internvl3_8b", "phi35v", "molmo7b", "qwen3vl_8b", "internvl35_8b"]
NAMES = {"qwen3b": "Qwen2.5-VL-3B", "qwen7b": "Qwen2.5-VL-7B", "internvl3_8b": "InternVL3-8B", "phi35v": "Phi-3.5-V",
         "molmo7b": "Molmo-7B-D", "qwen3vl_8b": "Qwen3-VL-8B", "internvl35_8b": "InternVL3.5-8B"}
ARMS = ["direct", "cot", "p1", "p2", "p3", "p1f", "p2f", "p3f"]
PROMPTS = ["p1", "p2", "p3", "p1f", "p2f", "p3f"]
SETS = ["cues", "syn", "blink", "cvb", "nat"]
SET_NAMES = {"cues": "Cue-factorial set (1,600)", "syn": "Synthetic Depth (500)", "blink": "BLINK Rel. Depth (124)",
             "cvb": "CV-Bench 3D Depth (600)", "nat": "BLINK + CV-Bench depth (724)"}
COT_PREFIX = "Let's think step by step."


# ============================================================================= loading

def load_model(m: str) -> dict[str, dict[str, dict[str, dict]]]:
    """set -> arm -> item_id -> record (only items present in every arm)."""
    D: dict[str, dict[str, dict[str, dict]]] = {s: defaultdict(dict) for s in ("cues", "syn", "blink", "cvb")}
    for r in rj(O / "e10" / f"{m}_cues.jsonl"):
        if r["condition"] in ("direct", "cot"):
            D["cues"][r["condition"]][r["item_id"]] = r | {"set": "cues", "arm": r["condition"]}
    for r in rj(O / "e10" / f"{m}.jsonl"):
        if r["task"] == "syn_depth" and r["condition"] in ("direct", "cot"):
            D["syn"][r["condition"]][r["item_id"]] = r | {"set": "syn", "arm": r["condition"], "phrasing": "front"}
    san = {}
    for s in ("cues", "syn", "nat"):
        files = [O / "e44" / f"{m}_{s}.jsonl"] + sorted((O / "e44").glob(f"{m}_{s}_shard*of*.jsonl"))
        for r in (x for f in files for x in rj(f)):
            if s != "nat" and r["arm"] == "direct":
                san.setdefault(s, []).append(r)
                continue
            D[r["set"]][r["arm"]][r["item_id"]] = r
    out = {}
    for s, A in D.items():
        if not all(a in A for a in ARMS):
            continue
        keys = sorted(set.intersection(*[set(A[a]) for a in ARMS]))
        out[s] = {a: {k: A[a][k] for k in keys} for a in ARMS}
    return {"data": out, "sanity_direct": san}


def correct(r: dict) -> float:
    return float(pe(r) == r["gold"])


# ============================================================================= bootstrap

def cluster_arrays(recs: dict[str, dict], keys: list[str]) -> tuple[np.ndarray, np.ndarray]:
    cl = np.array([str(recs[k]["image_id"]) for k in keys])
    uc, inv = np.unique(cl, return_inverse=True)
    return uc, inv


def acc_table(datas: dict[str, dict], s: str, nb: int, rng) -> dict:
    """Per-model and pooled accuracies / contrasts for one set with shared cluster draws across models."""
    models = [m for m in M7 if s in datas[m]]
    if not models:
        return {}
    keys = sorted(set.intersection(*[set(datas[m][s]["direct"]) for m in models]))
    ref = datas[models[0]][s]["direct"]
    uc, inv = cluster_arrays(ref, keys)
    C = len(uc)
    # per-cluster sums: [model, arm, cluster]
    S = np.zeros((len(models), len(ARMS), C))
    N = np.zeros(C)
    np.add.at(N, inv, 1.0)
    for i, m in enumerate(models):
        for j, a in enumerate(ARMS):
            v = np.array([correct(datas[m][s][a][k]) for k in keys])
            np.add.at(S[i, j], inv, v)
    est = 100 * S.sum(2) / N.sum()                      # [model, arm]
    W = rng.multinomial(C, np.full(C, 1.0 / C), size=nb)  # bootstrap cluster counts
    bs = 100 * np.einsum("bc,mac->bma", W, S) / (W @ N)[:, None, None]  # [b, model, arm]
    q = lambda v: [round(float(np.quantile(v, .025)), 1), round(float(np.quantile(v, .975)), 1)]
    out = {"n_items": len(keys), "n_clusters": int(C), "models": models, "per_model": {}, "pooled": {}}
    ia = {a: j for j, a in enumerate(ARMS)}
    contrasts = [(p, "direct") for p in PROMPTS] + [(p, "cot") for p in PROMPTS] + [("cot", "direct")]
    for i, m in enumerate(models):
        r = {f"acc:{a}": round(float(est[i, ia[a]]), 1) for a in ARMS}
        for a, b in contrasts:
            dv = bs[:, i, ia[a]] - bs[:, i, ia[b]]
            r[f"{a}-{b}"] = {"est": round(float(est[i, ia[a]] - est[i, ia[b]]), 1), "ci": q(dv),
                             "p_le0": round(float((dv <= 0).mean()), 4)}
        out["per_model"][m] = r
    pe_ = est.mean(0)
    pb = bs.mean(1)
    out["pooled"] = {f"acc:{a}": round(float(pe_[ia[a]]), 1) for a in ARMS}
    for a, b in contrasts:
        dv = pb[:, ia[a]] - pb[:, ia[b]]
        out["pooled"][f"{a}-{b}"] = {"est": round(float(pe_[ia[a]] - pe_[ia[b]]), 2), "ci": q(dv),
                                     "p_le0": round(float((dv <= 0).mean()), 4)}
    # per-model sign counts
    for a, b in contrasts:
        cis = [out["per_model"][m][f"{a}-{b}"]["ci"] for m in models]
        out["pooled"][f"{a}-{b}"]["n_ci_gt0"] = sum(c[0] > 0 for c in cis)
        out["pooled"][f"{a}-{b}"]["n_ci_lt0"] = sum(c[1] < 0 for c in cis)
    return out


# ============================================================================= cue-set AMEs

def ame_model(args: tuple[str, dict, int]) -> tuple[str, dict]:
    from r18_cue_ame import stats
    from rigor_e17f import glm_rows
    m, d, nb = args
    keys = sorted(d["direct"])
    data = {a: glm_rows([d[a][k] for k in keys], use_size=True) for a in ARMS}
    pair = np.array(data["direct"]["pair"])
    pairs = sorted(set(pair))
    pidx = {p: np.where(pair == p)[0] for p in pairs}
    est = {a: stats(data[a], np.arange(len(keys))) for a in ARMS}
    rng = np.random.default_rng(18)
    bs = {a: {c: [] for c in ("occlusion", "vertical", "size")} for a in ARMS}
    for _ in range(nb):
        s = np.concatenate([pidx[pairs[j]] for j in rng.integers(0, len(pairs), len(pairs))])
        for a in ARMS:
            st = stats(data[a], s)
            for c in bs[a]:
                bs[a][c].append(st[c]["ame"])
    q = lambda v: [round(float(np.quantile(v, .025)), 1), round(float(np.quantile(v, .975)), 1)]
    r = {}
    for c in ("occlusion", "vertical", "size"):
        r[c] = {a: {"est": round(est[a][c]["ame"], 1), "ci": q(bs[a][c])} for a in ARMS}
        for a in PROMPTS + ["cot"]:
            for b in ("direct", "cot"):
                if a == b:
                    continue
                dv = np.array(bs[a][c]) - np.array(bs[b][c])
                r[c][f"{a}-{b}"] = {"est": round(est[a][c]["ame"] - est[b][c]["ame"], 1), "ci": q(dv)}
    print(f"[ame] {m} done", flush=True)
    return m, r


# ============================================================================= written relation

def written(datas: dict[str, dict], nb_rng) -> dict:
    """claim_parse v3 written relation per model x set (cues, syn) x arm (cot, p1-p3); rigor_e32 definitions."""
    import re
    from rigor_e32 import cell_stats, rec
    QRX = re.compile(r"Is the (\w+ \w+) (in front of|behind) the (\w+ \w+)\?")
    out = {}
    for m in M7:
        for s in ("cues", "syn"):
            if s not in datas[m]:
                continue
            d = datas[m][s]
            for a in ["cot"] + PROMPTS:
                rs = []
                for k, r in d[a].items():
                    q = QRX.search(r["question"])
                    x, y = q.group(1).lower(), q.group(3).lower()
                    if s == "cues":
                        mt = r["meta"]
                        front, back = mt["front_name"].lower(), mt["back_name"].lower()
                    else:
                        front, back = (x, y) if r["gold"] == "A" else (y, x)
                    text = r.get("text") or ""
                    pre = r.get("prefix") or COT_PREFIX
                    body = text[len(pre):] if text.startswith(pre) else text
                    rs.append(rec(m, s, k, r["image_id"], r["gold"], pe(d["direct"][k]), pe(r), split_sents(body),
                                  front, back, x, r.get("phrasing") or "front"))
                cs = cell_stats(rs, nb_rng)
                keep = ("n", "acc_direct", "acc_cot", "first_coverage", "first_written_acc", "last_coverage",
                        "last_written_acc", "last_agree", "first_agree", "last_x", "last_x_ci", "y", "y_ci",
                        "DpCm", "DmCp")
                # agreement split by truth of the last written relation
                tl = [r_ for r_ in rs if r_["last_true"] is not None]
                at = [r_["last_agree"] for r_ in tl if r_["last_true"]]
                af = [r_["last_agree"] for r_ in tl if not r_["last_true"]]
                row = {k: cs.get(k) for k in keep}
                row |= {"agree_if_true": round(100 * float(np.mean(at)), 1) if at else None, "n_true": len(at),
                        "agree_if_false": round(100 * float(np.mean(af)), 1) if af else None, "n_false": len(af)}
                out.setdefault(m, {}).setdefault(s, {})[a] = row
    return out


# ============================================================================= sanity / costs

def sanity(m: str, L: dict) -> dict:
    out = {}
    for s, rs in L["sanity_direct"].items():
        ref = L["data"].get(s, {}).get("direct", {})
        ok = [pe(r) == pe(ref[r["item_id"]]) for r in rs if r["item_id"] in ref]
        out[f"direct_reread_{s}"] = {"n": len(ok), "agree": round(100 * float(np.mean(ok)), 2) if ok else None}
    # fresh BLINK Direct / CoT vs stored e6x
    if "blink" in L["data"]:
        st = {(r["item_id"], r["condition"]): r for r in rj(O / "e6x" / f"{m}_rest.jsonl") if r["task"] == "blink_depth"}
        st |= {(r["item_id"], "direct"): r for r in rj(O / "e6x" / f"{m}_direct.jsonl") if r["task"] == "blink_depth"}
        for a in ("direct", "cot"):
            new = L["data"]["blink"][a]
            pairs = [(pe(new[k]), pe(st[(k, a)]), new[k]["gold"]) for k in new if (k, a) in st]
            if pairs:
                out[f"blink_{a}_vs_e6x"] = {"n": len(pairs), "agree": round(100 * np.mean([p == q for p, q, _ in pairs]), 1),
                                            "acc_new": round(100 * np.mean([p == g for p, _, g in pairs]), 1),
                                            "acc_e6x": round(100 * np.mean([q == g for _, q, g in pairs]), 1)}
    # generation length / truncation per arm
    for s, A in L["data"].items():
        for a in ["cot"] + PROMPTS:
            gl = [r.get("gen_len") for r in A[a].values() if r.get("gen_len") is not None]
            tr = [bool(r.get("trunc")) or (r.get("gen_len") or 0) >= 512 for r in A[a].values()]
            if gl:
                emp = [not (r.get("text") or "")[len(r.get("prefix") or ""):].strip() for r in A[a].values()]
                out.setdefault("gen", {}).setdefault(s, {})[a] = {"mean_tok": round(float(np.mean(gl)), 1),
                                                                  "trunc_pct": round(100 * float(np.mean(tr)), 2),
                                                                  "empty_pct": round(100 * float(np.mean(emp)), 1)}
    return out


# ============================================================================= report

def f_ci(x: dict, d: int = 1) -> str:
    return f"{x['est']:+.{d}f} [{x['ci'][0]:+.1f}, {x['ci'][1]:+.1f}]"


def report(res: dict) -> list[str]:
    L = ["Exp-44 label-free prompt variants of the CoT prefix (pre-registered: outputs/e44/prereg.txt)", ""]
    pr = res["primary"]
    g = lambda d, k: f_ci(d[k], 2) if k in d else "n/a"
    L.append(f"PRIMARY (P1 - Direct, mean over 7 models): cue set {g(pr, 'cues')}; BLINK+CV-Bench depth "
             f"{g(pr, 'nat')} -> met: {pr['met']} (complete: {pr['complete_7_models']})")
    L.append(f"  weaker 'removes CoT loss' criterion: {pr['removes_cot_loss']}")
    sp = res.get("secondary_p1f", {})
    if sp:
        L.append(f"SECONDARY forced-continuation P1f - Direct: cue set {g(sp, 'cues')}; BLINK+CV-Bench {g(sp, 'nat')}")
    L.append("")
    for s in SETS:
        t = res["acc"].get(s)
        if not t:
            continue
        L.append(f"== {SET_NAMES[s]}  (n = {t['n_items']}, clusters = {t['n_clusters']})")
        for m in t["models"] + ["pooled"]:
            r = t["per_model"][m] if m != "pooled" else t["pooled"]
            nm = NAMES.get(m, "MEAN (7)")
            L.append(f"  {nm:15s} Dir {r['acc:direct']:5.1f} CoT {r['acc:cot']:5.1f} P1 {r['acc:p1']:5.1f} P2 {r['acc:p2']:5.1f} "
                     f"P3 {r['acc:p3']:5.1f} | P1-D {f_ci(r['p1-direct'])} P2-D {f_ci(r['p2-direct'])} P3-D {f_ci(r['p3-direct'])}"
                     f" | P1-CoT {f_ci(r['p1-cot'])} P2-CoT {f_ci(r['p2-cot'])} P3-CoT {f_ci(r['p3-cot'])} | CoT-D {f_ci(r['cot-direct'])}")
            L.append(f"  {'':15s} forced: P1f {r['acc:p1f']:5.1f} P2f {r['acc:p2f']:5.1f} P3f {r['acc:p3f']:5.1f} | "
                     f"P1f-D {f_ci(r['p1f-direct'])} P2f-D {f_ci(r['p2f-direct'])} P3f-D {f_ci(r['p3f-direct'])} | "
                     f"P1f-CoT {f_ci(r['p1f-cot'])} P2f-CoT {f_ci(r['p2f-cot'])} P3f-CoT {f_ci(r['p3f-cot'])}")
        p = t["pooled"]
        L.append("  per-model CI>0 / CI<0 counts: " + "; ".join(
            f"{c} {p[c]['n_ci_gt0']}/{p[c]['n_ci_lt0']}" for c in [f"{x}-{y}" for y in ("direct", "cot") for x in PROMPTS]))
        L.append("")
    if res.get("ame"):
        L.append("== Cue-set AMEs (points; probit GLM, twin-pair bootstrap)")
        for m in M7:
            if m not in res["ame"]:
                continue
            for c in ("occlusion", "vertical"):
                a = res["ame"][m][c]
                L.append(f"  {NAMES[m]:15s} {c:9s} Dir {a['direct']['est']:+.1f} CoT {a['cot']['est']:+.1f} P1 {a['p1']['est']:+.1f} "
                         f"P2 {a['p2']['est']:+.1f} P3 {a['p3']['est']:+.1f} | P1-D {f_ci(a['p1-direct'])} P2-D {f_ci(a['p2-direct'])} "
                         f"P3-D {f_ci(a['p3-direct'])} | P1f {a['p1f']['est']:+.1f} P1f-D {f_ci(a['p1f-direct'])} | P1-CoT {f_ci(a['p1-cot'])} CoT-D {f_ci(a['cot-direct'])}")
        L.append("")
    L.append("== Written relation (claim_parse v3; last claim; x = written acc - Direct on parsed items)")
    for m in M7:
        for s in ("cues", "syn"):
            w = res["written"].get(m, {}).get(s)
            if not w:
                continue
            L.append(f"  {NAMES[m]:15s} {s:4s} " + " | ".join(
                f"{a}: cov {w[a]['last_coverage']:.0f} first-cov {w[a]['first_coverage']:.0f} wr {w[a]['last_written_acc'] or float('nan'):.1f} "
                f"x {w[a]['last_x'] if w[a]['last_x'] is not None else float('nan'):+.1f} agree T/F {w[a]['agree_if_true']}/{w[a]['agree_if_false']}"
                for a in ["cot"] + PROMPTS))
    L.append("")
    L.append("== Sanity / costs")
    for m in M7:
        L.append(f"  {NAMES[m]:15s} {json.dumps(res['sanity'].get(m, {}))}")
    return L


def latex(res: dict) -> str:
    def c(x: dict) -> str:
        star = x["ci"][0] > 0 or x["ci"][1] < 0
        v = f"{x['est']:+.1f}"
        return (f"$\\mathbf{{{v}}}$" if star else f"${v}$") + f"{{\\scriptsize$[{x['ci'][0]:+.1f},{x['ci'][1]:+.1f}]$}}"
    Lx = [r"% Exp-44 (rigor_e44.py): label-free prompt variants of the CoT prefix. Not \input; bare tabular like the other",
          r"% supp tables. Suggested caption: \textbf{Label-free prompt variants of the CoT prefix (Exp-44, pre-registered).}",
          r"% The assistant prefix ``Let's think step by step.'' is replaced by P1 (occlusion first: ``First, look at where the",
          r"% two shapes overlap and say which shape's outline is complete there (that shape is in front). Then answer.''), P2",
          r"% (restate the question first) or P3 (both); greedy, <=512 tokens, same-turn readout. Accuracy (%) of Direct and CoT",
          r"% and differences in points with 95% paired cluster-bootstrap CIs (twin pairs / images; bold: CI excludes 0).",
          r"% Mean: unweighted over the seven models, same resampled clusters for all models. Pre-registered primary endpoint:",
          r"% P1-Direct on the cue set and on BLINK + CV-Bench depth, both CI lower bounds > 0. Forced: P1f-Direct, where",
          r"% empty continuations are regenerated with EOS suppressed for 16 tokens (pre-registered secondary).",
          r"% $^\dagger$: the P1 continuation is empty on > 50% of items (the model ends the turn after the instruction),",
          r"% so P1 there is Direct with the instruction in context. Wrap in \resizebox{\textwidth}{!}{...} (11 columns).",
          r"\begin{tabular}{@{}llccccccccc@{}}", r"\toprule",
          r"Set & Model & \texttt{Dir.} & \texttt{CoT} & P1$-$\texttt{Dir.} & P2$-$\texttt{Dir.} & P3$-$\texttt{Dir.} & "
          r"P1$-$\texttt{CoT} & P2$-$\texttt{CoT} & P3$-$\texttt{CoT} & P1f$-$\texttt{Dir.} \\", r"\midrule"]
    short = {"cues": "Cue set", "syn": "Syn. Depth", "nat": "BLINK+CVB"}
    for s in ("cues", "syn", "nat"):
        t = res["acc"].get(s)
        if not t:
            continue
        rows = t["models"] + ["pooled"]
        for k, m in enumerate(rows):
            r = t["per_model"][m] if m != "pooled" else t["pooled"]
            lab = (f"\\multirow{{{len(rows)}}}{{*}}{{{short[s]}}}" if k == 0 else "")
            nm = NAMES.get(m, r"\textit{Mean}")
            emp = res["sanity"].get(m, {}).get("gen", {}).get(s, {}).get("p1", {}).get("empty_pct", 0)
            nm += r"$^\dagger$" if emp > 50 else ""
            Lx.append(f"{lab} & {nm} & {r['acc:direct']:.1f} & {r['acc:cot']:.1f} & " +
                      " & ".join(c(r[x]) for x in ("p1-direct", "p2-direct", "p3-direct", "p1-cot", "p2-cot", "p3-cot", "p1f-direct")) + r" \\")
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
    # nat = blink + cvb pooled per model
    for m in M7:
        if "blink" in datas[m] and "cvb" in datas[m]:
            datas[m]["nat"] = {x: datas[m]["blink"][x] | datas[m]["cvb"][x] for x in ARMS}
    rng = np.random.default_rng(44)
    res: dict = {"prereg": str(O / "e44" / "prereg.txt"), "nb": a.nb, "acc": {}}
    for s in SETS:
        res["acc"][s] = acc_table(datas, s, a.nb, rng)
    pr = {s: res["acc"][s]["pooled"]["p1-direct"] for s in ("cues", "nat") if res["acc"].get(s)}
    res["secondary_p1f"] = {s: res["acc"][s]["pooled"]["p1f-direct"] for s in ("cues", "nat") if res["acc"].get(s)}
    complete = all(len(res["acc"].get(s, {}).get("models", [])) == 7 for s in ("cues", "nat"))
    met = complete and all(pr[s]["ci"][0] > 0 for s in ("cues", "nat"))
    rem = complete and all(pr[s]["ci"][1] >= 0 for s in ("cues", "nat")) and res["acc"]["cues"]["pooled"]["p1-cot"]["ci"][0] > 0
    res["primary"] = pr | {"met": met, "removes_cot_loss": rem, "complete_7_models": complete}
    if not a.no_ame:
        with Pool(7) as pool:
            out = pool.map(ame_model, [(m, datas[m]["cues"], a.nb_ame) for m in M7 if "cues" in datas[m]])
        res["ame"] = dict(out)
    res["written"] = written(datas, np.random.default_rng(32))
    res["sanity"] = {m: sanity(m, loaded[m]) for m in M7}
    B.mkdir(parents=True, exist_ok=True)
    json.dump(res, open(B / "e44_prompt_fix.json", "w"), indent=1)
    txt = "\n".join(report(res)) + "\n"
    (B / "rigor_e44.txt").write_text(txt)
    (PAPER / "tables" / "supp_e44.tex").write_text(latex(res))
    shutil.copy(B / "e44_prompt_fix.json", O / "e44" / "summary.json")
    shutil.copy(B / "rigor_e44.txt", O / "e44" / "rigor_e44.txt")
    # per-item correctness (for re-analysis)
    for s in ("cues", "syn", "blink", "cvb"):
        with open(O / "e44" / f"items_{s}.jsonl", "w") as f:
            for m in M7:
                if s not in datas[m]:
                    continue
                for k in datas[m][s]["direct"]:
                    f.write(json.dumps({"model": m, "set": s, "item_id": k, "image_id": datas[m][s]["direct"][k]["image_id"],
                                        "gold": datas[m][s]["direct"][k]["gold"],
                                        **{x: pe(datas[m][s][x][k]) for x in ARMS}}) + "\n")
    print(txt)


if __name__ == "__main__":
    main()
