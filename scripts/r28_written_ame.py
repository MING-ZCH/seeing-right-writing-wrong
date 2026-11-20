"""Exp-63a (R25 Argon leftover): cue AMEs of the WRITTEN relation vs Direct and CoT final answers (CPU only, 0 API calls).

Motivation: the Stage-1 AME (r18_cue_ame.py, Fig. 3a / Tab. supp_cue_ame) is computed from final answers, so a
CoT - Direct change mixes Stage 1 (which relation is written) with Stage 3 (how the conclusion maps it to an option).
Here the same probit / AME is fitted to three outcomes on the SAME cue-factorial items:
  D  Direct answer                says subject front (rigor_e17f.glm_rows coding)
  W  written relation             1 if the last relation of the conclusion-removed rationale R_cut puts the queried
                                  subject x in front, 0 if it puts the other shape in front (rigor_e54.cue_data coding:
                                  claim == x, i.e. the option the relation implies under 'in front' phrasing; for
                                  'behind' questions glm_rows recodes the answer the same way, so D, W, C share one scale)
  C  CoT final answer             says subject front
Regressors, ridge-0.1 probit, discrete AME (Phi(x_k=+1) - Phi(x_k=-1), points) and the twin-pair (image_id) cluster
bootstrap with 1,000 draws and seed 18 are those of r18_cue_ame.py; D, W, C are resampled jointly. Items: those with
a written relation (coverage reported); D and C are refitted on the same covered items.

Written-relation sources (variants):
  consensus   Exp-45 two-judge consensus (Claude Opus 5.5 + GPT-6-Astra, text only) on R_cut (primary; 9 models)
  opus / gpt  one judge alone (same R_cut texts)
  v3_trunc    claim_parse v3 rule parser on the same R_cut (all 13 models; the only R_cut source for the Exp-43 and
              Exp-48 models)
  v3_full     v3 on the full text, conclusion kept (Exp-32 / paper x definition; shows what the conclusion adds)
  judge_kept  Exp-48 two-judge consensus, conclusion kept (answer line stripped), MiMo-VL / GLM-4.1V only
-> outputs/bundle/r28_written_ame.json, outputs/bundle/rigor_r28_written_ame.txt
Usage: python r28_written_ame.py [--nb 1000] [--workers 30]
"""
from __future__ import annotations
import vtax_paths as _vtp  # noqa: E402

import argparse
import json
import re
import time
from collections import defaultdict
from multiprocessing import Pool
from pathlib import Path

import numpy as np

from r18_cue_ame import COL, CUES, ame, diff
from rigor_e17f import glm_rows, probit_fit

O = Path(f"{_vtp.OUTPUTS}")
B = Path("outputs/bundle")
M7 = ["qwen3b", "qwen7b", "internvl3_8b", "phi35v", "molmo7b", "qwen3vl_8b", "internvl35_8b"]
TIER_A = M7 + ["qwen3vl_32b", "qwen3vl_8b_think"]
TIER_B = ["internvl35_38b", "qwen32b", "mimo_vl_7b_rl|cot", "glm41v_9b_think|cot", "mimo_vl_7b_rl|think", "glm41v_9b_think|think"]
NAMES = {"qwen3b": "Qwen2.5-VL-3B", "qwen7b": "Qwen2.5-VL-7B", "internvl3_8b": "InternVL3-8B", "phi35v": "Phi-3.5-V",
         "molmo7b": "Molmo-7B-D", "qwen3vl_8b": "Qwen3-VL-8B", "internvl35_8b": "InternVL3.5-8B",
         "qwen3vl_32b": "Qwen3-VL-32B", "qwen3vl_8b_think": "Qwen3-VL-8B-Thinking", "internvl35_38b": "InternVL3.5-38B",
         "qwen32b": "Qwen2.5-VL-32B", "mimo_vl_7b_rl|cot": "MiMo-VL-7B-RL (short CoT)",
         "glm41v_9b_think|cot": "GLM-4.1V-9B-Thinking (short CoT)", "mimo_vl_7b_rl|think": "MiMo-VL-7B-RL (thinking)",
         "glm41v_9b_think|think": "GLM-4.1V-9B-Thinking (thinking)"}
SEED = 18
BOX = re.compile(r"<\|(?:begin|end)_of_box\|>")

PRESTATED = """# Exp-63a pre-stated questions (written before any written-relation AME was computed, 2026-11-19)
# Q1 (height): is the height ('lower is closer') AME already lower in the written relation than in Direct
#     (W - Direct < 0, 95% CI below 0) in all / most models, i.e. does Stage 1 (writing) carry the height loss seen in
#     the final answers?  Expected if the paper's Stage-1 claim holds: yes in most of the 13 models.
# Q2 (occlusion, Molmo-7B-D and Qwen3-VL-32B): is the significant final-answer occlusion AME loss
#     (paper: -26.1, -12.0) present in W - Direct, or does it appear only in C - W (the answer / conclusion step,
#     Stage 3)?  Prior evidence (Exp-45 / Exp-54) suggests Molmo's loss is at the answer step; 32B unknown.
# Q3 (shortcut models): do the occlusion AME gains of the height-reliant models appear in W already?
# Decision rule: a change is 'in W' if the W - Direct CI excludes 0 with the same sign as C - Direct, and
#     'at the answer step' if C - W excludes 0 with that sign while W - Direct does not.
# Primary source: Exp-45 two-judge consensus on R_cut (9 models); rule-parser R_cut (v3_trunc) for the 4 models
#     without R_cut judge labels, reported separately as lower-quality evidence.
"""


# ============================================================================= data

def _rows_e45(m: str) -> dict:
    from e45_uniform_two_channel import E45, rj
    return {r["item"]: r for r in rj(E45 / "items.jsonl") if r["group"] == "open" and r["model"] == m and r["ds"] == "cues"}


def load_tier_a(m: str) -> dict:
    """rigor_e54.cue_data (Exp-45 consensus + v3 kept) plus single-judge and v3-on-R_cut variants."""
    from e45_analyze import load_labels, shape_records
    from rigor_e54 import cue_data
    d = cue_data(m)
    rows = _rows_e45(m)
    lab = load_labels()
    out = {"consensus": d["y_w"]["consensus"], "v3_full": d["y_w"]["v3_kept"]}
    for spec, nm in (("claude-opus-5-5", "opus"), ("gpt-6-astra", "gpt")):
        rec = {r["item"]: r for r in shape_records(list(rows.values()), lab, spec=spec)}
        out[nm] = np.array([np.nan if not rec[k]["has"] else float(rec[k]["claim"] == rows[k]["x"]) for k in d["keys"]])
    out["v3_trunc"] = np.array([np.nan if rows[k]["v3_trunc"] is None else float(rows[k]["v3_trunc"] == rows[k]["x"])
                                for k in d["keys"]])
    return {"keys": d["keys"], "X": d["X"], "truth": d["truth"], "y_dir": d["y_dir"], "y_cot": d["y_cot"],
            "pair": d["pair"], "y_w": out}


def _v3_rcut(rec: dict) -> str | None:
    from e35_natural_two_channel import truncate
    from rigor_e32 import body, claims
    mt = rec["meta"]
    sents = body(BOX.sub("", rec.get("text") or ""))
    t, _ = truncate(sents, [mt["front_name"], mt["back_name"]])
    return claims(t, mt["front_name"], mt["back_name"])[1]


def _v3_full(rec: dict) -> str | None:
    from rigor_e32 import body, claims
    mt = rec["meta"]
    return claims(body(BOX.sub("", rec.get("text") or "")), mt["front_name"], mt["back_name"])[1]


def load_tier_b(m: str) -> dict:
    """Exp-43 (e10 layout) and Exp-48 (key 'model|arm') models: v3 on R_cut, v3 full, Exp-48 judges (conclusion kept)."""
    if "|" in m:
        import rigor_e48 as R48
        mm, arm = m.split("|")
        A = R48.load_arms(f"{mm}|cues|{arm}")
    else:
        from rigor_e31 import load_arms
        mm, arm = m, "cot"
        A = load_arms(m)
    keys = sorted(set(A["direct"]) & set(A["cot"]))
    D = glm_rows([A["direct"][k] for k in keys], use_size=True)
    C = glm_rows([A["cot"][k] for k in keys], use_size=True)
    assert np.array_equal(D["X"], C["X"])
    xs = [A["cot"][k]["meta"]["x_name"] for k in keys]
    wt = [_v3_rcut(A["cot"][k]) for k in keys]
    wf = [_v3_full(A["cot"][k]) for k in keys]
    out = {"v3_trunc": np.array([np.nan if w is None else float(w == x) for w, x in zip(wt, xs)]),
           "v3_full": np.array([np.nan if w is None else float(w == x) for w, x in zip(wf, xs)])}
    if "|" in m:
        from e48_judge import JUDGES as J48
        from e45_uniform_two_channel import rj
        lab = {j: {tuple(r["k"]): r["label"] for r in rj(O / "e48" / f"judge_syn_{j}.jsonl") if not r.get("api_error")}
               for j in J48}
        items = {r["item"]: r for r in rj(O / "e48" / "judge_items_syn.jsonl") if r["model"] == mm and r["arm"] == arm
                 and r["set"] == "cues"}
        jk = []
        for k in keys:
            it = items.get(k)
            a, b = (lab[J48[0]].get((mm, arm, "cues", k)), lab[J48[1]].get((mm, arm, "cues", k))) if it else (None, None)
            ok = it is not None and a == b and a in (it["x"], it["y"])
            jk.append(float(a == it["x"]) if ok else np.nan)
        out["judge_kept"] = np.array(jk)
    return {"keys": keys, "X": D["X"], "truth": D["truth"], "y_dir": D["y"], "y_cot": C["y"], "pair": np.array(D["pair"]),
            "y_w": out}


def load(m: str) -> dict:
    return load_tier_a(m) if m in TIER_A else load_tier_b(m)


# ============================================================================= statistics

def cue_stats(X: np.ndarray, y: np.ndarray) -> dict:
    b = probit_fit(X, y)
    return {c: {"ame": ame(X, b, COL[c]), "diff": diff(X, y, COL[c])} for c in CUES}


def q95(v) -> list[float]:
    v = np.asarray(v, float)
    return [round(float(np.quantile(v, .025)), 1), round(float(np.quantile(v, .975)), 1)]


def fit_set(X: np.ndarray, pair: np.ndarray, ys: dict, nb: int) -> dict:
    """r18_cue_ame bootstrap on the given items: estimates + joint twin-pair draws for every outcome in ys."""
    pairs = sorted(set(pair))
    pidx = {p: np.where(pair == p)[0] for p in pairs}
    est = {a: cue_stats(X, y) for a, y in ys.items()}
    rng = np.random.default_rng(SEED)
    bs = {a: {c: {k: [] for k in ("ame", "diff")} for c in CUES} for a in ys}
    for _ in range(nb):
        s = np.concatenate([pidx[pairs[j]] for j in rng.integers(0, len(pairs), len(pairs))])
        for a, y in ys.items():
            st = cue_stats(X[s], y[s])
            for c in CUES:
                for k in ("ame", "diff"):
                    bs[a][c][k].append(st[c][k])
    return {"est": est, "bs": bs, "n": len(X), "n_pairs": len(pairs)}


def summarize(fs: dict, contrasts: list[tuple[str, str]]) -> dict:
    out = {"n": fs["n"], "n_pairs": fs["n_pairs"]}
    for c in CUES:
        out[c] = {}
        for k in ("ame", "diff"):
            r = {a: {"est": round(fs["est"][a][c][k], 1), "ci": q95(fs["bs"][a][c][k])} for a in fs["est"]}
            for a1, a0 in contrasts:
                v1, v0 = np.array(fs["bs"][a1][c][k]), np.array(fs["bs"][a0][c][k])
                r[f"{a1}-{a0}"] = {"est": round(fs["est"][a1][c][k] - fs["est"][a0][c][k], 1), "ci": q95(v1 - v0),
                                   "p_le0": round(float(np.mean(v1 - v0 <= 0)), 4)}
            out[c][k] = r
    return out


def job(args: tuple[str, str, int]) -> tuple[str, str, dict]:
    m, var, nb = args
    d = load(m)
    if var == "_full":
        fs = fit_set(d["X"], d["pair"], {"D": d["y_dir"], "C": d["y_cot"]}, nb)
        return m, var, summarize(fs, [("C", "D")])
    yw = d["y_w"][var]
    cov = ~np.isnan(yw)
    idx = np.where(cov)[0]
    fs = fit_set(d["X"][idx], d["pair"][idx], {"D": d["y_dir"][idx], "W": yw[idx], "C": d["y_cot"][idx]}, nb)
    r = summarize(fs, [("W", "D"), ("C", "W"), ("C", "D")])
    tr = d["truth"][idx]
    r["coverage_pct"] = round(100 * float(cov.mean()), 1)
    r["n_all"] = int(len(yw))
    r["acc_direct_cov"] = round(100 * float(np.mean(d["y_dir"][idx] == tr)), 1)
    r["acc_written"] = round(100 * float(np.mean(yw[idx] == tr)), 1)
    r["acc_cot_cov"] = round(100 * float(np.mean(d["y_cot"][idx] == tr)), 1)
    r["acc_direct_all"] = round(100 * float(np.mean(d["y_dir"] == d["truth"])), 1)
    r["acc_cot_all"] = round(100 * float(np.mean(d["y_cot"] == d["truth"])), 1)
    r["cot_agrees_written_pct"] = round(100 * float(np.mean(d["y_cot"][idx] == yw[idx])), 1)
    r["cot_agrees_written_by_truth"] = {"W_true": round(100 * float(np.mean((d["y_cot"][idx] == yw[idx])[yw[idx] == tr])), 1),
                                        "W_false": round(100 * float(np.mean((d["y_cot"][idx] == yw[idx])[yw[idx] != tr])), 1)
                                        if (yw[idx] != tr).any() else None}
    # coverage by cue level (selection check): share covered when the cue favours the subject vs not
    r["coverage_by_cue"] = {c: {"plus": round(100 * float(cov[d["X"][:, COL[c]] > 0].mean()), 1),
                                "minus": round(100 * float(cov[d["X"][:, COL[c]] < 0].mean()), 1)} for c in CUES}
    return m, var, r


def variants_of(m: str) -> list[str]:
    if m in TIER_A:
        return ["consensus", "opus", "gpt", "v3_trunc", "v3_full"]
    return ["v3_trunc", "v3_full"] + (["judge_kept"] if "|" in m else [])


# ============================================================================= sanity references

def paper_ref(m: str) -> dict | None:
    """Stored final-answer AMEs (all items) for the reproduction check."""
    if m in M7:
        a = json.load(open(B / "cue_ame.json"))["models"][m]
        return {c: a[c]["ame"] for c in CUES}
    if m in ("qwen3vl_32b", "qwen3vl_8b_think"):
        a = json.load(open(B / "e31_cue_ladder.json"))["models"][m]["ame"]
        return {c: a[c]["ame"] for c in CUES}
    if m in ("internvl35_38b", "qwen32b"):
        a = json.load(open(B / "e43_big_open.json"))["models"][m]["ame"]
        return {c: a[c]["ame"] for c in CUES}
    mm, arm = m.split("|")
    a = json.load(open(B / "e48_reasoning_open.json"))["synthetic"][f"{mm}|cues|{arm}"]["ame"]
    return {c: a[c]["ame"] for c in CUES}


# ============================================================================= report

def f(x: dict) -> str:
    return f"{x['est']:+.1f} [{x['ci'][0]:+.1f}, {x['ci'][1]:+.1f}]"


def sig(x: dict) -> int:
    return 1 if x["ci"][0] > 0 else (-1 if x["ci"][1] < 0 else 0)


def report(res: dict) -> list[str]:
    L = []
    L.append("== Sanity: all-item Direct / CoT AME reproduce the stored final-answer AMEs (same seed, same draws)")
    for m, r in res.items():
        full, ref = r["_full"], r["paper_ref"]
        bad = []
        for c in CUES:
            for a, ka in (("D", "direct"), ("C", "cot")):
                if abs(full[c]["ame"][a]["est"] - ref[c][ka]["est"]) > 0.051 or any(abs(u - v) > 0.051 for u, v in zip(full[c]["ame"][a]["ci"], ref[c][ka]["ci"])):
                    bad.append(f"{c}/{a} {full[c]['ame'][a]['est']} {full[c]['ame'][a]['ci']} vs {ref[c][ka]['est']} {ref[c][ka]['ci']}")
        L.append(f"  {NAMES[m]:34s} n={full['n']} {'OK (est + CI identical)' if not bad else 'DIFF: ' + '; '.join(bad)}")
    L.append("")
    for var_set, title in ((("consensus",), "PRIMARY (Exp-45 two-judge consensus on R_cut)"),
                           (("v3_trunc",), "RULE PARSER v3 on R_cut (all models; only source for Exp-43 / Exp-48 models)")):
        L.append(f"== {title}: AME points on covered items; W-D = writing, C-W = answer step, C-D = total (covered)")
        for m, r in res.items():
            for var in var_set:
                if var not in r["variants"]:
                    continue
                v = r["variants"][var]
                L.append(f"  {NAMES[m]} [{var}] coverage {v['coverage_pct']}% (n {v['n']}/{v['n_all']}, pairs {v['n_pairs']}) | acc Dir {v['acc_direct_cov']} "
                         f"W {v['acc_written']} CoT {v['acc_cot_cov']} | CoT = W {v['cot_agrees_written_pct']}% (W true {v['cot_agrees_written_by_truth']['W_true']}, "
                         f"false {v['cot_agrees_written_by_truth']['W_false']})")
                for c in CUES:
                    a = v[c]["ame"]
                    L.append(f"     {c:9s} D {a['D']['est']:6.1f}  W {a['W']['est']:6.1f}  C {a['C']['est']:6.1f} | W-D {f(a['W-D'])}  C-W {f(a['C-W'])}  "
                             f"C-D {f(a['C-D'])} | all-item C-D {f(r['_full'][c]['ame']['C-D'])}")
        L.append("")
    L.append("== Robustness of W-D / C-W (height, occlusion) across written-relation sources")
    for m, r in res.items():
        parts = []
        for var, v in r["variants"].items():
            parts.append(f"{var} (cov {v['coverage_pct']}): vert W-D {f(v['vertical']['ame']['W-D'])} C-W {f(v['vertical']['ame']['C-W'])}; "
                         f"occ W-D {f(v['occlusion']['ame']['W-D'])} C-W {f(v['occlusion']['ame']['C-W'])}")
        L.append(f"  {NAMES[m]}: " + " || ".join(parts))
    L.append("")
    L.append("== Answers to the pre-stated questions")
    prim = {m: r["variants"].get("consensus") or r["variants"]["v3_trunc"] for m, r in res.items()}
    srcs = {m: ("consensus" if "consensus" in r["variants"] else "v3_trunc") for m, r in res.items()}
    core13 = [m for m in res if not m.endswith("|think")]
    for grp, ms in (("9 models with judge labels", TIER_A), ("13 open models (Exp-43/48 via v3 parser; MiMo/GLM short CoT)", core13)):
        ms = [m for m in ms if m in prim]
        neg = [m for m in ms if sig(prim[m]["vertical"]["ame"]["W-D"]) < 0]
        lt = [m for m in ms if prim[m]["vertical"]["ame"]["W-D"]["est"] < 0]
        cneg = [m for m in ms if sig(prim[m]["vertical"]["ame"]["C-D"]) < 0]
        share = [prim[m]["vertical"]["ame"]["W-D"]["est"] / prim[m]["vertical"]["ame"]["C-D"]["est"] for m in ms
                 if abs(prim[m]["vertical"]["ame"]["C-D"]["est"]) > 1]
        L.append(f"Q1 [{grp}]: height W-D < 0 (point) in {len(lt)}/{len(ms)}, CI < 0 in {len(neg)}/{len(ms)} "
                 f"(C-D CI < 0 in {len(cneg)}/{len(ms)}); W-D as share of C-D (covered): median {np.median(share):.2f}, range "
                 f"{min(share):.2f}-{max(share):.2f}. Not CI < 0: {[NAMES[m] for m in ms if m not in neg]}")
    for m in [m for m in ("molmo7b", "qwen3vl_32b") if m in prim]:
        o = prim[m]["occlusion"]["ame"]
        L.append(f"Q2 {NAMES[m]} [{srcs[m]}]: occlusion D {o['D']['est']} W {o['W']['est']} C {o['C']['est']} | W-D {f(o['W-D'])} | C-W {f(o['C-W'])} | "
                 f"C-D (covered) {f(o['C-D'])} | all-item C-D {f(res[m]['_full']['occlusion']['ame']['C-D'])}")
    for m in res:
        o = prim[m]["occlusion"]["ame"]
        if sig(res[m]["_full"]["occlusion"]["ame"]["C-D"]) > 0:
            L.append(f"Q3 {NAMES[m]} [{srcs[m]}]: occlusion gain C-D {f(o['C-D'])} | W-D {f(o['W-D'])} | C-W {f(o['C-W'])}")
    return L


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--nb", type=int, default=1000)
    ap.add_argument("--workers", type=int, default=30)
    ap.add_argument("--models", default=",".join(TIER_A + TIER_B))
    a = ap.parse_args()
    txt = B / "rigor_r28_written_ame.txt"
    txt.write_text(PRESTATED + f"# run started {time.strftime('%Y-%m-%d %H:%M:%S')}, nb={a.nb}, seed={SEED}\n")
    models = a.models.split(",")
    jobs = [(m, v, a.nb) for m in models for v in ["_full"] + variants_of(m)]
    t0 = time.time()
    with Pool(min(a.workers, len(jobs))) as pool:
        out = pool.map(job, jobs, chunksize=1)
    res: dict = {m: {"name": NAMES[m], "tier": "A" if m in TIER_A else "B", "variants": {}} for m in models}
    for m, v, r in out:
        if v == "_full":
            res[m]["_full"] = r
        else:
            res[m]["variants"][v] = r
    for m in models:
        res[m]["paper_ref"] = paper_ref(m)
    lines = report(res)
    doc = {"_doc": __doc__, "prestated": PRESTATED, "nb": a.nb, "seed": SEED, "models": res,
           "runtime_s": round(time.time() - t0, 1)}
    json.dump(doc, open(B / "r28_written_ame.json", "w"), indent=1)
    with open(txt, "a") as fh:
        fh.write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
