"""Exp-35: out-of-sample test of the two-channel fit (Exp-32) on natural-image depth items (CPU + text-only API).

build    Per model x dataset item records: CV-Bench 3D Depth (600, main pool: e2b rationales = Tab. 1 rationales,
         Tab. 1 Direct / CoT readouts from build_bundle.load_main; Thinking: e4 native trace + 'think' readout) and
         BLINK Relative Depth (124, held-out pool: e6x Direct / CoT as in heldout_breakdown.py; Thinking: e6x think).
         The rationale's conclusion is removed (truncate()): answer sentences (option letter, 'the answer is'),
         everything after the last answer sentence, a 'Conclusion' header and what follows it, and trailing
         concluding sentences (therefore / so / based on ...) or sentences that merely restate an option.
judge    Text-only judge (question, option labels, truncated rationale): which option the LAST relational depth claim
         names as closer to the camera, or NONE. Claude Opus 5.5 and GPT-6-Astra (api_client client, key from
         OPENAI_COMPAT_API_KEY, never printed). -> outputs/e35/judge_<judge>.jsonl (append, resumable)
analyze  Exp-32 cell statistics per model x dataset (consensus / each judge), out-of-sample prediction from the
         synthetic 21-point fit (two_channel.json primary|last_main; not refit), nulls, identity check, image bootstrap.
         -> bundle/e35_natural_two_channel.json, bundle/rigor_e35.txt, paper tables/supp_e35.tex

Usage: python e35_natural_two_channel.py build
       python e35_natural_two_channel.py judge --judge claude-opus-5-5 --rpm 90 --threads 24
       python e35_natural_two_channel.py analyze
"""
from __future__ import annotations
import vtax_paths as _vtp  # noqa: E402

import argparse
import json
import re
import threading
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from scipy.stats import pearsonr, spearmanr

import build_bundle as BB
from build_bundle import pe, rj
from e2f_claim_edit import split_sents

O = Path(f"{_vtp.OUTPUTS}")
E35 = O / "e35"
L = Path(f"{_vtp.LEGACY_RESULTS}")
B = Path("outputs/bundle")
PAPER = Path(f"{_vtp.PAPER}")
BLINK_PQ = Path(f"{_vtp.DATASETS}/heldout/BLINK/"
                "Relative_Depth/val-00000-of-00001.parquet")
M7 = ["qwen3b", "qwen7b", "internvl3_8b", "phi35v", "molmo7b", "qwen3vl_8b", "internvl35_8b"]
M8 = M7 + ["qwen3vl_8b_think"]
NAMES = {"qwen3b": "Qwen2.5-VL-3B", "qwen7b": "Qwen2.5-VL-7B", "internvl3_8b": "InternVL3-8B", "phi35v": "Phi-3.5-V",
         "molmo7b": "Molmo-7B-D", "qwen3vl_8b": "Qwen3-VL-8B", "internvl35_8b": "InternVL3.5-8B",
         "qwen3vl_8b_think": "Qwen3-VL-8B-Think."}
DSN = {"cvb3d": "CV-Bench 3D Depth", "blink": "BLINK Rel. Depth"}
JUDGES = ["claude-opus-5-5", "gpt-6-astra"]
PREFIX = "Let's think step by step."
NB = 2000

# ============================================================================= conclusion removal

ANS = re.compile(r"\b(?:the|final|correct|my|right)\s+(?:answer|option|choice)\b|\banswer\s*(?:is|would be|should be|:)|"
                 r"^\W*answer\b|\b(?:option|choice)\s*\(?\s*[AB]\b|(?<!point )(?<!points )(?<!circle )(?<!side )(?<!marker )(?<!labeled )(?<!label )(?<!box )"
                 r"(?<!object )(?<!location )(?<!dot )(?<!and )\(\s*[AB]\s*\)|\*\*\s*\(?[AB]\)?\s*\*\*|^\W*[AB]\W*$", re.I)
HEADER = re.compile(r"^\W*(?:\d+\.\s*)?\**\s*(?:conclusion|final answer|answer|summary)\b\W*$", re.I)
HEADER_INLINE = re.compile(r"^\W*(?:\d+\.\s*)?\*\*\s*(?:conclusion|final answer|summary)\s*\**\s*:", re.I)
CONC = re.compile(r"^\W*(?:\d+\.\s*)?\**\s*(?:therefore|thus|so\b|hence|in conclusion|to conclude|in summary|to summarize|"
                  r"overall|consequently|ultimately|based on|given (?:this|these|that|the)|considering|"
                  r"from (?:this|these|the)|it is (?:logical|reasonable|clear|safe) to conclude|"
                  r"we can (?:conclude|determine|infer|say)|putting (?:this|it) (?:all )?together|in this case,? the|conclusion)", re.I)
CMP = r"(?:closer|nearer|closest|nearest)"


def _norm(s: str) -> str:
    s = re.sub(r"[*_`#>]", "", s).strip().lower()
    s = re.sub(r"^(?:[-\u2022]\s*|\d+\.\s*)", "", s)
    s = re.sub(r"\s*\((?:highlighted|marked|in|labeled)[^)]*\)", "", s)
    s = re.sub(r"\b(?:highlighted|marked|outlined) (?:by|with|in) (?:a |the )?\w+ box\b", "", s)
    return re.sub(r"\s+", " ", s).strip(" .!")


def restates(s: str, opts: list[str]) -> bool:
    """Sentence is only 'X is closer (to the camera) (than Y)' or 'the object closer to the camera is X' for an option X."""
    t = _norm(s)
    t = re.sub(r"^(?:therefore|thus|so|hence|clearly|overall)\b,?\s*", "", t)
    o = "|".join(re.escape(x.lower()) for x in opts)
    obj = rf"(?:the |point |circle |object )?(?:{o})(?: object| point)?"
    pats = [rf"^{obj} (?:is|appears to be|seems to be|would be|should be|must be) (?:clearly |definitely |indeed |likely |probably )?"
            rf"(?:the )?{CMP}(?: one| object| point)?(?: to the (?:camera|viewer|observer))?(?: than {obj})?$",
            rf"^the (?:object|point|one) (?:that is |which is )?{CMP} to the (?:camera|viewer) is {obj}$",
            rf"^{obj} is {CMP}$"]
    return any(re.match(p, t) for p in pats)


def truncate(sents: list[str], opts: list[str]) -> tuple[list[str], dict]:
    """Remove the conclusion so that the remaining text cannot carry the answer itself."""
    info = Counter()
    s = [x for x in sents if x.strip()]
    hits = [j for j, x in enumerate(s) if ANS.search(x)]
    if hits:
        info["post_answer_dropped"] = len(s) - 1 - hits[-1]
        s = s[: hits[-1]]
        s2 = [x for x in s if not ANS.search(x)]
        info["answer_sents_dropped"] = len(hits) - 0
        s = s2
    hdr = [j for j, x in enumerate(s) if HEADER.match(x) or HEADER_INLINE.match(x)]
    if hdr:
        info["after_header_dropped"] = len(s) - hdr[-1]
        s = s[: hdr[-1]]
    while s and (CONC.match(s[-1]) or restates(s[-1], opts) or not re.search(r"[A-Za-z]{2,}", s[-1]) or s[-1].rstrip().endswith(":")):
        info["tail_dropped"] += 1
        s = s[:-1]
    return s, dict(info)


# ============================================================================= item records

QCVB = re.compile(r"the (.+?) \(highlighted by a (\w+) box\) or the (.+?) \(highlighted by a (\w+) box\)", re.I)


def body(text: str) -> list[str]:
    t = (text or "").strip()
    t = t[len(PREFIX):] if t.startswith(PREFIX) else t
    return split_sents(t)


def cvb_items(m: str) -> list[dict]:
    main = BB.load_main(O, L, m)
    if m == "qwen3vl_8b_think":
        src = {it["uid"]: it for it in json.load(open(O / "e4" / "qwen3vl_8b_think.json"))["items"] if it["task"] == "cvb_depth"}
        rat = {k: body(v["cot"]) for k, v in src.items()}
        qs = {k: v["question"] for k, v in src.items()}
    else:
        e2 = [r for r in rj(O / "e2b" / f"{m}.jsonl") if r["task"] == "cvb_depth"]
        rat = {r["item_id"]: r["sentences"] for r in e2}
        qs = {r["item_id"]: r["question"] for r in e2}
    out = []
    for k, d in main["direct"].items():
        if d[3] != "cvb_depth" or k not in main["cot"] or k not in rat:
            continue
        q = qs[k]
        mm = QCVB.search(q)
        oa = re.search(r"\(A\) (.+)", q).group(1).strip()
        ob = re.search(r"\(B\) (.+)", q).group(1).strip()
        col = {}
        if mm:
            col = {mm.group(1).lower(): mm.group(2), mm.group(3).lower(): mm.group(4)}
        gold = main["direct"][k][4] if main["direct"][k][0] else ("B" if main["direct"][k][4] == "A" else "A")
        out.append({"model": m, "ds": "cvb3d", "item": k, "cl": d[2], "gold": gold, "d_ans": d[4], "c_ans": main["cot"][k][4],
                    "question": q, "opts": {"A": oa, "B": ob},
                    "opt_desc": {"A": f"{oa} ({col.get(oa.lower(), '?')} box)", "B": f"{ob} ({col.get(ob.lower(), '?')} box)"},
                    "sents": rat[k]})
    return out


def blink_items(m: str) -> list[dict]:
    import pandas as pd
    df = pd.read_parquet(BLINK_PQ, columns=["idx", "prompt"])
    prm = dict(zip(df["idx"], df["prompt"]))
    if m == "qwen3vl_8b_think":
        R = defaultdict(dict)
        for r in rj(O / "e6x" / "qwen3vl_8b_think_think.jsonl"):
            R[r["item_id"]][r["condition"]] = r
        pairs = {k: (v["direct"], v["think"]) for k, v in R.items() if "direct" in v and "think" in v}
    else:
        D = {r["item_id"]: r for r in rj(O / "e6x" / f"{m}_direct.jsonl")}
        C = {r["item_id"]: r for r in rj(O / "e6x" / f"{m}_rest.jsonl") if r["condition"] == "cot"}
        pairs = {k: (D[k], C[k]) for k in D if k in C}
    out = []
    for k, (d, c) in pairs.items():
        if d["task"] != "blink_depth":
            continue
        out.append({"model": m, "ds": "blink", "item": k, "cl": d["image_id"], "gold": d["gold"], "d_ans": pe(d), "c_ans": pe(c),
                    "question": prm[k], "opts": {"A": "A", "B": "B"},
                    "opt_desc": {"A": "point A (A is closer)", "B": "point B (B is closer)"}, "sents": body(c.get("text"))})
    return out


def build() -> None:
    E35.mkdir(exist_ok=True)
    rows = []
    for m in M8:
        for r in cvb_items(m) + blink_items(m):
            opts = list(r["opts"].values()) + ([f"point {x}" for x in "AB"] if r["ds"] == "blink" else [])
            t, info = truncate(r["sents"], opts)
            r["trunc"] = t
            r["trunc_info"] = info
            rows.append(r)
    with open(E35 / "items.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    c = Counter((r["model"], r["ds"]) for r in rows)
    for (m, ds), n in sorted(c.items()):
        rs = [r for r in rows if r["model"] == m and r["ds"] == ds]
        d = np.mean([r["d_ans"] == r["gold"] for r in rs]) * 100
        cc = np.mean([r["c_ans"] == r["gold"] for r in rs]) * 100
        emp = np.mean([not r["trunc"] for r in rs]) * 100
        emp0 = np.mean([not r["sents"] for r in rs]) * 100
        print(f"{m:16s} {ds:6s} n={n} Direct {d:.1f} CoT {cc:.1f} CoT-D {cc - d:+.1f} | empty rationale {emp0:.1f}% empty after trunc {emp:.1f}%")


# ============================================================================= judge

PROMPT = """You will read a vision-language model's written reasoning about a depth question. You do NOT see the image; judge only what the text claims.

Question: {q}
Options:
(A) {a}
(B) {b}

Reasoning (the model's final answer and concluding sentences have been removed):
\"\"\"
{r}
\"\"\"

Task: find the LAST statement in the reasoning that claims which of the two options is closer to the camera (nearer to the viewer). Count any statement that states or directly implies the depth order of the two options, e.g. "X is closer/nearer to the camera", "Y is farther/further away (than X)", "X is in front of Y", "X blocks/occludes Y", "X is in the foreground while Y is in the background". A statement that only one option is "closer to the camera" or "farther away" counts (it implies the other is the opposite). Hedged claims ("X seems/should be closer") count. Do NOT count questions, hypotheticals ("if X were ..."), general rules ("objects lower in the image are usually closer"), or statements about other objects or about 2D position only (left/right/top/bottom) without a depth conclusion.

Reply in exactly this format:
QUOTE: <the claim, at most 25 words, or none>
CLOSER: <A, B, or NONE>
(A means the last such claim says option A is closer; B means option B is closer; NONE means the reasoning makes no such claim.)"""


def parse_closer(txt: str | None) -> str:
    m = re.findall(r"CLOSER\W*\s*\(?\s*(A|B|NONE)\b", (txt or "").upper())
    return m[-1] if m else "UNPARSED"


def key(r: dict) -> tuple:
    return (r["model"], r["ds"], r["item"])


def judge(jname: str, threads: int, rpm: float, limit: int | None, max_tokens: int = 4096, variant: str = "") -> None:
    import time
    import api_client
    rows = [r for r in rj(E35 / "items.jsonl") if r["trunc"]]
    p = E35 / f"judge_{jname}.jsonl"
    done = {key(r) for r in rj(p) if not r.get("api_error") and r["closer"] != "UNPARSED"}
    todo = [r for r in rows if key(r) not in done]
    todo.sort(key=lambda r: (r["ds"] != "blink", r["model"], r["item"]))
    if limit:
        todo = todo[:limit]
    print(f"[e35j] {jname}: {len(rows)} non-empty, done {len(done)}, to judge {len(todo)}", flush=True)
    lock, rl = threading.Lock(), threading.Lock()
    gap = 60.0 / rpm if rpm else 0.0
    nxt = [time.time()]
    cnt = Counter()
    f = open(p, "a")

    def go(r):
        if gap:
            with rl:
                t = max(nxt[0], time.time())
                nxt[0] = t + gap
            time.sleep(max(0.0, t - time.time()))
        rat = " ".join(r["trunc"])
        if variant == "brackets":  # Opus returns empty completions for some Thinking traces inside triple quotes
            rat = f"[model text begins] {rat} [model text ends]"
        txt = PROMPT.format(q=r["question"], a=r["opt_desc"]["A"], b=r["opt_desc"]["B"], r=rat)
        out = api_client.chat(jname, [{"role": "user", "content": txt}], max_tokens=max_tokens, effort="high")
        lab = parse_closer(out.get("text"))
        q = re.search(r"QUOTE:\s*(.*)", out.get("text") or "")
        rec = {"model": r["model"], "ds": r["ds"], "item": r["item"], "judge": jname, "closer": lab,
               "quote": q.group(1)[:200] if q else "", "raw": (out.get("text") or out.get("error", ""))[:300],
               "api_error": "error" in out, "variant": variant}
        with lock:
            f.write(json.dumps(rec) + "\n")
            f.flush()
            cnt["err" if rec["api_error"] else lab] += 1
            if sum(cnt.values()) % 200 == 0:
                print(f"[e35j] {jname}: {dict(cnt)}", flush=True)

    with ThreadPoolExecutor(threads) as ex:
        list(ex.map(go, todo))
    f.close()
    print(f"[e35j] {jname} finished: {dict(cnt)}", flush=True)


# ============================================================================= analysis

def labels() -> dict:
    """{judge: {key: label}} with the last non-error record per item; empty truncated rationale -> NONE."""
    out = {}
    for j in JUDGES:
        d = {}
        for r in rj(E35 / f"judge_{j}.jsonl"):
            if not r.get("api_error"):
                d[key(r)] = r["closer"]
        out[j] = d
    return out


def attach(rows: list[dict], lab: dict, spec: str) -> list[dict]:
    """Set r['claim'] in {A, B, None} per spec: consensus / one judge."""
    for r in rows:
        if not r["trunc"]:
            r["claim"], r["lab_status"] = None, "empty"
            continue
        a, b = lab[JUDGES[0]].get(key(r)), lab[JUDGES[1]].get(key(r))
        if spec == "consensus":
            if a is None or b is None:
                r["claim"], r["lab_status"] = None, "missing"
            elif "UNPARSED" in (a, b):
                r["claim"], r["lab_status"] = None, "unparsed"
            elif a == b and a in ("A", "B"):
                r["claim"], r["lab_status"] = a, "agree"
            elif a == b:
                r["claim"], r["lab_status"] = None, "agree_none"
            else:
                r["claim"], r["lab_status"] = None, "disagree"
        else:
            x = a if spec == JUDGES[0] else b
            r["claim"], r["lab_status"] = (x if x in ("A", "B") else None), ("missing" if x is None else "unparsed" if x == "UNPARSED" else "ok")
    return rows


def stats_arrays(rs: list[dict]) -> dict:
    return {"d": np.array([r["d_ans"] == r["gold"] for r in rs], float),
            "c": np.array([r["c_ans"] == r["gold"] for r in rs], float),
            "p": np.array([r["claim"] is not None for r in rs], float),
            "t": np.array([r["claim"] == r["gold"] for r in rs], float),
            "a": np.array([r["claim"] is not None and r["claim"] == r["c_ans"] for r in rs], float)}


def point(A: dict) -> dict:
    """Exp-32 quantities from per-item (or per-cluster summed) arrays. All in %."""
    d, c, p, t, a = (A[k] for k in "dcpta")
    npar = p.sum()
    out = {"n": int(len(d)), "acc_direct": 100 * d.mean(), "acc_cot": 100 * c.mean(), "coverage": 100 * p.mean(), "n_parsed": int(npar),
           "DpCm": int(((d == 1) & (c == 0)).sum()), "DmCp": int(((d == 0) & (c == 1)).sum())}
    out["y"] = out["acc_cot"] - out["acc_direct"]
    out["net"] = 100 * (out["DmCp"] - out["DpCm"]) / len(d)
    if npar:
        pm = p == 1
        out["written_acc"] = 100 * t[pm].mean()
        out["direct_on_parsed"] = 100 * d[pm].mean()
        out["cot_on_parsed"] = 100 * c[pm].mean()
        out["agree"] = 100 * a[pm].mean()
        tt, ff = pm & (t == 1), pm & (t == 0)
        out["n_true"], out["n_false"] = int(tt.sum()), int(ff.sum())
        out["p_agree_true"] = 100 * a[tt].mean() if tt.sum() else float("nan")
        out["p_agree_false"] = 100 * a[ff].mean() if ff.sum() else float("nan")
        out["x"] = out["written_acc"] - out["direct_on_parsed"]
        out["y_l4l"] = out["cot_on_parsed"] - out["direct_on_parsed"]
        W, aT, aF, D = out["written_acc"] / 100, out["p_agree_true"] / 100, out["p_agree_false"] / 100, out["direct_on_parsed"] / 100
        if np.isfinite(aT) and np.isfinite(aF):
            out["y_identity"] = 100 * ((aT + aF - 1) * W + (1 - aF) - D)
        if (~pm).sum():
            out["unparsed_cot_minus_direct"] = 100 * (c[~pm].mean() - d[~pm].mean())
    return out


def synth_fit() -> dict:
    tc = json.load(open(B / "two_channel.json"))
    pts = [c for k, c in tc["cells"].items() if c["ds"] in ("cues", "twins", "syn431") and c["model"] in M7]
    x = np.array([p["last_x"] for p in pts]); y = np.array([p["y"] for p in pts])
    sl, ic = np.polyfit(x, y, 1)
    f = tc["fits"]["primary|last_main"]
    assert abs(sl - f["ols_slope"]) < 5e-4 and abs(ic - f["ols_intercept"]) < 5e-3, (sl, ic)
    return {"slope": float(sl), "intercept": float(ic), "x": x, "y": y, "mean_y": float(y.mean()), "n": len(pts),
            "r": f["pearson_r"], "pts": pts}


def oos(obs: list[dict], fit: dict, sl=None, ic=None, mean_y=None) -> dict:
    sl = fit["slope"] if sl is None else sl
    ic = fit["intercept"] if ic is None else ic
    mean_y = fit["mean_y"] if mean_y is None else mean_y
    x = np.array([o["x"] for o in obs]); y = np.array([o["y"] for o in obs])
    yh = ic + sl * x
    e = y - yh
    out = {"n_points": len(x), "mae_fit": float(np.abs(e).mean()), "rmse_fit": float(np.sqrt((e ** 2).mean())), "bias_fit": float(e.mean()),
           "mae_null_mean_syn_y": float(np.abs(y - mean_y).mean()), "mae_null_zero": float(np.abs(y).mean()),
           "mae_identity_y_eq_x": float(np.abs(y - x).mean()),
           "r_pred_obs": float(pearsonr(yh, y)[0]) if np.ptp(x) > 0 and np.ptp(y) > 0 else float("nan"),
           "spearman_pred_obs": float(spearmanr(yh, y)[0]) if np.ptp(x) > 0 and np.ptp(y) > 0 else float("nan"),
           "sign_agree": int(np.sum(np.sign(yh) == np.sign(y))), "sign_agree_x": int(np.sum(np.sign(x) == np.sign(y)))}
    if len(x) > 2 and np.ptp(x) > 0:
        s2, i2 = np.polyfit(x, y, 1)
        out["natural_ols_descriptive"] = {"slope": float(s2), "intercept": float(i2)}
    return out


def analyze() -> None:
    fit = synth_fit()
    items = rj(E35 / "items.jsonl")
    lab = labels()
    lines = [f"Exp-35 natural-image two-channel test. Synthetic fit (21 points, two_channel.json primary|last_main, NOT refit): "
             f"slope {fit['slope']:.4f} intercept {fit['intercept']:+.4f}, mean synthetic y {fit['mean_y']:+.2f}"]
    res = {"_doc": __doc__, "synthetic_fit": {k: fit[k] for k in ("slope", "intercept", "mean_y", "n", "r")}, "specs": {}}
    # --- judge agreement
    ag = {}
    for (m, ds) in sorted({(r["model"], r["ds"]) for r in items}):
        ks = [key(r) for r in items if r["model"] == m and r["ds"] == ds and r["trunc"]]
        pr = [(lab[JUDGES[0]].get(k), lab[JUDGES[1]].get(k)) for k in ks]
        pr = [(a, b) for a, b in pr if a in ("A", "B", "NONE") and b in ("A", "B", "NONE")]
        ag[f"{m}|{ds}"] = agreement(pr) | {"n_nonempty": len(ks)}
    allp = []
    for r in items:
        if r["trunc"]:
            a, b = lab[JUDGES[0]].get(key(r)), lab[JUDGES[1]].get(key(r))
            if a in ("A", "B", "NONE") and b in ("A", "B", "NONE"):
                allp.append((a, b))
    ag["pooled"] = agreement(allp)
    res["judge_agreement"] = ag
    lines.append(f"JUDGE agreement pooled: {ag['pooled']}")
    for k, v in ag.items():
        if k != "pooled":
            lines.append(f"  {k:28s} n={v['n']} agree {v['pct_agree']:.1f}% kappa {v['kappa']:.3f} (A/B-only kappa {v['kappa_AB']:.3f})")
    # --- truncation summary
    tr = {}
    for (m, ds) in sorted({(r["model"], r["ds"]) for r in items}):
        rs = [r for r in items if r["model"] == m and r["ds"] == ds]
        tr[f"{m}|{ds}"] = {"n": len(rs), "empty_rationale": sum(not r["sents"] for r in rs), "empty_after_trunc": sum(not r["trunc"] for r in rs),
                           "with_answer_sentence": sum("answer_sents_dropped" in r["trunc_info"] for r in rs),
                           "post_answer_text_dropped": sum(r["trunc_info"].get("post_answer_dropped", 0) > 0 for r in rs),
                           "tail_dropped_mean": round(float(np.mean([r["trunc_info"].get("tail_dropped", 0) for r in rs])), 2),
                           "sents_mean": round(float(np.mean([len(r["sents"]) for r in rs])), 1),
                           "kept_mean": round(float(np.mean([len(r["trunc"]) for r in rs])), 1)}
    res["truncation"] = tr
    # synthetic robustness: conclusion removal applied to the synthetic rationales with the v3 rule parser; its refit
    # gives a definition-matched synthetic fit (secondary predictor; still no natural point is used)
    stc = synth_trunc_check(lines)
    res["synthetic_truncation_check"] = stc
    fit_m = {"slope": stc["slope_trunc"], "intercept": stc["intercept_trunc"], "x": np.array([p["x_trunc"] for p in stc["points"]]),
             "y": np.array([p["y"] for p in stc["points"]])}
    fit_m["mean_y"] = float(fit_m["y"].mean())
    rng_master = np.random.default_rng(35)
    for spec in ["consensus"] + JUDGES:
        attach(items, lab, spec)
        cells = {}
        groups = defaultdict(list)
        for r in items:
            groups[(r["model"], r["ds"])].append(r)
        for (m, ds), rs in sorted(groups.items()):
            c = point(stats_arrays(rs)) | {"model": m, "ds": ds,
                                           "lab_status": dict(Counter(r["lab_status"] for r in rs))}
            if m == "phi35v" and ds == "blink":
                c["note"] = "empty rationales (124/124): not parseable"
            cells[f"{m}|{ds}"] = c
        # out-of-sample points: cells with >= 20 parsed items
        valid = [k for k, c in cells.items() if c.get("n_parsed", 0) >= 20]
        main_pts = [k for k in valid if not k.startswith("qwen3vl_8b_think")]
        for k, c in cells.items():
            if "x" in c:
                c["y_hat_fit"] = fit["intercept"] + fit["slope"] * c["x"]
                c["pred_err"] = c["y"] - c["y_hat_fit"]
        sets = {"instruct": main_pts, "instruct+think": valid,
                "instruct_cvb3d": [k for k in main_pts if k.endswith("cvb3d")], "instruct_blink": [k for k in main_pts if k.endswith("blink")],
                "instruct_wo_molmo_phi": [k for k in main_pts if k.split("|")[0] not in ("molmo7b", "phi35v")]}
        st = {sn: oos([cells[k] for k in ks], fit) for sn, ks in sets.items() if len(ks) >= 3}
        # like-for-like y (parsed items) and identity check
        st["instruct_l4l"] = oos([{"x": cells[k]["x"], "y": cells[k]["y_l4l"]} for k in main_pts], fit)
        idc = [k for k in main_pts if "y_identity" in cells[k]]
        yid = np.array([cells[k]["y_identity"] for k in idc]); yo = np.array([cells[k]["y"] for k in idc]); yl = np.array([cells[k]["y_l4l"] for k in idc])
        st["identity_check"] = {"label": "consistency check, NOT a prediction: y_hat = (aT + aF - 1) W + (1 - aF) - D from the natural points' own agreement rates (equals y on parsed items by construction)",
                                "max_abs_identity_minus_y_l4l": float(np.abs(yid - yl).max()),
                                "mae_identity_vs_y_all": float(np.abs(yid - yo).mean()), "r_identity_vs_y_all": float(pearsonr(yid, yo)[0])}
        # bootstrap over images (jointly across models within dataset) + synthetic-point bootstrap of the fit
        bs = boot(items, groups, main_pts, fit, rng_master)
        st["instruct_matched_fit"] = oos([cells[k] for k in main_pts], fit_m) | {
            "label": "secondary: synthetic fit refit after the same conclusion removal (v3 rule parser), not the Exp-32 fit"}
        bsm = boot(items, groups, main_pts, fit_m, np.random.default_rng(351))
        res["specs"][spec] = {"cells": cells, "oos": st, "boot": bs, "boot_matched_fit": bsm["images_only"],
                              "points_instruct": main_pts, "points_with_think": valid}
        lines.append(f"\n===== spec {spec} =====")
        for k, c in cells.items():
            lines.append(f"CELL {c['model']:16s} {c['ds']:5s} n={c['n']} cov {c['coverage']:5.1f} Dir {c['acc_direct']:5.1f} CoT {c['acc_cot']:5.1f} "
                         f"Dir|p {c.get('direct_on_parsed', float('nan')):5.1f} CoT|p {c.get('cot_on_parsed', float('nan')):5.1f} "
                         f"W {c.get('written_acc', float('nan')):5.1f} agree T/F {c.get('p_agree_true', float('nan')):5.1f}/{c.get('p_agree_false', float('nan')):5.1f} "
                         f"(nT {c.get('n_true', 0)} nF {c.get('n_false', 0)}) | x {c.get('x', float('nan')):+6.1f} y {c['y']:+6.1f} y_l4l {c.get('y_l4l', float('nan')):+6.1f} "
                         f"yhat {c.get('y_hat_fit', float('nan')):+6.1f} err {c.get('pred_err', float('nan')):+6.1f} | D+C- {c['DpCm']} D-C+ {c['DmCp']} {c['lab_status']}")
        for sn, s in st.items():
            lines.append(f"OOS {sn:24s} {json.dumps({k: (round(v, 3) if isinstance(v, float) else v) for k, v in s.items()})}")
        lines.append(f"BOOT {json.dumps(bs)}")
        lines.append(f"BOOT matched fit {json.dumps(bsm['images_only'])}")
    json.dump(res, open(B / "e35_natural_two_channel.json", "w"), indent=1, default=float)
    (B / "rigor_e35.txt").write_text("\n".join(lines) + "\n")
    (PAPER / "tables" / "supp_e35.tex").write_text(table(res))
    print("\n".join(lines))


def agreement(pr: list[tuple]) -> dict:
    if not pr:
        return {"n": 0, "pct_agree": float("nan"), "kappa": float("nan"), "kappa_AB": float("nan")}
    cats = ["A", "B", "NONE"]
    M = np.zeros((3, 3))
    for a, b in pr:
        M[cats.index(a), cats.index(b)] += 1
    n = M.sum()
    po = np.trace(M) / n
    pe_ = (M.sum(1) @ M.sum(0)) / n ** 2
    k = (po - pe_) / (1 - pe_) if pe_ < 1 else float("nan")
    M2 = M[:2, :2]
    n2 = M2.sum()
    k2 = float("nan")
    if n2:
        po2 = np.trace(M2) / n2
        pe2 = (M2.sum(1) @ M2.sum(0)) / n2 ** 2
        k2 = (po2 - pe2) / (1 - pe2) if pe2 < 1 else float("nan")
    return {"n": int(n), "pct_agree": float(100 * po), "kappa": float(k), "kappa_AB": float(k2),
            "confusion_opus_rows_gpt_cols_A_B_NONE": M.astype(int).tolist(),
            "none_rate_opus": float(100 * M[2].sum() / n), "none_rate_gpt": float(100 * M[:, 2].sum() / n)}


def boot(items, groups, main_pts, fit, rng) -> dict:
    """Image bootstrap (same resampled images for every model within a dataset) of x, y and the out-of-sample stats;
    second variant also resamples the 21 synthetic points to propagate fit uncertainty."""
    imgs = {ds: sorted({r["cl"] for r in items if r["ds"] == ds}) for ds in ("cvb3d", "blink")}
    # per cell: cluster-summed arrays [n, d, c, p, t, d*p, c*p]
    agg = {}
    for k in main_pts:
        m, ds = k.split("|")
        rs = groups[(m, ds)]
        A = stats_arrays(rs)
        ix = {c: i for i, c in enumerate(imgs[ds])}
        inv = np.array([ix[r["cl"]] for r in rs])
        M = np.zeros((len(imgs[ds]), 7))
        np.add.at(M, inv, np.stack([np.ones(len(rs)), A["d"], A["c"], A["p"], A["t"] * A["p"], A["d"] * A["p"], A["c"] * A["p"]], 1))
        agg[k] = M
    sx, sy = fit["x"], fit["y"]
    keys = ("mae_fit", "mae_null_mean_syn_y", "mae_null_zero", "mae_identity_y_eq_x", "r_pred_obs", "sign_agree", "bias_fit")
    out = {k: [] for k in keys}
    out2 = {k: [] for k in ("mae_fit", "r_pred_obs", "sign_agree", "mae_fit_minus_null_mean", "mae_fit_minus_null_zero")}
    diffs = {"fit_minus_null_mean": [], "fit_minus_null_zero": [], "fit_minus_y_eq_x": []}
    cellx = {k: [] for k in main_pts}
    celly = {k: [] for k in main_pts}
    for _ in range(NB):
        sel = {ds: rng.integers(0, len(v), len(v)) for ds, v in imgs.items()}
        obs = []
        for k in main_pts:
            s = agg[k][sel[k.split("|")[1]]].sum(0)
            x = 100 * (s[4] - s[5]) / s[3]
            y = 100 * (s[2] - s[1]) / s[0]
            obs.append({"x": x, "y": y})
            cellx[k].append(x); celly[k].append(y)
        o = oos(obs, fit)
        for kk in keys:
            out[kk].append(o[kk])
        diffs["fit_minus_null_mean"].append(o["mae_fit"] - o["mae_null_mean_syn_y"])
        diffs["fit_minus_null_zero"].append(o["mae_fit"] - o["mae_null_zero"])
        diffs["fit_minus_y_eq_x"].append(o["mae_fit"] - o["mae_identity_y_eq_x"])
        j = rng.integers(0, len(sx), len(sx))
        sl, ic = np.polyfit(sx[j], sy[j], 1)
        o2 = oos(obs, fit, sl=sl, ic=ic, mean_y=float(sy[j].mean()))
        out2["mae_fit"].append(o2["mae_fit"]); out2["r_pred_obs"].append(o2["r_pred_obs"]); out2["sign_agree"].append(o2["sign_agree"])
        out2["mae_fit_minus_null_mean"].append(o2["mae_fit"] - o2["mae_null_mean_syn_y"])
        out2["mae_fit_minus_null_zero"].append(o2["mae_fit"] - o2["mae_null_zero"])
    q = lambda v: [round(float(np.nanquantile(v, .025)), 3), round(float(np.nanquantile(v, .975)), 3)]
    res = {"n_boot": NB, "images_only": {k: q(v) for k, v in out.items()} | {f"mae_diff_{k}": q(v) for k, v in diffs.items()}
           | {f"p_mae_fit_not_better_{k}": float(np.mean(np.array(v) >= 0)) for k, v in diffs.items()},
           "images_and_synthetic_fit": {k: q(v) for k, v in out2.items()},
           "cell_x_ci": {k: q(v) for k, v in cellx.items()}, "cell_y_ci": {k: q(v) for k, v in celly.items()}}
    return res


def synth_trunc_check(lines: list[str]) -> dict:
    """Apply the same conclusion removal to the synthetic rationales and recompute last-claim x (claim_parse v3)."""
    import rigor_e32 as R
    orig_rec = R.rec

    def rec2(model, ds, iid, cluster, gold, d_ans, c_ans, sents, front, back, x, phrasing, extra=None):
        t, _ = truncate(sents, [front, back])
        return orig_rec(model, ds, iid, cluster, gold, d_ans, c_ans, t, front, back, x, phrasing, extra)

    R.rec = rec2
    try:
        items = R.load_items()
    finally:
        R.rec = orig_rec
    rng = np.random.default_rng(32)
    groups = defaultdict(list)
    for r in items:
        if r["model"] not in M7:
            continue
        if r["ds"] == "syn":
            if r["overlap"]:
                groups[(r["model"], "syn431")].append(r)
        else:
            groups[(r["model"], r["ds"])].append(r)
    tc = json.load(open(B / "two_channel.json"))["cells"]
    pts = []
    for (m, ds), rs in sorted(groups.items()):
        c = R.cell_stats(rs, rng)
        pts.append({"model": m, "ds": ds, "x_trunc": c.get("last_x"), "x_orig": tc[f"{m}|{ds}"]["last_x"], "y": c["y"],
                    "cov_trunc": c["last_coverage"], "cov_orig": tc[f"{m}|{ds}"]["last_coverage"]})
    x = np.array([p["x_trunc"] for p in pts]); y = np.array([p["y"] for p in pts])
    sl, ic = np.polyfit(x, y, 1)
    out = {"n": len(pts), "slope_trunc": float(sl), "intercept_trunc": float(ic), "r_trunc": float(pearsonr(x, y)[0]),
           "mean_abs_x_change": float(np.mean([abs(p["x_trunc"] - p["x_orig"]) for p in pts])),
           "max_abs_x_change": float(np.max([abs(p["x_trunc"] - p["x_orig"]) for p in pts])), "points": pts}
    lines.append(f"\nSYNTHETIC conclusion-removal check (v3 rule parser, 21 points): slope {sl:.3f} int {ic:+.2f} r {out['r_trunc']:.3f}; "
                 f"mean |x change| {out['mean_abs_x_change']:.2f} max {out['max_abs_x_change']:.2f}")
    for p in pts:
        lines.append(f"  {p['model']:14s} {p['ds']:6s} x {p['x_orig']:+6.2f} -> {p['x_trunc']:+6.2f} cov {p['cov_orig']:.1f} -> {p['cov_trunc']:.1f}")
    return out


# ============================================================================= LaTeX table

def table(res: dict) -> str:
    S = res["specs"]["consensus"]
    C, st, bs = S["cells"], S["oos"], S["boot"]["images_only"]
    f = lambda v, d=1: "--" if v is None or not np.isfinite(v) else f"{v:.{d}f}"
    g = lambda v, d=1: "--" if v is None or not np.isfinite(v) else f"${v:+.{d}f}$"
    L_ = [r"\begin{tabular}{llrrrrrrrrrrrr}", r"\toprule",
          r"Model & Data & Cov. & \texttt{Dir.} & \texttt{Dir.}$|$p & \texttt{CoT} & $W$ & Agr.\ T/F & $x$ & $y$ & $y|$p & $\hat y$ & $y-\hat y$ & \texttt{D+C$-$}/\texttt{D$-$C+} \\",
          r"\midrule"]
    for ds in ("cvb3d", "blink"):
        for m in M8:
            k = f"{m}|{ds}"
            if k not in C:
                continue
            c = C[k]
            nm = NAMES[m] + ("$^\\dagger$" if m == "qwen3vl_8b_think" else "")
            if c.get("n_parsed", 0) < 20:
                L_.append(f"{nm} & {DSN[ds]} & {f(c['coverage'])} & {f(c['acc_direct'])} & -- & {f(c['acc_cot'])} & \\multicolumn{{3}}{{c}}{{not parseable (empty rationales)}} & "
                          f"{g(c['y'])} & -- & -- & -- & {c['DpCm']}/{c['DmCp']} \\\\")
                continue
            L_.append(f"{nm} & {DSN[ds]} & {f(c['coverage'])} & {f(c['acc_direct'])} & {f(c['direct_on_parsed'])} & {f(c['acc_cot'])} & "
                      f"{f(c['written_acc'])} & {f(c['p_agree_true'], 0)}/{f(c['p_agree_false'], 0)} & {g(c['x'])} & {g(c['y'])} & {g(c['y_l4l'])} & "
                      f"{g(c['y_hat_fit'])} & {g(c['pred_err'])} & {c['DpCm']}/{c['DmCp']} \\\\")
        L_.append(r"\midrule")
    L_.pop()
    a = st["instruct"]
    ag = res["judge_agreement"]["pooled"]
    sy = res["synthetic_fit"]
    L_ += [r"\bottomrule",
           r"\multicolumn{14}{p{0.97\linewidth}}{\scriptsize Written relation $W$: the option that the last depth claim of the CoT rationale names as closer "
           r"to the camera, after removing the answer and concluding sentences; text-only consensus of Claude Opus 5.5 and GPT-6-Astra "
           f"(agreement {ag['pct_agree']:.1f}\\%, $\\kappa={ag['kappa']:.2f}$); Cov.: share of items with a consensus claim. "
           r"$x = W - $\texttt{Dir.}$|$p (parsed items); $y=$ \texttt{CoT}$-$\texttt{Dir.} (all items); $y|$p on parsed items. "
           f"$\\hat y = {sy['intercept']:+.2f} + {sy['slope']:.3f}\\,x$: synthetic fit of \\cref{{tab:supp_two_channel}} (21 points), not refit. "
           f"Out of sample ({a['n_points']} instruct points): MAE {a['mae_fit']:.1f} pts vs.\\ "
           f"{a['mae_null_mean_syn_y']:.1f} (predict mean synthetic $y$; difference [{bs['mae_diff_fit_minus_null_mean'][0]:+.1f}, {bs['mae_diff_fit_minus_null_mean'][1]:+.1f}]), "
           f"{a['mae_null_zero']:.1f} (predict 0; [{bs['mae_diff_fit_minus_null_zero'][0]:+.1f}, {bs['mae_diff_fit_minus_null_zero'][1]:+.1f}]) "
           f"and {a['mae_identity_y_eq_x']:.1f} ($y=x$; [{bs['mae_diff_fit_minus_y_eq_x'][0]:+.1f}, {bs['mae_diff_fit_minus_y_eq_x'][1]:+.1f}]); "
           f"$r(\\hat y, y) = {a['r_pred_obs']:.2f}$ [{bs['r_pred_obs'][0]:.2f}, {bs['r_pred_obs'][1]:.2f}]; sign agreement {a['sign_agree']}/{a['n_points']} "
           f"(misses: InternVL3-8B and InternVL3.5-8B on CV-Bench, $|y| \\le 0.7$); "
           f"mean $y-\\hat y$ ${a['bias_fit']:+.1f}$ [{bs['bias_fit'][0]:+.1f}, {bs['bias_fit'][1]:+.1f}]. Brackets: image bootstrap ({S['boot']['n_boot']:,}), "
           f"MAE differences paired. Agr.\\ T/F: CoT answer agrees with the written relation when it is true / false (\\%). "
           r"$^\dagger$Native thinking trace, not in the out-of-sample statistics.} \\",
           r"\end{tabular}"]
    return "\n".join(L_) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("phase", choices=["build", "judge", "analyze"])
    ap.add_argument("--judge")
    ap.add_argument("--threads", type=int, default=16)
    ap.add_argument("--rpm", type=float, default=0.0)
    ap.add_argument("--limit", type=int)
    ap.add_argument("--max-tokens", type=int, default=4096)
    ap.add_argument("--variant", default="")
    a = ap.parse_args()
    if a.phase == "build":
        build()
    elif a.phase == "judge":
        judge(a.judge, a.threads, a.rpm, a.limit, a.max_tokens, a.variant)
    else:
        analyze()


if __name__ == "__main__":
    main()
