"""Exp-48: written relation on the natural depth sets by the Exp-35 two-judge extraction (CPU + text-only API).

build    outputs/e48/<model>_nat.jsonl (think and short-CoT arms) -> per-text records with the Exp-35 conclusion removal
         (e35_natural_two_channel.truncate on e2f split_sents; BLINK options "A"/"B" + "point A/B", CV-Bench object names
         with box colours parsed from the question by e35 QCVB) -> outputs/e48/judge_items_nat.jsonl
judge    Exp-35 PROMPT verbatim (question, option labels, truncated text -> option the LAST depth claim names closer, or
         NONE), Claude Opus 5.5 + GPT-6-Astra, effort high, api_client client (key from the environment, never printed)
         -> outputs/e48/judge_nat_<judge>.jsonl (append, resumable)
analyze  consensus label (both judges name the same option) and e35.point per model x arm x dataset (x, y, agreement,
         P(agree | true / false), coverage), image-cluster bootstrap 2,000 for x and y.
         -> outputs/bundle/e48_nat_two_channel.json (read by rigor_e48.py)
Usage: python e48_judge.py build | judge --judge claude-opus-5-5 --threads 16 | analyze
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

from build_bundle import pe, rj
from e2f_claim_edit import split_sents
from e35_natural_two_channel import PROMPT, QCVB, parse_closer, point, truncate

O = Path(f"{_vtp.OUTPUTS}")
E48 = O / "e48"
B = Path("outputs/bundle")
MODELS = ["mimo_vl_7b_rl", "glm41v_9b_think"]
ARMS = ["think", "cot"]
JUDGES = ["claude-opus-5-5", "gpt-6-astra"]
PREFIX = "Let's think step by step."
NB = 2000


def key(r: dict) -> tuple:
    return (r["model"], r["arm"], r["ds"], r["item"])


def build() -> None:
    rows = []
    for m in MODELS:
        by = defaultdict(dict)
        for r in rj(E48 / f"{m}_nat.jsonl"):
            by[r["item_id"]][r["condition"]] = r
        for k, d in by.items():
            dr = d["direct"]
            q = dr["question"]
            if dr["set"] == "blink":
                opts = {"A": "A", "B": "B"}
                desc = {"A": "point A (A is closer)", "B": "point B (B is closer)"}
                olist = ["A", "B", "point A", "point B"]
            else:
                oa = re.search(r"\(A\) (.+)", q).group(1).strip()
                ob = re.search(r"\(B\) (.+)", q).group(1).strip()
                mm = QCVB.search(q)
                col = {mm.group(1).lower(): mm.group(2), mm.group(3).lower(): mm.group(4)} if mm else {}
                opts = {"A": oa, "B": ob}
                desc = {"A": f"{oa} ({col.get(oa.lower(), '?')} box)", "B": f"{ob} ({col.get(ob.lower(), '?')} box)"}
                olist = [oa, ob]
            for arm in ARMS:
                if arm not in d:
                    continue
                c = d[arm]
                t = (c.get("text") or "").strip()
                t = t[len(PREFIX):] if t.startswith(PREFIX) else t
                t = re.sub(r"<\|(?:begin|end)_of_box\|>", "", t)
                sents = split_sents(t)
                tr, info = truncate(sents, olist)
                rows.append({"model": m, "arm": arm, "ds": "blink" if dr["set"] == "blink" else "cvb3d", "item": k,
                             "cl": dr["image_id"], "gold": dr["gold"], "d_ans": pe(dr), "c_ans": pe(c), "question": q,
                             "opts": opts, "opt_desc": desc, "sents": sents, "trunc": tr, "trunc_info": info})
    with open(E48 / "judge_items_nat.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    c = Counter((r["model"], r["arm"], r["ds"]) for r in rows)
    for g, n in sorted(c.items()):
        rs = [r for r in rows if (r["model"], r["arm"], r["ds"]) == g]
        print(g, n, "empty after truncation %.1f%%" % (100 * np.mean([not r["trunc"] for r in rs])))


def judge(jname: str, threads: int, limit: int | None) -> None:
    import api_client
    rows = [r for r in rj(E48 / "judge_items_nat.jsonl") if r["trunc"]]
    p = E48 / f"judge_nat_{jname}.jsonl"
    done = {tuple(r["k"]) for r in rj(p) if not r.get("api_error") and r["closer"] != "UNPARSED"}
    todo = [r for r in rows if key(r) not in done][:limit]
    print(f"[e48j] {jname}: {len(rows)} non-empty, done {len(done)}, to judge {len(todo)}", flush=True)
    lock = threading.Lock()
    cnt = Counter()
    f = open(p, "a")

    def go(r):
        rat = " ".join(r["trunc"])
        txt = PROMPT.format(q=r["question"], a=r["opt_desc"]["A"], b=r["opt_desc"]["B"], r=rat)
        out = api_client.chat(jname, [{"role": "user", "content": txt}], max_tokens=4096, effort="high")
        if not (out.get("text") or "").strip() and "error" not in out:  # Exp-35: bracketed delimiter for empty completions
            txt = PROMPT.format(q=r["question"], a=r["opt_desc"]["A"], b=r["opt_desc"]["B"],
                                r=f"[model text begins] {rat} [model text ends]")
            out = api_client.chat(jname, [{"role": "user", "content": txt}], max_tokens=4096, effort="high")
        lab = parse_closer(out.get("text"))
        rec = {"k": list(key(r)), "judge": jname, "closer": lab, "raw": (out.get("text") or out.get("error", ""))[:300],
               "api_error": "error" in out}
        with lock:
            f.write(json.dumps(rec) + "\n")
            f.flush()
            cnt["err" if rec["api_error"] else lab] += 1
            if sum(cnt.values()) % 200 == 0:
                print(f"[e48j] {jname}: {dict(cnt)}", flush=True)

    with ThreadPoolExecutor(threads) as ex:
        list(ex.map(go, todo))
    f.close()
    print(f"[e48j] {jname} finished: {dict(cnt)}", flush=True)


def analyze() -> dict:
    rows = rj(E48 / "judge_items_nat.jsonl")
    lab = {j: {tuple(r["k"]): r["closer"] for r in rj(E48 / f"judge_nat_{j}.jsonl") if not r.get("api_error")} for j in JUDGES}
    pairs = Counter()
    for r in rows:
        if not r["trunc"]:
            r["claim"] = None
            continue
        a, b = (lab[j].get(key(r)) for j in JUDGES)
        pairs[(a, b)] += 1
        r["claim"] = a if (a == b and a in ("A", "B")) else None
    both = [(lab[JUDGES[0]][key(r)], lab[JUDGES[1]][key(r)]) for r in rows if r["trunc"] and key(r) in lab[JUDGES[0]] and key(r) in lab[JUDGES[1]]]
    res = {"judge_agreement_pct": round(100 * float(np.mean([a == b for a, b in both])), 2) if both else None,
           "n_judged_both": len(both), "cells": {}}
    rng = np.random.default_rng(35)
    for g in sorted({(r["model"], r["arm"], r["ds"]) for r in rows}):
        rs = [r for r in rows if (r["model"], r["arm"], r["ds"]) == g]
        A = {"d": np.array([r["d_ans"] == r["gold"] for r in rs], float), "c": np.array([r["c_ans"] == r["gold"] for r in rs], float),
             "p": np.array([r["claim"] is not None for r in rs], float), "t": np.array([r["claim"] == r["gold"] for r in rs], float),
             "a": np.array([r["claim"] is not None and r["claim"] == r["c_ans"] for r in rs], float)}
        pt = point(A)
        cl = np.array([r["cl"] for r in rs])
        uc = np.unique(cl)
        idx = {c: np.where(cl == c)[0] for c in uc}
        bx, by_ = [], []
        for _ in range(NB):
            s = np.concatenate([idx[c] for c in rng.choice(uc, len(uc))])
            q = point({k: v[s] for k, v in A.items()})
            by_.append(q["y"])
            if "x" in q:
                bx.append(q["x"])
        pt["x_ci"] = [round(float(np.quantile(bx, .025)), 2), round(float(np.quantile(bx, .975)), 2)] if bx else None
        pt["y_ci"] = [round(float(np.quantile(by_, .025)), 2), round(float(np.quantile(by_, .975)), 2)]
        pt["empty_after_trunc_pct"] = round(100 * float(np.mean([not r["trunc"] for r in rs])), 1)
        res["cells"]["|".join(g)] = {k: (round(v, 3) if isinstance(v, float) else v) for k, v in pt.items()}
    json.dump(res, open(B / "e48_nat_two_channel.json", "w"), indent=1, default=float)
    print("judge agreement", res["judge_agreement_pct"], "n", res["n_judged_both"])
    for g, p in res["cells"].items():
        print(f"{g:32s} n {p['n']} cov {p['coverage']:.1f} written {p.get('written_acc', float('nan')):.1f} x {p.get('x', float('nan')):+.1f} {p['x_ci']} "
              f"y {p['y']:+.1f} {p['y_ci']} agree T/F {p.get('p_agree_true', float('nan')):.1f}/{p.get('p_agree_false', float('nan')):.1f} "
              f"(nF {p.get('n_false')})")
    return res


# ============================================================================= synthetic sets (Exp-38 protocol)
# Cue set and twins, writing arms think / cot: Exp-38 PROMPT verbatim ("a short text that describes an image", LAST
# front/behind statement, conclusion kept, final answer line stripped by e38 strip_answer), same two judges; consensus
# = both name the same shape. Used where the v3 rule parser covers few traces (GLM-4.1V native thinking).

SYN_ORDER = [("glm41v_9b_think", "think"), ("mimo_vl_7b_rl", "think"), ("glm41v_9b_think", "cot"), ("mimo_vl_7b_rl", "cot")]


def build_syn() -> None:
    from e38_closed_judge import strip_answer
    rows = []
    for m, arm in SYN_ORDER:
        for st in ("cues", "twins"):
            by = defaultdict(dict)
            for r in rj(E48 / f"{m}_{st}.jsonl"):
                by[r["item_id"]][r["condition"]] = r
            for k, d in by.items():
                c, dr = d[arm], d["direct"]
                mt = c["meta"]
                t = (c.get("text") or "").strip()
                t = t[len(PREFIX):].strip() if t.startswith(PREFIX) else t
                t = re.sub(r"<\|(?:begin|end)_of_box\|>", "", t)
                bd, nd = strip_answer(t)
                rows.append({"model": m, "arm": arm, "set": st, "item": k, "cl": c["image_id"], "phrasing": c["phrasing"],
                             "gold": c["gold"], "d_ans": pe(dr), "c_ans": pe(c), "x": mt["x_name"].lower(), "y": mt["y_name"].lower(),
                             "front": mt["front_name"].lower(), "body": bd, "n_restate_dropped": nd})
    with open(E48 / "judge_items_syn.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(Counter((r["model"], r["arm"], r["set"]) for r in rows))


def skey(r: dict) -> tuple:
    return (r["model"], r["arm"], r["set"], r["item"])


def judge_syn(jname: str, threads: int, limit: int | None) -> None:
    import api_client
    from e38_closed_judge import PROMPT as P38, parse_label
    rows = [r for r in rj(E48 / "judge_items_syn.jsonl") if r["body"]]
    p = E48 / f"judge_syn_{jname}.jsonl"
    done = {tuple(r["k"]) for r in rj(p) if not r.get("api_error") and r["label"] is not None}
    todo = [r for r in rows if skey(r) not in done][:limit]
    print(f"[e48s] {jname}: {len(rows)} non-empty, done {len(done)}, to judge {len(todo)}", flush=True)
    lock = threading.Lock()
    cnt = Counter()
    f = open(p, "a")

    def go(r):
        txt = P38.format(x=r["x"], y=r["y"], r=r["body"])
        out = api_client.chat(jname, [{"role": "user", "content": txt}], max_tokens=2048, effort="high")
        lab = parse_label(out.get("text"), r["x"], r["y"])
        rec = {"k": list(skey(r)), "judge": jname, "label": lab, "raw": (out.get("text") or out.get("error", ""))[:200],
               "api_error": "error" in out}
        with lock:
            f.write(json.dumps(rec) + "\n")
            f.flush()
            cnt["err" if rec["api_error"] else str(lab)] += 1
            if sum(cnt.values()) % 500 == 0:
                print(f"[e48s] {jname}: {sum(cnt.values())} {dict(Counter({('shape' if k not in ('none', 'unclear', 'None', 'err') else k): v for k, v in cnt.items()}))}", flush=True)

    with ThreadPoolExecutor(threads) as ex:
        list(ex.map(go, todo))
    f.close()
    print(f"[e48s] {jname} finished: {sum(cnt.values())}", flush=True)


def analyze_syn() -> dict:
    """Consensus last relation -> rigor_e32.cell_stats (x, y, agreement) and aT / aF with a pair-cluster bootstrap."""
    from rigor_e30 import implied
    from rigor_e32 import cell_stats
    rows = rj(E48 / "judge_items_syn.jsonl")
    lab = {j: {tuple(r["k"]): r["label"] for r in rj(E48 / f"judge_syn_{j}.jsonl") if not r.get("api_error")} for j in JUDGES}
    res = {"cells": {}}
    both = []
    for r in rows:
        a, b = (lab[j].get(skey(r)) for j in JUDGES)
        if a is not None and b is not None:
            both.append(a == b)
        r["claim"] = a if (a == b and a in (r["x"], r["y"])) else None
        r["n_judged"] = int(a is not None) + int(b is not None)
    res["judge_agreement_pct"] = round(100 * float(np.mean(both)), 2) if both else None
    res["n_judged_both"] = len(both)
    rng = np.random.default_rng(38)
    for g in sorted({(r["model"], r["arm"], r["set"]) for r in rows}):
        rs = [r for r in rows if (r["model"], r["arm"], r["set"]) == g]
        if not all(r["n_judged"] == 2 for r in rs if r["body"]):
            res["cells"]["|".join(g)] = {"complete": False, "n_judged_both": sum(r["n_judged"] == 2 for r in rs)}
            continue
        recs = []
        for r in rs:
            l = r["claim"]
            recs.append({"cl": r["cl"], "d_ok": r["d_ans"] == r["gold"], "c_ok": r["c_ans"] == r["gold"],
                         "last_true": None if l is None else l == r["front"],
                         "last_agree": None if l is None else implied(l, {"x_name": r["x"]}, r["phrasing"]) == r["c_ans"],
                         "first_true": None, "first_agree": None})
        cs = cell_stats(recs, np.random.default_rng(32))
        P = [x for x in recs if x["last_true"] is not None]
        T = [x for x in P if x["last_true"]]
        F = [x for x in P if not x["last_true"]]
        cs["p_agree_true"] = 100 * float(np.mean([x["last_agree"] for x in T])) if T else None
        cs["p_agree_false"] = 100 * float(np.mean([x["last_agree"] for x in F])) if F else None
        cs["n_true"], cs["n_false"] = len(T), len(F)
        cl = np.array([x["cl"] for x in recs])
        uc = np.unique(cl)
        idx = {c: np.where(cl == c)[0] for c in uc}
        bt, bf = [], []
        for _ in range(NB):
            s = np.concatenate([idx[c] for c in rng.choice(uc, len(uc))])
            ss = [recs[i] for i in s if recs[i]["last_true"] is not None]
            tt = [x["last_agree"] for x in ss if x["last_true"]]
            ff = [x["last_agree"] for x in ss if not x["last_true"]]
            if tt:
                bt.append(100 * np.mean(tt))
            if ff:
                bf.append(100 * np.mean(ff))
        cs["p_agree_true_ci"] = [round(float(np.quantile(bt, q)), 1) for q in (.025, .975)] if bt else None
        cs["p_agree_false_ci"] = [round(float(np.quantile(bf, q)), 1) for q in (.025, .975)] if bf else None
        cs["complete"] = True
        res["cells"]["|".join(g)] = {k: (round(v, 3) if isinstance(v, float) else v) for k, v in cs.items()}
    json.dump(res, open(B / "e48_syn_judge.json", "w"), indent=1, default=float)
    print("judge agreement", res["judge_agreement_pct"], "n", res["n_judged_both"])
    for g, c in res["cells"].items():
        if c.get("complete"):
            print(f"{g:30s} cov {c['last_coverage']:.1f} written {c['last_written_acc']:.1f} x {c['last_x']:+.2f} {c['last_x_ci']} y {c['y']:+.2f} "
                  f"agree {c['last_agree']:.1f} T/F {c['p_agree_true']:.1f}/{c['p_agree_false']:.1f} (nF {c['n_false']})")
        else:
            print(g, c)
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["build", "judge", "analyze", "build_syn", "judge_syn", "analyze_syn"])
    ap.add_argument("--judge", default=JUDGES[0])
    ap.add_argument("--threads", type=int, default=16)
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args()
    {"build": build, "judge": lambda: judge(a.judge, a.threads, a.limit), "analyze": analyze,
     "build_syn": build_syn, "judge_syn": lambda: judge_syn(a.judge, a.threads, a.limit), "analyze_syn": analyze_syn}[a.cmd]()


if __name__ == "__main__":
    main()
