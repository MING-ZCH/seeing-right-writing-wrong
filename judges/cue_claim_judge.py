"""E16: text-only judge parse of first depth-order claims on the cue-factorial CoT outputs.

Stratified sample: per model 15 CoT-wrong + 15 CoT-right prompts (seed 0). Each judge reads the rationale (no image)
and names the shape the FIRST order-stating sentence puts in front (or NONE). Compared with the rule parsers
v1 (e2f_claim_edit.first_claim), v2 (symmetric verbs excluded) and v3 (v2 + negation).

Usage: set -a; source configs/env.sh; set +a
       python -m judges.cue_claim_judge --judge claude-opus-5-5   (and gpt-6-astra) -> outputs/e16/<judge>.jsonl
       python -m judges.cue_claim_judge --analyze                 -> outputs/bundle/cue_judge.json
"""
from __future__ import annotations
from srww import paths as _vtp  # noqa: E402

import argparse
import json
import random
import re
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from srww.records import pe, rj
from srww.claim_parse import first_claim_v2, first_claim_v3
from srww.claim_parse import first_claim, split_sents
from srww.cue_glm import QRX, geom_meta

O = Path(f"{_vtp.OUTPUTS}")
B = Path("outputs/bundle")
M7 = ["qwen3b", "qwen7b", "internvl3_8b", "phi35v", "molmo7b", "qwen3vl_8b", "internvl35_8b"]
COT_PREFIX = "Let's think step by step."
PROMPT = """Below is a model's step-by-step reasoning about an image that shows two shapes: "{x}" and "{y}".

Find the FIRST sentence in the reasoning that states which of the two shapes is in front of the other (on top of it, covering, occluding or hiding part of it). Rules:
- A sentence that only says the shapes overlap, touch or intersect, without saying which one is on top, does NOT count.
- A negated statement counts for the other shape (e.g. "the {x} is not in front of the {y}" means the {y} is in front).
- Questions, hypotheses ("if ..."), restatements of the task, and statements about left/right/above/below or size only do NOT count.

Reasoning:
<<<
{r}
>>>

Reply with exactly one line: FRONT: <shape name, exactly "{x}" or "{y}"> or FRONT: NONE"""


def sample(seed: int = 0, per: int = 15, exclude: set | None = None) -> list[dict]:
    rng = random.Random(seed)
    exclude = exclude or set()
    out = []
    for m in M7:
        recs = [r for r in rj(O / "e10" / f"{m}_cues.jsonl") if r["condition"] == "cot" and r["task"].startswith("occ")]
        recs = [r for r in recs if (m, r["item_id"]) not in exclude]
        wrong = [r for r in recs if pe(r) != r["gold"]]
        right = [r for r in recs if pe(r) == r["gold"]]
        for grp, pool in (("cot_wrong", wrong), ("cot_right", right)):
            for r in rng.sample(pool, per):
                qm = QRX.search(r["question"])
                x, y = qm.group(1).lower(), qm.group(3).lower()
                g = geom_meta(r["meta"], x)
                sents = split_sents(r["text"][len(COT_PREFIX):])
                rules = {k: (f(sents) or (None, None))[1] for k, f in (("v1", first_claim), ("v2", first_claim_v2), ("v3", first_claim_v3))}
                out.append({"model": m, "key": r["item_id"], "group": grp, "x": x, "y": y, "phrasing": qm.group(2),
                            "front": g["front"], "lower": g["lower"], "larger": g["larger"], "pair": r["image_id"],
                            "text": r["text"][len(COT_PREFIX):].strip(), **{f"rule_{k}": v for k, v in rules.items()}})
    return out


def sample2() -> list[dict]:
    """Second stratified sample (25 + 25 per model), disjoint from the first."""
    first = {(r["model"], r["key"]) for r in sample()}
    return sample(seed=1, per=25, exclude=first)


def run_judge(judge: str, second: bool = False) -> None:
    from srww import api_client as api_client
    items = sample2() if second else sample()
    lock, res = threading.Lock(), []

    def go(c):
        r = api_client.chat(judge, [{"role": "user", "content": PROMPT.format(x=c["x"], y=c["y"], r=c["text"])}], max_tokens=2048, effort="high")
        txt = (r.get("text") or "").strip()
        m = re.search(r"FRONT:\s*\**\s*([a-z ]+)", txt, re.I)
        lab = m.group(1).strip().lower() if m else None
        lab = lab if lab in (c["x"], c["y"]) else ("none" if lab and lab.startswith("none") else None)
        with lock:
            res.append({k: v for k, v in c.items() if k != "text"} | {"judge": judge, "label": lab, "raw": (txt or r.get("error", ""))[:120]})

    with ThreadPoolExecutor(6) as ex:
        list(ex.map(go, items))
    (O / "e16").mkdir(exist_ok=True)
    with open(O / "e16" / f"{judge}{'_s2' if second else ''}.jsonl", "w") as f:
        for r in res:
            f.write(json.dumps(r) + "\n")
    print(f"[e16] {judge}: {len(res)} labels, unparsed {sum(r['label'] is None for r in res)}")


def kappa(a: list, b: list) -> float:
    cats = sorted(set(a) | set(b))
    po = np.mean([x == y for x, y in zip(a, b)])
    pe_ = sum((a.count(c) / len(a)) * (b.count(c) / len(b)) for c in cats)
    return float((po - pe_) / (1 - pe_)) if pe_ < 1 else 1.0


def analyze() -> None:
    J = {j: {r["key"] + r["model"]: r for r in rj(O / "e16" / f"{j}.jsonl")} for j in ("claude-opus-5-5", "gpt-6-astra")}
    keys = sorted(set(J["claude-opus-5-5"]) & set(J["gpt-6-astra"]))
    rows = [J["claude-opus-5-5"][k] | {"label_opus": J["claude-opus-5-5"][k]["label"], "label_gpt": J["gpt-6-astra"][k]["label"]} for k in keys]
    rel = lambda v, r: "none" if v is None or v == "none" else ("x" if v == r["x"] else "y")
    res = {"n": len(rows)}
    res["judge_vs_judge"] = {"agree": round(float(np.mean([rel(r["label_opus"], r) == rel(r["label_gpt"], r) for r in rows])), 3),
                             "kappa": round(kappa([rel(r["label_opus"], r) for r in rows], [rel(r["label_gpt"], r) for r in rows]), 3)}
    cons = [r for r in rows if rel(r["label_opus"], r) == rel(r["label_gpt"], r)]
    res["n_consensus"] = len(cons)
    for v in ("v1", "v2", "v3"):
        a = [rel(r[f"rule_{v}"], r) for r in cons]
        b = [rel(r["label_opus"], r) for r in cons]
        both = [(x, y) for x, y in zip(a, b) if x != "none" and y != "none"]
        res[f"rule_{v}_vs_consensus"] = {"agree_3way": round(float(np.mean([x == y for x, y in zip(a, b)])), 3), "kappa_3way": round(kappa(a, b), 3),
                                         "agree_when_both_claim": round(float(np.mean([x == y for x, y in both])), 3) if both else None,
                                         "n_both": len(both), "rule_none": a.count("none"), "judge_none": b.count("none")}
    # judge-based shares (consensus, claims only), per model and pooled
    def shares(rs):
        c = [r for r in rs if rel(r["label_opus"], r) != "none"]
        if not c:
            return None
        nm = lambda r: r["label_opus"]
        aff = lambda r: (nm(r) == r["x"]) == (r["phrasing"] == "in front of")
        return {"n": len(c), "true_front": round(100 * np.mean([nm(r) == r["front"] for r in c]), 1),
                "lower": round(100 * np.mean([nm(r) == r["lower"] for r in c]), 1),
                "larger": round(100 * np.mean([nm(r) == r["larger"] for r in c]), 1),
                "affirm": round(100 * np.mean([aff(r) for r in c]), 1),
                "affirm_cot_wrong": round(100 * np.mean([aff(r) for r in c if r["group"] == "cot_wrong"]), 1) if any(r["group"] == "cot_wrong" for r in c) else None}
    res["judge_shares_pooled"] = shares(cons)
    res["judge_shares_by_model"] = {m: shares([r for r in cons if r["model"] == m]) for m in M7}
    def rule_shares(rs, v):
        c = [r for r in rs if r[f"rule_{v}"]]
        aff = lambda r: (r[f"rule_{v}"] == r["x"]) == (r["phrasing"] == "in front of")
        return {"n": len(c), "true_front": round(100 * np.mean([r[f"rule_{v}"] == r["front"] for r in c]), 1),
                "affirm": round(100 * np.mean([aff(r) for r in c]), 1), "lower": round(100 * np.mean([r[f"rule_{v}"] == r["lower"] for r in c]), 1)}
    res["rule_shares_same_sample"] = {v: rule_shares(cons, v) for v in ("v1", "v2", "v3")}
    json.dump(res, open(B / "cue_judge.json", "w"), indent=1)
    print(json.dumps(res, indent=1))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--judge")
    ap.add_argument("--analyze", action="store_true")
    ap.add_argument("--second", action="store_true")
    a = ap.parse_args()
    analyze() if a.analyze else run_judge(a.judge, a.second)


if __name__ == "__main__":
    main()
