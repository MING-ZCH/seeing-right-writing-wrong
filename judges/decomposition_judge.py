"""Exp-54 judge calls (text only): subtype labels for the D+C- / D-C+ decomposition.

Written relation = Exp-45 two-judge consensus on the conclusion-removed body (reused, no new calls). New calls only:
  basis  items whose consensus relation is FALSE and D+C- (writing wrong) or TRUE and D-C+ (writing right):
         the main reason the text gives for its last relation (OCCLUSION / VERTICAL / SIZE / QUESTION / OTHER).
  step   items whose consensus relation is TRUE and D+C- (answering wrong) or FALSE and D-C+ (answering right):
         grammatical form of the last body relation (subject, FRONT/BEHIND, negated) and the answer stated by the
         removed conclusion (YES / NO / NONE).
Judges: Claude Opus 5.5 + GPT-6-Astra (effort high, max_tokens 2048, Exp-38/45 content-filter-safe framing).
Outputs: shared outputs/e54/tasks.jsonl, outputs/e54/judge_<judge>.jsonl (resumable, keyed by task id).

Usage: source configs/env.sh
       python -m judges.decomposition_judge build
       python -m judges.decomposition_judge judge --judge gpt-6-astra --threads 8 --rpm 55 [--key alt] [--shard 0/2]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor

from experiments.two_channel.uniform_analysis import load_labels, shape_records
from experiments.two_channel.uniform_two_channel import E45, JUDGES, M7, O, rj

E54 = O / "e54"

PROMPT_BASIS = """Below is a short text that describes an image showing two shapes: "{x}" and "{y}". The text was written to answer the question: "{q}". The text's final answer and concluding sentences have been removed.

Text:
\"\"\"
{r}
\"\"\"

The LAST statement in the text about which of the two shapes is in front of the other puts the "{f}" in front. What reason does the text give for putting the "{f}" in front (in that statement or in the sentences leading up to it)? Choose the ONE main reason:
- OCCLUSION: overlap or layering: one shape covers, hides, overlaps or is drawn over part of the other; outline completeness; visible or hidden parts (including "on top of" when it means layered over the other shape).
- VERTICAL: vertical position in the picture: one shape is lower or higher, above or below, nearer the bottom or the top of the image (including "on top" when it means higher up in the picture).
- SIZE: one shape is larger, bigger or more prominent, so it is closer.
- QUESTION: no visual reason is given: the text only asserts the relation, restates the question's wording as a fact, or says the shape is "described as" in front.
- OTHER: any other reason (color, brightness, shading, perspective, left/right position, a depth word without a stated cue), or you cannot tell.

Reply with exactly one line: BASIS: OCCLUSION or BASIS: VERTICAL or BASIS: SIZE or BASIS: QUESTION or BASIS: OTHER"""

PROMPT_STEP = """Below is a short text that describes an image showing two shapes: "{x}" and "{y}". The text was written to answer the question: "{q}" with the options (A) yes and (B) no. The text is split into its BODY and its CONCLUSION (the final answer and concluding sentences; it may be empty).

BODY:
\"\"\"
{r}
\"\"\"

CONCLUSION:
\"\"\"
{c}
\"\"\"

1. The LAST statement in the BODY about which of the two shapes is in front of or behind the other puts the "{f}" in front. Report its grammatical form: which shape is the grammatical subject of that statement; whether its relation word is of the FRONT kind (in front of, on top of, covering, over, closer, hiding, occluding) or the BEHIND kind (behind, under, beneath, covered by, hidden by, partly hidden, farther); and whether the statement is NEGATED ("is not in front of") or PLAIN.
2. Which answer to the question does the CONCLUSION state or imply: YES, NO, or NONE (empty, or no answer given)?

Reply with exactly two lines:
FORM: <subject shape, exactly "{x}" or "{y}"> | <FRONT or BEHIND> | <NEGATED or PLAIN>
CONCLUSION: YES or NO or NONE"""

BASIS_LABELS = ("OCCLUSION", "VERTICAL", "SIZE", "QUESTION", "OTHER")
REF_N = 100


def question(r: dict) -> str:
    rel = "in front of" if r["phrasing"] == "front" else "behind"
    return f"Is the {r['x']} {rel} the {r['y']}?"


def removed_sents(sents: list[str], trunc: list[str]) -> list[str]:
    """Sentences of the full rationale that the Exp-35 rule removed (order-preserving multiset difference)."""
    out, j = [], 0
    for s in sents:
        if j < len(trunc) and s == trunc[j]:
            j += 1
        else:
            out.append(s)
    return out


def open_records() -> tuple[list[dict], list[dict]]:
    rows = [r for r in rj(E45 / "items.jsonl") if r["group"] == "open" and r["model"] in M7]
    recs = shape_records(rows, load_labels())
    return rows, recs


def build() -> None:
    E54.mkdir(exist_ok=True)
    rows, recs = open_records()
    tasks, seen = [], set()
    for row, r in zip(rows, recs):
        if not r["has"]:
            continue
        dpcm = r["d_ok"] and not r["c_ok"]
        dmcp = (not r["d_ok"]) and r["c_ok"]
        if not (dpcm or dmcp):
            continue
        kind = "basis" if (dpcm and not r["true"]) or (dmcp and r["true"]) else "step"
        conc = " ".join(removed_sents(row["sents"], row["trunc"]))
        q = question(row)
        prompt = (PROMPT_BASIS if kind == "basis" else PROMPT_STEP).format(x=row["x"], y=row["y"], q=q, r=row["body"],
                                                                          f=r["claim"], c=conc)
        tid = hashlib.sha1(f"{kind}\x00{prompt}".encode()).hexdigest()[:20]
        if tid in seen:
            continue
        seen.add(tid)
        tasks.append({"tid": tid, "kind": kind, "x": row["x"], "y": row["y"], "prompt": prompt,
                      "first": f"{row['model']}|{row['ds']}|{row['item']}"})
    # reference sample for the step labels: 100 D+C+ items with a TRUE relation per model (seed 54), to give the
    # base rate of question forms (restated / converse / other shape) among items that are answered right
    for m in M7:
        pool = [(row, r) for row, r in zip(rows, recs) if r["model"] == m and r["has"] and r["true"]
                and r["d_ok"] and r["c_ok"]]
        pool.sort(key=lambda t: (t[0]["ds"], str(t[0]["item"])))
        for row, r in random.Random(54).sample(pool, min(REF_N, len(pool))):
            conc = " ".join(removed_sents(row["sents"], row["trunc"]))
            prompt = PROMPT_STEP.format(x=row["x"], y=row["y"], q=question(row), r=row["body"], f=r["claim"], c=conc)
            tid = hashlib.sha1(f"step\x00{prompt}".encode()).hexdigest()[:20]
            if tid not in seen:
                seen.add(tid)
                tasks.append({"tid": tid, "kind": "step", "x": row["x"], "y": row["y"], "prompt": prompt,
                              "first": f"{row['model']}|{row['ds']}|{row['item']}", "ref": True})
    with open(E54 / "tasks.jsonl", "w") as f:
        for t in tasks:
            f.write(json.dumps(t) + "\n")
    print(f"[e54] tasks {len(tasks)}: {Counter(t['kind'] for t in tasks)}")


def parse(kind: str, txt: str, x: str, y: str) -> dict | None:
    txt = (txt or "").replace("*", "")
    if kind == "basis":
        m = list(re.finditer(r"BASIS:\s*([A-Z]+)", txt, re.I))
        if not m or m[-1].group(1).upper() not in BASIS_LABELS:
            return None
        return {"basis": m[-1].group(1).upper()}
    fm = list(re.finditer(r"FORM:\s*\"?([A-Za-z ]+?)\"?\s*\|\s*(FRONT|BEHIND)\s*\|\s*(NEGATED|PLAIN)", txt, re.I))
    cm = list(re.finditer(r"CONCLUSION:\s*(YES|NO|NONE)", txt, re.I))
    if not fm or not cm:
        return None
    subj = fm[-1].group(1).strip().lower()
    subj = x if subj.startswith(x) else y if subj.startswith(y) else None
    if subj is None:
        return None
    return {"subj": subj, "pred": fm[-1].group(2).upper(), "neg": fm[-1].group(3).upper() == "NEGATED",
            "conc": cm[-1].group(1).upper()}


class Limiter:
    def __init__(self, rpm: float):
        self.dt, self.t, self.lock = 60.0 / rpm, time.time(), threading.Lock()

    def wait(self) -> None:
        with self.lock:
            now = time.time()
            self.t = max(self.t + self.dt, now)
            d = self.t - now
        if d > 0:
            time.sleep(d)


def judge(jname: str, threads: int, rpm: float, key: str, shard: str, limit: int | None) -> None:
    from srww import api_client as api_client
    if key == "alt":
        os.environ["OPENAI_COMPAT_API_KEY"] = os.environ["OPENAI_COMPAT_API_KEY_ALT"]
    tasks = rj(E54 / "tasks.jsonl")
    fn = E54 / f"judge_{jname}.jsonl"
    done = {r["tid"] for r in rj(fn) if r.get("label") is not None}
    todo = [t for t in tasks if t["tid"] not in done]
    if shard:
        i, n = map(int, shard.split("/"))
        todo = [t for t in todo if int(t["tid"], 16) % n == i]
    if limit:
        todo = todo[:limit]
    print(f"[e54] {jname} key={key} shard={shard or '-'}: {len(todo)} to judge ({len(done)} done)", flush=True)
    lim, lock, cnt, t0 = Limiter(rpm), threading.Lock(), Counter(), time.time()
    f = open(fn, "a")

    def go(t):
        lim.wait()
        out = api_client.chat(jname, [{"role": "user", "content": t["prompt"]}], max_tokens=2048, effort="high")
        raw = (out.get("text") or "").strip()
        err = out.get("error", "")[:200]
        lab = parse(t["kind"], raw, t["x"], t["y"]) if not err else None
        rec = {"tid": t["tid"], "kind": t["kind"], "judge": jname, "label": lab, "raw": raw[-300:], "error": err,
               "finish": out.get("finish_reason")}
        with lock:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            f.flush()
            cnt["err" if err else ("unlab" if lab is None else "ok")] += 1
            s = sum(cnt.values())
            if s % 200 == 0:
                print(f"[e54] {jname}: {s}/{len(todo)} {dict(cnt)} {time.time() - t0:.0f}s "
                      f"({60 * s / (time.time() - t0):.0f}/min)", flush=True)

    with ThreadPoolExecutor(threads) as ex:
        list(ex.map(go, todo))
    f.close()
    print(f"[e54] {jname}: finished {dict(cnt)} {time.time() - t0:.0f}s", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("phase", choices=["build", "judge"])
    ap.add_argument("--judge", choices=JUDGES)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--rpm", type=float, default=50)
    ap.add_argument("--key", default="main", choices=["main", "alt"])
    ap.add_argument("--shard", default="")
    ap.add_argument("--limit", type=int)
    a = ap.parse_args()
    if a.phase == "build":
        build()
    else:
        judge(a.judge, a.threads, a.rpm, a.key, a.shard, a.limit)


if __name__ == "__main__":
    main()
