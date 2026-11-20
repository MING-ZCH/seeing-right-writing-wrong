"""Exp-38 (part 1): two text-only judges extract the LAST written depth / occlusion relation from closed-model CoT.

Sources (closed / API models, written CoT condition only):
  e9h       hard occlusion set, 576 prompts (both phrasings) per model
  e9x       original occlusion twins (occ_front / occ_behind), 800 prompts per model
  e9c_cues  cue-factorial occlusion set, DeepSeek-V4.1-Flash only, 1,600 prompts
Rows may repeat after API errors: keep the last row with a non-empty pred per (condition, item_id, phrasing).

The final answer line(s) are removed before judging (text from the last "Answer:" on, plus trailing sentences that only
restate the chosen option). Each judge (Claude Opus 5.5, GPT-6-Astra; api_client client, max_tokens 2048, effort high, as in
e16_cue_judge.py / e30_judge.py) reads the remaining text and the two shape names (no image) and names the shape the
LAST depth / occlusion claim puts in front, or NONE / UNCLEAR.
The prompt calls the CoT 'a short text that describes an image': the Claude endpoint returns finish_reason=content_filter
(empty reply) on 17/24 pilot items when the prompt says 'a model's reasoning'.
-> outputs/e38/judge_<judge>.jsonl (resumable; key = set|model|item_id)
--first: same judges / prompt asking for the FIRST such statement, hard set + cue set only -> judge_<judge>_first.jsonl

Usage: source configs/env.sh
       python e38_closed_judge.py --judge claude-opus-5-5 [--threads 12] [--limit 20]
"""
from __future__ import annotations
import vtax_paths as _vtp  # noqa: E402

import argparse
import json
import random
import re
import threading
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

O = Path(f"{_vtp.OUTPUTS}")
OUT = O / "e38"
CLOSED = ["claude-opus-5-5", "gpt-6-astra", "gemini-4-argon", "qwen3.8-max-0902", "hy4", "deepseek-v4.1-flash"]
SETS = {"e9h": CLOSED, "e9x": CLOSED, "e9c_cues": ["deepseek-v4.1-flash"]}
JUDGES = ("claude-opus-5-5", "gpt-6-astra")

PROMPT = """Below is a short text that describes an image showing two shapes: "{x}" and "{y}". The text's final answer line has been removed.

Find the LAST statement in the text that says which of the two shapes is in front of the other: on top of it, covering, occluding, overlapping over or hiding part of it, closer to the viewer, or equivalently that the other shape is behind, underneath, covered, hidden or farther away. Rules:
- Resolve pronouns and short descriptions ("it", "the circle", "the left shape") to the two shapes.
- A negated statement counts for the other shape (e.g. "the {x} is not in front of the {y}" means the {y} is in front).
- A statement that only says the shapes overlap, touch or intersect, without saying which one is on top, does NOT count.
- Questions, hypotheses ("if ..."), restatements of the task, and alternatives that the text considers and rejects do NOT count. Statements about left/right/above/below or size alone do NOT count.
- A hedged statement ("appears to be", "probably") still counts.
- A sentence that only gives the answer choice (yes / no / an option letter) without naming which shape is in front does NOT count.
- If the last such statement is self-contradictory or you cannot tell which shape it puts in front, reply UNCLEAR.

Text:
\"\"\"
{r}
\"\"\"

Reply with exactly one line: FRONT: <shape name, exactly "{x}" or "{y}"> or FRONT: NONE or FRONT: UNCLEAR"""

PROMPT_FIRST = (PROMPT.replace("Find the LAST statement", "Find the FIRST statement")
                .replace("If the last such statement", "If the first such statement"))
FIRST_SETS = ("e9h", "e9c_cues")

SENT = re.compile(r"(?<=[.!?])\s+|\n+")
RESTATE = re.compile(r"\b(answer|option|choice)\b|^\W*\(?[AB]\)?\W*(yes|no)?\W*$|^\W*(yes|no)\b", re.I)
RELWORD = re.compile(r"front|behind|cover|on top|occlu|hidden|hide|closer|nearer|farther|underneath|beneath|obscur|block", re.I)


def rj(p: Path) -> list[dict]:
    return [json.loads(l) for l in open(p)] if p.exists() else []


def dedup(rows: list[dict]) -> dict:
    """Last row with a non-empty pred per (condition, item_id, phrasing)."""
    out = {}
    for r in rows:
        if r.get("pred"):
            out[(r["condition"], r["item_id"], r.get("phrasing"))] = r
    return out


def strip_answer(text: str) -> tuple[str, int]:
    """Remove the final 'Answer:' line(s) and trailing sentences that only restate the chosen option."""
    t = text or ""
    k = t.rfind("Answer:")
    if k < 0:
        m = list(re.finditer(r"answer\s*[:\uff1a]", t, re.I))
        k = m[-1].start() if m else -1
    if k >= 0:
        t = t[:k]
    t = t.rstrip(" \n*#-")
    parts = [p for p in SENT.split(t) if p and p.strip()]
    n_drop = 0
    while parts:
        s = parts[-1].strip(" *#-")
        if s and RESTATE.search(s) and not RELWORD.search(s):
            parts.pop()
            n_drop += 1
        else:
            break
    if n_drop:
        # cut the original text at the start of the first dropped sentence to keep formatting
        last = parts[-1] if parts else ""
        cut = t.rfind(last) + len(last) if last else 0
        t = t[:cut]
    return t.strip(), n_drop


def cot_items() -> list[dict]:
    items = []
    for st, models in SETS.items():
        for m in models:
            rows = rj(O / st / f"{m}.jsonl")
            for (cond, iid, ph), r in dedup(rows).items():
                if cond != "cot" or not r["task"].startswith("occ"):
                    continue
                mt = r["meta"]
                body, nd = strip_answer(r.get("text") or "")
                items.append({"key": f"{st}|{m}|{iid}", "set": st, "model": m, "item_id": iid, "phrasing": ph,
                              "x": mt["x_name"].lower(), "y": mt["y_name"].lower(), "front": mt["front_name"].lower(),
                              "body": body, "n_restate_dropped": nd})
    return items


def parse_label(txt: str, x: str, y: str) -> str | None:
    m = list(re.finditer(r"FRONT:\s*\**\s*\"?([A-Za-z ]+)", txt or "", re.I))
    if not m:
        return None
    lab = m[-1].group(1).strip().lower()
    for nm in (x, y):
        if lab.startswith(nm):
            return nm
    if lab.startswith("none"):
        return "none"
    if lab.startswith("unclear"):
        return "unclear"
    return None


def run(judge: str, threads: int, limit: int | None, reverse: bool = False, first: bool = False, priority: bool = False,
        per_model: int | None = None) -> None:
    import api_client
    OUT.mkdir(exist_ok=True)
    fn = OUT / f"judge_{judge}{'_first' if first else ''}.jsonl"
    done = {r["key"] for r in rj(fn) if r.get("label") is not None}
    items = [c for c in cot_items() if c["key"] not in done and (not first or c["set"] in FIRST_SETS)]
    prompt = PROMPT_FIRST if first else PROMPT
    if per_model:
        # fixed random subsample per set x model (seed 0), drawn from all CoT items so reruns pick the same keys
        keep = set()
        allk = defaultdict(list)
        for c in cot_items():
            allk[(c["set"], c["model"])].append(c["key"])
        for ks in allk.values():
            keep |= set(random.Random(0).sample(sorted(ks), min(per_model, len(ks))))
        items = [c for c in items if c["key"] in keep]
    if priority:
        # informative cells first; ceiling-model twins last and shuffled, so a stop leaves a random subsample per model
        rank = lambda c: 0 if c["set"] == "e9h" else 1 if c["set"] == "e9c_cues" else 2 if c["model"] == "deepseek-v4.1-flash" else 3
        rnd = random.Random(1)
        items = sorted(((rank(c), rnd.random() if rank(c) == 3 else 0, i, c) for i, c in enumerate(items)), key=lambda t: t[:3])
        items = [t[3] for t in items]
    if reverse:
        items = items[::-1]
    if limit:
        items = items[:limit]
    print(f"[e38] {judge}: {len(items)} to judge ({len(done)} done)", flush=True)
    lock = threading.Lock()
    t0, cnt = time.time(), [0, 0]

    def go(c):
        if not c["body"].strip():
            lab, raw, err = "none", "(empty after strip)", ""
        else:
            r = api_client.chat(judge, [{"role": "user", "content": prompt.format(x=c["x"], y=c["y"], r=c["body"])}],
                           max_tokens=2048, effort="high")
            raw = (r.get("text") or "").strip()
            err = r.get("error", "")[:200]
            lab = parse_label(raw, c["x"], c["y"]) if not err else None
        rec = {k: c[k] for k in ("key", "set", "model", "item_id", "phrasing", "x", "y", "front", "n_restate_dropped")}
        rec |= {"judge": judge, "label": lab, "raw": raw[-300:], "error": err, "body_chars": len(c["body"])}
        with lock:
            with open(fn, "a") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            cnt[0] += 1
            cnt[1] += lab is None
            if cnt[0] % 200 == 0:
                print(f"[e38] {judge}: {cnt[0]}/{len(items)} ({cnt[1]} unlabelled) {time.time() - t0:.0f}s", flush=True)

    with ThreadPoolExecutor(threads) as ex:
        list(ex.map(go, items))
    print(f"[e38] {judge}: finished {cnt[0]} ({cnt[1]} unlabelled) {time.time() - t0:.0f}s", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--judge", required=True, choices=JUDGES)
    ap.add_argument("--threads", type=int, default=12)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--priority", action="store_true", help="hard set, cue set, DeepSeek twins first; other twins shuffled last")
    ap.add_argument("--per-model", type=int, default=None, help="fixed random subsample of N CoT texts per set x model")
    ap.add_argument("--first", action="store_true", help="FIRST instead of LAST claim (hard set + cue set only)")
    ap.add_argument("--reverse", action="store_true", help="judge in reverse order (second process on another key)")
    a = ap.parse_args()
    run(a.judge, a.threads, a.limit, a.reverse, a.first, a.priority, a.per_model)


if __name__ == "__main__":
    main()
