"""Exp-62 judges via api_client (key from OPENAI_COMPAT_API_KEY, never printed).

  obj   vision, stimulus filter: is each COCO cutout (rendered alone on gray at its pasted resolution) clearly
        recognisable as its category name?                           -> stimuli/judge_obj_<judge>.jsonl
  bg    vision, stimulus filter: does the background crop contain a prominent object?  -> stimuli/judge_bg_<judge>.jsonl
  rel   text only (no image, no gold, conclusion removed): which of the two objects does the LAST relational depth
        statement in R_cut put closer to the camera?  Fixed per-model sample (seed 62, 120 items per phrasing);
        deterministic prefilter (no depth word -> NONE, no call); identical (objects, body) judged once.
                                                                     -> outputs/e62/judge_rel_<judge>.jsonl

Usage: source configs/env.sh
       python -m judges.coco_twins_judge obj --judge claude-opus-5-5
       python -m judges.coco_twins_judge rel-build && python -m judges.coco_twins_judge rel --judge gpt-6-astra --threads 16
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import re
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from PIL import Image

from data_gen import coco_occlusion_twins as S

O = S.SHARE / "code/vtax/outputs/e62"
JUDGES = ("claude-opus-5-5", "gpt-6-astra")
M8 = ["qwen3b", "qwen7b", "internvl3_8b", "phi35v", "molmo7b", "qwen3vl_8b", "internvl35_8b", "qwen3vl_32b"]
PER_PHRASING = 120

PROMPT_OBJ = """The image shows a single object that was cut out of a photograph and placed on a plain gray background.

Would a typical viewer, without any hint, recognise this object as a "{cat}"? Answer NO if the object is too small, too dark, too blurry, only a fragment, or looks more like something else.

Reply with exactly one line: OBJ: YES or OBJ: NO"""

PROMPT_BG = """Look at this photograph. It will be used as a background behind two other objects.

Does it contain any prominent, distinct object that draws attention, for example an animal, a person, a vehicle, a food item, a piece of furniture, an appliance, a sign with large text, or any other object occupying more than about 5% of the image? Scenery such as sky, water, grass, sand, snow, walls, floors, roads, fences, trees or distant buildings does not count.

Reply with exactly one line: BG: CLEAN (no prominent object) or BG: OBJECT"""

PROMPT_REL = """Below is a short text written about a photograph that shows two objects: FIRST = "{a}" and SECOND = "{b}".

Text:
\"\"\"
{r}
\"\"\"

Find the LAST statement in the text that implies which of the two objects is closer to the camera. It may say so directly (e.g. "the {a} is closer than the {b}", "the {b} is farther away") or through occlusion or layering (e.g. "the {a} is in front of the {b}", "the {a} partially covers / blocks / overlaps the {b}", "the {b} is behind the {a}"). Report which object that statement puts closer to the camera. If the text contains no such statement, answer NONE.

Reply with exactly one line: REL: FIRST or REL: SECOND or REL: NONE"""

DEPTH_RX = re.compile(r"front|behind|closer|nearer|near\b|farther|further|far\b|away|distan|depth|occlu|overlap|cover|"
                      r"block|obscur|hidden|hide|foreground|background|on top|in back|layer", re.I)


class Limiter:
    """Global requests-per-minute limiter."""

    def __init__(self, rpm: float):
        self.dt, self.t, self.lock = 60.0 / rpm, time.time(), threading.Lock()

    def wait(self) -> None:
        with self.lock:
            now = time.time()
            self.t = max(self.t + self.dt, now)
            d = self.t - now
        if d > 0:
            time.sleep(d)


def parse(kind: str, raw: str) -> str | None:
    rx = {"obj": r"OBJ:\s*(YES|NO)", "bg": r"BG:\s*(CLEAN|OBJECT)", "rel": r"REL:\s*(FIRST|SECOND|NONE)"}[kind]
    m = re.findall(rx, raw.upper())
    return m[-1] if m else None


def run(kind: str, jname: str, tasks: list[dict], fn: Path, threads: int, rpm: float) -> None:
    """Resumable judge loop over tasks (each with 'tid' and 'messages')."""
    from srww import api_client as api_client
    done = {r["tid"] for r in S_read(fn) if r.get("label") is not None}
    todo = [t for t in tasks if t["tid"] not in done]
    print(f"[e62] {kind} {jname}: {len(todo)} to judge ({len(done)} done)", flush=True)
    lim, lock, cnt, t0 = Limiter(rpm), threading.Lock(), Counter(), time.time()
    f = open(fn, "a")

    def go(t):
        lim.wait()
        out = api_client.chat(jname, t["messages"], max_tokens=2048, effort="high")
        raw = (out.get("text") or "").strip()
        err = out.get("error", "")[:200]
        lab = parse(kind, raw) if not err else None
        with lock:
            f.write(json.dumps({"tid": t["tid"], "judge": jname, "label": lab, "raw": raw[-200:], "error": err},
                               ensure_ascii=False) + "\n")
            f.flush()
            cnt["err" if err else ("unlab" if lab is None else lab)] += 1
            s = sum(cnt.values())
            if s % 100 == 0:
                print(f"[e62] {kind} {jname}: {s}/{len(todo)} {dict(cnt)} {time.time() - t0:.0f}s", flush=True)

    with ThreadPoolExecutor(threads) as ex:
        list(ex.map(go, todo))
    f.close()
    print(f"[e62] {kind} {jname}: finished {dict(cnt)} {time.time() - t0:.0f}s", flush=True)


def S_read(p: Path) -> list[dict]:
    return [json.loads(l) for l in open(p)] if p.exists() else []


# ============================================================================= stimulus filters

def obj_tasks() -> list[dict]:
    from srww import api_client as api_client
    c = json.load(open(S.ST / "candidates.json"))
    out = []
    for o in c["objects"]:
        im, _ = S.scaled(S.cutout(o), 0.05 * S.W * S.H)  # typical pasted area
        bg = Image.new("RGB", (max(320, im.width + 40), max(320, im.height + 40)), (128, 128, 128))
        bg.paste(im, ((bg.width - im.width) // 2, (bg.height - im.height) // 2), im)
        out.append({"tid": str(o["ann_id"]), "messages": [{"role": "user", "content": [
            api_client.img_part(bg), {"type": "text", "text": PROMPT_OBJ.format(cat=o["cat"])}]}]})
    return out


def bg_tasks() -> list[dict]:
    from srww import api_client as api_client
    c = json.load(open(S.ST / "candidates.json"))
    return [{"tid": b["file"], "messages": [{"role": "user", "content": [
        api_client.img_part(S.bg_image(b)), {"type": "text", "text": PROMPT_BG}]}]} for b in c["backgrounds"]]


# ============================================================================= written relation

def rel_key(a: str, b: str, body: str) -> str:
    return hashlib.sha1(f"{a}\x00{b}\x00{body}".encode()).hexdigest()[:16]


def rel_sample(items: list[dict]) -> set[str]:
    """Fixed item sample (same uids for every model): PER_PHRASING per phrasing, seed 62."""
    out = set()
    for ph in ("closer", "farther", "which"):
        us = sorted(it["uid"] for it in items if it["phrasing"] == ph)
        out |= set(random.Random(62).sample(us, min(PER_PHRASING, len(us))))
    return out


def rel_build() -> None:
    items = {it["uid"]: it for it in S_read(S.ST / "items_final.jsonl")}
    samp = rel_sample(list(items.values()))
    tasks, rows = {}, []
    for m in M8:
        recs = [r for r in S_read(O / f"{m}.jsonl") if r["arm"] == "cot" and r["uid"] in samp]
        for r in recs:
            it = items[r["uid"]]
            a, b = it["meta"]["large"]["cat"], it["meta"]["small"]["cat"]
            body = r["rcut"]
            pre = DEPTH_RX.search(body.replace(S.POD_ST, "")) is None
            k = rel_key(a, b, body)
            rows.append({"model": m, "uid": r["uid"], "tid": k, "prefilter_none": pre, "first": a, "second": b})
            if not pre:
                tasks.setdefault(k, {"tid": k, "a": a, "b": b, "body": body})
    json.dump({"rows": rows, "n_tasks": len(tasks)}, open(O / "rel_rows.json", "w"))
    with open(O / "rel_tasks.jsonl", "w") as f:
        for t in tasks.values():
            f.write(json.dumps(t) + "\n")
    print(f"[e62] rel: {len(rows)} rows ({Counter(r['model'] for r in rows)}), {sum(r['prefilter_none'] for r in rows)} "
          f"prefiltered NONE, {len(tasks)} unique texts to judge")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("phase", choices=["obj", "bg", "rel-build", "rel"])
    ap.add_argument("--judge", choices=JUDGES, default="claude-opus-5-5")
    ap.add_argument("--threads", type=int, default=12)
    ap.add_argument("--rpm", type=float, default=60)
    ap.add_argument("--limit", type=int)
    a = ap.parse_args()
    if a.phase == "rel-build":
        rel_build()
        return
    if a.phase == "obj":
        ts, fn = obj_tasks(), S.ST / f"judge_obj_{a.judge}.jsonl"
    elif a.phase == "bg":
        ts, fn = bg_tasks(), S.ST / f"judge_bg_{a.judge}.jsonl"
    else:
        ts = [{"tid": t["tid"], "messages": [{"role": "user", "content": PROMPT_REL.format(a=t["a"], b=t["b"], r=t["body"])}]}
              for t in S_read(O / "rel_tasks.jsonl")]
        fn = O / f"judge_rel_{a.judge}.jsonl"
    run(a.phase, a.judge, ts[: a.limit] if a.limit else ts, fn, a.threads, a.rpm)


if __name__ == "__main__":
    main()
