"""E60 judges via api_client (Claude Opus 5.5 + GPT-6-Astra).

  text  vision: does the ORIGINAL image contain legible text? (item filter, one call per candidate image and judge)
  rel   text only: horizontal relation of X to Y implied by the LAST such statement in R_cut (no image, no gold).
        Deterministic prefilter (no left/right word -> NONE, no call); identical (X, Y, R_cut) judged once.

Outputs (resumable): outputs/e60/judge_text_<judge>.jsonl (uid), outputs/e60/rel_tasks.jsonl,
outputs/e60/judge_rel_<judge>.jsonl (tid).

Usage: source configs/env.sh
       python -m judges.mirrored_cvbench_judge text --judge claude-opus-5-5 --threads 8
       python -m judges.mirrored_cvbench_judge rel-build && python -m judges.mirrored_cvbench_judge rel --judge gpt-6-astra --threads 8
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

from srww import vllm_utils as vx
from data_gen.mirrored_cvbench import E60, JUDGES, M7, candidates, r_cut, stored_r

PROMPT_TEXT = """Look at this photograph. Does it contain any legible text: letters, words or numbers that a viewer could read or recognize as writing (for example on signs, shirts, logos, screens, books, newspapers, packaging, license plates or scoreboards)? Ignore colored rectangles drawn on top of the photograph.

Reply with exactly one line: TEXT: YES or TEXT: NO"""

PROMPT_REL = """Below is a short text written about a photograph. The text mentions two objects: X = "{x}" and Y = "{y}".

Text:
\"\"\"
{r}
\"\"\"

Find the LAST statement in the text that implies whether X is to the left or to the right of Y in the picture. It may state it directly (e.g. "X is to the left of Y", "Y is on the right of X") or place both objects on sides of the picture (e.g. "X is on the left side of the image, while Y is on the right"). Report on which side of Y that statement puts X. If the text contains no such statement, answer NONE.

Reply with exactly one line: REL: LEFT or REL: RIGHT or REL: NONE"""

CAP = 2500  # unique texts (two judges -> <= 5,000 calls)


class Limiter:
    """Simple global requests-per-minute limiter."""

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
    """Label from the reply's last matching line."""
    rx = r"TEXT:\s*(YES|NO)" if kind == "text" else r"REL:\s*(LEFT|RIGHT|NONE)"
    m = re.findall(rx, raw.upper())
    return m[-1] if m else None


def run(kind: str, jname: str, tasks: list[dict], fn: str, threads: int, rpm: float) -> None:
    """Resumable judge loop over tasks (each with 'tid' and 'messages')."""
    from srww import api_client as api_client
    done = {r["tid"] for r in vx.read_jsonl(fn) if r.get("label") is not None}
    todo = [t for t in tasks if t["tid"] not in done]
    print(f"[e60] {kind} {jname}: {len(todo)} to judge ({len(done)} done)", flush=True)
    lim, lock, cnt, t0 = Limiter(rpm), threading.Lock(), Counter(), time.time()
    f = open(fn, "a")

    def go(t):
        lim.wait()
        out = api_client.chat(jname, t["messages"], max_tokens=2048, effort="high")
        raw = (out.get("text") or "").strip()
        err = out.get("error", "")[:200]
        lab = parse(kind, raw) if not err else None
        with lock:
            f.write(json.dumps({"tid": t["tid"], "uid": t.get("uid"), "judge": jname, "label": lab, "raw": raw[-200:],
                                "error": err}, ensure_ascii=False) + "\n")
            f.flush()
            cnt["err" if err else ("unlab" if lab is None else "ok")] += 1
            s = sum(cnt.values())
            if s % 100 == 0:
                print(f"[e60] {kind} {jname}: {s}/{len(todo)} {dict(cnt)} {time.time() - t0:.0f}s", flush=True)

    with ThreadPoolExecutor(threads) as ex:
        list(ex.map(go, todo))
    f.close()
    print(f"[e60] {kind} {jname}: finished {dict(cnt)} {time.time() - t0:.0f}s", flush=True)


def text_tasks() -> list[dict]:
    """One vision task per candidate image (long side <= 1024 px)."""
    from srww import api_client as api_client
    out = []
    for c in candidates(with_images=True):
        im = c["image"].copy()
        im.thumbnail((1024, 1024))
        out.append({"tid": c["uid"], "uid": c["uid"],
                    "messages": [{"role": "user", "content": [api_client.img_part(im), {"type": "text", "text": PROMPT_TEXT}]}]})
    return out


def has_lr(text: str) -> bool:
    """Deterministic prefilter: any word starting with left / right."""
    return re.search(r"\b(left|right)", text, re.I) is not None


def rel_key(x: str, y: str, body: str) -> str:
    """Task id of a (X, Y, R_cut) triple."""
    return hashlib.sha1(f"{x}\x00{y}\x00{body}".encode()).hexdigest()[:16]


def rel_build() -> None:
    """Relation tasks for kept items x 7 models (deduplicated, capped at CAP unique texts)."""
    items = [r for r in vx.read_jsonl(f"{E60}/items.jsonl") if r["keep"]]
    tasks: dict[str, dict] = {}
    n_pre = 0
    for m in M7:
        rs = stored_r(m)
        for it in items:
            if it["uid"] not in rs:
                continue
            body = r_cut(rs[it["uid"]])
            if not has_lr(body):
                n_pre += 1
                continue
            k = rel_key(it["x"], it["y"], body)
            tasks.setdefault(k, {"tid": k, "x": it["x"], "y": it["y"], "body": body})
    ks = sorted(tasks)
    sampled = len(ks) > CAP
    if sampled:
        ks = sorted(random.Random(60).sample(ks, CAP))
    vx.write_jsonl(f"{E60}/rel_tasks.jsonl", [tasks[k] for k in ks])
    print(f"[e60] rel tasks: {len(tasks)} unique texts, {n_pre} prefiltered NONE, judged {len(ks)} (sampled={sampled})")


def main() -> None:
    """CLI entry point."""
    ap = argparse.ArgumentParser()
    ap.add_argument("phase", choices=["text", "rel-build", "rel"])
    ap.add_argument("--judge", choices=JUDGES)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--rpm", type=float, default=50)
    ap.add_argument("--limit", type=int)
    a = ap.parse_args()
    os.makedirs(E60, exist_ok=True)
    if a.phase == "text":
        ts = text_tasks()
        ts = ts[: a.limit] if a.limit else ts
        run("text", a.judge, ts, f"{E60}/judge_text_{a.judge}.jsonl", a.threads, a.rpm)
    elif a.phase == "rel-build":
        rel_build()
    else:
        ts = [{"tid": t["tid"], "messages": [{"role": "user", "content": PROMPT_REL.format(x=t["x"], y=t["y"], r=t["body"])}]}
              for t in vx.read_jsonl(f"{E60}/rel_tasks.jsonl")]
        run("rel", a.judge, ts[: a.limit] if a.limit else ts, f"{E60}/judge_rel_{a.judge}.jsonl", a.threads, a.rpm)


if __name__ == "__main__":
    main()
