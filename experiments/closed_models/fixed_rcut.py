"""Exp-61: open-model fixed-rationale twin swap run on closed APIs (Exp-50 regenerated the rationale).

For each Exp-50 prompt (150 hard-set + 150 cue-set, seed 0) the model's own stored CoT reply R under the real image v
is truncated (variants: 'removed' = Exp-35/45 rule, primary; 'kept' = Exp-38 strip_answer; 'strict' = Exp-45
strict_body) and supplied as the model's own previous assistant turn:
    user[image + question + e9x COT instruction], assistant[R_var], user['Answer with the option letter only.']
under the real image v and under the identity-swapped twin v_cf (same R_var). Direct-low arm: letter only, same lowest
reasoning setting, under v and v_cf (cue set; hard set only for models without e9h 'direct_nr').

Phases
  build  -> outputs/e61/items.jsonl (one row per model x prompt with the three bodies and Exp-50 relation labels)
  pilot  -> outputs/e61/pilot.jsonl (effort levels, multi-turn and prefill acceptance; <= 4 calls per effort level)
  gen    -> outputs/e61/gen_<model>.jsonl (resumable; error rows retried on the next run)

Usage: source configs/env.sh
       python -m experiments.closed_models.fixed_rcut build
       python -m experiments.closed_models.fixed_rcut pilot --model gpt-6-astra
       python -m experiments.closed_models.fixed_rcut gen --model claude-opus-5-5 --key alt --workers 8
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from experiments.closed_models.image_blind import E50, O, load_gen, rj

E61 = O / "e61"
MAIN = ["claude-opus-5-5", "gpt-6-astra", "gemini-4-argon", "qwen3.8-max-0902"]
OPTIONAL = ["deepseek-v4.1-flash", "hy4"]
VARIANTS = ("removed", "kept", "strict")
LOW = {"claude-opus-5-5": "minimal", "gpt-6-astra": "low", "gemini-4-argon": "low", "qwen3.8-max-0902": "none",
       "deepseek-v4.1-flash": "none", "hy4": "none"}
ASK = "Answer with the option letter only."
NR_EFFORT_E9H = {"claude-opus-5-5": "minimal", "gpt-6-astra": "low", "qwen3.8-max-0902": "none",
                 "deepseek-v4.1-flash": "none"}


def norm(s: str) -> str:
    return re.sub(r"\s+", " ", s or "").strip()


def bhash(body: str) -> str:
    return hashlib.sha1(norm(body).encode()).hexdigest()[:16]


def cut_original(text: str, all_sents: list[str], kept: list[str]) -> str:
    """Cut the original reply after the last kept sentence if the kept sentences are a contiguous prefix of the
    (non-empty) sentence list; otherwise join the kept sentences with newlines."""
    s = [x for x in all_sents if x.strip()]
    if not kept:
        return ""
    if s[: len(kept)] == kept:
        pos = 0
        for sent in kept:
            k = text.find(sent, pos)
            if k < 0:
                return "\n".join(kept)
            pos = k + len(sent)
        return text[:pos].rstrip()
    return "\n".join(kept)


# ============================================================================= build

def build() -> None:
    from srww.claim_parse import split_sents
    from srww.rcut import truncate
    from judges.closed_relation_judge import strip_answer
    from experiments.two_channel.uniform_two_channel import strict_body
    E61.mkdir(exist_ok=True)
    S = json.load(open(E50 / "sample.json"))
    P = {(p["set"], p["uid"]): p for v in S.values() for p in v}
    bundle = json.load(open(Path("outputs/bundle/e50_closed_image_blind.json")))
    lab = {(r["model"], r["set"], r["uid"]): r for r in bundle["rows"] if r["cond"] == "real"}
    out = []
    for g in load_gen():
        if g["cond"] != "real" or g["model"] not in MAIN + OPTIONAL:
            continue
        p = P[(g["set"], g["uid"])]
        mt = p["meta"]
        front, back = mt["front_name"].lower(), mt["back_name"].lower()
        text = g.get("text") or ""
        sents = split_sents(text)
        tr, info = truncate(sents, [front, back])
        kept, _ = strip_answer(text)
        removed = cut_original(text, sents, tr)
        strict_join = strict_body({"trunc": tr, "front": front, "back": back, "group": "closed"})
        strict_sents = strict_join.split("\n") if strict_join else []
        strict = cut_original(text, sents, strict_sents)
        L = lab[(g["model"], g["set"], g["uid"])]
        bodies = {"removed": removed, "kept": kept, "strict": strict}
        out.append({"model": g["model"], "set": g["set"], "uid": g["uid"], "twin_uid": p["twin_uid"], "group": p["group"],
                    "phrasing": p["phrasing"], "gold": p["gold"], "question": p["question"], "image": p["image"],
                    "twin_image": p["twin_image"], "x": L["x"], "y": L["y"], "real_front": L["real_front"],
                    "stored_pred": g.get("pred"), "stored_source": g.get("source"), "trunc_info": info,
                    "rel_kept": L["rel_kept"], "rel_removed": L["rel_removed"],
                    "lab_kept": L["lab_kept"], "lab_removed": L["lab_removed"],
                    **{f"body_{v}": b for v, b in bodies.items()}, **{f"bh_{v}": bhash(b) for v, b in bodies.items()},
                    "removed_eq_kept": norm(removed) == norm(kept), "strict_eq_removed": norm(strict) == norm(removed),
                    "removed_join_eq_e50": norm(removed) == norm("\n".join(tr))})
    with open(E61 / "items.jsonl", "w") as f:
        for r in out:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    c = Counter()
    for r in out:
        c[(r["model"], "uniq_bodies")] += len({r[f"bh_{v}"] for v in VARIANTS if r[f"body_{v}"].strip()})
        c[(r["model"], "removed_eq_kept")] += r["removed_eq_kept"]
        c[(r["model"], "strict_eq_removed")] += r["strict_eq_removed"]
        c[(r["model"], "strict_empty")] += not r["body_strict"].strip()
        c[(r["model"], "removed_empty")] += not r["body_removed"].strip()
    for k, v in sorted(c.items()):
        print(k, v)


def load_items(model: str | None = None) -> list[dict]:
    return [r for r in rj(E61 / "items.jsonl") if model is None or r["model"] == model]


# ============================================================================= API

def messages(it: dict, img_path: str, body: str | None, prefill: bool = False) -> list:
    from srww import api_client as api_client
    from PIL import Image
    from experiments.closed_models.occlusion_audit import COT, DIRECT
    img = api_client.img_part(Image.open(img_path).convert("RGB"))
    if body is None:  # Direct-low
        return [{"role": "user", "content": [img, {"type": "text", "text": it["question"] + "\n" + DIRECT}]}]
    m = [{"role": "user", "content": [img, {"type": "text", "text": it["question"] + "\n" + COT}]}]
    if prefill:
        return m + [{"role": "assistant", "content": body + "\nAnswer: ("}]
    return m + [{"role": "assistant", "content": body}, {"role": "user", "content": ASK}]


def call(model: str, msgs: list, effort: str, max_tokens: int = 4096) -> dict:
    """api_client.chat with extra backoff rounds on 429 / 5xx / network errors."""
    from srww import api_client as api_client
    r = {}
    for k in range(4):
        r = api_client.chat(model, msgs, max_tokens=max_tokens, effort=effort)
        if "error" not in r:
            return r
        e = r["error"]
        if e.startswith("HTTP 4") and not e.startswith("HTTP 429"):
            return r
        time.sleep(30 * (k + 1))
    return r


def summarize(r: dict) -> dict:
    from experiments.closed_models.occlusion_audit import parse_letter
    if "error" in r:
        return {"error": r["error"][:300]}
    u = r.get("usage") or {}
    return {"text": r["text"][:2000], "pred": parse_letter(r["text"], ["A", "B"]),
            "reasoning_tokens": (u.get("completion_tokens_details") or {}).get("reasoning_tokens"),
            "completion_tokens": u.get("completion_tokens"), "prompt_tokens": u.get("prompt_tokens"),
            "raw_model": r.get("raw_model"), "finish_reason": r.get("finish_reason"),
            "reasoning_chars": len(r.get("reasoning") or "")}


def set_key(key: str) -> None:
    if key == "alt":
        os.environ["OPENAI_COMPAT_API_KEY"] = os.environ["OPENAI_COMPAT_API_KEY_ALT"]


# ============================================================================= pilot

def pilot(model: str, key: str, efforts: list[str]) -> None:
    set_key(key)
    its = [r for r in load_items(model) if r["set"] == "hard" and r["body_removed"].strip()]
    it = its[0]
    f = open(E61 / "pilot.jsonl", "a")
    for eff in efforts:
        for kind in ("multiturn", "prefill"):
            msgs = messages(it, it["image"], it["body_removed"], prefill=(kind == "prefill"))
            r = summarize(call(model, msgs, eff))
            rec = {"model": model, "effort": eff, "kind": kind, "uid": it["uid"], "stored_pred": it["stored_pred"],
                   "time": time.strftime("%H:%M:%S"), **r}
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            f.flush()
            print(json.dumps({k: (v[:200] if isinstance(v, str) else v) for k, v in rec.items()}, ensure_ascii=False))
            if "error" in r:
                break
    f.close()


# ============================================================================= gen

def jobs_for(model: str, sets: tuple, variants: tuple, direct_sets: tuple, prefill_n: int) -> list[dict]:
    jobs, seen = [], set()
    for it in load_items(model):
        if it["set"] not in sets:
            continue
        for v in variants:
            body = it[f"body_{v}"]
            if not body.strip():
                continue
            for arm in ("v", "vcf"):
                k = ("rcut", it["set"], it["uid"], arm, it[f"bh_{v}"])
                if k not in seen:
                    seen.add(k)
                    jobs.append({"kind": "rcut", "it": it, "arm": arm, "bh": it[f"bh_{v}"], "body": body})
        if it["set"] in direct_sets:
            for arm in ("v", "vcf"):
                jobs.append({"kind": "direct", "it": it, "arm": arm, "bh": "", "body": None})
    if prefill_n:
        hard = [it for it in load_items(model) if it["set"] == "hard" and it["body_removed"].strip()][:prefill_n]
        for it in hard:
            for arm in ("v", "vcf"):
                jobs.append({"kind": "prefill", "it": it, "arm": arm, "bh": it["bh_removed"], "body": it["body_removed"]})
    return jobs


def gen(model: str, key: str, workers: int, sets: tuple, variants: tuple, direct_sets: tuple, prefill_n: int,
        limit: int | None, effort: str | None, shard: str = "") -> None:
    set_key(key)
    eff = effort or LOW[model]
    fn = E61 / f"gen_{model}.jsonl"
    done = {(r["kind"], r["set"], r["uid"], r["arm"], r["bh"]) for r in rj(fn) if "error" not in r}
    jobs = [j for j in jobs_for(model, sets, variants, direct_sets, prefill_n)
            if (j["kind"], j["it"]["set"], j["it"]["uid"], j["arm"], j["bh"]) not in done]
    if shard:
        i, n = map(int, shard.split("/"))
        jobs = [j for j in jobs if int(hashlib.md5(j["it"]["uid"].encode()).hexdigest(), 16) % n == i]
    if limit:
        jobs = jobs[:limit]
    print(f"[e61] {model}: {len(jobs)} calls to run ({len(done)} done), effort {eff}", flush=True)
    lock = threading.Lock()
    f = open(fn, "a")
    t0, cnt = time.time(), Counter()

    def run(j):
        it = j["it"]
        img = it["image"] if j["arm"] == "v" else it["twin_image"]
        msgs = messages(it, img, j["body"], prefill=(j["kind"] == "prefill"))
        r = summarize(call(model, msgs, eff))
        rec = {"model": model, "kind": j["kind"], "set": it["set"], "uid": it["uid"], "arm": j["arm"], "bh": j["bh"],
               "gold": it["gold"], "image": img, "effort": eff, **r}
        with lock:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            f.flush()
            cnt["err" if "error" in rec else ("nopred" if not rec.get("pred") else "ok")] += 1
            s = sum(cnt.values())
            if s % 100 == 0:
                print(f"[e61] {model}: {s}/{len(jobs)} {dict(cnt)} {time.time() - t0:.0f}s", flush=True)

    with ThreadPoolExecutor(workers) as ex:
        list(ex.map(run, jobs))
    f.close()
    print(f"[e61] {model}: finished {dict(cnt)} {time.time() - t0:.0f}s", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("phase", choices=["build", "pilot", "gen", "count"])
    ap.add_argument("--model", choices=MAIN + OPTIONAL)
    ap.add_argument("--key", default="main", choices=["main", "alt"])
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--efforts", default="")
    ap.add_argument("--effort", default=None)
    ap.add_argument("--sets", default="hard,cues")
    ap.add_argument("--variants", default=",".join(VARIANTS))
    ap.add_argument("--direct-sets", default="cues")
    ap.add_argument("--prefill-n", type=int, default=0)
    ap.add_argument("--limit", type=int)
    ap.add_argument("--shard", default="", help="i/n: only prompts with md5(uid) %% n == i")
    a = ap.parse_args()
    if a.phase == "build":
        build()
    elif a.phase == "pilot":
        pilot(a.model, a.key, a.efforts.split(","))
    elif a.phase == "count":
        for m in MAIN + OPTIONAL:
            ds = tuple(x for x in a.direct_sets.split(",") if x)
            print(m, Counter(j["kind"] for j in jobs_for(m, tuple(a.sets.split(",")), tuple(a.variants.split(",")), ds, 0)))
    else:
        gen(a.model, a.key, a.workers, tuple(a.sets.split(",")), tuple(a.variants.split(",")),
            tuple(x for x in a.direct_sets.split(",") if x), a.prefill_n, a.limit, a.effort, a.shard)


if __name__ == "__main__":
    main()
