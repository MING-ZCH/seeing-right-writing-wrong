"""Exp-50 (plan S3): do closed models' CoT rationales and answers depend on the image at all?

Six closed / API models (Exp-38 settings: reasoning high; Gemini thinking level high, see gen())
on a fixed random sample (seed 0) of 150 hard-set prompts and 150 cue-set prompts (75 "in front of" + 75 "behind" each),
same CoT prompt as e9x, under three images:
  real   the item's image (hard set: e9h CoT replies reused)
  gray   uniform gray (128, 128, 128) image of the same size
  twin   the identity-swapped twin image, same question (hard set: the e9h CoT reply to the twin prompt reused)

Phases
  sample   -> outputs/e50/sample.json (+ cue set regenerated with e12_cue_set.py --n 400 --seed 1 in outputs/e50/occ_cues;
              the shared items/occ_cues copy holds only 284/800 images; regenerated images match those 284 pixel-for-pixel
              and the meta / gold of all 1,600 e9c_cues prompts)
  gen      -> outputs/e50/gen_<model>.jsonl (resumable; error rows are retried on the next run)
  judge    -> outputs/e50/judge_<judge>.jsonl: last written relation, (i) answer line removed, conclusion kept
              (Exp-38 prompt, key hkey(x, y, body, "pre")) and (ii) conclusion removed with the Exp-35 rule (Exp-45 prompt,
              key hkey(x, y, body)). Labels already in outputs/e38 (same texts) and outputs/e45 (same hkey) are reused.
              Texts without a depth / occlusion keyword are NONE without a call (Exp-45 prefilter). Under (ii), a text whose
              truncated body equals the kept body (whitespace-normalised) takes the kept label.
  presence -> outputs/e50/presence_<judge>.jsonl: does the full reply describe both shapes as visible with a depth relation,
              say they are absent / the image is blank, or neither (all gray texts + 25 real texts per model x set). Needed
              because the relation judges count premise denials ("X is not behind Y, since neither is present") as relations.
  analyze  -> bundle/e50_closed_image_blind.json, bundle/rigor_e50.txt, paper tables/supp_e50.tex

Usage: source configs/env.sh
       python e50_closed_image_blind.py sample
       python e50_closed_image_blind.py gen --model claude-opus-5-5 --workers 12 [--key alt]
       python e50_closed_image_blind.py judge --judge gpt-6-astra --threads 16 [--key alt]
       python e50_closed_image_blind.py analyze
"""
from __future__ import annotations
import vtax_paths as _vtp  # noqa: E402

import argparse
import json
import os
import random
import re
import threading
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

W = f"{_vtp.DATA_ROOT}"
O = Path(f"{_vtp.W}/outputs")
E50 = O / "e50"
HARD = Path(f"{_vtp.W}/exps/items/occlusion_hard")
CUES = E50 / "occ_cues"
CLOSED = ["claude-opus-5-5", "gpt-6-astra", "gemini-4-argon", "qwen3.8-max-0902", "hy4", "deepseek-v4.1-flash"]
JUDGES = ("claude-opus-5-5", "gpt-6-astra")
CONDS = ("real", "gray", "twin")
N_PER_PHRASING = 75
GRAY = (128, 128, 128)


def rj(p: Path) -> list[dict]:
    return [json.loads(l) for l in open(p)] if p.exists() else []


# ============================================================================= sample

def _prompts(items_dir: Path, set_name: str) -> list[dict]:
    """All prompts of an occlusion set (base + twin images, both phrasings), e9x_api.load_items conventions."""
    items = {}
    for fn in ("items.jsonl", "items_cf.jsonl"):
        for r in map(json.loads, open(items_dir / fn)):
            items[r["item_id"]] = r
    out = []
    for iid, r in sorted(items.items()):
        tw = items[iid[:-3] if iid.endswith("_cf") else iid + "_cf"]
        assert tw["question"] == r["question"] and tw["gold"] != r["gold"]
        for ph in ("front", "behind"):
            q, g = r["question"], r["gold"]
            if ph == "behind":
                q, g = q.replace(" in front of ", " behind ", 1), ("B" if g == "A" else "A")
            out.append({"uid": f"{iid}_{ph}", "twin_uid": f"{tw['item_id']}_{ph}", "set": set_name, "phrasing": ph,
                        "question": q + "\n(A) yes\n(B) no", "gold": g, "image": str(items_dir / r["image_paths"][0]),
                        "twin_image": str(items_dir / tw["image_paths"][0]), "group": iid.replace("_cf", ""),
                        "meta": r["meta"], "twin_meta": tw["meta"]})
    return out


def sample() -> None:
    E50.mkdir(exist_ok=True)
    assert (CUES / "items.jsonl").exists(), "regenerate the cue set first: e12_cue_set.py --out-dir outputs/e50/occ_cues"
    res = {}
    for name, d in (("hard", HARD), ("cues", CUES)):
        P = _prompts(d, name)
        rng = random.Random(0)
        pick = []
        for ph in ("front", "behind"):
            pick += rng.sample([p for p in P if p["phrasing"] == ph], N_PER_PHRASING)
        res[name] = pick
        print(name, len(P), "prompts ->", len(pick), "gold A", sum(p["gold"] == "A" for p in pick),
              "images", len({p["image"] for p in pick}), "pairs", len({p["group"] for p in pick}))
    json.dump(res, open(E50 / "sample.json", "w"), indent=1)


def load_sample() -> list[dict]:
    S = json.load(open(E50 / "sample.json"))
    return S["hard"] + S["cues"]


# ============================================================================= gen

def e9h_reuse(model: str) -> dict:
    """e9h CoT reply per uid: last row with a parsed letter, else the last non-error row (Exp-38 dedup)."""
    best = {}
    for r in rj(O / "e9h" / f"{model}.jsonl"):
        if r["condition"] != "cot" or "error" in r:
            continue
        if r.get("pred") or r["item_id"] not in best or not best[r["item_id"]].get("pred"):
            best[r["item_id"]] = r
    return best


def gen(model: str, workers: int, key: str, limit: int | None) -> None:
    import io  # noqa: F401
    from PIL import Image
    import api_client
    from e9x_api import COT, parse_letter
    if key == "alt":
        os.environ["OPENAI_COMPAT_API_KEY"] = os.environ["OPENAI_COMPAT_API_KEY_ALT"]
    # Gemini: on 2026-11-16 16:40 the route was google-vertex again, which rejects reasoning_effort ("Thinking budget is
    # not supported ... use thinking level") and honours thinking_config (api_client default); set OPENAI_COMPAT_GEMINI_REASONING_EFFORT=1
    # only when the request goes through a router. Records keep raw_model and reasoning_tokens for the check.
    fn = E50 / f"gen_{model}.jsonl"
    done = {(r["set"], r["cond"], r["uid"]) for r in rj(fn) if "error" not in r}
    reuse = e9h_reuse(model)
    lock = threading.Lock()
    f = open(fn, "a")
    jobs = []
    for p in load_sample():
        for cond in CONDS:
            if (p["set"], cond, p["uid"]) in done:
                continue
            src_uid = p["twin_uid"] if cond == "twin" else p["uid"]
            if p["set"] == "hard" and cond != "gray" and src_uid in reuse:
                r = reuse[src_uid]
                rec = {"model": model, "set": p["set"], "cond": cond, "uid": p["uid"], "phrasing": p["phrasing"],
                       "gold": p["gold"], "image": p["twin_image"] if cond == "twin" else p["image"], "source": "e9h",
                       "src_item_id": src_uid, "effort": r.get("effort"), "text": r.get("text") or "", "pred": r.get("pred"),
                       "reasoning_tokens": r.get("reasoning_tokens"), "completion_tokens": r.get("completion_tokens"),
                       "raw_model": r.get("raw_model"), "finish_reason": None}
                assert r["gold"] == (p["gold"] if cond == "real" else ("B" if p["gold"] == "A" else "A"))
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                continue
            jobs.append((p, cond))
    f.flush()
    if limit:
        jobs = jobs[:limit]
    print(f"[e50] {model}: {len(jobs)} calls to run ({len(done)} done)", flush=True)
    t0, cnt = time.time(), Counter()

    def run(job):
        p, cond = job
        real = Image.open(p["twin_image"] if cond == "twin" else p["image"]).convert("RGB")
        img = Image.new("RGB", real.size, GRAY) if cond == "gray" else real
        text = p["question"] + "\n" + COT
        r = api_client.chat(model, [{"role": "user", "content": [api_client.img_part(img), {"type": "text", "text": text}]}],
                       max_tokens=16384, effort="high")
        rec = {"model": model, "set": p["set"], "cond": cond, "uid": p["uid"], "phrasing": p["phrasing"], "gold": p["gold"],
               "image": "gray" if cond == "gray" else (p["twin_image"] if cond == "twin" else p["image"]),
               "image_size": list(real.size), "source": "e50", "effort": "high"}
        if "error" in r:
            rec["error"] = r["error"]
        else:
            rec.update(text=r["text"], pred=parse_letter(r["text"], ["A", "B"]),
                       reasoning_tokens=(r["usage"].get("completion_tokens_details") or {}).get("reasoning_tokens"),
                       completion_tokens=r["usage"].get("completion_tokens"), raw_model=r["raw_model"],
                       finish_reason=r.get("finish_reason"), reasoning_chars=len(r.get("reasoning") or ""))
        with lock:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            f.flush()
            cnt["err" if "error" in rec else ("nopred" if not rec.get("pred") else "ok")] += 1
            s = sum(cnt.values())
            if s % 50 == 0:
                print(f"[e50] {model}: {s}/{len(jobs)} {dict(cnt)} {time.time() - t0:.0f}s", flush=True)

    with ThreadPoolExecutor(workers) as ex:
        list(ex.map(run, jobs))
    f.close()
    print(f"[e50] {model}: finished {dict(cnt)} {time.time() - t0:.0f}s", flush=True)


def load_gen() -> list[dict]:
    """Final record per model x set x cond x uid (last non-error row; error rows kept only if nothing else)."""
    out = {}
    for m in CLOSED:
        for r in rj(E50 / f"gen_{m}.jsonl"):
            k = (m, r["set"], r["cond"], r["uid"])
            if "error" not in r or k not in out or "error" in out[k]:
                out[k] = r
    return list(out.values())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("phase", choices=["sample", "gen", "judge", "presence", "analyze"])
    ap.add_argument("--model", choices=CLOSED)
    ap.add_argument("--judge", choices=JUDGES)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--threads", type=int, default=16)
    ap.add_argument("--key", default="main", choices=["main", "alt"])
    ap.add_argument("--limit", type=int)
    ap.add_argument("--shard", default="", help="i/n: judge only texts with int(hkey) %% n == i")
    a = ap.parse_args()
    if a.phase == "sample":
        sample()
    elif a.phase == "gen":
        gen(a.model, a.workers, a.key, a.limit)
    elif a.phase == "judge":
        from e50_judge import judge
        judge(a.judge, a.threads, a.key, a.limit, a.shard)
    elif a.phase == "presence":
        from e50_judge import presence
        presence(a.judge, a.threads, a.key)
    else:
        from e50_analyze import analyze
        analyze()


if __name__ == "__main__":
    main()
