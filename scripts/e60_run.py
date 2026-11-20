"""E60 (GPU): fixed-rationale readout under the original vs horizontally mirrored natural image.

For every left/right candidate (outputs/e60/candidates.jsonl; the item filter is applied in rigor_e60.py) and both
images v / mirror(v), read the forced answer with three assistant contexts:
  direct  "Answer: ("
  rcut    R_cut + "\\nAnswer: ("   (R_cut = legacy strip_answer of the model's own stored R; the paper's R_cut)
  full    R + "\\nAnswer: ("
Engines / readouts = vx.make_llm / vx.readout (as the stored main-pool runs).
-> outputs/e60/<model>.jsonl (one record per item x image x context)

Usage: python e60_run.py --model qwen7b [--limit 3] [--dry]
"""
from __future__ import annotations

import argparse
import functools
import json
import os
import time

import vx
from vx import C
from e60_items import E60, candidates, r_cut, stored_r

IMGS = ("orig", "mirror")
CTXS = ("direct", "rcut", "full")


def build_jobs(model: str, limit: int | None) -> tuple[list[dict], list[dict]]:
    """Items with images and the (item index, image, context, assistant text) job list."""
    items = candidates(with_images=True)
    if limit:
        items = [it for s in ("cvb", "blink") for it in [x for x in items if x["set"] == s][:limit]]
    rs = stored_r(model)
    jobs = []
    for i, it in enumerate(items):
        r = rs.get(it["uid"])
        for img in IMGS:
            for ctx in CTXS:
                if ctx != "direct" and r is None:
                    continue
                text = "" if ctx == "direct" else (r if ctx == "full" else r_cut(r))
                jobs.append({"i": i, "img": img, "ctx": ctx, "text": text})
    return items, jobs


def main() -> None:
    """CLI entry point."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--dry", action="store_true", help="build prompts only (CPU), no engine")
    args = ap.parse_args()
    t0 = time.time()
    items, jobs = build_jobs(args.model, args.limit)
    print(f"[e60] {args.model}: {len(items)} items, {len(jobs)} readouts", flush=True)
    refs: dict[tuple[int, str], dict] = {}
    msgs, ch = [], []
    for j in jobs:
        it = items[j["i"]]
        key = (j["i"], j["img"])
        if key not in refs:
            refs[key] = vx.img_ref(vx.image_variant(it, "original" if j["img"] == "orig" else "mirror"))
        base = "direct" if j["ctx"] == "direct" else "cot"
        msgs.append(C.build_messages(it, base, "none", j["text"], refs[key]))
        ch.append(it["choices"])
    if args.dry:
        for j, m in list(zip(jobs, msgs))[:6]:
            print(j["img"], j["ctx"], json.dumps([{"role": x["role"], "content": x["content"] if isinstance(x["content"], str)
                                                   else [p.get("text", "<image>") for p in x["content"]]} for x in m])[:600])
        print(f"[e60] dry run ok ({time.time() - t0:.0f}s)")
        return
    if os.environ.get("E22_NO_MM_CACHE"):
        import vllm
        vllm.LLM = functools.partial(vllm.LLM, disable_mm_preprocessor_cache=True)
    if os.environ.get("E28_MAX_BATCHED"):
        import vllm
        vllm.LLM = functools.partial(vllm.LLM, max_num_batched_tokens=int(os.environ["E28_MAX_BATCHED"]))
    llm = vx.make_llm(args.model)
    t1 = time.time()
    outs = vx.readout(llm, args.model, msgs, ch)
    recs = []
    for j, o in zip(jobs, outs):
        it = items[j["i"]]
        recs.append({"model": args.model, "item_id": it["uid"], "set": it["set"], "img": j["img"], "ctx": j["ctx"],
                     "gold": it["gold"], "gold_mirror": it["gold_mirror"], "letters": it["choices"],
                     "image_id": it["group"], "ctx_chars": len(j["text"]), **o})
    sfx = f"_limit{args.limit}" if args.limit else ""
    vx.write_jsonl(f"{E60}/{args.model}{sfx}.jsonl", recs)
    json.dump({"load_s": t1 - t0, "readout_s": time.time() - t1, "n": len(recs)},
              open(f"{E60}/{args.model}{sfx}.timing.json", "w"))
    print(f"[e60] wrote {len(recs)} records ({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
