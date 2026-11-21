"""E6x: frozen-protocol held-out benchmarks (BLINK val spatial tasks, 3DSRBench real-image subset).

Stage ``direct``: Direct readout only (outputs/e6x/<model>_direct.jsonl). Predictions are then written by
``experiments/behavior/heldout_predict.py`` (timestamped, with input sha256) before stage ``rest`` runs:
  cot    greedy R (512) + same-turn readout
  vote4  4 samples at T=0.7 + same-turn readout per sample (original e6 protocol)
  desc   question-blind description, then q and readout (E5x)

Usage: python -m experiments.behavior.heldout_benchmarks --model qwen3b --stage direct|rest [--limit 5]
"""
from __future__ import annotations

import argparse
import base64
import io
import os
import random
import time

import pandas as pd
from PIL import Image

from srww import vllm_utils as vx
from srww.vllm_utils import C
from experiments.behavior.reasoning_formats import DESC_PROMPT

H = f"{vx.SHARE}/datasets/heldout"
DSR_CATS = ["location_closer_to_camera", "location_above", "height_higher", "orientation_in_front_of",
            "orientation_on_the_left"]


def load_heldout(limit: int | None = None) -> list[dict]:
    """Held-out items in the original item format (question includes lettered options)."""
    items = []
    for task, sub in (("blink_relation", "Spatial_Relation"), ("blink_depth", "Relative_Depth")):
        d = pd.read_parquet(f"{H}/BLINK/{sub}/val-00000-of-00001.parquet")
        for _, r in d.iterrows():
            n = len(list(r["choices"]))
            items.append({"uid": r["idx"], "dataset": "blink", "task": task, "question": r["prompt"],
                          "choices": C.letters(n), "gold": r["answer"].strip("() "),
                          "image": Image.open(io.BytesIO(r["image_1"]["bytes"])).convert("RGB"), "group": r["idx"]})
    d = pd.read_csv(f"{H}/3DSRBench/3dsrbench_v1_vlmevalkit.tsv", sep="\t")
    rng = random.Random(0)
    for cat in DSR_CATS:
        sub = d[d["category"] == cat].drop_duplicates("image_url")
        rows = sub.iloc[sorted(rng.sample(range(len(sub)), min(100, len(sub))))]
        for _, r in rows.iterrows():
            opts = [(L, str(r[L])) for L in "ABCD" if str(r[L]) not in ("nan", "None", "")]
            q = r["question"] + " Select from the following choices.\n" + "\n".join(f"({L}) {o}" for L, o in opts)
            items.append({"uid": f"3dsr_{r['index']}", "dataset": "3dsr", "task": f"3dsr_{cat}", "question": q,
                          "choices": [L for L, _ in opts], "gold": r["answer"],
                          "image": Image.open(io.BytesIO(base64.b64decode(r["image"]))).convert("RGB"),
                          "group": r["image_url"]})
    if limit:
        by = {}
        for it in items:
            by.setdefault(it["task"], []).append(it)
        items = [x for v in by.values() for x in v[:limit]]
    return items


def main() -> None:
    """CLI entry point."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--stage", required=True, choices=["direct", "rest"])
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()
    t0 = time.time()
    sfx = f"_limit{args.limit}" if args.limit else ""
    if args.stage == "rest" and not args.limit:
        preds = [f for f in os.listdir(f"{vx.OUT}/e6x") if f.startswith(f"predictions_{args.model}_")]
        assert preds, "run experiments.behavior.heldout_predict on the Direct outputs before stage rest"
    items = load_heldout(args.limit)
    print(f"[e6x] {args.model}: {len(items)} held-out items", flush=True)
    llm = vx.make_llm(args.model)
    refs = [vx.img_ref(it["image"]) for it in items]
    ch = [it["choices"] for it in items]
    meta = lambda it: {"model": args.model, "task": it["task"], "item_id": it["uid"], "gold": it["gold"],
                       "letters": it["choices"], "image_id": it["group"]}
    recs = []
    if args.stage == "direct":
        outs = vx.readout(llm, args.model, [C.build_messages(it, "direct", "none", "", r) for it, r in zip(items, refs)], ch)
        recs = [meta(it) | {"condition": "direct"} | o for it, o in zip(items, outs)]
        vx.write_jsonl(f"{vx.OUT}/e6x/{args.model}_direct{sfx}.jsonl", recs)
        print(f"[e6x] direct done ({time.time() - t0:.0f}s)", flush=True)
        return
    # greedy CoT
    gens = vx.generate(llm, args.model, [C.cot_messages(it, r) for it, r in zip(items, refs)])
    cots = [C.COT_PREFIX + " " + g["text"] for g in gens]
    outs = vx.readout(llm, args.model, [C.build_messages(it, "cot", "none", c, r) for it, c, r in zip(items, cots, refs)], ch)
    recs += [meta(it) | {"condition": "cot", "text": c, "gen_len": g["len"]} | o
             for it, c, g, o in zip(items, cots, gens, outs)]
    print(f"[e6x] cot done ({time.time() - t0:.0f}s)", flush=True)
    # 4 samples at T=0.7
    samp = vx.generate(llm, args.model, [C.cot_messages(it, r) for it, r in zip(items, refs)], temperature=0.7, n=4)
    flat = [(i, j) for i in range(len(items)) for j in range(4)]
    sc = [C.COT_PREFIX + " " + samp[i]["samples"][j]["text"] for i, j in flat]
    outs = vx.readout(llm, args.model, [C.build_messages(items[i], "cot", "none", s, refs[i]) for (i, _), s in zip(flat, sc)],
                      [ch[i] for i, _ in flat])
    for i, it in enumerate(items):
        o4 = outs[4 * i: 4 * i + 4]
        recs.append(meta(it) | {"condition": "vote4", "vote_outs": o4,
                                "vote_stated": [C.parse_free_answer(sc[4 * i + j], it["choices"]) for j in range(4)],
                                "gen_len": [samp[i]["samples"][j]["len"] for j in range(4)]})
    print(f"[e6x] vote4 done ({time.time() - t0:.0f}s)", flush=True)
    # question-blind description
    gm = [[{"role": "user", "content": [r, {"type": "text", "text": DESC_PROMPT}]}] for r in refs]
    if args.model.startswith("molmo"):
        gens = vx.generate(llm, args.model, [m + [{"role": "assistant", "content": ""}] for m in gm], cont=True)
    else:
        gens = vx.generate(llm, args.model, gm, cont=False)
    ro = [m + [{"role": "assistant", "content": g["text"]}, {"role": "user", "content": [{"type": "text", "text": it["question"]}]},
               {"role": "assistant", "content": C.FORCE_PREFIX}] for m, g, it in zip(gm, gens, items)]
    outs = vx.readout(llm, args.model, ro, ch)
    recs += [meta(it) | {"condition": "desc", "text": g["text"], "gen_len": g["len"]} | o
             for it, g, o in zip(items, gens, outs)]
    vx.write_jsonl(f"{vx.OUT}/e6x/{args.model}_rest{sfx}.jsonl", recs)
    print(f"[e6x] wrote {len(recs)} records ({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
