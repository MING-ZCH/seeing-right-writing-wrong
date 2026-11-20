"""E5x: zero-training reasoning formats that keep the question wording out of the perceptual claims.

Conditions (all end with the forced one-token readout ``Answer: (``):
  desc    question-blind description: turn 1 asks only for the scene layout (objects, positions, sizes,
          occlusion); turn 2 gives q and reads the answer.
  sscot   structured spatial state: the assistant continues a JSON object list
          (name, bbox_2d, in_front_of / occluded_by) for the objects named in q, then one sentence
          derived from the state; the answer is read after a closing turn.
Both are compared with Direct and the stored CoT on the same items (E1x / legacy).

Usage: python e5x.py --model qwen3b [--limit 5] [--tasks ...]
"""
from __future__ import annotations

import argparse
import time

import vx
from vx import C

DESC_PROMPT = ("Describe the spatial layout of this image: list each object, where it is in the image, "
               "its apparent size, and which objects overlap or hide parts of other objects.")
SSCOT_PREFIX = ("I will first write the scene state for the objects in the question, then derive the answer "
                "from that state only.\nScene state (JSON): {\"objects\": [")


def main() -> None:
    """CLI entry point."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--tasks", default=",".join(vx.SPATIAL))
    ap.add_argument("--conds", default="desc,sscot")
    args = ap.parse_args()
    t0 = time.time()
    items = vx.load_items(args.tasks.split(","), limit=args.limit)
    llm = vx.make_llm(args.model)
    refs = [vx.img_ref(C.get_image(it)) for it in items]
    recs = []
    for cond in args.conds.split(","):
        if cond == "desc":
            gm = [[{"role": "user", "content": [r, {"type": "text", "text": DESC_PROMPT}]}] for r in refs]
            if args.model.startswith("molmo"):
                gm = [m + [{"role": "assistant", "content": ""}] for m in gm]
                gens = vx.generate(llm, args.model, gm, cont=True)
            else:
                gens = vx.generate(llm, args.model, gm, cont=False)
            ro = [[{"role": "user", "content": [r, {"type": "text", "text": DESC_PROMPT}]},
                   {"role": "assistant", "content": g["text"]},
                   {"role": "user", "content": [{"type": "text", "text": it["question"]}]},
                   {"role": "assistant", "content": C.FORCE_PREFIX}] for it, r, g in zip(items, refs, gens)]
            texts = [g["text"] for g in gens]
        else:
            gm = [[{"role": "user", "content": [r, {"type": "text", "text": it["question"]}]},
                   {"role": "assistant", "content": SSCOT_PREFIX}] for it, r in zip(items, refs)]
            gens = vx.generate(llm, args.model, gm, cont=True)
            texts = [SSCOT_PREFIX + g["text"] for g in gens]
            ro = [C.build_messages(it, "cot", "turn", t, r) for it, r, t in zip(items, refs, texts)]
        outs = vx.readout(llm, args.model, ro, [it["choices"] for it in items])
        for it, g, t, o in zip(items, gens, texts, outs):
            recs.append({"model": args.model, "task": it["task"], "item_id": it["uid"], "condition": cond,
                         "gold": it["gold"], "letters": it["choices"], "image_id": it["group"], "text": t,
                         "gen_len": g["len"], "gen_trunc": g["trunc"],
                         "stated": C.parse_free_answer(t, it["choices"]), **o})
        print(f"[e5x] {cond} done ({time.time() - t0:.0f}s)", flush=True)
    sfx = f"_limit{args.limit}" if args.limit else ""
    vx.write_jsonl(f"{vx.OUT}/e5x/{args.model}{sfx}.jsonl", recs)
    print(f"[e5x] wrote {len(recs)} records ({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
