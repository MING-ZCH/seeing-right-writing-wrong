"""E2d: image counterfactual for depth on the cue-decoupled occlusion set (200 scenes + identity-swapped twins).

Twins share the question and geometry; only which object is in front changes, so the gold answer flips.
For each image we run Direct, CoT (greedy R + same-turn readout) and the question-blind description
condition. A reasoning trace that tracks occlusion should flip its stated answer between twins.

Usage: python -m experiments.occlusion.twins_readout --model qwen3b [--limit 5]
"""
from __future__ import annotations

import argparse
import json
import time

from srww import vllm_utils as vx
from srww.vllm_utils import C
from experiments.behavior.reasoning_formats import DESC_PROMPT

ITEMS = f"{vx.W}/data/items/occlusion"


def load_occ(limit: int | None = None, polar: bool = False, items_dir: str | None = None) -> list[dict]:
    """Occlusion items (base + twin) in the original item format; cluster id = twin pair."""
    items_dir = items_dir or ITEMS
    out = []
    for fn in ("items.jsonl", "items_cf.jsonl"):
        rows = [json.loads(l) for l in open(f"{items_dir}/{fn}")][:limit]
        for r in rows:
            pair = r["item_id"].replace("_cf", "")
            img = r.get("image_paths", [r.get("image")])[0] if r.get("image_paths") or r.get("image") else None
            q, g = r["question"], r["gold"]
            if polar:
                q, g = q.replace(" in front of ", " behind ", 1), ("B" if g == "A" else "A")
            out.append({"uid": r["item_id"], "dataset": "occlusion", "task": r["task"],
                        "question": q + "\n(A) yes\n(B) no", "choices": ["A", "B"], "gold": g,
                        "image_path": img if img.startswith("/") else f"{items_dir}/{img}", "group": pair,
                        "meta": r["meta"]})
    return out


def main() -> None:
    """CLI entry point."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--polar", action="store_true", help='ask "Is X behind Y?" (gold flips)')
    ap.add_argument("--items-dir", default=ITEMS, help="occlusion item directory (default: original set)")
    ap.add_argument("--out-dir", default=f"{vx.OUT}/e2d", help="output directory")
    ap.add_argument("--no-desc", action="store_true", help="skip the question-blind description condition")
    args = ap.parse_args()
    t0 = time.time()
    items = load_occ(args.limit, args.polar, args.items_dir)
    llm = vx.make_llm(args.model)
    refs = [vx.img_ref(C.get_image(it)) for it in items]
    ch = [it["choices"] for it in items]
    meta = lambda it: {"model": args.model, "task": it["task"], "item_id": it["uid"], "gold": it["gold"],
                       "letters": it["choices"], "image_id": it["group"], "meta": it["meta"]}
    recs = []
    outs = vx.readout(llm, args.model, [C.build_messages(it, "direct", "none", "", r) for it, r in zip(items, refs)], ch)
    recs += [meta(it) | {"condition": "direct"} | o for it, o in zip(items, outs)]
    gens = vx.generate(llm, args.model, [C.cot_messages(it, r) for it, r in zip(items, refs)])
    cots = [C.COT_PREFIX + " " + g["text"] for g in gens]
    outs = vx.readout(llm, args.model, [C.build_messages(it, "cot", "none", c, r) for it, c, r in zip(items, cots, refs)], ch)
    recs += [meta(it) | {"condition": "cot", "text": c, "gen_len": g["len"], "stated": C.parse_free_answer(c, it["choices"])} | o
             for it, c, g, o in zip(items, cots, gens, outs)]
    if args.no_desc:
        sfx = ("_behind" if args.polar else "") + (f"_limit{args.limit}" if args.limit else "")
        vx.write_jsonl(f"{args.out_dir}/{args.model}{sfx}.jsonl", recs)
        print(f"[e2d] wrote {len(recs)} records ({time.time() - t0:.0f}s)", flush=True)
        return
    gm = [[{"role": "user", "content": [r, {"type": "text", "text": DESC_PROMPT}]}] for r in refs]
    if args.model.startswith("molmo"):
        gd = vx.generate(llm, args.model, [m + [{"role": "assistant", "content": ""}] for m in gm], cont=True)
    else:
        gd = vx.generate(llm, args.model, gm, cont=False)
    ro = [m + [{"role": "assistant", "content": g["text"]}, {"role": "user", "content": [{"type": "text", "text": it["question"]}]},
               {"role": "assistant", "content": C.FORCE_PREFIX}] for m, g, it in zip(gm, gd, items)]
    outs = vx.readout(llm, args.model, ro, ch)
    recs += [meta(it) | {"condition": "desc", "text": g["text"], "gen_len": g["len"]} | o for it, g, o in zip(items, gd, outs)]
    sfx = ("_behind" if args.polar else "") + (f"_limit{args.limit}" if args.limit else "")
    vx.write_jsonl(f"{args.out_dir}/{args.model}{sfx}.jsonl", recs)
    print(f"[e2d] wrote {len(recs)} records ({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
