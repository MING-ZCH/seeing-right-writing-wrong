"""E1x: exact full-vocabulary option probabilities for the original condition matrix.

Reuses each model's stored greedy rationale R (legacy e1 output) and recomputes the forced
``Answer: (`` readout under every original condition plus two new turn-matched long-filler
conditions (``filler512+turn``, ``filler1024+turn``). Each record keeps the legacy top-20 letter
scores (for agreement checks) and the exact full-vocab logprobs.

Usage: python e1x.py --model qwen3b [--limit 5]
"""
from __future__ import annotations

import argparse
import time

import vx
from vx import C

EXTRA = [("filler512", "turn"), ("filler1024", "turn")]


def main() -> None:
    """CLI entry point."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--conds", default=None, help="comma list of condition names (default: all)")
    args = ap.parse_args()
    t0 = time.time()
    leg = vx.legacy_e1(args.model)
    items = [it for it in vx.load_items(limit=args.limit) if it["uid"] in leg]
    print(f"[e1x] {args.model}: {len(items)} items", flush=True)
    llm = vx.make_llm(args.model)
    tok = llm.get_tokenizer()
    is_molmo = args.model.startswith("molmo")
    refs = [vx.img_ref(C.get_image(it)) for it in items]
    cots = [leg[it["uid"]]["cot"] for it in items]
    fillers = [C.filler_text(tok, max(leg[it["uid"]]["cot_len"], 1)) for it in items]
    f512, f1024 = C.filler_text(tok, 512), C.filler_text(tok, 1024)
    conds = [c for c in list(C.CONDITIONS) + EXTRA
             if not (is_molmo and ("reimage" in c[1] or c[0] == "filler1024"))]
    if args.conds:
        keep = set(args.conds.split(","))
        conds = [c for c in conds if C.cond_name(*c) in keep]
    recs = []
    for base, suffix in conds:
        name = C.cond_name(base, suffix)
        ctx = {"direct": [""] * len(items), "cot": cots, "filler": fillers,
               "filler512": [f512] * len(items), "filler1024": [f1024] * len(items)}[base]
        msgs = [C.build_messages(it, base, suffix, ctx[i], refs[i]) for i, it in enumerate(items)]
        outs = vx.readout(llm, args.model, msgs, [it["choices"] for it in items])
        for it, o in zip(items, outs):
            recs.append({"model": args.model, "task": it["task"], "item_id": it["uid"], "condition": name,
                         "gold": it["gold"], "letters": it["choices"], "image_id": it["group"], **o})
        print(f"[e1x] {name} done ({time.time() - t0:.0f}s)", flush=True)
    sfx = f"_limit{args.limit}" if args.limit else ""
    vx.write_jsonl(f"{vx.OUT}/e1x/{args.model}{sfx}.jsonl", recs)
    print(f"[e1x] wrote {len(recs)} records ({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
