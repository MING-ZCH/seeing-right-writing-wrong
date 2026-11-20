"""E3x: same-item 2x2 conclusion swap crossed with image ablation.

On two-option items whose stored R states a parsable option, append the original swap sentence
``Therefore, the answer is (Y) <option>.`` to R_cut (original ``strip_answer``) for Y in {gold, wrong},
and also test the conclusion alone (no R body). Readout images: original, gray, and (left/right items)
mirrored. Direct readouts under gray / mirror are stored as image-prior covariates.

Selectivity S = F(gold) - F(wrong) is computed on the same items; visual contribution
dvis = S(original) - S(gray); under mirror the image supports the opposite option.

Usage: python e3x.py --model qwen7b [--limit 5]
"""
from __future__ import annotations
import vtax_paths as _vtp  # noqa: E402

import argparse
import sys
import time

import vx
from vx import C

sys.path.insert(0, f"{_vtp.LEGACY_SCRIPTS}")
from e5_content_edit import strip_answer  # noqa: E402
from e2a import mirror_item  # noqa: E402


def main() -> None:
    """CLI entry point."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()
    t0 = time.time()
    leg = vx.legacy_e1(args.model)
    items = [it for it in vx.load_items(limit=args.limit)
             if it["uid"] in leg and len(it["choices"]) == 2 and leg[it["uid"]]["cot_free"] in it["choices"]]
    print(f"[e3x] {args.model}: {len(items)} binary items with parsable R", flush=True)
    llm = vx.make_llm(args.model)
    jobs = []  # (item index, image variant, context kind, Y)
    for i, it in enumerate(items):
        variants = ["original", "gray"] + (["mirror"] if mirror_item(it) else [])
        wrong = [c for c in it["choices"] if c != it["gold"]][0]
        for var in variants:
            jobs.append((i, var, "direct", None))
            for kind in ("rcut", "concl"):
                for y in (it["gold"], wrong):
                    jobs.append((i, var, kind, y))
    refs: dict[tuple[int, str], dict] = {}
    msgs, choices = [], []
    for i, var, kind, y in jobs:
        it = items[i]
        if (i, var) not in refs:
            refs[(i, var)] = vx.img_ref(vx.image_variant(it, var))
        ref = refs[(i, var)]
        if kind == "direct":
            msgs.append(C.build_messages(it, "direct", "none", "", ref))
        else:
            ow = vx.option_words(it)
            body = strip_answer(leg[it["uid"]]["cot"]) if kind == "rcut" else C.COT_PREFIX
            ctx = f"{body} Therefore, the answer is ({y}) {ow.get(y, '')}.".replace(" .", ".")
            msgs.append(C.build_messages(it, "cot", "none", ctx, ref))
        choices.append(it["choices"])
    print(f"[e3x] {len(msgs)} readouts", flush=True)
    outs = vx.readout(llm, args.model, msgs, choices)
    recs = []
    for (i, var, kind, y), o in zip(jobs, outs):
        it = items[i]
        mi = mirror_item(it)
        recs.append({"model": args.model, "task": it["task"], "item_id": it["uid"], "image": var, "context": kind,
                     "Y": y, "gold": it["gold"], "gold_mirror": mi["gold"] if mi else it["gold"],
                     "r_stated": leg[it["uid"]]["cot_free"], "letters": it["choices"],
                     "image_id": it["group"], **o})
    sfx = f"_limit{args.limit}" if args.limit else ""
    vx.write_jsonl(f"{vx.OUT}/e3x/{args.model}{sfx}.jsonl", recs)
    print(f"[e3x] wrote {len(recs)} records ({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
