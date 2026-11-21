"""E2a: does the rationale track the image? Regenerate R under image ablations and counterfactuals.

Variants of the generation input (all items of the 2,850 spatial pool unless noted):
  gray    same-resolution mid-gray image
  none    no image (text only)
  cfq     real image, question with the two objects swapped (synthetic sets; gold flips)
  mirror  horizontally mirrored image (left/right items only; gold flips)
For each variant we store R, its stated option, and forced readouts:
  gray/none : real image + R_variant (same turn)  -> how much of the answer R carries without pixels
  cfq/mirror: Direct and CoT (variant input + R_variant) readouts under the counterfactual input.

Usage: python -m experiments.behavior.rationale_ablation --model qwen3b [--limit 5]
"""
from __future__ import annotations

import argparse
import re
import time

from srww import vllm_utils as vx
from srww.vllm_utils import C

SWAP_RX = re.compile(r"^Is the (.+?) (to the \w+ of|in front of|above|below|behind) the (.+?)\?")


def cf_question(item: dict) -> dict | None:
    """Synthetic item with the two objects swapped (gold flips); None if not applicable."""
    if item["task"] not in ("syn_depth", "syn_relpos"):
        return None
    head, _, rest = item["question"].partition("\n")
    m = SWAP_RX.match(head)
    if not m:
        return None
    q2 = f"Is the {m.group(3)} {m.group(2)} the {m.group(1)}?"
    return {**item, "question": q2 + "\n" + rest, "gold": "B" if item["gold"] == "A" else "A"}


def polar_question(item: dict) -> dict | None:
    """Syn. Depth item asked with the opposite predicate ("behind" instead of "in front of"); gold flips."""
    if item["task"] != "syn_depth" or " in front of " not in item["question"]:
        return None
    return {**item, "question": item["question"].replace(" in front of ", " behind ", 1),
            "gold": "B" if item["gold"] == "A" else "A"}


def lr_kind(item: dict) -> str | None:
    """'options' if options are left/right words, 'question' if a yes/no left/right question, else None."""
    ow = {k: v.lower() for k, v in vx.option_words(item).items()}
    if sorted(ow.values()) == ["left", "right"]:
        return "options"
    if item["task"] == "syn_relpos" and re.search(r"\b(left|right)\b", item["question"].split("\n")[0]):
        return "question"
    return None


def mirror_item(item: dict) -> dict | None:
    """Item whose gold flips under a horizontal mirror; None if the answer does not depend on left/right."""
    k = lr_kind(item)
    if k is None:
        return None
    if k == "options":
        ow = {kk: v.lower() for kk, v in vx.option_words(item).items()}
        target = {"left": "right", "right": "left"}[ow[item["gold"]]]
        g = [kk for kk, v in ow.items() if v == target][0]
    else:
        g = "B" if item["gold"] == "A" else "A"
    return {**item, "gold": g}


def main() -> None:
    """CLI entry point."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--variants", default="gray,none,cfq,mirror")
    ap.add_argument("--tag", default="", help="output suffix, e.g. polar")
    args = ap.parse_args()
    t0 = time.time()
    leg = vx.legacy_e1(args.model)
    items = [it for it in vx.load_items(limit=args.limit) if it["uid"] in leg]
    llm = vx.make_llm(args.model)
    is_molmo = args.model.startswith("molmo")
    real = [vx.img_ref(C.get_image(it)) for it in items]
    recs = []
    for var in args.variants.split(","):
        if var == "none" and is_molmo:
            continue
        if var in ("gray", "none"):
            sel = list(range(len(items)))
            gen_items = [items[i] for i in sel]
            refs = [vx.img_ref(vx.image_variant(items[i], "gray")) for i in sel] if var == "gray" else [real[i] for i in sel]
        elif var in ("cfq", "polar"):
            fn = cf_question if var == "cfq" else polar_question
            sel = [i for i, it in enumerate(items) if fn(it)]
            gen_items = [fn(items[i]) for i in sel]
            refs = [real[i] for i in sel]
        else:
            sel = [i for i, it in enumerate(items) if mirror_item(it)]
            gen_items = [mirror_item(items[i]) for i in sel]
            refs = [vx.img_ref(vx.image_variant(items[i], "mirror")) for i in sel]
        if not sel:
            continue
        msgs = [C.cot_messages(it, r) for it, r in zip(gen_items, refs)]
        if var == "none":
            msgs = [vx.strip_images(m) for m in msgs]
        gens = vx.generate(llm, args.model, msgs)
        cots = [C.COT_PREFIX + " " + g["text"] for g in gens]
        print(f"[e2a] {var}: generated {len(sel)} R ({time.time() - t0:.0f}s)", flush=True)
        if var in ("gray", "none"):
            ro_msgs = [C.build_messages(items[i], "cot", "none", cots[k], real[i]) for k, i in enumerate(sel)]
            ro = vx.readout(llm, args.model, ro_msgs, [items[i]["choices"] for i in sel])
            ro_direct = [None] * len(sel)
        else:
            ro_msgs = [C.build_messages(gi, "cot", "none", cots[k], refs[k]) for k, gi in enumerate(gen_items)]
            ro = vx.readout(llm, args.model, ro_msgs, [gi["choices"] for gi in gen_items])
            d_msgs = [C.build_messages(gi, "direct", "none", "", refs[k]) for k, gi in enumerate(gen_items)]
            ro_direct = vx.readout(llm, args.model, d_msgs, [gi["choices"] for gi in gen_items])
        for k, i in enumerate(sel):
            gi = gen_items[k]
            recs.append({"model": args.model, "task": items[i]["task"], "item_id": items[i]["uid"],
                         "variant": var, "gold": gi["gold"], "gold_orig": items[i]["gold"],
                         "letters": gi["choices"], "image_id": items[i]["group"], "question": gi["question"],
                         "cot": cots[k], "cot_len": gens[k]["len"], "cot_trunc": gens[k]["trunc"],
                         "stated": C.parse_free_answer(cots[k], gi["choices"]),
                         "readout_cot": ro[k], "readout_direct": ro_direct[k]})
        print(f"[e2a] {var}: readouts done ({time.time() - t0:.0f}s)", flush=True)
    sfx = (f"_{args.tag}" if args.tag else "") + (f"_limit{args.limit}" if args.limit else "")
    vx.write_jsonl(f"{vx.OUT}/e2a/{args.model}{sfx}.jsonl", recs)
    print(f"[e2a] wrote {len(recs)} records ({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
