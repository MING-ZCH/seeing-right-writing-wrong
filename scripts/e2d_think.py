"""E2d for Qwen3-VL-8B-Thinking: occlusion twins with native thinking, both phrasings.

Direct = empty think block then "Answer: ("; think = native <think> trace (<= 4096 tokens) closed with
</think> then "Answer: (". Same records as e2d.py (condition direct / cot), written to
outputs/e2d/qwen3vl_8b_think.jsonl and _behind.jsonl.

Usage: python e2d_think.py [--polar] [--limit 5]
"""
from __future__ import annotations

import argparse
import math
import time

import vx
from vx import C
from e2d import load_occ


def main() -> None:
    """CLI entry point."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--polar", action="store_true")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--max-think", type=int, default=4096)
    ap.add_argument("--items-dir", default=None, help="occlusion item directory (default: original set)")
    ap.add_argument("--out-dir", default=f"{vx.OUT}/e2d", help="output directory")
    args = ap.parse_args()
    from vllm import SamplingParams
    t0 = time.time()
    model = "qwen3vl_8b_think"
    items = load_occ(args.limit, args.polar, args.items_dir)
    llm = vx.make_llm(model)
    tok = llm.get_tokenizer()
    prompts = [tok.apply_chat_template([{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": it["question"]}]}],
                                       tokenize=False, add_generation_prompt=True) for it in items]
    pre = "" if prompts[0].rstrip().endswith("<think>") else "<think>\n"
    imgs = [C.get_image(it) for it in items]
    gen = llm.generate([{"prompt": p + pre, "multi_modal_data": {"image": im}} for p, im in zip(prompts, imgs)],
                       SamplingParams(temperature=0.0, max_tokens=args.max_think, stop=["</think>"]))
    thinks = [g.outputs[0].text.strip() for g in gen]
    sp1 = SamplingParams(temperature=0.0, max_tokens=1, logprobs=20)
    recs = []
    for cond, texts in (("direct", [p + pre + "\n</think>\n\n" + C.FORCE_PREFIX for p in prompts]),
                        ("cot", [p + pre + t + "\n</think>\n\n" + C.FORCE_PREFIX for p, t in zip(prompts, thinks)])):
        res = llm.generate([{"prompt": t, "multi_modal_data": {"image": im}} for t, im in zip(texts, imgs)], sp1)
        for i, (it, r) in enumerate(zip(items, res)):
            top = {lp.decoded_token: lp.logprob for lp in r.outputs[0].logprobs[0].values()}
            sc = C.letter_scores(top, it["choices"])
            sc = {k: (v if v > -math.inf else None) for k, v in sc.items()}
            rec = {"model": model, "task": it["task"], "item_id": it["uid"], "gold": it["gold"], "letters": it["choices"],
                   "image_id": it["group"], "meta": it["meta"], "condition": cond, "top20": sc}
            if cond == "cot":
                rec.update(text=thinks[i], gen_len=len(gen[i].outputs[0].token_ids))
            recs.append(rec)
    sfx = ("_behind" if args.polar else "") + (f"_limit{args.limit}" if args.limit else "")
    vx.write_jsonl(f"{args.out_dir}/{model}{sfx}.jsonl", recs)
    print(f"[e2d_think] wrote {len(recs)} records ({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
