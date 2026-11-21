"""E4: Qwen3-VL-8B Instruct / Thinking under the original protocol (vLLM >= 0.11, V1 engine).

Instruct: greedy CoT R (continuing "Let's think step by step.", <=512 tokens) and the core matrix
  direct, cot, cot+turn, filler, filler+turn, filler512, cot+restate, cot+reimage, cot+reimage_restate.
Thinking: native thinking T (generation prompt opens <think>, <=4096 tokens); readouts
  direct (empty think block), think (T closed with </think> then "Answer: ("), and filler(|T|)+turn.
Output mirrors the legacy e1 JSON schema (top-20 letter scores) at outputs/e4/<model>.json so that
e2a / e3x / e2b can reuse it; image budget aligned to Qwen2.5-VL (max_pixels = 1280*28*28).

Usage: python -m experiments.behavior.qwen3vl_main_pool --model qwen3vl_8b [--limit 5]
"""
from __future__ import annotations

import argparse
import json
import math
import os
import time

from srww import vllm_utils as vx
from srww.vllm_utils import C

CORE = [("direct", "none"), ("cot", "none"), ("cot", "turn"), ("filler", "none"), ("filler", "turn"),
        ("filler512", "none"), ("cot", "restate"), ("cot", "reimage"), ("cot", "reimage_restate")]
MAX_PIXELS = 1280 * 28 * 28


def make_llm_v1(model: str):
    from vllm import LLM
    return LLM(model=C.model_path(model), trust_remote_code=True, dtype="bfloat16", max_model_len=16384,
               gpu_memory_utilization=0.85, seed=0, limit_mm_per_prompt={"image": 2}, enable_prefix_caching=True,
               mm_processor_kwargs={"max_pixels": MAX_PIXELS}, max_logprobs=20)


def scores_of(r, choices: list[str]) -> dict:
    top = {lp.decoded_token: lp.logprob for lp in r.outputs[0].logprobs[0].values()}
    sc = C.letter_scores(top, choices)
    return {k: (v if v > -math.inf else None) for k, v in sc.items()}


def main() -> None:
    """CLI entry point."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="qwen3vl_8b | qwen3vl_8b_think | internvl35_8b (any V1 model)")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--max-think", type=int, default=4096)
    args = ap.parse_args()
    from vllm import SamplingParams
    t0 = time.time()
    items = vx.load_items(limit=args.limit)
    llm = vx.make_llm(args.model)
    tok = llm.get_tokenizer()
    refs = [vx.img_ref(C.get_image(it)) for it in items]
    kw = dict(add_generation_prompt=False, continue_final_message=True, use_tqdm=True)
    sp1 = SamplingParams(temperature=0.0, max_tokens=1, logprobs=20)
    recs = [{k: v for k, v in it.items() if k not in ("image", "image_path")} | {"scores": {}} for it in items]
    if not args.model.endswith("_think"):
        gen = llm.chat([C.cot_messages(it, r) for it, r in zip(items, refs)],
                       SamplingParams(temperature=0.0, max_tokens=512), **kw)
        cots = [C.COT_PREFIX + " " + g.outputs[0].text.strip() for g in gen]
        lens = [len(g.outputs[0].token_ids) for g in gen]
        trunc = [g.outputs[0].finish_reason == "length" for g in gen]
        print(f"[e4] CoT generated ({time.time() - t0:.0f}s)", flush=True)
        fillers = [C.filler_text(tok, max(n, 1)) for n in lens]
        f512 = C.filler_text(tok, 512)
        for base, suffix in CORE:
            ctx = {"direct": [""] * len(items), "cot": cots, "filler": fillers, "filler512": [f512] * len(items)}[base]
            msgs = [C.build_messages(it, base, suffix, ctx[i], refs[i]) for i, it in enumerate(items)]
            res = llm.chat(msgs, sp1, **kw)
            for rec, it, r in zip(recs, items, res):
                rec["scores"][C.cond_name(base, suffix)] = scores_of(r, it["choices"])
            print(f"[e4] {C.cond_name(base, suffix)} done ({time.time() - t0:.0f}s)", flush=True)
    else:
        # Thinking: build prompts as text from the chat template (generation prompt opens the think block).
        def base_prompt(it: dict) -> str:
            m = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": it["question"]}]}]
            return tok.apply_chat_template(m, tokenize=False, add_generation_prompt=True)

        prompts = [base_prompt(it) for it in items]
        opens_think = prompts[0].rstrip().endswith("<think>")
        pre = "" if opens_think else "<think>\n"
        imgs = [C.get_image(it) for it in items]
        gen = llm.generate([{"prompt": p + pre, "multi_modal_data": {"image": im}} for p, im in zip(prompts, imgs)],
                           SamplingParams(temperature=0.0, max_tokens=args.max_think, stop=["</think>"]))
        cots = [g.outputs[0].text.strip() for g in gen]
        lens = [len(g.outputs[0].token_ids) for g in gen]
        trunc = [g.outputs[0].finish_reason == "length" for g in gen]
        print(f"[e4] thinking generated ({time.time() - t0:.0f}s), opens_think={opens_think}", flush=True)
        fillers = [C.filler_text(tok, max(n, 1)) for n in lens]
        conds = {
            "direct": [p + pre + "\n</think>\n\n" + C.FORCE_PREFIX for p in prompts],
            "think": [p + pre + t + "\n</think>\n\n" + C.FORCE_PREFIX for p, t in zip(prompts, cots)],
        }
        for name, texts in conds.items():
            res = llm.generate([{"prompt": t, "multi_modal_data": {"image": im}} for t, im in zip(texts, imgs)], sp1)
            for rec, it, r in zip(recs, items, res):
                rec["scores"][name] = scores_of(r, it["choices"])
            print(f"[e4] {name} done ({time.time() - t0:.0f}s)", flush=True)
        for name, ctx in (("cot+turn", cots), ("filler+turn", fillers)):
            # T (or filler of |T| tokens) as a closed assistant turn, then the final-answer turn
            msgs = [C.build_messages(it, "cot", "turn", ctx[i], refs[i]) for i, it in enumerate(items)]
            res = llm.chat(msgs, sp1, **kw)
            for rec, it, r in zip(recs, items, res):
                rec["scores"][name] = scores_of(r, it["choices"])
            print(f"[e4] {name} done ({time.time() - t0:.0f}s)", flush=True)
    for i, rec in enumerate(recs):
        rec.update(cot=cots[i], cot_len=lens[i], cot_truncated=trunc[i],
                   cot_free=C.parse_free_answer(cots[i], items[i]["choices"]))
    sfx = f"_limit{args.limit}" if args.limit else ""
    os.makedirs(f"{vx.OUT}/e4", exist_ok=True)
    json.dump({"model": args.model, "conditions": list(recs[0]["scores"]), "items": recs},
              open(f"{vx.OUT}/e4/{args.model}{sfx}.json", "w"))
    print(f"[e4] wrote ({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
