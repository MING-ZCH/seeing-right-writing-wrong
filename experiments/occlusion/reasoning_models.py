"""Exp-48: reasoning-tuned open VLMs from other families (MiMo-VL-7B-RL, GLM-4.1V-9B-Thinking).

Protocol = Exp-31 / experiments/occlusion/twins_readout_thinking.py (Qwen3-VL-8B-Thinking), applied to each model's own chat template:
  direct  generation prompt + empty think block + "Answer: (" (neither template has a thinking switch, so
          "thinking disabled" = empty think block, as for Qwen3-VL-8B-Thinking).
  think   generation prompt + think opener; native trace, greedy, <= 4096 tokens, stop at "</think>"; the block is
          closed and the answer read at "Answer: (".
  cot     empty think block + assistant prefix "Let's think step by step." (e10 protocol: greedy <= 512 tokens,
          same-turn readout "<prefix> <continuation>\\nAnswer: (").
GLM-4.1V writes "<think>...</think><answer>...</answer>"; its answer opener "<answer>" is placed after the think block
in every arm. Readout: MiMo-VL (Qwen2.5-VL architecture) on vLLM 0.8.4 V0 with exact full-vocabulary letter scores
(vx.ExactReadout, like qwen7b); GLM-4.1V on vLLM 0.11 V1, top-20 letter scores (like qwen3vl_8b_think).
Sets: cues (1,600 = 800 cue-factorial images x 2 phrasings), twins (800 = original occlusion twins x 2 phrasings),
nat (BLINK Relative Depth val 124 + CV-Bench 3D Depth 600; e44_prompt_fix.load_set("nat")).
Records use the e10 cue layout (item_id with _behind, base_id, phrasing, question, condition) ->
outputs/e48/<model>_<set>.jsonl.

Usage: python -m experiments.occlusion.reasoning_models --model mimo_vl_7b_rl --set cues [--limit 4] [--arms direct,think,cot]
"""
from __future__ import annotations
from srww import paths as _vtp  # noqa: E402

import argparse
import json
import math
import os
import time

from srww import vllm_utils as vx
from srww.vllm_utils import C

C.MODELS["mimo_vl_7b_rl"] = f"{vx.SHARE}/models/MiMo-VL-7B-RL"
C.MODELS["glm41v_9b_think"] = f"{vx.SHARE}/models/GLM-4.1V-9B-Thinking"
FMT = {
    # think opener, empty think block, think closer, answer opener
    "mimo_vl_7b_rl": dict(open="<think>\n", empty="<think>\n\n</think>\n\n", close="\n</think>\n\n", ans=""),
    "glm41v_9b_think": dict(open="<think>", empty="<think></think><answer>", close="</think><answer>", ans=""),
}
V1 = {"glm41v_9b_think"}


def make_llm(model: str):
    """vLLM engine with the vx.make_llm settings of the matching architecture."""
    import torch
    torch.backends.cuda.matmul.allow_tf32 = True
    util = float(os.environ.get("VX_GPU_UTIL", "0.85"))
    if model in V1:
        os.environ["VLLM_USE_V1"] = "1"
        # vLLM's flash-attn kernel for the GLM-4.1V ViT fails on this driver ("PTX ... unsupported toolchain");
        # run the ViT on torch SDPA (patch must live in this process, as for qwen3vl_2b/4b in vx.make_llm).
        os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
        import vllm.model_executor.models.glm4_1v as _g
        from vllm.platforms import _Backend
        _g.get_vit_attn_backend = lambda *a, **k: _Backend.TORCH_SDPA
        _g.check_upstream_fa_availability = lambda *a, **k: False
        from vllm import LLM
        return LLM(model=C.model_path(model), trust_remote_code=True, dtype="bfloat16", max_model_len=16384,
                   gpu_memory_utilization=util, seed=0, limit_mm_per_prompt={"image": 1, "video": 0},
                   enable_prefix_caching=True, max_logprobs=20)
    os.environ["VLLM_USE_V1"] = "0"
    from vllm import LLM
    return LLM(model=C.model_path(model), trust_remote_code=True, dtype="bfloat16", max_model_len=12288,
               gpu_memory_utilization=util, seed=0, limit_mm_per_prompt={"image": 1}, enable_prefix_caching=True,
               max_logprobs=20, mm_processor_kwargs={"max_pixels": 1280 * 28 * 28})


def load(name: str, limit: int | None) -> list[dict]:
    """Items with key / phrasing / set (cue and twin layouts as experiments/interventions/structured_state.py --set cues)."""
    from experiments.occlusion.twins_readout import load_occ, ITEMS
    if name in ("cues", "twins"):
        d = os.environ.get("CUES_DIR", f"{_vtp.ITEMS}/occ_cues") if name == "cues" else ITEMS
        items = [dict(it, phrasing="front", key=it["uid"]) for it in load_occ(limit, items_dir=d)]
        items += [dict(it, phrasing="behind", key=it["uid"] + "_behind") for it in load_occ(limit, polar=True, items_dir=d)]
        return [dict(it, set=name) for it in items]
    from experiments.interventions.prompt_variants import load_set
    return load_set("nat", limit)


def readout(llm, model: str, prompts: list[str], imgs: list, choices: list[list[str]]) -> list[dict]:
    """One-token readout after prompts ending in "Answer: (": top-20 letter scores (+ exact scores under V0)."""
    from vllm import SamplingParams
    inp = [{"prompt": p, "multi_modal_data": {"image": im}} for p, im in zip(prompts, imgs)]
    if model in V1:
        res = llm.generate(inp, SamplingParams(temperature=0.0, max_tokens=1, logprobs=20))
        ex = None
    else:
        ex = vx.ExactReadout(llm.get_tokenizer())
        res = llm.generate(inp, [ex.sampling_params(i) for i in range(len(inp))])
    out = []
    for i, r in enumerate(res):
        top = {lp.decoded_token: lp.logprob for lp in r.outputs[0].logprobs[0].values()}
        t20 = C.letter_scores(top, choices[i])
        mass = sum(math.exp(v) for v in t20.values() if v > -math.inf)
        o = {"top20": {k: (v if v > -math.inf else None) for k, v in t20.items()}, "top20_letter_mass": round(mass, 5),
             "top1_token": max(top, key=top.get) if top else None}
        if ex is not None:
            o |= ex.decode(i, choices[i])
        out.append(o)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=list(FMT))
    ap.add_argument("--set", required=True, choices=["cues", "twins", "nat"])
    ap.add_argument("--arms", default="direct,think,cot")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--max-think", type=int, default=4096)
    ap.add_argument("--max-cot", type=int, default=512)
    ap.add_argument("--out-dir", default=f"{vx.OUT}/e48")
    ap.add_argument("--probe", action="store_true", help="print raw native generations (special tokens kept) and exit")
    args = ap.parse_args()
    from vllm import SamplingParams
    t0 = time.time()
    f = FMT[args.model]
    items = load(args.set, args.limit)
    print(f"[e48] {args.model} {args.set}: {len(items)} items", flush=True)
    llm = make_llm(args.model)
    tok = llm.get_tokenizer()
    # MiMo-VL: the tokenizer carries a text-only template; the multimodal one is in chat_template.json
    ctj = os.path.join(C.model_path(args.model), "chat_template.json")
    tmpl = json.load(open(ctj))["chat_template"] if os.path.exists(ctj) else None
    prompts = [tok.apply_chat_template([{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": it["question"]}]}],
                                       tokenize=False, add_generation_prompt=True, chat_template=tmpl) for it in items]
    print("[e48] prompt[0] repr:", repr(prompts[0][-200:]), flush=True)
    imgs = [C.get_image(it) for it in items]
    ch = [it["choices"] for it in items]
    if args.probe:
        gen = llm.generate([{"prompt": p, "multi_modal_data": {"image": im}} for p, im in zip(prompts, imgs)],
                           SamplingParams(temperature=0.0, max_tokens=3000, skip_special_tokens=False))
        for it, g in zip(items, gen):
            print(f"[probe] {it['key']} gold {it['gold']} len {len(g.outputs[0].token_ids)} finish {g.outputs[0].finish_reason}\n"
                  f"{g.outputs[0].text!r}\n", flush=True)
        return
    arms = args.arms.split(",")
    stop_extra = ["</answer>"] if args.model.startswith("glm") else []
    ctx: dict[str, list[str]] = {}
    gen_info: dict[str, list[dict]] = {}
    if "direct" in arms:
        ctx["direct"] = [p + f["empty"] + f["ans"] + C.FORCE_PREFIX for p in prompts]
    if "think" in arms:
        gen = llm.generate([{"prompt": p + f["open"], "multi_modal_data": {"image": im}} for p, im in zip(prompts, imgs)],
                           SamplingParams(temperature=0.0, max_tokens=args.max_think, stop=["</think>"]))
        texts = [g.outputs[0].text.strip() for g in gen]
        ctx["think"] = [p + f["open"] + t + f["close"] + f["ans"] + C.FORCE_PREFIX for p, t in zip(prompts, texts)]
        gen_info["think"] = [{"text": t, "gen_len": len(g.outputs[0].token_ids), "trunc": g.outputs[0].finish_reason == "length"}
                             for t, g in zip(texts, gen)]
        print(f"[e48] think generated ({time.time() - t0:.0f}s)", flush=True)
    if "cot" in arms:
        pre = [p + f["empty"] + f["ans"] + C.COT_PREFIX for p in prompts]
        gen = llm.generate([{"prompt": p, "multi_modal_data": {"image": im}} for p, im in zip(pre, imgs)],
                           SamplingParams(temperature=0.0, max_tokens=args.max_cot, stop=stop_extra or None))
        raw = [g.outputs[0].text for g in gen]
        cots = [C.COT_PREFIX + " " + t.strip() for t in raw]
        ctx["cot"] = [p + f["empty"] + f["ans"] + c + "\n" + C.FORCE_PREFIX for p, c in zip(prompts, cots)]
        gen_info["cot"] = [{"text": c, "gen_len": len(g.outputs[0].token_ids), "trunc": g.outputs[0].finish_reason == "length"}
                           for c, g in zip(cots, gen)]
        print(f"[e48] cot generated ({time.time() - t0:.0f}s)", flush=True)
    recs = []
    for cond in arms:
        outs = readout(llm, args.model, ctx[cond], imgs, ch)
        for i, (it, o) in enumerate(zip(items, outs)):
            rec = {"model": args.model, "set": it["set"], "task": it["task"], "item_id": it["key"], "base_id": it["uid"],
                   "phrasing": it["phrasing"], "condition": cond, "gold": it["gold"], "letters": it["choices"],
                   "image_id": it["group"], "question": it["question"], "meta": it.get("meta"), **o}
            if cond in gen_info:
                rec |= gen_info[cond][i]
            recs.append(rec)
        print(f"[e48] {cond} read ({time.time() - t0:.0f}s)", flush=True)
    sfx = f"_limit{args.limit}" if args.limit else ""
    os.makedirs(args.out_dir, exist_ok=True)
    vx.write_jsonl(f"{args.out_dir}/{args.model}_{args.set}{sfx}.jsonl", recs)
    json.dump({"model": args.model, "set": args.set, "n": len(items), "arms": arms, "seconds": round(time.time() - t0),
               "model_path": C.model_path(args.model), "fmt": f, "max_think": args.max_think, "max_cot": args.max_cot},
              open(f"{args.out_dir}/{args.model}_{args.set}{sfx}.timing.json", "w"), indent=1)
    print(f"[e48] wrote {len(recs)} records ({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
