"""E1: behavioural factorial (vLLM).

Stage 1: greedy CoT generation R (continuing "Let's think step by step.").
Stage 2: single-token forced answer ("Answer: (") logprobs under all CONDITIONS.
Writes one JSON per model with per-item records.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(__file__))
import common as C  # noqa: E402

MOLMO_TEMPLATE = (
    "{% for m in messages %}{% if m['role'] == 'user' %}{{ '' if loop.first else ' ' }}User: {{ m['content'] if m['content'] is string "
    "else (m['content'] | selectattr('type','equalto','text') | map(attribute='text') | join('\n')) }}"
    "{% else %} Assistant: {{ m['content'] }}{% endif %}{% endfor %}"
)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=list(C.MODELS))
    ap.add_argument("--datasets", default="cvb_relation,cvb_depth,cvb_distance,cvb_count,syn_relpos,syn_depth,mmstar")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--max_cot_tokens", type=int, default=512)
    ap.add_argument("--out", default=None)
    ap.add_argument("--cot_file", default=None, help="save CoT here (if absent) or load it (if present)")
    ap.add_argument("--wait_cot", action="store_true", help="wait for --cot_file written by another process")
    ap.add_argument("--tp", type=int, default=1)
    ap.add_argument("--part", default=None, help="k/n: compute only every n-th condition starting at k")
    args = ap.parse_args()

    import torch
    torch.backends.cuda.matmul.allow_tf32 = True
    from vllm import LLM, SamplingParams

    out = args.out or f"{C.PROJ}/results/v2/e1/{args.model}{'_pilot' if args.limit else ''}.json"
    os.makedirs(os.path.dirname(out), exist_ok=True)
    is_molmo = args.model.startswith("molmo")
    is_internvl = args.model.startswith("internvl")

    kw: dict = dict(model=C.model_path(args.model), trust_remote_code=True, dtype="bfloat16",
                    max_model_len=12288, gpu_memory_utilization=0.85, seed=0,
                    limit_mm_per_prompt={"image": 1 if is_molmo else 2}, enable_prefix_caching=True,
                    tensor_parallel_size=args.tp)
    if args.model.startswith("qwen"):
        kw["mm_processor_kwargs"] = {"max_pixels": 1280 * 28 * 28}
    if is_internvl:
        kw["mm_processor_kwargs"] = {"max_dynamic_patch": 6}
    if args.model == "phi35v":
        kw["mm_processor_kwargs"] = {"num_crops": 4}
    if is_molmo:
        kw["max_model_len"] = 4096
    llm = LLM(**kw)
    tok = llm.get_tokenizer()
    chat_kw = dict(add_generation_prompt=False, continue_final_message=True, use_tqdm=True)
    if is_molmo:
        chat_kw["chat_template"] = MOLMO_TEMPLATE

    items = C.load_items(args.datasets.split(","), args.limit)
    print(f"[e1] {args.model}: {len(items)} items", flush=True)
    img_refs = [{"type": "image_url", "image_url": {"url": C.image_data_url(C.get_image(it))}} for it in items]

    t0 = time.time()
    if args.cot_file and args.wait_cot:
        while not os.path.exists(args.cot_file + ".done"):
            time.sleep(20)
    if args.cot_file and os.path.exists(args.cot_file + ".done"):
        cd = json.load(open(args.cot_file))
        assert cd["uids"] == [it["uid"] for it in items]
        cots, cot_len, truncated = cd["cots"], cd["cot_len"], cd["truncated"]
    else:
        gen = llm.chat([C.cot_messages(it, r) for it, r in zip(items, img_refs)],
                       SamplingParams(temperature=0.0, max_tokens=args.max_cot_tokens), **chat_kw)
        cots = [C.COT_PREFIX + " " + g.outputs[0].text.strip() for g in gen]
        cot_len = [len(g.outputs[0].token_ids) for g in gen]
        truncated = [g.outputs[0].finish_reason == "length" for g in gen]
        if args.cot_file:
            json.dump({"uids": [it["uid"] for it in items], "cots": cots, "cot_len": cot_len, "truncated": truncated},
                      open(args.cot_file, "w"))
            open(args.cot_file + ".done", "w").close()
    fillers = [C.filler_text(tok, max(n, 1)) for n in cot_len]
    print(f"[e1] CoT done in {time.time()-t0:.0f}s", flush=True)

    conds = [c for c in C.CONDITIONS if not (is_molmo and ("reimage" in c[1] or c[0] == "filler1024"))]
    if args.part:
        k, n = map(int, args.part.split("/"))
        conds = conds[k::n]
    recs = [{k: v for k, v in it.items() if k not in ("image",)} | {
        "cot": cots[i], "cot_len": cot_len[i], "cot_truncated": truncated[i],
        "cot_free": C.parse_free_answer(cots[i], it["choices"]), "scores": {}} for i, it in enumerate(items)]

    sp = SamplingParams(temperature=0.0, max_tokens=1, logprobs=20)
    for base, suffix in conds:
        name = C.cond_name(base, suffix)
        ctx = {"direct": [""] * len(items), "cot": cots, "filler": fillers,
               "filler512": [C.filler_text(tok, 512)] * len(items),
               "filler1024": [C.filler_text(tok, 1024)] * len(items)}[base]
        msgs = [C.build_messages(it, base, suffix, ctx[i], img_refs[i]) for i, it in enumerate(items)]
        res = llm.chat(msgs, sp, **chat_kw)
        for i, r in enumerate(res):
            top = {lp.decoded_token: lp.logprob for lp in r.outputs[0].logprobs[0].values()}
            sc = C.letter_scores(top, items[i]["choices"])
            recs[i]["scores"][name] = sc
        print(f"[e1] {name} done ({time.time()-t0:.0f}s)", flush=True)

    json.dump({"model": args.model, "conditions": [C.cond_name(*c) for c in conds], "items": recs},
              open(out, "w"), default=lambda x: None if x != x else x)
    print(f"[e1] wrote {out} in {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
