"""E5: edit the content of the model's own CoT text R (vLLM, forced single-token answer).

cot_noans : R with its concluding answer sentence(s) removed.
cot_swap  : cot_noans + "Therefore, the answer is (Y) <option>." where Y differs from the answer R stated
            (2-choice items with a parsed R answer only).
cot_noans+restate : cot_noans followed by the restate turn.
Tests whether the CoT loss is carried by the stated answer (copying) or by the preceding reasoning.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import time

from srww import protocol as C  # noqa: E402
from srww.rcut import strip_answer  # noqa: E402,F401


def option_words(item: dict) -> dict[str, str]:
    out = {}
    for line in item["question"].split("\n"):
        m = re.match(r"\(([A-D])\)\s*(.+)", line.strip())
        if m:
            out[m.group(1)] = m.group(2).strip()
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    args = ap.parse_args()
    from vllm import LLM, SamplingParams
    from experiments.behavior.behavior_factorial import MOLMO_TEMPLATE

    e1 = json.load(open(f"{C.PROJ}/results/v2/e1/{args.model}.json"))
    recs = {it["uid"]: it for it in e1["items"]}
    names = ["cvb_relation", "cvb_depth", "cvb_distance", "cvb_count", "syn_relpos", "syn_depth", "mmstar"]
    items = [it for it in C.load_items(names) if it["uid"] in recs]
    kw: dict = dict(model=C.model_path(args.model), trust_remote_code=True, dtype="bfloat16", max_model_len=12288,
                    gpu_memory_utilization=0.85, seed=0, limit_mm_per_prompt={"image": 1}, enable_prefix_caching=True)
    if args.model.startswith("qwen"):
        kw["mm_processor_kwargs"] = {"max_pixels": 1280 * 28 * 28}
    if args.model.startswith("internvl"):
        kw["mm_processor_kwargs"] = {"max_dynamic_patch": 6}
    if args.model == "phi35v":
        kw["mm_processor_kwargs"] = {"num_crops": 4}
    if args.model.startswith("molmo"):
        kw["max_model_len"] = 4096
    llm = LLM(**kw)
    chat_kw = dict(add_generation_prompt=False, continue_final_message=True, use_tqdm=True)
    if args.model.startswith("molmo"):
        chat_kw["chat_template"] = MOLMO_TEMPLATE
    refs = [{"type": "image_url", "image_url": {"url": C.image_data_url(C.get_image(it))}} for it in items]

    ctx = {"cot_noans": [], "cot_swap": []}
    swap_ok = []
    for it in items:
        r = recs[it["uid"]]
        na = strip_answer(r["cot"])
        ctx["cot_noans"].append(na)
        ra = r["cot_free"]
        ow = option_words(it)
        if len(it["choices"]) == 2 and ra in it["choices"]:
            y = [c for c in it["choices"] if c != ra][0]
            ctx["cot_swap"].append(f"{na} Therefore, the answer is ({y}) {ow.get(y, '')}.".replace(" .", "."))
            swap_ok.append(y)
        else:
            ctx["cot_swap"].append(None); swap_ok.append(None)

    sp = SamplingParams(temperature=0.0, max_tokens=1, logprobs=20)
    out = {it["uid"]: {"noans_len_frac": len(ctx["cot_noans"][i]) / max(len(recs[it["uid"]]["cot"]), 1),
                       "noans_has_letter": bool(re.search(r"\([A-D]\)", ctx["cot_noans"][i])),
                       "swap_to": swap_ok[i], "scores": {}} for i, it in enumerate(items)}
    t0 = time.time()
    for name, base_key, suffix in [("cot_noans", "cot_noans", "none"), ("cot_noans+restate", "cot_noans", "restate"),
                                   ("cot_swap", "cot_swap", "none")]:
        idx = [i for i in range(len(items)) if ctx[base_key][i] is not None]
        msgs = [C.build_messages(items[i], "cot", suffix, ctx[base_key][i], refs[i]) for i in idx]
        res = llm.chat(msgs, sp, **chat_kw)
        for i, r in zip(idx, res):
            top = {lp.decoded_token: lp.logprob for lp in r.outputs[0].logprobs[0].values()}
            out[items[i]["uid"]]["scores"][name] = C.letter_scores(top, items[i]["choices"])
        print(f"[e5] {name} done ({time.time()-t0:.0f}s)", flush=True)
    os.makedirs(f"{C.PROJ}/results/v2/e5", exist_ok=True)
    json.dump({"model": args.model, "items": out, "example_noans": ctx["cot_noans"][:5], "example_swap": ctx["cot_swap"][:5]},
              open(f"{C.PROJ}/results/v2/e5/{args.model}.json", "w"))
    print("[e5] done", flush=True)


if __name__ == "__main__":
    main()
