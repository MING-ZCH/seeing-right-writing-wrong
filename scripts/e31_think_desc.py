"""E31: question-blind description arm on the cue-factorial set for Qwen3-VL-8B-Thinking (e28 desc arm, Thinking layout).

Turn 1 = image + e5x.DESC_PROMPT; the generation prompt opens the native think block; the trace (<= 4096 tokens, stop at
</think>) is closed with </think> and the description is generated greedily (<= 512 tokens). Generated once per image
(800) and shared by both phrasings, as in e28_run.py. Turn 2 = the question as a new user turn; the chat template drops
the turn-1 trace, so the answer step sees image + description only; readout as the e2d_think Direct arm (empty think
block, then "Answer: (", top-20 letter scores). Direct is re-read on the same engine (sanity vs outputs/e31).
-> outputs/e31/qwen3vl_8b_think_cues_desc.jsonl

Usage: python e31_think_desc.py [--limit 5]
"""
from __future__ import annotations

import argparse
import math
import os
import time

import vx
from vx import C
from e2d import load_occ
from e5x import DESC_PROMPT

MODEL = "qwen3vl_8b_think"


def main() -> None:
    """CLI entry point."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--max-think", type=int, default=4096)
    ap.add_argument("--out-dir", default=f"{vx.OUT}/e31")
    args = ap.parse_args()
    from vllm import SamplingParams
    t0 = time.time()
    d = os.environ.get("CUES_DIR", "/dev/shm/occ_cues")
    items = [dict(it, phrasing="front", key=it["uid"]) for it in load_occ(args.limit, items_dir=d)]
    items += [dict(it, phrasing="behind", key=it["uid"] + "_behind") for it in load_occ(args.limit, polar=True, items_dir=d)]
    llm = vx.make_llm(MODEL)
    tok = llm.get_tokenizer()
    imgs = {it["uid"]: C.get_image(it) for it in items}
    order = list(dict.fromkeys(it["uid"] for it in items))
    u_img = {"type": "image"}
    p1 = tok.apply_chat_template([{"role": "user", "content": [u_img, {"type": "text", "text": DESC_PROMPT}]}],
                                 tokenize=False, add_generation_prompt=True)
    pre = "" if p1.rstrip().endswith("<think>") else "<think>\n"
    g1 = llm.generate([{"prompt": p1 + pre, "multi_modal_data": {"image": imgs[u]}} for u in order],
                      SamplingParams(temperature=0.0, max_tokens=args.max_think, stop=["</think>"]))
    thinks = [g.outputs[0].text.strip() for g in g1]
    print(f"[e31d] think done ({time.time() - t0:.0f}s)", flush=True)
    g2 = llm.generate([{"prompt": p1 + pre + t + "\n</think>\n\n", "multi_modal_data": {"image": imgs[u]}} for u, t in zip(order, thinks)],
                      SamplingParams(temperature=0.0, max_tokens=512))
    desc = {u: {"text": b.outputs[0].text.strip(), "len": len(b.outputs[0].token_ids), "trunc": b.outputs[0].finish_reason == "length",
                "think": t, "think_len": len(a.outputs[0].token_ids), "think_trunc": a.outputs[0].finish_reason == "length"}
            for u, t, a, b in zip(order, thinks, g1, g2)}
    print(f"[e31d] descriptions done ({time.time() - t0:.0f}s)", flush=True)
    sp1 = SamplingParams(temperature=0.0, max_tokens=1, logprobs=20)
    close = "\n</think>\n\n" + C.FORCE_PREFIX
    ro_desc, ro_dir = [], []
    for it in items:
        m = [{"role": "user", "content": [u_img, {"type": "text", "text": DESC_PROMPT}]},
             {"role": "assistant", "content": desc[it["uid"]]["text"]},
             {"role": "user", "content": [{"type": "text", "text": it["question"]}]}]
        p2 = tok.apply_chat_template(m, tokenize=False, add_generation_prompt=True)
        ro_desc.append(p2 + ("" if p2.rstrip().endswith("<think>") else "<think>\n") + close)
        pd = tok.apply_chat_template([{"role": "user", "content": [u_img, {"type": "text", "text": it["question"]}]}],
                                     tokenize=False, add_generation_prompt=True)
        ro_dir.append(pd + ("" if pd.rstrip().endswith("<think>") else "<think>\n") + close)
    print(f"[e31d] readout prompt example:\n{ro_desc[0][-600:]}", flush=True)
    recs = []
    for cond, texts in (("desc", ro_desc), ("direct_reread", ro_dir)):
        res = llm.generate([{"prompt": t, "multi_modal_data": {"image": imgs[it["uid"]]}} for t, it in zip(texts, items)], sp1)
        for it, r in zip(items, res):
            top = {lp.decoded_token: lp.logprob for lp in r.outputs[0].logprobs[0].values()}
            sc = C.letter_scores(top, it["choices"])
            sc = {k: (v if v > -math.inf else None) for k, v in sc.items()}
            rec = {"model": MODEL, "set": "cues", "task": it["task"], "item_id": it["key"], "base_id": it["uid"],
                   "phrasing": it["phrasing"], "gold": it["gold"], "letters": it["choices"], "image_id": it["group"],
                   "question": it["question"], "meta": it["meta"], "condition": cond, "top20": sc}
            if cond == "desc":
                g = desc[it["uid"]]
                rec |= {"text": g["text"], "gen_len": g["len"], "trunc": g["trunc"], "think": g["think"],
                        "think_len": g["think_len"], "think_trunc": g["think_trunc"]}
            recs.append(rec)
    sfx = f"_limit{args.limit}" if args.limit else ""
    vx.write_jsonl(f"{args.out_dir}/{MODEL}_cues_desc{sfx}.jsonl", recs)
    print(f"[e31d] wrote {len(recs)} records ({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
