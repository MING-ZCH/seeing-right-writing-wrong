"""E44: label-free prompt variants of the CoT prefix (perception-first / restate-question).

The CoT arm's assistant prefix "Let's think step by step." is replaced by a fixed instruction; everything else is the
seven-model CoT protocol (experiments/interventions/structured_state.py): user turn = image + question, greedy continuation (<= 512 tokens), same-turn
forced readout "<prefix> <continuation>\nAnswer: (" via C.build_messages(item, "cot", "none", ...).

  p1  occlusion-first (pre-registered primary, outputs/e44/prereg.txt)
  p2  restate-question
  p3  p1 + p2 combined
  p1f / p2f / p3f  forced-continuation variant (prereg addendum): items whose continuation is empty are regenerated
      with EOS suppressed for the first 16 tokens (min_tokens), all other items identical to p1 / p2 / p3
  cot / direct  the paper's CoT / Direct arms (run only on the natural sets; cues / syn reuse the stored e10 runs)

Sets: cues (cue-factorial, 1,600 queries), syn (Synthetic Depth 500, single order as e10 / Tab. 1),
nat (BLINK val Relative Depth 124 + CV-Bench 3D Depth 600).
-> outputs/e44/<model>_<set>.jsonl (one record per item x arm)

Usage: python -m experiments.interventions.prompt_variants --model qwen3vl_8b --set cues [--limit 3]
"""
from __future__ import annotations
from srww import paths as _vtp  # noqa: E402

import argparse
import io
import json
import os
import time

from srww import vllm_utils as vx
from srww.vllm_utils import C

PROMPTS = {
    "p1": "First, look at where the two shapes overlap and say which shape's outline is complete there "
          "(that shape is in front). Then answer.",
    "p2": "First, restate the question in my own words using the same shapes and the same relation, "
          "then decide it from the image.",
    "p3": "First, restate the question in my own words using the same shapes and the same relation. "
          "Then look at where the two shapes overlap and say which shape's outline is complete there "
          "(that shape is in front). Then decide it from the image and answer.",
    "cot": C.COT_PREFIX,
}
PROMPT_ARMS = ["p1", "p2", "p3"]
MAX_TOK = 512
MIN_TOK = 16
N_SANITY = 64


def load_set(name: str, limit: int | None) -> list[dict]:
    """Items with ``key`` (unique query id), ``phrasing`` and ``set``."""
    from experiments.occlusion.twins_readout import load_occ
    if name == "cues":
        d = os.environ.get("CUES_DIR", f"{_vtp.ITEMS}/occ_cues")
        items = [dict(it, phrasing="front", key=it["uid"]) for it in load_occ(limit, items_dir=d)]
        items += [dict(it, phrasing="behind", key=it["uid"] + "_behind") for it in load_occ(limit, polar=True, items_dir=d)]
        return [dict(it, set="cues") for it in items]
    if name == "syn":
        return [dict(it, phrasing="front", key=it["uid"], set="syn") for it in vx.load_items(["syn_depth"], limit=limit)]
    if name == "nat":
        import pandas as pd
        from PIL import Image
        d = pd.read_parquet(f"{vx.SHARE}/datasets/heldout/BLINK/Relative_Depth/val-00000-of-00001.parquet")
        items = []
        for _, r in d.iterrows():
            items.append({"uid": r["idx"], "dataset": "blink", "task": "blink_depth", "question": r["prompt"],
                          "choices": C.letters(len(list(r["choices"]))), "gold": r["answer"].strip("() "),
                          "image": Image.open(io.BytesIO(r["image_1"]["bytes"])).convert("RGB"), "group": r["idx"]})
        items = items[:limit] if limit else items
        cvb = vx.load_items(["cvb_depth"])
        items += cvb[:limit] if limit else cvb
        return [dict(it, phrasing=None, key=it["uid"], set="blink" if it["task"] == "blink_depth" else "cvb")
                for it in items]
    raise ValueError(name)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--set", required=True, choices=["cues", "syn", "nat"])
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--shard", default=None, help="i/n: run items[i::n] only (output suffix _shard<i>of<n>)")
    args = ap.parse_args()
    sfx = f"_limit{args.limit}" if args.limit else ""
    if args.shard:
        si, sn = map(int, args.shard.split("/"))
        sfx += f"_shard{si}of{sn}"
    out = f"{vx.OUT}/e44/{args.model}_{args.set}{sfx}.jsonl"
    if not args.force and (os.path.exists(out) or os.path.exists(out + ".lock")):
        print(f"[e44] skip: {out} exists or is locked", flush=True)
        return
    os.makedirs(os.path.dirname(out), exist_ok=True)
    open(out + ".lock", "w").write(str(os.getpid()))
    import functools
    import vllm
    if os.environ.get("E22_NO_MM_CACHE"):
        # vLLM 0.8.4 V1 mm_input_cache P0/P1 desync with Molmo on the cue set (Exp-22)
        vllm.LLM = functools.partial(vllm.LLM, disable_mm_preprocessor_cache=True)
    if os.environ.get("E28_MAX_BATCHED"):
        # Qwen3-VL ViT OOM on large batches at max_num_batched_tokens=16384 (Exp-28)
        vllm.LLM = functools.partial(vllm.LLM, max_num_batched_tokens=int(os.environ["E28_MAX_BATCHED"]))
    t0 = time.time()
    tm: dict[str, float] = {}
    items = load_set(args.set, args.limit)
    if args.shard:
        items = items[si::sn]
    n = len(items)
    arms = ["p1", "p2", "p3"] + (["cot"] if args.set == "nat" else [])
    llm = vx.make_llm(args.model)
    refs = [vx.img_ref(C.get_image(it)) for it in items]
    tm["setup"] = time.time() - t0
    print(f"[e44] {args.model} {args.set}: {n} prompts, arms {arms}", flush=True)
    # generation: all arms in one batch
    t1 = time.time()
    gmsgs = [[{"role": "user", "content": [refs[i], {"type": "text", "text": it["question"]}]},
              {"role": "assistant", "content": PROMPTS[a]}] for a in arms for i, it in enumerate(items)]
    gens = vx.generate(llm, args.model, gmsgs, max_tokens=MAX_TOK)
    G = {a: gens[k * n:(k + 1) * n] for k, a in enumerate(arms)}
    tm["gen"] = time.time() - t1
    # forced-continuation variant (prereg addendum): where a prompt's continuation is empty (the model ends the turn
    # right after the instruction), regenerate with EOS / stop tokens suppressed for the first MIN_TOK tokens
    t1 = time.time()
    from vllm import SamplingParams
    sp = SamplingParams(temperature=0.0, max_tokens=MAX_TOK, min_tokens=MIN_TOK, seed=0)
    for a in PROMPT_ARMS:
        G[a + "f"] = [dict(g, forced=False) for g in G[a]]
        idx = [i for i, g in enumerate(G[a]) if not g["text"].strip()]
        if idx:
            res = llm.chat([gmsgs[arms.index(a) * n + i] for i in idx], sp, **vx.chat_kw(args.model))
            for i, r in zip(idx, res):
                o = r.outputs[0]
                G[a + "f"][i] = {"text": o.text.strip(), "len": len(o.token_ids), "trunc": o.finish_reason == "length",
                                 "forced": True}
        print(f"[e44] {a}: {len(idx)}/{n} empty continuations regenerated with min_tokens={MIN_TOK}", flush=True)
    tm["gen_forced"] = time.time() - t1
    # same-turn readouts (+ Direct: all items on nat, first N_SANITY items elsewhere as a sanity re-read)
    t1 = time.time()
    n_dir = n if args.set == "nat" else min(N_SANITY, n)
    rmsgs = [C.build_messages(items[i], "direct", "none", "", refs[i]) for i in range(n_dir)]
    rmsgs += [C.build_messages(it, "cot", "none", PROMPTS[a] + " " + G[a][i]["text"], refs[i])
              for a in arms for i, it in enumerate(items)]
    ch = [items[i]["choices"] for i in range(n_dir)] + [it["choices"] for a in arms for it in items]
    fidx = [(a, i) for a in PROMPT_ARMS for i in range(n) if G[a + "f"][i]["forced"]]
    rmsgs += [C.build_messages(items[i], "cot", "none", PROMPTS[a] + " " + G[a + "f"][i]["text"], refs[i]) for a, i in fidx]
    ch += [items[i]["choices"] for _, i in fidx]
    ro = vx.readout(llm, args.model, rmsgs, ch)
    tm["readout"] = time.time() - t1
    nb = n_dir + len(arms) * n
    RO = {(a, i): ro[n_dir + arms.index(a) * n + i] for a in arms for i in range(n)}
    for j, (a, i) in enumerate(fidx):
        RO[(a + "f", i)] = ro[nb + j]
    for a in PROMPT_ARMS:
        for i in range(n):
            RO.setdefault((a + "f", i), RO[(a, i)])
    tok = llm.get_tokenizer()
    recs = []
    base = lambda it: {"model": args.model, "set": it["set"], "task": it["task"], "item_id": it["key"], "base_id": it["uid"],
                       "phrasing": it["phrasing"], "gold": it["gold"], "letters": it["choices"], "image_id": it["group"],
                       "question": it["question"], "meta": it.get("meta")}
    for i in range(n_dir):
        recs.append(base(items[i]) | {"arm": "direct", **ro[i]})
    for a in arms + [p + "f" for p in PROMPT_ARMS]:
        pa = a.rstrip("f") if a.endswith("f") else a
        for i, it in enumerate(items):
            g = G[a][i]
            recs.append(base(it) | {"arm": a, **RO[(a, i)], "prefix": PROMPTS[pa], "text": PROMPTS[pa] + " " + g["text"],
                                    "gen_len": g["len"], "trunc": g["trunc"], "forced": g.get("forced", False),
                                    "body_tok": len(tok.encode(g["text"], add_special_tokens=False))})
    vx.write_jsonl(out, recs)
    tm["total"] = time.time() - t0
    json.dump(tm | {"n": n, "arms": arms}, open(out.replace(".jsonl", ".timing.json"), "w"), indent=1)
    os.remove(out + ".lock")
    print(f"[e44] wrote {len(recs)} records to {out}; timing {json.dumps({k: round(v) for k, v in tm.items()})}", flush=True)


if __name__ == "__main__":
    main()
