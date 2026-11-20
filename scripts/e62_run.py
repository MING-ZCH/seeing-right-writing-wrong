"""Exp-62 (GPU): Direct, CoT and the fixed-rationale twin swap on natural-object occlusion twins (COCO composites).

Items: outputs/e62/stimuli/items_final.jsonl (e62_stimuli.py). Each item = image v x phrasing (closer / farther /
which); its twin item has the same question on the twin image v_cf (paste order swapped), so the gold flips.
Per item (paper protocol, as e2d.py / e14_twin_imgswap.py / e60_run.py):
  direct  user[v, q] + assistant "Answer: ("                                   (forced readout)
  cot     greedy R (<= 512 tokens) continuing "Let's think step by step." under v; same-turn readout R + "\\nAnswer: ("
  rcut    R_cut = legacy strip_answer(R) (the paper's R_cut), read under v and under v_cf (same question)
  strict  R_cut with trailing concluding sentences (therefore / so / thus / overall / based on ...) also removed,
          read under v and v_cf (robustness variant; identical to rcut when nothing is removed)
  full    R as generated, read under v_cf (under v this is the cot readout)
-> outputs/e62/<model>.jsonl (one record per item x arm x image)

Usage: python e62_run.py --model qwen7b [--limit 4]
"""
from __future__ import annotations
import vtax_paths as _vtp  # noqa: E402

import argparse
import functools
import json
import os
import re
import sys
import time

import vx
from vx import C

sys.path.insert(0, f"{_vtp.LEGACY_SCRIPTS}")
from e5_content_edit import strip_answer  # noqa: E402

E62 = f"{vx.OUT}/e62"
CONCL = re.compile(r"^\W*(?:therefore|thus|hence|so\b|in conclusion|to conclude|overall|in summary|based on|"
                   r"consequently|as a result|this means|this suggests|this indicates|given (?:this|these|that)|"
                   r"conclusion|final answer|answer)", re.I)


def split_sents(text: str) -> list[str]:
    """Sentence / line split that keeps the separators attached (join with '' restores the text)."""
    return [s for s in re.split(r"(?<=[.!?])(?=\s)|(?=\n)", text) if s]


def strict_cut(body: str) -> str:
    """Drop trailing concluding sentences from R_cut (never cuts into the 'Let's think step by step.' prefix)."""
    sents = split_sents(body)
    while len(sents) > 1 and (CONCL.match(sents[-1].strip()) or re.fullmatch(r"\W*\d*\.?\W*", sents[-1])):
        sents.pop()
    out = "".join(sents).rstrip()
    return out if len(out) >= len(C.COT_PREFIX) else C.COT_PREFIX


def load_items(limit: int | None) -> list[dict]:
    """Final items; ``limit`` keeps the first n pairs."""
    items = vx.read_jsonl(f"{E62}/stimuli/items_final.jsonl")
    if limit:
        keep = sorted({it["pair"] for it in items})[:limit]
        items = [it for it in items if it["pair"] in keep]
    return items


def main() -> None:
    """CLI entry point."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--max-cot", type=int, default=512)
    args = ap.parse_args()
    t0 = time.time()
    items = load_items(args.limit)
    by_uid = {it["uid"]: it for it in items}
    print(f"[e62] {args.model}: {len(items)} items", flush=True)
    if os.environ.get("E22_NO_MM_CACHE"):
        import vllm
        vllm.LLM = functools.partial(vllm.LLM, disable_mm_preprocessor_cache=True)
    if os.environ.get("E28_MAX_BATCHED"):
        import vllm
        vllm.LLM = functools.partial(vllm.LLM, max_num_batched_tokens=int(os.environ["E28_MAX_BATCHED"]))
    llm = vx.make_llm(args.model)
    t1 = time.time()
    refs: dict[str, dict] = {}

    def ref(uid: str) -> dict:
        it = by_uid[uid]
        if it["image_uid"] not in refs:
            refs[it["image_uid"]] = vx.img_ref(C.get_image(it))
        return refs[it["image_uid"]]

    ch = [it["choices"] for it in items]
    base = lambda it: {"model": args.model, "uid": it["uid"], "image_uid": it["image_uid"], "pair": it["pair"],
                       "phrasing": it["phrasing"], "gold": it["gold"], "letters": it["choices"]}
    recs = []
    outs = vx.readout(llm, args.model, [C.build_messages(it, "direct", "none", "", ref(it["uid"])) for it in items], ch)
    recs += [base(it) | {"arm": "direct", "img": "v"} | o for it, o in zip(items, outs)]
    print(f"[e62] direct done ({time.time() - t0:.0f}s)", flush=True)
    gens = vx.generate(llm, args.model, [C.cot_messages(it, ref(it["uid"])) for it in items], max_tokens=args.max_cot)
    cots = [C.COT_PREFIX + " " + g["text"] for g in gens]
    cuts = [strip_answer(c) for c in cots]
    stricts = [strict_cut(c) for c in cuts]
    print(f"[e62] cot generated ({time.time() - t0:.0f}s)", flush=True)
    jobs = []  # (item index, arm, img, context text)
    for i, it in enumerate(items):
        tw = it["twin_item"] if it["twin_item"] in by_uid else None
        jobs.append((i, "cot", "v", cots[i]))
        for arm, body in (("rcut", cuts[i]), ("strict", stricts[i])):
            jobs.append((i, arm, "v", body))
            if tw:
                jobs.append((i, arm, "vcf", body))
        if tw:
            jobs.append((i, "full", "vcf", cots[i]))
    msgs = [C.build_messages(items[i], "cot", "none", body, ref(items[i]["uid"] if img == "v" else items[i]["twin_item"]))
            for i, arm, img, body in jobs]
    outs = vx.readout(llm, args.model, msgs, [items[i]["choices"] for i, *_ in jobs])
    for (i, arm, img, body), o in zip(jobs, outs):
        it = items[i]
        r = base(it) | {"arm": arm, "img": img, "ctx_chars": len(body)} | o
        if arm == "cot":
            r |= {"text": cots[i], "gen_len": gens[i]["len"], "trunc": gens[i]["trunc"], "rcut": cuts[i],
                  "strict": stricts[i], "stated": C.parse_free_answer(cots[i], it["choices"])}
        recs.append(r)
    sfx = f"_limit{args.limit}" if args.limit else ""
    vx.write_jsonl(f"{E62}/{args.model}{sfx}.jsonl", recs)
    json.dump({"load_s": t1 - t0, "total_s": time.time() - t0, "n_items": len(items), "n_records": len(recs),
               "model_path": C.model_path(args.model), "max_cot": args.max_cot},
              open(f"{E62}/{args.model}{sfx}.timing.json", "w"), indent=1)
    print(f"[e62] wrote {len(recs)} records ({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
