"""Exp-63 evaluation (vLLM, same engine / readout as the paper: vx.make_llm + vx.readout; prereg outputs/e63/prereg.txt).

Per model (qwen3vl_8b, qwen3vl_8b_e63match, qwen3vl_8b_e63cf) and set (e63_common.EVAL_SETS):
  direct     "Answer: (" under v (all sets); e60 also under the mirror (twin sets read v_cf from the twin item's record)
  base_rcut  the BASE model's stored R_cut fixed: under v (all sets with a stored R), v_cf and gray (twin / mirror sets)
  cot        own greedy CoT R (C.cot_messages, <= 512 tokens), same-turn readout under v
  own_rcut   own R_cut = legacy strip_answer(R) under v and v_cf (twin / mirror sets)
Records carry everything the endpoint analysis needs (gold, gold_cf, twin uid, cluster group, phrasing, base D / C flags), so
the analysis never reloads images. With --vcheck the same engine also reads out the fixed 32-example training check
sample (e63_gen.vcheck -> outputs/e63/vcheck_<model>.json), compared there with the HF LoRA model.
-> outputs/e63/eval/<model>__<sets>.jsonl ; ``--merge`` concatenates them into outputs/e63/eval/<model>.jsonl.

Usage: python -m experiments.lora_pilot.evaluate --model qwen3vl_8b_e63cf --sets coco,e60,heldout [--vcheck] [--limit 3] [--dry]
       python -m experiments.lora_pilot.evaluate --model qwen3vl_8b_e63cf --merge
"""
from __future__ import annotations

import argparse
import functools
import glob
import json
import os
import time
from collections import Counter

from experiments.lora_pilot import lora_common as E
from experiments.lora_pilot.lora_common import C, vx

EV = f"{E.E63}/eval"


def build_jobs(items: list[dict], set_name: str) -> list[tuple[int, str, str]]:
    """(item index, arm, img) for the arms that need no generation."""
    jobs = []
    twin = set_name in E.TWIN_SETS
    for i, it in enumerate(items):
        jobs.append((i, "direct", "v"))
        if it["cf"] and it["cf"]["kind"] == "mirror":
            jobs.append((i, "direct", "cf"))
        if it["base"]["rcut"] is not None:
            for img in (("v", "cf", "gray") if twin else ("v",)):
                jobs.append((i, "base_rcut", img))
    return jobs


def meta(model: str, it: dict) -> dict:
    """Per-record item fields (no images)."""
    cf = it.get("cf") or {}
    return {"model": model, "set": it["set"], "uid": it["uid"], "task": it.get("task"), "group": it["group"],
            "phrasing": it.get("phrasing"), "gold": it["gold"], "gold_cf": cf.get("gold"), "twin_uid": cf.get("uid"),
            "cf_kind": cf.get("kind"), "letters": it["choices"], "base_direct_ok": it["base"]["direct_ok"],
            "base_cot_ok": it["base"]["cot_ok"]}


def run_set(llm, model: str, set_name: str, limit: int | None, dry: bool, max_cot: int) -> list[dict]:
    t0 = time.time()
    items = E.load_eval_set(set_name, limit)
    E.prefetch([it.get("image_path") for it in items if "image" not in it] + [(it["cf"] or {}).get("image_path") for it in items])
    twin = set_name in E.TWIN_SETS
    refs: dict[tuple[str, str], dict] = {}

    def ref(i: int, img: str) -> dict:
        it = items[i]
        key = (it["cf"]["image_path"] if img == "cf" and it["cf"].get("image_path") else it["uid"], img)
        if key not in refs:
            refs[key] = vx.img_ref(E.image_of(it, img))
        return refs[key]

    jobs = build_jobs(items, set_name)
    msgs = [C.build_messages(items[i], "direct", "none", "", ref(i, img)) if arm == "direct" else
            C.build_messages(items[i], "cot", "none", items[i]["base"]["rcut"], ref(i, img)) for i, arm, img in jobs]
    cot_msgs = [C.cot_messages(it, ref(i, "v")) for i, it in enumerate(items)]
    n_own = len(items) * (3 if twin else 1)
    print(f"[e63 eval] {model} {set_name}: {len(items)} items, {len(jobs)} fixed readouts, {len(cot_msgs)} CoT gens, "
          f"{n_own} own readouts; missing base R {sum(it['base']['rcut'] is None for it in items)} "
          f"({time.time() - t0:.0f}s)", flush=True)
    if dry:
        print("  arms:", dict(Counter((a, g) for _, a, g in jobs)))
        for (i, a, g), m in list(zip(jobs, msgs))[:3]:
            print("  ", a, g, json.dumps([{"role": x["role"], "content": x["content"] if isinstance(x["content"], str)
                                         else [p.get("text", "<image>") for p in x["content"]]} for x in m])[:400])
        os.makedirs(f"{E.E63}/eval_dry", exist_ok=True)
        skel = [meta(model, items[i]) | {"arm": a, "img": g} for i, a, g in jobs]
        skel += [meta(model, it) | {"arm": a, "img": g} for it in items
                 for a, g in ([("cot", "v")] + ([("own_rcut", "v"), ("own_rcut", "cf")] if twin else []))]
        vx.write_jsonl(f"{E.E63}/eval_dry/{model}__{set_name}.jobs.jsonl", skel)
        return []
    outs = vx.readout(llm, model, msgs, [items[i]["choices"] for i, *_ in jobs])
    recs = [meta(model, items[i]) | {"arm": a, "img": g, "ctx_chars": len(items[i]["base"]["rcut"] or "")} | o
            for (i, a, g), o in zip(jobs, outs)]
    print(f"[e63 eval] {set_name} fixed readouts done ({time.time() - t0:.0f}s)", flush=True)
    gens = vx.generate(llm, model, cot_msgs, max_tokens=max_cot)
    cots = [C.COT_PREFIX + " " + g["text"] for g in gens]
    cuts = [E.r_cut(c) for c in cots]
    own = []
    for i in range(len(items)):
        own.append((i, "cot", "v", cots[i]))
        if twin:
            own += [(i, "own_rcut", "v", cuts[i]), (i, "own_rcut", "cf", cuts[i])]
    outs = vx.readout(llm, model, [C.build_messages(items[i], "cot", "none", body, ref(i, img)) for i, _, img, body in own],
                      [items[i]["choices"] for i, *_ in own])
    for (i, a, g, body), o in zip(own, outs):
        r = meta(model, items[i]) | {"arm": a, "img": g, "ctx_chars": len(body)} | o
        if a == "cot":
            r |= {"text": cots[i], "rcut": cuts[i], "gen_len": gens[i]["len"], "trunc": gens[i]["trunc"],
                  "stated": C.parse_free_answer(cots[i], items[i]["choices"])}
        recs.append(r)
    print(f"[e63 eval] {set_name} done: {len(recs)} records ({time.time() - t0:.0f}s)", flush=True)
    return recs


def merge(model: str) -> None:
    parts = sorted(p for p in glob.glob(f"{EV}/{model}__*.jsonl") if "_limit" not in p)
    recs, seen = [], set()
    for p in parts:
        for r in vx.read_jsonl(p):
            k = (r["set"], r["uid"], r["arm"], r["img"])
            assert k not in seen, f"duplicate record {k} in {p}"
            seen.add(k)
            recs.append(r)
    vx.write_jsonl(f"{EV}/{model}.jsonl", recs)
    print(f"[e63 eval] merged {len(parts)} parts -> {model}.jsonl ({len(recs)} records; "
          f"{dict(Counter(r['set'] for r in recs))})")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=E.EVAL_MODELS)
    ap.add_argument("--sets", default=",".join(E.EVAL_SETS))
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--max-cot", type=int, default=512)
    ap.add_argument("--vcheck", action="store_true", help="also read out the 32-example training check sample")
    ap.add_argument("--dry", action="store_true", help="build prompts only (CPU), no engine")
    ap.add_argument("--merge", action="store_true")
    a = ap.parse_args()
    os.makedirs(EV, exist_ok=True)
    if a.merge:
        return merge(a.model)
    t0 = time.time()
    sets = a.sets.split(",")
    assert all(s in E.EVAL_SETS for s in sets), sets
    llm = None
    if not a.dry:
        if a.model != E.BASE:
            p = C.model_path(a.model)
            assert os.path.exists(f"{p}/e63_merge.json"), f"merged checkpoint missing: {p}"
        if os.environ.get("E22_NO_MM_CACHE"):
            import vllm
            vllm.LLM = functools.partial(vllm.LLM, disable_mm_preprocessor_cache=True)
        llm = vx.make_llm(a.model)
    t1 = time.time()
    if a.vcheck and (not a.dry or os.path.exists(f"{E.E63}/train_examples/check_sample.jsonl")):
        from experiments.lora_pilot import generate as e63_gen
        e63_gen.vcheck(a.model, a.dry, llm=llm)
    recs = []
    for s in sets:
        recs += run_set(llm, a.model, s, a.limit, a.dry, a.max_cot)
    if a.dry:
        print(f"[e63 eval] dry ok ({time.time() - t0:.0f}s)")
        return
    sfx = f"_limit{a.limit}" if a.limit else ""
    tag = "+".join(sets)
    vx.write_jsonl(f"{EV}/{a.model}__{tag}{sfx}.jsonl", recs)
    json.dump({"load_s": t1 - t0, "total_s": time.time() - t0, "n_records": len(recs), "sets": sets,
               "model_path": C.model_path(a.model)}, open(f"{EV}/{a.model}__{tag}{sfx}.timing.json", "w"), indent=1)
    print(f"[e63 eval] wrote {len(recs)} records ({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
