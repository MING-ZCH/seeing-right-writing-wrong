"""Exp-63 rationales and training examples (base qwen3vl_8b; vLLM via vx.make_llm, as experiments/fixed_rationale/coco_twins_swap.py).

Stages
  gen     (GPU, sharded by pair) for every training item (image v or v_cf, its question): greedy CoT R continuing
          "Let's think step by step." (C.cot_messages, <= 512 tokens), R = COT_PREFIX + " " + text,
          R_cut = legacy strip_answer(R); base readouts (vx.readout): direct (own image), cot (R, own image),
          rcut_own (R_cut, own image), rcut_twin (R_cut, the pair's other image, same question)
          -> outputs/e63/gen/shard<k>of<n>.jsonl
  build   (CPU) merge shards -> outputs/e63/gen/rationales.jsonl; two arms with identical images, counts, rationales:
            cf     per pair one R* drawn from {R_cut(v), R_cut(v_cf)} (random.Random(63), pairs in sorted order);
                   examples [v, q, R*] -> gold(v) and [v_cf, q, R*] -> gold(v_cf)
            match  [v, q, R_cut(v)] -> gold(v) and [v_cf, q, R_cut(v_cf)] -> gold(v_cf)
          -> outputs/e63/train_examples/{cf,match}.jsonl, build_report.json, and the fixed 32-example check sample
             outputs/e63/train_examples/check_sample.jsonl (16 per arm, seed 63)
  vcheck  (GPU) vLLM readout of the check sample with the training prompt (C.build_messages(item, 'cot', 'none',
          R, image)): prompt token ids (as fed to the engine) + top-20 letter logprobs -> outputs/e63/vcheck_<model>.json
          (read by experiments/lora_pilot/train.py at step 0; repeated for the merged checkpoints by experiments/lora_pilot/evaluate.py)
Dry runs (CPU): ``gen --dry`` / ``vcheck --dry`` build all prompts and print samples, no engine.

Usage: python -m experiments.lora_pilot.generate gen --shard 0 --nshard 8 [--limit-pairs 4] [--dry]
       python -m experiments.lora_pilot.generate build
       python -m experiments.lora_pilot.generate vcheck [--model qwen3vl_8b] [--dry]
"""
from __future__ import annotations

import argparse
import json
import os
import random
import time
from collections import Counter

from experiments.lora_pilot import lora_common as E
from experiments.lora_pilot.lora_common import C, vx

GEN = f"{E.E63}/gen"
EX = f"{E.E63}/train_examples"


def train_messages(it: dict, rationale: str, ref: dict) -> list[dict]:
    """Training / evaluation readout prompt: assistant turn = R_cut + '\\nAnswer: (' (continued)."""
    return C.build_messages(it, "cot", "none", rationale, ref)


def _show(msgs: list[dict]) -> str:
    return json.dumps([{"role": m["role"], "content": m["content"] if isinstance(m["content"], str)
                        else [p.get("text", "<image>") for p in m["content"]]} for m in msgs])


def _engine():
    import functools
    if os.environ.get("E22_NO_MM_CACHE"):
        import vllm
        vllm.LLM = functools.partial(vllm.LLM, disable_mm_preprocessor_cache=True)
    return vx.make_llm(E.BASE)


# ----------------------------------------------------------------------------- gen

def gen(shard: int, nshard: int, limit_pairs: int | None, dry: bool, max_cot: int) -> None:
    t0 = time.time()
    items = E.load_train_items(limit_pairs)
    pairs = sorted({it["pair"] for it in items})
    mine = {p for i, p in enumerate(pairs) if i % nshard == shard}
    items = [it for it in items if it["pair"] in mine]
    by_uid = {it["uid"]: it for it in items}
    print(f"[e63 gen] shard {shard}/{nshard}: {len(items)} items, {len(mine)} pairs", flush=True)
    refs: dict[str, dict] = {}

    def ref(uid: str) -> dict:
        it = by_uid[uid]
        if it["image_uid"] not in refs:
            refs[it["image_uid"]] = vx.img_ref(E.load_image(it["image_path"]))
        return refs[it["image_uid"]]

    E.prefetch([it["image_path"] for it in items])
    ch = [it["choices"] for it in items]
    cot_msgs = [C.cot_messages(it, ref(it["uid"])) for it in items]
    direct_msgs = [C.build_messages(it, "direct", "none", "", ref(it["uid"])) for it in items]
    if dry:
        for m in cot_msgs[:2] + direct_msgs[:2]:
            print(_show(m)[:500])
        fake = [C.COT_PREFIX + " The red circle covers part of the blue square. Therefore, the answer is (A) yes."] * len(items)
        cuts = [E.r_cut(c) for c in fake]
        m2 = [train_messages(it, cuts[i], ref(it["twin_item"])) for i, it in enumerate(items)]
        print(_show(m2[0])[:600])
        print(f"[e63 gen] dry ok: {len(cot_msgs)} CoT prompts, {len(direct_msgs) + 3 * len(items)} readouts "
              f"({time.time() - t0:.0f}s)")
        return
    llm = _engine()
    t1 = time.time()
    d_out = vx.readout(llm, E.BASE, direct_msgs, ch)
    gens = vx.generate(llm, E.BASE, cot_msgs, max_tokens=max_cot)
    cots = [C.COT_PREFIX + " " + g["text"] for g in gens]
    cuts = [E.r_cut(c) for c in cots]
    print(f"[e63 gen] generated ({time.time() - t0:.0f}s)", flush=True)
    jobs = []
    for i, it in enumerate(items):
        jobs += [(i, "cot", it["uid"], cots[i]), (i, "rcut_own", it["uid"], cuts[i]), (i, "rcut_twin", it["twin_item"], cuts[i])]
    outs = vx.readout(llm, E.BASE, [train_messages(items[i], body, ref(src)) for i, _, src, body in jobs], [items[i]["choices"] for i, *_ in jobs])
    ro: dict[int, dict] = {i: {"direct": d_out[i]} for i in range(len(items))}
    for (i, arm, _, _), o in zip(jobs, outs):
        ro[i][arm] = o
    recs = []
    for i, it in enumerate(items):
        tw = by_uid[it["twin_item"]]
        recs.append({"uid": it["uid"], "pair": it["pair"], "kind": it["kind"], "phrasing": it["phrasing"],
                     "img_role": it["img_role"], "gold": it["gold"], "gold_twin": tw["gold"], "R": cots[i], "rcut": cuts[i],
                     "gen_len": gens[i]["len"], "trunc": gens[i]["trunc"], "stated": C.parse_free_answer(cots[i], it["choices"]),
                     "readouts": {k: {"top20": v["top20"], "pred": v["pred_top20"]} for k, v in ro[i].items()}})
    sfx = f"_limit{limit_pairs}" if limit_pairs else ""
    vx.write_jsonl(f"{GEN}/shard{shard}of{nshard}{sfx}.jsonl", recs)
    json.dump({"load_s": t1 - t0, "total_s": time.time() - t0, "n_items": len(items)},
              open(f"{GEN}/shard{shard}of{nshard}{sfx}.timing.json", "w"))
    print(f"[e63 gen] wrote {len(recs)} ({time.time() - t0:.0f}s)", flush=True)


# ----------------------------------------------------------------------------- build

def build(nshard: int, sfx: str = "") -> None:
    items = {it["uid"]: it for it in E.load_train_items()}
    recs = {}
    for k in range(nshard):
        for r in vx.read_jsonl(f"{GEN}/shard{k}of{nshard}{sfx}.jsonl"):
            recs[r["uid"]] = r
    missing = sorted(set(items) - set(recs))
    if sfx:
        items = {u: it for u, it in items.items() if u in recs}
        missing = []
    assert not missing, f"{len(missing)} training items without a rationale, e.g. {missing[:3]}"
    vx.write_jsonl(f"{GEN}/rationales{sfx}.jsonl", [recs[u] for u in sorted(recs)])
    by_pair: dict[str, dict] = {}
    for u, it in items.items():
        by_pair.setdefault(it["pair"], {})[it["img_role"]] = it
    rng = random.Random(E.SEED)
    arms: dict[str, list[dict]] = {"cf": [], "match": []}
    pick = Counter()
    for p in sorted(by_pair):
        a, b = by_pair[p]["v"], by_pair[p]["cf"]
        ra, rb = recs[a["uid"]]["rcut"], recs[b["uid"]]["rcut"]
        star = rng.choice(("v", "cf"))
        pick[star] += 1
        rs = ra if star == "v" else rb
        for arm, texts in (("cf", ((rs, star), (rs, star))), ("match", ((ra, "v"), (rb, "cf")))):
            for it, (txt, src) in zip((a, b), texts):
                arms[arm].append({"ex_id": f"{arm}_{it['uid']}", "arm": arm, "pair": p, "uid": it["uid"], "kind": it["kind"],
                                  "phrasing": it["phrasing"], "img_role": it["img_role"], "image_rel": it["image_rel"],
                                  "question": it["question"], "choices": it["choices"], "gold": it["gold"],
                                  "rationale": txt, "rationale_src": src, "target": it["gold"]})
    os.makedirs(EX, exist_ok=True)
    for arm, xs in arms.items():
        vx.write_jsonl(f"{EX}/{arm}{sfx}.jsonl", xs)
    # report: rationale statistics and the base model on the training prompts
    rs = list(recs.values())
    def acc(key: str, gold_key: str) -> float:
        v = [r["readouts"][key]["pred"] == r[gold_key] for r in rs]
        return round(100 * sum(v) / len(v), 2)
    lens = sorted(r["gen_len"] for r in rs)
    rep = {"n_items": len(rs), "n_pairs": len(by_pair), "cf_pick": dict(pick),
           "n_examples": {k: len(v) for k, v in arms.items()},
           "same_images_and_order": [x["uid"] for x in arms["cf"]] == [x["uid"] for x in arms["match"]],
           "cf_text_identical_within_pair": all(arms["cf"][i]["rationale"] == arms["cf"][i + 1]["rationale"]
                                               for i in range(0, len(arms["cf"]), 2)),
           "trunc_pct": round(100 * sum(r["trunc"] for r in rs) / len(rs), 2),
           "rcut_prefix_only_pct": round(100 * sum(r["rcut"].strip() == C.COT_PREFIX for r in rs) / len(rs), 2),
           "gen_len_median": lens[len(lens) // 2], "rcut_chars_median": sorted(len(r["rcut"]) for r in rs)[len(rs) // 2],
           "pair_rcut_identical_pct": round(100 * sum(recs[by_pair[p]["v"]["uid"]]["rcut"] == recs[by_pair[p]["cf"]["uid"]]["rcut"]
                                                     for p in by_pair) / len(by_pair), 2),
           "base_acc": {"direct": acc("direct", "gold"), "cot": acc("cot", "gold"), "rcut_own": acc("rcut_own", "gold"),
                        "rcut_twin_vs_twin_gold": acc("rcut_twin", "gold_twin")},
           "base_acc_by_kind": {k: {a: round(100 * sum(r["readouts"][a]["pred"] == r["gold"] for r in rs if r["kind"] == k)
                                            / max(1, sum(r["kind"] == k for r in rs)), 2) for a in ("direct", "cot", "rcut_own")}
                                for k in ("depth", "lr")}}
    # base image effect on the training pairs with own R_cut fixed (indicator, item level)
    ie = [float(r["readouts"]["rcut_own"]["pred"] == r["gold"]) - float(r["readouts"]["rcut_twin"]["pred"] == r["gold"]) for r in rs]
    de = []
    for p, d in by_pair.items():
        for role, other in (("v", "cf"), ("cf", "v")):
            r, o = recs[d[role]["uid"]], recs[d[other]["uid"]]
            de.append(float(r["readouts"]["direct"]["pred"] == r["gold"]) - float(o["readouts"]["direct"]["pred"] == r["gold"]))
    rep["base_image_effect_indicator"] = {"direct": round(100 * sum(de) / len(de), 2), "rcut": round(100 * sum(ie) / len(ie), 2)}
    # fixed check sample: 16 examples per arm = 8 pairs (4 depth + 4 lr), seed 63
    pr = random.Random(E.SEED)
    ps = pr.sample(sorted(p for p in by_pair if p.startswith("e63d")), 4) + pr.sample(sorted(p for p in by_pair if p.startswith("e63h")), 4)
    chk = [x for arm in ("cf", "match") for x in arms[arm] if x["pair"] in ps]
    vx.write_jsonl(f"{EX}/check_sample{sfx}.jsonl", chk)
    rep["check_sample"] = {"n": len(chk), "pairs": ps}
    json.dump(rep, open(f"{EX}/build_report{sfx}.json", "w"), indent=1)
    print(json.dumps(rep, indent=1))


# ----------------------------------------------------------------------------- vcheck

def example_item(x: dict) -> dict:
    """Protocol item for a training example (image_path resolved against the current share mount)."""
    return {"uid": x["uid"], "question": x["question"], "choices": x["choices"], "gold": x["target"],
            "image_path": f"{E.STIM}/{x['image_rel']}"}


def vcheck_messages(sfx: str = "") -> tuple[list[dict], list[list[dict]]]:
    xs = vx.read_jsonl(f"{EX}/check_sample{sfx}.jsonl")
    msgs = [train_messages(example_item(x), x["rationale"], vx.img_ref(E.load_image(example_item(x)["image_path"]))) for x in xs]
    return xs, msgs


def vcheck(model: str, dry: bool, sfx: str = "", llm=None) -> dict:
    """Readout of the check sample: prompt token ids fed to the engine + top-20 letter logprobs."""
    xs, msgs = vcheck_messages(sfx)
    if dry:
        print(_show(msgs[0])[:800])
        print(f"[e63 vcheck] dry ok: {len(msgs)} prompts")
        return {}
    from vllm import SamplingParams
    llm = llm or vx.make_llm(model)
    res = llm.chat(msgs, SamplingParams(temperature=0.0, max_tokens=1, logprobs=20), **vx.chat_kw(model))
    out = []
    for x, r in zip(xs, res):
        top = {lp.decoded_token: lp.logprob for lp in r.outputs[0].logprobs[0].values()}
        out.append({"ex_id": x["ex_id"], "target": x["target"], "prompt_token_ids": list(r.prompt_token_ids),
                    "letters": C.letter_scores(top, x["choices"])})
    rec = {"model": model, "model_path": C.model_path(model), "n": len(out), "rows": out}
    json.dump(rec, open(f"{E.E63}/vcheck_{model}{sfx}.json", "w"))
    print(f"[e63 vcheck] {model}: {len(out)} rows -> vcheck_{model}{sfx}.json", flush=True)
    return rec


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["gen", "build", "vcheck"])
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshard", type=int, default=8)
    ap.add_argument("--limit-pairs", type=int, default=None)
    ap.add_argument("--max-cot", type=int, default=512)
    ap.add_argument("--model", default=E.BASE)
    ap.add_argument("--dry", action="store_true")
    a = ap.parse_args()
    os.makedirs(GEN, exist_ok=True)
    sfx = f"_limit{a.limit_pairs}" if a.limit_pairs else ""
    if a.stage == "gen":
        gen(a.shard, a.nshard, a.limit_pairs, a.dry, a.max_cot)
    elif a.stage == "build":
        build(a.nshard, sfx)
    else:
        vcheck(a.model, a.dry, sfx)


if __name__ == "__main__":
    main()
