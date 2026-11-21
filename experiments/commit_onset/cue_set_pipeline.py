"""E28: cue-factorial evidence for the competent models.

Parts (one vLLM engine per process; all prompts / readouts reuse the existing protocol code):
  desc   question-blind description arm (cues only): turn 1 = image + e5x.DESC_PROMPT (exactly as the twins arm of
         experiments/occlusion/twins_readout.py, Tab. S14 'Desc.'), greedy description; turn 2 = the question as a new user turn; forced readout
         "Answer: (". The description is question-blind, so it is generated once per image (800) and shared by both
         phrasings (e2d generated it twice with identical greedy text). Direct is re-read on the same engine (sanity).
  traj   sentence-level trajectory of the stored CoT rationale (cues: e10 *_cues.jsonl, the CoT arm behind the cue GLM;
         twins: e2d front + _behind, the CoT arm behind Tab. S13): e2b split, prefix R_{1:j} closed as an assistant turn,
         "Now give your final answer.", forced readout (experiments/commit_onset/sentence_trajectory.py layout), j = 0..J (J <= 30).
  edit   claim edit at the commit sentence j* (e20 rule: first j >= 1 with P_{j-1}(gold) >= .5 and P_k(gold) < .5 for all
         k >= j) on D+C- items: sentence j* -> orig (kept) / correct ("The <front> is in front of the <back>, covering part
         of it.", e2f template) / neutral ("Let me look at the two shapes in the image carefully.", e2f), list marker kept;
         everything after j* removed; greedy same-turn continuation (<= 512 tokens, e2f 'same' mode); same-turn readout.
-> outputs/e28/<model>_<set>_{desc,traj,edit}.jsonl (+ .timing.json)

Usage: python -m experiments.commit_onset.cue_set_pipeline --model qwen3vl_8b --set cues --parts desc,traj,edit [--limit 5]
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
from experiments.commit_onset.sentence_trajectory import body, split_sentences
from experiments.occlusion.twins_readout import load_occ
from experiments.behavior.reasoning_formats import DESC_PROMPT
from experiments.claim_edits.first_claim_edit import HEAD, NEUTRAL

MAX_SENT = 30


def am(lp: dict) -> str | None:
    fin = {k: v for k, v in lp.items() if v is not None and math.isfinite(v)}
    return max(fin, key=fin.get) if fin else None


def p_gold(lp: dict, gold: str) -> float:
    pr = vx.probs_from_lp(lp) or {}
    return float(pr.get(gold, 0.0))


def commit_idx(pg: list[float]) -> int | None:
    """0-based sentence index of j* (e20_commit_judge.commit_idx on the trajectory list pg[0..J])."""
    js = next((j for j in range(1, len(pg)) if pg[j - 1] >= 0.5 and all(v < 0.5 for v in pg[j:])), None)
    return None if js is None else js - 1


def load_set(name: str, limit: int | None) -> list[dict]:
    d = os.environ.get("CUES_DIR", f"{_vtp.ITEMS}/occ_cues") if name == "cues" else None
    items = [dict(it, phrasing="front", key=it["uid"]) for it in load_occ(limit, items_dir=d)]
    items += [dict(it, phrasing="behind", key=it["uid"] + "_behind") for it in load_occ(limit, polar=True, items_dir=d)]
    return items


def source_records(model: str, name: str) -> dict:
    """{(key, condition): record} of the stored Direct / CoT runs."""
    out = {}
    if name == "cues":
        for r in vx.read_jsonl(f"{vx.OUT}/e10/{model}_cues.jsonl"):
            if r["condition"] in ("direct", "cot"):
                out[(r["item_id"], r["condition"])] = r
    else:
        for fn, sfx in ((f"{model}.jsonl", ""), (f"{model}_behind.jsonl", "_behind")):
            for r in vx.read_jsonl(f"{vx.OUT}/e2d/{fn}"):
                if r["condition"] in ("direct", "cot"):
                    out[(r["item_id"] + sfx, r["condition"])] = r
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--set", required=True, choices=["cues", "twins"])
    ap.add_argument("--parts", default="desc,traj,edit")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()
    parts = args.parts.split(",")
    sfx = f"_limit{args.limit}" if args.limit else ""
    stem = f"{vx.OUT}/e28/{args.model}_{args.set}"
    os.makedirs(f"{vx.OUT}/e28", exist_ok=True)
    t0 = time.time()
    tm: dict = {}
    if os.environ.get("E22_NO_MM_CACHE"):
        # vLLM 0.8.4 V1 mm_input_cache desync with Molmo on the cue set (Exp-22): preprocessor cache off
        import functools
        import vllm
        vllm.LLM = functools.partial(vllm.LLM, disable_mm_preprocessor_cache=True)
    if os.environ.get("E28_MAX_BATCHED"):
        # Qwen3-VL cue set: the ViT OOMed on a 1,600-prompt Direct batch at max_num_batched_tokens=16384
        import functools
        import vllm
        vllm.LLM = functools.partial(vllm.LLM, max_num_batched_tokens=int(os.environ["E28_MAX_BATCHED"]))
    items = load_set(args.set, args.limit)
    src = source_records(args.model, args.set)
    items = [it for it in items if (it["key"], "direct") in src and (it["key"], "cot") in src]
    llm = vx.make_llm(args.model)
    refs = [vx.img_ref(C.get_image(it)) for it in items]
    ch = [it["choices"] for it in items]
    n = len(items)
    print(f"[e28] {args.model} {args.set}: {n} prompts, parts {parts}", flush=True)
    tm["load"] = time.time() - t0

    def base(it: dict) -> dict:
        d, c = src[(it["key"], "direct")], src[(it["key"], "cot")]
        dp, cp = am(d["lp_sum"]), am(c["lp_sum"])
        return {"model": args.model, "set": args.set, "task": it["task"], "item_id": it["key"], "base_id": it["uid"],
                "phrasing": it["phrasing"], "gold": it["gold"], "letters": it["choices"], "image_id": it["group"],
                "question": it["question"], "meta": it["meta"], "src_direct": dp, "src_cot": cp,
                "group": ("D+" if dp == it["gold"] else "D-") + ("C+" if cp == it["gold"] else "C-")}

    # ------------------------------------------------------------------ desc
    if "desc" in parts and args.set == "cues":
        t1 = time.time()
        uimg = {}
        for i, it in enumerate(items):
            uimg.setdefault(it["uid"], i)
        order = list(uimg)
        gm = [[{"role": "user", "content": [refs[uimg[u]], {"type": "text", "text": DESC_PROMPT}]}] for u in order]
        if args.model.startswith("molmo"):
            gd = vx.generate(llm, args.model, [m + [{"role": "assistant", "content": ""}] for m in gm], cont=True)
        else:
            gd = vx.generate(llm, args.model, gm, cont=False)
        desc = dict(zip(order, gd))
        tm["desc_gen"] = time.time() - t1
        t1 = time.time()
        ro = [[{"role": "user", "content": [r, {"type": "text", "text": DESC_PROMPT}]},
               {"role": "assistant", "content": desc[it["uid"]]["text"]},
               {"role": "user", "content": [{"type": "text", "text": it["question"]}]},
               {"role": "assistant", "content": C.FORCE_PREFIX}] for it, r in zip(items, refs)]
        outs = vx.readout(llm, args.model, ro, ch)
        ro_d = vx.readout(llm, args.model, [C.build_messages(it, "direct", "none", "", r) for it, r in zip(items, refs)], ch)
        tm["desc_readout"] = time.time() - t1
        recs = []
        for it, o, od in zip(items, outs, ro_d):
            g = desc[it["uid"]]
            recs.append(base(it) | {"condition": "desc", "text": g["text"], "gen_len": g["len"], "trunc": g["trunc"], **o})
            recs.append(base(it) | {"condition": "direct_reread", **od})
        vx.write_jsonl(f"{stem}_desc{sfx}.jsonl", recs)
        print(f"[e28] desc: {len(order)} descriptions, {len(recs)} records ({time.time() - t0:.0f}s)", flush=True)

    # ------------------------------------------------------------------ traj
    traj_by = {}
    if "traj" in parts or "edit" in parts:
        tp = f"{stem}_traj{sfx}.jsonl"
        if "traj" not in parts and os.path.exists(tp):
            traj_by = {r["item_id"]: r for r in vx.read_jsonl(tp)}
        else:
            t1 = time.time()
            msgs, cks, keys, sents_all = [], [], [], []
            for i, it in enumerate(items):
                sents = split_sentences(body(src[(it["key"], "cot")]["text"]))[:MAX_SENT]
                sents_all.append(sents)
                for j in range(len(sents) + 1):
                    prefix = (C.COT_PREFIX + " " + " ".join(sents[:j])).strip()
                    msgs.append(C.build_messages(it, "cot", "turn", prefix, refs[i]))
                    cks.append(it["choices"])
                    keys.append((i, j))
            print(f"[e28] traj: {len(msgs)} prefix readouts", flush=True)
            outs = vx.readout(llm, args.model, msgs, cks)
            tr: dict[int, list] = {}
            for (i, j), o in zip(keys, outs):
                tr.setdefault(i, []).append({"j": j, "lp_sum": o["lp_sum"], "top20": o["top20"], "pred_exact": o["pred_exact"]})
            recs = []
            for i, it in enumerate(items):
                pg = [p_gold(t["lp_sum"], it["gold"]) for t in tr[i]]
                js = commit_idx(pg)
                recs.append(base(it) | {"sentences": sents_all[i], "traj": tr[i], "p_gold": pg, "commit_idx": js,
                                        "commit_sentence": sents_all[i][js] if js is not None else None})
            vx.write_jsonl(tp, recs)
            traj_by = {r["item_id"]: r for r in recs}
            tm["traj"] = time.time() - t1
            print(f"[e28] traj done ({time.time() - t0:.0f}s)", flush=True)

    # ------------------------------------------------------------------ edit at j*
    if "edit" in parts:
        t1 = time.time()
        jobs, gen_msgs = [], []
        for i, it in enumerate(items):
            r = traj_by.get(it["key"])
            if r is None or r["group"] != "D+C-" or r["commit_idx"] is None:
                continue
            p, sents = r["commit_idx"], r["sentences"]
            head = HEAD.match(sents[p]).group(0)
            front, back = it["meta"]["front_name"], it["meta"]["back_name"]
            new = {"orig": sents[p], "correct": head + f"The {front} is in front of the {back}, covering part of it.",
                   "neutral": head + NEUTRAL}
            for cond, s in new.items():
                prefix = (C.COT_PREFIX + " " + " ".join(sents[:p] + [s])).strip()
                m = C.cot_messages(it, refs[i])
                m[-1] = {"role": "assistant", "content": prefix}
                jobs.append((i, cond, prefix, p))
                gen_msgs.append(m)
        print(f"[e28] edit: {len(jobs)} continuations", flush=True)
        gens = vx.generate(llm, args.model, gen_msgs, cont=True) if jobs else []
        ro = [C.build_messages(items[i], "cot", "none", prefix + " " + g["text"], refs[i]) for (i, _, prefix, _), g in zip(jobs, gens)]
        outs = vx.readout(llm, args.model, ro, [items[i]["choices"] for i, *_ in jobs]) if jobs else []
        recs = []
        for (i, cond, prefix, p), g, o in zip(jobs, gens, outs):
            it = items[i]
            recs.append(base(it) | {"cond": cond, "commit_idx": p, "n_sent": len(traj_by[it["key"]]["sentences"]),
                                    "orig_sentence": traj_by[it["key"]]["sentences"][p], "prefix": prefix,
                                    "continuation": g["text"], "cont_len": g["len"], "trunc": g["trunc"],
                                    "stated": C.parse_free_answer(g["text"], it["choices"]), **o})
        vx.write_jsonl(f"{stem}_edit{sfx}.jsonl", recs)
        tm["edit"] = time.time() - t1
        print(f"[e28] edit: {len(recs)} records ({time.time() - t0:.0f}s)", flush=True)
    tm["total"] = time.time() - t0
    json.dump(tm, open(f"{stem}_{'-'.join(parts)}{sfx}.timing.json", "w"), indent=1)
    print(f"[e28] done {json.dumps({k: round(v) for k, v in tm.items()})}", flush=True)


if __name__ == "__main__":
    main()
