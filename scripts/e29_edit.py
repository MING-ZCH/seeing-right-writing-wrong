"""E29 claim edits on the cue-factorial set / twins (round-17 external: Opus M2, Gemini W2).

Same protocol as the Exp-28 edit at j* (e28_run.py): sentence s_p of the stored CoT rationale is replaced (list marker
kept), everything after it removed, greedy same-turn continuation (<= 512 tokens), same-turn readout "Answer: (".
Arms (all applied to every selected item, so template x question phrasing is fully crossed):
  orig     s_p kept (regeneration baseline)
  neutral  "Let me look at the two shapes in the image carefully." (e2f)
  F0       "The <front> is in front of the <back>, covering part of it."      (Exp-28 template, 'in front of' wording)
  F1       "Looking at the two shapes carefully, the <front> is in front of the <back>."   (neutral-style paraphrase)
  F2       mined: most frequent true 'in front of' sentence of the model's own cue-set D+C+ rationales (rule in mine())
  B0       "The <back> is behind the <front>, partly hidden by it."           ('behind' wording)
  B1       "Looking at the two shapes carefully, the <back> is behind the <front>."
  B2       mined: most frequent true 'behind' sentence of the model's own D+C+ rationales
  ECHO     the question's own subject, object and predicate with the true polarity:
           "The <x> is [not] in front of / behind the <y>."
Parts:
  jstar  p = j* (e20 rule, Exp-28 trajectories) on every D+C- item with j*.
  dcp    control without D+C- selection: up to 100 D+C+ items per phrasing (seed 292) with a first claim
         (e20 first_claim_idx), p = that sentence.
  early  p = earliest consensus image-contradicting relational sentence before j* (outputs/e29/earliest_<m>_<set>.json
         from rigor_e29.py prej2), arms orig / neutral / F0 / B0 / ECHO.
-> outputs/e29/<model>_<set>_edit_<part>.jsonl (+ .timing.json, templates in the records)

Usage: python e29_edit.py --model molmo7b --set cues --parts jstar,dcp [--limit 5]
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
import time
from collections import Counter

import vx
from vx import C
from e2f_claim_edit import HEAD, NEUTRAL
from e28_run import am, load_set, source_records

E28 = f"{vx.OUT}/e28"
E29 = f"{vx.OUT}/e29"
FIXED = {"F0": "The {front} is in front of the {back}, covering part of it.",
         "F1": "Looking at the two shapes carefully, the {front} is in front of the {back}.",
         "B0": "The {back} is behind the {front}, partly hidden by it.",
         "B1": "Looking at the two shapes carefully, the {back} is behind the {front}."}
# e20_commit_judge.first_claim_idx (copied: importing e20 pulls build_bundle, which clashes with legacy common in the pod)
OBJ = re.compile(r"\b(red|blue|green|yellow|purple|orange|cyan|gray|grey|pink|brown|black|white)\s+"
                 r"(circle|square|triangle|diamond|hexagon|pentagon|star|rectangle)s?\b", re.I)
ACT = re.compile(r"\b(overlap\w*|cover\w*|on top of|obscur\w*|block\w*|in front of)\b", re.I)
PAS = re.compile(r"\b(covered|hidden|obscured|blocked|overlapped|occluded)\s+by\b|\bbehind\b|\bunderneath\b|\bbeneath\b", re.I)


def first_claim_idx(sents: list[str]) -> int | None:
    for j, s in enumerate(sents):
        objs = {(o.group(1) + " " + o.group(2)).lower() for o in OBJ.finditer(s)}
        if len(objs) >= 2 and (ACT.search(s) or PAS.search(s)) and not re.search(r"\?|whether|if the|determine|asks", s, re.I):
            return j
    return None


META = re.compile(r"^(given|therefore|so\b|thus|this means|hence|in conclusion|the question|the description|it states|"
                  r"it is described|based on|the image shows|we are|looking at|answer|final|since)", re.I)


def mine(traj: list[dict], pred: str) -> tuple[str, int, list]:
    """Most frequent true-relation sentence with predicate `pred` in the model's D+C+ rationales.

    Shape names -> {front}/{back}; list markers and ** removed; excluded: conclusion / meta openers (META), negations,
    questions, quotes, 'the answer'; the sentence must name each shape once and state the true relation in the order
    {front} .. 'in front of' .. {back} (pred=front) or {back} .. 'behind' .. {front} (pred=behind), without the other
    predicate word. Counted once per rationale.
    """
    cnt = Counter()
    for t in traj:
        if t["group"] != "D+C+":
            continue
        f, b = t["meta"]["front_name"], t["meta"]["back_name"]
        seen = set()
        for s in t["sentences"]:
            s2 = HEAD.sub("", s).replace("**", "").strip()
            s2 = re.sub(r"\b(an?|the)\s+" + re.escape(f) + r"\b", "the {front}", s2, flags=re.I)
            s2 = re.sub(r"\b(an?|the)\s+" + re.escape(b) + r"\b", "the {back}", s2, flags=re.I)
            if s2.count("{front}") != 1 or s2.count("{back}") != 1:
                continue
            if META.match(s2) or re.search(r"\bnot\b|n't|\?|\"|the answer", s2, re.I):
                continue
            if pred == "front":
                ok = "in front of" in s2 and "behind" not in s2 and s2.index("{front}") < s2.index("in front of") < s2.index("{back}")
            else:
                ok = "behind" in s2 and "in front" not in s2 and s2.index("{back}") < s2.index("behind") < s2.index("{front}")
            if ok and s2 not in seen:
                seen.add(s2)
                cnt[s2] += 1
    (top, n), = cnt.most_common(1)
    return top[0].upper() + top[1:], n, cnt.most_common(5)


def echo(it: dict) -> str:
    mt = it["meta"]
    x, y = mt["x_name"], mt["y_name"]
    if it["phrasing"] == "front":
        return f"The {x} is in front of the {y}." if x == mt["front_name"] else f"The {x} is not in front of the {y}."
    return f"The {x} is behind the {y}." if x == mt["back_name"] else f"The {x} is not behind the {y}."


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--set", required=True, choices=["cues", "twins"])
    ap.add_argument("--parts", default="jstar,dcp")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()
    parts = args.parts.split(",")
    sfx = f"_limit{args.limit}" if args.limit else ""
    os.makedirs(E29, exist_ok=True)
    t0 = time.time()
    tm: dict = {}
    import functools
    import vllm
    if os.environ.get("E22_NO_MM_CACHE"):
        vllm.LLM = functools.partial(vllm.LLM, disable_mm_preprocessor_cache=True)
    if os.environ.get("E28_MAX_BATCHED"):
        vllm.LLM = functools.partial(vllm.LLM, max_num_batched_tokens=int(os.environ["E28_MAX_BATCHED"]))
    items = load_set(args.set, None)
    src = source_records(args.model, args.set)
    items = [it for it in items if (it["key"], "direct") in src and (it["key"], "cot") in src]
    by_key = {it["key"]: it for it in items}
    traj = vx.read_jsonl(f"{E28}/{args.model}_{args.set}_traj.jsonl")
    T = {r["item_id"]: r for r in traj}
    cue_traj = traj if args.set == "cues" else vx.read_jsonl(f"{E28}/{args.model}_cues_traj.jsonl")
    tmpl = dict(FIXED)
    mined = {}
    for k, pred in (("F2", "front"), ("B2", "behind")):
        s, n, top5 = mine(cue_traj, pred)
        tmpl[k] = s
        mined[k] = {"template": s, "count": n, "top5": top5}
    print(f"[e29] templates {json.dumps(tmpl)}", flush=True)

    jobs = []  # (part, item_key, p, arm, sentence)
    if "jstar" in parts:
        sel = [r for r in traj if r["group"] == "D+C-" and r["commit_idx"] is not None]
        sel = sel[:args.limit] if args.limit else sel
        for r in sel:
            jobs += [("jstar", r["item_id"], r["commit_idx"], a, None) for a in ["orig", "neutral", *tmpl, "ECHO"]]
    if "dcp" in parts:
        rng = random.Random(292)
        for ph in ("front", "behind"):
            pool = sorted((r for r in traj if r["group"] == "D+C+" and r["phrasing"] == ph and r["sentences"]
                           and first_claim_idx(r["sentences"]) is not None), key=lambda r: r["item_id"])
            sel = rng.sample(pool, min(100, len(pool)))
            sel = sel[:args.limit] if args.limit else sel
            for r in sel:
                jobs += [("dcp", r["item_id"], first_claim_idx(r["sentences"]), a, None) for a in ["orig", "neutral", *tmpl, "ECHO"]]
    if "early" in parts:
        ek = json.load(open(f"{E29}/earliest_{args.model}_{args.set}.json"))
        sel = sorted(ek.items())[:args.limit] if args.limit else sorted(ek.items())
        for iid, k in sel:
            if k < T[iid]["commit_idx"]:
                jobs += [("early", iid, k, a, None) for a in ["orig", "neutral", "F0", "B0", "ECHO"]]
    print(f"[e29] {args.model} {args.set}: {len(jobs)} continuations {dict(Counter((j[0], j[3]) for j in jobs))}", flush=True)

    llm = vx.make_llm(args.model)
    tm["load"] = time.time() - t0
    t1 = time.time()
    refs = {}
    gen_msgs, metas = [], []
    for part, iid, p, arm, _ in jobs:
        it, r = by_key[iid], T[iid]
        if iid not in refs:
            refs[iid] = vx.img_ref(C.get_image(it))
        sents = r["sentences"]
        head = HEAD.match(sents[p]).group(0)
        mt = it["meta"]
        if arm == "orig":
            s = sents[p]
        elif arm == "neutral":
            s = head + NEUTRAL
        elif arm == "ECHO":
            s = head + echo(it)
        else:
            s = head + tmpl[arm].format(front=mt["front_name"], back=mt["back_name"])
        prefix = (C.COT_PREFIX + " " + " ".join(sents[:p] + [s])).strip()
        m = C.cot_messages(it, refs[iid])
        m[-1] = {"role": "assistant", "content": prefix}
        gen_msgs.append(m)
        metas.append((part, iid, p, arm, s, prefix))
    gens = vx.generate(llm, args.model, gen_msgs, cont=True) if jobs else []
    tm["gen"] = time.time() - t1
    t1 = time.time()
    ro = [C.build_messages(by_key[iid], "cot", "none", prefix + " " + g["text"], refs[iid])
          for (_, iid, _, _, _, prefix), g in zip(metas, gens)]
    outs = vx.readout(llm, args.model, ro, [by_key[iid]["choices"] for _, iid, *_ in metas]) if jobs else []
    tm["readout"] = time.time() - t1
    recs = {}
    for (part, iid, p, arm, s, prefix), g, o in zip(metas, gens, outs):
        it, r = by_key[iid], T[iid]
        d, c = src[(iid, "direct")], src[(iid, "cot")]
        recs.setdefault(part, []).append(
            {"model": args.model, "set": args.set, "part": part, "item_id": iid, "base_id": it["uid"], "image_id": it["group"],
             "phrasing": it["phrasing"], "gold": it["gold"], "meta": it["meta"], "group": r["group"],
             "src_direct": am(d["lp_sum"]), "src_cot": am(c["lp_sum"]), "commit_idx": r["commit_idx"], "edit_idx": p,
             "n_sent": len(r["sentences"]), "arm": arm, "orig_sentence": r["sentences"][p], "new_sentence": s,
             "continuation": g["text"], "cont_len": g["len"], "trunc": g["trunc"],
             "stated": C.parse_free_answer(g["text"], it["choices"]), **o})
    for part, rs in recs.items():
        vx.write_jsonl(f"{E29}/{args.model}_{args.set}_edit_{part}{sfx}.jsonl", rs)
        print(f"[e29] {part}: {len(rs)} records", flush=True)
    tm["total"] = time.time() - t0
    json.dump({"timing": tm, "templates": tmpl, "mined": mined, "n_jobs": len(jobs)},
              open(f"{E29}/{args.model}_{args.set}_edit_{'-'.join(parts)}{sfx}.timing.json", "w"), indent=1)
    print(f"[e29] done {json.dumps({k: round(v) for k, v in tm.items()})}", flush=True)


if __name__ == "__main__":
    main()
