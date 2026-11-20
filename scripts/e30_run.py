"""E30 GPU parts (round-18 external reviews, section B). One vLLM engine per process; protocol code reused.

  argedit  Argument-order edit (Opus M2). Exp-29 protocol (e29_edit.py): sentence s_p replaced (list marker kept), text
           after it removed, greedy same-turn continuation (<= 512), same-turn readout. Items = Exp-29 selections:
           jstar (all D+C- with j*) and dcp (D+C+ control, 100 + 100, seed 292, edit at the first claim). Arms
           (all TRUE; F = true front shape, K = true back shape):
             orig, neutral,
             FS "The F is in front of the K."        predicate 'in front of', subject F
             FO "The K has the F in front of it."    predicate 'in front of', subject K
             FN "The K is not in front of the F."    predicate 'in front of', subject K, negated
             BS "The K is behind the F."             predicate 'behind',      subject K
             BO "The F has the K behind it."         predicate 'behind',      subject F
             BN "The F is not behind the K."         predicate 'behind',      subject F, negated
           Subject match (analysis) = sentence subject is the question's subject X.
  ndesc    Occlusion-neutral description arm (Carbon MAJOR-2), cue set: turn 1 = image + NDESC_PROMPT (shape, colour,
           2D position only), greedy description once per image, question in a new user turn, forced readout. Fillers
           (legacy filler_text, n = the Exp-28 occlusion-prompted description length of that image, per model):
             filler_desc   image + DESC_PROMPT (Exp-28 wording) / filler / question turn / readout
             filler_ndesc  image + NDESC_PROMPT / filler / question turn / readout
             filler_turn   image + question / filler / "Now give your final answer." / readout (Fil(|T|)+turn layout)
           plus a Direct re-read.
  gray     Gray-image prefix readout (Opus M5): 120 cue-set D+C- rationales with j* per model (Qwen3-VL-8B: the 120 of
           the Exp-29 pre-j* analysis; Molmo-7B-D: seed 300), e28 traj layout, j = 0..J, real and uniform-gray image.
-> outputs/e30/<model>_<set>_{argedit_<part>,ndesc,gray}.jsonl (+ timing)

Usage: python e30_run.py --model molmo7b --set cues --parts argedit,ndesc,gray [--limit 3]
"""
from __future__ import annotations

import argparse
import json
import os
import random
import time
from collections import Counter

from PIL import Image

import vx
from vx import C
from e2b import body, split_sentences
from e5x import DESC_PROMPT
from e2f_claim_edit import HEAD, NEUTRAL
from e28_run import MAX_SENT, am, load_set, p_gold, commit_idx, source_records
from e29_edit import first_claim_idx

E28, E29, E30 = f"{vx.OUT}/e28", f"{vx.OUT}/e29", f"{vx.OUT}/e30"
NDESC_PROMPT = ("Describe this image: list each object with its shape, its color, and its position in the image "
                "(left, right, top, bottom, or center).")
ARG = {"FS": ("The {F} is in front of the {K}.", "front", "F", False),
       "FO": ("The {K} has the {F} in front of it.", "front", "K", False),
       "FN": ("The {K} is not in front of the {F}.", "front", "K", True),
       "BS": ("The {K} is behind the {F}.", "behind", "K", False),
       "BO": ("The {F} has the {K} behind it.", "behind", "F", False),
       "BN": ("The {F} is not behind the {K}.", "behind", "F", True)}


def patch_vllm() -> None:
    import functools
    import vllm
    if os.environ.get("E22_NO_MM_CACHE"):
        vllm.LLM = functools.partial(vllm.LLM, disable_mm_preprocessor_cache=True)
    if os.environ.get("E28_MAX_BATCHED"):
        vllm.LLM = functools.partial(vllm.LLM, max_num_batched_tokens=int(os.environ["E28_MAX_BATCHED"]))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--set", required=True, choices=["cues", "twins"])
    ap.add_argument("--parts", default="argedit")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()
    parts = args.parts.split(",")
    sfx = f"_limit{args.limit}" if args.limit else ""
    os.makedirs(E30, exist_ok=True)
    stem = f"{E30}/{args.model}_{args.set}"
    t0 = time.time()
    tm: dict = {}
    patch_vllm()
    items = load_set(args.set, None)
    src = source_records(args.model, args.set)
    items = [it for it in items if (it["key"], "direct") in src and (it["key"], "cot") in src]
    by_key = {it["key"]: it for it in items}
    T = {r["item_id"]: r for r in vx.read_jsonl(f"{E28}/{args.model}_{args.set}_traj.jsonl")}
    llm = vx.make_llm(args.model)
    tm["load"] = time.time() - t0
    refs: dict = {}

    def ref(key: str) -> dict:
        if key not in refs:
            refs[key] = vx.img_ref(C.get_image(by_key[key]))
        return refs[key]

    def base(it: dict) -> dict:
        d, c = src[(it["key"], "direct")], src[(it["key"], "cot")]
        dp, cp = am(d["lp_sum"]), am(c["lp_sum"])
        return {"model": args.model, "set": args.set, "item_id": it["key"], "base_id": it["uid"], "image_id": it["group"],
                "phrasing": it["phrasing"], "gold": it["gold"], "meta": it["meta"], "src_direct": dp, "src_cot": cp,
                "group": ("D+" if dp == it["gold"] else "D-") + ("C+" if cp == it["gold"] else "C-")}

    # ------------------------------------------------------------------ argument-order edit
    if "argedit" in parts:
        t1 = time.time()
        traj = sorted(T.values(), key=lambda r: r["item_id"])
        jobs = []
        sel = [r for r in traj if r["group"] == "D+C-" and r["commit_idx"] is not None]
        for r in sel[:args.limit] if args.limit else sel:
            jobs += [("jstar", r["item_id"], r["commit_idx"], a) for a in ["orig", "neutral", *ARG]]
        rng = random.Random(292)  # = e29_edit dcp selection
        for ph in ("front", "behind"):
            pool = sorted((r for r in traj if r["group"] == "D+C+" and r["phrasing"] == ph and r["sentences"]
                           and first_claim_idx(r["sentences"]) is not None), key=lambda r: r["item_id"])
            s2 = rng.sample(pool, min(100, len(pool)))
            for r in s2[:args.limit] if args.limit else s2:
                jobs += [("dcp", r["item_id"], first_claim_idx(r["sentences"]), a) for a in ["orig", "neutral", *ARG]]
        print(f"[e30] argedit {args.model} {args.set}: {len(jobs)} continuations {dict(Counter((j[0], j[3]) for j in jobs))}", flush=True)
        gen_msgs, metas = [], []
        for part, iid, p, arm in jobs:
            it, sents = by_key[iid], T[iid]["sentences"]
            head = HEAD.match(sents[p]).group(0)
            F, K = it["meta"]["front_name"], it["meta"]["back_name"]
            if arm == "orig":
                s = sents[p]
            elif arm == "neutral":
                s = head + NEUTRAL
            else:
                s = head + ARG[arm][0].format(F=F, K=K)
            prefix = (C.COT_PREFIX + " " + " ".join(sents[:p] + [s])).strip()
            m = C.cot_messages(it, ref(iid))
            m[-1] = {"role": "assistant", "content": prefix}
            gen_msgs.append(m)
            metas.append((part, iid, p, arm, s, prefix))
        gens = vx.generate(llm, args.model, gen_msgs, cont=True)
        ro = [C.build_messages(by_key[iid], "cot", "none", prefix + " " + g["text"], ref(iid)) for (_, iid, _, _, _, prefix), g in zip(metas, gens)]
        outs = vx.readout(llm, args.model, ro, [by_key[iid]["choices"] for _, iid, *_ in metas])
        recs: dict = {}
        for (part, iid, p, arm, s, prefix), g, o in zip(metas, gens, outs):
            it = by_key[iid]
            mt = it["meta"]
            info = {}
            if arm in ARG:
                _, sp, subj, neg = ARG[arm]
                sname = mt["front_name"] if subj == "F" else mt["back_name"]
                info = {"sent_pred": sp, "sent_subject": sname, "negated": neg, "subject_match": sname == mt["x_name"],
                        "pred_match": sp == it["phrasing"]}
            recs.setdefault(part, []).append(base(it) | {"part": part, "commit_idx": T[iid]["commit_idx"], "edit_idx": p,
                                                         "arm": arm, "orig_sentence": T[iid]["sentences"][p], "new_sentence": s,
                                                         "continuation": g["text"], "cont_len": g["len"], "trunc": g["trunc"],
                                                         "stated": C.parse_free_answer(g["text"], it["choices"]), **info, **o})
        for part, rs in recs.items():
            vx.write_jsonl(f"{stem}_argedit_{part}{sfx}.jsonl", rs)
            print(f"[e30] argedit {part}: {len(rs)} records", flush=True)
        tm["argedit"] = time.time() - t1

    # ------------------------------------------------------------------ occlusion-neutral description + fillers
    if "ndesc" in parts and args.set == "cues":
        t1 = time.time()
        keep = {it["uid"] for it in items[:args.limit]} if args.limit else None
        its = [it for it in items if keep is None or it["uid"] in keep]
        dlen = {}
        for r in vx.read_jsonl(f"{E28}/{args.model}_cues_desc.jsonl"):
            if r["condition"] == "desc":
                dlen[r["base_id"]] = int(r["gen_len"])
        uimg = {}
        for it in its:
            uimg.setdefault(it["uid"], it["key"])
        order = list(uimg)
        gm = [[{"role": "user", "content": [ref(uimg[u]), {"type": "text", "text": NDESC_PROMPT}]}] for u in order]
        if args.model.startswith("molmo"):
            gd = vx.generate(llm, args.model, [m + [{"role": "assistant", "content": ""}] for m in gm], cont=True)
        else:
            gd = vx.generate(llm, args.model, gm, cont=False)
        nd = dict(zip(order, gd))
        tok = llm.get_tokenizer()
        fill = {u: C.filler_text(tok, dlen[u]) for u in order}
        flen = {u: len(tok.encode(fill[u], add_special_tokens=False)) for u in order}
        tm["ndesc_gen"] = time.time() - t1
        arms = {
            "ndesc": lambda it, r: [{"role": "user", "content": [r, {"type": "text", "text": NDESC_PROMPT}]},
                                    {"role": "assistant", "content": nd[it["uid"]]["text"]},
                                    {"role": "user", "content": [{"type": "text", "text": it["question"]}]},
                                    {"role": "assistant", "content": C.FORCE_PREFIX}],
            "filler_desc": lambda it, r: [{"role": "user", "content": [r, {"type": "text", "text": DESC_PROMPT}]},
                                          {"role": "assistant", "content": fill[it["uid"]]},
                                          {"role": "user", "content": [{"type": "text", "text": it["question"]}]},
                                          {"role": "assistant", "content": C.FORCE_PREFIX}],
            "filler_ndesc": lambda it, r: [{"role": "user", "content": [r, {"type": "text", "text": NDESC_PROMPT}]},
                                           {"role": "assistant", "content": fill[it["uid"]]},
                                           {"role": "user", "content": [{"type": "text", "text": it["question"]}]},
                                           {"role": "assistant", "content": C.FORCE_PREFIX}],
            "filler_turn": lambda it, r: C.build_messages(it, "filler", "turn", fill[it["uid"]], r),
            "direct_reread": lambda it, r: C.build_messages(it, "direct", "none", "", r),
        }
        recs = []
        for arm, fn in arms.items():
            outs = vx.readout(llm, args.model, [fn(it, ref(it["key"])) for it in its], [it["choices"] for it in its])
            for it, o in zip(its, outs):
                extra = {}
                if arm == "ndesc":
                    g = nd[it["uid"]]
                    extra = {"text": g["text"], "gen_len": g["len"], "trunc": g["trunc"]}
                elif arm.startswith("filler"):
                    extra = {"filler_len": flen[it["uid"]], "target_len": dlen[it["uid"]]}
                recs.append(base(it) | {"condition": arm, **extra, **o})
            print(f"[e30] ndesc arm {arm} done ({time.time() - t0:.0f}s)", flush=True)
        vx.write_jsonl(f"{stem}_ndesc{sfx}.jsonl", recs)
        tm["ndesc"] = time.time() - t1

    # ------------------------------------------------------------------ gray-image prefix readout
    if "gray" in parts:
        t1 = time.time()
        if args.model == "qwen3vl_8b" and args.set == "cues":
            ids = sorted({r["item_id"] for r in vx.read_jsonl(f"{E29}/judge_gpt-6-astra_prej2.jsonl")})
        else:
            pool = sorted(r["item_id"] for r in T.values() if r["group"] == "D+C-" and r["commit_idx"] is not None)
            ids = sorted(random.Random(300).sample(pool, min(120, len(pool))))
        ids = ids[:args.limit] if args.limit else ids
        msgs, cks, keys, n_split_diff = [], [], [], 0
        for iid in ids:
            it = by_key[iid]
            img = C.get_image(it)
            gref = vx.img_ref(Image.new("RGB", img.size, vx.GRAY))
            sents = T[iid]["sentences"]
            n_split_diff += sents != split_sentences(body(src[(iid, "cot")]["text"]))[:MAX_SENT]
            for j in range(len(sents) + 1):
                prefix = (C.COT_PREFIX + " " + " ".join(sents[:j])).strip()
                for v, r in (("real", ref(iid)), ("gray", gref)):
                    msgs.append(C.build_messages(it, "cot", "turn", prefix, r))
                    cks.append(it["choices"])
                    keys.append((iid, j, v))
        print(f"[e30] gray: {len(ids)} rationales, {len(msgs)} readouts, sentence split differs from e28 on {n_split_diff}", flush=True)
        outs = vx.readout(llm, args.model, msgs, cks)
        tr: dict = {}
        for (iid, j, v), o in zip(keys, outs):
            tr.setdefault((iid, v), []).append({"j": j, "lp_sum": o["lp_sum"], "pred_exact": o["pred_exact"]})
        recs = []
        for iid in ids:
            it = by_key[iid]
            row = base(it) | {"n_sent": len(T[iid]["sentences"]), "commit_idx_e28": T[iid]["commit_idx"],
                              "p_gold_e28": T[iid]["p_gold"]}
            for v in ("real", "gray"):
                pg = [p_gold(t["lp_sum"], it["gold"]) for t in tr[(iid, v)]]
                row[f"p_gold_{v}"] = pg
                row[f"commit_idx_{v}"] = commit_idx(pg)
            recs.append(row)
        vx.write_jsonl(f"{stem}_gray{sfx}.jsonl", recs)
        tm["gray"] = time.time() - t1
    tm["total"] = time.time() - t0
    json.dump(tm, open(f"{stem}_{'-'.join(parts)}{sfx}.timing.json", "w"), indent=1)
    print(f"[e30] done {json.dumps({k: round(v) for k, v in tm.items()})}", flush=True)


if __name__ == "__main__":
    main()
