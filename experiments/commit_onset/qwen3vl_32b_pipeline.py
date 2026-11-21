"""E57 GPU parts: the sentence-level pipeline on Qwen3-VL-32B. One vLLM engine per process; every
prompt layout and readout is the protocol code of the earlier runs (imported, not re-implemented).

Source rationales = stored greedy CoT runs: cue set outputs/e10/<m>_cues.jsonl (Exp-31 for 32B), twins
outputs/e2d/<m>{,_behind}.jsonl or, if absent, outputs/scale/<m>{,_behind}.jsonl (Exp-16 ladder for 32B).

Parts (outputs/e57/<model>_<set>_<part>.jsonl):
  reread   Direct re-read of every query (sanity vs the stored Direct readout).
  traj     e28_run traj layout: e2b split (<= 30 sentences), prefix R_{1:j} closed as an assistant turn, "Now give your
           final answer.", forced readout, j = 0..J; j* = e20 rule (commit_idx).
  argedit  e30_run argedit: at j* of every D+C- with j* (part jstar) and at the first claim of 100 + 100 D+C+ items
           (seed 292, part dcp), sentence -> orig / neutral / FS FO FN BS BO BN (all true), text after it removed,
           greedy same-turn continuation (<= 512), same-turn readout. Needs this run's traj file.
  uncond   e2f same-turn claim edit on ALL four Direct x CoT cells: first two-object claim (e2f first_claim) ->
           orig (kept) / correct / wrong / neutral (e2f templates), text after it removed, greedy continuation (<= 512),
           same-turn readout.
  imgswap  e14 own-rationale image swap: R full and R_cut (legacy strip_answer), read out under the own image, the
           identity-swapped twin image (same question, gold flips) and, for R_cut, a uniform gray image.

Usage: python -m experiments.commit_onset.qwen3vl_32b_pipeline --model qwen3vl_32b --set cues --parts reread,traj,argedit,uncond,imgswap [--limit 3]
"""
from __future__ import annotations

import argparse
import json
import os
import random
import time
from collections import Counter

from PIL import Image

from srww import vllm_utils as vx
from srww.vllm_utils import C
from experiments.commit_onset.sentence_trajectory import body, split_sentences
from experiments.claim_edits.first_claim_edit import HEAD, NEUTRAL, edited, first_claim
from experiments.commit_onset.cue_set_pipeline import MAX_SENT, am, commit_idx, load_set, p_gold
from experiments.claim_edits.commit_claim_edit import first_claim_idx
from experiments.claim_edits.argument_order_edit import ARG, patch_vllm

from srww.rcut import strip_answer  # noqa: E402

E57 = f"{vx.OUT}/e57"
UNCOND_CONDS = ("orig", "correct", "wrong", "neutral")


def source_records(model: str, name: str) -> dict:
    """{(key, condition): record} of the stored Direct / CoT runs (cue set: e10; twins: e2d, else scale)."""
    out = {}
    if name == "cues":
        for r in vx.read_jsonl(f"{vx.OUT}/e10/{model}_cues.jsonl"):
            if r["condition"] in ("direct", "cot"):
                out[(r["item_id"], r["condition"])] = r
        return out
    d = "e2d" if os.path.exists(f"{vx.OUT}/e2d/{model}.jsonl") else "scale"
    for fn, sfx in ((f"{model}.jsonl", ""), (f"{model}_behind.jsonl", "_behind")):
        for r in vx.read_jsonl(f"{vx.OUT}/{d}/{fn}"):
            if r["condition"] in ("direct", "cot"):
                out[(r["item_id"] + sfx, r["condition"])] = r
    return out


def twin_of(k: str) -> str:
    """cue000 <-> cue000_cf / occ001 <-> occ001_cf, keeping an optional _behind suffix (= e14)."""
    base, beh = (k[:-7], "_behind") if k.endswith("_behind") else (k, "")
    return (base[:-3] if base.endswith("_cf") else base + "_cf") + beh


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--set", required=True, choices=["cues", "twins"])
    ap.add_argument("--parts", default="reread,traj")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()
    parts = args.parts.split(",")
    sfx = f"_limit{args.limit}" if args.limit else ""
    os.makedirs(E57, exist_ok=True)
    stem = f"{E57}/{args.model}_{args.set}"
    t0 = time.time()
    tm: dict = {}
    patch_vllm()
    items = load_set(args.set, args.limit)
    src = source_records(args.model, args.set)
    items = [it for it in items if (it["key"], "direct") in src and (it["key"], "cot") in src]
    by_key = {it["key"]: it for it in items}
    llm = vx.make_llm(args.model)
    tm["load"] = time.time() - t0
    print(f"[e57] {args.model} {args.set}: {len(items)} prompts, parts {parts} (engine {tm['load']:.0f}s)", flush=True)
    refs: dict = {}

    def ref(key: str) -> dict:
        if key not in refs:
            refs[key] = vx.img_ref(C.get_image(by_key[key]))
        return refs[key]

    def base(it: dict) -> dict:
        d, c = src[(it["key"], "direct")], src[(it["key"], "cot")]
        dp, cp = am(d["lp_sum"]), am(c["lp_sum"])
        return {"model": args.model, "set": args.set, "item_id": it["key"], "base_id": it["uid"], "image_id": it["group"],
                "phrasing": it["phrasing"], "gold": it["gold"], "letters": it["choices"], "meta": it["meta"],
                "src_direct": dp, "src_cot": cp,
                "group": ("D+" if dp == it["gold"] else "D-") + ("C+" if cp == it["gold"] else "C-")}

    def sents_of(it: dict) -> list[str]:
        return split_sentences(body(src[(it["key"], "cot")]["text"]))[:MAX_SENT]

    # ------------------------------------------------------------------ Direct re-read
    if "reread" in parts:
        t1 = time.time()
        outs = vx.readout(llm, args.model, [C.build_messages(it, "direct", "none", "", ref(it["key"])) for it in items],
                          [it["choices"] for it in items])
        vx.write_jsonl(f"{stem}_reread{sfx}.jsonl", [base(it) | {"condition": "direct_reread", **o} for it, o in zip(items, outs)])
        tm["reread"] = time.time() - t1
        print(f"[e57] reread done ({time.time() - t0:.0f}s)", flush=True)

    # ------------------------------------------------------------------ commit trajectories
    T: dict = {}
    tp = f"{stem}_traj{sfx}.jsonl"
    if "traj" in parts:
        t1 = time.time()
        msgs, cks, keys, sents_all = [], [], [], []
        for i, it in enumerate(items):
            sents = sents_of(it)
            sents_all.append(sents)
            for j in range(len(sents) + 1):
                prefix = (C.COT_PREFIX + " " + " ".join(sents[:j])).strip()
                msgs.append(C.build_messages(it, "cot", "turn", prefix, ref(it["key"])))
                cks.append(it["choices"])
                keys.append((i, j))
        print(f"[e57] traj: {len(msgs)} prefix readouts", flush=True)
        outs = vx.readout(llm, args.model, msgs, cks)
        tr: dict = {}
        for (i, j), o in zip(keys, outs):
            tr.setdefault(i, []).append({"j": j, "lp_sum": o["lp_sum"], "top20": o["top20"], "pred_exact": o["pred_exact"]})
        recs = []
        for i, it in enumerate(items):
            pg = [p_gold(t["lp_sum"], it["gold"]) for t in tr[i]]
            js = commit_idx(pg)
            recs.append(base(it) | {"sentences": sents_all[i], "traj": tr[i], "p_gold": pg, "commit_idx": js,
                                    "commit_sentence": sents_all[i][js] if js is not None else None})
        vx.write_jsonl(tp, recs)
        T = {r["item_id"]: r for r in recs}
        tm["traj"] = time.time() - t1
        print(f"[e57] traj done ({time.time() - t0:.0f}s)", flush=True)
    elif os.path.exists(tp):
        T = {r["item_id"]: r for r in vx.read_jsonl(tp)}

    def run_edits(jobs: list[tuple], tag: str) -> list[tuple]:
        """jobs: (meta, item, new sentence list up to the edit). Greedy same-turn continuation + same-turn readout."""
        gen_msgs, prefixes = [], []
        for _, it, ss in jobs:
            prefix = (C.COT_PREFIX + " " + " ".join(ss)).strip()
            m = C.cot_messages(it, ref(it["key"]))
            m[-1] = {"role": "assistant", "content": prefix}
            gen_msgs.append(m)
            prefixes.append(prefix)
        print(f"[e57] {tag}: {len(jobs)} continuations", flush=True)
        gens = vx.generate(llm, args.model, gen_msgs, cont=True) if jobs else []
        ro = [C.build_messages(it, "cot", "none", p + " " + g["text"], ref(it["key"])) for (_, it, _), p, g in zip(jobs, prefixes, gens)]
        outs = vx.readout(llm, args.model, ro, [it["choices"] for _, it, _ in jobs]) if jobs else []
        return list(zip(jobs, prefixes, gens, outs))

    # ------------------------------------------------------------------ six-form argument-order edit (e30)
    if "argedit" in parts:
        t1 = time.time()
        traj = sorted(T.values(), key=lambda r: r["item_id"])
        sel = [("jstar", r["item_id"], r["commit_idx"]) for r in traj if r["group"] == "D+C-" and r["commit_idx"] is not None]
        rng = random.Random(292)  # = e29_edit / e30_run dcp selection
        for ph in ("front", "behind"):
            pool = sorted((r for r in traj if r["group"] == "D+C+" and r["phrasing"] == ph and r["sentences"]
                           and first_claim_idx(r["sentences"]) is not None), key=lambda r: r["item_id"])
            sel += [("dcp", r["item_id"], first_claim_idx(r["sentences"])) for r in rng.sample(pool, min(100, len(pool)))]
        jobs = []
        for part, iid, p in sel:
            it, sents = by_key[iid], T[iid]["sentences"]
            head = HEAD.match(sents[p]).group(0)
            F, K = it["meta"]["front_name"], it["meta"]["back_name"]
            for arm in ["orig", "neutral", *ARG]:
                s = sents[p] if arm == "orig" else head + (NEUTRAL if arm == "neutral" else ARG[arm][0].format(F=F, K=K))
                jobs.append(((part, p, arm, s), it, sents[:p] + [s]))
        recs: dict = {}
        for ((part, p, arm, s), it, _), prefix, g, o in run_edits(jobs, "argedit"):
            mt, info = it["meta"], {}
            if arm in ARG:
                _, sp, subj, neg = ARG[arm]
                sname = mt["front_name"] if subj == "F" else mt["back_name"]
                info = {"sent_pred": sp, "sent_subject": sname, "negated": neg, "subject_match": sname == mt["x_name"],
                        "pred_match": sp == it["phrasing"]}
            iid = it["key"]
            recs.setdefault(part, []).append(base(it) | {"part": part, "commit_idx": T[iid]["commit_idx"], "edit_idx": p, "arm": arm,
                                                         "orig_sentence": T[iid]["sentences"][p], "new_sentence": s,
                                                         "continuation": g["text"], "cont_len": g["len"], "trunc": g["trunc"],
                                                         "stated": C.parse_free_answer(g["text"], it["choices"]), **info, **o})
        for part, rs in recs.items():
            vx.write_jsonl(f"{stem}_argedit_{part}{sfx}.jsonl", rs)
        tm["argedit"] = time.time() - t1
        print(f"[e57] argedit done {dict(Counter(p for p, *_ in sel))} ({time.time() - t0:.0f}s)", flush=True)

    # ------------------------------------------------------------------ unconditional first-claim edit (e2f same)
    if "uncond" in parts:
        t1 = time.time()
        jobs = []
        for it in items:
            sents = sents_of(it)
            fc = first_claim(sents)
            if fc is None:
                continue
            j, claimed = fc
            F, K = it["meta"]["front_name"], it["meta"]["back_name"]
            for cond in UNCOND_CONDS:
                jobs.append(((cond, j, len(sents), sents[j], int(claimed != F.lower())), it, edited(sents, j, cond, F, K)))
        recs = []
        for ((cond, j, n, s0, wrong0), it, _), prefix, g, o in run_edits(jobs, "uncond"):
            recs.append(base(it) | {"cond": cond, "j": j, "n_sent": n, "orig_sentence": s0, "orig_claim_wrong": wrong0,
                                    "prefix": prefix, "continuation": g["text"], "cont_len": g["len"], "trunc": g["trunc"],
                                    "stated": C.parse_free_answer(g["text"], it["choices"]), **o})
        vx.write_jsonl(f"{stem}_uncond{sfx}.jsonl", recs)
        tm["uncond"] = time.time() - t1
        print(f"[e57] uncond done: {len(recs)} records ({time.time() - t0:.0f}s)", flush=True)

    # ------------------------------------------------------------------ fixed-rationale image swap (e14)
    if "imgswap" in parts:
        t1 = time.time()
        jobs, msgs, ch = [], [], []
        gray: dict = {}
        for it in items:
            tk = twin_of(it["key"])
            if tk not in by_key:
                continue
            r = src[(it["key"], "cot")]["text"]
            for ctx, txt in (("full", r), ("cut", strip_answer(r))):
                for img in ("own", "twin") + (("gray",) if ctx == "cut" else ()):
                    if img == "gray":
                        if it["uid"] not in gray:
                            gray[it["uid"]] = vx.img_ref(Image.new("RGB", C.get_image(it).size, vx.GRAY))
                        im = gray[it["uid"]]
                    else:
                        im = ref(it["key"] if img == "own" else tk)
                    jobs.append((it, tk, ctx, img, len(txt)))
                    msgs.append(C.build_messages(it, "cot", "none", txt, im))
                    ch.append(it["choices"])
        print(f"[e57] imgswap: {len(msgs)} readouts", flush=True)
        outs = vx.readout(llm, args.model, msgs, ch)
        recs = [base(it) | {"twin_id": tk, "ctx": ctx, "img": img, "ctx_chars": nch, **o} for (it, tk, ctx, img, nch), o in zip(jobs, outs)]
        vx.write_jsonl(f"{stem}_imgswap{sfx}.jsonl", recs)
        tm["imgswap"] = time.time() - t1
        print(f"[e57] imgswap done ({time.time() - t0:.0f}s)", flush=True)

    # ------------------------------------------------------------------ exploratory (prereg addendum 2)
    if "imgswap_trunc" in parts:
        # Exp-45 conclusion-removed body (e35 truncate rule) under the own vs the twin image
        t1 = time.time()
        rt = {}
        for r in vx.read_jsonl(f"{vx.OUT}/e45/items.jsonl"):
            if r["group"] == "open" and r["model"] == args.model and r["ds"] == args.set:
                k = r["item"].replace("|front", "").replace("|behind", "_behind")
                rt[k] = (C.COT_PREFIX + " " + r["body"]).strip()
        jobs, msgs = [], []
        for it in items:
            tk = twin_of(it["key"])
            if tk not in by_key or it["key"] not in rt:
                continue
            for img in ("own", "twin"):
                jobs.append((it, tk, img))
                msgs.append(C.build_messages(it, "cot", "none", rt[it["key"]], ref(it["key"] if img == "own" else tk)))
        print(f"[e57] imgswap_trunc: {len(msgs)} readouts ({len(rt)} truncated bodies)", flush=True)
        outs = vx.readout(llm, args.model, msgs, [it["choices"] for it, _, _ in jobs])
        vx.write_jsonl(f"{stem}_imgswap_trunc{sfx}.jsonl", [base(it) | {"twin_id": tk, "ctx": "trunc", "img": img, **o}
                                                           for (it, tk, img), o in zip(jobs, outs)])
        tm["imgswap_trunc"] = time.time() - t1
    if "trajtwin" in parts:
        # trajectory layout (prefix closed as a turn) under the twin image, every prefix of every query
        t1 = time.time()
        TT = T or {r["item_id"]: r for r in vx.read_jsonl(f"{vx.OUT}/e28/{args.model}_{args.set}_traj.jsonl")}
        msgs, cks, keys = [], [], []
        for it in items:
            tk = twin_of(it["key"])
            if tk not in by_key or it["key"] not in TT:
                continue
            sents = TT[it["key"]]["sentences"]
            for j in range(len(sents) + 1):
                prefix = (C.COT_PREFIX + " " + " ".join(sents[:j])).strip()
                msgs.append(C.build_messages(it, "cot", "turn", prefix, ref(tk)))
                cks.append(it["choices"])
                keys.append((it["key"], j))
        print(f"[e57] trajtwin: {len(msgs)} prefix readouts", flush=True)
        outs = vx.readout(llm, args.model, msgs, cks)
        pg: dict = {}
        for (k, j), o in zip(keys, outs):
            pg.setdefault(k, []).append(p_gold(o["lp_sum"], by_key[k]["gold"]))
        vx.write_jsonl(f"{stem}_trajtwin{sfx}.jsonl", [base(by_key[k]) | {"twin_id": twin_of(k), "p_gold_twin": v,
                                                                          "p_gold_own": TT[k]["p_gold"], "commit_idx": TT[k]["commit_idx"]}
                                                       for k, v in pg.items()])
        tm["trajtwin"] = time.time() - t1

    tm["total"] = time.time() - t0
    json.dump(tm, open(f"{stem}_{'-'.join(parts)}{sfx}.timing.json", "w"), indent=1)
    print(f"[e57] done {json.dumps({k: round(v) for k, v in tm.items()})}", flush=True)


if __name__ == "__main__":
    main()
