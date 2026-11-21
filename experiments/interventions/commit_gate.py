"""E22: label-free, training-free commit-sentence gate for spatial CoT.

The rationale is generated with the paper's CoT arm (user turn = image + question, assistant prefix
"Let's think step by step.", greedy, <= 512 tokens). It is processed sentence by sentence (sentence split of e2b).
After every new sentence s_j the forced readout P_j at "Answer: (" is taken on the prefix with the turn closed
("Now give your final answer.", as in the e2b trajectories). The model's own Direct answer d is read first.

Arms (one record per item x arm; final answer = same-turn readout after the full rationale, as in the CoT arm):
  cot     plain greedy rationale (one-shot greedy; equals sentence-wise greedy decoding up to re-tokenization).
  gate    readout gate: if s_j moves P(d) from >= 0.5 to < 0.5, draw K = 4 samples of s_j (T = 0.7) and keep the first
          whose prefix keeps P(d) >= 0.5; if all K flip, drop s_j and keep the greedy text after it (the later sentences
          are re-gated on the new prefix). After an accepted resample, decoding continues greedily from the new prefix.
  rand    random-resample control: same number and kind of events per item as ``gate`` (resample / drop), at random
          sentence positions (uniform over the plain rationale's sentences), first T = 0.7 sample kept, no criterion.
  tgate   text gate: s_j is gated when its parsed depth-order claim (claim_parse v3 on the sentence) implies an answer
          different from d; same resample / drop rule, criterion = parsed claim of the candidate.
Sets: cues (cue-factorial, 1,600 prompts), twins (original twins, 800), syn (Synthetic Depth 500 + swapped-object
question 'cfq' 500).
-> outputs/e22/<model>_<set>.jsonl

Usage: python -m experiments.interventions.commit_gate --model qwen3vl_8b --set cues [--limit 10] [--arms cot,gate,rand,tgate]
"""
from __future__ import annotations
from srww import paths as _vtp  # noqa: E402

import argparse
import json
import os
import random
import re
import time

from srww import vllm_utils as vx
from srww.vllm_utils import C
from experiments.commit_onset.sentence_trajectory import _ENUM, _SPLIT
from srww.claim_parse import first_claim_v3

K = 4
T_SAMPLE = 0.7
MAX_TOK = 512
MAX_EVENTS = 8
CAND_TOK = 160
QRX = re.compile(r"Is the (\w+ \w+) (in front of|behind) the (\w+ \w+)\?")


# ----------------------------------------------------------------------------- items

def load_set(name: str, limit: int | None) -> list[dict]:
    """Items with key / phrasing; subject x and other y parsed from the question."""
    from experiments.occlusion.twins_readout import load_occ
    from experiments.behavior.rationale_ablation import cf_question
    if name in ("cues", "twins"):
        d = os.environ.get("CUES_DIR", f"{_vtp.ITEMS}/occ_cues") if name == "cues" else None
        items = [dict(it, phrasing="front", key=it["uid"]) for it in load_occ(limit, items_dir=d)]
        items += [dict(it, phrasing="behind", key=it["uid"] + "_behind") for it in load_occ(limit, polar=True, items_dir=d)]
    elif name == "syn":
        base = vx.load_items(["syn_depth"], limit=limit)
        items = [dict(it, phrasing="front", key=it["uid"], order="orig") for it in base]
        items += [dict(cf_question(it), phrasing="front", key=it["uid"] + "_cfq", order="cfq") for it in base]
    else:
        raise ValueError(name)
    for it in items:
        m = QRX.search(it["question"])
        it["x"], it["y"] = m.group(1).lower(), m.group(3).lower()
    return items


# ----------------------------------------------------------------------------- text helpers

def sentence_ends(text: str, final: bool) -> list[int]:
    """End offsets (just after the sentence text) of complete sentences in ``text`` (e2b split; bare list markers merge
    into the next sentence). If ``final``, the trailing piece counts as a sentence."""
    ends, seg = [], 0
    for m in _SPLIT.finditer(text):
        piece = text[seg:m.start()].strip()
        if piece and not _ENUM.match(piece):
            ends.append(m.start())
            seg = m.end()
        elif not piece:
            seg = m.end()
    if final and text[seg:].strip():
        ends.append(len(text.rstrip()))
    return ends


def claim_answer(sent: str, it: dict) -> str | None:
    """Answer letter implied by the sentence's depth-order claim (None if no parsable two-object claim)."""
    c = first_claim_v3([sent])
    if c is None or c[1] not in (it["x"], it["y"]):
        return None
    x_front = c[1] == it["x"]
    yes = x_front if it["phrasing"] == "front" else not x_front
    return "A" if yes else "B"


# ----------------------------------------------------------------------------- engine wrappers

def gen_raw(llm, model: str, msgs: list, max_tokens: list[int], temperature: float = 0.0, n: int = 1,
            seeds: list[int] | None = None) -> list[list[dict]]:
    """Continue the final assistant message; raw (unstripped) text per sample."""
    from vllm import SamplingParams
    sps = [SamplingParams(temperature=temperature, max_tokens=max(1, mt), n=n, seed=(seeds[i] if seeds else 0))
           for i, mt in enumerate(max_tokens)]
    res = llm.chat(msgs, sps, **vx.chat_kw(model, True))
    return [[{"text": o.text, "len": len(o.token_ids), "eos": o.finish_reason != "length"} for o in r.outputs] for r in res]


class Runner:
    """Holds the engine, per-item image refs and the Direct answers; counts compute."""

    def __init__(self, model: str, items: list[dict]):
        self.model, self.items = model, items
        if os.environ.get("E22_NO_MM_CACHE"):
            # vLLM 0.8.4 V1 mm_input_cache P0/P1 desync (KeyError on an image hash) seen with Molmo on the cue set
            import functools
            import vllm
            vllm.LLM = functools.partial(vllm.LLM, disable_mm_preprocessor_cache=True)
        self.llm = vx.make_llm(model)
        self.tok = self.llm.get_tokenizer()
        self.refs = [vx.img_ref(C.get_image(it)) for it in items]

    def ntok(self, s: str) -> int:
        return len(self.tok.encode(s, add_special_tokens=False))

    def cont_msgs(self, i: int, body: str) -> list[dict]:
        return [{"role": "user", "content": [self.refs[i], {"type": "text", "text": self.items[i]["question"]}]},
                {"role": "assistant", "content": C.COT_PREFIX + body}]

    def p_dir(self, pairs: list[tuple[int, str]], d_ans: list[str]) -> list[float]:
        """Closed-turn forced readout P(Direct answer) after prefix body (e2b layout)."""
        if not pairs:
            return []
        msgs = [C.build_messages(self.items[i], "cot", "turn", (C.COT_PREFIX + b).strip(), self.refs[i]) for i, b in pairs]
        outs = vx.readout(self.llm, self.model, msgs, [self.items[i]["choices"] for i, _ in pairs])
        ps = []
        for (i, _), o in zip(pairs, outs):
            pr = vx.probs_from_lp(o["lp_sum"]) or {}
            ps.append(float(pr.get(d_ans[i], 0.0)))
        return ps

    def final(self, bodies: list[str]) -> list[dict]:
        """Same-turn readout after the full rationale (CoT arm)."""
        msgs = [C.build_messages(it, "cot", "none", C.COT_PREFIX + " " + b.strip(), r)
                for it, b, r in zip(self.items, bodies, self.refs)]
        return vx.readout(self.llm, self.model, msgs, [it["choices"] for it in self.items])


# ----------------------------------------------------------------------------- gated decoding

def run_arm(R: Runner, arm: str, d_ans: list[str], plain: list[dict], p0: list[float], plain_traj: list[list[float]],
            sched: dict[int, list[tuple[int, str]]] | None = None, seed0: int = 0) -> list[dict]:
    """Event-driven sentence-wise decoding for gate / rand / tgate. ``plain`` = greedy continuation from the trigger
    (round 1 is shared with the cot arm); ``plain_traj`` = its prefix readouts (gate only)."""
    n = len(R.items)
    st = [{"acc": "", "p": p0[i], "events": [], "gen_tok": 0, "n_acc": 0, "done": False,
           "n_readout": 1 + len(sentence_ends(plain[i]["text"], True)) if plain_traj else 0,
           "pending": {"text": plain[i]["text"], "eos": plain[i]["eos"], "traj": plain_traj[i] if plain_traj else None},
           "sched": list(sched.get(i, [])) if sched else [], "traj": [p0[i]]} for i in range(n)]
    for i in range(n):
        st[i]["gen_tok"] = plain[i]["len"]
    rnd = 0
    while True:
        act = [i for i in range(n) if not st[i]["done"]]
        if not act:
            break
        rnd += 1
        # 1) greedy continuation for tracks without pending text
        need = [i for i in act if st[i]["pending"] is None]
        if need:
            budget = [MAX_TOK - R.ntok(st[i]["acc"]) for i in need]
            outs = gen_raw(R.llm, R.model, [R.cont_msgs(i, st[i]["acc"]) for i in need], budget)
            for i, b, o in zip(need, budget, outs):
                o = o[0]
                st[i]["gen_tok"] += o["len"]
                st[i]["pending"] = {"text": o["text"], "eos": o["eos"] or b <= 1, "traj": None}
        # 2) readouts of every sentence prefix of the pending text (gate only, if not cached)
        ends = {i: sentence_ends(st[i]["pending"]["text"], True) for i in act}
        if arm == "gate":
            pairs = [(i, st[i]["acc"] + st[i]["pending"]["text"][:e]) for i in act if st[i]["pending"]["traj"] is None
                     for e in ends[i]]
            ps = R.p_dir(pairs, d_ans)
            k = 0
            for i in act:
                if st[i]["pending"]["traj"] is None:
                    st[i]["pending"]["traj"] = [None] + ps[k:k + len(ends[i])]
                    st[i]["n_readout"] += len(ends[i])
                    k += len(ends[i])
        # 3) scan for the first triggering sentence
        trig = {}
        for i in act:
            s = st[i]
            txt, e = s["pending"]["text"], ends[i]
            prev, hit = s["p"], None
            for k, end in enumerate(e):
                start = e[k - 1] if k else 0
                sent = txt[start:end]
                if len(s["events"]) >= MAX_EVENTS:
                    break
                if arm == "gate":
                    pk = s["pending"]["traj"][k + 1]
                    fire = prev >= 0.5 > pk
                elif arm == "tgate":
                    ca = claim_answer(sent, R.items[i])
                    fire = ca is not None and ca != d_ans[i]
                else:
                    fire = bool(s["sched"]) and s["sched"][0][0] == s["n_acc"] + k + 1
                if fire:
                    hit = k
                    break
                if arm == "gate":
                    prev = pk
            if hit is None:
                s["acc"] += txt.rstrip() if e else ""
                s["n_acc"] += len(e)
                if arm == "gate":
                    s["traj"] += s["pending"]["traj"][1:]
                    s["p"] = s["pending"]["traj"][-1] if e else s["p"]
                s["done"] = True
                continue
            start = e[hit - 1] if hit else 0
            if arm == "gate":
                s["traj"] += s["pending"]["traj"][1:hit + 1]
                s["p"] = s["pending"]["traj"][hit] if hit else s["p"]
            s["acc"] += txt[:start]
            s["n_acc"] += hit
            trig[i] = {"pos": s["n_acc"] + 1, "greedy": txt[start:e[hit]], "rest": txt[e[hit]:],
                       "eos": s["pending"]["eos"], "p_before": s["p"],
                       "p_greedy": s["pending"]["traj"][hit + 1] if arm == "gate" else None,
                       "kind": s["sched"].pop(0)[1] if arm == "rand" else None}
        if not trig:
            continue
        # 4) K samples of the triggering sentence (rand: one sample, none for drop events)
        ti = sorted(trig)
        kk = 1 if arm == "rand" else K
        tis = [i for i in ti if not (arm == "rand" and trig[i]["kind"] == "drop")]
        remain = [MAX_TOK - R.ntok(st[i]["acc"]) for i in tis]
        budget = [min(CAND_TOK, r) for r in remain]
        seeds = [seed0 + 7919 * len(st[i]["events"]) + i for i in tis]
        samp = gen_raw(R.llm, R.model, [R.cont_msgs(i, st[i]["acc"]) for i in tis], budget, T_SAMPLE, kk, seeds) if tis else []
        cands = {i: [] for i in ti}
        for i, b, rm, outs in zip(tis, budget, remain, samp):
            cl = []
            for o in outs:
                st[i]["gen_tok"] += o["len"]
                fin = o["eos"] or (o["len"] >= b and rm <= CAND_TOK)
                e = sentence_ends(o["text"], fin)
                cut = e[0] if e else len(o["text"].rstrip())
                cl.append({"text": o["text"][:cut], "last": fin and len(e) <= 1})
            cands[i] = cl
        if arm == "gate":
            pairs = [(i, st[i]["acc"] + c["text"]) for i in ti for c in cands[i]]
            ps = R.p_dir(pairs, d_ans)
            k = 0
            for i in ti:
                for c in cands[i]:
                    c["p"] = ps[k]
                    k += 1
                st[i]["n_readout"] += len(cands[i])
        for i in ti:
            s, tg = st[i], trig[i]
            if arm == "gate":
                ok = [j for j, c in enumerate(cands[i]) if c["p"] >= 0.5]
            elif arm == "tgate":
                ok = [j for j, c in enumerate(cands[i]) if claim_answer(c["text"], R.items[i]) in (None, d_ans[i])]
            else:
                ok = [0] if tg["kind"] == "resample" else []
            ev = {"pos": tg["pos"], "greedy": tg["greedy"], "p_before": tg["p_before"], "p_greedy": tg["p_greedy"],
                  "cands": [c["text"] for c in cands[i]], "cand_p": [c.get("p") for c in cands[i]]}
            if ok:
                c = cands[i][ok[0]]
                ev |= {"type": "resample", "tries": ok[0] + 1}
                s["acc"] += c["text"]
                s["n_acc"] += 1
                if arm == "gate":
                    s["p"] = c["p"]
                    s["traj"].append(c["p"])
                s["pending"] = None
                if c["last"] or R.ntok(s["acc"]) >= MAX_TOK - 1:
                    s["done"] = True
            else:
                ev |= {"type": "drop", "tries": kk}
                rest = tg["rest"]
                s["pending"] = {"text": rest, "eos": tg["eos"], "traj": None} if rest.strip() else None
                if s["pending"] is None and tg["eos"]:
                    s["done"] = True
            s["events"].append(ev)
        print(f"[e22] {arm} round {rnd}: {len(act)} active, {len(trig)} triggered", flush=True)
    return st


# ----------------------------------------------------------------------------- main

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--set", required=True, choices=["cues", "twins", "syn"])
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--arms", default="cot,gate,rand,tgate")
    ap.add_argument("--force", action="store_true", help="rerun even if the output exists or another run holds the lock")
    args = ap.parse_args()
    sfx = f"_limit{args.limit}" if args.limit else ""
    out = f"{vx.OUT}/e22/{args.model}_{args.set}{sfx}.jsonl"
    if not args.force and (os.path.exists(out) or os.path.exists(out + ".lock")):
        print(f"[e22] skip: {out} exists or is locked", flush=True)
        return
    os.makedirs(os.path.dirname(out), exist_ok=True)
    open(out + ".lock", "w").write(str(os.getpid()))
    t0 = time.time()
    tm = {}
    items = load_set(args.set, args.limit)
    R = Runner(args.model, items)
    n = len(items)
    print(f"[e22] {args.model} {args.set}: {n} prompts", flush=True)
    # Direct
    ro_d = vx.readout(R.llm, args.model, [C.build_messages(it, "direct", "none", "", r) for it, r in zip(items, R.refs)],
                      [it["choices"] for it in items])
    d_ans = [o["pred_exact"] for o in ro_d]
    tm["direct"] = time.time() - t0
    # plain greedy CoT (round 1 for every arm) + its prefix trajectory
    t1 = time.time()
    plain = [o[0] for o in gen_raw(R.llm, args.model, [R.cont_msgs(i, "") for i in range(n)], [MAX_TOK] * n)]
    tm["cot_gen"] = time.time() - t1
    t1 = time.time()
    ends = [sentence_ends(p["text"], True) for p in plain]
    pairs = [(i, "") for i in range(n)] + [(i, plain[i]["text"][:e]) for i in range(n) for e in ends[i]]
    ps = R.p_dir(pairs, d_ans)
    p0, rest = ps[:n], ps[n:]
    traj, k = [], 0
    for i in range(n):
        traj.append([None] + rest[k:k + len(ends[i])])
        k += len(ends[i])
    tm["cot_traj"] = time.time() - t1
    arms = args.arms.split(",")
    bodies = {"cot": [p["text"] for p in plain]}
    states = {"cot": [{"events": [], "gen_tok": plain[i]["len"], "n_readout": 1 + len(ends[i]), "traj": [p0[i]] + traj[i][1:],
                       "n_acc": len(ends[i])} for i in range(n)]}
    for arm in ("gate", "tgate"):
        if arm in arms:
            t1 = time.time()
            states[arm] = run_arm(R, arm, d_ans, plain, p0, traj if arm == "gate" else None, seed0=1 if arm == "gate" else 2)
            bodies[arm] = [s["acc"] for s in states[arm]]
            tm[arm] = time.time() - t1
    if "rand" in arms and "gate" in arms:
        t1 = time.time()
        rng = random.Random(22)
        sched = {}
        for i in range(n):
            kinds = [e["type"] for e in states["gate"][i]["events"]]
            J = max(1, len(ends[i]))
            pos = sorted(rng.sample(range(1, J + 1), min(len(kinds), J)))
            sched[i] = list(zip(pos, kinds[:len(pos)]))
        states["rand"] = run_arm(R, "rand", d_ans, plain, p0, None, sched=sched, seed0=3)
        bodies["rand"] = [s["acc"] for s in states["rand"]]
        tm["rand"] = time.time() - t1
    recs = []
    base = lambda it: {"model": args.model, "set": args.set, "task": it["task"], "item_id": it["key"], "base_id": it["uid"],
                       "phrasing": it["phrasing"], "order": it.get("order"), "gold": it["gold"], "letters": it["choices"],
                       "image_id": it["group"], "question": it["question"], "meta": it.get("meta"), "x": it["x"], "y": it["y"]}
    for i, it in enumerate(items):
        recs.append(base(it) | {"arm": "direct", **ro_d[i]})
    for arm, bs in bodies.items():
        t1 = time.time()
        ro = R.final(bs)
        tm[f"final_{arm}"] = time.time() - t1
        for i, it in enumerate(items):
            s = states[arm][i]
            ev = [{k: v for k, v in e.items()} for e in s["events"]]
            recs.append(base(it) | {"arm": arm, **ro[i], "text": C.COT_PREFIX + " " + bs[i].strip(), "direct_ans": d_ans[i],
                                    "p0": p0[i], "traj": s.get("traj"), "events": ev, "gen_tok": s["gen_tok"],
                                    "body_tok": R.ntok(bs[i]), "n_readout": s["n_readout"], "n_sent": s.get("n_acc")})
    vx.write_jsonl(out, recs)
    tm["total"] = time.time() - t0
    json.dump(tm, open(out.replace(".jsonl", ".timing.json"), "w"), indent=1)
    os.remove(out + ".lock")
    print(f"[e22] wrote {len(recs)} records to {out}; timing {json.dumps({k: round(v) for k, v in tm.items()})}", flush=True)


if __name__ == "__main__":
    main()
