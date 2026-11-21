"""E22b: location vs selection controls for the commit-sentence gate (Exp-22c).

Reuses the e22 run of the same cell (outputs/e22/<model>_<set>.jsonl): the model's Direct answer d, the plain greedy
CoT rationale and its closed-turn prefix readouts P_j(d), and the gate's per-item event count. Every new arm therefore
starts from the identical plain rationale as e22's cot / gate / rand arms. Prompts, decoding (greedy, <= 512 tokens,
T = 0.7 samples of <= 160 tokens cut at the first sentence end), sentence split, readouts and the 8-event cap are e22's.

Arms (final answer = same-turn readout after the full rationale, as in the CoT arm):
  locfree  LOCATION-MATCHED, CRITERION-FREE: fires where the gate's criterion fires (sentence moves the closed-turn
           P(d) from >= 0.5 to < 0.5, evaluated on the evolving trajectory), draws ONE T = 0.7 sample with no acceptance
           check, keeps it, continues greedily, and fires again at later flagged sentences. Seeds as e22's gate
           (seed0 = 1), so the first event's sample can coincide with the gate's first candidate.
  randfree RANDOM-LOCATION, COUNT-MATCHED, CRITERION-FREE: per item exactly as many single T = 0.7 samples as
           ``locfree``, at random sentences of the evolving text: with r events left and S sentences in the pending
           greedy text, the next event is the earliest of r positions drawn without replacement from the S (all S if
           r >= S); no cap at the plain rationale's length. If the text ends with events left, the last sentence is
           resampled again (``fallback``). Seed0 = 4.
  randsel  RANDOM-LOCATION, CRITERION-MATCHED: per item as many events as e22's gate (resamples + drops), at random
           positions drawn as in ``randfree``; each event draws up to K = 4 samples and keeps the first whose
           closed-turn P(d) >= 0.5, else drops the sentence and keeps the greedy remainder (the gate's rule). Seed0 = 5.
Diagnostic readouts (not used for decisions, counted separately as n_diag): p_before / p_greedy / p_after for
randfree events, p_before / p_greedy for randsel events.
-> outputs/e22b/<model>_<set>.jsonl

Usage: python -m experiments.interventions.commit_gate_controls --model molmo7b --set cues [--limit 10] [--arms locfree,randfree,randsel]
"""
from __future__ import annotations

import argparse
import json
import os
import random
import time

from srww import vllm_utils as vx
from srww.vllm_utils import C
from experiments.interventions.commit_gate import CAND_TOK, K, MAX_EVENTS, MAX_TOK, T_SAMPLE, Runner, gen_raw, load_set, sentence_ends

RAND_ARMS = ("randfree", "randsel")


def load_e22(model: str, set_name: str, sfx: str) -> dict[str, dict[str, dict]]:
    by: dict[str, dict[str, dict]] = {}
    for line in open(f"{vx.OUT}/e22/{model}_{set_name}{sfx}.jsonl"):
        r = json.loads(line)
        by.setdefault(r["arm"], {})[r["item_id"]] = r
    return by


def run_arm_b(R: Runner, arm: str, d_ans: list[str], plain: list[dict], p0: list[float],
              plain_traj: list[list[float]] | None, n_target: list[int] | None, seed0: int) -> list[dict]:
    """Event-driven sentence-wise decoding for locfree / randfree / randsel (mirrors e22_gate.run_arm)."""
    n = len(R.items)
    rngs = [random.Random(f"e22b-{arm}-{it['key']}") for it in R.items]
    st = [{"acc": "", "p": p0[i], "events": [], "gen_tok": plain[i]["len"], "n_acc": 0, "done": False,
           "n_readout": (1 + len(sentence_ends(plain[i]["text"], True))) if arm == "locfree" else 0, "n_diag": 0,
           "pending": {"text": plain[i]["text"], "eos": plain[i]["eos"],
                       "traj": plain_traj[i] if arm == "locfree" else None},
           "left": n_target[i] if n_target is not None else None, "traj": [p0[i]], "n_fallback": 0} for i in range(n)]
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
        ends = {i: sentence_ends(st[i]["pending"]["text"], True) for i in act}
        # 2) prefix readouts of the pending text (locfree only: the gate's criterion)
        if arm == "locfree":
            pairs = [(i, st[i]["acc"] + st[i]["pending"]["text"][:e]) for i in act if st[i]["pending"]["traj"] is None
                     for e in ends[i]]
            ps = R.p_dir(pairs, d_ans)
            k = 0
            for i in act:
                if st[i]["pending"]["traj"] is None:
                    st[i]["pending"]["traj"] = [None] + ps[k:k + len(ends[i])]
                    st[i]["n_readout"] += len(ends[i])
                    k += len(ends[i])
        # 3) find the triggering sentence
        trig = {}
        for i in act:
            s = st[i]
            txt, e = s["pending"]["text"], ends[i]
            hit = None
            if arm == "locfree":
                prev = s["p"]
                for k in range(len(e)):
                    if len(s["events"]) >= MAX_EVENTS:
                        break
                    pk = s["pending"]["traj"][k + 1]
                    if prev >= 0.5 > pk:
                        hit = k
                        break
                    prev = pk
            elif s["left"] > 0 and e:
                hit = min(rngs[i].sample(range(len(e)), min(s["left"], len(e))))
            if hit is None and arm in RAND_ARMS and s["left"] > 0:
                # text exhausted with events left: resample the last committed sentence again
                full = s["acc"] + txt.rstrip()
                ea = sentence_ends(full, True)
                if ea:
                    start = ea[-2] if len(ea) > 1 else 0
                    s["acc"] = full[:start]
                    s["n_acc"] += len(e) - 1
                    s["n_fallback"] += 1
                    trig[i] = {"pos": s["n_acc"] + 1, "greedy": full[start:ea[-1]], "rest": "", "eos": True,
                               "p_before": None, "p_greedy": None, "n_pending": 0, "fallback": True}
                    continue
            if hit is None:
                s["acc"] += txt.rstrip() if e else ""
                s["n_acc"] += len(e)
                if arm == "locfree":
                    s["traj"] += s["pending"]["traj"][1:]
                    s["p"] = s["pending"]["traj"][-1] if e else s["p"]
                s["done"] = True
                continue
            start = e[hit - 1] if hit else 0
            if arm == "locfree":
                s["traj"] += s["pending"]["traj"][1:hit + 1]
                s["p"] = s["pending"]["traj"][hit] if hit else s["p"]
            s["acc"] += txt[:start]
            s["n_acc"] += hit
            trig[i] = {"pos": s["n_acc"] + 1, "greedy": txt[start:e[hit]], "rest": txt[e[hit]:], "eos": s["pending"]["eos"],
                       "p_before": s["p"] if arm == "locfree" else None,
                       "p_greedy": s["pending"]["traj"][hit + 1] if arm == "locfree" else None,
                       "n_pending": len(e), "fallback": False}
        if not trig:
            continue
        # 4) samples of the triggering sentence
        ti = sorted(trig)
        kk = K if arm == "randsel" else 1
        remain = [MAX_TOK - R.ntok(st[i]["acc"]) for i in ti]
        budget = [min(CAND_TOK, r) for r in remain]
        seeds = [seed0 + 7919 * len(st[i]["events"]) + i for i in ti]
        samp = gen_raw(R.llm, R.model, [R.cont_msgs(i, st[i]["acc"]) for i in ti], budget, T_SAMPLE, kk, seeds)
        cands = {}
        for i, b, rm, outs in zip(ti, budget, remain, samp):
            cl = []
            for o in outs:
                st[i]["gen_tok"] += o["len"]
                fin = o["eos"] or (o["len"] >= b and rm <= CAND_TOK)
                e = sentence_ends(o["text"], fin)
                cut = e[0] if e else len(o["text"].rstrip())
                cl.append({"text": o["text"][:cut], "last": fin and len(e) <= 1})
            cands[i] = cl
        # 5) readouts: decision (locfree: the kept sample; randsel: all K) and diagnostics (random arms)
        pairs = [(i, st[i]["acc"] + c["text"]) for i in ti for c in cands[i]]
        ps = R.p_dir(pairs, d_ans)
        k = 0
        for i in ti:
            for c in cands[i]:
                c["p"] = ps[k]
                k += 1
            if arm == "randfree":
                st[i]["n_diag"] += len(cands[i])
            else:
                st[i]["n_readout"] += len(cands[i])
        if arm in RAND_ARMS:
            pairs = [(i, st[i]["acc"]) for i in ti] + [(i, st[i]["acc"] + trig[i]["greedy"]) for i in ti]
            ps = R.p_dir(pairs, d_ans)
            for j, i in enumerate(ti):
                trig[i]["p_before"], trig[i]["p_greedy"] = ps[j], ps[len(ti) + j]
                st[i]["n_diag"] += 2
        # 6) accept / drop
        for i in ti:
            s, tg = st[i], trig[i]
            if arm == "randsel":
                ok = [j for j, c in enumerate(cands[i]) if c["p"] >= 0.5]
            else:
                ok = [0]
            ev = {"pos": tg["pos"], "greedy": tg["greedy"], "p_before": tg["p_before"], "p_greedy": tg["p_greedy"],
                  "cands": [c["text"] for c in cands[i]], "cand_p": [c["p"] for c in cands[i]],
                  "n_pending": tg["n_pending"], "fallback": tg["fallback"]}
            if s["left"] is not None:
                s["left"] -= 1
            more = s["left"] is not None and s["left"] > 0
            if ok:
                c = cands[i][ok[0]]
                ev |= {"type": "resample", "tries": ok[0] + 1}
                s["acc"] += c["text"]
                s["n_acc"] += 1
                s["p"] = c["p"]
                if arm == "locfree":
                    s["traj"].append(c["p"])
                s["pending"] = None
                if c["last"] or R.ntok(s["acc"]) >= MAX_TOK - 1:
                    if more:
                        s["pending"] = {"text": "", "eos": True, "traj": None}
                    else:
                        s["done"] = True
            else:
                ev |= {"type": "drop", "tries": kk}
                rest = tg["rest"]
                s["pending"] = {"text": rest, "eos": tg["eos"], "traj": None} if rest.strip() else None
                if s["pending"] is None and tg["eos"]:
                    if more:
                        s["pending"] = {"text": "", "eos": True, "traj": None}
                    else:
                        s["done"] = True
            s["events"].append(ev)
        print(f"[e22b] {arm} round {rnd}: {len(act)} active, {len(trig)} triggered", flush=True)
    return st


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--set", required=True, choices=["cues", "twins", "syn"])
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--arms", default="locfree,randfree,randsel")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    sfx = f"_limit{args.limit}" if args.limit else ""
    out = f"{vx.OUT}/e22b/{args.model}_{args.set}{sfx}.jsonl"
    if not args.force and (os.path.exists(out) or os.path.exists(out + ".lock")):
        print(f"[e22b] skip: {out} exists or is locked", flush=True)
        return
    os.makedirs(os.path.dirname(out), exist_ok=True)
    open(out + ".lock", "w").write(str(os.getpid()))
    t0 = time.time()
    tm: dict[str, float] = {}
    E = load_e22(args.model, args.set, "")
    items = load_set(args.set, args.limit)
    items = [it for it in items if it["key"] in E["cot"]]
    R = Runner(args.model, items)
    n = len(items)
    print(f"[e22b] {args.model} {args.set}: {n} prompts", flush=True)
    cot = [E["cot"][it["key"]] for it in items]
    d_ans = [r["direct_ans"] for r in cot]
    p0 = [r["p0"] for r in cot]
    pre = len(C.COT_PREFIX) + 1
    # raw greedy continuation = leading space + stored stripped body (e22 stored COT_PREFIX + " " + body.strip())
    plain = [{"text": " " + r["text"][pre:], "len": r["gen_tok"], "eos": r["gen_tok"] < MAX_TOK} for r in cot]
    ends = [sentence_ends(p["text"], True) for p in plain]
    traj = [[None] + r["traj"][1:] for r in cot]
    bad = [i for i in range(n) if len(traj[i]) != 1 + len(ends[i])]
    if bad:
        ps = R.p_dir([(i, plain[i]["text"][:e]) for i in bad for e in ends[i]], d_ans)
        k = 0
        for i in bad:
            traj[i] = [None] + ps[k:k + len(ends[i])]
            k += len(ends[i])
    # sanity: stored Direct answers and first-sentence readouts reproduce
    chk = list(range(min(64, n)))
    ro = vx.readout(R.llm, args.model, [C.build_messages(items[i], "direct", "none", "", R.refs[i]) for i in chk],
                    [items[i]["choices"] for i in chk])
    p1 = R.p_dir([(i, plain[i]["text"][:ends[i][0]]) for i in chk if ends[i]], d_ans)
    s1 = [traj[i][1] for i in chk if ends[i]]
    sanity = {"n_traj_recomputed": len(bad), "direct_agree": sum(o["pred_exact"] == d_ans[i] for o, i in zip(ro, chk)) / len(chk),
              "p1_max_abs_diff": max(abs(a - b) for a, b in zip(p1, s1)) if p1 else None,
              "p1_mean_abs_diff": sum(abs(a - b) for a, b in zip(p1, s1)) / max(1, len(p1))}
    print(f"[e22b] sanity {sanity}", flush=True)
    tm["setup"] = time.time() - t0
    arms = args.arms.split(",")
    states, bodies = {}, {}
    if "locfree" in arms:
        t1 = time.time()
        states["locfree"] = run_arm_b(R, "locfree", d_ans, plain, p0, traj, None, seed0=1)
        tm["locfree"] = time.time() - t1
    if "randfree" in arms:
        t1 = time.time()
        tgt = [len(s["events"]) for s in states["locfree"]]
        states["randfree"] = run_arm_b(R, "randfree", d_ans, plain, p0, None, tgt, seed0=4)
        tm["randfree"] = time.time() - t1
    if "randsel" in arms:
        t1 = time.time()
        tgt = [len(E["gate"][it["key"]]["events"]) for it in items]
        states["randsel"] = run_arm_b(R, "randsel", d_ans, plain, p0, None, tgt, seed0=5)
        tm["randsel"] = time.time() - t1
    recs = []
    for arm, sts in states.items():
        bs = [s["acc"] for s in sts]
        t1 = time.time()
        ro = R.final(bs)
        tm[f"final_{arm}"] = time.time() - t1
        for i, it in enumerate(items):
            s = sts[i]
            base = {k: cot[i][k] for k in ("model", "set", "task", "item_id", "base_id", "phrasing", "order", "gold", "letters",
                                           "image_id", "question", "meta", "x", "y")}
            recs.append(base | {"arm": arm, **ro[i], "text": C.COT_PREFIX + " " + bs[i].strip(), "direct_ans": d_ans[i],
                                "p0": p0[i], "traj": s["traj"] if arm == "locfree" else None, "events": s["events"],
                                "gen_tok": s["gen_tok"], "body_tok": R.ntok(bs[i]), "n_readout": s["n_readout"],
                                "n_diag": s["n_diag"], "n_sent": s["n_acc"], "n_fallback": s["n_fallback"],
                                "n_target": None if s["left"] is None else len(s["events"]) + s["left"]})
    vx.write_jsonl(out, recs)
    tm["total"] = time.time() - t0
    json.dump(tm | {"sanity": sanity}, open(out.replace(".jsonl", ".timing.json"), "w"), indent=1)
    os.remove(out + ".lock")
    print(f"[e22b] wrote {len(recs)} records to {out}; timing {json.dumps({k: round(v) for k, v in tm.items()})}", flush=True)


if __name__ == "__main__":
    main()
