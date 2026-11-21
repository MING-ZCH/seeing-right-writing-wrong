"""E2f: causal claim edit. Replace the rationale's first two-object depth claim and let the model continue.

Items: Direct-right items on Synthetic Depth (groups from the legacy top-20 readouts, as in E2c / the judge) and on
the cue-decoupled occlusion twins ("in front" phrasing; groups from the E2d Direct / CoT readouts).
Edit point j: the first sentence of the stored R that names both objects with an occlusion/front verb (same parse as
srww.claim_parse.first_claim and judges/occlusion_claim_judge.py). The sentence's list marker / bold heading is kept.
Conditions at j:
  orig         keep s_j (regeneration control)
  correct      s_j -> "The <front> is in front of the <back>, covering part of it."   (ground truth)
  wrong        s_j -> "The <back> is in front of the <front>, covering part of it."
  neutral      s_j -> "Let me look at the two shapes in the image carefully."         (deletes the claim)
  neutral_ins  neutral sentence inserted before s_j (s_j kept)                         (edit-perturbation control)
Continuation modes:
  same   R_{<j} + edit stays in the assistant turn; greedy continuation (<=512 tokens); same-turn readout.
  closed R_{<j} + edit closed as an assistant turn + user "Continue your reasoning." (as in E2c); readout after a
         closing "Now give your final answer." turn.

  inplace s_p edited inside the full stored R (rest of R fixed, no regeneration); same-turn forced readout.
         Positions p: the claim sentence j, and the E2b commit sentence j* (Synthetic Depth items with a commit).
         Conditions orig / correct / wrong / neutral_len (14-word neutral sentence, same word count as the claims).

Usage: python -m experiments.claim_edits.first_claim_edit --model qwen3b [--limit 20] [--sets syn,occ] [--modes same,closed,inplace]
       python -m experiments.claim_edits.first_claim_edit --analyze   (CPU; writes outputs/bundle/claim_edit.json)
"""
from __future__ import annotations
from srww import paths as _vtp  # noqa: E402

import argparse
import json
import math
import re
import time
from pathlib import Path
from srww.claim_parse import QRX, first_claim, split_sents  # noqa: F401

HEAD = re.compile(r"^\s*((?:\d+[.)]|[-*])\s*)?(\*\*[^*]{1,60}\*\*:?\s*)?")
NEUTRAL = "Let me look at the two shapes in the image carefully."
NEUTRAL_LEN = "Let me look carefully at the two shapes and their colors in this image."
CONT = "Continue your reasoning."
CONDS = ("orig", "correct", "wrong", "neutral", "neutral_ins")
INPLACE_CONDS = ("orig", "correct", "wrong", "neutral_len")


def commit_sentence(traj: list[dict], gold: str) -> int | None:
    """Sentence index of the commit j* (first prefix after which P(gold) stays < 0.5), as in error_onset."""
    pg = []
    for t in sorted(traj, key=lambda x: x["j"]):
        lp = t.get("lp_sum") or t["lp_bare"]
        fin = {k: v for k, v in lp.items() if v is not None and math.isfinite(v)}
        m = max(fin.values())
        z = sum(math.exp(v - m) for v in fin.values())
        pg.append(math.exp(lp[gold] - m) / z if gold in fin else 0.0)
    js = next((j for j in range(1, len(pg)) if pg[j - 1] >= 0.5 and all(v < 0.5 for v in pg[j:])), None)
    return None if js is None else js - 1


ANS_RX = re.compile(r"answer|\([A-D]\)|\b[A-D]\)|option [A-D]\b", re.I)


def n_keep_without_answer(sents: list[str]) -> int:
    """Number of leading sentences kept after dropping trailing answer-mentioning sentences (<= 4, as strip_answer)."""
    n = len(sents)
    for _ in range(4):
        if n > 0 and ANS_RX.search(sents[n - 1]):
            n -= 1
        else:
            break
    return n


def replaced(sents: list[str], p: int, cond: str, front: str, back: str) -> list[str]:
    """Full sentence list with sentence p replaced (list marker / heading kept)."""
    if cond == "orig":
        return list(sents)
    head = HEAD.match(sents[p]).group(0)
    new = {"correct": f"The {front} is in front of the {back}, covering part of it.",
           "wrong": f"The {back} is in front of the {front}, covering part of it.", "neutral_len": NEUTRAL_LEN}[cond]
    return sents[:p] + [head + new] + sents[p + 1:]


CLAIM_RX = re.compile(r"The (\w+ \w+) is in front of the (\w+ \w+), covering part of it\.$")


def edited(sents: list[str], j: int, cond: str, front: str, back: str) -> list[str]:
    """Rationale sentences up to and including the edited claim position."""
    head = HEAD.match(sents[j]).group(0)
    claim = {"correct": f"The {front} is in front of the {back}, covering part of it.",
             "wrong": f"The {back} is in front of the {front}, covering part of it.",
             "neutral": NEUTRAL}
    if cond == "orig":
        return sents[: j + 1]
    if cond == "neutral_ins":
        return sents[:j] + [head + NEUTRAL, sents[j][len(head):]]
    return sents[:j] + [head + claim[cond]]


# ----------------------------------------------------------------------------- GPU run

def build_items(model: str, sets: list[str], limit: int | None, groups: tuple = ("D+C-", "D+C+")) -> list[dict]:
    from srww import vllm_utils as vx
    from experiments.commit_onset.sentence_trajectory import body, split_sentences
    out = []
    if "syn" in sets:
        leg = vx.legacy_e1(model)
        for it in vx.load_items(["syn_depth"], limit=limit):
            r = leg.get(it["uid"])
            q = QRX.search(it["question"])
            if not r or not q:
                continue
            am = lambda sc: max({k: v for k, v in sc.items() if v is not None and math.isfinite(v)}.items(),
                                key=lambda kv: kv[1])[0]
            d_ok, c_ok = am(r["scores"]["direct"]) == it["gold"], am(r["scores"]["cot"]) == it["gold"]
            x, y = q.group(1).lower(), q.group(2).lower()
            front, back = (x, y) if it["gold"] == "A" else (y, x)
            sents = split_sentences(body(r["cot"]))[:30]
            out.append({"set": "syn", "item": it, "sents": sents, "front": front, "back": back,
                        "group": ("D+" if d_ok else "D-") + ("C+" if c_ok else "C-")})
    if "occ" in sets:
        from experiments.occlusion.twins_readout import load_occ
        recs = vx.read_jsonl(f"{vx.OUT}/e2d/{model}.jsonl")
        by = {(r["item_id"], r["condition"]): r for r in recs}
        for it in load_occ(limit):
            d, c = by.get((it["uid"], "direct")), by.get((it["uid"], "cot"))
            if not d or not c:
                continue
            pe = lambda r: max(r["lp_sum"], key=r["lp_sum"].get)
            front, back = it["meta"]["front_name"], it["meta"]["back_name"]
            sents = split_sentences(body(c["text"]))[:30]
            out.append({"set": "occ", "item": it, "sents": sents, "front": front, "back": back,
                        "group": ("D+" if pe(d) == it["gold"] else "D-") + ("C+" if pe(c) == it["gold"] else "C-")})
    keep = []
    for x in out:
        fc = first_claim(x["sents"])
        if fc is None or x["group"] not in groups:
            continue
        x["j"], x["claimed"] = fc
        keep.append(x)
    return keep


def run(args) -> None:
    from srww import vllm_utils as vx
    from srww.vllm_utils import C
    t0 = time.time()
    items = build_items(args.model, args.sets.split(","), args.limit, tuple(args.groups.split(",")))
    print(f"[e2f] {args.model}: {len(items)} items with a parsed claim "
          f"({sum(x['group'] == 'D+C-' for x in items)} D+C-)", flush=True)
    llm = vx.make_llm(args.model)
    is_molmo = args.model.startswith("molmo")
    sfx = args.tag + (f"_limit{args.limit}" if args.limit else "")
    for mode in args.modes.split(","):
        if mode == "inplace":
            inplace(llm, args.model, items, sfx)
            continue
        jobs, gen_msgs = [], []
        for x in items:
            it = x["item"]
            ref = vx.img_ref(C.get_image(it))
            for cond in CONDS:
                prefix = (C.COT_PREFIX + " " + " ".join(edited(x["sents"], x["j"], cond, x["front"], x["back"]))).strip()
                if mode == "same":
                    m = C.cot_messages(it, ref)
                    m[-1] = {"role": "assistant", "content": prefix}
                else:
                    m = [{"role": "user", "content": [ref, {"type": "text", "text": it["question"]}]},
                         {"role": "assistant", "content": prefix},
                         {"role": "user", "content": [{"type": "text", "text": CONT}]}]
                    if is_molmo:
                        m = m + [{"role": "assistant", "content": ""}]
                jobs.append((x, cond, prefix, ref))
                gen_msgs.append(m)
        print(f"[e2f] mode={mode}: {len(jobs)} continuations", flush=True)
        try:
            gens = vx.generate(llm, args.model, gen_msgs, cont=(mode == "same" or is_molmo))
            if mode == "same":
                ro = [C.build_messages(x["item"], "cot", "none", prefix + " " + g["text"], ref)
                      for (x, _, prefix, ref), g in zip(jobs, gens)]
            else:
                ro = [(m[:-1] if is_molmo else m) + [{"role": "assistant", "content": g["text"]},
                                                     {"role": "user", "content": [{"type": "text", "text": C.FINAL_INSTR}]},
                                                     {"role": "assistant", "content": C.FORCE_PREFIX}]
                      for m, g in zip(gen_msgs, gens)]
            outs = vx.readout(llm, args.model, ro, [x["item"]["choices"] for x, *_ in jobs])
        except Exception as e:  # keep the other mode's results (Phi multi-turn chat crashed in E2c)
            print(f"[e2f] mode={mode} failed: {e!r}", flush=True)
            continue
        recs = []
        for (x, cond, prefix, _), g, o in zip(jobs, gens, outs):
            it = x["item"]
            recs.append({"model": args.model, "set": x["set"], "task": it["task"], "item_id": it["uid"],
                         "image_id": it["group"], "group": x["group"], "mode": mode, "cond": cond, "j": x["j"],
                         "n_sent": len(x["sents"]), "orig_sentence": x["sents"][x["j"]],
                         "orig_claim_wrong": int(x["claimed"] != x["front"]), "gold": it["gold"],
                         "letters": it["choices"], "prefix": prefix, "continuation": g["text"], "cont_len": g["len"],
                         "trunc": g["trunc"], "stated": C.parse_free_answer(g["text"], it["choices"]), **o})
        vx.write_jsonl(f"{vx.OUT}/e2f/{args.model}_{mode}{sfx}.jsonl", recs)
        print(f"[e2f] wrote {len(recs)} records mode={mode} ({time.time() - t0:.0f}s)", flush=True)


def inplace(llm, model: str, items: list[dict], sfx: str) -> None:
    """In-place sentence swap inside the full stored R; one-token readout only."""
    from srww import vllm_utils as vx
    from srww.vllm_utils import C
    tok = llm.get_tokenizer()
    e2b = {r["item_id"]: r for r in vx.read_jsonl(f"{vx.OUT}/e2b/{model}.jsonl")}
    jobs, msgs = [], []
    for x in items:
        it = x["item"]
        ref = vx.img_ref(C.get_image(it))
        pos = {"claim": x["j"]}
        if x["set"] == "syn" and it["uid"] in e2b:
            js = commit_sentence(e2b[it["uid"]]["traj"], it["gold"])
            if js is not None and js < len(x["sents"]):
                pos["commit"] = js
        n_cut = n_keep_without_answer(x["sents"])
        for pname, p in pos.items():
            for rvar, n in (("full", len(x["sents"])), ("cut", n_cut)):
                if p >= n:
                    continue
                for cond in INPLACE_CONDS:
                    full = C.COT_PREFIX + " " + " ".join(replaced(x["sents"], p, cond, x["front"], x["back"])[:n])
                    jobs.append((x, pname, p, cond, full, rvar))
                    msgs.append(C.build_messages(it, "cot", "none", full, ref))
    print(f"[e2f] inplace: {len(jobs)} readouts", flush=True)
    outs = vx.readout(llm, model, msgs, [x["item"]["choices"] for x, *_ in jobs])
    recs = []
    for (x, pname, p, cond, full, rvar), o in zip(jobs, outs):
        it = x["item"]
        recs.append({"model": model, "set": x["set"], "task": it["task"], "item_id": it["uid"], "image_id": it["group"],
                     "group": x["group"], "mode": "inplace", "rvar": rvar, "pos": pname, "p": p, "j_claim": x["j"], "cond": cond,
                     "n_sent": len(x["sents"]), "orig_sentence": x["sents"][p],
                     "orig_claim_wrong": int(x["claimed"] != x["front"]), "gold": it["gold"], "letters": it["choices"],
                     "n_tok_orig_sentence": len(tok.encode(x["sents"][p], add_special_tokens=False)),
                     "n_tok_new_sentence": len(tok.encode(replaced(x["sents"], p, cond, x["front"], x["back"])[p],
                                                          add_special_tokens=False)), **o})
    vx.write_jsonl(f"{vx.OUT}/e2f/{model}_inplace{sfx}.jsonl", recs)
    print(f"[e2f] wrote {len(recs)} records mode=inplace", flush=True)


# ----------------------------------------------------------------------------- analysis (CPU)

def analyze(out_dir: Path, O: Path) -> None:
    import numpy as np
    from srww.common import cluster_bootstrap_ci
    MODELS = ["qwen3b", "qwen7b", "internvl3_8b", "phi35v", "molmo7b", "qwen3vl_8b"]

    def pe(r):
        lp = r.get("lp_sum") or r["top20"]
        fin = {k: v for k, v in lp.items() if v is not None and math.isfinite(v)}
        return max(fin, key=fin.get) if fin else None

    def ci(v, g):
        v = np.asarray(v, float)
        if len(v) == 0:
            return {"n": 0}
        m, lo, hi = cluster_bootstrap_ci(v, g, 2000)
        return {"mean": round(100 * m, 1), "lo": round(100 * lo, 1), "hi": round(100 * hi, 1), "n": int(len(v)),
                "n_clusters": int(len(set(g)))}

    res: dict = {}
    lines = []
    for m in MODELS:
        for mode in ("same", "closed"):
            recs = [json.loads(l) for l in open(O / "e2f" / f"{m}_{mode}.jsonl")] if (O / "e2f" / f"{m}_{mode}.jsonl").exists() else []
            if not recs:
                continue
            idx = {(r["set"], r["item_id"], r["cond"]): r for r in recs}
            for st in ("syn", "occ", "all"):
                for grp in ("D+C-", "D+C+"):
                    for sub in ("all", "orig_claim_matches_group"):
                        base = [r for r in recs if r["cond"] == "orig" and r["group"] == grp and (st == "all" or r["set"] == st)]
                        if sub != "all":  # D+C-: original claim wrong; D+C+: original claim right (rule-based parse)
                            base = [r for r in base if r["orig_claim_wrong"] == int(grp == "D+C-")]
                        if len(base) < 5:
                            continue
                        g = [r["set"] + ":" + r["image_id"] for r in base]
                        row = {"n_items": len(base)}
                        corr = {}
                        for cond in CONDS:
                            rr = [idx.get((r["set"], r["item_id"], cond)) for r in base]
                            corr[cond] = [float(x is not None and pe(x) == x["gold"]) for x in rr]
                            row[f"acc:{cond}"] = ci(corr[cond], g)
                        for a, b in (("correct", "neutral"), ("correct", "orig"), ("wrong", "neutral"), ("wrong", "orig"),
                                     ("neutral", "orig"), ("neutral_ins", "orig"), ("correct", "wrong")):
                            row[f"diff:{a}-{b}"] = ci(np.array(corr[a]) - np.array(corr[b]), g)
                        # does the continuation (re)state a two-object claim, and is it wrong?
                        for cond in ("orig", "neutral", "correct", "wrong"):
                            st_wrong, st_has = [], []
                            for r in base:
                                x, c = idx.get((r["set"], r["item_id"], cond)), idx.get((r["set"], r["item_id"], "correct"))
                                if x is None or c is None:
                                    continue
                                fm = CLAIM_RX.search(c["prefix"])
                                fc = first_claim(split_sents(x["continuation"])) if fm else None
                                st_has.append(float(fc is not None))
                                if fc is not None:
                                    st_wrong.append(float(fc[1] != fm.group(1).lower()))
                            row[f"cont_has_claim:{cond}"] = round(100 * float(np.mean(st_has)), 1) if st_has else None
                            row[f"cont_claim_wrong:{cond}"] = round(100 * float(np.mean(st_wrong)), 1) if st_wrong else None
                        res[f"{m}|{mode}|{st}|{grp}|{sub}"] = row
                        if sub == "all":
                            lines.append(f"{m:13s} {mode:6s} {st:3s} {grp} n={len(base):3d} " + " ".join(
                                f"{c}={row['acc:' + c]['mean']:.1f}" for c in CONDS) +
                                f" | corr-neu {row['diff:correct-neutral']['mean']:+.1f} [{row['diff:correct-neutral']['lo']:+.1f},{row['diff:correct-neutral']['hi']:+.1f}]"
                                f" wrong-neu {row['diff:wrong-neutral']['mean']:+.1f} [{row['diff:wrong-neutral']['lo']:+.1f},{row['diff:wrong-neutral']['hi']:+.1f}]")
        p = O / "e2f" / f"{m}_inplace.jsonl"
        recs = [json.loads(l) for l in open(p)] if p.exists() else []
        idx = {(r["set"], r["item_id"], r["pos"], r.get("rvar", "full"), r["cond"]): r for r in recs}
        for pos, rvar in [(p_, v_) for p_ in ("claim", "commit") for v_ in ("full", "cut")]:
            for st in ("syn", "occ", "all"):
                for grp in ("D+C-", "D+C+"):
                    base = [r for r in recs if r["cond"] == "orig" and r["pos"] == pos and r["group"] == grp
                            and r.get("rvar", "full") == rvar and (st == "all" or r["set"] == st)]
                    if len(base) < 5:
                        continue
                    g = [r["set"] + ":" + r["image_id"] for r in base]
                    row = {"n_items": len(base)}
                    if pos == "commit":
                        row["share_commit_is_claim_sentence"] = round(100 * float(np.mean([r["p"] == r["j_claim"] for r in base])), 1)
                    corr = {}
                    for cond in INPLACE_CONDS:
                        rr = [idx.get((r["set"], r["item_id"], pos, rvar, cond)) for r in base]
                        corr[cond] = [float(x is not None and pe(x) == x["gold"]) for x in rr]
                        row[f"acc:{cond}"] = ci(corr[cond], g)
                    for a, b in (("correct", "neutral_len"), ("wrong", "neutral_len"), ("neutral_len", "orig"), ("correct", "wrong")):
                        row[f"diff:{a}-{b}"] = ci(np.array(corr[a]) - np.array(corr[b]), g)
                    res[f"{m}|inplace_{rvar}_{pos}|{st}|{grp}|all"] = row
                    lines.append(f"{m:13s} inpl-{rvar}-{pos:6s} {st:3s} {grp} n={len(base):3d} " + " ".join(
                        f"{c}={row['acc:' + c]['mean']:.1f}" for c in INPLACE_CONDS) +
                        f" | corr-neu {row['diff:correct-neutral_len']['mean']:+.1f} [{row['diff:correct-neutral_len']['lo']:+.1f},{row['diff:correct-neutral_len']['hi']:+.1f}]"
                        f" wrong-neu {row['diff:wrong-neutral_len']['mean']:+.1f} [{row['diff:wrong-neutral_len']['lo']:+.1f},{row['diff:wrong-neutral_len']['hi']:+.1f}]")
    json.dump(res, open(out_dir / "claim_edit.json", "w"), indent=1)
    (out_dir / "claim_edit.txt").write_text("acc = P(final answer == gold), readout after the continuation\n" + "\n".join(lines) + "\n")
    print("\n".join(lines))


def main() -> None:
    """CLI entry point."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--model")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--sets", default="syn,occ")
    ap.add_argument("--modes", default="same,closed")
    ap.add_argument("--analyze", action="store_true")
    ap.add_argument("--groups", default="D+C-,D+C+", help="Direct x CoT cells to edit (e.g. D-C+,D-C-)")
    ap.add_argument("--tag", default="", help="output file suffix (e.g. _dminus)")
    ap.add_argument("--outputs", default=f"{_vtp.OUTPUTS}")
    args = ap.parse_args()
    if args.analyze:
        analyze(Path("outputs/bundle"), Path(args.outputs))
    else:
        run(args)


if __name__ == "__main__":
    main()
