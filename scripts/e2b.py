"""E2b: sentence-level early-answering trajectory of the stored greedy rationale R.

For every spatial item, split R (after the CoT trigger) into sentences s_1..s_J and read the forced
answer after each prefix R_{1:j} (j = 0..J) with the turn closed (``cot+turn`` layout). Stores exact
P_j over options; the commit sentence and sentence types are derived offline (analyze_e2.py).

Usage: python e2b.py --model qwen3b [--limit 5] [--max-sent 30]
"""
from __future__ import annotations

import argparse
import re
import time

import vx
from vx import C

_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z(\"'*\d])|\n+")
_ENUM = re.compile(r"^\s*(\d+|[a-z])[.)]\s*$|^\s*[-*]+\s*$")


def split_sentences(text: str) -> list[str]:
    """Split rationale text into sentences; bare list markers are merged into the next piece."""
    raw = [s.strip() for s in _SPLIT.split(text) if s and s.strip()]
    out: list[str] = []
    carry = ""
    for s in raw:
        if _ENUM.match(s):
            carry = (carry + " " + s).strip()
            continue
        out.append((carry + " " + s).strip() if carry else s)
        carry = ""
    return out


def body(cot: str) -> str:
    """Rationale text without the CoT trigger."""
    return cot[len(C.COT_PREFIX):].strip() if cot.startswith(C.COT_PREFIX) else cot.strip()


def main() -> None:
    """CLI entry point."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--max-sent", type=int, default=30)
    args = ap.parse_args()
    t0 = time.time()
    leg = vx.legacy_e1(args.model)
    items = [it for it in vx.load_items(limit=args.limit) if it["uid"] in leg]
    llm = vx.make_llm(args.model)
    msgs, choices, keys, sents_all = [], [], [], {}
    for it in items:
        sents = split_sentences(body(leg[it["uid"]]["cot"]))[: args.max_sent]
        sents_all[it["uid"]] = sents
        ref = vx.img_ref(C.get_image(it))
        for j in range(len(sents) + 1):
            prefix = (C.COT_PREFIX + " " + " ".join(sents[:j])).strip()
            msgs.append(C.build_messages(it, "cot", "turn", prefix, ref))
            choices.append(it["choices"])
            keys.append((it["uid"], j))
    print(f"[e2b] {args.model}: {len(items)} items, {len(msgs)} prefix readouts", flush=True)
    outs = vx.readout(llm, args.model, msgs, choices)
    traj: dict[str, list] = {}
    for (uid, j), o in zip(keys, outs):
        traj.setdefault(uid, []).append({"j": j, "lp_bare": o["lp_bare"], "lp_sum": o["lp_sum"], "pred_exact": o["pred_exact"],
                                         "pred_top20": o["pred_top20"]})
    recs = []
    for it in items:
        recs.append({"model": args.model, "task": it["task"], "item_id": it["uid"], "gold": it["gold"],
                     "letters": it["choices"], "image_id": it["group"], "question": it["question"],
                     "cot_free": leg[it["uid"]]["cot_free"], "sentences": sents_all[it["uid"]],
                     "traj": traj[it["uid"]]})
    sfx = f"_limit{args.limit}" if args.limit else ""
    vx.write_jsonl(f"{vx.OUT}/e2b/{args.model}{sfx}.jsonl", recs)
    print(f"[e2b] wrote {len(recs)} records ({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
