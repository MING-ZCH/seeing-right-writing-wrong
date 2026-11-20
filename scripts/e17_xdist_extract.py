"""E17x (GPU): layer-20 states for cross-distribution probe arbitration (round-10 Opus M4.3).

Items: the original occlusion twins (both phrasings) and Synthetic Depth from outputs/e10/<model>.jsonl (greedy CoT
stored there). First claim parsed with claim_parse v3. Saves the residual stream at layer 20 for the original image at
d (Direct "Answer: ("), r (last token of the claim's relation phrase) and a (CoT "Answer: (").
Writes /dev/shm/e17x/<model>_<shard>.npy (float16 [N, 3, D]) and .jsonl metadata.

Usage: python e17_xdist_extract.py --model qwen3vl_8b --shard 0 --nshard 2 [--limit 4]
"""
from __future__ import annotations

import argparse
import json
import os
import re
import time

import numpy as np

import vx
from vx import C
from e2d import load_occ
from e15_probe_extract import Runner
from e17_probe_ctrl_lite import claim_spans

OUTD = "/dev/shm/e17x"
LAYER = 20
QRX = re.compile(r"Is the (\w+ \w+) (in front of|behind) the (\w+ \w+)\?")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshard", type=int, default=1)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()
    t0 = time.time()
    its = {it["uid"]: dict(it, phrasing="front", key=it["uid"], set="syn") for it in vx.load_items(["syn_depth"])}
    its |= {it["uid"]: dict(it, phrasing="front", key=it["uid"], set="twins") for it in load_occ(None)}
    its |= {it["uid"] + "_behind": dict(it, phrasing="behind", key=it["uid"] + "_behind", set="twins") for it in load_occ(None, polar=True)}
    by = {}
    for r in vx.read_jsonl(f"{vx.OUT}/e10/{args.model}.jsonl"):
        if r["condition"] in ("direct", "cot") and r["item_id"] in its:
            by.setdefault(r["item_id"], {})[r["condition"]] = r
    items = [(its[k], d) for k, d in sorted(by.items()) if "direct" in d and "cot" in d]
    items = items[args.shard::args.nshard][: args.limit]
    print(f"[e17x] {args.model} shard {args.shard}: {len(items)} items", flush=True)
    R = Runner(args.model)
    os.makedirs(OUTD, exist_ok=True)
    arr = None
    meta = []
    for i, (it, rr) in enumerate(items):
        img = C.get_image(it)
        ref = vx.img_ref(img)
        cot = rr["cot"]["text"]
        cs = claim_spans(cot)
        hd, _ = R.forward(R.render(C.build_messages(it, "direct", "none", "", ref)), img, [])
        ctext = R.render(C.build_messages(it, "cot", "none", cot, ref))
        c0 = ctext.rfind(cot + "\n" + C.FORCE_PREFIX)
        if cs:
            hc, _ = R.forward(ctext, img, [c0 + cs["rel_end"]])
        else:
            h1, _ = R.forward(ctext, img, [])
            hc = np.concatenate([h1, h1], 0)
        h = np.stack([hd[-1, LAYER], hc[0, LAYER], hc[-1, LAYER]], 0)
        if arr is None:
            arr = np.zeros((len(items),) + h.shape, np.float16)
        arr[i] = h
        qm = QRX.search(it["question"])
        x, y_ = qm.group(1).lower(), qm.group(3).lower()
        if it["set"] == "syn":
            y_true = int(it["gold"] == "A")  # "Is X in front of Y?" -> yes = X in front
        else:
            y_true = int(it["meta"]["x_name"].lower() == it["meta"]["front_name"].lower())
        meta.append({"key": it["key"], "set": it["set"], "pair": it["group"], "phrasing": it["phrasing"], "x": x, "y_obj": y_,
                     "y_true": y_true, "named": cs["named"] if cs else None, "has_claim": int(bool(cs)), "gold": rr["cot"]["gold"],
                     "lp_direct": rr["direct"].get("lp_sum") or rr["direct"]["top20"], "lp_cot": rr["cot"].get("lp_sum") or rr["cot"]["top20"]})
        if i % 100 == 0:
            print(f"[e17x] {i}/{len(items)} ({time.time() - t0:.0f}s)", flush=True)
    sfx = f"_{args.shard}" + (f"_limit{args.limit}" if args.limit else "")
    np.save(f"{OUTD}/{args.model}{sfx}.npy", arr)
    with open(f"{OUTD}/{args.model}{sfx}.jsonl", "w") as f:
        for m in meta:
            f.write(json.dumps(m) + "\n")
    print(f"[e17x] done {arr.shape} ({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
