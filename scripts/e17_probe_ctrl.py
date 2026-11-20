"""E17 (GPU): image controls for the within-claim truth probe (round-6 Opus M1).

Cue-factorial set, both phrasings. The stored greedy CoT text (outputs/e10/<model>_cues.jsonl) is teacher-forced
under three images: the original, a uniform gray image, and the identity-swapped twin (same layout, front and back
shapes swapped, so the true relation flips while the text is fixed). First claim parsed with claim_parse v3.
Residual stream (all layers) saved at, in the CoT pass:
  pre    token before the first token of the claim's first object mention (the state that writes the subject noun)
  s      last token of that object mention
  rprev  token before the first token of the relation phrase (the state that writes the relation word)
  r      last token of the relation phrase (the state after reading it)
  a      last token ("Answer: (" after the rationale)
plus d (Direct prompt, last token) for the original image.
Writes /dev/shm/e17/<model>_<shard>_{orig,gray,twin}.npy (float16 [N, P, L+1, D]) and <model>_<shard>.jsonl.

Usage: python e17_probe_ctrl.py --model qwen3vl_8b --shard 0 --nshard 2 [--limit 4]
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
from PIL import Image

import vx
from vx import C
from e17_probe_ctrl_lite import claim_spans
from e2d import load_occ
from e15_probe_extract import Runner

OUTD = "/dev/shm/e17"
POS_COT = ("pre", "s", "rprev", "r", "a")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshard", type=int, default=1)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()
    t0 = time.time()
    cues = os.environ.get("CUES_DIR", "/dev/shm/occ_cues")
    its = [dict(it, phrasing="front", key=it["uid"]) for it in load_occ(None, items_dir=cues)]
    its += [dict(it, phrasing="behind", key=it["uid"] + "_behind") for it in load_occ(None, polar=True, items_dir=cues)]
    img_of = {it["uid"]: it["image_path"] for it in its}
    by = {}
    for r in vx.read_jsonl(f"{vx.OUT}/e10/{args.model}_cues.jsonl"):
        if r["condition"] in ("direct", "cot"):
            by.setdefault(r["item_id"], {})[r["condition"]] = r
    items = [(it, by[it["key"]]) for it in its if it["key"] in by and "cot" in by[it["key"]]]
    items = items[args.shard::args.nshard][: args.limit]
    print(f"[e17] {args.model} shard {args.shard}/{args.nshard}: {len(items)} items", flush=True)
    R = Runner(args.model)
    os.makedirs(OUTD, exist_ok=True)
    arrs = {k: None for k in ("orig", "gray", "twin")}
    meta = []
    for i, (it, rr) in enumerate(items):
        img = C.get_image(it)
        twin_uid = it["uid"][:-3] if it["uid"].endswith("_cf") else it["uid"] + "_cf"
        imgs = {"orig": img, "gray": Image.new("RGB", img.size, (128, 128, 128)), "twin": Image.open(img_of[twin_uid]).convert("RGB")}
        cot = rr["cot"]["text"]
        cs = claim_spans(cot)
        ref = vx.img_ref(img)
        hd, _ = R.forward(R.render(C.build_messages(it, "direct", "none", "", ref)), img, [])
        ctext = R.render(C.build_messages(it, "cot", "none", cot, ref))
        c0 = ctext.rfind(cot + "\n" + C.FORCE_PREFIX)
        info = {}
        for cond, im in imgs.items():
            if cs:
                # token before the first token of the subject / relation phrase (BPE attaches the leading space)
                chars = [(c0 + cs["subj_start"], -1), c0 + cs["subj_end"], (c0 + cs["rel_start"], -1), c0 + cs["rel_end"]]
                h, tk = R.forward(ctext, im, chars)
                if cond == "orig":
                    info = {"tok_pre": R.last_info["tok_str"][0], "tok_s": R.last_info["tok_str"][1], "tok_rprev": R.last_info["tok_str"][2],
                            "tok_r": R.last_info["tok_str"][3]}
            else:
                h1, tk = R.forward(ctext, im, [])
                h = np.concatenate([h1] * 5, 0)
            info[f"ab_{cond}"] = R.last_info["ab"]
            if cond == "orig":
                h = np.concatenate([hd[-1:], h], 0)
            if arrs[cond] is None:
                arrs[cond] = np.zeros((len(items),) + h.shape, np.float16)
            arrs[cond][i] = h
        mt = it["meta"]
        meta.append({"key": it["key"], "pair": it["group"], "phrasing": it["phrasing"], "x": mt["x_name"].lower(),
                     "front": mt["front_name"].lower(), "y_true": int(mt["x_name"].lower() == mt["front_name"].lower()),
                     "named": cs["named"] if cs else None, "has_claim": int(bool(cs)), "negated": cs["negated"] if cs else None,
                     "gold": rr["cot"]["gold"], "lp_direct": rr["direct"].get("lp_sum") or rr["direct"]["top20"],
                     "lp_cot": rr["cot"].get("lp_sum") or rr["cot"]["top20"], "front_above": mt.get("front_above"),
                     "front_larger": mt.get("front_larger"), **info})
        if i % 50 == 0:
            print(f"[e17] {i}/{len(items)} ({time.time() - t0:.0f}s)", flush=True)
    sfx = f"_{args.shard}" + (f"_limit{args.limit}" if args.limit else "")
    for cond, a in arrs.items():
        np.save(f"{OUTD}/{args.model}{sfx}_{cond}.npy", a)
    with open(f"{OUTD}/{args.model}{sfx}.jsonl", "w") as fo:
        for m in meta:
            fo.write(json.dumps(m) + "\n")
    print(f"[e17] done ({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
