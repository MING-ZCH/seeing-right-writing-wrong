"""Cue-factorial occlusion set: 2 (size: front larger / smaller) x 2 (vertical: front lower / higher) x identity twins.

Occlusion is the only valid depth cue: in each cell size and vertical position are fixed by design, so across the four
cells they are balanced (front larger in half, front lower in half). Per cell n/4 base items, polarity balanced (asked
object is front -> yes in half). Each base item has an identity-swapped twin with identical geometry (same cell, answer
flips). Drawing, slot geometry and twin construction follow e2_occlusion_set.py; area ratio front/back is in
[1.25, 1.8] (front larger) or [0.55, 0.8] (front smaller).

Usage: python e12_cue_set.py --out-dir <W>/exps/items/occ_cues --n 400
"""
from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path

import numpy as np
from PIL import Image

from e2_occlusion_set import COLORS, SHAPES, bbox_of, mask_of, polygon_at, render

RATIO = {"larger": (1.25, 1.8), "smaller": (0.55, 0.8)}


def sample_layout(rng: random.Random, size: int, front_above: bool, ratio: tuple[float, float]) -> dict:
    """Slot geometry: front centre above/below the back centre (>= 18 px), objects inside the canvas."""
    for _ in range(5000):
        a_back = rng.uniform(0.04, 0.075) * size * size
        a_front = a_back * rng.uniform(*ratio)
        r_b, r_f = math.sqrt(a_back / math.pi), math.sqrt(a_front / math.pi)
        dist = rng.uniform(0.55, 1.05) * (r_b + r_f) * 0.75
        ang = math.radians(rng.uniform(25, 70))
        dy = dist * math.sin(ang) * (-1 if front_above else 1)
        dx = dist * math.cos(ang) * rng.choice([-1, 1])
        cx_b, cy_b = rng.uniform(0.3, 0.7) * size, rng.uniform(0.3, 0.7) * size
        geo = {"a_back": a_back, "a_front": a_front, "back_c": (cx_b, cy_b), "front_c": (cx_b + dx, cy_b + dy), "dx": dx}
        if abs(dy) < 18:
            continue
        if all(12 < cx - 1.6 * r and 12 < cy - 1.6 * r and cx + 1.6 * r < size - 12 and cy + 1.6 * r < size - 12
               for (cx, cy), r in ((geo["back_c"], r_b), (geo["front_c"], r_f))):
            return geo
    raise RuntimeError("layout sampling failed")


def make_pair(idx: int, rng: random.Random, size: int, front_above: bool, front_larger: bool, ask_front: bool):
    """One base item and its identity-swapped twin (records + images)."""
    ratio = RATIO["larger" if front_larger else "smaller"]
    while True:
        geo = sample_layout(rng, size, front_above, ratio)
        sh, co = rng.sample(SHAPES, 2), rng.sample(sorted(COLORS), 2)
        ident = [{"shape": sh[0], "color": co[0]}, {"shape": sh[1], "color": co[1]}]
        polys = {"front": polygon_at(ident[0]["shape"], geo["a_front"], *geo["front_c"]),
                 "back": polygon_at(ident[1]["shape"], geo["a_back"], *geo["back_c"])}
        polys_sw = {"front": polygon_at(ident[1]["shape"], geo["a_front"], *geo["front_c"]),
                    "back": polygon_at(ident[0]["shape"], geo["a_back"], *geo["back_c"])}
        ok, stats = True, {}
        for tag, pol in (("orig", polys), ("twin", polys_sw)):
            mf, mb = mask_of(pol["front"], size), mask_of(pol["back"], size)
            hidden = (mf & mb).sum() / mb.sum()
            front_out = (mf & ~mb).sum() / mf.sum()
            larger_ok = (mf.sum() > mb.sum()) == front_larger
            if not (0.15 <= hidden <= 0.55 and front_out >= 0.35 and larger_ok):
                ok = False
            bf, bb = bbox_of(mf), bbox_of(mb)
            stats[tag] = {"area_front_px": int(mf.sum()), "area_back_px": int(mb.sum()), "back_hidden_frac": float(hidden),
                          "front_outside_frac": float(front_out), "bbox_front": bf, "bbox_back": bb}
        if ok:
            break
    recs, imgs = [], []
    for tag, front_id, back_id, pol in (("orig", ident[0], ident[1], polys), ("twin", ident[1], ident[0], polys_sw)):
        fname, bname = f"{front_id['color']} {front_id['shape']}", f"{back_id['color']} {back_id['shape']}"
        x_name, y_name = (fname, bname) if ask_front else (bname, fname)
        if tag == "twin":
            x_name, y_name = recs[0]["meta"]["x_name"], recs[0]["meta"]["y_name"]
        imgs.append(render({"poly": pol["front"], **front_id}, {"poly": pol["back"], **back_id}, size))
        recs.append({
            "item_id": f"cue{idx:03d}" + ("_cf" if tag == "twin" else ""),
            "task": "occ_cues" + ("_cf" if tag == "twin" else ""),
            "question": f"Is the {x_name} in front of the {y_name}?", "options": ["yes", "no"],
            "gold": "A" if x_name == fname else "B",
            "meta": {"front_name": fname, "back_name": bname, "x_name": x_name, "y_name": y_name,
                     "front_above": front_above, "front_larger": front_larger,
                     "cell": ("front_larger" if front_larger else "front_smaller") + "|" + ("front_higher" if front_above else "front_lower"),
                     "dx_sign": int(np.sign(geo["dx"])),
                     "area_ratio_front_over_back": stats[tag]["area_front_px"] / stats[tag]["area_back_px"], **stats[tag]}})
    return recs[0], recs[1], imgs[0], imgs[1]


def _make(job):
    i, seed, size, fa, fl, af = job
    return make_pair(i, random.Random(seed), size, fa, fl, af)


def main() -> None:
    """CLI entry point."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--n", type=int, default=400, help="base items (multiple of 8)")
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()
    rng = random.Random(args.seed)
    out = Path(args.out_dir)
    (out / "images").mkdir(parents=True, exist_ok=True)
    cells = [(fa, fl, af) for fa in (True, False) for fl in (True, False) for af in (True, False)]
    plan = [c for c in cells for _ in range(args.n // 8)]
    rng.shuffle(plan)
    from multiprocessing import Pool
    jobs = [(i, args.seed * 100003 + i, args.size, fa, fl, af) for i, (fa, fl, af) in enumerate(plan)]
    with Pool(min(48, len(jobs))) as pool:
        made = pool.map(_make, jobs)
    base, twins = [], []
    for r0, r1, im0, im1 in made:
        for r, im, bucket in ((r0, im0, base), (r1, im1, twins)):
            im.save(out / "images" / f"{r['item_id']}.png")
            r["image_paths"] = [f"images/{r['item_id']}.png"]
            r["image_id"] = r["item_id"].replace("_cf", "")
            bucket.append(r)
    for name, recs in (("items.jsonl", base), ("items_cf.jsonl", twins)):
        with open(out / name, "w") as f:
            for r in recs:
                f.write(json.dumps(r) + "\n")
    allr = base + twins
    report = {"n_base": len(base), "n_twins": len(twins), "seed": args.seed,
              "cells": {c: sum(r["meta"]["cell"] == c for r in base) for c in sorted({r["meta"]["cell"] for r in base})},
              "yes_base": sum(r["gold"] == "A" for r in base),
              "ratio_larger": [round(min(r["meta"]["area_ratio_front_over_back"] for r in allr if r["meta"]["front_larger"]), 3),
                               round(max(r["meta"]["area_ratio_front_over_back"] for r in allr if r["meta"]["front_larger"]), 3)],
              "ratio_smaller": [round(min(r["meta"]["area_ratio_front_over_back"] for r in allr if not r["meta"]["front_larger"]), 3),
                                round(max(r["meta"]["area_ratio_front_over_back"] for r in allr if not r["meta"]["front_larger"]), 3)],
              "hidden_frac_range": [round(min(r["meta"]["back_hidden_frac"] for r in allr), 3),
                                    round(max(r["meta"]["back_hidden_frac"] for r in allr), 3)]}
    (out / "generation_report.json").write_text(json.dumps(report, indent=2))
    sheet = Image.new("RGB", (4 * 256, 4 * 256), (255, 255, 255))
    for k, r in enumerate(base[:8] + twins[:8]):
        sheet.paste(Image.open(out / r["image_paths"][0]).resize((256, 256)), ((k % 4) * 256, (k // 4) * 256))
    sheet.save(out / "contact_sheet.png")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
