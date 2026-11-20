"""Difficulty-calibrated occlusion set (E2-hard): cue-decoupled depth ordering at several difficulty levels.

Same design as ``e2_occlusion_set.py`` (the original set is never modified):

* the front object's pixel area is never larger than the back object's, so occlusion is the only
  depth cue;
* every item has an identity-swapped twin with identical geometry, distractors and pixel noise, so
  the gold answer flips while the image statistics stay matched;
* balanced over vertical direction of the front object and over answer polarity.

Difficulty is raised with five knobs that do not reintroduce a size cue:

* overlap: how much of the back object the front object hides (thin sliver at hard levels);
* contrast: object colours are blended towards the background and drawn from confusable
  colour pairs (names stay unambiguous: "red" vs "orange");
* distractors: extra non-overlapping shapes elsewhere in the scene;
* blur / sensor noise;
* object size: smaller objects (fraction of canvas area).

Levels (``--levels``): L0 reproduces the original easy setting, L1-L4 are progressively harder.

Usage: python e2_occlusion_hard.py --out-dir items/occlusion_hard --n 200 --levels L1,L2,L3
"""

from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

from e2_occlusion_set import BG, COLORS, OUTLINE, SHAPES, bbox_of, mask_of, polygon_at

# Colour pairs that stay nameable but are perceptually close (used at low-contrast levels).
CONFUSABLE = [("red", "orange"), ("blue", "purple"), ("green", "yellow"), ("orange", "yellow"),
              ("red", "purple"), ("blue", "green")]

# level -> generator parameters
LEVELS: dict[str, dict[str, Any]] = {
    # hidden: fraction of the back object hidden by the front object
    # area: back-object area as a fraction of the canvas
    # contrast: 1.0 = saturated colours, lower blends the object towards the background
    # confusable: draw the two object colours from a confusable pair
    "L0": dict(hidden=(0.15, 0.55), area=(0.045, 0.090), contrast=1.00, confusable=False,
               distractors=0, blur=0.0, noise=0.0, outline=3),
    "L1": dict(hidden=(0.10, 0.20), area=(0.030, 0.050), contrast=0.85, confusable=False,
               distractors=2, blur=0.4, noise=3.0, outline=2),
    "L2": dict(hidden=(0.06, 0.12), area=(0.018, 0.032), contrast=0.70, confusable=True,
               distractors=3, blur=0.8, noise=5.0, outline=2),
    "L3": dict(hidden=(0.035, 0.075), area=(0.010, 0.020), contrast=0.55, confusable=True,
               distractors=4, blur=1.1, noise=7.0, outline=1),
    # L3b sits between L3 and L4: in the pilot, closed-model Direct accuracy is at ceiling up to L3
    # (100%) and drops to 75% at L4, so the transition is sampled with one intermediate level.
    "L3b": dict(hidden=(0.028, 0.060), area=(0.0075, 0.0155), contrast=0.50, confusable=True,
                distractors=5, blur=1.25, noise=8.0, outline=1),
    "L4": dict(hidden=(0.020, 0.045), area=(0.006, 0.013), contrast=0.45, confusable=True,
               distractors=5, blur=1.4, noise=9.0, outline=1),
}


def blend(color: tuple[int, int, int], contrast: float) -> tuple[int, int, int]:
    """Blend a colour towards the background; contrast 1.0 keeps it saturated."""
    return tuple(int(round(contrast * c + (1 - contrast) * b)) for c, b in zip(color, BG))


def sample_layout(rng: random.Random, size: int, front_above: bool, cfg: dict,
                  ratio_min: float, max_tries: int = 4000) -> dict[str, Any]:
    """Rejection-sample slot geometry for a difficulty level.

    Constraints: front area <= back area (ratio in [ratio_min, 1]); both objects inside the canvas
    with a 12 px margin; front centre above (below) the back centre. The hidden fraction is checked
    on the rasterised masks in :func:`make_pair`; here we only pre-select a plausible separation.
    """
    lo, hi = cfg["hidden"]
    for _ in range(max_tries):
        a_back = rng.uniform(*cfg["area"]) * size * size
        a_front = a_back * rng.uniform(ratio_min, 1.0)
        r_b, r_f = math.sqrt(a_back / math.pi), math.sqrt(a_front / math.pi)
        # smaller hidden fraction -> larger centre distance
        frac = 1.0 - math.sqrt(max(lo, min(hi, rng.uniform(lo, hi))))
        dist = (r_b + r_f) * rng.uniform(0.55 + 0.45 * frac, 0.75 + 0.45 * frac)
        ang = math.radians(rng.uniform(25, 70))
        dy = dist * math.sin(ang) * (-1 if front_above else 1)
        dx = dist * math.cos(ang) * rng.choice([-1, 1])
        geo = {"a_back": a_back, "a_front": a_front, "back_c": (0.0, 0.0), "front_c": (0.0, 0.0),
               "dx": dx, "dy": dy}
        if abs(dy) < max(8.0, 0.06 * (r_b + r_f)):
            continue
        cx_b = rng.uniform(0.3, 0.7) * size
        cy_b = rng.uniform(0.3, 0.7) * size
        geo["back_c"], geo["front_c"] = (cx_b, cy_b), (cx_b + dx, cy_b + dy)
        ok = True
        for (cx, cy), r in ((geo["back_c"], r_b), (geo["front_c"], r_f)):
            if cx - 1.6 * r < 12 or cy - 1.6 * r < 12 or cx + 1.6 * r > size - 12 or cy + 1.6 * r > size - 12:
                ok = False
        if ok:
            return geo
    raise RuntimeError("layout sampling failed")


def sample_distractors(rng: random.Random, size: int, cfg: dict, targets: list[np.ndarray],
                       used_names: set[str]) -> list[dict[str, Any]]:
    """Non-overlapping extra shapes placed away from the two target objects."""
    out: list[dict[str, Any]] = []
    occupied = np.zeros((size, size), bool)
    for m in targets:
        occupied |= m
    for _ in range(cfg["distractors"]):
        for _try in range(400):
            shape, color = rng.choice(SHAPES), rng.choice(sorted(COLORS))
            if f"{color} {shape}" in used_names:
                continue
            area = rng.uniform(*cfg["area"]) * size * size
            r = math.sqrt(area / math.pi)
            cx, cy = rng.uniform(1.8 * r, size - 1.8 * r), rng.uniform(1.8 * r, size - 1.8 * r)
            poly = polygon_at(shape, area, cx, cy)
            m = mask_of(poly, size)
            grown = np.asarray(Image.fromarray((m * 255).astype(np.uint8)).filter(ImageFilter.MaxFilter(9))) > 0
            if (grown & occupied).any():
                continue
            occupied |= m
            out.append({"poly": poly, "shape": shape, "color": color})
            used_names.add(f"{color} {shape}")
            break
    return out


def render(front: dict, back: dict, distractors: list[dict], size: int, cfg: dict,
           noise: np.ndarray | None) -> Image.Image:
    """Draw distractors, then the back object, then the occluding front object; blur and add noise."""
    img = Image.new("RGB", (size, size), BG)
    d = ImageDraw.Draw(img)
    outline = blend(OUTLINE, min(1.0, cfg["contrast"] + 0.25))
    for o in distractors + [back, front]:
        d.polygon(o["poly"], fill=blend(COLORS[o["color"]], cfg["contrast"]), outline=outline, width=cfg["outline"])
    if cfg["blur"] > 0:
        img = img.filter(ImageFilter.GaussianBlur(cfg["blur"]))
    if noise is not None:
        arr = np.clip(np.asarray(img, np.float32) + noise, 0, 255).astype(np.uint8)
        img = Image.fromarray(arr)
    return img


def make_pair(idx: int, level: str, rng: random.Random, size: int, front_above: bool, ask_front: bool,
              ratio_min: float) -> tuple[dict, dict, Image.Image, Image.Image]:
    """One item and its identity-swapped twin at a given difficulty level."""
    cfg = LEVELS[level]
    lo, hi = cfg["hidden"]
    while True:
        geo = sample_layout(rng, size, front_above, cfg, ratio_min)
        if cfg["confusable"]:
            co = list(rng.choice(CONFUSABLE))
            rng.shuffle(co)
            sh = rng.sample(SHAPES, 2)
        else:
            sh, co = rng.sample(SHAPES, 2), rng.sample(sorted(COLORS), 2)
        ident = [{"shape": sh[0], "color": co[0]}, {"shape": sh[1], "color": co[1]}]
        polys = {slot: polygon_at(ident_i["shape"], geo[f"a_{slot}"], *geo[f"{slot}_c"])
                 for slot, ident_i in (("front", ident[0]), ("back", ident[1]))}
        m_f, m_b = mask_of(polys["front"], size), mask_of(polys["back"], size)
        polys_sw = {"front": polygon_at(ident[1]["shape"], geo["a_front"], *geo["front_c"]),
                    "back": polygon_at(ident[0]["shape"], geo["a_back"], *geo["back_c"])}
        m_f2, m_b2 = mask_of(polys_sw["front"], size), mask_of(polys_sw["back"], size)
        ok, stats = True, {}
        for tag, mf, mb in (("orig", m_f, m_b), ("twin", m_f2, m_b2)):
            if not mb.sum() or not mf.sum():
                ok = False
                break
            hidden = (mf & mb).sum() / mb.sum()
            front_out = (mf & ~mb).sum() / mf.sum()
            # thin sliver but still unambiguous: a few hidden pixels are not enough
            if not (lo <= hidden <= hi and front_out >= 0.55 and mf.sum() <= mb.sum() and (mf & mb).sum() >= 40):
                ok = False
            stats[tag] = {"area_front_px": int(mf.sum()), "area_back_px": int(mb.sum()),
                          "back_hidden_frac": float(hidden), "front_outside_frac": float(front_out),
                          "bbox_front": bbox_of(mf), "bbox_back": bbox_of(mb)}
        if ok:
            break
    used = {f"{i['color']} {i['shape']}" for i in ident}
    distr = sample_distractors(rng, size, cfg, [m_f | m_f2, m_b | m_b2], used)
    noise = None
    if cfg["noise"] > 0:
        noise = np.random.default_rng(1000 + idx).normal(0, cfg["noise"], (size, size, 3)).astype(np.float32)
    recs, imgs = [], []
    for tag, front_id, back_id, pol in (("orig", ident[0], ident[1], polys), ("twin", ident[1], ident[0], polys_sw)):
        fname = f"{front_id['color']} {front_id['shape']}"
        bname = f"{back_id['color']} {back_id['shape']}"
        x_name, y_name = (fname, bname) if ask_front else (bname, fname)
        if tag == "twin":  # the twin keeps the question text, so the answer flips
            x_name, y_name = recs[0]["meta"]["x_name"], recs[0]["meta"]["y_name"]
        answer_yes = x_name == fname
        imgs.append(render({"poly": pol["front"], **front_id}, {"poly": pol["back"], **back_id},
                           distr, size, cfg, noise))
        recs.append({
            "item_id": f"occ{level.lower()}{idx:03d}" + ("_cf" if tag == "twin" else ""),
            "task": "occ_decoupled" + ("_cf" if tag == "twin" else ""),
            "question": f"Is the {x_name} in front of the {y_name}?",
            "options": ["yes", "no"], "gold": "A" if answer_yes else "B",
            "meta": {"front_name": fname, "back_name": bname, "x_name": x_name, "y_name": y_name,
                     "front_above": front_above, "dx_sign": int(np.sign(geo["dx"])), "level": level,
                     "n_distractors": len(distr),
                     "area_ratio_front_over_back": stats[tag]["area_front_px"] / stats[tag]["area_back_px"],
                     "twin_id": f"occ{level.lower()}{idx:03d}" + ("" if tag == "twin" else "_cf"), **stats[tag]},
        })
    return recs[0], recs[1], imgs[0], imgs[1]


def main() -> None:
    """CLI entry point."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", default="items/occlusion_hard")
    ap.add_argument("--n", type=int, default=200, help="total number of base items (split over levels)")
    ap.add_argument("--levels", default="L1,L2,L3,L4")
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--ratio-min", type=float, default=0.55, help="min front/back area ratio")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    levels = args.levels.split(",")
    per = args.n // len(levels)
    assert per % 4 == 0, "items per level must be a multiple of 4 (balanced cells)"
    rng = random.Random(args.seed)
    out = Path(args.out_dir)
    (out / "images").mkdir(parents=True, exist_ok=True)
    cells = [(fa, af) for fa in (True, False) for af in (True, False)]
    base, twins = [], []
    for level in levels:
        plan = [c for c in cells for _ in range(per // 4)]
        rng.shuffle(plan)
        for i, (front_above, ask_front) in enumerate(plan):
            r0, r1, im0, im1 = make_pair(i, level, rng, args.size, front_above, ask_front, args.ratio_min)
            for r, im, bucket in ((r0, im0, base), (r1, im1, twins)):
                p = out / "images" / f"{r['item_id']}.png"
                im.save(p)
                r["image_paths"] = [f"images/{p.name}"]
                r["image_id"] = r["item_id"].replace("_cf", "")
                bucket.append(r)
        print(f"[gen] {level}: {per} base items", flush=True)
    for name, recs in (("items.jsonl", base), ("items_cf.jsonl", twins)):
        with open(out / name, "w") as f:
            for r in recs:
                f.write(json.dumps(r) + "\n")
    report: dict[str, Any] = {"levels": levels, "n_base": len(base), "n_twins": len(twins), "seed": args.seed,
                              "size": args.size, "ratio_min": args.ratio_min, "per_level": {}}
    for level in levels:
        rs = [r for r in base + twins if r["meta"]["level"] == level]
        ratios = [r["meta"]["area_ratio_front_over_back"] for r in rs]
        hid = [r["meta"]["back_hidden_frac"] for r in rs]
        report["per_level"][level] = {
            "n": len(rs), "yes_rate": float(np.mean([r["gold"] == "A" for r in rs])),
            "front_above_rate": float(np.mean([r["meta"]["front_above"] for r in rs])),
            "max_area_ratio": max(ratios), "mean_area_ratio": float(np.mean(ratios)),
            "violations_area": sum(x > 1.0 for x in ratios),
            "hidden_frac": [min(hid), float(np.mean(hid)), max(hid)],
            "mean_back_area_frac": float(np.mean([r["meta"]["area_back_px"] / args.size ** 2 for r in rs])),
            "params": LEVELS[level],
        }
    (out / "generation_report.json").write_text(json.dumps(report, indent=2))
    # contact sheet: one row per level (4 base items with their twins)
    cell = 220
    sheet = Image.new("RGB", (8 * cell, len(levels) * cell), (255, 255, 255))
    dr = ImageDraw.Draw(sheet)
    for row, level in enumerate(levels):
        bs = [r for r in base if r["meta"]["level"] == level][:4]
        ts = {r["item_id"]: r for r in twins if r["meta"]["level"] == level}
        for k, r in enumerate(bs):
            for j, rec in enumerate((r, ts[r["item_id"] + "_cf"])):
                im = Image.open(out / rec["image_paths"][0]).resize((cell, cell))
                sheet.paste(im, ((2 * k + j) * cell, row * cell))
        dr.text((4, row * cell + 4), level, fill=(0, 0, 0))
    sheet.save(out / "contact_sheet.png")
    print(json.dumps(report["per_level"], indent=2, default=str))


if __name__ == "__main__":
    main()
