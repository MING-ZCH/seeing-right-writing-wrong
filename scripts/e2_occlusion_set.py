"""Cue-decoupled occlusion set (E2): 200 depth-ordering items + identity-swapped twins.

In the original Synthetic Depth Ordering set the front object both overlaps
and is drawn larger than the back object, so size and occlusion are collinear.
Here the front object's pixel area is never larger than the back object's
(area ratio front/back in ``[ratio_min, 1]``), so occlusion is the only depth
cue. The design is balanced over

* vertical direction of the front object (above / below the back object's
  centre; 100 each),
* answer polarity (asked object is front -> yes, back -> no; 100 each),
* horizontal offset sign (random, recorded).

Each item has a counterfactual twin with the two objects' identities
(shape + colour) swapped while geometry (slot areas, positions, drawing order)
is unchanged; the twin's answer is flipped, and the front object stays the
smaller one.

Question format follows the original binary synthetic sets:
``Is the {X} in front of the {Y}?`` with ``(A) yes`` / ``(B) no``.
"""

from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw

SHAPES = ("circle", "square", "triangle", "hexagon", "pentagon")
COLORS = {"red": (220, 50, 47), "blue": (38, 110, 210), "green": (60, 160, 70), "yellow": (235, 195, 40),
          "purple": (140, 80, 180), "orange": (240, 130, 30)}
BG = (238, 238, 238)
OUTLINE = (30, 30, 30)


def unit_polygon(shape: str) -> np.ndarray:
    """Vertices of a shape with unit area centred at the origin."""
    if shape == "circle":
        n, phase = 96, 0.0
    elif shape == "square":
        n, phase = 4, math.pi / 4
    elif shape == "triangle":
        n, phase = 3, -math.pi / 2
    elif shape == "pentagon":
        n, phase = 5, -math.pi / 2
    elif shape == "hexagon":
        n, phase = 6, 0.0
    else:
        raise ValueError(shape)
    ang = phase + 2 * math.pi * np.arange(n) / n
    pts = np.stack([np.cos(ang), np.sin(ang)], 1)
    x, y = pts[:, 0], pts[:, 1]
    area = 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))
    pts = pts / math.sqrt(area)
    pts[:, 1] -= pts[:, 1].mean()
    return pts


def polygon_at(shape: str, area: float, cx: float, cy: float) -> list[tuple[float, float]]:
    """Polygon of a given pixel area centred at (cx, cy)."""
    pts = unit_polygon(shape) * math.sqrt(area)
    return [(float(cx + x), float(cy + y)) for x, y in pts]


def mask_of(poly: list[tuple[float, float]], size: int) -> np.ndarray:
    """Boolean raster mask of a polygon."""
    m = Image.new("L", (size, size), 0)
    ImageDraw.Draw(m).polygon(poly, fill=255)
    return np.asarray(m) > 0


def render(front: dict[str, Any], back: dict[str, Any], size: int) -> Image.Image:
    """Draw the back object first, then the occluding front object."""
    img = Image.new("RGB", (size, size), BG)
    d = ImageDraw.Draw(img)
    for o in (back, front):
        d.polygon(o["poly"], fill=COLORS[o["color"]], outline=OUTLINE, width=3)
    return img


def bbox_of(mask: np.ndarray) -> list[int]:
    """Tight bbox ``[x1, y1, x2, y2]`` of a mask."""
    ys, xs = np.nonzero(mask)
    return [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]


def sample_layout(rng: random.Random, size: int, front_above: bool, ratio_min: float,
                  max_tries: int = 2000) -> dict[str, Any]:
    """Rejection-sample slot geometry satisfying the occlusion constraints.

    Constraints: front area <= back area; 15-55% of the back object hidden;
    at least 35% of the front object outside the back object's footprint;
    both objects fully inside the canvas with a 12 px margin; front centre
    above (or below) the back centre by at least 18 px.
    """
    for _ in range(max_tries):
        a_back = rng.uniform(0.045, 0.09) * size * size
        a_front = a_back * rng.uniform(ratio_min, 1.0)
        r_b, r_f = math.sqrt(a_back / math.pi), math.sqrt(a_front / math.pi)
        dist = rng.uniform(0.55, 1.05) * (r_b + r_f) * 0.75
        ang = math.radians(rng.uniform(25, 70))
        dy = dist * math.sin(ang) * (-1 if front_above else 1)
        dx = dist * math.cos(ang) * rng.choice([-1, 1])
        cx_b = rng.uniform(0.3, 0.7) * size
        cy_b = rng.uniform(0.3, 0.7) * size
        geo = {"a_back": a_back, "a_front": a_front, "back_c": (cx_b, cy_b), "front_c": (cx_b + dx, cy_b + dy),
               "dx": dx, "dy": dy}
        if abs(dy) < 18:
            continue
        ok = True
        for (cx, cy), r in ((geo["back_c"], r_b), (geo["front_c"], r_f)):
            if cx - 1.6 * r < 12 or cy - 1.6 * r < 12 or cx + 1.6 * r > size - 12 or cy + 1.6 * r > size - 12:
                ok = False
        if ok:
            return geo
    raise RuntimeError("layout sampling failed")


def make_pair(idx: int, rng: random.Random, size: int, front_above: bool, ask_front: bool,
              ratio_min: float) -> tuple[dict[str, Any], dict[str, Any], Image.Image, Image.Image]:
    """Create one item and its identity-swapped twin (records + images)."""
    while True:
        geo = sample_layout(rng, size, front_above, ratio_min)
        sh = rng.sample(SHAPES, 2)
        co = rng.sample(sorted(COLORS), 2)
        ident = [{"shape": sh[0], "color": co[0]}, {"shape": sh[1], "color": co[1]}]
        polys = {}
        for slot, ident_i in (("front", ident[0]), ("back", ident[1])):
            c = geo[f"{slot}_c"]
            polys[slot] = polygon_at(ident_i["shape"], geo[f"a_{slot}"], c[0], c[1])
        m_f, m_b = mask_of(polys["front"], size), mask_of(polys["back"], size)
        # Swapped twin geometry: same slots, swapped identities.
        polys_sw = {"front": polygon_at(ident[1]["shape"], geo["a_front"], *geo["front_c"]),
                    "back": polygon_at(ident[0]["shape"], geo["a_back"], *geo["back_c"])}
        m_f2, m_b2 = mask_of(polys_sw["front"], size), mask_of(polys_sw["back"], size)
        ok = True
        stats = {}
        for tag, mf, mb in (("orig", m_f, m_b), ("twin", m_f2, m_b2)):
            hidden = (mf & mb).sum() / mb.sum()
            front_out = (mf & ~mb).sum() / mf.sum()
            if not (0.15 <= hidden <= 0.55 and front_out >= 0.35 and mf.sum() <= mb.sum()):
                ok = False
            stats[tag] = {"area_front_px": int(mf.sum()), "area_back_px": int(mb.sum()),
                          "back_hidden_frac": float(hidden), "front_outside_frac": float(front_out),
                          "bbox_front": bbox_of(mf), "bbox_back": bbox_of(mb)}
        if ok:
            break
    recs, imgs = [], []
    for tag, front_id, back_id, pol in (("orig", ident[0], ident[1], polys), ("twin", ident[1], ident[0], polys_sw)):
        fname = f"{front_id['color']} {front_id['shape']}"
        bname = f"{back_id['color']} {back_id['shape']}"
        x_name, y_name = (fname, bname) if ask_front else (bname, fname)
        # Twin keeps the same question text, so the answer flips.
        if tag == "twin":
            x_name, y_name = recs[0]["meta"]["x_name"], recs[0]["meta"]["y_name"]
        answer_yes = x_name == fname
        imgs.append(render({"poly": pol["front"], **front_id}, {"poly": pol["back"], **back_id}, size))
        recs.append({
            "item_id": f"occ{idx:03d}" + ("_cf" if tag == "twin" else ""),
            "task": "occ_decoupled" + ("_cf" if tag == "twin" else ""),
            "question": f"Is the {x_name} in front of the {y_name}?",
            "options": ["yes", "no"], "gold": "A" if answer_yes else "B",
            "meta": {"front_name": fname, "back_name": bname, "x_name": x_name, "y_name": y_name,
                     "front_above": front_above, "dx_sign": int(np.sign(geo["dx"])),
                     "area_ratio_front_over_back": stats[tag]["area_front_px"] / stats[tag]["area_back_px"],
                     "twin_id": f"occ{idx:03d}" + ("" if tag == "twin" else "_cf"), **stats[tag],
                     "objects": [{"name": fname, "bbox": stats[tag]["bbox_front"], "occludes": [bname]},
                                 {"name": bname, "bbox": stats[tag]["bbox_back"], "occludes": []}]},
        })
    return recs[0], recs[1], imgs[0], imgs[1]


def main() -> None:
    """CLI entry point."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", default="items/occlusion")
    ap.add_argument("--n", type=int, default=200, help="number of base items (multiple of 4)")
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--ratio-min", type=float, default=0.55, help="min front/back area ratio")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    assert args.n % 4 == 0
    rng = random.Random(args.seed)
    out = Path(args.out_dir)
    (out / "images").mkdir(parents=True, exist_ok=True)
    cells = [(fa, af) for fa in (True, False) for af in (True, False)]
    plan = [c for c in cells for _ in range(args.n // 4)]
    rng.shuffle(plan)
    base, twins = [], []
    for i, (front_above, ask_front) in enumerate(plan):
        r0, r1, im0, im1 = make_pair(i, rng, args.size, front_above, ask_front, args.ratio_min)
        for r, im, bucket in ((r0, im0, base), (r1, im1, twins)):
            p = out / "images" / f"{r['item_id']}.png"
            im.save(p)
            r["image_paths"] = [f"images/{p.name}"]
            r["image_id"] = r["item_id"].replace("_cf", "")
            bucket.append(r)
    for name, recs in (("items.jsonl", base), ("items_cf.jsonl", twins)):
        with open(out / name, "w") as f:
            for r in recs:
                f.write(json.dumps(r) + "\n")
    ratios = [r["meta"]["area_ratio_front_over_back"] for r in base + twins]
    report = {
        "n_base": len(base), "n_twins": len(twins),
        "yes_base": sum(r["gold"] == "A" for r in base), "yes_twin": sum(r["gold"] == "A" for r in twins),
        "front_above_base": sum(r["meta"]["front_above"] for r in base),
        "max_area_ratio": max(ratios), "mean_area_ratio": float(np.mean(ratios)),
        "violations_area": sum(x > 1.0 for x in ratios),
        "hidden_frac_range": [min(r["meta"]["back_hidden_frac"] for r in base + twins),
                              max(r["meta"]["back_hidden_frac"] for r in base + twins)],
        "seed": args.seed,
    }
    (out / "generation_report.json").write_text(json.dumps(report, indent=2))
    sheet = Image.new("RGB", (4 * 256, 4 * 256), (255, 255, 255))
    for k, r in enumerate(base[:8] + twins[:8]):
        im = Image.open(out / r["image_paths"][0]).resize((256, 256))
        sheet.paste(im, ((k % 4) * 256, (k // 4) * 256))
    sheet.save(out / "contact_sheet.png")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
