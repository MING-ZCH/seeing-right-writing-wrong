"""Held-out CoT-Direct split into depth (324: BLINK Relative Depth, 3DSR closer, 3DSR in front of) and
non-depth (443: BLINK Spatial Relation, 3DSR above, higher, left) items, plus the 767 total, per model.

Uses heldout_breakdown.model_rows (frozen Direct and held-out CoT rows) and the same image-cluster bootstrap.
-> outputs/bundle/heldout_split.json
Usage: python heldout_split.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from build_bundle import ci
from heldout_breakdown import DEPTH, MODELS, model_rows

B = Path("outputs/bundle")
SUBSETS = {"all767": lambda t: True, "depth324": lambda t: t in DEPTH, "nondepth443": lambda t: t not in DEPTH,
           "left100": lambda t: t == "3dsr_orientation_on_the_left"}


def main() -> None:
    res = {}
    for m in MODELS:
        rows = model_rows(m)
        res[m] = {}
        for s, f in SUBSETS.items():
            sub = [r for r in rows if f(r["task"])]
            g = [r["img"] for r in sub]
            d = np.array([float(r["pd"] == r["gold"]) for r in sub])
            c = np.array([float(r["pc"] == r["gold"]) for r in sub])
            res[m][s] = {"n": len(sub), "direct": ci(d, g), "cot": ci(c, g), "cot_minus_direct": ci(c - d, g)}
            x = res[m][s]["cot_minus_direct"]
            print(f"{m:14s} {s:12s} n={len(sub):3d} D={100 * d.mean():.1f} C={100 * c.mean():.1f} "
                  f"C-D={x['mean']:+.1f} [{x['lo']:+.1f},{x['hi']:+.1f}]")
    json.dump(res, open(B / "heldout_split.json", "w"), indent=1)


if __name__ == "__main__":
    main()
