"""Exp-62 example figure: 5 natural-object occlusion twin pairs (v | v_cf), front object named under each image.

-> paper/figures/supp_e62_examples.{pdf,png}

Usage: python e62_fig.py [--pairs e62p001 e62p020 ...]
"""
from __future__ import annotations
import vtax_paths as _vtp  # noqa: E402

import argparse
import json
import random
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from PIL import Image  # noqa: E402

ST = Path(f"{_vtp.OUTPUTS}/e62/stimuli")
FIG = Path(__file__).resolve().parent.parent / f"{_vtp.PAPER}" / "figures" / "supp_e62_examples"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", nargs="*", default=None)
    a = ap.parse_args()
    items = [json.loads(l) for l in open(ST / "items_final.jsonl")]
    meta = {it["image_uid"]: it for it in items if it["phrasing"] == "closer"}
    pairs = a.pairs or sorted(random.Random(62).sample(sorted({it["pair"] for it in items}), 5))
    fig, axes = plt.subplots(2, len(pairs), figsize=(2.3 * len(pairs), 3.9))
    for j, p in enumerate(pairs):
        for i, t in enumerate("ab"):
            it = meta[p + t]
            ax = axes[i, j]
            ax.imshow(Image.open(ST / "images" / f"{p}{t}.png"))
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_xlabel(f"front: {it['front_name']}", fontsize=7.5, labelpad=2)
            if j == 0:
                ax.set_ylabel(["$v$", "$v_\\mathrm{cf}$"][i], fontsize=10)
        axes[0, j].set_title(f"{meta[p + 'a']['meta']['large']['cat']} / {meta[p + 'a']['meta']['small']['cat']}",
                             fontsize=8)
    plt.tight_layout(pad=0.3, h_pad=0.4, w_pad=0.3)
    FIG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(f"{FIG}.pdf", dpi=300)
    fig.savefig(f"{FIG}.png", dpi=200)
    print("pairs", pairs, "->", FIG)


if __name__ == "__main__":
    main()
