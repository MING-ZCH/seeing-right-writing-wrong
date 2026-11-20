"""Items file for steering: D+C- wrong-claim steering items where the layer-26 within-claim r-probe (CV) disagrees with
the label (sign from that probe is then anti-oracle). Writes /dev/shm/e17/<model>_rp26_items.json (mixed-items format).
Usage: python e17_rp26.py --model qwen3vl_8b
"""
from __future__ import annotations

import argparse
import json

import numpy as np

from e17_m1 import within
from e17_probe_analyze import load, meta_of


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    a = ap.parse_args()
    _, meta = meta_of(a.model)
    y = np.array([m["y_true"] for m in meta])
    pair = np.array([m["pair"] for m in meta])
    hc = np.array([m["has_claim"] for m in meta], bool)
    claim = np.array([int(m["named"] == m["x"]) if m["has_claim"] else -1 for m in meta])
    pr, _, _, _ = within(load(a.model, "orig", 26)[:, 4], y, claim, pair, hc)
    keys = set(json.load(open(f"/dev/shm/e17/{a.model}_steer_items.json"))["DCm_wrong"])
    out = {m["key"]: {"y": int(y[i]), "rprobe_y": int(pr[i]), "direct_answer_y": int(y[i]), "direct_probe_y": int(y[i]), "group": "D+C-"}
           for i, m in enumerate(meta) if m["key"] in keys and pr[i] != y[i]}
    json.dump(out, open(f"/dev/shm/e17/{a.model}_rp26_items.json", "w"))
    print(f"[rp26] {len(out)} items")


if __name__ == "__main__":
    main()
