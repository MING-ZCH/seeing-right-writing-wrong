"""Readers for the per-item output records (readout predictions, option probabilities, main-pool readouts).

Readouts: exact full-vocabulary log-probabilities (``lp_sum``, vLLM V0 engine) when present, else the top-20
log-probabilities (``top20``; Molmo, Qwen3-VL and InternVL3.5 on the V1 engine, whose top-20 option mass is >= 0.99).
"""
from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path

TASKS = ["cvb_relation", "cvb_depth", "cvb_distance", "syn_relpos", "syn_depth"]


def rj(p: Path) -> list[dict]:
    return [json.loads(l) for l in open(p)] if p.exists() else []


def am(lp: dict) -> str | None:
    fin = {k: v for k, v in lp.items() if v is not None and math.isfinite(v)}
    return max(fin, key=fin.get) if fin else None


def pe(r: dict) -> str | None:
    """Prediction of a readout record: exact (lp_sum) if present else top-20."""
    if "lp_sum" in r:
        return am(r["lp_sum"])
    return am(r["top20"])


def probs(lp: dict) -> dict:
    fin = {k: v for k, v in lp.items() if v is not None and math.isfinite(v)}
    if not fin:
        return {k: 1 / len(lp) for k in lp}
    m = max(fin.values())
    z = sum(math.exp(v - m) for v in fin.values())
    return {k: (math.exp(lp[k] - m) / z if k in fin else 0.0) for k in lp}


def load_main(O: Path, L: Path, m: str) -> dict:
    """{condition: {uid: (correct, pgold, image_id, task, pred)}} on the 2,850 pool."""
    out = defaultdict(dict)
    recs = rj(O / "e1x" / f"{m}.jsonl")
    if recs:
        for r in recs:
            p = pe(r)
            pg = probs(r["lp_sum"] if "lp_sum" in r else r["top20"])[r["gold"]]
            out[r["condition"]][r["item_id"]] = (float(p == r["gold"]), pg, r["image_id"], r["task"], p)
        return out
    src = L / "e1" / f"{m}.json"
    if not src.exists():
        src = O / "e4" / f"{m}.json"
    if not src.exists():
        return out
    for it in json.load(open(src))["items"]:
        if it["task"] not in TASKS:
            continue
        for c, sc in it["scores"].items():
            p = am(sc)
            c = "cot" if c == "think" else c  # Thinking model: native think trace plays the CoT role
            out[c][it["uid"]] = (float(p == it["gold"]), probs(sc)[it["gold"]], it["group"], it["task"], p)
    return out


def groups_from(main: dict, L: Path, O: Path, m: str) -> dict:
    src = L / "e1" / f"{m}.json"
    if not src.exists():
        src = O / "e4" / f"{m}.json"
    G = {}
    if not src.exists():
        return G
    for it in json.load(open(src))["items"]:
        if it["task"] not in TASKS:
            continue
        d, c = am(it["scores"]["direct"]), am(it["scores"].get("cot") or it["scores"]["think"])
        G[it["uid"]] = {"direct": d, "cot": c, "cot_free": it.get("cot_free"),
                        "group": ("D+" if d == it["gold"] else "D-") + ("C+" if c == it["gold"] else "C-")}
    return G
