"""Shared helpers for E2: D+/C- grouping from E1 outputs and counterfactual images."""

from __future__ import annotations

import random
from collections import defaultdict
from pathlib import Path
from typing import Any

from srww.common import Item, load_items, lr_option_kind, mirror_image, mirrored_gold, read_jsonl, variant_images


def e1_index(e1_path: str | Path, pred_key: str = "pred_sum") -> dict[str, dict[str, dict[str, Any]]]:
    """Index E1 records as ``{condition: {item_id: record}}``."""
    idx: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for r in read_jsonl(e1_path):
        r["_ok"] = r.get(pred_key) == r["gold"]
        r["_pred"] = r.get(pred_key)
        idx[r["condition"]][r["item_id"]] = r
    return idx


def make_groups(items: list[Item], e1: dict[str, dict[str, dict[str, Any]]], seed: int = 0,
                cot_cond: str = "cot", max_per_group: int | None = None) -> dict[str, str]:
    """Assign items to D+C-, matched D+C+, and D-C+ groups.

    D+C+ is subsampled per task to match the D+C- count of that task.

    Returns:
        ``{item_id: group}`` for selected items.
    """
    rng = random.Random(seed)
    by_task: dict[str, dict[str, list[str]]] = defaultdict(lambda: defaultdict(list))
    for it in items:
        d, c = e1["direct"].get(it.item_id), e1[cot_cond].get(it.item_id)
        if d is None or c is None:
            continue
        g = ("D+" if d["_ok"] else "D-") + ("C+" if c["_ok"] else "C-")
        by_task[it.task][g].append(it.item_id)
    out: dict[str, str] = {}
    for task, gs in by_task.items():
        n_dc = len(gs["D+C-"])
        for gid in gs["D+C-"]:
            out[gid] = "D+C-"
        for gid in rng.sample(gs["D+C+"], min(n_dc, len(gs["D+C+"]))):
            out[gid] = "D+C+"
        for gid in gs["D-C+"]:
            out[gid] = "D-C+"
    if max_per_group:
        cap: dict[str, list[str]] = defaultdict(list)
        for k, g in out.items():
            cap[g].append(k)
        keep = set()
        for g, ks in cap.items():
            keep.update(rng.sample(ks, min(max_per_group, len(ks))))
        out = {k: v for k, v in out.items() if k in keep}
    return out


def load_twins(paths: list[str] | None) -> dict[str, Item]:
    """Map base item_id -> counterfactual twin item (from ``items_cf.jsonl``)."""
    out: dict[str, Item] = {}
    for p in paths or []:
        for it in load_items(p):
            base = it.meta.get("twin_id")
            if base:
                out[base] = it
    return out


def counterfactual(item: Item, twins: dict[str, Item]) -> tuple[list[Any], str, str] | None:
    """Counterfactual images and gold for an item.

    Synthetic items use the re-rendered identity-swapped twin; CV-Bench / other
    left-right items use a horizontal mirror. Returns ``None`` when no
    answer-changing counterfactual exists (e.g. CV-Bench depth).

    Returns:
        ``(images, gold, kind)``.
    """
    if item.item_id in twins:
        tw = twins[item.item_id]
        return variant_images(tw, "original"), tw.gold, "rerender_twin"
    if lr_option_kind(item) is not None:
        return [mirror_image(im) for im in variant_images(item, "original")], mirrored_gold(item), "mirror"
    return None
