"""Exp-17e analyses (round-3 reviews): Fig.3 S split by R_cut source cell, own-rationale image swap on twins (E14),
unconditional claim-edit effect over all four Direct x CoT cells (E2f same-turn, D+ and D- runs).

Usage: python rigor_e17e.py   (writes outputs/bundle/{swap_split,imgswap,claim_edit_uncond}.json and a .txt summary)
"""
from __future__ import annotations
import vtax_paths as _vtp  # noqa: E402

import json
import math
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import norm

from build_bundle import ci, groups_from, pe, rj

O = Path(f"{_vtp.OUTPUTS}")
L = Path(f"{_vtp.LEGACY_RESULTS}")
B = Path("outputs/bundle")
M7 = ["qwen3b", "qwen7b", "internvl3_8b", "phi35v", "molmo7b", "qwen3vl_8b", "internvl35_8b"]
CELLS = ["D+C+", "D+C-", "D-C+", "D-C-"]


def fmt(d: dict) -> str:
    return f"{d['mean']:+.1f} [{d['lo']:+.1f},{d['hi']:+.1f}] (n={d['n']})" if d and "mean" in d else "--"


# ----------------------------------------------------------------------------- Fig.3 S by source cell

def swap_split(lines: list[str]) -> dict:
    res = {}
    for m in M7:
        recs = rj(O / "e3x" / f"{m}.jsonl")
        if not recs:
            continue
        G = groups_from({}, L, O, m)
        idx, meta = defaultdict(dict), {}
        for r in recs:
            idx[(r["item_id"], r["image"], r["context"])][r["Y"]] = r
            meta[r["item_id"]] = r
        out = {}
        for ctx in ("rcut", "concl"):
            for img in ("original", "gray", "mirror"):
                for cell in CELLS + ["all"]:
                    vals, fg_, fw_, g = [], [], [], []
                    for uid, r0 in meta.items():
                        if cell != "all" and G.get(uid, {}).get("group") != cell:
                            continue
                        d = idx.get((uid, img, ctx))
                        gold = r0["gold"]
                        w = [c for c in r0["letters"] if c != gold][0]
                        if not d or gold not in d or w not in d:
                            continue
                        fg, fw = float(pe(d[gold]) == gold), float(pe(d[w]) == w)
                        vals.append(fg - fw); fg_.append(fg); fw_.append(fw); g.append(r0["image_id"])
                    if len(vals) >= 10:
                        out[f"{ctx}:{img}:{cell}"] = {"S": ci(vals, g), "F_gold": ci(fg_, g), "F_wrong": ci(fw_, g)}
        res[m] = out
        lines.append(f"SWAP {m:14s} rcut S orig: " + " | ".join(f"{c} {fmt(out.get(f'rcut:original:{c}', {}).get('S'))}" for c in CELLS) +
                     " || mirror: " + " | ".join(f"{c} {fmt(out.get(f'rcut:mirror:{c}', {}).get('S'))}" for c in ("D+C+", "D+C-")))
    return res


# ----------------------------------------------------------------------------- E14 own-R image swap

def imgswap(lines: list[str]) -> dict:
    res = {}
    for m in M7:
        recs = rj(O / "e14" / f"{m}.jsonl")
        if not recs:
            continue
        by = {(r["item_id"], r["ctx"], r["img"]): r for r in recs}
        out = {}
        for ctx in ("full", "cut"):
            for sub in ("wrongR", "rightR", "all"):
                dv, own, tw, flip, g = [], [], [], [], []
                for (k, c, im), r in by.items():
                    if c != ctx or im != "own" or (k, ctx, "twin") not in by:
                        continue
                    cot_ok = COT_OK.get(m, {}).get(k)
                    if sub == "wrongR" and cot_ok is not False:
                        continue
                    if sub == "rightR" and cot_ok is not True:
                        continue
                    a, b = pe(r), pe(by[(k, ctx, "twin")])
                    own.append(float(a == r["gold"])); tw.append(float(b == r["gold"]))
                    dv.append(own[-1] - tw[-1]); flip.append(float(a != b)); g.append(r["image_id"])
                if len(dv) >= 10:
                    out[f"{ctx}:{sub}"] = {"acc_own": ci(own, g), "acc_goldv_under_twin": ci(tw, g), "dvis": ci(dv, g),
                                           "flip": ci(flip, g)}
        res[m] = out
        lines.append(f"IMGSWAP {m:14s} " + " | ".join(f"{k}: own {v['acc_own']['mean']:.1f} twin->gold_v {v['acc_goldv_under_twin']['mean']:.1f} dvis {fmt(v['dvis'])}"
                                                      for k, v in out.items()))
    return res


COT_OK: dict = {}


def load_cot_ok() -> None:
    """Per twin record: is R's first two-object claim right (True) / wrong (False) w.r.t. its own image (None: no claim).

    Selection uses R's content, not the readout, so the own-image accuracy is not fixed by construction.
    """
    from e2f_claim_edit import first_claim, split_sents
    for m in M7:
        p = O / "e10" / f"{m}.jsonl"
        if not p.exists():
            continue
        d = {}
        for r in rj(p):
            if r["condition"] != "cot" or not r["task"].startswith("occ"):
                continue
            fc = first_claim(split_sents(r["text"][len("Let's think step by step."):]))
            d[r["item_id"]] = None if not fc else fc[1] == r["meta"]["front_name"].lower()
        COT_OK[m] = d


# ----------------------------------------------------------------------------- unconditional claim edit

def sdt(pred: list[str], gold: list[str]) -> float:
    p, g = np.array([x == "A" for x in pred]), np.array([x == "A" for x in gold])
    h = (p[g].sum() + .5) / (g.sum() + 1)
    f = (p[~g].sum() + .5) / ((~g).sum() + 1)
    return float(norm.ppf(h) - norm.ppf(f))


def claim_uncond(lines: list[str]) -> dict:
    res = {}
    for m in M7:
        recs = rj(O / "e2f" / f"{m}_same.jsonl") + rj(O / "e2f" / f"{m}_same_dminus.jsonl")
        if not recs:
            continue
        have_dm = (O / "e2f" / f"{m}_same_dminus.jsonl").exists()
        out = {}
        for st in ("syn", "occ", "all"):
            items = defaultdict(dict)
            for r in recs:
                if st == "all" or r["set"] == st:
                    items[(r["set"], r["item_id"])][r["cond"]] = r
            keys = [k for k, v in items.items() if all(c in v for c in ("orig", "correct", "neutral", "wrong"))]
            if len(keys) < 20:
                continue
            other = lambda y: "B" if y == "A" else "A"
            gold = [items[k]["orig"]["gold"] for k in keys]
            grp = [items[k]["orig"]["group"] for k in keys]
            pred = {"direct": [y if gp.startswith("D+") else other(y) for y, gp in zip(gold, grp)],
                    "cot": [y if gp.endswith("C+") else other(y) for y, gp in zip(gold, grp)]}
            for c in ("orig", "correct", "neutral", "wrong"):
                pred[c] = [pe(items[k][c]) for k in keys]
            g = [items[k]["orig"]["image_id"] for k in keys]
            row = {"n": len(keys), "cells": {c: grp.count(c) for c in CELLS}}
            for c, p in pred.items():
                acc = [float(a == y) for a, y in zip(p, gold)]
                row[c] = {"acc": ci(acc, g), "dprime": round(sdt(p, gold), 2)}
            for c in ("correct", "neutral", "orig"):
                for ref in ("direct", "cot"):
                    row[f"{c}-{ref}"] = ci([float(a == y) - float(b == y) for a, b, y in zip(pred[c], pred[ref], gold)], g)
            out[st] = row
        out["complete_cells"] = have_dm
        res[m] = out
        if "all" in out:
            r = out["all"]
            lines.append(f"CLAIM_UNCOND {m:14s} n={r['n']} cells={r['cells']} acc D {r['direct']['acc']['mean']:.1f} CoT {r['cot']['acc']['mean']:.1f} "
                         f"orig {r['orig']['acc']['mean']:.1f} neutral {r['neutral']['acc']['mean']:.1f} correct {r['correct']['acc']['mean']:.1f} | "
                         f"d' D {r['direct']['dprime']:.2f} CoT {r['cot']['dprime']:.2f} neutral {r['neutral']['dprime']:.2f} correct {r['correct']['dprime']:.2f} | "
                         f"neutral-D {fmt(r['neutral-direct'])} correct-D {fmt(r['correct-direct'])} {'(D+ cells only)' if not have_dm else ''}")
    return res


def main() -> None:
    lines: list[str] = []
    load_cot_ok()
    json.dump(swap_split(lines), open(B / "swap_split.json", "w"), indent=1)
    json.dump(imgswap(lines), open(B / "imgswap.json", "w"), indent=1)
    json.dump(claim_uncond(lines), open(B / "claim_edit_uncond.json", "w"), indent=1)
    (B / "rigor_e17e.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
