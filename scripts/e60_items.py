"""E60 items: natural left/right items whose answer flips under a horizontal mirror (CPU, dev machine).

Phases:
  build     candidates (CV-Bench 2D Relation {left,right}; BLINK Spatial Relation left/right yes/no) with mirrored gold,
            queried object names X / Y and the frame-of-reference / BLINK filters -> outputs/e60/candidates.jsonl
  finalize  merge the two vision text judges (e60_judge.py text) -> outputs/e60/items.jsonl (keep flag + reason)
  spot      render 20 random kept items (seed 60) as original | mirrored with gold / mirrored gold -> outputs/e60/spot*.png

Usage: SHARE=<DATA_ROOT> python e60_items.py build
"""
from __future__ import annotations
import vtax_paths as _vtp  # noqa: E402

import argparse
import io
import json
import os
import random
import re

import vx
from vx import C

E60 = f"{vx.OUT}/e60"
JUDGES = ("claude-opus-5-5", "gpt-6-astra")
FRAME_RX = re.compile(r"\b(his|her|its|their|my|your)\s+(left|right)\b|perspective|facing|point of view", re.I)
CVB_RX = re.compile(r"where is the (.+?) located with respect to the (.+?)\?")
BLINK_RX = re.compile(r"^Is the (.+?) (?:at the |on the )?(left|right)(?: side)? of the (.+?)\?")
# Gold "yes" BLINK items whose image-frame relation agrees with the question (eye check before any E60 output).
BLINK_KEEP = {f"val_Spatial_Relation_{i}" for i in (3, 9, 32, 39, 40, 61, 88, 93)}


def _strip_box(s: str) -> str:
    """Object name without the '(annotated by the red box)' suffix."""
    return re.sub(r"\s*\(annotated by the \w+ box\)", "", s).strip()


def load_blink() -> list[dict]:
    """BLINK Spatial Relation val items (original item format, as e6x.load_heldout)."""
    import pandas as pd
    from PIL import Image
    d = pd.read_parquet(f"{vx.SHARE}/datasets/heldout/BLINK/Spatial_Relation/val-00000-of-00001.parquet")
    return [{"uid": r["idx"], "dataset": "blink", "task": "blink_relation", "question": r["prompt"],
             "choices": C.letters(len(list(r["choices"]))), "gold": r["answer"].strip("() "),
             "image": Image.open(io.BytesIO(r["image_1"]["bytes"])).convert("RGB"), "group": r["idx"]}
            for _, r in d.iterrows()]


def candidates(with_images: bool = False) -> list[dict]:
    """All left/right candidates with mirrored gold and filter fields (images attached if requested)."""
    out = []
    for it in vx.load_items(["cvb_relation"]):
        ow = {k: v.lower() for k, v in vx.option_words(it).items()}
        if sorted(ow.values()) != ["left", "right"]:
            continue
        q = it["question"].split("\n")[0]
        m = CVB_RX.search(q)
        target = {"left": "right", "right": "left"}[ow[it["gold"]]]
        rec = {"uid": it["uid"], "task": it["task"], "set": "cvb", "question": it["question"], "choices": it["choices"],
               "gold": it["gold"], "gold_mirror": [k for k, v in ow.items() if v == target][0], "group": it["group"],
               "x": _strip_box(m.group(1)) if m else None, "y": _strip_box(m.group(2)) if m else None,
               "frame_flag": bool(FRAME_RX.search(q)), "blink_keep": None}
        if with_images:
            rec["image"] = C.get_image(it)
        out.append(rec)
    for it in load_blink():
        q = it["question"].split("\n")[0]
        if not re.search(r"\b(left|right)\b", q.lower()):
            continue
        m = BLINK_RX.search(q)
        rec = {"uid": it["uid"], "task": it["task"], "set": "blink", "question": it["question"], "choices": it["choices"],
               "gold": it["gold"], "gold_mirror": "B" if it["gold"] == "A" else "A", "group": it["group"],
               "x": m.group(1) if m else None, "y": m.group(3) if m else None,
               "frame_flag": bool(FRAME_RX.search(q)), "blink_keep": it["uid"] in BLINK_KEEP}
        if with_images:
            rec["image"] = it["image"]
        out.append(rec)
    return out


M7 = ["qwen3b", "qwen7b", "internvl3_8b", "phi35v", "molmo7b", "qwen3vl_8b", "internvl35_8b"]


def stored_r(model: str) -> dict[str, str]:
    """The model's own stored greedy CoT R per uid: main pool (legacy e1 / e4) and BLINK (e6x rest, condition cot)."""
    out = {u: it["cot"] for u, it in vx.legacy_e1(model).items() if u.startswith("cvb_Relation")}
    for r in vx.read_jsonl(f"{vx.OUT}/e6x/{model}_rest.jsonl"):
        if r["condition"] == "cot" and r["task"] == "blink_relation":
            out[r["item_id"]] = r["text"]
    return out


def r_cut(r: str) -> str:
    """The paper's R_cut (legacy strip_answer)."""
    import sys
    sys.path.insert(0, f"{_vtp.LEGACY_SCRIPTS}")
    from e5_content_edit import strip_answer
    return strip_answer(r)


def finalize() -> None:
    """Merge the text judges into the keep decision."""
    cands = vx.read_jsonl(f"{E60}/candidates.jsonl")
    lab = {j: {r["uid"]: r["label"] for r in vx.read_jsonl(f"{E60}/judge_text_{j}.jsonl")} for j in JUDGES}
    out = []
    for c in cands:
        labs = {j: lab[j].get(c["uid"]) for j in JUDGES}
        reasons = []
        if any(v != "NO" for v in labs.values()):
            reasons.append("text")
        if c["frame_flag"]:
            reasons.append("frame")
        if c["set"] == "blink" and not c["blink_keep"]:
            reasons.append("blink_no_flip" if c["gold"] == "B" else "blink_frame")
        if not c["x"] or not c["y"]:
            reasons.append("unparsed_objects")
        out.append({**c, "text_labels": labs, "keep": not reasons, "exclude": reasons})
    vx.write_jsonl(f"{E60}/items.jsonl", out)
    for s in ("cvb", "blink"):
        xs = [o for o in out if o["set"] == s]
        from collections import Counter
        print(s, len(xs), "kept", sum(o["keep"] for o in xs), Counter(r for o in xs for r in o["exclude"]))


def spot(n: int = 20) -> None:
    """Contact sheets of original | mirrored for n random kept CV-Bench items (gold / mirrored gold printed)."""
    from PIL import Image, ImageDraw, ImageOps
    keep = {r["uid"]: r for r in vx.read_jsonl(f"{E60}/items.jsonl") if r["keep"] and r["set"] == "cvb"}
    pick = sorted(random.Random(60).sample(sorted(keep), n))
    imgs = {c["uid"]: c["image"] for c in candidates(with_images=True) if c["uid"] in pick}
    tiles = []
    for u in pick:
        r = keep[u]
        ow = vx.option_words(r)
        a = imgs[u].copy()
        a.thumbnail((300, 300))
        b = ImageOps.mirror(a)
        t = Image.new("RGB", (620, 360), "white")
        t.paste(a, (5, 5))
        t.paste(b, (315, 5))
        d = ImageDraw.Draw(t)
        d.text((5, 310), f"{u}: X={r['x']} | Y={r['y']}", fill="black")
        d.text((5, 325), f"orig gold {r['gold']}={ow[r['gold']]}   mirror gold {r['gold_mirror']}={ow[r['gold_mirror']]}",
               fill="black")
        tiles.append(t)
    for k in range(0, len(tiles), 6):
        sheet = Image.new("RGB", (620 * 2, 360 * 3), "white")
        for j, t in enumerate(tiles[k:k + 6]):
            sheet.paste(t, ((j % 2) * 620, (j // 2) * 360))
        sheet.save(f"{E60}/spot{k // 6}.png")
    json.dump(pick, open(f"{E60}/spot_ids.json", "w"))
    print("spot items:", pick)


def main() -> None:
    """CLI entry point."""
    ap = argparse.ArgumentParser()
    ap.add_argument("phase", choices=["build", "finalize", "spot"])
    a = ap.parse_args()
    os.makedirs(E60, exist_ok=True)
    if a.phase == "build":
        cs = candidates()
        vx.write_jsonl(f"{E60}/candidates.jsonl", cs)
        print(len(cs), "candidates;", sum(c["set"] == "cvb" for c in cs), "cvb;",
              sum(c["frame_flag"] for c in cs), "frame flags;", sum(not c["x"] for c in cs), "unparsed objects")
    elif a.phase == "finalize":
        finalize()
    else:
        spot()


if __name__ == "__main__":
    main()
