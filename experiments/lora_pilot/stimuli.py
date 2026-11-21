"""Exp-63 training stimuli (CPU): rendered depth twin pairs and left/right mirror pairs, seed 63.

Generator: data_gen.cue_factorial_set.make_pair (cue-factorial occlusion scenes, 512 px), per-pair seeds 63 * 100003 + i (the cue
set used seed 1, i < 400; e2_occlusion_set seed 0 with one global RNG), so no scene is re-drawn.

depth  1,512 identity-swapped twin pairs (v = scene, v_cf = twin, same question, gold flips). Balanced over
       front above/below x front larger/smaller x asked object front/back x phrasing (63 per cell):
         front   "Is the X in front of the Y?\\n(A) yes\\n(B) no"
         behind  "Is the X behind the Y?\\n(A) yes\\n(B) no"
         which   "Which object is closer to the camera?\\n(A) the X\\n(B) the Y"
lr     1,008 horizontal-mirror pairs (v = scene, v_cf = ImageOps.mirror(v), same question, gold flips); only scenes
       whose object centres (bbox centres of the full shapes AND visible-pixel centroids) differ by >= 40 px
       horizontally. Balanced over phrasing x X-left-in-v x X-is-front-object (84 per cell):
         left    "Is the X to the left of the Y?\\n(A) yes\\n(B) no"
         right   "Is the X to the right of the Y?\\n(A) yes\\n(B) no"
         where   CV-Bench 2D template without boxes, "(A) left\\n(B) right"
Outputs ({W}/outputs/e63/train_stimuli/): images/*.png, items.jsonl (one row per image x question, protocol schema:
question with option lines, choices, gold, image_rel, pair / group, twin_item), manifest.json (counts, balance, per-image
file and pixel sha256, overlap check vs every evaluation image set), contact_depth.png / contact_lr.png.

Usage: python -m experiments.lora_pilot.stimuli build [--n-depth 1512 --n-lr 1008]
       python -m experiments.lora_pilot.stimuli check          (pixel-hash overlap + near-duplicate check; updates manifest.json)
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import random
import time
from collections import Counter
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageOps

from experiments.lora_pilot import lora_common as E
from data_gen.occlusion_twins import COLORS
from data_gen.cue_factorial_set import make_pair

SIZE = 512
MIN_DX = 40
LR_TEMPLATE = ("Considering the relative positions of the {x} and the {y} in the image provided, where is the {x} "
               "located with respect to the {y}? Select from the following choices.\n(A) left\n(B) right")


def pixel_sha(img: Image.Image) -> str:
    """sha256 of mode, size and raw RGB bytes (format-independent image identity)."""
    im = img.convert("RGB")
    return hashlib.sha256(f"RGB{im.size}".encode() + im.tobytes()).hexdigest()


def file_sha(p: str) -> str:
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def centres(img: Image.Image, rec: dict) -> dict:
    """x of bbox centres (full shapes) and visible-pixel centroids of the front / back objects."""
    m = rec["meta"]
    a = np.asarray(img)
    out = {}
    for role, name, bb in (("front", m["front_name"], m["bbox_front"]), ("back", m["back_name"], m["bbox_back"])):
        col = np.array(COLORS[name.split()[0]])
        ys, xs = np.nonzero((a == col).all(-1))
        out[role] = {"bbox_cx": (bb[0] + bb[2]) / 2, "vis_cx": float(xs.mean()), "vis_cy": float(ys.mean()), "vis_px": int(len(xs))}
    return out


# ----------------------------------------------------------------------------- jobs

def _depth_job(job):
    """One depth twin pair: rows (images saved by the worker)."""
    i, seed, fa, fl, af, phr = job
    r0, r1, im0, im1 = make_pair(i, random.Random(seed), SIZE, fa, fl, af)
    rows = depth_items(i, phr, r0, r1)
    if phr == "front":
        assert [r["gold"] for r in rows] == [r0["gold"], r1["gold"]]
    for r, im in zip(rows, (im0, im1)):
        im.save(f"{E.STIM}/{r['image_rel']}")
    return rows, 1


def _lr_job(job):
    """One left/right pair; rejection-samples scenes until both centre definitions differ by >= MIN_DX px."""
    i, seed, fa, fl, phr, x_left, x_front = job
    rng = random.Random(seed)
    tries = 0
    while True:
        tries += 1
        r0, _, im0, _ = make_pair(i, rng, SIZE, fa, fl, True)
        c = centres(im0, r0)
        dxb = c["front"]["bbox_cx"] - c["back"]["bbox_cx"]
        dxv = c["front"]["vis_cx"] - c["back"]["vis_cx"]
        if abs(dxb) >= MIN_DX and abs(dxv) >= MIN_DX and np.sign(dxb) == np.sign(dxv):
            break
    rows, imgs = lr_items(i, phr, x_left, x_front, r0, im0, c)
    for r in rows:
        imgs[r["img_role"]].save(f"{E.STIM}/{r['image_rel']}")
    return rows, tries


# ----------------------------------------------------------------------------- build

def depth_items(i: int, phr: str, r0: dict, r1: dict) -> list[dict]:
    pid = f"e63d{i:04d}"
    m = r0["meta"]
    x, y = m["x_name"], m["y_name"]
    rows = []
    for role, r in (("v", r0), ("cf", r1)):
        front = r["meta"]["front_name"]
        if phr == "front":
            q, g = f"Is the {x} in front of the {y}?\n(A) yes\n(B) no", "A" if x == front else "B"
        elif phr == "behind":
            q, g = f"Is the {x} behind the {y}?\n(A) yes\n(B) no", "A" if y == front else "B"
        else:
            q, g = f"Which object is closer to the camera?\n(A) the {x}\n(B) the {y}", "A" if x == front else "B"
        img_uid = f"{pid}{role}"
        rows.append({"uid": f"{img_uid}_{phr}", "image_uid": img_uid, "pair": pid, "group": pid, "kind": "depth",
                     "phrasing": phr, "img_role": role, "dataset": "e63train", "task": "e63_depth", "question": q,
                     "choices": ["A", "B"], "gold": g, "image_rel": f"images/{img_uid}.png",
                     "meta": {**{k: v for k, v in r["meta"].items()}, "src_item": r["item_id"]}})
    rows[0]["twin_item"], rows[1]["twin_item"] = rows[1]["uid"], rows[0]["uid"]
    return rows


def lr_items(i: int, phr: str, x_left: bool, x_front: bool, r0: dict, im0: Image.Image, c: dict):
    """Rows and images (v, mirror) of one left/right pair; v is the render or its mirror so that X-left-in-v = x_left."""
    pid = f"e63h{i:04d}"
    m = r0["meta"]
    x, y = (m["front_name"], m["back_name"]) if x_front else (m["back_name"], m["front_name"])
    xr, yr = ("front", "back") if x_front else ("back", "front")
    x_left_render = c[xr]["vis_cx"] < c[yr]["vis_cx"]
    flipped = x_left_render != x_left
    v = ImageOps.mirror(im0) if flipped else im0
    imgs = {"v": v, "cf": ImageOps.mirror(v)}
    rows = []
    for role in ("v", "cf"):
        xl = x_left if role == "v" else not x_left
        if phr == "left":
            q, g = f"Is the {x} to the left of the {y}?\n(A) yes\n(B) no", "A" if xl else "B"
        elif phr == "right":
            q, g = f"Is the {x} to the right of the {y}?\n(A) yes\n(B) no", "B" if xl else "A"
        else:
            q, g = LR_TEMPLATE.format(x=x, y=y), "A" if xl else "B"
        img_uid = f"{pid}{role}"
        mirrored = flipped != (role == "cf")
        rows.append({"uid": f"{img_uid}_{phr}", "image_uid": img_uid, "pair": pid, "group": pid, "kind": "lr",
                     "phrasing": phr, "img_role": role, "dataset": "e63train", "task": "e63_lr", "question": q,
                     "choices": ["A", "B"], "gold": g, "image_rel": f"images/{img_uid}.png",
                     "meta": {"x_name": x, "y_name": y, "x_is_front": x_front, "x_left": xl, "render_mirrored": mirrored,
                              "front_name": m["front_name"], "back_name": m["back_name"], "cell": m["cell"],
                              "dx_vis_abs": abs(c["front"]["vis_cx"] - c["back"]["vis_cx"]),
                              "dx_bbox_abs": abs(c["front"]["bbox_cx"] - c["back"]["bbox_cx"]),
                              "back_hidden_frac": m["back_hidden_frac"], "src_item": r0["item_id"]}})
    rows[0]["twin_item"], rows[1]["twin_item"] = rows[1]["uid"], rows[0]["uid"]
    return rows, imgs


def contact(rows: list[dict], out: Path, n: int = 8, seed: int = E.SEED) -> None:
    """v | v_cf tiles with question and gold for n random pairs."""
    by = {}
    for r in rows:
        by.setdefault(r["pair"], {})[r["img_role"]] = r
    pick = random.Random(seed).sample(sorted(by), n)
    sheet = Image.new("RGB", (2 * 520, n * 300), "white")
    d = ImageDraw.Draw(sheet)
    for k, p in enumerate(pick):
        for j, role in enumerate(("v", "cf")):
            r = by[p][role]
            im = Image.open(f"{E.STIM}/{r['image_rel']}").resize((256, 256))
            sheet.paste(im, (j * 520, k * 300))
            d.text((j * 520 + 262, k * 300 + 5), f"{r['uid']}\n" + r["question"].replace("? ", "?\n") + f"\ngold {r['gold']}",
                   fill="black")
    sheet.save(out)


def build(n_depth: int, n_lr: int) -> None:
    t0 = time.time()
    out = Path(E.STIM)
    (out / "images").mkdir(parents=True, exist_ok=True)
    rng = random.Random(E.SEED)
    dcells = [(fa, fl, af, ph) for fa in (True, False) for fl in (True, False) for af in (True, False)
              for ph in ("front", "behind", "which")]
    assert n_depth % len(dcells) == 0
    dplan = [c for c in dcells for _ in range(n_depth // len(dcells))]
    rng.shuffle(dplan)
    djobs = [(i, E.SEED * 100003 + i, *c) for i, c in enumerate(dplan)]
    lcells = [(ph, xl, xf) for ph in ("left", "right", "where") for xl in (True, False) for xf in (True, False)]
    assert n_lr % len(lcells) == 0
    lplan = [c for c in lcells for _ in range(n_lr // len(lcells))]
    rng.shuffle(lplan)
    scene = [(fa, fl) for fa in (True, False) for fl in (True, False)]
    ljobs = [(i, E.SEED * 100003 + 50000 + i, *scene[i % 4], *c) for i, c in enumerate(lplan)]
    with Pool(min(32, os.cpu_count() or 8)) as pool:
        dmade = pool.map(_depth_job, djobs)
        lmade = pool.map(_lr_job, ljobs)
    rows, img_hash = [r for rs, _ in dmade for r in rs], {}
    rows += [r for rs, _ in lmade for r in rs]
    tries = [t for _, t in lmade]
    for r in rows:
        p = out / r["image_rel"]
        img_hash[r["image_uid"]] = {"file_sha256": file_sha(str(p)), "pixel_sha256": pixel_sha(Image.open(p))}
    E.vx.write_jsonl(f"{out}/items.jsonl", rows)
    rep: dict = {"n_pairs": {"depth": n_depth, "lr": n_lr}, "n_items": len(rows), "n_images": len(img_hash),
                 "seed": E.SEED, "pair_seeds": "depth: 63*100003+i; lr: 63*100003+50000+i (i = pair index)",
                 "size": SIZE, "min_dx_px": MIN_DX, "lr_scene_tries_mean": float(np.mean(tries))}
    for kind in ("depth", "lr"):
        ks = [r for r in rows if r["kind"] == kind]
        rep[kind] = {"phrasing": dict(Counter(r["phrasing"] for r in ks if r["img_role"] == "v")),
                     "gold_A_v": dict(Counter(r["phrasing"] for r in ks if r["img_role"] == "v" and r["gold"] == "A")),
                     "gold_A_cf": dict(Counter(r["phrasing"] for r in ks if r["img_role"] == "cf" and r["gold"] == "A")),
                     "gold_flips_in_pair": all(a["gold"] != b["gold"] for a, b in zip(ks[0::2], ks[1::2])),
                     "same_question_in_pair": all(a["question"] == b["question"] for a, b in zip(ks[0::2], ks[1::2]))}
        if kind == "depth":
            rep[kind]["cells"] = dict(Counter(r["meta"]["cell"] for r in ks if r["img_role"] == "v"))
            rep[kind]["hidden_frac_range"] = [round(min(r["meta"]["back_hidden_frac"] for r in ks), 3),
                                              round(max(r["meta"]["back_hidden_frac"] for r in ks), 3)]
        else:
            dv = [r["meta"]["dx_vis_abs"] for r in ks]
            rep[kind]["dx_vis_abs_px"] = {"min": round(min(dv), 1), "median": round(float(np.median(dv)), 1), "max": round(max(dv), 1)}
            rep[kind]["x_is_front"] = sum(r["meta"]["x_is_front"] for r in ks if r["img_role"] == "v")
            rep[kind]["render_mirrored_v"] = sum(r["meta"]["render_mirrored"] for r in ks if r["img_role"] == "v")
    dup = Counter(h["pixel_sha256"] for h in img_hash.values())
    rep["within_train_duplicate_images"] = sum(c - 1 for c in dup.values() if c > 1)
    contact([r for r in rows if r["kind"] == "depth"], out / "contact_depth.png")
    contact([r for r in rows if r["kind"] == "lr"], out / "contact_lr.png")
    man = {"created": time.strftime("%Y-%m-%d %H:%M:%S"), "generator": "experiments/lora_pilot/stimuli.py (data_gen.cue_factorial_set.make_pair)",
           "items_sha256": file_sha(f"{out}/items.jsonl"), "report": rep, "images": img_hash}
    json.dump(man, open(out / "manifest.json", "w"), indent=1)
    print(json.dumps(rep, indent=1))
    print(f"[e63 stimuli] {len(rows)} items, {len(img_hash)} images ({time.time() - t0:.0f}s)")


# ----------------------------------------------------------------------------- overlap check

def eval_images():
    """(set name, image id, PIL image) for every evaluation image set (originals; mirrors are added by the caller)."""
    W, SH = E.vx.W, E.vx.SHARE
    dirs = {"cue_factorial(e50/occ_cues)": f"{W}/outputs/e50/occ_cues/images",
            "cue_factorial(data/items/occ_cues)": f"{W}/data/items/occ_cues/images",
            "cue_factorial(e28/cue_items)": f"{W}/outputs/e28/cue_items/images",
            "occlusion_twins": f"{W}/data/items/occlusion/images",
            "hard_set": f"{W}/data/items/occlusion_hard/images",
            "hard_set_pilot": f"{W}/data/items/occlusion_hard_pilot/images",
            "synthetic(spatial_probing incl. Synthetic Depth)": f"{SH}/datasets/spatial_probing/images",
            "synthetic_real(spatial_probing/real_images)": f"{SH}/datasets/spatial_probing/real_images",
            "coco_twins(e62)": f"{W}/outputs/e62/stimuli/images"}
    for name, d in dirs.items():
        if not os.path.isdir(d):
            print(f"[check] missing {d}")
            continue
        fs = [f for f in sorted(os.listdir(d)) if f.lower().endswith((".png", ".jpg", ".jpeg"))]
        E.prefetch([f"{d}/{f}" for f in fs], workers=48, as_bytes=True)
        print(f"[check] {name}: {len(fs)} files read", flush=True)
        for f in fs:
            yield name, f, E.load_image(f"{d}/{f}")
    for it in E.C.load_cvbench():
        yield "cv_bench", it["uid"], it["image"]
    from experiments.behavior.heldout_benchmarks import load_heldout
    for it in load_heldout():
        yield "heldout(BLINK+3DSR)", it["uid"], it["image"]


def thumb(img: Image.Image) -> np.ndarray:
    return np.asarray(img.convert("RGB").resize((32, 32), Image.BILINEAR), dtype=np.float32).ravel() / 255.0


def check() -> None:
    t0 = time.time()
    man = json.load(open(f"{E.STIM}/manifest.json"))
    train = {h["pixel_sha256"]: u for u, h in man["images"].items()}
    rows = E.vx.read_jsonl(f"{E.STIM}/items.jsonl")
    tr_uids = sorted({r["image_uid"] for r in rows})
    E.prefetch([f"{E.STIM}/images/{u}.png" for u in tr_uids], workers=48, as_bytes=True)
    tr_th = np.stack([thumb(E.load_image(f"{E.STIM}/images/{u}.png")) for u in tr_uids])
    tr_pix = {pixel_sha(E.load_image(f"{E.STIM}/images/{u}.png")): u for u in tr_uids}
    assert tr_pix == train, "manifest pixel hashes differ from the images on disk"
    print(f"[check] {len(tr_uids)} training images re-hashed = manifest", flush=True)
    per_set: dict = {}
    hits = []
    ev_th, ev_names = [], []
    for name, iid, img in eval_images():
        s = per_set.setdefault(name, {"n_images": 0, "n_hashes_incl_mirror": 0, "overlap": 0})
        s["n_images"] += 1
        rendered = not name.startswith(("cv_bench", "heldout", "coco", "synthetic_real"))
        for tag, im in (("orig", img), ("mirror", ImageOps.mirror(img))):
            s["n_hashes_incl_mirror"] += 1
            h = pixel_sha(im)
            if h in train:
                s["overlap"] += 1
                hits.append({"set": name, "image": iid, "variant": tag, "train_image": train[h]})
            if rendered and im.size == (SIZE, SIZE):
                ev_th.append(thumb(im))
                ev_names.append(f"{name}:{iid}:{tag}")
    ev = np.stack(ev_th)
    # nearest-neighbour L2 distance (32x32 RGB thumbnails) train -> rendered eval, and train -> train (reference)
    def nn(a: np.ndarray, b: np.ndarray, same: bool = False) -> np.ndarray:
        d = (a ** 2).sum(1)[:, None] + (b ** 2).sum(1)[None, :] - 2 * a @ b.T
        if same:
            np.fill_diagonal(d, np.inf)
        return np.sqrt(np.maximum(d.min(1), 0))
    d_ev = nn(tr_th, ev)
    d_tr = nn(tr_th, tr_th, same=True)
    # exclude each image's own pair partner (twin / mirror) from the train->train reference
    near = {"train_to_rendered_eval_nn_l2": {"min": round(float(d_ev.min()), 3), "p1": round(float(np.percentile(d_ev, 1)), 3),
                                             "median": round(float(np.median(d_ev)), 3)},
            "train_to_train_nn_l2 (incl. own twin/mirror)": {"min": round(float(d_tr.min()), 3),
                                                              "median": round(float(np.median(d_tr)), 3)},
            "n_rendered_eval_thumbs": len(ev_names), "thumb": "32x32 RGB bilinear, values in [0,1], L2 over 3072 dims",
            "closest_pair": [tr_uids[int(d_ev.argmin())], ev_names[int(np.argmin(((tr_th[int(d_ev.argmin())] - ev) ** 2).sum(1)))]]}
    man["overlap_check"] = {"checked": time.strftime("%Y-%m-%d %H:%M:%S"), "method": "pixel sha256 (RGB bytes + size) of "
                            "every evaluation image and its horizontal mirror vs every training image",
                            "per_set": per_set, "total_overlap": len(hits), "hits": hits, "near_duplicate": near}
    json.dump(man, open(f"{E.STIM}/manifest.json", "w"), indent=1)
    print(json.dumps(man["overlap_check"], indent=1))
    print(f"[e63 check] {sum(s['n_images'] for s in per_set.values())} eval images, overlap {len(hits)} ({time.time() - t0:.0f}s)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("phase", choices=["build", "check"])
    ap.add_argument("--n-depth", type=int, default=1512)
    ap.add_argument("--n-lr", type=int, default=1008)
    a = ap.parse_args()
    build(a.n_depth, a.n_lr) if a.phase == "build" else check()


if __name__ == "__main__":
    main()
