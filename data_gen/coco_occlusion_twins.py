"""Exp-62 stimuli (CPU): natural-object occlusion twins built from COCO val2017 instances.

Each pair pastes two segmented COCO objects (different categories) onto a low-clutter COCO background crop. The two
images of a pair share the background, the object scales and the object centroids; only the paste order differs
(P over Q vs Q over P), so occlusion is the only cue that changes between twins.

Design per pair (canvas 640x480):
  * small object full area a_s ~ U[3.5%, 6.5%] of the canvas; large object a_l = a_s * (1 + f), where
    f = overlap / a_s is the realised overlap fraction (15-40% of the smaller object's area). Hence in the twin with
    the large object behind, the visible areas are near-equal; in the other twin the front object is visibly larger.
    Full-area "front larger" is 50/50 by construction (one twin each).
  * vertical: one object's centroid is higher by >= 6% of the canvas height; "front higher" is 50/50 by construction
    (one twin each); which object (large / small) is higher is balanced across pairs.
  * horizontal side (left / right of the large object) randomised.
Questions (same text for both twins, gold flips): closer / farther yes-no and a two-option "which is closer".

Phases:
  select    annotation-only candidate objects + background crops -> stimuli/candidates.json
  download  fetch the needed val2017 JPEGs from the HF mirror (merve/coco) into datasets/coco2017_val/val2017
  build     compose pairs -> stimuli/images/*.png, stimuli/pairs.jsonl, stimuli/items.jsonl
  sheets    hand-check contact sheets for 40 random images (seed 62) -> stimuli/handcheck/*.png
  finalize  drop pairs listed in stimuli/handcheck/drop.json -> stimuli/items_final.jsonl + manifest.json
            (the released drop list is data/coco_composites/manifest.json: dropped_pairs)
stimuli/ = outputs/e62/stimuli; COCO under $VTAX_DATA_ROOT/datasets/coco2017_val.

Usage: python -m data_gen.coco_occlusion_twins select|download|build|sheets|finalize
"""
from __future__ import annotations
from srww import paths as _vtp  # noqa: E402

import argparse
import hashlib
import json
import math
import os
import random
import shutil
import subprocess
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

SHARE = Path(f"{_vtp.DATA_ROOT}")
COCO = SHARE / "datasets/coco2017_val"
ANN = COCO / "annotations/instances_val2017.json"
IMGDIR = COCO / "val2017"
LOCAL = Path(f"{_vtp.SCRATCH}/e62_tmp")
POD_ST = f"{_vtp.OUTPUTS}/e62/stimuli"
ST = Path(POD_ST)
MIRROR = "https://huggingface.co/datasets/merve/coco/resolve/main/val2017"
W, H = 640, 480
N_PAIRS = 240
EXCLUDE_CATS = {"dining table", "bed", "couch", "bench", "toilet", "sink", "refrigerator", "oven", "tv", "train",
                "airplane", "bus", "truck", "boat", "fire hydrant", "parking meter", "traffic light", "stop sign",
                "kite", "skis", "snowboard", "surfboard", "baseball bat", "tennis racket", "fork", "knife", "spoon",
                "toothbrush", "hair drier", "person"}
RENAME = {"sports ball": "ball", "potted plant": "potted plant", "cell phone": "cell phone"}


def poly_mask(segs: list[list[float]], w: int, h: int) -> np.ndarray:
    """Rasterise COCO polygons to a boolean mask."""
    m = Image.new("L", (w, h), 0)
    d = ImageDraw.Draw(m)
    for s in segs:
        if len(s) >= 6:
            d.polygon([(s[i], s[i + 1]) for i in range(0, len(s), 2)], fill=255)
    return np.asarray(m) > 0


def hull_area(pts: np.ndarray) -> float:
    """Convex hull area of 2D points."""
    from scipy.spatial import ConvexHull
    return float(ConvexHull(pts).volume)


# ============================================================================= select

def select() -> None:
    """Annotation-only selection of isolated, compact objects and low-clutter background crops."""
    d = json.load(open(ANN))
    cats = {c["id"]: c["name"] for c in d["categories"]}
    imgs = {i["id"]: i for i in d["images"]}
    by_img = defaultdict(list)
    for a in d["annotations"]:
        by_img[a["image_id"]].append(a)
    objs = []
    for a in d["annotations"]:
        name = cats[a["category_id"]]
        if a["iscrowd"] or name in EXCLUDE_CATS or not isinstance(a["segmentation"], list) or len(a["segmentation"]) != 1:
            continue
        im = imgs[a["image_id"]]
        x, y, w, h = a["bbox"]
        if a["area"] < 6000 or w < 60 or h < 60 or max(w, h) / min(w, h) > 2.8:
            continue
        if x < 4 or y < 4 or x + w > im["width"] - 4 or y + h > im["height"] - 4:
            continue
        pts = np.array(a["segmentation"][0]).reshape(-1, 2)
        sol = a["area"] / max(hull_area(pts), 1.0)
        if sol < 0.75:
            continue
        # isolation: no other instance bbox overlaps the bbox expanded by 5% (> 4% of its area) (avoids source occlusion / clutter)
        ex = (x - 0.05 * w, y - 0.05 * h, x + 1.05 * w, y + 1.05 * h)
        clash = False
        for b in by_img[a["image_id"]]:
            if b["id"] == a["id"]:
                continue
            bx, by, bw, bh = b["bbox"]
            iw = min(ex[2], bx + bw) - max(ex[0], bx)
            ih = min(ex[3], by + bh) - max(ex[1], by)
            if iw > 0 and ih > 0 and iw * ih > 0.04 * w * h:
                clash = True
                break
        if clash:
            continue
        objs.append({"ann_id": a["id"], "image_id": a["image_id"], "file": im["file_name"], "cat": RENAME.get(name, name),
                     "bbox": a["bbox"], "area": a["area"], "solidity": round(sol, 3), "seg": a["segmentation"]})
    # backgrounds: 4:3 crops with <= 2% annotated pixels, width >= 55% of the image
    bgs = []
    used_obj_imgs = {o["image_id"] for o in objs}
    for iid, im in imgs.items():
        anns = by_img.get(iid, [])
        if iid in used_obj_imgs or any(a["iscrowd"] for a in anns) or len(anns) > 6:
            continue
        iw, ih = im["width"], im["height"]
        segs = [s for a in anns for s in a["segmentation"]]
        m = poly_mask(segs, iw, ih) if segs else np.zeros((ih, iw), bool)
        ii = np.pad(m.cumsum(0).cumsum(1), ((1, 0), (1, 0)))
        best = None
        for frac in (0.8, 0.7, 0.6, 0.55):
            cw = int(frac * iw)
            ch = int(cw * 3 / 4)
            if ch > ih:
                ch = int(frac * ih)
                cw = int(ch * 4 / 3)
            for x0 in np.linspace(0, iw - cw, 6).astype(int):
                for y0 in np.linspace(0, ih - ch, 6).astype(int):
                    cov = (ii[y0 + ch, x0 + cw] - ii[y0, x0 + cw] - ii[y0 + ch, x0] + ii[y0, x0]) / (cw * ch)
                    if cov <= 0.02 and (best is None or cov < best[0]):
                        best = (float(cov), [int(x0), int(y0), int(cw), int(ch)])
            if best:
                break
        if best:
            bgs.append({"image_id": iid, "file": im["file_name"], "crop": best[1], "coverage": round(best[0], 4)})
    cc = Counter(o["cat"] for o in objs)
    print(f"objects {len(objs)} in {len(used_obj_imgs)} images, {len(cc)} categories; backgrounds {len(bgs)}")
    print(cc.most_common())
    # cap objects per category (diversity) and sample backgrounds
    rng = random.Random(62)
    rng.shuffle(objs)
    keep, per = [], Counter()
    for o in objs:
        if per[o["cat"]] < 20:
            keep.append(o)
            per[o["cat"]] += 1
    rng.shuffle(bgs)
    bgs = bgs[:840]
    ST.mkdir(parents=True, exist_ok=True)
    json.dump({"objects": keep, "backgrounds": bgs, "ann_sha256": sha256(ANN)}, open(ST / "candidates.json", "w"))
    files = sorted({o["file"] for o in keep} | {b["file"] for b in bgs})
    print(f"kept objects {len(keep)} ({len(per)} categories); backgrounds {len(bgs)}; files to download {len(files)}")


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


# ============================================================================= download

def download(workers: int = 24) -> None:
    """Fetch only the needed val2017 JPEGs (resumable)."""
    c = json.load(open(ST / "candidates.json"))
    files = sorted({o["file"] for o in c["objects"]} | {b["file"] for b in c["backgrounds"]})
    IMGDIR.mkdir(parents=True, exist_ok=True)
    tmp = Path(f"{_vtp.SCRATCH}/e62_tmp/val2017")
    tmp.mkdir(parents=True, exist_ok=True)

    def get(fn: str) -> str:
        out = IMGDIR / fn
        if out.exists() and out.stat().st_size > 1000:
            return "skip"
        t = tmp / fn
        r = subprocess.run(["curl", "-sS", "-L", "--retry", "4", "-m", "300", "-o", str(t), f"{MIRROR}/{fn}"],
                           capture_output=True)
        if r.returncode or not t.exists() or t.stat().st_size < 1000:
            return "fail"
        try:
            Image.open(t).verify()
        except Exception:
            return "bad"
        subprocess.run(["cp", str(t), str(out)])
        return "ok"

    with ThreadPoolExecutor(workers) as ex:
        res = list(ex.map(get, files))
    print(Counter(res))
    sums = {fn: sha256(IMGDIR / fn) for fn in files if (IMGDIR / fn).exists()}
    json.dump({"source": MIRROR, "n": len(sums), "sha256": sums}, open(COCO / "val2017_subset_sha256.json", "w"), indent=0)
    print(f"sha256 recorded for {len(sums)} files")


# ============================================================================= build

def src(fn: str) -> Path:
    """Local copy of a val2017 JPEG if present (same bytes as IMGDIR; download() copies tmp -> IMGDIR)."""
    p = LOCAL / "val2017" / fn
    return p if p.exists() else IMGDIR / fn


def judge_pass(kind: str, ok: str) -> set[str]:
    """Task ids accepted by both stimulus-filter judges."""
    sets = []
    for j in ("claude-opus-5-5", "gpt-6-astra"):
        rows = [json.loads(l) for l in open(ST / f"judge_{kind}_{j}.jsonl")]
        sets.append({r["tid"] for r in rows if r.get("label") == ok})
    return sets[0] & sets[1]


def cutout(o: dict) -> Image.Image:
    """RGBA cutout of a COCO instance (alpha = polygon eroded by 1 px and feathered)."""
    im = Image.open(src(o["file"])).convert("RGB")
    m = Image.fromarray((poly_mask(o["seg"], im.width, im.height) * 255).astype(np.uint8))
    m = m.filter(ImageFilter.MinFilter(3)).filter(ImageFilter.GaussianBlur(0.8))
    x, y, w, h = o["bbox"]
    box = (int(math.floor(x)) - 2, int(math.floor(y)) - 2, int(math.ceil(x + w)) + 2, int(math.ceil(y + h)) + 2)
    rgba = im.crop(box)
    rgba.putalpha(m.crop(box))
    return rgba


def scaled(rgba: Image.Image, target_area: float) -> tuple[Image.Image, np.ndarray]:
    """Resize a cutout so that its (alpha > 0.5) area equals target_area; returns image and boolean mask."""
    a0 = (np.asarray(rgba.split()[-1]) > 127).sum()
    s = math.sqrt(target_area / a0)
    im = rgba.resize((max(1, round(rgba.width * s)), max(1, round(rgba.height * s))), Image.LANCZOS)
    return im, np.asarray(im.split()[-1]) > 127


def centroid(mask: np.ndarray) -> tuple[float, float]:
    ys, xs = np.nonzero(mask)
    return float(xs.mean()), float(ys.mean())


def place(m_l: np.ndarray, m_s: np.ndarray, rng: random.Random, small_higher: bool, f_target: float,
          tries: int = 300) -> dict | None:
    """Find top-left positions so that overlap / area_small ~ f_target, both objects inside the canvas, and the
    requested object higher by >= 6% of H. Returns positions, centroids and realised overlap."""
    cl, cs = centroid(m_l), centroid(m_s)
    a_s = m_s.sum()
    for _ in range(tries):
        ang = math.radians(rng.uniform(20, 70))
        sx = rng.choice([-1, 1])
        sy = -1 if small_higher else 1  # direction of the small object's centroid relative to the large one
        ux, uy = sx * math.cos(ang), sy * math.sin(ang)
        # canvas position of the large object's centroid
        Lx = rng.uniform(0.3, 0.7) * W
        Ly = rng.uniform(0.35, 0.65) * H
        lo, hi = 0.0, 0.5 * (math.hypot(*m_l.shape) + math.hypot(*m_s.shape))
        best = None
        for _ in range(22):
            dist = 0.5 * (lo + hi)
            Sx, Sy = Lx + ux * dist, Ly + uy * dist
            pl = (round(Lx - cl[0]), round(Ly - cl[1]))
            ps = (round(Sx - cs[0]), round(Sy - cs[1]))
            ov = overlap(m_l, pl, m_s, ps) / a_s
            best = (pl, ps, ov, (Lx, Ly), (Sx, Sy))
            if ov > f_target:
                lo = dist
            else:
                hi = dist
        pl, ps, ov, Lc, Sc = best
        if not (0.15 <= ov <= 0.40 and abs(ov - f_target) < 0.04):
            continue
        if abs(Lc[1] - Sc[1]) < 0.06 * H:
            continue
        ok = all(8 <= p[0] and 8 <= p[1] and p[0] + m.shape[1] <= W - 8 and p[1] + m.shape[0] <= H - 8
                 for p, m in ((pl, m_l), (ps, m_s)))
        if ok:
            return {"pos_l": pl, "pos_s": ps, "overlap_frac_small": float(ov), "cent_l": Lc, "cent_s": Sc}
    return None


def canvas_mask(m: np.ndarray, pos: tuple[int, int]) -> np.ndarray:
    c = np.zeros((H, W), bool)
    x, y = pos
    c[y:y + m.shape[0], x:x + m.shape[1]] = m
    return c


def overlap(m1: np.ndarray, p1: tuple[int, int], m2: np.ndarray, p2: tuple[int, int]) -> int:
    """Pixel overlap of two masks at integer positions (clipped to a padded canvas)."""
    x0, y0 = min(p1[0], p2[0]), min(p1[1], p2[1])
    x1 = max(p1[0] + m1.shape[1], p2[0] + m2.shape[1])
    y1 = max(p1[1] + m1.shape[0], p2[1] + m2.shape[0])
    a = np.zeros((y1 - y0, x1 - x0), bool)
    b = np.zeros_like(a)
    a[p1[1] - y0:p1[1] - y0 + m1.shape[0], p1[0] - x0:p1[0] - x0 + m1.shape[1]] = m1
    b[p2[1] - y0:p2[1] - y0 + m2.shape[0], p2[0] - x0:p2[0] - x0 + m2.shape[1]] = m2
    return int((a & b).sum())


def bg_image(b: dict) -> Image.Image:
    im = Image.open(src(b["file"])).convert("RGB")
    x, y, w, h = b["crop"]
    return im.crop((x, y, x + w, y + h)).resize((W, H), Image.LANCZOS)


def qtexts(x: str, y: str, o1: str, o2: str) -> dict[str, str]:
    return {"closer": f"Is the {x} closer to the camera than the {y}?\n(A) yes\n(B) no",
            "farther": f"Is the {x} farther from the camera than the {y}?\n(A) yes\n(B) no",
            "which": f"Which object is closer to the camera?\n(A) the {o1}\n(B) the {o2}"}


def build() -> None:
    """Compose N_PAIRS twin pairs (seed 62)."""
    c = json.load(open(ST / "candidates.json"))
    ok_obj, ok_bg = judge_pass("obj", "YES"), judge_pass("bg", "CLEAN")
    objs = [o for o in c["objects"] if (IMGDIR / o["file"]).exists() and str(o["ann_id"]) in ok_obj]
    ann = json.load(open(ANN))
    names = {k["id"]: RENAME.get(k["name"], k["name"]) for k in ann["categories"]}
    img_cats = defaultdict(set)
    for a in ann["annotations"]:
        img_cats[a["image_id"]].add(names[a["category_id"]])
    bgs = []
    for b in c["backgrounds"]:
        if not (IMGDIR / b["file"]).exists() or b["file"] not in ok_bg:
            continue
        b["cats"] = img_cats[b["image_id"]]
        g = np.asarray(bg_image(b).convert("L"), np.float32)
        b["gray_std"] = float(g.std())
        b["gray_mean"] = float(g.mean())
        if 18 <= b["gray_std"] and 40 <= b["gray_mean"] <= 215:  # textured, not near-black / blown out
            bgs.append(b)
    by_cat = defaultdict(list)
    for o in objs:
        by_cat[o["cat"]].append(o)
    cats = sorted(by_cat)
    print(f"{len(objs)} objects ({len(cats)} categories), {len(bgs)} usable backgrounds")
    rng = random.Random(62)
    out_img = LOCAL / "stim_images"
    if out_img.exists():
        shutil.rmtree(out_img)
    out_img.mkdir(parents=True)
    plan = [(lh, xs) for lh in (True, False) for xs in (True, False)] * (N_PAIRS // 4 + 1)
    rng.shuffle(plan)
    pairs, items = [], []
    used_obj = Counter()
    k = 0
    bg_i = 0
    while len(pairs) < N_PAIRS:
        small_higher, ask_large = plan[k % len(plan)]
        k += 1
        # category weight = min(#instances, 8): varied categories without re-using rare instances too often
        c1, c2 = None, None
        while c1 == c2:
            c1, c2 = rng.choices(cats, weights=[min(len(by_cat[c]), 8) for c in cats], k=2)
        o1 = min(by_cat[c1], key=lambda o: (used_obj[o["ann_id"]], rng.random()))
        o2 = min(by_cat[c2], key=lambda o: (used_obj[o["ann_id"]], rng.random()))
        lg, sm = (o1, o2) if rng.random() < 0.5 else (o2, o1)
        f_target = rng.uniform(0.18, 0.37)
        a_s = rng.uniform(0.035, 0.065) * W * H
        im_l, m_l = scaled(cutout(lg), a_s * (1 + f_target))
        im_s, m_s = scaled(cutout(sm), a_s)
        # upscaling limit (source resolution): linear scale <= 1.8
        if im_l.width > 1.8 * (lg["bbox"][2] + 4) or im_s.width > 1.8 * (sm["bbox"][2] + 4):
            continue
        geo = place(m_l, m_s, rng, small_higher, f_target)
        if geo is None:
            continue
        # background must not contain an annotated instance of either pair category
        for _ in range(len(bgs)):
            b = bgs[bg_i % len(bgs)]
            bg_i += 1
            if not ({lg["cat"], sm["cat"]} & b["cats"]):
                break
        pid = f"e62p{len(pairs):03d}"
        cm_l, cm_s = canvas_mask(m_l, geo["pos_l"]), canvas_mask(m_s, geo["pos_s"])
        ov = int((cm_l & cm_s).sum())
        a_l, a_sm = int(cm_l.sum()), int(cm_s.sum())
        nl, ns = lg["cat"], sm["cat"]
        x, y = (nl, ns) if ask_large else (ns, nl)
        w1, w2 = (nl, ns) if rng.random() < 0.5 else (ns, nl)
        qs = qtexts(x, y, w1, w2)
        base = bg_image(b)
        for tag, front in (("a", "large"), ("b", "small")):
            img = base.copy()
            order = [("small", im_s, geo["pos_s"]), ("large", im_l, geo["pos_l"])]
            if front == "small":
                order = order[::-1]
            for _, im, pos in order:
                img.paste(im, pos, im)
            uid = f"{pid}{tag}"
            img.save(out_img / f"{uid}.png")
            front_name = nl if front == "large" else ns
            back_name = ns if front == "large" else nl
            vis_front = a_l if front == "large" else a_sm
            vis_back = (a_sm if front == "large" else a_l) - ov
            front_cent = geo["cent_l"] if front == "large" else geo["cent_s"]
            back_cent = geo["cent_s"] if front == "large" else geo["cent_l"]
            meta = {"pair": pid, "twin_uid": f"{pid}{'b' if tag == 'a' else 'a'}", "front_name": front_name,
                    "back_name": back_name, "front_is_large": front == "large", "front_higher": front_cent[1] < back_cent[1],
                    "front_left": front_cent[0] < back_cent[0], "vis_ratio_front_over_back": vis_front / vis_back,
                    "full_ratio_large_over_small": a_l / a_sm, "overlap_frac_small": ov / a_sm,
                    "large": {"cat": nl, "ann_id": lg["ann_id"], "file": lg["file"], "area_px": a_l, "pos": geo["pos_l"]},
                    "small": {"cat": ns, "ann_id": sm["ann_id"], "file": sm["file"], "area_px": a_sm, "pos": geo["pos_s"]},
                    "background": {"file": b["file"], "crop": b["crop"]}, "x_name": x, "y_name": y,
                    "which_opts": [w1, w2], "ask_large": ask_large, "small_higher": small_higher}
            for ph, q in qs.items():
                if ph == "closer":
                    g = "A" if x == front_name else "B"
                elif ph == "farther":
                    g = "A" if x == back_name else "B"
                else:
                    g = "A" if w1 == front_name else "B"
                items.append({"uid": f"{uid}_{ph}", "image_uid": uid, "twin_item": f"{meta['twin_uid']}_{ph}",
                              "pair": pid, "phrasing": ph, "question": q, "choices": ["A", "B"], "gold": g,
                              "image_path": f"{POD_ST}/images/{uid}.png", "front_name": front_name,
                              "back_name": back_name, "meta": meta})
        used_obj[lg["ann_id"]] += 1
        used_obj[sm["ann_id"]] += 1
        pairs.append({"pair": pid, "large": nl, "small": ns, "overlap_frac_small": ov / a_sm,
                      "full_ratio": a_l / a_sm, "small_higher": small_higher, "ask_large": ask_large,
                      "bg": b["file"], "f_target": f_target})
        if len(pairs) % 40 == 0:
            print(len(pairs), "pairs", flush=True)
    if (ST / "images").exists():
        shutil.rmtree(ST / "images")
    shutil.copytree(out_img, ST / "images")
    with open(ST / "pairs.jsonl", "w") as f:
        for p in pairs:
            f.write(json.dumps(p) + "\n")
    with open(ST / "items.jsonl", "w") as f:
        for it in items:
            f.write(json.dumps(it) + "\n")
    report(items, pairs, "build")


def report(items: list[dict], pairs: list[dict], tag: str) -> dict:
    imgs = {it["image_uid"]: it["meta"] for it in items}
    r = {"tag": tag, "n_pairs": len(pairs), "n_images": len(imgs), "n_items": len(items),
         "n_categories": len({p["large"] for p in pairs} | {p["small"] for p in pairs}),
         "front_higher_frac": float(np.mean([m["front_higher"] for m in imgs.values()])),
         "front_large_frac": float(np.mean([m["front_is_large"] for m in imgs.values()])),
         "front_left_frac": float(np.mean([m["front_left"] for m in imgs.values()])),
         "small_higher_pair_frac": float(np.mean([p["small_higher"] for p in pairs])),
         "gold_A_frac": {ph: float(np.mean([it["gold"] == "A" for it in items if it["phrasing"] == ph]))
                         for ph in ("closer", "farther", "which")},
         "overlap_frac_small": [float(np.min([p["overlap_frac_small"] for p in pairs])),
                                float(np.median([p["overlap_frac_small"] for p in pairs])),
                                float(np.max([p["overlap_frac_small"] for p in pairs]))],
         "vis_ratio_front_over_back_quartiles": [float(np.quantile([m["vis_ratio_front_over_back"] for m in imgs.values()], q))
                                                 for q in (0, 0.25, 0.5, 0.75, 1)],
         "category_counts": Counter([p["large"] for p in pairs] + [p["small"] for p in pairs]).most_common()}
    print(json.dumps({k: v for k, v in r.items() if k != "category_counts"}, indent=1))
    return r


# ============================================================================= hand check

def sheets(n: int = 40) -> None:
    """Contact sheets (10 tiles each, 2 x 5) of n random images with names and front object printed."""
    items = [json.loads(l) for l in open(ST / "items.jsonl")]
    imgs = {it["image_uid"]: it for it in items}
    pick = sorted(random.Random(62).sample(sorted(imgs), n))
    out = ST / "handcheck"
    out.mkdir(exist_ok=True)
    tw, th = 480, 380
    for s in range(0, n, 10):
        sheet = Image.new("RGB", (tw * 2, th * 5), "white")
        for j, u in enumerate(pick[s:s + 10]):
            it = imgs[u]
            im = Image.open(ST / "images" / f"{u}.png").resize((tw, 360))
            t = Image.new("RGB", (tw, th), "white")
            t.paste(im, (0, 0))
            ImageDraw.Draw(t).text((4, 362), f"{u}: front={it['front_name']} | back={it['back_name']}", fill="black")
            sheet.paste(t, ((j % 2) * tw, (j // 2) * th))
        sheet.save(out / f"sheet{s // 10}.png")
    json.dump(pick, open(out / "picked.json", "w"))
    print("picked", pick)


def finalize() -> None:
    items = [json.loads(l) for l in open(ST / "items.jsonl")]
    pairs = [json.loads(l) for l in open(ST / "pairs.jsonl")]
    dp = ST / "handcheck/drop.json"
    drop = set(json.load(open(dp))["drop_pairs"]) if dp.exists() else set()
    items = [it for it in items if it["pair"] not in drop]
    pairs = [p for p in pairs if p["pair"] not in drop]
    with open(ST / "items_final.jsonl", "w") as f:
        for it in items:
            f.write(json.dumps(it) + "\n")
    r = report(items, pairs, "final")
    imgs = sorted({it["image_uid"] for it in items})
    man = {"exp": "e62", "n_pairs": len(pairs), "n_images": len(imgs), "dropped_pairs": sorted(drop), "report": r,
           "coco_annotations": str(ANN), "coco_annotations_sha256": sha256(ANN),
           "coco_image_sha256_file": str(COCO / "val2017_subset_sha256.json"),
           "images_sha256": {u: sha256(ST / "images" / f"{u}.png") for u in imgs},
           "items_final_sha256": sha256(ST / "items_final.jsonl"), "builder": "data_gen/coco_occlusion_twins.py", "seed": 62}
    json.dump(man, open(ST / "manifest.json", "w"), indent=1)
    print(f"final: {len(pairs)} pairs, {len(imgs)} images, {len(items)} items; dropped {len(drop)}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("phase", choices=["select", "download", "build", "sheets", "finalize"])
    ap.add_argument("--workers", type=int, default=24)
    a = ap.parse_args()
    {"select": select, "download": lambda: download(a.workers), "build": build, "sheets": sheets,
     "finalize": finalize}[a.phase]()


if __name__ == "__main__":
    main()
