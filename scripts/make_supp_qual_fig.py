"""Generate the one-page supplementary qualitative figure (Track F1, Fig. S1 / fig:supp_qual).

Two parts on a single full-width canvas (6.875 in wide, 7.70 in = 0.87 textheight tall):
  (a) Stimulus gallery: one small example per evaluation set with its verbatim question
      and cue/confound design note:
        1. Cue-factorial set (twin pair cue018 / cue018_cf: identical geometry, swapped
           identities, 2x2 size x height balanced so occlusion is the sole valid cue).
        2. Synthetic Depth (syn_depth_10 / depth_0010.png: front shape always drawn larger,
           creating a 98.6% visible-area size confound).
        3. Hard occlusion set for closed models (occl4026, tier L4: thin overlap sliver,
           low contrast, confusable colours, distractors, blur and sensor noise).
        4. COCO occlusion twin pair (e62p016a / e62p016b: segmented COCO cutouts with only
           the paste order swapped).
        5. Mirrored CV-Bench 2D left/right pair (cvb_Relation_1276_COCO: horizontal mirror
           flips left/right ground truth on natural photographs).
  (b) Three real, verbatim annotated rationales from saved outputs:
        1. Stage 1 error onset & Stage 2 lock-in: Qwen3-VL-8B cue-set D+C- rationale
           (cue320) where Direct is right, showing per-sentence forced readout P_j(gold)
           and highlighting the commit sentence j* (first sentence after which P_j(gold) < 0.5
           and stays there).
        2. Stage 3 converse mis-mapping: Molmo-7B-D cue-set D+C- rationale (cue348_behind)
           that writes a true body relation ("in front of", "partially covers") and then
           mis-maps the conclusion on a converse "behind" question.
        3. Closed-model answer-step re-grounding: Claude Opus 5.5 on hard occlusion
           (occl4026_front) with its own conclusion-removed rationale R_cut(v) held fixed,
           answering gold_v under v and overriding the fixed text to give the image-consistent
           answer under the swapped twin v_cf (alongside Gemini-4-Argon, GPT-6-Astra, and
           text-locked Qwen3.8-Max on the same item).

Writes:
  - ../paper/figures/supp_qual_examples.pdf
  - ../paper/figures/supp_qual_examples.png (300 dpi)
  - ../paper/figures/supp_qual_examples.data.json

Run from vtax_exps/:
    python make_supp_qual_fig.py
"""
from __future__ import annotations
import vtax_paths as _vtp  # noqa: E402

import argparse
import io
import json
import math
import os
import textwrap
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageOps

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_BUNDLE_DIR = Path(_vtp.OUTPUTS) / "bundle"
DEFAULT_FIG_DIR = SCRIPT_DIR.parent / f"{_vtp.PAPER}" / "figures"


def resolve_share_root() -> Path:
    """Resolve the shared network storage root across dev machine and pod mounts."""
    env_share = os.environ.get("VTAX_DATA_ROOT")
    if env_share and Path(env_share).exists():
        return Path(env_share)
    for cand in (
        Path(f"{_vtp.DATA_ROOT}"),
        Path(f"{_vtp.DATA_ROOT}"),
    ):
        if cand.exists():
            return cand
    raise FileNotFoundError("Could not locate shared network storage root.")


# Tab-style muted palette (#B9BEC6 #9BBAD9 #E2B28F #DC8A7E)
COL_SLATE = "#B9BEC6"
COL_SLATE_DARK = "#5A6270"
COL_ICE = "#9BBAD9"
COL_ICE_DARK = "#3B6998"
COL_PEACH = "#E2B28F"
COL_PEACH_DARK = "#965E32"
COL_CORAL = "#DC8A7E"
COL_CORAL_DARK = "#B85244"

COL_TEXT = "#1F242B"
COL_MUTED = "#555D68"
COL_CARD_BG = "#F8F9FB"
COL_CARD_EDGE = "#CFD5DF"
COL_SUBBOX_BG = "#FFFFFF"
COL_SUBBOX_EDGE = "#DCE1E9"

TINT_ICE = "#EDF3FA"
TINT_PEACH = "#FDF4E8"
TINT_CORAL = "#FCEFEB"
TINT_SLATE = "#F0F2F6"
TINT_GOOD = "#ECF5EF"
EDGE_GOOD = "#8BB99B"
COL_GOOD = "#266D42"
COL_BAD = "#B24035"


def configure_matplotlib() -> None:
    """Configure sans-serif typography with Type-42 TrueType vector font embedding."""
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
        "mathtext.fontset": "dejavusans",
        "font.size": 6.5,
        "text.color": COL_TEXT,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def rel_path(p: Path | str) -> str:
    """Strip machine-specific prefixes for clean reproducibility audit JSON."""
    s = str(p)
    for anchor in ("vtax/", "datasets/", "outputs/bundle/"):
        k = s.find(anchor)
        if k >= 0:
            return s[k:]
    return Path(s).name


def read_jsonl(path: Path, needle: str | None = None) -> list[dict[str, Any]]:
    """Load JSON lines from a file, optionally pre-filtering lines by substring."""
    out: list[dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            if needle is not None and needle not in line:
                continue
            out.append(json.loads(line))
    return out


def probs_from_lp(lp: dict[str, Any]) -> dict[str, float]:
    """Convert option log-probabilities into normalized probabilities."""
    fin = {k: float(v) for k, v in lp.items() if v is not None and math.isfinite(float(v))}
    m = max(fin.values())
    z = sum(math.exp(v - m) for v in fin.values())
    return {k: (math.exp(fin[k] - m) / z if k in fin else 0.0) for k in lp}


def argmax_lp(lp: dict[str, Any]) -> str:
    """Return the option letter with the highest finite log-probability."""
    fin = {k: float(v) for k, v in lp.items() if v is not None and math.isfinite(float(v))}
    return max(fin, key=fin.get)


def crop_foreground_square(im_a: np.ndarray, im_b: np.ndarray | None = None,
                           pad: int = 34) -> tuple[np.ndarray, np.ndarray | None]:
    """Compute a shared square crop tightly framing foreground shapes."""
    h, w, _ = im_a.shape
    bg_a = im_a[0, 0]
    ys_a, xs_a = np.where(np.any(np.abs(im_a.astype(int) - bg_a.astype(int)) > 8, axis=-1))
    x0, x1 = int(xs_a.min()), int(xs_a.max())
    y0, y1 = int(ys_a.min()), int(ys_a.max())
    if im_b is not None:
        bg_b = im_b[0, 0]
        ys_b, xs_b = np.where(np.any(np.abs(im_b.astype(int) - bg_b.astype(int)) > 8, axis=-1))
        x0, x1 = min(x0, int(xs_b.min())), max(x1, int(xs_b.max()))
        y0, y1 = min(y0, int(ys_b.min())), max(y1, int(ys_b.max()))
    x0, y0 = max(0, x0 - pad), max(0, y0 - pad)
    x1, y1 = min(w, x1 + pad), min(h, y1 + pad)
    side = min(max(x1 - x0, y1 - y0), min(h, w))
    cx, cy = (x0 + x1) // 2, (y0 + y1) // 2
    sx0 = max(0, min(w - side, cx - side // 2))
    sy0 = max(0, min(h - side, cy - side // 2))
    ca = im_a[sy0: sy0 + side, sx0: sx0 + side]
    cb = im_b[sy0: sy0 + side, sx0: sx0 + side] if im_b is not None else None
    return ca, cb


def crop_bbox_square(im: np.ndarray, boxes: list[list[int]],
                     pad: int = 24) -> tuple[np.ndarray, tuple[int, int, int]]:
    """Square crop around the union of [x1, y1, x2, y2] bounding boxes."""
    h, w, _ = im.shape
    x0 = max(0, min(b[0] for b in boxes) - pad)
    y0 = max(0, min(b[1] for b in boxes) - pad)
    x1 = min(w, max(b[2] for b in boxes) + pad)
    y1 = min(h, max(b[3] for b in boxes) + pad)
    side = min(max(x1 - x0, y1 - y0), min(h, w))
    cx, cy = (x0 + x1) // 2, (y0 + y1) // 2
    sx0 = max(0, min(w - side, cx - side // 2))
    sy0 = max(0, min(h - side, cy - side // 2))
    return im[sy0: sy0 + side, sx0: sx0 + side], (sx0, sy0, side)


def find_t_junction(img: np.ndarray, tol: int = 38) -> tuple[float, float] | None:
    """Locate a foreground-foreground-background occlusion T-junction pixel."""
    from scipy.ndimage import binary_dilation
    bg = img[0, 0].astype(int)
    flat = img.reshape(-1, 3).astype(int)
    fg = np.abs(flat - bg).sum(1) > tol
    cols, counts = np.unique(flat[fg], axis=0, return_counts=True)
    if len(cols) < 2:
        return None
    c1 = cols[counts.argmax()]
    rest = [(n, c) for c, n in zip(cols, counts) if np.abs(c - c1).sum() > 3 * tol]
    if not rest:
        return None
    c2 = max(rest, key=lambda t: t[0])[1]
    im = img.astype(int)
    m1 = np.abs(im - c1).sum(-1) < tol
    m2 = np.abs(im - c2).sum(-1) < tol
    mb = np.abs(im - bg).sum(-1) < tol
    st = np.ones((5, 5), bool)
    tj = binary_dilation(m1, st) & binary_dilation(m2, st) & binary_dilation(mb, st)
    ys, xs = np.nonzero(tj)
    if not len(ys):
        return None
    k = int(np.argmin(ys))
    return float(xs[k]), float(ys[k])


def load_cvbench_single(share_root: Path, idx: int, task: str = "Relation") -> Image.Image:
    """Fast direct Arrow lookup of a single CV-Bench RGB image without loading all rows."""
    ds_dir = share_root / "datasets" / "cv_bench" / "hf_dataset"
    arrow_file = ds_dir / "data-00000-of-00001.arrow"
    if arrow_file.exists():
        import pyarrow as pa
        import pyarrow.compute as pc
        with pa.memory_map(str(arrow_file), "r") as source:
            table = pa.ipc.open_stream(source).read_all()
        mask = pc.and_(pc.equal(table["idx"], idx), pc.equal(table["task"], task))
        sub = table.filter(mask)
        if len(sub) > 0:
            img_cell = sub["image"][0].as_py()
            if isinstance(img_cell, dict) and img_cell.get("bytes"):
                return Image.open(io.BytesIO(img_cell["bytes"])).convert("RGB")
    from datasets import load_from_disk
    ds = load_from_disk(str(ds_dir))
    match = [i for i, (ix, tk) in enumerate(zip(ds["idx"], ds["task"])) if ix == idx and tk == task]
    if not match:
        raise KeyError(f"CV-Bench item idx={idx} task={task} not found")
    return ds.select([match[0]])[0]["image"].convert("RGB")


def gather_all_data(share_root: Path, bundle_dir: Path) -> dict[str, Any]:
    """Load and verify all stimuli, verbatim texts, and forced readouts from disk."""
    vtax = share_root / "code" / "vtax"
    out_dir = vtax / "outputs"

    # Cue-factorial items dir (outputs/e50/occ_cues or outputs/e28/cue_items)
    cues_dir = out_dir / "e50" / "occ_cues"
    if not (cues_dir / "items.jsonl").exists():
        cues_dir = out_dir / "e28" / "cue_items"
    cue_base = {r["item_id"]: r for r in read_jsonl(cues_dir / "items.jsonl", "cue018")}
    cue_cf = {r["item_id"]: r for r in read_jsonl(cues_dir / "items_cf.jsonl", "cue018_cf")}

    # (a.1) Cue-factorial twin pair: cue018 / cue018_cf
    a1_v = cue_base["cue018"]
    a1_cf = cue_cf["cue018_cf"]
    a1_im_v = np.asarray(Image.open(cues_dir / "images" / "cue018.png").convert("RGB"))
    a1_im_cf = np.asarray(Image.open(cues_dir / "images" / "cue018_cf.png").convert("RGB"))
    a1_crop_v, a1_crop_cf = crop_foreground_square(a1_im_v, a1_im_cf, pad=34)

    # (a.2) Synthetic Depth: syn_depth_10 (depth_0010.png)
    syn_json_path = share_root / "datasets" / "spatial_probing" / "depth_ordering.json"
    syn_list = json.loads(syn_json_path.read_text(encoding="utf-8"))
    syn_rec = syn_list[10]
    syn_img_path = share_root / "datasets" / "spatial_probing" / "images" / "depth_0010.png"
    syn_geo = json.loads((bundle_dir / "syn_depth_geometry.json").read_text(encoding="utf-8"))["items"]["syn_depth_10"]
    syn_im = np.asarray(Image.open(syn_img_path).convert("RGB"))
    syn_crop, _ = crop_foreground_square(syn_im, None, pad=30)
    syn_px_front = int(syn_geo["yellow triangle"]["px"])
    syn_px_back = int(syn_geo["orange square"]["px"])
    syn_area_ratio = syn_px_front / syn_px_back

    # (a.3) Hard occlusion set: occl4026 / occl4026_cf
    hard_dir = vtax / "exps" / "items" / "occlusion_hard"
    hard_base = {r["item_id"]: r for r in read_jsonl(hard_dir / "items.jsonl", "occl4026")}
    hard_cf = {r["item_id"]: r for r in read_jsonl(hard_dir / "items_cf.jsonl", "occl4026_cf")}
    a3_v = hard_base["occl4026"]
    a3_cf = hard_cf["occl4026_cf"]
    a3_im_v = np.asarray(Image.open(hard_dir / "images" / "occl4026.png").convert("RGB"))
    a3_im_cf = np.asarray(Image.open(hard_dir / "images" / "occl4026_cf.png").convert("RGB"))

    # (a.4) COCO occlusion twin pair: e62p016a / e62p016b
    e62_dir = out_dir / "e62" / "stimuli"
    e62_items = {(r["image_uid"], r["phrasing"]): r for r in read_jsonl(e62_dir / "items_final.jsonl", "e62p016")}
    a4_v = e62_items[("e62p016a", "closer")]
    a4_cf = e62_items[("e62p016b", "closer")]
    a4_im_v = np.asarray(Image.open(e62_dir / "images" / "e62p016a.png").convert("RGB"))
    a4_im_cf = np.asarray(Image.open(e62_dir / "images" / "e62p016b.png").convert("RGB"))
    a4_crop_v = a4_im_v[140:460, 40:360]
    a4_crop_cf = a4_im_cf[140:460, 40:360]

    # (a.5) Mirrored CV-Bench left/right pair: cvb_Relation_1276_COCO
    e60_items = {r["uid"]: r for r in read_jsonl(out_dir / "e60" / "items.jsonl", "cvb_Relation_1276_COCO")}
    a5_rec = e60_items["cvb_Relation_1276_COCO"]
    a5_pil_v = load_cvbench_single(share_root, 1276, "Relation")
    a5_pil_mir = ImageOps.mirror(a5_pil_v)
    a5_im_v = np.asarray(a5_pil_v)
    a5_im_mir = np.asarray(a5_pil_mir)

    # (b.1) Qwen3-VL-8B cue-set D+C- rationale: cue320
    e10_q8 = {(r["item_id"], r["condition"]): r
              for r in read_jsonl(out_dir / "e10" / "qwen3vl_8b_cues.jsonl", '"cue320"')}
    traj_q8 = {r["item_id"]: r
               for r in read_jsonl(out_dir / "e28" / "qwen3vl_8b_cues_traj.jsonl", '"cue320"')}
    trajtwin_q8 = {r["item_id"]: r
                   for r in read_jsonl(out_dir / "e57" / "qwen3vl_8b_cues_trajtwin.jsonl", '"cue320"')}
    j_opus = {(r["item_id"], r["idx"]): r["label"]
              for r in read_jsonl(out_dir / "e28" / "judge_claude-opus-5-5_commit_cues.jsonl", '"cue320"')}
    j_gpt = {(r["item_id"], r["idx"]): r["label"]
             for r in read_jsonl(out_dir / "e28" / "judge_gpt-6-astra_commit_cues.jsonl", '"cue320"')}

    b1_dir = e10_q8[("cue320", "direct")]
    b1_cot = e10_q8[("cue320", "cot")]
    b1_tr = traj_q8["cue320"]
    b1_tt = trajtwin_q8["cue320"]
    b1_im_raw = np.asarray(Image.open(cues_dir / "images" / "cue320.png").convert("RGB"))
    b1_crop, _ = crop_foreground_square(b1_im_raw, None, pad=32)

    for s in b1_tr["sentences"]:
        assert s in b1_cot["text"], f"Sentence not verbatim in cue320 raw CoT: {s!r}"
    assert j_opus[("cue320", b1_tr["commit_idx"])] == "INCORRECT"
    assert j_gpt[("cue320", b1_tr["commit_idx"])] == "INCORRECT"

    # (b.2) Molmo-7B-D cue-set Stage-3 converse mis-mapping: cue348_behind
    e10_m7 = {(r["item_id"], r["condition"]): r
              for r in read_jsonl(out_dir / "e10" / "molmo7b_cues.jsonl", '"cue348_behind"')}
    traj_m7 = {r["item_id"]: r
               for r in read_jsonl(out_dir / "e28" / "molmo7b_cues_traj.jsonl", '"cue348_behind"')}
    e54_decomp = json.loads((bundle_dir / "e54_decomp.json").read_text(encoding="utf-8"))
    b2_e54 = next(
        r for r in e54_decomp["_records"]
        if r["model"] == "molmo7b" and r["ds"] == "cues" and r["item"] == "cue348_behind"
    )
    assert b2_e54["dm"] and b2_e54["true"] and b2_e54["cat"] == "b_map" and b2_e54["qform"] == "converse"

    b2_dir = e10_m7[("cue348_behind", "direct")]
    b2_cot = e10_m7[("cue348_behind", "cot")]
    b2_tr = traj_m7["cue348_behind"]
    b2_im_raw = np.asarray(Image.open(cues_dir / "images" / "cue348.png").convert("RGB"))
    b2_crop, _ = crop_foreground_square(b2_im_raw, None, pad=32)
    for s in b2_tr["sentences"]:
        assert s in b2_cot["text"], f"Sentence not verbatim in cue348_behind raw CoT: {s!r}"

    # (b.3) Closed-model fixed R_cut + twin swap on Hard occlusion: occl4026_front
    e61_items = {(r["model"], r["set"], r["uid"]): r
                 for r in read_jsonl(out_dir / "e61" / "items.jsonl", '"occl4026_front"')}
    b3_it_claude = e61_items[("claude-opus-5-5", "hard", "occl4026_front")]
    gen_claude: dict[tuple[str, str, str, str, str], dict[str, Any]] = {}
    for r in read_jsonl(out_dir / "e61" / "gen_claude-opus-5-5.jsonl", '"occl4026_front"'):
        if "error" not in r:
            gen_claude[(r["kind"], r["set"], r["uid"], r["arm"], r["bh"])] = r
    b3_rv = gen_claude[("rcut", "hard", "occl4026_front", "v", b3_it_claude["bh_removed"])]
    b3_rcf = gen_claude[("rcut", "hard", "occl4026_front", "vcf", b3_it_claude["bh_removed"])]

    e61_bundle = json.loads((bundle_dir / "e61_closed_fixed_rcut.json").read_text(encoding="utf-8"))
    b3_peers = {
        r["model"]: r for r in e61_bundle["rows"]
        if r["set"] == "hard" and r["uid"] == "occl4026_front"
    }
    boxes_4026 = [a3_v["meta"]["bbox_front"], a3_v["meta"]["bbox_back"]]
    b3_crop_v, crop_box_4026 = crop_bbox_square(a3_im_v, boxes_4026, pad=26)
    b3_crop_cf, _ = crop_bbox_square(a3_im_cf, boxes_4026, pad=26)

    return {
        "paths": {
            "cues_dir": cues_dir,
            "syn_json": syn_json_path,
            "syn_img": syn_img_path,
            "hard_dir": hard_dir,
            "e62_dir": e62_dir,
            "out_dir": out_dir,
            "bundle_dir": bundle_dir,
        },
        "a1": {"v": a1_v, "cf": a1_cf, "crop_v": a1_crop_v, "crop_cf": a1_crop_cf},
        "a2": {"rec": syn_rec, "crop": syn_crop, "px_front": syn_px_front,
               "px_back": syn_px_back, "area_ratio": syn_area_ratio},
        "a3": {"v": a3_v, "cf": a3_cf, "im_v": a3_im_v, "crop_box": crop_box_4026},
        "a4": {"v": a4_v, "cf": a4_cf, "crop_v": a4_crop_v, "crop_cf": a4_crop_cf},
        "a5": {"rec": a5_rec, "im_v": a5_im_v, "im_mir": a5_im_mir},
        "b1": {
            "dir": b1_dir, "cot": b1_cot, "traj": b1_tr, "trajtwin": b1_tt, "crop": b1_crop,
            "p_dir": probs_from_lp(b1_dir["lp_sum"])[b1_dir["gold"]],
            "p_cot": probs_from_lp(b1_cot["lp_sum"])[b1_cot["gold"]],
            "pred_dir": argmax_lp(b1_dir["lp_sum"]),
            "pred_cot": argmax_lp(b1_cot["lp_sum"]),
        },
        "b2": {
            "dir": b2_dir, "cot": b2_cot, "traj": b2_tr, "e54": b2_e54, "crop": b2_crop,
            "p_dir": probs_from_lp(b2_dir["lp_sum"])[b2_dir["gold"]],
            "p_cot": probs_from_lp(b2_cot["lp_sum"])[b2_cot["gold"]],
            "pred_dir": argmax_lp(b2_dir["lp_sum"]),
            "pred_cot": argmax_lp(b2_cot["lp_sum"]),
        },
        "b3": {
            "item": b3_it_claude, "rv": b3_rv, "rcf": b3_rcf, "peers": b3_peers,
            "crop_v": b3_crop_v, "crop_cf": b3_crop_cf,
        },
    }


def add_rbox(ax: plt.Axes, x: float, y: float, w: float, h: float,
             fc: str, ec: str, lw: float = 0.75, radius: float = 0.05,
             ls: str = "-", zorder: int = 1) -> mpatches.FancyBboxPatch:
    """Draw a rounded rectangle in physical inch coordinates on the canvas axis."""
    patch = mpatches.FancyBboxPatch(
        (x, y), w, h,
        boxstyle=f"round,pad=0.0,rounding_size={radius}",
        facecolor=fc, edgecolor=ec, linewidth=lw, linestyle=ls,
        zorder=zorder, clip_on=False,
    )
    ax.add_patch(patch)
    return patch


def add_img_inset(fig: plt.Figure, fig_w: float, fig_h: float,
                  x: float, y: float, w: float, h: float,
                  img: np.ndarray, inset_boxes: list[tuple[float, float, float, float]],
                  edge_col: str = "#B2BAC6", lw: float = 0.7) -> plt.Axes:
    """Place an image inset at exact physical inch coordinates [x, y, w, h]."""
    inset_boxes.append((x, y, x + w, y + h))
    iax = fig.add_axes([x / fig_w, y / fig_h, w / fig_w, h / fig_h], zorder=4)
    iax.imshow(img)
    iax.set_xticks([])
    iax.set_yticks([])
    for sp in iax.spines.values():
        sp.set_visible(True)
        sp.set_edgecolor(edge_col)
        sp.set_linewidth(lw)
    return iax


def draw_prob_bar(ax: plt.Axes, x0: float, ymid: float, w: float, bh: float,
                  p_val: float, val_x: float, bold_val: bool = False) -> None:
    """Draw a horizontal P_j(gold) bar with a 0.5 threshold reference and numeric label."""
    ax.add_patch(mpatches.Rectangle(
        (x0, ymid - bh / 2), w, bh,
        facecolor="#E6E9EF", edgecolor="#C5CCD8", linewidth=0.45, zorder=2,
    ))
    fill_w = max(0.012, p_val * w)
    bcol = COL_ICE if p_val >= 0.5 else COL_CORAL
    bedge = COL_ICE_DARK if p_val >= 0.5 else COL_CORAL_DARK
    ax.add_patch(mpatches.Rectangle(
        (x0, ymid - bh / 2), fill_w, bh,
        facecolor=bcol, edgecolor=bedge, linewidth=0.5, zorder=3,
    ))
    ax.text(
        val_x, ymid, f"{p_val:.2f}",
        fontsize=6.05, fontweight="bold" if bold_val else "normal",
        ha="left", va="center", color=COL_BAD if p_val < 0.5 else COL_ICE_DARK, zorder=3,
    )


def build_figure(data: dict[str, Any], fig_dir: Path) -> tuple[list[Path], dict[str, Any]]:
    """Render the full-width one-page qualitative figure and run automated layout checks."""
    W_IN, H_IN = 6.875, 7.68
    fig = plt.figure(figsize=(W_IN, H_IN))
    ax = fig.add_axes([0.0, 0.0, 1.0, 1.0], zorder=0)
    ax.set_xlim(0.0, W_IN)
    ax.set_ylim(0.0, H_IN)
    ax.set_axis_off()

    inset_boxes: list[tuple[float, float, float, float]] = []
    # Track (text_artist, container_rect_inches [x0, y0, x1, y1]) for automated overflow verification
    bounded_texts: list[tuple[matplotlib.text.Text, tuple[float, float, float, float]]] = []

    def btext(x: float, y: float, s: str, box: tuple[float, float, float, float], **kw: Any) -> matplotlib.text.Text:
        t = ax.text(x, y, s, **kw)
        bounded_texts.append((t, box))
        return t

    full_canvas = (0.0, 0.0, W_IN, H_IN)

    # =========================================================================
    # PART (a): STIMULUS GALLERY (y = 5.18 .. 7.64)
    # =========================================================================
    btext(
        0.04, 7.63,
        "(a) Stimulus gallery across the five evaluation sets",
        full_canvas,
        fontsize=8.0, fontweight="bold", ha="left", va="top", color=COL_TEXT,
    )
    btext(
        W_IN - 0.04, 7.615,
        "$v$: original   $v_{\\mathrm{cf}}$ / $v_{\\mathrm{mir}}$: counterfactual twin   (binary A/B)",
        full_canvas,
        fontsize=6.1, ha="right", va="top", color=COL_MUTED,
    )

    y_card0, h_card = 5.16, 2.28
    card_specs = [
        ("a1", 0.035, 1.515, "Cue-factorial twins"),
        ("a2", 1.605, 1.055, "Synthetic Depth"),
        ("a3", 2.715, 1.135, "Hard occlusion"),
        ("a4", 3.905, 1.445, "COCO occl. twins"),
        ("a5", 5.405, 1.435, "Mirrored CV-Bench"),
    ]
    card_rect: dict[str, tuple[float, float, float, float]] = {}

    for key, cx0, cw, title in card_specs:
        cbox = (cx0, y_card0, cx0 + cw, y_card0 + h_card)
        card_rect[key] = cbox
        add_rbox(ax, cx0, y_card0, cw, h_card, COL_CARD_BG, COL_CARD_EDGE, lw=0.75, radius=0.055, zorder=1)
        add_rbox(ax, cx0, y_card0 + h_card - 0.215, cw, 0.215, "#EAEFF6", COL_CARD_EDGE, lw=0.65, radius=0.055, zorder=2)
        btext(
            cx0 + cw / 2, y_card0 + h_card - 0.108, title, cbox,
            fontsize=6.85, fontweight="bold", ha="center", va="center", color=COL_TEXT, zorder=3,
        )

    im_y = 6.55

    # --- (a.1) Cue-factorial twins ---
    cx0, cw = 0.035, 1.515
    cbox = card_rect["a1"]
    ts = 0.61
    im1_x, im2_x = cx0 + 0.072, cx0 + cw - 0.072 - ts
    add_img_inset(fig, W_IN, H_IN, im1_x, im_y, ts, ts, data["a1"]["crop_v"], inset_boxes)
    add_img_inset(fig, W_IN, H_IN, im2_x, im_y, ts, ts, data["a1"]["crop_cf"], inset_boxes)
    btext(cx0 + cw / 2, im_y + ts / 2, "\u21c4", cbox, fontsize=7.6, fontweight="bold",
          ha="center", va="center", color=COL_SLATE_DARK, zorder=5)
    btext(im1_x + ts / 2, 6.45, "$v$: (A) yes", cbox, fontsize=6.05, fontweight="bold",
          ha="center", va="center", color=COL_ICE_DARK)
    btext(im2_x + ts / 2, 6.45, "$v_{\\mathrm{cf}}$: (B) no", cbox, fontsize=6.05, fontweight="bold",
          ha="center", va="center", color=COL_CORAL_DARK)

    qbox_a1 = (cx0 + 0.045, 5.86, cx0 + cw - 0.045, 6.35)
    add_rbox(ax, qbox_a1[0], qbox_a1[1], qbox_a1[2] - qbox_a1[0], qbox_a1[3] - qbox_a1[1],
             COL_SUBBOX_BG, COL_SUBBOX_EDGE, lw=0.6, radius=0.035, zorder=2)
    q_a1 = f"Q: \u201c{data['a1']['v']['question']}\u201d"
    btext(cx0 + cw / 2, 6.105, "\n".join(textwrap.wrap(q_a1, 27)) + "\n(A) yes   (B) no", qbox_a1,
          fontsize=6.0, ha="center", va="center", color=COL_TEXT, linespacing=1.12, zorder=3)

    nbox_a1 = (cx0 + 0.045, 5.33, cx0 + cw - 0.045, 5.80)
    add_rbox(ax, nbox_a1[0], nbox_a1[1], nbox_a1[2] - nbox_a1[0], nbox_a1[3] - nbox_a1[1],
             TINT_ICE, COL_ICE, lw=0.7, radius=0.035, zorder=2)
    btext(cx0 + cw / 2, 5.565,
          "Same geometry, IDs swapped;\n2\u00d72 size\u00d7height balanced\n\u2192 occlusion is only cue",
          nbox_a1, fontsize=6.0, fontweight="bold", ha="center", va="center",
          color=COL_ICE_DARK, linespacing=1.12, zorder=3)
    btext(cx0 + cw / 2, 5.235, "occ_cues \u00b7 cue018, cue018_cf", cbox,
          fontsize=6.0, ha="center", va="center", color=COL_MUTED, zorder=3)

    # --- (a.2) Synthetic Depth ---
    cx0, cw = 1.605, 1.055
    cbox = card_rect["a2"]
    ts2 = 0.61
    add_img_inset(fig, W_IN, H_IN, cx0 + (cw - ts2) / 2, im_y, ts2, ts2, data["a2"]["crop"], inset_boxes)
    btext(cx0 + cw / 2, 6.45, "Gold: (A) yes", cbox, fontsize=6.05, fontweight="bold",
          ha="center", va="center", color=COL_ICE_DARK)

    qbox_a2 = (cx0 + 0.04, 5.86, cx0 + cw - 0.04, 6.35)
    add_rbox(ax, qbox_a2[0], qbox_a2[1], qbox_a2[2] - qbox_a2[0], qbox_a2[3] - qbox_a2[1],
             COL_SUBBOX_BG, COL_SUBBOX_EDGE, lw=0.6, radius=0.035, zorder=2)
    q_a2 = f"Q: \u201c{data['a2']['rec']['question']}\u201d"
    btext(cx0 + cw / 2, 6.105, "\n".join(textwrap.wrap(q_a2, 18)) + "\n(A) yes  (B) no", qbox_a2,
          fontsize=6.0, ha="center", va="center", color=COL_TEXT, linespacing=1.08, zorder=3)

    nbox_a2 = (cx0 + 0.04, 5.33, cx0 + cw - 0.04, 5.80)
    add_rbox(ax, nbox_a2[0], nbox_a2[1], nbox_a2[2] - nbox_a2[0], nbox_a2[3] - nbox_a2[1],
             TINT_PEACH, COL_PEACH, lw=0.75, radius=0.035, zorder=2)
    btext(cx0 + cw / 2, 5.565,
          f"Size confound:\nfront drawn larger\n(98.6%; {data['a2']['area_ratio']:.1f}\u00d7 here)",
          nbox_a2, fontsize=6.0, fontweight="bold", ha="center", va="center",
          color=COL_PEACH_DARK, linespacing=1.11, zorder=3)
    btext(cx0 + cw / 2, 5.235, "syn_depth \u00b7 syn_depth_10", cbox,
          fontsize=6.0, ha="center", va="center", color=COL_MUTED, zorder=3)

    # --- (a.3) Hard occlusion (closed models) ---
    cx0, cw = 2.715, 1.135
    cbox = card_rect["a3"]
    ts3 = 0.61
    iax_a3 = add_img_inset(fig, W_IN, H_IN, cx0 + (cw - ts3) / 2, im_y, ts3, ts3, data["a3"]["im_v"], inset_boxes)
    sx0, sy0, side_crop = data["a3"]["crop_box"]
    iax_a3.add_patch(mpatches.Rectangle(
        (sx0, sy0), side_crop, side_crop,
        fill=False, edgecolor=COL_CORAL_DARK, linewidth=0.95, linestyle=(0, (2.2, 1.5)),
    ))
    btext(cx0 + cw / 2, 6.45, "Gold: (A) yes [L4]", cbox, fontsize=6.05, fontweight="bold",
          ha="center", va="center", color=COL_SLATE_DARK)

    qbox_a3 = (cx0 + 0.04, 5.86, cx0 + cw - 0.04, 6.35)
    add_rbox(ax, qbox_a3[0], qbox_a3[1], qbox_a3[2] - qbox_a3[0], qbox_a3[3] - qbox_a3[1],
             COL_SUBBOX_BG, COL_SUBBOX_EDGE, lw=0.6, radius=0.035, zorder=2)
    q_a3 = f"Q: \u201c{data['a3']['v']['question']}\u201d"
    btext(cx0 + cw / 2, 6.105, "\n".join(textwrap.wrap(q_a3, 20)) + "\n(A) yes  (B) no", qbox_a3,
          fontsize=6.0, ha="center", va="center", color=COL_TEXT, linespacing=1.08, zorder=3)

    nbox_a3 = (cx0 + 0.04, 5.33, cx0 + cw - 0.04, 5.80)
    add_rbox(ax, nbox_a3[0], nbox_a3[1], nbox_a3[2] - nbox_a3[0], nbox_a3[3] - nbox_a3[1],
             TINT_SLATE, COL_SLATE, lw=0.75, radius=0.035, zorder=2)
    btext(cx0 + cw / 2, 5.565,
          "3.8% overlap, 5 dist.,\nlow contrast + noise;\nfront \u2264 back (see b.3)",
          nbox_a3, fontsize=6.0, fontweight="bold", ha="center", va="center",
          color=COL_SLATE_DARK, linespacing=1.11, zorder=3)
    btext(cx0 + cw / 2, 5.235, "occ_hard \u00b7 occl4026", cbox,
          fontsize=6.0, ha="center", va="center", color=COL_MUTED, zorder=3)

    # --- (a.4) COCO occlusion twin pair ---
    cx0, cw = 3.905, 1.445
    cbox = card_rect["a4"]
    ts4 = 0.585
    im4_1, im4_2 = cx0 + 0.065, cx0 + cw - 0.065 - ts4
    add_img_inset(fig, W_IN, H_IN, im4_1, im_y + 0.012, ts4, ts4, data["a4"]["crop_v"], inset_boxes)
    add_img_inset(fig, W_IN, H_IN, im4_2, im_y + 0.012, ts4, ts4, data["a4"]["crop_cf"], inset_boxes)
    btext(cx0 + cw / 2, im_y + 0.012 + ts4 / 2, "\u21c4", cbox, fontsize=7.5, fontweight="bold",
          ha="center", va="center", color=COL_SLATE_DARK, zorder=5)
    btext(im4_1 + ts4 / 2, 6.45, "$v$: (A) yes", cbox, fontsize=6.05, fontweight="bold",
          ha="center", va="center", color=COL_ICE_DARK)
    btext(im4_2 + ts4 / 2, 6.45, "$v_{\\mathrm{cf}}$: (B) no", cbox, fontsize=6.05, fontweight="bold",
          ha="center", va="center", color=COL_CORAL_DARK)

    qbox_a4 = (cx0 + 0.045, 5.86, cx0 + cw - 0.045, 6.35)
    add_rbox(ax, qbox_a4[0], qbox_a4[1], qbox_a4[2] - qbox_a4[0], qbox_a4[3] - qbox_a4[1],
             COL_SUBBOX_BG, COL_SUBBOX_EDGE, lw=0.6, radius=0.035, zorder=2)
    q_a4_line = data["a4"]["v"]["question"].splitlines()[0]
    btext(cx0 + cw / 2, 6.105, "\n".join(textwrap.wrap(f"Q: \u201c{q_a4_line}\u201d", 25)) + "\n(A) yes   (B) no", qbox_a4,
          fontsize=6.0, ha="center", va="center", color=COL_TEXT, linespacing=1.12, zorder=3)

    nbox_a4 = (cx0 + 0.045, 5.33, cx0 + cw - 0.045, 5.80)
    add_rbox(ax, nbox_a4[0], nbox_a4[1], nbox_a4[2] - nbox_a4[0], nbox_a4[3] - nbox_a4[1],
             TINT_ICE, COL_ICE, lw=0.7, radius=0.035, zorder=2)
    btext(cx0 + cw / 2, 5.565,
          "Segmented COCO cutouts;\nonly paste order swapped\n(cat over vs. under laptop)",
          nbox_a4, fontsize=6.0, fontweight="bold", ha="center", va="center",
          color=COL_ICE_DARK, linespacing=1.12, zorder=3)
    btext(cx0 + cw / 2, 5.235, "e62_coco \u00b7 e62p016a/b", cbox,
          fontsize=6.0, ha="center", va="center", color=COL_MUTED, zorder=3)

    # --- (a.5) Mirrored CV-Bench left/right pair ---
    cx0, cw = 5.405, 1.435
    cbox = card_rect["a5"]
    w5 = 0.595
    h5 = w5 * (427.0 / 640.0)
    im5_y = im_y + (ts - h5) / 2
    im5_1, im5_2 = cx0 + 0.055, cx0 + cw - 0.055 - w5
    add_img_inset(fig, W_IN, H_IN, im5_1, im5_y, w5, h5, data["a5"]["im_v"], inset_boxes)
    add_img_inset(fig, W_IN, H_IN, im5_2, im5_y, w5, h5, data["a5"]["im_mir"], inset_boxes)
    btext(cx0 + cw / 2, im5_y + h5 / 2, "\u21c4", cbox, fontsize=7.4, fontweight="bold",
          ha="center", va="center", color=COL_SLATE_DARK, zorder=5)
    btext(im5_1 + w5 / 2, 6.45, "$v$: (A) left", cbox, fontsize=6.05, fontweight="bold",
          ha="center", va="center", color=COL_ICE_DARK)
    btext(im5_2 + w5 / 2, 6.45, "$v_{\\mathrm{mir}}$: (B) right", cbox, fontsize=6.05, fontweight="bold",
          ha="center", va="center", color=COL_CORAL_DARK)

    qbox_a5 = (cx0 + 0.045, 5.86, cx0 + cw - 0.045, 6.35)
    add_rbox(ax, qbox_a5[0], qbox_a5[1], qbox_a5[2] - qbox_a5[0], qbox_a5[3] - qbox_a5[1],
             COL_SUBBOX_BG, COL_SUBBOX_EDGE, lw=0.6, radius=0.035, zorder=2)
    q_a5_full = data["a5"]["rec"]["question"].splitlines()[0]
    assert "where is the bear located with respect to the bird?" in q_a5_full
    q_a5_elided = "Q: \u201c...where is the bear located with respect to the bird? ...\u201d"
    btext(cx0 + cw / 2, 6.105, "\n".join(textwrap.wrap(q_a5_elided, 25)) + "\n(A) left   (B) right", qbox_a5,
          fontsize=6.0, ha="center", va="center", color=COL_TEXT, linespacing=1.10, zorder=3)

    nbox_a5 = (cx0 + 0.045, 5.33, cx0 + cw - 0.045, 5.80)
    add_rbox(ax, nbox_a5[0], nbox_a5[1], nbox_a5[2] - nbox_a5[0], nbox_a5[3] - nbox_a5[1],
             TINT_ICE, COL_ICE, lw=0.7, radius=0.035, zorder=2)
    btext(cx0 + cw / 2, 5.565,
          "Natural photo horizontally\nmirrored; 2D left/right\ntruth flips (A \u2194 B)",
          nbox_a5, fontsize=6.0, fontweight="bold", ha="center", va="center",
          color=COL_ICE_DARK, linespacing=1.12, zorder=3)
    btext(cx0 + cw / 2, 5.235, "cvb \u00b7 cvb_Relation_1276_COCO", cbox,
          fontsize=6.0, ha="center", va="center", color=COL_MUTED, zorder=3)

    # =========================================================================
    # PART (b): THREE REAL VERBATIM ANNOTATED RATIONALES (y = 0.04 .. 5.06)
    # =========================================================================
    btext(
        0.04, 5.06,
        "(b) Verbatim rationales and forced readouts across failure stages",
        full_canvas,
        fontsize=8.0, fontweight="bold", ha="left", va="top", color=COL_TEXT,
    )
    btext(
        W_IN - 0.04, 5.045,
        "exact saved outputs; $P_j(\\mathrm{gold})$ measured at forced \u201cAnswer: (\u201d",
        full_canvas,
        fontsize=6.1, ha="right", va="top", color=COL_MUTED,
    )

    bx0, bw = 0.035, W_IN - 0.070  # 6.805 in
    rx0 = bx0 + 1.43
    tx_tag = rx0 + 0.035
    tx_sent = rx0 + 0.44
    bar_x0 = bx0 + bw - 1.15
    bar_w = 0.68
    val_x = bar_x0 + bar_w + 0.055

    # -------------------------------------------------------------------------
    # Card (b.1): Stage 1 Error Onset & Trajectory (Qwen3-VL-8B, cue320)
    # y = 2.81 .. 4.89 (height = 2.08 in)
    # -------------------------------------------------------------------------
    y1_0, h1 = 2.81, 2.08
    box_b1 = (bx0, y1_0, bx0 + bw, y1_0 + h1)
    hdr_b1 = (bx0, y1_0 + h1 - 0.215, bx0 + bw, y1_0 + h1)
    add_rbox(ax, bx0, y1_0, bw, h1, COL_CARD_BG, COL_CARD_EDGE, lw=0.8, radius=0.055, zorder=1)
    add_rbox(ax, bx0, y1_0 + h1 - 0.215, bw, 0.215, "#EAEFF6", COL_CARD_EDGE, lw=0.65, radius=0.055, zorder=2)
    btext(
        bx0 + 0.075, y1_0 + h1 - 0.108,
        "(b.1) Stage 1 commit flip at $j^*$ & Stage 2 lock-in",
        hdr_b1, fontsize=6.9, fontweight="bold", ha="left", va="center", color=COL_TEXT, zorder=3,
    )
    btext(
        bx0 + bw - 0.075, y1_0 + h1 - 0.108,
        "Qwen3-VL-8B \u00b7 occ_cues (D+C\u2212) \u00b7 item: cue320 \u00b7 Gold: (B) no",
        hdr_b1, fontsize=6.05, ha="right", va="center", color=COL_MUTED, zorder=3,
    )

    # Left column of (b.1)
    b1_im_s = 0.72
    b1_im_x = bx0 + 0.34
    b1_im_y = y1_0 + 1.05
    iax_b1 = add_img_inset(fig, W_IN, H_IN, b1_im_x, b1_im_y, b1_im_s, b1_im_s, data["b1"]["crop"], inset_boxes)
    tj_b1 = find_t_junction(data["b1"]["crop"])
    if tj_b1 is not None:
        side_b1 = data["b1"]["crop"].shape[0]
        tx, ty = tj_b1
        iax_b1.annotate(
            "", xy=(tx, ty), xytext=(max(8.0, tx - 0.22 * side_b1), max(8.0, ty - 0.22 * side_b1)),
            arrowprops={"arrowstyle": "-|>", "color": COL_TEXT, "lw": 0.9, "mutation_scale": 6.0},
        )
    btext(b1_im_x + b1_im_s / 2, b1_im_y - 0.082, "True front: green square", box_b1,
          fontsize=6.0, fontweight="bold", ha="center", va="center", color=COL_SLATE_DARK)

    qbox_b1 = (bx0 + 0.055, y1_0 + 0.455, bx0 + 1.365, y1_0 + 0.895)
    add_rbox(ax, qbox_b1[0], qbox_b1[1], qbox_b1[2] - qbox_b1[0], qbox_b1[3] - qbox_b1[1],
             COL_SUBBOX_BG, COL_SUBBOX_EDGE, lw=0.6, radius=0.035, zorder=2)
    b1_q_line = data["b1"]["traj"]["question"].splitlines()[0]
    btext(0.5 * (qbox_b1[0] + qbox_b1[2]), 0.5 * (qbox_b1[1] + qbox_b1[3]),
          "\n".join(textwrap.wrap(f"Q: \u201c{b1_q_line}\u201d", 23)) + "\n(A) yes   (B) no", qbox_b1,
          fontsize=6.0, ha="center", va="center", color=COL_TEXT, linespacing=1.10, zorder=3)

    dbox_b1 = (bx0 + 0.055, y1_0 + 0.255, bx0 + 1.365, y1_0 + 0.425)
    add_rbox(ax, dbox_b1[0], dbox_b1[1], dbox_b1[2] - dbox_b1[0], dbox_b1[3] - dbox_b1[1],
             TINT_GOOD, EDGE_GOOD, lw=0.65, radius=0.03, zorder=2)
    btext(0.5 * (dbox_b1[0] + dbox_b1[2]), 0.5 * (dbox_b1[1] + dbox_b1[3]),
          f"Direct: ({data['b1']['pred_dir']}) no \u2713 (P={data['b1']['p_dir']:.2f})", dbox_b1,
          fontsize=6.05, fontweight="bold", ha="center", va="center", color=COL_GOOD, zorder=3)

    cbox_b1 = (bx0 + 0.055, y1_0 + 0.06, bx0 + 1.365, y1_0 + 0.23)
    add_rbox(ax, cbox_b1[0], cbox_b1[1], cbox_b1[2] - cbox_b1[0], cbox_b1[3] - cbox_b1[1],
             TINT_CORAL, COL_CORAL, lw=0.7, radius=0.03, zorder=2)
    btext(0.5 * (cbox_b1[0] + cbox_b1[2]), 0.5 * (cbox_b1[1] + cbox_b1[3]),
          f"CoT: ({data['b1']['pred_cot']}) yes \u2717 (P={data['b1']['p_cot']:.2f})", cbox_b1,
          fontsize=6.05, fontweight="bold", ha="center", va="center", color=COL_BAD, zorder=3)

    # Right sentence-by-sentence trajectory table in (b.1)
    hdr_y = y1_0 + h1 - 0.305
    btext(tx_tag, hdr_y, "Step", box_b1, fontsize=6.05, fontweight="bold",
          ha="left", va="center", color=COL_SLATE_DARK)
    btext(tx_sent, hdr_y, "Verbatim sentence prefix $R_{1:j}$", box_b1, fontsize=6.05, fontweight="bold",
          ha="left", va="center", color=COL_SLATE_DARK)
    btext(bar_x0 + bar_w / 2, hdr_y, "$P_j(\\mathrm{gold})$", box_b1, fontsize=6.05, fontweight="bold",
          ha="center", va="center", color=COL_SLATE_DARK)

    b1_sents = data["b1"]["traj"]["sentences"]
    b1_pg = data["b1"]["traj"]["p_gold"]
    j_star = data["b1"]["traj"]["commit_idx"]

    rows_b1: list[tuple[str, str, float, bool]] = [
        ("$j{=}0$", "Let's think step by step.", float(b1_pg[0]), False),
    ]
    for i, s in enumerate(b1_sents):
        lbl = f"$s_{{{i + 1}}}$" + (" $j^*$" if i == j_star else "")
        rows_b1.append((lbl, s, float(b1_pg[i + 1]), i == j_star))

    wrapped_rows_b1 = []
    for lbl, s, pg_val, is_jstar in rows_b1:
        lines = textwrap.wrap(s, width=70)
        wrapped_rows_b1.append((lbl, "\n".join(lines), len(lines), pg_val, is_jstar))

    top_cursor = y1_0 + h1 - 0.395
    bot_limit = y1_0 + 0.21
    avail_h = top_cursor - bot_limit
    weights = [1.0 if n_ln == 1 else 1.58 for _, _, n_ln, _, _ in wrapped_rows_b1]
    unit_h = avail_h / sum(weights)

    ax.plot([bar_x0 + 0.5 * bar_w, bar_x0 + 0.5 * bar_w], [bot_limit + 0.01, top_cursor],
            color="#8D95A2", linestyle=":", linewidth=0.75, zorder=2)

    sent_col_box_b1 = (tx_sent - 0.02, bot_limit - 0.02, bar_x0 - 0.06, top_cursor + 0.02)
    cur_y = top_cursor
    for idx_r, (lbl, wtext, n_ln, pg_val, is_jstar) in enumerate(wrapped_rows_b1):
        rh = weights[idx_r] * unit_h
        ymid = cur_y - rh / 2.0
        if is_jstar:
            add_rbox(ax, rx0, cur_y - rh + 0.008, bw - 1.48, rh - 0.016,
                     TINT_CORAL, COL_CORAL, lw=0.85, radius=0.025, zorder=2)
        elif idx_r % 2 == 1:
            add_rbox(ax, rx0, cur_y - rh + 0.008, bw - 1.48, rh - 0.016,
                     "#F2F4F8", "none", lw=0.0, radius=0.02, zorder=1)

        btext(
            tx_tag, ymid, lbl, box_b1,
            fontsize=6.0, fontweight="bold" if is_jstar else "normal",
            ha="left", va="center", color=COL_BAD if is_jstar else COL_SLATE_DARK, zorder=3,
        )
        btext(
            tx_sent, ymid, wtext, sent_col_box_b1,
            fontsize=6.05, fontweight="bold" if is_jstar else "normal",
            style="italic" if idx_r == 0 else "normal",
            ha="left", va="center", color=COL_TEXT, linespacing=1.05, zorder=3,
        )
        draw_prob_bar(ax, bar_x0, ymid, bar_w, min(0.085, rh * 0.60), pg_val, val_x, bold_val=is_jstar)
        cur_y -= rh

    tt_own = data["b1"]["trajtwin"]["p_gold_own"]
    tt_twin = data["b1"]["trajtwin"]["p_gold_twin"]
    pre_ie_min = 100.0 * min(tt_own[k] - tt_twin[k] for k in range(j_star + 1))
    pre_ie_max = 100.0 * max(tt_own[k] - tt_twin[k] for k in range(j_star + 1))
    post_ie_max = 100.0 * max(abs(tt_own[k] - tt_twin[k]) for k in range(j_star + 1, len(tt_own)))
    btext(
        rx0 + 0.02, y1_0 + 0.105,
        f"\u25b8 Consensus audit at $j^*=s_5$: INCORRECT. Prefix twin image effect: "
        f"{pre_ie_min:.0f}\u2013{pre_ie_max:.0f} pts before $j^*$ \u2192 {post_ie_max:.1f} pts after $j^*$.",
        box_b1, fontsize=6.0, ha="left", va="center", color=COL_MUTED, zorder=3,
    )

    # -------------------------------------------------------------------------
    # Card (b.2): Stage 3 Converse Mis-Mapping (Molmo-7B-D, cue348_behind)
    # y = 1.43 .. 2.73 (height = 1.30 in)
    # -------------------------------------------------------------------------
    y2_0, h2 = 1.43, 1.30
    box_b2 = (bx0, y2_0, bx0 + bw, y2_0 + h2)
    hdr_b2 = (bx0, y2_0 + h2 - 0.215, bx0 + bw, y2_0 + h2)
    add_rbox(ax, bx0, y2_0, bw, h2, COL_CARD_BG, COL_CARD_EDGE, lw=0.8, radius=0.055, zorder=1)
    add_rbox(ax, bx0, y2_0 + h2 - 0.215, bw, 0.215, "#EAEFF6", COL_CARD_EDGE, lw=0.65, radius=0.055, zorder=2)
    btext(
        bx0 + 0.075, y2_0 + h2 - 0.108,
        "(b.2) Stage 3 converse mis-mapping after a true written relation",
        hdr_b2, fontsize=6.9, fontweight="bold", ha="left", va="center", color=COL_TEXT, zorder=3,
    )
    btext(
        bx0 + bw - 0.075, y2_0 + h2 - 0.108,
        "Molmo-7B-D \u00b7 occ_cues (D+C\u2212, converse) \u00b7 cue348_behind \u00b7 Gold: (A) yes",
        hdr_b2, fontsize=6.05, ha="right", va="center", color=COL_MUTED, zorder=3,
    )

    # Left column of (b.2)
    b2_im_s = 0.53
    add_img_inset(fig, W_IN, H_IN, bx0 + 0.06, y2_0 + 0.49, b2_im_s, b2_im_s, data["b2"]["crop"], inset_boxes)
    dbox_b2 = (bx0 + 0.625, y2_0 + 0.775, bx0 + 1.365, y2_0 + 1.015)
    add_rbox(ax, dbox_b2[0], dbox_b2[1], dbox_b2[2] - dbox_b2[0], dbox_b2[3] - dbox_b2[1],
             TINT_GOOD, EDGE_GOOD, lw=0.6, radius=0.028, zorder=2)
    btext(0.5 * (dbox_b2[0] + dbox_b2[2]), 0.5 * (dbox_b2[1] + dbox_b2[3]),
          f"Direct: ({data['b2']['pred_dir']}) yes \u2713\nP(gold)={data['b2']['p_dir']:.2f}", dbox_b2,
          fontsize=6.0, fontweight="bold", ha="center", va="center", color=COL_GOOD, linespacing=1.08, zorder=3)

    cbox_b2 = (bx0 + 0.625, y2_0 + 0.495, bx0 + 1.365, y2_0 + 0.735)
    add_rbox(ax, cbox_b2[0], cbox_b2[1], cbox_b2[2] - cbox_b2[0], cbox_b2[3] - cbox_b2[1],
             TINT_CORAL, COL_CORAL, lw=0.65, radius=0.028, zorder=2)
    btext(0.5 * (cbox_b2[0] + cbox_b2[2]), 0.5 * (cbox_b2[1] + cbox_b2[3]),
          f"CoT: ({data['b2']['pred_cot']}) no \u2717\nP(gold)={data['b2']['p_cot']:.2f}", cbox_b2,
          fontsize=6.0, fontweight="bold", ha="center", va="center", color=COL_BAD, linespacing=1.08, zorder=3)

    qbox_b2 = (bx0 + 0.055, y2_0 + 0.055, bx0 + 1.365, y2_0 + 0.445)
    add_rbox(ax, qbox_b2[0], qbox_b2[1], qbox_b2[2] - qbox_b2[0], qbox_b2[3] - qbox_b2[1],
             COL_SUBBOX_BG, COL_SUBBOX_EDGE, lw=0.6, radius=0.03, zorder=2)
    b2_q_line = data["b2"]["traj"]["question"].splitlines()[0]
    btext(0.5 * (qbox_b2[0] + qbox_b2[2]), 0.5 * (qbox_b2[1] + qbox_b2[3]),
          "\n".join(textwrap.wrap(f"Q: \u201c{b2_q_line}\u201d", 23)) + "\n(A) yes   (B) no", qbox_b2,
          fontsize=6.0, ha="center", va="center", color=COL_TEXT, linespacing=1.08, zorder=3)

    # Right aligned rows of (b.2) sharing the exact Step / Verbatim / P_j(gold) column geometry
    b2_sents = data["b2"]["traj"]["sentences"]
    b2_pg = data["b2"]["traj"]["p_gold"]
    top_b2 = y2_0 + h2 - 0.255
    bot_b2 = y2_0 + 0.205
    ax.plot([bar_x0 + 0.5 * bar_w, bar_x0 + 0.5 * bar_w], [bot_b2, top_b2],
            color="#8D95A2", linestyle=":", linewidth=0.75, zorder=2)

    rows_b2 = [
        ("$s_1$", b2_sents[0], float(b2_pg[1]), "none"),
        ("$s_2$", b2_sents[1], float(b2_pg[2]), "true_body"),
        ("$s_3$", b2_sents[2], float(b2_pg[3]), "true_body"),
        ("$s_4$", "\n".join(textwrap.wrap(b2_sents[3], 68)), float(b2_pg[4]), "converse"),
        ("$s_{5..6}$", f"{b2_sents[4]} {b2_sents[5]}", float(b2_pg[6]), "concl"),
    ]
    row_h_b2 = [0.148, 0.148, 0.148, 0.238, 0.148]
    sent_col_box_b2 = (tx_sent - 0.02, bot_b2 - 0.03, bar_x0 - 0.06, top_b2 + 0.03)
    cy2 = top_b2
    for (lbl, txt_r, pv, style_tag), rh in zip(rows_b2, row_h_b2):
        ymid = cy2 - rh / 2.0
        if style_tag == "true_body":
            add_rbox(ax, rx0, cy2 - rh + 0.008, bw - 1.48, rh - 0.016,
                     TINT_ICE, COL_ICE, lw=0.8, radius=0.025, zorder=2)
            tcol = COL_ICE_DARK
        elif style_tag == "converse":
            add_rbox(ax, rx0, cy2 - rh + 0.008, bw - 1.48, rh - 0.016,
                     TINT_PEACH, COL_PEACH, lw=0.85, radius=0.025, zorder=2)
            tcol = COL_PEACH_DARK
        elif style_tag == "concl":
            add_rbox(ax, rx0, cy2 - rh + 0.008, bw - 1.48, rh - 0.016,
                     TINT_CORAL, COL_CORAL, lw=0.75, radius=0.025, zorder=2)
            tcol = COL_BAD
        else:
            tcol = COL_SLATE_DARK

        btext(tx_tag, ymid, lbl, box_b2, fontsize=6.0,
              fontweight="bold" if style_tag != "none" else "normal",
              ha="left", va="center", color=tcol, zorder=3)
        btext(tx_sent, ymid, txt_r, sent_col_box_b2, fontsize=6.05,
              fontweight="bold" if style_tag in ("true_body", "converse") else "normal",
              ha="left", va="center", color=COL_TEXT, linespacing=1.05, zorder=3)
        draw_prob_bar(ax, bar_x0, ymid, bar_w, min(0.085, rh * 0.58), pv, val_x,
                      bold_val=(style_tag in ("true_body", "converse")))
        cy2 -= rh

    btext(
        rx0 + 0.02, y2_0 + 0.10,
        "\u25b8 $s_{2..3}$ ($R_{\\mathrm{cut}}$ body) state the TRUE relation; $s_4$ fails to invert "
        "\u201cblue square in front\u201d \u21d2 \u201cred pentagon behind\u201d.",
        box_b2, fontsize=6.0, ha="left", va="center", color=COL_MUTED, zorder=3,
    )

    # -------------------------------------------------------------------------
    # Card (b.3): Closed-Model Answer-Step Re-Grounding (Claude Opus 5.5, occl4026_front)
    # y = 0.04 .. 1.35 (height = 1.31 in)
    # -------------------------------------------------------------------------
    y3_0, h3 = 0.04, 1.31
    box_b3 = (bx0, y3_0, bx0 + bw, y3_0 + h3)
    hdr_b3 = (bx0, y3_0 + h3 - 0.215, bx0 + bw, y3_0 + h3)
    add_rbox(ax, bx0, y3_0, bw, h3, COL_CARD_BG, COL_CARD_EDGE, lw=0.8, radius=0.055, zorder=1)
    add_rbox(ax, bx0, y3_0 + h3 - 0.215, bw, 0.215, "#EAEFF6", COL_CARD_EDGE, lw=0.65, radius=0.055, zorder=2)
    btext(
        bx0 + 0.075, y3_0 + h3 - 0.108,
        "(b.3) Answer-step re-grounding under fixed $R_{\\mathrm{cut}}(v)$ + twin $v_{\\mathrm{cf}}$",
        hdr_b3, fontsize=6.9, fontweight="bold", ha="left", va="center", color=COL_TEXT, zorder=3,
    )
    btext(
        bx0 + bw - 0.075, y3_0 + h3 - 0.108,
        "Claude Opus 5.5 \u00b7 occ_hard (L4) \u00b7 occl4026_front \u00b7 Gold: A\u2192B",
        hdr_b3, fontsize=6.05, ha="right", va="center", color=COL_MUTED, zorder=3,
    )

    # Left column of (b.3): zoomed overlap crops for v and v_cf (dashed box from a.3)
    z_s = 0.53
    zx1 = bx0 + 0.06
    zx2 = bx0 + 0.82
    zy = y3_0 + 0.49
    add_img_inset(fig, W_IN, H_IN, zx1, zy, z_s, z_s, data["b3"]["crop_v"], inset_boxes,
                  edge_col=COL_ICE_DARK, lw=0.8)
    add_img_inset(fig, W_IN, H_IN, zx2, zy, z_s, z_s, data["b3"]["crop_cf"], inset_boxes,
                  edge_col=COL_CORAL_DARK, lw=0.8)
    btext(bx0 + 0.705, zy + z_s / 2, "\u21c4", box_b3, fontsize=7.2, fontweight="bold",
          ha="center", va="center", color=COL_SLATE_DARK, zorder=5)
    btext(zx1 + z_s / 2, zy - 0.085, "$v$ (gold A)", box_b3, fontsize=6.0, fontweight="bold",
          ha="center", va="center", color=COL_ICE_DARK)
    btext(zx2 + z_s / 2, zy - 0.085, "$v_{\\mathrm{cf}}$ (gold B)", box_b3, fontsize=6.0, fontweight="bold",
          ha="center", va="center", color=COL_CORAL_DARK)
    btext(bx0 + 0.705, y3_0 + 0.175,
          "Target pair zoom from (a.3)\nQ: same [(A) yes / (B) no]",
          (bx0 + 0.04, y3_0 + 0.03, bx0 + 1.38, y3_0 + 0.33),
          fontsize=6.0, ha="center", va="center", color=COL_MUTED, linespacing=1.08, zorder=3)

    # Right column of (b.3): Fixed R_cut(v) subbox + Turn-2 answer subbox
    rw3 = bw - 1.48
    rcut_box = (rx0, y3_0 + 0.59, rx0 + rw3, y3_0 + 1.045)
    add_rbox(ax, rcut_box[0], rcut_box[1], rw3, rcut_box[3] - rcut_box[1],
             TINT_SLATE, COL_SLATE, lw=0.75, radius=0.03, zorder=2)
    b3_body_one_line = " ".join(ln.strip() for ln in data["b3"]["item"]["body_removed"].splitlines() if ln.strip())
    b3_lines = textwrap.wrap(f"Fixed assistant Rcut(v): \u201c{b3_body_one_line}\u201d", width=88)
    b3_lines[0] = b3_lines[0].replace("Rcut(v):", "$R_{\\mathrm{cut}}(v)$:")
    btext(rx0 + 0.06, 0.5 * (rcut_box[1] + rcut_box[3]), "\n".join(b3_lines), rcut_box,
          fontsize=6.0, ha="left", va="center", color=COL_TEXT, linespacing=1.08, zorder=3)

    t2_box = (rx0, y3_0 + 0.055, rx0 + rw3, y3_0 + 0.545)
    add_rbox(ax, t2_box[0], t2_box[1], rw3, t2_box[3] - t2_box[1],
             TINT_GOOD, EDGE_GOOD, lw=0.75, radius=0.03, zorder=2)

    rv_txt = data["b3"]["rv"]["text"].strip()
    rcf_txt = " ".join(data["b3"]["rcf"]["text"].strip().splitlines())
    rt_v = data["b3"]["rv"]["reasoning_tokens"]
    rt_cf = data["b3"]["rcf"]["reasoning_tokens"]
    peers = data["b3"]["peers"]

    btext(
        rx0 + 0.06, y3_0 + 0.435,
        f"Turn-2 + $v$: \u201c{rv_txt}\u201d \u2713 ({rt_v} tok.) | Peers: "
        f"Gemini-4-Argon {peers['gemini-4-argon']['removed_v']}\u2192{peers['gemini-4-argon']['removed_vcf']} \u2713, "
        f"GPT-6-Astra {peers['gpt-6-astra']['removed_v']}\u2192{peers['gpt-6-astra']['removed_vcf']} \u2713, "
        f"Qwen3.8-Max {peers['qwen3.8-max-0902']['removed_v']}\u2192{peers['qwen3.8-max-0902']['removed_vcf']} \u2717",
        t2_box, fontsize=6.0, fontweight="bold", ha="left", va="center", color=COL_TEXT, zorder=3,
    )
    rcf_lines = textwrap.wrap(
        f"Turn-2 + twin vcf (same fixed Rcut): \u201c{rcf_txt}\u201d \u2713 (gold = B, {rt_cf} reas. tok.)",
        width=88,
    )
    rcf_lines[0] = rcf_lines[0].replace("twin vcf (same fixed Rcut):", "twin $v_{\\mathrm{cf}}$ (same fixed $R_{\\mathrm{cut}}$):")
    btext(
        rx0 + 0.06, y3_0 + 0.205, "\n".join(rcf_lines), t2_box,
        fontsize=6.0, fontweight="bold", ha="left", va="center", color=COL_GOOD, linespacing=1.08, zorder=3,
    )

    # =========================================================================
    # Automated typography, non-overlap, box-containment, and font-type checks
    # =========================================================================
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    dpi = fig.dpi

    min_pt = min(t.get_fontsize() for t in ax.texts)
    assert min_pt >= 6.0, f"Found font smaller than 6 pt: {min_pt}"

    bboxes_in: list[tuple[str, tuple[float, float, float, float]]] = []
    box_overflows: list[str] = []
    img_overlaps: list[str] = []

    for t, cbox_lim in bounded_texts:
        ext = t.get_window_extent(renderer)
        tx0, ty0, tx1, ty1 = ext.x0 / dpi, ext.y0 / dpi, ext.x1 / dpi, ext.y1 / dpi
        label_short = t.get_text().replace("\n", " ")[:36]
        bboxes_in.append((label_short, (tx0, ty0, tx1, ty1)))

        tol = 0.025  # 1.8 pt tolerance for glyph bearings
        if (tx0 < cbox_lim[0] - tol or tx1 > cbox_lim[2] + tol
                or ty0 < cbox_lim[1] - tol or ty1 > cbox_lim[3] + tol):
            box_overflows.append(
                f"{label_short!r} [{tx0:.3f},{ty0:.3f},{tx1:.3f},{ty1:.3f}] outside {tuple(round(v, 3) for v in cbox_lim)}"
            )

        for ix0, iy0, ix1, iy1 in inset_boxes:
            ox0, oy0 = max(tx0 + 0.01, ix0), max(ty0 + 0.01, iy0)
            ox1, oy1 = min(tx1 - 0.01, ix1), min(ty1 - 0.01, iy1)
            if ox1 > ox0 and oy1 > oy0:
                img_overlaps.append(f"{label_short!r} overlaps inset [{ix0:.2f},{iy0:.2f},{ix1:.2f},{iy1:.2f}]")

    overlaps: list[tuple[str, str]] = []
    for i in range(len(bboxes_in)):
        for j in range(i + 1, len(bboxes_in)):
            a0, a1 = bboxes_in[i][1], bboxes_in[j][1]
            pad_in = 0.012  # ~0.85 pt ascender/descender line-box touch tolerance
            ix0, iy0 = max(a0[0] + pad_in, a1[0] + pad_in), max(a0[1] + pad_in, a1[1] + pad_in)
            ix1, iy1 = min(a0[2] - pad_in, a1[2] - pad_in), min(a0[3] - pad_in, a1[3] - pad_in)
            if ix1 > ix0 and iy1 > iy0:
                overlaps.append((bboxes_in[i][0], bboxes_in[j][0]))

    # Export PDF, 300-dpi PNG, and reproducibility .data.json
    fig_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = fig_dir / "supp_qual_examples.pdf"
    png_path = fig_dir / "supp_qual_examples.png"
    json_path = fig_dir / "supp_qual_examples.data.json"
    fig.savefig(pdf_path)
    fig.savefig(png_path, dpi=300)
    plt.close(fig)

    pdf_bytes = pdf_path.read_bytes()
    has_type3 = b"/Subtype /Type3" in pdf_bytes or b"/Subtype/Type3" in pdf_bytes
    assert not has_type3, "PDF contains Type-3 fonts!"

    audit = {
        "figure": "supp_qual_examples",
        "canvas_inches": [W_IN, H_IN],
        "min_fontsize_pt": min_pt,
        "pdf_has_type3_fonts": has_type3,
        "text_overlap_pairs": overlaps,
        "box_overflows": box_overflows,
        "image_overlaps": img_overlaps,
        "part_a_stimuli": {
            "cue_factorial": {
                "set": "occ_cues",
                "item_ids": ["cue018", "cue018_cf"],
                "images": [
                    rel_path(data["paths"]["cues_dir"] / "images" / "cue018.png"),
                    rel_path(data["paths"]["cues_dir"] / "images" / "cue018_cf.png"),
                ],
                "question": data["a1"]["v"]["question"],
                "options": data["a1"]["v"]["options"],
                "gold_v": data["a1"]["v"]["gold"],
                "gold_vcf": data["a1"]["cf"]["gold"],
                "cell": data["a1"]["v"]["meta"]["cell"],
                "area_ratio_front_over_back": round(float(data["a1"]["v"]["meta"]["area_ratio_front_over_back"]), 3),
            },
            "synthetic_depth": {
                "set": "syn_depth",
                "item_id": "syn_depth_10",
                "image": rel_path(data["paths"]["syn_img"]),
                "question": data["a2"]["rec"]["question"],
                "answer": data["a2"]["rec"]["answer"],
                "gold": "A",
                "visible_area_px": {"front_yellow_triangle": data["a2"]["px_front"],
                                    "back_orange_square": data["a2"]["px_back"]},
                "visible_area_ratio": round(data["a2"]["area_ratio"], 2),
                "dataset_confound_note": "front shape drawn larger; larger visible area on 98.6% of images",
            },
            "hard_occlusion": {
                "set": "occlusion_hard",
                "item_id": "occl4026",
                "twin_id": "occl4026_cf",
                "level": data["a3"]["v"]["meta"]["level"],
                "image": rel_path(data["paths"]["hard_dir"] / "images" / "occl4026.png"),
                "question": data["a3"]["v"]["question"],
                "gold_v": data["a3"]["v"]["gold"],
                "back_hidden_frac": round(float(data["a3"]["v"]["meta"]["back_hidden_frac"]), 4),
                "area_ratio_front_over_back": round(float(data["a3"]["v"]["meta"]["area_ratio_front_over_back"]), 3),
                "n_distractors": data["a3"]["v"]["meta"]["n_distractors"],
            },
            "coco_occlusion_twins": {
                "set": "e62_coco_twins",
                "pair_id": "e62p016",
                "item_ids": ["e62p016a_closer", "e62p016b_closer"],
                "images": [
                    rel_path(data["paths"]["e62_dir"] / "images" / "e62p016a.png"),
                    rel_path(data["paths"]["e62_dir"] / "images" / "e62p016b.png"),
                ],
                "question": data["a4"]["v"]["question"],
                "gold_v": data["a4"]["v"]["gold"],
                "gold_vcf": data["a4"]["cf"]["gold"],
                "front_v": data["a4"]["v"]["front_name"],
                "front_vcf": data["a4"]["cf"]["front_name"],
            },
            "mirrored_cvbench": {
                "set": "cvb_relation",
                "item_id": "cvb_Relation_1276_COCO",
                "question_full": data["a5"]["rec"]["question"],
                "question_displayed": q_a5_elided + "\n(A) left   (B) right",
                "gold_v": data["a5"]["rec"]["gold"],
                "gold_mirror": data["a5"]["rec"]["gold_mirror"],
                "x": data["a5"]["rec"]["x"],
                "y": data["a5"]["rec"]["y"],
            },
        },
        "part_b_rationales": {
            "b1_stage1_commit": {
                "model": "qwen3vl_8b",
                "model_label": "Qwen3-VL-8B",
                "set": "occ_cues",
                "group": "D+C-",
                "item_id": "cue320",
                "image": rel_path(data["paths"]["cues_dir"] / "images" / "cue320.png"),
                "sources": [
                    "vtax/outputs/e10/qwen3vl_8b_cues.jsonl",
                    "vtax/outputs/e28/qwen3vl_8b_cues_traj.jsonl",
                    "vtax/outputs/e57/qwen3vl_8b_cues_trajtwin.jsonl",
                ],
                "question": data["b1"]["traj"]["question"],
                "gold": data["b1"]["traj"]["gold"],
                "front_name": data["b1"]["traj"]["meta"]["front_name"],
                "back_name": data["b1"]["traj"]["meta"]["back_name"],
                "direct_pred": data["b1"]["pred_dir"],
                "direct_p_gold": round(data["b1"]["p_dir"], 4),
                "cot_pred": data["b1"]["pred_cot"],
                "cot_p_gold": round(data["b1"]["p_cot"], 4),
                "cot_full_verbatim": data["b1"]["cot"]["text"],
                "sentences_verbatim": b1_sents,
                "p_gold_trajectory": [round(float(x), 4) for x in b1_pg],
                "commit_idx_0based": j_star,
                "commit_sentence_1based": j_star + 1,
                "commit_sentence_verbatim": b1_sents[j_star],
                "two_judge_consensus_at_jstar": "INCORRECT",
                "prefix_twin_p_gold_own": [round(float(x), 4) for x in tt_own],
                "prefix_twin_p_gold_twin": [round(float(x), 4) for x in tt_twin],
            },
            "b2_stage3_converse": {
                "model": "molmo7b",
                "model_label": "Molmo-7B-D",
                "set": "occ_cues",
                "group": "D+C-",
                "item_id": "cue348_behind",
                "image": rel_path(data["paths"]["cues_dir"] / "images" / "cue348.png"),
                "sources": [
                    "vtax/outputs/e10/molmo7b_cues.jsonl",
                    "vtax/outputs/e28/molmo7b_cues_traj.jsonl",
                    "outputs/bundle/e54_decomp.json",
                ],
                "question": data["b2"]["traj"]["question"],
                "gold": data["b2"]["traj"]["gold"],
                "front_name": data["b2"]["traj"]["meta"]["front_name"],
                "back_name": data["b2"]["traj"]["meta"]["back_name"],
                "direct_pred": data["b2"]["pred_dir"],
                "direct_p_gold": round(data["b2"]["p_dir"], 4),
                "cot_pred": data["b2"]["pred_cot"],
                "cot_p_gold": round(data["b2"]["p_cot"], 4),
                "cot_full_verbatim": data["b2"]["cot"]["text"],
                "sentences_verbatim": b2_sents,
                "p_gold_trajectory": [round(float(x), 4) for x in b2_pg],
                "e54_classification": data["b2"]["e54"],
            },
            "b3_closed_regrounding": {
                "model": "claude-opus-5-5",
                "model_label": "Claude Opus 5.5",
                "set": "occlusion_hard",
                "item_id": "occl4026_front",
                "twin_id": "occl4026_cf",
                "sources": [
                    "vtax/outputs/e61/items.jsonl",
                    "vtax/outputs/e61/gen_claude-opus-5-5.jsonl",
                    "outputs/bundle/e61_closed_fixed_rcut.json",
                ],
                "question": data["b3"]["item"]["question"],
                "gold_v": data["b3"]["item"]["gold"],
                "gold_vcf": "B",
                "fixed_rcut_removed_verbatim": data["b3"]["item"]["body_removed"],
                "reply_under_v_verbatim": data["b3"]["rv"]["text"],
                "reply_under_v_pred": data["b3"]["rv"]["pred"],
                "reply_under_v_reasoning_tokens": rt_v,
                "reply_under_vcf_verbatim": data["b3"]["rcf"]["text"],
                "reply_under_vcf_pred": data["b3"]["rcf"]["pred"],
                "reply_under_vcf_reasoning_tokens": rt_cf,
                "same_item_peers_removed_variant": {
                    m: {"v": peers[m]["removed_v"], "vcf": peers[m]["removed_vcf"]}
                    for m in ("gemini-4-argon", "gpt-6-astra", "qwen3.8-max-0902")
                },
            },
        },
    }
    json_path.write_text(json.dumps(audit, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return [pdf_path, png_path, json_path], audit


def parse_args() -> argparse.Namespace:
    """Parse command-line options."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--share", type=Path, default=None, help="Shared network storage root")
    ap.add_argument("--bundle-dir", type=Path, default=DEFAULT_BUNDLE_DIR)
    ap.add_argument("--fig-dir", type=Path, default=DEFAULT_FIG_DIR)
    return ap.parse_args()


def main() -> None:
    """Entry point: gather verbatim outputs, build figure, verify fonts/overlaps/bounds."""
    args = parse_args()
    configure_matplotlib()
    share_root = args.share or resolve_share_root()
    data = gather_all_data(share_root, args.bundle_dir)
    paths, audit = build_figure(data, args.fig_dir)
    print("Generated files:")
    for p in paths:
        print(f"  {p}")
    print(f"Min font size: {audit['min_fontsize_pt']} pt | Type-3 fonts: {audit['pdf_has_type3_fonts']}")
    print(f"Overlaps: {len(audit['text_overlap_pairs'])} | Box overflows: {len(audit['box_overflows'])} | Image overlaps: {len(audit['image_overlaps'])}")
    for pair in audit["text_overlap_pairs"]:
        print("  OVERLAP:", pair)
    for ov in audit["box_overflows"]:
        print("  BOX OVERFLOW:", ov)
    for io_msg in audit["image_overlaps"]:
        print("  IMG OVERLAP:", io_msg)


if __name__ == "__main__":
    main()
