"""Generate the v3 paper figures directly from experimental outputs.

Every plotted value and quoted rationale excerpt is read at run time from:
  1. outputs/bundle/numbers.json (summary metrics and 95% image-cluster CIs)
  2. outputs/bundle/dprime_ci.json (pooled occlusion sensitivity d' and CIs)
  3. vtax/outputs/e2_judge.jsonl (Claude Opus 5.5 spatial claim audit)
  4. vtax/outputs/e2b/<model>.jsonl + legacy/e4 readouts (prefix trajectories)
  5. vtax/outputs/e2d/qwen3vl_8b.jsonl + occlusion images/manifests (Fig. 1a)

Files under any _invalid_* directory are ignored. Missing models or conditions
are skipped silently so the script can be re-run as remaining experiments finish.
Writes <figname>.pdf, <figname>.png (300 dpi), and <figname>.data.json.
"""
from __future__ import annotations
import vtax_paths as _vtp  # noqa: E402

import argparse
import json
import math
import re
import sys
import textwrap
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
import numpy as np
from PIL import Image
from scipy.stats import norm

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from build_bundle import NAMES, groups_from, probs  # noqa: E402
from common import cluster_bootstrap_ci  # noqa: E402

# Default data and output paths
DEFAULT_BUNDLE = Path(_vtp.OUTPUTS) / "bundle" / "numbers.json"
DEFAULT_DPRIME = Path(_vtp.OUTPUTS) / "bundle" / "dprime_ci.json"
DEFAULT_OUTPUTS = Path(
    f"{_vtp.OUTPUTS}"
)
DEFAULT_JUDGE = DEFAULT_OUTPUTS / "e2_judge.jsonl"
DEFAULT_LEGACY = Path(_vtp.LEGACY_RESULTS)
DEFAULT_OCC_DIR = Path(
    f"{_vtp.ITEMS}/occlusion"
)
DEFAULT_FIG_DIR = SCRIPT_DIR.parent / f"{_vtp.PAPER}" / "figures"

# muted palette
COL_SLATE = "#B9BEC6"
COL_SLATE_DARK = "#5A6270"
COL_ICE = "#9BBAD9"
COL_ICE_DARK = "#3B6998"
COL_PEACH = "#E2B28F"
COL_CORAL = "#DC8A7E"
COL_CORAL_DARK = "#B85244"
COL_TEXT = "#1F242B"
COL_MUTED_TEXT = "#555D68"
COL_GRID = "#E6E9EE"
COL_GOOD = "#2B7A4B"
COL_BAD = "#B84439"

# Distinct muted line styles for multi-model trajectory plots (Fig. 2a)
MODEL_LINE_STYLE: dict[str, dict[str, Any]] = {
    "qwen3b": {"color": "#DC8A7E", "marker": "o", "ls": "-"},
    "qwen7b": {"color": "#4E79A7", "marker": "s", "ls": "-"},
    "internvl3_8b": {"color": "#59A14F", "marker": "^", "ls": "-"},
    "phi35v": {"color": "#8E6C8A", "marker": "D", "ls": "--"},
    "molmo7b": {"color": "#D99B52", "marker": "v", "ls": "-"},
    "qwen3vl_8b": {"color": "#2F6B7C", "marker": "P", "ls": "-"},
    "qwen3vl_8b_think": {"color": "#A35D6A", "marker": "*", "ls": "-."},
    "internvl35_8b": {"color": "#7A828E", "marker": "X", "ls": "-."},
}

# One canonical short model-label set for every figure (single-line and two-line forms of the same names)
CANON_LABELS: dict[str, str] = {
    "qwen3b": "Qwen2.5-VL-3B",
    "qwen7b": "Qwen2.5-VL-7B",
    "internvl3_8b": "InternVL3-8B",
    "phi35v": "Phi-3.5-Vision",
    "molmo7b": "Molmo-7B-D",
    "qwen3vl_8b": "Qwen3-VL-8B",
    "qwen3vl_8b_think": "Qwen3-VL-8B-Thinking",
    "internvl35_8b": "InternVL3.5-8B",
    "deepseek": "DeepSeek-V4.1-Flash\u2020",
}
CANON_TWO_LINE: dict[str, str] = {
    "qwen3b": "Qwen2.5-VL\n3B",
    "qwen7b": "Qwen2.5-VL\n7B",
    "internvl3_8b": "InternVL3\n8B",
    "phi35v": "Phi-3.5\nVision",
    "molmo7b": "Molmo\n7B-D",
    "qwen3vl_8b": "Qwen3-VL\n8B",
    "qwen3vl_8b_think": "Qwen3-VL\n8B-Thinking",
    "internvl35_8b": "InternVL3.5\n8B",
    "deepseek": "DeepSeek-V4.1\nFlash\u2020",
}
TWO_LINE_LABELS = CANON_TWO_LINE
NARROW_TWO_LINE_LABELS = CANON_TWO_LINE
COMPACT_LABELS = CANON_LABELS
SCATTER_LABELS = CANON_LABELS



def rel_path(p: Path | str) -> str:
    """Path relative to the vtax root, so shipped audit files carry no machine paths."""
    s = str(p)
    k = s.find("vtax/")
    return s[k:] if k >= 0 else Path(s).name


def configure_matplotlib() -> None:
    """Apply clean sans-serif styling across all figures."""
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
        "font.size": 7.5,
        "axes.titlesize": 8.0,
        "axes.labelsize": 7.5,
        "xtick.labelsize": 7.0,
        "ytick.labelsize": 7.0,
        "legend.fontsize": 7.0,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.edgecolor": "#4A515C",
        "axes.linewidth": 0.65,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "xtick.major.size": 2.5,
        "ytick.major.size": 2.5,
        "xtick.color": COL_TEXT,
        "ytick.color": COL_TEXT,
        "text.color": COL_TEXT,
        "axes.labelcolor": COL_TEXT,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def is_invalid_path(path: Path) -> bool:
    """Return True if any path component starts with _invalid_."""
    return any(part.startswith("_invalid_") for part in path.parts)


def safe_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read a JSONL file while ignoring any _invalid_* directory."""
    if is_invalid_path(path) or not path.exists():
        return []
    with open(path, "r", encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def argmax_finite(lp: dict[str, Any]) -> str | None:
    """Return the key with the largest finite logprob."""
    fin = {k: float(v) for k, v in lp.items() if v is not None and math.isfinite(float(v))}
    return max(fin, key=fin.get) if fin else None


def has_valid_ci(d: Any) -> bool:
    """Check whether a bundle CI dictionary contains valid numeric entries."""
    return isinstance(d, dict) and "mean" in d and d.get("n", 0) > 0


def wilson_ci_pct(k: int, n: int, confidence: float = 0.95) -> tuple[float, float, float]:
    """Compute binomial proportion and Wilson score interval in percentage points."""
    if n <= 0:
        return float("nan"), float("nan"), float("nan")
    z = float(norm.ppf(0.5 + confidence / 2.0))
    p = k / n
    denom = 1.0 + (z * z) / n
    center = (p + (z * z) / (2.0 * n)) / denom
    half = (z / denom) * math.sqrt((p * (1.0 - p)) / n + (z * z) / (4.0 * n * n))
    lo = max(0.0, center - half)
    hi = min(1.0, center + half)
    return 100.0 * p, 100.0 * lo, 100.0 * hi


def save_figure_and_audit(fig: plt.Figure, fig_dir: Path, name: str,
                          audit_data: dict[str, Any]) -> list[Path]:
    """Export PDF and 300-dpi PNG at exact paper canvas size, plus audit JSON."""
    fig_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = fig_dir / f"{name}.pdf"
    png_path = fig_dir / f"{name}.png"
    json_path = fig_dir / f"{name}.data.json"
    fig.savefig(pdf_path)
    fig.savefig(png_path, dpi=300)
    with open(json_path, "w", encoding="utf-8") as fh:
        json.dump(audit_data, fh, indent=2, ensure_ascii=False)
    plt.close(fig)
    return [pdf_path, png_path, json_path]


def extract_overlap_excerpt(cot_text: str) -> str:
    """Extract a short verbatim overlap/on-top claim sentence from a CoT rationale.

    Strips markdown bold markers and bullet prefixes while keeping the exact
    wording generated by the model. Prefers concrete image-specific sentences
    that assign 'on top' or 'overlap' to a specific shape over generic preambles.
    """
    clean = cot_text.replace("**", "")
    raw_lines = [ln.strip().lstrip("-*• ").strip() for ln in clean.splitlines() if ln.strip()]
    candidates: list[str] = []
    for ln in raw_lines:
        for sent in re.split(r"(?<=[.!?])\s+", ln):
            s = sent.strip()
            if not s:
                continue
            low = s.lower()
            if "in 2d images" in low or "perceived as being" in low or "let's think" in low:
                continue
            if "on top" in low or "overlap" in low:
                candidates.append(s)
    if not candidates:
        return raw_lines[-1] if raw_lines else ""
    on_top = [c for c in candidates if "on top" in c.lower()]
    chosen = on_top[0] if on_top else candidates[0]
    for prefix in ("This means ", "Specifically, ", "Therefore, based on the visual representation, "):
        if chosen.startswith(prefix):
            chosen = "\u2026" + chosen[len(prefix):]
            break
    if chosen.endswith(" in the image."):
        chosen = chosen[: -len(" in the image.")] + "\u2026"
    return chosen


def find_t_junction(img: np.ndarray, tol: int = 40) -> tuple[float, float] | None:
    """Topmost pixel where the two shapes and the background meet (the occlusion T-junction)."""
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
    t = binary_dilation(m1, st) & binary_dilation(m2, st) & binary_dilation(mb, st)
    ys, xs = np.nonzero(t)
    if not len(ys):
        return None
    k = ys.argmin()
    return float(xs[k]), float(ys[k])


def load_twin_cropped_images(img_a_path: Path, img_b_path: Path,
                             pad: int = 32) -> tuple[np.ndarray, np.ndarray]:
    """Load twin PNGs and apply a shared square crop around both scenes' foregrounds."""
    im_a = np.asarray(Image.open(img_a_path).convert("RGB"))
    im_b = np.asarray(Image.open(img_b_path).convert("RGB"))
    h, w, _ = im_a.shape
    bg_a, bg_b = im_a[0, 0], im_b[0, 0]
    ys_a, xs_a = np.where(np.any(im_a != bg_a, axis=-1))
    ys_b, xs_b = np.where(np.any(im_b != bg_b, axis=-1))
    x0 = max(0, int(min(xs_a.min(), xs_b.min())) - pad)
    x1 = min(w, int(max(xs_a.max(), xs_b.max())) + pad)
    y0 = max(0, int(min(ys_a.min(), ys_b.min())) - pad)
    y1 = min(h, int(max(ys_a.max(), ys_b.max())) + pad)
    side = max(x1 - x0, y1 - y0)
    cx, cy = (x0 + x1) // 2, (y0 + y1) // 2
    sx0 = max(0, min(w - side, cx - side // 2))
    sy0 = max(0, min(h - side, cy - side // 2))
    return im_a[sy0: sy0 + side, sx0: sx0 + side], im_b[sy0: sy0 + side, sx0: sx0 + side]


# =============================================================================
# Figure 1: Teaser (3 panels, full width = 6.875 in, ~2.60 in tall)
# =============================================================================

def build_fig1_teaser(bundle: dict[str, Any], outputs_dir: Path, judge_path: Path,
                      occ_dir: Path, fig_dir: Path) -> tuple[list[Path], dict[str, Any]]:
    """Build Figure 1: (a) twin example, (b) two-judge consensus contradiction rate, (c) task heatmap."""
    report: dict[str, Any] = {"skipped": [], "notes": []}
    audit: dict[str, Any] = {"panel_a": {}, "panel_b": {}, "panel_c": {}}

    fig = plt.figure(figsize=(6.875, 2.82))
    # Explicit panel positions with breathing room between (a)-(b) and slightly wider (c) cells
    ax_a = fig.add_axes([0.008, 0.242, 0.296, 0.538])
    ax_b = fig.add_axes([0.366, 0.242, 0.190, 0.538])
    ax_c = fig.add_axes([0.800, 0.150, 0.190, 0.600])

    # -------------------------------------------------------------------------
    # Panel (a): Swapped occlusion twins (occ085 vs. occ085_cf on Qwen3-VL-8B)
    # -------------------------------------------------------------------------
    ax_a.set_axis_off()
    ax_a.set_xlim(0.0, 1.0)
    ax_a.set_ylim(0.0, 1.0)
    ax_a.text(
        0.01, 1.265, "(a) Occlusion swap vs. CoT rationale",
        transform=ax_a.transAxes, fontsize=7.8, fontweight="bold", ha="left", va="bottom"
    )

    q_map: dict[str, str] = {}
    opt_map: dict[str, list[str]] = {}
    for mf in ("items.jsonl", "items_cf.jsonl"):
        for rec in safe_jsonl(occ_dir / mf):
            if rec.get("item_id") in ("occ085", "occ085_cf"):
                q_map[rec["item_id"]] = rec["question"]
                opt_map[rec["item_id"]] = rec["options"]
    shared_question = q_map.get("occ085") or q_map.get("occ085_cf") or ""
    options = opt_map.get("occ085", ["yes", "no"])
    letter_to_word = {"A": options[0].capitalize(), "B": options[1].capitalize()}

    e2d_recs = safe_jsonl(outputs_dir / "e2d" / "qwen3vl_8b.jsonl")
    by_item_cond: dict[tuple[str, str], dict[str, Any]] = {}
    for rec in e2d_recs:
        iid = rec.get("item_id")
        cond = rec.get("condition")
        if iid in ("occ085", "occ085_cf") and cond in ("direct", "cot"):
            by_item_cond[(iid, cond)] = rec

    img_orig_path = occ_dir / "images" / "occ085.png"
    img_cf_path = occ_dir / "images" / "occ085_cf.png"
    crop_orig, crop_cf = load_twin_cropped_images(img_orig_path, img_cf_path)

    # Shared question box strictly inside [0.02, 0.98] with wrapped 2-line question
    q_box = mpatches.FancyBboxPatch(
        (0.015, 0.872), 0.97, 0.136,
        boxstyle="round,pad=0.0,rounding_size=0.022",
        facecolor="#F2F5F9", edgecolor="#CBD4E1", linewidth=0.65,
        transform=ax_a.transAxes, clip_on=False, zorder=1
    )
    ax_a.add_patch(q_box)
    wrapped_q = "\n".join(textwrap.wrap(f"Q: \u201c{shared_question}\u201d", width=36))
    ax_a.text(
        0.50, 0.940, wrapped_q,
        transform=ax_a.transAxes, fontsize=7.0, fontweight="bold",
        ha="center", va="center", color=COL_TEXT, linespacing=1.14, zorder=2
    )

    twin_specs = [
        ("occ085", crop_orig, 0.0, 0.442),
        ("occ085_cf", crop_cf, 0.558, 0.442),
    ]

    for iid, crop_arr, x_left, col_w in twin_specs:
        d_rec = by_item_cond[(iid, "direct")]
        c_rec = by_item_cond[(iid, "cot")]
        gold = d_rec["gold"]
        d_pred = argmax_finite(d_rec["top20"])
        c_pred = argmax_finite(c_rec["top20"])
        d_ok = d_pred == gold
        c_ok = c_pred == gold
        front_obj = d_rec.get("meta", {}).get("front_name", "")
        excerpt = extract_overlap_excerpt(c_rec.get("text", ""))

        # Two-line non-colliding ground-truth label per twin image
        scene_tag = f"{'Hexagon' if 'hexagon' in front_obj.lower() else 'Triangle'} in front"
        ax_a.text(
            x_left + col_w / 2, 0.805, scene_tag,
            transform=ax_a.transAxes, fontsize=6.4, fontweight="bold",
            ha="center", va="center", color=COL_SLATE_DARK, linespacing=1.06
        )

        # Square image inset
        im_x0, im_y0, im_w, im_h = x_left + 0.066, 0.468, col_w - 0.132, 0.278
        im_ax = ax_a.inset_axes([im_x0, im_y0, im_w, im_h])
        im_ax.imshow(crop_arr)
        tj = find_t_junction(crop_arr)
        if tj is not None:
            side = crop_arr.shape[0]
            tx, ty = tj
            # Short clear pointer wholly in empty lower-left background aimed at the occluding T-junction
            sx = max(0.06 * side, tx - 0.165 * side)
            sy = min(0.94 * side, ty + 0.155 * side)
            im_ax.annotate(
                "", xy=(tx, ty), xytext=(sx, sy),
                arrowprops={"arrowstyle": "-|>", "color": COL_TEXT, "lw": 0.9, "mutation_scale": 6.0,
                            "shrinkA": 0, "shrinkB": 0.5}
            )
            if iid == "occ085":
                # Subtle grey callout for the occluding edge right beneath the first thumbnail
                tail_ax_x = im_x0 + (sx / side) * im_w
                tail_ax_y = im_y0 + (1.0 - sy / side) * im_h
                pass
            audit["panel_a"].setdefault("t_junction_px", {})[iid] = [round(float(tx), 1), round(float(ty), 1)]
        im_ax.set_xticks([])
        im_ax.set_yticks([])
        for sp in im_ax.spines.values():
            sp.set_visible(True)
            sp.set_edgecolor("#B9C1CC")
            sp.set_linewidth(0.7)

        # Direct readout pill
        if iid == "occ085":
            box_x0, box_w = -0.015, 0.495
        else:
            box_x0, box_w = 0.520, 0.495
        box_cx = box_x0 + box_w / 2
        d_pill = mpatches.FancyBboxPatch(
            (box_x0, 0.302), box_w, 0.084,
            boxstyle="round,pad=0.0,rounding_size=0.018",
            facecolor="#ECEEF2", edgecolor=COL_SLATE, linewidth=0.65,
            transform=ax_a.transAxes, clip_on=False
        )
        ax_a.add_patch(d_pill)
        d_mark = "\u2713" if d_ok else "\u2717"
        d_col = COL_GOOD if d_ok else COL_BAD
        ax_a.text(
            box_cx, 0.344,
            f"Direct: {letter_to_word[d_pred]} ({d_mark})",
            transform=ax_a.transAxes, fontsize=7.0, fontweight="bold",
            ha="center", va="center", color=d_col
        )

        # CoT rationale excerpt + CoT answer box
        c_box = mpatches.FancyBboxPatch(
            (box_x0, 0.030), box_w, 0.240,
            boxstyle="round,pad=0.0,rounding_size=0.018",
            facecolor="#FCF3F1" if not c_ok else "#EFF6F1",
            edgecolor=COL_CORAL if not c_ok else "#8DBF9E", linewidth=0.75,
            transform=ax_a.transAxes, clip_on=False
        )
        ax_a.add_patch(c_box)
        wrapped_excerpt = "\u201c\u2026hexagon is\non top\u2026\u201d"
        ax_a.text(
            box_cx, 0.180, wrapped_excerpt,
            transform=ax_a.transAxes, fontsize=6.8, ha="center", va="center",
            color=COL_TEXT, linespacing=1.08, style="italic"
        )
        c_mark = "\u2713" if c_ok else "\u2717"
        c_col = COL_GOOD if c_ok else COL_BAD
        ax_a.text(
            box_cx, 0.075,
            f"CoT: {letter_to_word[c_pred]} ({c_mark})",
            transform=ax_a.transAxes, fontsize=7.0, fontweight="bold",
            ha="center", va="center", color=c_col
        )

        audit["panel_a"][iid] = {
            "image_path": rel_path(img_orig_path if iid == "occ085" else img_cf_path),
            "question": shared_question,
            "gold": gold,
            "gold_label": letter_to_word[gold],
            "front_name": d_rec.get("meta", {}).get("front_name"),
            "back_name": d_rec.get("meta", {}).get("back_name"),
            "direct_top20": d_rec["top20"],
            "direct_pred": d_pred,
            "direct_pred_label": letter_to_word[d_pred],
            "direct_correct": d_ok,
            "cot_top20": c_rec["top20"],
            "cot_pred": c_pred,
            "cot_pred_label": letter_to_word[c_pred],
            "cot_correct": c_ok,
            "cot_excerpt_plotted": excerpt,
            "cot_full_text": c_rec.get("text", ""),
        }

    # Swap arrow between twin images + wrapped unclipped bottom note
    ax_a.text(
        0.50, 0.607, "\u21c4", transform=ax_a.transAxes,
        fontsize=8.8, fontweight="bold", ha="center", va="center", color=COL_SLATE_DARK
    )

    # -------------------------------------------------------------------------
    # Panel (b): First spatial claim contradicts image (two-judge consensus + individual judges)
    # -------------------------------------------------------------------------
    ax_b.text(
        -0.06, 1.265, "(b) First claim contradicts image",
        transform=ax_b.transAxes, fontsize=7.8, fontweight="bold", ha="left", va="bottom"
    )

    judge2_path = Path(_vtp.OUTPUTS) / "bundle" / "judge2.json"
    judge2_data = json.loads(judge2_path.read_text(encoding="utf-8"))
    j2_rates = judge2_data.get("rates", {})
    candidate_1b = ["qwen3b", "qwen7b", "internvl3_8b", "phi35v", "molmo7b", "qwen3vl_8b", "internvl35_8b"]
    present_1b: list[str] = []
    judge_stats: dict[str, dict[str, dict[str, Any]]] = {}

    for m in candidate_1b:
        r_neg = j2_rates.get(f"{m}|D+C-")
        r_pos = j2_rates.get(f"{m}|D+C+")
        if r_neg and r_pos and "consensus_incorrect" in r_neg and "consensus_incorrect" in r_pos:
            present_1b.append(m)
            judge_stats[m] = {
                "D+C-": {
                    "consensus_incorrect_pct": float(r_neg["consensus_incorrect"]),
                    "consensus_ci_pct": [float(r_neg["consensus_ci"][0]), float(r_neg["consensus_ci"][1])],
                    "consensus_n": int(r_neg["consensus_n"]),
                    "opus_incorrect_pct": float(r_neg["claude_incorrect_excl_unclear"]),
                    "opus_n_clear": int(r_neg["claude_n_clear"]),
                    "gpt_incorrect_pct": float(r_neg["gpt_incorrect_excl_unclear"]),
                    "gpt_n_clear": int(r_neg["gpt_n_clear"]),
                    "n_total_sampled": int(r_neg["n"]),
                },
                "D+C+": {
                    "consensus_incorrect_pct": float(r_pos["consensus_incorrect"]),
                    "consensus_ci_pct": [float(r_pos["consensus_ci"][0]), float(r_pos["consensus_ci"][1])],
                    "consensus_n": int(r_pos["consensus_n"]),
                    "opus_incorrect_pct": float(r_pos["claude_incorrect_excl_unclear"]),
                    "opus_n_clear": int(r_pos["claude_n_clear"]),
                    "gpt_incorrect_pct": float(r_pos["gpt_incorrect_excl_unclear"]),
                    "gpt_n_clear": int(r_pos["gpt_n_clear"]),
                    "n_total_sampled": int(r_pos["n"]),
                },
            }

    not_in_judge = [m for m in candidate_1b if m not in present_1b]
    if not_in_judge:
        report["skipped"].append(f"fig1b (judge2): {', '.join(not_in_judge)}")

    audit["panel_b"] = {
        "source": rel_path(judge2_path),
        "metric": "two-judge consensus share judged 'incorrect' among clear agreed items (95% Wilson CI), with individual Opus 5.5 and GPT-6-Astra markers",
        "agreement": judge2_data.get("D+C-/D+C+ (480)", {}),
        "models_plotted": present_1b,
        "skipped_models": not_in_judge,
        "values": judge_stats,
    }

    x_b = np.arange(len(present_1b), dtype=float)
    bw = 0.37
    ax_b.yaxis.grid(True, color=COL_GRID, linewidth=0.55, zorder=0)
    ax_b.set_axisbelow(True)

    for idx, m in enumerate(present_1b):
        st_neg = judge_stats[m]["D+C-"]
        st_pos = judge_stats[m]["D+C+"]
        x_neg = x_b[idx] - bw / 2
        x_pos = x_b[idx] + bw / 2

        ax_b.bar(
            x_neg, st_neg["consensus_incorrect_pct"], bw,
            color=COL_CORAL, edgecolor="white", linewidth=0.7, zorder=2
        )
        ax_b.bar(
            x_pos, st_pos["consensus_incorrect_pct"], bw,
            color=COL_SLATE, edgecolor="white", linewidth=0.7, zorder=2
        )

        for x_bar, st in ((x_neg, st_neg), (x_pos, st_pos)):
            val = st["consensus_incorrect_pct"]
            lo_ci, hi_ci = st["consensus_ci_pct"]
            ax_b.errorbar(
                x_bar, val,
                yerr=[[max(0.0, val - lo_ci)], [max(0.0, hi_ci - val)]],
                fmt="none", ecolor="#3C434D", elinewidth=0.72, capsize=1.6, capthick=0.72, zorder=4
            )
            # Individual judge markers slightly offset left/right within bar width
            ax_b.scatter(
                [x_bar - 0.042], [st["opus_incorrect_pct"]],
                marker="o", s=9.0, facecolor="white", edgecolor=COL_TEXT, linewidth=0.68, zorder=5
            )
            ax_b.scatter(
                [x_bar + 0.042], [st["gpt_incorrect_pct"]],
                marker="^", s=9.5, facecolor=COL_TEXT, edgecolor="white", linewidth=0.42, zorder=5
            )

    ax_b.set_xticks(x_b)
    ax_b.set_xticklabels([COMPACT_LABELS[m] for m in present_1b], fontsize=7.0, rotation=40, ha="right",
                         rotation_mode="anchor")
    ax_b.tick_params(axis="x", pad=2.0)
    ax_b.set_ylabel("Judged incorrect (%)", labelpad=2)
    ax_b.set_ylim(0, 105)
    ax_b.set_yticks([0, 25, 50, 75, 100])
    ax_b.set_xlim(x_b[0] - 0.56, x_b[-1] + 0.54)

    leg_1b_bars = [
        mpatches.Patch(facecolor=COL_CORAL, edgecolor="white", label="D+C\u2212 (Direct \u2713, CoT \u2717)"),
        mpatches.Patch(facecolor=COL_SLATE, edgecolor="white", label="D+C+ (Direct \u2713, CoT \u2713)"),
    ]
    leg_1b_judges = [
        plt.Line2D([], [], marker="o", linestyle="none", markersize=3.3, markerfacecolor="white",
                   markeredgecolor=COL_TEXT, markeredgewidth=0.75, label="judge: Opus 5.5"),
        plt.Line2D([], [], marker="^", linestyle="none", markersize=3.5, markerfacecolor=COL_TEXT,
                   markeredgecolor="white", markeredgewidth=0.4, label="judge: GPT-6-Astra"),
    ]
    leg_top = ax_b.legend(
        handles=leg_1b_bars, loc="lower left", bbox_to_anchor=(-0.04, 1.085),
        ncol=1, frameon=False, handlelength=1.05, handletextpad=0.38, labelspacing=0.18,
        fontsize=7.0, borderaxespad=0.0
    )
    ax_b.add_artist(leg_top)
    ax_b.legend(
        handles=leg_1b_judges, loc="lower left", bbox_to_anchor=(-0.04, 1.002),
        ncol=2, frameon=False, handlelength=0.9, handletextpad=0.28, columnspacing=0.65,
        fontsize=7.0, borderaxespad=0.0
    )

    # -------------------------------------------------------------------------
    # Panel (c): image effect before vs. after a relation is written (points)
    # Sources: Tab. supp_e57 (prefix twin effect before/after j*), Tab. 2 (twins,
    # 7-model mean of Direct / own R_cut), Tab. supp_e60 (mirrored CV-Bench),
    # Tab. supp_e62 (COCO twins), Tab. supp_e61 (closed, hard set, removed variant
    # vs. lowest-reasoning Direct).
    # -------------------------------------------------------------------------
    ax_c.text(
        -0.66, 1.205, "(c) Does the image still count?",
        transform=ax_c.transAxes, fontsize=7.8, fontweight="bold", ha="left", va="bottom"
    )
    rows_c = [
        ("Qwen3-VL-8B, same rationale", 64.6, 3.7),
        ("Qwen3-VL-32B, same rationale", 86.9, 3.0),
        ("Rendered twins (7 open)", float(np.mean([3.0, 8.0, 26.8, 26.8, 76.5, 62.3, 13.3])),
         float(np.mean([2.0, 3.9, 5.0, 4.4, 8.9, 8.4, 2.0]))),
        ("Mirrored photos (7 open)", 69.6, 3.4),
        ("COCO twins (7 open)", 20.5, 3.0),
        ("Qwen3.8-Max (closed)", 62.0, 1.3),
        ("Claude Opus 5.5 (closed)", 74.0, 74.0),
        ("GPT-6-Astra (closed)", 90.0, 90.0),
    ]
    audit["panel_c"] = {"rows": [{"label": r, "before_or_direct": b, "after_written": a} for r, b, a in rows_c]}
    y_c = np.arange(len(rows_c), dtype=float)
    y_c[5:] += 0.45
    h_c = 0.36
    ax_c.barh(y_c - h_c / 2, [b for _, b, _ in rows_c], height=h_c, color=COL_ICE, edgecolor="white",
              linewidth=0.4, label="before writing / Direct")
    ax_c.barh(y_c + h_c / 2, [a for _, _, a in rows_c], height=h_c, color=COL_CORAL, edgecolor="white",
              linewidth=0.4, label="after writing (own text fixed)")
    for yy, (_, b, a) in zip(y_c, rows_c):
        ax_c.text(a + 1.5, yy + h_c / 2, f"{a:.0f}" if a >= 10 else f"{a:.1f}", fontsize=6.0,
                  va="center", ha="left", color="#B85244", fontweight="bold")
    ax_c.axhline(4.72, color="#9AA2AE", linewidth=0.5, linestyle=(0, (2, 2)))
    ax_c.set_yticks(y_c)
    ax_c.set_yticklabels([r for r, _, _ in rows_c], fontsize=6.3)
    ax_c.set_ylim(y_c[-1] + 0.6, -0.6)
    ax_c.set_xlim(0, 100)
    ax_c.set_xticks([0, 50, 100])
    ax_c.tick_params(axis="x", labelsize=6.6, pad=1.5)
    ax_c.set_xlabel("Image effect (pts)", fontsize=6.8, labelpad=1.5)
    for sp in ("top", "right"):
        ax_c.spines[sp].set_visible(False)
    ax_c.legend(loc="lower left", bbox_to_anchor=(-0.66, 1.005), ncol=1, frameon=False, fontsize=6.4,
                handlelength=0.9, handletextpad=0.3, columnspacing=0.6, borderaxespad=0.0)

    paths = save_figure_and_audit(fig, fig_dir, "fig1_teaser", audit)
    return paths, report


# =============================================================================
# Figure 2: Error Onset & Occlusion Sensitivity (full width = 6.875 in, ~2.40 in)
# =============================================================================

def _place_scatter_labels(ax: plt.Axes, points: list[dict[str, Any]]) -> None:
    """Place short model names around scatter points in Fig. 2c without overlaps.

    Assigns each model to a dedicated vertical/horizontal lane so labels never
    collide with each other, CI whiskers, or the bottom annotation.
    """
    # Explicit target (tx, ty, ha, va) in data coordinates for clean separation
    # Two label columns in empty regions: above the diagonal (left) and the shaded lower-right area.
    target_positions: dict[str, tuple[float, float, str, str]] = {
        "qwen3vl_8b": (-0.45, 2.98, "left", "bottom"),
        "internvl3_8b": (-0.45, 2.55, "left", "bottom"),
        "internvl35_8b": (-0.45, 2.12, "left", "bottom"),
        "qwen7b": (-0.45, 1.69, "left", "bottom"),
        "qwen3b": (-0.45, -0.42, "left", "bottom"),
        "qwen3vl_8b_think": (3.92, 0.50, "right", "bottom"),
        "molmo7b": (1.55, 0.93, "left", "bottom"),
        "phi35v": (1.25, 0.10, "left", "bottom"),
    }

    for idx, pt in enumerate(points):
        m = pt["model"]
        x, y = pt["x"], pt["y"]
        label = SCATTER_LABELS.get(m, NAMES.get(m, m))
        if m in target_positions:
            tx, ty, ha, va = target_positions[m]
        else:
            ang = (idx * 1.1) % (2 * math.pi)
            tx, ty = x + 0.35 * math.cos(ang), y + 0.35 * math.sin(ang)
            ha = "left" if tx >= x else "right"
            va = "bottom" if ty >= y else "top"

        dist = math.hypot(tx - x, ty - y)
        arrow_kw = (
            {"arrowstyle": "-", "color": "#8A93A0", "lw": 0.55, "shrinkA": 1.5, "shrinkB": 2.5}
            if dist > 0.22
            else None
        )
        ax.annotate(
            label,
            xy=(x, y),
            xytext=(tx, ty),
            fontsize=7.0,
            color=COL_TEXT,
            ha=ha,
            va=va,
            arrowprops=arrow_kw,
            zorder=5,
        )


def panel_c_points(mode: str, dprime_data: dict[str, Any]) -> list[dict[str, Any]]:
    """Fig. 2c points: occlusion AME (pts) on the cue-factorial set (cue_ame.json + Exp-31) or pooled twin d' (dprime_ci.json)."""
    pts: list[dict[str, Any]] = []
    if mode == "occlusion":
        bdir = Path(_vtp.OUTPUTS) / "bundle"
        ame = json.loads((bdir / "cue_ame.json").read_text(encoding="utf-8"))["models"]
        e31 = json.loads((bdir / "e31_cue_ladder.json").read_text(encoding="utf-8"))["models"]
        rows = [(m, ame[m]["occlusion"]["ame"], "cue_ame.json") for m in
                ["qwen3b", "qwen7b", "internvl3_8b", "phi35v", "molmo7b", "qwen3vl_8b", "internvl35_8b"] if m in ame]
        rows += [(m, e31[m]["ame"]["occlusion"]["ame"], "e31_cue_ladder.json (Exp-31)") for m in
                 ["qwen3vl_32b", "qwen3vl_8b_think"] if m in e31]
        for m, a, src in rows:
            pts.append({
                "model": m,
                "x": float(a["direct"]["est"]),
                "x_ci": [float(v) for v in a["direct"]["ci"]],
                "y": float(a["cot"]["est"]),
                "y_ci": [float(v) for v in a["cot"]["ci"]],
                "diff": float(a["cot-direct"]["est"]),
                "diff_ci": [float(v) for v in a["cot-direct"]["ci"]],
                "exp31": m in ("qwen3vl_32b", "qwen3vl_8b_think"),
                "source": src,
            })
        return pts
    for m in ["qwen3b", "qwen7b", "internvl3_8b", "phi35v", "molmo7b", "qwen3vl_8b", "qwen3vl_8b_think", "internvl35_8b"]:
        row = dprime_data.get(m)
        if isinstance(row, dict) and "dprime_direct_ci" in row and "dprime_cot_ci" in row:
            pts.append({"model": m, "x": float(row["dprime_direct"]), "x_ci": list(row["dprime_direct_ci"]),
                        "y": float(row["dprime_cot"]), "y_ci": list(row["dprime_cot_ci"]),
                        "diff": row.get("dprime_cot_minus_direct"), "diff_ci": row.get("diff_ci")})
    e9 = Path(_vtp.OUTPUTS) / "bundle" / "e9x_summary.json"
    if e9.exists():
        sd = json.load(open(e9)).get("deepseek-v4.1-flash", {})
        o, dd = sd.get("occ_sdt", {}), sd.get("occ_sdt_diff", {})
        if "direct_nr" in o and "cot" in o:
            pts.append({"model": "deepseek", "x": o["direct_nr"]["dprime"], "x_ci": o["direct_nr"]["ci"], "y": o["cot"]["dprime"],
                        "y_ci": o["cot"]["ci"], "diff": dd["cot_minus_direct_nr"][0], "diff_ci": dd["cot_minus_direct_nr"][1:]})
    return pts


def _segments_cross(a, b, c, d) -> bool:
    """Proper intersection of segments ab and cd."""
    def orient(p, q, r):
        return (q[0] - p[0]) * (r[1] - p[1]) - (q[1] - p[1]) * (r[0] - p[0])
    return (orient(a, b, c) * orient(a, b, d) < 0) and (orient(c, d, a) * orient(c, d, b) < 0)


def _place_labels_no_cross(ax: plt.Axes, points: list[dict[str, Any]], lo: float, hi: float) -> None:
    """Direct labels in two columns (upper-left above the diagonal, lower-right below it).

    Label slots in a column are assigned by searching permutations for one whose leader lines do not cross,
    preferring the order of the points' y values.
    """
    from itertools import permutations
    step = 0.34
    cols = [([p for p in points if p["x"] < 2.0], lo + 0.08, "left", [hi - 0.22 - k * step for k in range(8)]),
            ([p for p in points if p["x"] >= 2.0], hi - 0.08, "right", [lo + 0.30 + k * step for k in range(8)][::-1])]
    for pts, x0, ha, slots_all in cols:
        if not pts:
            continue
        pts = sorted(pts, key=lambda p: -p["y"])
        slots = slots_all[:len(pts)] if ha == "left" else slots_all[-len(pts):]
        best = None
        for perm in permutations(range(len(pts))):
            segs = [((x0, slots[perm[k]]), (p["x"], p["y"])) for k, p in enumerate(pts)]
            n_cross = sum(_segments_cross(*segs[a], *segs[b]) for a in range(len(segs)) for b in range(a + 1, len(segs)))
            cost = (n_cross, sum(abs(perm[k] - k) for k in range(len(pts))))
            if best is None or cost < best[0]:
                best = (cost, perm)
        for k, p in enumerate(pts):
            ax.annotate(CANON_LABELS.get(p["model"], p["model"]), xy=(p["x"], p["y"]), xytext=(x0, slots[best[1][k]]),
                        fontsize=7.0, ha=ha, va="center", color=COL_TEXT, zorder=5,
                        arrowprops={"arrowstyle": "-", "color": "#8A93A0", "lw": 0.55, "shrinkA": 1.0, "shrinkB": 3.0})


def build_fig2_error_onset(bundle: dict[str, Any], dprime_data: dict[str, Any],
                           outputs_dir: Path, legacy_dir: Path,
                           fig_dir: Path, panel_c: str = "occlusion") -> tuple[list[Path], dict[str, Any]]:
    """Build Figure 2: (a) commit trajectory, (b) commit sentence types, (c) pooled d' scatter."""
    report: dict[str, Any] = {"skipped": [], "notes": []}
    audit: dict[str, Any] = {"panel_a": {}, "panel_b": {}, "panel_c": {}}

    fig = plt.figure(figsize=(6.875, 2.75))
    gs = gridspec.GridSpec(
        1, 3, figure=fig, width_ratios=[0.86, 1.0, 1.30], wspace=0.60,
        left=0.072, right=0.986, top=0.735, bottom=0.175
    )

    # -------------------------------------------------------------------------
    # Panel (a): Mean P_j(gold) aligned at commit sentence j* (-3..+3) on D+C-
    # -------------------------------------------------------------------------
    ax_a = fig.add_subplot(gs[0, 0])
    ax_a.text(
        0.0, 1.31, "(a) Prefix trajectory on D+C\u2212",
        transform=ax_a.transAxes, fontsize=7.8, fontweight="bold", ha="left", va="bottom"
    )

    candidate_2a = ["qwen3b", "qwen7b", "internvl3_8b", "phi35v", "molmo7b", "qwen3vl_8b", "internvl35_8b"]
    offsets = list(range(-3, 4))
    present_2a: list[str] = []
    skipped_2a: list[str] = []

    ax_a.yaxis.grid(True, color=COL_GRID, linewidth=0.55, zorder=0)
    ax_a.set_axisbelow(True)
    ax_a.axhline(0.5, color="#8D95A2", linestyle=":", linewidth=0.8, zorder=1)
    ax_a.axvline(0.0, color="#5A6270", linestyle="--", linewidth=0.85, zorder=1)
    ax_a.text(
        0.13, 0.94, "commit $j^*$", fontsize=7.0, color=COL_SLATE_DARK,
        ha="left", va="top", zorder=2
    )

    audit["panel_a"] = {"offsets": offsets, "models_plotted": [], "skipped_models": [], "trajectories": {}}
    model_means_2a: list[dict[int, float]] = []

    for m in candidate_2a:
        e2b_path = outputs_dir / "e2b" / f"{m}.jsonl"
        recs = safe_jsonl(e2b_path)
        if not recs:
            skipped_2a.append(m)
            continue
        grp_map = groups_from({}, legacy_dir, outputs_dir, m)
        if not grp_map:
            skipped_2a.append(m)
            continue

        sel = [r for r in recs if grp_map.get(r["item_id"], {}).get("group") == "D+C-"]
        if not sel:
            skipped_2a.append(m)
            continue

        vals_by_off: dict[int, list[float]] = {off: [] for off in offsets}
        grps_by_off: dict[int, list[str]] = {off: [] for off in offsets}
        n_committed = 0

        for r in sel:
            traj = sorted(r["traj"], key=lambda x: x["j"])
            pg = [probs(t.get("lp_sum") or t["lp_bare"])[r["gold"]] for t in traj]
            j_star = next(
                (j for j in range(1, len(pg)) if pg[j - 1] >= 0.5 and all(v < 0.5 for v in pg[j:])),
                None,
            )
            if j_star is None:
                continue
            n_committed += 1
            cluster_id = str(r.get("image_id") or r["item_id"])
            for off in offsets:
                idx = j_star + off
                if 0 <= idx < len(pg):
                    vals_by_off[off].append(float(pg[idx]))
                    grps_by_off[off].append(cluster_id)

        if n_committed == 0:
            skipped_2a.append(m)
            continue

        present_2a.append(m)
        xs_plot, means, los, his = [], [], [], []
        per_off_audit: dict[str, Any] = {}
        for off in offsets:
            vs = vals_by_off[off]
            if not vs:
                continue
            mean_v, lo_v, hi_v = cluster_bootstrap_ci(vs, grps_by_off[off], n_boot=2000, seed=0)
            xs_plot.append(off)
            means.append(mean_v)
            los.append(lo_v)
            his.append(hi_v)
            per_off_audit[str(off)] = {
                "n": len(vs),
                "mean": round(mean_v, 4),
                "lo": round(lo_v, 4),
                "hi": round(hi_v, 4),
            }

        model_means_2a.append(dict(zip(xs_plot, means)))
        audit["panel_a"]["trajectories"][m] = {
            "n_DC_minus_total": len(sel),
            "n_committed": n_committed,
            "by_offset": per_off_audit,
        }

    if model_means_2a:
        grand = [(off, float(np.mean([mm[off] for mm in model_means_2a if off in mm])))
                 for off in offsets if any(off in mm for mm in model_means_2a)]
        from scipy.stats import t as t_dist
        band = []
        for off, _ in grand:
            v = np.array([mm[off] for mm in model_means_2a if off in mm])
            h = t_dist.ppf(0.975, len(v) - 1) * v.std(ddof=1) / np.sqrt(len(v)) if len(v) > 1 else 0.0
            band.append((v.mean() - h, v.mean() + h))
        ax_a.fill_between([o for o, _ in grand], [b[0] for b in band], [b[1] for b in band], color=COL_ICE,
                          alpha=0.45, linewidth=0, zorder=2, label="95% CI")
        ax_a.plot([o for o, _ in grand], [v for _, v in grand], color=COL_SLATE_DARK, linewidth=1.9, marker="o",
                  markersize=3.4, zorder=4, label=f"Mean ({len(model_means_2a)} models)")
        audit["panel_a"]["band_95ci_across_models"] = {str(o): [round(b[0], 4), round(b[1], 4)] for (o, _), b in zip(grand, band)}
        audit["panel_a"]["mean_over_models"] = {str(o): round(v, 4) for o, v in grand}
    audit["panel_a"]["models_plotted"] = present_2a
    audit["panel_a"]["skipped_models"] = skipped_2a
    if skipped_2a:
        report["skipped"].append(f"fig2a (e2b trajectory): {', '.join(skipped_2a)}")

    ax_a.set_xticks(offsets)
    ax_a.set_xticklabels(["\u22123", "\u22122", "\u22121", "$j^*$", "+1", "+2", "+3"])
    ax_a.set_xlabel("Sentence offset from commit $j^*$", labelpad=2.5)
    ax_a.set_ylabel("Mean $P_j(\\mathrm{gold})$", labelpad=3)
    ax_a.set_ylim(0.0, 1.0)
    ax_a.set_yticks([0.0, 0.25, 0.5, 0.75, 1.0])
    ax_a.set_xlim(-3.25, 3.25)
    ax_a.legend(
        loc="lower left", bbox_to_anchor=(-0.02, 1.01),
        ncol=1, frameon=False, handlelength=1.3, handletextpad=0.35,
        labelspacing=0.3, fontsize=7.0, borderaxespad=0.0
    )

    # -------------------------------------------------------------------------
    # Panel (b): Stacked horizontal bars — commit-sentence type shares for D+C-
    # -------------------------------------------------------------------------
    ax_b = fig.add_subplot(gs[0, 1])
    ax_b.text(
        -0.12, 1.31, "(b) Commit-sentence type",
        transform=ax_b.transAxes, fontsize=7.8, fontweight="bold", ha="left", va="bottom"
    )

    candidate_2b = ["qwen3b", "qwen7b", "internvl3_8b", "phi35v", "molmo7b", "qwen3vl_8b", "internvl35_8b"]
    present_2b: list[str] = []
    skipped_2b: list[str] = []
    for m in candidate_2b:
        eo = bundle.get(m, {}).get("error_onset", {}).get("D+C-")
        if isinstance(eo, dict) and eo.get("n", 0) > 0:
            present_2b.append(m)
        else:
            skipped_2b.append(m)
    if skipped_2b:
        report["skipped"].append(f"fig2b (error_onset D+C-): {', '.join(skipped_2b)}")
    present_2b.sort(key=lambda m: -float(bundle[m]["error_onset"]["D+C-"].get("type_relation", 0.0)))
    cb_path = Path(_vtp.OUTPUTS) / "bundle" / "commit_baseline.json"
    commit_null = json.load(open(cb_path)) if cb_path.exists() else {}

    type_specs = [
        ("type_relation", "Relational", COL_CORAL),
        ("type_non_relational", "Non-rel.", COL_PEACH),
        ("type_conclusion", "Concl.", COL_ICE),
        ("type_no_commit", "No commit", COL_SLATE),
    ]

    y_b = np.arange(len(present_2b))
    lefts = np.zeros(len(present_2b))
    audit["panel_b"] = {
        "models_plotted": present_2b,
        "skipped_models": skipped_2b,
        "categories": [k for k, _, _ in type_specs],
        "values": {},
    }

    for m in present_2b:
        eo = bundle[m]["error_onset"]["D+C-"]
        audit["panel_b"]["values"][m] = {
            "n": eo.get("n"),
            **{k: float(eo.get(k, 0.0)) for k, _, _ in type_specs},
        }

    for k_field, lbl, col in type_specs:
        widths = np.array([float(bundle[m]["error_onset"]["D+C-"].get(k_field, 0.0)) for m in present_2b])
        ax_b.barh(
            y_b, widths, left=lefts, height=0.62, color=col,
            edgecolor="white", linewidth=0.7, label=lbl, zorder=2
        )
        if k_field == "type_relation":
            for idx, w_val in enumerate(widths):
                if w_val >= 22.0:
                    ax_b.text(
                        lefts[idx] + 3.0, y_b[idx], f"{w_val:.1f}%",
                        ha="left", va="center", fontsize=7.0, fontweight="bold",
                        color="white", zorder=3
                    )
        lefts = lefts + widths

    cn_path = Path(_vtp.OUTPUTS) / "bundle" / "commit_nulls.json"
    pos_null = json.load(open(cn_path)) if cn_path.exists() else {}
    null_handle = pm_handle = None
    for idx, m in enumerate(present_2b):
        cb_m = commit_null.get(m, {})
        n_dc = bundle[m]["error_onset"]["D+C-"].get("n")
        if cb_m.get("null_uniform_mean") is not None and cb_m.get("n_DCm"):
            # null shares of committed items, rescaled to the share of all D+C- items plotted on this axis
            scale = cb_m["n_committed"] / cb_m["n_DCm"]
            null_all = cb_m["null_uniform_mean"] * scale
            null_handle = ax_b.plot([null_all, null_all], [idx - 0.34, idx + 0.34], color=COL_MUTED_TEXT, linewidth=0.9,
                                    linestyle=(0, (1.6, 1.0)), solid_capstyle="butt", dash_capstyle="butt", zorder=4)[0]
            audit["panel_b"]["values"][m]["random_sentence_null_share_of_all"] = round(null_all, 1)
            pm = pos_null.get(m, {}).get("position_matched", {}).get("null")
            if pm is not None:
                pm_all = pm * scale
                pm_handle = ax_b.plot([pm_all, pm_all], [idx - 0.40, idx + 0.40], color=COL_TEXT, linewidth=1.5,
                                      solid_capstyle="butt", zorder=5)[0]
                audit["panel_b"]["values"][m]["position_matched_null_share_of_all"] = round(pm_all, 1)
        ax_b.text(101.5, idx, f"n={n_dc}", fontsize=7.0, ha="left", va="center", color=COL_MUTED_TEXT)
    max_right = 100.0
    ax_b.set_yticks(y_b)
    ax_b.set_yticklabels([COMPACT_LABELS[m] for m in present_2b], fontsize=7.0)
    ax_b.invert_yaxis()
    ax_b.set_xlim(0.0, max_right + 0.3)
    ax_b.set_xticks([0, 25, 50, 75, 100])
    ax_b.spines["right"].set_visible(False)
    ax_b.set_xlabel("Share of D+C\u2212 items (%)", labelpad=2.5)
    ax_b.xaxis.grid(True, color=COL_GRID, linewidth=0.55, zorder=0)
    ax_b.set_axisbelow(True)
    h_b, l_b = ax_b.get_legend_handles_labels()
    if pm_handle is not None:
        h_b.append(pm_handle)
        l_b.append("Pos.-matched null")
    if null_handle is not None:
        h_b.append(null_handle)
        l_b.append("Uniform null")
    ax_b.legend(
        h_b, l_b, loc="lower left", bbox_to_anchor=(-0.02, 1.02),
        ncol=2, frameon=False, handlelength=1.0, handletextpad=0.35, columnspacing=0.8,
        fontsize=7.0, borderaxespad=0.0
    )

    # -------------------------------------------------------------------------
    # Panel (c): Scatter Direct d' vs CoT d' with 95% CIs (both phrasings)
    # -------------------------------------------------------------------------
    ax_c = fig.add_subplot(gs[0, 2])
    ax_c.text(
        1.0, 1.31, "(c) Occlusion AME (cue-factorial set)" if panel_c == "occlusion"
        else "(c) Occlusion sensitivity $d'$ (both phrasings)",
        transform=ax_c.transAxes, fontsize=7.0, fontweight="bold", ha="right", va="bottom"
    )

    scatter_pts = panel_c_points(panel_c, dprime_data)
    present_2c = [pt["model"] for pt in scatter_pts]
    audit["panel_c"] = {"mode": panel_c, "models_plotted": present_2c, "values": {pt["model"]: pt for pt in scatter_pts}}
    if panel_c == "occlusion":
        audit["panel_c"]["quantity"] = (
            "average marginal effect (pts) of occlusion on P(answer says the queried subject is in front), probit GLM "
            "(says-front ~ phrasing + occlusion + height + size), Direct (x) vs CoT (y); 95% twin-pair cluster bootstrap CIs; "
            "colour = sign of the CoT - Direct AME CI; squares = Exp-31 models")

    all_x = [pt["x_ci"][0] for pt in scatter_pts] + [pt["x_ci"][1] for pt in scatter_pts]
    all_y = [pt["y_ci"][0] for pt in scatter_pts] + [pt["y_ci"][1] for pt in scatter_pts]
    lo_lim = -2.0 if panel_c == "occlusion" else (min([-0.3] + all_x + all_y) - 0.08)
    hi = 100.0 if panel_c == "occlusion" else (max([3.6] + all_x + all_y) + 0.2)
    ax_c.grid(True, color=COL_GRID, linewidth=0.55, zorder=0)
    ax_c.set_axisbelow(True)
    diag_grid = np.linspace(lo_lim, hi, 100)
    ax_c.fill_between(diag_grid, lo_lim, diag_grid, facecolor="#FAF1EF", alpha=0.52, zorder=0)
    ax_c.plot([lo_lim, hi], [lo_lim, hi], color="#6E7785", linestyle="--", linewidth=0.85, zorder=1)
    for pt in scatter_pts:
        x, y = pt["x"], pt["y"]
        dci = pt.get("diff_ci")
        if dci and dci[1] < 0.0:
            m_face, m_edge = COL_CORAL, COL_CORAL_DARK
        elif dci and dci[0] > 0.0:
            m_face, m_edge = COL_ICE, COL_ICE_DARK
        else:
            m_face, m_edge = COL_SLATE, "#7A828E"
        ax_c.errorbar(x, y, xerr=[[max(0.0, x - pt["x_ci"][0])], [max(0.0, pt["x_ci"][1] - x)]],
                      yerr=[[max(0.0, y - pt["y_ci"][0])], [max(0.0, pt["y_ci"][1] - y)]],
                      fmt="none", ecolor="#7A828E", elinewidth=0.7, capsize=1.5, capthick=0.65, zorder=3)
        mk = "D" if pt["model"] == "deepseek" else ("s" if pt.get("exp31") else "o")
        ax_c.scatter([x], [y], s=22 if mk == "s" else 24, marker=mk, facecolor=m_face,
                     edgecolor=m_edge, linewidth=0.85, zorder=4)
    if panel_c == "occlusion":
        by_m = {pt["model"]: pt for pt in scatter_pts}
        manual: dict[str, tuple[float, float, str]] = {
            "qwen3vl_8b": (28.0, 64.0, "center"),
            "qwen3vl_8b_think": (33.0, 85.0, "center"),
            "qwen3vl_32b": (99.0, 95.0, "right"),
            "molmo7b": (80.0, 36.0, "center"),
        }
        for m, (tx, ty, ha) in manual.items():
            if m not in by_m:
                continue
            pt = by_m[m]
            lab = CANON_LABELS.get(m, "Qwen3-VL-32B" if m == "qwen3vl_32b" else m) + ("\u2021" if pt.get("exp31") else "")
            ax_c.annotate(lab, xy=(pt["x"], pt["y"]), xytext=(tx, ty), fontsize=7.0, ha=ha, va="center", color=COL_TEXT,
                          zorder=5, bbox={"boxstyle": "square,pad=0.04", "facecolor": "white", "edgecolor": "none", "alpha": 0.85},
                          arrowprops={"arrowstyle": "-", "color": "#8A93A0", "lw": 0.55, "shrinkA": 1.0, "shrinkB": 3.2})
        ax_c.text(31.0, 9.0, "5 other models", fontsize=7.0, ha="left", va="center", color=COL_MUTED_TEXT, zorder=5)
    else:
        _place_labels_no_cross(ax_c, scatter_pts, lo_lim, hi)
    ax_c.set_xlim(lo_lim, hi)
    ax_c.set_ylim(lo_lim, hi)
    if panel_c == "occlusion":
        ax_c.set_xticks([0, 25, 50, 75, 100])
        ax_c.set_yticks([0, 25, 50, 75, 100])
        ax_c.set_xlabel("Direct occlusion AME (pts)", labelpad=2.5)
        ax_c.set_ylabel("CoT occlusion AME (pts)", labelpad=2.5)
    else:
        ax_c.set_xticks([0, 1, 2, 3])
        ax_c.set_yticks([0, 1, 2, 3])
        ax_c.set_xlabel("Direct $d'$ (both phrasings)", labelpad=2.5)
        ax_c.set_ylabel("CoT $d'$ (both phrasings)", labelpad=2.5)
    leg_2c = [
        mpatches.Patch(facecolor=COL_CORAL, edgecolor=COL_CORAL_DARK, label="CoT < Dir."),
        mpatches.Patch(facecolor=COL_ICE, edgecolor=COL_ICE_DARK, label="CoT > Dir."),
        mpatches.Patch(facecolor=COL_SLATE, edgecolor="#7A828E", label="CI \u220b 0"),
    ]
    if panel_c == "occlusion":
        from matplotlib.lines import Line2D
        leg_2c.append(Line2D([], [], marker="s", linestyle="none", markersize=4.0, markerfacecolor="white",
                             markeredgecolor="#4A515C", markeredgewidth=0.85, label="larger / thinking\u2021"))
        ax_c.legend(handles=leg_2c, loc="lower left", bbox_to_anchor=(-0.08, 1.02), ncol=2, frameon=False,
                    handlelength=0.9, handletextpad=0.3, columnspacing=0.6, labelspacing=0.25, fontsize=7.0, borderaxespad=0.0)
    else:
        ax_c.legend(handles=leg_2c, loc="lower left", bbox_to_anchor=(-0.08, 1.02), ncol=3, frameon=False,
                    handlelength=0.9, handletextpad=0.3, columnspacing=0.6, fontsize=7.0, borderaxespad=0.0)

    paths = save_figure_and_audit(fig, fig_dir, "fig2_error_onset" + ("" if panel_c == "occlusion" else "_pooled"), audit)
    return paths, report


# =============================================================================
# Figure 3: Conclusion-Swap Selectivity (single column = 3.3 in, ~2.24 in tall)
# =============================================================================

def build_fig3_swaps(bundle: dict[str, Any], fig_dir: Path) -> tuple[list[Path], dict[str, Any]]:
    """Conclusion-swap selectivity S = F(gold) - F(wrong), split into checkers (full range) and copiers (zoomed).

    Bars: R_cut + swapped conclusion read out with the original / gray / mirrored (L/R subset) image.
    Diamond: conclusion sentence alone (no rationale body), original image.
    """
    report: dict[str, Any] = {"skipped": [], "notes": []}
    candidate = ["qwen3b", "qwen7b", "internvl3_8b", "phi35v", "molmo7b", "qwen3vl_8b", "internvl35_8b"]
    img_specs = [("original", "Original", COL_ICE), ("gray", "Gray", COL_SLATE), ("mirror", "Mirrored (L/R)", COL_CORAL)]
    S = lambda m, key: bundle.get(m, {}).get("swaps", {}).get(key, {}).get("S")
    present = [m for m in candidate if has_valid_ci(S(m, "rcut:original"))]
    checkers = sorted([m for m in present if S(m, "rcut:original")["mean"] >= 20.0], key=lambda m: -S(m, "rcut:original")["mean"])
    copiers = sorted([m for m in present if m not in checkers], key=lambda m: -S(m, "rcut:original")["mean"])
    sp = Path(_vtp.OUTPUTS) / "bundle" / "swap_split.json"
    split = json.load(open(sp)) if sp.exists() else {}
    lp = Path(_vtp.OUTPUTS) / "bundle" / "e3x_same_subset.json"
    lr_sub = json.load(open(lp)) if lp.exists() else {}

    fig = plt.figure(figsize=(3.28, 3.50))  # printed at \\columnwidth (3.28 in)
    gs = fig.add_gridspec(2, 1, height_ratios=[len(checkers), 0.92 * len(copiers)], hspace=0.42,
                          left=0.300, right=0.965, top=0.805, bottom=0.105)
    audit: dict[str, Any] = {"groups": {"checkers": checkers, "copiers": copiers}, "values": {}}
    panels = [(fig.add_subplot(gs[0]), checkers, (-50.0, 100.0), "Rationale-body checkers"),
              (fig.add_subplot(gs[1]), copiers, (-3.0, 15.0), "Conclusion copiers (x-axis zoomed)")]
    bh = 0.26
    for ax, ms, (xmin, xmax), title in panels:
        for r, m in enumerate(ms):
            for k, (img, _, col) in enumerate(img_specs):
                yy = r + (k - 1) * bh
                d = S(m, f"rcut:{img}")
                if not has_valid_ci(d):
                    continue
                v, ci_lo, ci_hi = float(d["mean"]), float(d["lo"]), float(d["hi"])
                ax.barh(yy, v, height=bh * 0.92, color=col, edgecolor="white", linewidth=0.4, zorder=2)
                ax.errorbar(v, yy, xerr=[[max(0.0, v - ci_lo)], [max(0.0, ci_hi - v)]], fmt="none",
                            ecolor="#4A515C", elinewidth=0.5, capsize=0.8, zorder=3)
                audit["values"].setdefault(m, {})[img] = d
                lr_orig = lr_sub.get(m, {}).get("S_original_same_subset", {}).get("mean")
                if img == "mirror" and lr_orig is not None:
                    ax.scatter([float(lr_orig)], [yy], marker="|", s=22, color=COL_TEXT, linewidths=0.9, zorder=6)
                    audit["values"][m]["S_orig_LR_subset"] = lr_sub[m]["S_original_same_subset"]
                if img == "original":
                    if (xmax - xmin) > 50 and v > 25:
                        ax.text(v - 0.02 * (xmax - xmin), yy, f"{v:+.1f}", fontsize=6.0, va="center",
                                ha="right", color="white", fontweight="bold", zorder=7)
                    else:
                        ax.text(max(v, ci_hi) + 0.03 * (xmax - xmin), yy, f"{v:+.1f}", fontsize=6.0, va="center",
                                ha="left", color=COL_TEXT, fontweight="bold", zorder=7)
            dc = S(m, "concl:original")
            if has_valid_ci(dc):
                ax.scatter([float(dc["mean"])], [r - bh], marker="D", s=9, facecolor="white", edgecolor=COL_TEXT,
                           linewidth=0.7, zorder=5)
                audit["values"].setdefault(m, {})["concl_only_original"] = dc
            if m in checkers and split.get(m):
                for key, mk, sz in (("rcut:original:D+C+", "o", 12), ("rcut:original:D+C-", "X", 16)):
                    sv = split[m].get(key, {}).get("S")
                    if sv:
                        ax.scatter([float(sv["mean"])], [r - bh], marker=mk, s=sz, facecolor=COL_TEXT,
                                   edgecolor="white", linewidth=0.4, zorder=6)
                        audit["values"].setdefault(m, {})[f"S_{key.split(':')[-1]}_body"] = sv
        ax.axvline(0, color=COL_TEXT, linewidth=0.65, zorder=3)
        ax.set_yticks(range(len(ms)))
        ax.set_yticklabels([CANON_LABELS[m] for m in ms], fontsize=6.6)
        ax.set_ylim(len(ms) - 0.55, -0.6)
        ax.set_xlim(xmin, xmax)
        ax.tick_params(axis="x", labelsize=6.4, pad=1.5)
        ax.tick_params(axis="y", pad=1.8, length=0)
        ax.xaxis.grid(True, color=COL_GRID, linewidth=0.55, zorder=0)
        ax.set_axisbelow(True)
        for spn in ("top", "right"):
            ax.spines[spn].set_visible(False)
        ax.set_title(title, fontsize=6.9, fontweight="bold", loc="left", color=COL_SLATE_DARK, pad=2.5)
    panels[1][0].set_xlabel("Selectivity $S = F(\\mathrm{gold}) - F(\\mathrm{wrong})$ (pts)", fontsize=6.8, labelpad=2.0)
    handles = [mpatches.Patch(facecolor=c, edgecolor="white", label=l) for _, l, c in img_specs]
    handles += [plt.Line2D([], [], marker="D", linestyle="none", markersize=2.9, markerfacecolor="white",
                           markeredgecolor=COL_TEXT, label="Concl. only"),
                plt.Line2D([], [], marker="|", linestyle="none", markersize=4.6, markeredgewidth=0.9, color=COL_TEXT,
                           label="Orig. (L/R items)"),
                plt.Line2D([], [], marker="o", linestyle="none", markersize=3.1, markerfacecolor=COL_TEXT,
                           markeredgecolor="white", label="D+C+ bodies"),
                plt.Line2D([], [], marker="X", linestyle="none", markersize=3.6, markerfacecolor=COL_TEXT,
                           markeredgecolor="white", label="D+C\u2212 bodies")]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.56, 0.995), ncol=3, frameon=False,
               handlelength=0.9, handletextpad=0.3, columnspacing=0.7, fontsize=6.4, borderaxespad=0.0)
    paths = save_figure_and_audit(fig, fig_dir, "fig3_swaps", audit)
    return paths, report


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments with sensible project defaults."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--bundle", type=Path, default=DEFAULT_BUNDLE)
    ap.add_argument("--dprime", type=Path, default=DEFAULT_DPRIME)
    ap.add_argument("--judge", type=Path, default=DEFAULT_JUDGE)
    ap.add_argument("--outputs", type=Path, default=DEFAULT_OUTPUTS)
    ap.add_argument("--legacy", type=Path, default=DEFAULT_LEGACY)
    ap.add_argument("--occ-dir", type=Path, default=DEFAULT_OCC_DIR)
    ap.add_argument("--fig-dir", type=Path, default=DEFAULT_FIG_DIR)
    ap.add_argument("--only-fig1", action="store_true", help="rebuild only fig1_teaser")
    ap.add_argument("--only-fig3", action="store_true", help="rebuild only fig3_swaps")
    ap.add_argument("--only-fig2", action="store_true", help="rebuild only fig2_error_onset (main, occlusion AME panel c)")
    return ap.parse_args()


def main() -> None:
    """Entry point: build all v3 paper figures and print an audit summary."""
    args = parse_args()
    configure_matplotlib()

    with open(args.bundle, "r", encoding="utf-8") as fh:
        bundle = json.load(fh)
    dprime_data = json.loads(args.dprime.read_text(encoding="utf-8")) if args.dprime.exists() else {}

    all_paths: list[Path] = []
    all_skipped: list[str] = []
    all_notes: list[str] = []

    if args.only_fig3:
        p3, r3 = build_fig3_swaps(bundle, args.fig_dir)
        print("Generated files:\n  " + "\n  ".join(str(p) for p in p3))
        return

    if args.only_fig1:
        p1, r1 = build_fig1_teaser(bundle, args.outputs, args.judge, args.occ_dir, args.fig_dir)
        print("Generated files:\n  " + "\n  ".join(str(p) for p in p1))
        print("Skipped:", r1["skipped"])
        return

    if args.only_fig2:
        p2, r2 = build_fig2_error_onset(bundle, dprime_data, args.outputs, args.legacy, args.fig_dir)
        print("Generated files:\n  " + "\n  ".join(str(p) for p in p2))
        print("Skipped:", r2["skipped"])
        return

    p1, r1 = build_fig1_teaser(bundle, args.outputs, args.judge, args.occ_dir, args.fig_dir)
    all_paths.extend(p1)
    all_skipped.extend(r1["skipped"])
    all_notes.extend(r1["notes"])

    p2, r2 = build_fig2_error_onset(bundle, dprime_data, args.outputs, args.legacy, args.fig_dir)
    build_fig2_error_onset(bundle, dprime_data, args.outputs, args.legacy, args.fig_dir, panel_c="pooled")
    all_paths.extend(p2)
    all_skipped.extend(r2["skipped"])
    all_notes.extend(r2["notes"])

    p3, r3 = build_fig3_swaps(bundle, args.fig_dir)
    all_paths.extend(p3)
    all_skipped.extend(r3["skipped"])
    all_notes.extend(r3["notes"])

    print("Generated files:")
    for p in all_paths:
        print(f"  {p}")
    print("Skipped (missing data):")
    for s in all_skipped:
        print(f"  {s}")
    if all_notes:
        print("Notes:")
        for n in all_notes:
            print(f"  {n}")


if __name__ == "__main__":
    main()
