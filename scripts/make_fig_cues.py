"""Generate the two-panel cue-decomposition and scale-ladder figure (Fig. 3).

Outputs written to ../paper/figures/:
  - fig_cues.{pdf,png,data.json}:
      (a) Probit-GLM CoT - Direct d' components (occlusion / vertical / size)
          per model with 95% twin-pair bootstrap CIs on the cue-factorial set.
      (b) Qwen3-VL scale ladder (2B, 4B, 8B, 32B) on the original twins:
          CoT - Direct occlusion-minus-size component with 95% CIs, plus
          front-lower Direct and CoT accuracy markers.
  - fig_cues_cells.{pdf,png}: per-cell alternative plot.

Run from vtax_exps/:
    python3 make_fig_cues.py
"""
from __future__ import annotations
import vtax_paths as _vtp  # noqa: E402

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as mpatches  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
import numpy as np  # noqa: E402

SCRIPT_DIR = Path(__file__).resolve().parent
B = Path(_vtp.OUTPUTS) / "bundle"
FIG = SCRIPT_DIR.parent / f"{_vtp.PAPER}" / "figures"

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

MODELS = [
    "qwen3b",
    "qwen7b",
    "internvl3_8b",
    "phi35v",
    "internvl35_8b",
    "molmo7b",
    "qwen3vl_8b",
    "deepseek",
]
EXP31 = ["qwen3vl_32b", "qwen3vl_8b_think"]
CANON_LABELS["qwen3vl_32b"] = "Qwen3-VL-32B"
LADDER_SIZES = ["2b", "4b", "8b", "32b"]
LADDER_LABELS = {"2b": "2B", "4b": "4B", "8b": "8B", "32b": "32B"}

COL_OCC = "#3B6998"
COL_VERT = "#E2B28F"
COL_SIZE = "#B9BEC6"
COL_TEXT = "#1F242B"
COL_MUTED = "#4A515C"
COL_SPINE = "#3C434D"
COL_GRID = "#E6E9EE"
COL_ACC_COT = "#B85244"

COMPS = [
    ("occlusion", "Occlusion (valid cue)", COL_OCC),
    ("vertical", "Height (lower = closer; shortcut)", COL_VERT),
    ("size", "Size (larger = closer)", COL_SIZE),
]
COMP_MARKER = {"occlusion": ("o", "#24476B"), "vertical": ("s", "#A86E45"), "size": ("^", "#6E7582")}
CELLS = [
    ("front_larger|front_lower", "larger, lower", "#3B6998"),
    ("front_larger|front_higher", "larger, higher", "#9BBAD9"),
    ("front_smaller|front_lower", "smaller, lower", "#E2B28F"),
    ("front_smaller|front_higher", "smaller, higher", "#B85244"),
]


def key_for_cues(model: str) -> str:
    """Map canonical model ID to cues.json key."""
    return "api:deepseek-v4.1-flash" if model == "deepseek" else model


def style() -> None:
    """Configure single-column typography (~6.8-7.6 pt, matching Figs. 1-2) and 0.7 pt spines."""
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
            "font.size": 6.8,
            "axes.titlesize": 7.6,
            "axes.labelsize": 7.0,
            "xtick.labelsize": 6.8,
            "ytick.labelsize": 6.8,
            "legend.fontsize": 6.8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.edgecolor": COL_SPINE,
            "axes.linewidth": 0.7,
            "xtick.major.width": 0.65,
            "ytick.major.width": 0.65,
            "xtick.major.size": 2.5,
            "ytick.major.size": 2.5,
            "xtick.color": COL_TEXT,
            "ytick.color": COL_TEXT,
            "text.color": COL_TEXT,
            "axes.labelcolor": COL_TEXT,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def glm_components() -> dict[str, dict[str, dict[str, Any]]]:
    """Load CoT - Direct d' components from probit GLM bundles."""
    with open(B / "glm_decomp.json", "r", encoding="utf-8") as fh:
        g = json.load(fh)["cues"]
    out: dict[str, dict[str, dict[str, Any]]] = {
        m: {
            c: {"d": g[m]["cot-direct"][c]["dprime"], "ci": g[m]["cot-direct"][c]["ci"]}
            for c, _, _ in COMPS
        }
        for m in g
    }
    with open(B / "deepseek_glm.json", "r", encoding="utf-8") as fh:
        ds = json.load(fh)["cot-direct"]
    out["deepseek"] = {
        c: {"d": ds[c]["dprime"], "ci": ds[c]["ci"]} for c, _, _ in COMPS
    }
    return out


def ame_components() -> dict[str, dict[str, dict[str, Any]]]:
    """Load CoT - Direct average marginal effects (pts) for the seven open models (cue_ame.json) and Exp-31 models."""
    with open(B / "cue_ame.json", "r", encoding="utf-8") as fh:
        a = json.load(fh)["models"]
    out = {
        m: {c: {"d": a[m][c]["ame"]["cot-direct"]["est"], "ci": a[m][c]["ame"]["cot-direct"]["ci"],
                "direct": a[m][c]["ame"]["direct"]["est"], "cot": a[m][c]["ame"]["cot"]["est"]} for c, _, _ in COMPS}
        for m in a
    }
    with open(B / "e31_cue_ladder.json", "r", encoding="utf-8") as fh:
        e = json.load(fh)["models"]
    for m in EXP31:
        out[m] = {c: {"d": e[m]["ame"][c]["ame"]["cot-direct"]["est"], "ci": e[m]["ame"][c]["ame"]["cot-direct"]["ci"],
                      "direct": e[m]["ame"][c]["ame"]["direct"]["est"], "cot": e[m]["ame"][c]["ame"]["cot"]["est"]}
                  for c, _, _ in COMPS}
    return out


def load_ladder() -> dict[str, dict[str, Any]]:
    """Load Qwen3-VL scale ladder occlusion-minus-size components and front-lower accuracies."""
    with open(B / "ladder_cues.json", "r", encoding="utf-8") as fh:
        raw = json.load(fh)
    out: dict[str, dict[str, Any]] = {}
    for sz in LADDER_SIZES:
        halves = raw[sz]["halves"]
        occ_sz = halves["cot-direct"]["occ_minus_size"]
        out[sz] = {
            "label": f"Qwen3-VL-{LADDER_LABELS[sz]}",
            "occ_minus_size": {"d": occ_sz["d"], "ci": occ_sz["ci"]},
            "vertical": {
                "d": halves["cot-direct"]["vertical"]["d"],
                "ci": halves["cot-direct"]["vertical"]["ci"],
            },
            "front_lower_acc": {
                "direct": halves["direct"]["front_lower"]["acc"],
                "cot": halves["cot"]["front_lower"]["acc"],
            },
            "front_higher_acc": {
                "direct": halves["direct"]["front_higher"]["acc"],
                "cot": halves["cot"]["front_higher"]["acc"],
            },
        }
    return out


def build_fig_cues(fig_dir: Path = FIG) -> None:
    """Build two-panel Figure 3 at exact single-column width (3.28 in)."""
    comp = ame_components()
    ladder = load_ladder()

    opens = sorted([m for m in MODELS if m in comp], key=lambda m: comp[m]["occlusion"]["d"])
    extra = sorted([m for m in EXP31 if m in comp], key=lambda m: comp[m]["occlusion"]["d"])
    present = opens + extra

    # Exact \columnwidth = 3.28125 in -> ~6.8-7.6 pt typography matching Figs. 1-2
    fig = plt.figure(figsize=(3.28, 4.30))

    # -------------------------------------------------------------------------
    # Panel (a): CoT - Direct average marginal effects (probability scale)
    # -------------------------------------------------------------------------
    fig.text(
        0.024,
        0.984,
        "(a) Cue effects on the answer, CoT \u2212 Direct (AME)",
        fontsize=7.6,
        fontweight="bold",
        ha="left",
        va="top",
        color=COL_TEXT,
    )

    ax_a = fig.add_axes([0.390, 0.432, 0.592, 0.408])
    xlim_a = (-60.0, 21.0)
    ax_a.set_xlim(*xlim_a)
    ax_a.tick_params(axis="y", pad=2.5)

    y_positions = [float(i) for i in range(len(opens))]
    divider_y = len(opens) - 0.42
    y_positions += [float(len(opens)) + 0.62 + k for k in range(len(extra))]

    h = 0.27
    for j, (c, lab, col) in enumerate(COMPS):
        mk, edge = COMP_MARKER[c]
        for i, m in enumerate(present):
            d_entry = comp[m][c]
            val = float(d_entry["d"])
            ci_lo, ci_hi = float(d_entry["ci"][0]), float(d_entry["ci"][1])
            y = y_positions[i] + (j - 1) * h
            ax_a.plot([ci_lo, ci_hi], [y, y], color=edge, linewidth=0.75, solid_capstyle="butt", zorder=2)
            ax_a.plot(
                val, y, linestyle="none", marker=mk, markersize=3.6 if mk != "^" else 3.9,
                markerfacecolor=col, markeredgecolor=edge, markeredgewidth=0.6, zorder=3,
                label=lab if i == 0 else None,
            )

    ax_a.axvline(0.0, color="#2F3640", linewidth=0.75, zorder=1)
    if extra:
        ax_a.axhline(divider_y, color="#8D95A2", linewidth=0.7, linestyle=(0, (2.5, 2.0)), zorder=1)
        ax_a.text(xlim_a[0] + 1.0, divider_y + 0.08, "\u2021 larger / thinking (cue set)", fontsize=6.8, color=COL_MUTED,
                  ha="left", va="top", zorder=4)

    ax_a.xaxis.grid(True, color=COL_GRID, linewidth=0.55, zorder=0)
    ax_a.set_axisbelow(True)
    ax_a.set_yticks(y_positions)
    ax_a.set_yticklabels([CANON_LABELS[m] + ("\u2021" if m in EXP31 else "") for m in present])
    ax_a.set_ylim(y_positions[-1] + 0.50, -0.50)
    ax_a.set_xticks([-60, -40, -20, 0, 20])
    ax_a.set_xlabel("CoT \u2212 Direct, probability-scale effect (pts)", labelpad=2.0)

    handles_a, labels_a = ax_a.get_legend_handles_labels()
    fig.legend(
        handles_a,
        labels_a,
        loc="upper left",
        bbox_to_anchor=(0.020, 0.952),
        ncol=1,
        frameon=False,
        handlelength=0.95,
        handletextpad=0.35,
        labelspacing=0.22,
        borderaxespad=0.0,
    )

    # -------------------------------------------------------------------------
    # Panel (b): where CoT losses end (D+C- items; Tab. supp_e54a pooled rows:
    # false last body relation / true relation with a mis-mapped conclusion / no relation)
    # -------------------------------------------------------------------------
    fig.text(0.024, 0.340, "(b) Where CoT losses end (Direct \u2713, CoT \u2717)",
             fontsize=7.6, fontweight="bold", ha="left", va="top", color=COL_TEXT)
    loss_rows = [
        ("Qwen3-VL-8B", 96, 2, 2), ("InternVL3.5-8B", 83, 6, 11), ("Qwen2.5-VL-3B", 66, 7, 27),
        ("Qwen2.5-VL-7B", 64, 22, 15), ("InternVL3-8B", 52, 23, 25), ("Phi-3.5-Vision", 38, 28, 34),
        ("Molmo-7B-D", 23, 61, 16), ("All seven", 57, 24, 20),
    ]
    ax_b = fig.add_axes([0.390, 0.072, 0.570, 0.212])
    seg_cols = ("#DC8A7E", "#E2B28F", "#C9CED6")
    seg_labels = ("false relation", "true relation, wrong conclusion", "no relation")
    for r_i, (name, a, b, c) in enumerate(loss_rows):
        tot = float(a + b + c)
        vals = [100.0 * v / tot for v in (a, b, c)]
        left = 0.0
        for k, v in enumerate(vals):
            ax_b.barh(r_i, v, left=left, height=0.68, color=seg_cols[k], edgecolor="white", linewidth=0.6)
            if k < 2 and v >= 12:
                ax_b.text(left + v / 2, r_i, f"{(a, b)[k]}", ha="center", va="center", fontsize=6.2,
                          color="white" if k == 0 else COL_TEXT, fontweight="bold")
            left += v
    ax_b.axhline(len(loss_rows) - 1.5, color="#9AA2AE", linewidth=0.5, linestyle=(0, (2, 2)))
    ax_b.set_yticks(range(len(loss_rows)))
    ax_b.set_yticklabels([r[0] for r in loss_rows])
    ax_b.set_ylim(len(loss_rows) - 0.5, -0.5)
    ax_b.set_xlim(0, 100)
    ax_b.set_xticks([0, 50, 100])
    ax_b.set_xlabel("Share of D+C\u2212 losses (%)", labelpad=1.5)
    for sp in ("top", "right"):
        ax_b.spines[sp].set_visible(False)
    fig.legend([mpatches.Patch(facecolor=c, edgecolor="white") for c in seg_cols], seg_labels,
               loc="upper left", bbox_to_anchor=(0.03, 0.312), ncol=3, frameon=False, fontsize=6.0,
               handlelength=0.8, handletextpad=0.25, columnspacing=0.5, borderaxespad=0.0)

    fig_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(fig_dir / "fig_cues.pdf")
    fig.savefig(fig_dir / "fig_cues.png", dpi=300)
    plt.close(fig)

    audit_payload = {
        "panel_a": {
            "models_ordered": [CANON_LABELS[m] for m in present],
            "exp31_models": [CANON_LABELS[m] for m in extra],
            "ame_points": {CANON_LABELS[m]: comp[m] for m in present},
            "quantity": (
                "average marginal effect (pts) of each cue on P(answer says the queried subject is in front): mean over items "
                "of Phi(x_cue=+1) - Phi(x_cue=-1) from the probit GLM (says-front ~ phrasing + occlusion + height + size); "
                "plotted value d = CoT - Direct; direct / cot = per-arm AME"
            ),
            "ci": "95% twin-pair (image_id) cluster bootstrap, Direct and CoT resampled jointly (1,000 draws)",
            "sources": "cue_ame.json (r18_cue_ame.py, Tab. S24) for the seven open instruct models; "
                       "e31_cue_ladder.json (rigor_e31.py, Exp-31) for Qwen3-VL-32B and Qwen3-VL-8B-Thinking",
            "sort": "by occlusion CoT - Direct AME, within group",
            "dropped": "DeepSeek-V4.1-Flash (no AME computed; only probit weights in deepseek_glm.json)",
        },
        "panel_b": {
            "quantity": "share of D+C- items by the last relation of the conclusion-removed rationale (two-judge consensus)",
            "source": "Tab. supp_e54a pooled rows (cue set, twins, Synthetic Depth 431); rows normalized to 100",
            "rows": loss_rows,
        },
    }
    with open(fig_dir / "fig_cues.data.json", "w", encoding="utf-8") as fh:
        json.dump(audit_payload, fh, indent=2, ensure_ascii=False)


def cells(cu: dict[str, Any], fig_dir: Path = FIG) -> None:
    """Render alternative per-cell CoT - Direct d' plot."""
    present = [
        m for m in MODELS if cu.get(key_for_cues(m), {}).get("cues", {}).get("cells")
    ]
    fig, ax = plt.subplots(figsize=(3.3, 2.4))
    fig.subplots_adjust(left=0.14, right=0.995, top=0.76, bottom=0.2)
    x, w = np.arange(len(present)), 0.19
    for j, (cell, lab, col) in enumerate(CELLS):
        vals, lo, hi = [], [], []
        for m in present:
            c = cu[key_for_cues(m)]["cues"]["cells"].get(cell, {}).get("cot-direct")
            vals.append(c["dprime"] if c else np.nan)
            lo.append(c["dprime"] - c["dprime_ci"][0] if c else 0)
            hi.append(c["dprime_ci"][1] - c["dprime"] if c else 0)
        pos = x + (j - 1.5) * w
        ax.bar(
            pos,
            vals,
            w,
            color=col,
            edgecolor="white",
            linewidth=0.4,
            label=lab,
            zorder=2,
        )
        ax.errorbar(
            pos,
            vals,
            yerr=[lo, hi],
            fmt="none",
            ecolor=COL_SPINE,
            elinewidth=0.6,
            capsize=1.0,
            zorder=3,
        )
    ax.axhline(0, color="#2F3640", linewidth=0.7, zorder=3)
    ax.yaxis.grid(True, color=COL_GRID, linewidth=0.5, zorder=0)
    ax.set_axisbelow(True)
    ax.set_xticks(x)
    ax.set_xticklabels(
        [CANON_TWO_LINE[m] for m in present], linespacing=0.95, fontsize=7.0
    )
    ax.set_ylabel("CoT \u2212 Direct $d'$", labelpad=1.5)
    ax.set_xlim(-0.5, len(present) - 0.5)
    ax.legend(
        title="front object is",
        title_fontsize=7.0,
        loc="lower left",
        bbox_to_anchor=(-0.16, 1.0),
        ncol=2,
        frameon=False,
        handlelength=0.9,
        handletextpad=0.3,
        columnspacing=0.55,
        borderaxespad=0.2,
    )
    fig.savefig(fig_dir / "fig_cues_cells.pdf")
    fig.savefig(fig_dir / "fig_cues_cells.png", dpi=300)
    plt.close(fig)


def main() -> None:
    """Entry point for generating cue decomposition figures."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--fig-dir", type=Path, default=FIG)
    ap.add_argument("--cells", action="store_true", help="also regenerate fig_cues_cells")
    args = ap.parse_args()
    style()
    with open(B / "cues.json", "r", encoding="utf-8") as fh:
        cu = json.load(fh)
    build_fig_cues(args.fig_dir)
    if args.cells:
        cells(cu, args.fig_dir)
    print("wrote fig_cues.{pdf,png,data.json}" + (" and fig_cues_cells.{pdf,png}" if args.cells else ""))


if __name__ == "__main__":
    main()
