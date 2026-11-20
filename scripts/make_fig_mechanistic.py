"""Redraw of supplementary Fig. S3 (fig2_mechanistic_audit): answer-position attention, probe transfer, patching.

The original PDF (2026-11-12 handoff) has no surviving generator; this script redraws it from the stored
probe / attention / patching summaries of the original v2 runs (results/v2/e2, results/v2/e3, results/v2/summary_v2.json).

    python make_fig_mechanistic.py            # extract -> outputs/bundle/mechanistic_audit.json, plot -> outputs/fig_redraw/
    python make_fig_mechanistic.py --paper    # also overwrite paper/figures/fig2_mechanistic_audit.{pdf,png,data.json}

Definitions (all checked against the numbers printed in the original figure and its caption):
    (a) top: mean answer-token attention over layers l >= l* (att_mean rows 20..35 = layers 21..36, l* = 21) on
        Synthetic Relative Position, spans image1 / question / context ("inserted"); legend accuracy = hf_acc.
    (a) bottom: Synthetic Depth Direct-probe transfer at the last layer; band = range over Filler+turn, Filler-512 and CoT
        at l* in 3B, labelled with 2-decimal truncation as in the original (0.518-0.524 -> "0.51-0.52").
    (b) Direct-probe transfer per layer (x = l / L), Synthetic Relative Position.
    (c) residual patching mean P(gold) per patched block (x = b / (L - 1)); onset = first block whose mean P(gold)
        exceeds the unpatched first-block value by more than 0.05.
"""
from __future__ import annotations
import vtax_paths as _vtp  # noqa: E402

import json
import math
import shutil
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
V2 = Path(_vtp.LEGACY_RESULTS)
B = Path(_vtp.OUTPUTS) / "bundle"
OUT = Path(_vtp.OUTPUTS) / "fig_redraw"
PAPER_FIG = ROOT / f"{_vtp.PAPER}" / "figures"
NAME = "fig2_mechanistic_audit"
LSTAR_3B = 21
ONSET_RISE = 0.05


def extract() -> dict:
    e2 = {m: json.load(open(V2 / "e2" / f"{m}.json")) for m in ("qwen3b", "qwen7b")}
    sd = {m: json.load(open(V2 / "e2" / f"{m}_syndepth.json"))["datasets"]["syn_depth"] for m in ("qwen3b", "qwen7b")}
    s3 = json.load(open(V2 / "summary_v2.json"))["e3"]
    rel3 = e2["qwen3b"]["datasets"]["syn_relpos"]
    spans = e2["qwen3b"]["spans"]
    att = {}
    for c in ("direct", "filler", "cot"):
        a = np.array(rel3["att_mean"][c])[LSTAR_3B - 1:].mean(0)
        att[c] = {"image": float(a[spans.index("image1")]), "question": float(a[spans.index("question")]),
                  "inserted": float(a[spans.index("context")]), "acc": rel3["hf_acc"][c]}
    last = {m: {c: sd[m]["probes"][c]["transfer"][-1] for c in ("filler+turn", "filler512", "cot")} for m in sd}
    mid = [sd["qwen3b"]["probes"][c]["transfer"][LSTAR_3B] for c in ("filler+turn", "filler512", "cot")]
    probes = {m: {c: e2[m]["datasets"]["syn_relpos"]["probes"][c]["transfer"] for c in ("direct", "cot", "filler512", "filler512+reimage")}
              for m in e2}
    patch = {m: {s: {"n": s3[m][s]["n"], "curve": s3[m][s]["mean_curve"]} for s in ("flip_cot", "flip_filler512", "mirror_cot")}
             for m in s3}
    for m, d in patch.items():
        y = np.array(d["flip_cot"]["curve"])
        b = int(np.argmax(y - y[0] > ONSET_RISE))
        d["onset_frac"] = b / (len(y) - 1)
    return {"source": "results/v2/e2/{qwen3b,qwen7b}{,_syndepth}.json, results/v2/summary_v2.json (e3)",
            "lstar_3b": LSTAR_3B, "n_layers": {"qwen3b": 36, "qwen7b": 28}, "attention_qwen3b_syn_relpos": att,
            "syn_depth_last_layer_transfer": last, "syn_depth_mid_layer_3b": mid, "syn_relpos_transfer": probes, "patching": patch}


def trunc2(x: float) -> str:
    return f"{math.floor(x * 100) / 100:.2f}"


def plot(D: dict, out: Path) -> dict:
    plt.rcParams.update({"font.size": 8, "axes.spines.top": False, "axes.spines.right": False, "pdf.fonttype": 42,
                         "font.family": "DejaVu Sans"})
    fig = plt.figure(figsize=(10.2, 3.9))
    gs = fig.add_gridspec(2, 3, width_ratios=[1, 1.05, 1.05], wspace=0.32, hspace=0.85, left=0.065, right=0.99, top=0.83, bottom=0.135)
    blue_d, blue_l, peach = "#3c6493", "#a9c4e4", "#e9b893"
    green, salmon, red, grey = "#8fc29b", "#e07d6c", "#c43c3c", "#666a7a"

    # (a) top: attention mass
    A = D["attention_qwen3b_syn_relpos"]
    ax = fig.add_subplot(gs[0, 0])
    groups = ["image", "question", "inserted"]
    w = 0.27
    for j, (c, lab, col) in enumerate((("direct", "Direct", blue_d), ("filler", "Filler", blue_l), ("cot", "CoT", peach))):
        ys = [A[c][g] for g in groups]
        x = np.arange(3) + (j - 1) * w
        ax.bar(x, ys, w, color=col, label=f"{lab} ({100 * A[c]['acc']:.1f}%)".replace(".0%", "%"))
        for xi, g, y in zip(x, groups, ys):
            if g != "image" and y > 0.001:
                ax.text(xi, y + 0.01, f"{y:.3f}".lstrip("0"), ha="center", va="bottom", fontsize=6.5)
    ax.text(0, max(A[c]["image"] for c in A) + 0.012, f"≤{A['direct']['image']:.3f}".replace("0.", "."), ha="center", fontsize=6.5)
    ax.set_xticks(range(3)); ax.set_xticklabels(["Image", "Question", "Inserted"])
    ax.set_ylabel("Attn. mass"); ax.set_ylim(0, 0.33); ax.set_yticks([0, 0.15, 0.30])
    ax.legend(ncol=3, fontsize=6.5, frameon=False, loc="lower center", bbox_to_anchor=(0.5, 1.0), handlelength=1.0, columnspacing=0.8)
    ax.set_title("(a) Attention & depth-probe transfer", fontsize=9, fontweight="bold", pad=22)

    # (a) bottom: Synthetic Depth transfer
    T, mid = D["syn_depth_last_layer_transfer"], D["syn_depth_mid_layer_3b"]
    ax = fig.add_subplot(gs[1, 0])
    for j, (c, lab, col) in enumerate((("filler+turn", "Filler+turn", green), ("filler512", "Filler-512", blue_l), ("cot", "Short CoT", salmon))):
        ys = [T["qwen3b"][c], T["qwen7b"][c]]
        x = np.arange(2) + (j - 1) * w
        ax.bar(x, ys, w, color=col, label=lab)
        for xi, y in zip(x, ys):
            ax.text(xi, y + 0.03, f"{y:.2f}", ha="center", va="bottom", fontsize=6.5)
    ax.axhspan(min(mid), max(mid), color=grey, alpha=0.25, lw=0)
    ax.axhline(float(np.mean(mid)), color=grey, ls="--", lw=0.8)
    ax.text(0.5, 0.33, f"3B mid-depth ℓ*: {trunc2(min(mid))}–{trunc2(max(mid))}", ha="center", fontsize=6.5)
    ax.set_xticks(range(2)); ax.set_xticklabels(["Last layer L (3B)", "Last layer L (7B)"])
    ax.set_ylabel("Syn. Depth\ntransfer"); ax.set_ylim(0.3, 1.12); ax.set_yticks([0.5, 0.75, 1.0])
    ax.legend(ncol=3, fontsize=6.5, frameon=False, loc="lower center", bbox_to_anchor=(0.5, 1.0), handlelength=1.0, columnspacing=0.8)

    # (b) layerwise Direct-probe transfer
    P = D["syn_relpos_transfer"]
    ax = fig.add_subplot(gs[:, 1])
    series = [("qwen3b", "direct", "Direct (3B)", blue_d, "-"), ("qwen3b", "filler512", "Filler-512 (3B)", "#2f7d4f", "-"),
              ("qwen3b", "filler512+reimage", "F512+reim (3B)", "#e0803a", "--"), ("qwen3b", "cot", "CoT (3B)", red, "-"),
              ("qwen7b", "cot", "CoT (7B)", red, ":"), ("qwen7b", "filler512", "Filler-512 (7B)", "#2f7d4f", ":")]
    for m, c, lab, col, ls in series:
        y = np.array(P[m][c]); x = np.arange(len(y)) / (len(y) - 1)
        ax.plot(x, y, color=col, ls=ls, lw=1.6, label=lab)
    ls_frac = D["lstar_3b"] / D["n_layers"]["qwen3b"]
    ax.axvline(ls_frac, color="k", ls=":", lw=0.9)
    ax.text(ls_frac - 0.12, 0.88, "ℓ* (3B)", fontsize=7, color=grey, fontweight="bold")
    ax.set_xlim(0, 1); ax.set_ylim(0.38, 1.04)
    ax.set_xlabel("Normalized layer depth (ℓ / L)"); ax.set_ylabel("Direct-probe transfer acc.")
    ax.grid(alpha=0.3, lw=0.5)
    ax.legend(ncol=3, fontsize=6.5, frameon=False, loc="lower center", bbox_to_anchor=(0.5, 1.0), handlelength=1.8, columnspacing=0.8)
    ax.set_title("(b) Mid-depth drift for all inserted text", fontsize=9, fontweight="bold", pad=30)

    # (c) residual patching
    Q = D["patching"]
    ax = fig.add_subplot(gs[:, 2])
    for m, s, lab, col, ls in (("qwen3b", "flip_cot", "Direct→CoT (3B", red, "-"), ("qwen3b", "flip_filler512", "Direct→F512 (3B", "#4a7bb7", "-"),
                               ("qwen3b", "mirror_cot", "Mirror→CoT (3B", grey, "--"), ("qwen7b", "flip_cot", "Direct→CoT (7B", "#8e2a2a", ":")):
        y = np.array(Q[m][s]["curve"]); x = np.arange(len(y)) / (len(y) - 1)
        ax.plot(x, y, color=col, ls=ls, lw=1.8, label=f"{lab}, n={Q[m][s]['n']})")
    ax.axvline(ls_frac, color="k", ls=":", lw=0.9)
    ax.text(ls_frac - 0.12, 0.83, "ℓ* (3B)", fontsize=7, color=grey, fontweight="bold")
    ax.axhline(0.5, color="0.6", ls="--", lw=0.7)
    o3, o7 = Q["qwen3b"]["onset_frac"], Q["qwen7b"]["onset_frac"]
    y7 = np.array(Q["qwen7b"]["flip_cot"]["curve"])
    ax.annotate(f"Onset: {100 * o3:.0f}% (3B)\n{100 * o7:.0f}% (7B)", xy=(o7, float(np.interp(o7, np.arange(len(y7)) / (len(y7) - 1), y7))),
                xytext=(0.28, 0.2), fontsize=7, color=red, fontweight="bold", ha="center",
                arrowprops={"arrowstyle": "->", "color": red, "lw": 0.8})
    ax.set_xlim(0, 1); ax.set_ylim(-0.02, 1.06)
    ax.set_xlabel("Patched block b / (L − 1)"); ax.set_ylabel("Patched P(gold)")
    ax.grid(alpha=0.3, lw=0.5)
    ax.legend(ncol=2, fontsize=6.5, frameon=False, loc="lower center", bbox_to_anchor=(0.5, 1.0), handlelength=1.8, columnspacing=0.8)
    ax.set_title("(c) Late answer-state decisiveness", fontsize=9, fontweight="bold", pad=30)

    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / f"{NAME}.pdf")
    fig.savefig(out / f"{NAME}.png", dpi=110)
    plt.close(fig)
    # same keys and rounding as the original data.json in paper/figures
    data = {"attention_qwen3b": {c: [round(A[c][g], 3) for g in groups] for c in ("direct", "filler", "cot")},
            "syn_depth_last_layer_transfer": {k: [round(T["qwen3b"][c], 2), round(T["qwen7b"][c], 2)]
                                              for k, c in (("filler_turn", "filler+turn"), ("filler_512", "filler512"), ("cot", "cot"))},
            "syn_depth_mid_layer_range_3b": [float(trunc2(min(mid))), float(trunc2(max(mid)))]}
    (out / f"{NAME}.data.json").write_text(json.dumps(data))
    return data


def main() -> None:
    D = extract()
    (B / "mechanistic_audit.json").write_text(json.dumps(D, indent=1))
    data = plot(json.load(open(B / "mechanistic_audit.json")), OUT)
    print(json.dumps(data))
    print("onset", {m: round(D["patching"][m]["onset_frac"], 3) for m in D["patching"]})
    if "--paper" in sys.argv:
        for suf in (".pdf", ".png", ".data.json"):
            shutil.copy(OUT / f"{NAME}{suf}", PAPER_FIG / f"{NAME}{suf}")


if __name__ == "__main__":
    main()
