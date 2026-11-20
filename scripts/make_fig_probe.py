"""Within-claim truth probe figure and table (Exp-17j setting: cue set, claim_parse v3 strata, layer 20).

Reads $W/outputs/e17/<model>_ctrl.json and outputs/bundle/probe_controls.json.
fig_probe_cot.{pdf,png,data.json} (full width):
  (a-c) per model, balanced accuracy per layer at the relation token r (95% CI band), the token that writes the relation
        word (r-1) and the token that writes the subject (s-1), original image; gray-image refit at r; filler-matched
        token; layer-0 value at r as a reference line.
  (d)   recall on D+C- wrong claims at layer 20 (original r, answer token, gray refit r, filler, layer 0, CoT text).
supp_probe_cot.tex: layer-20 balanced accuracy / D+C- wrong-claim recall at s-1, s, r-1, r and the answer token.
"""
from __future__ import annotations
import vtax_paths as _vtp  # noqa: E402

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

E17 = Path(f"{_vtp.OUTPUTS}/e17")
B = Path("outputs/bundle")
T = Path(f"{_vtp.TABLES}")
FIG = Path(f"{_vtp.FIGURES}")
MODELS = {"qwen3vl_8b": "Qwen3-VL-8B", "molmo7b": "Molmo-7B-D", "qwen7b": "Qwen2.5-VL-7B"}
LAYER = 20
COL = {"r": "#B85244", "rprev": "#E2A08F", "pre": "#8C6BB1", "gray": "#3B6998", "filler": "#7A828E", "l0": "#2F3640", "a": "#5E9E6E", "text": "#E2B28F"}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--fig-dir", type=Path, default=FIG)
    args = ap.parse_args()
    fig_dir = args.fig_dir
    fig_dir.mkdir(parents=True, exist_ok=True)

    ctrl = {m: json.load(open(E17 / f"{m}_ctrl.json")) for m in MODELS}
    pc = json.load(open(B / "probe_controls.json"))
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 7.5, "axes.labelsize": 7.5, "xtick.labelsize": 7.0,
                         "ytick.labelsize": 7.0, "legend.fontsize": 7.0, "axes.spines.top": False, "axes.spines.right": False,
                         "pdf.fonttype": 42})
    fig = plt.figure(figsize=(6.875, 2.8))
    gs = fig.add_gridspec(1, 4, width_ratios=[1, 1, 1, 1.25], wspace=0.38, left=0.065, right=0.99, top=0.70, bottom=0.155)
    data = {"setting": "cue-factorial set; within-claim truth probe (claim_parse v3 strata), GroupKFold by twin pair; "
                       "balanced accuracy = mean recall on wrong- and right-claim items", "layer_for_panel_d": LAYER, "models": {}}
    axes = []
    for j, (m, name) in enumerate(MODELS.items()):
        ax = fig.add_subplot(gs[0, j], sharey=axes[0] if axes else None)
        axes.append(ax)
        lay = ctrl[m]["layers"]
        L = np.array([r["layer"] for r in lay])
        get = lambda f: np.array([f(r) for r in lay], float)
        r_b = get(lambda r: r["orig"]["r"]["bal"])
        r_lo = get(lambda r: r["orig"]["r"]["bal_ci"][0])
        r_hi = get(lambda r: r["orig"]["r"]["bal_ci"][1])
        ax.fill_between(L, r_lo, r_hi, color=COL["r"], alpha=0.18, linewidth=0)
        ax.plot(L, r_b, color=COL["r"], linewidth=1.6, marker="o", markersize=2.6, label="Relation token $r$ (95% CI)")
        series = {"r": r_b.tolist(), "r_ci": [r_lo.tolist(), r_hi.tolist()]}
        for key, path, lab, ls in (("rprev", ("orig", "rprev"), "Token writing the relation ($r{-}1$)", "-"),
                                   ("pre", ("orig", "pre"), "Token writing the subject ($s{-}1$)", "-"),
                                   ("gray", ("gray_refit", "r"), "Gray image, refit at $r$", "--")):
            v = get(lambda r, p=path: r[p[0]][p[1]]["bal"])
            ax.plot(L, v, color=COL[key], linewidth=1.0, linestyle=ls, marker="o" if ls == "-" else None, markersize=1.8, label=lab)
            series[key] = v.tolist()
        fl = get(lambda r: (r.get("filler") or {}).get("bal", np.nan))
        ax.plot(L, fl, color=COL["filler"], linewidth=0.9, linestyle=(0, (1, 1.2)), label="Filler-matched token")
        series["filler"] = fl.tolist()
        l0 = lay[0]["orig"]["r"]["bal"]
        ax.axhline(l0, color=COL["l0"], linewidth=0.7, linestyle="-.", label="Layer 0 at $r$ (lexical)")
        ax.axhline(0.5, color="#B9BEC6", linewidth=0.5, linestyle=":")
        ax.axvline(LAYER, color="#CBD0D7", linewidth=0.6, zorder=0)
        ax.set_title(f"({'abc'[j]}) {name}", fontsize=7.5, fontweight="bold", loc="left")
        ax.set_xlabel("Layer", labelpad=1)
        ax.set_xticks([0, 10, 20, 30] if m == "qwen3vl_8b" else [0, 10, 20, 28])
        ax.set_ylim(0.45, 1.0)
        ax.grid(True, color="#E6E9EE", linewidth=0.5)
        if j == 0:
            ax.set_ylabel("Within-claim balanced accuracy")
        else:
            plt.setp(ax.get_yticklabels(), visible=False)
        data["models"][name] = {"layers": L.tolist(), **series, "layer0_r": l0}
    # (d) D+C- wrong-claim recall at layer 20
    ax = fig.add_subplot(gs[0, 3])
    conds = [("r", "$r$", lambda p: p["orig"]["r"]), ("a", "Answer", lambda p: p["orig"]["a"]),
             ("gray", "Gray $r$", lambda p: p["gray_refit"]["r"]), ("filler", "Filler", lambda p: p["filler"]),
             ("l0", "Layer 0", lambda p: p["layer0_r"]), ("text", "Text", lambda p: p["baselines"]["text"])]
    w = 0.13
    xs = np.arange(len(MODELS))
    data["panel_d_recall_DCm_wrong"] = {}
    for k, (key, lab, f) in enumerate(conds):
        vals, err = [], [[], []]
        for m in MODELS:
            d = f(pc[m])
            v = d["recall_DCm_wrong"]
            ci = d.get("recall_DCm_wrong_ci")
            vals.append(v)
            err[0].append(v - ci[0] if ci else 0)
            err[1].append(ci[1] - v if ci else 0)
            data["panel_d_recall_DCm_wrong"].setdefault(MODELS[m], {})[lab] = {"recall": v, "ci": ci}
        ax.bar(xs + (k - 2.5) * w, vals, w * 0.92, color=COL[key], label=lab, zorder=2)
        ax.errorbar(xs + (k - 2.5) * w, vals, yerr=err, fmt="none", ecolor="#3C434D", elinewidth=0.5, capsize=0.8, zorder=3)
    ax.axhline(0.5, color="#B9BEC6", linewidth=0.5, linestyle=":")
    ax.set_xticks(xs)
    ax.set_xticklabels(["Qwen3-VL\n8B", "Molmo\n7B-D", "Qwen2.5-VL\n7B"], linespacing=0.95)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Recall, D+C$-$ wrong claims", labelpad=1)
    ax.set_title(f"(d) Wrong claims, L{LAYER}", fontsize=7.5, fontweight="bold", loc="left")
    ax.grid(True, axis="y", color="#E6E9EE", linewidth=0.5)
    leg_d = ax.legend(loc="lower center", bbox_to_anchor=(0.48, 1.13), ncol=3, frameon=False, handlelength=1.0,
                      columnspacing=0.6, handletextpad=0.3, labelspacing=0.2,
                      title="(d)", title_fontproperties={"weight": "bold", "size": 7.0})
    leg_d._legend_box.align = "left"
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper left", bbox_to_anchor=(0.025, 0.99), ncol=3, frameon=False, handlelength=1.45,
               columnspacing=0.60, handletextpad=0.35, labelspacing=0.25)
    fig.savefig(fig_dir / "fig_probe_cot.pdf")
    fig.savefig(fig_dir / "fig_probe_cot.png", dpi=300)
    json.dump(data, open(fig_dir / "fig_probe_cot.data.json", "w"), indent=1)
    # table
    pos = [("pre", "Token writing the subject ($s{-}1$)"), ("s", "Subject noun ($s$)"), ("rprev", "Token writing the relation ($r{-}1$)"),
           ("r", "Relation token ($r$)"), ("a", "Answer token")]
    fc = lambda x: (f"{x['bal']:.2f} [{x['bal_ci'][0]:.2f}, {x['bal_ci'][1]:.2f}] / "
                    f"{x['recall_DCm_wrong']:.2f} [{x['recall_DCm_wrong_ci'][0]:.2f}, {x['recall_DCm_wrong_ci'][1]:.2f}]")
    lines = [r"\begin{tabular}{l" + "c" * len(MODELS) + "}", r"\toprule",
             r"Position (original image, layer " + str(LAYER) + r") & " + " & ".join(f"{n} ($n_\\mathrm{{D+C-}}={pc[m]['orig']['r']['n_DCm_wrong']}$)" for m, n in MODELS.items()) + r" \\",
             r"\midrule"]
    for k, lab in pos:
        lines.append(lab + " & " + " & ".join(fc(pc[m]["orig"][k]) for m in MODELS) + r" \\")
    lines += [r"\bottomrule", r"\multicolumn{" + str(len(MODELS) + 1) + r"}{l}{\scriptsize Cells: balanced accuracy [95\% CI] / recall on D+C$-$ wrong claims [95\% CI]; twin-pair bootstrap.} \\",
              r"\end{tabular}"]
    (T / "supp_probe_cot.tex").write_text("\n".join(lines) + "\n")
    print({MODELS[m]: pc[m]["best_layer"] for m in MODELS})


if __name__ == "__main__":
    main()
