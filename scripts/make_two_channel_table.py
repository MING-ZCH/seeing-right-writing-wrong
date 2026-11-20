"""Supplementary table of the 21 (+1) two-channel points (Exp-32)."""
import vtax_paths as _vtp  # noqa: E402
import json
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path

B = Path("outputs/bundle")
T = Path(f"{_vtp.TABLES}")
NAMES = {"qwen3b": "Qwen2.5-VL-3B", "qwen7b": "Qwen2.5-VL-7B", "internvl3_8b": "InternVL3-8B",
         "phi35v": "Phi-3.5-V", "molmo7b": "Molmo-7B-D", "qwen3vl_8b": "Qwen3-VL-8B",
         "internvl35_8b": "InternVL3.5-8B", "qwen3vl_8b_think": "Qwen3-VL-8B-Think."}
DS = {"cues": "Cue set", "twins": "Twins", "syn431": "Syn.\\ Depth 431"}


def r1(v: float) -> str:
    return str(Decimal(repr(v)).quantize(Decimal("0.1"), rounding=ROUND_HALF_UP))


def main() -> None:
    d = json.load(open(B / "two_channel.json"))
    cells = d["cells"]
    rows = []
    for m in NAMES:
        for ds in DS:
            c = cells.get(f"{m}|{ds}")
            if c is None:
                continue
            rows.append(
                f"{NAMES[m]} & {DS[ds]} & {c['n']} & {c['last_n_parsed']} ({c['last_coverage']:.1f}) & "
                f"{r1(100 * (c['DpCp'] + c['DpCm']) / c['n'])} & {r1(100 * (c['DpCp'] + c['DmCp']) / c['n'])} & {c['last_direct_on_parsed']:.1f} & {c['last_written_acc']:.1f} & {c['last_agree']:.1f} & "
                f"{c['last_x']:+.1f} & {'+' if c['DmCp'] >= c['DpCm'] else '-'}{r1(abs(100 * (c['DmCp'] - c['DpCm']) / c['n']))} & {c['DpCm']} / {c['DmCp']} \\\\")
    f = d["fits"]["primary|last_main"]
    g = d["fits"]["primary|last_l4l"]
    nul = d["shared_direct_null"]
    lines = [r"\begin{tabular}{llcccccccccc}", r"\toprule",
             r"Model & Set & $n$ & parsed ($\%$) & \texttt{Dir.} & \texttt{CoT} & \texttt{Dir.}\,(parsed) & written & agree & $x$ & $y$ & \texttt{D+C$-$} / \texttt{D$-$C+} \\",
             r"\midrule", *rows, r"\bottomrule",
             r"\multicolumn{12}{p{0.97\linewidth}}{\scriptsize Accuracies in \%. Written: accuracy of the last written relation (claim\_parse v3) over items with a parsed relation; agree: share of those items whose \texttt{CoT} answer agrees with it. "
             r"$x$ = written $-$ \texttt{Dir.}\,(parsed), both on the items with a parsed relation; $y$ = \texttt{CoT} $-$ \texttt{Direct} on all items (points). "
             f"Fit over the $21$ instruct-model points: Pearson $r={f['pearson_r']:.2f}$, Spearman $\\rho={f['spearman_rho']:.2f}$, OLS slope ${f['ols_slope']:.2f}$ "
             f"[{f['slope_ci_point_boot'][0]:.2f}, {f['slope_ci_point_boot'][1]:.2f}] (point bootstrap; model-cluster bootstrap [{f['slope_ci_model_cluster_boot'][0]:.2f}, {f['slope_ci_model_cluster_boot'][1]:.2f}]), intercept ${f['ols_intercept']:.1f}$; "
             f"with $y$ on the parsed items only, $r={g['pearson_r']:.2f}$, slope ${g['ols_slope']:.2f}$ [{g['slope_ci_point_boot'][0]:.2f}, {g['slope_ci_point_boot'][1]:.2f}]. "
             f"Shared-\\texttt{{Direct}} null (written accuracy permuted across models within each set, $10{{,}}000$ draws): mean $r={nul['r_main_null_mean']:.2f}$, $95\\%$ range [{nul['r_main_null_95'][0]:.2f}, {nul['r_main_null_95'][1]:.2f}], $p<10^{{-4}}$. "
             r"The \texttt{Qwen3-VL-8B-Thinking} twin row is not in the fit.} \\",
             r"\end{tabular}"]
    (T / "supp_two_channel.tex").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
