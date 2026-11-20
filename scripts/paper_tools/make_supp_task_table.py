"""Build the supplementary task-breakdown table (moved Table 1 panel B) from bundle/numbers.json."""
import vtax_paths as _vtp  # noqa: E402
import json
from pathlib import Path

PAPER = Path(_vtp.PAPER)
BUNDLE = Path(_vtp.OUTPUTS) / "bundle" / "numbers.json"
OUT = PAPER / "tables" / "tab_s_tasks.tex"
MODELS = [("qwen3b", "Qwen2.5-VL-3B"), ("qwen7b", "Qwen2.5-VL-7B"), ("internvl3_8b", "InternVL3-8B"),
          ("phi35v", "Phi-3.5-V"), ("molmo7b", "Molmo-7B-D"), ("qwen3vl_8b", "Qwen3-VL-8B"),
          ("qwen3vl_8b_think", "Qwen3-VL-8B-Think."), ("internvl35_8b", "InternVL3.5-8B")]
TASKS = [("cvb_relation", "CVB 2D Relation ($650$)"), ("cvb_depth", "CVB 3D Depth ($600$)"),
         ("cvb_distance", "CVB 3D Distance ($600$)"), ("syn_relpos", "Syn.\\ RelPos ($500$)"),
         ("syn_depth", "Syn.\\ Depth ($500$)")]


def fmt(x: float, sign: bool = False) -> str:
    s = f"{x:+.1f}" if sign else f"{x:.1f}"
    return s.replace("+0.0", "0.0").replace("-0.0", "0.0")


def parse(s: str):
    # "73.7 [70.2, 77.1]" -> (73.7, 70.2, 77.1)
    v, ci = s.split(" [")
    lo, hi = ci.rstrip("]").split(", ")
    return float(v), float(lo), float(hi)


def main() -> None:
    d = json.load(open(BUNDLE))
    lines = ["\\begin{tabular}{@{}llccc@{}}", "\\toprule",
             "Model & Task & \\texttt{Direct} / \\texttt{CoT} (\\texttt{Fil+t}) & $\\Delta$(\\texttt{CoT}$-$\\texttt{Direct}) & $\\Delta$(\\texttt{CoT}$-$\\texttt{Fil+t}) \\\\",
             "\\midrule"]
    for mi, (m, name) in enumerate(MODELS):
        beh = d[m]["behavioral"]
        for ti, (t, tname) in enumerate(TASKS):
            r = beh[f"task:{t}"]
            g = lambda k: (r[k]["mean"], r[k]["lo"], r[k]["hi"]) if isinstance(r[k], dict) else parse(r[k])
            dr, co, ft = g("direct")[0], g("cot")[0], g("filler+turn")[0]
            a = g("cot_minus_direct"); b = g("cot_minus_fillerturn")
            lab = name if ti == 0 else ""
            lines.append(f"{lab} & {tname} & ${fmt(dr)}$ / ${fmt(co)}$ (${fmt(ft)}$) & "
                         f"${fmt(a[0], True)}$\\,{{\\scriptsize$[{fmt(a[1], True)}, {fmt(a[2], True)}]$}} & "
                         f"${fmt(b[0], True)}$\\,{{\\scriptsize$[{fmt(b[1], True)}, {fmt(b[2], True)}]$}} \\\\")
        if mi < len(MODELS) - 1:
            lines.append("\\midrule")
    lines += ["\\bottomrule", "\\end{tabular}"]
    open(OUT, "w").write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
