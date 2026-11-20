"""Worst-case bounds for the rescue-sentence audit and its comparators (round 11b).

Reads the two-judge files in outputs/e20 and writes bundle/rescue_bounds.json.
Bounds count every unjudgeable sentence (judges disagree, unclear, no claim, or
unparsed) as consistent or as contradicting, as in the D+C- commit audit.
"""
import vtax_paths as _vtp  # noqa: E402
import json
from pathlib import Path

O = Path(f"{_vtp.OUTPUTS}/e20")
B = Path(_vtp.OUTPUTS) / "bundle"
J2 = ("claude-opus-5-5", "gpt-6-astra")
M7 = ["qwen3b", "qwen7b", "internvl3_8b", "phi35v", "molmo7b", "qwen3vl_8b", "internvl35_8b"]
SETS = {"rescue_DmCp": "_rescue", "DmCm_rel_rescuepos": "_rescue_dcm",
        "DpCp_rel_rescuepos": "_rescue_dcprel", "DpCp_rel_commitpos": "_relcomp"}


def rj(p: Path) -> list[dict]:
    return [json.loads(line) for line in open(p) if line.strip()]


def main() -> None:
    res = {}
    for name, suf in SETS.items():
        J = {j: {(r["model"], r["item_id"]): r for r in rj(O / f"{j}{suf}.jsonl")} for j in J2}
        ks = [k for k in set(J[J2[0]]) & set(J[J2[1]]) if k[0] in M7]
        lab = lambda j, k: J[j][k]["label"]
        cor = sum(lab(J2[0], k) == lab(J2[1], k) == "CORRECT" for k in ks)
        inc = sum(lab(J2[0], k) == lab(J2[1], k) == "INCORRECT" for k in ks)
        n, nj = len(ks), cor + inc
        res[name] = {"n": n, "judgeable": nj, "consistent_k": cor, "contradict_k": inc,
                     "consistent_pct": round(100 * cor / nj, 1),
                     "consistent_worst_case": [round(100 * cor / n, 1), round(100 * (cor + n - nj) / n, 1)],
                     "contradict_worst_case": [round(100 * inc / n, 1), round(100 * (inc + n - nj) / n, 1)]}
        print(name, res[name])
    json.dump(res, open(B / "rescue_bounds.json", "w"), indent=1)


if __name__ == "__main__":
    main()
