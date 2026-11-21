"""Cue AMEs: scale-free cue weights on the cue-factorial set (CPU only, stored e10 outputs).

Probit coefficients are identified only relative to the latent residual variance, so CoT - Direct changes in the
probit weights mix a cue-specific change with any added image-independent answer variance. This script reports,
per model, arm (Direct, CoT) and cue (occlusion, vertical, size), two quantities on the probability scale:

ame   Average marginal effect from the probit GLM of srww.cue_glm.glm_rows / probit_fit (says-subject-front ~ phrasing
      + occlusion + vertical + size): mean over items of Phi(x_k=+1) - Phi(x_k=-1), in percentage points.
diff  Nonparametric marginal contrast P(says front | cue = +1) - P(says front | cue = -1); the cue-factorial set is
      balanced, so this is the cue's main effect on the answer. The per-cue AUC of the binary answer equals
      (1 + diff/100) / 2.

CIs: twin-pair (image_id) cluster bootstrap, NB draws, Direct and CoT resampled jointly.
-> outputs/bundle/cue_ame.json, outputs/bundle/cue_ame.txt
Usage: python -m experiments.cue_analysis.cue_ame [--nb 1000]
"""
from __future__ import annotations
from srww import paths as _vtp  # noqa: E402

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.stats import norm

from srww.records import rj
from srww.cue_glm import glm_rows, probit_fit

O = Path(f"{_vtp.OUTPUTS}")
B = Path("outputs/bundle")
M7 = ["qwen3b", "qwen7b", "internvl3_8b", "phi35v", "molmo7b", "qwen3vl_8b", "internvl35_8b"]
CUES = ["occlusion", "vertical", "size"]
COL = {"occlusion": 2, "vertical": 3, "size": 4}


def ame(X: np.ndarray, b: np.ndarray, j: int) -> float:
    """Discrete average marginal effect of the +-1 coded column j, in points."""
    hi, lo = X.copy(), X.copy()
    hi[:, j], lo[:, j] = 1.0, -1.0
    return 100.0 * float(np.mean(norm.cdf(hi @ b) - norm.cdf(lo @ b)))


def diff(X: np.ndarray, y: np.ndarray, j: int) -> float:
    return 100.0 * float(y[X[:, j] > 0].mean() - y[X[:, j] < 0].mean())


def stats(d: dict, idx: np.ndarray) -> dict:
    X, y = d["X"][idx], d["y"][idx]
    b = probit_fit(X, y)
    return {c: {"ame": ame(X, b, COL[c]), "diff": diff(X, y, COL[c]), "d": 2 * float(b[COL[c]])} for c in CUES}


def q95(v) -> list[float]:
    return [round(float(np.quantile(v, .025)), 1), round(float(np.quantile(v, .975)), 1)]


def one_model(args: tuple[str, int]) -> tuple[str, dict, list[str]]:
    m, nb = args
    lines: list[str] = []
    if True:
        R = rj(O / "e10" / f"{m}_cues.jsonl")
        by = {c: {r["item_id"]: r for r in R if r["condition"] == c} for c in ("direct", "cot")}
        keys = sorted(set(by["direct"]) & set(by["cot"]))
        data = {c: glm_rows([by[c][k] for k in keys], use_size=True) for c in by}
        pair = np.array(data["direct"]["pair"])
        pairs = sorted(set(pair))
        pidx = {p: np.where(pair == p)[0] for p in pairs}
        allidx = np.arange(len(keys))
        est = {c: stats(data[c], allidx) for c in data}
        rng = np.random.default_rng(18)
        bs = {c: {cue: {k: [] for k in ("ame", "diff")} for cue in CUES} for c in data}
        for _ in range(nb):
            s = np.concatenate([pidx[pairs[j]] for j in rng.integers(0, len(pairs), len(pairs))])
            for c in data:
                st = stats(data[c], s)
                for cue in CUES:
                    for k in ("ame", "diff"):
                        bs[c][cue][k].append(st[cue][k])
        r = {"n": len(keys)}
        for cue in CUES:
            r[cue] = {}
            for k in ("ame", "diff"):
                dv, cv = np.array(bs["direct"][cue][k]), np.array(bs["cot"][cue][k])
                r[cue][k] = {
                    "direct": {"est": round(est["direct"][cue][k], 1), "ci": q95(dv)},
                    "cot": {"est": round(est["cot"][cue][k], 1), "ci": q95(cv)},
                    "cot-direct": {"est": round(est["cot"][cue][k] - est["direct"][cue][k], 1), "ci": q95(cv - dv)},
                    "retained": round(est["cot"][cue][k] / est["direct"][cue][k], 2) if abs(est["direct"][cue][k]) > 1e-9 else None,
                }
            r[cue]["probit_d"] = {"direct": round(est["direct"][cue]["d"], 2), "cot": round(est["cot"][cue]["d"], 2)}
            r[cue]["auc_from_diff"] = {c: round(0.5 + est[c][cue]["diff"] / 200.0, 3) for c in ("direct", "cot")}
        f = lambda x: f"{x['est']:+.1f} [{x['ci'][0]:+.1f}, {x['ci'][1]:+.1f}]"
        for cue in CUES:
            A, D = r[cue]["ame"], r[cue]["diff"]
            lines.append(f"{m:14s} {cue:9s} AME Dir {f(A['direct'])} CoT {f(A['cot'])} CoT-Dir {f(A['cot-direct'])} kept {A['retained']} | "
                         f"diff Dir {f(D['direct'])} CoT {f(D['cot'])} CoT-Dir {f(D['cot-direct'])} kept {D['retained']} | "
                         f"AUC Dir {r[cue]['auc_from_diff']['direct']} CoT {r[cue]['auc_from_diff']['cot']} | probit d' {r[cue]['probit_d']}")
        print("\n".join(lines[-3:]), flush=True)
    return m, r, lines


def main() -> None:
    ap = argparse.ArgumentParser(description="Cue AMEs (occlusion, vertical, size) of Direct and CoT answers on the cue-factorial set.")
    ap.add_argument("--nb", type=int, default=1000)
    a = ap.parse_args()
    from multiprocessing import Pool
    with Pool(len(M7)) as pool:
        out = pool.map(one_model, [(m, a.nb) for m in M7])
    res = {m: r for m, r, _ in out}
    lines = [ln for _, _, ls in out for ln in ls]
    json.dump({"nb": a.nb, "models": res}, open(B / "cue_ame.json", "w"), indent=1)
    (B / "cue_ame.txt").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
