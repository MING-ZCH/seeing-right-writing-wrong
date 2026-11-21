"""Write pre-registered E6 predictions from held-out Direct outputs and the frozen main-pool gate.

For each model and held-out task: p-bar (exact Direct P(gold)), predicted 4-vote accuracy (item-level
plurality formula), constant-p-bar vote prediction, and the fraction routed by the frozen tau.
The file name carries a UTC timestamp; it stores sha256 of the Direct outputs, the tau file and this code.

Run after `heldout_benchmarks --stage direct` and before `--stage rest`. The frozen tau (main-pool gate, 5-fold CV)
is data/heldout/gate_frozen_tau.json. Needs the full Direct records of a fresh run (fields ``letters`` and
``pred_exact``, which are trimmed from the released outputs).

Usage: python -m experiments.behavior.heldout_predict [--models qwen7b] [--outputs outputs] [--tau data/heldout/gate_frozen_tau.json]
"""
from __future__ import annotations
from srww import paths as _vtp  # noqa: E402

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

from srww.common import margin_top2, plurality_win_prob, sha256_file, utc_stamp


def probs(lp: dict, letters: list[str]) -> np.ndarray:
    v = np.array([lp[l] for l in letters], dtype=float)
    v = np.exp(v - v.max())
    return v / v.sum()


def main() -> None:
    """CLI entry point."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--outputs", default=_vtp.OUTPUTS)
    ap.add_argument("--tau", default=f"{_vtp.ROOT}/data/heldout/gate_frozen_tau.json")
    ap.add_argument("--models", default="qwen3b,qwen7b,internvl3_8b,phi35v,molmo7b")
    args = ap.parse_args()
    O = Path(args.outputs)
    taus = json.load(open(args.tau))["models"]
    for m in args.models.split(","):
        p = O / "e6x" / f"{m}_direct.jsonl"
        if not p.exists():
            print(f"skip {m}: no Direct outputs")
            continue
        recs = [json.loads(l) for l in open(p)]
        by = defaultdict(list)
        for r in recs:
            by[r["task"]].append(r)
            by["__pooled__"].append(r)
        tau = {k: v["tau"] for k, v in taus.get(m, {}).items()}
        pred = {}
        for task, rs in by.items():
            pv = [probs(r["lp_sum"], r["letters"]) for r in rs]
            pg = np.array([v[r["letters"].index(r["gold"])] for v, r in zip(pv, rs)])
            vote = np.mean([plurality_win_prob(v, r["letters"].index(r["gold"]), 4) for v, r in zip(pv, rs)])
            pbar = float(pg.mean())
            const = float(np.mean([plurality_win_prob(np.array([pbar, 1 - pbar]), 0, 4) for _ in [0]]))
            marg = np.array([margin_top2(v) for v in pv])
            pred[task] = {"n": len(rs), "direct_acc_observed": float(np.mean([r["pred_exact"] == r["gold"] for r in rs])),
                          "pbar": pbar, "pred_vote4_item": float(vote), "pred_vote4_const_pbar": const,
                          "routed_frac": {k: float((marg < t).mean()) for k, t in tau.items()}}
        stamp = utc_stamp()
        out = O / "e6x" / f"predictions_{m}_{stamp}.json"
        json.dump({"created_utc": stamp, "model": m, "tau": tau,
                   "sha256": {"direct": sha256_file(p), "tau": sha256_file(args.tau), "code": sha256_file(__file__)},
                   "rule": "gate: use Direct if P(c1)-P(c2) >= tau else alternative (cot | vote4)",
                   "predictions": pred}, open(out, "w"), indent=1)
        print(f"wrote {out}")
        for t, v in pred.items():
            print(f"  {t}: n={v['n']} direct={100 * v['direct_acc_observed']:.1f} pbar={100 * v['pbar']:.1f} "
                  f"vote4_pred={100 * v['pred_vote4_item']:.1f} routed={v['routed_frac']}")


if __name__ == "__main__":
    main()
