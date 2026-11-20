"""Summarize closed/API model runs (E9x original occlusion set, E9h difficulty-calibrated set).

Reports per task and per difficulty level: accuracy with image-cluster bootstrap CIs for every
condition present (direct = letter only with hidden reasoning, cot = written rationale,
direct_nr = letter only with reasoning_effort="none"), condition differences, unparsed counts,
pooled d' on the judgement "the question's subject is in front" with twin-pair bootstrap CIs,
twin flip rates and reasoning-token counts.

By default every model file found in the output directory is summarized.

Usage:
  python analyze_e9x.py                      # original set -> outputs/bundle/e9x_summary.{json,txt}
  python analyze_e9x.py --dir e9h --out outputs/bundle/e9h_summary
"""
from __future__ import annotations
import vtax_paths as _vtp  # noqa: E402

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import norm

from common import cluster_bootstrap_ci

ROOT = Path(f"{_vtp.OUTPUTS}")
EXCLUDE: set[str] = set()
CONDS = ("direct", "direct_nr", "cot")
DIFFS = (("cot", "direct"), ("cot", "direct_nr"), ("direct_nr", "direct"))


def load(path: Path) -> dict:
    """Records keyed by (item_id, condition); error rows and superseded retries are dropped."""
    recs = {}
    for r in map(json.loads, open(path)):
        if "error" in r:
            continue
        recs[(r["item_id"], r["condition"])] = r
    return recs


def ci(v, g) -> list[float]:
    """Percent accuracy with a 95% image-cluster bootstrap CI."""
    if not len(v):
        return []
    m, lo, hi = cluster_bootstrap_ci(np.asarray(v, float), g, 2000)
    return [round(100 * m, 1), round(100 * lo, 1), round(100 * hi, 1)]


def sdt(resp: np.ndarray, truth: np.ndarray) -> tuple[float, float]:
    """d' and criterion c with a 0.5 correction."""
    h = (resp[truth].sum() + .5) / (truth.sum() + 1)
    f = (resp[~truth].sum() + .5) / ((~truth).sum() + 1)
    return float(norm.ppf(h) - norm.ppf(f)), float(-(norm.ppf(h) + norm.ppf(f)) / 2)


def occ_pairs(R: dict, cond: str, level: str | None = None) -> dict[str, list[tuple[int, int]]]:
    """Per twin-pair (says-subject-is-in-front, subject-really-in-front) responses, both phrasings."""
    out: dict[str, list[tuple[int, int]]] = defaultdict(list)
    for (uid, c), r in R.items():
        if c != cond or not r["task"].startswith("occ_") or r["pred"] is None:
            continue
        if level and (r.get("meta") or {}).get("level") != level:
            continue
        says = (r["pred"] == "A") if r["phrasing"] == "front" else (r["pred"] == "B")
        out[r["image_id"]].append((int(says), int(r["meta"]["front_name"] == r["meta"]["x_name"])))
    return out


def dprime_boot(P: dict[str, list[tuple[int, int]]], n_boot: int = 2000) -> dict:
    """Pooled d' / c with a bootstrap CI over twin pairs."""
    keys = sorted(P)
    if not keys:
        return {}
    arr = np.array([x for k in keys for x in P[k]])
    d, c = sdt(arr[:, 0].astype(bool), arr[:, 1].astype(bool))
    rng = np.random.default_rng(0)
    b = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(keys), len(keys))
        a = np.array([x for i in idx for x in P[keys[i]]])
        b.append(sdt(a[:, 0].astype(bool), a[:, 1].astype(bool))[0])
    return {"dprime": round(d, 2), "ci": [round(float(np.quantile(b, .025)), 2), round(float(np.quantile(b, .975)), 2)],
            "c": round(c, 2), "n_pairs": len(keys), "n_resp": int(len(arr))}


def dprime_diff(P1: dict, P2: dict, n_boot: int = 2000) -> list[float]:
    """d'(P1) - d'(P2) with a paired bootstrap CI over the twin pairs present in both."""
    keys = sorted(set(P1) & set(P2))
    if not keys:
        return []
    est = []
    for P in (P1, P2):
        a = np.array([x for k in keys for x in P[k]])
        est.append(sdt(a[:, 0].astype(bool), a[:, 1].astype(bool))[0])
    rng = np.random.default_rng(0)
    b = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(keys), len(keys))
        row = []
        for P in (P1, P2):
            a = np.array([x for i in idx for x in P[keys[i]]])
            row.append(sdt(a[:, 0].astype(bool), a[:, 1].astype(bool))[0])
        b.append(row[0] - row[1])
    return [round(est[0] - est[1], 2), round(float(np.quantile(b, .025)), 2), round(float(np.quantile(b, .975)), 2)]


def twin_flip(R: dict, cond: str, level: str | None = None) -> float | None:
    """Fraction of twin pairs whose answer changes when the two object identities are swapped."""
    flips = []
    for (uid, c), r in R.items():
        if c != cond or not r["task"].startswith("occ_") or r["pred"] is None or "_cf_" in uid:
            continue
        if level and (r.get("meta") or {}).get("level") != level:
            continue
        twin = uid.replace("_front", "_cf_front").replace("_behind", "_cf_behind")
        if (twin, cond) in R and R[(twin, cond)]["pred"] is not None:
            flips.append(r["pred"] != R[(twin, cond)]["pred"])
    return round(100 * float(np.mean(flips)), 1) if flips else None


def summarize(R: dict) -> dict:
    """Full summary for one model."""
    conds = [c for c in CONDS if any(k[1] == c for k in R)]
    res: dict = {"n_records": len(R), "conditions": conds}
    tasks = sorted({r["task"] for r in R.values()})
    levels = sorted({(r.get("meta") or {}).get("level") for r in R.values()
                     if r["task"].startswith("occ_") and (r.get("meta") or {}).get("level")})
    for t in tasks + (["occ_both"] if any(t.startswith("occ_") for t in tasks) else []):
        sel = lambda r: (r["task"] == t or (t == "occ_both" and r["task"].startswith("occ_")))
        keys = sorted({k[0] for k, r in R.items() if sel(r)})
        row: dict = {}
        for c in conds:
            ks = [k for k in keys if (k, c) in R]
            if not ks:
                continue
            g = [R[(k, c)]["image_id"] for k in ks]
            acc = np.array([R[(k, c)]["pred"] == R[(k, c)]["gold"] for k in ks], float)
            row[c] = {"n": len(ks), "acc": ci(acc, g),
                      "unparsed": int(sum(R[(k, c)]["pred"] is None for k in ks)),
                      "reasoning_tokens": round(float(np.mean([R[(k, c)].get("reasoning_tokens") or 0 for k in ks])), 1),
                      "frac_with_reasoning": round(float(np.mean([(R[(k, c)].get("reasoning_tokens") or 0) > 0
                                                                  for k in ks])), 3),
                      "completion_tokens": round(float(np.mean([R[(k, c)].get("completion_tokens") or 0 for k in ks])), 1)}
        for a, b in DIFFS:
            ks = [k for k in keys if (k, a) in R and (k, b) in R]
            if not ks or a not in row or b not in row:
                continue
            g = [R[(k, a)]["image_id"] for k in ks]
            d = np.array([(R[(k, a)]["pred"] == R[(k, a)]["gold"]) - (R[(k, b)]["pred"] == R[(k, b)]["gold"])
                          for k in ks], float)
            row[f"{a}_minus_{b}"] = ci(d, g)
        if row:
            res[t] = row
    if any(t.startswith("occ_") for t in tasks):
        pairs = {c: occ_pairs(R, c) for c in conds}
        res["occ_sdt"] = {c: dprime_boot(pairs[c]) | ({"twin_flip": twin_flip(R, c)} if pairs[c] else {})
                          for c in conds if pairs[c]}
        res["occ_sdt_diff"] = {f"{a}_minus_{b}": dprime_diff(pairs[a], pairs[b])
                               for a, b in DIFFS if pairs.get(a) and pairs.get(b)}
        for lv in levels:
            lp = {c: occ_pairs(R, c, lv) for c in conds}
            keys_lv = {k[0] for k, r in R.items() if r["task"].startswith("occ_")
                       and (r.get("meta") or {}).get("level") == lv}
            row = {}
            for c in conds:
                ks = [k for k in sorted(keys_lv) if (k, c) in R]
                if not ks:
                    continue
                g = [R[(k, c)]["image_id"] for k in ks]
                acc = np.array([R[(k, c)]["pred"] == R[(k, c)]["gold"] for k in ks], float)
                row[c] = {"n": len(ks), "acc": ci(acc, g)} | dprime_boot(lp[c]) | {"twin_flip": twin_flip(R, c, lv)}
            for a, b in DIFFS:
                if lp.get(a) and lp.get(b):
                    row[f"dprime_{a}_minus_{b}"] = dprime_diff(lp[a], lp[b])
            res[f"level_{lv}"] = row
    return res


def main() -> None:
    """CLI entry point."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="e9x", help="output subdirectory under outputs/")
    ap.add_argument("--models", default=None, help="comma-separated model list (default: all files found)")
    ap.add_argument("--out", default=None, help="output prefix (default outputs/bundle/<dir>_summary)")
    args = ap.parse_args()
    d = ROOT / args.dir
    models = args.models.split(",") if args.models else sorted(
        p.stem for p in d.glob("*.jsonl") if "_limit" not in p.stem and p.stem not in EXCLUDE)
    out, lines = {}, []
    for m in models:
        p = d / f"{m}.jsonl"
        if not p.exists():
            print(f"[warn] missing {p}")
            continue
        R = load(p)
        if not R:
            continue
        out[m] = summarize(R)
        lines.append(f"== {m}")
        for k, v in out[m].items():
            lines.append(f"  {k}: {json.dumps(v)}")
    prefix = args.out or f"outputs/bundle/{args.dir}_summary"
    Path(prefix).parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(f"{prefix}.json", "w"), indent=1)
    Path(f"{prefix}.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"[analyze_e9x] wrote {prefix}.json / .txt for {len(out)} models")


if __name__ == "__main__":
    main()
