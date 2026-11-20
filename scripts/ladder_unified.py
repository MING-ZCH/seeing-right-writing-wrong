"""Unified Qwen3-VL scale-ladder table on the original twins, one run per size.

Runs: 2B / 4B / 32B from outputs/scale (open-model protocol); 8B from outputs/e10 (the re-run behind the per-half and
AUC ladder rows of ladder_cues.json), and, for reference, the 8B main twin run outputs/e2d (Tab. S14 / S15).
Per size: Direct / CoT accuracy (both phrasings), front-lower accuracy, pooled d' (subject-in-front, both phrasings,
log-linear correction), pair-matched d'_pair = ln[(BC+.5)/(BW+.5)]/1.702 over (scene, phrasing) units of the two twins,
with CoT-Direct differences and 95% scene-bootstrap CIs (2,000; both twins and both phrasings of a scene kept together).
Per-half occlusion-minus-size and AUC changes are copied from ladder_cues.json (rigor_e17g.py).
-> outputs/bundle/ladder_unified.json
Usage: python ladder_unified.py
"""
from __future__ import annotations
import vtax_paths as _vtp  # noqa: E402

import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import norm

from build_bundle import pe, rj

O = Path(f"{_vtp.OUTPUTS}")
B = Path("outputs/bundle")
LP = 1.702


def records(src: str) -> list[dict]:
    """Twin records with phrasing, scene, twin flag, says-front, truth, correct."""
    out = []
    if src.startswith("scale_"):
        s = src[6:]
        files = ((O / "scale" / f"qwen3vl_{s}.jsonl", "front"), (O / "scale" / f"qwen3vl_{s}_behind.jsonl", "behind"))
    elif src == "e10_8b":
        files = ((O / "e10" / "qwen3vl_8b.jsonl", None),)
    else:
        files = ((O / "e2d" / "qwen3vl_8b.jsonl", "front"), (O / "e2d" / "qwen3vl_8b_behind.jsonl", "behind"))
    for fn, ph0 in files:
        for r in rj(fn):
            if not r["task"].startswith("occ") or r["condition"] not in ("direct", "cot"):
                continue
            ph = ph0 or r["phrasing"]
            iid = r["item_id"].replace("_behind", "")
            p = pe(r)
            mt = r["meta"]
            out.append({"cond": r["condition"], "ph": ph, "scene": iid.replace("_cf", ""), "cf": iid.endswith("_cf"),
                        "says": (p == "A") if ph == "front" else (p == "B"), "truth": mt["front_name"] == mt["x_name"],
                        "ok": p == r["gold"], "lower": not mt["front_above"]})
    return out


def dprime(s: np.ndarray, t: np.ndarray) -> float:
    h = (s[t].sum() + .5) / (t.sum() + 1)
    f = (s[~t].sum() + .5) / ((~t).sum() + 1)
    return float(norm.ppf(h) - norm.ppf(f))


def analyze(src: str, nboot: int = 2000) -> dict:
    R = records(src)
    scenes = sorted({r["scene"] for r in R})
    sid = {s: i for i, s in enumerate(scenes)}
    res, arrs = {}, {}
    for c in ("direct", "cot"):
        rs = [r for r in R if r["cond"] == c]
        says = np.array([r["says"] for r in rs]); tru = np.array([r["truth"] for r in rs])
        ok = np.array([r["ok"] for r in rs]); low = np.array([r["lower"] for r in rs])
        sc = np.array([sid[r["scene"]] for r in rs])
        units = defaultdict(dict)
        for r in rs:
            units[(r["scene"], r["ph"])]["cf" if r["cf"] else "o"] = r["says"] == r["truth"]
        uk = sorted(k for k, v in units.items() if len(v) == 2)
        oc = np.array([0 if units[k]["o"] and units[k]["cf"] else 3 if not (units[k]["o"] or units[k]["cf"]) else 1 for k in uk])
        usc = np.array([sid[k[0]] for k in uk])
        arrs[c] = (says, tru, ok, low, sc, oc, usc)
        res[c] = {"n": len(rs), "acc": 100 * ok.mean(), "acc_front_lower": 100 * ok[low].mean(), "dprime": dprime(says, tru),
                  "dpair": np.log(((oc == 0).sum() + .5) / ((oc == 3).sum() + .5)) / LP, "n_units": len(uk),
                  "BC": int((oc == 0).sum()), "BW": int((oc == 3).sum())}
    rng = np.random.default_rng(0)
    idx = {c: [np.where(arrs[c][4] == i)[0] for i in range(len(scenes))] for c in arrs}
    uidx = {c: [np.where(arrs[c][6] == i)[0] for i in range(len(scenes))] for c in arrs}
    bs = defaultdict(list)
    for _ in range(nboot):
        js = rng.integers(0, len(scenes), len(scenes))
        st = {}
        for c in arrs:
            says, tru, ok, low, sc, oc, usc = arrs[c]
            i = np.concatenate([idx[c][j] for j in js]); u = np.concatenate([uidx[c][j] for j in js])
            st[c] = (dprime(says[i], tru[i]), np.log(((oc[u] == 0).sum() + .5) / ((oc[u] == 3).sum() + .5)) / LP, ok[i].mean())
        for k, j in (("dprime", 0), ("dpair", 1), ("acc", 2)):
            bs[k].append(st["cot"][j] - st["direct"][j])
    res["cot-direct"] = {k: {"est": res["cot"][k] - res["direct"][k],
                             "ci": [float(np.quantile(v, .025)) * (100 if k == "acc" else 1), float(np.quantile(v, .975)) * (100 if k == "acc" else 1)]}
                         for k, v in bs.items()}
    return res


def main() -> None:
    lc = json.load(open(B / "ladder_cues.json"))
    out = {"_doc": __doc__}
    for size, src in (("2b", "scale_2b"), ("4b", "scale_4b"), ("8b", "e10_8b"), ("8b_main_e2d", "e2d_8b"), ("32b", "scale_32b")):
        r = analyze(src)
        if size in lc:
            h, g = lc[size]["halves"], lc[size]["glm"]
            r["per_half_occ_minus_size"] = h["cot-direct"]["occ_minus_size"]
            r["auc"] = {"direct": g["direct"]["auc"]["auc"], "cot": g["cot"]["auc"]["auc"], "diff": g["cot-direct"]["auc"]}
            r["halves_acc"] = {c: {k: h[c][k]["acc"] for k in ("front_lower", "front_higher")} for c in ("direct", "cot")}
        out[size] = r
        d, c, dd = r["direct"], r["cot"], r["cot-direct"]
        f = lambda x, nd=2: f"{x['est']:+.{nd}f} [{x['ci'][0]:+.{nd}f},{x['ci'][1]:+.{nd}f}]"
        print(f"{size:12s} src={src:10s} acc {d['acc']:.2f}->{c['acc']:.2f} lowacc {d['acc_front_lower']:.1f}->{c['acc_front_lower']:.1f} "
              f"d' {d['dprime']:.2f}->{c['dprime']:.2f} {f(dd['dprime'])} | d'pair {d['dpair']:.2f}->{c['dpair']:.2f} {f(dd['dpair'])} "
              f"(BC/BW {d['BC']}/{d['BW']} -> {c['BC']}/{c['BW']}, units {d['n_units']})"
              + (f" | half {r['per_half_occ_minus_size']} AUC {r['auc']['direct']}->{r['auc']['cot']} {r['auc']['diff']}" if "auc" in r else ""))
    json.dump(out, open(B / "ladder_unified.json", "w"), indent=1, default=float)


if __name__ == "__main__":
    main()
