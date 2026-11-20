"""Cue-congruence analysis: does free-text CoT fall back on monocular heuristics (lower = closer, larger = closer)?

Sets and cells
  twins (original occlusion twins, e10 main runs + e2d desc): vertical (front lower = congruent / front higher);
        size is always incongruent (front never larger), so it is split by area ratio (<0.77 vs >=0.77) instead.
  syn   (Synthetic Depth, e10 main runs): vertical split from colour segmentation of the flat-palette images;
        size is collinear with occlusion by construction (front drawn larger) and is not split.
  cues  (cue-factorial set, e10 --set cues): 2 size x 2 vertical cells from the generator metadata.
Metrics per model x set x cell x condition: accuracy and d' (twins/cues: pooled "subject is in front" judgement over
both phrasings; syn: yes/no), CoT - Direct (and State - Direct, desc - Direct) with cluster bootstrap CIs
(twin pairs / images), congruence contrasts and the size x vertical interaction on cues, and for the first parsed
two-object claim in CoT (and the state's front field) how often it names the lower / larger object as front.

Usage: python e13_cue_analysis.py   (writes outputs/bundle/cues.json, cues.txt)
"""
from __future__ import annotations
import vtax_paths as _vtp  # noqa: E402

import json
import math
import re
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.stats import norm

from e10_state import match_obj, parse_state
from e2f_claim_edit import first_claim, split_sents

SH = f"{_vtp.DATA_ROOT}"
O = Path(f"{_vtp.W}/outputs")
B = Path("outputs/bundle")
MODELS = ["qwen3b", "qwen7b", "internvl3_8b", "phi35v", "molmo7b", "qwen3vl_8b", "internvl35_8b"]
CONDS = ["direct", "cot", "filler_R+turn", "state", "desc"]
QRX = re.compile(r"Is the (\w+ \w+) (in front of|behind) the (\w+ \w+)\?")
N_BOOT = 1000


# ----------------------------------------------------------------------------- Synthetic Depth geometry

def syn_geometry() -> dict:
    """Per Synthetic Depth item: centre-y and visible pixel count of each named object (cached)."""
    cache = B / "syn_depth_geometry.json"
    if cache.exists():
        return json.load(open(cache))
    d = json.load(open(f"{SH}/datasets/spatial_probing/depth_ordering.json"))
    imgs, tops = [], []
    for ex in d:
        im = np.asarray(Image.open(ex["image_path"].replace("<DATA_ROOT>", SH)).convert("RGB"))
        flat = im.reshape(-1, 3)
        cnt = Counter(map(tuple, flat[(flat != 240).any(1) & (flat > 0).any(1)].tolist()))
        imgs.append(im)
        tops.append([c for c, _ in cnt.most_common(2)])
    # palette: colour word -> RGB, from the colours shared by all images that mention the word
    pal: dict[str, Counter] = {}
    for ex, t in zip(d, tops):
        for w in re.findall(r"the (\w+) \w+", ex["question"]):
            pal.setdefault(w, Counter()).update(t)
    palette = {w: c.most_common(1)[0][0] for w, c in pal.items()}
    out = {}
    for i, (ex, im) in enumerate(zip(d, imgs)):
        q = QRX.search(ex["question"])
        x, y = q.group(1), q.group(3)
        rec = {}
        for name in (x, y):
            m = (im == np.array(palette[name.split()[0]])).all(-1)
            ys = np.nonzero(m)[0]
            rec[name] = {"cy": float(ys.mean()) if len(ys) else None, "px": int(m.sum())}
        out[f"syn_depth_{i}"] = rec
    json.dump({"palette": {k: list(v) for k, v in palette.items()}, "items": out}, open(cache, "w"))
    return json.load(open(cache))


# ----------------------------------------------------------------------------- helpers

def pe(r: dict) -> str | None:
    lp = r.get("lp_sum") or r["top20"]
    fin = {k: v for k, v in lp.items() if v is not None and math.isfinite(v)}
    return max(fin, key=fin.get) if fin else None


def sdt(resp: np.ndarray, tru: np.ndarray) -> tuple[float, float]:
    h = (resp[tru].sum() + .5) / (tru.sum() + 1)
    f = (resp[~tru].sum() + .5) / ((~tru).sum() + 1)
    return float(norm.ppf(h) - norm.ppf(f)), float(-(norm.ppf(h) + norm.ppf(f)) / 2)


def geom(r: dict, sg: dict) -> dict | None:
    """front/back names, lower object, larger object (None if unknown) and the cell labels for a record."""
    q = QRX.search(r["question"])
    if not q:
        return None
    x, y = q.group(1).lower(), q.group(3).lower()
    mt = r.get("meta")
    if mt:
        front, back = mt["front_name"].lower(), mt["back_name"].lower()
        cyf = (mt["bbox_front"][1] + mt["bbox_front"][3]) / 2
        cyb = (mt["bbox_back"][1] + mt["bbox_back"][3]) / 2
        lower = front if cyf > cyb else back
        larger = front if mt["area_front_px"] > mt["area_back_px"] else back
        ratio = mt["area_front_px"] / mt["area_back_px"]
    else:
        front = x if r["gold"] == "A" else y
        back = y if front == x else x
        g = sg["items"].get(r["base_id"] if "base_id" in r else r["item_id"])
        if not g or g[x]["cy"] is None or g[y]["cy"] is None:
            return None
        lower = x if g[x]["cy"] > g[y]["cy"] else y
        larger, ratio = None, None
    return {"x": x, "y": y, "front": front, "back": back, "lower": lower, "larger": larger, "ratio": ratio,
            "vert": "front_lower" if lower == front else "front_higher",
            "size": None if larger is None else ("front_larger" if larger == front else "front_smaller")}


def local(p: Path) -> Path:
    """Copy a network storage file to local disk once (re-copy if the size changed)."""
    import shutil
    dst = Path("/tmp/e13cache") / f"{p.parent.name}_{p.name}"
    dst.parent.mkdir(exist_ok=True)
    if not dst.exists() or dst.stat().st_size != p.stat().st_size:
        shutil.copy(p, dst)
    return dst


_CACHE: dict = {}


def load(model: str, which: str) -> list[dict]:
    """e10 records (+ e2d desc for the original twins)."""
    main = which != "cues"
    if (model, main) in _CACHE:
        return _CACHE[(model, main)]
    fn = O / "e10" / (f"{model}.jsonl" if main else f"{model}_cues.jsonl")
    if not fn.exists():
        return []
    R = [json.loads(l) for l in open(local(fn))]
    if main:
        for ph, f in (("front", f"{model}.jsonl"), ("behind", f"{model}_behind.jsonl")):
            p = O / "e2d" / f
            if p.exists():
                for r in map(json.loads, open(local(p))):
                    if r["condition"] == "desc":
                        R.append(r | {"item_id": r["item_id"] + ("_behind" if ph == "behind" else ""), "base_id": r["item_id"],
                                      "phrasing": ph, "question": ("Is the {} behind the {}?" if ph == "behind" else
                                                                   "Is the {} in front of the {}?").format(r["meta"]["x_name"], r["meta"]["y_name"])})
    _CACHE[(model, main)] = R
    return R


# ----------------------------------------------------------------------------- per-cell statistics

def cell_stats(rows: dict, conds: list[str], keys: list[str], kind: str, rng) -> dict:
    """rows[(key, cond)] -> (correct, says_subject_front_or_yes, truth, cluster). Returns acc, d', c, deltas."""
    clusters = sorted({rows[(k, "direct")][3] for k in keys})
    by_cl = {c: [k for k in keys if rows[(k, "direct")][3] == c] for c in clusters}
    out = {}
    arr = {c: np.array([rows[(k, c)][:3] for k in keys], float) for c in conds if all((k, c) in rows for k in keys)}
    cl_idx = {c: np.array([keys.index(k) for k in by_cl[c]]) for c in clusters}

    def stat(a):
        d, cc = sdt(a[:, 1].astype(bool), a[:, 2].astype(bool))
        return a[:, 0].mean(), d, cc

    est = {c: stat(a) for c, a in arr.items()}
    boots = {c: [] for c in arr}
    for _ in range(N_BOOT):
        idx = np.concatenate([cl_idx[clusters[i]] for i in rng.integers(0, len(clusters), len(clusters))])
        for c, a in arr.items():
            boots[c].append(stat(a[idx]))
    q = lambda v: [round(float(np.quantile(v, .025)), 3), round(float(np.quantile(v, .975)), 3)]
    for c in arr:
        b = np.array(boots[c])
        out[c] = {"acc": round(100 * est[c][0], 1), "acc_ci": [round(100 * x, 1) for x in q(b[:, 0])],
                  "dprime": round(est[c][1], 2), "dprime_ci": [round(x, 2) for x in q(b[:, 1])], "c": round(est[c][2], 2)}
    for c in [c for c in arr if c != "direct"]:
        b0, b1 = np.array(boots["direct"]), np.array(boots[c])
        out[f"{c}-direct"] = {"acc": round(100 * (est[c][0] - est["direct"][0]), 1),
                              "acc_ci": [round(100 * x, 1) for x in q(b1[:, 0] - b0[:, 0])],
                              "dprime": round(est[c][1] - est["direct"][1], 2), "dprime_ci": [round(x, 2) for x in q(b1[:, 1] - b0[:, 1])]}
    out["n_items"], out["n_clusters"] = len(keys), len(clusters)
    out["_boots"] = {c: np.array(boots[c]) for c in arr}
    return out


def analyze_set(model: str, which: str, sg: dict, rng) -> dict:
    R = load(model, which)
    if not R:
        return {}
    sel = (lambda r: r["task"] == "syn_depth") if which == "syn" else (lambda r: r["task"].startswith("occ"))
    rows, meta, claims = {}, {}, []
    for r in R:
        if not sel(r):
            continue
        g = geom(r, sg)
        if g is None:
            continue
        p = pe(r)
        if which == "syn":
            says = p == "A"
            tru = r["gold"] == "A"
        else:
            says = (p == "A") if r["phrasing"] == "front" else (p == "B")
            tru = g["front"] == g["x"]
        rows[(r["item_id"], r["condition"])] = (float(p == r["gold"]), float(says), float(tru), r["image_id"])
        meta[r["item_id"]] = g
        if r["condition"] in ("cot", "state") and r.get("text"):
            if r["condition"] == "cot":
                fc = first_claim(split_sents(r["text"][len("Let's think step by step."):]))
                named = fc[1] if fc else None
            else:
                named = match_obj(parse_state(r["text"])["front"], g["x"], g["y"])
            claims.append({"cond": r["condition"], "named": named, **g})
    conds = [c for c in CONDS if any(k[1] == c for k in rows)]
    keys_all = sorted({k for k, c in rows if c == "direct" and all((k, cc) in rows for cc in conds)})
    if which == "cues":
        cells = {f"{s}|{v}": [k for k in keys_all if meta[k]["size"] == s and meta[k]["vert"] == v]
                 for s in ("front_larger", "front_smaller") for v in ("front_lower", "front_higher")}
        cells |= {v: [k for k in keys_all if meta[k]["vert"] == v] for v in ("front_lower", "front_higher")}
        cells |= {s: [k for k in keys_all if meta[k]["size"] == s] for s in ("front_larger", "front_smaller")}
    else:
        cells = {v: [k for k in keys_all if meta[k]["vert"] == v] for v in ("front_lower", "front_higher")}
        if which == "twins":
            cells |= {"ratio<0.77": [k for k in keys_all if meta[k]["ratio"] < 0.77],
                      "ratio>=0.77": [k for k in keys_all if meta[k]["ratio"] >= 0.77]}
    cells["all"] = keys_all
    res = {"cells": {}}
    boots = {}
    for name, ks in cells.items():
        if len(ks) < 20:
            continue
        st = cell_stats(rows, conds, ks, which, rng)
        boots[name] = st.pop("_boots")
        res["cells"][name] = st
    # congruence contrasts on CoT - Direct d' (independent bootstraps per cell -> difference of draws)
    def contrast(a, b, c="cot"):
        if a not in boots or b not in boots or c not in boots[a]:
            return None
        da = boots[a][c][:, 1] - boots[a]["direct"][:, 1]
        db = boots[b][c][:, 1] - boots[b]["direct"][:, 1]
        v = da - db
        mean = res["cells"][a][f"{c}-direct"]["dprime"] - res["cells"][b][f"{c}-direct"]["dprime"]
        return {"mean": round(mean, 2), "ci": [round(float(np.quantile(v, .025)), 2), round(float(np.quantile(v, .975)), 2)]}
    res["contrast_vertical"] = {c: contrast("front_higher", "front_lower", c) for c in conds if c != "direct"}
    if which == "cues":
        res["contrast_size"] = {c: contrast("front_smaller", "front_larger", c) for c in conds if c != "direct"}
        inter = {}
        for c in [c for c in conds if c != "direct"]:
            dd = {k: boots[k][c][:, 1] - boots[k]["direct"][:, 1] for k in boots if "|" in k}
            if len(dd) == 4:
                v = (dd["front_smaller|front_higher"] - dd["front_smaller|front_lower"]) - (dd["front_larger|front_higher"] - dd["front_larger|front_lower"])
                pt = {k: res["cells"][k][f"{c}-direct"]["dprime"] for k in dd}
                m = (pt["front_smaller|front_higher"] - pt["front_smaller|front_lower"]) - (pt["front_larger|front_higher"] - pt["front_larger|front_lower"])
                inter[c] = {"mean": round(m, 2), "ci": [round(float(np.quantile(v, .025)), 2), round(float(np.quantile(v, .975)), 2)]}
        res["interaction_size_x_vertical"] = inter
        # heuristic reliance per condition: acc(congruent) - acc(incongruent) per cue
        res["reliance"] = {c: {"vertical": round(res["cells"]["front_lower"][c]["acc"] - res["cells"]["front_higher"][c]["acc"], 1),
                               "size": round(res["cells"]["front_larger"][c]["acc"] - res["cells"]["front_smaller"][c]["acc"], 1)}
                           for c in conds}
    else:
        res["reliance"] = {c: {"vertical": round(res["cells"]["front_lower"][c]["acc"] - res["cells"]["front_higher"][c]["acc"], 1)}
                           for c in conds}
    # first claim / state front field
    cl = {}
    for c in ("cot", "state"):
        xs = [x for x in claims if x["cond"] == c]
        named = [x for x in xs if x["named"]]
        if not named:
            continue
        d = {"n": len(xs), "parsed": len(named), "names_true_front": round(100 * np.mean([x["named"] == x["front"] for x in named]), 1),
             "names_lower": round(100 * np.mean([x["named"] == x["lower"] for x in named]), 1)}
        if named[0]["larger"] is not None:
            d["names_larger"] = round(100 * np.mean([x["named"] == x["larger"] for x in named]), 1)
        for v in ("front_lower", "front_higher"):
            s = [x for x in named if x["vert"] == v]
            if s:
                d[f"true_front|{v}"] = round(100 * np.mean([x["named"] == x["front"] for x in s]), 1)
        if which == "cues":
            for k in ("front_larger|front_lower", "front_larger|front_higher", "front_smaller|front_lower", "front_smaller|front_higher"):
                s = [x for x in named if f"{x['size']}|{x['vert']}" == k]
                if s:
                    d[f"true_front|{k}"] = round(100 * np.mean([x["named"] == x["front"] for x in s]), 1)
        cl[c] = d
    res["claims"] = cl
    res["decomposition"] = decompose(res, boots, conds, which)
    return res


def decompose(res: dict, boots: dict, conds: list[str], which: str) -> dict:
    """d' components: occlusion = mean over cue cells; vertical = half (front-lower - front-higher); size likewise.

    A reader that uses only occlusion has equal d' in congruent and incongruent cells (vertical component 0); one that
    uses only "lower = closer" has d' of equal size and opposite sign (occlusion component 0).
    """
    if which == "cues":
        cells = ["front_larger|front_lower", "front_larger|front_higher", "front_smaller|front_lower", "front_smaller|front_higher"]
        if not all(c in boots for c in cells):
            return {}
        comp = {"occlusion": lambda d: sum(d) / 4,
                "vertical": lambda d: ((d[0] - d[1]) + (d[2] - d[3])) / 4,
                "size": lambda d: ((d[0] - d[2]) + (d[1] - d[3])) / 4}
    else:
        cells = ["front_lower", "front_higher"]
        if not all(c in boots for c in cells):
            return {}
        comp = {"occlusion": lambda d: (d[0] + d[1]) / 2, "vertical": lambda d: (d[0] - d[1]) / 2}
    out = {}
    q = lambda v: [round(float(np.quantile(v, .025)), 2), round(float(np.quantile(v, .975)), 2)]
    for name, f in comp.items():
        pt = {c: f([res["cells"][k][c]["dprime"] for k in cells]) for c in conds}
        bt = {c: f([boots[k][c][:, 1] for k in cells]) for c in conds}
        row = {c: {"d": round(pt[c], 2), "ci": q(bt[c])} for c in conds}
        for c in [c for c in conds if c != "direct"]:
            row[f"{c}-direct"] = {"d": round(pt[c] - pt["direct"], 2), "ci": q(bt[c] - bt["direct"])}
        out[name] = row
    return out


def analyze_api(model: str, rng) -> dict:
    """Closed model on the cue-factorial set (e9x_api.py --effort none: direct and cot, no hidden reasoning)."""
    p = O / "e9c_cues" / f"{model}.jsonl"
    if not p.exists():
        return {}
    seen, rows, meta = set(), {}, {}
    for r in map(json.loads, open(local(p))):
        k = (r["item_id"], r["condition"])
        if "error" in r or k in seen or r.get("pred") is None:
            continue
        seen.add(k)
        mt = r["meta"]
        says = (r["pred"] == "A") if r["phrasing"] == "front" else (r["pred"] == "B")
        rows[k] = (float(r["pred"] == r["gold"]), float(says), float(mt["front_name"] == mt["x_name"]), r["image_id"])
        meta[r["item_id"]] = f"{'front_larger' if mt['front_larger'] else 'front_smaller'}|{'front_higher' if mt['front_above'] else 'front_lower'}"
    conds = ["direct", "cot"]
    keys = sorted({k for k, c in rows if c == "direct" and (k, "cot") in rows})
    res, boots = {"cells": {}, "n_records": len(rows)}, {}
    cells = {c: [k for k in keys if meta[k] == c] for c in sorted(set(meta.values()))}
    cells["all"] = keys
    for name, ks in cells.items():
        if len(ks) < 20:
            continue
        st = cell_stats(rows, conds, ks, "cues", rng)
        boots[name] = st.pop("_boots")
        res["cells"][name] = st
    res["decomposition"] = decompose(res, boots, conds, "cues")
    return res


def main() -> None:
    rng = np.random.default_rng(0)
    sg = syn_geometry()
    res, lines = {}, []
    for m in MODELS:
        res[m] = {}
        for which in ("twins", "syn", "cues"):
            src = "cues" if which == "cues" else "main"
            r = analyze_set(m, which if which != "twins" else "twins", sg, rng) if src == "main" else analyze_set(m, "cues", sg, rng)
            if not r:
                continue
            res[m][which] = r
            c = r["cells"]
            def cd(k, cond="cot"):
                return f"{c[k][cond + '-direct']['dprime']:+.2f} {c[k][cond + '-direct']['dprime_ci']}" if k in c and cond + "-direct" in c[k] else "--"
            dc = r.get("decomposition", {})
            fmtd = lambda comp: (f"{comp}: D {dc[comp]['direct']['d']:+.2f} CoT {dc[comp]['cot']['d']:+.2f} CoT-D {dc[comp]['cot-direct']['d']:+.2f} {dc[comp]['cot-direct']['ci']}"
                                 if comp in dc else "")
            lines.append(f"{m:14s} {which:5s} DECOMP " + " | ".join(fmtd(c) for c in ("occlusion", "vertical", "size") if c in dc))
            if which == "cues":
                lines.append(f"{m:14s} cues  D d' " + " ".join(f"{k.replace('front_', '')}={c[k]['direct']['dprime']:.2f}" for k in c if '|' in k) +
                             " | CoT-D d' " + " ".join(f"{k.replace('front_', '')}={cd(k)}" for k in c if '|' in k) +
                             f" | inter {r['interaction_size_x_vertical'].get('cot')} vert {r['contrast_vertical'].get('cot')} size {r['contrast_size'].get('cot')}")
            else:
                lines.append(f"{m:14s} {which:5s} D d' lower={c['front_lower']['direct']['dprime']:.2f} higher={c['front_higher']['direct']['dprime']:.2f} "
                             f"| CoT-D d' lower={cd('front_lower')} higher={cd('front_higher')} | contrast {r['contrast_vertical'].get('cot')} "
                             f"| claims {r['claims'].get('cot')}")
    api = analyze_api("deepseek-v4.1-flash", rng)
    if api:
        res["api:deepseek-v4.1-flash"] = {"cues": api}
        dc = api["decomposition"]
        lines.append("deepseek-v4.1-flash (effort none) cues DECOMP " + " | ".join(
            f"{c}: D {dc[c]['direct']['d']:+.2f} CoT {dc[c]['cot']['d']:+.2f} CoT-D {dc[c]['cot-direct']['d']:+.2f} {dc[c]['cot-direct']['ci']}"
            for c in ("occlusion", "vertical", "size") if c in dc) + f" | n_records {api['n_records']}")
    json.dump(res, open(B / "cues.json", "w"), indent=1)
    (B / "cues.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
