"""Held-out breakdown: frozen 767 rows, pooled 667 (excl. 3DSR orientation_on_the_left), depth-valid subset
(BLINK Relative Depth + 3DSR location_closer_to_camera + 3DSR orientation_in_front_of, n=324).

Per model and subset: Direct, CoT, CoT-Direct, Sym-Direct, CoT-if-consistent-else-Direct, frozen gate(CoT) vs Direct
and vs random routing at the same rate, flag precision/recall (CoT order-inconsistent as a predictor of CoT error),
and d'/c for the binary depth tasks (Direct vs CoT; option A as "signal"). Image-cluster bootstrap (2,000).
Also Qwen3-VL-8B-Thinking (Direct vs native think, outputs/e6x/qwen3vl_8b_think_think.jsonl) when present.

Usage: python heldout_breakdown.py  (writes outputs/bundle/heldout_breakdown.json/.txt)
"""
from __future__ import annotations
import vtax_paths as _vtp  # noqa: E402

import glob
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import norm

from build_bundle import ci, pe, probs, rj

O = Path(f"{_vtp.OUTPUTS}")
B = Path("outputs/bundle")
MODELS = ["qwen3b", "qwen7b", "internvl3_8b", "phi35v", "molmo7b", "qwen3vl_8b", "internvl35_8b"]
DEPTH = {"blink_depth", "3dsr_location_closer_to_camera", "3dsr_orientation_in_front_of"}
SUBSETS = {"all767": lambda t: True, "pooled667": lambda t: t != "3dsr_orientation_on_the_left",
           "depth324": lambda t: t in DEPTH, "blink_depth": lambda t: t == "blink_depth"}


def sdt_ab(pred: list[str], gold: list[str]) -> dict:
    """d', c with option A as the signal response (log-linear correction)."""
    p, g = np.array([x == "A" for x in pred]), np.array([x == "A" for x in gold])
    h = (p[g].sum() + .5) / (g.sum() + 1)
    f = (p[~g].sum() + .5) / ((~g).sum() + 1)
    return {"dprime": round(float(norm.ppf(h) - norm.ppf(f)), 2), "c": round(float(-(norm.ppf(h) + norm.ppf(f)) / 2), 2)}


def dprime_boot(pd: list[str], pc: list[str], gold: list[str], img: list[str], n: int = 2000) -> dict:
    """d' Direct, CoT and CoT - Direct with image-cluster bootstrap CI."""
    rng = np.random.default_rng(0)
    cl = sorted(set(img))
    idx = {c: [i for i, x in enumerate(img) if x == c] for c in cl}
    diffs = []
    for _ in range(n):
        s = np.concatenate([idx[cl[j]] for j in rng.integers(0, len(cl), len(cl))])
        a = sdt_ab([pd[i] for i in s], [gold[i] for i in s])["dprime"]
        b = sdt_ab([pc[i] for i in s], [gold[i] for i in s])["dprime"]
        diffs.append(b - a)
    d0, d1 = sdt_ab(pd, gold), sdt_ab(pc, gold)
    return {"direct": d0, "cot": d1, "diff": round(d1["dprime"] - d0["dprime"], 2),
            "diff_ci": [round(float(np.quantile(diffs, .025)), 2), round(float(np.quantile(diffs, .975)), 2)]}


def model_rows(m: str) -> list[dict]:
    """One row per held-out item with predictions under each method."""
    D = {r["item_id"]: r for r in rj(O / "e6x" / f"{m}_direct.jsonl")}
    rest = rj(O / "e6x" / f"{m}_rest.jsonl")
    Cc = {r["item_id"]: r for r in rest if r["condition"] == "cot"}
    swp = rj(O / "e5c" / f"{m}_heldout.jsonl")
    sd = {r["item_id"]: r for r in swp if r["condition"] == "direct_swap"}
    sc = {r["item_id"]: r for r in swp if r["condition"] == "cot_swap"}
    pred = json.load(open(sorted(glob.glob(str(O / "e6x" / f"predictions_{m}_*.json")))[0]))
    tau = pred["tau"].get("cot")
    fl = {}
    if (O / "e6x" / f"{m}_filler.jsonl").exists():
        for r in rj(O / "e6x" / f"{m}_filler.jsonl"):
            fl.setdefault(r["item_id"], {})[r["condition"]] = pe(r)
    rows = []
    for k, d in D.items():
        if k not in Cc:
            continue
        pd0 = probs(d.get("lp_sum") or d["top20"])
        pc0 = probs(Cc[k].get("lp_sum") or Cc[k]["top20"])
        srt = sorted(pd0.values(), reverse=True)
        row = {"item": k, "task": d["task"], "gold": d["gold"], "img": d["image_id"], "pd": pe(d), "pc": pe(Cc[k]),
               "margin": srt[0] - (srt[1] if len(srt) > 1 else 0), "tau": tau, "n_opt": len(d["letters"]),
               "cot_empty": not Cc[k].get("text", "").replace("Let's think step by step.", "").strip()}
        if k in fl and "filler_R+turn" in fl[k]:
            row |= {"pf": fl[k]["filler_R+turn"], "pp": fl[k]["cot_prefix_only"]}
        if k in sd and k in sc and len(d["letters"]) == 2:
            gold, letters = d["gold"], d["letters"]
            other = [x for x in letters if x != gold][0]
            fm = {gold: other, other: gold}
            pd1 = probs(sd[k].get("lp_sum") or sd[k]["top20"])
            pc1 = probs(sc[k].get("lp_sum") or sc[k]["top20"])
            sym = {x: pd0[x] + pd1[fm[x]] for x in letters}
            pc1_arg = fm[max(pc1, key=pc1.get)]
            row |= {"sym": max(sym, key=sym.get), "cons_else_d": row["pc"] if row["pc"] == pc1_arg else row["pd"],
                    "flag": row["pc"] != pc1_arg}
        rows.append(row)
    return rows


def summarize(rows: list[dict]) -> dict:
    g = [r["img"] for r in rows]
    acc = lambda key: [float(r[key] == r["gold"]) for r in rows]
    d, c = np.array(acc("pd")), np.array(acc("pc"))
    out = {"n": len(rows), "direct": ci(d, g), "cot": ci(c, g), "cot_minus_direct": ci(c - d, g)}
    if rows and rows[0]["tau"] is not None:
        route = np.array([r["margin"] < r["tau"] for r in rows])
        gate = np.where(route, c, d)
        rnd = d + route.mean() * (c - d)
        out |= {"gate_minus_direct": ci(gate - d, g), "gate_minus_random": ci(gate - rnd, g), "routed": round(float(route.mean()), 3)}
    sw = [r for r in rows if "sym" in r]
    if sw:
        gs = [r["img"] for r in sw]
        ds = np.array([float(r["pd"] == r["gold"]) for r in sw])
        out |= {"n_swap": len(sw),
                "sym_minus_direct": ci(np.array([float(r["sym"] == r["gold"]) for r in sw]) - ds, gs),
                "cons_else_direct_minus_direct": ci(np.array([float(r["cons_else_d"] == r["gold"]) for r in sw]) - ds, gs)}
        fl = [r["flag"] for r in sw]
        er = [r["pc"] != r["gold"] for r in sw]
        out["flag"] = {"precision": ci([float(e) for f, e in zip(fl, er) if f], [r["img"] for r, f in zip(sw, fl) if f]),
                       "recall": ci([float(f) for f, e in zip(fl, er) if e], [r["img"] for r, e in zip(sw, er) if e]),
                       "base_error": ci([float(e) for e in er], gs), "n_flagged": int(sum(fl)), "n_err": int(sum(er))}
    for key, sel in (("sdt", lambda r: r["n_opt"] == 2), ("sdt_depth_tasks", lambda r: r["n_opt"] == 2 and r["task"] in DEPTH)):
        bin_rows = [r for r in rows if sel(r)]
        if len(bin_rows) >= 50:
            out[key] = dprime_boot([r["pd"] for r in bin_rows], [r["pc"] for r in bin_rows], [r["gold"] for r in bin_rows],
                                   [r["img"] for r in bin_rows]) | {"n": len(bin_rows)}
    fr = [r for r in rows if "pf" in r]
    if len(fr) >= 50 and len(fr) == len(rows):
        gf = [r["img"] for r in fr]
        a = lambda key: np.array([float(r[key] == r["gold"]) for r in fr])
        out["format"] = {"filler_turn": ci(a("pf"), gf), "prefix_only": ci(a("pp"), gf),
                         "cot_minus_filler_turn": ci(a("pc") - a("pf"), gf), "filler_turn_minus_direct": ci(a("pf") - a("pd"), gf),
                         "cot_minus_prefix_only": ci(a("pc") - a("pp"), gf), "prefix_only_minus_direct": ci(a("pp") - a("pd"), gf),
                         "empty_cot_share": round(100 * float(np.mean([r["cot_empty"] for r in fr])), 1)}
    return out


def think_rows() -> list[dict]:
    p = O / "e6x" / "qwen3vl_8b_think_think.jsonl"
    if not p.exists():
        return []
    R = defaultdict(dict)
    for r in rj(p):
        R[r["item_id"]][r["condition"]] = r
    return [{"item": k, "task": v["direct"]["task"], "gold": v["direct"]["gold"], "img": v["direct"]["image_id"], "pd": pe(v["direct"]),
             "pc": pe(v["think"]), "margin": 1.0, "tau": None, "n_opt": len(v["direct"]["letters"])}
            for k, v in R.items() if "direct" in v and "think" in v]


def main() -> None:
    res, lines = {}, []
    for m in MODELS + ["qwen3vl_8b_think"]:
        rows = think_rows() if m == "qwen3vl_8b_think" else model_rows(m)
        if not rows:
            continue
        res[m] = {s: summarize([r for r in rows if f(r["task"])]) for s, f in SUBSETS.items()}
        for s in SUBSETS:
            x = res[m][s]
            f = lambda d: f"{d['mean']:+.1f} [{d['lo']:+.1f},{d['hi']:+.1f}]" if isinstance(d, dict) and "mean" in d else "--"
            line = f"{m:16s} {s:10s} n={x['n']:3d} D={x['direct']['mean']:.1f} CoT-D={f(x['cot_minus_direct'])}"
            if "sym_minus_direct" in x:
                line += f" Sym-D={f(x['sym_minus_direct'])} Cons-D={f(x['cons_else_direct_minus_direct'])}"
            if "gate_minus_direct" in x:
                line += f" Gate-D={f(x['gate_minus_direct'])} Gate-Rnd={f(x['gate_minus_random'])}"
            if "flag" in x:
                line += f" flagP={x['flag']['precision']['mean']:.1f} R={x['flag']['recall']['mean']:.1f} base={x['flag']['base_error']['mean']:.1f}"
            if "format" in x:
                fm = x["format"]
                line += (f" | FORMAT CoT-F+t={f(fm['cot_minus_filler_turn'])} F+t-D={f(fm['filler_turn_minus_direct'])} "
                         f"prefix-D={f(fm['prefix_only_minus_direct'])} CoT-prefix={f(fm['cot_minus_prefix_only'])} empty={fm['empty_cot_share']}%")
            if "sdt" in x:
                line += f" d' {x['sdt']['direct']['dprime']:.2f}->{x['sdt']['cot']['dprime']:.2f} ({x['sdt']['diff']:+.2f} {x['sdt']['diff_ci']}) c {x['sdt']['direct']['c']:+.2f}->{x['sdt']['cot']['c']:+.2f}"
            lines.append(line)
    from build_bundle import load_main
    nat = {}
    for m in MODELS + ["qwen3vl_8b_think"]:
        mn = load_main(O, Path(f"{_vtp.LEGACY_RESULTS}"), m)
        ks = [k for k, v in mn.get("direct", {}).items() if v[3] == "cvb_depth" and k in mn.get("cot", {})]
        if not ks:
            continue
        other = lambda p: "B" if p == "A" else "A"
        gold = [mn["direct"][k][4] if mn["direct"][k][0] else other(mn["direct"][k][4]) for k in ks]
        nat[m] = {"cvb_depth": dprime_boot([mn["direct"][k][4] for k in ks], [mn["cot"][k][4] for k in ks], gold,
                                           [mn["direct"][k][2] for k in ks]) | {"n": len(ks)}}
        if m in res and "sdt" in res[m]["blink_depth"]:
            nat[m]["blink_depth"] = res[m]["blink_depth"]["sdt"] | {"n": res[m]["blink_depth"]["n"]}
        for t, x in nat[m].items():
            lines.append(f"NATURAL SDT {m:16s} {t:11s} n={x['n']} d' {x['direct']['dprime']:.2f}->{x['cot']['dprime']:.2f} "
                         f"({x['diff']:+.2f} {x['diff_ci']}) c {x['direct']['c']:+.2f}->{x['cot']['c']:+.2f}")
    res["natural_sdt"] = nat
    json.dump(res, open(B / "heldout_breakdown.json", "w"), indent=1)
    (B / "heldout_breakdown.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
