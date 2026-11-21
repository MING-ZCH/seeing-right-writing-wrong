"""Exp-58 Part A: label-free 'BodyRel' readout.

For each item, if the conclusion-removed rationale body has a consensus written relation (Exp-45 two-judge consensus on
synthetic sets, Exp-35 consensus on CV-Bench 3D Depth / BLINK Relative Depth), answer the option implied by that relation
for the asked question; otherwise fall back to (i) the CoT answer (primary) or (ii) the Direct answer.
Mapping (synthetic yes/no questions "Is X in front of / behind Y?"): yes iff the relation puts X in front and the question
asks "in front of", or puts the other shape in front and the question asks "behind" (e45_analyze.implied); natural
questions ask which option is closer, so the judged option is the answer.
Paired cluster bootstrap (clusters = Exp-45 / Exp-35 'cl': cue twin pair, twin pair, Syn. Depth image, natural image),
2,000 draws shared by all models within a set; natural pooled resamples clusters within each benchmark.
Pre-registration: outputs/e58/prereg.txt (written before any BodyRel number was computed).
-> outputs/bundle/e58_bodyrel.json, outputs/bundle/e58_bodyrel.txt

Usage: python -m experiments.interventions.body_relation_readout
"""
from __future__ import annotations

from collections import defaultdict

import numpy as np

from experiments.two_channel import natural_two_channel as N
from experiments.two_channel.uniform_analysis import agg, from_sums, implied, load_labels, natural_records, shape_records
from experiments.two_channel.uniform_two_channel import E45, M7, rj

EXTRA = ["qwen3vl_32b", "qwen3vl_8b_think"]
SETS = ["cues", "twins", "syn431", "cvb3d", "blink", "natural"]
NB = 2000
SEED = 58


# ============================================================================= records

def synthetic_records() -> list[dict]:
    rows = [r for r in rj(E45 / "items.jsonl") if r["group"] == "open"]
    recs = shape_records(rows, load_labels())
    out = []
    for row, r in zip(rows, recs):
        ans = implied(r["claim"], row["x"], row["phrasing"]) if r["has"] else None
        out.append({"model": r["model"], "ds": r["ds"], "cl": r["cl"], "item": r["item"], "has": r["has"],
                    "true": r["true"], "body_ans": ans, "body_ok": ans == row["gold"] if ans else None,
                    "d_ok": r["d_ok"], "c_ok": r["c_ok"], "status": r["status"]})
    return out


def natural_items() -> list[dict]:
    items = N.rj(N.E35 / "items.jsonl")
    N.attach(items, N.labels(), "consensus")
    out = []
    for r in items:
        ans = r["claim"]
        out.append({"model": r["model"], "ds": r["ds"], "cl": f"{r['ds']}|{r['cl']}", "item": r["item"],
                    "has": ans is not None, "true": (ans == r["gold"]) if ans else None, "body_ans": ans,
                    "body_ok": (ans == r["gold"]) if ans else None,
                    "d_ok": r["d_ans"] == r["gold"], "c_ok": r["c_ans"] == r["gold"], "status": r["lab_status"]})
    return out


def add_readouts(recs: list[dict]) -> None:
    for r in recs:
        r["br_c"] = bool(r["body_ok"]) if r["has"] else bool(r["c_ok"])
        r["br_d"] = bool(r["body_ok"]) if r["has"] else bool(r["d_ok"])


# ============================================================================= sanity vs the Exp-45 bundle

def sanity_s13(recs_syn: list[dict], recs_nat: list[dict], e45_cells: dict) -> dict:
    rows = [r for r in recs_syn] + [r for r in recs_nat if not (r["model"] == "phi35v" and r["ds"] == "blink")]
    g = defaultdict(list)
    for r in rows:
        if r["model"] in M7 + EXTRA:
            g[(r["model"], r["ds"])].append(r)
    res, bad_bundle = {}, []
    for (m, ds), rs in sorted(g.items()):
        if not (ds in ("cues", "twins", "syn431", "cvb3d", "blink")):
            continue
        st = from_sums(agg([{**r, "agree": False} if r["has"] else {**r, "agree": None} for r in rs]).sum(0))
        grp = "natural" if ds in ("cvb3d", "blink") else "open"
        ref = e45_cells.get(f"{grp}|{m}|{ds}")
        row = {k: st.get(k) for k in ("coverage", "W", "acc_direct", "direct_cov", "acc_cot")}
        if ref is not None:
            for k in row:
                if row[k] is not None and abs(row[k] - ref[k]) > 6e-4:  # bundle stores 3 decimals
                    bad_bundle.append((m, ds, k, row[k], ref[k]))
            row["A_bundle"] = ref.get("A")
        res[f"{m}|{ds}"] = row
    return {"cells": res, "mismatch_vs_bundle": bad_bundle, "n_cells": len(res)}


def sanity_A(recs_syn: list[dict], e45_cells: dict) -> list:
    """Recompute A with the agreement of the CoT answer (needs c_ans) from shape_records directly."""
    rows = [r for r in rj(E45 / "items.jsonl") if r["group"] == "open"]
    recs = shape_records(rows, load_labels())
    g = defaultdict(list)
    for r in recs:
        g[(r["model"], r["ds"])].append(r)
    bad = []
    for (m, ds), rs in g.items():
        o = from_sums(agg(rs).sum(0))
        ref = e45_cells[f"open|{m}|{ds}"]
        for k in ("A", "aT", "aF", "x", "y"):
            if k in ref and abs(o[k] - ref[k]) > 6e-4:
                bad.append((m, ds, k, o[k], ref[k]))
    return bad


# ============================================================================= bootstrap

def cluster_weights(cl_lists: list[list[str]], nb: int, rng) -> tuple[dict, np.ndarray]:
    """Stratified cluster bootstrap: one stratum per list. Returns cluster index map and (nb, C) weight matrix."""
    idx, mats = {}, []
    for cls in cl_lists:
        u = sorted(set(cls))
        base = len(idx)
        for i, c in enumerate(u):
            idx[c] = base + i
        draws = rng.integers(0, len(u), (nb, len(u)))
        W = np.zeros((nb, len(u)))
        for b in range(nb):
            W[b] = np.bincount(draws[b], minlength=len(u))
        mats.append(W)
    return idx, np.concatenate(mats, 1)


KEYS = ("d_ok", "c_ok", "br_c", "br_d")


def set_stats(by_model: dict, strata: list[list[str]], rng, nb: int = NB, mean_models: list[str] = M7) -> dict:
    """by_model: model -> records. Shared cluster draws across models. Returns per-model and 7-model-mean results."""
    cidx, W = cluster_weights(strata, nb, rng)
    C = W.shape[1]
    out, draws = {}, {}
    for m, rs in by_model.items():
        inv = np.array([cidx[r["cl"]] for r in rs])
        V = np.array([[float(r[k]) for k in KEYS] for r in rs])
        S = np.zeros((C, len(KEYS)))
        np.add.at(S, inv, V)
        n = np.bincount(inv, minlength=C).astype(float)
        est = V.mean(0) * 100
        bd = (W @ S) / (W @ n)[:, None] * 100  # (nb, 4)
        d = {"brc_dir": bd[:, 2] - bd[:, 0], "brc_cot": bd[:, 2] - bd[:, 1], "brd_dir": bd[:, 3] - bd[:, 0],
             "brd_cot": bd[:, 3] - bd[:, 1], "cot_dir": bd[:, 1] - bd[:, 0]}
        e = {"brc_dir": est[2] - est[0], "brc_cot": est[2] - est[1], "brd_dir": est[3] - est[0], "brd_cot": est[3] - est[1],
             "cot_dir": est[1] - est[0]}
        o = {"n": len(rs), "n_clusters": int((n > 0).sum()), "coverage": 100 * float(np.mean([r["has"] for r in rs])),
             "acc_direct": est[0], "acc_cot": est[1], "acc_brc": est[2], "acc_brd": est[3],
             "W": 100 * float(np.mean([r["body_ok"] for r in rs if r["has"]])) if any(r["has"] for r in rs) else None}
        for k in d:
            o[k] = float(e[k])
            o[f"{k}_ci"] = [float(np.quantile(d[k], .025)), float(np.quantile(d[k], .975))]
        out[m] = o
        draws[m] = d
    mm = [m for m in mean_models if m in by_model]
    if len(mm) == len(mean_models):
        mean = {"models": mm}
        for k in ("acc_direct", "acc_cot", "acc_brc", "acc_brd", "coverage"):
            mean[k] = float(np.mean([out[m][k] for m in mm]))
        for k in ("brc_dir", "brc_cot", "brd_dir", "brd_cot", "cot_dir"):
            v = np.mean([draws[m][k] for m in mm], 0)
            mean[k] = float(np.mean([out[m][k] for m in mm]))
            mean[f"{k}_ci"] = [float(np.quantile(v, .025)), float(np.quantile(v, .975))]
            mean[f"{k}_n_pos"] = int(sum(out[m][f"{k}_ci"][0] > 0 for m in mm))
            mean[f"{k}_n_neg"] = int(sum(out[m][f"{k}_ci"][1] < 0 for m in mm))
        out["mean7"] = mean
    return out


def run(lines: list[str]) -> dict:
    import json
    from experiments.two_channel.uniform_two_channel import B
    syn = synthetic_records()
    nat = natural_items()
    add_readouts(syn)
    add_readouts(nat)
    e45 = json.load(open(B / "e45_uniform_two_channel.json"))["cells"]
    # ---- sanity
    mism_map = [(r["model"], r["ds"], r["item"]) for r in syn + nat if r["has"] and r["body_ok"] != r["true"]]
    s13 = sanity_s13(syn, nat, e45)
    badA = sanity_A(syn, e45)
    lines.append(f"SANITY mapping: BodyRel answer correct == relation true on {sum(r['has'] for r in syn + nat)} covered items, "
                 f"{len(mism_map)} mismatches")
    lines.append(f"SANITY Exp-45 cells ({s13['n_cells']} cells): coverage / W / Dir / Dir|c / CoT vs bundle mismatches "
                 f"{s13['mismatch_vs_bundle']}; "
                 f"A / aT / aF / x / y recomputed vs bundle mismatches {badA}")
    for k, v in s13["cells"].items():
        lines.append(f"  S13 {k:26s} cov {v['coverage']:5.1f} W {v['W']:5.1f} Dir {v['acc_direct']:5.1f} Dir|c {v['direct_cov']:5.1f} "
                     f"CoT {v['acc_cot']:5.1f} A {v.get('A_bundle') if v.get('A_bundle') is None else round(v['A_bundle'], 1)}")
    # ---- per set
    res = {"_doc": __doc__, "sanity": {"mapping_mismatch": len(mism_map), "s13": s13, "A_mismatch": badA}, "sets": {}}
    rng = np.random.default_rng(SEED)
    allr = syn + nat
    for variant in ("primary", "with_phi_blink"):
        for ds in SETS:
            if variant == "with_phi_blink" and ds not in ("blink", "natural"):
                continue
            keep = lambda r: (variant == "with_phi_blink" or not (r["model"] == "phi35v" and r["ds"] == "blink"))
            if ds == "natural":
                rs = [r for r in allr if r["ds"] in ("cvb3d", "blink") and keep(r)]
                strata = [[r["cl"] for r in rs if r["ds"] == "cvb3d"], [r["cl"] for r in rs if r["ds"] == "blink"]]
            else:
                rs = [r for r in allr if r["ds"] == ds and keep(r)]
                strata = [[r["cl"] for r in rs]]
            bm = defaultdict(list)
            for r in rs:
                if r["model"] in M7 + EXTRA:
                    bm[r["model"]].append(r)
            st = set_stats(dict(bm), strata, rng, mean_models=M7 if ds != "blink" or variant == "with_phi_blink"
                           else [m for m in M7 if m != "phi35v"])
            key = ds if variant == "primary" else f"{ds}_with_phi_blink"
            res["sets"][key] = st
    # ---- print
    f1 = lambda v: f"{v:+.1f}"
    ci = lambda v: f"[{v[0]:+.1f}, {v[1]:+.1f}]"
    for key, st in res["sets"].items():
        for m in M7 + EXTRA + ["mean7"]:
            if m not in st:
                continue
            o = st[m]
            lines.append(f"BODYREL {key:24s} {m:16s} n {o.get('n', '-')!s:>5s} cov {o['coverage']:5.1f} Dir {o['acc_direct']:5.1f} CoT {o['acc_cot']:5.1f} "
                         f"BR(fbC) {o['acc_brc']:5.1f} BR(fbD) {o['acc_brd']:5.1f} | BRc-Dir {f1(o['brc_dir'])} {ci(o['brc_dir_ci'])} "
                         f"BRc-CoT {f1(o['brc_cot'])} {ci(o['brc_cot_ci'])} | BRd-Dir {f1(o['brd_dir'])} {ci(o['brd_dir_ci'])} "
                         f"BRd-CoT {f1(o['brd_cot'])} {ci(o['brd_cot_ci'])} | CoT-Dir {f1(o['cot_dir'])}"
                         + (f" | CI>0/<0 BRc-Dir {o['brc_dir_n_pos']}/{o['brc_dir_n_neg']} BRc-CoT {o['brc_cot_n_pos']}/{o['brc_cot_n_neg']} "
                            f"BRd-Dir {o['brd_dir_n_pos']}/{o['brd_dir_n_neg']}" if m == "mean7" else ""))
    p1, p2 = res["sets"]["cues"]["mean7"], res["sets"]["natural"]["mean7"]
    verdict = lambda o: "BEATS Direct" if o["brc_dir_ci"][0] > 0 else ("BELOW Direct" if o["brc_dir_ci"][1] < 0 else "inconclusive")
    res["primary"] = {"P1_cues_in_sample": {"est": p1["brc_dir"], "ci": p1["brc_dir_ci"], "verdict": verdict(p1)},
                      "P2_natural_held_out": {"est": p2["brc_dir"], "ci": p2["brc_dir_ci"], "verdict": verdict(p2)}}
    lines.append(f"PRIMARY P1 cue set (in-sample) BodyRel(fb CoT) - Direct, 7-model mean {p1['brc_dir']:+.2f} {ci(p1['brc_dir_ci'])} -> {verdict(p1)}")
    lines.append(f"PRIMARY P2 natural pooled (held-out) BodyRel(fb CoT) - Direct, 7-model mean {p2['brc_dir']:+.2f} {ci(p2['brc_dir_ci'])} -> {verdict(p2)}")
    return res


def jdefault(o):
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, np.bool_):
        return bool(o)
    return str(o)


def main() -> None:
    import argparse
    import json
    from experiments.two_channel.uniform_two_channel import B
    argparse.ArgumentParser(description="Label-free BodyRel readout (CPU; reads outputs/e35, outputs/e45 and the "
                                        "e45 bundle file).").parse_args()
    lines: list[str] = []
    res = run(lines)
    json.dump(res, open(B / "e58_bodyrel.json", "w"), indent=1, default=jdefault)
    (B / "e58_bodyrel.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
