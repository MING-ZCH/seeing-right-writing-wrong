"""Track X2 analyses (review_supp_r1/FIX_PLAN.md): X2a truncation robustness of the fixed own-rationale twin swap and
X2b D+C+ conflict-direction prefix swap. CPU only; sections whose GPU outputs are missing report what exists (rule
statistics of the dry-run contexts; X2b from the stored Exp-28 / Exp-57 readouts).

x2a   per model x arm (full / rcut / uniform / uniform_join / strict / strict62; outputs/x2/trunc_swap/<m>.jsonl from
      x2_trunc_swap.py): image effect = 100 [1(answer = gold_v | R_arm, v) - 1(answer = gold_v | R_arm, v_cf)]
      (argmax of the exact / top-20 readout, as rigor_e17e for E14), twin-pair bootstrap (build_bundle.ci: 2,000
      resamples, seed 0; cluster = scene pair, both images and phrasings); probability-scale effect; accuracy under v;
      effect - effect(rcut) paired over queries, overall and on the queries whose context the rule changed. Rule
      statistics from outputs/x2/trunc_ctx/<m>.jsonl: share of rationales changed vs R and vs R_cut, characters, empty
      bodies. Sanity: rcut / full reproduce E14 (bundle imgswap.json cut:all / full:all = main Tab. 2 own-R_cut column;
      per-record agreement with outputs/e14/<m>.jsonl) and, for Qwen3-VL-32B, the Exp-57 twin imgswap records.
      Direct twin effect (2 acc - 100) for reference.
x2b   per model x anchor (first_true primary / judge_true / jstar; outputs/x2/dcp_anchors.jsonl from
      x2_dcp_prefix_swap.py): twin-image effect 100 [P(gold | prefix, v) - P(gold | prefix, v_cf)] just before vs just
      after the anchor sentence and the paired change, twin-pair bootstrap (2,000, seed 0); accuracy scale (P > .5);
      subsets clean / first claim / phrasing; event profile at offsets -2..+3 around the anchor. Sources: 'stored' =
      Exp-28 (8B) / Exp-57 (32B) trajectories + Exp-57 trajtwin (turn layout; no GPU needed); 'gpu_turn' / 'gpu_same' =
      outputs/x2/dcp_prefix/<m>.jsonl, with a replicate check of gpu_turn against the stored readouts.
--selftest: the x2a code on E14 records relabelled as rcut / full (must give imgswap.json exactly) and the x2b code
      on the stored readouts relabelled as GPU turn-layout records (must equal 'stored', replicate diff 0); temp dir.
-> outputs/bundle/x2_trunc_swap.json, outputs/bundle/x2_dcp_prefix.json, outputs/bundle/rigor_x2.txt
Usage: python rigor_x2.py [--only x2a,x2b] [--selftest]
"""
from __future__ import annotations
import vtax_paths as _vtp  # noqa: E402

import argparse
import json
import tempfile
from collections import Counter
from pathlib import Path

import numpy as np

from build_bundle import ci, pe, probs, rj

O = Path(f"{_vtp.OUTPUTS}")
B = Path("outputs/bundle")
M7 = ["qwen3b", "qwen7b", "internvl3_8b", "phi35v", "molmo7b", "qwen3vl_8b", "internvl35_8b"]
M8 = M7 + ["qwen3vl_32b"]
M2 = ["qwen3vl_8b", "qwen3vl_32b"]
NAMES = {"qwen3b": "Qwen2.5-VL-3B", "qwen7b": "Qwen2.5-VL-7B", "internvl3_8b": "InternVL3-8B", "phi35v": "Phi-3.5-V",
         "molmo7b": "Molmo-7B-D", "qwen3vl_8b": "Qwen3-VL-8B", "internvl35_8b": "InternVL3.5-8B", "qwen3vl_32b": "Qwen3-VL-32B"}
ARMS = ("full", "rcut", "uniform", "uniform_join", "strict", "strict62")
ANCH = ("first_true", "judge_true", "jstar")
TRAJ = {"qwen3vl_8b": O / "e28" / "qwen3vl_8b_cues_traj.jsonl", "qwen3vl_32b": O / "e57" / "qwen3vl_32b_cues_traj.jsonl"}
PATHS = {"x2": O / "x2"}
fm = lambda x: f"{x['mean']:.1f} [{x['lo']:.1f}, {x['hi']:.1f}]" if x and "mean" in x else "--"
fa = lambda x: f"{x['mean']:+.1f} [{x['lo']:+.1f}, {x['hi']:+.1f}]" if x and "mean" in x else "--"


def pgold(r: dict) -> float:
    """P(gold) of a readout record (exact lp_sum, else top-20), renormalized over the options."""
    return probs(r["lp_sum"] if "lp_sum" in r else r["top20"])[r["gold"]]


def pair_of(base_id: str) -> str:
    return base_id.replace("_cf", "")


# ============================================================================= X2a

def ctx_stats(rows: list[dict]) -> dict:
    """Rule statistics of the dry-run contexts (x2_trunc_swap.ctx_stats plus characters removed)."""
    pre = "Let's think step by step."
    n = len(rows)
    out = {"n": n, "uniform_leading_run_pct": round(100 * sum(r["uniform_leading_run"] for r in rows) / n, 1)}
    for a in ARMS:
        out[a] = {"changed_vs_full_pct": round(100 * sum(r["ctx"][a] != r["ctx"]["full"] for r in rows) / n, 1),
                  "changed_vs_rcut_pct": round(100 * sum(r["ctx"][a] != r["ctx"]["rcut"] for r in rows) / n, 1),
                  "mean_chars": round(sum(len(r["ctx"][a]) for r in rows) / n, 1),
                  "chars_removed_vs_full_pct": round(100 * (1 - sum(len(r["ctx"][a]) for r in rows) / sum(len(r["ctx"]["full"]) for r in rows)), 1),
                  "empty_body": sum(r["ctx"][a].strip() == pre for r in rows)}
    return out


def direct_ref(m: str) -> dict | None:
    """Direct twin image effect 2 acc - 100 (main Tab. 2 'Direct' column; Qwen3-VL-32B: Exp-57)."""
    if m == "qwen3vl_32b":
        return json.load(open(B / "e57_sentence32b.json"))["imgswap"]["twins"]["all"]["direct_effect"]
    a = json.load(open(B / "relation_sensitivity.json"))[m]["occ"]["direct_acc"]
    return {"mean": 2 * a["mean"] - 100, "lo": 2 * a["lo"] - 100, "hi": 2 * a["hi"] - 100, "n": a["n"]}


def sanity_ref(m: str) -> tuple[dict, dict]:
    """Stored own-R readouts ({(item_id, ctx in full/cut, img): record}) and the stored effects for the default rule."""
    if m == "qwen3vl_32b":
        recs = rj(O / "e57" / "qwen3vl_32b_twins_imgswap.jsonl")
        st = json.load(open(B / "e57_sentence32b.json"))["imgswap"]["twins"]["all"]
        eff = {"cut": st["cut"]["image_effect"], "full": st["full"]["image_effect"], "src": "e57_sentence32b.json imgswap.twins.all"}
    else:
        recs = rj(O / "e14" / f"{m}.jsonl")
        st = json.load(open(B / "imgswap.json"))[m]
        eff = {"cut": st["cut:all"]["dvis"], "full": st["full:all"]["dvis"], "src": "imgswap.json (rigor_e17e; main Tab. 2)"}
    return {(r["item_id"], r["ctx"], r["img"]): r for r in recs if r["img"] in ("own", "twin")}, eff


def x2a_model(m: str) -> dict:
    x2 = PATHS["x2"]
    ctx = rj(x2 / "trunc_ctx" / f"{m}.jsonl")
    res: dict = {"ctx_stats": ctx_stats(ctx) if ctx else None, "direct_effect": direct_ref(m)}
    R = rj(x2 / "trunc_swap" / f"{m}.jsonl")
    if not R:
        res["status"] = "pending (no GPU readouts yet)"
        return res
    by = {(r["item_id"], r["arm"], r["img"]): r for r in R}
    keys = sorted({r["item_id"] for r in R if all((r["item_id"], a, i) in by for a in ARMS for i in ("own", "twin"))})
    g = [by[(k, "rcut", "own")]["image_id"] for k in keys]
    dv, dp = {}, {}
    arms = {}
    for a in ARMS:
        own = np.array([float(pe(by[(k, a, "own")]) == by[(k, a, "own")]["gold"]) for k in keys])
        tw = np.array([float(pe(by[(k, a, "twin")]) == by[(k, a, "own")]["gold"]) for k in keys])
        dv[a] = own - tw
        dp[a] = np.array([pgold(by[(k, a, "own")]) - pgold(by[(k, a, "twin")]) for k in keys])
        arms[a] = {"image_effect": ci(dv[a], g), "prob_effect": ci(dp[a], g), "acc_own": ci(own, g),
                   "acc_goldv_under_twin": ci(tw, g),
                   "flip": ci([float(pe(by[(k, a, "own")]) != pe(by[(k, a, "twin")])) for k in keys], g)}
    C = {r["item_id"]: r for r in ctx}
    for a in ARMS:
        arms[a]["minus_rcut"] = ci(dv[a] - dv["rcut"], g)
        arms[a]["prob_minus_rcut"] = ci(dp[a] - dp["rcut"], g)
        ch = [i for i, k in enumerate(keys) if k in C and C[k]["ctx"][a] != C[k]["ctx"]["rcut"]]
        if len(ch) >= 10:
            gc = [g[i] for i in ch]
            arms[a]["changed"] = {"n": len(ch), "image_effect": ci(dv[a][ch], gc), "image_effect_rcut": ci(dv["rcut"][ch], gc),
                                  "minus_rcut": ci(dv[a][ch] - dv["rcut"][ch], gc)}
    res["n_queries"] = len(keys)
    res["arms"] = arms
    ref, eff = sanity_ref(m)
    san = {"stored": eff}
    for a, c in (("rcut", "cut"), ("full", "full")):
        pairs = [(by[(k, a, i)], ref[(k, c, i)]) for k in keys for i in ("own", "twin") if (k, c, i) in ref]
        san[a] = {"n_readouts": len(pairs), "pred_agree_pct": round(100 * float(np.mean([pe(x) == pe(y) for x, y in pairs])), 2) if pairs else None,
                  "max_abs_pgold_diff": round(float(max(abs(pgold(x) - pgold(y)) for x, y in pairs)), 4) if pairs else None,
                  "effect_new": arms[a]["image_effect"], "effect_stored": eff[c],
                  "abs_diff": round(abs(arms[a]["image_effect"]["mean"] - eff[c]["mean"]), 3)}
    san["reproduced"] = san["rcut"]["abs_diff"] <= 1.0 and (san["rcut"]["pred_agree_pct"] or 0) >= 98.0
    res["sanity"] = san
    return res


def x2a(lines: list[str]) -> dict:
    res = {}
    for m in M8:
        r = x2a_model(m)
        res[m] = r
        cs = r["ctx_stats"]
        if cs:
            lines.append(f"X2A-RULES {NAMES[m]:15s} n={cs['n']} changed vs R_cut: " +
                         " ".join(f"{a} {cs[a]['changed_vs_rcut_pct']:.1f}%" for a in ARMS if a != "rcut") +
                         f" | chars removed vs R: " + " ".join(f"{a} {cs[a]['chars_removed_vs_full_pct']:.1f}%" for a in ARMS if a != "full") +
                         f" | empty uniform {cs['uniform']['empty_body']} strict {cs['strict']['empty_body']} | uniform leading run {cs['uniform_leading_run_pct']}%")
        if "arms" not in r:
            lines.append(f"X2A {NAMES[m]:15s} {r['status']} | Direct twin effect {fm(r['direct_effect'])}")
            continue
        A = r["arms"]
        lines.append(f"X2A {NAMES[m]:15s} n={r['n_queries']} image effect | Direct {fm(r['direct_effect'])} | " +
                     " | ".join(f"{a} {fm(A[a]['image_effect'])}" for a in ARMS))
        lines.append(f"X2A {NAMES[m]:15s} minus R_cut (paired)    | " + " | ".join(f"{a} {fa(A[a]['minus_rcut'])}" for a in ARMS if a != "rcut") +
                     " || on changed queries: " + " | ".join(f"{a} {A[a]['changed']['n']}: {fm(A[a]['changed']['image_effect_rcut'])} -> {fm(A[a]['changed']['image_effect'])}"
                                                         for a in ("uniform", "strict", "strict62") if "changed" in A[a]))
        lines.append(f"X2A {NAMES[m]:15s} prob-scale effect       | " + " | ".join(f"{a} {fm(A[a]['prob_effect'])}" for a in ARMS))
        s = r["sanity"]
        lines.append(f"X2A-SANITY {NAMES[m]:15s} rcut {fm(s['rcut']['effect_new'])} vs stored {fm(s['stored']['cut'])} (|d| {s['rcut']['abs_diff']}, "
                     f"pred agree {s['rcut']['pred_agree_pct']}% of {s['rcut']['n_readouts']}, max |dP| {s['rcut']['max_abs_pgold_diff']}) | full "
                     f"{fm(s['full']['effect_new'])} vs {fm(s['stored']['full'])} (agree {s['full']['pred_agree_pct']}%) | "
                     f"{'REPRODUCED' if s['reproduced'] else 'NOT REPRODUCED'} [{s['stored']['src']}]")
    return res


# ============================================================================= X2b

def stored_source(m: str) -> tuple[dict, dict]:
    """Trajectories and {(item_id, j, img): P(gold)} from the stored turn-layout readouts (traj + trajtwin)."""
    T = {r["item_id"]: r for r in rj(TRAJ[m])}
    P = {}
    for r in rj(O / "e57" / f"{m}_cues_trajtwin.jsonl"):
        for j, (po, pt) in enumerate(zip(r["p_gold_own"], r["p_gold_twin"])):
            P[(r["item_id"], j, "v")] = po
            P[(r["item_id"], j, "vcf")] = pt
    return T, P


def gpu_source(m: str) -> dict[str, dict]:
    """{layout: {(item_id, j, img): P(gold)}} from the x2_dcp_prefix_swap.py readouts (empty if not run yet)."""
    out: dict[str, dict] = {}
    for r in rj(PATHS["x2"] / "dcp_prefix" / f"{m}.jsonl"):
        out.setdefault(r["layout"], {})[(r["item_id"], r["j"], r["img"])] = pgold(r)
    return out


def anchor_block(A: list[dict], P: dict) -> dict:
    """Before / after effects for one anchor set on one source (rows missing a readout are dropped and counted)."""
    rows = [a for a in A if all((a["item_id"], j, i) in P for j in (a["idx"], a["idx"] + 1) for i in ("v", "vcf"))]
    if len(rows) < 10:
        return {"n": len(rows), "n_missing": len(A) - len(rows)}
    g = [pair_of(a["base_id"]) for a in rows]
    get = lambda a, j, i: P[(a["item_id"], j, i)]
    pvb = np.array([get(a, a["idx"], "v") for a in rows]); pcb = np.array([get(a, a["idx"], "vcf") for a in rows])
    pva = np.array([get(a, a["idx"] + 1, "v") for a in rows]); pca = np.array([get(a, a["idx"] + 1, "vcf") for a in rows])
    eb, ea = pvb - pcb, pva - pca
    ab = (pvb > .5).astype(float) - (pcb > .5); aa = (pva > .5).astype(float) - (pca > .5)
    return {"n": len(rows), "n_missing": len(A) - len(rows), "n_clusters": len(set(g)),
            "effect_before": ci(eb, g), "effect_after": ci(ea, g), "change": ci(ea - eb, g),
            "acc_effect_before": ci(ab, g), "acc_effect_after": ci(aa, g), "acc_change": ci(aa - ab, g),
            "p_v_before": round(float(pvb.mean()), 3), "p_v_after": round(float(pva.mean()), 3),
            "p_vcf_before": round(float(pcb.mean()), 3), "p_vcf_after": round(float(pca.mean()), 3)}


def profile(A: list[dict], T: dict, P: dict) -> dict:
    """Effect at prefix j = anchor + offset (offset 0 = just before, 1 = just after the anchor sentence)."""
    out = {}
    for off in range(-2, 4):
        rows = [a for a in A if 0 <= a["idx"] + off <= len(T[a["item_id"]]["sentences"])
                and (a["item_id"], a["idx"] + off, "vcf") in P]
        if len(rows) >= 10:
            out[off] = ci([P[(a["item_id"], a["idx"] + off, "v")] - P[(a["item_id"], a["idx"] + off, "vcf")] for a in rows],
                          [pair_of(a["base_id"]) for a in rows])
    return out


SUBSETS = {"all": lambda a: True, "clean": lambda a: a["clean"], "first_claim": lambda a: a["first_claim"],
           "front": lambda a: a["phrasing"] == "front", "behind": lambda a: a["phrasing"] == "behind"}


def replicate(Pg: dict, Ps: dict) -> dict:
    """GPU turn-layout readouts vs the stored readouts of the same prefixes."""
    ks = [k for k in Pg if k in Ps]
    if not ks:
        return {"n": 0}
    d = np.array([Pg[k] - Ps[k] for k in ks])
    return {"n": len(ks), "max_abs_diff": round(float(np.abs(d).max()), 4), "mean_abs_diff": round(float(np.abs(d).mean()), 5),
            "share_abs_diff_lt_0.01": round(100 * float((np.abs(d) < .01).mean()), 2),
            "pred_agree_pct": round(100 * float(np.mean([(Pg[k] > .5) == (Ps[k] > .5) for k in ks])), 2)}


def x2b(lines: list[str]) -> dict:
    p = PATHS["x2"] / "dcp_anchors.jsonl"
    allA = rj(p)
    if not allA:
        raise SystemExit(f"{p} missing: run `python x2_dcp_prefix_swap.py anchors` first")
    res = {}
    for m in M2:
        A = [a for a in allA if a["model"] == m]
        T, Ps = stored_source(m)
        srcs = {"stored": Ps} | {f"gpu_{lay}": P for lay, P in gpu_source(m).items()}
        out = {"anchors": dict(Counter(a["anchor"] for a in A)), "sources": {}}
        for sn, P in srcs.items():
            blk = {}
            for an in ANCH:
                Aa = [a for a in A if a["anchor"] == an]
                for sub, f in SUBSETS.items():
                    if an == "jstar" and sub in ("clean", "first_claim"):
                        continue
                    blk[f"{an}|{sub}"] = anchor_block([a for a in Aa if f(a)], P)
                if sn == "stored":
                    blk[f"{an}|profile"] = profile(Aa, T, P)
            out["sources"][sn] = blk
        if "gpu_turn" in srcs:
            out["replicate_gpu_turn_vs_stored"] = replicate(srcs["gpu_turn"], Ps)
        res[m] = out
        for sn, blk in out["sources"].items():
            for an in ANCH:
                x = blk[f"{an}|all"]
                if "change" not in x:
                    lines.append(f"X2B {NAMES[m]:13s} {sn:8s} {an:10s} n={x['n']} (missing {x['n_missing']})")
                    continue
                lines.append(f"X2B {NAMES[m]:13s} {sn:8s} {an:10s} n={x['n']:4d} ({x['n_clusters']} pairs) twin-image effect before "
                             f"{fm(x['effect_before'])} -> after {fm(x['effect_after'])}, change {fa(x['change'])} | P(gold|v) "
                             f"{x['p_v_before']:.3f}->{x['p_v_after']:.3f} P(gold|v_cf) {x['p_vcf_before']:.3f}->{x['p_vcf_after']:.3f} | "
                             f"acc scale {fm(x['acc_effect_before'])} -> {fm(x['acc_effect_after'])}")
            for sub in ("clean", "first_claim", "front", "behind"):
                x = blk.get(f"first_true|{sub}", {})
                if "change" in x:
                    lines.append(f"X2B {NAMES[m]:13s} {sn:8s} first_true|{sub:11s} n={x['n']:4d} before {fm(x['effect_before'])} -> after "
                                 f"{fm(x['effect_after'])}, change {fa(x['change'])}")
            if sn == "stored":
                for an in ("first_true", "jstar"):
                    pr = blk[f"{an}|profile"]
                    lines.append(f"X2B {NAMES[m]:13s} stored   {an:10s} profile (offset from anchor; 0 = before, 1 = after): " +
                                 " ".join(f"{o:+d}: {v['mean']:.1f} (n {v['n']})" for o, v in pr.items()))
        if "replicate_gpu_turn_vs_stored" in out:
            lines.append(f"X2B-REPLICATE {NAMES[m]} gpu_turn vs stored: {out['replicate_gpu_turn_vs_stored']}")
    return res


# ============================================================================= self-test

def selftest() -> None:
    """Run both sections on relabelled stored readouts in a temp dir and check the known answers."""
    tmp = Path(tempfile.mkdtemp(prefix="x2_selftest_"))
    real = PATHS["x2"]
    for d in ("trunc_ctx", "trunc_swap", "dcp_prefix"):
        (tmp / d).mkdir(parents=True)
    (tmp / "dcp_anchors.jsonl").write_text((real / "dcp_anchors.jsonl").read_text())
    m = "qwen7b"
    E = rj(O / "e14" / f"{m}.jsonl")
    with open(tmp / "trunc_swap" / f"{m}.jsonl", "w") as f:
        for r in E:
            arms = ["full"] if r["ctx"] == "full" else [a for a in ARMS if a != "full"]
            for a in arms:
                f.write(json.dumps({**r, "arm": a, "read_as": "rcut" if r["ctx"] == "cut" else "full"}) + "\n")
    (tmp / "trunc_ctx" / f"{m}.jsonl").write_text((real / "trunc_ctx" / f"{m}.jsonl").read_text())
    for mm in M2:
        T, Ps = stored_source(mm)
        A = [a for a in rj(real / "dcp_anchors.jsonl") if a["model"] == mm]
        need = {(a["item_id"], j) for a in A for j in (a["idx"], a["idx"] + 1)}
        with open(tmp / "dcp_prefix" / f"{mm}.jsonl", "w") as f:
            for k, j in sorted(need):
                for img in ("v", "vcf"):
                    p = Ps[(k, j, img)]
                    lp = {"A": float(np.log(max(p, 1e-12))), "B": float(np.log(max(1 - p, 1e-12)))}
                    if T[k]["gold"] == "B":
                        lp = {"A": lp["B"], "B": lp["A"]}
                    f.write(json.dumps({"item_id": k, "j": j, "img": img, "layout": "turn", "gold": T[k]["gold"], "lp_sum": lp}) + "\n")
    PATHS["x2"] = tmp
    try:
        lines: list[str] = []
        ra = x2a_model(m)
        rb = x2b(lines)
    finally:
        PATHS["x2"] = real
    st = json.load(open(B / "imgswap.json"))[m]
    assert ra["arms"]["rcut"]["image_effect"] == st["cut:all"]["dvis"], (ra["arms"]["rcut"]["image_effect"], st["cut:all"]["dvis"])
    assert ra["arms"]["full"]["image_effect"] == st["full:all"]["dvis"]
    assert ra["sanity"]["rcut"]["pred_agree_pct"] == 100.0 and ra["sanity"]["reproduced"]
    assert ra["arms"]["uniform"]["minus_rcut"]["mean"] == 0.0
    for mm in M2:
        S, G = rb[mm]["sources"]["stored"], rb[mm]["sources"]["gpu_turn"]
        for an in ANCH:
            for key in ("effect_before", "effect_after", "change"):
                a, b = S[f"{an}|all"][key], G[f"{an}|all"][key]
                assert abs(a["mean"] - b["mean"]) < 1e-6 and a["n"] == b["n"], (mm, an, key, a, b)
        rep = rb[mm]["replicate_gpu_turn_vs_stored"]
        assert rep["max_abs_diff"] < 1e-6 and rep["pred_agree_pct"] == 100.0, rep
    j57 = json.load(open(B / "e57_sentence32b.json"))["expl"]
    for mm in M2:
        ref = j57[f"{mm}|trajtwin"]["D+C-"]
        x = rb[mm]["sources"]["stored"]["jstar|all"]
        assert (x["effect_before"]["mean"], x["effect_after"]["mean"]) == (ref["before_jstar"]["mean"], ref["after_jstar"]["mean"]), (mm, x, ref)
    print(f"[rigor_x2 selftest] PASS: x2a reproduces imgswap.json for {m} (rcut {fm(ra['arms']['rcut']['image_effect'])}, "
          f"full {fm(ra['arms']['full']['image_effect'])}); x2b GPU path equals the stored path; jstar reproduces Exp-57 "
          f"before / after ({tmp})", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="x2a,x2b")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        selftest()
        return
    lines: list[str] = []
    secs = a.only.split(",")
    if "x2a" in secs:
        json.dump(x2a(lines), open(B / "x2_trunc_swap.json", "w"), indent=1, default=float)
    if "x2b" in secs:
        json.dump(x2b(lines), open(B / "x2_dcp_prefix.json", "w"), indent=1, default=float)
    old = (B / "rigor_x2.txt").read_text().splitlines() if (B / "rigor_x2.txt").exists() else []
    keep = [l for l in old if not any(l.startswith(f"{s.upper()}") for s in secs)]
    (B / "rigor_x2.txt").write_text("\n".join(keep + lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
