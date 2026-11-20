"""Exp-63 analysis (CPU): all pre-registered endpoints of outputs/e63/prereg.txt with 95% cluster-bootstrap CIs.

Inputs: outputs/e63/eval/<model>.jsonl for qwen3vl_8b (base), qwen3vl_8b_e63match, qwen3vl_8b_e63cf (e63_eval.py
--merge); stored base outputs for the sanity block and the written-relation labels (Exp-60 / Exp-62 judges; rule parser
claim_parse.first_claim_v3 on the last ordered depth sentence for cues / occ); ckpt/<arm>/ training logs and the
HF-LoRA vs merged-vLLM check sample.
Metrics (rigor_e60 / rigor_e62 conventions): pred = argmax lp_sum (= top-20 under V1); P(g) = two-option renormalised
probability (missing -> 0.5); E_c = 100 * mean[P_c(gold_v | v) - P_c(gold_v | v_cf)] (Ep) and the indicator version
(Ei); for Direct on twin sets v_cf is read from the twin item's Direct record. CIs: 2,000 cluster-bootstrap draws,
seed 63, percentile 95%; one draw matrix per item list, shared by all three models, so model differences are paired.
-> outputs/e63/rigor_e63.{txt,json} (copies in vtax_exps/outputs/e63/)

Usage: python rigor_e63.py [--eval-dir DIR] [--out-dir DIR]
       python rigor_e63.py --selftest      (synthetic records from the e63_eval.py --dry job skeletons; checks the code)
"""
from __future__ import annotations
import vtax_paths as _vtp  # noqa: E402

import argparse
import json
import math
import os
import random
import shutil
import sys
from collections import Counter
from pathlib import Path

import numpy as np

import e63_common as E
from e63_common import C, vx

NB = 2000
BASE, MATCH, CF = E.EVAL_MODELS
SHORT = {BASE: "base", MATCH: "match", CF: "cf"}
DIFFS = (("cf-base", CF, BASE), ("match-base", MATCH, BASE), ("cf-match", CF, MATCH))
NATURAL = ("coco", "e60")
REPO = Path(__file__).resolve().parent


# ----------------------------------------------------------------------------- basics

def pe(r: dict) -> str | None:
    return E.pe(r)


def p2(lp: dict | None, g: str) -> float:
    """Two-option renormalised probability of g (missing -> 0.5), as rigor_e62.p2."""
    fin = {k: v for k, v in (lp or {}).items() if v is not None and math.isfinite(v) and v > -1e3}
    if not fin:
        return 0.5
    m = max(fin.values())
    z = sum(math.exp(v - m) for v in fin.values())
    return math.exp(fin[g] - m) / z if g in fin else 0.0


class Boot:
    """Cluster bootstrap over a fixed item list (draws shared by every array evaluated on that list)."""

    def __init__(self, groups: list[str]):
        keys = sorted(set(groups))
        pos = {k: i for i, k in enumerate(keys)}
        self.inv = np.array([pos[g] for g in groups])
        self.c = len(keys)
        self.idx = np.random.default_rng(E.SEED).integers(0, self.c, size=(NB, self.c))

    def ci(self, v: np.ndarray, mask: np.ndarray | None = None, scale: float = 100.0) -> dict:
        m = np.isfinite(v) if mask is None else (mask & np.isfinite(v))
        if m.sum() == 0:
            return {"mean": None, "lo": None, "hi": None, "n": 0}
        s = np.bincount(self.inv[m], weights=v[m], minlength=self.c)
        n = np.bincount(self.inv[m], minlength=self.c).astype(float)
        with np.errstate(invalid="ignore", divide="ignore"):
            d = scale * s[self.idx].sum(1) / n[self.idx].sum(1)
        d = d[np.isfinite(d)]
        return {"mean": round(float(scale * v[m].mean()), 3), "lo": round(float(np.percentile(d, 2.5)), 3),
                "hi": round(float(np.percentile(d, 97.5)), 3), "n": int(m.sum())}


def fmt(c: dict | None) -> str:
    if not c or c.get("mean") is None:
        return "n/a"
    return f"{c['mean']:+.1f} [{c['lo']:+.1f}, {c['hi']:+.1f}] (n={c['n']})"


# ----------------------------------------------------------------------------- written relation of the base R_cut

def _last_depth_claim(body: str) -> str | None:
    from claim_parse import first_claim_v3
    from e2f_claim_edit import split_sents
    sents = split_sents(body[len(C.COT_PREFIX):] if body.startswith(C.COT_PREFIX) else body)
    for s in reversed(sents):
        c = first_claim_v3([s])
        if c:
            return c[1]
    return None


def written_letters() -> dict[tuple[str, str], str | None]:
    """(set, uid) -> answer letter implied by the relation written in the BASE R_cut (None = no / unparsed relation).

    cues / occ: rule parser (last ordered two-object depth sentence, claim_parse v3); coco: Exp-62 two-judge consensus
    (FIRST / SECOND, qwen3vl_8b sample of 360); e60: Exp-60 two-judge consensus (LEFT / RIGHT)."""
    out: dict = {}
    for s, f in (("cues", f"{vx.OUT}/e10/{BASE}_cues.jsonl"), ("occ", f"{vx.OUT}/e10/{BASE}.jsonl")):
        for r in vx.read_jsonl(f):
            if r["condition"] != "cot" or not r["task"].startswith("occ"):
                continue
            cl = _last_depth_claim(E.r_cut(r["text"]))
            x, y = r["meta"]["x_name"].lower(), r["meta"]["y_name"].lower()
            if cl not in (x, y):
                out[(s, r["item_id"])] = None
                continue
            front_is_x = cl == x
            out[(s, r["item_id"])] = ("A" if front_is_x else "B") if r["phrasing"] == "front" else ("B" if front_is_x else "A")
    j62 = ("claude-opus-5-5", "gpt-6-astra")
    lab = {j: {r["tid"]: r["label"] for r in vx.read_jsonl(f"{vx.OUT}/e62/judge_rel_{j}.jsonl") if r.get("label")} for j in j62}
    items62 = {it["uid"]: it for it in vx.read_jsonl(f"{vx.OUT}/e62/stimuli/items_final.jsonl")}
    rr = json.load(open(f"{vx.OUT}/e62/rel_rows.json"))["rows"] if os.path.exists(f"{vx.OUT}/e62/rel_rows.json") else []
    from rigor_e62 import implied_letter
    for row in rr:
        if row["model"] != BASE:
            continue
        a, b = (lab[j].get(row["tid"]) for j in j62)
        l = "NONE" if row["prefilter_none"] else (a if a is not None and a == b else None)
        if l in ("FIRST", "SECOND"):
            out[("coco", row["uid"])] = implied_letter(items62[row["uid"]], row["first"] if l == "FIRST" else row["second"])
        else:
            out[("coco", row["uid"])] = None
    from e60_items import E60, stored_r
    from e60_judge import has_lr, rel_key
    j60 = ("claude-opus-5-5", "gpt-6-astra")
    lab = {j: {r["tid"]: r["label"] for r in vx.read_jsonl(f"{E60}/judge_rel_{j}.jsonl") if r.get("label")} for j in j60}
    rs = stored_r(BASE)
    for it in vx.read_jsonl(f"{E60}/items.jsonl"):
        if not (it["keep"] and it["set"] == "cvb"):
            continue
        body = E.r_cut(rs[it["uid"]])
        l = None
        if has_lr(body):
            tid = rel_key(it["x"], it["y"], body)
            a, b = (lab[j].get(tid) for j in j60)
            l = a if a is not None and a == b else None
        wl = {v.lower(): k for k, v in vx.option_words(it).items()}.get(l.lower()) if l in ("LEFT", "RIGHT") else None
        out[("e60", it["uid"])] = wl
    return out


# ----------------------------------------------------------------------------- arrays

def load(eval_dir: str) -> dict[str, dict]:
    out = {}
    for m in E.EVAL_MODELS:
        p = f"{eval_dir}/{m}.jsonl"
        out[m] = {(r["set"], r["uid"], r["arm"], r["img"]): r for r in vx.read_jsonl(p)}
        print(f"[rigor_e63] {m}: {len(out[m])} records from {p}")
    return out


def item_list(R: dict, s: str) -> list[dict]:
    """Items of set s present (Direct under v) for all three models; meta from the base model's records."""
    base = {u: r for (ss, u, a, g), r in R[BASE].items() if ss == s and a == "direct" and g == "v"}
    keep = [u for u in sorted(base) if all((s, u, "direct", "v") in R[m] for m in E.EVAL_MODELS)]
    return [base[u] for u in keep]


def arrays(R: dict, m: str, items: list[dict], s: str, wl: dict) -> dict[str, np.ndarray]:
    rec = R[m]
    A: dict[str, list] = {k: [] for k in ("acc_d", "acc_cot", "acc_brcut", "Ep_direct", "Ei_direct", "Ep_base_rcut",
                                          "Ei_base_rcut", "Ep_own_rcut", "Ei_own_rcut", "gray_agree", "gray_p_written",
                                          "gray_acc", "trunc")}
    nan = float("nan")
    for it in items:
        u, g = it["uid"], it["gold"]
        get = lambda a, i: rec.get((s, u, a, i))  # noqa: E731
        d, cot, br = get("direct", "v"), get("cot", "v"), get("base_rcut", "v")
        A["acc_d"].append(float(pe(d) == g))
        A["acc_cot"].append(float(pe(cot) == g) if cot else nan)
        A["trunc"].append(float(cot["trunc"]) if cot else nan)
        A["acc_brcut"].append(float(pe(br) == g) if br else nan)
        if s in E.TWIN_SETS:
            dcf = get("direct", "cf") if it["cf_kind"] == "mirror" else rec.get((s, it["twin_uid"], "direct", "v"))
            for arm, rv, rc in (("direct", d, dcf), ("base_rcut", br, get("base_rcut", "cf")),
                                ("own_rcut", get("own_rcut", "v"), get("own_rcut", "cf"))):
                ok = rv is not None and rc is not None
                A[f"Ep_{arm}"].append(p2(rv["lp_sum"], g) - p2(rc["lp_sum"], g) if ok else nan)
                A[f"Ei_{arm}"].append(float(pe(rv) == g) - float(pe(rc) == g) if ok else nan)
            gr, w = get("base_rcut", "gray"), wl.get((s, u))
            A["gray_acc"].append(float(pe(gr) == g) if gr else nan)
            A["gray_agree"].append(float(pe(gr) == w) if gr and w else nan)
            A["gray_p_written"].append(p2(gr["lp_sum"], w) if gr and w else nan)
        else:
            for k in ("Ep_direct", "Ei_direct", "Ep_base_rcut", "Ei_base_rcut", "Ep_own_rcut", "Ei_own_rcut",
                      "gray_agree", "gray_p_written", "gray_acc"):
                A[k].append(nan)
    out = {k: np.array(v, dtype=float) for k, v in A.items()}
    out["cot_minus_d"] = out["acc_cot"] - out["acc_d"]
    return out


# ----------------------------------------------------------------------------- sanity

def sanity(R: dict) -> dict:
    """Base model re-read in Exp-63 vs stored paper outputs (argmax agreement)."""
    rb = R[BASE]
    out: dict = {}

    def agree(pairs: list[tuple]) -> dict:
        v = [a == b for a, b in pairs if a is not None and b is not None]
        return {"n": len(v), "agree_pct": round(100 * float(np.mean(v)), 2) if v else None}

    e62 = {(r["uid"], r["arm"], r["img"]): r for r in vx.read_jsonl(f"{vx.OUT}/e62/{BASE}.jsonl")}
    for arm, sarm in (("direct", "direct"), ("base_rcut", "rcut")):
        for img, simg in (("v", "v"), ("cf", "vcf")):
            if arm == "direct" and img == "cf":
                continue
            out[f"coco_{arm}_{img}"] = agree([(pe(r), pe(e62[(u, sarm, simg)])) for (s, u, a, g), r in rb.items()
                                              if s == "coco" and a == arm and g == img and (u, sarm, simg) in e62])
    e60 = {(r["item_id"], r["img"], r["ctx"]): r for r in vx.read_jsonl(f"{vx.OUT}/e60/{BASE}.jsonl")}
    for arm, ctx in (("direct", "direct"), ("base_rcut", "rcut")):
        for img, simg in (("v", "orig"), ("cf", "mirror")):
            out[f"e60_{arm}_{img}"] = agree([(pe(r), pe(e60[(u, simg, ctx)])) for (s, u, a, g), r in rb.items()
                                             if s == "e60" and a == arm and g == img and (u, simg, ctx) in e60])
    e14 = {(r["item_id"], r["ctx"], r["img"]): r for r in vx.read_jsonl(f"{vx.OUT}/e14/{BASE}.jsonl")}
    for img, simg in (("v", "own"), ("cf", "twin")):
        out[f"occ_base_rcut_{img}_vs_e14_cut"] = agree([(pe(r), pe(e14[(u, "cut", simg)])) for (s, u, a, g), r in rb.items()
                                                        if s == "occ" and a == "base_rcut" and g == img and (u, "cut", simg) in e14])
    for s, f in (("cues", f"{vx.OUT}/e10/{BASE}_cues.jsonl"), ("occ", f"{vx.OUT}/e10/{BASE}.jsonl")):
        st = {(r["item_id"], r["condition"]): r for r in vx.read_jsonl(f)}
        for arm, cond in (("direct", "direct"), ("cot", "cot")):
            out[f"{s}_{arm}_vs_e10"] = agree([(pe(r), pe(st[(u, cond)])) for (ss, u, a, g), r in rb.items()
                                             if ss == s and a == arm and g == "v" and (u, cond) in st])
    e4 = vx.legacy_e1(BASE)
    for arm in ("direct", "cot"):
        out[f"main_{arm}_vs_e4"] = agree([(pe(r), E._pred(e4[u]["scores"][arm])) for (s, u, a, g), r in rb.items()
                                          if s == "main" and a == arm and u in e4])
    st = {r["item_id"]: r for r in vx.read_jsonl(f"{vx.OUT}/e6x/{BASE}_direct.jsonl")}
    out["heldout_direct_vs_e6x"] = agree([(pe(r), pe(st[u])) for (s, u, a, g), r in rb.items()
                                          if s == "heldout" and a == "direct" and u in st])
    return out


def train_block() -> dict:
    """Training-log endpoints and HF-LoRA (final) vs merged-vLLM agreement on the check sample."""
    out = {}
    for arm in E.ARMS:
        d = f"{E.E63}/ckpt/{arm}"
        log = vx.read_jsonl(f"{d}/train_log.jsonl")
        r: dict = {"steps_logged": len(log), "first": log[0] if log else None, "last": log[-1] if log else None}
        if os.path.exists(f"{d}/vcheck_hf_init.json"):
            r["init_vs_vllm_base"] = json.load(open(f"{d}/vcheck_hf_init.json")).get("compare")
        vp = f"{E.E63}/vcheck_{E.MODEL_OF_ARM[arm]}.json"
        if os.path.exists(f"{d}/vcheck_hf_final.json") and os.path.exists(vp):
            hf = {x["ex_id"]: x for x in json.load(open(f"{d}/vcheck_hf_final.json"))["rows"]}
            dm, ag, n = 0.0, 0, 0
            for v in json.load(open(vp))["rows"]:
                h = hf.get(v["ex_id"])
                if h is None:
                    continue
                n += 1
                vl = {k: (x if x is not None and x > -1e9 else -1e4) for k, x in v["letters"].items()}
                dm = max(dm, abs(h["letters"][h["target"]] - vl[h["target"]]))
                ag += max(h["letters"], key=h["letters"].get) == max(vl, key=vl.get)
            r["final_hf_lora_vs_merged_vllm"] = {"n": n, "argmax_agree": ag, "max_abs_dlogp_target": round(dm, 4)}
        out[arm] = r
    return out


# ----------------------------------------------------------------------------- main analysis

def analyze(eval_dir: str, out_dir: str, with_sanity: bool = True) -> dict:
    lines: list[str] = []
    P = lambda s: (lines.append(s), print(s))  # noqa: E731
    R = load(eval_dir)
    wl = written_letters()
    res: dict = {"sets": {}, "pooled": {}, "decisions": {}}
    pooled_nat, pooled_gray = [], []
    for s in E.EVAL_SETS:
        items = item_list(R, s)
        if not items:
            P(f"[{s}] no items")
            continue
        boot = Boot([it["group"] for it in items])
        A = {m: arrays(R, m, items, s, wl) for m in E.EVAL_MODELS}
        n_w = sum(wl.get((s, it["uid"])) is not None for it in items)
        S: dict = {"n_items": len(items), "n_clusters": boot.c, "n_written_relation": n_w, "models": {}, "diffs": {}}
        P(f"\n=== {s}: {len(items)} items, {boot.c} clusters" + (f", base R_cut with a written relation {n_w}" if s in E.TWIN_SETS else ""))
        keys = ["acc_d", "acc_cot", "cot_minus_d", "acc_brcut"]
        if s in E.TWIN_SETS:
            keys += ["Ep_direct", "Ei_direct", "Ep_base_rcut", "Ei_base_rcut", "Ep_own_rcut", "Ei_own_rcut", "gray_agree",
                     "gray_p_written", "gray_acc"]
        for m in E.EVAL_MODELS:
            S["models"][SHORT[m]] = {k: boot.ci(A[m][k]) for k in keys}
            S["models"][SHORT[m]]["trunc_pct"] = round(100 * float(np.nanmean(A[m]["trunc"])), 2) if np.isfinite(A[m]["trunc"]).any() else None
            P(f"  {SHORT[m]:5s} " + " | ".join(f"{k} {fmt(S['models'][SHORT[m]][k])}" for k in keys))
        for name, a, b in DIFFS:
            S["diffs"][name] = {k: boot.ci(A[a][k] - A[b][k]) for k in keys}
            P(f"  d({name}) " + " | ".join(f"{k} {fmt(S['diffs'][name][k])}" for k in keys if not k.startswith("gray_p")))
        # rescue (base D / C groups; base R_cut fixed, original image)
        bd = np.array([it["base_direct_ok"] is True for it in items])
        bc = np.array([it["base_cot_ok"] is True for it in items])
        known = np.array([it["base_direct_ok"] is not None and it["base_cot_ok"] is not None for it in items])
        S["rescue"] = {}
        for gname, mask in (("D+C-", known & bd & ~bc), ("D+C+", known & bd & bc)):
            S["rescue"][gname] = {SHORT[m]: boot.ci(A[m]["acc_brcut"], mask) for m in E.EVAL_MODELS}
            S["rescue"][gname]["cf-base"] = boot.ci(A[CF]["acc_brcut"] - A[BASE]["acc_brcut"], mask)
            P(f"  rescue {gname}: " + " | ".join(f"{k} {fmt(v)}" for k, v in S["rescue"][gname].items()))
        # per phrasing image effect (secondary)
        if s in ("coco", "cues", "occ"):
            S["by_phrasing"] = {}
            for ph in sorted({it["phrasing"] for it in items}):
                mask = np.array([it["phrasing"] == ph for it in items])
                S["by_phrasing"][ph] = {SHORT[m]: {k: boot.ci(A[m][k], mask) for k in ("Ep_base_rcut", "Ei_base_rcut", "acc_d")}
                                        for m in E.EVAL_MODELS}
                P(f"  phrasing {ph}: " + " | ".join(f"{SHORT[m]} Ep_base_rcut {fmt(S['by_phrasing'][ph][SHORT[m]]['Ep_base_rcut'])}"
                                                   for m in E.EVAL_MODELS))
        res["sets"][s] = S
        # pooled pieces
        nat_mask = np.array([s in ("heldout", "coco") or (s == "main" and str(it.get("task", "")).startswith("cvb")) for it in items])
        if nat_mask.any():
            pooled_nat.append((s, [it for it, k in zip(items, nat_mask) if k], {m: A[m]["acc_d"][nat_mask] for m in E.EVAL_MODELS}))
        if s in E.TWIN_SETS:
            gm = np.isfinite(A[BASE]["gray_agree"])
            pooled_gray.append((s, [it for it, k in zip(items, gm) if k], {m: A[m]["gray_agree"][gm] for m in E.EVAL_MODELS}))
    # pooled H2 / H4
    for name, parts in (("natural_direct_acc", pooled_nat), ("gray_agree_written", pooled_gray)):
        if not parts:
            continue
        groups = [f"{s}:{it['group']}" for s, its, _ in parts for it in its]
        boot = Boot(groups)
        V = {m: np.concatenate([v[m] for *_, v in parts]) for m in E.EVAL_MODELS}
        res["pooled"][name] = {"sets": [s for s, *_ in parts], "n": len(groups),
                               **{SHORT[m]: boot.ci(V[m]) for m in E.EVAL_MODELS},
                               **{d: boot.ci(V[a] - V[b]) for d, a, b in DIFFS}}
        P(f"\n=== pooled {name} ({res['pooled'][name]['sets']}): " + " | ".join(
            f"{k} {fmt(res['pooled'][name][k])}" for k in ("base", "match", "cf", "cf-base", "match-base", "cf-match")))
    # decisions (prereg)
    dec = res["decisions"]
    h1 = {}
    for s, k in (("coco", "Ep_base_rcut"), ("e60", "Ei_base_rcut")):
        S = res["sets"].get(s)
        if not S:
            h1[s] = None
            continue
        cb, cm = S["diffs"]["cf-base"][k], S["diffs"]["cf-match"][k]
        h1[s] = {"metric": k, "cf-base": cb, "cf-match": cm, "rise_ge_20": cb["mean"] is not None and cb["mean"] >= 20,
                 "cf-base_lo_gt_0": cb["lo"] is not None and cb["lo"] > 0, "cf-match_lo_gt_0": cm["lo"] is not None and cm["lo"] > 0}
        h1[s]["pass"] = h1[s]["rise_ge_20"] and h1[s]["cf-base_lo_gt_0"] and h1[s]["cf-match_lo_gt_0"]
        h1[s]["pass_vs_base_only"] = h1[s]["rise_ge_20"] and h1[s]["cf-base_lo_gt_0"]
        k2 = "Ei_base_rcut" if k == "Ep_base_rcut" else "Ep_base_rcut"
        h1[s]["robustness_other_metric"] = {"metric": k2, "cf-base": S["diffs"]["cf-base"][k2], "cf-match": S["diffs"]["cf-match"][k2]}
    ok = [v["pass"] for v in h1.values() if v]
    vb = [v["pass_vs_base_only"] for v in h1.values() if v]
    h1["verdict"] = ("SUPPORTED" if len(ok) == 2 and all(ok) else
                     "PARTIAL" if (sum(ok) == 1 or (len(vb) == 2 and all(vb))) else "NOT SUPPORTED")
    dec["H1"] = h1
    nat = res["pooled"].get("natural_direct_acc")
    dec["H2"] = {"cf-base": nat["cf-base"] if nat else None,
                 "verdict": None if not nat else ("NON-INFERIOR" if nat["cf-base"]["lo"] > -2 else "NOT NON-INFERIOR")}
    h3 = {}
    for s in ("main", "cues"):
        S = res["sets"].get(s)
        if S:
            c = S["diffs"]["cf-base"]["cot_minus_d"]
            h3[s] = {"cf-base": c, "pass": c["lo"] is not None and c["lo"] > 0}
    h3["verdict"] = "SUPPORTED" if h3 and all(v["pass"] for v in h3.values()) else (
        "PARTIAL" if any(v["pass"] for v in h3.values()) else "NOT SUPPORTED")
    dec["H3"] = h3
    g = res["pooled"].get("gray_agree_written")
    dec["H4"] = {"cf": g["cf"] if g else None,
                 "verdict": None if not g else ("SUPPORTED" if g["cf"]["mean"] > 50 and g["cf"]["lo"] > 50 else "NOT SUPPORTED")}
    P("\n=== DECISIONS (prereg)")
    for k, v in dec.items():
        P(f"  {k}: {v['verdict']} | " + json.dumps({kk: vv for kk, vv in v.items() if kk != "verdict"}, default=str)[:900])
    if with_sanity:
        res["sanity_base_vs_stored"] = sanity(R)
        P("\n=== SANITY base re-read vs stored: " + json.dumps(res["sanity_base_vs_stored"]))
        res["training"] = train_block()
        P("=== TRAINING " + json.dumps(res["training"], default=str)[:2000])
    os.makedirs(out_dir, exist_ok=True)
    json.dump(res, open(f"{out_dir}/rigor_e63.json", "w"), indent=1, default=str)
    open(f"{out_dir}/rigor_e63.txt", "w").write("\n".join(lines) + "\n")
    return res


# ----------------------------------------------------------------------------- selftest

def selftest() -> None:
    """Synthetic eval records from the e63_eval.py --dry job skeletons (base: no image effect under base_rcut; cf: a
    strong one; match: half of it) -> the analysis must run end to end and recover the planted ordering."""
    dry = f"{E.E63}/eval_dry"
    jobs = [r for f in sorted(os.listdir(dry)) if f.startswith(f"{BASE}__") for r in vx.read_jsonl(f"{dry}/{f}")]
    assert jobs, "run e63_eval.py --model qwen3vl_8b --dry first"
    tmp = Path("/tmp/e63_selftest")
    shutil.rmtree(tmp, ignore_errors=True)
    (tmp / "eval").mkdir(parents=True)
    rng = random.Random(0)
    strength = {BASE: 0.0, MATCH: 0.4, CF: 0.9}
    for m in E.EVAL_MODELS:
        recs = []
        for j in jobs:
            g = j["gold"] if j["img"] != "cf" else (j["gold_cf"] or j["gold"])
            if j["arm"] in ("direct", "cot"):
                p_right = 0.8
            elif j["img"] == "gray":
                p_right = 0.5
            else:  # base_rcut / own_rcut: base answers by the text (= gold of v), fine-tuned models follow the image
                p_right = strength[m] if j["img"] == "cf" else 0.9
                if j["img"] == "cf" and rng.random() > strength[m]:
                    g = j["gold"]
                    p_right = 1.0
            L = j["letters"]
            pick = g if rng.random() < p_right else rng.choice([x for x in L if x != g] or L)
            lp = {x: (-0.05 if x == pick else -3.0) for x in L}
            recs.append(j | {"model": m, "top20": lp, "lp_sum": lp, "lp_bare": lp, "pred_top20": pick, "pred_exact": pick,
                             "trunc": False})
        vx.write_jsonl(str(tmp / "eval" / f"{m}.jsonl"), recs)
    res = analyze(str(tmp / "eval"), str(tmp), with_sanity=False)
    co = res["sets"].get("coco", {}).get("diffs", {})
    ok = bool(co) and co["cf-base"]["Ei_base_rcut"]["mean"] > co["match-base"]["Ei_base_rcut"]["mean"] > 0
    print(f"[rigor_e63 selftest] planted order cf > match > base recovered on coco: {ok}")
    assert ok


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-dir", default=f"{E.E63}/eval")
    ap.add_argument("--out-dir", default=E.E63)
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        return selftest()
    analyze(a.eval_dir, a.out_dir)
    local = Path(_vtp.OUTPUTS) / "e63"
    local.mkdir(parents=True, exist_ok=True)
    for f in ("rigor_e63.txt", "rigor_e63.json"):
        shutil.copy2(f"{a.out_dir}/{f}", local / f)


if __name__ == "__main__":
    sys.setrecursionlimit(10000)
    main()
