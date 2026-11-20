"""Exp-29 analyses (round-17 external reviews: Opus M2/M4/M5, Gemini W2).

edit     Task 1: phrasing-congruent true-relation templates at j* (outputs/e29/<m>_<set>_edit_jstar.jsonl) and on D+C+
         items at the first claim (…_edit_dcp.jsonl): accuracy per template x question phrasing, paired differences vs
         Neutral and vs F0 (the Exp-28 template), the phrasing gap, and answer polarity. Cluster bootstrap over images.
early    Task 2 (Qwen3-VL-8B cues): earliest consensus image-contradicting relational sentence (judge_*_prej2.jsonl,
         e20 prompt) -> outputs/e29/earliest_qwen3vl_8b_cues.json; edit there (…_edit_early.jsonl) vs at j* on the same items.
ontop    Task 3: re-judge with the depth/vertical prompt (judge_*_ontop.jsonl) vs the Exp-28 / e20 labels; rates with and
         without "on top" sentences, per reading, and the vertical-true / depth-false share of commits.
desc     Task 5: full-text depth-claim audit of descriptions and CoT rationales (judge_*_desc.jsonl); share with a
         false depth claim, and whether the description-arm answer follows the described relation.
reimage  Task 4: Molmo-7B-D HF readouts (molmo7b_cues_reimage_hf_shard*.jsonl): accuracy and probit occlusion weight
         for Direct / CoT / CoT+turn / CoT+reimage, paired twin-pair bootstrap.
-> bundle/e29_*.json, bundle/rigor_e29.txt, bundle/supp_e29.tex (copied to paper tables/, not \\input)
Usage: python rigor_e29.py [--only edit,early,ontop,desc,reimage] [--earliest-only]
"""
from __future__ import annotations
import vtax_paths as _vtp  # noqa: E402

import argparse
import json
import re
import shutil
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from build_bundle import pe, rj
from rigor_e28 import J2, NAMES, cboot, pct, q95, wilson

O = Path(f"{_vtp.OUTPUTS}")
E28, E29 = O / "e28", O / "e29"
B = Path("outputs/bundle")
PAPER_TAB = Path(f"{_vtp.TABLES}")
COMP = ["molmo7b", "qwen3vl_8b"]
ARMS = ["orig", "neutral", "F0", "F1", "F2", "B0", "B1", "B2", "ECHO"]
ONTOP = re.compile(r"\bon top\b|\batop\b", re.I)
fa = lambda x: f"{x['mean']:+.1f} [{x['ci'][0]:+.1f}, {x['ci'][1]:+.1f}]"
fm = lambda x: f"{x['mean']:.1f} [{x['ci'][0]:.1f}, {x['ci'][1]:.1f}]"


# ============================================================================= Task 1: templates

def says_yes(r: dict) -> bool:
    return pe(r) == "A"


def edit_part(lines: list[str], m: str, st: str, part: str) -> dict | None:
    p = E29 / f"{m}_{st}_edit_{part}.jsonl"
    R = rj(p)
    if not R:
        return None
    rng = np.random.default_rng(29)
    idx = {(r["item_id"], r["arm"]): r for r in R}
    arms = [a for a in ARMS if any(k[1] == a for k in idx)]
    keys = sorted({r["item_id"] for r in R if all((r["item_id"], a) in idx for a in arms)})
    g = np.array([idx[(k, "orig")]["image_id"] for k in keys])
    ph = np.array([idx[(k, "orig")]["phrasing"] for k in keys])
    gold_yes = np.array([idx[(k, "orig")]["gold"] == "A" for k in keys])
    acc = {a: np.array([float(pe(idx[(k, a)]) == idx[(k, a)]["gold"]) for k in keys]) for a in arms}
    yes = {a: np.array([float(says_yes(idx[(k, a)])) for k in keys]) for a in arms}
    tm = json.load(open(next(E29.glob(f"{m}_{st}_edit_*{part}*.timing.json")))) if list(E29.glob(f"{m}_{st}_edit_*{part}*.timing.json")) else {}
    res = {"n": len(keys), "n_images": len(set(g)), "templates": tm.get("templates"), "mined": tm.get("mined")}
    for sub in ("all", "front", "behind"):
        s = np.ones(len(keys), bool) if sub == "all" else ph == sub
        gg = g[s]
        r = {"n": int(s.sum())}
        for a in arms:
            r[f"acc_{a}"] = cboot(acc[a][s], gg, rng)
        for a in arms:
            if a != "neutral":
                r[f"{a}-neutral"] = cboot(acc[a][s] - acc["neutral"][s], gg, rng)
            if a not in ("F0", "orig", "neutral"):
                r[f"{a}-F0"] = cboot(acc[a][s] - acc["F0"][s], gg, rng)
        # congruent: F-arms on 'in front of' questions, B-arms on 'behind' questions
        if sub == "all":
            for k_ in ("0", "1", "2"):
                cong = np.where(ph == "front", acc["F" + k_], acc["B" + k_])
                inc = np.where(ph == "front", acc["B" + k_], acc["F" + k_])
                r[f"cong{k_}"] = cboot(cong, gg, rng)
                r[f"incong{k_}"] = cboot(inc, gg, rng)
                r[f"cong{k_}-neutral"] = cboot(cong - acc["neutral"], gg, rng)
                r[f"cong{k_}-incong{k_}"] = cboot(cong - inc, gg, rng)
        # answer polarity: share of 'yes' answers by gold
        for a in arms:
            r[f"yes_{a}|gold_yes"] = round(100 * float(yes[a][s & gold_yes].mean()), 1) if (s & gold_yes).any() else None
            r[f"yes_{a}|gold_no"] = round(100 * float(yes[a][s & ~gold_yes].mean()), 1) if (s & ~gold_yes).any() else None
        res[sub] = r
    # phrasing gap of the congruent template: (front gain over neutral) - (behind gain over neutral)
    fr, be = ph == "front", ph == "behind"
    ug = np.unique(g)
    gi = {u: np.where(g == u)[0] for u in ug}

    def gap(a_f: str, a_b: str, sel=None) -> dict:
        def stat(ix):
            f_, b_ = ix[fr[ix]], ix[be[ix]]
            if len(f_) == 0 or len(b_) == 0:
                return np.nan
            return (acc[a_f][f_] - acc["neutral"][f_]).mean() - (acc[a_b][b_] - acc["neutral"][b_]).mean()
        est = stat(np.arange(len(keys)))
        bs = [stat(np.concatenate([gi[u] for u in rng.choice(ug, len(ug))])) for _ in range(2000)]
        return {"mean": round(100 * float(est), 1), "ci": [round(100 * x, 1) for x in q95(bs, 4)]}

    res["gap_F0_F0"] = gap("F0", "F0")
    for k_ in ("0", "1", "2"):
        res[f"gap_F{k_}_B{k_}"] = gap("F" + k_, "B" + k_)
    res["gap_ECHO"] = gap("ECHO", "ECHO")
    A, F_, Bh = res["all"], res["front"], res["behind"]
    lines.append(f"EDIT-{part.upper()} {st:5s} {m:11s} n={res['n']} (front {F_['n']}, behind {Bh['n']}) images {res['n_images']} | templates {res['templates']}")
    for a in arms:
        lines.append(f"   {a:7s} all {fm(A[f'acc_{a}'])} | front {fm(F_[f'acc_{a}'])} | behind {fm(Bh[f'acc_{a}'])}"
                     + ("" if a == "neutral" else f" | -neutral: front {fa(F_[f'{a}-neutral'])} behind {fa(Bh[f'{a}-neutral'])}")
                     + (f" | -F0: front {fa(F_[f'{a}-F0'])} behind {fa(Bh[f'{a}-F0'])}" if f"{a}-F0" in Bh else "")
                     + f" | yes|gold yes/no front {F_[f'yes_{a}|gold_yes']}/{F_[f'yes_{a}|gold_no']} behind {Bh[f'yes_{a}|gold_yes']}/{Bh[f'yes_{a}|gold_no']}")
    lines.append(f"   congruent (F on front-Q, B on behind-Q): " + "; ".join(
        f"k={k_}: cong {fm(A[f'cong{k_}'])} incong {fm(A[f'incong{k_}'])} cong-neutral {fa(A[f'cong{k_}-neutral'])} cong-incong {fa(A[f'cong{k_}-incong{k_}'])}"
        for k_ in "012"))
    lines.append(f"   phrasing gap (front gain - behind gain over neutral): F0/F0 {fa(res['gap_F0_F0'])} | F0/B0 {fa(res['gap_F0_B0'])} | "
                 f"F1/B1 {fa(res['gap_F1_B1'])} | F2/B2 {fa(res['gap_F2_B2'])} | ECHO {fa(res['gap_ECHO'])}")
    return res


def edit(lines: list[str]) -> dict:
    out = {}
    for m in COMP:
        for st in ("cues", "twins"):
            for part in ("jstar", "dcp"):
                r = edit_part(lines, m, st, part)
                if r:
                    out[f"{m}|{st}|{part}"] = r
    return out


# ============================================================================= Task 2: earliest contradicting sentence

def consensus(a: str | None, b: str | None) -> str:
    return a if (a == b and a in ("CORRECT", "INCORRECT")) else "NONCONS"


def earliest(lines: list[str]) -> dict:
    J = {j: {(r["item_id"], r["idx"]): r for r in rj(E29 / f"judge_{j}_prej2.jsonl")} for j in J2}
    keys = sorted(set(J[J2[0]]) & set(J[J2[1]]))
    per = defaultdict(list)
    for k in keys:
        per[k[0]].append(k)
    ek, rows = {}, Counter()
    for iid, ks in per.items():
        js = J[J2[0]][ks[0]]["commit_idx"]
        con = sorted(k[1] for k in ks if consensus(J[J2[0]][k]["label"], J[J2[1]][k]["label"]) == "INCORRECT")
        if con:
            ek[iid] = con[0]
            rows["before_jstar" if con[0] < js else "at_jstar"] += 1
        else:
            rows["none"] += 1
    json.dump(ek, open(E29 / "earliest_qwen3vl_8b_cues.json", "w"), indent=1)
    n = len(per)
    e28 = sum(1 for iid, ks in per.items() if J[J2[0]][ks[0]].get("in_e28_sample"))
    lines.append(f"EARLIEST qwen3vl_8b cues: {n} D+C- rationales ({e28} Exp-28 sample + {n - e28} new) | earliest consensus contradicting "
                 f"relational sentence before j* {rows['before_jstar']} ({100 * rows['before_jstar'] / n:.1f}% {wilson(rows['before_jstar'], n)}), "
                 f"at j* {rows['at_jstar']}, none {rows['none']}")
    return {"n": n, "n_e28_sample": e28, **dict(rows), "before_jstar_pct": pct(rows["before_jstar"], n)}


def early(lines: list[str], ek_info: dict) -> dict:
    R = rj(E29 / "qwen3vl_8b_cues_edit_early.jsonl")
    Jst = {(r["item_id"], r["arm"]): r for r in rj(E29 / "qwen3vl_8b_cues_edit_jstar.jsonl")}
    if not R or not Jst:
        lines.append("EARLY: missing edit files")
        return {}
    rng = np.random.default_rng(31)
    idx = {(r["item_id"], r["arm"]): r for r in R}
    keys = sorted({r["item_id"] for r in R})
    g = np.array([idx[(k, "orig")]["image_id"] for k in keys])
    ph = np.array([idx[(k, "orig")]["phrasing"] for k in keys])
    ok = lambda r: float(pe(r) == r["gold"])
    arms = ["orig", "neutral", "F0", "B0", "ECHO"]
    a_e = {a: np.array([ok(idx[(k, a)]) for k in keys]) for a in arms}
    a_j = {a: np.array([ok(Jst[(k, a)]) for k in keys]) for a in arms}
    cong_e = np.where(ph == "front", a_e["F0"], a_e["B0"])
    cong_j = np.where(ph == "front", a_j["F0"], a_j["B0"])
    res = {"n": len(keys), "n_images": len(set(g)), "mean_gap_sentences": round(float(np.mean([idx[(k, 'orig')]["commit_idx"] - idx[(k, 'orig')]["edit_idx"] for k in keys])), 2)}
    for a in arms:
        res[f"early_{a}"] = cboot(a_e[a], g, rng)
        res[f"jstar_{a}"] = cboot(a_j[a], g, rng)
        res[f"early-jstar_{a}"] = cboot(a_e[a] - a_j[a], g, rng)
    res["early_cong"], res["jstar_cong"] = cboot(cong_e, g, rng), cboot(cong_j, g, rng)
    res["early-jstar_cong"] = cboot(cong_e - cong_j, g, rng)
    res["early_cong-neutral"] = cboot(cong_e - a_e["neutral"], g, rng)
    res["early_F0-neutral"] = cboot(a_e["F0"] - a_e["neutral"], g, rng)
    lines.append(f"EARLY qwen3vl_8b cues: items with earliest contradiction before j* n={res['n']} (images {res['n_images']}, mean j*-k {res['mean_gap_sentences']}) | "
                 + " | ".join(f"{a}: early {fm(res[f'early_{a}'])} vs j* {fm(res[f'jstar_{a}'])} diff {fa(res[f'early-jstar_{a}'])}" for a in arms)
                 + f" | congruent: early {fm(res['early_cong'])} vs j* {fm(res['jstar_cong'])} diff {fa(res['early-jstar_cong'])}; "
                 f"early cong-neutral {fa(res['early_cong-neutral'])}, early F0-neutral {fa(res['early_F0-neutral'])}")
    # all 120 rationales: edit at the earliest contradicting sentence (k < j*: early run; k = j*: j* run) and none
    allk = json.load(open(E29 / "earliest_qwen3vl_8b_cues.json"))
    J1 = {r["item_id"]: r for r in rj(E29 / "judge_gpt-6-astra_prej2.jsonl")}
    ids = sorted({r["item_id"] for r in rj(E29 / "judge_gpt-6-astra_prej2.jsonl")})
    rows = []
    for iid in ids:
        if (iid, "neutral") not in Jst:
            continue
        src = idx if (iid, "neutral") in idx else Jst
        typ = "before" if (iid, "neutral") in idx else ("at_jstar" if iid in allk else "none")
        rows.append((iid, Jst[(iid, "orig")]["image_id"], typ, ok(src[(iid, "neutral")]), ok(Jst[(iid, "neutral")]),
                     ok(src[(iid, "F0")]), ok(Jst[(iid, "F0")])))
    gg = [r[1] for r in rows]
    res["all120"] = {"n": len(rows), "types": dict(Counter(r[2] for r in rows)),
                     "neutral_earliest": cboot([r[3] for r in rows], gg, rng), "neutral_jstar": cboot([r[4] for r in rows], gg, rng),
                     "neutral_earliest-jstar": cboot([r[3] - r[4] for r in rows], gg, rng),
                     "F0_earliest": cboot([r[5] for r in rows], gg, rng), "F0_jstar": cboot([r[6] for r in rows], gg, rng)}
    a = res["all120"]
    lines.append(f"EARLY-ALL qwen3vl_8b cues (n={a['n']}, {a['types']}; 'none' items edited at j*): neutral at earliest contradiction {fm(a['neutral_earliest'])} "
                 f"vs at j* {fm(a['neutral_jstar'])} diff {fa(a['neutral_earliest-jstar'])} | F0 {fm(a['F0_earliest'])} vs {fm(a['F0_jstar'])}")
    return res


# ============================================================================= Task 3: 'on top'

def own_contra(r: dict) -> str:
    """Contradiction under the reading the sentence uses (charitable for AMBIGUOUS: false only if false under both)."""
    rd, d, v = r["reading"], r["depth_true"], r["vertical_true"]
    if rd == "NONE" or "UNPARSED" in (rd, d, v):
        return "NA"
    if rd == "DEPTH":
        return {"FALSE": "YES", "TRUE": "NO"}.get(d, "NA")
    if rd == "VERTICAL":
        return {"FALSE": "YES", "TRUE": "NO"}.get(v, "NA")
    if rd == "BOTH":
        return "YES" if "FALSE" in (d, v) else "NO" if d == v == "TRUE" else "NA"
    # AMBIGUOUS
    if d == "FALSE" and v in ("FALSE", "NA"):
        return "YES"
    return "NO" if "TRUE" in (d, v) else "NA"


def depth_contra(r: dict) -> str:
    if r["reading"] in ("NONE", "VERTICAL"):
        return "NA"
    return {"FALSE": "YES", "TRUE": "NO"}.get(r["depth_true"], "NA")


def cons2(a: str, b: str) -> str:
    return a if a == b and a in ("YES", "NO") else "NC"


def rate2(labs: list[str]) -> dict:
    k, c = labs.count("YES"), labs.count("NO")
    n = len(labs)
    return pct(k, k + c) | {"n_items": n, "worst_case": [round(100 * k / n, 1) if n else None, round(100 * (n - c) / n, 1) if n else None]}


def old_labels() -> dict:
    """{(src, model, item_id, group, idx): consensus} from the Exp-28 / e20 judge files."""
    out = {}
    for src, fn in (("e28_commit_cues", "judge_{}_commit_cues.jsonl"), ("e28_commit_twins", "judge_{}_commit_twins.jsonl"),
                    ("e28_prej_cues", "judge_{}_prej_cues.jsonl")):
        L = {j: {(r["model"], r["item_id"], r["group"], r["idx"]): r["label"] for r in rj(E28 / fn.format(j))} for j in J2}
        for k in L[J2[0]]:
            out[(src,) + k] = consensus(L[J2[0]][k], L[J2[1]].get(k))
    L = {j: {} for j in J2}
    for j in J2:
        for fn in (f"{j}.jsonl", f"{j}_relcomp.jsonl"):
            for r in rj(O / "e20" / fn):
                L[j][(r["model"], r["item_id"], r["group"], r["idx"])] = r["label"]
    for k in L[J2[0]]:
        out[("e20_syn",) + k] = consensus(L[J2[0]][k], L[J2[1]].get(k))
    return out


def ontop(lines: list[str]) -> dict:
    J = {j: {(r["src"], r["model"], r["item_id"], r["group"], r["idx"]): r for r in rj(E29 / f"judge_{j}_ontop.jsonl")} for j in J2}
    if not J[J2[0]] or not J[J2[1]]:
        lines.append("ONTOP: judge files missing")
        return {}
    OL = old_labels()
    keys = sorted(set(J[J2[0]]) & set(J[J2[1]]))
    meta = {k: J[J2[1]][k] for k in keys}
    is_top = {k: bool(ONTOP.search(meta[k]["sentence"])) for k in keys}
    reading = {k: (J[J2[0]][k]["reading"] if J[J2[0]][k]["reading"] == J[J2[1]][k]["reading"] else "NC") for k in keys}
    dc = {k: cons2(depth_contra(J[J2[0]][k]), depth_contra(J[J2[1]][k])) for k in keys}
    oc = {k: cons2(own_contra(J[J2[0]][k]), own_contra(J[J2[1]][k])) for k in keys}
    old = {k: {"INCORRECT": "YES", "CORRECT": "NO"}.get(OL.get(k, "NONCONS"), "NC") for k in keys}
    vt_df = {k: all(J[j][k]["depth_true"] == "FALSE" and J[j][k]["vertical_true"] == "TRUE" for j in J2) for k in keys}
    agree_read = np.mean([J[J2[0]][k]["reading"] == J[J2[1]][k]["reading"] for k in keys])
    res = {"n": len(keys), "reading_agreement": round(100 * float(agree_read), 1),
           "unparsed": {j: sum("UNPARSED" in (J[j][k]["reading"], J[j][k]["depth_true"], J[j][k]["vertical_true"]) for k in keys) for j in J2},
           "api_errors": {j: sum(J[j][k]["api_error"] for k in keys) for j in J2}, "rows": {}}
    groups = []
    for src in ("e28_commit_cues", "e28_commit_twins", "e28_prej_cues", "e20_syn"):
        ms = sorted({k[1] for k in keys if k[0] == src})
        for g in sorted({k[3] for k in keys if k[0] == src}):
            for m in ms + (["pooled_comp"] if src.startswith("e28") else ["pooled_comp", "pooled7"]):
                groups.append((src, g, m))
    for src, g, m in groups:
        ks = [k for k in keys if k[0] == src and k[3] == g and (k[1] == m or (m == "pooled_comp" and k[1] in COMP) or m == "pooled7")]
        if not ks:
            continue
        top = [k for k in ks if is_top[k]]
        notop = [k for k in ks if not is_top[k]]
        row = {"n": len(ks), "n_ontop": len(top), "ontop_share": round(100 * len(top) / len(ks), 1),
               "old": rate2([old[k] for k in ks]), "old_without_ontop": rate2([old[k] for k in notop]), "old_ontop_only": rate2([old[k] for k in top]),
               "new_depth": rate2([dc[k] for k in ks]), "new_depth_without_ontop": rate2([dc[k] for k in notop]),
               "new_own": rate2([oc[k] for k in ks]), "new_own_without_ontop": rate2([oc[k] for k in notop]),
               "new_own_ontop_only": rate2([oc[k] for k in top]), "new_depth_ontop_only": rate2([dc[k] for k in top]),
               "reading": dict(Counter(reading[k] for k in ks)), "reading_ontop": dict(Counter(reading[k] for k in top)),
               "vertical_true_depth_false": pct(sum(vt_df[k] for k in ks), len(ks)),
               "vertical_true_depth_false_ontop": pct(sum(vt_df[k] for k in top), len(top))}
        for rd in ("DEPTH", "VERTICAL", "AMBIGUOUS", "BOTH"):
            sub = [k for k in ks if reading[k] == rd]
            if sub:
                row[f"by_reading_{rd}"] = {"n": len(sub), "own": rate2([oc[k] for k in sub]), "depth": rate2([dc[k] for k in sub])}
        res["rows"][f"{src}|{g}|{m}"] = row
        f = lambda x: f"{x['k']}/{x['n']} = {x['pct']}%" if x["n"] else "--"
        lines.append(f"ONTOP {src:16s} {g:8s} {m:12s} n={row['n']:3d} on-top {row['n_ontop']} ({row['ontop_share']}%) | OLD {f(row['old'])} "
                     f"w/o on-top {f(row['old_without_ontop'])} on-top only {f(row['old_ontop_only'])} | NEW depth-reading {f(row['new_depth'])} "
                     f"w/o on-top {f(row['new_depth_without_ontop'])} | NEW own-reading {f(row['new_own'])} {row['new_own']['wilson']} "
                     f"w/o on-top {f(row['new_own_without_ontop'])} on-top only {f(row['new_own_ontop_only'])} | reading {row['reading']} "
                     f"| vertical-true & depth-false {row['vertical_true_depth_false']['k']} (on-top {row['vertical_true_depth_false_ontop']['k']}/{len(top)})")
    lines.append(f"ONTOP reading agreement {res['reading_agreement']}% n={res['n']} unparsed {res['unparsed']} api errors {res['api_errors']}")
    return res


# ============================================================================= Task 5: descriptions

def claimed_role(fc: str, front: str, back: str) -> str:
    fc = fc.lower()
    if fc in ("none", "unparsed") or fc.startswith("none"):
        return "none"
    if fc.startswith("mixed"):
        return "mixed"
    hf, hb = front.lower() in fc, back.lower() in fc
    if hf and not hb:
        return "front"
    if hb and not hf:
        return "back"
    # colour-only or shape-only mention
    cf, sf = front.lower().split()
    cb, sb = back.lower().split()
    mf = (cf in fc) + (sf in fc)
    mb = (cb in fc) + (sb in fc)
    return "front" if mf > mb else "back" if mb > mf else "unclear"


def implied_answer(role: str, it_meta: dict, phrasing: str) -> str | None:
    """Answer letter implied by a claimed front shape ('front' = true front)."""
    if role not in ("front", "back"):
        return None
    claimed_front = it_meta["front_name"] if role == "front" else it_meta["back_name"]
    x = it_meta["x_name"]
    yes = (claimed_front == x) if phrasing == "front" else (claimed_front != x)
    return "A" if yes else "B"


def desc(lines: list[str]) -> dict:
    J = {j: {(r["model"], r["kind"], r["item_id"]): r for r in rj(E29 / f"judge_{j}_desc.jsonl")} for j in J2}
    if not J[J2[0]] or not J[J2[1]]:
        lines.append("DESC: judge files missing")
        return {}
    keys = sorted(set(J[J2[0]]) & set(J[J2[1]]))
    res = {"n": len(keys), "rows": {}}
    cons = {}
    for k in keys:
        a, b = J[J2[0]][k], J[J2[1]][k]
        la, lb = a["depth_claim"], b["depth_claim"]
        ra = claimed_role(a["front_claimed"], a["front_name"], a["back_name"])
        rb = claimed_role(b["front_claimed"], b["front_name"], b["back_name"])
        cons[k] = {"claim": la if la == lb else "NC", "role": ra if ra == rb else "NC", "ontop_only": a["on_top_only"] == b["on_top_only"] == "YES",
                   "any_incorrect": "INCORRECT" in (la, lb) or "MIXED" in (la, lb)}
    res["agree_claim"] = round(100 * float(np.mean([J[J2[0]][k]["depth_claim"] == J[J2[1]][k]["depth_claim"] for k in keys])), 1)
    for m in COMP:
        Dm = {r["item_id"]: r for r in rj(E28 / f"{m}_cues_desc.jsonl") if r["condition"] == "desc"}
        Dr = {r["item_id"]: r for r in rj(O / "e10" / f"{m}_cues.jsonl") if r["condition"] == "direct"}
        Cr = {r["item_id"]: r for r in rj(O / "e10" / f"{m}_cues.jsonl") if r["condition"] == "cot"}
        for kind, strata in (("desc", ("desc_err", "desc_ok")), ("cot", ("D+C-", "D+C+rel"))):
            for s in strata:
                ks = [k for k in keys if k[0] == m and k[1] == kind and J[J2[1]][k]["stratum"] == s]
                if not ks:
                    continue
                labs = [cons[k]["claim"] for k in ks]
                n_inc, n_cor = labs.count("INCORRECT"), labs.count("CORRECT")
                row = {"n": len(ks), "claim_counts": dict(Counter(labs)), "incorrect_of_correct_or_incorrect": pct(n_inc, n_inc + n_cor),
                       "incorrect_of_all": pct(n_inc, len(ks)), "noclaim_of_all": pct(labs.count("NOCLAIM"), len(ks)),
                       "ontop_only": sum(cons[k]["ontop_only"] for k in ks)}
                # answers vs implied answers (items = image x phrasing for desc; the rationale's own item for CoT)
                fol, n_def, fol_err, n_err_def, err_false, err_true, err_none, err_n = 0, 0, 0, 0, 0, 0, 0, 0
                for k in ks:
                    role = cons[k]["role"]
                    if kind == "desc":
                        its = [Dm[i] for i in (k[2], k[2] + "_behind") if i in Dm]
                    else:
                        its = [Cr[k[2]] | {"phrasing": "behind" if k[2].endswith("_behind") else "front"}]
                    for it in its:
                        ans = pe(it)
                        wrong = ans != it["gold"]
                        d_ok = pe(Dr[it["item_id"]]) == it["gold"]
                        ia = implied_answer(role, it["meta"], it["phrasing"])
                        if ia is not None:
                            n_def += 1
                            fol += ia == ans
                        if wrong and d_ok:
                            err_n += 1
                            if ia is not None:
                                n_err_def += 1
                                fol_err += ia == ans
                            err_false += role == "back"
                            err_true += role == "front"
                            err_none += role not in ("front", "back")
                row["answer_follows_claim"] = pct(fol, n_def)
                row["direct_right_arm_wrong_items"] = err_n
                row["errors_answer_follows_claim"] = pct(fol_err, n_err_def)
                row["errors_with_false_claim"] = pct(err_false, err_n)
                row["errors_with_true_claim"] = pct(err_true, err_n)
                row["errors_without_claim_or_unclear"] = pct(err_none, err_n)
                res["rows"][f"{m}|{kind}|{s}"] = row
                f = lambda x: f"{x['k']}/{x['n']} = {x['pct']}% {x['wilson']}" if x["n"] else "--"
                lines.append(f"DESC {m:11s} {kind:4s} {s:8s} n={len(ks)} | consensus INCORRECT / judgeable {f(row['incorrect_of_correct_or_incorrect'])} "
                             f"of all {row['incorrect_of_all']['pct']}% (NOCLAIM {row['noclaim_of_all']['pct']}%) {row['claim_counts']} | answer follows the "
                             f"claimed front {f(row['answer_follows_claim'])} | Direct-right/arm-wrong items {err_n}: claim false {f(row['errors_with_false_claim'])}, "
                             f"claim true {row['errors_with_true_claim']['pct']}%, none/unclear {row['errors_without_claim_or_unclear']['pct']}%, answer follows claim "
                             f"{f(row['errors_answer_follows_claim'])}")
        # stratum weights -> share of all descriptions with a false depth claim over images in the two strata
        r_e, r_o = res["rows"].get(f"{m}|desc|desc_err"), res["rows"].get(f"{m}|desc|desc_ok")
        if r_e and r_o:
            ex = next(J[J2[1]][k] for k in keys if k[0] == m and k[1] == "desc")
            ne, no, nall = ex["n_err_strata"], ex["n_ok_strata"], ex["n_images"]
            w = (ne * r_e["incorrect_of_all"]["pct"] + no * r_o["incorrect_of_all"]["pct"]) / (ne + no)
            res["rows"][f"{m}|desc|weighted"] = {"n_err_images": ne, "n_ok_images": no, "n_images": nall,
                                                "incorrect_of_all_weighted": round(w, 1)}
            lines.append(f"DESC {m:11s} weighted over {ne} error + {no} all-right images (of {nall}): {w:.1f}% of descriptions carry a consensus false depth claim")
    lines.append(f"DESC judge agreement on DEPTH_CLAIM {res['agree_claim']}% (n={res['n']})")
    return res


# ============================================================================= Task 4: reimage

def reimage(lines: list[str], nb_glm: int) -> dict:
    from rigor_e28 import glm_arms
    R = [r for p in sorted(E29.glob("molmo7b_cues_reimage_hf_shard*.jsonl")) for r in rj(p)]
    if not R:
        lines.append("REIMAGE: missing")
        return {}
    rng = np.random.default_rng(32)
    conds = ["direct", "cot", "cot_turn", "cot_reimage"]
    by = {c: {r["item_id"]: r for r in R if r["condition"] == c} for c in conds}
    keys = sorted(set.intersection(*[set(v) for v in by.values()]))
    S = {c: {r["item_id"]: r for r in rj(O / "e10" / "molmo7b_cues.jsonl") if r["condition"] == c} for c in ("direct", "cot")}
    g = [by["direct"][k]["image_id"] for k in keys]
    acc = {c: np.array([float(pe(by[c][k]) == by[c][k]["gold"]) for k in keys]) for c in conds}
    res = {"n": len(keys), "agree_vllm_direct": round(100 * float(np.mean([pe(by['direct'][k]) == pe(S['direct'][k]) for k in keys])), 2),
           "agree_vllm_cot": round(100 * float(np.mean([pe(by['cot'][k]) == pe(S['cot'][k]) for k in keys])), 2),
           "vllm_acc_direct": round(100 * float(np.mean([pe(S['direct'][k]) == S['direct'][k]['gold'] for k in keys])), 1),
           "vllm_acc_cot": round(100 * float(np.mean([pe(S['cot'][k]) == S['cot'][k]['gold'] for k in keys])), 1),
           "mean_img_tokens": {c: round(float(np.mean([by[c][k]["n_img_tokens"] for k in keys])), 0) for c in conds}}
    for c in conds:
        res[f"acc_{c}"] = cboot(acc[c], g, rng)
    for a1, a0 in (("cot", "direct"), ("cot_turn", "direct"), ("cot_reimage", "direct"), ("cot_reimage", "cot"), ("cot_reimage", "cot_turn"), ("cot_turn", "cot")):
        res[f"acc_{a1}-{a0}"] = cboot(acc[a1] - acc[a0], g, rng)
    for k in keys:
        for c in conds:
            by[c][k].setdefault("question", S["cot"][k]["question"])
    G = glm_arms({"direct": [by["direct"][k] for k in keys], "cot": [by["cot"][k] for k in keys],
                  "desc": [by["cot_reimage"][k] for k in keys]}, rng, nb_glm)
    G2 = glm_arms({"direct": [by["direct"][k] for k in keys], "cot": [by["cot_turn"][k] for k in keys],
                   "desc": [by["cot_reimage"][k] for k in keys]}, rng, nb_glm)
    res["glm"] = {"direct": G["direct"], "cot": G["cot"], "cot_turn": G2["cot"], "cot_reimage": G["desc"],
                  "cot-direct": G["cot-direct"], "cot_reimage-direct": G["desc-direct"], "cot_reimage-cot": G["desc-cot"],
                  "cot_turn-direct": G2["cot-direct"], "cot_reimage-cot_turn": G2["desc-cot"]}
    f = lambda x: f"{x['d']:+.2f} [{x['ci'][0]:+.2f}, {x['ci'][1]:+.2f}]"
    gl = res["glm"]
    lines.append(f"REIMAGE molmo7b cues HF n={res['n']} | sanity: HF Direct = vLLM Direct {res['agree_vllm_direct']}%, HF CoT = vLLM CoT {res['agree_vllm_cot']}% "
                 f"(vLLM acc {res['vllm_acc_direct']} / {res['vllm_acc_cot']}) | image tokens {res['mean_img_tokens']}")
    lines.append("   acc " + " | ".join(f"{c} {fm(res[f'acc_{c}'])}" for c in conds))
    lines.append("   diff " + " | ".join(f"{a} {fa(res[f'acc_{a}'])}" for a in ("cot-direct", "cot_turn-direct", "cot_reimage-direct", "cot_reimage-cot", "cot_reimage-cot_turn")))
    lines.append(f"   occlusion weight Direct {gl['direct']['occlusion']['d']:.2f} CoT {gl['cot']['occlusion']['d']:.2f} CoT+turn {gl['cot_turn']['occlusion']['d']:.2f} "
                 f"CoT+reimage {gl['cot_reimage']['occlusion']['d']:.2f} | CoT-Dir {f(gl['cot-direct']['occlusion'])} reimage-Dir {f(gl['cot_reimage-direct']['occlusion'])} "
                 f"reimage-CoT {f(gl['cot_reimage-cot']['occlusion'])} reimage-turn {f(gl['cot_reimage-cot_turn']['occlusion'])} | vertical CoT-Dir {f(gl['cot-direct']['vertical'])} "
                 f"reimage-Dir {f(gl['cot_reimage-direct']['vertical'])}")
    return res


# ============================================================================= table

def table(R: dict) -> str:
    fx = lambda x: f"{x['mean']:.1f}"
    fd = lambda x: f"${x['mean']:+.1f}$ {{\\scriptsize$[{x['ci'][0]:+.1f}, {x['ci'][1]:+.1f}]$}}"
    L = [r"\begin{table*}[t]", r"\centering", r"\scriptsize", r"\setlength{\tabcolsep}{3pt}",
         r"\caption{\textbf{Phrasing-congruent claim edits, edits at the earliest contradicting sentence, and CoT+reimage} (cue-factorial set). "
         r"(a) Accuracy (\%) after replacing the commit sentence $j^\ast$ of \texttt{D+C$-$} rationales and regenerating, by question predicate. Orig.: regeneration from the unedited sentence. "
         r"Neutral: ``Let me look at the two shapes in the image carefully.''; F0: ``The $\langle$front$\rangle$ is in front of the $\langle$back$\rangle$, covering part of it.'' (\cref{tab:supp_e28}); "
         r"F1/B1: ``Looking at the two shapes carefully, the $\langle$front$\rangle$ is in front of the $\langle$back$\rangle$.'' / ``\dots the $\langle$back$\rangle$ is behind the $\langle$front$\rangle$.''; "
         r"B0: ``The $\langle$back$\rangle$ is behind the $\langle$front$\rangle$, partly hidden by it.''; F2/B2: the model's most frequent true ``in front of'' / ``behind'' sentence in its own \texttt{D+C$+$} rationales "
         r"(``The $\langle$front$\rangle$ is positioned in front of the $\langle$back$\rangle$.'' / ``The $\langle$back$\rangle$ is positioned behind the $\langle$front$\rangle$.''); "
         r"Echo: the question's own subject, predicate and object with the true polarity. Bracketed: congruent template (F on ``in front of'', B on ``behind'' questions) minus Neutral, "
         r"$95\%$ image-cluster bootstrap CI. \texttt{D+C$+$} rows: control, the first claim of $100$ correct rationales per predicate replaced the same way. Orig., Neutral, and F0 were re-run with the other arms (\texttt{Molmo-7B-D} cue-set Neutral $55.9\%$ here, $56.7\%$ in \cref{tab:supp_e28}). "
         r"(b) \texttt{Qwen3-VL-8B}: rationales whose earliest two-judge image-contradicting relational sentence precedes $j^\ast$, edited there or at $j^\ast$. "
         r"(c) \texttt{Molmo-7B-D}, $1{,}600$ cue-set queries, stored CoT rationales read out with HuggingFace (one engine for all four arms; HF \texttt{Direct} matches \texttt{vLLM} \texttt{Direct} on $95.8\%$ of queries, accuracy $87.1\%$ vs.\ $88.2\%$).}",
         r"\label{tab:supp_e29}"]
    L += [r"\resizebox{\textwidth}{!}{%", r"\begin{tabular}{llccccccccccc}", r"\toprule",
          r"(a) & Question & $n$ & Orig. & Neutral & F0 & F1 & F2 & B0 & B1 & B2 & Echo & Congruent$-$Neutral (F0/B0) \\", r"\midrule"]
    E = R.get("edit", {})
    for m in COMP:
        for st in ("cues", "twins"):
            e = E.get(f"{m}|{st}|jstar")
            if not e:
                continue
            for ph in ("front", "behind"):
                r = e[ph]
                cg = r["F0-neutral"] if ph == "front" else r["B0-neutral"]
                nm = NAMES[m] + (r" (twins)" if st == "twins" else "")
                L.append(f"{nm} & {'in front of' if ph == 'front' else 'behind'} & {r['n']} & " + " & ".join(fx(r[f'acc_{a}']) for a in ARMS)
                         + f" & {fd(cg)} \\\\")
    for m in COMP:
        e = E.get(f"{m}|cues|dcp")
        if e:
            for ph in ("front", "behind"):
                r = e[ph]
                cg = r["F0-neutral"] if ph == "front" else r["B0-neutral"]
                L.append(f"{NAMES[m]} \\texttt{{D+C$+$}} (first claim) & {'in front of' if ph == 'front' else 'behind'} & {r['n']} & "
                         + " & ".join(fx(r[f'acc_{a}']) for a in ARMS) + f" & {fd(cg)} \\\\")
    L += [r"\bottomrule", r"\end{tabular}}"]
    ea = R.get("early", {})
    if ea:
        L += [r"\\[4pt]", r"\begin{tabular}{lcccccc}", r"\toprule",
              r"(b) \texttt{Qwen3-VL-8B} & $n$ & Orig. & Neutral & F0 & Congruent (F0/B0) & Echo \\", r"\midrule",
              f"Edit at earliest contradicting sentence & {ea['n']} & {fx(ea['early_orig'])} & {fx(ea['early_neutral'])} & {fx(ea['early_F0'])} & {fx(ea['early_cong'])} & {fx(ea['early_ECHO'])} \\\\",
              f"Edit at $j^\\ast$ (same items) & {ea['n']} & {fx(ea['jstar_orig'])} & {fx(ea['jstar_neutral'])} & {fx(ea['jstar_F0'])} & {fx(ea['jstar_cong'])} & {fx(ea['jstar_ECHO'])} \\\\",
              f"Difference & & {fd(ea['early-jstar_orig'])} & {fd(ea['early-jstar_neutral'])} & {fd(ea['early-jstar_F0'])} & {fd(ea['early-jstar_cong'])} & {fd(ea['early-jstar_ECHO'])} \\\\",
              r"\bottomrule", r"\end{tabular}"]
    ri = R.get("reimage", {})
    if ri:
        g = ri["glm"]
        f2 = lambda x: f"${x['d']:+.2f}$ {{\\scriptsize$[{x['ci'][0]:+.2f}, {x['ci'][1]:+.2f}]$}}"
        L += [r"\hfill", r"\begin{tabular}{lcccc}", r"\toprule", r"(c) \texttt{Molmo-7B-D} & Direct & CoT & CoT+turn & CoT+reimage \\", r"\midrule",
              f"Accuracy & {fx(ri['acc_direct'])} & {fx(ri['acc_cot'])} & {fx(ri['acc_cot_turn'])} & {fx(ri['acc_cot_reimage'])} \\\\",
              f"$-$Direct & & {fd(ri['acc_cot-direct'])} & {fd(ri['acc_cot_turn-direct'])} & {fd(ri['acc_cot_reimage-direct'])} \\\\",
              f"Occlusion $d'$ & {g['direct']['occlusion']['d']:.2f} & {g['cot']['occlusion']['d']:.2f} & {g['cot_turn']['occlusion']['d']:.2f} & {g['cot_reimage']['occlusion']['d']:.2f} \\\\",
              f"$\\Delta$occl.\\ $-$Direct & & {f2(g['cot-direct']['occlusion'])} & {f2(g['cot_turn-direct']['occlusion'])} & {f2(g['cot_reimage-direct']['occlusion'])} \\\\",
              r"\bottomrule", r"\end{tabular}"]
    L += [r"\end{table*}"]
    ot, ds = R.get("ontop", {}).get("rows"), R.get("desc", {}).get("rows")
    if ot or ds:
        fp = lambda c: f"{c['k']}/{c['n']} ({c['pct']:.1f})" if c["n"] else "--"
        L += ["", r"\begin{table*}[t]", r"\centering", r"\scriptsize", r"\setlength{\tabcolsep}{3pt}",
              r"\caption{\textbf{``On top'' re-judging and the description-arm audit.} "
              r"(d) Sentences re-judged with a prompt that separates a depth/layering reading (in front, behind, covering) from a vertical reading (higher, lower), asks which reading the sentence uses "
              r"(depth, vertical, both, or ambiguous) and whether it is true under each. Two-judge consensus contradicting / judgeable: Orig.\ = the original prompt (\cref{tab:supp_e28}, \cref{tab:supp_commit_judge}); "
              r"Depth = false as a depth claim (vertical-only sentences excluded); Own reading = false under the reading the sentence uses (ambiguous sentences count as false only if false under both readings); "
              r"``w/o on top'' drops sentences containing ``on top''/``atop''; V-true/D-false = both judges mark the sentence true as a vertical claim and false as a depth claim. Syn.\ Depth rows: the all-$500$-image sample of \cref{tab:supp_commit_judge}, not the $431$ overlapping images. "
              r"(e) Full texts judged for their depth claim (which shape is in front): question-blind descriptions of images where the description arm answered a Direct-correct item wrongly (Desc.\ err.) "
              r"or both phrasings correctly (Desc.\ ok), and full CoT rationales of the \texttt{D+C$-$} / relational \texttt{D+C$+$} commit samples. "
              r"Errors: on \texttt{Direct}-correct items that the arm answered wrongly, the share whose text states a false depth claim / whose answer follows the text's claimed front shape (texts with a claim).}",
              r"\label{tab:supp_e29b}"]
    if ot:
        L += [r"\resizebox{\textwidth}{!}{%", r"\begin{tabular}{lllcccccccc}", r"\toprule",
              r"(d) Set & Sentences & Model & $n$ & On top (\%) & Orig. & Orig.\ w/o on top & Depth & Own reading & Own w/o on top & V-true/D-false \\", r"\midrule"]
        sets = [("e28_commit_cues", "Cue-fact."), ("e28_commit_twins", "Twins"), ("e28_prej_cues", r"Cue-fact.\ pre-$j^\ast$"), ("e20_syn", r"Syn.\ Depth")]
        for src, sn in sets:
            for g, gn in (("D+C-", r"\texttt{D+C$-$} $j^\ast$"), ("D+C+rel", r"rel.\ \texttt{D+C$+$}"), ("pre_jstar", "relational")):
                for m in COMP + ["pooled7"]:
                    r = ot.get(f"{src}|{g}|{m}")
                    if not r:
                        continue
                    mn = {"pooled7": "seven models"}.get(m, NAMES.get(m, m))
                    L.append(f"{sn} & {gn} & {mn} & {r['n']} & {r['ontop_share']:.1f} & {fp(r['old'])} & {fp(r['old_without_ontop'])} & {fp(r['new_depth'])} & "
                             f"{fp(r['new_own'])} & {fp(r['new_own_without_ontop'])} & {r['vertical_true_depth_false']['k']} \\\\")
        L += [r"\bottomrule", r"\end{tabular}}"]
    if ds:
        L += [r"\\[4pt]", r"\begin{tabular}{llcccc}", r"\toprule",
              r"(e) Model & Texts & $n$ & False depth claim / judgeable & No depth claim (\%) & Errors: claim false / follows \\", r"\midrule"]
        for m in COMP:
            for kind, s, nm in (("desc", "desc_err", "Desc.\\ err."), ("desc", "desc_ok", "Desc.\\ ok"), ("cot", "D+C-", r"CoT \texttt{D+C$-$}"), ("cot", "D+C+rel", r"CoT rel.\ \texttt{D+C$+$}")):
                r = ds.get(f"{m}|{kind}|{s}")
                if not r:
                    continue
                L.append(f"{NAMES[m]} & {nm} & {r['n']} & {fp(r['incorrect_of_correct_or_incorrect'])} & {r['noclaim_of_all']['pct']:.1f} & "
                         f"{fp(r['errors_with_false_claim'])} / {fp(r['errors_answer_follows_claim'])} \\\\")
        L += [r"\bottomrule", r"\end{tabular}"]
    if ot or ds:
        L += [r"\end{table*}"]
    return "\n".join(L) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="edit,early,ontop,desc,reimage")
    ap.add_argument("--earliest-only", action="store_true")
    ap.add_argument("--nb-glm", type=int, default=1000)
    a = ap.parse_args()
    if a.earliest_only:
        lines: list[str] = []
        earliest(lines)
        print("\n".join(lines))
        return
    names = {"edit": "e29_edit.json", "early": "e29_early.json", "ontop": "e29_ontop.json", "desc": "e29_desc.json", "reimage": "e29_reimage.json"}
    sec = E29 / "rigor_sections"
    sec.mkdir(exist_ok=True)
    for k in a.only.split(","):
        lines = []
        if k == "early":
            ei = earliest(lines)
            res = early(lines, ei) | {"earliest": ei}
        else:
            res = {"edit": edit, "ontop": ontop, "desc": desc}[k](lines) if k != "reimage" else reimage(lines, a.nb_glm)
        json.dump(res, open(B / names[k], "w"), indent=1)
        (sec / f"{k}.txt").write_text("\n".join(lines) + "\n")
        print("\n".join(lines))
    (B / "rigor_e29.txt").write_text("".join((sec / f"{k}.txt").read_text() for k in names if (sec / f"{k}.txt").exists()))
    allj = {k: json.load(open(B / fn)) for k, fn in names.items() if (B / fn).exists()}
    tex = table(allj)
    (B / "supp_e29.tex").write_text(tex)
    shutil.copy(B / "supp_e29.tex", PAPER_TAB / "supp_e29.tex")


if __name__ == "__main__":
    main()
