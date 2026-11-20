"""Exp-61 analysis: closed models with their own truncated rationale held fixed, image swapped at the answer step.

-> outputs/bundle/e61_closed_fixed_rcut.json, outputs/bundle/rigor_e61.txt, paper/tables/supp_e61.tex
Image effect IE = P(answer = gold_v | v) - P(answer = gold_v | v_cf); failed / letter-free replies count as not gold_v.
"""
from __future__ import annotations
import vtax_paths as _vtp  # noqa: E402

import json
import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from e50_analyze import NAMES, implied_letter
from e50_closed_image_blind import O, rj
from e61_fixed_rcut import E61, MAIN, NR_EFFORT_E9H, OPTIONAL, VARIANTS, load_items

B = Path("outputs/bundle")
PAPER = Path(f"{_vtp.PAPER}")
NBOOT = 2000
SETS = ("hard", "cues")
SETN = {"hard": "Hard occl.", "cues": "Cue set"}


def pct(v) -> float:
    return 100.0 * float(np.mean(v)) if len(v) else float("nan")


def boot(groups: list, fn, n: int = NBOOT, seed: int = 61) -> list[float]:
    rng = np.random.default_rng(seed)
    u = sorted(set(groups))
    vals = []
    for _ in range(n):
        v = fn([u[i] for i in rng.choice(len(u), len(u), replace=True)])
        if v is not None and not np.isnan(v):
            vals.append(v)
    return [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))] if vals else [float("nan")] * 2


LASTLINE = re.compile(r"^\W*\(?\s*([AB])\s*\)?\W*$", re.M)
BOLD = re.compile(r"\*\*\s*\(?([AB])\)?\s*\*\*")
CORRECTION = re.compile(r"\bcorrect(?:ion|ing)?\b|\bearlier\b|\bpreviously\b|\bmistake\b|\bI (?:had|swapped|misread|described)|"
                        r"\bwas wrong\b|\bactually\b|\bre-?examin|\blooking again\b|\bon closer\b", re.I)


def parse(text: str) -> str | None:
    """e9x parse_letter, then a letter alone on a line or in bold (the last-line misses noted in Exp-38)."""
    from e9x_api import parse_letter
    p = parse_letter(text, ["A", "B"])
    if p:
        return p
    m = LASTLINE.findall(text or "") or BOLD.findall(text or "")
    return m[-1] if m else None


def latest(fn: Path) -> dict:
    out = {}
    for r in rj(fn):
        k = (r["kind"], r["set"], r["uid"], r["arm"], r["bh"])
        if "error" not in r or k not in out or "error" in out[k]:
            out[k] = r
    for r in out.values():
        if "error" not in r:
            r["pred_e9x"] = r.get("pred")
            r["pred"] = parse(r.get("text", ""))
            r["correction"] = bool(CORRECTION.search(r.get("text", "")))
    return out


def e9h_direct(model: str, cond: str) -> dict:
    d = {}
    for r in rj(O / "e9h" / f"{model}.jsonl"):
        if r["condition"] == cond and "error" not in r and (r.get("pred") or r["item_id"] not in d):
            d[r["item_id"]] = r
    return d


def analyze() -> None:
    e50 = json.load(open(B / "e50_closed_image_blind.json"))
    E50R = {(r["model"], r["set"], r["cond"], r["uid"]): r for r in e50["rows"]}
    models = [m for m in MAIN + OPTIONAL if (E61 / f"gen_{m}.jsonl").exists()]
    res = {"cells": {}, "split": {}, "sanity": {}, "reasoning": {}, "failures": {}, "calls": {}}
    rows = []
    for m in models:
        G = latest(E61 / f"gen_{m}.jsonl")
        allrecs = rj(E61 / f"gen_{m}.jsonl")
        res["calls"][m] = {"attempts": len(allrecs), "errors_all_attempts": sum("error" in r for r in allrecs),
                           "final_errors": sum("error" in r for r in G.values()),
                           "by_kind": dict(Counter(r["kind"] for r in G.values()))}
        dnr = e9h_direct(m, "direct_nr") if m in NR_EFFORT_E9H else {}
        dhi = e9h_direct(m, "direct")
        for it in load_items(m):
            st, uid = it["set"], it["uid"]
            row = {"model": m, "set": st, "uid": uid, "group": it["group"], "gold": it["gold"],
                   "phrasing": it["phrasing"], "x": it["x"], "y": it["y"], "real_front": it["real_front"],
                   "stored_pred": it["stored_pred"]}
            for v in VARIANTS:
                for arm in ("v", "vcf"):
                    r = G.get(("rcut", st, uid, arm, it[f"bh_{v}"])) if it[f"body_{v}"].strip() else None
                    row[f"{v}_{arm}"] = r.get("pred") if r and "error" not in r else None
                    row[f"{v}_{arm}_ok"] = bool(r and "error" not in r)
                    row[f"{v}_{arm}_rt"] = (r or {}).get("reasoning_tokens")
                    row[f"{v}_{arm}_corr"] = bool(r and r.get("correction"))
                    row[f"{v}_{arm}_chars"] = len((r or {}).get("text", ""))
                row[f"{v}_has_body"] = bool(it[f"body_{v}"].strip())
            for v in ("kept", "removed"):
                rel = it[f"rel_{v}"]
                row[f"rel_{v}"] = rel
                row[f"relcls_{v}"] = "none" if rel is None else ("true" if rel == it["real_front"] else "false")
                row[f"impl_{v}"] = implied_letter(rel, it["x"], it["y"], it["phrasing"])
            # Direct-low
            for arm in ("v", "vcf"):
                r = G.get(("direct", st, uid, arm, ""))
                if r is not None:
                    row[f"dlow_{arm}"], row[f"dlow_src"] = (r.get("pred") if "error" not in r else None), "e61"
                    row[f"dlow_{arm}_rt"] = r.get("reasoning_tokens")
                elif st == "hard" and dnr:
                    rr = dnr.get(uid if arm == "v" else it["twin_uid"])
                    row[f"dlow_{arm}"], row["dlow_src"] = (rr or {}).get("pred"), "e9h_direct_nr"
                    row[f"dlow_{arm}_rt"] = (rr or {}).get("reasoning_tokens")
                else:
                    row[f"dlow_{arm}"] = None
                    row.setdefault("dlow_src", None)
                if st == "hard":
                    rr = dhi.get(uid if arm == "v" else it["twin_uid"])
                    row[f"dhigh_{arm}"] = (rr or {}).get("pred")
            for arm, cond in (("v", "real"), ("vcf", "twin")):
                rr = E50R.get((m, st, cond, uid))
                row[f"regen_{arm}"] = (rr or {}).get("pred")
            rows.append(row)

    def ie(rs, a, b):
        return pct([r[a] == r["gold"] for r in rs]) - pct([r[b] == r["gold"] for r in rs])

    for m in models:
        for st in SETS:
            rs = [r for r in rows if r["model"] == m and r["set"] == st]
            if not rs:
                continue
            byg = defaultdict(list)
            for r in rs:
                byg[r["group"]].append(r)
            gs = list(byg)

            def sub(g, cond=lambda r: True):
                return [r for x in g for r in byg[x] if cond(r)]

            c = {"n": len(rs)}
            arms = {"regen": ("regen_v", "regen_vcf"), "dlow": ("dlow_v", "dlow_vcf")}
            if st == "hard":
                arms["dhigh"] = ("dhigh_v", "dhigh_vcf")
            for v in VARIANTS:
                arms[v] = (f"{v}_v", f"{v}_vcf")
            for name, (a, b) in arms.items():
                has = (lambda r, v=name: r[f"{v}_has_body"]) if name in VARIANTS else (lambda r: True)
                if all(r.get(a) is None and r.get(b) is None for r in rs):
                    continue
                c[name] = {"ie": ie(sub(gs, has), a, b), "ci": boot(gs, lambda g, a=a, b=b, has=has: ie(sub(g, has), a, b)),
                           "acc_v": pct([r[a] == r["gold"] for r in sub(gs, has)]),
                           "acc_vcf_vs_gold_v": pct([r[b] == r["gold"] for r in sub(gs, has)]),
                           "n": len(sub(gs, has)),
                           "missing_v": sum(r.get(a) is None for r in sub(gs, has)),
                           "missing_vcf": sum(r.get(b) is None for r in sub(gs, has))}
                if name in VARIANTS:
                    c[name]["answer_changed"] = pct([r[a] != r[b] for r in sub(gs, has)])
                    c[name]["correction_text_v"] = pct([r[f"{a}_corr"] for r in sub(gs, has)])
                    c[name]["correction_text_vcf"] = pct([r[f"{b}_corr"] for r in sub(gs, has)])
                    c[name]["longer_than_letter_vcf"] = pct([r[f"{b}_chars"] > 20 for r in sub(gs, has)])
                    vb = [r for r in sub(gs, has) if r[a] and r[b]]
                    c[name]["ie_valid"] = ie(vb, a, b)
                    c[name]["n_valid"] = len(vb)
            # paired differences, primary variant
            for v in VARIANTS:
                if v not in c:
                    continue
                for ref in ("regen", "dlow", "dhigh"):
                    if ref not in c:
                        continue
                    a, b = arms[ref]
                    fa, fb = arms[v]
                    has = lambda r, v=v: r[f"{v}_has_body"]
                    fn = lambda g, a=a, b=b, fa=fa, fb=fb, has=has: ie(sub(g, has), a, b) - ie(sub(g, has), fa, fb)
                    c[f"{ref}_minus_{v}"] = {"d": fn(gs), "ci": boot(gs, fn)}
            res["cells"][f"{m}|{st}"] = c
            # split by the last written relation (Exp-50 two-judge consensus, same body)
            for v in ("removed", "kept"):
                for cls in ("true", "false", "none"):
                    sel = lambda r, v=v, cls=cls: r[f"relcls_{v}"] == cls
                    srs = sub(gs, sel)
                    d = {"n": len(srs)}
                    if srs:
                        a, b = f"{v}_v", f"{v}_vcf"
                        d["ie"] = ie(srs, a, b)
                        d["ci"] = boot([r["group"] for r in srs], lambda g, a=a, b=b, sel=sel: ie(sub(g, sel), a, b))
                        d["acc_v"] = pct([r[a] == r["gold"] for r in srs])
                        d["acc_vcf_vs_gold_v"] = pct([r[b] == r["gold"] for r in srs])
                        if cls != "none":
                            d["agree_v"] = pct([r[a] == r[f"impl_{v}"] for r in srs])
                            d["agree_vcf"] = pct([r[b] == r[f"impl_{v}"] for r in srs])
                        d["regen_ie"] = ie(srs, "regen_v", "regen_vcf")
                        d["dlow_ie"] = ie(srs, "dlow_v", "dlow_vcf") if any(r.get("dlow_v") for r in srs) else None
                    res["split"][f"{m}|{st}|{v}|{cls}"] = d
            # sanity: v-arm reproduces the stored answer of R under v
            for v in VARIANTS:
                sr = [r for r in rs if r["stored_pred"] and r[f"{v}_has_body"]]
                res["sanity"][f"{m}|{st}|{v}"] = {"n": len(sr), "v_eq_stored": pct([r[f"{v}_v"] == r["stored_pred"] for r in sr]),
                                                  "vcf_eq_stored": pct([r[f"{v}_vcf"] == r["stored_pred"] for r in sr])}
            # reasoning tokens
            for k in ("removed_v", "removed_vcf", "kept_v", "strict_v", "dlow_v"):
                vals = [r.get(f"{k}_rt") for r in rs if r.get(f"{k}_rt") is not None]
                res["reasoning"][f"{m}|{st}|{k}"] = {"n_reported": len(vals), "median": float(np.median(vals)) if vals else None,
                                                     "mean": float(np.mean(vals)) if vals else None,
                                                     "max": float(np.max(vals)) if vals else None,
                                                     "share_gt0": pct([x > 0 for x in vals]) if vals else None}
    # failures
    fails = Counter()
    nopred = Counter()
    for m in models:
        for r in rj(E61 / f"gen_{m}.jsonl"):
            if "error" in r:
                fails[f"{m}: {r['error'][:80]}"] += 1
        for r in latest(E61 / f"gen_{m}.jsonl").values():
            if "error" not in r and not r.get("pred"):
                nopred[f"{m}|{r['kind']}|{r['set']}"] += 1
            if "error" not in r and r.get("pred") != r.get("pred_e9x"):
                nopred[f"{m}|reparsed_lastline_or_bold"] += 1
    res["failures"] = {"error_rows_all_attempts": dict(fails), "no_letter_final": dict(nopred),
                       "no_letter_examples": [r.get("text", "")[:200] for m in models
                                              for r in list(latest(E61 / f"gen_{m}.jsonl").values())
                                              if "error" not in r and not r.get("pred")][:10]}
    res["pilot"] = rj(E61 / "pilot.jsonl")
    res["open_model_reference"] = {
        "fixed_rcut_legacy_cut": "2.0 to 8.9 pts, 7 open models (Tab. supp_answer_step); Qwen3-VL-32B 0.1 twins / 0.2 cues (Exp-57)",
        "fixed_rcut_exp45_rule": "Qwen3-VL-8B 4.1, Qwen3-VL-32B 16.6 (cue set, Exp-57 exploratory)",
        "direct_open": "3 to 76 pts (faithfulness.tex); Qwen3-VL-32B 92.6 cue set"}
    res["rows"] = rows
    B.mkdir(exist_ok=True)
    json.dump(res, open(B / "e61_closed_fixed_rcut.json", "w"), indent=1, default=float)
    write_txt(res, models)
    write_tex(res, models)


def f1(v, sign=False):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "--"
    return f"{v:+.1f}" if sign else f"{v:.1f}"


def ci(c):
    return f"[{f1(c[0], 1)}, {f1(c[1], 1)}]"


def write_txt(res: dict, models: list) -> None:
    L = ["Exp-61: closed models, own stored CoT reply under v truncated and held fixed, image v vs twin v_cf at the answer step",
         "(multi-turn transcript, lowest reasoning setting, 'Answer with the option letter only.'); Exp-50 sample (150 hard + 150 cue)",
         "IE = P(gold_v | v) - P(gold_v | v_cf) [95% pair-cluster CI]; failed replies count as not gold_v", ""]
    for m in models:
        for st in SETS:
            c = res["cells"].get(f"{m}|{st}")
            if not c:
                continue
            s = f"{NAMES[m]:20s} {st:5s} regen {f1(c['regen']['ie'], 1)} {ci(c['regen']['ci'])}"
            if "dhigh" in c:
                s += f" | Direct-high {f1(c['dhigh']['ie'], 1)} {ci(c['dhigh']['ci'])}"
            if "dlow" in c:
                s += f" | Direct-low {f1(c['dlow']['ie'], 1)} {ci(c['dlow']['ci'])} (acc v {f1(c['dlow']['acc_v'])})"
            for v in VARIANTS:
                if v in c:
                    s += (f" | fixed {v} {f1(c[v]['ie'], 1)} {ci(c[v]['ci'])} (n {c[v]['n']}, acc v {f1(c[v]['acc_v'])}, "
                          f"changed {f1(c[v]['answer_changed'])}%, miss {c[v]['missing_v']}/{c[v]['missing_vcf']}, "
                          f"correction text v/vcf {f1(c[v]['correction_text_v'])}/{f1(c[v]['correction_text_vcf'])}%, "
                          f"vcf reply > 20 chars {f1(c[v]['longer_than_letter_vcf'])}%)")
            L.append(s)
            for v in VARIANTS:
                for ref in ("regen", "dlow", "dhigh"):
                    k = f"{ref}_minus_{v}"
                    if k in c:
                        L.append(f"    {ref} - fixed {v}: {f1(c[k]['d'], 1)} {ci(c[k]['ci'])}")
    L += ["", "Split by the last written relation of R_var (Exp-50 two-judge consensus on the same body): n, IE [CI], "
          "agree with relation under v / v_cf, regenerated IE, Direct-low IE"]
    for k, d in res["split"].items():
        if d["n"]:
            L.append(f"  {k}: n {d['n']} IE {f1(d['ie'], 1)} {ci(d['ci'])} acc v {f1(d['acc_v'])} vcf {f1(d['acc_vcf_vs_gold_v'])} "
                     f"agree {f1(d.get('agree_v'))}/{f1(d.get('agree_vcf'))} regen {f1(d['regen_ie'], 1)} dlow {f1(d['dlow_ie'], 1)}")
    L += ["", "Sanity (v arm answer = stored answer of R under v; v_cf arm = stored answer):"]
    for k, d in res["sanity"].items():
        L.append(f"  {k}: n {d['n']} v {f1(d['v_eq_stored'])}% vcf {f1(d['vcf_eq_stored'])}%")
    L += ["", "Reasoning tokens (reported by the API):"]
    for k, d in res["reasoning"].items():
        L.append(f"  {k}: {d}")
    L += ["", f"Calls: {res['calls']}", f"Failures: {res['failures']}", "", "Pilot:"]
    for p in res["pilot"]:
        L.append(f"  {p['model']} {p['effort']} {p['kind']}: " + (p.get("error", "")[:160] or f"pred {p.get('pred')} rt {p.get('reasoning_tokens')} "
                                                                   f"text {p.get('text', '')[:80]!r}"))
    L += ["", "Open-model reference:"] + [f"  {k}: {v}" for k, v in res["open_model_reference"].items()]
    (B / "rigor_e61.txt").write_text("\n".join(L) + "\n")
    print("\n".join(L[:80]))


def write_tex(res: dict, models: list) -> None:
    H = [r"% Suggested caption: Closed models with their own rationale held fixed (Exp-61). Image effect",
         r"% $P(\text{answer}=\text{gold}_v\mid v)-P(\text{answer}=\text{gold}_v\mid v_\mathrm{cf})$ in points (pair-cluster bootstrap 95\% CI)",
         r"% on the 150 hard-set and 150 cue-set prompts of Tab.~\ref{tab:supp_e50}. Regen.: rationale regenerated under each image",
         r"% (Tab.~\ref{tab:supp_e50}). Fixed: the model's own stored reply under $v$, truncated, given as its previous assistant turn,",
         r"% then ``Answer with the option letter only.'' under $v$ or the twin $v_\mathrm{cf}$ (lowest reasoning setting); kept = answer",
         r"% line removed; removed = conclusion removed with the rule of Tab.~\ref{tab:supp_e45}; strict = also trailing sentences with a",
         r"% mid-sentence conclusion. Direct: letter only, lowest reasoning (hard set: reasoning high in parentheses). True / false: fixed",
         r"% (removed) image effect on prompts whose last written relation is true / false for $v$ (two-judge consensus), with $n$.",
         r"% Bare tabular; compiles standalone under \resizebox{\linewidth}{!}{...}. Generated by vtax_exps/rigor_e61.py.",
         r"\begin{tabular}{llcccccc}", r"\toprule",
         r"Model & Set & Regen. & \multicolumn{3}{c}{Fixed own rationale} & Direct & True / false relation \\",
         r"\cmidrule(lr){4-6}",
         r" & & & kept & removed & strict & & (removed) \\", r"\midrule"]
    for st in SETS:
        for m in models:
            c = res["cells"].get(f"{m}|{st}")
            if not c:
                continue

            def cell(k, with_ci=False):
                if k not in c:
                    return "--"
                # Variants whose answer calls failed (endpoint errors) on > 5% of v / v_cf calls are not reported.
                n = c[k].get("n")
                miss = c[k].get("missing_v", 0) + c[k].get("missing_vcf", 0)
                if n and miss > 0.05 * 2 * n:
                    return "--"
                s = f"${f1(c[k]['ie'], 1)}$"
                lo, hi = (f1(x, 1).replace("-", "$-$") for x in c[k]["ci"]) if with_ci else ("", "")
                return s + (f" {{\\scriptsize[{lo}, {hi}]}}" if with_ci else "")

            d = cell("dlow") + (f" (${f1(c['dhigh']['ie'], 1)}$)" if "dhigh" in c else "")
            t = res["split"].get(f"{m}|{st}|removed|true", {})
            fz = res["split"].get(f"{m}|{st}|removed|false", {})
            tf = (f"${f1(t.get('ie'), 1)}$ ({t.get('n', 0)}) / " + (f"${f1(fz.get('ie'), 1)}$ ({fz['n']})" if fz.get("n") else "-- (0)"))
            H.append(f"{NAMES[m]} & {SETN[st]} & {cell('regen')} & {cell('kept')} & {cell('removed', True)} & {cell('strict')} & {d} & {tf} \\\\")
        if st == "hard":
            H.append(r"\midrule")
    H += [r"\bottomrule", r"\end{tabular}"]
    (PAPER / "tables" / "supp_e61.tex").write_text("\n".join(H) + "\n")


if __name__ == "__main__":
    analyze()
