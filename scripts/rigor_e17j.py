"""Exp-17j CPU analyses (round-6 Opus M2, M3, M5).

twostage  (M2) cue set, per model:
  (a) CoT answer probit: says-subject-front ~ cues (occlusion, vertical, size) with no text terms / + phrasing
      (question affirmation) / + first-named shape (is X the first shape named in the rationale); d' = 2 beta.
  (b) claim formation: probit of "first claim (v3) puts X in front" ~ phrasing + cues; claim-to-answer: share of CoT
      answers that follow the claim, and the answer probit ~ claim + phrasing + cues (residual image weight).
  (c) joint ML fit (not a projection): Direct weights b and CoT weights s*b + delta on ONE cue (occlusion or vertical);
      retention ratios r_occ, r_vert and r_occ / r_vert. Twin-pair bootstrap.
affirm    (M3) per model on the 560 judge-labelled cue claims (two judges, consensus): logit of "claim puts X in front"
          ~ X front + X lower + X larger + phrasing; same items with rule v2 / v3 labels; per-model rule-judge kappa.
natural   (M5) held-out cross-model means per 3DSRBench category and BLINK task: CoT - Direct and CoT - Filler+turn.

Usage: python rigor_e17j.py --only twostage,affirm,natural
"""
from __future__ import annotations
import vtax_paths as _vtp  # noqa: E402

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from scipy.stats import norm

from build_bundle import pe, rj
from claim_parse import first_claim_v3
from e2f_claim_edit import OBJ, split_sents
from rigor_e17f import QRX, geom_meta, logit_fit, probit_fit

O = Path(f"{_vtp.OUTPUTS}")
B = Path("outputs/bundle")
M7 = ["qwen3b", "qwen7b", "internvl3_8b", "phi35v", "molmo7b", "qwen3vl_8b", "internvl35_8b"]
RNG = np.random.default_rng(0)
COT_PREFIX = "Let's think step by step."
q = lambda v: [round(float(np.nanquantile(v, .025)), 3), round(float(np.nanquantile(v, .975)), 3)]


def rows_for(m: str) -> dict:
    """Per condition arrays on the cue set (both phrasings)."""
    out = {}
    for c in ("direct", "cot"):
        R = [r for r in rj(O / "e10" / f"{m}_cues.jsonl") if r["condition"] == c]
        A = []
        for r in R:
            qm = QRX.search(r["question"])
            x, behind = qm.group(1).lower(), qm.group(2) == "behind"
            g = geom_meta(r["meta"], x)
            says = (pe(r) == "A") != behind
            claim, first_x = 0.0, 0.0
            if c == "cot":
                body = r["text"][len(COT_PREFIX):]
                fc = first_claim_v3(split_sents(body))
                claim = (1.0 if fc[1] == x else -1.0) if fc else 0.0
                o = OBJ.search(body)
                first_x = (1.0 if (o.group(1) + " " + o.group(2)).lower() == x else -1.0) if o else 0.0
            A.append((float(says), -1.0 if behind else 1.0, 1.0 if g["x_front"] else -1.0, 1.0 if g["x_lower"] else -1.0,
                      1.0 if g["x_larger"] else -1.0, claim, first_x, r["image_id"]))
        a = np.array([t[:7] for t in A])
        out[c] = {"y": a[:, 0], "ph": a[:, 1], "occ": a[:, 2], "vert": a[:, 3], "size": a[:, 4], "claim": a[:, 5], "first": a[:, 6],
                  "pair": np.array([t[7] for t in A])}
    return out


def design(d: dict, terms: list[str], sel=None) -> tuple[np.ndarray, np.ndarray]:
    s = np.arange(len(d["y"])) if sel is None else sel
    X = np.column_stack([np.ones(len(s))] + [d[t][s] for t in terms])
    return X, d["y"][s]


def twostage(lines: list[str]) -> dict:
    res = {}
    for m in M7:
        D = rows_for(m)
        pairs = sorted(set(D["direct"]["pair"]))
        pidx = {c: {p: np.where(D[c]["pair"] == p)[0] for p in pairs} for c in D}
        cues = ["occ", "vert", "size"]
        specs = {"direct": ("direct", ["ph"] + cues, None),
                 "cot_no_text": ("cot", cues, None), "cot_phr": ("cot", ["ph"] + cues, None),
                 "cot_phr_first": ("cot", ["ph", "first"] + cues, None)}

        def run(sel):
            out = {}
            for name, (c, terms, _) in specs.items():
                X, y = design(D[c], terms, sel[c])
                b = probit_fit(X, y)
                out[name] = {t: 2 * b[1 + terms.index(t)] for t in terms}
            # claim formation (items with a claim) and claim -> answer
            cc = D["cot"]
            s = sel["cot"][cc["claim"][sel["cot"]] != 0]
            Xc = np.column_stack([np.ones(len(s)), cc["ph"][s], cc["occ"][s], cc["vert"][s], cc["size"][s]])
            bc = probit_fit(Xc, (cc["claim"][s] > 0).astype(float))
            out["claim"] = {"ph": 2 * bc[1], "occ": 2 * bc[2], "vert": 2 * bc[3], "size": 2 * bc[4]}
            Xa = np.column_stack([np.ones(len(s)), cc["claim"][s], cc["ph"][s], cc["occ"][s], cc["vert"][s], cc["size"][s]])
            ba = probit_fit(Xa, cc["y"][s])
            out["answer_given_claim"] = {"claim": 2 * ba[1], "ph": 2 * ba[2], "occ": 2 * ba[3], "vert": 2 * ba[4], "size": 2 * ba[5]}
            out["follow"] = float(np.mean((cc["y"][s] > 0.5) == (cc["claim"][s] > 0)))
            out["claim_correct"] = float(np.mean((cc["claim"][s] > 0) == (cc["occ"][s] > 0)))
            # joint ML: CoT weights = s * Direct weights + delta on one cue
            dd = D["direct"]
            sd_, sc_ = sel["direct"], sel["cot"]
            Xd = np.column_stack([np.ones(len(sd_)), dd["ph"][sd_], dd["occ"][sd_], dd["vert"][sd_], dd["size"][sd_]])
            Xk = np.column_stack([np.ones(len(sc_)), cc["ph"][sc_], cc["occ"][sc_], cc["vert"][sc_], cc["size"][sc_]])
            yd, yk = dd["y"][sd_], cc["y"][sc_]
            for which, j in (("occ", 2), ("vert", 3)):
                def nll(p):
                    a_d, p_d, bo, bv, bs, a_c, p_c, sc, dl = p
                    zd = Xd @ np.array([a_d, p_d, bo, bv, bs])
                    bco = np.array([a_c, p_c, sc * bo, sc * bv, sc * bs])
                    bco[j] += dl
                    zk = Xk @ bco
                    return -(yd * norm.logcdf(zd) + (1 - yd) * norm.logcdf(-zd)).sum() - (yk * norm.logcdf(zk) + (1 - yk) * norm.logcdf(-zk)).sum() \
                        + 0.05 * (p[2:5] ** 2).sum()
                p0 = np.array([0, 0, .5, .5, 0, 0, 0, .5, 0], float)
                r = minimize(nll, p0, method="L-BFGS-B")
                out[f"joint_delta_{which}"] = 2 * r.x[8]
                out[f"joint_s_{which}"] = r.x[7]
            out["retain_occ"] = out["cot_phr"]["occ"] / out["direct"]["occ"]
            out["retain_vert"] = out["cot_phr"]["vert"] / out["direct"]["vert"]
            out["retain_ratio"] = out["retain_occ"] / out["retain_vert"] if abs(out["retain_vert"]) > 1e-6 else np.nan
            return out

        full = {c: np.arange(len(D[c]["y"])) for c in D}
        est = run(full)
        boots = []
        for _ in range(150):
            js = RNG.integers(0, len(pairs), len(pairs))
            boots.append(run({c: np.concatenate([pidx[c][pairs[j]] for j in js]) for c in D}))

        def flat(o, pre=""):
            f = {}
            for k, v in o.items():
                if isinstance(v, dict):
                    f |= flat(v, pre + k + ".")
                else:
                    f[pre + k] = v
            return f
        fe, fb = flat(est), [flat(b) for b in boots]
        res[m] = {k: {"est": round(float(v), 3), "ci": q(np.array([b[k] for b in fb]))} for k, v in fe.items()}
        r = res[m]
        g = lambda k: f"{r[k]['est']:+.2f}{r[k]['ci']}"
        lines.append(f"TWOSTAGE {m:14s} Direct occ {r['direct.occ']['est']:+.2f} vert {r['direct.vert']['est']:+.2f} | CoT occ no-text {r['cot_no_text.occ']['est']:+.2f} "
                     f"+phr {r['cot_phr.occ']['est']:+.2f} +first {r['cot_phr_first.occ']['est']:+.2f}; vert {r['cot_no_text.vert']['est']:+.2f}/{r['cot_phr.vert']['est']:+.2f}/"
                     f"{r['cot_phr_first.vert']['est']:+.2f}; phr {g('cot_phr.ph')} first {g('cot_phr_first.first')} | claim occ {g('claim.occ')} vert {g('claim.vert')} ph {g('claim.ph')} "
                     f"| follow {r['follow']['est']:.3f} ans|claim: claim {g('answer_given_claim.claim')} occ {g('answer_given_claim.occ')} vert {g('answer_given_claim.vert')} "
                     f"| joint d_occ {g('joint_delta_occ')} d_vert {g('joint_delta_vert')} | retain occ {r['retain_occ']['est']:.2f} vert {r['retain_vert']['est']:.2f} ratio {g('retain_ratio')}")
    return res


def kappa(a, b):
    cats = sorted(set(a) | set(b))
    po = np.mean([x == y for x, y in zip(a, b)])
    pe_ = sum((a.count(c) / len(a)) * (b.count(c) / len(b)) for c in cats)
    return float((po - pe_) / (1 - pe_)) if pe_ < 1 else 1.0


def affirm(lines: list[str]) -> dict:
    rows = {}
    for sfx in ("", "_s2"):
        J = {j: {(r["model"], r["key"]): r for r in rj(O / "e16" / f"{j}{sfx}.jsonl")} for j in ("claude-opus-5-5", "gpt-6-astra")}
        for k in set(J["claude-opus-5-5"]) & set(J["gpt-6-astra"]):
            a, b = J["claude-opus-5-5"][k], J["gpt-6-astra"][k]
            rows[k] = a | {"label_opus": a["label"], "label_gpt": b["label"]}
    rel = lambda v, r: "none" if v in (None, "none") else ("x" if v == r["x"] else "y")
    res = {}
    for m in M7:
        rs = [r for k, r in rows.items() if k[0] == m]
        cons = [r for r in rs if rel(r["label_opus"], r) == rel(r["label_gpt"], r)]
        out = {"n_labelled": len(rs), "n_consensus": len(cons),
               "kappa_judges": round(kappa([rel(r["label_opus"], r) for r in rs], [rel(r["label_gpt"], r) for r in rs]), 3)}
        for v in ("v2", "v3"):
            out[f"kappa_rule_{v}"] = round(kappa([rel(r[f"rule_{v}"], r) for r in cons], [rel(r["label_opus"], r) for r in cons]), 3)
        for lab in ("judge", "v2", "v3"):
            get = (lambda r: r["label_opus"]) if lab == "judge" else (lambda r, v=lab: r[f"rule_{v}"])
            c = [r for r in cons if rel(get(r), r) != "none"]
            if len(c) < 20:
                continue
            y = np.array([float(get(r) == r["x"]) for r in c])
            X = np.column_stack([np.ones(len(c)), [1.0 if r["front"] == r["x"] else -1.0 for r in c], [1.0 if r["lower"] == r["x"] else -1.0 for r in c],
                                 [1.0 if r["larger"] == r["x"] else -1.0 for r in c], [1.0 if r["phrasing"] == "in front of" else -1.0 for r in c]])
            b = logit_fit(X, y)
            pairs = sorted({r["pair"] for r in c})
            pidx = {p: [i for i, r in enumerate(c) if r["pair"] == p] for p in pairs}
            bb = np.array([logit_fit(X[s], y[s]) for s in (np.concatenate([pidx[pairs[j]] for j in RNG.integers(0, len(pairs), len(pairs))]) for _ in range(500))])
            aff = np.mean([(get(r) == r["x"]) == (r["phrasing"] == "in front of") for r in c])
            out[lab] = {"n": len(c), "affirm_share": round(100 * float(aff), 1), "b_affirm": round(float(b[4]), 2), "b_affirm_ci": q(bb[:, 4]),
                        "b_front": round(float(b[1]), 2), "b_lower": round(float(b[2]), 2), "b_larger": round(float(b[3]), 2)}
        lab_ok = [out.get(l, {}).get("b_affirm_ci", [0, 0])[0] > 0 for l in ("judge", "v2", "v3")]
        out["robust_positive"] = bool(all(lab_ok))
        res[m] = out
        lines.append(f"AFFIRM-J {m:14s} n={out['n_consensus']} kappa judges {out['kappa_judges']} rule v2 {out['kappa_rule_v2']} v3 {out['kappa_rule_v3']} | "
                     + " | ".join(f"{l}: aff {out[l]['affirm_share']} b {out[l]['b_affirm']:+.2f}{out[l]['b_affirm_ci']}" for l in ("judge", "v2", "v3") if l in out)
                     + f" | robust {out['robust_positive']}")
    return res


def natural(lines: list[str]) -> dict:
    import heldout_breakdown as HB
    rows = {m: {r["item"]: r for r in HB.model_rows(m)} for m in M7}
    for m in M7:
        p = O / "e6x" / f"{m}_filler2.jsonl"
        if p.exists():
            for r in rj(p):
                if r["condition"] == "filler_R+turn" and r["item_id"] in rows[m]:
                    rows[m][r["item_id"]]["pf"] = pe(r)
    rows["qwen3vl_8b_think"] = {r["item"]: r for r in HB.think_rows()}
    depth = {"blink_depth", "3dsr_location_closer_to_camera", "3dsr_orientation_in_front_of"}
    subsets = [(t, (lambda tt, t=t: tt == t)) for t in ("blink_depth", "blink_relation", "3dsr_location_closer_to_camera",
                                                          "3dsr_orientation_in_front_of", "3dsr_height_higher", "3dsr_location_above",
                                                          "3dsr_orientation_on_the_left")]
    subsets += [("depth324", lambda tt: tt in depth), ("3dsr_depth", lambda tt: tt in depth - {"blink_depth"})]

    def boot(diff: np.ndarray, groups: list, n: int = 4000) -> dict:
        """Image-cluster bootstrap, fixed seed per cell (identical for every table that reads this file)."""
        rng = np.random.default_rng(0)
        cl = sorted(set(groups))
        idx = {c: np.where(np.array(groups) == c)[0] for c in cl}
        bs = [diff[np.concatenate([idx[cl[j]] for j in rng.integers(0, len(cl), len(cl))])].mean() for _ in range(n)]
        return {"mean": round(100 * float(diff.mean()), 1), "ci": [round(100 * float(np.quantile(bs, .025)), 1), round(100 * float(np.quantile(bs, .975)), 1)]}

    res = {}
    for t, sel in subsets:
        for label, models, key in (("cot-direct|8 open", M7 + ["qwen3vl_8b_think"], "pd"), ("cot-direct|7 (no Phi)", [m for m in M7 if m != "phi35v"] + ["qwen3vl_8b_think"], "pd"),
                                   ("cot-direct|7 instruct", M7, "pd"),
                                   ("cot-filler_turn|7", M7, "pf"), ("cot-filler_turn|6 (no Phi)", [m for m in M7 if m != "phi35v"], "pf")):
            items = sorted(k for k, r in rows[M7[0]].items() if sel(r["task"]) and all(k in rows[m] and key in rows[m][k] for m in models))
            if not items:
                continue
            diff = np.array([np.mean([float(rows[m][k]["pc"] == rows[m][k]["gold"]) - float(rows[m][k][key] == rows[m][k]["gold"]) for m in models]) for k in items])
            cb = boot(diff, [rows[M7[0]][k]["img"] for k in items])
            per = {m: round(100 * float(np.mean([float(rows[m][k]["pc"] == rows[m][k]["gold"]) - float(rows[m][k][key] == rows[m][k]["gold"]) for k in items])), 1) for m in models}
            res.setdefault(t, {})[label] = cb | {"n_items": len(items), "per_model": per}
            lines.append(f"NATURAL {t:32s} {label:28s} n={len(items)} {cb['mean']:+.1f} {cb['ci']}")
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="twostage,affirm,natural")
    a = ap.parse_args()
    lines: list[str] = []
    for k in a.only.split(","):
        json.dump({"twostage": twostage, "affirm": affirm, "natural": natural}[k](lines), open(B / f"e17j_{k}.json", "w"), indent=1)
    old = (B / "rigor_e17j.txt").read_text().splitlines() if (B / "rigor_e17j.txt").exists() else []
    pref = {"twostage": "TWOSTAGE", "affirm": "AFFIRM-J", "natural": "NATURAL"}
    keep = [x for x in old if not any(x.startswith(pref[k]) for k in a.only.split(","))]
    (B / "rigor_e17j.txt").write_text("\n".join(keep + lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
