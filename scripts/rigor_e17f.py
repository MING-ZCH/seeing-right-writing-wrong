"""Exp-17f analyses for the round-4 Opus review.

1. relation_sensitivity: answer-step image sensitivity with the text fixed, per relation family and model
   (R_cut + appended conclusion: gray dvis = S_orig - S_gray, mirror dS = S_orig - S_mirror on left/right items;
   twins: own cut rationale under v vs v_cf from imgswap.json), against Direct accuracy on that family.
2. cue_claims: first relational claim on the cue-factorial CoT outputs (rule parse): share naming the true front /
   lower / larger object with twin-pair CIs, logistic regression of "claims the subject is in front" on
   (subject is front, subject is lower, subject is larger); "on top" wording vs vertical position; teaser check.
3. glm: clamp-free probit decomposition (says-subject-front ~ phrasing + occlusion + vertical + size, d' component =
   2 x coefficient), twin-pair cluster bootstrap; AUC of the continuous readout; Qwen3-VL ladder points.
5. commit_nulls: uniform, non-conclusion and position-matched random-sentence nulls for the commit sentence.
8. closed_facts: DeepSeek hidden vs no reasoning on twins; Qwen3.8-Max hard set.

Usage: python rigor_e17f.py   (writes outputs/bundle/{relation_sensitivity,cue_claims,glm_decomp,commit_nulls,closed_facts}.json
                               and rigor_e17f.txt; figure paper/figures/fig_answer_step.{pdf,png})
"""
from __future__ import annotations
import vtax_paths as _vtp  # noqa: E402

import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from scipy.stats import binomtest, norm, spearmanr

import build_bundle as BB
from build_bundle import ci, pe, rj
from e2f_claim_edit import first_claim, split_sents

O = Path(f"{_vtp.OUTPUTS}")
W = O.parent
L = Path(f"{_vtp.LEGACY_RESULTS}")
B = Path("outputs/bundle")
FIG = Path(f"{_vtp.FIGURES}")
M7 = ["qwen3b", "qwen7b", "internvl3_8b", "phi35v", "molmo7b", "qwen3vl_8b", "internvl35_8b"]
QRX = re.compile(r"Is the (\w+ \w+) (in front of|behind) the (\w+ \w+)\?")
ONTOP = re.compile(r"\bon top\b", re.I)
RNG = np.random.default_rng(0)


def q(v) -> list[float]:
    return [round(float(np.quantile(v, .025)), 3), round(float(np.quantile(v, .975)), 3)]


# ============================================================================= 1. relation sensitivity

FAMILY_NAMES = {"LR": "2D left/right", "UD": "2D above/below", "depth": "depth", "distance": "3D distance", "occ": "occlusion twins"}


def relation_sensitivity(lines: list[str]) -> dict:
    res = {}
    imgswap = json.load(open(B / "imgswap.json"))
    for m in M7:
        recs = [r for r in rj(O / "e3x" / f"{m}.jsonl") if r["context"] == "rcut"]
        main = BB.load_main(O, L, m)
        idx, meta = defaultdict(dict), {}
        for r in recs:
            idx[(r["item_id"], r["image"])][r["Y"]] = r
            meta[r["item_id"]] = r
        lr = {u for (u, im) in idx if im == "mirror"}

        def fam(u: str) -> str:
            t = meta[u]["task"]
            if t in ("cvb_relation", "syn_relpos"):
                return "LR" if u in lr else "UD"
            return "depth" if t in ("cvb_depth", "syn_depth") else "distance"

        def S(u: str, im: str):
            d = idx.get((u, im))
            r0 = meta[u]
            g = r0["gold"]
            w = [c for c in r0["letters"] if c != g][0]
            if not d or g not in d or w not in d:
                return None
            return float(pe(d[g]) == g) - float(pe(d[w]) == w)

        out = {}
        for f in ("LR", "UD", "depth", "distance"):
            us = [u for u in meta if fam(u) == f]
            so = {u: S(u, "original") for u in us}
            sg = {u: S(u, "gray") for u in us}
            ks = [u for u in us if so[u] is not None and sg[u] is not None]
            if len(ks) < 20:
                continue
            row = {"n": len(ks), "S_orig": ci([so[u] for u in ks], [meta[u]["image_id"] for u in ks]),
                   "dvis_gray": ci([so[u] - sg[u] for u in ks], [meta[u]["image_id"] for u in ks]),
                   "direct_acc": ci([main["direct"][u][0] for u in ks if u in main.get("direct", {})],
                                    [meta[u]["image_id"] for u in ks if u in main.get("direct", {})])}
            if f == "LR":
                sm = {u: S(u, "mirror") for u in us}
                km = [u for u in ks if sm[u] is not None]
                row["S_orig_LRsubset"] = ci([so[u] for u in km], [meta[u]["image_id"] for u in km])
                row["S_mirror"] = ci([sm[u] for u in km], [meta[u]["image_id"] for u in km])
                row["dS_mirror"] = ci([so[u] - sm[u] for u in km], [meta[u]["image_id"] for u in km])
            out[f] = row
        # occlusion twins: own cut rationale, image v -> v_cf
        e10 = [r for r in rj(O / "e10" / f"{m}.jsonl") if r["condition"] == "direct" and r["task"].startswith("occ")]
        out["occ"] = {"n": len(e10), "direct_acc": ci([float(pe(r) == r["gold"]) for r in e10], [r["image_id"] for r in e10]),
                      "dvis_cut_swap": imgswap.get(m, {}).get("cut:all", {}).get("dvis"),
                      "dvis_cut_swap_rightR": imgswap.get(m, {}).get("cut:rightR", {}).get("dvis"),
                      "dvis_full_swap": imgswap.get(m, {}).get("full:all", {}).get("dvis")}
        res[m] = out
        f2 = lambda d: f"{d['mean']:+.1f} [{d['lo']:+.1f},{d['hi']:+.1f}]" if d and "mean" in d else "--"
        lines.append(f"RELSENS {m:14s} " + " | ".join(
            f"{k}: D {v['direct_acc']['mean']:.1f} S {v['S_orig']['mean']:.1f} dvisG {f2(v['dvis_gray'])}"
            + (f" dSmir {f2(v['dS_mirror'])} (S_LR {v['S_orig_LRsubset']['mean']:.1f}->{v['S_mirror']['mean']:.1f})" if k == "LR" else "")
            for k, v in out.items() if k != "occ") + f" | occ: D {out['occ']['direct_acc']['mean']:.1f} cut-swap {f2(out['occ']['dvis_cut_swap'])}")
    # hypothesis tests
    pts = [(m, f, v["direct_acc"]["mean"], v["dvis_gray"]["mean"], v["S_orig"]["mean"]) for m, d in res.items()
           for f, v in d.items() if f != "occ"]
    rho, p = spearmanr([x[2] for x in pts], [x[3] for x in pts])
    checkers = ["internvl35_8b", "internvl3_8b", "qwen7b"]
    ptc = [x for x in pts if x[0] in checkers]
    rho_c, p_c = spearmanr([x[2] for x in ptc], [x[3] for x in ptc])
    occ_pts = [(m, d["occ"]["direct_acc"]["mean"], d["occ"]["dvis_cut_swap"]["mean"]) for m, d in res.items() if d["occ"]["dvis_cut_swap"]]
    rho_o, p_o = spearmanr([x[1] for x in occ_pts], [x[2] for x in occ_pts])
    res["_tests"] = {"spearman_direct_acc_vs_gray_dvis_all": [round(float(rho), 3), float(p), len(pts)],
                     "spearman_checkers_only": [round(float(rho_c), 3), float(p_c), len(ptc)],
                     "spearman_twins_direct_acc_vs_cut_swap": [round(float(rho_o), 3), float(p_o), len(occ_pts)]}
    lines.append(f"RELSENS tests {res['_tests']}")
    return res


def fig_answer_step(rs: dict) -> None:
    """Answer-step image sensitivity vs Direct accuracy per relation family (gray ablation) + twin cut swap."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from make_figures_v3 import CANON_LABELS, MODEL_LINE_STYLE
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 7.5, "axes.labelsize": 7.5, "xtick.labelsize": 7.0,
                         "ytick.labelsize": 7.0, "legend.fontsize": 7.0, "axes.spines.top": False, "axes.spines.right": False,
                         "pdf.fonttype": 42})
    fam_mark = {"LR": "o", "UD": "^", "depth": "s", "distance": "D", "occ": "X"}
    fig, axes = plt.subplots(1, 2, figsize=(6.875, 2.55), gridspec_kw={"width_ratios": [1.45, 1.0], "wspace": 0.28})
    fig.subplots_adjust(left=0.065, right=0.985, top=0.80, bottom=0.17)
    ax = axes[0]
    for m in M7:
        col = MODEL_LINE_STYLE[m]["color"]
        for f in ("LR", "UD", "depth", "distance"):
            v = rs[m].get(f)
            if not v:
                continue
            x, y = v["direct_acc"]["mean"], v["dvis_gray"]["mean"]
            ax.errorbar(x, y, yerr=[[y - v["dvis_gray"]["lo"]], [v["dvis_gray"]["hi"] - y]], fmt="none", ecolor=col,
                        elinewidth=0.5, alpha=0.7, zorder=2)
            ax.scatter([x], [y], marker=fam_mark[f], s=24, facecolor=col, edgecolor="white", linewidth=0.4, zorder=3)
    ax.axhline(0, color="#2F3640", linewidth=0.6)
    ax.set_xlabel("Direct accuracy on the relation family (%)")
    ax.set_ylabel("Answer-step image use: $S_{orig}-S_{gray}$ (pts)")
    ax.set_title("(a) Gray image under fixed $R_{cut}$ + conclusion", fontsize=7.5, loc="left", fontweight="bold")
    ax.grid(True, color="#E6E9EE", linewidth=0.5, zorder=0)
    ax = axes[1]
    for m in M7:
        col = MODEL_LINE_STYLE[m]["color"]
        v = rs[m]["occ"]
        d = v["dvis_cut_swap"]
        if d:
            ax.errorbar(v["direct_acc"]["mean"], d["mean"], yerr=[[d["mean"] - d["lo"]], [d["hi"] - d["mean"]]], fmt="none",
                        ecolor=col, elinewidth=0.6, zorder=2)
            ax.scatter([v["direct_acc"]["mean"]], [d["mean"]], marker="X", s=30, facecolor=col, edgecolor="white",
                       linewidth=0.4, zorder=3)
        lr = rs[m].get("LR")
        if lr and "dS_mirror" in lr:
            ax.scatter([lr["direct_acc"]["mean"]], [lr["dS_mirror"]["mean"] / 2], marker="o", s=24, facecolor="white",
                       edgecolor=col, linewidth=0.9, zorder=3)
    ax.axhline(0, color="#2F3640", linewidth=0.6)
    ax.set_xlabel("Direct accuracy (%)")
    ax.set_ylabel("Image effect, text fixed (pts)")
    ax.set_title("(b) Twin swap, cut $R$ (×); mirror $\\Delta S/2$ (○)", fontsize=7.5, loc="left", fontweight="bold")
    ax.grid(True, color="#E6E9EE", linewidth=0.5, zorder=0)
    from matplotlib.lines import Line2D
    h = [Line2D([], [], marker="o", linestyle="none", markersize=4.5, markerfacecolor=MODEL_LINE_STYLE[m]["color"],
                markeredgecolor="white", label=CANON_LABELS[m]) for m in M7]
    h += [Line2D([], [], marker=fam_mark[f], linestyle="none", markersize=4.5, markerfacecolor="white", markeredgecolor="black",
                 label=FAMILY_NAMES[f]) for f in ("LR", "UD", "depth", "distance")]
    fig.legend(handles=h, loc="upper center", ncol=6, frameon=False, bbox_to_anchor=(0.52, 1.0), handletextpad=0.2,
               columnspacing=0.7)
    fig.savefig(FIG / "fig_answer_step.pdf")
    fig.savefig(FIG / "fig_answer_step.png", dpi=300)


def fig_answer_step_compact(rs: dict) -> None:
    """Single-column version: colour/marker = relation family; filled = rationale-body checkers, hollow = copiers."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 7.5, "axes.labelsize": 7.5, "xtick.labelsize": 7.0,
                         "ytick.labelsize": 7.0, "legend.fontsize": 7.0, "axes.spines.top": False, "axes.spines.right": False,
                         "pdf.fonttype": 42})
    fam = {"LR": ("o", "#3B6998"), "UD": ("^", "#7FA7CF"), "depth": ("s", "#B85244"), "distance": ("D", "#E2B28F"),
           "occ": ("X", "#2F3640")}
    checkers = {"internvl3_8b", "internvl35_8b", "qwen7b"}
    fig, ax = plt.subplots(figsize=(3.28, 2.45))
    fig.subplots_adjust(left=0.15, right=0.98, top=0.80, bottom=0.17)
    for m in M7:
        filled = m in checkers
        for f in ("LR", "UD", "depth", "distance"):
            v = rs[m].get(f)
            if not v:
                continue
            mk, col = fam[f]
            ax.scatter([v["direct_acc"]["mean"]], [v["dvis_gray"]["mean"]], marker=mk, s=22, facecolor=col if filled else "white",
                       edgecolor=col, linewidth=0.9, zorder=3)
        d = rs[m]["occ"]["dvis_cut_swap"]
        if d:
            ax.scatter([rs[m]["occ"]["direct_acc"]["mean"]], [d["mean"]], marker="X", s=26, facecolor=fam["occ"][1] if filled else "white",
                       edgecolor=fam["occ"][1], linewidth=0.8, zorder=3)
    ax.axhline(0, color="#2F3640", linewidth=0.6)
    ax.grid(True, color="#E6E9EE", linewidth=0.5, zorder=0)
    ax.set_xlabel("Direct accuracy on the relation (%)", labelpad=1.5)
    ax.set_ylabel("Answer-step image effect,\ntext fixed (pts)", labelpad=1.5)
    t = rs["_tests"]
    ax.text(0.02, 0.97, f"gray $R_{{cut}}$: $\\rho$ = {t['spearman_direct_acc_vs_gray_dvis_all'][0]:.2f}\ntwin cut-$R$ swap: $\\rho$ = "
            f"{t['spearman_twins_direct_acc_vs_cut_swap'][0]:.2f}", transform=ax.transAxes, fontsize=7.0, va="top", ha="left",
            color="#4A515C")
    h = [Line2D([], [], marker=fam[f][0], linestyle="none", markersize=4.5, markerfacecolor=fam[f][1], markeredgecolor=fam[f][1],
                label=l) for f, l in (("LR", "2D left/right (gray)"), ("UD", "2D above/below (gray)"), ("depth", "depth (gray)"),
                                      ("distance", "3D distance (gray)"), ("occ", "occlusion twins (swap)"))]
    h.append(Line2D([], [], marker="o", linestyle="none", markersize=4.5, markerfacecolor="white", markeredgecolor="#7A828E",
                    label="hollow = copier"))
    fig.legend(handles=h, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.55, 1.0), handletextpad=0.2,
               columnspacing=0.8, labelspacing=0.25)
    fig.savefig(FIG / "fig_answer_step_compact.pdf")
    fig.savefig(FIG / "fig_answer_step_compact.png", dpi=300)


# ============================================================================= 2. cue-set claims

def geom_meta(mt: dict, x: str) -> dict:
    front, back = mt["front_name"].lower(), mt["back_name"].lower()
    cyf = (mt["bbox_front"][1] + mt["bbox_front"][3]) / 2
    cyb = (mt["bbox_back"][1] + mt["bbox_back"][3]) / 2
    lower = front if cyf > cyb else back
    larger = front if mt["area_front_px"] > mt["area_back_px"] else back
    return {"front": front, "lower": lower, "upper": back if lower == front else front, "larger": larger,
            "x_front": x == front, "x_lower": x == lower, "x_larger": x == larger}


def logit_fit(X: np.ndarray, y: np.ndarray, ridge: float = 1e-3) -> np.ndarray:
    b = np.zeros(X.shape[1])
    for _ in range(50):
        p = 1 / (1 + np.exp(-X @ b))
        g = X.T @ (y - p) - ridge * b
        H = -(X.T * (p * (1 - p))) @ X - ridge * np.eye(len(b))
        step = np.linalg.solve(H, g)
        b -= step
        if np.abs(step).max() < 1e-8:
            break
    return b


def claims_for(m: str, which: str) -> list[dict]:
    fn = O / "e10" / (f"{m}_cues.jsonl" if which == "cues" else f"{m}.jsonl")
    out = []
    for r in rj(fn):
        if r["condition"] != "cot" or not r["task"].startswith("occ"):
            continue
        qm = QRX.search(r["question"])
        x = qm.group(1).lower()
        sents = split_sents(r["text"][len("Let's think step by step."):])
        fc = first_claim(sents)
        g = geom_meta(r["meta"], x)
        out.append({"named": fc[1] if fc else None, "sent": sents[fc[0]] if fc else None, "x": x, "pair": r["image_id"],
                    **g})
    return out


def cue_claims(lines: list[str]) -> dict:
    res = {}
    for which in ("cues", "twins"):
        for m in M7:
            cl = [c for c in claims_for(m, which) if c["named"]]
            if not cl:
                continue
            pairs = sorted({c["pair"] for c in cl})
            pidx = {p: [i for i, c in enumerate(cl) if c["pair"] == p] for p in pairs}
            arr = {k: np.array([float(c["named"] == c[k]) for c in cl]) for k in ("front", "lower", "larger")}
            row = {"n_parsed": len(cl)}
            boots = {k: [] for k in arr}
            for _ in range(2000):
                s = np.concatenate([pidx[pairs[j]] for j in RNG.integers(0, len(pairs), len(pairs))])
                for k in arr:
                    boots[k].append(arr[k][s].mean())
            for k in arr:
                if which == "twins" and k == "larger":
                    continue  # identical to 100 - names_true_front on the twins (front never larger)
                row[f"names_{k}"] = {"pct": round(100 * arr[k].mean(), 1), "ci": [round(100 * v, 1) for v in q(boots[k])],
                                     "binom_p_vs_50": float(binomtest(int(arr[k].sum()), len(cl), 0.5).pvalue)}
            # logistic regression: claims subject x is front ~ x front + x lower (+ x larger)
            y = np.array([float(c["named"] == c["x"]) for c in cl])
            cols = ["intercept", "x_front", "x_lower"] + (["x_larger"] if which == "cues" else [])
            X = np.column_stack([np.ones(len(cl))] + [np.array([1.0 if c[k] else -1.0 for c in cl]) for k in cols[1:]])
            b = logit_fit(X, y)
            bb = []
            for _ in range(1000):
                s = np.concatenate([pidx[pairs[j]] for j in RNG.integers(0, len(pairs), len(pairs))])
                bb.append(logit_fit(X[s], y[s]))
            bb = np.array(bb)
            row["logit"] = {k: {"b": round(float(b[i]), 3), "ci": [round(v, 3) for v in q(bb[:, i])]} for i, k in enumerate(cols)}
            # "on top" wording
            ot = [c for c in cl if ONTOP.search(c["sent"])]
            nt = [c for c in cl if not ONTOP.search(c["sent"])]
            row["on_top"] = {"share_of_claims": round(100 * len(ot) / len(cl), 1),
                             "names_upper_if_on_top": round(100 * np.mean([c["named"] == c["upper"] for c in ot]), 1) if ot else None,
                             "names_upper_otherwise": round(100 * np.mean([c["named"] == c["upper"] for c in nt]), 1) if nt else None,
                             "true_front_if_on_top": round(100 * np.mean([c["named"] == c["front"] for c in ot]), 1) if ot else None,
                             "true_front_otherwise": round(100 * np.mean([c["named"] == c["front"] for c in nt]), 1) if nt else None,
                             "n_on_top": len(ot)}
            res.setdefault(which, {})[m] = row
            lg = row["logit"]
            lines.append(f"CLAIMS {which:5s} {m:14s} n={len(cl)} front {row['names_front']['pct']} lower {row['names_lower']['pct']} {row['names_lower']['ci']}"
                         + (f" larger {row['names_larger']['pct']} {row['names_larger']['ci']}" if "names_larger" in row else "")
                         + " | logit " + " ".join(f"{k}={v['b']:+.2f}{v['ci']}" for k, v in lg.items() if k != "intercept")
                         + f" | on-top {row['on_top']['share_of_claims']}% upper|ontop {row['on_top']['names_upper_if_on_top']} vs {row['on_top']['names_upper_otherwise']}")
    # teaser check (Fig. 1a: Qwen3-VL-8B, occ085 / occ085_cf, e2d CoT)
    items = {}
    for fn in ("items.jsonl", "items_cf.jsonl"):
        for r in map(json.loads, open(W / "exps" / "items" / "occlusion" / fn)):
            items[r["item_id"]] = r
    e2d = {r["item_id"]: r for r in rj(O / "e2d" / "qwen3vl_8b.jsonl") if r["condition"] == "cot"}
    tz = {}
    for k in ("occ085", "occ085_cf"):
        r, it = e2d[k], items[k]
        x = QRX.search(it["question"]).group(1).lower()
        g = geom_meta(it["meta"], x)
        sents = split_sents(r["text"][len("Let's think step by step."):])
        fc = first_claim(sents)
        tz[k] = {"first_claim": sents[fc[0]] if fc else None, "named_front": fc[1] if fc else None, "true_front": g["front"],
                 "upper_object": g["upper"], "names_upper": (fc[1] == g["upper"]) if fc else None,
                 "uses_on_top": bool(fc and ONTOP.search(sents[fc[0]]))}
    res["teaser_qwen3vl"] = tz
    lines.append(f"TEASER {json.dumps(tz)}")
    return res


# ============================================================================= 3. probit GLM

def probit_fit(X: np.ndarray, y: np.ndarray, ridge: float = 0.1) -> np.ndarray:
    """Probit MLE with a weak N(0, 1/ridge) prior on non-intercept terms (finite under quasi-separation)."""
    def nll(b):
        z = X @ b
        return -(y * norm.logcdf(z) + (1 - y) * norm.logcdf(-z)).sum() + 0.5 * ridge * (b[1:] ** 2).sum()

    def grad(b):
        z = X @ b
        lam1 = np.exp(norm.logpdf(z) - norm.logcdf(z))
        lam0 = np.exp(norm.logpdf(z) - norm.logcdf(-z))
        g = -(X.T @ (y * lam1 - (1 - y) * lam0))
        g[1:] += ridge * b[1:]
        return g
    return minimize(nll, np.zeros(X.shape[1]), jac=grad, method="BFGS", options={"gtol": 1e-7, "maxiter": 500}).x


def resp_prob(r: dict) -> float:
    p = BB.probs(r.get("lp_sum") or r["top20"])
    return p.get("A", 0.0)


def glm_rows(recs: list[dict], use_size: bool) -> dict:
    """Arrays for the probit: says subject front; phrasing, occlusion, vertical, size codes (+-1); pair; score."""
    rows = []
    for r in recs:
        qm = QRX.search(r["question"]) if "question" in r else None
        mt = r["meta"]
        x = (qm.group(1) if qm else mt["x_name"]).lower()
        g = geom_meta(mt, x)
        behind = (r.get("phrasing") == "behind") or (qm is not None and qm.group(2) == "behind")
        pa = resp_prob(r)
        says = (pe(r) == "A") != behind
        score = (pa if not behind else 1 - pa)
        rows.append((float(says), -1.0 if behind else 1.0, 1.0 if g["x_front"] else -1.0, 1.0 if g["x_lower"] else -1.0,
                     1.0 if g["x_larger"] else -1.0, r["image_id"], score))
    a = np.array([x[:5] for x in rows])
    cols = [np.ones(len(a)), a[:, 1], a[:, 2], a[:, 3]] + ([a[:, 4]] if use_size else [])
    return {"y": a[:, 0], "X": np.column_stack(cols), "pair": [x[5] for x in rows], "score": np.array([x[6] for x in rows]),
            "truth": a[:, 2] > 0}


def auc(score: np.ndarray, truth: np.ndarray) -> float:
    from scipy.stats import rankdata
    r = rankdata(score)
    n1, n0 = truth.sum(), (~truth).sum()
    return float((r[truth].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def glm_set(recs_by_cond: dict, use_size: bool, n_boot: int = 300) -> dict:
    names = ["intercept", "phrasing", "occlusion", "vertical"] + (["size"] if use_size else [])
    data = {c: glm_rows(rs, use_size) for c, rs in recs_by_cond.items()}
    pairs = sorted(set(data[next(iter(data))]["pair"]))
    est = {c: probit_fit(d["X"], d["y"]) for c, d in data.items()}
    pidx = {c: {p: np.where(np.array(d["pair"]) == p)[0] for p in pairs} for c, d in data.items()}
    boots = {c: [] for c in data}
    aucb = {c: [] for c in data}
    for _ in range(n_boot):
        js = RNG.integers(0, len(pairs), len(pairs))
        for c, d in data.items():
            s = np.concatenate([pidx[c][pairs[j]] for j in js])
            boots[c].append(probit_fit(d["X"][s], d["y"][s]))
            aucb[c].append(auc(d["score"][s], d["truth"][s]))
    out = {}
    for c in data:
        bb = np.array(boots[c])
        out[c] = {k: {"dprime": round(2 * float(est[c][i]), 2), "ci": [round(2 * v, 2) for v in q(bb[:, i])]}
                  for i, k in enumerate(names) if k not in ("intercept", "phrasing")}
        out[c]["max_abs_coef"] = round(float(np.abs(est[c][2:]).max()), 2)
        out[c]["separated"] = bool(np.abs(est[c][2:]).max() > 2.5)
        out[c]["auc"] = {"auc": round(auc(data[c]["score"], data[c]["truth"]), 3), "ci": q(aucb[c])}
        out[c]["acc"] = round(100 * float(np.mean(data[c]["y"] == data[c]["truth"])), 1)
    if "direct" in data:
        for c in [c for c in data if c != "direct"]:
            d0, d1 = np.array(boots["direct"]), np.array(boots[c])
            out[f"{c}-direct"] = {k: {"dprime": round(2 * float(est[c][i] - est["direct"][i]), 2),
                                      "ci": [round(2 * v, 2) for v in q(d1[:, i] - d0[:, i])]}
                                  for i, k in enumerate(names) if k not in ("intercept", "phrasing")}
            out[f"{c}-direct"]["auc"] = {"diff": round(out[c]["auc"]["auc"] - out["direct"]["auc"]["auc"], 3),
                                         "ci": q(np.array(aucb[c]) - np.array(aucb["direct"]))}
    return out


def glm(lines: list[str]) -> dict:
    res = {"cues": {}, "twins": {}, "ladder": {}}
    for m in M7:
        R = rj(O / "e10" / f"{m}_cues.jsonl")
        by = {c: [r for r in R if r["condition"] == c] for c in ("direct", "cot")}
        res["cues"][m] = g = glm_set(by, use_size=True)
        d = g["cot-direct"]
        lines.append(f"GLM cues {m:14s} Direct occ {g['direct']['occlusion']['dprime']:+.2f} vert {g['direct']['vertical']['dprime']:+.2f} "
                     f"size {g['direct']['size']['dprime']:+.2f} (sep {g['direct']['separated']}) | CoT-D occ {d['occlusion']['dprime']:+.2f} {d['occlusion']['ci']} "
                     f"vert {d['vertical']['dprime']:+.2f} {d['vertical']['ci']} size {d['size']['dprime']:+.2f} {d['size']['ci']} | CoT vert {g['cot']['vertical']['dprime']:+.2f} {g['cot']['vertical']['ci']} "
                     f"| AUC {g['direct']['auc']['auc']}->{g['cot']['auc']['auc']}")
        R = [r for r in rj(O / "e10" / f"{m}.jsonl") if r["task"].startswith("occ")]
        by = {c: [r for r in R if r["condition"] == c] for c in ("direct", "cot")}
        res["twins"][m] = g = glm_set(by, use_size=False)
        d = g["cot-direct"]
        lines.append(f"GLM twins {m:14s} CoT-D occ {d['occlusion']['dprime']:+.2f} {d['occlusion']['ci']} vert {d['vertical']['dprime']:+.2f} {d['vertical']['ci']} "
                     f"| CoT vert {g['cot']['vertical']['dprime']:+.2f} {g['cot']['vertical']['ci']}")
    for s in ("2b", "4b", "32b"):
        recs = []
        for ph, fn in (("front", f"qwen3vl_{s}.jsonl"), ("behind", f"qwen3vl_{s}_behind.jsonl")):
            for r in rj(O / "scale" / fn):
                mt = r["meta"]
                qq = f"Is the {mt['x_name']} {'in front of' if ph == 'front' else 'behind'} the {mt['y_name']}?"
                recs.append(r | {"phrasing": ph, "question": qq})
        by = {c: [r for r in recs if r["condition"] == c] for c in ("direct", "cot")}
        res["ladder"][s] = g = glm_set(by, use_size=False)
        d = g["cot-direct"]
        lines.append(f"GLM ladder {s:4s} Direct occ {g['direct']['occlusion']['dprime']:+.2f} {g['direct']['occlusion']['ci']} (max|b| {g['direct']['max_abs_coef']}) "
                     f"acc {g['direct']['acc']} AUC {g['direct']['auc']['auc']} | CoT occ {g['cot']['occlusion']['dprime']:+.2f} acc {g['cot']['acc']} AUC {g['cot']['auc']['auc']} "
                     f"| CoT-D occ {d['occlusion']['dprime']:+.2f} {d['occlusion']['ci']} AUC diff {d['auc']['diff']:+.3f} {d['auc']['ci']}")
    return res


# ============================================================================= 5. commit nulls

def commit_nulls(lines: list[str], n_perm: int = 4000) -> dict:
    from rigor_e17 import stype
    rng = np.random.default_rng(1)
    base = json.load(open(B / "commit_baseline.json"))
    res = {}
    for m in M7:
        G = BB.groups_from({}, L, O, m)
        recs = [r for r in rj(O / "e2b" / f"{m}.jsonl") if G.get(r["item_id"], {}).get("group") == "D+C-"]
        items = []
        for r in recs:
            ts = [stype(s) for s in r["sentences"]]
            pg = [BB.probs(t.get("lp_sum") or t["lp_bare"])[r["gold"]] for t in sorted(r["traj"], key=lambda x: x["j"])]
            js = next((j for j in range(1, len(pg)) if pg[j - 1] >= 0.5 and all(v < 0.5 for v in pg[j:])), None)
            if js is None or not ts:
                continue
            items.append({"types": ts, "j": js - 1})
        rel = [np.array([t == "relation" for t in it["types"]]) for it in items]
        obs = np.mean([it["types"][it["j"]] == "relation" for it in items])
        # position-matched: take the sentence at the same relative position in a random other D+C- rationale
        relpos = np.array([it["j"] / max(1, len(it["types"]) - 1) for it in items])
        lens = np.array([len(x) for x in rel])
        null = np.zeros(n_perm)
        for k in range(len(items)):
            others = rng.integers(0, len(items), n_perm)
            pos = np.rint(relpos[k] * (lens[others] - 1)).astype(int)
            null += np.array([rel[o][p] for o, p in zip(others, pos)])
        null /= len(items)
        b = base[m]
        res[m] = {"obs_committed": round(100 * obs, 1),
                  "uniform": {"null": b["null_uniform_mean"], "excess": round(b["commit_relational_share_of_committed"] - b["null_uniform_mean"], 1)},
                  "nonconclusion": {"obs": b["nonconcl_obs"], "null": b["nonconcl_null_mean"], "excess": round(b["nonconcl_obs"] - b["nonconcl_null_mean"], 1)},
                  "position_matched": {"null": round(100 * null.mean(), 1), "null_95": [round(100 * v, 1) for v in q(null)],
                                       "excess": round(100 * (obs - null.mean()), 1),
                                       "perm_p": float((np.sum(null >= obs) + 1) / (n_perm + 1))}}
        r = res[m]
        lines.append(f"COMMIT {m:14s} obs {r['obs_committed']} | uniform +{r['uniform']['excess']} | nonconcl +{r['nonconclusion']['excess']} "
                     f"| position-matched null {r['position_matched']['null']} excess {r['position_matched']['excess']:+.1f} p {r['position_matched']['perm_p']:.4f}")
    return res


# ============================================================================= 8. closed facts

def closed_facts(lines: list[str]) -> dict:
    s = json.load(open(B / "e9x_summary.json"))
    ds = s["deepseek-v4.1-flash"]
    res = {"deepseek_twins": {"occ_sdt": ds["occ_sdt"], "occ_sdt_diff": ds["occ_sdt_diff"]}}
    p = O / "e9h" / "summary.json"
    if p.exists():
        h = json.load(open(p))
        qm = h.get("qwen3.8-max-0902", {})
        res["qwen38max_hard"] = {"occ_sdt": qm.get("occ_sdt"), "occ_sdt_diff": qm.get("occ_sdt_diff"),
                                 "occ_both": qm.get("occ_both")}
    lines.append(f"CLOSED {json.dumps(res)[:1500]}")
    return res


def main() -> None:
    import sys
    lines: list[str] = []
    if "--fig-only" in sys.argv:
        fig_answer_step(json.load(open(B / "relation_sensitivity.json")))
        fig_answer_step_compact(json.load(open(B / "relation_sensitivity.json")))
        return
    if "--glm-only" in sys.argv:
        json.dump(glm(lines), open(B / "glm_decomp.json", "w"), indent=1)
        old = (B / "rigor_e17f.txt").read_text().splitlines()
        (B / "rigor_e17f.txt").write_text("\n".join([x for x in old if not x.startswith("GLM")] + lines) + "\n")
        print("\n".join(lines))
        return
    rs = relation_sensitivity(lines)
    json.dump(rs, open(B / "relation_sensitivity.json", "w"), indent=1)
    fig_answer_step(rs)
    json.dump(cue_claims(lines), open(B / "cue_claims.json", "w"), indent=1)
    json.dump(commit_nulls(lines), open(B / "commit_nulls.json", "w"), indent=1)
    json.dump(closed_facts(lines), open(B / "closed_facts.json", "w"), indent=1)
    json.dump(glm(lines), open(B / "glm_decomp.json", "w"), indent=1)
    (B / "rigor_e17f.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
