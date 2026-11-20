"""E17 summary: image controls and matched baselines for the within-claim truth probe (round-6 Opus M1).

Reads $W/outputs/e17/<model>_ctrl.json (pod analysis) and e17 meta; recomputes the text-only (TF-IDF of the CoT up to
the relation word) and cue-only (subject lower/larger, phrasing) baselines with claim_parse v3 strata, reporting the
same two metrics as the probe: balanced accuracy and recall on D+C- wrong claims.
Writes outputs/bundle/probe_controls.json and paper/tables/supp_probe_controls.tex.
"""
from __future__ import annotations
import vtax_paths as _vtp  # noqa: E402

import glob
import json
import math
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold

from build_bundle import rj
from e17_probe_ctrl_lite import claim_spans

O = Path(f"{_vtp.OUTPUTS}")
B = Path("outputs/bundle")
T = Path(f"{_vtp.TABLES}")
MODELS = {"qwen3vl_8b": "Qwen3-VL-8B", "molmo7b": "Molmo-7B-D", "qwen7b": "Qwen2.5-VL-7B"}
RNG = np.random.default_rng(0)


def pe(lp):
    fin = {k: v for k, v in lp.items() if v is not None and math.isfinite(v)}
    return max(fin, key=fin.get) if fin else None


def metrics(pr, y, claim, sel, dcm, pair, n=500):
    m = sel & (pr >= 0)
    w, r = m & (claim != y), m & (claim == y)
    dw = w & dcm
    ok = (pr == y).astype(float)
    u = np.unique(pair[m])
    idx = {g: np.where((pair == g) & m)[0] for g in u}
    stat = lambda ii: [(ok[ii][w[ii]].mean() + ok[ii][r[ii]].mean()) / 2, ok[ii][dw[ii]].mean() if dw[ii].any() else np.nan]
    pt = stat(np.where(m)[0])
    bs = np.array([stat(np.concatenate([idx[g] for g in RNG.choice(u, len(u))])) for _ in range(n)])
    q = lambda j: [round(float(np.nanquantile(bs[:, j], .025)), 3), round(float(np.nanquantile(bs[:, j], .975)), 3)]
    return {"bal": round(float(pt[0]), 3), "bal_ci": q(0), "recall_DCm_wrong": round(float(pt[1]), 3), "recall_DCm_wrong_ci": q(1)}


def baselines(model: str) -> dict:
    meta = []
    for f in sorted(glob.glob(str(O / "e17" / "meta" / f"{model}_[0-9].jsonl"))):
        meta += [json.loads(l) for l in open(f)]
    cot = {r["item_id"]: r["text"] for r in rj(O / "e10" / f"{model}_cues.jsonl") if r["condition"] == "cot"}
    y = np.array([m["y_true"] for m in meta])
    pair = np.array([m["pair"] for m in meta])
    hc = np.array([m["has_claim"] for m in meta], bool)
    claim = np.array([int(m["named"] == m["x"]) if m["has_claim"] else -1 for m in meta])
    dcm = hc & np.array([pe(m["lp_direct"]) == m["gold"] for m in meta]) & ~np.array([pe(m["lp_cot"]) == m["gold"] for m in meta])
    txt = []
    for m in meta:
        cs = claim_spans(cot[m["key"]]) if m["has_claim"] else None
        txt.append(cot[m["key"]][: cs["rel_end"] + 1] if cs else "")
    txt = np.array(txt, dtype=object)
    xl = np.array([(m["y_true"] == 1) == (not m["front_above"]) for m in meta], float) * 2 - 1
    xg = np.array([(m["y_true"] == 1) == bool(m["front_larger"]) for m in meta], float) * 2 - 1
    ph = np.array([m["phrasing"] == "front" for m in meta], float) * 2 - 1
    Z = np.column_stack([xl, xg, ph, xl * ph, xg * ph, xl * xg])
    out = {}
    for name in ("text", "cue"):
        pr = np.full(len(y), -1)
        for c in (0, 1):
            idx = np.where(hc & (claim == c))[0]
            for tr, te in GroupKFold(5).split(idx, y[idx], pair[idx]):
                if name == "text":
                    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2)
                    clf = LogisticRegression(C=1.0, max_iter=3000, class_weight="balanced").fit(vec.fit_transform(txt[idx[tr]]), y[idx[tr]])
                    pr[idx[te]] = clf.predict(vec.transform(txt[idx[te]]))
                else:
                    clf = LogisticRegression(C=1.0, max_iter=3000, class_weight="balanced").fit(Z[idx[tr]], y[idx[tr]])
                    pr[idx[te]] = clf.predict(Z[idx[te]])
        out[name] = metrics(pr, y, claim, hc, dcm, pair)
    return out


def main() -> None:
    res = {}
    for m, name in MODELS.items():
        d = json.load(open(O / "e17" / f"{m}_ctrl.json"))
        lay = d["layers"]
        best = max(lay, key=lambda r: r["orig"]["r"]["bal"])
        l0 = lay[0]
        res[m] = {"best_layer": best["layer"], "n_claims": d["n_claims"],
                  "orig": best["orig"], "gray_refit": best["gray_refit"], "gray_xfer": best["gray_xfer"], "twin_xfer": best["twin_xfer"],
                  "filler": best.get("filler"), "layer0_r": l0["orig"]["r"], "layer0_a": l0["orig"]["a"],
                  "direct_probe_at_answer": best["direct_probe_at_answer"], "baselines": baselines(m),
                  "per_layer": [{"layer": r["layer"], "orig_r": r["orig"]["r"]["bal"], "orig_a": r["orig"]["a"]["bal"], "orig_rprev": r["orig"]["rprev"]["bal"],
                                 "orig_s": r["orig"]["s"]["bal"], "orig_pre": r["orig"]["pre"]["bal"], "gray_refit_r": r["gray_refit"]["r"]["bal"],
                                 "gray_xfer_r": r["gray_xfer"]["r"]["bal"], "twin_follows_shown_r": r["twin_xfer"]["r"]["follows_shown_image"],
                                 "filler": (r.get("filler") or {}).get("bal")} for r in lay]}
        r = res[m]
        print(m, "L", r["best_layer"], "| orig r", r["orig"]["r"]["bal"], r["orig"]["r"]["recall_DCm_wrong"], "| a", r["orig"]["a"]["bal"], r["orig"]["a"]["recall_DCm_wrong"],
              "| rprev", r["orig"]["rprev"]["bal"], "s", r["orig"]["s"]["bal"], "pre", r["orig"]["pre"]["bal"], "| gray refit r", r["gray_refit"]["r"]["bal"],
              r["gray_refit"]["r"]["recall_DCm_wrong"], "xfer", r["gray_xfer"]["r"]["bal"], "| twin follows shown", r["twin_xfer"]["r"]["follows_shown_image"],
              r["twin_xfer"]["r"]["follows_shown_image_DCm_wrong"], "| filler", (r["filler"] or {}).get("bal"), (r["filler"] or {}).get("recall_DCm_wrong"),
              "| L0 r", r["layer0_r"]["bal"], r["layer0_r"]["recall_DCm_wrong"], "| text", r["baselines"]["text"], "| cue", r["baselines"]["cue"],
              "| Dprobe@a", r["direct_probe_at_answer"])
    json.dump(res, open(B / "probe_controls.json", "w"), indent=1)
    f = lambda x: f"{x['bal']:.2f} / {x['recall_DCm_wrong']:.2f}" if x else "--"
    fc = lambda x: f"{x['bal']:.2f} [{x['bal_ci'][0]:.2f}, {x['bal_ci'][1]:.2f}] / {x['recall_DCm_wrong']:.2f} [{x['recall_DCm_wrong_ci'][0]:.2f}, {x['recall_DCm_wrong_ci'][1]:.2f}]"
    rows = [("Original image, relation token ($r$)", lambda r: fc(r["orig"]["r"])),
            ("Original image, answer token ($a$)", lambda r: fc(r["orig"]["a"])),
            ("Original image, token writing the relation ($r{-}1$)", lambda r: f(r["orig"]["rprev"])),
            ("Original image, subject noun ($s$)", lambda r: f(r["orig"]["s"])),
            ("Original image, token writing the subject ($s{-}1$)", lambda r: f(r["orig"]["pre"])),
            ("Gray image, probe refit ($r$)", lambda r: fc(r["gray_refit"]["r"])),
            ("Gray image, original probe ($r$)", lambda r: f(r["gray_xfer"]["r"])),
            ("Twin image, original probe ($r$): follows shown image", lambda r: f"{r['twin_xfer']['r']['follows_shown_image']:.2f} (all items)"),
            ("Twin image, original probe ($a$): follows shown image", lambda r: f"{r['twin_xfer']['a']['follows_shown_image']:.2f} (all items)"),
            ("Filler-matched token", lambda r: f(r["filler"])),
            ("Layer 0 (lexical), $r$", lambda r: f(r["layer0_r"])),
            ("CoT text up to the relation word (TF-IDF)", lambda r: f(r["baselines"]["text"])),
            ("Cue/phrasing covariates", lambda r: f(r["baselines"]["cue"]))]
    lines = [r"\begin{tabular}{l" + "c" * len(MODELS) + "}", r"\toprule",
             "Within-claim truth probe (balanced acc.\\ / recall on D+C$-$ wrong claims) & " + " & ".join(f"{n} (L{res[m]['best_layer']})" for m, n in MODELS.items()) + r" \\",
             r"\midrule"]
    for lab, fn in rows:
        lines.append(lab + " & " + " & ".join(fn(res[m]) for m in MODELS) + r" \\")
    lines.append(r"\midrule")
    lines.append("Direct-answer probe at $a$: share following the wrong CoT answer (D+C$-$) & " +
                 " & ".join(f"{res[m]['direct_probe_at_answer']['follows_cot_answer_DCm_wrong']:.2f}" for m in MODELS) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (T / "supp_probe_controls.tex").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
