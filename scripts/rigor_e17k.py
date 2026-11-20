"""Exp-17k CPU analyses (round-7 Opus M4-M6, m15).

locus     per model on the cue set and the original twins: Direct and CoT accuracy, P(first claim correct) (claim_parse
          v3, items with a claim), P(answer correct | claim correct), P(answer correct | claim wrong), with a
          per-model classification of where the loss arises.
deepseek  DeepSeek-V4.1-Flash reasoning-free baselines (hard set and twins) with CIs.
fisher    BLINK binding errors, CoT-wrong vs CoT-right (Fisher exact and permutation).
stale     rewrite heldout_mean.json / heldout_extra_means.json as mirrors of e17j_natural.json (one computation).

Usage: python rigor_e17k.py
"""
from __future__ import annotations
import vtax_paths as _vtp  # noqa: E402

import json
from pathlib import Path

import numpy as np
from scipy.stats import fisher_exact

from build_bundle import pe, rj
from claim_parse import first_claim_v3
from e2f_claim_edit import split_sents
from rigor_e17f import QRX, geom_meta

O = Path(f"{_vtp.OUTPUTS}")
B = Path("outputs/bundle")
M7 = ["qwen3b", "qwen7b", "internvl3_8b", "phi35v", "molmo7b", "qwen3vl_8b", "internvl35_8b"]
RNG = np.random.default_rng(0)


def boot(v: np.ndarray, g: np.ndarray, n: int = 2000) -> dict:
    if len(v) == 0:
        return {"mean": None, "ci": None, "n": 0}
    u = np.unique(g)
    idx = {x: np.where(g == x)[0] for x in u}
    bs = [v[np.concatenate([idx[x] for x in RNG.choice(u, len(u))])].mean() for _ in range(n)]
    return {"mean": round(100 * float(v.mean()), 1), "ci": [round(100 * float(np.quantile(bs, .025)), 1), round(100 * float(np.quantile(bs, .975)), 1)], "n": int(len(v))}


def locus(lines: list[str]) -> dict:
    res = {}
    for which in ("cues", "twins"):
        for m in M7:
            fn = O / "e10" / (f"{m}_cues.jsonl" if which == "cues" else f"{m}.jsonl")
            R = [r for r in rj(fn) if r["task"].startswith("occ") and r["condition"] in ("direct", "cot")]
            by = {}
            for r in R:
                by.setdefault(r["item_id"], {})[r["condition"]] = r
            rows = []
            for k, d in by.items():
                if "direct" not in d or "cot" not in d:
                    continue
                r = d["cot"]
                x = QRX.search(r["question"]).group(1).lower()
                g = geom_meta(r["meta"], x)
                fc = first_claim_v3(split_sents(r["text"][len("Let's think step by step."):]))
                rows.append((pe(d["direct"]) == r["gold"], pe(r) == r["gold"], fc is not None, bool(fc and fc[1] == g["front"]), r["image_id"]))
            a = np.array([t[:4] for t in rows], bool)
            gp = np.array([t[4] for t in rows])
            dacc, cacc, has, cc = a[:, 0], a[:, 1], a[:, 2], a[:, 3]
            out = {"n": len(rows), "direct_acc": boot(dacc.astype(float), gp), "cot_acc": boot(cacc.astype(float), gp),
                   "claim_rate": round(100 * float(has.mean()), 1),
                   "p_claim_correct": boot(cc[has].astype(float), gp[has]),
                   "p_ans_given_claim_correct": boot(cacc[has & cc].astype(float), gp[has & cc]),
                   "p_ans_given_claim_wrong": boot(cacc[has & ~cc].astype(float), gp[has & ~cc]),
                   "cot_acc_claim_items": boot(cacc[has].astype(float), gp[has]),
                   "direct_acc_claim_items": boot(dacc[has].astype(float), gp[has])}
            # same claim-bearing subset for all three, so claim-stage + post-claim = Direct - CoT exactly (rounded inputs)
            D, Cl, A = out["direct_acc_claim_items"]["mean"], out["p_claim_correct"]["mean"], out["cot_acc_claim_items"]["mean"]
            claim_loss, post_loss = round(D - Cl, 1), round(Cl - A, 1)
            out["claim_stage_loss"], out["post_claim_loss"] = claim_loss, post_loss
            if D < 58:
                cls = "Direct near chance: no loss to localize"
            elif claim_loss < 3 and post_loss < 3:
                cls = "little loss on this set"
            elif post_loss >= 3 and claim_loss < 3:
                cls = "loss after the first claim (later text or readout)"
            elif claim_loss >= 3 and post_loss < 3:
                cls = "loss at claim formation"
            else:
                cls = "loss both at the first claim and after it"
            out["classification"] = cls
            res.setdefault(which, {})[m] = out
            lines.append(f"LOCUS {which:5s} {m:14s} D {D} | claim {Cl} (rate {out['claim_rate']}) | P(ans|claim ok) {out['p_ans_given_claim_correct']['mean']} "
                         f"P(ans|claim wrong) {out['p_ans_given_claim_wrong']['mean']} | CoT {out['cot_acc']['mean']} (claim items {A}) | claim-stage {claim_loss:+.1f} post-claim {post_loss:+.1f} -> {cls}")
    return res


def deepseek(lines: list[str]) -> dict:
    h = json.load(open(O / "e9h" / "summary.json"))["deepseek-v4.1-flash"]
    t = json.load(open(B / "e9x_summary.json"))["deepseek-v4.1-flash"]
    res = {"hard": {"occ_sdt": h["occ_sdt"], "occ_sdt_diff": h["occ_sdt_diff"], "acc": {k: v.get("acc") for k, v in h["occ_both"].items() if isinstance(v, dict)}},
           "twins": {"occ_sdt": t["occ_sdt"], "occ_sdt_diff": t["occ_sdt_diff"]}}
    nr = h["occ_sdt"]["direct_nr"]
    res["hard_baseline_above_chance"] = bool(nr["ci"][0] > 0)
    lines.append(f"DEEPSEEK hard direct_nr d' {nr['dprime']} {nr['ci']} acc {res['hard']['acc']['direct_nr']}; hidden reasoning "
                 f"{-h['occ_sdt_diff']['direct_nr_minus_direct'][0]:+.2f} [{-h['occ_sdt_diff']['direct_nr_minus_direct'][2]:+.2f}, {-h['occ_sdt_diff']['direct_nr_minus_direct'][1]:+.2f}]; "
                 f"twins {-t['occ_sdt_diff']['direct_nr_minus_direct'][0]:+.2f} [{-t['occ_sdt_diff']['direct_nr_minus_direct'][2]:+.2f}, {-t['occ_sdt_diff']['direct_nr_minus_direct'][1]:+.2f}]")
    return res


def fisher(lines: list[str]) -> dict:
    J = {j: {(r["model"], r["item_id"]): r for r in rj(O / "e19" / f"{j}.jsonl")} for j in ("claude-opus-5-5", "gpt-6-astra")}
    keys = sorted(set(J["claude-opus-5-5"]) & set(J["gpt-6-astra"]))
    err = {k: all(J[j][k]["label"] in ("SWAPPED", "MISNAMED") for j in J) for k in keys}
    grp = {k: J["gpt-6-astra"][k]["group"] for k in keys}
    a = sum(err[k] for k in keys if grp[k] == "cot_wrong"); n1 = sum(grp[k] == "cot_wrong" for k in keys)
    b = sum(err[k] for k in keys if grp[k] == "cot_right"); n2 = sum(grp[k] == "cot_right" for k in keys)
    odds, p = fisher_exact([[a, n1 - a], [b, n2 - b]])
    lab = np.array([err[k] for k in keys], float)
    g = np.array([grp[k] == "cot_wrong" for k in keys])
    obs = lab[g].mean() - lab[~g].mean()
    perm = [(lambda s: lab[s].mean() - lab[~s].mean())(RNG.permutation(g)) for _ in range(10000)]
    pp = float((np.sum(np.abs(perm) >= abs(obs)) + 1) / 10001)
    res = {"cot_wrong": [a, n1], "cot_right": [b, n2], "diff_pts": round(100 * obs, 1), "fisher_p": round(float(p), 3), "odds_ratio": round(float(odds), 2),
           "perm_p_two_sided": round(pp, 3)}
    lines.append(f"FISHER binding {a}/{n1} vs {b}/{n2}: diff {res['diff_pts']} pts, Fisher p {res['fisher_p']}, permutation p {res['perm_p_two_sided']}")
    return res


def stale(lines: list[str]) -> dict:
    nat = json.load(open(B / "e17j_natural.json"))
    note = "mirror of e17j_natural.json (image-cluster bootstrap, seed 0 per cell, 4,000 resamples); kept for backward compatibility"
    json.dump({"_note": note, "blink_depth": nat["blink_depth"], "depth324": nat["depth324"]}, open(B / "heldout_mean.json", "w"), indent=1)
    json.dump({"_note": note, "blink_relation": nat["blink_relation"], "3dsr_depth": nat["3dsr_depth"]}, open(B / "heldout_extra_means.json", "w"), indent=1)
    lines.append("STALE heldout_mean.json and heldout_extra_means.json rewritten as mirrors of e17j_natural.json")
    return {"rewritten": ["heldout_mean.json", "heldout_extra_means.json"], "note": note}


def main() -> None:
    lines: list[str] = []
    out = {"locus": locus(lines), "deepseek": deepseek(lines), "fisher": fisher(lines), "stale": stale(lines)}
    json.dump(out, open(B / "e17k.json", "w"), indent=1)
    (B / "rigor_e17k.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
