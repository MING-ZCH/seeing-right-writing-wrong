"""Exp-22 analysis: commit-sentence gate vs Direct / plain CoT / random-resample control / text gate.

Per model x set x arm (outputs/e22/<model>_<set>.jsonl, all arms from one engine run):
  accuracy with paired cluster-bootstrap CIs vs Direct and vs CoT (clusters: twin pair / Synthetic Depth image);
  pooled d' on "the queried subject is in front" (cues / twins: over both phrasings; syn: over both object orders);
  Synthetic Depth pair-matched d'_pair = ln[(BC+.5)/(BW+.5)]/1.702 (rigor_e17p);
  cue-set probit-GLM occlusion / vertical / size components (rigor_e17f.glm_set);
  churn vs Direct (D+C-, D-C+), repairs of CoT's D+C- items, rescue loss on CoT's D-C+ items;
  gating statistics (share gated, events, resamples / drops, tries, cap hits), compute (generated tokens, readouts);
  rationale-answer consistency (final answer = answer implied by the last parsed depth-order claim, claim_parse v3).
-> outputs/bundle/e22_gate.json, outputs/bundle/rigor_e22.txt, outputs/bundle/supp_gate.tex
Usage: python rigor_e22.py [--nboot 2000] [--glm-boot 200]
"""
from __future__ import annotations
import vtax_paths as _vtp  # noqa: E402

import argparse
import json
import math
import re
from pathlib import Path

import numpy as np
from scipy.stats import norm

from claim_parse import first_claim_v3

# sentence split of e2b.py (importing e2b pulls in the pod-side vx / legacy common)
_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z(\"'*\d])|\n+")
_ENUM = re.compile(r"^\s*(\d+|[a-z])[.)]\s*$|^\s*[-*]+\s*$")


def split_sentences(text: str) -> list[str]:
    raw = [s.strip() for s in _SPLIT.split(text) if s and s.strip()]
    out, carry = [], ""
    for s in raw:
        if _ENUM.match(s):
            carry = (carry + " " + s).strip()
            continue
        out.append((carry + " " + s).strip() if carry else s)
        carry = ""
    return out

O = Path(f"{_vtp.OUTPUTS}")
B = Path("outputs/bundle")
MODELS = ["qwen3vl_8b", "qwen7b", "internvl3_8b", "molmo7b"]
LABEL = {"qwen3vl_8b": "Qwen3-VL-8B", "qwen7b": "Qwen2.5-VL-7B", "internvl3_8b": "InternVL3-8B", "molmo7b": "Molmo-7B-D"}
SETS = ["cues", "twins", "syn"]
SLABEL = {"cues": "Cue-factorial", "twins": "Orig. twins", "syn": "Syn. Depth (both orders)"}
ARMS = ["direct", "cot", "gate", "rand", "tgate"]
ALABEL = {"direct": "Direct", "cot": "CoT", "gate": "Gate", "rand": "Rand.", "tgate": "Text gate"}
LOGIT_PROBIT = 1.702
MAX_EVENTS = 8
COT_PREFIX = "Let's think step by step."
SFX = ""


def pred(r: dict) -> str | None:
    lp = r.get("lp_sum") or r["top20"]
    fin = {k: v for k, v in lp.items() if v is not None and math.isfinite(v)}
    return max(fin, key=fin.get) if fin else None


CONCL = re.compile(r"\b(answer is|answer:|therefore|thus|hence|so the|in conclusion|the correct (option|answer)|\([AB]\))", re.I)


def sent_claim(s: str, r: dict) -> str | None:
    """Answer implied by one sentence's two-object depth-order claim (claim_parse v3)."""
    c = first_claim_v3([s])
    if c is None or c[1] not in (r["x"], r["y"]):
        return None
    x_front = c[1] == r["x"]
    yes = x_front if r["phrasing"] == "front" else not x_front
    return "A" if yes else "B"


def claim_answers(r: dict) -> list[str]:
    """Answers implied by every parsable claim in the rationale, in order."""
    sents = split_sentences(r["text"][len(COT_PREFIX):].strip())
    return [a for a in (sent_claim(s, r) for s in sents) if a is not None]


def last_claim_answer(r: dict) -> str | None:
    """Answer implied by the last sentence with a parsable two-object depth-order claim."""
    ca = claim_answers(r)
    return ca[-1] if ca else None


def dprime(says: np.ndarray, tru: np.ndarray) -> float:
    h = (says[tru].sum() + .5) / (tru.sum() + 1)
    f = (says[~tru].sum() + .5) / ((~tru).sum() + 1)
    return float(norm.ppf(h) - norm.ppf(f))


def q95(v) -> list[float]:
    return [float(np.nanquantile(v, .025)), float(np.nanquantile(v, .975))]


def load(m: str, s: str) -> dict | None:
    p = O / "e22" / f"{m}_{s}{SFX}.jsonl"
    if not p.exists():
        return None
    R = [json.loads(l) for l in open(p)]
    by = {}
    for r in R:
        by.setdefault(r["arm"], {})[r["item_id"]] = r
    arms = [a for a in ARMS if a in by]
    keys = sorted(by["direct"])
    clusters = sorted({by["direct"][k]["image_id"] for k in keys})
    cid = {c: i for i, c in enumerate(clusters)}
    d = {"by": by, "arms": arms, "keys": keys, "cl": np.array([cid[by["direct"][k]["image_id"]] for k in keys]),
         "ncl": len(clusters)}
    r0 = [by["direct"][k] for k in keys]
    if s == "syn":
        # truth: queried subject (x) is in front <=> gold yes (both orders)
        d["tru"] = np.array([r["gold"] == "A" for r in r0])
    else:
        d["tru"] = np.array([r["meta"]["front_name"].lower() == r["x"] for r in r0])
    d["behind"] = np.array([r["phrasing"] == "behind" for r in r0])
    for a in arms:
        p = [pred(by[a][k]) for k in keys]
        d[f"corr_{a}"] = np.array([pp == by["direct"][k]["gold"] for pp, k in zip(p, keys)])
        d[f"says_{a}"] = np.array([(pp == "A") != b for pp, b in zip(p, d["behind"])])
    if s == "syn":
        # per image: index of original and swapped-order item
        o_idx, s_idx = {}, {}
        for i, k in enumerate(keys):
            (s_idx if k.endswith("_cfq") else o_idx)[by["direct"][k]["image_id"]] = i
        imgs = sorted(set(o_idx) & set(s_idx))
        d["pair_o"] = np.array([o_idx[g] for g in imgs])
        d["pair_s"] = np.array([s_idx[g] for g in imgs])
        d["pair_cl"] = np.array([cid[g] for g in imgs])
    return d


def stats(d: dict, s: str, idx: np.ndarray, pidx: np.ndarray | None) -> dict[str, float]:
    """All bootstrap statistics on one (re)sample of items (``idx``) / Synthetic Depth pairs (``pidx``)."""
    out = {}
    for a in d["arms"]:
        out[f"acc_{a}"] = 100 * d[f"corr_{a}"][idx].mean()
        out[f"dp_{a}"] = dprime(d[f"says_{a}"][idx], d["tru"][idx])
        if s == "syn" and pidx is not None:
            co = d[f"corr_{a}"][d["pair_o"][pidx]]
            cs = d[f"corr_{a}"][d["pair_s"][pidx]]
            bc, bw = (co & cs).sum(), (~co & ~cs).sum()
            out[f"dpair_{a}"] = math.log((bc + .5) / (bw + .5)) / LOGIT_PROBIT
    for a in d["arms"]:
        for ref in ("direct", "cot", "rand", "tgate"):
            if ref in ("rand", "tgate") and a != "gate":
                continue
            if a != ref and ref in d["arms"]:
                out[f"dacc_{a}-{ref}"] = out[f"acc_{a}"] - out[f"acc_{ref}"]
                out[f"ddp_{a}-{ref}"] = out[f"dp_{a}"] - out[f"dp_{ref}"]
                if f"dpair_{a}" in out:
                    out[f"ddpair_{a}-{ref}"] = out[f"dpair_{a}"] - out[f"dpair_{ref}"]
    return out


def boot(d: dict, s: str, nb: int, seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    members = [np.where(d["cl"] == c)[0] for c in range(d["ncl"])]
    pmembers = [np.where(d["pair_cl"] == c)[0] for c in range(d["ncl"])] if s == "syn" else None
    full = stats(d, s, np.arange(len(d["keys"])), np.arange(len(d["pair_o"])) if s == "syn" else None)
    bs = {k: [] for k in full}
    for _ in range(nb):
        cs = rng.integers(0, d["ncl"], d["ncl"])
        idx = np.concatenate([members[c] for c in cs])
        pidx = np.concatenate([pmembers[c] for c in cs]) if s == "syn" else None
        for k, v in stats(d, s, idx, pidx).items():
            bs[k].append(v)
    return {k: {"est": round(float(v), 3), "ci": [round(x, 3) for x in q95(bs[k])]} for k, v in full.items()}


def descriptives(d: dict) -> dict:
    by, keys = d["by"], d["keys"]
    out = {}
    cD = d["corr_direct"]
    cC = d.get("corr_cot")
    for a in d["arms"]:
        if a == "direct":
            continue
        rs = [by[a][k] for k in keys]
        cA = d[f"corr_{a}"]
        ev = [r.get("events") or [] for r in rs]
        nev = np.array([len(e) for e in ev])
        types = [x["type"] for e in ev for x in e]
        tries = [x["tries"] for e in ev for x in e if x["type"] == "resample"]
        gen = np.array([r["gen_tok"] for r in rs], float)
        gen_cot = np.array([by["cot"][k]["gen_tok"] for k in keys], float)
        nro = np.array([r["n_readout"] for r in rs], float)
        cas = [claim_answers(r) for r in rs]
        la = [c[-1] if c else None for c in cas]
        pa = [pred(r) for r in rs]
        has = [x is not None for x in la]
        contra = [any(x != p for x in c) for c, p in zip(cas, pa)]
        evs = [x for e in ev for x in e]
        greedy_concl = [bool(CONCL.search(x["greedy"])) for x in evs]
        greedy_claim = [sent_claim(x["greedy"], r) is not None for e, r in zip(ev, rs) for x in e]
        row = {
            "n": len(rs),
            "D+C-": int((cD & ~cA).sum()), "D-C+": int((~cD & cA).sum()),
            "net_vs_direct": int(cA.sum() - cD.sum()),
            "agree_direct_pct": round(100 * float(np.mean([p == r["direct_ans"] for p, r in zip(pa, rs)])), 1),
            "body_tok_mean": round(float(np.mean([r["body_tok"] for r in rs])), 1),
            "gen_tok_mean": round(float(gen.mean()), 1),
            "gen_tok_ratio_vs_cot": round(float(gen.sum() / gen_cot.sum()), 2),
            "readouts_mean": round(float(nro.mean()), 2),
            "consistency_pct": round(100 * float(np.mean([p == x for p, x, h in zip(pa, la, has) if h])), 1) if any(has) else None,
            "claim_parsed_pct": round(100 * float(np.mean(has)), 1),
            "any_claim_contradicts_answer_pct": round(100 * float(np.mean(contra)), 1),
            "states_letter_pct": round(100 * float(np.mean([bool(re.search(r"\([AB]\)", r["text"])) for r in rs])), 1),
        }
        # claim truth vs geometry: a parsed claim is true iff its implied answer equals the gold answer
        tru = [x == r["gold"] for c, r in zip(cas, rs) for x in c]
        row["claims_true_pct"] = round(100 * float(np.mean(tru)), 1) if tru else None
        row["claims_per_item"] = round(len(tru) / len(rs), 2)
        if evs:
            row |= {"event_sent_conclusion_pct": round(100 * float(np.mean(greedy_concl)), 1),
                    "event_sent_claim_pct": round(100 * float(np.mean(greedy_claim)), 1)}
            g_t, k_t = [], []
            for e, r in zip(ev, rs):
                for x in e:
                    ga = sent_claim(x["greedy"], r)
                    if ga is not None:
                        g_t.append(ga == r["gold"])
                    if x["type"] == "resample":
                        ka = sent_claim(x["cands"][x["tries"] - 1], r)
                        if ka is not None:
                            k_t.append(ka == r["gold"])
            row["replaced_sent_claim_true_pct"] = round(100 * float(np.mean(g_t)), 1) if g_t else None
            row["kept_sample_claim_true_pct"] = round(100 * float(np.mean(k_t)), 1) if k_t else None
            row["kept_sample_with_claim"] = len(k_t)
        if a != "cot":
            row |= {"gated_pct": round(100 * float((nev > 0).mean()), 1), "events_mean": round(float(nev.mean()), 2),
                    "events_mean_gated": round(float(nev[nev > 0].mean()), 2) if (nev > 0).any() else 0.0,
                    "resamples": int(types.count("resample")), "drops": int(types.count("drop")),
                    "tries_mean": round(float(np.mean(tries)), 2) if tries else None,
                    "cap_hit_pct": round(100 * float((nev >= MAX_EVENTS).mean()), 1)}
        if cC is not None and a != "cot":
            dcm = cD & ~cC   # CoT's D+C- items
            dcp = ~cD & cC   # CoT's D-C+ rescues
            row |= {"cot_DpCm": int(dcm.sum()), "repaired": int((dcm & cA).sum()),
                    "repaired_pct": round(100 * float((dcm & cA).sum() / max(1, dcm.sum())), 1),
                    "cot_DmCp": int(dcp.sum()), "rescues_lost": int((dcp & ~cA).sum()),
                    "rescues_lost_pct": round(100 * float((dcp & ~cA).sum() / max(1, dcp.sum())), 1),
                    "cot_DpCp_broken": int((cD & cC & ~cA).sum())}
            if a == "gate":
                # where the gate acted: items whose plain CoT trajectory ever crosses P(Direct) 0.5 from above
                cot_rs = [by["cot"][k] for k in keys]
                cross = np.array([any(t0 >= .5 > t1 for t0, t1 in zip(r["traj"][:-1], r["traj"][1:])) for r in cot_rs])
                row["cot_traj_cross_pct"] = round(100 * float(cross.mean()), 1)
                row["p0_below_half_pct"] = round(100 * float(np.mean([r["p0"] < .5 for r in cot_rs])), 1)
        out[a] = row
    return out


def glm(d: dict, nb: int) -> dict:
    import rigor_e17f as F
    F.RNG = np.random.default_rng(0)
    recs = {a: [d["by"][a][k] for k in d["keys"]] for a in d["arms"]}
    g = F.glm_set(recs, use_size=True, n_boot=nb)
    return g


def fmt(x: dict, scale: float = 1.0, nd: int = 1) -> str:
    return f"{scale * x['est']:+.{nd}f} [{scale * x['ci'][0]:+.{nd}f}, {scale * x['ci'][1]:+.{nd}f}]"


def latex(res: dict) -> str:
    """(a) accuracy with paired CIs; (b) sensitivity, gating cost, rescues and consistency."""
    SL = {"cues": "Cues", "twins": "Twins", "syn": "Syn.\\ Depth"}
    L = [r"\begin{table*}[t]", r"\centering", r"\fontsize{6.5}{7.8}\selectfont", r"\setlength{\tabcolsep}{3pt}",
         r"\caption{\textbf{Commit-sentence gate (label-free, training-free).} The rationale is processed sentence by sentence "
         r"with the CoT arm's prompt and greedy decoding; after each sentence the forced readout at \texttt{``Answer: (''} "
         r"(turn closed) is read. \emph{Gate}: a sentence that moves $P(\text{Direct answer})$ from $\ge0.5$ to $<0.5$ is "
         r"resampled ($T{=}0.7$, up to $K{=}4$; first non-flipping sample kept); if all $K$ flip, it is dropped and the greedy text "
         r"after it is kept and re-gated (\cref{sec:supp_gate}). \emph{Rand.}: up to the same "
         r"number of events per item (capped at the plain rationale's sentence count; $2$--$19\%$ fewer in total) at random "
         r"sentences, one sample, no criterion; count-matched controls are in \cref{tab:supp_gate_loc}. \emph{Text}: criterion is the sentence's parsed "
         r"depth-order claim (rule-based parser v3) vs.\ the Direct answer. Final answer read as in CoT (same turn). Cues: cue-factorial "
         r"set ($1{,}600$ prompts); Twins: original occlusion twins ($800$); Syn.\ Depth: $500$ images $\times$ both object orders. "
         r"Accuracy (\%) and differences (points) with $95\%$ paired cluster-bootstrap CIs (twin pairs / images; $2{,}000$ resamples). "
         r"Direct and CoT are re-run in this experiment; Direct matches the main runs except for one Molmo-7B-D answer per set ("
         r"$1{,}410/1{,}600 = 88.1\%$ on Cues and $706/800 = 88.25\%$ on Twins here, against $1{,}411/1{,}600 = 88.2\%$ and "
         r"$705/800 = 88.1\%$ in main-paper Tab.~2 and \cref{tab:supp_twins}).}",
         r"\label{tab:supp_gate}"]
    L += [r"\begin{tabular}{@{}llccccc@{\hspace{6pt}}ccc@{}}", r"\toprule",
          r"\multicolumn{10}{@{}l}{\textbf{(a) Accuracy}} \\",
          r"Model & Set & Direct & CoT & Gate & Rand. & Text & Gate$-$Direct & Gate$-$CoT & Gate$-$Rand. \\", r"\midrule"]
    for m in MODELS:
        for s in SETS:
            r = res.get(m, {}).get(s)
            if not r:
                continue
            b = r["boot"]
            cell = lambda k: fmt(b[k]).replace("-", "$-$") if k in b else "--"
            L.append(f"{LABEL[m] if s == 'cues' else ''} & {SL[s]} & " + " & ".join(f"{b[f'acc_{a}']['est']:.1f}" for a in ("direct", "cot", "gate", "rand", "tgate"))
                     + f" & {cell('dacc_gate-direct')} & {cell('dacc_gate-cot')} & {cell('dacc_gate-rand')} \\\\")
        L.append(r"\addlinespace[1pt]")
    L[-1] = r"\bottomrule"
    L += [r"\end{tabular}", r"\par\vspace{4pt}",
          r"\begin{tabular}{@{}llccccccccc@{}}", r"\toprule",
          r"\multicolumn{11}{@{}l}{\textbf{(b) Sensitivity, cost, rescues, consistency}} \\",
          r"Model & Set & \shortstack{$d'$\\D / CoT / Gate} & \shortstack{$d'$\\Gate$-$Direct} & \shortstack{$d'_{\text{pair}}$\\D / CoT / Gate} & "
          r"\shortstack{Gated\\(\%)} & \shortstack{Tokens\\$\times$CoT} & \shortstack{Read-\\outs} & Repaired & \shortstack{Rescues\\lost} & "
          r"\shortstack{Consist.\\CoT / Gate} \\", r"\midrule"]
    for m in MODELS:
        for s in SETS:
            r = res.get(m, {}).get(s)
            if not r:
                continue
            b, g, c = r["boot"], r["desc"]["gate"], r["desc"]["cot"]
            dp = f"{b['dp_direct']['est']:.2f} / {b['dp_cot']['est']:.2f} / {b['dp_gate']['est']:.2f}"
            dpp = (f"{b['dpair_direct']['est']:.2f} / {b['dpair_cot']['est']:.2f} / {b['dpair_gate']['est']:.2f}" if s == "syn" else "--")
            L.append(f"{LABEL[m] if s == 'cues' else ''} & {SL[s]} & {dp} & {fmt(b['ddp_gate-direct'], nd=2).replace('-', '$-$')} & {dpp} & "
                     f"{g['gated_pct']:.1f} & {g['gen_tok_ratio_vs_cot']:.1f} & {g['readouts_mean']:.1f} & "
                     f"{g['repaired']}/{g['cot_DpCm']} & {g['rescues_lost']}/{g['cot_DmCp']} & "
                     f"{c['consistency_pct']:.1f} / {g['consistency_pct']:.1f} \\\\")
        L.append(r"\addlinespace[1pt]")
    L[-1] = r"\bottomrule"
    L += [r"\end{tabular}",
          r"\par\vspace{2pt}\raggedright\footnotesize $d'$ on ``the queried object is in front'', pooled over both phrasings "
          r"(Cues, Twins) or both object orders (Syn.\ Depth); $d'_{\text{pair}}=\ln[(\mathrm{BC}+.5)/(\mathrm{BW}+.5)]/1.702$ over "
          r"the two orders of each image. Gated: items with $\ge1$ gate event. Tokens$\times$: generated tokens incl.\ discarded "
          r"samples relative to plain CoT. Readouts: forced readouts per item (plain-CoT trajectory included). Repaired: CoT "
          r"\texttt{D+C$-$} items the gate answers correctly. Rescues lost: CoT \texttt{D$-$C+} items (Direct wrong, CoT right) "
          r"the gate answers wrongly. Consist.: final answer agrees with the last parsed depth-order claim (items with a claim).",
          r"\end{table*}"]
    return "\n".join(L) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--nboot", type=int, default=2000)
    ap.add_argument("--glm-boot", type=int, default=200)
    ap.add_argument("--suffix", default="", help="input file suffix (smoke runs), e.g. _limit10")
    ap.add_argument("--out", default=str(B))
    ap.add_argument("--from-json", action="store_true", help="only rewrite supp_gate.tex from an existing e22_gate.json")
    args = ap.parse_args()
    global SFX
    SFX = args.suffix
    out = Path(args.out)
    if args.from_json:
        (out / "supp_gate.tex").write_text(latex(json.load(open(out / "e22_gate.json"))))
        return
    res, lines = {}, []
    for m in MODELS:
        for s in SETS:
            d = load(m, s)
            if d is None:
                lines.append(f"MISSING {m} {s}")
                continue
            r = {"n": len(d["keys"]), "n_clusters": d["ncl"], "arms": d["arms"], "boot": boot(d, s, args.nboot),
                 "desc": descriptives(d)}
            if s == "cues" and args.glm_boot > 0:
                r["glm"] = glm(d, args.glm_boot)
            res.setdefault(m, {})[s] = r
            b, g = r["boot"], r["desc"]
            lines.append(f"== {m} {s} n={r['n']} clusters={r['n_clusters']}")
            lines.append("  acc  " + "  ".join(f"{a} {b[f'acc_{a}']['est']:.1f}" for a in d["arms"]))
            for a in d["arms"]:
                if a == "direct":
                    continue
                lines.append(f"  {a:5s} -Direct {fmt(b[f'dacc_{a}-direct'])}" + (f"  -CoT {fmt(b[f'dacc_{a}-cot'])}" if a != "cot" else "")
                             + (f"  -Rand {fmt(b['dacc_gate-rand'])}  -TextGate {fmt(b['dacc_gate-tgate'])}" if a == "gate" else ""))
            lines.append("  d'   " + "  ".join(f"{a} {b[f'dp_{a}']['est']:.2f}" for a in d["arms"]))
            for a in d["arms"]:
                if a == "direct":
                    continue
                lines.append(f"  d' {a:5s} -Direct {fmt(b[f'ddp_{a}-direct'], nd=2)}" + (f"  -CoT {fmt(b[f'ddp_{a}-cot'], nd=2)}" if a != "cot" else "")
                             + (f"  -Rand {fmt(b['ddp_gate-rand'], nd=2)}" if a == "gate" else ""))
            if s == "syn":
                lines.append("  d'pair " + "  ".join(f"{a} {b[f'dpair_{a}']['est']:.2f}" for a in d["arms"]))
                for a in d["arms"]:
                    if a == "direct":
                        continue
                    lines.append(f"  d'pair {a:5s} -Direct {fmt(b[f'ddpair_{a}-direct'], nd=2)}"
                                 + (f"  -CoT {fmt(b[f'ddpair_{a}-cot'], nd=2)}" if a != "cot" else ""))
            for a, row in g.items():
                lines.append(f"  desc {a:5s} " + json.dumps(row))
            if "glm" in r:
                for a in d["arms"]:
                    gg = r["glm"][a]
                    lines.append(f"  GLM {a:5s} occ {gg['occlusion']['dprime']:+.2f} {gg['occlusion']['ci']} vert {gg['vertical']['dprime']:+.2f} "
                                 f"{gg['vertical']['ci']} size {gg['size']['dprime']:+.2f} | AUC {gg['auc']['auc']}")
                    if a != "direct":
                        dd = r["glm"][f"{a}-direct"]
                        lines.append(f"      {a}-Direct occ {dd['occlusion']['dprime']:+.2f} {dd['occlusion']['ci']} vert {dd['vertical']['dprime']:+.2f} {dd['vertical']['ci']}")
    out.mkdir(parents=True, exist_ok=True)
    json.dump(res, open(out / "e22_gate.json", "w"), indent=1)
    (out / "rigor_e22.txt").write_text("\n".join(lines) + "\n")
    (out / "supp_gate.tex").write_text(latex(res))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
