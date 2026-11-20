"""Exp-63b (R25 Argon leftover): what happens at false relational sentences that come BEFORE the commit sentence j*?

j* (e20 rule) = first sentence after which the forced-readout P(gold) is < .5 and stays there (P before j* >= .5).
Concern: an earlier false sentence precedes j* in 58% of Qwen3-VL-8B cue-set D+C- items (Exp-29), so j* may not be
where the error starts. Questions (stated before computing):
  B1 share of D+C- rationales with an earlier false relational sentence (k < j*);
  B2 readout at / after the earliest false sentence k: P(gold) after k, its change, whether P stayed >= .5 up to j*
     (NOTE: P_{j*-1} >= .5 holds by the definition of j*, so 'recovered before j*' is partly by construction;
     the non-circular quantities are the size of the move at k vs at j* vs at true sentences, and the twin-image
     effect right after k, i.e. whether the readout still follows the image once the false sentence is written);
  B3 is k the same proposition as j* (same shape claimed in front; near-verbatim repetition) or a different form;
  B4 j* contradiction rate (image-conditioned judges) with vs without an earlier false relation.
Interpretation target: j* = the sentence where a false relation is written in the form the readout uses, while earlier
false sentences often leave the readout (and its dependence on the image) intact.

Sources (no new judge calls; the e20 commit prompt is image-conditioned):
  judge   two-judge consensus (Claude Opus 5.5 + GPT-6-Astra, e20 prompt, image + previous sentence) on every
          'relational' sentence (rigor_e17.stype regex) before j*: Qwen3-VL-8B cue set 120 D+C- rationales (Exp-29
          prej2, incl. j*), Exp-28 samples of 40 (Molmo-7B-D cue set; both models twins; Syn. Depth = e20 audit sample);
          j* labels: Exp-30 full cue audit (all D+C- with j*), Exp-28 twin commit sample, e20 Syn. Depth audit,
          Exp-57 Qwen3-VL-32B audit (all). 'On top' re-judge (Exp-29 ontop prompt: reading depth / vertical) where present.
  parser  claim_parse v3 per sentence (depth relation with both shape names; claimed front shape) on EVERY sentence of
          every D+C- trajectory with j* (cue set: Qwen3-VL-8B, Molmo-7B-D, InternVL3-8B, Qwen3-VL-32B; twins: Qwen3-VL-8B,
          Molmo-7B-D, Qwen3-VL-32B; Syn. Depth main pool: 7 models). Validated against the judge labels.
  readout e28 / e57 trajectories (p_gold per prefix; e2b for Syn. Depth); twin-image prefix readout (Exp-57 trajtwin,
          Qwen3-VL-8B and -32B cue set): image effect = P(gold | own image) - P(original gold | twin image).
CIs: image-cluster bootstrap (2,000 draws, seed 63).
-> outputs/bundle/r28_jstar_condition.json, outputs/bundle/rigor_r28_jstar.txt
Usage: python r28_jstar_condition.py
"""
from __future__ import annotations
import vtax_paths as _vtp  # noqa: E402

import json
import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

import build_bundle as BB
from claim_parse import NEG, ORD, _objs
from e2f_claim_edit import OBJ, PAS, SKIP
from rigor_e17 import stype

O = Path(f"{_vtp.OUTPUTS}")
L = Path(f"{_vtp.LEGACY_RESULTS}")
B = Path("outputs/bundle")
J2 = ("claude-opus-5-5", "gpt-6-astra")
M7 = ["qwen3b", "qwen7b", "internvl3_8b", "phi35v", "molmo7b", "qwen3vl_8b", "internvl35_8b"]
NAMES = {"qwen3b": "Qwen2.5-VL-3B", "qwen7b": "Qwen2.5-VL-7B", "internvl3_8b": "InternVL3-8B", "phi35v": "Phi-3.5-V",
         "molmo7b": "Molmo-7B-D", "qwen3vl_8b": "Qwen3-VL-8B", "internvl35_8b": "InternVL3.5-8B", "qwen3vl_32b": "Qwen3-VL-32B"}
SETN = {"cues": "cue set", "twins": "twins", "syn": "Syn. Depth"}
NB, SEED = 2000, 63
QRX = re.compile(r"Is the (.+?) (in front of|behind) the (.+?)\?", re.I)
INFER = re.compile(r"\b(so|therefore|thus|hence|this means|which means|means that|meaning|indicat\w+|suggest\w*|implies|"
                   r"consequently|confirm\w*|this shows|we can conclude)\b", re.I)
DEPTHW = re.compile(r"\b(in front|behind|closer|nearer|farther|further away|foreground|background)\b", re.I)


def rj(p: Path) -> list[dict]:
    return [json.loads(l) for l in open(p)] if p.exists() else []


# ============================================================================= sentence parsing

def parse_sent(s: str, front: str, back: str) -> dict | None:
    """claim_parse v3 on one sentence + its grammatical form: claimed front, surface subject, predicate, negation."""
    objs = _objs(s)
    if len(set(objs)) < 2 or SKIP.search(s):
        return None
    a, p = ORD.search(s), PAS.search(s)
    if not a and not p:
        return None
    use_pas = p is not None and (a is None or p.start() < a.start())
    m = p if use_pas else a
    before = [(o.group(1) + " " + o.group(2)).lower().replace("grey", "gray") for o in OBJ.finditer(s) if o.end() <= m.start()]
    after = [o for o in OBJ.finditer(s) if o.start() >= m.end()]
    subj = before[-1] if before and after else objs[0]
    other = [o for o in objs if o != subj][0]
    claimed = other if use_pas else subj
    neg = bool(NEG.search(s[: m.start()][-25:]))
    if neg:
        claimed = other if claimed == subj else subj
    if claimed not in (front, back):
        return None
    w = m.group(0).lower()
    pred = "FRONT" if w.startswith("in front") else ("BEHIND" if w == "behind" else "OCCL")
    return {"claimed": claimed, "subj": subj, "pred": pred, "neg": neg}


def form(s: str, ps: dict | None, x: str, phrasing: str) -> dict:
    f = {"infer": bool(INFER.search(s)), "depth_word": bool(DEPTHW.search(s)), "stype": stype(s)}
    if ps:
        qpred = (phrasing == "front" and ps["pred"] == "FRONT") or (phrasing == "behind" and ps["pred"] == "BEHIND")
        f |= {"pred": ps["pred"], "neg": ps["neg"], "subj_is_x": ps["subj"] == x, "q_pred": qpred,
              "restates_q": qpred and ps["subj"] == x, "explicit": ps["pred"] in ("FRONT", "BEHIND")}
    return f


TOK = re.compile(r"[a-z]+")
STOP = set("the a an is of to and in on it its that this which as with be are by at part partially".split())


def jacc(a: str, b: str) -> float:
    A = {t for t in TOK.findall(a.lower()) if t not in STOP}
    Bs = {t for t in TOK.findall(b.lower()) if t not in STOP}
    return len(A & Bs) / max(1, len(A | Bs))


def logit(p: float) -> float:
    p = min(max(p, 1e-6), 1 - 1e-6)
    return float(np.log(p / (1 - p)))


# ============================================================================= trajectories

def commit_of(pg: list[float]) -> int | None:
    js = next((j for j in range(1, len(pg)) if pg[j - 1] >= 0.5 and all(v < 0.5 for v in pg[j:])), None)
    return None if js is None else js - 1


def traj_units(m: str, st: str) -> list[dict]:
    """All trajectories of a model x set: sentences, p_gold per prefix, j*, front / back / x / phrasing, group, cluster."""
    out = []
    if st == "syn":
        G = BB.groups_from({}, L, O, m)
        for r in rj(O / "e2b" / f"{m}.jsonl"):
            if r["task"] != "syn_depth" or not r["sentences"]:
                continue
            q = QRX.search(r["question"])
            x, ph, y = q.group(1).lower(), ("front" if q.group(2).lower() == "in front of" else "behind"), q.group(3).lower()
            yes_front = (r["gold"] == "A") == (ph == "front")
            front, back = (x, y) if yes_front else (y, x)
            pg = [BB.probs(t.get("lp_sum") or t["lp_bare"])[r["gold"]] for t in sorted(r["traj"], key=lambda t: t["j"])]
            out.append({"key": r["item_id"], "cl": r["item_id"], "group": G.get(r["item_id"], {}).get("group"), "sents": r["sentences"],
                        "pg": pg, "js": commit_of(pg), "front": front, "back": back, "x": x, "phrasing": ph})
        return out
    src = O / ("e57" if m == "qwen3vl_32b" else "e28") / f"{m}_{st}_traj.jsonl"
    for r in rj(src):
        mt = r["meta"]
        key = r["item_id"]
        front, back, x = mt["front_name"].lower(), mt["back_name"].lower(), mt["x_name"].lower()
        pg = r["p_gold"]
        js = r["commit_idx"]
        assert js == commit_of(pg), (m, st, key)
        out.append({"key": key, "cl": r["base_id"], "group": r["group"], "sents": r["sentences"], "pg": pg, "js": js,
                    "front": front, "back": back, "x": x, "phrasing": r["phrasing"]})
    return out


# ============================================================================= judge labels

def cons(a: str | None, b: str | None) -> str:
    return a if (a == b and a in ("CORRECT", "INCORRECT", "NOCLAIM")) else "NONCONS"


def load_prej(m: str, st: str) -> tuple[dict, set]:
    """(key, idx) -> consensus label for judged pre-j* (and j*) sentences; set of keys whose pre-j* sentences were judged."""
    files = {("qwen3vl_8b", "cues"): [O / "e29" / f"judge_{{j}}_prej2.jsonl"],
             ("molmo7b", "cues"): [O / "e28" / f"judge_{{j}}_prej_cues.jsonl"],
             ("qwen3vl_8b", "twins"): [O / "e28" / f"judge_{{j}}_prej_twins.jsonl"],
             ("molmo7b", "twins"): [O / "e28" / f"judge_{{j}}_prej_twins.jsonl"],
             ("qwen3vl_8b", "syn"): [O / "e28" / f"judge_{{j}}_prej_syn.jsonl"],
             ("molmo7b", "syn"): [O / "e28" / f"judge_{{j}}_prej_syn.jsonl"]}.get((m, st), [])
    lab = defaultdict(dict)
    for tmpl in files:
        for j in J2:
            for r in rj(Path(str(tmpl).format(j=j))):
                if r.get("model") == m and not r.get("api_error"):
                    lab[(r["item_id"], int(r["idx"]))][j] = r.get("label")
    # sample membership: rationales without any relational pre-j* sentence are absent from the prej files
    if (m, st) == ("qwen3vl_8b", "cues"):
        sample = {k[0] for k in lab}
    elif st in ("cues", "twins") and files:
        sample = {r["item_id"] for r in rj(O / "e28" / f"judge_{J2[1]}_commit_{st}.jsonl") if r["model"] == m and r["group"] == "D+C-"}
    elif st == "syn" and files:
        sample = {r["item_id"] for r in rj(O / "e20" / f"{J2[1]}.jsonl") if r["model"] == m and r["group"] == "D+C-"}
    else:
        sample = set()
    return {k: cons(v.get(J2[0]), v.get(J2[1])) for k, v in lab.items()}, sample


def load_jstar_labels(m: str, st: str) -> dict:
    """item key -> consensus label of the j* sentence (image-conditioned e20 prompt)."""
    srcs = []
    if st == "cues" and m in ("qwen3vl_8b", "molmo7b"):
        srcs = [O / "e30" / f"judge_{{j}}_commit_e20.jsonl", O / "e28" / f"judge_{{j}}_commit_cues.jsonl"]
    elif st == "twins" and m in ("qwen3vl_8b", "molmo7b"):
        srcs = [O / "e28" / f"judge_{{j}}_commit_twins.jsonl"]
    elif st == "syn":
        srcs = [O / "e20" / f"{{j}}.jsonl"]
    elif m == "qwen3vl_32b":
        srcs = [O / "e57" / f"judge_{{j}}_commit_{st}_e20.jsonl"]
    lab = defaultdict(dict)
    for tmpl in srcs:
        for j in J2:
            for r in rj(Path(str(tmpl).format(j=j))):
                if r.get("model") != m or r.get("group") != "D+C-" or str(r.get("api_error")) == "True":
                    continue
                lab[(r["item_id"], int(r["idx"]))].setdefault(j, r.get("label"))
    return {k: cons(v.get(J2[0]), v.get(J2[1])) for k, v in lab.items()}


def load_ontop(m: str, st: str) -> dict:
    """(item, idx) -> (reading, depth_true) consensus from the Exp-29 'on top' re-judge, where present."""
    lab = defaultdict(dict)
    for j in J2:
        for r in rj(O / "e29" / f"judge_{j}_ontop.jsonl"):
            if r.get("model") == m and r.get("set") == st and not r.get("api_error"):
                lab[(r["item_id"], int(r["idx"]))][j] = (r.get("reading"), r.get("depth_true"))
    out = {}
    for k, v in lab.items():
        a, b = v.get(J2[0]), v.get(J2[1])
        out[k] = a if a == b else ("NONCONS", None)
    return out


# ============================================================================= statistics

def cboot(vals, cl, nb: int = NB, seed: int = SEED, scale: float = 100.0) -> dict | None:
    vals = np.asarray(vals, float)
    cl = np.asarray(cl)
    ok = ~np.isnan(vals)
    vals, cl = vals[ok], cl[ok]
    if len(vals) == 0:
        return None
    u = np.unique(cl)
    gi = {c: np.where(cl == c)[0] for c in u}
    rng = np.random.default_rng(seed)
    bs = [vals[np.concatenate([gi[c] for c in rng.choice(u, len(u))])].mean() for _ in range(nb)]
    return {"mean": round(scale * float(vals.mean()), 1 if scale == 100 else 3), "n": int(len(vals)),
            "ci": [round(scale * float(np.quantile(bs, .025)), 1 if scale == 100 else 3),
                   round(scale * float(np.quantile(bs, .975)), 1 if scale == 100 else 3)]}


def fmt(x: dict | None, pct: bool = True) -> str:
    if not x:
        return "NA"
    u = "%" if pct else ""
    return f"{x['mean']}{u} [{x['ci'][0]}, {x['ci'][1]}] (n {x['n']})"


# ============================================================================= per unit analysis

def analyse_unit(m: str, st: str, twin: dict | None) -> dict:
    T = [u for u in traj_units(m, st)]
    dcm = [u for u in T if u["group"] == "D+C-" and u["js"] is not None]
    dcp = [u for u in T if u["group"] == "D+C+"]
    prej, judged = load_prej(m, st)
    jl = load_jstar_labels(m, st)
    ot = load_ontop(m, st)
    res = {"model": m, "set": st, "n_traj": len(T), "n_dcm": sum(u["group"] == "D+C-" for u in T), "n_dcm_jstar": len(dcm)}
    rows = []
    for u in dcm:
        s, js, pg = u["sents"], u["js"], u["pg"]
        ps = [parse_sent(x, u["front"], u["back"]) for x in s]
        fal = [k for k in range(js) if ps[k] and ps[k]["claimed"] == u["back"]]
        tru = [k for k in range(js) if ps[k] and ps[k]["claimed"] == u["front"]]
        jkey_ = u["key"]
        in_j = jkey_ in judged
        lab = lambda k: prej.get((jkey_, k))
        jfal = [k for k in range(js) if lab(k) == "INCORRECT"] if in_j else []
        jcor = [k for k in range(js) if lab(k) == "CORRECT"] if in_j else []
        jany = [k for k in range(js) if lab(k) is not None] if in_j else []
        jnc = [k for k in range(js) if lab(k) in ("NONCONS",)] if in_j else []
        jstar_lab = jl.get((jkey_, js)) or (lab(js) if in_j else None)
        tw = twin.get(jkey_) if twin else None
        rows.append({"key": jkey_, "cl": u["cl"], "js": js, "n": len(s), "pg": pg, "sents": s, "ps": ps, "x": u["x"],
                     "phrasing": u["phrasing"], "front": u["front"], "back": u["back"], "pfal": fal, "ptru": tru,
                     "in_j": in_j, "jfal": jfal, "jcor": jcor, "jany": jany, "jnc": jnc, "jstar_lab": jstar_lab, "tw": tw,
                     "ot": {k: ot.get((jkey_, k)) for k in range(js + 1)}})
    res["judge"] = judge_block(rows) if any(r["in_j"] for r in rows) else None
    res["parser"] = parser_block(rows, dcp)
    return res


def move_stats(rows: list[dict], pick, tag: str) -> dict:
    """Readout at the picked sentence k (pick(row) -> k or None): P after, dP, dlogit, P >= .5, min P from k to j*-1,
    twin-image effect before / after k (if available)."""
    P_after, dP, dL, ge, stay, cl, ie_b, ie_a, iecl = [], [], [], [], [], [], [], [], []
    for r in rows:
        k = pick(r)
        if k is None:
            continue
        pg = r["pg"]
        P_after.append(pg[k + 1]); dP.append(pg[k + 1] - pg[k]); dL.append(logit(pg[k + 1]) - logit(pg[k]))
        ge.append(float(pg[k + 1] >= 0.5))
        stay.append(float(min(pg[k + 1: r["js"] + 1]) >= 0.5) if k < r["js"] else np.nan)
        cl.append(r["cl"])
        if r["tw"] is not None:
            po, pt = r["tw"]["p_gold_own"], r["tw"]["p_gold_twin"]
            ie_b.append(po[k] - pt[k]); ie_a.append(po[k + 1] - pt[k + 1]); iecl.append(r["cl"])
    out = {"tag": tag, "n": len(P_after), "P_after": cboot(P_after, cl, scale=1.0), "dP": cboot(dP, cl, scale=1.0),
           "dlogit": cboot(dL, cl, scale=1.0), "P_after_ge_.5": cboot(ge, cl), "never_below_.5_until_jstar": cboot(stay, cl),
           "median_dP": round(float(np.median(dP)), 3) if dP else None}
    if ie_b:
        out["image_effect_before"] = cboot(ie_b, iecl)
        out["image_effect_after"] = cboot(ie_a, iecl)
        out["image_effect_change"] = cboot(np.array(ie_a) - np.array(ie_b), iecl)
    return out


def form_compare(rows: list[dict], pick_k, label: str) -> dict:
    """Earliest false sentence k vs j* on the same rationales: same claimed front, repetition, grammatical form."""
    rec = defaultdict(list)
    cl = []
    for r in rows:
        k = pick_k(r)
        if k is None:
            continue
        js = r["js"]
        pk, pj = r["ps"][k], r["ps"][js]
        fk, fj = form(r["sents"][k], pk, r["x"], r["phrasing"]), form(r["sents"][js], pj, r["x"], r["phrasing"])
        cl.append(r["cl"])
        rec["jstar_parsed"].append(float(pj is not None))
        rec["k_parsed"].append(float(pk is not None))
        both = pk is not None and pj is not None
        rec["same_front_if_both"].append(float(pk["claimed"] == pj["claimed"]) if both else np.nan)
        rec["jstar_claims_back_if_parsed"].append(float(pj["claimed"] == r["back"]) if pj else np.nan)
        jac = jacc(r["sents"][k], r["sents"][js])
        rec["jaccard"].append(jac)
        rec["near_verbatim(jacc>=.6)"].append(float(jac >= 0.6))
        for nm, f in (("k", fk), ("jstar", fj)):
            rec[f"{nm}_infer_marker"].append(float(f["infer"]))
            rec[f"{nm}_depth_word"].append(float(f["depth_word"]))
            rec[f"{nm}_explicit_front_behind"].append(float(f.get("explicit", False)) if "pred" in f else np.nan)
            rec[f"{nm}_restates_question"].append(float(f.get("restates_q", False)) if "pred" in f else np.nan)
            rec[f"{nm}_subject_is_queried"].append(float(f.get("subj_is_x", False)) if "pred" in f else np.nan)
            rec[f"{nm}_negated"].append(float(f.get("neg", False)) if "pred" in f else np.nan)
            rec[f"{nm}_stype_conclusion"].append(float(f["stype"] == "conclusion"))
            rec[f"{nm}_pred_OCCL"].append(float(f.get("pred") == "OCCL") if "pred" in f else np.nan)
        rec["gap_sentences"].append(js - k)
        rec["n_false_before_jstar"].append(len(r["jfal"] if label == "judge" else r["pfal"]))
    out = {"n": len(cl)}
    for k_, v in rec.items():
        sc = 1.0 if k_ in ("jaccard", "gap_sentences", "n_false_before_jstar") else 100.0
        out[k_] = cboot(v, cl, scale=sc)
    return out


def judge_block(rows: list[dict]) -> dict:
    R = [r for r in rows if r["in_j"]]
    cl = [r["cl"] for r in R]
    has = [float(bool(r["jfal"])) for r in R]
    worst = [float(bool(r["jfal"] or r["jnc"])) for r in R]
    # false sentence that is also a depth claim naming the back shape (parser) or read as depth & false ('on top' re-judge)
    def depth_false(r, k):
        ok = r["ps"][k] is not None and r["ps"][k]["claimed"] == r["back"]
        o = r["ot"].get(k)
        return ok or (o is not None and o[0] in ("DEPTH", "BOTH") and o[1] == "FALSE")
    hasd = [float(any(depth_false(r, k) for k in r["jfal"])) for r in R]
    out = {"n_rationales": len(R), "n_prej_judged_sentences": int(sum(len(r["jany"]) for r in R)),
           "earlier_false": cboot(has, cl), "earlier_false_worst_case": cboot(worst, cl),
           "earlier_false_depth": cboot(hasd, cl)}
    # where the earliest false sentence falls relative to j*
    E = [r for r in R if r["jfal"]]
    out["earliest_false_gap"] = Counter(min(5, r["js"] - r["jfal"][0]) for r in E)
    out["move_earliest_false"] = move_stats(E, lambda r: r["jfal"][0], "earliest consensus-false sentence before j*")
    out["move_all_false"] = move_stats([dict(r, _k=k) for r in E for k in r["jfal"]], lambda r: r["_k"], "every consensus-false pre-j* sentence")
    out["move_true"] = move_stats([dict(r, _k=k) for r in R for k in r["jcor"]], lambda r: r["_k"], "every consensus-true pre-j* sentence")
    unj = [dict(r, _k=k) for r in R for k in range(r["js"]) if k not in r["jany"]]
    out["move_unjudged"] = move_stats(unj, lambda r: r["_k"], "pre-j* sentences not judged (non-relational by regex)")
    out["move_jstar_with_earlier"] = move_stats(E, lambda r: r["js"], "j* (rationales with an earlier false sentence)")
    out["move_jstar_all"] = move_stats(R, lambda r: r["js"], "j* (all judged rationales)")
    out["form_earliest_vs_jstar"] = form_compare(E, lambda r: r["jfal"][0], "judge")
    # j* contradiction conditional on an earlier false sentence
    def jrate(sub):
        v = [np.nan if r["jstar_lab"] in (None, "NONCONS") else float(r["jstar_lab"] == "INCORRECT") for r in sub]
        return cboot(v, [r["cl"] for r in sub])
    out["jstar_false_given_earlier"] = jrate(E)
    out["jstar_false_given_none"] = jrate([r for r in R if not r["jfal"]])
    out["jstar_label_counts"] = {"earlier": dict(Counter(r["jstar_lab"] for r in E)),
                                 "none": dict(Counter(r["jstar_lab"] for r in R if not r["jfal"]))}
    # validation of the parser on judged sentences
    agree, n = Counter(), 0
    for r in R:
        for k in r["jany"]:
            lab = "INCORRECT" if k in r["jfal"] else ("CORRECT" if k in r["jcor"] else "OTHER")
            p = r["ps"][k]
            pl = "none" if p is None else ("false" if p["claimed"] == r["back"] else "true")
            agree[(lab, pl)] += 1
    out["parser_vs_judge_prej"] = {f"{a}|{b}": v for (a, b), v in sorted(agree.items())}
    # sequences: rationales where the readout first dropped below .5 at an earlier false sentence and recovered
    out["dip_below_.5_at_earliest_false_then_recover"] = cboot([float(r["pg"][r["jfal"][0] + 1] < 0.5) for r in E], [r["cl"] for r in E])
    out["examples"] = [{"key": r["key"], "js": r["js"], "earliest_false": r["jfal"][0],
                        "k_sentence": r["sents"][r["jfal"][0]][:240], "jstar_sentence": r["sents"][r["js"]][:240],
                        "P_after_k": round(r["pg"][r["jfal"][0] + 1], 3), "P_after_jstar": round(r["pg"][r["js"] + 1], 3)}
                       for r in E[:6]]
    return out


def parser_block(rows: list[dict], dcp: list[dict]) -> dict:
    cl = [r["cl"] for r in rows]
    has = [float(bool(r["pfal"])) for r in rows]
    E = [r for r in rows if r["pfal"]]
    out = {"n_rationales": len(rows), "earlier_false": cboot(has, cl),
           "jstar_parsed_depth_claim": cboot([float(r["ps"][r["js"]] is not None) for r in rows], cl),
           "jstar_parsed_false": cboot([float(r["ps"][r["js"]] is not None and r["ps"][r["js"]]["claimed"] == r["back"]) for r in rows], cl)}
    out["move_earliest_false"] = move_stats(E, lambda r: r["pfal"][0], "earliest parser-false sentence before j*")
    out["move_all_false"] = move_stats([dict(r, _k=k) for r in E for k in r["pfal"]], lambda r: r["_k"], "every parser-false pre-j* sentence")
    out["move_true"] = move_stats([dict(r, _k=k) for r in rows for k in r["ptru"]], lambda r: r["_k"], "every parser-true pre-j* sentence")
    other = [dict(r, _k=k) for r in rows for k in range(r["js"]) if r["ps"][k] is None]
    out["move_noclaim"] = move_stats(other, lambda r: r["_k"], "pre-j* sentences without a parsed depth claim")
    out["move_jstar_with_earlier"] = move_stats(E, lambda r: r["js"], "j* (rationales with an earlier parser-false sentence)")
    out["move_jstar_all"] = move_stats(rows, lambda r: r["js"], "j* (all)")
    out["form_earliest_vs_jstar"] = form_compare(E, lambda r: r["pfal"][0], "parser")
    # where is the FIRST false sentence written with an explicit front/behind predicate (resp. in the question's own
    # subject + predicate) relative to j*?  ('after' = only after j*; 'none' = never in the rationale)
    for nm, test in (("explicit", lambda r, p, s: p["pred"] in ("FRONT", "BEHIND")),
                     ("restates_q", lambda r, p, s: form(s, p, r["x"], r["phrasing"]).get("restates_q", False))):
        pos = []
        for r in rows:
            ks = [k for k, p in enumerate(r["ps"]) if p and p["claimed"] == r["back"] and test(r, p, r["sents"][k])]
            pos.append("none" if not ks else ("before" if ks[0] < r["js"] else ("at" if ks[0] == r["js"] else "after")))
        c = Counter(pos)
        out[f"first_false_{nm}_vs_jstar"] = {k: c.get(k, 0) for k in ("before", "at", "after", "none")}
        out[f"first_false_{nm}_at_jstar"] = cboot([float(p == "at") for p in pos], cl)
        out[f"first_false_{nm}_before_jstar"] = cboot([float(p == "before") for p in pos], cl)
        pre = [dict(r, _k=k) for r in rows for k in range(r["js"])
               if r["ps"][k] and r["ps"][k]["claimed"] == r["back"] and test(r, r["ps"][k], r["sents"][k])]
        out[f"move_false_{nm}_before_jstar"] = move_stats(pre, lambda r: r["_k"], f"parser-false pre-j* sentences, {nm} form")

    def jrate(sub):
        v = [np.nan if r["jstar_lab"] in (None, "NONCONS") else float(r["jstar_lab"] == "INCORRECT") for r in sub]
        return cboot(v, [r["cl"] for r in sub])
    out["jstar_false_given_earlier"] = jrate(E)
    out["jstar_false_given_none"] = jrate([r for r in rows if not r["pfal"]])
    # control: D+C+ rationales (CoT ends right) that contain a parser-false depth sentence: the false sentence did not commit
    fcp, fcl, dpc, pcl = [], [], [], []
    for u in dcp:
        ps = [parse_sent(x, u["front"], u["back"]) for x in u["sents"]]
        f_ = [k for k, p in enumerate(ps) if p and p["claimed"] == u["back"]]
        fcp.append(float(bool(f_))); fcl.append(u["cl"])
        for k in f_:
            dpc.append(u["pg"][k + 1] - u["pg"][k]); pcl.append(u["cl"])
    out["dcp_with_false_sentence"] = cboot(fcp, fcl)
    out["dcp_false_sentence_dP"] = cboot(dpc, pcl, scale=1.0)
    out["dcp_n"] = len(dcp)
    return out


def load_twin(m: str) -> dict:
    R = rj(O / "e57" / f"{m}_cues_trajtwin.jsonl")
    out = {}
    for r in R:
        out[r["item_id"]] = r
    return out


# ============================================================================= report

def report(res: dict) -> list[str]:
    L = []
    for k, r in res["units"].items():
        L.append(f"== {NAMES[r['model']]} {SETN[r['set']]}: trajectories {r['n_traj']}, D+C- {r['n_dcm']}, with j* {r['n_dcm_jstar']}")
        for src in ("judge", "parser"):
            b = r.get(src)
            if not b:
                continue
            L.append(f"  [{src}] rationales {b['n_rationales']}" + (f", judged pre-j* sentences {b['n_prej_judged_sentences']}" if src == "judge" else ""))
            L.append(f"    B1 earlier false relational sentence before j*: {fmt(b['earlier_false'])}"
                     + (f" | worst case (non-consensus = false) {fmt(b['earlier_false_worst_case'])} | false AND a depth claim naming the back shape "
                        f"{fmt(b['earlier_false_depth'])} | gap j*-k {dict(sorted(b['earliest_false_gap'].items()))} (5 = >=5)" if src == "judge" else
                        f" | j* parsed as depth claim {fmt(b['jstar_parsed_depth_claim'])}, as a false one {fmt(b['jstar_parsed_false'])}"))
            for nm in ("move_earliest_false", "move_all_false", "move_true", "move_unjudged" if src == "judge" else "move_noclaim",
                       "move_jstar_with_earlier", "move_jstar_all"):
                mv = b.get(nm)
                if not mv or not mv["n"]:
                    continue
                L.append(f"    B2 {mv['tag']:58s} n {mv['n']:4d} | P(gold) after {fmt(mv['P_after'], False)} | dP {fmt(mv['dP'], False)} "
                         f"(median {mv['median_dP']}) | dlogit {fmt(mv['dlogit'], False)} | P after >= .5 {fmt(mv['P_after_ge_.5'])} | "
                         f"never < .5 until j* {fmt(mv['never_below_.5_until_jstar'])}"
                         + (f" | twin image effect before {fmt(mv['image_effect_before'])} after {fmt(mv['image_effect_after'])} change {fmt(mv['image_effect_change'])}" if "image_effect_before" in mv else ""))
            fc = b["form_earliest_vs_jstar"]
            if fc["n"]:
                g = lambda kk: fmt(fc.get(kk))
                L.append(f"    B3 earliest false k vs j* (n {fc['n']}): j* parsed {g('jstar_parsed')}, k parsed {g('k_parsed')}; same claimed front (both parsed) "
                         f"{g('same_front_if_both')}; near-verbatim (Jaccard >= .6) {g('near_verbatim(jacc>=.6)')}, mean Jaccard {fmt(fc['jaccard'], False)}; "
                         f"gap {fmt(fc['gap_sentences'], False)} sentences; false sentences before j* {fmt(fc['n_false_before_jstar'], False)}")
                L.append(f"       form k / j*: inference marker {g('k_infer_marker')} / {g('jstar_infer_marker')}; explicit front/behind predicate (parsed) "
                         f"{g('k_explicit_front_behind')} / {g('jstar_explicit_front_behind')}; occlusion-verb predicate {g('k_pred_OCCL')} / {g('jstar_pred_OCCL')}; "
                         f"restates the question's subject + predicate {g('k_restates_question')} / {g('jstar_restates_question')}; subject = queried shape "
                         f"{g('k_subject_is_queried')} / {g('jstar_subject_is_queried')}; depth word {g('k_depth_word')} / {g('jstar_depth_word')}; "
                         f"'answer/conclusion' sentence {g('k_stype_conclusion')} / {g('jstar_stype_conclusion')}")
            L.append(f"    B4 j* judged false | earlier false {fmt(b['jstar_false_given_earlier'])} vs | none {fmt(b['jstar_false_given_none'])}"
                     + (f" | counts {b['jstar_label_counts']}" if src == "judge" else ""))
            if src == "judge":
                L.append(f"    parser vs judge on judged pre-j* sentences (judge|parser): {b['parser_vs_judge_prej']}")
                L.append(f"    readout dipped below .5 right after the earliest false sentence (then recovered, by j* definition): "
                         f"{fmt(b['dip_below_.5_at_earliest_false_then_recover'])}")
            else:
                L.append(f"    control D+C+ (n {b['dcp_n']}): rationales with a parser-false depth sentence {fmt(b['dcp_with_false_sentence'])}, "
                         f"dP at it {fmt(b['dcp_false_sentence_dP'], False)}")
                L.append(f"    first false sentence with an explicit front/behind predicate vs j*: {b['first_false_explicit_vs_jstar']} "
                         f"(at j* {fmt(b['first_false_explicit_at_jstar'])}, before {fmt(b['first_false_explicit_before_jstar'])}) | "
                         f"first false sentence in the question's own subject + predicate: {b['first_false_restates_q_vs_jstar']} "
                         f"(at j* {fmt(b['first_false_restates_q_at_jstar'])})")
                for nm in ("explicit", "restates_q"):
                    mv = b[f"move_false_{nm}_before_jstar"]
                    if mv["n"]:
                        L.append(f"    pre-j* false sentences in {nm} form: n {mv['n']} | dP {fmt(mv['dP'], False)} (median {mv['median_dP']}) | "
                                 f"P after {fmt(mv['P_after'], False)}"
                                 + (f" | twin image effect before {fmt(mv['image_effect_before'])} after {fmt(mv['image_effect_after'])}"
                                    if "image_effect_before" in mv else ""))
        L.append("")
    return L


UNITS = [("qwen3vl_8b", "cues"), ("molmo7b", "cues"), ("internvl3_8b", "cues"), ("qwen3vl_32b", "cues"),
         ("qwen3vl_8b", "twins"), ("molmo7b", "twins"), ("qwen3vl_32b", "twins")] + [(m, "syn") for m in M7]


def main() -> None:
    res = {"_doc": __doc__, "units": {}}
    for m, st in UNITS:
        twin = load_twin(m) if st == "cues" and m in ("qwen3vl_8b", "qwen3vl_32b") else None
        r = analyse_unit(m, st, twin)
        res["units"][f"{m}|{st}"] = r
        print(f"done {m} {st}: D+C- with j* {r['n_dcm_jstar']}", flush=True)
    lines = report(res)
    hdr = __doc__.split("Sources")[0]
    (B / "rigor_r28_jstar.txt").write_text("# " + hdr.replace("\n", "\n# ") + "\n" + "\n".join(lines) + "\n")
    json.dump(res, open(B / "r28_jstar_condition.json", "w"), indent=1, default=lambda o: dict(o) if isinstance(o, Counter) else str(o))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
