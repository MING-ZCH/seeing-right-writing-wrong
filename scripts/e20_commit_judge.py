"""E20: judge the commit sentence j* itself against the image (round-7 Opus M2).

Synthetic Depth, per model: up to 40 D+C- items with a commit sentence (P(gold) >= 0.5 before it and < 0.5 from it on;
same rule as commit_baseline) and 40 D+C+ items whose sentence at the matched relative position (relative position
of a random D+C- commit of the same model) is judged. Each judge sees the image, the preceding sentence (context) and
the target sentence. Labels: CORRECT / INCORRECT / NOCLAIM (no checkable spatial claim) / UNCLEAR; markdown-robust parse.
Also records whether j* is the audited first two-object claim (the e2_judge parse).

Usage: python e20_commit_judge.py --judge claude-opus-5-5   (key from env)  ->  outputs/e20/<judge>.jsonl
       python e20_commit_judge.py --analyze                 ->  outputs/bundle/commit_judge.json
"""
from __future__ import annotations
import vtax_paths as _vtp  # noqa: E402

import argparse
import json
import random
import re
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

import build_bundle as B

O = Path(f"{_vtp.OUTPUTS}")
L = Path(f"{_vtp.LEGACY_RESULTS}")
SYN = f"{_vtp.DATASETS}/spatial_probing"
M7 = ["qwen3b", "qwen7b", "internvl3_8b", "phi35v", "molmo7b", "qwen3vl_8b", "internvl35_8b"]
PROMPT = """Here is an image with two shapes and part of a model's reasoning about it.

Previous sentence (context only): "{c}"
Target sentence: "{s}"

Consider only what the TARGET sentence claims about the spatial relation between the two shapes (which one is in front of, on top of, covering or hiding the other, or their relative position or size). Does that claim match the image?
Reply with exactly one word: CORRECT, INCORRECT, NOCLAIM (the target sentence makes no checkable spatial claim about the shapes), or UNCLEAR."""
OBJ = re.compile(r"\b(red|blue|green|yellow|purple|orange|cyan|gray|grey|pink|brown|black|white)\s+"
                 r"(circle|square|triangle|diamond|hexagon|pentagon|star|rectangle)s?\b", re.I)
ACT = re.compile(r"\b(overlap\w*|cover\w*|on top of|obscur\w*|block\w*|in front of)\b", re.I)
PAS = re.compile(r"\b(covered|hidden|obscured|blocked|overlapped|occluded)\s+by\b|\bbehind\b|\bunderneath\b|\bbeneath\b", re.I)


def first_claim_idx(sents: list[str]) -> int | None:
    """Index of the sentence audited by e2_judge (first two-object occlusion sentence)."""
    for j, s in enumerate(sents):
        objs = {(o.group(1) + " " + o.group(2)).lower() for o in OBJ.finditer(s)}
        if len(objs) >= 2 and (ACT.search(s) or PAS.search(s)) and not re.search(r"\?|whether|if the|determine|asks", s, re.I):
            return j
    return None


def commit_idx(r: dict) -> int | None:
    pg = [B.probs(t.get("lp_sum") or t["lp_bare"])[r["gold"]] for t in sorted(r["traj"], key=lambda x: x["j"])]
    js = next((j for j in range(1, len(pg)) if pg[j - 1] >= 0.5 and all(v < 0.5 for v in pg[j:])), None)
    return None if js is None else js - 1


def sample(seed: int = 0) -> list[dict]:
    rng = random.Random(seed)
    syn = json.load(open(f"{SYN}/depth_ordering.json"))
    img = lambda iid: syn[int(iid.split("_")[-1])]["image_path"].replace("<DATA_ROOT>",
                                                                          f"{_vtp.DATA_ROOT}")
    out = []
    for m in M7:
        G = B.groups_from({}, L, O, m)
        recs = [r for r in B.rj(O / "e2b" / f"{m}.jsonl") if r["task"] == "syn_depth"]
        dcm = [r for r in recs if G.get(r["item_id"], {}).get("group") == "D+C-" and commit_idx(r) is not None and r["sentences"]]
        dcp = [r for r in recs if G.get(r["item_id"], {}).get("group") == "D+C+" and len(r["sentences"]) >= 2]
        dcm = rng.sample(dcm, min(40, len(dcm)))
        relpos = []
        for r in dcm:
            j = commit_idx(r)
            relpos.append(j / max(1, len(r["sentences"]) - 1))
            fc = first_claim_idx(r["sentences"])
            out.append({"model": m, "item_id": r["item_id"], "group": "D+C-", "idx": j, "n_sent": len(r["sentences"]),
                        "is_first_claim": fc == j, "first_claim_idx": fc, "sentence": r["sentences"][j],
                        "context": r["sentences"][j - 1] if j > 0 else "(none)", "image": img(r["item_id"])})
        for r in rng.sample(dcp, min(40, len(dcp))):
            p = rng.choice(relpos) if relpos else 0.5
            j = int(round(p * (len(r["sentences"]) - 1)))
            out.append({"model": m, "item_id": r["item_id"], "group": "D+C+", "idx": j, "n_sent": len(r["sentences"]),
                        "is_first_claim": first_claim_idx(r["sentences"]) == j, "first_claim_idx": first_claim_idx(r["sentences"]),
                        "sentence": r["sentences"][j], "context": r["sentences"][j - 1] if j > 0 else "(none)", "image": img(r["item_id"])})
    return out


def sample_relcomp() -> list[dict]:
    """D+C+ comparator restricted to relational sentences (rigor_e17.stype), nearest to the matched relative position."""
    from rigor_e17 import stype
    base = sample()
    rng = random.Random(1)
    syn = json.load(open(f"{SYN}/depth_ordering.json"))
    img = lambda iid: syn[int(iid.split("_")[-1])]["image_path"].replace("<DATA_ROOT>",
                                                                          f"{_vtp.DATA_ROOT}")
    out = []
    for m in M7:
        relpos = [r["idx"] / max(1, r["n_sent"] - 1) for r in base if r["model"] == m and r["group"] == "D+C-"]
        G = B.groups_from({}, L, O, m)
        recs = [r for r in B.rj(O / "e2b" / f"{m}.jsonl") if r["task"] == "syn_depth" and G.get(r["item_id"], {}).get("group") == "D+C+"]
        recs = [r for r in recs if any(stype(x) == "relation" for x in r["sentences"])]
        for r in rng.sample(recs, min(40, len(recs))):
            target = rng.choice(relpos) * (len(r["sentences"]) - 1)
            rel = [j for j, x in enumerate(r["sentences"]) if stype(x) == "relation"]
            j = min(rel, key=lambda t: abs(t - target))
            out.append({"model": m, "item_id": r["item_id"], "group": "D+C+rel", "idx": j, "n_sent": len(r["sentences"]),
                        "is_first_claim": first_claim_idx(r["sentences"]) == j, "first_claim_idx": first_claim_idx(r["sentences"]),
                        "sentence": r["sentences"][j], "context": r["sentences"][j - 1] if j > 0 else "(none)", "image": img(r["item_id"])})
    return out


def sample_rescue() -> list[dict]:
    """D-C+ Synthetic Depth items: the sentence at which P(gold) rises to >= 0.5 and stays (commit-to-gold), up to 40/model."""
    rng = random.Random(2)
    syn = json.load(open(f"{SYN}/depth_ordering.json"))
    img = lambda iid: syn[int(iid.split("_")[-1])]["image_path"].replace("<DATA_ROOT>",
                                                                          f"{_vtp.DATA_ROOT}")
    out = []
    for m in M7:
        G = B.groups_from({}, L, O, m)
        recs = []
        for r in B.rj(O / "e2b" / f"{m}.jsonl"):
            if r["task"] != "syn_depth" or G.get(r["item_id"], {}).get("group") != "D-C+" or not r["sentences"]:
                continue
            pg = [B.probs(t.get("lp_sum") or t["lp_bare"])[r["gold"]] for t in sorted(r["traj"], key=lambda x: x["j"])]
            js = next((j for j in range(1, len(pg)) if pg[j - 1] < 0.5 and all(v >= 0.5 for v in pg[j:])), None)
            if js is not None:
                recs.append((r, js - 1))
        for r, j in rng.sample(recs, min(40, len(recs))):
            out.append({"model": m, "item_id": r["item_id"], "group": "D-C+rescue", "idx": j, "n_sent": len(r["sentences"]),
                        "is_first_claim": first_claim_idx(r["sentences"]) == j, "first_claim_idx": first_claim_idx(r["sentences"]),
                        "sentence": r["sentences"][j], "context": r["sentences"][j - 1] if j > 0 else "(none)", "image": img(r["item_id"])})
    return out


def sample_rescue_cmp(group: str) -> list[dict]:
    """Comparator for the rescue audit (round-11 Opus M3/m9): relational sentences from Synthetic Depth D-C- (or D+C+)
    rationales, position-matched to the rescue sentences of the same model (relative position drawn from that model's
    rescue sentences; nearest relational sentence), same n per model as the rescue sample."""
    from rigor_e17 import stype
    resc = sample_rescue()
    rng = random.Random({"D-C-": 3, "D+C+": 4}[group])
    syn = json.load(open(f"{SYN}/depth_ordering.json"))
    img = lambda iid: syn[int(iid.split("_")[-1])]["image_path"].replace("<DATA_ROOT>",
                                                                          f"{_vtp.DATA_ROOT}")
    out = []
    for m in M7:
        rr = [r for r in resc if r["model"] == m]
        relpos = [r["idx"] / max(1, r["n_sent"] - 1) for r in rr]
        G = B.groups_from({}, L, O, m)
        recs = [r for r in B.rj(O / "e2b" / f"{m}.jsonl") if r["task"] == "syn_depth" and G.get(r["item_id"], {}).get("group") == group]
        recs = [r for r in recs if any(stype(x) == "relation" for x in r["sentences"])]
        for r in rng.sample(recs, min(len(rr), len(recs))):
            target = rng.choice(relpos) * (len(r["sentences"]) - 1)
            rel = [j for j, x in enumerate(r["sentences"]) if stype(x) == "relation"]
            j = min(rel, key=lambda t: abs(t - target))
            out.append({"model": m, "item_id": r["item_id"], "group": f"{group}rel_rescuepos", "idx": j, "n_sent": len(r["sentences"]),
                        "is_first_claim": first_claim_idx(r["sentences"]) == j, "first_claim_idx": first_claim_idx(r["sentences"]),
                        "sentence": r["sentences"][j], "context": r["sentences"][j - 1] if j > 0 else "(none)", "image": img(r["item_id"])})
    return out


def parse(txt: str) -> str:
    w = re.sub(r"[^A-Za-z ]", " ", txt or "").strip().upper().split()
    for tok in w[:6]:
        if tok in ("CORRECT", "INCORRECT", "NOCLAIM", "UNCLEAR"):
            return tok
    return "UNPARSED"


def judge_items(judge: str, items: list[dict]) -> list[dict]:
    import api_client
    from PIL import Image
    lock, res = threading.Lock(), []

    def go(c):
        im = Image.open(c["image"]).convert("RGB")
        r = api_client.chat(judge, [{"role": "user", "content": [api_client.img_part(im), {"type": "text", "text": PROMPT.format(c=c["context"], s=c["sentence"])}]}],
                       max_tokens=2048, effort="high")
        with lock:
            res.append({k: v for k, v in c.items() if k != "image"} | {"judge": judge, "label": parse(r.get("text")),
                                                                         "raw": (r.get("text") or r.get("error", ""))[:80],
                                                                         "api_error": "error" in r})

    with ThreadPoolExecutor(6) as ex:
        list(ex.map(go, items))
    return res


def run(judge: str, relcomp: bool = False, rescue: bool = False, cmp: str | None = None) -> None:
    if cmp:
        items = sample_rescue_cmp({"dcm": "D-C-", "dcprel": "D+C+"}[cmp])
    else:
        items = sample_rescue() if rescue else sample_relcomp() if relcomp else sample()
    res = judge_items(judge, items)
    (O / "e20").mkdir(exist_ok=True)
    suffix = f"_rescue_{cmp}" if cmp else f"{'_relcomp' if relcomp else ''}{'_rescue' if rescue else ''}"
    with open(O / "e20" / f"{judge}{suffix}.jsonl", "w") as f:
        for r in res:
            f.write(json.dumps(r) + "\n")
    print(f"[e20] {judge}{suffix}: {len(res)}; unparsed {sum(r['label'] == 'UNPARSED' for r in res)}; api errors {sum(r['api_error'] for r in res)}")


def retry(judge: str, suffix: str) -> None:
    """Re-judge records whose call failed (HTTP / network error); keeps the original file as <name>.pre_retry.jsonl."""
    import shutil
    p = O / "e20" / f"{judge}{suffix}.jsonl"
    recs = B.rj(p)
    bad = [i for i, r in enumerate(recs) if r["label"] == "UNPARSED" and (r.get("api_error") or r["raw"].startswith(("HTTP", "URLError", "TimeoutError")))]
    if not bad:
        print(f"[e20 retry] {p.name}: nothing to retry")
        return
    bk = p.with_name(p.stem + ".pre_retry.jsonl")
    if not bk.exists():
        shutil.copy(p, bk)
    syn = json.load(open(f"{SYN}/depth_ordering.json"))
    img = lambda iid: syn[int(iid.split("_")[-1])]["image_path"].replace("<DATA_ROOT>",
                                                                          f"{_vtp.DATA_ROOT}")
    items = [{k: v for k, v in recs[i].items() if k not in ("judge", "label", "raw", "api_error")} | {"image": img(recs[i]["item_id"])} for i in bad]
    new = {(r["model"], r["item_id"], r["group"]): r for r in judge_items(judge, items)}
    for i in bad:
        k = (recs[i]["model"], recs[i]["item_id"], recs[i]["group"])
        recs[i] = new[k] | {"retried": True}
    with open(p, "w") as f:
        for r in recs:
            f.write(json.dumps(r) + "\n")
    print(f"[e20 retry] {p.name}: retried {len(bad)}; still failing {sum(new[k]['api_error'] for k in new)}")


def wilson(k: int, n: int) -> list[float]:
    if n == 0:
        return [float("nan")] * 2
    z, p = 1.96, k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [round(100 * (c - h), 1), round(100 * (c + h), 1)]


def analyze() -> None:
    J = {j: {(r["model"], r["item_id"], r["group"]): r for r in B.rj(O / "e20" / f"{j}.jsonl")} for j in ("claude-opus-5-5", "gpt-6-astra")}
    keys = sorted(set(J["claude-opus-5-5"]) & set(J["gpt-6-astra"]))
    lab = lambda j, k: J[j][k]["label"]
    res = {"n": len(keys), "agree": round(float(np.mean([lab("claude-opus-5-5", k) == lab("gpt-6-astra", k) for k in keys])), 3)}
    clear = [k for k in keys if {lab("claude-opus-5-5", k), lab("gpt-6-astra", k)} <= {"CORRECT", "INCORRECT"}]
    a = [int(lab("claude-opus-5-5", k) == "INCORRECT") for k in clear]
    b = [int(lab("gpt-6-astra", k) == "INCORRECT") for k in clear]
    po = np.mean(np.array(a) == np.array(b))
    pe_ = np.mean(a) * np.mean(b) + (1 - np.mean(a)) * (1 - np.mean(b))
    res["kappa_clear"] = round(float((po - pe_) / (1 - pe_)), 3)
    rows = {}
    for m in M7 + ["pooled"]:
        for g in ("D+C-", "D+C+"):
            ks = [k for k in keys if k[2] == g and (m == "pooled" or k[0] == m)]
            if not ks:
                continue
            r = {"n": len(ks)}
            for j, jl in (("claude-opus-5-5", "opus"), ("gpt-6-astra", "gpt")):
                cl = [k for k in ks if lab(j, k) in ("CORRECT", "INCORRECT")]
                kk = sum(lab(j, k) == "INCORRECT" for k in cl)
                r[f"{jl}_incorrect"] = round(100 * kk / max(1, len(cl)), 1)
                r[f"{jl}_ci"] = wilson(kk, len(cl))
                r[f"{jl}_noclaim"] = round(100 * np.mean([lab(j, k) == "NOCLAIM" for k in ks]), 1)
            cons = [k for k in ks if lab("claude-opus-5-5", k) == lab("gpt-6-astra", k) and lab("gpt-6-astra", k) in ("CORRECT", "INCORRECT")]
            kk = sum(lab("gpt-6-astra", k) == "INCORRECT" for k in cons)
            r["consensus_incorrect"] = round(100 * kk / max(1, len(cons)), 1)
            r["consensus_ci"] = wilson(kk, len(cons))
            r["consensus_n"] = len(cons)
            r["is_first_claim"] = round(100 * np.mean([J["gpt-6-astra"][k]["is_first_claim"] for k in ks]), 1)
            if g == "D+C-":
                for sub, f in (("commit_is_first_claim", True), ("commit_later_or_other", False)):
                    ss = [k for k in cons if J["gpt-6-astra"][k]["is_first_claim"] == f]
                    r[f"{sub}_consensus_incorrect"] = round(100 * np.mean([lab("gpt-6-astra", k) == "INCORRECT" for k in ss]), 1) if ss else None
                    r[f"{sub}_n"] = len(ss)
            rows[f"{m}|{g}"] = r
    res["rows"] = rows
    json.dump(res, open(Path("outputs/bundle") / "commit_judge.json", "w"), indent=1)
    for k, r in rows.items():
        print(k, {x: r[x] for x in ("n", "opus_incorrect", "gpt_incorrect", "consensus_incorrect", "consensus_n", "is_first_claim")},
              {x: r.get(x) for x in ("commit_is_first_claim_consensus_incorrect", "commit_later_or_other_consensus_incorrect", "commit_later_or_other_n")})
    print("agree", res["agree"], "kappa", res["kappa_clear"])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--judge")
    ap.add_argument("--analyze", action="store_true")
    ap.add_argument("--relcomp", action="store_true")
    ap.add_argument("--rescue", action="store_true")
    ap.add_argument("--cmp", choices=["dcm", "dcprel"], help="rescue comparator: D-C- / D+C+ relational, rescue-position-matched")
    ap.add_argument("--retry", help="file suffix to retry failed calls for, e.g. _rescue")
    a = ap.parse_args()
    if a.analyze:
        analyze()
    elif a.retry:
        retry(a.judge, a.retry)
    else:
        run(a.judge, a.relcomp, a.rescue, a.cmp)


if __name__ == "__main__":
    main()
