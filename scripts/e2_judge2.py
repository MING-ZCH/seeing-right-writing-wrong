"""Second judge + D-C+ extension for the claim-truth audit (same prompt and settings as e2_judge.py).

Jobs: (a) gpt-6-astra on the 480 claims in outputs/e2_judge.jsonl (judged by claude-opus-5-5);
      (b) a D-C+ sample (<= --per per model, seed 0) judged by both claude-opus-5-5 and gpt-6-astra.
Writes outputs/e2_judge2.jsonl (one row per claim x judge). --analyze writes outputs/bundle/judge2.json.
Run from vtax_exps/.
"""
from __future__ import annotations
import vtax_paths as _vtp  # noqa: E402

import argparse
import json
import random
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image

import build_bundle as B
from e2_judge import L, O, PROMPT, SYN, kappa
from e2f_claim_edit import QRX, first_claim

MODELS = ["qwen3b", "qwen7b", "internvl3_8b", "phi35v", "molmo7b", "qwen3vl_8b"]
JUDGES = ["claude-opus-5-5", "gpt-6-astra"]


def image_path(item_id: str, syn: list) -> str:
    i = int(item_id.split("_")[-1])
    return syn[i]["image_path"].replace("<DATA_ROOT>",
                                        f"{_vtp.DATA_ROOT}")


def claims_for_group(m: str, grp: str) -> list[dict]:
    """First parsed two-object claim per Syn. Depth item of a D/C group (same parse as claim_truth)."""
    G = B.groups_from({}, L, O, m)
    out = []
    for r in B.rj(O / "e2b" / f"{m}.jsonl"):
        if r["task"] != "syn_depth" or G.get(r["item_id"], {}).get("group") != grp:
            continue
        q = QRX.search(r["question"])
        if not q:
            continue
        x, y = q.group(1).lower(), q.group(2).lower()
        front = x if r["gold"] == "A" else y
        fc = first_claim(r["sentences"])
        if fc is None:
            continue
        out.append({"model": m, "item_id": r["item_id"], "group": grp, "sentence": r["sentences"][fc[0]],
                    "heuristic_wrong": int(fc[1] != front)})
    return out


def judge_all(jobs: list[dict], syn: list) -> list[dict]:
    import api_client
    lock = threading.Lock()
    res = []

    def run(c):
        img = Image.open(image_path(c["item_id"], syn)).convert("RGB")
        r = api_client.chat(c["judge_name"], [{"role": "user", "content": [api_client.img_part(img),
                                                                       {"type": "text", "text": PROMPT.format(s=c["sentence"])}]}],
                       max_tokens=2048, effort="high")
        word = (r.get("text") or "").strip().strip("*_` \n").lower()
        lab = "correct" if word.startswith("correct") else "incorrect" if word.startswith("incorrect") else "unclear"
        with lock:
            res.append(c | {"judge": lab, "judge_raw": (r.get("text") or r.get("error", ""))[:80]})
            if len(res) % 50 == 0:
                print(f"[judge2] {len(res)}/{len(jobs)}", flush=True)

    with ThreadPoolExecutor(12) as ex:
        list(ex.map(run, jobs))
    return res


def analyze(out_dir: Path) -> None:
    j1 = [json.loads(l) for l in open(O / "e2_judge.jsonl")]
    j2 = [json.loads(l) for l in open(O / "e2_judge2.jsonl")]
    lab = {}
    for r in j1:
        lab[(r["model"], r["item_id"], "claude-opus-5-5")] = r
    for r in j2:
        lab[(r["model"], r["item_id"], r["judge_name"])] = r
    keys = sorted({(m, i) for m, i, _ in lab})
    rows = []
    for m, i in keys:
        a, b = lab.get((m, i, JUDGES[0])), lab.get((m, i, JUDGES[1]))
        if a and b:
            rows.append({"model": m, "item_id": i, "group": a["group"], "rule": a["heuristic_wrong"],
                         "j1": a["judge"], "j2": b["judge"]})
    res = {"n_claims_both_judges": len(rows)}

    def agree(sel, name):
        both = [r for r in sel if r["j1"] != "unclear" and r["j2"] != "unclear"]
        x1 = [int(r["j1"] == "incorrect") for r in both]
        x2 = [int(r["j2"] == "incorrect") for r in both]
        out = {"n": len(sel), "n_both_clear": len(both),
               "unclear_j1": sum(r["j1"] == "unclear" for r in sel), "unclear_j2": sum(r["j2"] == "unclear" for r in sel),
               "judge_judge_agreement": round(float(np.mean(np.array(x1) == np.array(x2))), 3) if both else None,
               "judge_judge_kappa": round(kappa(x1, x2), 3) if both else None,
               "three_way_agreement_incl_unclear": round(float(np.mean([r["j1"] == r["j2"] for r in sel])), 3)}
        for jn, key in (("claude", "j1"), ("gpt", "j2")):
            cl = [r for r in sel if r[key] != "unclear"]
            h = [r["rule"] for r in cl]
            j = [int(r[key] == "incorrect") for r in cl]
            out[f"rule_vs_{jn}_agreement"] = round(float(np.mean(np.array(h) == np.array(j))), 3) if cl else None
            out[f"rule_vs_{jn}_kappa"] = round(kappa(h, j), 3) if cl else None
        res[name] = out

    agree([r for r in rows if r["group"] in ("D+C-", "D+C+")], "D+C-/D+C+ (480)")
    agree(rows, "all incl. D-C+")

    def wilson(k, n):
        if n == 0:
            return [None, None]
        p, z = k / n, 1.96
        d = 1 + z * z / n
        c = (p + z * z / (2 * n)) / d
        h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
        return [round(100 * (c - h), 1), round(100 * (c + h), 1)]

    rates = {}
    for m in MODELS:
        for g in ("D+C-", "D+C+", "D-C+"):
            sel = [r for r in rows if r["model"] == m and r["group"] == g]
            if not sel:
                continue
            d = {"n": len(sel), "rule_wrong": round(100 * float(np.mean([r["rule"] for r in sel])), 1)}
            for jn, key in (("claude", "j1"), ("gpt", "j2")):
                cl = [r for r in sel if r[key] != "unclear"]
                k = sum(r[key] == "incorrect" for r in cl)
                d[f"{jn}_incorrect_excl_unclear"] = round(100 * k / len(cl), 1) if cl else None
                d[f"{jn}_ci"] = wilson(k, len(cl))
                d[f"{jn}_n_clear"] = len(cl)
                d[f"{jn}_bound_unclear_as_correct"] = round(100 * k / len(sel), 1)
                d[f"{jn}_bound_unclear_as_incorrect"] = round(100 * (len(sel) - sum(r[key] == "correct" for r in sel)) / len(sel), 1)
            both = [r for r in sel if r["j1"] != "unclear" and r["j2"] != "unclear" and r["j1"] == r["j2"]]
            d["consensus_incorrect"] = round(100 * float(np.mean([r["j1"] == "incorrect" for r in both])), 1) if both else None
            d["consensus_n"] = len(both)
            d["consensus_ci"] = wilson(sum(r["j1"] == "incorrect" for r in both), len(both))
            rates[f"{m}|{g}"] = d
    res["rates"] = rates
    json.dump(res, open(out_dir / "judge2.json", "w"), indent=1)
    print(json.dumps({k: v for k, v in res.items() if k != "rates"}, indent=1))
    for k, v in rates.items():
        print(k, v)


def main() -> None:
    """CLI entry point."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--per", type=int, default=40)
    ap.add_argument("--analyze", action="store_true")
    args = ap.parse_args()
    if args.analyze:
        analyze(Path("outputs/bundle"))
        return
    syn = json.load(open(f"{SYN}/depth_ordering.json"))
    j1 = [json.loads(l) for l in open(O / "e2_judge.jsonl")]
    keep = ("model", "item_id", "group", "sentence", "heuristic_wrong")
    jobs = [{k: r[k] for k in keep} | {"judge_name": "gpt-6-astra"} for r in j1]
    rng = random.Random(0)
    for m in MODELS:
        cl = claims_for_group(m, "D-C+")
        smp = rng.sample(cl, min(args.per, len(cl)))
        print(f"[judge2] {m}: D-C+ parsed {len(cl)}, sampled {len(smp)}", flush=True)
        jobs += [c | {"judge_name": jn} for c in smp for jn in JUDGES]
    print(f"[judge2] {len(jobs)} judge calls", flush=True)
    res = judge_all(jobs, syn)
    with open(O / "e2_judge2.jsonl", "w") as f:
        for r in res:
            f.write(json.dumps(r) + "\n")
    print(f"[judge2] wrote {len(res)}; unclear {sum(r['judge'] == 'unclear' for r in res)}", flush=True)


if __name__ == "__main__":
    main()
