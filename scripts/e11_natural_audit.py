"""Natural-image claim audit: first two-object depth claim in stored CoT rationales, judged by two LLM judges.

Sets: BLINK Relative Depth (val, 124; rationales from the held-out e6x runs) and CV-Bench 3D Depth (600; legacy e1 /
e4 rationales). Groups from the same runs (Direct vs CoT prediction). Sample per model: BLINK all D+C- and <= 30 D+C+,
CV-Bench <= 40 D+C- and <= 30 D+C+ (seed 0). Judges: claude-opus-5-5 and gpt-6-astra (reasoning_effort=high), with the
e2_judge protocol adapted to depth ("which of the two is closer to the camera"); markdown-robust label parsing.

Usage (dev machine, from vtax_exps/, API env loaded): SHARE=<dev share> python e11_natural_audit.py [--analyze]
"""
from __future__ import annotations
import vtax_paths as _vtp  # noqa: E402

import argparse
import io
import json
import math
import random
import re
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

O = Path(f"{_vtp.OUTPUTS}")
H = f"{_vtp.DATASETS}/heldout"
L = Path(f"{_vtp.LEGACY_RESULTS}")
MODELS = ["qwen3b", "qwen7b", "internvl3_8b", "phi35v", "molmo7b", "qwen3vl_8b", "internvl35_8b"]
JUDGES = ["claude-opus-5-5", "gpt-6-astra"]
PROMPT = ("Here is an image and one sentence that a model wrote about it:\n\"{s}\"\n"
          "Consider only what the sentence claims about which of the two marked points or objects is closer to the camera "
          "(or in front of the other). Does that claim match the image? Reply with exactly one word: correct, incorrect, "
          "or unclear.")
DEPTH = re.compile(r"\b(closer|nearer|farther|further|in front|behind|foreground|background|distance|distant|depth|"
                   r"deeper|occlud\w*|overlap\w*|cover\w*|blocks?)\b", re.I)
SKIP = re.compile(r"\?|whether|determine|\basks?\b|which (one|point|object)|we need|need to", re.I)
PT = re.compile(r"\b(?:point|circle|label(?:ed)?|marker|spot)\s+([AB])\b|\(([AB])\)|\b([AB])\b(?=\s+(?:is|appears|seems|lies|sits))")
_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z(\"'*\d])|\n+")


def sentences(text: str) -> list[str]:
    t = text[len("Let's think step by step."):] if text.startswith("Let's think step by step.") else text
    return [s.strip() for s in _SPLIT.split(t) if s and s.strip()]


def first_claim(text: str, kind: str, names: tuple[str, str] | None) -> str | None:
    for s in sentences(text):
        if not DEPTH.search(s) or SKIP.search(s):
            continue
        if kind == "blink":
            labs = {next(g for g in m.groups() if g) for m in PT.finditer(s)}
            if {"A", "B"} <= labs:
                return s
        elif all(n.lower() in s.lower() for n in names):
            return s
    return None


def label(raw: str) -> str:
    w = (raw or "").strip().strip("*_` \n").lower()
    return "correct" if w.startswith("correct") else "incorrect" if w.startswith("incorrect") else "unclear"


def argmax(lp: dict) -> str | None:
    fin = {k: v for k, v in lp.items() if v is not None and math.isfinite(v)}
    return max(fin, key=fin.get) if fin else None


def build_claims() -> tuple[list[dict], dict]:
    """All parsed claims per model / set / group, plus coverage counts."""
    claims, cov = [], {}
    for m in MODELS:
        # BLINK Relative Depth (held-out runs)
        d = {r["item_id"]: r for r in map(json.loads, open(O / "e6x" / f"{m}_direct.jsonl")) if r["task"] == "blink_depth"}
        c = {r["item_id"]: r for r in map(json.loads, open(O / "e6x" / f"{m}_rest.jsonl"))
             if r["task"] == "blink_depth" and r["condition"] == "cot"}
        for k in d:
            if k not in c:
                continue
            pd_, pc = argmax(d[k].get("lp_sum") or d[k]["top20"]), argmax(c[k].get("lp_sum") or c[k]["top20"])
            g = ("D+" if pd_ == d[k]["gold"] else "D-") + ("C+" if pc == d[k]["gold"] else "C-")
            s = first_claim(c[k]["text"], "blink", None)
            cov.setdefault(f"{m}|blink|{g}", [0, 0])
            cov[f"{m}|blink|{g}"][0] += 1
            if s:
                cov[f"{m}|blink|{g}"][1] += 1
                claims.append({"model": m, "set": "blink", "item_id": k, "group": g, "sentence": s})
        # CV-Bench 3D Depth (legacy / e4 runs)
        src = L / "e1" / f"{m}.json"
        if not src.exists():
            src = O / "e4" / f"{m}.json"
        for it in json.load(open(src))["items"]:
            if it["task"] != "cvb_depth":
                continue
            sc = it["scores"]
            g = ("D+" if argmax(sc["direct"]) == it["gold"] else "D-") + ("C+" if argmax(sc["cot"]) == it["gold"] else "C-")
            opts = dict(re.findall(r"\(([AB])\)\s*(.+)", it["question"]))
            s = first_claim(it["cot"], "cvb", (opts.get("A", "").strip(), opts.get("B", "").strip()))
            cov.setdefault(f"{m}|cvb|{g}", [0, 0])
            cov[f"{m}|cvb|{g}"][0] += 1
            if s:
                cov[f"{m}|cvb|{g}"][1] += 1
                claims.append({"model": m, "set": "cvb", "item_id": it["uid"], "group": g, "sentence": s})
    return claims, cov


def sample(claims: list[dict]) -> list[dict]:
    rng = random.Random(0)
    caps = {("blink", "D+C-"): 10 ** 6, ("blink", "D+C+"): 30, ("cvb", "D+C-"): 40, ("cvb", "D+C+"): 30}
    out = []
    for m in MODELS:
        for (st, g), cap in caps.items():
            pool = [c for c in claims if c["model"] == m and c["set"] == st and c["group"] == g]
            out += rng.sample(pool, min(cap, len(pool)))
    return out


def images(sample_: list[dict]) -> dict:
    """PIL image per (set, item_id)."""
    import pandas as pd
    from PIL import Image
    imgs = {}
    d = pd.read_parquet(f"{H}/BLINK/Relative_Depth/val-00000-of-00001.parquet")
    for _, r in d.iterrows():
        imgs[("blink", r["idx"])] = Image.open(io.BytesIO(r["image_1"]["bytes"])).convert("RGB")
    need = {c["item_id"] for c in sample_ if c["set"] == "cvb"}
    if need:
        import vx
        for it in vx.load_items(["cvb_depth"]):
            if it["uid"] in need:
                imgs[("cvb", it["uid"])] = vx.C.get_image(it)
    return imgs


def run() -> None:
    import api_client
    claims, cov = build_claims()
    smp = sample(claims)
    imgs = images(smp)
    jobs = [c | {"judge_name": j} for c in smp for j in JUDGES]
    print(f"[e11] {len(claims)} parsed claims, {len(smp)} sampled, {len(jobs)} judge calls", flush=True)
    lock, res = threading.Lock(), []

    def one(c):
        r = {}
        for _ in range(3):
            r = api_client.chat(c["judge_name"], [{"role": "user", "content": [api_client.img_part(imgs[(c["set"], c["item_id"])]),
                                                                           {"type": "text", "text": PROMPT.format(s=c["sentence"])}]}],
                           max_tokens=2048, effort="high")
            if r.get("text"):
                break
        raw = (r.get("text") or r.get("error", ""))[:80]
        with lock:
            res.append(c | {"judge": label(raw), "judge_raw": raw})
            if len(res) % 100 == 0:
                print(f"[e11] {len(res)}/{len(jobs)}", flush=True)

    with ThreadPoolExecutor(12) as ex:
        list(ex.map(one, jobs))
    with open(O / "e11_natural_audit.jsonl", "w") as f:
        for r in res:
            f.write(json.dumps(r) + "\n")
    json.dump({k: v for k, v in cov.items()}, open(O / "e11_natural_coverage.json", "w"), indent=1)
    print(f"[e11] wrote {len(res)}; unclear {sum(r['judge'] == 'unclear' for r in res)}; "
          f"errors {sum(r['judge_raw'].startswith('HTTP') for r in res)}", flush=True)


def analyze(out_dir: Path) -> None:
    from e2_judge import kappa

    def wilson(k, n):
        if n == 0:
            return [None, None]
        p, z = k / n, 1.96
        d = 1 + z * z / n
        c = (p + z * z / (2 * n)) / d
        h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
        return [round(max(0.0, 100 * (c - h)), 1), round(min(100.0, 100 * (c + h)), 1)]

    R = [json.loads(l) for l in open(O / "e11_natural_audit.jsonl")]
    cov = json.load(open(O / "e11_natural_coverage.json"))
    lab = {(r["model"], r["set"], r["item_id"], r["judge_name"]): r for r in R}
    keys = sorted({(r["model"], r["set"], r["item_id"], r["group"]) for r in R})
    res = {"rates": {}, "coverage": {k: {"n_items": v[0], "parsed": v[1], "parsed_share": round(100 * v[1] / v[0], 1) if v[0] else None}
                                     for k, v in cov.items()}}
    both = []
    for m, st, i, g in keys:
        a, b = lab.get((m, st, i, JUDGES[0])), lab.get((m, st, i, JUDGES[1]))
        if a and b:
            both.append((m, st, g, a["judge"], b["judge"]))
    cl = [x for x in both if x[3] != "unclear" and x[4] != "unclear"]
    x1, x2 = [int(x[3] == "incorrect") for x in cl], [int(x[4] == "incorrect") for x in cl]
    res["agreement"] = {"n": len(both), "n_both_clear": len(cl), "agreement": round(float(np.mean(np.array(x1) == np.array(x2))), 3),
                        "kappa": round(kappa(x1, x2), 3), "unclear_opus": sum(x[3] == "unclear" for x in both),
                        "unclear_gpt": sum(x[4] == "unclear" for x in both)}
    for st in ("blink", "cvb"):
        s = [x for x in cl if x[1] == st]
        a1, a2 = [int(x[3] == "incorrect") for x in s], [int(x[4] == "incorrect") for x in s]
        res["agreement"][st] = {"n_both_clear": len(s), "agreement": round(float(np.mean(np.array(a1) == np.array(a2))), 3) if s else None,
                                "kappa": round(kappa(a1, a2), 3) if s else None}
    lines = []
    for m in MODELS:
        for st in ("blink", "cvb"):
            for g in ("D+C-", "D+C+"):
                sel = [x for x in both if x[0] == m and x[1] == st and x[2] == g]
                if not sel:
                    continue
                d = {"n": len(sel)}
                for jn, j in (("opus", 3), ("gpt", 4)):
                    c_ = [x for x in sel if x[j] != "unclear"]
                    k = sum(x[j] == "incorrect" for x in c_)
                    d[f"{jn}_incorrect"] = round(100 * k / len(c_), 1) if c_ else None
                    d[f"{jn}_ci"] = wilson(k, len(c_))
                    d[f"{jn}_n_clear"] = len(c_)
                    d[f"{jn}_bounds"] = [round(100 * k / len(sel), 1),
                                         round(100 * (len(sel) - sum(x[j] == "correct" for x in sel)) / len(sel), 1)]
                cons = [x for x in sel if x[3] == x[4] != "unclear"]
                k = sum(x[3] == "incorrect" for x in cons)
                d["consensus_incorrect"] = round(100 * k / len(cons), 1) if cons else None
                d["consensus_ci"] = wilson(k, len(cons))
                d["consensus_n"] = len(cons)
                res["rates"][f"{m}|{st}|{g}"] = d
                lines.append(f"{m:14s} {st:5s} {g} n={len(sel):3d} opus {d['opus_incorrect']} {d['opus_ci']} gpt {d['gpt_incorrect']} "
                             f"{d['gpt_ci']} cons {d['consensus_incorrect']} {d['consensus_ci']} (n {len(cons)})")
    json.dump(res, open(out_dir / "natural_audit.json", "w"), indent=1)
    print(json.dumps(res["agreement"]))
    print("\n".join(lines))


def main() -> None:
    """CLI entry point."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--analyze", action="store_true")
    args = ap.parse_args()
    analyze(Path("outputs/bundle")) if args.analyze else run()


if __name__ == "__main__":
    main()
