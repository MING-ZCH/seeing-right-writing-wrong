"""Audit the heuristic occlusion-claim parse (srww.claim_parse.first_claim rule) with an LLM judge that sees the image.

Samples up to --per items per (model, group) on Synthetic Depth, sends the image + the parsed sentence to the
judge, and writes outputs/e2_judge.jsonl; prints agreement / Cohen's kappa and judge-based contradiction rates.
"""
from __future__ import annotations
from srww import paths as _vtp  # noqa: E402

import argparse
import json
import random
import re
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image

from srww import api_client as api_client
from srww import records as B

O = Path(f"{_vtp.OUTPUTS}")
L = Path(f"{_vtp.LEGACY_RESULTS}")
SYN = f"{_vtp.DATASETS}/spatial_probing"
PROMPT = ("Here is an image and one sentence that a model wrote about it:\n\"{s}\"\n"
          "Consider only what the sentence claims about which shape is in front of, on top of, covering, or "
          "overlapping the other. Does that claim match the image? Reply with exactly one word: correct, incorrect, "
          "or unclear.")
OBJ = re.compile(r"\b(red|blue|green|yellow|purple|orange|cyan|gray|grey|pink|brown|black|white)\s+"
                 r"(circle|square|triangle|diamond|hexagon|pentagon|star|rectangle)s?\b", re.I)
ACT = re.compile(r"\b(overlap\w*|cover\w*|on top of|obscur\w*|block\w*|in front of)\b", re.I)
PAS = re.compile(r"\b(covered|hidden|obscured|blocked|overlapped|occluded)\s+by\b|\bbehind\b|\bunderneath\b|\bbeneath\b", re.I)
QRX = re.compile(r"Is the (\w+ \w+) in front of the (\w+ \w+)\?")


def parsed_claims(m: str) -> list[dict]:
    main = B.load_main(O, L, m)
    G = B.groups_from(main, L, O, m)
    syn = json.load(open(f"{SYN}/depth_ordering.json"))
    out = []
    for r in B.rj(O / "e2b" / f"{m}.jsonl"):
        if r["task"] != "syn_depth" or G.get(r["item_id"], {}).get("group") not in ("D+C-", "D+C+"):
            continue
        q = QRX.search(r["question"])
        if not q:
            continue
        x, y = q.group(1).lower(), q.group(2).lower()
        front = x if r["gold"] == "A" else y
        for s in r["sentences"]:
            objs = [(o.group(1) + " " + o.group(2)).lower().replace("grey", "gray") for o in OBJ.finditer(s)]
            if len(set(objs)) < 2 or not (ACT.search(s) or PAS.search(s)) or re.search(r"\?|whether|if the|determine|asks", s, re.I):
                continue
            first = objs[0]
            claimed = first if not PAS.search(s) else [o for o in objs if o != first][0]
            i = int(r["item_id"].split("_")[-1])
            out.append({"model": m, "item_id": r["item_id"], "group": G[r["item_id"]]["group"], "sentence": s,
                        "heuristic_wrong": int(claimed != front),
                        "image": syn[i]["image_path"].replace("<DATA_ROOT>",
                                                              f"{_vtp.DATA_ROOT}")})
            break
    return out


def kappa(a: list[int], b: list[int]) -> float:
    a, b = np.array(a), np.array(b)
    po = (a == b).mean()
    pe = a.mean() * b.mean() + (1 - a.mean()) * (1 - b.mean())
    return float((po - pe) / (1 - pe)) if pe < 1 else float("nan")


def main() -> None:
    """CLI entry point."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--judge", default="claude-opus-5-5")
    ap.add_argument("--per", type=int, default=40)
    ap.add_argument("--models", default="qwen3b,qwen7b,internvl3_8b,phi35v")
    ap.add_argument("--out", default="e2_judge.jsonl", help="file name under outputs/")
    args = ap.parse_args()
    rng = random.Random(0)
    sample = []
    for m in args.models.split(","):
        cl = parsed_claims(m)
        for grp in ("D+C-", "D+C+"):
            g = [c for c in cl if c["group"] == grp]
            sample += rng.sample(g, min(args.per, len(g)))
    print(f"[judge] {len(sample)} claims", flush=True)
    lock = threading.Lock()
    res = []

    def run(c):
        img = Image.open(c["image"]).convert("RGB")
        r = api_client.chat(args.judge, [{"role": "user", "content": [api_client.img_part(img), {"type": "text", "text": PROMPT.format(s=c["sentence"])}]}],
                       max_tokens=2048, effort="high")
        word = (r.get("text") or "").strip().strip("*_` \n").lower()
        lab = "correct" if word.startswith("correct") else "incorrect" if word.startswith("incorrect") else "unclear"
        with lock:
            res.append(c | {"judge": lab, "judge_raw": (r.get("text") or r.get("error", ""))[:80]})

    with ThreadPoolExecutor(12) as ex:
        list(ex.map(run, sample))
    with open(O / args.out, "w") as f:
        for r in res:
            f.write(json.dumps({k: v for k, v in r.items() if k != "image"}) + "\n")
    clear = [r for r in res if r["judge"] != "unclear"]
    h = [r["heuristic_wrong"] for r in clear]
    j = [int(r["judge"] == "incorrect") for r in clear]
    print(f"[judge] unclear {len(res) - len(clear)}/{len(res)}; agreement {np.mean(np.array(h) == np.array(j)):.3f}; kappa {kappa(h, j):.3f}")
    for m in args.models.split(","):
        for grp in ("D+C-", "D+C+"):
            s = [r for r in clear if r["model"] == m and r["group"] == grp]
            if s:
                print(f"  {m} {grp}: n={len(s)} judge-incorrect {np.mean([r['judge'] == 'incorrect' for r in s]):.2f} "
                      f"heuristic-wrong {np.mean([r['heuristic_wrong'] for r in s]):.2f}")


if __name__ == "__main__":
    main()
