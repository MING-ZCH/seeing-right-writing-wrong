"""E19: point-binding audit of CoT rationales on BLINK Relative Depth (round-6 Opus M5).

Sample per model (Phi-3.5-Vision excluded: empty rationales): 12 CoT-wrong + 8 CoT-right items (seed 0).
Each judge sees the image (points A and B are marked) and the rationale, and labels how the rationale refers to the
marked points: CORRECT (describes what A and B actually mark), SWAPPED (A and B exchanged), MISNAMED (attaches a point
to a different object or region), NONE (never says what the points mark), UNCLEAR.
Rule statistics on all rationales: share mentioning both point labels.

Usage: SHARE=... python e19_blink_binding.py --judge gpt-6-astra   (key from env)
       python e19_blink_binding.py --analyze
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

from build_bundle import pe, rj

O = Path(f"{_vtp.OUTPUTS}")
B = Path("outputs/bundle")
M6 = ["qwen3b", "qwen7b", "internvl3_8b", "molmo7b", "qwen3vl_8b", "internvl35_8b"]
PROMPT = """The image has two marked points labelled A and B. A model wrote the following reasoning to decide which point is closer to the camera.

Reasoning:
<<<
{r}
>>>

Look at the image. How does the reasoning refer to the marked points?
- CORRECT: it says what A and B mark, and both descriptions match the image.
- SWAPPED: its descriptions of A and B are exchanged (what it says about A is true of B and vice versa).
- MISNAMED: at least one point is attached to an object or region it does not mark (not a clean swap).
- NONE: it never says what the points mark.
- UNCLEAR: cannot tell.
Reply with exactly one word: CORRECT, SWAPPED, MISNAMED, NONE or UNCLEAR."""


def sample() -> list[dict]:
    import io

    import pandas as pd
    from PIL import Image
    d = pd.read_parquet(f"{_vtp.DATASETS}/heldout/BLINK/Relative_Depth/val-00000-of-00001.parquet")
    imgs = {r["idx"]: {"image": Image.open(io.BytesIO(r["image_1"]["bytes"])).convert("RGB")} for _, r in d.iterrows()}
    rng = random.Random(0)
    out = []
    for m in M6:
        cot = [r for r in rj(O / "e6x" / f"{m}_rest.jsonl") if r["condition"] == "cot" and r["task"] == "blink_depth"]
        wrong = [r for r in cot if pe(r) != r["gold"]]
        right = [r for r in cot if pe(r) == r["gold"]]
        for grp, pool, n in (("cot_wrong", wrong, 12), ("cot_right", right, 8)):
            for r in rng.sample(pool, min(n, len(pool))):
                out.append({"model": m, "item_id": r["item_id"], "group": grp, "text": r["text"].replace("Let's think step by step.", "").strip(),
                            "image": imgs[r["item_id"]]["image"]})
    return out


def run(judge: str) -> None:
    import api_client
    items = sample()
    lock, res = threading.Lock(), []

    def go(c):
        r = api_client.chat(judge, [{"role": "user", "content": [api_client.img_part(c["image"]), {"type": "text", "text": PROMPT.format(r=c["text"])}]}],
                       max_tokens=2048, effort="high")
        w = (r.get("text") or "").strip().strip("*_` .\n").upper()
        lab = next((x for x in ("CORRECT", "SWAPPED", "MISNAMED", "NONE", "UNCLEAR") if w.startswith(x)), "UNPARSED")
        with lock:
            res.append({k: v for k, v in c.items() if k not in ("image", "text")} | {"judge": judge, "label": lab})

    with ThreadPoolExecutor(6) as ex:
        list(ex.map(go, items))
    (O / "e19").mkdir(exist_ok=True)
    with open(O / "e19" / f"{judge}.jsonl", "w") as f:
        for r in res:
            f.write(json.dumps(r) + "\n")
    print(f"[e19] {judge}: {len(res)}")


def analyze() -> None:
    J = {j: {(r["model"], r["item_id"]): r for r in rj(O / "e19" / f"{j}.jsonl")} for j in ("claude-opus-5-5", "gpt-6-astra")}
    keys = sorted(set(J["claude-opus-5-5"]) & set(J["gpt-6-astra"]))
    res = {"n": len(keys), "agree": round(float(np.mean([J["claude-opus-5-5"][k]["label"] == J["gpt-6-astra"][k]["label"] for k in keys])), 3)}
    for grp in ("cot_wrong", "cot_right"):
        ks = [k for k in keys if J["gpt-6-astra"][k]["group"] == grp]
        for j in J:
            labs = [J[j][k]["label"] for k in ks]
            res[f"{grp}|{j}"] = {l: round(100 * labs.count(l) / len(labs), 1) for l in ("CORRECT", "SWAPPED", "MISNAMED", "NONE", "UNCLEAR", "UNPARSED")} | {"n": len(labs)}
        cons = [J["gpt-6-astra"][k]["label"] for k in ks if J["gpt-6-astra"][k]["label"] == J["claude-opus-5-5"][k]["label"]]
        res[f"{grp}|consensus"] = {l: round(100 * cons.count(l) / max(1, len(cons)), 1) for l in ("CORRECT", "SWAPPED", "MISNAMED", "NONE", "UNCLEAR")} | {"n": len(cons)}
        bind_err = [k for k in ks if J["gpt-6-astra"][k]["label"] in ("SWAPPED", "MISNAMED") and J["claude-opus-5-5"][k]["label"] in ("SWAPPED", "MISNAMED")]
        res[f"{grp}|binding_error_both_judges"] = round(100 * len(bind_err) / max(1, len(ks)), 1)
    # rule: share of all BLINK-depth rationales mentioning both point labels
    rule = {}
    for m in M6:
        cot = [r for r in rj(O / "e6x" / f"{m}_rest.jsonl") if r["condition"] == "cot" and r["task"] == "blink_depth"]
        both = [bool(re.search(r"\b(point )?A\b", r["text"])) and bool(re.search(r"\b(point )?B\b", r["text"])) for r in cot]
        rule[m] = round(100 * float(np.mean(both)), 1)
    res["rule_mentions_both_labels"] = rule
    json.dump(res, open(B / "blink_binding.json", "w"), indent=1)
    print(json.dumps(res, indent=1))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--judge")
    ap.add_argument("--analyze", action="store_true")
    a = ap.parse_args()
    analyze() if a.analyze else run(a.judge)


if __name__ == "__main__":
    main()
