"""Shared data loading, prompt construction and answer scoring for the v2 experiments.

All conditions share identical image + question tokens in the first user turn; they differ only
in what follows (empty / model CoT text R / length-matched filler) and in an optional extra user
turn (instruction only / restated question / re-supplied image / both).
"""
from __future__ import annotations
import vtax_paths as _vtp  # noqa: E402

import base64
import io
import json
import os
import random
import re
from typing import Any

from PIL import Image

ROOT = f"{_vtp.DATA_ROOT}"
POD_PREFIX = f"{_vtp.DATA_ROOT}"
PROJ = os.environ.get("VTAX_PROJ", f"{_vtp.ROOT}")

MODELS = {
    "qwen3b": f"{ROOT}/models/Qwen2.5-VL-3B-Instruct",
    "qwen7b": f"{ROOT}/models/Qwen2.5-VL-7B-Instruct",
    "internvl3_8b": f"{ROOT}/models/hf_cache/models--OpenGVLab--InternVL3-8B",
    "internvl25_8b": f"{ROOT}/models/hf_cache/models--OpenGVLab--InternVL2_5-8B",
    "molmo7b": f"{ROOT}/models/Molmo-7B-D-0924",
    "phi35v": f"{ROOT}/hf_cache/models--microsoft--Phi-3.5-vision-instruct",
    "qwen32b": f"{ROOT}/models/Qwen2.5-VL-32B-Instruct",
}

FORCE_PREFIX = "Answer: ("
FINAL_INSTR = "Now give your final answer."
COT_PREFIX = "Let's think step by step."

# Neutral, non-spatial filler prose (no relational / positional words).
FILLER_SENTENCES = [
    "The history of paper making spans many centuries and several cultures.",
    "Early writers recorded ideas on bark, cloth, clay and animal skin.",
    "Later, workers learned to soak plant fibers and press them into thin sheets.",
    "The method spread slowly through trade and the movement of skilled craftsmen.",
    "Mills powered by water made production faster and cheaper.",
    "Printed books became common, and literacy rates rose over time.",
    "Today most paper is produced from wood pulp in large industrial plants.",
    "Recycling programs recover a large share of used paper every year.",
    "Some artists still prefer handmade sheets for their texture and strength.",
    "Libraries preserve old documents by controlling humidity and temperature.",
    "Scholars study watermarks to estimate the age of historical manuscripts.",
    "The chemistry of acid-free paper helps documents last for generations.",
]


def resolve(path: str) -> str:
    return path.replace(POD_PREFIX, ROOT)


def model_path(name: str) -> str:
    p = MODELS[name]
    base = os.path.basename(p).replace("models--OpenGVLab--", "").replace("models--microsoft--", "")
    if os.path.exists(f"/dev/shm/models/{base}/.done"):
        return f"/dev/shm/models/{base}"
    snap = os.path.join(p, "snapshots")
    if os.path.isdir(snap):
        return os.path.join(snap, sorted(os.listdir(snap))[0])
    return p


def letters(n: int) -> list[str]:
    return [chr(ord("A") + i) for i in range(n)]


# ----------------------------------------------------------------------------- data

def load_cvbench(tasks: tuple[str, ...] = ("Relation", "Depth", "Distance", "Count")) -> list[dict]:
    from datasets import load_from_disk

    local = "/dev/shm/cv_bench_hf"
    ds = load_from_disk(local if os.path.exists(local) else f"{ROOT}/datasets/cv_bench/hf_dataset")
    ds = ds.remove_columns([c for c in ds.column_names if c not in ("idx", "task", "type", "prompt", "choices", "answer", "image", "source")])
    ds = ds.select([i for i, t in enumerate(ds["task"]) if t in tasks])
    out = []
    for ex in ds:
        gold = ex["answer"].strip("() ")
        out.append({
            "uid": f"cvb_{ex['task']}_{ex['idx']}_{ex['source']}",
            "dataset": "cvbench",
            "task": f"cvb_{ex['task'].lower()}",
            "question": ex["prompt"],
            "choices": letters(len(ex["choices"])),
            "gold": gold,
            "image": ex["image"].convert("RGB"),
            "group": f"cvb_{ex['source']}_{ex['idx']}_{ex['type']}",
        })
    return out


def load_synthetic(task: str) -> list[dict]:
    fn = {"relpos": "relative_position.json", "depth": "depth_ordering.json"}[task]
    data = json.load(open(f"{ROOT}/datasets/spatial_probing/{fn}"))
    out = []
    for i, ex in enumerate(data):
        p = resolve(ex["image_path"])
        out.append({
            "uid": f"syn_{task}_{i}",
            "dataset": "synthetic",
            "task": f"syn_{task}",
            "question": ex["question"] + "\n(A) yes\n(B) no",
            "choices": ["A", "B"],
            "gold": "A" if ex["answer"] == "yes" else "B",
            "image_path": p,
            "group": os.path.basename(p),
            "relation": ex.get("metadata", {}).get("relation"),
        })
    return out


def load_mmstar(n: int = 500, seed: int = 0) -> list[dict]:
    data = json.load(open(f"{ROOT}/datasets/mmstar/val.json"))
    random.Random(seed).shuffle(data)
    out = []
    for ex in data[:n]:
        q = ex["question"]
        opts = re.findall(r"(?:^|\s)([A-D]):", q.split("Options:")[-1]) if "Options:" in q else []
        n_opt = max(len(set(opts)), 2)
        out.append({
            "uid": f"mmstar_{ex['index']}",
            "dataset": "mmstar",
            "task": "mmstar_" + ex["category"].replace(" ", "_"),
            "question": q,
            "choices": letters(n_opt),
            "gold": ex["answer"].strip(),
            "image_path": resolve(ex["image_path"]),
            "group": f"mmstar_{ex['index']}",
        })
    return out


DATASETS = {
    "cvb_relation": lambda: load_cvbench(("Relation",)),
    "cvb_depth": lambda: load_cvbench(("Depth",)),
    "cvb_distance": lambda: load_cvbench(("Distance",)),
    "cvb_count": lambda: load_cvbench(("Count",)),
    "syn_relpos": lambda: load_synthetic("relpos"),
    "syn_depth": lambda: load_synthetic("depth"),
    "mmstar": lambda: load_mmstar(500),
}


def load_items(names: list[str], limit: int | None = None, seed: int = 0) -> list[dict]:
    items = []
    for n in names:
        xs = DATASETS[n]()
        if limit and len(xs) > limit:
            xs = random.Random(seed).sample(xs, limit)
        items.extend(xs)
    return items


def get_image(item: dict) -> Image.Image:
    if "image" in item:
        return item["image"]
    return Image.open(item["image_path"]).convert("RGB")


def image_data_url(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


# ----------------------------------------------------------------------------- filler

def filler_text(tokenizer, n_tokens: int) -> str:
    """Neutral prose truncated to exactly n_tokens tokens (under `tokenizer`)."""
    parts, ids = [], []
    k = 0
    while len(ids) < n_tokens:
        parts.append(FILLER_SENTENCES[k % len(FILLER_SENTENCES)])
        k += 1
        ids = tokenizer.encode(" ".join(parts), add_special_tokens=False)
    return tokenizer.decode(ids[:n_tokens])


# ----------------------------------------------------------------------------- conditions
# base context: direct (nothing), cot (model text R), filler (length-matched to R)
# suffix: none | turn (instruction-only user turn) | restate | reimage | reimage_restate

CONDITIONS = [
    ("direct", "none"), ("direct", "restate"),
    ("cot", "none"), ("cot", "turn"), ("cot", "restate"), ("cot", "reimage"), ("cot", "reimage_restate"),
    ("filler", "none"), ("filler", "turn"), ("filler", "restate"), ("filler", "reimage"),
    ("filler512", "none"), ("filler512", "restate"), ("filler512", "reimage"),
    ("filler1024", "none"), ("filler1024", "restate"),
]


def cond_name(base: str, suffix: str) -> str:
    return base if suffix == "none" else f"{base}+{suffix}"


def build_messages(item: dict, base: str, suffix: str, context: str, img_ref: Any) -> list[dict]:
    """OpenAI-style chat messages. `img_ref` is the image content part (dict).

    The final assistant message is always a partial answer ending in FORCE_PREFIX, to be continued.
    """
    q = item["question"]
    if base == "direct" and suffix == "restate":
        user = [img_ref, {"type": "text", "text": f"{q}\n\nQuestion again: {q}\n{FINAL_INSTR}"}]
        return [{"role": "user", "content": user}, {"role": "assistant", "content": FORCE_PREFIX}]
    msgs = [{"role": "user", "content": [img_ref, {"type": "text", "text": q}]}]
    if base == "direct":
        msgs.append({"role": "assistant", "content": FORCE_PREFIX})
        return msgs
    if suffix == "none":
        msgs.append({"role": "assistant", "content": f"{context}\n{FORCE_PREFIX}"})
        return msgs
    msgs.append({"role": "assistant", "content": context})
    extra = {
        "turn": [{"type": "text", "text": FINAL_INSTR}],
        "restate": [{"type": "text", "text": f"Question again: {q}\n{FINAL_INSTR}"}],
        "reimage": [img_ref, {"type": "text", "text": f"Here is the image again.\n{FINAL_INSTR}"}],
        "reimage_restate": [img_ref, {"type": "text", "text": f"Here is the image again.\nQuestion again: {q}\n{FINAL_INSTR}"}],
    }[suffix]
    msgs.append({"role": "user", "content": extra})
    msgs.append({"role": "assistant", "content": FORCE_PREFIX})
    return msgs


def cot_messages(item: dict, img_ref: Any) -> list[dict]:
    return [
        {"role": "user", "content": [img_ref, {"type": "text", "text": item["question"]}]},
        {"role": "assistant", "content": COT_PREFIX},
    ]


# ----------------------------------------------------------------------------- scoring

def parse_free_answer(text: str, choices: list[str]) -> str | None:
    pat = "|".join(choices)
    for rx in [rf"answer is:?\s*\(?({pat})\b", rf"\(({pat})\)", rf"(?:^|\s)({pat})[\.\):]"]:
        m = re.findall(rx, text)
        if m:
            return m[-1]
    return None


def letter_scores(top_logprobs: dict[str, float], choices: list[str]) -> dict[str, float]:
    """Max logprob per choice letter among top-k decoded tokens; missing letters get -inf."""
    import math

    out = {c: -math.inf for c in choices}
    for tok, lp in top_logprobs.items():
        t = tok.strip().strip("(").strip()
        if t in out and lp > out[t]:
            out[t] = lp
    return out


def normalized_p(scores: dict[str, float], gold: str) -> float:
    import math

    finite = {k: v for k, v in scores.items() if v > -math.inf}
    if gold not in finite:
        return 0.0
    m = max(finite.values())
    z = sum(math.exp(v - m) for v in finite.values())
    return math.exp(finite[gold] - m) / z
