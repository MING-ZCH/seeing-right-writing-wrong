"""Build canonical item manifests (JSONL + PNG images of ``srww.common.Item`` records) from public benchmarks.

Sources:
  cvbench   HF ``nyu-visionx/CV-Bench`` test split (2D Relation / 3D Depth /
            3D Distance / 2D Count).
  blink     HF ``BLINK-Benchmark/BLINK`` val split (Spatial_Relation,
            Relative_Depth, Multi-view_Reasoning).
  hf        Generic HF dataset with column mapping (3DSRBench, What'sUp).
  jsonl     Existing local manifest (e.g. the original synthetic sets) with a
            field mapping and image root.

Examples::

    python -m data_gen.benchmark_items cvbench --out items/cvbench.jsonl
    python -m data_gen.benchmark_items blink --out items/blink_val.jsonl
    python -m data_gen.benchmark_items hf --hf-name ccvl/3DSRBench --split test --question-col question \
        --option-cols A B C D --answer-col answer --image-col image \
        --filter-col image_source --filter-regex coco --sample 500 --task-prefix 3dsr \
        --out items/3dsr_real500.jsonl
    python -m data_gen.benchmark_items jsonl --path <syn_depth.jsonl> --image-root <image dir> \
        --map item_id=id question=question options=options gold=answer image=image \
        --task syn_depth --out items/syn_depth.jsonl
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import random
import re
from pathlib import Path
from typing import Any

from srww.common import LETTERS, Item, image_hash, read_jsonl

CVB_TASKS = {"Relation": "cvb_relation", "Depth": "cvb_depth", "Distance": "cvb_distance", "Count": "cvb_count"}
BLINK_CONFIGS = {"Spatial_Relation": "blink_spatial_relation", "Relative_Depth": "blink_relative_depth",
                 "Multi-view_Reasoning": "blink_multiview"}


def save_image(img: Any, img_dir: Path) -> tuple[str, str]:
    """Save a PIL image / bytes / base64 string as PNG; return (relative path, hash)."""
    from PIL import Image

    if isinstance(img, dict) and "bytes" in img:
        img = img["bytes"]
    if isinstance(img, str) and len(img) > 512:
        img = base64.b64decode(img)
    if isinstance(img, (bytes, bytearray)):
        img = Image.open(io.BytesIO(img))
    if isinstance(img, str):
        img = Image.open(img)
    img = img.convert("RGB")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    data = buf.getvalue()
    h = image_hash(data)
    img_dir.mkdir(parents=True, exist_ok=True)
    p = img_dir / f"{h}.png"
    if not p.exists():
        p.write_bytes(data)
    return f"{img_dir.name}/{p.name}", h


def gold_letter(ans: Any, options: list[str]) -> str:
    """Resolve the gold letter from ``(A)``, ``A``, an index, or the option text."""
    if isinstance(ans, int):
        return LETTERS[ans]
    s = str(ans).strip()
    m = re.match(r"^\(?([A-H])\)?$", s)
    if m:
        return m.group(1)
    low = [o.strip().lower() for o in options]
    if s.lower() in low:
        return LETTERS[low.index(s.lower())]
    raise ValueError(f"cannot resolve gold {ans!r} among {options}")


def write_items(items: list[Item], out: Path) -> None:
    """Write a manifest and print per-task counts."""
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        for it in items:
            f.write(json.dumps(it.to_json(), ensure_ascii=False) + "\n")
    counts: dict[str, int] = {}
    for it in items:
        counts[it.task] = counts.get(it.task, 0) + 1
    print(f"wrote {len(items)} items to {out}: {counts}")


def build_cvbench(args: argparse.Namespace) -> list[Item]:
    """CV-Bench test split."""
    from datasets import load_dataset

    ds = load_dataset("nyu-visionx/CV-Bench", split="test", revision=args.hf_revision)
    img_dir = Path(args.out).parent / "images"
    items = []
    for r in ds:
        task = CVB_TASKS.get(r["task"])
        if task is None or (args.tasks and task not in args.tasks):
            continue
        rel, h = save_image(r["image"], img_dir)
        opts = [str(c) for c in r["choices"]]
        meta = {k: r.get(k) for k in ("type", "filename", "source", "source_dataset", "source_filename",
                                      "target_class", "target_size", "bbox") if k in r}
        items.append(Item(item_id=args.id_template.format(idx=r["idx"], task=task), task=task, image_paths=[rel],
                          question=r["question"], options=opts, gold=gold_letter(r["answer"], opts), image_id=h,
                          prompt=r["prompt"] if args.use_dataset_prompt else None, meta=meta))
    return items


def build_blink(args: argparse.Namespace) -> list[Item]:
    """BLINK val split for the three spatial subtasks."""
    from datasets import load_dataset

    img_dir = Path(args.out).parent / "images"
    items = []
    for cfg, task in BLINK_CONFIGS.items():
        ds = load_dataset("BLINK-Benchmark/BLINK", cfg, split="val", revision=args.hf_revision)
        for r in ds:
            rels, hs = [], []
            for k in ("image_1", "image_2", "image_3", "image_4"):
                if r.get(k) is not None:
                    rel, h = save_image(r[k], img_dir)
                    rels.append(rel)
                    hs.append(h)
            opts = [str(c) for c in r["choices"]]
            items.append(Item(item_id=str(r["idx"]), task=task, image_paths=rels, question=r["question"],
                              options=opts, gold=gold_letter(r["answer"], opts), image_id="+".join(hs),
                              meta={"sub_task": r.get("sub_task")}))
    return items


def build_hf_generic(args: argparse.Namespace) -> list[Item]:
    """Generic HF dataset with explicit column mapping."""
    from datasets import load_dataset

    ds = load_dataset(args.hf_name, args.hf_config, split=args.split, revision=args.hf_revision)
    img_dir = Path(args.out).parent / "images"
    rows = list(range(len(ds)))
    if args.filter_col:
        rx = re.compile(args.filter_regex)
        rows = [i for i in rows if rx.search(str(ds[i][args.filter_col]))]
    if args.dedup_col:
        seen, keep = set(), []
        for i in rows:
            key = ds[i][args.dedup_col]
            if key not in seen:
                seen.add(key)
                keep.append(i)
        rows = keep
    if args.sample and len(rows) > args.sample:
        rows = sorted(random.Random(args.seed).sample(rows, args.sample))
    items = []
    for i in rows:
        r = ds[i]
        opts = [str(r[c]) for c in args.option_cols if r.get(c) not in (None, "", "nan")]
        img = r[args.image_col]
        if args.image_root and isinstance(img, str) and len(img) < 512:
            img = str(Path(args.image_root) / img)
        rel, h = save_image(img, img_dir)
        task = args.task_prefix + (f"_{r[args.task_col]}" if args.task_col else "")
        items.append(Item(item_id=f"{args.task_prefix}_{r[args.id_col] if args.id_col else i}", task=task,
                          image_paths=[rel], question=str(r[args.question_col]), options=opts,
                          gold=gold_letter(r[args.answer_col], opts), image_id=h,
                          meta={"row": i, **({args.filter_col: r[args.filter_col]} if args.filter_col else {})}))
    return items


def build_from_jsonl(args: argparse.Namespace) -> list[Item]:
    """Map an existing manifest (e.g. legacy synthetic sets) to canonical items."""
    fmap = dict(kv.split("=", 1) for kv in args.map)
    img_dir = Path(args.out).parent / "images"
    items = []
    for r in read_jsonl(args.path):
        opts = r[fmap.get("options", "options")]
        if isinstance(opts, dict):
            opts = [opts[k] for k in sorted(opts)]
        opts = [str(o) for o in opts]
        img = r[fmap.get("image", "image")]
        if isinstance(img, str) and args.image_root and not Path(img).is_absolute():
            img = str(Path(args.image_root) / img)
        rel, h = save_image(img, img_dir)
        meta = {k: v for k, v in r.items() if k not in fmap.values()}
        items.append(Item(item_id=str(r[fmap.get("item_id", "item_id")]), task=args.task or r.get("task"),
                          image_paths=[rel], question=str(r[fmap.get("question", "question")]), options=opts,
                          gold=gold_letter(r[fmap.get("gold", "gold")], opts),
                          image_id=str(r.get(fmap["image_id"])) if "image_id" in fmap else h,
                          prompt=r.get(fmap["prompt"]) if "prompt" in fmap else None, meta=meta))
    return items


def main() -> None:
    """CLI entry point."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("source", choices=["cvbench", "blink", "hf", "jsonl"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--hf-revision", default=None)
    ap.add_argument("--tasks", nargs="*", default=None)
    ap.add_argument("--id-template", default="cvb_{idx}")
    ap.add_argument("--use-dataset-prompt", action="store_true",
                    help="use CV-Bench's 'prompt' field verbatim as q")
    ap.add_argument("--hf-name")
    ap.add_argument("--hf-config", default=None)
    ap.add_argument("--split", default="test")
    ap.add_argument("--question-col", default="question")
    ap.add_argument("--option-cols", nargs="*", default=["A", "B", "C", "D"])
    ap.add_argument("--answer-col", default="answer")
    ap.add_argument("--image-col", default="image")
    ap.add_argument("--image-root", default=None)
    ap.add_argument("--id-col", default=None)
    ap.add_argument("--task-col", default=None)
    ap.add_argument("--task-prefix", default="hf")
    ap.add_argument("--filter-col", default=None)
    ap.add_argument("--filter-regex", default=".*")
    ap.add_argument("--dedup-col", default=None, help="keep first row per value (e.g. circular-eval base index)")
    ap.add_argument("--sample", type=int, default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--path")
    ap.add_argument("--map", nargs="*", default=[])
    ap.add_argument("--task", default=None)
    args = ap.parse_args()
    fn = {"cvbench": build_cvbench, "blink": build_blink, "hf": build_hf_generic, "jsonl": build_from_jsonl}
    write_items(fn[args.source](args), Path(args.out))


if __name__ == "__main__":
    main()
