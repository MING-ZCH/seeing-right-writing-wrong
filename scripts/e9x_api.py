"""E9x: closed / API VLMs on the depth audit (black-box answers only, reasoning_effort=high).

Items: occlusion twins under both phrasings (800), Synthetic Depth Ordering sample (250, seed 0; front object
larger), BLINK val Relative Depth (124) and Spatial Relation (143, 2D control).
Conditions:
  direct  question + "Answer with the option's letter only." (hidden reasoning allowed by the API)
  cot     question + written step-by-step description ending with "Answer: (X)"
Results stream to outputs/e9x/<model>.jsonl (resumable). Key from OPENAI_COMPAT_API_KEY.

Usage: python e9x_api.py --model claude-opus-5-5 [--limit 5] [--workers 12]
"""
from __future__ import annotations
import vtax_paths as _vtp  # noqa: E402

import argparse
import io
import json
import os
import random
import re
import threading
from concurrent.futures import ThreadPoolExecutor

import pandas as pd
from PIL import Image

import api_client

W = f"{_vtp.DATA_ROOT}"
OCC = f"{_vtp.W}/exps/items/occlusion"
SYN = f"{W}/datasets/spatial_probing"
BLINK = f"{W}/datasets/heldout/BLINK"
OUT = f"{_vtp.W}/outputs/e9x"
DIRECT = "Answer with the option's letter only."
COT = ("Let's think step by step: first describe what you see that matters for the question, then reason, "
       "and end with 'Answer: (X)' where X is the option letter.")


def parse_letter(text: str, letters: list[str]) -> str | None:
    pat = "|".join(letters)
    for rx in (rf"Answer:\s*\(?({pat})\)?", rf"\(({pat})\)", rf"^\s*\(?({pat})\b"):
        m = re.findall(rx, text or "")
        if m:
            return m[-1]
    return None


def load_items(limit: int | None, items_dir: str | None = None, tasks: str | None = None) -> list[dict]:
    """Occlusion twins (both phrasings) + Syn. Depth + BLINK; ``tasks`` keeps task-name prefixes."""
    occ = items_dir or OCC
    items = []
    for fn, ph in (("items.jsonl", "front"), ("items_cf.jsonl", "front"), ("items.jsonl", "behind"), ("items_cf.jsonl", "behind")):
        for r in map(json.loads, open(f"{occ}/{fn}")):
            q, g = r["question"], r["gold"]
            if ph == "behind":
                q, g = q.replace(" in front of ", " behind ", 1), ("B" if g == "A" else "A")
            items.append({"uid": f"{r['item_id']}_{ph}", "task": f"occ_{ph}", "question": q + "\n(A) yes\n(B) no",
                          "letters": ["A", "B"], "gold": g, "image": f"{occ}/{r['image_paths'][0]}",
                          "group": r["item_id"].replace("_cf", ""), "meta": r["meta"], "phrasing": ph})
    keep = tuple(tasks.split(",")) if tasks else None
    if keep and not any(t.startswith(keep) for t in ("syn_depth", "blink_depth", "blink_relation")):
        return [it for it in items if it["task"].startswith(keep)]
    syn = json.load(open(f"{SYN}/depth_ordering.json"))
    idx = sorted(random.Random(0).sample(range(len(syn)), 250))
    for i in idx:
        ex = syn[i]
        items.append({"uid": f"syn_depth_{i}", "task": "syn_depth", "question": ex["question"] + "\n(A) yes\n(B) no",
                      "letters": ["A", "B"], "gold": "A" if ex["answer"] == "yes" else "B",
                      "image": ex["image_path"].replace("<DATA_ROOT>", W),
                      "group": f"syn_depth_{i}"})
    for task, sub in (("blink_depth", "Relative_Depth"), ("blink_relation", "Spatial_Relation")):
        d = pd.read_parquet(f"{BLINK}/{sub}/val-00000-of-00001.parquet")
        for _, r in d.iterrows():
            items.append({"uid": r["idx"], "task": task, "question": r["prompt"],
                          "letters": [chr(65 + i) for i in range(len(list(r["choices"])))],
                          "gold": r["answer"].strip("() "), "image_bytes": r["image_1"]["bytes"], "group": r["idx"]})
    if keep:
        items = [it for it in items if it["task"].startswith(keep)]
    if limit:
        by = {}
        for it in items:
            by.setdefault(it["task"], []).append(it)
        items = [x for v in by.values() for x in v[:limit]]
    return items


def main() -> None:
    """CLI entry point."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--effort", default="high")
    ap.add_argument("--items-dir", default=OCC, help="occlusion item directory (default: original set)")
    ap.add_argument("--tasks", default=None, help="comma-separated task prefixes to keep (e.g. occ)")
    ap.add_argument("--out-dir", default=OUT, help="output directory (e.g. .../outputs/e9h)")
    ap.add_argument("--tag", default="", help="suffix added to the output file name")
    ap.add_argument("--no-reasoning-direct", action="store_true",
                    help='only run condition "direct_nr": letter-only answer with reasoning_effort="none" '
                         "(occlusion twins, Syn. Depth, BLINK Relative Depth)")
    ap.add_argument("--reverse", action="store_true", help="process jobs in reverse order (second process)")
    ap.add_argument("--level-priority", default=None,
                    help="e.g. L4,L3b,L3: run occlusion jobs level by level, twin pair by twin pair (rate-limited models)")
    ap.add_argument("--nr-effort", default="none",
                    help='effort for direct_nr; models without "none" take their lowest level (claude: minimal, '
                         "gpt: low); reasoning tokens are stored per record so absence can be verified")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    out = f"{args.out_dir}/{args.model}{args.tag}{'_limit' + str(args.limit) if args.limit else ''}.jsonl"
    done = set()
    if os.path.exists(out):
        for r in map(json.loads, open(out)):
            if "error" not in r:
                done.add((r["item_id"], r["condition"]))
    items = load_items(args.limit, args.items_dir, args.tasks)
    if args.no_reasoning_direct:
        conds = ("direct_nr",)
        items = [it for it in items if it["task"] != "blink_relation"]
    else:
        conds = ("direct", "cot")
    jobs = [(it, c) for it in items for c in conds if (it["uid"], c) not in done]
    if args.level_priority:
        pri = {lv: i for i, lv in enumerate(args.level_priority.split(","))}
        jobs.sort(key=lambda j: (pri.get((j[0].get("meta") or {}).get("level"), len(pri)), j[0]["group"], j[0]["uid"], j[1]))
    if args.reverse:
        jobs = jobs[::-1]
    print(f"[e9x] {args.model}: {len(jobs)} calls to run ({len(done)} done)", flush=True)
    lock = threading.Lock()

    def run(job):
        it, cond = job
        img = Image.open(io.BytesIO(it["image_bytes"])) if "image_bytes" in it else Image.open(it["image"])
        text = it["question"] + "\n" + (COT if cond == "cot" else DIRECT)
        eff = args.nr_effort if cond == "direct_nr" else args.effort
        r = api_client.chat(args.model, [{"role": "user", "content": [api_client.img_part(img.convert("RGB")), {"type": "text", "text": text}]}],
                       max_tokens=16384 if cond != "direct_nr" else 512, effort=eff)
        rec = {"model": args.model, "item_id": it["uid"], "task": it["task"], "condition": cond, "gold": it["gold"], "effort": eff,
               "letters": it["letters"], "image_id": it["group"], "meta": it.get("meta"), "phrasing": it.get("phrasing")}
        if "error" in r:
            rec["error"] = r["error"]
        else:
            rec.update(text=r["text"], pred=parse_letter(r["text"], it["letters"]),
                       reasoning_tokens=(r["usage"].get("completion_tokens_details") or {}).get("reasoning_tokens"),
                       completion_tokens=r["usage"].get("completion_tokens"), raw_model=r["raw_model"],
                       reasoning_chars=len(r.get("reasoning") or ""))
        with lock, open(out, "a") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    with ThreadPoolExecutor(args.workers) as ex:
        list(ex.map(run, jobs))
    print(f"[e9x] {args.model}: finished", flush=True)


if __name__ == "__main__":
    main()
