"""E29 judge calls.

Tasks (Claude Opus 5.5 + GPT-6-Astra, effort high, api_client client as judges/commit_sentence_judge.py; key from OPENAI_COMPAT_API_KEY):
  prej2   Qwen3-VL-8B cue-set D+C- rationales: every relational sentence before j* and j* itself, judged with the e20
          prompt (identical to Exp-28 prej), for the 40 Exp-28 commit-sample items plus 80 more D+C- items with j*
          (seed 290). Exp-28 labels are reused for the sentences they already cover. Feeds the Task-2 edit at the
          earliest image-contradicting relational sentence.
  ontop   Re-judge with a prompt that separates DEPTH (layering) from VERTICAL (2D position) readings: Exp-28 cue-set and
          twin commit samples (D+C- j* and relational D+C+ comparators), Exp-28 cue-set pre-j* sentences, and the e20
          Synthetic Depth audit (D+C- commits of seven models + relational D+C+ comparators).
  desc    Full-text depth-claim audit: question-blind descriptions of the Exp-28 cue-set description arm (per model
          50 images with >= 1 Direct-right / Desc-wrong item and 50 images with Direct and Desc right on both phrasings,
          seed 291) and, as the CoT comparator, the full CoT rationales of the Exp-28 cue commit sample (40 D+C- and
          40 relational D+C+ per model), with the same prompt.
-> outputs/e29/judge_<judge>_<task>.jsonl

Usage: python -m judges.claim_edit_judge --task ontop --judge claude-opus-5-5 [--dry] [--limit 4]
"""
from __future__ import annotations
from srww import paths as _vtp  # noqa: E402

import argparse
import json
import random
import re
import threading
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from srww import records as B
from judges.commit_sentence_judge import PROMPT as PROMPT_E20, first_claim_idx, parse as parse_e20
from srww.claim_parse import stype

O = Path(f"{_vtp.OUTPUTS}")
E28 = O / "e28"
E29 = O / "e29"
SYN = f"{_vtp.DATASETS}/spatial_probing"
IMG = {"cues": E28 / "cue_items" / "images", "twins": O.parent / "exps" / "items" / "occlusion" / "images"}
J2 = ("claude-opus-5-5", "gpt-6-astra")
COMP = ["qwen3vl_8b", "molmo7b"]

PROMPT_ONTOP = """Here is an image with two shapes and part of a model's reasoning about it.

Previous sentence (context only): "{c}"
Target sentence: "{s}"

Spatial wording can describe two different relations between the shapes:
- DEPTH (layering): which shape is in front of or behind the other, covers, hides, overlaps, or is partly hidden by the other. In the image, the shape whose outline is complete where they overlap is in front; the shape that is cut off is behind.
- VERTICAL (2D position): which shape is higher or lower, above or below, in the picture.
Phrases such as "on top of" or "over" can mean either.

For the TARGET sentence only:
1. READING: which relation between the two shapes does it assert? DEPTH, VERTICAL, BOTH (one of each), AMBIGUOUS (a phrase such as "on top of" that can be read either way and nothing in the two sentences settles it), or NONE (no checkable relation between the two shapes).
2. DEPTH_TRUE: read as a depth/layering claim, is it true in the image? TRUE, FALSE, or NA (it has no depth reading).
3. VERTICAL_TRUE: read as a vertical-position claim, is it true in the image? TRUE, FALSE, or NA (it has no vertical reading).

Reply with exactly three lines and nothing else:
READING: <DEPTH|VERTICAL|BOTH|AMBIGUOUS|NONE>
DEPTH_TRUE: <TRUE|FALSE|NA>
VERTICAL_TRUE: <TRUE|FALSE|NA>"""

PROMPT_DESC = """Here is an image with two shapes and a text a model wrote about the image.

Text: "{d}"

Consider only what the text claims about the DEPTH (layering) relation between the two shapes: which shape is in front of or behind the other, covers, hides, overlaps, or is partly hidden by the other. In the image, the shape whose outline is complete where they overlap is in front; the shape that is cut off is behind. Ignore claims about vertical position (higher/lower, above/below in the picture). Count "on top of" or "over" as a depth claim only if the text uses it for layering or overlap.

Reply with exactly three lines and nothing else:
FRONT_CLAIMED: <the shape the text says is in front, as color and shape (e.g. red circle); NONE if it makes no depth claim; MIXED if its depth claims conflict>
DEPTH_CLAIM: <CORRECT|INCORRECT|MIXED|NOCLAIM>
ON_TOP_ONLY: <YES if its only depth wording is "on top of" or "over", otherwise NO>"""


def syn_img(iid: str, _cache: dict = {}) -> str:
    if "syn" not in _cache:
        _cache["syn"] = json.load(open(f"{SYN}/depth_ordering.json"))
    return _cache["syn"][int(iid.split("_")[-1])]["image_path"].replace(
        "<DATA_ROOT>", f"{_vtp.DATA_ROOT}")


def base_of(item_id: str) -> str:
    return item_id[:-7] if item_id.endswith("_behind") else item_id


def set_img(st: str, item_id: str) -> str:
    return syn_img(item_id) if st == "syn" else str(IMG[st] / f"{base_of(item_id)}.png")


# ============================================================================= samples

def sample_prej2() -> list[dict]:
    """Qwen3-VL-8B cue D+C-: Exp-28 commit sample (40) + 80 more (seed 290); all pre-j* relational sentences + j*."""
    T = B.rj(E28 / "qwen3vl_8b_cues_traj.jsonl")
    old = {r["item_id"] for r in B.rj(E28 / "judge_gpt-6-astra_commit_cues.jsonl") if r["model"] == "qwen3vl_8b" and r["group"] == "D+C-"}
    dcm = [r for r in T if r["group"] == "D+C-" and r["commit_idx"] is not None and r["sentences"]]
    rest = sorted((r for r in dcm if r["item_id"] not in old), key=lambda r: r["item_id"])
    new = random.Random(290).sample(rest, 80)
    out = []
    for r in [x for x in dcm if x["item_id"] in old] + new:
        s, js = r["sentences"], r["commit_idx"]
        for j in range(js + 1):
            if j == js or stype(s[j]) == "relation":
                out.append({"model": r["model"], "set": "cues", "item_id": r["item_id"], "image_id": r["image_id"],
                            "group": "jstar" if j == js else "pre_jstar", "idx": j, "n_sent": len(s), "commit_idx": js,
                            "in_e28_sample": r["item_id"] in old, "sentence": s[j], "context": s[j - 1] if j > 0 else "(none)",
                            "stype": stype(s[j]), "image": set_img("cues", r["item_id"])})
    return out


def sample_ontop() -> list[dict]:
    out = []
    for st in ("cues", "twins"):
        for r in B.rj(E28 / f"judge_gpt-6-astra_commit_{st}.jsonl"):
            out.append({k: v for k, v in r.items() if k not in ("judge", "label", "raw", "api_error")}
                       | {"src": f"e28_commit_{st}", "image": set_img(st, r["item_id"])})
    for r in B.rj(E28 / "judge_gpt-6-astra_prej_cues.jsonl"):
        out.append({k: v for k, v in r.items() if k not in ("judge", "label", "raw", "api_error")}
                   | {"src": "e28_prej_cues", "image": set_img("cues", r["item_id"])})
    for fn, grp in (("gpt-6-astra.jsonl", "D+C-"), ("gpt-6-astra_relcomp.jsonl", "D+C+rel")):
        for r in B.rj(O / "e20" / fn):
            if r["group"] != grp:
                continue
            out.append({k: v for k, v in r.items() if k not in ("judge", "label", "raw", "api_error", "retried")}
                       | {"set": "syn", "src": "e20_syn", "image": syn_img(r["item_id"])})
    return out


def sample_desc() -> list[dict]:
    rng = random.Random(291)
    out = []
    for m in COMP:
        D = [r for r in B.rj(E28 / f"{m}_cues_desc.jsonl") if r["condition"] == "desc"]
        R = {(r["item_id"], r["condition"]): r for r in B.rj(O / "e10" / f"{m}_cues.jsonl") if r["condition"] == "direct"}
        by = {}
        for r in D:
            d = B.pe(R[(r["item_id"], "direct")]) == r["gold"]
            s = B.pe(r) == r["gold"]
            by.setdefault(r["base_id"], {"text": r["text"], "meta": r["meta"], "image_id": r["image_id"], "cells": []})["cells"].append((d, s))
        err = sorted(u for u, v in by.items() if any(d and not s for d, s in v["cells"]))
        ok = sorted(u for u, v in by.items() if all(d and s for d, s in v["cells"]))
        for stratum, us in (("desc_err", rng.sample(err, min(50, len(err)))), ("desc_ok", rng.sample(ok, min(50, len(ok))))):
            for u in us:
                out.append({"model": m, "kind": "desc", "stratum": stratum, "item_id": u, "base_id": u,
                            "image_id": by[u]["image_id"], "text": by[u]["text"], "front_name": by[u]["meta"]["front_name"],
                            "back_name": by[u]["meta"]["back_name"], "n_err_strata": len(err), "n_ok_strata": len(ok),
                            "n_images": len(by), "image": str(IMG["cues"] / f"{u}.png")})
        T = {r["item_id"]: r for r in B.rj(E28 / f"{m}_cues_traj.jsonl")}
        for r in B.rj(E28 / "judge_gpt-6-astra_commit_cues.jsonl"):
            if r["model"] != m:
                continue
            t = T[r["item_id"]]
            out.append({"model": m, "kind": "cot", "stratum": r["group"], "item_id": r["item_id"], "base_id": t["base_id"],
                        "image_id": t["image_id"], "text": " ".join(t["sentences"]), "front_name": t["meta"]["front_name"],
                        "back_name": t["meta"]["back_name"], "commit_idx": t["commit_idx"], "image": set_img("cues", r["item_id"])})
    return out


# ============================================================================= judging

def parse_ontop(txt: str) -> dict:
    t = (txt or "").upper().replace("*", "")
    g = lambda k, opts: (re.search(rf"{k}\s*:\s*({'|'.join(opts)})\b", t) or [None, "UNPARSED"])[1]
    return {"reading": g("READING", ["DEPTH", "VERTICAL", "BOTH", "AMBIGUOUS", "NONE"]),
            "depth_true": g("DEPTH_TRUE", ["TRUE", "FALSE", "NA"]), "vertical_true": g("VERTICAL_TRUE", ["TRUE", "FALSE", "NA"])}


def parse_desc(txt: str) -> dict:
    t = (txt or "").replace("*", "")
    fc = re.search(r"FRONT_CLAIMED\s*:\s*([^\n]+)", t, re.I)
    dc = re.search(r"DEPTH_CLAIM\s*:\s*(CORRECT|INCORRECT|MIXED|NOCLAIM)\b", t, re.I)
    ot = re.search(r"ON_TOP_ONLY\s*:\s*(YES|NO)\b", t, re.I)
    return {"front_claimed": fc.group(1).strip().strip('."').lower() if fc else "UNPARSED",
            "depth_claim": dc.group(1).upper() if dc else "UNPARSED", "on_top_only": ot.group(1).upper() if ot else "UNPARSED"}


def judge(task: str, jname: str, items: list[dict]) -> list[dict]:
    from srww import api_client as api_client
    from PIL import Image
    lock, res = threading.Lock(), []

    def go(c):
        im = Image.open(c["image"]).convert("RGB")
        if task == "desc":
            txt, parser = PROMPT_DESC.format(d=c["text"]), parse_desc
        elif task == "ontop":
            txt, parser = PROMPT_ONTOP.format(c=c["context"], s=c["sentence"]), parse_ontop
        else:
            txt, parser = PROMPT_E20.format(c=c["context"], s=c["sentence"]), lambda x: {"label": parse_e20(x)}
        r = api_client.chat(jname, [{"role": "user", "content": [api_client.img_part(im), {"type": "text", "text": txt}]}],
                       max_tokens=2048, effort="high")
        with lock:
            res.append({k: v for k, v in c.items() if k != "image"} | {"judge": jname, **parser(r.get("text")),
                                                                     "raw": (r.get("text") or r.get("error", ""))[:200],
                                                                     "api_error": "error" in r})

    with ThreadPoolExecutor(6) as ex:
        list(ex.map(go, items))
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True, choices=["prej2", "ontop", "desc"])
    ap.add_argument("--judge")
    ap.add_argument("--dry", action="store_true")
    ap.add_argument("--limit", type=int)
    a = ap.parse_args()
    items = {"prej2": sample_prej2, "ontop": sample_ontop, "desc": sample_desc}[a.task]()
    key = lambda x: (x.get("src") or x.get("kind") or x.get("group"), x["model"], x.get("group") or x.get("stratum"))
    print(f"[e29j] {a.task}: {len(items)} items {dict(Counter(key(x) for x in items))}", flush=True)
    if a.task == "prej2":
        # reuse Exp-28 labels (same prompt / judge settings) for sentences already judged there
        old = {}
        for t in ("prej", "commit"):
            for r in B.rj(E28 / f"judge_{a.judge}_{t}_cues.jsonl") if a.judge else []:
                if r["model"] == "qwen3vl_8b" and r["group"] in ("pre_jstar", "D+C-"):
                    old[(r["item_id"], r["idx"])] = r
        todo = [x for x in items if (x["item_id"], x["idx"]) not in old]
        print(f"[e29j] prej2: {len(items) - len(todo)} labels reused from Exp-28, {len(todo)} to judge", flush=True)
    else:
        todo = items
    if a.limit:
        todo = todo[:a.limit]
    if a.dry:
        return
    E29.mkdir(exist_ok=True)
    res = judge(a.task, a.judge, todo)
    if a.task == "prej2":
        for x in items:
            k = (x["item_id"], x["idx"])
            if k in old:
                res.append({kk: v for kk, v in x.items() if kk != "image"} | {"judge": a.judge, "label": old[k]["label"],
                                                                              "raw": old[k]["raw"], "api_error": old[k]["api_error"],
                                                                              "reused_e28": True})
    sfx = f"_limit{a.limit}" if a.limit else ""
    p = E29 / f"judge_{a.judge}_{a.task}{sfx}.jsonl"
    with open(p, "w") as f:
        for r in res:
            f.write(json.dumps(r) + "\n")
    bad = sum(r["api_error"] for r in res)
    unp = sum(("UNPARSED" in (r.get("label"), r.get("reading"), r.get("depth_claim"), r.get("depth_true"))) for r in res)
    print(f"[e29j] {p.name}: {len(res)}; unparsed {unp}; api errors {bad}", flush=True)


if __name__ == "__main__":
    main()
