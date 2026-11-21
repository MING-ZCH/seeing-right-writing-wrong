"""Exp-45: two-channel analysis under ONE conclusion-removal protocol (synthetic open, natural, closed).

Protocol (all points): the CoT text is split into sentences (e2f split_sents) and the conclusion is removed with the
Exp-35 rule exactly (e35_natural_two_channel.truncate: answer sentences, text after the last answer sentence, a
'Conclusion' header and what follows, trailing concluding sentences 'therefore / so / based on ...' and trailing
restatements of an option). The LAST remaining depth / occlusion claim is extracted by two text-only judges
(Claude Opus 5.5, GPT-6-Astra; effort high); a relation counts only when both name the same object (Exp-35 / Exp-38
consensus). Shapes (synthetic open, closed): the Exp-38 prompt ('a short text that describes an image', which avoids
the Claude content filter), with the removal sentence updated. Natural images: Exp-35 items and labels reused (same
rule, same judges, same consensus).

build    item records -> outputs/e45/items.jsonl (open synthetic 7 models x cues / twins / Syn. Depth 431, Qwen3-VL-32B and
         Qwen3-VL-8B-Thinking cue set, Thinking twins; closed e9h / e9x / e9c_cues)
judge    -> outputs/e45/judge_<judge>.jsonl, keyed by a hash of (object names, truncated text), resumable.
         Texts whose truncated body has no depth / occlusion word are labelled NONE without a call (prefilter);
         validated by judging a random sample of them (--prefilter-check).
         --pre: same judges on the answer-line-stripped text (Exp-38 strip_answer, conclusion kept) for a random
         validation subsample, to measure the judge-level coverage drop on open models.
analyze  -> bundle/e45_uniform_two_channel.json, bundle/rigor_e45.txt

Usage: source configs/env.sh
       python -m experiments.two_channel.uniform_two_channel build
       python -m experiments.two_channel.uniform_two_channel judge --judge gpt-6-astra --threads 24 [--key alt]
       python -m experiments.two_channel.uniform_two_channel analyze
"""
from __future__ import annotations
from srww import paths as _vtp  # noqa: E402

import argparse
import hashlib
import json
import os
import random
import re
import threading
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from scipy.stats import pearsonr, spearmanr

O = Path(f"{_vtp.OUTPUTS}")
E45 = O / "e45"
B = Path("outputs/bundle")
M7 = ["qwen3b", "qwen7b", "internvl3_8b", "phi35v", "molmo7b", "qwen3vl_8b", "internvl35_8b"]
EXTRA = ["qwen3vl_32b", "qwen3vl_8b_think"]
CLOSED = ["claude-opus-5-5", "gpt-6-astra", "gemini-4-argon", "qwen3.8-max-0902", "hy4", "deepseek-v4.1-flash"]
JUDGES = ("claude-opus-5-5", "gpt-6-astra")
NAMES = {"qwen3b": "Qwen2.5-VL-3B", "qwen7b": "Qwen2.5-VL-7B", "internvl3_8b": "InternVL3-8B", "phi35v": "Phi-3.5-V",
         "molmo7b": "Molmo-7B-D", "qwen3vl_8b": "Qwen3-VL-8B", "internvl35_8b": "InternVL3.5-8B",
         "qwen3vl_8b_think": "Qwen3-VL-8B-Think.", "qwen3vl_32b": "Qwen3-VL-32B",
         "claude-opus-5-5": "Claude Opus 5.5", "gpt-6-astra": "GPT-6-Astra", "gemini-4-argon": "Gemini-4-Argon",
         "qwen3.8-max-0902": "Qwen3.8-Max", "hy4": "HY4", "deepseek-v4.1-flash": "DeepSeek-V4.1-Flash"}
DSN = {"cues": "Cue set", "twins": "Twins", "syn431": "Syn.\\ Depth 431", "cvb3d": "CVB 3D Depth", "blink": "BLINK Depth",
       "e9h": "Hard occl.", "e9x": "Twins", "e9c_cues": "Cue set"}

PROMPT = """Below is a short text that describes an image showing two shapes: "{x}" and "{y}". The text's final answer and concluding sentences have been removed.

Find the LAST statement in the text that says which of the two shapes is in front of the other: on top of it, covering, occluding, overlapping over or hiding part of it, closer to the viewer, or equivalently that the other shape is behind, underneath, covered, hidden or farther away. Rules:
- Resolve pronouns and short descriptions ("it", "the circle", "the left shape") to the two shapes.
- A negated statement counts for the other shape (e.g. "the {x} is not in front of the {y}" means the {y} is in front).
- A statement that only says the shapes overlap, touch or intersect, without saying which one is on top, does NOT count.
- Questions, hypotheses ("if ..."), restatements of the task, and alternatives that the text considers and rejects do NOT count. Statements about left/right/above/below or size alone do NOT count.
- A hedged statement ("appears to be", "probably") still counts.
- A sentence that only gives the answer choice (yes / no / an option letter) without naming which shape is in front does NOT count.
- If the last such statement is self-contradictory or you cannot tell which shape it puts in front, reply UNCLEAR.

Text:
\"\"\"
{r}
\"\"\"

Reply with exactly one line: FRONT: <shape name, exactly "{x}" or "{y}"> or FRONT: NONE or FRONT: UNCLEAR"""
PROMPT_PRE = PROMPT.replace("The text's final answer and concluding sentences have been removed.",
                            "The text's final answer line has been removed.")  # = Exp-38 prompt
assert PROMPT_PRE != PROMPT
RELKW = re.compile(r"front|behind|cover|on top|top of|atop|occlu|hidden|hide|hiding|closer|nearer|farther|further|"
                   r"underneath|beneath|obscur|block|overlap|over\b|under\b|foreground|background|layer|visible|partial|"
                   r"cut off|obstruct|depth|distance|near|far\b|ahead|stack|eclips|conceal|mask|tuck|rest", re.I)


MIDCONC = re.compile(r"(,|;|—|-)\s*(so|therefore|thus|hence|which means|meaning|indicating that|confirming)\b|"
                     r"\b(this|that|which) (means|indicates|confirms|shows)\b", re.I)


def strict_body(r: dict) -> str:
    """Sensitivity variant: after the Exp-35 rule, also drop trailing sentences that contain a mid-sentence conclusion
    ('..., so the circle is in front', 'This means ...'), then re-apply the Exp-35 tail rule."""
    from srww.rcut import truncate
    s = list(r["trunc"])
    while s and MIDCONC.search(s[-1]):
        s = s[:-1]
        s, _ = truncate(s, [r["front"], r["back"]])
    return " ".join(s) if r["group"] == "open" else "\n".join(s)


def rj(p: Path) -> list[dict]:
    return [json.loads(l) for l in open(p)] if p.exists() else []


def hkey(x: str, y: str, body: str, tag: str = "") -> str:
    return hashlib.sha1(f"{tag}\x00{x}\x00{y}\x00{body}".encode()).hexdigest()[:20]


# ============================================================================= build

def _v3_last(sents: list[str], front: str, back: str) -> str | None:
    from experiments.two_channel.two_channel_fit import claims
    return claims(sents, front, back)[1]


def build() -> None:
    from experiments.two_channel import two_channel_fit as R
    from srww.records import pe
    from srww.claim_parse import split_sents
    from srww.rcut import truncate
    from judges.closed_relation_judge import dedup, strip_answer
    E45.mkdir(exist_ok=True)
    rows = []
    orig = R.rec

    def cap(model, ds, iid, cluster, gold, d_ans, c_ans, sents, front, back, x, phrasing, extra=None):
        r = orig(model, ds, iid, cluster, gold, d_ans, c_ans, sents, front, back, x, phrasing, extra)
        t, info = truncate(sents, [front, back])
        return r | {"front": front, "back": back, "x": x, "y": back if x == front else front,
                    "sents": sents, "trunc": t, "trunc_info": info,
                    "v3_full": _v3_last(sents, front, back), "v3_trunc": _v3_last(t, front, back)}

    R.rec = cap
    try:
        open_items = R.load_items()  # 7 models x cues / twins / syn (500) + Thinking twins
        open_items += R.items_cues("qwen3vl_32b")
        from experiments.two_channel.two_channel_fit import load_arms
        A = load_arms("qwen3vl_8b_think")
        for k in sorted(set(A["direct"]) & set(A["cot"])):
            c = A["cot"][k]
            mt = c["meta"]
            open_items.append(cap("qwen3vl_8b_think", "cues", k, c["image_id"], c["gold"], pe(A["direct"][k]), pe(c),
                                  R.body(c.get("text") or ""), mt["front_name"], mt["back_name"], mt["x_name"], c["phrasing"]))
    finally:
        R.rec = orig
    for r in open_items:
        if r["ds"] == "syn":
            if not r.get("overlap"):
                continue
            r["ds"] = "syn431"
        r["group"] = "open"
        r["body_pre"] = " ".join(r["sents"])
        rows.append(r)
    # closed models (Exp-38 sources; Direct + CoT valid)
    for st, models in {"e9h": CLOSED, "e9x": CLOSED, "e9c_cues": ["deepseek-v4.1-flash"]}.items():
        for m in models:
            G = defaultdict(dict)
            for (cond, iid, ph), r in dedup(rj(O / st / f"{m}.jsonl")).items():
                if r["task"].startswith("occ"):
                    G[iid][cond] = r
            for iid, d in G.items():
                if "cot" not in d or "direct" not in d:
                    continue
                c, dr = d["cot"], d["direct"]
                mt = c["meta"]
                x, front, back = mt["x_name"].lower(), mt["front_name"].lower(), mt["back_name"].lower()
                text = c.get("text") or ""
                sents = split_sents(text)
                t, info = truncate(sents, [front, back])
                pre, _ = strip_answer(text)
                nr = d.get("direct_nr")
                rows.append({"group": "closed", "model": m, "ds": st, "item": iid, "cl": c["image_id"], "gold": c["gold"],
                             "phrasing": c["phrasing"], "d_ok": dr["pred"] == c["gold"], "c_ok": c["pred"] == c["gold"],
                             "c_ans": c["pred"], "nr_ok": None if nr is None else nr["pred"] == c["gold"],
                             "front": front, "back": back, "x": x, "y": back if x == front else front,
                             "sents": sents, "trunc": t, "trunc_info": info, "body_pre": pre,
                             "key38": f"{st}|{m}|{iid}",
                             "v3_full": _v3_last(split_sents(pre), front, back), "v3_trunc": _v3_last(t, front, back)})
    for r in rows:
        r["body"] = " ".join(r["trunc"]) if r["group"] == "open" else _join_closed(r["trunc"])
        r["hkey"] = hkey(r["x"], r["y"], r["body"])
        r["has_kw"] = bool(RELKW.search(r["body"]))
    with open(E45 / "items.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    c = Counter((r["group"], r["model"], r["ds"]) for r in rows)
    for k, n in sorted(c.items()):
        rs = [r for r in rows if (r["group"], r["model"], r["ds"]) == k]
        print(f"{k[0]:6s} {k[1]:20s} {k[2]:8s} n={n:5d} empty {sum(not r['body'] for r in rs):4d} no-kw {sum(r['body'] != '' and not r['has_kw'] for r in rs):4d} "
              f"v3 full {np.mean([r['v3_full'] is not None for r in rs]) * 100:5.1f} trunc {np.mean([r['v3_trunc'] is not None for r in rs]) * 100:5.1f}")
    need = {r["hkey"] for r in rows if r["body"] and r["has_kw"]}
    print(f"rows {len(rows)}; unique texts to judge {len(need)}; prefiltered (no keyword) {len({r['hkey'] for r in rows if r['body'] and not r['has_kw']})}; "
          f"empty {sum(not r['body'] for r in rows)}")


def _join_closed(sents: list[str]) -> str:
    return "\n".join(sents)


# ============================================================================= judge

def parse_label(txt: str, x: str, y: str) -> str | None:
    m = list(re.finditer(r"FRONT:\s*\**\s*\"?([A-Za-z ]+)", txt or "", re.I))
    if not m:
        return None
    lab = m[-1].group(1).strip().lower()
    for nm in (x, y):
        if lab.startswith(nm):
            return nm
    if lab.startswith("none"):
        return "none"
    if lab.startswith("unclear"):
        return "unclear"
    return None


def judge(jname: str, threads: int, key: str, mode: str, limit: int | None, n_sample: int, shard: str = "") -> None:
    from srww import api_client as api_client
    if key == "alt":
        os.environ["OPENAI_COMPAT_API_KEY"] = os.environ["OPENAI_COMPAT_API_KEY_ALT"]
    rows = rj(E45 / "items.jsonl")
    fn = E45 / (f"judge_{jname}.jsonl" if mode == "main" else f"judge_{jname}_{mode}.jsonl")
    done = {r["hkey"] for r in rj(fn) if r.get("label") is not None}
    todo, seen = [], set()
    if mode == "main":
        src = [r for r in rows if r["body"] and r["has_kw"]]
        pr = lambda r: (r["group"] != "closed", r["ds"] == "e9x", r["model"], r["item"])
        for r in sorted(src, key=pr):
            if r["hkey"] not in done and r["hkey"] not in seen:
                seen.add(r["hkey"]); todo.append((r["hkey"], r["x"], r["y"], r["body"], PROMPT))
    elif mode == "prefilter":  # random sample of keyword-free texts, judged normally
        src = sorted({r["hkey"]: r for r in rows if r["body"] and not r["has_kw"]}.values(), key=lambda r: r["hkey"])
        for r in random.Random(45).sample(src, min(n_sample, len(src))):
            if r["hkey"] not in done:
                todo.append((r["hkey"], r["x"], r["y"], r["body"], PROMPT))
    elif mode == "strict":  # sensitivity variant: only texts that differ from the primary truncation
        pr = lambda r: (r["group"] != "closed", r["ds"] == "e9x", r["model"], r["item"])
        for r in sorted(rows, key=pr):
            b = strict_body(r)
            if not b or not RELKW.search(b):
                continue
            hk = hkey(r["x"], r["y"], b)
            if hk != r["hkey"] and hk not in done and hk not in seen:
                seen.add(hk); todo.append((hk, r["x"], r["y"], b, PROMPT))
    elif mode == "pre":  # validation subsample: conclusion kept (answer line stripped), Exp-38 prompt
        for r in val_sample(rows, n_sample):
            hk = hkey(r["x"], r["y"], r["body_pre"], "pre")
            if hk not in done and r["body_pre"].strip():
                todo.append((hk, r["x"], r["y"], r["body_pre"], PROMPT_PRE))
    if shard:
        i, n = map(int, shard.split("/"))
        todo = [t for t in todo if int(t[0], 16) % n == i]
    if limit:
        todo = todo[:limit]
    print(f"[e45] {jname} {mode}: {len(todo)} to judge ({len(done)} done)", flush=True)
    lock = threading.Lock()
    t0, cnt = time.time(), Counter()
    f = open(fn, "a")

    def go(t):
        hk, x, y, body, prompt = t
        out = api_client.chat(jname, [{"role": "user", "content": prompt.format(x=x, y=y, r=body)}], max_tokens=2048, effort="high")
        raw = (out.get("text") or "").strip()
        err = out.get("error", "")[:200]
        lab = parse_label(raw, x, y) if not err else None
        rec = {"hkey": hk, "judge": jname, "label": lab, "raw": raw[-200:], "error": err}
        with lock:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            f.flush()
            cnt["err" if err else ("unlab" if lab is None else "ok")] += 1
            s = sum(cnt.values())
            if s % 250 == 0:
                print(f"[e45] {jname} {mode}: {s}/{len(todo)} {dict(cnt)} {time.time() - t0:.0f}s ({60 * s / (time.time() - t0):.0f}/min)", flush=True)

    with ThreadPoolExecutor(threads) as ex:
        list(ex.map(go, todo))
    f.close()
    print(f"[e45] {jname} {mode}: finished {dict(cnt)} {time.time() - t0:.0f}s", flush=True)


def val_sample(rows: list[dict], n: int) -> list[dict]:
    """Fixed random validation subsample: n open-model synthetic texts and n closed texts (seed 45)."""
    out = []
    for g in ("open", "closed"):
        src = sorted([r for r in rows if r["group"] == g], key=lambda r: (r["model"], r["ds"], str(r["item"])))
        out += random.Random(45).sample(src, min(n, len(src)))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("phase", choices=["build", "judge", "analyze"])
    ap.add_argument("--judge", choices=JUDGES)
    ap.add_argument("--threads", type=int, default=16)
    ap.add_argument("--key", default="main", choices=["main", "alt"])
    ap.add_argument("--mode", default="main", choices=["main", "prefilter", "pre", "strict"])
    ap.add_argument("--limit", type=int)
    ap.add_argument("--n-sample", type=int, default=150)
    ap.add_argument("--shard", default="", help="i/n: judge only texts with int(hkey) %% n == i (parallel processes)")
    a = ap.parse_args()
    if a.phase == "build":
        build()
    elif a.phase == "judge":
        judge(a.judge, a.threads, a.key, a.mode, a.limit, a.n_sample, a.shard)
    else:
        from experiments.two_channel.uniform_analysis import analyze
        analyze()


if __name__ == "__main__":
    main()
