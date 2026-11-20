"""Exp-51: last written answer-relevant claim in CoT texts on VSI-Bench / MindCube-Tiny (CPU + text-only API).

build    outputs/e51/<model>_<bench>.jsonl (cot arm) -> per-text records under two protocols (pre-registered):
           rm  conclusion removed: Exp-35 truncate rule generalized to options A-F (answer sentences, everything after
               the last answer sentence, Conclusion header + rest, trailing concluding / option-restating sentences)
           kp  conclusion kept: only answer sentences (option letter / "the answer ...") removed
         -> outputs/e51/judge_items.jsonl
judge    Claude Opus 5.5 / GPT-6-Astra (effort high, api_client client, key from the environment, never printed): which single
         option the LAST answer-relevant claim identifies, or NONE -> outputs/e51/judge_<proto>_<judge>.jsonl (append,
         resumable; texts identical across models / protocols are judged once per judge via a text hash)
Usage: python e51_judge.py build | judge --judge claude-opus-5-5 --proto rm [--key alt] [--shard 0/2] --threads 16
"""
from __future__ import annotations
import vtax_paths as _vtp  # noqa: E402

import argparse
import hashlib
import json
import os
import re
import threading
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from build_bundle import pe, rj
from e2f_claim_edit import split_sents
from e35_natural_two_channel import CONC, HEADER, HEADER_INLINE

O = Path(f"{_vtp.OUTPUTS}")
E51 = O / "e51"
M7 = ["qwen3b", "qwen7b", "internvl3_8b", "phi35v", "qwen3vl_8b", "internvl35_8b", "molmo7b"]
BENCHES = ["vsi", "mc"]
JUDGES = ["claude-opus-5-5", "gpt-6-astra"]
PREFIX = "Let's think step by step."

# Exp-35 ANS rule with option letters A-F instead of A/B
ANS = re.compile(r"\b(?:the|final|correct|my|right|best)\s+(?:answer|option|choice)\b|\banswer\s*(?:is|would be|should be|:)|"
                 r"^\W*answer\b|\b(?:option|choice)\s*\(?\s*[A-F]\b|(?<!point )(?<!image )(?<!frame )(?<!and )\(\s*[A-F]\s*\)|"
                 r"\*\*\s*\(?[A-F]\)?\s*\*\*|^\W*\(?[A-F]\)?\W*$", re.I)


def _norm(s: str) -> str:
    s = re.sub(r"[*_`#>]", "", s).strip().lower()
    s = re.sub(r"^(?:[-\u2022]\s*|\d+\.\s*)", "", s)
    return re.sub(r"\s+", " ", s).strip(" .!")


def restates(s: str, opts: dict[str, str]) -> bool:
    """Sentence is only an option text (optionally 'so/thus/therefore, <option>')."""
    t = re.sub(r"^(?:therefore|thus|so|hence|clearly|overall)\b,?\s*", "", _norm(s))
    return any(t == _norm(v) for v in opts.values())


def truncate(sents: list[str], opts: dict[str, str]) -> list[str]:
    s = [x for x in sents if x.strip()]
    hits = [j for j, x in enumerate(s) if ANS.search(x)]
    if hits:
        s = [x for x in s[: hits[-1]] if not ANS.search(x)]
    hdr = [j for j, x in enumerate(s) if HEADER.match(x) or HEADER_INLINE.match(x)]
    if hdr:
        s = s[: hdr[-1]]
    while s and (CONC.match(s[-1]) or restates(s[-1], opts) or not re.search(r"[A-Za-z]{2,}", s[-1])
                 or s[-1].rstrip().endswith(":")):
        s = s[:-1]
    return s


def kept(sents: list[str]) -> list[str]:
    return [x for x in sents if x.strip() and not ANS.search(x)]


PROMPT = """You will read a short text written in response to a multiple-choice spatial question about {media}. You do NOT see the {media_short}; judge only what the text claims.

Question:
{q}

Text ({removed}):
\"\"\"
{r}
\"\"\"

Task: find the LAST statement in the text that asserts a spatial fact which, taken at face value, picks out exactly one of the options as correct. Examples: for a direction question "the sofa is on my left" picks the option "left"; for a distance question "the chair is the closest to the TV" (or distance estimates that make one option the closest) picks "chair"; for an order question "the cup appears first, then the door, then the heater, then the ceiling light" picks the option with that order; for a yes/no question a statement that directly asserts or denies the asked relation picks "Yes" or "No"; for a "what is to my right" question "the door would be on my right" picks "door". Hedged claims ("seems", "likely") count. Do NOT count questions, hypotheticals that the text does not adopt, general rules, restatements of the question, or statements compatible with more than one option.

Reply in exactly this format:
QUOTE: <the statement, at most 30 words, or none>
OPTION: <one option letter, or NONE>"""

MEDIA = {"vsi": ("a video of an indoor scene (shown as frames)", "video"),
         "mc": ("a few images of a scene taken from different viewpoints", "images")}
REMOVED = {"rm": "any final answer, option letters and concluding sentences have been removed",
           "kp": "any explicitly stated final answer and option letters have been removed"}


def parse_option(txt: str | None, letters: list[str]) -> str:
    m = re.findall(r"OPTION\W*\s*\(?\s*([A-F]|NONE)\b", (txt or "").upper())
    if not m:
        return "UNPARSED"
    return m[-1] if m[-1] == "NONE" or m[-1] in letters else "UNPARSED"


def hk(proto: str, bench: str, q: str, body: str) -> str:
    return hashlib.sha1(f"{proto}|{bench}|{q}|{body}".encode()).hexdigest()[:16]


def build() -> None:
    items = {b: {it["key"]: it for it in rj(E51 / f"items_{b}.jsonl")} for b in BENCHES}
    rows = []
    for m in M7:
        for b in BENCHES:
            recs = rj(E51 / f"{m}_{b}.jsonl")
            if not recs:
                print("missing", m, b)
                continue
            by: dict = {}
            for r in recs:
                by.setdefault(r["item_id"], {})[r["condition"]] = r
            for k, d in by.items():
                it = items[b][k]
                c = d["cot"]
                t = (c.get("text") or "").strip()
                t = t[len(PREFIX):].strip() if t.startswith(PREFIX) else t
                sents = split_sents(t)
                rm, kp = truncate(sents, it["options"]), kept(sents)
                q = it["question"].replace("These are frames of a video.\n", "")
                rows.append({"model": m, "bench": b, "item": k, "qtype": it["qtype"], "cl": it["cluster"], "gold": it["gold"],
                             "letters": it["choices"], "d_ans": pe(d["direct"]), "c_ans": pe(c), "f_ans": pe(d["filler_R+turn"]),
                             "question": q, "rm": " ".join(rm), "kp": " ".join(kp),
                             "hk_rm": hk("rm", b, q, " ".join(rm)) if rm else None,
                             "hk_kp": hk("kp", b, q, " ".join(kp)) if kp else None,
                             "trunc": c.get("trunc"), "gen_len": c.get("gen_len")})
    with open(E51 / "judge_items.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    for g, n in sorted(Counter((r["model"], r["bench"]) for r in rows).items()):
        rs = [r for r in rows if (r["model"], r["bench"]) == g]
        print(g, n, "empty rm %.1f%% kp %.1f%%" % (100 * sum(not r["rm"] for r in rs) / n, 100 * sum(not r["kp"] for r in rs) / n))
    print("unique texts rm", len({r["hk_rm"] for r in rows if r["hk_rm"]}), "kp", len({r["hk_kp"] for r in rows if r["hk_kp"]}))


def judge(jname: str, proto: str, threads: int, limit: int | None, shard: str | None, key: str) -> None:
    import api_client
    if key == "alt":
        os.environ["OPENAI_COMPAT_API_KEY"] = os.environ["OPENAI_COMPAT_API_KEY_ALT"]
    rows = [r for r in rj(E51 / "judge_items.jsonl") if r[f"hk_{proto}"]]
    p = E51 / f"judge_{proto}_{jname}.jsonl"
    done = {r["hk"] for r in rj(p) if not r.get("api_error") and r["option"] != "UNPARSED"}
    todo, seen = [], set()
    for r in rows:
        h = r[f"hk_{proto}"]
        if h in done or h in seen:
            continue
        if shard:
            si, sn = map(int, shard.split("/"))
            if int(h, 16) % sn != si:
                continue
        seen.add(h)
        todo.append(r)
    todo = todo[:limit] if limit else todo
    print(f"[e51j] {jname} {proto}: {len(rows)} texts, done {len(done)}, to judge {len(todo)}", flush=True)
    lock = threading.Lock()
    cnt: Counter = Counter()
    f = open(p, "a")

    def go(r):
        med, ms = MEDIA[r["bench"]]
        txt = PROMPT.format(media=med, media_short=ms, q=r["question"], removed=REMOVED[proto], r=r[proto])
        out = api_client.chat(jname, [{"role": "user", "content": txt}], max_tokens=4096, effort="high")
        if not (out.get("text") or "").strip() and "error" not in out:  # Exp-35: bracketed delimiter for empty completions
            txt = PROMPT.format(media=med, media_short=ms, q=r["question"], removed=REMOVED[proto],
                                r=f"[text begins] {r[proto]} [text ends]")
            out = api_client.chat(jname, [{"role": "user", "content": txt}], max_tokens=4096, effort="high")
        lab = parse_option(out.get("text"), r["letters"])
        qm = re.search(r"QUOTE:\s*(.*)", out.get("text") or "")
        rec = {"hk": r[f"hk_{proto}"], "judge": jname, "option": lab, "quote": qm.group(1)[:200] if qm else "",
               "raw": (out.get("text") or out.get("error", ""))[:300], "api_error": "error" in out,
               "finish": out.get("finish_reason")}
        with lock:
            f.write(json.dumps(rec) + "\n")
            f.flush()
            cnt["err" if rec["api_error"] else ("letter" if lab not in ("NONE", "UNPARSED") else lab)] += 1
            if sum(cnt.values()) % 500 == 0:
                print(f"[e51j] {jname} {proto}: {sum(cnt.values())} {dict(cnt)}", flush=True)

    with ThreadPoolExecutor(threads) as ex:
        list(ex.map(go, todo))
    f.close()
    print(f"[e51j] {jname} {proto} finished: {dict(cnt)}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["build", "judge"])
    ap.add_argument("--judge", default=JUDGES[0])
    ap.add_argument("--proto", default="rm", choices=["rm", "kp"])
    ap.add_argument("--threads", type=int, default=16)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--shard", default=None)
    ap.add_argument("--key", default="main", choices=["main", "alt"])
    a = ap.parse_args()
    if a.cmd == "build":
        build()
    else:
        judge(a.judge, a.proto, a.threads, a.limit, a.shard, a.key)


if __name__ == "__main__":
    main()
