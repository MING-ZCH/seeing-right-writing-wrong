"""Exp-50 judge phase: last written depth / occlusion relation in closed-model CoT under real / gray / twin images.

Two protocols per text (see e50_closed_image_blind.py): 'kept' (Exp-38: answer line removed, conclusion kept) and
'removed' (Exp-35 rule via e35_natural_two_channel.truncate, Exp-45 prompt). Cached labels are taken from Exp-38
(key set|model|item_id, same reply text) and Exp-45 (same hkey) before any call.
"""
from __future__ import annotations

import json
import os
import re
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor

from e50_closed_image_blind import CLOSED, E50, O, load_gen, rj

E38 = O / "e38"
E45 = O / "e45"


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", s or "").strip()


def text_records() -> list[dict]:
    """One record per generated CoT reply with both judged bodies and their cache keys."""
    from e2f_claim_edit import split_sents
    from e35_natural_two_channel import truncate
    from e38_closed_judge import strip_answer
    from e45_uniform_two_channel import RELKW, hkey
    S = json.load(open(E50 / "sample.json"))
    P = {p["uid"] + "|" + p["set"]: p for v in S.values() for p in v}
    out = []
    for r in load_gen():
        if "error" in r:
            continue
        p = P[r["uid"] + "|" + r["set"]]
        mt = p["meta"]
        x, front, back = mt["x_name"].lower(), mt["front_name"].lower(), mt["back_name"].lower()
        y = back if x == front else front
        text = r.get("text") or ""
        kept, _ = strip_answer(text)
        t, info = truncate(split_sents(text), [front, back])
        removed = "\n".join(t)
        out.append({"model": r["model"], "set": r["set"], "cond": r["cond"], "uid": r["uid"], "source": r["source"],
                    "src_item_id": r.get("src_item_id"), "x": x, "y": y, "front": front, "back": back,
                    "kept": kept, "removed": removed, "same_body": _norm(kept) == _norm(removed),
                    "kw_kept": bool(RELKW.search(kept)), "kw_removed": bool(RELKW.search(removed)),
                    "hk_kept": hkey(x, y, kept, "pre"), "hk_removed": hkey(x, y, removed)})
    return out


def caches(judge: str) -> tuple[dict, dict]:
    """(labels by hkey from Exp-45 + Exp-50 files, Exp-38 labels by set|model|item_id with body length)."""
    lab = {}
    for fn in (E45 / f"judge_{judge}.jsonl", E45 / f"judge_{judge}_pre.jsonl", E50 / f"judge_{judge}.jsonl"):
        for r in rj(fn):
            if r.get("label") is not None:
                lab[r["hkey"]] = r["label"]
    e38 = {r["key"]: r for r in rj(E38 / f"judge_{judge}.jsonl") if r.get("label") is not None}
    return lab, e38


def label_of(rec: dict, proto: str, lab: dict, e38: dict) -> tuple[str | None, str]:
    """Label for one text under one protocol, and its source (prefilter / e38 / e45-e50 cache / same-body)."""
    body, kw, hk = rec[proto], rec[f"kw_{proto}"], rec[f"hk_{proto}"]
    if not body.strip() or not kw:
        return "none", "prefilter"
    if proto == "kept" and rec["source"] == "e9h":
        k = f"e9h|{rec['model']}|{rec['src_item_id']}"
        if k in e38 and e38[k]["body_chars"] == len(body):
            return e38[k]["label"], "e38"
    if hk in lab:
        return lab[hk], "cache"
    if proto == "removed" and rec["same_body"]:
        lk, src = label_of(rec, "kept", lab, e38)
        return lk, ("same_body:" + src) if lk is not None else ""
    return None, ""


PRESENCE = """Below is a short text that describes an image and answers a question about two shapes: "{x}" and "{y}".

Classify what the text says about the image:
- VISIBLE_RELATION: the text describes both shapes as present in the image and states which one is in front of / on top of / covering the other (or which one is behind).
- NOT_VISIBLE: the text says that one or both shapes are absent or not visible, or that the image is blank, uniform or empty (whatever it then answers).
- OTHER: anything else (for example both shapes described as present, but no statement about which one is in front).

Text:
\"\"\"
{r}
\"\"\"

Reply with exactly one line: CLASS: VISIBLE_RELATION or CLASS: NOT_VISIBLE or CLASS: OTHER"""


def presence_items() -> list[dict]:
    """All gray-image texts + a fixed random 25 real-image texts per model x set (seed 50), full reply text."""
    import random
    from e45_uniform_two_channel import hkey
    G = load_gen()
    S = json.load(open(E50 / "sample.json"))
    P = {p["uid"] + "|" + p["set"]: p for v in S.values() for p in v}
    out, real = [], {}
    for r in G:
        if "error" in r or not (r.get("text") or "").strip():
            continue
        p = P[r["uid"] + "|" + r["set"]]
        mt = p["meta"]
        x, front, back = mt["x_name"].lower(), mt["front_name"].lower(), mt["back_name"].lower()
        it = {"model": r["model"], "set": r["set"], "cond": r["cond"], "uid": r["uid"], "x": x,
              "y": back if x == front else front, "text": r["text"]}
        it["hk"] = hkey(it["x"], it["y"], it["text"], "presence")
        if r["cond"] == "gray":
            out.append(it)
        elif r["cond"] == "real":
            real.setdefault((r["model"], r["set"]), []).append(it)
    for k in sorted(real):
        v = sorted(real[k], key=lambda t: t["uid"])
        out += random.Random(50).sample(v, min(25, len(v)))
    return out


def parse_presence(txt: str) -> str | None:
    m = list(re.finditer(r"CLASS:\s*\**\s*(VISIBLE_RELATION|NOT_VISIBLE|OTHER)", txt or "", re.I))
    return m[-1].group(1).upper() if m else None


def presence(jname: str, threads: int, key: str) -> None:
    import api_client
    if key == "alt":
        os.environ["OPENAI_COMPAT_API_KEY"] = os.environ["OPENAI_COMPAT_API_KEY_ALT"]
    fn = E50 / f"presence_{jname}.jsonl"
    done = {r["hkey"] for r in rj(fn) if r.get("label")}
    todo = {it["hk"]: it for it in presence_items() if it["hk"] not in done}
    print(f"[e50] presence {jname}: {len(todo)} calls", flush=True)
    f = open(fn, "a")
    lock = threading.Lock()

    def go(it):
        out = api_client.chat(jname, [{"role": "user", "content": PRESENCE.format(x=it["x"], y=it["y"], r=it["text"])}],
                         max_tokens=2048, effort="high")
        raw = (out.get("text") or "").strip()
        rec = {"hkey": it["hk"], "judge": jname, "label": parse_presence(raw), "raw": raw[-200:],
               "error": out.get("error", "")[:200], "finish_reason": out.get("finish_reason")}
        with lock:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            f.flush()

    with ThreadPoolExecutor(threads) as ex:
        list(ex.map(go, todo.values()))
    f.close()
    print(f"[e50] presence {jname}: finished", flush=True)


def judge(jname: str, threads: int, key: str, limit: int | None, shard: str = "") -> None:
    import api_client
    from e45_uniform_two_channel import PROMPT, PROMPT_PRE, parse_label
    if key == "alt":
        os.environ["OPENAI_COMPAT_API_KEY"] = os.environ["OPENAI_COMPAT_API_KEY_ALT"]
    recs = text_records()
    lab, e38 = caches(jname)
    todo, seen = [], set()
    for proto in ("kept", "removed"):  # kept first: removed texts with the same body then reuse it on the next run
        for r in sorted(recs, key=lambda r: (r["cond"] == "real", r["model"], r["set"], r["uid"])):
            if proto == "removed" and r["same_body"]:
                continue
            lbl, _ = label_of(r, proto, lab, e38)
            hk = r[f"hk_{proto}"]
            if lbl is None and hk not in seen:
                seen.add(hk)
                todo.append((hk, r["x"], r["y"], r[proto], PROMPT_PRE if proto == "kept" else PROMPT))
    if shard:
        i, n = map(int, shard.split("/"))
        todo = [t for t in todo if int(t[0], 16) % n == i]
    if limit:
        todo = todo[:limit]
    print(f"[e50] judge {jname}: {len(todo)} calls ({len(recs)} texts)", flush=True)
    fn = E50 / f"judge_{jname}.jsonl"
    f = open(fn, "a")
    lock = threading.Lock()
    t0, cnt = time.time(), Counter()

    def go(t):
        hk, x, y, body, prompt = t
        out = api_client.chat(jname, [{"role": "user", "content": prompt.format(x=x, y=y, r=body)}], max_tokens=2048, effort="high")
        raw = (out.get("text") or "").strip()
        err = out.get("error", "")[:200]
        lbl = parse_label(raw, x, y) if not err else None
        rec = {"hkey": hk, "judge": jname, "label": lbl, "raw": raw[-200:], "error": err,
               "finish_reason": out.get("finish_reason")}
        with lock:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            f.flush()
            cnt["err" if err else ("unlab" if lbl is None else "ok")] += 1
            s = sum(cnt.values())
            if s % 200 == 0:
                print(f"[e50] judge {jname}: {s}/{len(todo)} {dict(cnt)} {time.time() - t0:.0f}s", flush=True)

    with ThreadPoolExecutor(threads) as ex:
        list(ex.map(go, todo))
    f.close()
    print(f"[e50] judge {jname}: finished {dict(cnt)} {time.time() - t0:.0f}s", flush=True)
