"""E30 judge calls: full-sample audits on the cue-factorial set.

Tasks (Claude Opus 5.5 + GPT-6-Astra, effort high, api_client client; key from OPENAI_COMPAT_API_KEY, never printed):
  commit   ALL cue-set D+C- commit sentences j* (Exp-28 trajectories, e20 rule) of Qwen3-VL-8B and Molmo-7B-D, plus
           relational D+C+ comparators under the Exp-28 rule applied to ALL D+C+ rationales with >= 1 relational
           sentence (srww.claim_parse.stype): target = relational sentence nearest to a relative position drawn (seed 30) from
           the same model's full list of commit positions. Each sentence is judged with
             e20   the original prompt (CORRECT / INCORRECT / NOCLAIM / UNCLEAR), and
             ontop the Exp-29 depth/vertical prompt (READING, DEPTH_TRUE, VERTICAL_TRUE).
           Labels from Exp-28 (commit / prej, e20 prompt), Exp-29 prej2 (e20 prompt, Qwen3-VL j*) and Exp-29 ontop are
           reused when (model, item_id, idx, sentence) match.
  descall  ALL 800 question-blind descriptions per model of the Exp-28 cue description arm with the Exp-29 description
           prompt (FRONT_CLAIMED, DEPTH_CLAIM, ON_TOP_ONLY); Exp-29 desc labels reused.
-> outputs/e30/judge_<judge>_<task>[_<prompt>].jsonl (appended per call; resumable)

Usage: python -m judges.cue_audit_judge --task commit --prompt e20 --judge claude-opus-5-5 [--dry] [--threads 6]
       python -m judges.cue_audit_judge --task descall --judge gpt-6-astra
"""
from __future__ import annotations
from srww import paths as _vtp  # noqa: E402

import argparse
import json
import random
import threading
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from srww import records as B
from judges.commit_sentence_judge import PROMPT as PROMPT_E20, first_claim_idx, parse as parse_e20
from judges.claim_edit_judge import IMG, PROMPT_DESC, PROMPT_ONTOP, parse_desc, parse_ontop
from srww.claim_parse import stype

O = Path(f"{_vtp.OUTPUTS}")
E28, E29, E30 = O / "e28", O / "e29", O / "e30"
COMP = ["qwen3vl_8b", "molmo7b"]


def img(base_id: str) -> str:
    return str(IMG["cues"] / f"{base_id}.png")


def sample_commit() -> list[dict]:
    """All D+C- commits and all position-matched relational D+C+ comparators, cue set."""
    rng = random.Random(30)
    out = []
    for m in COMP:
        T = sorted(B.rj(E28 / f"{m}_cues_traj.jsonl"), key=lambda r: r["item_id"])
        dcm = [r for r in T if r["group"] == "D+C-" and r["commit_idx"] is not None and r["sentences"]]
        relpos = [r["commit_idx"] / max(1, len(r["sentences"]) - 1) for r in dcm]
        for r in dcm:
            j, s = r["commit_idx"], r["sentences"]
            out.append({"model": m, "set": "cues", "item_id": r["item_id"], "base_id": r["base_id"], "image_id": r["image_id"],
                        "phrasing": r["phrasing"], "group": "D+C-", "idx": j, "n_sent": len(s), "commit_idx": j,
                        "first_claim_idx": first_claim_idx(s), "stype": stype(s[j]), "sentence": s[j],
                        "context": s[j - 1] if j > 0 else "(none)"})
        for r in T:
            s = r["sentences"]
            if r["group"] != "D+C+" or not any(stype(x) == "relation" for x in s):
                continue
            target = rng.choice(relpos) * (len(s) - 1)
            rel = [j for j, x in enumerate(s) if stype(x) == "relation"]
            j = min(rel, key=lambda t: abs(t - target))
            out.append({"model": m, "set": "cues", "item_id": r["item_id"], "base_id": r["base_id"], "image_id": r["image_id"],
                        "phrasing": r["phrasing"], "group": "D+C+rel", "idx": j, "n_sent": len(s), "commit_idx": None,
                        "first_claim_idx": first_claim_idx(s), "stype": "relation", "sentence": s[j],
                        "context": s[j - 1] if j > 0 else "(none)"})
    return out


def sample_descall() -> list[dict]:
    out = []
    for m in COMP:
        seen = set()
        for r in B.rj(E28 / f"{m}_cues_desc.jsonl"):
            if r["condition"] != "desc" or r["base_id"] in seen:
                continue
            seen.add(r["base_id"])
            out.append({"model": m, "kind": "desc", "item_id": r["base_id"], "base_id": r["base_id"], "image_id": r["image_id"],
                        "text": r["text"], "front_name": r["meta"]["front_name"], "back_name": r["meta"]["back_name"]})
    return out


def reuse_e20(jname: str) -> dict:
    """{(model, item_id, idx): (label, raw, sentence)} from earlier runs with the e20 prompt on cue-set sentences."""
    out = {}
    for fn in (E28 / f"judge_{jname}_commit_cues.jsonl", E28 / f"judge_{jname}_prej_cues.jsonl", E29 / f"judge_{jname}_prej2.jsonl"):
        for r in B.rj(fn):
            if r.get("api_error") or r["label"] == "UNPARSED":
                continue
            out[(r["model"], r["item_id"], r["idx"])] = (r["label"], r["raw"], r["sentence"], fn.name)
    return out


def reuse_ontop(jname: str) -> dict:
    out = {}
    for r in B.rj(E29 / f"judge_{jname}_ontop.jsonl"):
        if r.get("set") == "syn" or r.get("api_error") or "UNPARSED" in (r["reading"], r["depth_true"], r["vertical_true"]):
            continue
        if not str(r.get("src", "")).endswith("cues"):
            continue
        out[(r["model"], r["item_id"], r["idx"])] = ({k: r[k] for k in ("reading", "depth_true", "vertical_true")}, r["raw"], r["sentence"], r["src"])
    return out


def reuse_desc(jname: str) -> dict:
    out = {}
    for r in B.rj(E29 / f"judge_{jname}_desc.jsonl"):
        if r["kind"] == "desc" and not r.get("api_error") and r["depth_claim"] != "UNPARSED":
            out[(r["model"], r["item_id"])] = ({k: r[k] for k in ("front_claimed", "depth_claim", "on_top_only")}, r["raw"], r["text"])
    return out


def key_of(x: dict) -> tuple:
    return (x["model"], x["item_id"], x.get("idx", -1), x.get("group", x.get("kind")))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True, choices=["commit", "descall"])
    ap.add_argument("--prompt", choices=["e20", "ontop"], default="e20")
    ap.add_argument("--judge", required=True)
    ap.add_argument("--threads", type=int, default=6)
    ap.add_argument("--rpm", type=float, default=0.0, help="max request starts per minute for this process (0 = no limit)")
    ap.add_argument("--dry", action="store_true")
    ap.add_argument("--limit", type=int)
    a = ap.parse_args()
    items = sample_commit() if a.task == "commit" else sample_descall()
    print(f"[e30j] {a.task}: {len(items)} items {dict(Counter((x['model'], x.get('group', x.get('kind'))) for x in items))}", flush=True)
    E30.mkdir(exist_ok=True)
    tag = f"{a.task}_{a.prompt}" if a.task == "commit" else a.task
    p = E30 / f"judge_{a.judge}_{tag}{'_limit%d' % a.limit if a.limit else ''}.jsonl"
    done = {key_of(r) for r in B.rj(p) if not r.get("api_error")} if p.exists() else set()
    if a.task == "commit":
        old = reuse_e20(a.judge) if a.prompt == "e20" else reuse_ontop(a.judge)
        reused = [x for x in items if (x["model"], x["item_id"], x["idx"]) in old and old[(x["model"], x["item_id"], x["idx"])][2] == x["sentence"]]
    else:
        old = reuse_desc(a.judge)
        reused = [x for x in items if (x["model"], x["item_id"]) in old and old[(x["model"], x["item_id"])][2] == x["text"]]
    rk = {key_of(x) for x in reused}
    todo = [x for x in items if key_of(x) not in rk and key_of(x) not in done]
    # commits first so that a partial run still covers the primary sample
    todo.sort(key=lambda x: (x.get("group") != "D+C-", x["model"], x["item_id"]))
    if a.limit:
        todo = todo[:a.limit]
    print(f"[e30j] {p.name}: reused {len(reused)}, already done {len(done)}, to judge {len(todo)}", flush=True)
    if a.dry:
        return
    lock = threading.Lock()
    f = open(p, "a")
    with lock:
        for x in reused:
            if key_of(x) in done:
                continue
            if a.task == "commit":
                lab, raw, _, src = old[(x["model"], x["item_id"], x["idx"])]
                lab = {"label": lab} if a.prompt == "e20" else lab
            else:
                lab, raw, _ = old[(x["model"], x["item_id"])]
                src = "e29_desc"
            f.write(json.dumps({k: v for k, v in x.items()} | {"judge": a.judge, **lab, "raw": raw, "api_error": False,
                                                                "reused": src}) + "\n")
        f.flush()

    import time
    from srww import api_client as api_client
    from PIL import Image
    cnt = Counter()
    gap = 60.0 / a.rpm if a.rpm else 0.0
    nxt = [time.time()]
    rl = threading.Lock()

    def go(x):
        if gap:
            # client-side limiter: the platform caps requests per minute per project (HTTP 429 otherwise)
            with rl:
                t = max(nxt[0], time.time())
                nxt[0] = t + gap
            time.sleep(max(0.0, t - time.time()))
        im = Image.open(img(x["base_id"])).convert("RGB")
        if a.task == "descall":
            txt, parser = PROMPT_DESC.format(d=x["text"]), parse_desc
        elif a.prompt == "ontop":
            txt, parser = PROMPT_ONTOP.format(c=x["context"], s=x["sentence"]), parse_ontop
        else:
            txt, parser = PROMPT_E20.format(c=x["context"], s=x["sentence"]), lambda t: {"label": parse_e20(t)}
        r = api_client.chat(a.judge, [{"role": "user", "content": [api_client.img_part(im), {"type": "text", "text": txt}]}],
                       max_tokens=2048, effort="high")
        rec = x | {"judge": a.judge, **parser(r.get("text")), "raw": (r.get("text") or r.get("error", ""))[:200],
                   "api_error": "error" in r}
        with lock:
            f.write(json.dumps(rec) + "\n")
            f.flush()
            cnt["err" if rec["api_error"] else "ok"] += 1
            if sum(cnt.values()) % 200 == 0:
                print(f"[e30j] {p.name}: {dict(cnt)}", flush=True)

    with ThreadPoolExecutor(a.threads) as ex:
        list(ex.map(go, todo))
    f.close()
    print(f"[e30j] {p.name}: finished {dict(cnt)}", flush=True)


if __name__ == "__main__":
    main()
