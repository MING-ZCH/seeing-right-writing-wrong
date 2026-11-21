"""E10: structured grounded-state condition vs Direct / free CoT / filler on Synthetic Depth and occlusion twins.

The user turn is unchanged. In the ``state`` condition the assistant turn starts with STATE_PREFIX (a JSON skeleton
asking for each object's name and bounding box, the occluded object, and the front object); the model completes the
JSON greedily (stop at the closing code fence, <= 300 tokens) and the answer is read same-turn after "Answer: (".
All conditions are re-run in the same engine so the readout is identical:
  direct, cot (greedy R, same-turn readout), state, filler_S+turn (filler with |state| tokens), filler_R+turn (|R|).
Items: Synthetic Depth (500) + occlusion twins under both phrasings (2 x 400).

Usage: python -m experiments.interventions.structured_state --model qwen3b [--limit 5]
       python -m experiments.interventions.structured_state --analyze      (CPU; writes outputs/bundle/state.json)
"""
from __future__ import annotations
from srww import paths as _vtp  # noqa: E402

import argparse
import json
import math
import os
import re
import time
from pathlib import Path

STATE_PREFIX = ("Let me first write down the visual state as JSON in the format "
                "{\"objects\": [{\"name\": ..., \"bbox\": [x1, y1, x2, y2]}, {\"name\": ..., \"bbox\": [x1, y1, x2, y2]}], "
                "\"occluded\": <name of the object whose outline is cut off by the other>, "
                "\"front\": <name of the object in front>}.\n```json\n{\"objects\": [")
QRX = re.compile(r"Is the (\w+ \w+) (in front of|behind) the (\w+ \w+)\?")
CONDS = ("direct", "cot", "state", "filler_S+turn", "filler_R+turn")


# ----------------------------------------------------------------------------- GPU run

def run(args) -> None:
    from srww import vllm_utils as vx
    from srww.vllm_utils import C
    from vllm import SamplingParams
    from experiments.occlusion.twins_readout import load_occ
    t0 = time.time()
    if args.set == "cues":
        cues = os.environ.get("CUES_DIR", f"{_vtp.ITEMS}/occ_cues")
        items = [dict(it, phrasing="front", key=it["uid"]) for it in load_occ(args.limit, items_dir=cues)]
        items += [dict(it, phrasing="behind", key=it["uid"] + "_behind") for it in load_occ(args.limit, polar=True, items_dir=cues)]
    else:
        items = [dict(it, phrasing="front", key=it["uid"]) for it in vx.load_items(["syn_depth"], limit=args.limit)]
        items += [dict(it, phrasing="front", key=it["uid"]) for it in load_occ(args.limit)]
        items += [dict(it, phrasing="behind", key=it["uid"] + "_behind") for it in load_occ(args.limit, polar=True)]
    print(f"[e10] {args.model}: {len(items)} items", flush=True)
    llm = vx.make_llm(args.model)
    tok = llm.get_tokenizer()
    refs = [vx.img_ref(C.get_image(it)) for it in items]
    ch = [it["choices"] for it in items]
    # generations: free CoT and structured state
    gens = vx.generate(llm, args.model, [C.cot_messages(it, r) for it, r in zip(items, refs)])
    cots = [C.COT_PREFIX + " " + g["text"] for g in gens]
    smsgs = [[{"role": "user", "content": [r, {"type": "text", "text": it["question"]}]},
              {"role": "assistant", "content": STATE_PREFIX}] for it, r in zip(items, refs)]
    sres = llm.chat(smsgs, SamplingParams(temperature=0.0, max_tokens=300, stop=["```"]), **vx.chat_kw(args.model))
    states = [STATE_PREFIX + r.outputs[0].text.rstrip() + "\n```" for r in sres]
    s_len = [len(r.outputs[0].token_ids) for r in sres]
    ctx = {"direct": [("direct", "none", "")] * len(items),
           "cot": [("cot", "none", c) for c in cots],
           "state": [("cot", "none", s) for s in states],
           "filler_S+turn": [("filler", "turn", C.filler_text(tok, max(1, n))) for n in s_len],
           "filler_R+turn": [("filler", "turn", C.filler_text(tok, max(1, g["len"]))) for g in gens]}
    recs = []
    for cond in CONDS:
        msgs = [C.build_messages(it, b, s, c, r) for it, (b, s, c), r in zip(items, ctx[cond], refs)]
        outs = vx.readout(llm, args.model, msgs, ch)
        for i, (it, o) in enumerate(zip(items, outs)):
            rec = {"model": args.model, "task": it["task"], "item_id": it["key"], "base_id": it["uid"],
                   "phrasing": it["phrasing"], "condition": cond, "gold": it["gold"], "letters": it["choices"],
                   "image_id": it["group"], "question": it["question"], "meta": it.get("meta"), **o}
            if cond == "cot":
                rec |= {"text": cots[i], "gen_len": gens[i]["len"]}
            if cond == "state":
                rec |= {"text": states[i], "gen_len": s_len[i]}
            recs.append(rec)
        print(f"[e10] {cond} done ({time.time() - t0:.0f}s)", flush=True)
    sfx = ("_cues" if args.set == "cues" else "") + (f"_limit{args.limit}" if args.limit else "")
    vx.write_jsonl(f"{vx.OUT}/e10/{args.model}{sfx}.jsonl", recs)
    print(f"[e10] wrote {len(recs)} records ({time.time() - t0:.0f}s)", flush=True)


# ----------------------------------------------------------------------------- parsing (CPU)

def truth(r: dict) -> tuple[str, str, str, str] | None:
    """(subject x, other y, front object, back object) for a record."""
    q = QRX.search(r["question"])
    if not q:
        return None
    x, y = q.group(1).lower(), q.group(3).lower()
    if r.get("meta"):
        front = r["meta"]["front_name"].lower()
    else:  # Synthetic Depth: only "in front" questions
        front = x if r["gold"] == "A" else y
    back = y if front == x else x
    return x, y, front, back


def match_obj(v, x: str, y: str) -> str | None:
    """Map a JSON field value to one of the two object names (None if unparsable / ambiguous)."""
    if not isinstance(v, str):
        return None
    s = v.lower().replace("grey", "gray")
    hx, hy = x in s, y in s
    if hx != hy:
        return x if hx else y
    sx, sy = x.split()[1] in s, y.split()[1] in s  # shape word
    if sx != sy:
        return x if sx else y
    cx, cy = x.split()[0] in s, y.split()[0] in s  # color word
    if cx != cy:
        return x if cx else y
    return None


def parse_state(text: str) -> dict:
    """Extract the 'occluded' / 'front' fields (JSON first, regex fallback)."""
    body = text[text.rfind("```json") + 7:] if "```json" in text else text
    body = body.split("```")[0].strip()
    out = {"json_ok": False, "front": None, "occluded": None, "n_boxes": 0}
    try:
        d = json.loads(body[: body.rfind("}") + 1])
        out["json_ok"] = True
        out["front"], out["occluded"] = d.get("front"), d.get("occluded")
        objs = [o for o in d.get("objects", []) if isinstance(o, dict)] if isinstance(d.get("objects"), list) else []
        out["n_boxes"] = sum(isinstance(o.get("bbox", o.get("bounding_box")), list) for o in objs)
        for k in ("front", "occluded"):  # per-object booleans / strings instead of top-level fields
            if isinstance(out[k], dict):
                out[k] = out[k].get("name")
            if not isinstance(out[k], str):
                hits = [o.get("name") for o in objs if o.get(k) is True]
                strs = [o.get(k) for o in objs if isinstance(o.get(k), str)]
                out[k] = hits[0] if len(hits) == 1 else (strs[0] if len(set(strs)) == 1 else None)
    except Exception:
        for k in ("front", "occluded"):
            m = re.search(rf'"{k}"\s*:\s*"([^"]+)"', body)
            out[k] = m.group(1) if m else None
        out["n_boxes"] = len(re.findall(r'"bbox"\s*:\s*\[', body))
    return out


# ----------------------------------------------------------------------------- analysis (CPU)

def analyze(O: Path, out_dir: Path) -> None:
    import numpy as np
    from scipy.stats import norm
    from srww.common import cluster_bootstrap_ci
    MODELS = ["qwen3b", "qwen7b", "internvl3_8b", "phi35v", "molmo7b", "qwen3vl_8b", "internvl35_8b"]
    rng = np.random.default_rng(0)

    def pe(r):
        lp = r.get("lp_sum") or r["top20"]
        fin = {k: v for k, v in lp.items() if v is not None and math.isfinite(v)}
        return max(fin, key=fin.get) if fin else None

    def sdt(resp, tru):
        h = (resp[tru].sum() + .5) / (tru.sum() + 1)
        f = (resp[~tru].sum() + .5) / ((~tru).sum() + 1)
        return norm.ppf(h) - norm.ppf(f), -(norm.ppf(h) + norm.ppf(f)) / 2

    def ci(v, g):
        m, lo, hi = cluster_bootstrap_ci(np.asarray(v, float), list(g), 2000)
        return {"mean": round(100 * m, 1), "lo": round(100 * lo, 1), "hi": round(100 * hi, 1), "n": len(v)}

    res, lines = {}, []
    for m in MODELS:
        p = O / "e10" / f"{m}.jsonl"
        if not p.exists():
            continue
        R = [json.loads(l) for l in open(p)]
        by = {(r["item_id"], r["condition"]): r for r in R}
        mr = {}
        # --- twins: pooled d' on "subject X is in front" over both phrasings, resampling twin pairs
        tw = sorted({r["item_id"] for r in R if r["task"].startswith("occ")})
        pairs = sorted({by[(k, "direct")]["image_id"] for k in tw})
        resp = {c: {} for c in CONDS}
        for k in tw:
            r0 = by[(k, "direct")]
            x, y, front, back = truth(r0)
            for c in CONDS:
                r = by[(k, c)]
                says = (pe(r) == "A") if r["phrasing"] == "front" else (pe(r) == "B")
                resp[c].setdefault(r0["image_id"], []).append((int(says), int(front == x), float(pe(r) == r["gold"])))
        est = {}
        for c in CONDS:
            a = np.array([t for pp in pairs for t in resp[c][pp]])
            d, cc = sdt(a[:, 0].astype(bool), a[:, 1].astype(bool))
            est[c] = {"dprime": round(float(d), 2), "c": round(float(cc), 2), "acc": round(100 * a[:, 2].mean(), 1)}
        boots = {c: [] for c in CONDS}
        for _ in range(2000):
            idx = rng.integers(0, len(pairs), len(pairs))
            for c in CONDS:
                a = np.array([t for i in idx for t in resp[c][pairs[i]]])
                boots[c].append(sdt(a[:, 0].astype(bool), a[:, 1].astype(bool))[0])
        q = lambda v: [round(float(np.quantile(v, .025)), 2), round(float(np.quantile(v, .975)), 2)]
        for c in CONDS:
            est[c]["dprime_ci"] = q(boots[c])
        for a_, b_ in (("state", "direct"), ("state", "cot"), ("state", "filler_S+turn"), ("cot", "direct")):
            dv = np.array(boots[a_]) - np.array(boots[b_])
            est[f"d_dprime:{a_}-{b_}"] = {"mean": round(est[a_]["dprime"] - est[b_]["dprime"], 2), "ci": q(dv)}
        mr["twins"] = est
        # --- accuracy deltas (twins: pair clusters; syn: items)
        for st, sel in (("twins_acc", lambda r: r["task"].startswith("occ")), ("syn_depth", lambda r: r["task"] == "syn_depth")):
            keys = sorted({r["item_id"] for r in R if sel(r)})
            g = [by[(k, "direct")]["image_id"] for k in keys]
            acc = {c: np.array([float(pe(by[(k, c)]) == by[(k, c)]["gold"]) for k in keys]) for c in CONDS}
            row = {f"acc:{c}": ci(acc[c], g) for c in CONDS}
            for a_, b_ in (("state", "direct"), ("state", "cot"), ("state", "filler_S+turn"), ("cot", "direct")):
                row[f"diff:{a_}-{b_}"] = ci(acc[a_] - acc[b_], g)
            if st == "syn_depth":
                gy = np.array([by[(k, "direct")]["gold"] == "A" for k in keys])
                for c in CONDS:
                    py = np.array([pe(by[(k, c)]) == "A" for k in keys])
                    d, cc = sdt(py, gy)
                    row[f"sdt:{c}"] = {"dprime": round(float(d), 2), "c": round(float(cc), 2)}
            mr[st] = row
        # --- state parse and field truth
        srec = [r for r in R if r["condition"] == "state"]
        par = []
        for r in srec:
            x, y, front, back = truth(r)
            ps = parse_state(r["text"])
            f_, o_ = match_obj(ps["front"], x, y), match_obj(ps["occluded"], x, y)
            # answer implied by the state's front field
            implied = None
            if f_:
                says_front = f_ == x
                implied = ("A" if says_front else "B") if r["phrasing"] == "front" else ("B" if says_front else "A")
            mt = r.get("meta") or {}
            larger = (front if mt.get("area_front_px", 0) > mt.get("area_back_px", 0) else back) if mt else None
            upper = (front if mt.get("front_above") else back) if mt else None
            par.append({"set": "twins" if r["task"].startswith("occ") else "syn", "json_ok": ps["json_ok"],
                        "front_is_larger": None if (f_ is None or larger is None) else f_ == larger,
                        "front_is_upper": None if (f_ is None or upper is None) else f_ == upper,
                        "front_parsed": f_ is not None, "occl_parsed": o_ is not None,
                        "front_true": None if f_ is None else f_ == front, "occl_true": None if o_ is None else o_ == back,
                        "consistent_fields": None if (f_ is None or o_ is None) else f_ != o_,
                        "answer_follows_state": None if implied is None else pe(r) == implied,
                        "trunc": r["gen_len"] >= 300})
        st_sum = {}
        for s in ("syn", "twins", "all"):
            P = [p for p in par if s == "all" or p["set"] == s]
            f = lambda k: round(100 * float(np.mean([p[k] for p in P if p[k] is not None])), 1) if any(p[k] is not None for p in P) else None
            st_sum[s] = {"n": len(P), "json_ok": round(100 * float(np.mean([p["json_ok"] for p in P])), 1),
                         "front_parsed": round(100 * float(np.mean([p["front_parsed"] for p in P])), 1),
                         "occl_parsed": round(100 * float(np.mean([p["occl_parsed"] for p in P])), 1),
                         "front_true": f("front_true"), "occl_true": f("occl_true"),
                         "fields_consistent": f("consistent_fields"), "answer_follows_state_front": f("answer_follows_state"),
                         "front_is_larger_object": f("front_is_larger"), "front_is_upper_object": f("front_is_upper"),
                         "truncated": round(100 * float(np.mean([p["trunc"] for p in P])), 1),
                         "mean_state_tokens": round(float(np.mean([r["gen_len"] for r in srec if (s == "all" or (r["task"].startswith("occ")) == (s == "twins"))])), 1)}
        mr["state_parse"] = st_sum
        res[m] = mr
        t, sd = mr["twins"], mr["syn_depth"]
        lines.append(f"{m:14s} twins d' D {t['direct']['dprime']} CoT {t['cot']['dprime']} State {t['state']['dprime']} "
                     f"FilS {t['filler_S+turn']['dprime']} | S-D {t['d_dprime:state-direct']['mean']:+.2f} {t['d_dprime:state-direct']['ci']} "
                     f"S-CoT {t['d_dprime:state-cot']['mean']:+.2f} {t['d_dprime:state-cot']['ci']} | twins acc D {t['direct']['acc']} CoT {t['cot']['acc']} S {t['state']['acc']}"
                     f" | syn acc D {sd['acc:direct']['mean']} CoT {sd['acc:cot']['mean']} S {sd['acc:state']['mean']} "
                     f"S-D {sd['diff:state-direct']['mean']:+.1f} [{sd['diff:state-direct']['lo']:+.1f},{sd['diff:state-direct']['hi']:+.1f}] "
                     f"S-CoT {sd['diff:state-cot']['mean']:+.1f} [{sd['diff:state-cot']['lo']:+.1f},{sd['diff:state-cot']['hi']:+.1f}]"
                     f" | parse json {st_sum['all']['json_ok']} front_true {st_sum['twins']['front_true']}/{st_sum['syn']['front_true']} "
                     f"follows {st_sum['all']['answer_follows_state_front']}")
    json.dump(res, open(out_dir / "state.json", "w"), indent=1)
    (out_dir / "state.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


def main() -> None:
    """CLI entry point."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--model")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--analyze", action="store_true")
    ap.add_argument("--set", default="main", choices=["main", "cues"])
    ap.add_argument("--outputs", default=f"{_vtp.OUTPUTS}")
    args = ap.parse_args()
    if args.analyze:
        analyze(Path(args.outputs), Path("outputs/bundle"))
    else:
        run(args)


if __name__ == "__main__":
    main()
