"""E15 (GPU): hidden states for the "seeing right, writing wrong inside the CoT pass" probe.

For each occlusion prompt (cue-factorial set + original twins, both phrasings) with a stored greedy CoT
(outputs/e10/<model>{,_cues}.jsonl), run HF forward passes (teacher-forced, bf16) and save the residual stream
(all layers) at five positions:
  d  Direct prompt, last token ("Answer: (")
  s  CoT pass, last token of the first object mention in the first two-object relational claim
  r  CoT pass, last token of the relational phrase of that claim (overlap / cover / on top of / in front of / behind ...)
  a  CoT pass, last token ("Answer: (" after the rationale)
  f  filler pass (same-turn filler with |R| tokens), token at the same offset from the assistant start as r
Items without a parsed claim keep s/r/f = d (flag has_claim = 0).
Writes /dev/shm/e15/<model>_<shard>.npy (float16 [N, 5, L+1, D]) and <model>_<shard>.jsonl (metadata).

Usage: python e15_probe_extract.py --model qwen3vl_8b --shard 0 --nshard 3 [--limit 4]
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import torch

import vx
from vx import C
from e2d import load_occ
from e2f_claim_edit import ACT, OBJ, PAS, first_claim, split_sents

OUTD = "/dev/shm/e15"
POS = ("d", "s", "r", "a", "f")


def load_model(name: str):
    path = C.model_path(name)
    if name == "molmo7b":
        from transformers import AutoModelForCausalLM, AutoProcessor
        proc = AutoProcessor.from_pretrained(path, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(path, trust_remote_code=True, torch_dtype=torch.bfloat16).cuda().eval()
        return model, proc
    from transformers import AutoProcessor
    if name.startswith("qwen3vl"):
        from transformers import Qwen3VLForConditionalGeneration as M
    else:
        from transformers import Qwen2_5_VLForConditionalGeneration as M
    model = M.from_pretrained(path, torch_dtype=torch.bfloat16, attn_implementation="sdpa").cuda().eval()
    proc = AutoProcessor.from_pretrained(path, max_pixels=1280 * 28 * 28)
    return model, proc


class Runner:
    """Model-specific prompt rendering, char->token mapping and forward pass."""

    def __init__(self, name: str):
        self.name = name
        self.model, self.proc = load_model(name)
        self.tok = self.proc.tokenizer

    def render(self, msgs: list[dict]) -> str:
        if self.name == "molmo7b":
            from jinja2 import Template
            return " " + Template(vx.MOLMO_TEMPLATE).render(messages=msgs)
        hm = [{"role": m["role"], "content": [({"type": "image"} if p.get("type") == "image_url" else p) for p in m["content"]]
               if isinstance(m["content"], list) else m["content"]} for m in msgs]
        return self.proc.apply_chat_template(hm, tokenize=False, add_generation_prompt=False, continue_final_message=True)

    @torch.no_grad()
    def forward(self, text: str, img, char_pos: list[int | None]) -> tuple[np.ndarray, list[int]]:
        h, toks, self.last_info = self._forward(text, img, char_pos)
        return h, toks

    def _forward(self, text: str, img, char_pos: list[int | None]):
        """Hidden states [len(char_pos)+1, L+1, D] at the tokens covering char_pos (None -> last token) plus the last token."""
        enc = self.tok(text, return_offsets_mapping=True, add_special_tokens=False)
        ids, offs = enc["input_ids"], enc["offset_mapping"]
        if self.name == "molmo7b":
            out = self.proc.process(images=[img], tokens=ids)
            full = out["input_ids"]
            delta = len(full) - len(ids)
            assert full[-len(ids):].tolist() == ids
            mapk = lambda k: k + delta
            inputs = {k: v.unsqueeze(0).cuda() for k, v in out.items()}
            inputs["images"] = inputs["images"].to(torch.bfloat16)
            res = self.model(**inputs, output_hidden_states=True)
        else:
            inputs = self.proc(text=[text], images=[img], return_tensors="pt")
            full = inputs["input_ids"][0].tolist()
            pad = self.tok.convert_tokens_to_ids("<|image_pad|>")
            kp = ids.index(pad)
            delta = len(full) - len(ids)
            assert full[kp + delta + 1:] == ids[kp + 1:]
            mapk = lambda k: k if k < kp else k + delta
            res = self.model(**{k: v.cuda() for k, v in inputs.items()}, output_hidden_states=True)
        hs = torch.stack(res.hidden_states, 0)[:, 0]  # [L+1, T, D]
        n = hs.shape[1]
        toks = []
        for p in char_pos + [None]:
            if p is None:
                toks.append(n - 1)
                continue
            p, shift = p if isinstance(p, tuple) else (p, 0)  # (char, token offset): offset -1 = token before
            k = next(i for i, (a, b) in enumerate(offs) if a <= p < b)
            toks.append(mapk(k) + shift)
        lp = torch.log_softmax(res.logits[0, -1].float(), -1)
        ab = {c: float(lp[self.tok.convert_tokens_to_ids(c)]) for c in ("A", "B")}
        strs = [self.tok.decode([full[t]]) for t in toks]
        return hs[:, toks].transpose(0, 1).float().cpu().numpy().astype(np.float16), toks, {"ab": ab, "tok_str": strs}

    def tokpos(self, text: str, p: int) -> int:
        enc = self.tok(text, return_offsets_mapping=True, add_special_tokens=False)
        return next(i for i, (a, b) in enumerate(enc["offset_mapping"]) if a <= p < b)


def claim_chars(cot: str) -> tuple[int, int, str] | None:
    """(char of subject-noun end, char of relational-phrase end, claimed-front object) within the CoT text."""
    body = cot[len(C.COT_PREFIX):]
    sents = split_sents(body)
    fc = first_claim(sents)
    if fc is None:
        return None
    j, named = fc
    pos = len(C.COT_PREFIX)
    for k in range(j + 1):
        pos = cot.find(sents[k], pos)
        if k < j:
            pos += len(sents[k])
    s = sents[j]
    rel = [m for m in (ACT.search(s), PAS.search(s)) if m]
    m_rel = min(rel, key=lambda m: m.start())
    m_obj = OBJ.search(s)
    return pos + m_obj.end() - 1, pos + m_rel.end() - 1, named


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshard", type=int, default=1)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()
    t0 = time.time()
    cues = os.environ.get("CUES_DIR", "/dev/shm/occ_cues")
    items = []
    for which, d in (("cues", cues), ("twins", None)):
        kw = {"items_dir": d} if d else {}
        its = [dict(it, phrasing="front", key=it["uid"], set=which) for it in load_occ(None, **kw)]
        its += [dict(it, phrasing="behind", key=it["uid"] + "_behind", set=which) for it in load_occ(None, polar=True, **kw)]
        recs = vx.read_jsonl(f"{vx.OUT}/e10/{args.model}{'_cues' if which == 'cues' else ''}.jsonl")
        by = {}
        for r in recs:
            if r["condition"] in ("direct", "cot") and r["task"].startswith("occ"):
                by.setdefault(r["item_id"], {})[r["condition"]] = r
        for it in its:
            if it["key"] in by and "cot" in by[it["key"]]:
                items.append((it, by[it["key"]]))
    items = items[args.shard::args.nshard][: args.limit]
    print(f"[e15] {args.model} shard {args.shard}/{args.nshard}: {len(items)} items", flush=True)
    R = Runner(args.model)
    os.makedirs(OUTD, exist_ok=True)
    arr, meta = None, []
    for i, (it, rr) in enumerate(items):
        img = C.get_image(it)
        ref = vx.img_ref(img)
        cot = rr["cot"]["text"]
        cc = claim_chars(cot)
        # Direct
        dtext = R.render(C.build_messages(it, "direct", "none", "", ref))
        hd, _ = R.forward(dtext, img, [])
        info = {"hf_ab_direct": R.last_info["ab"]}
        # CoT pass
        ctext = R.render(C.build_messages(it, "cot", "none", cot, ref))
        cstart = ctext.rfind(cot + "\n" + C.FORCE_PREFIX)
        if cc:
            hc, tk = R.forward(ctext, img, [cstart + cc[0], cstart + cc[1]])
            off = R.tokpos(ctext, cstart + cc[1]) - R.tokpos(ctext, cstart)
            info |= {"tok_s": R.last_info["tok_str"][0], "tok_r": R.last_info["tok_str"][1]}
        else:
            hc, tk = R.forward(ctext, img, [])
            hc = np.concatenate([hd, hd, hc], 0)
            off = None
        info["hf_ab_cot"] = R.last_info["ab"]
        # filler pass (same length, same turn), token at the matched offset
        n_f = max(1, int(rr["cot"].get("gen_len") or 1))
        ftxt = C.filler_text(R.tok, n_f)
        ftext = R.render(C.build_messages(it, "filler", "none", ftxt, ref))
        fstart = ftext.rfind(ftxt + "\n" + C.FORCE_PREFIX)
        if off is not None:
            fk = R.tokpos(ftext, fstart) + off
            enc = R.tok(ftext, return_offsets_mapping=True, add_special_tokens=False)
            fend = R.tokpos(ftext, fstart + len(ftxt) - 1)
            fk = min(fk, fend)
            hf, _ = R.forward(ftext, img, [enc["offset_mapping"][fk][0]])
            hf = hf[:1]
            info["tok_f"] = R.last_info["tok_str"][0]
        else:
            hf = hd
        h = np.concatenate([hd[-1:], hc[:1], hc[1:2], hc[-1:], hf[:1]], 0)  # d, s, r, a, f
        if arr is None:
            arr = np.zeros((len(items),) + h.shape, np.float16)
        arr[i] = h
        mt = it["meta"]
        meta.append({"key": it["key"], "set": it["set"], "pair": it["group"], "phrasing": it["phrasing"],
                     "x": mt["x_name"].lower(), "front": mt["front_name"].lower(), "y_true": int(mt["x_name"].lower() == mt["front_name"].lower()),
                     "named": cc[2] if cc else None, "has_claim": int(bool(cc)), "gold": rr["cot"]["gold"],
                     "lp_direct": rr["direct"].get("lp_sum") or rr["direct"]["top20"], "lp_cot": rr["cot"].get("lp_sum") or rr["cot"]["top20"],
                     "front_above": mt.get("front_above"), "front_larger": mt.get("front_larger"), "cot_tokens": tk if cc else None, **info})
        if i % 50 == 0:
            print(f"[e15] {i}/{len(items)} ({time.time() - t0:.0f}s)", flush=True)
    sfx = f"_{args.shard}" + (f"_limit{args.limit}" if args.limit else "")
    np.save(f"{OUTD}/{args.model}{sfx}.npy", arr)
    with open(f"{OUTD}/{args.model}{sfx}.jsonl", "w") as fo:
        for m in meta:
            fo.write(json.dumps(m) + "\n")
    print(f"[e15] done {arr.shape} ({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
