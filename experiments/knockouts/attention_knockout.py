"""E52 (GPU): attention-edge knockout at the forced answer position + Direct-probe transport states.

For a stratified sample of stored greedy CoT rationales (outputs/e10, cue-factorial set + occlusion twins) the
forced readout P(gold) at "Answer: (" is recomputed in HF transformers (bf16, text decoder on a patched eager
attention) under knockouts applied to the answer-prefix query rows ("\\nAnswer: ("):
  a: rows -> image tokens blocked     b: rows -> rationale body blocked     c: rows -> conclusion blocked
  d: top-5% visual heads (calibration split, Direct) zeroed at the rows     r1/r2: layer-matched random heads
  secondary: ac, a_all (image blocked from the whole assistant turn), a_lo/a_hi/c_lo/c_hi (layer halves),
  a_full (addendum 1: image blocked from every position after the image), c_afull (addendum 2: c + a_full)
Direct reference: none, a, a_full, d, r1, r2.  Filler reference: filler_R+turn, no knockout.
Pre-registration: outputs/e52/prereg.txt.

Stages:
  python -m experiments.knockouts.attention_knockout --model qwen3vl_8b --stage calib            # sample + calibration split -> visual heads
  python -m experiments.knockouts.attention_knockout --model qwen3vl_8b --stage main --shard 0 --nshard 2
Outputs (shared network storage): outputs/e52/{sample_<m>.json, heads_<m>.json, <m>_<shard>.jsonl, h_<m>_<shard>.npz,
att_<m>_<shard>.npz, calib_<m>.npz}
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import time
from collections import defaultdict

import numpy as np
import torch

from srww import vllm_utils as vx
from srww.vllm_utils import C
from srww.claim_parse import first_claim_v2
from experiments.occlusion.twins_readout import load_occ
from srww.claim_parse import split_sents

OUTD = os.environ.get("E52_OUT", f"{vx.OUT}/e52")
CUES_DIR = os.environ.get("E52_CUES", f"{vx.OUT}/e50/occ_cues")
TWIN_DIR = f"{vx.W}/data/items/occlusion"
SEED = 52
ALLOC = (("D+C+", 150), ("D+C-", 100), ("D-C+", 100), ("D-C-", 50))
N_CALIB_PAIRS = 25
FAST = os.environ.get("E52_FAST", "0") == "1"
VIS_ATTN = os.environ.get("E52_VIS_ATTN", "sdpa")
DTYPE = torch.float32 if os.environ.get("E52_DTYPE") == "fp32" else torch.bfloat16
COT_CONDS = ("none", "a", "b", "c", "ac", "a_all", "a_full", "c_afull", "d", "r1", "r2", "a_lo", "a_hi", "c_lo", "c_hi")
DIR_CONDS = ("none", "a", "a_full", "d", "r1", "r2")


# ============================================================================= items / sample

def pred(lp: dict) -> str:
    fin = {k: v for k, v in lp.items() if v is not None and math.isfinite(v)}
    return max(fin, key=fin.get)


def p_gold(lp: dict, gold: str) -> float:
    fin = {k: v for k, v in lp.items() if v is not None and math.isfinite(v)}
    m = max(fin.values())
    z = sum(math.exp(v - m) for v in fin.values())
    return math.exp(fin[gold] - m) / z if gold in fin else 0.0


def last_relation(cot: str, x: str, y: str) -> tuple[int, int, str] | None:
    """(char start of body, char start of last relational sentence, claimed-front shape) within the CoT text."""
    pre = C.COT_PREFIX
    body = cot[len(pre):] if cot.startswith(pre) else cot
    off0 = len(cot) - len(body)
    sents = split_sents(body)
    pos, starts = off0, []
    for s in sents:
        p = cot.find(s, pos)
        starts.append(p)
        pos = p + len(s)
    for j in range(len(sents) - 1, -1, -1):
        fc = first_claim_v2([sents[j]], negation=True)
        if fc and fc[1] in (x, y):
            return (starts[0] if sents else len(cot)), starts[j], fc[1]
    return None


def written_letter(question: str, claimed: str, x: str, y: str) -> str:
    """Letter implied by the written relation ((A) yes / (B) no)."""
    behind = " behind the " in question.split("\n")[0]
    yes = (claimed == y) if behind else (claimed == x)
    return "A" if yes else "B"


def load_set(model: str, which: str) -> list[dict]:
    d = CUES_DIR if which == "cues" else TWIN_DIR
    its = [dict(it, key=it["uid"]) for it in load_occ(None, items_dir=d)]
    its += [dict(it, key=it["uid"] + "_behind") for it in load_occ(None, polar=True, items_dir=d)]
    recs = vx.read_jsonl(f"{vx.OUT}/e10/{model}{'_cues' if which == 'cues' else ''}.jsonl")
    by = defaultdict(dict)
    for r in recs:
        if r["condition"] in ("direct", "cot", "filler_R+turn") and r["task"].startswith("occ"):
            by[r["item_id"]][r["condition"]] = r
    out = []
    for it in its:
        rr = by.get(it["key"], {})
        if "cot" not in rr or "direct" not in rr:
            continue
        assert rr["cot"]["question"] == it["question"] and rr["cot"]["gold"] == it["gold"], it["key"]
        lpd = rr["direct"].get("lp_sum") or rr["direct"]["top20"]
        lpc = rr["cot"].get("lp_sum") or rr["cot"]["top20"]
        cell = ("D+" if pred(lpd) == it["gold"] else "D-") + ("C+" if pred(lpc) == it["gold"] else "C-")
        mt = it["meta"]
        x, y = mt["x_name"].lower(), mt["y_name"].lower()
        rel = last_relation(rr["cot"]["text"], x, y)
        fil = rr.get("filler_R+turn")
        out.append({"key": it["key"], "set": which, "pair": it["group"], "cell": cell, "gold": it["gold"],
                    "x": x, "y": y, "front": mt["front_name"].lower(), "y_true": int(x == mt["front_name"].lower()),
                    "question": it["question"], "image_path": it["image_path"], "cot": rr["cot"]["text"],
                    "gen_len": int(rr["cot"].get("gen_len") or 1),
                    "p_direct_stored": p_gold(lpd, it["gold"]), "p_cot_stored": p_gold(lpc, it["gold"]),
                    "p_filler_stored": p_gold(fil.get("lp_sum") or fil["top20"], it["gold"]) if fil else None,
                    "rel": rel, "written": written_letter(it["question"], rel[2], x, y) if rel else None})
    return out


def make_sample(model: str) -> dict:
    """Calibration pairs (reserved first) and the stratified evaluation sample, per set."""
    res = {"calib": [], "eval": [], "N_cell": {}, "n_cell": {}}
    for which in ("cues", "twins"):
        rows = load_set(model, which)
        rng = random.Random(f"{SEED}-{model}-{which}")
        pairs = sorted({r["pair"] for r in rows})
        calib_pairs = set(rng.sample(pairs, N_CALIB_PAIRS))
        cal = [r for r in rows if r["pair"] in calib_pairs]
        rest = [r for r in rows if r["pair"] not in calib_pairs]
        bycell = defaultdict(list)
        for r in rest:
            bycell[r["cell"]].append(r)
        for v in bycell.values():
            rng.shuffle(v)
        take, short = {}, 0
        for c, n in ALLOC:
            take[c] = min(n, len(bycell[c]))
            short += n - take[c]
        for c in ("D+C+", "D-C-", "D+C-", "D-C+"):
            add = min(short, len(bycell[c]) - take[c])
            take[c] += add
            short -= add
        ev = [r for c, _ in ALLOC for r in bycell[c][: take[c]]]
        res["calib"] += cal
        res["eval"] += ev
        res["N_cell"][which] = {c: sum(r["cell"] == c for r in rows) for c, _ in ALLOC}
        res["n_cell"][which] = take
    return res


# ============================================================================= patched attention

class KO:
    """Global knockout state read by the patched attention function."""
    mask: torch.Tensor | None = None       # bool [T, T], True = blocked (query row, key col)
    layers: set | None = None              # None = all layers
    heads: dict | None = None              # layer -> LongTensor of heads zeroed at `rows`
    rows: torch.Tensor | None = None
    rec_img: torch.Tensor | None = None    # LongTensor of image key positions (record last-row mass)
    rec_spans: dict | None = None          # name -> LongTensor of key positions (head-mean last-row mass)
    att: dict = {}
    stock: bool = False                    # Molmo: route to the original attention (numerical reference)


ORIG_EAGER = {}


def ko_core(li: int, query, key_states, value_states, attention_mask, scaling):
    """Attention with fp32 scores + knockout mask / head ablation / attention recording. Returns [B, H, T, d]."""
    # scores in fp32: bf16 scores (stock eager) distort Qwen2.5-VL readouts (large q.k logits, no QK-norm)
    w = torch.matmul(query.float(), key_states.float().transpose(2, 3)) * scaling
    T = w.shape[-1]
    if attention_mask is not None:
        w = w + attention_mask[:, :, -w.shape[-2]:, :T].float()
    else:
        w = w.masked_fill(torch.ones(T, T, dtype=torch.bool, device=w.device).triu(1), float("-inf"))
    on_layer = KO.layers is None or li in KO.layers
    if KO.mask is not None and on_layer:
        w = w.masked_fill(KO.mask, float("-inf"))
    w = torch.nn.functional.softmax(w, dim=-1, dtype=torch.float32)
    if KO.rec_img is not None:
        last = w[0, :, -1]
        KO.att.setdefault("img", {})[li] = last[:, KO.rec_img].sum(-1).cpu()
        for nm, ix in (KO.rec_spans or {}).items():
            KO.att.setdefault(nm, {})[li] = last[:, ix].sum(-1).mean().item() if len(ix) else 0.0
    w = w.to(query.dtype)
    o = torch.matmul(w, value_states)
    if KO.heads and on_layer and li in KO.heads:
        o[:, KO.heads[li].unsqueeze(1), KO.rows.unsqueeze(0)] = 0
    return o, w


def ko_eager(module, query, key, value, attention_mask, scaling, dropout=0.0, **kwargs):
    li = getattr(module, "_e52", None)
    if li is None:
        return ORIG_EAGER["fn"](module, query, key, value, attention_mask, scaling, dropout, **kwargs)
    from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import repeat_kv
    o, w = ko_core(li, query, repeat_kv(key, module.num_key_value_groups), repeat_kv(value, module.num_key_value_groups),
                   attention_mask, scaling)
    return o.transpose(1, 2).contiguous(), w


def ko_molmo(self, q, k, v, attn_mask=None, dropout_p=0.0, response_dropout_p=0.0, is_causal=False):
    """Replacement for MolmoBlock._scaled_dot_product_attention (q [B, H, T, d], k/v [B, kvH, T, d])."""
    if KO.stock:
        return ORIG_EAGER["molmo"](self, q, k, v, attn_mask, dropout_p, response_dropout_p, is_causal)
    rep_ = q.size(1) // k.size(1)
    if rep_ > 1:
        k = k.repeat_interleave(rep_, dim=1)
        v = v.repeat_interleave(rep_, dim=1)
    o, _ = ko_core(self.layer_id, q, k, v, attn_mask, 1.0 / math.sqrt(q.size(-1)))
    return o


def install_patch(model, name: str) -> int:
    if name == "molmo7b":
        blocks = [m for m in model.modules() if hasattr(m, "layer_id") and hasattr(m, "_scaled_dot_product_attention")]
        cls = type(blocks[0])
        ORIG_EAGER["molmo"] = cls._scaled_dot_product_attention
        cls._scaled_dot_product_attention = ko_molmo
        return len(blocks)
    if name.startswith("qwen3vl"):
        import transformers.models.qwen3_vl.modeling_qwen3_vl as M
    else:
        import transformers.models.qwen2_5_vl.modeling_qwen2_5_vl as M
    ORIG_EAGER["fn"] = M.eager_attention_forward
    M.eager_attention_forward = ko_eager
    n = 0
    for nm, mod in model.named_modules():
        if nm.endswith("self_attn") and "language_model" in nm and hasattr(mod, "layer_idx"):
            mod._e52 = mod.layer_idx
            n += 1
    return n


def load_model(name: str):
    from transformers import AutoProcessor
    path = C.model_path(name)
    if name == "molmo7b":
        from transformers import AutoModelForCausalLM
        proc = AutoProcessor.from_pretrained(path, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(path, trust_remote_code=True, torch_dtype=DTYPE).cuda().eval()
        return model, proc
    from transformers import Qwen2VLImageProcessor
    if name.startswith("qwen3vl"):
        from transformers import Qwen3VLForConditionalGeneration as Mdl
        proc = AutoProcessor.from_pretrained(path)
        if not FAST:
            proc.image_processor = Qwen2VLImageProcessor.from_pretrained(path)
    else:
        from transformers import Qwen2_5_VLForConditionalGeneration as Mdl
        proc = AutoProcessor.from_pretrained(path, max_pixels=1280 * 28 * 28)
        if not FAST:
            proc.image_processor = Qwen2VLImageProcessor.from_pretrained(path, max_pixels=1280 * 28 * 28)
    model = Mdl.from_pretrained(path, torch_dtype=DTYPE,
                                attn_implementation={"text_config": "eager", "vision_config": VIS_ATTN, "": "eager"}).cuda().eval()
    return model, proc


# ============================================================================= runner

class Runner:
    def __init__(self, name: str):
        self.name = name
        self.model, self.proc = load_model(name)
        self.tok = self.proc.tokenizer
        self.n_layers = install_patch(self.model, name)
        self.molmo = name == "molmo7b"
        if self.molmo:
            self.n_heads = self.model.config.num_attention_heads
            assert self.n_layers == self.model.config.num_hidden_layers
            self.pad = self.tok.convert_tokens_to_ids("<im_patch>")
        else:
            cfg = self.model.config.text_config
            self.n_heads = cfg.num_attention_heads
            assert self.n_layers == cfg.num_hidden_layers, (self.n_layers, cfg.num_hidden_layers)
            self.pad = self.tok.convert_tokens_to_ids("<|image_pad|>")
        self.letters = [self.tok.convert_tokens_to_ids(c) for c in ("A", "B")]

    def render(self, msgs: list[dict]) -> str:
        if self.molmo:
            from jinja2 import Template
            return " " + Template(vx.MOLMO_TEMPLATE).render(messages=msgs)
        hm = [{"role": m["role"], "content": [({"type": "image"} if p.get("type") == "image_url" else p) for p in m["content"]]
               if isinstance(m["content"], list) else m["content"]} for m in msgs]
        return self.proc.apply_chat_template(hm, tokenize=False, add_generation_prompt=False, continue_final_message=True)

    def encode(self, text: str, img, spans: dict[str, tuple[int, int]]):
        """Model inputs + token index sets for char spans {name: (a, b)} (token assigned by its first char)."""
        enc = self.tok(text, return_offsets_mapping=True, add_special_tokens=False)
        ids, offs = enc["input_ids"], enc["offset_mapping"]
        if self.molmo:
            out = self.proc.process(images=[img], tokens=ids)
            full = out["input_ids"].tolist()
            delta = len(full) - len(ids)
            assert full[delta:] == ids
            idx = {nm: [k + delta for k, (s, _) in enumerate(offs) if a <= s < b] for nm, (a, b) in spans.items()}
            idx["img"] = [i for i, t in enumerate(full) if t == self.pad]
            idx = {k: torch.tensor(v, dtype=torch.long, device="cuda") for k, v in idx.items()}
            inputs = {k: v.unsqueeze(0).cuda() for k, v in out.items()}
            inputs["images"] = inputs["images"].to(DTYPE)
            return inputs, idx, len(full)
        inputs = self.proc(text=[text], images=[img], return_tensors="pt")
        full = inputs["input_ids"][0].tolist()
        kp = ids.index(self.pad)
        delta = len(full) - len(ids)
        assert full[kp + delta + 1:] == ids[kp + 1:] and full[:kp] == ids[:kp]
        mapk = lambda k: k if k < kp else k + delta
        idx = {nm: [mapk(k) for k, (s, _) in enumerate(offs) if a <= s < b and ids[k] != self.pad] for nm, (a, b) in spans.items()}
        idx["img"] = [i for i, t in enumerate(full) if t == self.pad]
        dev = "cuda"
        idx = {k: torch.tensor(v, dtype=torch.long, device=dev) for k, v in idx.items()}
        return {k: v.to(dev) for k, v in inputs.items()}, idx, len(full)

    @torch.no_grad()
    def run(self, inputs: dict, hidden: bool = False):
        kw = {"last_logits_only": True} if self.molmo else {"logits_to_keep": 1}
        out = self.model(**inputs, output_hidden_states=hidden, **kw)
        lp = torch.log_softmax(out.logits[0, -1].float(), -1)
        ab = [float(lp[t]) for t in self.letters]
        h = None
        if hidden:
            h = torch.stack([x[0, -1] for x in out.hidden_states], 0).float().cpu().numpy().astype(np.float16)
        return ab, h

    @torch.no_grad()
    def run_sdpa(self, inputs: dict) -> list[float]:
        """Same forward with the stock SDPA text attention (numerical reference for the patched path)."""
        if self.molmo:
            KO.stock = True
            try:
                ab, _ = self.run(inputs)
            finally:
                KO.stock = False
            return ab
        cfg = self.model.config.text_config
        cfg._attn_implementation = "sdpa"
        try:
            ab, _ = self.run(inputs)
        finally:
            cfg._attn_implementation = "eager"
        return ab

    def set_ko(self, T: int, rows=None, cols=None, layers=None, heads=None):
        KO.layers, KO.heads, KO.rows = layers, heads, rows
        KO.mask = None
        if cols is not None and len(cols) and rows is not None and len(rows):
            m = torch.zeros(T, T, dtype=torch.bool, device="cuda")
            m[rows.unsqueeze(1), cols.unsqueeze(0)] = True
            KO.mask = m

    def clear(self):
        KO.mask, KO.layers, KO.heads, KO.rows, KO.rec_img, KO.rec_spans = None, None, None, None, None, None
        KO.att = {}

    @torch.no_grad()
    def reduced_run(self, inputs: dict, keep: list[int]):
        """Forward on the kept token positions only, with their original (m)RoPE position ids (no image)."""
        if self.molmo:
            ix = torch.tensor(keep, device="cuda")
            ids = inputs["input_ids"][:, ix]
            out = self.model(input_ids=ids, position_ids=ix.unsqueeze(0), attention_mask=torch.ones_like(ids), last_logits_only=True)
            lp = torch.log_softmax(out.logits[0, -1].float(), -1)
            return [float(lp[t]) for t in self.letters]
        mm = self.model.model
        pos, _ = mm.get_rope_index(input_ids=inputs["input_ids"], image_grid_thw=inputs.get("image_grid_thw"),
                                   attention_mask=inputs.get("attention_mask"))
        ix = torch.tensor(keep, device="cuda")
        ids = inputs["input_ids"][:, ix]
        out = self.model(input_ids=ids, position_ids=pos[:, :, ix], attention_mask=torch.ones_like(ids), logits_to_keep=1)
        lp = torch.log_softmax(out.logits[0, -1].float(), -1)
        return [float(lp[t]) for t in self.letters]


def pg2(ab: list[float], gold: str) -> float:
    m = max(ab)
    e = [math.exp(v - m) for v in ab]
    return e["AB".index(gold)] / sum(e)


def post_image(idx: dict, T: int) -> torch.Tensor:
    """Every query position after the last image token (question, rationale, answer rows)."""
    return torch.arange(int(idx["img"].max()) + 1, T, device="cuda")


def set_c_afull(R, T: int, rows: torch.Tensor, idx: dict) -> None:
    """Conclusion blocked from the answer rows + image blocked from every position after it."""
    R.set_ko(T, post_image(idx, T), idx["img"])
    KO.mask[rows.unsqueeze(1), idx["conc"].unsqueeze(0)] = True


def head_dict(pairs: list[list[int]]) -> dict:
    d = defaultdict(list)
    for l, h in pairs:
        d[l].append(h)
    return {l: torch.tensor(v, dtype=torch.long, device="cuda") for l, v in d.items()}


# ============================================================================= stages

def stage_calib(args) -> None:
    os.makedirs(OUTD, exist_ok=True)
    smp = make_sample(args.model)
    json.dump(smp, open(f"{OUTD}/sample_{args.model}.json", "w"))
    print(f"[e52] sample {args.model}: calib {len(smp['calib'])} eval {len(smp['eval'])} n_cell {smp['n_cell']}", flush=True)
    R = Runner(args.model)
    cal = smp["calib"][: args.limit]
    att, H, meta = [], [], []
    for i, it in enumerate(cal):
        img = C.get_image(it)
        text = R.render(C.build_messages(it, "direct", "none", "", vx.img_ref(img)))
        a0 = text.rfind(C.FORCE_PREFIX)
        inputs, idx, T = R.encode(text, img, {"ans": (a0, len(text))})
        R.clear()
        KO.rec_img = idx["img"]
        ab, h = R.run(inputs, hidden=True)
        att.append(torch.stack([KO.att["img"][l] for l in range(R.n_layers)], 0).numpy())
        H.append(h)
        meta.append({"key": it["key"], "set": it["set"], "pair": it["pair"], "y_true": it["y_true"], "gold": it["gold"],
                     "p_direct_hf": pg2(ab, it["gold"]), "p_direct_stored": it["p_direct_stored"]})
        R.clear()
        if i % 50 == 0:
            print(f"[e52] calib {i}/{len(cal)}", flush=True)
    att = np.stack(att, 0)  # [N, L, H]
    score = att.mean(0)
    k = int(round(0.05 * score.size))
    order = np.argsort(-score.ravel())
    vis = [[int(i // R.n_heads), int(i % R.n_heads)] for i in order[:k]]
    vis_set = {tuple(v) for v in vis}
    per_layer = defaultdict(int)
    for l, _ in vis:
        per_layer[l] += 1
    rand = {}
    for s in (SEED, SEED + 1):
        rng = random.Random(f"rand-{s}-{args.model}")
        sel = []
        for l, n in sorted(per_layer.items()):
            pool = [h for h in range(R.n_heads) if (l, h) not in vis_set]
            sel += [[l, h] for h in rng.sample(pool, n)]
        rand[str(s)] = sel
    np.savez_compressed(f"{OUTD}/calib_{args.model}.npz", att=att.astype(np.float16), h=np.stack(H, 0))
    json.dump({"visual": vis, "random": rand, "k": k, "n_heads_total": int(score.size), "n_layers": R.n_layers,
               "n_heads": R.n_heads, "score_visual_mean": float(score.ravel()[order[:k]].mean()),
               "score_all_mean": float(score.mean()), "per_layer": dict(per_layer), "meta": meta},
              open(f"{OUTD}/heads_{args.model}.json", "w"))
    print(f"[e52] heads: k={k}, layers {dict(sorted(per_layer.items()))}", flush=True)


def stage_main(args) -> None:
    smp = json.load(open(f"{OUTD}/sample_{args.model}.json"))
    hd = json.load(open(f"{OUTD}/heads_{args.model}.json"))
    items = smp["eval"][args.shard::args.nshard][: args.limit]
    R = Runner(args.model)
    L = R.n_layers
    lo, hi = set(range(L // 2)), set(range(L // 2, L))
    heads = {"d": head_dict(hd["visual"]), "r1": head_dict(hd["random"][str(SEED)]), "r2": head_dict(hd["random"][str(SEED + 1)])}
    sfx = f"_{args.shard}" + (f"_limit{args.limit}" if args.limit else "")
    fo = open(f"{OUTD}/{args.model}{sfx}.jsonl", "w")
    Hd, Hc, Hf, Ad, Ac, Af = [], [], [], [], [], []
    t0 = time.time()
    for i, it in enumerate(items):
        img = C.get_image(it)
        ref = vx.img_ref(img)
        rec = {k: it[k] for k in ("key", "set", "pair", "cell", "gold", "y_true", "written", "p_direct_stored", "p_cot_stored", "p_filler_stored")}
        rec["has_rel"] = it["rel"] is not None
        # ---------------- Direct
        text = R.render(C.build_messages(it, "direct", "none", "", ref))
        a0 = text.rfind(C.FORCE_PREFIX)
        inputs, idx, T = R.encode(text, img, {"ans": (a0, len(text))})
        rows = idx["ans"]
        R.clear()
        KO.rec_img = idx["img"]
        ab, h = R.run(inputs, hidden=True)
        Ad.append(torch.stack([KO.att["img"][l] for l in range(L)], 0).numpy())
        Hd.append(h)
        dres = {"none": ab}
        for cnd in DIR_CONDS[1:]:
            R.clear()
            if cnd == "a":
                R.set_ko(T, rows, idx["img"])
            elif cnd == "a_full":
                R.set_ko(T, post_image(idx, T), idx["img"])
            else:
                R.set_ko(T, rows, heads=heads[cnd])
            dres[cnd], _ = R.run(inputs)
        rec["direct"] = dres
        # ---------------- CoT
        cot = it["cot"]
        text = R.render(C.build_messages(it, "cot", "none", cot, ref))
        cs = text.rfind(cot + "\n" + C.FORCE_PREFIX)
        assert cs > 0
        ce = cs + len(cot)
        spans = {"ans": (ce, len(text)), "turn": (cs, len(text))}
        if it["rel"]:
            b0, c0, _ = it["rel"]
            spans["body"] = (cs + b0, cs + c0)
            spans["conc"] = (cs + c0, ce)
        inputs, idx, T = R.encode(text, img, spans)
        rows = idx["ans"]
        rec["n_tok"] = {k: int(len(v)) for k, v in idx.items()} | {"T": T}
        R.clear()
        KO.rec_img = idx["img"]
        KO.rec_spans = {k: idx[k] for k in ("body", "conc") if k in idx}
        ab, h = R.run(inputs, hidden=True)
        Ac.append(torch.stack([KO.att["img"][l] for l in range(L)], 0).numpy())
        rec["att_cot"] = {k: [KO.att[k][l] for l in range(L)] for k in ("body", "conc") if k in KO.att}
        Hc.append(h)
        cres = {"none": ab}
        for cnd in COT_CONDS[1:]:
            R.clear()
            if cnd in ("b", "c", "ac", "c_lo", "c_hi", "c_afull") and not it["rel"]:
                continue
            if cnd == "a":
                R.set_ko(T, rows, idx["img"])
            elif cnd == "b":
                R.set_ko(T, rows, idx["body"])
            elif cnd == "c":
                R.set_ko(T, rows, idx["conc"])
            elif cnd == "ac":
                R.set_ko(T, rows, torch.cat([idx["img"], idx["conc"]]))
            elif cnd == "a_all":
                R.set_ko(T, idx["turn"], idx["img"])
            elif cnd == "a_full":
                R.set_ko(T, post_image(idx, T), idx["img"])
            elif cnd == "c_afull":
                set_c_afull(R, T, rows, idx)
            elif cnd in ("d", "r1", "r2"):
                R.set_ko(T, rows, heads=heads[cnd])
            elif cnd in ("a_lo", "a_hi"):
                R.set_ko(T, rows, idx["img"], layers=lo if cnd.endswith("lo") else hi)
            elif cnd in ("c_lo", "c_hi"):
                R.set_ko(T, rows, idx["conc"], layers=lo if cnd.endswith("lo") else hi)
            cres[cnd], _ = R.run(inputs)
        rec["cot"] = cres
        # ---------------- sanity S2 / S3 on the first 30 prompts of the worker
        if i < 30:
            R.clear()
            R.set_ko(T, rows, torch.zeros(0, dtype=torch.long, device="cuda"))
            KO.heads = {}
            rec["s2_identity"], _ = R.run(inputs)
            R.clear()
            rec["s2_sdpa"] = R.run_sdpa(inputs)
            R.clear()
            allk = torch.arange(1, T, device="cuda")
            keepset = set(rows.tolist())
            cols = torch.tensor([k for k in allk.tolist() if k not in keepset], device="cuda")
            R.set_ko(T, rows, cols)
            rec["s3_bos_ko"], _ = R.run(inputs)
            R.clear()
            rec["s3_reduced"] = R.reduced_run(inputs, [0] + rows.tolist())
        # ---------------- filler_R+turn
        R.clear()
        ftxt = C.filler_text(R.tok, max(1, it["gen_len"]))
        text = R.render(C.build_messages(it, "filler", "turn", ftxt, ref))
        fs = text.rfind(ftxt)
        a0 = text.rfind(C.FORCE_PREFIX)
        inputs, idx, T = R.encode(text, img, {"ans": (a0, len(text)), "fill": (fs, fs + len(ftxt))})
        KO.rec_img = idx["img"]
        KO.rec_spans = {"fill": idx["fill"]}
        ab, h = R.run(inputs, hidden=True)
        Af.append(torch.stack([KO.att["img"][l] for l in range(L)], 0).numpy())
        rec["att_fill"] = [KO.att["fill"][l] for l in range(L)]
        Hf.append(h)
        rec["filler"] = {"none": ab}
        R.clear()
        fo.write(json.dumps(rec) + "\n")
        fo.flush()
        if i % 20 == 0:
            print(f"[e52] {args.model} shard {args.shard}: {i}/{len(items)} ({time.time() - t0:.0f}s)", flush=True)
    fo.close()
    np.savez_compressed(f"{OUTD}/h_{args.model}{sfx}.npz", direct=np.stack(Hd), cot=np.stack(Hc), filler=np.stack(Hf),
                        keys=np.array([it["key"] for it in items]))
    np.savez_compressed(f"{OUTD}/att_{args.model}{sfx}.npz", direct=np.stack(Ad).astype(np.float16),
                        cot=np.stack(Ac).astype(np.float16), filler=np.stack(Af).astype(np.float16))
    print(f"[e52] done {len(items)} ({time.time() - t0:.0f}s)", flush=True)


def stage_extra(args) -> None:
    """Addendum 2: c_afull for runs made before it was added (CoT none recomputed as a check)."""
    smp = json.load(open(f"{OUTD}/sample_{args.model}.json"))
    items = smp["eval"][args.shard::args.nshard][: args.limit]
    R = Runner(args.model)
    sfx = f"_{args.shard}" + (f"_limit{args.limit}" if args.limit else "")
    t0 = time.time()
    with open(f"{OUTD}/{args.model}_x{sfx}.jsonl", "w") as fo:
        for i, it in enumerate(items):
            if not it["rel"]:
                continue
            img = C.get_image(it)
            cot = it["cot"]
            text = R.render(C.build_messages(it, "cot", "none", cot, vx.img_ref(img)))
            cs = text.rfind(cot + "\n" + C.FORCE_PREFIX)
            ce = cs + len(cot)
            b0, c0, _ = it["rel"]
            inputs, idx, T = R.encode(text, img, {"ans": (ce, len(text)), "conc": (cs + c0, ce)})
            R.clear()
            none, _ = R.run(inputs)
            set_c_afull(R, T, idx["ans"], idx)
            ca, _ = R.run(inputs)
            R.clear()
            fo.write(json.dumps({"key": it["key"], "none": none, "c_afull": ca}) + "\n")
            if i % 50 == 0:
                print(f"[e52x] {i}/{len(items)} ({time.time() - t0:.0f}s)", flush=True)
    print(f"[e52x] done ({time.time() - t0:.0f}s)", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--stage", choices=("calib", "main", "sample", "extra"), required=True)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshard", type=int, default=1)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()
    if args.stage == "sample":
        smp = make_sample(args.model)
        print({k: smp[k] for k in ("N_cell", "n_cell")}, len(smp["calib"]), len(smp["eval"]),
              sum(r["rel"] is not None for r in smp["eval"]))
        return
    {"calib": stage_calib, "main": stage_main, "extra": stage_extra}[args.stage](args)


if __name__ == "__main__":
    main()
