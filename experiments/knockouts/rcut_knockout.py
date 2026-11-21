"""E59 (GPU): Exp-52 attention-edge knockouts on R_cut (rationale with its concluding sentence removed) for all seven
open instruct models, plus main-pool 2D left/right items for the InternVL checkers and Qwen2.5-VL-7B.

Arms per prompt (teacher-forced, one forward pass per condition; patched eager attention with fp32 scores, Exp-52):
  direct: none, a_full                      (a_full = image blocked from every query position after the image)
  cut:    none, a_full, a, rel              (a = answer rows -> image; rel = answer rows -> last remaining relational
                                             sentence of R_cut, through the end of R_cut)
  full:   none, a_full, c, c_afull          (occlusion only; Exp-52 conclusion span; reproduction of Exp-52)
  e3x:    R_cut + "Therefore, the answer is (Y) <option>." for Y = gold / wrong (2D left/right; reproduces e3x)
Sanity on the first N_SAN prompts of each worker: empty knockout (identity) and stock SDPA where switchable.
Pre-registration: outputs/e59/prereg.txt.

Stages:
  python -m experiments.knockouts.rcut_knockout --stage sample --model qwen7b          # CPU (dev machine): samples -> outputs/e59/
  python -m experiments.knockouts.rcut_knockout --stage main --model qwen7b --set occ --shard 0 --nshard 2
  python -m experiments.knockouts.rcut_knockout --stage main --model internvl3_8b --set lr
Env: E59_DEV=cpu for a CPU dry run; E59_OUT overrides the output directory.
"""
from __future__ import annotations
from srww import paths as _vtp  # noqa: E402

import argparse
import json
import math
import os
import random
import re
import sys
import time
from collections import defaultdict

import numpy as np
import torch

from srww import vllm_utils as vx
from srww.vllm_utils import C
from experiments.knockouts import attention_knockout as E52
from experiments.knockouts.attention_knockout import KO, ko_core
from srww.claim_parse import split_sents

from srww.rcut import strip_answer  # noqa: E402

OUTD = os.environ.get("E59_OUT", f"{vx.OUT}/e59")
DEV = os.environ.get("E59_DEV", "cuda")
DTYPE = torch.bfloat16
SEED_LR = 59
N_SAN = 20
LR_MODELS = ("internvl3_8b", "internvl35_8b", "qwen7b")
ALLOC = E52.ALLOC
LRX = re.compile(r"\b(left|right)\b", re.I)


# ============================================================================= samples (CPU)

def lr_relation(cut: str) -> int | None:
    """Char start of the last sentence of R_cut (after the CoT prefix) that mentions left / right."""
    pre = C.COT_PREFIX
    body = cut[len(pre):] if cut.startswith(pre) else cut
    off0 = len(cut) - len(body)
    sents = split_sents(body)
    pos, starts = off0, []
    for s in sents:
        p = cut.find(s, pos)
        starts.append(p)
        pos = p + len(s)
    for j in range(len(sents) - 1, -1, -1):
        if LRX.search(sents[j]):
            return starts[j]
    return None


def stage_sample(model: str) -> None:
    os.makedirs(OUTD, exist_ok=True)
    smp = E52.make_sample(model)
    old = f"{vx.OUT}/e52/sample_{model}.json"
    if os.path.exists(old):
        o = json.load(open(old))
        assert [r["key"] for r in o["eval"]] == [r["key"] for r in smp["eval"]], "sample differs from Exp-52"
        print(f"[e59] {model}: sample identical to Exp-52", flush=True)
    stored14 = {r["item_id"]: r for r in vx.read_jsonl(f"{vx.OUT}/e14/{model}.jsonl") if r["ctx"] == "cut" and r["img"] == "own"}
    for r in smp["eval"]:
        cut = strip_answer(r["cot"])
        r["cut"] = cut
        r["rel_cut"] = E52.last_relation(cut, r["x"], r["y"])
        r["written_cut"] = E52.written_letter(r["question"], r["rel_cut"][2], r["x"], r["y"]) if r["rel_cut"] else None
        s14 = stored14.get(r["key"]) if r["set"] == "twins" else None
        r["e14_cut_lp"] = (s14.get("lp_sum") or s14["top20"]) if s14 else None
    smp.pop("calib")
    json.dump(smp, open(f"{OUTD}/sample_{model}.json", "w"))
    ev = smp["eval"]
    print(f"[e59] {model}: occ {len(ev)}; rel full {sum(r['rel'] is not None for r in ev)}, rel cut "
          f"{sum(r['rel_cut'] is not None for r in ev)}; cut == prefix only {sum(r['cut'].strip() == C.COT_PREFIX for r in ev)};"
          f" mean chars full {np.mean([len(r['cot']) for r in ev]):.0f} cut {np.mean([len(r['cut']) for r in ev]):.0f};"
          f" e14 stored {sum(r['e14_cut_lp'] is not None for r in ev)}", flush=True)
    if model in LR_MODELS:
        stage_sample_lr(model)


def load_main(model: str) -> dict:
    """{condition: {uid: (correct, p_gold)}} from the Tab. 1 records (srww.records.load_main sources)."""
    out = defaultdict(dict)
    recs = vx.read_jsonl(f"{vx.OUT}/e1x/{model}.jsonl")
    if recs:
        for r in recs:
            lp = r.get("lp_sum") or r["top20"]
            out[r["condition"]][r["item_id"]] = (float(E52.pred(lp) == r["gold"]), E52.p_gold(lp, r["gold"]))
        return out
    for it in vx.legacy_e1(model).values():
        for c, sc in it["scores"].items():
            out[c][it["uid"]] = (float(E52.pred(sc) == it["gold"]), E52.p_gold(sc, it["gold"]))
    return out


def stage_sample_lr(model: str) -> None:
    main = load_main(model)
    leg = vx.legacy_e1(model)
    e3 = defaultdict(dict)
    meta = {}
    for r in vx.read_jsonl(f"{vx.OUT}/e3x/{model}.jsonl"):
        if r["context"] == "rcut":
            e3[(r["item_id"], r["image"])][r["Y"]] = r
            meta[r["item_id"]] = r
    lr = sorted({u for (u, im) in e3 if im == "mirror"})
    rows = []
    for u in lr:
        m0 = meta[u]
        assert m0["task"] in ("cvb_relation", "syn_relpos"), m0["task"]
        d, c = main["direct"].get(u), main["cot"].get(u)
        if d is None or c is None:
            continue
        cot = leg[u]["cot"]
        cut = strip_answer(cot)
        st = e3[(u, "original")]
        rows.append({"key": u, "set": "lr", "pair": m0["image_id"], "task": m0["task"], "gold": m0["gold"],
                     "letters": m0["letters"], "cell": ("D+" if d[0] else "D-") + ("C+" if c[0] else "C-"),
                     "p_direct_stored": d[1], "p_cot_stored": c[1], "question": leg[u]["question"], "cot": cot, "cut": cut,
                     "rel_cut": lr_relation(cut), "e3x_lp": {y: (v.get("lp_sum") or v["top20"]) for y, v in st.items()}})
    rng = random.Random(f"{SEED_LR}-{model}-lr")
    bycell = defaultdict(list)
    for r in rows:
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
    res = {"eval": ev, "N_cell": {"lr": {c: len(bycell[c]) for c, _ in ALLOC}}, "n_cell": {"lr": take}}
    json.dump(res, open(f"{OUTD}/lr_{model}.json", "w"))
    print(f"[e59] {model} LR: pool {len(rows)} N_cell {res['N_cell']['lr']} n_cell {take}; rel cut "
          f"{sum(r['rel_cut'] is not None for r in ev)}/{len(ev)}; tasks "
          f"{ {t: sum(r['task'] == t for r in ev) for t in ('cvb_relation', 'syn_relpos')} }", flush=True)


# ============================================================================= patched attention (all families)

CALLS = [0]
ORIG = {}


def ko_eager(module, query, key, value, attention_mask, scaling, dropout=0.0, **kwargs):
    """Patched transformers eager_attention_forward (Qwen2.5-VL / Qwen3-VL / Qwen2 / Qwen3 decoders)."""
    li = getattr(module, "_e59", None)
    if li is None:
        return ORIG["fn"](module, query, key, value, attention_mask, scaling, dropout, **kwargs)
    CALLS[0] += 1
    g = module.num_key_value_groups
    k = key.repeat_interleave(g, dim=1) if g > 1 else key
    v = value.repeat_interleave(g, dim=1) if g > 1 else value
    o, w = ko_core(li, query, k, v, attention_mask, scaling)
    return o.transpose(1, 2).contiguous(), w


def ko_molmo(self, q, k, v, attn_mask=None, dropout_p=0.0, response_dropout_p=0.0, is_causal=False):
    if KO.stock:
        return ORIG["molmo"](self, q, k, v, attn_mask, dropout_p, response_dropout_p, is_causal)
    CALLS[0] += 1
    rep_ = q.size(1) // k.size(1)
    if rep_ > 1:
        k = k.repeat_interleave(rep_, dim=1)
        v = v.repeat_interleave(rep_, dim=1)
    o, _ = ko_core(self.layer_id, q, k, v, attn_mask, 1.0 / math.sqrt(q.size(-1)))
    return o


def make_phi_forward(mod_globals: dict):
    """Replacement for Phi3Attention.forward (Phi-3.5-Vision remote code), no KV cache."""
    apply_rotary_pos_emb = mod_globals["apply_rotary_pos_emb"]

    def forward(self, hidden_states, attention_mask=None, position_ids=None, past_key_value=None,
                output_attentions=False, use_cache=False):
        if KO.stock:
            return ORIG["phi"](self, hidden_states, attention_mask, position_ids, past_key_value, output_attentions, use_cache)
        CALLS[0] += 1
        bsz, q_len, _ = hidden_states.size()
        qkv = self.qkv_proj(hidden_states)
        qp = self.num_heads * self.head_dim
        kvp = self.num_key_value_heads * self.head_dim
        q = qkv[..., :qp].view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = qkv[..., qp:qp + kvp].view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        v = qkv[..., qp + kvp:].view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        cos, sin = self.rotary_emb(v, position_ids, seq_len=q_len)
        q, k = apply_rotary_pos_emb(q, k, cos, sin, position_ids)
        if self.num_key_value_groups > 1:
            k = k.repeat_interleave(self.num_key_value_groups, dim=1)
            v = v.repeat_interleave(self.num_key_value_groups, dim=1)
        o, _ = ko_core(self.layer_idx, q, k, v, attention_mask, 1.0 / math.sqrt(self.head_dim))
        o = o.transpose(1, 2).contiguous().reshape(bsz, q_len, self.hidden_size)
        return self.o_proj(o), None, past_key_value

    return forward


def install_patch(model, name: str) -> int:
    if name == "molmo7b":
        blocks = [m for m in model.modules() if hasattr(m, "layer_id") and hasattr(m, "_scaled_dot_product_attention")]
        cls = type(blocks[0])
        ORIG["molmo"] = cls._scaled_dot_product_attention
        cls._scaled_dot_product_attention = ko_molmo
        return len(blocks)
    if name == "phi35v":
        attn = [m for m in model.modules() if type(m).__name__ == "Phi3Attention"]
        cls = type(attn[0])
        ORIG["phi"] = cls.forward
        cls.forward = make_phi_forward(sys.modules[cls.__module__].__dict__)
        return len(attn)
    if name.startswith("qwen3vl"):
        import transformers.models.qwen3_vl.modeling_qwen3_vl as M
    elif name.startswith("qwen"):
        import transformers.models.qwen2_5_vl.modeling_qwen2_5_vl as M
    elif name == "internvl3_8b":
        import transformers.models.qwen2.modeling_qwen2 as M
    elif name == "internvl35_8b":
        import transformers.models.qwen3.modeling_qwen3 as M
    else:
        raise ValueError(name)
    ORIG["fn"] = M.eager_attention_forward
    M.eager_attention_forward = ko_eager
    n = 0
    for nm, mod in model.named_modules():
        if nm.endswith("self_attn") and "language_model" in nm and hasattr(mod, "layer_idx"):
            mod._e59 = mod.layer_idx
            n += 1
    return n


# ============================================================================= InternVL image preprocessing (as vLLM)

def _closest_ratio(aspect, ratios, w, h, size):
    best_diff, best = float("inf"), (1, 1)
    area = w * h
    for r in ratios:
        diff = abs(aspect - r[0] / r[1])
        if diff < best_diff:
            best_diff, best = diff, r
        elif diff == best_diff and area > 0.5 * size * size * r[0] * r[1]:
            best = r
    return best


def internvl_pixels(img, max_num: int = 6, size: int = 448) -> torch.Tensor:
    import torchvision.transforms as T
    from torchvision.transforms.functional import InterpolationMode
    w, h = img.size
    ratios = sorted({(i, j) for n in range(1, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1)
                     if 1 <= i * j <= max_num}, key=lambda x: x[0] * x[1])
    r = _closest_ratio(w / h, ratios, w, h, size)
    tw, th = size * r[0], size * r[1]
    rs = img.resize((tw, th))
    tiles = [rs.crop(((i % (tw // size)) * size, (i // (tw // size)) * size, ((i % (tw // size)) + 1) * size,
                      ((i // (tw // size)) + 1) * size)) for i in range(r[0] * r[1])]
    if len(tiles) != 1:
        tiles.append(img.resize((size, size)))
    tf = T.Compose([T.Lambda(lambda x: x.convert("RGB")), T.Resize((size, size), interpolation=InterpolationMode.BICUBIC),
                    T.ToTensor(), T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
    return torch.stack([tf(t) for t in tiles])


# ============================================================================= runner

def model_path(name: str) -> str:
    """C.model_path, skipping /dev/shm copies without weights (InternVL3-8B)."""
    import glob
    p = C.model_path(name)
    shards = sorted(glob.glob(f"{p}/*.safetensors"))
    if len(shards) > 1 and not glob.glob(f"{p}/*.safetensors.index.json"):
        # sharded checkpoint without an index (vLLM globs shards; HF needs the index): symlinked copy + index in /tmp
        from safetensors import safe_open
        q = f"/tmp/e59_models/{name}"
        os.makedirs(q, exist_ok=True)
        for f in os.listdir(p):
            if not os.path.exists(f"{q}/{f}"):
                os.symlink(f"{p}/{f}", f"{q}/{f}")
        wm = {}
        for s in shards:
            with safe_open(s, "pt") as fh:
                wm.update({k: os.path.basename(s) for k in fh.keys()})
        json.dump({"metadata": {}, "weight_map": wm}, open(f"{q}/model.safetensors.index.json", "w"))
        p = q
    return p


def load_model(name: str):
    path = model_path(name)
    print(f"[e59] weights: {path}", flush=True)
    if name == "molmo7b":
        from transformers import AutoModelForCausalLM, AutoProcessor
        proc = AutoProcessor.from_pretrained(path, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(path, trust_remote_code=True, torch_dtype=DTYPE).to(DEV).eval()
        return model, proc, proc.tokenizer
    if name.startswith("internvl"):
        from transformers import AutoModel, AutoTokenizer
        tok = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
        model = AutoModel.from_pretrained(path, torch_dtype=DTYPE, trust_remote_code=True, use_flash_attn=False,
                                          low_cpu_mem_usage=True).to(DEV).eval()
        return model, None, tok
    if name == "phi35v":
        from transformers import AutoModelForCausalLM, AutoProcessor
        proc = AutoProcessor.from_pretrained(path, trust_remote_code=True, num_crops=4)
        model = AutoModelForCausalLM.from_pretrained(path, trust_remote_code=True, torch_dtype=DTYPE,
                                                     _attn_implementation="eager").to(DEV).eval()
        return model, proc, proc.tokenizer
    from transformers import AutoProcessor, Qwen2VLImageProcessor
    if name.startswith("qwen3vl"):
        from transformers import Qwen3VLForConditionalGeneration as Mdl
        proc = AutoProcessor.from_pretrained(path)
        proc.image_processor = Qwen2VLImageProcessor.from_pretrained(path)
    else:
        from transformers import Qwen2_5_VLForConditionalGeneration as Mdl
        proc = AutoProcessor.from_pretrained(path, max_pixels=1280 * 28 * 28)
        proc.image_processor = Qwen2VLImageProcessor.from_pretrained(path, max_pixels=1280 * 28 * 28)
    model = Mdl.from_pretrained(path, torch_dtype=DTYPE,
                                attn_implementation={"text_config": "eager", "vision_config": "sdpa", "": "eager"}).to(DEV).eval()
    return model, proc, proc.tokenizer


class Runner:
    def __init__(self, name: str):
        self.name = name
        self.model, self.proc, self.tok = load_model(name)
        self.n_layers = install_patch(self.model, name)
        self.fam = ("molmo" if name == "molmo7b" else "internvl" if name.startswith("internvl") else
                    "phi" if name == "phi35v" else "qwen")
        if self.fam == "qwen":
            assert self.n_layers == self.model.config.text_config.num_hidden_layers
        elif self.fam == "internvl":
            assert self.n_layers == self.model.config.llm_config.num_hidden_layers
            self.ctx_id = self.tok.convert_tokens_to_ids("<IMG_CONTEXT>")
            self.model.img_context_token_id = self.ctx_id
            self.n_img_tok = self.model.num_image_token
        else:
            assert self.n_layers == self.model.config.num_hidden_layers, (self.n_layers, self.model.config.num_hidden_layers)
        self.letters = [self.tok.convert_tokens_to_ids(c) for c in ("A", "B")]
        assert all(isinstance(t, int) and t >= 0 for t in self.letters), self.letters
        self._vit = (None, None)
        print(f"[e59] {name}: {self.n_layers} patched layers, letters {self.letters}", flush=True)

    # ---------------------------------------------------------------- prompt
    def render(self, msgs: list[dict]) -> tuple[str, str]:
        """(prompt text, image marker string whose tail after the last occurrence is plain text)."""
        if self.fam == "molmo":
            from jinja2 import Template
            return " " + Template(vx.MOLMO_TEMPLATE).render(messages=msgs), ""
        if self.fam == "phi":
            pm = []
            for m in msgs:
                if isinstance(m["content"], list):
                    txt = "\n".join(p["text"] for p in m["content"] if p.get("type") == "text")
                    has_img = any(p.get("type") == "image_url" for p in m["content"])
                    pm.append({"role": m["role"], "content": ("<|image_1|>\n" if has_img else "") + txt})
                else:
                    pm.append(m)
            return self.tok.apply_chat_template(pm, tokenize=False, add_generation_prompt=False, continue_final_message=True), "<|image_1|>"
        hm = [{"role": m["role"], "content": [({"type": "image"} if p.get("type") == "image_url" else p) for p in m["content"]]
               if isinstance(m["content"], list) else m["content"]} for m in msgs]
        if self.fam == "internvl":
            t = self.tok.apply_chat_template(hm, tokenize=False, add_generation_prompt=False, continue_final_message=True)
            assert t.count("<image>") == 1, t[:300]
            return t, "<image>"
        return self.proc.apply_chat_template(hm, tokenize=False, add_generation_prompt=False, continue_final_message=True), "<|image_pad|>"

    def encode(self, text: str, marker: str, img, spans: dict[str, tuple[int, int]]):
        """Model inputs, token index sets for char spans (token assigned by its first char) and the image tokens."""
        if self.fam == "molmo":
            enc = self.tok(text, return_offsets_mapping=True, add_special_tokens=False)
            out = self.proc.process(images=[img], tokens=enc["input_ids"])
            full = out["input_ids"].tolist()
            inputs = {k: v.unsqueeze(0).to(DEV) for k, v in out.items()}
            inputs["images"] = inputs["images"].to(DTYPE)
            img_idx = [i for i, t in enumerate(full) if t == self.tok.convert_tokens_to_ids("<im_patch>")]
            tail0, tail_special = 0, False
        elif self.fam == "phi":
            inputs = self.proc(text, [img], return_tensors="pt")
            full = inputs["input_ids"][0].tolist()
            inputs = {k: v.to(DEV) for k, v in inputs.items()}
            inputs["pixel_values"] = inputs["pixel_values"].to(DTYPE)
            img_idx = [i for i, t in enumerate(full) if t < 0]
            tail0, tail_special = text.rfind(marker) + len(marker), True
        elif self.fam == "internvl":
            pv = internvl_pixels(img)
            exp = "<img>" + "<IMG_CONTEXT>" * (self.n_img_tok * pv.shape[0]) + "</img>"
            a = text.rfind(marker)
            text_x = text[:a] + exp + text[a + len(marker):]
            shift = len(exp) - len(marker)
            spans = {k: (s + shift, e + shift) for k, (s, e) in spans.items()}
            text = text_x
            full = self.tok(text, add_special_tokens=False)["input_ids"]
            with torch.no_grad():
                vit = self.model.extract_feature(pv.to(DEV, DTYPE))
            inputs = {"input_ids": torch.tensor([full], device=DEV), "vit": vit}
            img_idx = [i for i, t in enumerate(full) if t == self.ctx_id]
            assert len(img_idx) == self.n_img_tok * pv.shape[0]
            tail0, tail_special = a + len(exp), False
        else:
            inputs = self.proc(text=[text], images=[img], return_tensors="pt")
            full = inputs["input_ids"][0].tolist()
            inputs = {k: v.to(DEV) for k, v in inputs.items()}
            pad = self.tok.convert_tokens_to_ids("<|image_pad|>")
            img_idx = [i for i, t in enumerate(full) if t == pad]
            tail0, tail_special = text.rfind(marker) + len(marker), False
        tail = text[tail0:]
        enc = self.tok(tail, return_offsets_mapping=True, add_special_tokens=tail_special)
        tids, offs = enc["input_ids"], enc["offset_mapping"]
        L = 0
        while L < len(tids) and L < len(full) and tids[-1 - L] == full[-1 - L]:
            L += 1
        n0 = len(tids) - L  # leading tail tokens that differ (boundary effects); spans must not touch them
        base = len(full) - len(tids)
        idx = {}
        for nm, (s, e) in spans.items():
            ks = [k for k, (a_, _) in enumerate(offs) if s - tail0 <= a_ < e - tail0]
            assert all(k >= n0 for k in ks), (nm, n0)
            idx[nm] = ks
        idx = {k: torch.tensor([base + j for j in v], dtype=torch.long, device=DEV) for k, v in idx.items()}
        idx["img"] = torch.tensor(img_idx, dtype=torch.long, device=DEV)
        assert len(img_idx) > 0 and max(img_idx) < len(full) - L, "image tokens must precede the matched text tail"
        self.last_tail = {"n_mismatch_lead": n0, "n_tail": len(tids)}
        return inputs, idx, len(full)

    # ---------------------------------------------------------------- forward
    @torch.no_grad()
    def run(self, inputs: dict) -> list[float]:
        CALLS[0] = 0
        if self.fam == "molmo":
            out = self.model(**inputs, last_logits_only=True)
            logits = out.logits[0, -1]
        elif self.fam == "qwen":
            logits = self.model(**inputs, logits_to_keep=1).logits[0, -1]
        elif self.fam == "internvl":
            vit = inputs["vit"]
            ids = inputs["input_ids"]
            lm = self.model.language_model
            emb = lm.get_input_embeddings()(ids).clone()
            sel = ids[0] == self.ctx_id
            emb[0, sel] = vit.reshape(-1, vit.shape[-1]).to(emb.dtype)
            logits = lm(inputs_embeds=emb, use_cache=False, logits_to_keep=1).logits[0, -1]
        else:
            h = self.model.model(input_ids=inputs["input_ids"], pixel_values=inputs["pixel_values"],
                                 image_sizes=inputs["image_sizes"], use_cache=False).last_hidden_state
            logits = self.model.lm_head(h[:, -1])[0]
        if not KO.stock:
            assert CALLS[0] == self.n_layers, f"patched attention called {CALLS[0]}x, expected {self.n_layers}"
        lp = torch.log_softmax(logits.float(), -1)
        return [float(lp[t]) for t in self.letters]

    @torch.no_grad()
    def run_ref(self, inputs: dict) -> list[float] | None:
        """Stock attention reference: SDPA (Qwen / InternVL), original eager (Phi), original Molmo attention."""
        if self.fam in ("molmo", "phi"):
            KO.stock = True
            try:
                return self.run(inputs)
            finally:
                KO.stock = False
        cfg = self.model.config.text_config if self.fam == "qwen" else self.model.language_model.config
        old = cfg._attn_implementation
        cfg._attn_implementation = "sdpa"
        KO.stock = True
        try:
            return self.run(inputs)
        finally:
            cfg._attn_implementation = old
            KO.stock = False

    def set_ko(self, T: int, rows=None, cols=None):
        KO.layers, KO.heads, KO.rows, KO.rec_img, KO.rec_spans = None, None, rows, None, None
        KO.mask = None
        if cols is not None and len(cols) and rows is not None and len(rows):
            m = torch.zeros(T, T, dtype=torch.bool, device=DEV)
            m[rows.unsqueeze(1), cols.unsqueeze(0)] = True
            KO.mask = m

    def clear(self):
        KO.mask, KO.layers, KO.heads, KO.rows, KO.rec_img, KO.rec_spans = None, None, None, None, None, None
        KO.att = {}


KO.stock = False


def post_image(idx: dict, T: int) -> torch.Tensor:
    return torch.arange(int(idx["img"].max()) + 1, T, device=DEV)


def cond_mask(R: Runner, cnd: str, T: int, idx: dict) -> bool:
    """Set the knockout for a condition; False if the condition does not apply to this prompt."""
    R.clear()
    rows = idx["ans"]
    if cnd == "none":
        return True
    if cnd == "a_full":
        R.set_ko(T, post_image(idx, T), idx["img"])
    elif cnd == "a":
        R.set_ko(T, rows, idx["img"])
    elif cnd in ("rel", "c"):
        if "span" not in idx or not len(idx["span"]):
            return False
        R.set_ko(T, rows, idx["span"])
    elif cnd == "c_afull":
        if "span" not in idx or not len(idx["span"]):
            return False
        R.set_ko(T, post_image(idx, T), idx["img"])
        KO.mask[rows.unsqueeze(1), idx["span"].unsqueeze(0)] = True
    else:
        raise ValueError(cnd)
    return True


def item_of(r: dict, items: dict) -> dict:
    if r["set"] == "lr":
        it = items[r["key"]]
        assert it["question"] == r["question"] and it["gold"] == r["gold"], r["key"]
        return it
    p = r["image_path"]
    if not os.path.exists(p):
        p = p.replace("<DATA_ROOT>", f"{_vtp.DATA_ROOT}")
    return {"question": r["question"], "image_path": p, "gold": r["gold"], "choices": ["A", "B"]}


def ctx_pass(R: Runner, it: dict, img, ctx: str, span: tuple[int, int] | None, conds: tuple[str, ...], san: bool) -> dict:
    """Teacher-forced passes with assistant context `ctx` ("" = Direct); span = (a, b) chars within ctx."""
    msgs = C.build_messages(it, "cot" if ctx else "direct", "none", ctx, {"type": "image_url"})
    text, marker = R.render(msgs)
    if ctx:
        cs = text.rfind(ctx + "\n" + C.FORCE_PREFIX)
        assert cs > 0
        ce = cs + len(ctx)
    else:
        ce = text.rfind(C.FORCE_PREFIX)
        cs = ce
    spans = {"ans": (ce, len(text)), "ctx": (cs, ce)}
    if span:
        spans["span"] = (cs + span[0], cs + span[1])
    inputs, idx, T = R.encode(text, marker, img, spans)
    out = {}
    for cnd in conds:
        if cond_mask(R, cnd, T, idx):
            out[cnd] = R.run(inputs)
    R.clear()
    if san:
        R.set_ko(T, idx["ans"], torch.zeros(0, dtype=torch.long, device=DEV))
        out["s2_identity"] = R.run(inputs)
        R.clear()
        out["s2_ref"] = R.run_ref(inputs)
    out["_n"] = {k: int(len(v)) for k, v in idx.items()} | {"T": T, "lead_mismatch": R.last_tail["n_mismatch_lead"]}
    return out


def stage_conflict(args, R: Runner, rows: list[dict], items: dict) -> None:
    """Addendum 1: R_cut + appended gold / wrong conclusion under none / a_full / a (2D left/right)."""
    sfx = f"_conflict_{args.shard}" + (f"_limit{args.limit}" if args.limit else "")
    t0 = time.time()
    with open(f"{OUTD}/{args.model}{sfx}.jsonl", "w") as fo:
        for i, r in enumerate(rows):
            it = item_of(r, items)
            img = C.get_image(it)
            ow = vx.option_words(it)
            wrong = [c for c in it["choices"] if c != it["gold"]][0]
            rec = {k: r.get(k) for k in ("key", "set", "pair", "cell", "gold", "task")} | {"wrong": wrong, "conf": {}}
            for y in (it["gold"], wrong):
                ctx = f"{r['cut']} Therefore, the answer is ({y}) {ow.get(y, '')}.".replace(" .", ".")
                rec["conf"][y] = ctx_pass(R, it, img, ctx, None, ("none", "a_full", "a"), False)
            fo.write(json.dumps(rec) + "\n")
            fo.flush()
            if i % 50 == 0:
                print(f"[e59] {args.model} conflict shard {args.shard}: {i}/{len(rows)} ({time.time() - t0:.0f}s)", flush=True)
    print(f"[e59] done {args.model} conflict shard {args.shard}: {len(rows)} ({time.time() - t0:.0f}s)", flush=True)


def stage_main(args) -> None:
    fn = f"{OUTD}/{'sample' if args.set == 'occ' else 'lr'}_{args.model}.json"
    smp = json.load(open(fn))
    rows = smp["eval"][args.shard::args.nshard][: args.limit]
    items = {}
    if args.set in ("lr", "conflict"):
        keys = {r["key"] for r in rows}
        items = {it["uid"]: it for it in vx.load_items(["cvb_relation", "syn_relpos"]) if it["uid"] in keys}
    R = Runner(args.model)
    if args.set == "conflict":
        stage_conflict(args, R, rows, items)
        return
    sfx = f"_{args.set}_{args.shard}" + (f"_limit{args.limit}" if args.limit else "")
    fo = open(f"{OUTD}/{args.model}{sfx}.jsonl", "w")
    t0 = time.time()
    for i, r in enumerate(rows):
        it = item_of(r, items)
        img = C.get_image(it)
        san = i < N_SAN
        rec = {k: r.get(k) for k in ("key", "set", "pair", "cell", "gold", "y_true", "written", "written_cut", "task",
                                     "p_direct_stored", "p_cot_stored", "p_filler_stored", "e14_cut_lp", "e3x_lp")}
        rec["has_rel_cut"] = r["rel_cut"] is not None
        rec["direct"] = ctx_pass(R, it, img, "", None, ("none", "a_full"), san)
        cut = r["cut"]
        if args.set == "occ":
            sp = (r["rel_cut"][1], len(cut)) if r["rel_cut"] else None
        else:
            sp = (r["rel_cut"], len(cut)) if r["rel_cut"] is not None else None
        rec["cut"] = ctx_pass(R, it, img, cut, sp, ("none", "a_full", "a", "rel"), san)
        if args.set == "occ":
            rec["has_rel"] = r["rel"] is not None
            cot = r["cot"]
            spf = (r["rel"][1], len(cot)) if r["rel"] else None
            rec["full"] = ctx_pass(R, it, img, cot, spf, ("none", "a_full", "c", "c_afull"), san)
        else:
            ow = vx.option_words(it)
            wrong = [c for c in it["choices"] if c != it["gold"]][0]
            rec["e3x"] = {}
            for y in (it["gold"], wrong):
                ctx = f"{cut} Therefore, the answer is ({y}) {ow.get(y, '')}.".replace(" .", ".")
                rec["e3x"][y] = ctx_pass(R, it, img, ctx, None, ("none",), False)["none"]
        fo.write(json.dumps(rec) + "\n")
        fo.flush()
        if i % 20 == 0:
            print(f"[e59] {args.model} {args.set} shard {args.shard}: {i}/{len(rows)} ({time.time() - t0:.0f}s)", flush=True)
    fo.close()
    print(f"[e59] done {args.model} {args.set} shard {args.shard}: {len(rows)} ({time.time() - t0:.0f}s)", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--stage", choices=("sample", "main"), required=True)
    ap.add_argument("--set", choices=("occ", "lr", "conflict"), default="occ")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshard", type=int, default=1)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()
    if args.stage == "sample":
        stage_sample(args.model)
    else:
        os.makedirs(OUTD, exist_ok=True)
        stage_main(args)


if __name__ == "__main__":
    main()
