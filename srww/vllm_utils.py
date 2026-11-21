"""Shared GPU helpers for the experiments, built on the original v2 protocol code.

Data loading, prompt construction (``build_messages``), CoT generation prompts, filler text and
top-20 letter scoring are imported unchanged from ``legacy/scripts/v2/common.py`` so that every new
number uses the exact protocol behind the paper. This module adds:

* exact full-vocabulary option log-probabilities at ``Answer: (`` (vLLM V0 logits processor that reads
  the raw logits row before sampling), alongside the legacy top-20 readout;
* greedy rationale generation under image variants (original / gray / mirror / none);
* JSONL helpers.
"""
from __future__ import annotations
from srww import paths as _vtp  # noqa: E402

import io
import json
import math
import os
import re
from typing import Any

SHARE = os.environ.get("VTAX_DATA_ROOT", f"{_vtp.DATA_ROOT}")
W = os.environ.get("VTAX_ROOT", f"{_vtp.W}")
LEGACY = f"{W}/legacy"
from srww import protocol as C  # noqa: E402

C.ROOT = SHARE
C.MODELS["qwen3vl_8b"] = f"{SHARE}/models/Qwen3-VL-8B-Instruct"
C.MODELS["qwen3vl_8b_think"] = f"{SHARE}/models/Qwen3-VL-8B-Thinking"
C.MODELS["internvl35_8b"] = f"{SHARE}/models/InternVL3_5-8B"
# Qwen3-VL scale ladder (same family / training recipe, different size)
C.MODELS["qwen3vl_2b"] = f"{SHARE}/models/Qwen3-VL-2B-Instruct"
C.MODELS["qwen3vl_4b"] = f"{SHARE}/models/Qwen3-VL-4B-Instruct"
C.MODELS["qwen3vl_32b"] = f"{SHARE}/models/Qwen3-VL-32B-Instruct"
# Exp-43: >= 30B non-Qwen3 open models (InternVL3.5-38B: same family / template as internvl35_8b; qwen32b = Qwen2.5-VL-32B in legacy C.MODELS)
C.MODELS["internvl35_38b"] = f"{SHARE}/models/InternVL3_5-38B"
C.MODELS["internvl3_38b"] = f"{SHARE}/models/InternVL3-38B"
C.MODELS["qwen32b"] = f"{SHARE}/models/Qwen2.5-VL-32B-Instruct"

SPATIAL = ["cvb_relation", "cvb_depth", "cvb_distance", "syn_relpos", "syn_depth"]
OUT = f"{W}/outputs"
GRAY = (128, 128, 128)

MOLMO_TEMPLATE = (
    "{% for m in messages %}{% if m['role'] == 'user' %}{{ '' if loop.first else ' ' }}User: {{ m['content'] if m['content'] is string "
    "else (m['content'] | selectattr('type','equalto','text') | map(attribute='text') | join('\n')) }}"
    "{% else %} Assistant: {{ m['content'] }}{% endif %}{% endfor %}"
)


# ----------------------------------------------------------------------------- io

def read_jsonl(path: str) -> list[dict]:
    """Read a JSONL file (missing file -> empty list)."""
    if not os.path.exists(path):
        return []
    with open(path) as f:
        return [json.loads(l) for l in f if l.strip()]


def write_jsonl(path: str, recs: list[dict]) -> None:
    """Write records to JSONL (overwrites)."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        for r in recs:
            f.write(json.dumps(r, default=lambda x: None) + "\n")


def legacy_e1(model: str) -> dict[str, dict]:
    """Original per-item E1 records keyed by uid (R text, cot_len, top-20 scores)."""
    p = f"{_vtp.LEGACY_RESULTS}/e1/{model}.json"
    if not os.path.exists(p):
        p = f"{OUT}/e4/{model}.json"
    d = json.load(open(p))
    return {it["uid"]: it for it in d["items"]}


def load_items(tasks: list[str] = SPATIAL, limit: int | None = None) -> list[dict]:
    """Items via the original loaders; ``limit`` keeps the first n per task (smoke runs)."""
    out = []
    for t in tasks:
        xs = C.DATASETS[t]()
        out.extend(xs[:limit] if limit else xs)
    return out


# ----------------------------------------------------------------------------- images

def image_variant(item: dict, variant: str):
    """PIL image under a variant: original | gray | mirror."""
    from PIL import Image, ImageOps
    img = C.get_image(item)
    if variant == "original":
        return img
    if variant == "gray":
        return Image.new("RGB", img.size, GRAY)
    if variant == "mirror":
        return ImageOps.mirror(img)
    raise ValueError(variant)


def img_ref(img) -> dict:
    """OpenAI-style image content part."""
    return {"type": "image_url", "image_url": {"url": C.image_data_url(img)}}


def strip_images(msgs: list[dict]) -> list[dict]:
    """Remove image parts (no-image condition)."""
    out = []
    for m in msgs:
        if isinstance(m["content"], list):
            parts = [p for p in m["content"] if p.get("type") != "image_url"]
            out.append({"role": m["role"], "content": parts})
        else:
            out.append(m)
    return out


# ----------------------------------------------------------------------------- engine

def is_v1(model: str) -> bool:
    """Models run on the vLLM V1 engine (no per-request logits processors; top-20 readout).

    Qwen3-VL / InternVL3.5 need vLLM >= 0.11 (V1 only). Molmo must use V1: its V0 implementation in
    vLLM 0.8.4 puts ~1% of mass on option letters vs 99.9% under V1 (the engine of the original runs).
    """
    return model.startswith(("qwen3vl", "internvl35", "molmo"))


def make_llm(model: str, max_images: int = 2, max_logprobs: int = 20, tp: int | None = None):
    """vLLM engine with the original v2 settings (V0 engine for per-request logits processors).

    ``tp`` (or env ``VX_TP``) sets tensor parallelism; needed for models that do not fit on one GPU.
    """
    import torch
    torch.backends.cuda.matmul.allow_tf32 = True
    tp = tp or int(os.environ.get("VX_TP", "1"))
    if is_v1(model):
        os.environ["VLLM_USE_V1"] = "1"
        from vllm import LLM
        kw = dict(model=C.model_path(model), trust_remote_code=True, dtype="bfloat16", max_model_len=16384,
                  gpu_memory_utilization=float(os.environ.get("VX_GPU_UTIL", "0.85")), seed=0,
                  limit_mm_per_prompt={"image": max_images}, enable_prefix_caching=True, max_logprobs=max_logprobs,
                  tensor_parallel_size=tp, disable_custom_all_reduce=tp > 1)  # custom all-reduce failed on our nodes
        if model.startswith("qwen3vl"):
            kw["mm_processor_kwargs"] = {"max_pixels": 1280 * 28 * 28}
        if model in ("qwen3vl_2b", "qwen3vl_4b"):
            # ViT head dim 64 selects vLLM's flash-attn kernel, which fails on this driver ("PTX ...
            # unsupported toolchain"); the 8B/32B ViT (head dim 72) already runs on torch SDPA, so use
            # SDPA here too. The patch must live in this process (no engine-core subprocess).
            os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
            import vllm.model_executor.models.qwen3_vl as _q3
            from vllm.platforms import _Backend
            _q3.get_vit_attn_backend = lambda head_size, dtype: _Backend.TORCH_SDPA
        if model.startswith("internvl35"):
            kw["mm_processor_kwargs"] = {"max_dynamic_patch": 6}
            # vLLM 0.11's flash-attn kernel for InternViT fails on this driver ("PTX ... unsupported
            # toolchain"); TRITON_ATTN for the decoder makes the ViT fall back to torch SDPA.
            os.environ.setdefault("VLLM_ATTENTION_BACKEND", "TRITON_ATTN")
        if model.startswith("molmo"):
            kw.update(max_model_len=4096, limit_mm_per_prompt={"image": 1},
                      mm_processor_kwargs={"message_format": "none"})
            # vLLM 0.8.4 wraps every Molmo prompt as "User: <prompt> Assistant:" (message_format="role"),
            # which appends a second "Assistant:" after our "Answer: (" prefix. MOLMO_TEMPLATE already renders
            # the native "User: ... Assistant: ..." format, so switch the wrapper off (front-end process).
            os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
            import vllm.model_executor.models.molmo as _molmo
            for _cls in vars(_molmo).values():
                if isinstance(_cls, type) and isinstance(vars(_cls).get("message_format"), property):
                    _cls.message_format = property(lambda self: "none")
        return LLM(**kw)
    os.environ.setdefault("VLLM_USE_V1", "0")
    from vllm import LLM
    is_molmo = model.startswith("molmo")
    kw: dict = dict(model=C.model_path(model), trust_remote_code=True, dtype="bfloat16", max_model_len=12288,
                    gpu_memory_utilization=float(os.environ.get("VX_GPU_UTIL", "0.85")), seed=0,
                    limit_mm_per_prompt={"image": 1 if is_molmo else max_images}, enable_prefix_caching=True,
                    max_logprobs=max_logprobs, tensor_parallel_size=tp, disable_custom_all_reduce=tp > 1)
    if model.startswith("qwen") and not model.startswith("qwen3vl"):
        kw["mm_processor_kwargs"] = {"max_pixels": 1280 * 28 * 28}
    if model.startswith("internvl"):
        kw["mm_processor_kwargs"] = {"max_dynamic_patch": 6}
    if model == "phi35v":
        kw["mm_processor_kwargs"] = {"num_crops": 4}
    if is_molmo:
        kw["max_model_len"] = 4096
    return LLM(**kw)


def chat_kw(model: str, cont: bool = True) -> dict:
    """Keyword arguments for ``llm.chat`` (continue the final assistant message)."""
    kw = dict(add_generation_prompt=not cont, continue_final_message=cont, use_tqdm=True)
    if model.startswith("molmo"):
        kw["chat_template"] = MOLMO_TEMPLATE
    return kw


def letter_token_ids(tok, letters: str = "ABCDEF") -> dict[str, dict[str, int]]:
    """Single-token ids per letter for variants: bare, space-prefixed, lower, space-lower."""
    vocab = tok.get_vocab()
    out: dict[str, dict[str, int]] = {}
    for L in letters:
        out[L] = {}
        l = L.lower()
        for name, pieces in (("bare", [L]), ("sp", ["Ġ" + L, "▁" + L]), ("low", [l]), ("splow", ["Ġ" + l, "▁" + l])):
            for p in pieces:
                if p in vocab:
                    out[L][name] = vocab[p]
                    break
    return out


class ExactReadout:
    """Full-vocabulary log-probabilities of letter tokens at the first generated position."""

    def __init__(self, tok):
        self.ids = letter_token_ids(tok)
        self.flat = sorted({i for v in self.ids.values() for i in v.values()})
        self.store: dict[int, list[float]] = {}

    def sampling_params(self, key: int):
        """SamplingParams (greedy, 1 token, top-20 logprobs) with a capturing logits processor."""
        import torch
        from vllm import SamplingParams
        flat = self.flat
        store = self.store

        def proc(past_ids, logits):
            if len(past_ids) == 0:
                lp = torch.log_softmax(logits.float(), dim=-1)
                store[key] = lp[torch.tensor(flat, device=lp.device)].tolist()
            return logits

        return SamplingParams(temperature=0.0, max_tokens=1, logprobs=20, logits_processors=[proc])

    def decode(self, key: int, choices: list[str]) -> dict[str, Any]:
        """Per-option exact logprobs: ``bare`` (main) and ``sum`` (logsumexp over variants)."""
        row = dict(zip(self.flat, self.store.pop(key)))
        bare = {c: row[self.ids[c]["bare"]] for c in choices}
        summ = {}
        for c in choices:
            vals = [row[i] for i in self.ids[c].values()]
            m = max(vals)
            summ[c] = m + math.log(sum(math.exp(v - m) for v in vals))
        return {"lp_bare": bare, "lp_sum": summ}


def readout(llm, model: str, msgs: list[list[dict]], choices: list[list[str]], tag: str = "") -> list[dict]:
    """Forced one-token readout for each message list (ending in ``Answer: (``).

    Returns per prompt: legacy top-20 letter scores, exact bare/sum logprobs, and both predictions.
    Under the V1 engine (Qwen3-VL) the exact fields repeat the top-20 scores (missing -> -1e4) and
    ``exact`` is False.
    """
    v1 = is_v1(model)
    if v1:
        from vllm import SamplingParams
        res = llm.chat(msgs, SamplingParams(temperature=0.0, max_tokens=1, logprobs=20), **chat_kw(model))
    else:
        ex = ExactReadout(llm.get_tokenizer())
        sps = [ex.sampling_params(i) for i in range(len(msgs))]
        res = llm.chat(msgs, sps, **chat_kw(model))
    out = []
    for i, r in enumerate(res):
        top = {lp.decoded_token: lp.logprob for lp in r.outputs[0].logprobs[0].values()}
        t20 = C.letter_scores(top, choices[i])
        if v1:
            lp = {k: (v if v > -math.inf else -1e4) for k, v in t20.items()}
            e = {"lp_bare": lp, "lp_sum": lp}
        else:
            e = ex.decode(i, choices[i])
        fin = {k: v for k, v in t20.items() if v > -math.inf}
        out.append({"top20": {k: (v if v > -math.inf else None) for k, v in t20.items()},
                    "pred_top20": max(fin, key=fin.get) if fin else None,
                    "lp_bare": e["lp_bare"], "lp_sum": e["lp_sum"],
                    "pred_exact": max(e["lp_sum"], key=e["lp_sum"].get)})
    return out


def generate(llm, model: str, msgs: list[list[dict]], max_tokens: int = 512, temperature: float = 0.0,
             n: int = 1, seed: int = 0, cont: bool = True) -> list[dict]:
    """Continue the final assistant message (``cont``) or open a new assistant turn; returns text/len/trunc."""
    from vllm import SamplingParams
    sp = SamplingParams(temperature=temperature, max_tokens=max_tokens, n=n, seed=seed)
    res = llm.chat(msgs, sp, **chat_kw(model, cont))
    out = []
    for r in res:
        outs = [{"text": o.text.strip(), "len": len(o.token_ids), "trunc": o.finish_reason == "length"}
                for o in r.outputs]
        out.append(outs[0] if n == 1 else {"samples": outs})
    return out


# ----------------------------------------------------------------------------- probabilities

def probs_from_lp(lp: dict[str, float | None]) -> dict[str, float] | None:
    """Renormalize option logprobs (None/-inf treated as missing)."""
    fin = {k: v for k, v in lp.items() if v is not None and math.isfinite(v)}
    if not fin:
        return None
    m = max(fin.values())
    z = sum(math.exp(v - m) for v in fin.values())
    return {k: (math.exp(lp[k] - m) / z if k in fin else 0.0) for k in lp}


def option_words(item: dict) -> dict[str, str]:
    """Option text per letter parsed from ``(X) text`` lines of the question."""
    out = {}
    for line in item["question"].split("\n"):
        m = re.match(r"\(([A-F])\)\s*(.+)", line.strip())
        if m:
            out[m.group(1)] = m.group(2).strip()
    return out
