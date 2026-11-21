"""Exp-63 LoRA fine-tune of the answer step of Qwen3-VL-8B-Instruct (HF transformers 4.57.1, no peft).

Each training example is exactly the evaluation readout prompt C.build_messages(item, 'cot', 'none', R_cut, image):
user [image, question with option lines], assistant "R_cut\\nAnswer: (" continued (chat template with
continue_final_message, as vLLM's llm.chat(continue_final_message=True)); the target is the bare letter token vLLM
would generate next. Loss = cross-entropy at that one next-token position (logits_to_keep=1, left padding).
Images go through the HF Qwen3-VL processor exactly as vLLM 0.11 processes them (see load_processor: the
max_pixels kwarg of vx.make_llm is a no-op for Qwen3-VL there). The CPU check (``cpucheck``) compares these token
ids / pixel tensors / grids with vLLM 0.11's own chat + multimodal preprocessing.

LoRA (self-contained, ``LoRALinear``): r=16, alpha=32, dropout 0.05 on the language model's q/k/v/o/gate/up/down
projections (252 modules); vision tower, merger, embeddings, norms, lm_head frozen. AdamW lr 1e-4 (betas 0.9/0.999,
weight decay 0), 3% linear warmup + cosine decay to 0, grad-norm clip 1.0, effective batch 16 (micro x accum x world),
1 epoch, seed 63, bf16 weights (LoRA params fp32), SDPA, gradient checkpointing. Pairs are shuffled once with
random.Random(63) and the two examples of a pair stay adjacent in the same micro-batch (identical order in both arms).
Multi-GPU: torchrun (LoRA gradients all-reduced by hand; no DDP wrapper around the 8B model).

Outputs: outputs/e63/ckpt/<arm>/{lora.safetensors, lora_config.json, train_log.jsonl, vcheck_hf_{init,final}.json,
summary.json}; merged full checkpoint (base safetensors shards with W += (alpha/r) B A on the 252 target weights, same
key names / index, all config / tokenizer / processor / chat-template files copied) at --merged-out
(default /dev/shm/e63_<arm>), loadable by vLLM like the base model (model keys qwen3vl_8b_e63<arm>, e63_common).

Usage (2 GPUs):  torchrun --nproc_per_node 2 experiments/lora_pilot/train.py train --arm cf [--max-steps N]
              python -m experiments.lora_pilot.train merge --arm cf               (re-merge from saved LoRA weights; CPU ok)
       (CPU): python -m experiments.lora_pilot.train selftest                     (LoRA wrapper + safetensors merge unit checks)
              python -m experiments.lora_pilot.train metacheck                    (meta-device Qwen3-VL: target modules vs checkpoint keys)
              python -m experiments.lora_pilot.train cpucheck [--n 5]             (HF training prompt ids == vLLM chat prompt ids)
              python -m experiments.lora_pilot.train dry --arm cf                 (batches of the real data loader, processor only)
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
import shutil
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from experiments.lora_pilot import lora_common as E
from experiments.lora_pilot.lora_common import C, vx

TARGET_RX = re.compile(r"^model\.language_model\.layers\.\d+\.(self_attn\.(q|k|v|o)_proj|mlp\.(gate|up|down)_proj)$")
IMG = {"type": "image_url", "image_url": {"url": "<image>"}}
EX = f"{E.E63}/train_examples"
CKPT = f"{E.E63}/ckpt"


# ----------------------------------------------------------------------------- LoRA

class LoRALinear(nn.Module):
    """y = W x + (alpha / r) * B A dropout(x); A ~ kaiming-uniform (as peft), B = 0, so the wrapper is exact at init."""

    def __init__(self, base: nn.Linear, r: int, alpha: float, dropout: float, gen: torch.Generator | None = None):
        super().__init__()
        self.base = base
        self.r, self.alpha, self.scale = r, alpha, alpha / r
        dev = base.weight.device if base.weight.device.type != "meta" else "cpu"
        a = torch.empty(r, base.in_features, dtype=torch.float32)
        bound = 1 / math.sqrt(base.in_features)  # kaiming_uniform_(a=sqrt(5)) bound for fan_in
        a.uniform_(-bound, bound, generator=gen)
        self.lora_A = nn.Parameter(a.to(dev))
        self.lora_B = nn.Parameter(torch.zeros(base.out_features, r, dtype=torch.float32, device=dev))
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.base(x)
        z = F.linear(F.linear(self.drop(x).to(self.lora_A.dtype), self.lora_A), self.lora_B)
        return y + (self.scale * z).to(y.dtype)

    def delta(self) -> torch.Tensor:
        return self.scale * (self.lora_B @ self.lora_A)


def apply_lora(model: nn.Module, r: int = 16, alpha: float = 32, dropout: float = 0.05, rx: re.Pattern = TARGET_RX,
               seed: int = E.SEED) -> dict[str, LoRALinear]:
    """Freeze every parameter, then wrap the target nn.Linear modules (deterministic A init from ``seed``)."""
    for p in model.parameters():
        p.requires_grad_(False)
    gen = torch.Generator().manual_seed(seed)
    names = [n for n, m in model.named_modules() if rx.match(n) and isinstance(m, nn.Linear)]
    out = {}
    for n in names:
        parent, child = n.rsplit(".", 1)
        pm = model.get_submodule(parent)
        lm = LoRALinear(getattr(pm, child), r, alpha, dropout, gen)
        setattr(pm, child, lm)
        out[n] = lm
    return out


def lora_state(loras: dict[str, LoRALinear]) -> dict[str, torch.Tensor]:
    sd = {}
    for n, m in loras.items():
        sd[f"{n}.lora_A"] = m.lora_A.detach().cpu().contiguous()
        sd[f"{n}.lora_B"] = m.lora_B.detach().cpu().contiguous()
    return sd


def merge_checkpoint(base_dir: str, lora_sd: dict[str, torch.Tensor], scale: float, out_dir: str) -> dict:
    """Base safetensors shards with W <- W + scale * B @ A on every LoRA target; other files copied unchanged."""
    from safetensors.torch import load_file, save_file
    base, out = Path(base_dir), Path(out_dir)
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    idx_p = base / "model.safetensors.index.json"
    shards = sorted(set(json.load(open(idx_p))["weight_map"].values())) if idx_p.exists() else ["model.safetensors"]
    want = {k[: -len(".lora_A")] + ".weight": k[: -len(".lora_A")] for k in lora_sd if k.endswith(".lora_A")}
    done, max_rel = set(), 0.0
    for sh in shards:
        sd = load_file(str(base / sh))
        for k in list(sd):
            if k in want:
                n = want[k]
                ct = torch.float64 if sd[k].dtype == torch.float64 else torch.float32
                d = scale * (lora_sd[f"{n}.lora_B"].to(ct) @ lora_sd[f"{n}.lora_A"].to(ct))
                w = sd[k].to(ct)
                sd[k] = (w + d).to(sd[k].dtype).contiguous()
                max_rel = max(max_rel, float(d.norm() / w.norm()))
                done.add(k)
        save_file(sd, str(out / sh), metadata={"format": "pt"})
    missing = sorted(set(want) - done)
    assert not missing, f"LoRA targets missing in checkpoint: {missing[:3]}"
    for f in base.iterdir():
        if f.is_file() and not f.name.endswith(".safetensors") and f.name != ".done":
            shutil.copy2(f, out / f.name)
    info = {"base": str(base), "n_merged": len(done), "scale": scale, "max_rel_delta_norm": max_rel,
            "created": time.strftime("%Y-%m-%d %H:%M:%S")}
    json.dump(info, open(out / "e63_merge.json", "w"), indent=1)
    return info


# ----------------------------------------------------------------------------- prompts / batches

def example_item(x: dict) -> dict:
    return {"uid": x["uid"], "question": x["question"], "choices": x["choices"], "gold": x["target"],
            "image_path": f"{E.STIM}/{x['image_rel']}"}


def hf_conversation(msgs: list[dict]) -> list[dict]:
    """OpenAI-style messages -> the conversation vLLM renders (content format 'openai': string content becomes one
    text part, image_url parts become {'type': 'image'})."""
    conv = []
    for m in msgs:
        parts = [{"type": "text", "text": m["content"]}] if isinstance(m["content"], str) else \
            [{"type": "image"} if p["type"] == "image_url" else {"type": "text", "text": p["text"]} for p in m["content"]]
        conv.append({"role": m["role"], "content": parts})
    return conv


def prompt_text(tok, chat_template: str, item: dict, rationale: str) -> str:
    """Chat-template text of the readout prompt (image placeholder not yet expanded)."""
    msgs = C.build_messages(item, "cot", "none", rationale, IMG)
    return tok.apply_chat_template(hf_conversation(msgs), chat_template=chat_template, tokenize=False,
                                   add_generation_prompt=False, continue_final_message=True)


def load_processor(path: str):
    """Qwen3-VL processor as vLLM 0.11 effectively builds it (use_fast); left padding for last-position logits.

    vx.make_llm passes mm_processor_kwargs={"max_pixels": 1280*28*28}, but with vLLM 0.11 + transformers 4.57.1 that
    kwarg does not change the Qwen3-VL image processor (size stays at preprocessor_config.json: longest_edge 16777216):
    a 2000x1500 image gets grid 1x94x124 in vLLM with or without it (outputs/e63/cpu_maxpixels_probe.json). The
    training inputs therefore use the processor default, i.e. exactly what the paper's vLLM runs saw; cpucheck
    asserts grid / pixel equality with vLLM's own preprocessing, including a > 1M-pixel image.
    """
    from transformers import AutoProcessor
    proc = AutoProcessor.from_pretrained(path, use_fast=True)
    proc.tokenizer.padding_side = "left"
    return proc


def letter_ids(tok) -> dict[str, int]:
    return {L: v["bare"] for L, v in vx.letter_token_ids(tok, "AB").items()}


def make_batch(proc, xs: list[dict]) -> tuple[dict, torch.Tensor]:
    """Processor inputs (left padded) and target token ids for a list of training examples."""
    tok = proc.tokenizer
    lid = letter_ids(tok)
    texts = [prompt_text(tok, proc.chat_template, example_item(x), x["rationale"]) for x in xs]
    imgs = [E.load_image(example_item(x)["image_path"]) for x in xs]
    inp = proc(text=texts, images=imgs, padding=True, return_tensors="pt")
    return dict(inp), torch.tensor([lid[x["target"]] for x in xs])


def ordered_examples(arm: str, sfx: str = "") -> list[dict]:
    """Examples of an arm in training order: pairs shuffled with random.Random(63), pair members adjacent."""
    xs = vx.read_jsonl(f"{EX}/{arm}{sfx}.jsonl")
    by: dict[str, list[dict]] = {}
    for x in xs:
        by.setdefault(x["pair"], []).append(x)
    pairs = sorted(by)
    random.Random(E.SEED).shuffle(pairs)
    return [x for p in pairs for x in sorted(by[p], key=lambda x: x["img_role"] != "v")]


# ----------------------------------------------------------------------------- model

def load_model(path: str, device: str):
    from transformers import Qwen3VLForConditionalGeneration
    m = Qwen3VLForConditionalGeneration.from_pretrained(path, dtype=torch.bfloat16, attn_implementation="sdpa")
    m.config.use_cache = False
    return m.to(device)


def last_logits(model, inp: dict, device: str) -> torch.Tensor:
    inp = {k: v.to(device) for k, v in inp.items()}
    out = model(**inp, logits_to_keep=1, use_cache=False)
    return out.logits[:, -1, :].float()


@torch.no_grad()
def hf_vcheck(model, proc, device: str, sfx: str = "", bs: int = 4) -> list[dict]:
    """HF readout of the fixed check sample (same prompts as experiments/lora_pilot/generate.py vcheck): prompt ids + letter logprobs."""
    was = model.training
    model.eval()
    xs = vx.read_jsonl(f"{EX}/check_sample{sfx}.jsonl")
    lid = letter_ids(proc.tokenizer)
    rows = []
    for i in range(0, len(xs), bs):
        chunk = xs[i:i + bs]
        inp, _ = make_batch(proc, chunk)
        lp = torch.log_softmax(last_logits(model, inp, device), -1)
        for j, x in enumerate(chunk):
            ids = inp["input_ids"][j][inp["attention_mask"][j].bool()].tolist()
            rows.append({"ex_id": x["ex_id"], "target": x["target"], "prompt_token_ids": ids,
                         "letters": {L: float(lp[j, t]) for L, t in lid.items()}})
    model.train(was)
    return rows


def compare_vcheck(hf_rows: list[dict], vllm_path: str) -> dict:
    """Token-id equality and letter-logprob agreement between the HF forward and the stored vLLM readout."""
    if not os.path.exists(vllm_path):
        return {"vllm_file": None}
    vr = {r["ex_id"]: r for r in json.load(open(vllm_path))["rows"]}
    same, dmax, agree, n = 0, 0.0, 0, 0
    for h in hf_rows:
        v = vr.get(h["ex_id"])
        if v is None:
            continue
        n += 1
        same += h["prompt_token_ids"] == v["prompt_token_ids"]
        vl = {k: (x if x is not None and x > -1e9 else -1e4) for k, x in v["letters"].items()}
        dmax = max(dmax, abs(h["letters"][h["target"]] - vl[h["target"]]))
        agree += max(h["letters"], key=h["letters"].get) == max(vl, key=vl.get)
    return {"vllm_file": vllm_path, "n": n, "ids_equal": same, "max_abs_dlogp_target": round(dmax, 4), "argmax_agree": agree}


# ----------------------------------------------------------------------------- train

def _dist():
    import torch.distributed as dist
    if "RANK" in os.environ and int(os.environ.get("WORLD_SIZE", "1")) > 1:
        import datetime
        dist.init_process_group("nccl", timeout=datetime.timedelta(minutes=60))  # rank 0 merges 17 GB while others wait
        return dist, dist.get_rank(), dist.get_world_size(), int(os.environ["LOCAL_RANK"])
    return None, 0, 1, 0


def train(a) -> None:
    dist, rank, world, lrank = _dist()
    device = f"cuda:{lrank}"
    torch.cuda.set_device(device)
    random.seed(E.SEED)
    torch.manual_seed(E.SEED)
    sfx = f"_limit{a.limit_pairs}" if a.limit_pairs else ""
    out = Path(f"{CKPT}/{a.arm}{sfx}")
    out.mkdir(parents=True, exist_ok=True)
    log = (lambda s: print(f"[e63 train {a.arm} r{rank}] {s}", flush=True))
    path = C.model_path(E.BASE)
    proc = load_processor(path)
    xs = ordered_examples(a.arm, sfx)
    E.prefetch([example_item(x)["image_path"] for x in xs], as_bytes=True)
    assert a.batch % (a.micro * world) == 0 and a.micro % 2 == 0
    accum = a.batch // (a.micro * world)
    n_steps = len(xs) // a.batch
    if a.max_steps:
        n_steps = min(n_steps, a.max_steps)
    warm = max(1, math.ceil(0.03 * n_steps))
    log(f"{len(xs)} examples, world {world}, micro {a.micro}, accum {accum}, steps {n_steps}, warmup {warm}, base {path}")
    t0 = time.time()
    model = load_model(path, device)
    loras = apply_lora(model, a.r, a.alpha, a.dropout)
    params = [p for m in loras.values() for p in (m.lora_A, m.lora_B)]
    if dist:
        for p in params:
            dist.broadcast(p.data, 0)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.train()
    n_tr = sum(p.numel() for p in params)
    log(f"loaded ({time.time() - t0:.0f}s); LoRA modules {len(loras)}, trainable params {n_tr:,}")
    if rank == 0:
        hf0 = hf_vcheck(model, proc, device, sfx)
        cmp0 = compare_vcheck(hf0, f"{E.E63}/vcheck_{E.BASE}{sfx}.json")
        json.dump({"compare": cmp0, "rows": hf0}, open(out / "vcheck_hf_init.json", "w"))
        log(f"step-0 check vs vLLM base: {cmp0}")
        if a.strict and cmp0.get("n") and cmp0["ids_equal"] != cmp0["n"]:
            raise SystemExit("prompt token ids differ between HF training inputs and vLLM; aborting (use --no-strict to override)")
    opt = torch.optim.AdamW(params, lr=a.lr, betas=(0.9, 0.999), weight_decay=0.0)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / warm if s < warm else 0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, n_steps - warm))))
    flog = open(out / "train_log.jsonl", "w") if rank == 0 else None
    lid = letter_ids(proc.tokenizer)
    ab = torch.tensor([lid["A"], lid["B"]], device=device)
    win = torch.zeros(4, device=device)  # loss sum, correct, n, p_target sum
    for step in range(n_steps):
        g = xs[step * a.batch:(step + 1) * a.batch]
        mine = g[rank * a.micro * accum:(rank + 1) * a.micro * accum]
        for k in range(accum):
            mb = mine[k * a.micro:(k + 1) * a.micro]
            inp, tgt = make_batch(proc, mb)
            tgt = tgt.to(device)
            logits = last_logits(model, inp, device)
            loss = F.cross_entropy(logits, tgt)
            (loss / accum).backward()
            with torch.no_grad():
                two = logits[:, ab]
                is_b = (tgt == lid["B"]).long()
                win += torch.stack([loss.detach() * len(mb), (two.argmax(1) == is_b).float().sum(),
                                    torch.tensor(float(len(mb)), device=device),
                                    torch.softmax(two, 1).gather(1, is_b[:, None]).sum()])
        if dist:
            flat = torch.cat([p.grad.flatten() for p in params])
            dist.all_reduce(flat)
            flat /= world
            o = 0
            for p in params:
                p.grad.copy_(flat[o:o + p.numel()].view_as(p))
                o += p.numel()
        gn = torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        sched.step()
        opt.zero_grad(set_to_none=True)
        if (step + 1) % a.log_every == 0 or step + 1 == n_steps:
            w = win.clone()
            if dist:
                dist.all_reduce(w)
            rec = {"step": step + 1, "examples": (step + 1) * a.batch, "lr": sched.get_last_lr()[0],
                   "loss": float(w[0] / w[2]), "acc": float(w[1] / w[2]), "p_target": float(w[3] / w[2]),
                   "grad_norm": float(gn), "n": int(w[2]), "elapsed_s": round(time.time() - t0, 1)}
            if flog:
                flog.write(json.dumps(rec) + "\n")
                flog.flush()
                log(json.dumps(rec))
            win.zero_()
    if rank == 0:
        from safetensors.torch import save_file
        sd = lora_state(loras)
        save_file(sd, str(out / "lora.safetensors"), metadata={"format": "pt"})
        cfg = {"r": a.r, "alpha": a.alpha, "dropout": a.dropout, "scale": a.alpha / a.r, "targets": TARGET_RX.pattern,
               "n_modules": len(loras), "n_params": n_tr, "base": path, "arm": a.arm, "steps": n_steps, "batch": a.batch,
               "micro": a.micro, "accum": accum, "world": world, "lr": a.lr, "warmup_steps": warm, "seed": E.SEED,
               "examples_file": f"{EX}/{a.arm}{sfx}.jsonl"}
        json.dump(cfg, open(out / "lora_config.json", "w"), indent=1)
        hf1 = hf_vcheck(model, proc, device, sfx)
        json.dump({"rows": hf1}, open(out / "vcheck_hf_final.json", "w"))
        del model
        torch.cuda.empty_cache()
        info = merge_checkpoint(path, sd, a.alpha / a.r, a.merged_out or f"/dev/shm/e63_{a.arm}{sfx}")
        json.dump({"train_s": time.time() - t0, "merge": info, "config": cfg}, open(out / "summary.json", "w"), indent=1)
        log(f"saved LoRA to {out}, merged -> {info} ({time.time() - t0:.0f}s)")
    if dist:
        dist.barrier()
        dist.destroy_process_group()


def merge_cmd(a) -> None:
    from safetensors.torch import load_file
    sfx = f"_limit{a.limit_pairs}" if a.limit_pairs else ""
    d = f"{CKPT}/{a.arm}{sfx}"
    cfg = json.load(open(f"{d}/lora_config.json"))
    print(merge_checkpoint(C.model_path(E.BASE), load_file(f"{d}/lora.safetensors"), cfg["scale"],
                           a.merged_out or f"/dev/shm/e63_{a.arm}{sfx}"))


# ----------------------------------------------------------------------------- CPU checks

@torch.no_grad()
def selftest() -> None:
    """LoRA wrapper: exact at init, merge equality (module and safetensors level), only LoRA params trainable."""
    import tempfile
    from safetensors.torch import load_file, save_file
    torch.manual_seed(0)

    class Tiny(nn.Module):
        def __init__(self):
            super().__init__()
            self.model = nn.Module()
            self.model.language_model = nn.Module()
            self.model.language_model.layers = nn.ModuleList()
            for _ in range(2):
                L = nn.Module()
                L.self_attn = nn.Module()
                for n, (i, o) in {"q_proj": (32, 32), "k_proj": (32, 8), "v_proj": (32, 8), "o_proj": (32, 32)}.items():
                    setattr(L.self_attn, n, nn.Linear(i, o, bias=False))
                L.mlp = nn.Module()
                for n, (i, o) in {"gate_proj": (32, 48), "up_proj": (32, 48), "down_proj": (48, 32)}.items():
                    setattr(L.mlp, n, nn.Linear(i, o, bias=False))
                self.model.language_model.layers.append(L)
            self.model.visual = nn.Linear(32, 32)
            self.lm_head = nn.Linear(32, 10, bias=False)

        def forward(self, x):
            h = x
            for L in self.model.language_model.layers:
                a = L.self_attn
                q, k, v = a.q_proj(h), a.k_proj(h), a.v_proj(h)
                h = h + a.o_proj(q * torch.sigmoid(k.repeat(1, 4) + v.repeat(1, 4)))
                h = h + L.mlp.down_proj(F.silu(L.mlp.gate_proj(h)) * L.mlp.up_proj(h))
            return self.lm_head(self.model.visual(h) + h)

    m = Tiny().double()
    x = torch.randn(5, 32, dtype=torch.float64)
    ref = m(x)
    base_sd = {k: v.clone() for k, v in m.state_dict().items()}
    loras = apply_lora(m, r=4, alpha=8, dropout=0.05)
    for L in loras.values():
        L.lora_A.data = L.lora_A.data.double()
        L.lora_B.data = L.lora_B.data.double()
    m.eval()
    res = {"n_wrapped": len(loras), "init_max_abs_diff": float((m(x) - ref).abs().max())}
    tr = [n for n, p in m.named_parameters() if p.requires_grad]
    res["trainable_only_lora"] = all(n.endswith(("lora_A", "lora_B")) for n in tr) and len(tr) == 2 * len(loras)
    for L in loras.values():
        L.lora_B.data = torch.randn_like(L.lora_B) * 0.1
    y = m(x)
    res["perturbed_changes_output"] = float((y - ref).abs().max())
    m2 = Tiny().double()
    m2.load_state_dict(base_sd)
    for n, L in loras.items():
        m2.get_submodule(n).weight.data += L.delta().double()
    res["module_merge_max_abs_diff"] = float((m2(x) - y).abs().max())
    with tempfile.TemporaryDirectory() as td:
        bd, od = Path(td) / "base", Path(td) / "merged"
        bd.mkdir()
        keys = sorted(base_sd)
        half = len(keys) // 2
        wm = {}
        for i, ks in enumerate((keys[:half], keys[half:])):
            fn = f"model-0000{i + 1}-of-00002.safetensors"
            save_file({k: base_sd[k].contiguous() for k in ks}, str(bd / fn))
            wm.update({k: fn for k in ks})
        json.dump({"metadata": {}, "weight_map": wm}, open(bd / "model.safetensors.index.json", "w"))
        json.dump({"dummy": 1}, open(bd / "config.json", "w"))
        info = merge_checkpoint(str(bd), lora_state(loras), loras[next(iter(loras))].scale, str(od))
        m3 = Tiny().double()
        msd = {}
        for f in od.glob("*.safetensors"):
            msd.update(load_file(str(f)))
        m3.load_state_dict(msd)
        res["ckpt_merge_max_abs_diff"] = float((m3(x) - y).abs().max())
        res["ckpt_merge_n"] = info["n_merged"]
        res["ckpt_files"] = sorted(p.name for p in od.iterdir())
    ok = (res["init_max_abs_diff"] == 0.0 and res["trainable_only_lora"] and res["perturbed_changes_output"] > 1e-3
          and res["module_merge_max_abs_diff"] < 1e-10 and res["ckpt_merge_max_abs_diff"] < 1e-10 and res["ckpt_merge_n"] == 14)
    res["PASS"] = ok
    print(json.dumps(res, indent=1))
    json.dump(res, open(f"{E.E63}/cpu_selftest_lora.json", "w"), indent=1)
    assert ok


def metacheck() -> None:
    """Instantiate Qwen3-VL-8B on the meta device (no weights): LoRA targets, trainable count, checkpoint key names."""
    import inspect
    from transformers import AutoConfig, Qwen3VLForConditionalGeneration
    path = C.model_path(E.BASE)
    cfg = AutoConfig.from_pretrained(path)
    with torch.device("meta"):
        m = Qwen3VLForConditionalGeneration._from_config(cfg, dtype=torch.bfloat16)
    names = [n for n, mod in m.named_modules() if TARGET_RX.match(n) and isinstance(mod, nn.Linear)]
    keys = set(json.load(open(f"{path}/model.safetensors.index.json"))["weight_map"])
    own = {n for n, _ in m.named_parameters()}
    loras = apply_lora(m)
    n_tr = sum(p.numel() for p in m.parameters() if p.requires_grad)
    vis = [n for n, p in m.named_parameters() if p.requires_grad and ("visual" in n or "lm_head" in n or "embed" in n)]
    res = {"n_targets": len(names), "expected": 36 * 7, "targets_in_ckpt": sum(f"{n}.weight" in keys for n in names),
           "hf_param_names_eq_ckpt_keys": own == keys, "n_trainable": n_tr, "trainable_in_vision_or_head": len(vis),
           "example_targets": names[:7], "has_logits_to_keep": "logits_to_keep" in inspect.signature(Qwen3VLForConditionalGeneration.forward).parameters}
    res["PASS"] = (len(names) == 252 and res["targets_in_ckpt"] == 252 and res["hf_param_names_eq_ckpt_keys"]
                   and not vis and res["has_logits_to_keep"])
    print(json.dumps(res, indent=1))
    json.dump(res, open(f"{E.E63}/cpu_metacheck.json", "w"), indent=1)
    assert res["PASS"]


def vllm_reference(path: str):
    """vLLM 0.11's own chat + multimodal preprocessing on CPU (no weights): returns f(msgs) -> (prompt str, ids, mm)."""
    from vllm.config import ModelConfig
    from vllm.entrypoints.chat_utils import (apply_hf_chat_template, parse_chat_messages,
                                             resolve_chat_template_content_format)
    from vllm.inputs.preprocess import InputPreprocessor
    from vllm.multimodal import MULTIMODAL_REGISTRY
    from vllm.transformers_utils.tokenizer import get_tokenizer
    mc = ModelConfig(model=path, trust_remote_code=True, dtype="bfloat16", max_model_len=16384, seed=0,
                     limit_mm_per_prompt={"image": 2}, mm_processor_kwargs={"max_pixels": E.MAX_PIXELS})
    tok = get_tokenizer(path, trust_remote_code=True)
    fmt = resolve_chat_template_content_format(None, None, "auto", tok, model_config=mc)
    ip = InputPreprocessor(mc, tok, MULTIMODAL_REGISTRY)

    def run(msgs: list[dict]):
        conv, mm, _ = parse_chat_messages(msgs, mc, tok, content_format=fmt)
        s = apply_hf_chat_template(tok, conv, None, None, model_config=mc, add_generation_prompt=False,
                                   continue_final_message=True)
        ids = tok.encode(s, add_special_tokens=False)
        prompt = {"prompt_token_ids": ids, "multi_modal_data": mm}
        res = ip.preprocess(prompt)
        return s, ids, res
    return run, fmt


def cpucheck(n: int) -> None:
    """Training prompts (HF processor path) vs vLLM chat prompts (vLLM preprocessing path) on n training items + 2
    natural images; rationales: stored base cue-set R_cut (stand-ins for the generated ones)."""
    path = C.model_path(E.BASE)
    proc = load_processor(path)
    run, fmt = vllm_reference(path)
    items = E.load_train_items()
    rng = random.Random(E.SEED)
    pick = rng.sample([it for it in items if it["kind"] == "depth"], (n + 1) // 2) + \
        rng.sample([it for it in items if it["kind"] == "lr"], n // 2)
    cues = [r["text"] for r in vx.read_jsonl(f"{vx.OUT}/e10/{E.BASE}_cues.jsonl") if r["condition"] == "cot"]
    rats = [E.r_cut(t) for t in rng.sample(cues, n)]
    xs = [{"ex_id": it["uid"], "uid": it["uid"], "question": it["question"], "choices": it["choices"], "target": it["gold"],
           "image_rel": it["image_rel"], "rationale": r} for it, r in zip(pick, rats)]
    cases = [(example_item(x), x["rationale"], E.load_image(example_item(x)["image_path"])) for x in xs]
    for it in E.load_eval_set("heldout")[:400:200]:
        cases.append((it, E.r_cut(it["base"]["R"]), it["image"]))
    big = dict(cases[-1][0], uid=cases[-1][0]["uid"] + "_upscaled2000x1500")
    cases.append((big, cases[-1][1], cases[-1][2].resize((2000, 1500))))  # > 1280*28*28 px: max_pixels would matter
    rows = []
    for k, (it, rat, img) in enumerate(cases):
        msgs = C.build_messages(it, "cot", "none", rat, vx.img_ref(img))
        s_v, ids_pre, res = run(msgs)
        s_h = prompt_text(proc.tokenizer, proc.chat_template, it, rat)
        hin = proc(text=[s_h], images=[img], return_tensors="pt")
        h_ids = hin["input_ids"][0].tolist()
        v_ids = list(res["prompt_token_ids"])
        mmk = res["mm_kwargs"]["image"][0]
        v_pix, v_thw = mmk["pixel_values"].data, mmk["image_grid_thw"].data
        row = {"case": it["uid"], "image_size": list(img.size), "text_equal": s_v == s_h, "ids_equal": h_ids == v_ids,
               "n_ids": len(h_ids), "grid_thw_hf": hin["image_grid_thw"][0].tolist(), "grid_thw_vllm": v_thw.tolist(),
               "pixel_values_equal_bf16": bool(torch.equal(hin["pixel_values"].to(v_pix.dtype), v_pix)),
               "pixel_dtype_vllm": str(v_pix.dtype),
               "pixel_max_abs_diff_fp32_vs_vllm": float((hin["pixel_values"].float() - v_pix.float()).abs().max()),
               "ends_with": proc.tokenizer.decode(h_ids[-6:]), "target_bare_id": letter_ids(proc.tokenizer)[it["gold"]]}
        rows.append(row)
        print(json.dumps(row), flush=True)
    s_bat, tg = make_batch(proc, xs[:4])
    left_ok = all(int(s_bat["attention_mask"][j, -1]) == 1 for j in range(len(tg)))
    unb = [proc(text=[prompt_text(proc.tokenizer, proc.chat_template, example_item(x), x["rationale"])],
                images=[E.load_image(example_item(x)["image_path"])], return_tensors="pt")["input_ids"][0].tolist() for x in xs[:4]]
    bat = [s_bat["input_ids"][j][s_bat["attention_mask"][j].bool()].tolist() for j in range(len(tg))]
    res = {"content_format": fmt, "rows": rows, "batched_left_padded_last_is_real": left_ok, "batched_ids_eq_unbatched": bat == unb,
           # vLLM keeps pixel_values in the model dtype (bf16); the bf16 vision tower casts the HF fp32 tensor the same way
           "PASS": all(r["text_equal"] and r["ids_equal"] and r["pixel_values_equal_bf16"]
                       and [r["grid_thw_hf"]] == (r["grid_thw_vllm"] if isinstance(r["grid_thw_vllm"][0], list) else [r["grid_thw_vllm"]])
                       for r in rows)
           and left_ok and bat == unb}
    json.dump(res, open(f"{E.E63}/cpu_prompt_check.json", "w"), indent=1)
    print(json.dumps({k: v for k, v in res.items() if k != "rows"}, indent=1))
    assert res["PASS"]


def dry(a) -> None:
    """Real data-loader batches on CPU (processor only): shapes, targets, ordering invariants."""
    sfx = f"_limit{a.limit_pairs}" if a.limit_pairs else ""
    proc = load_processor(C.model_path(E.BASE))
    xs = ordered_examples(a.arm, sfx)
    other = ordered_examples("match" if a.arm == "cf" else "cf", sfx)
    print(f"[e63 dry] {a.arm}: {len(xs)} examples; same image order as other arm: "
          f"{[x['uid'] for x in xs] == [x['uid'] for x in other]}; pairs adjacent: "
          f"{all(xs[i]['pair'] == xs[i + 1]['pair'] for i in range(0, len(xs), 2))}")
    for k in range(2):
        inp, tgt = make_batch(proc, xs[k * a.micro:(k + 1) * a.micro])
        print({kk: tuple(v.shape) for kk, v in inp.items()}, "targets", tgt.tolist(),
              "last tokens", [proc.tokenizer.decode(inp["input_ids"][j, -3:]) for j in range(len(tgt))])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["train", "merge", "selftest", "metacheck", "cpucheck", "dry"])
    ap.add_argument("--arm", choices=E.ARMS, default="cf")
    ap.add_argument("--r", type=int, default=16)
    ap.add_argument("--alpha", type=float, default=32)
    ap.add_argument("--dropout", type=float, default=0.05)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--micro", type=int, default=4)
    ap.add_argument("--log-every", type=int, default=20)
    ap.add_argument("--max-steps", type=int, default=None)
    ap.add_argument("--limit-pairs", type=int, default=None)
    ap.add_argument("--merged-out", default=None)
    ap.add_argument("--no-strict", dest="strict", action="store_false")
    ap.add_argument("--n", type=int, default=5)
    a = ap.parse_args()
    os.makedirs(E.E63, exist_ok=True)
    {"train": lambda: train(a), "merge": lambda: merge_cmd(a), "selftest": selftest, "metacheck": metacheck,
     "cpucheck": lambda: cpucheck(a.n), "dry": lambda: dry(a)}[a.cmd]()


if __name__ == "__main__":
    main()
