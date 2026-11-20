"""E57 optional probe (A6): within-claim truth probe with gray / twin image controls on Qwen3-VL-32B.

Runs the unchanged e17_probe_ctrl.py extraction and e17_probe_analyze.py analysis; the only change is that the 32B model
is loaded with device_map="auto" over the visible GPUs (it does not fit on one H20 with all hidden states).
Extraction -> /dev/shm/e57probe/qwen3vl_32b_<shard>_{orig,gray,twin}.npy; analysis -> outputs/e57/qwen3vl_32b_ctrl.json.

Usage: python e57_probe.py extract --shard 0 --nshard 2 [--limit 4]
       python e57_probe.py analyze [--jobs 10]
"""
from __future__ import annotations

import sys

import torch

import e15_probe_extract as E15
import vx

MODEL = "qwen3vl_32b"
IND = "/dev/shm/e57probe"


def load_model_auto(name: str):
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
    path = vx.C.model_path(name)
    model = Qwen3VLForConditionalGeneration.from_pretrained(path, torch_dtype=torch.bfloat16, attn_implementation="sdpa",
                                                            device_map="auto").eval()
    proc = AutoProcessor.from_pretrained(path, max_pixels=1280 * 28 * 28)
    return model, proc


def main() -> None:
    phase = sys.argv[1]
    rest = sys.argv[2:]
    if phase == "extract":
        E15.load_model = load_model_auto
        import e17_probe_ctrl as E17
        E17.OUTD = IND
        sys.argv = ["e17_probe_ctrl.py", "--model", MODEL] + rest
        E17.main()
    else:
        import e17_probe_analyze as A
        A.IND = IND
        A.OUT = f"{vx.OUT}/e57"
        sys.argv = ["e17_probe_analyze.py", "--model", MODEL] + rest
        A.main()


if __name__ == "__main__":
    main()
