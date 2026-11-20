"""Decompress the shipped per-item outputs (``*.gz`` under outputs/ and data/) next to the compressed copies.

The scripts read plain ``.jsonl`` / ``.json`` files; run this once after unzipping. Idempotent (skips files whose
decompressed copy already exists with the recorded size).

Usage: python tools/unpack_outputs.py
"""
from __future__ import annotations

import gzip
import os
import shutil
from pathlib import Path

ROOT = Path(os.environ.get("VTAX_ROOT", Path(__file__).resolve().parents[1]))


def main() -> None:
    n = 0
    for top in ("outputs", "data"):
        for gz in sorted((ROOT / top).rglob("*.gz")):
            dst = gz.with_suffix("")
            if dst.exists():
                continue
            with gzip.open(gz, "rb") as fi, open(dst, "wb") as fo:
                shutil.copyfileobj(fi, fo, 1 << 20)
            n += 1
    print(f"decompressed {n} files")


if __name__ == "__main__":
    main()
