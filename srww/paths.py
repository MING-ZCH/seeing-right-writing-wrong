"""Locations used by the code of this repository.

Everything is derived from ``VTAX_ROOT`` (default: the repository root, found from this file). Run the scripts from
the repository root (``cd $VTAX_ROOT``): several of them use paths relative to it (``outputs/bundle/...``).

VTAX_ROOT        repository root (srww/, experiments/, data/, outputs/)
VTAX_DATA_ROOT   external data: ``datasets/`` (public benchmarks, COCO val2017) and ``models/`` (checkpoints);
                 default ``$VTAX_ROOT/external``
VTAX_SCRATCH     scratch space of the GPU / API runners; default ``/tmp/vtax``
"""
from __future__ import annotations

import os
from pathlib import Path

ROOT = os.environ.get("VTAX_ROOT") or str(Path(__file__).resolve().parents[1])
W = ROOT
OUTPUTS = f"{ROOT}/outputs"
LEGACY_RESULTS = f"{OUTPUTS}/legacy_v2"
ITEMS = f"{ROOT}/data/items"
DATA_ROOT = os.environ.get("VTAX_DATA_ROOT", f"{ROOT}/external")
DATASETS = f"{DATA_ROOT}/datasets"
MODELS = f"{DATA_ROOT}/models"
SCRATCH = os.environ.get("VTAX_SCRATCH", "/tmp/vtax")


def resolve(p: str) -> str:
    """Map a placeholder path stored in the released outputs (``<DATA_ROOT>/...``, ``<VTAX_ROOT>/...``) to a real path."""
    return (p.replace("<DATA_ROOT>", DATA_ROOT).replace("<VTAX_ROOT>", ROOT).replace("<SCRATCH>", SCRATCH)
            if isinstance(p, str) else p)
