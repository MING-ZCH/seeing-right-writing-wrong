"""Locations used by the scripts of this package.

Everything is derived from ``VTAX_ROOT`` (default: the package root, found from this file). Run the scripts from
the package root (``cd $VTAX_ROOT``): several of them use paths relative to it (``outputs/bundle/...``).

VTAX_ROOT        package root (scripts/, outputs/, data/, reference/)
VTAX_DATA_ROOT   external data: ``datasets/`` (public benchmarks, COCO val2017) and ``models/`` (checkpoints);
                 default ``$VTAX_ROOT/external``
VTAX_REGEN       where regenerated ``tables/`` and ``figures/`` are written; default ``$VTAX_ROOT/regenerated``
VTAX_SCRATCH     scratch space of the GPU / API runners; default ``/tmp/vtax``
"""
from __future__ import annotations

import os
from pathlib import Path


def _root() -> str:
    if os.environ.get("VTAX_ROOT"):
        return os.environ["VTAX_ROOT"]
    here = Path(__file__).resolve().parent
    for q in (here, *here.parents):
        if (q / "scripts" / "vtax_paths.py").is_file() and (q / "outputs").is_dir():
            return str(q)
    return str(here.parent)


ROOT = _root()
W = ROOT
OUTPUTS = f"{ROOT}/outputs"
LEGACY_RESULTS = f"{OUTPUTS}/legacy_v2"
SCRIPTS = f"{ROOT}/scripts"
LEGACY_SCRIPTS = f"{SCRIPTS}/legacy_v2"
ITEMS = f"{ROOT}/data/items"
PAPER = os.environ.get("VTAX_REGEN", f"{ROOT}/regenerated")
TABLES = f"{PAPER}/tables"
FIGURES = f"{PAPER}/figures"
DATA_ROOT = os.environ.get("VTAX_DATA_ROOT", f"{ROOT}/external")
DATASETS = f"{DATA_ROOT}/datasets"
MODELS = f"{DATA_ROOT}/models"
SCRATCH = os.environ.get("VTAX_SCRATCH", "/tmp/vtax")
# Hugging Face ids of the public benchmarks (see docs/REPRODUCE.md, "Data"); set before running the download / item builders.
CVBENCH_HF_ID = os.environ.get("CVBENCH_HF_ID", "")
VSIBENCH_HF_ID = os.environ.get("VSIBENCH_HF_ID", "")


def resolve(p: str) -> str:
    """Map a placeholder path stored in the released outputs (``<DATA_ROOT>/...``, ``<VTAX_ROOT>/...``) to a real path."""
    return (p.replace("<DATA_ROOT>", DATA_ROOT).replace("<VTAX_ROOT>", ROOT).replace("<SCRATCH>", SCRATCH)
            if isinstance(p, str) else p)


for _d in (PAPER, TABLES, FIGURES):
    os.makedirs(_d, exist_ok=True)
