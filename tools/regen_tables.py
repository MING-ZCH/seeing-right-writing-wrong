"""Regenerate the paper tables / figures listed in tools/targets.json from the shipped outputs (CPU only).

Runs each generator from the package root, writes tables / figures to ``$VTAX_REGEN`` (default regenerated/),
logs to regenerated/logs/, then runs tools/check_tables.py. Generators that also rebuild a bundle file overwrite
it in outputs/bundle/; the shipped copies are saved to regenerated/bundle_shipped/ first and every rebuilt bundle
file is compared with its shipped copy (JSON-equal) in regenerated/bundle_check.json.

Usage: python tools/regen_tables.py [--only RUN_ID ...] [--list] [--no-check]
"""
from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(os.environ.get("VTAX_ROOT", Path(__file__).resolve().parents[1]))
REGEN = Path(os.environ.get("VTAX_REGEN", ROOT / "regenerated"))


def close(a, b) -> bool:
    """JSON equality with NaN == NaN and floats equal up to 1e-12 (relative; summation-order noise)."""
    if isinstance(a, float) and isinstance(b, (int, float)) or isinstance(b, float) and isinstance(a, (int, float)):
        if math.isnan(a) and math.isnan(b):
            return True
        return a == b or abs(a - b) <= 1e-12 * max(abs(a), abs(b))
    if type(a) is not type(b):
        return False
    if isinstance(a, dict):
        return a.keys() == b.keys() and all(close(a[k], b[k]) for k in a)
    if isinstance(a, list):
        return len(a) == len(b) and all(close(x, y) for x, y in zip(a, b))
    return a == b


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*", default=None)
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--no-check", action="store_true")
    a = ap.parse_args()
    T = json.load(open(ROOT / "tools" / "targets.json"))
    runs = {k: v for k, v in T["runs"].items() if v.get("default", True) or (a.only and k in a.only)}
    if a.only:
        runs = {k: v for k, v in runs.items() if k in a.only}
    if a.list:
        for k, v in T["runs"].items():
            print(f"{k:22s} {v['command']}" + ("" if v.get("default", True) else "   (not run by default)"))
        return
    if not any((ROOT / "outputs").rglob("*.jsonl")) and any((ROOT / "outputs").rglob("*.gz")):
        subprocess.check_call([sys.executable, str(ROOT / "tools" / "unpack_outputs.py")])
    (REGEN / "logs").mkdir(parents=True, exist_ok=True)
    keep = REGEN / "bundle_shipped"
    if not keep.exists():
        shutil.copytree(ROOT / "outputs" / "bundle", keep)
    env = dict(os.environ, VTAX_ROOT=str(ROOT), VTAX_REGEN=str(REGEN), MPLBACKEND="Agg", PYTHONHASHSEED="0",
               CUDA_VISIBLE_DEVICES="")
    status = {}
    for rid, r in runs.items():
        t0 = time.time()
        with open(REGEN / "logs" / f"{rid}.log", "w") as log:
            rc = subprocess.call([sys.executable] + r["argv"], cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
        status[rid] = {"exit": rc, "seconds": round(time.time() - t0, 1)}
        print(f"{rid:22s} exit={rc} {status[rid]['seconds']:7.1f}s   {r['command']}", flush=True)
    bcheck = {}
    for p in sorted((ROOT / "outputs" / "bundle").glob("*.json")):
        q = keep / p.name
        if q.is_file() and p.read_bytes() != q.read_bytes():
            try:
                x, y = json.load(open(p)), json.load(open(q))
                bcheck[p.name] = "JSON-equal" if x == y else "equal up to float noise / NaN" if close(x, y) else "differs"
            except json.JSONDecodeError:
                bcheck[p.name] = "differs"
    json.dump({"runs": status, "rebuilt_bundle_files_vs_shipped": bcheck}, open(REGEN / "bundle_check.json", "w"), indent=1)
    if bcheck:
        print("bundle files rewritten by the runs (vs shipped copy):", bcheck)
    if not a.no_check:
        raise SystemExit(subprocess.call([sys.executable, str(ROOT / "tools" / "check_tables.py")], cwd=ROOT, env=env))
    raise SystemExit(1 if any(s["exit"] for s in status.values()) else 0)


if __name__ == "__main__":
    main()
