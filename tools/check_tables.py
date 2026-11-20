"""Compare regenerated tables / figure data with the copies the paper was compiled from (reference/).

Tables: every number printed in the paper table (LaTeX comments and layout arguments removed; signs ignored, so
``$-5.6$`` and ``-5.6`` match) must occur in the regenerated table as often as in the paper copy (several paper
tables are condensed versions of the generated ones: columns or rows were deleted, cells never edited). For
hand-built tables and as a second level, numbers are also looked up among the numeric leaves of the bundle files
the table is built from, rounded to the printed number of decimals. Figures: ``<name>.data.json`` (the values each
figure plots) must be JSON-equal to the reference copy.

Usage: python tools/check_tables.py [--json regenerated/check_report.json]
"""
from __future__ import annotations

import argparse
import json
import os
import re
from collections import Counter
from pathlib import Path

ROOT = Path(os.environ.get("VTAX_ROOT", Path(__file__).resolve().parents[1]))
REGEN = Path(os.environ.get("VTAX_REGEN", ROOT / "regenerated"))
LAYOUT = re.compile(r"\\(?:multicolumn|multirow|cmidrule|cline|hspace|vspace|rule|resizebox|scalebox|fitto|setlength|"
                    r"renewcommand|arraystretch|tabcolsep|begin|end|rowcolor|cellcolor|color|textcolor)\s*"
                    r"(?:\([^)]*\))?(?:\{[^{}]*\}|\[[^\]]*\])*")
NUM = re.compile(r"(?<![\w.])\d+(?:\.\d+)?(?![\w])")


def numbers(tex: str) -> Counter:
    """Multiset of unsigned numbers printed in a LaTeX table."""
    lines = []
    for line in tex.splitlines():
        line = re.sub(r"(?<!\\)%.*$", "", line)
        if line.strip():
            lines.append(line)
    t = "\n".join(lines).replace("{,}", "").replace("\\,", " ")
    t = LAYOUT.sub(" ", t)
    return Counter(NUM.findall(t))


def leaves(x, out: list) -> None:
    if isinstance(x, bool):
        return
    if isinstance(x, (int, float)):
        out.append(float(x))
    elif isinstance(x, dict):
        for v in x.values():
            leaves(v, out)
    elif isinstance(x, list):
        for v in x:
            leaves(v, out)


def bundle_strings(files: list[str]) -> set[str]:
    """Numeric leaves of bundle files, formatted with 0-3 decimals (also x100 for shares)."""
    vals: list[float] = []
    for f in files:
        p = ROOT / f
        if p.is_file() and p.suffix == ".json":
            try:
                leaves(json.load(open(p)), vals)
            except (json.JSONDecodeError, UnicodeDecodeError):
                pass
    out = set()
    for v in vals:
        for s in (v, 100 * v):
            a = abs(s)
            if a > 1e7:
                continue
            for d in range(4):
                out.add(f"{a:.{d}f}")
    return out


def plotted(x):
    """Figure data without provenance strings (keys ending in 'path' / 'source'), i.e. the plotted values."""
    if isinstance(x, dict):
        return {k: plotted(v) for k, v in x.items()
                if not (isinstance(v, str) and (k.endswith("path") or k in ("source", "sources")))}
    if isinstance(x, list):
        return [plotted(v) for v in x]
    return x


def check_table(ref: Path, regen: Path | None, bundle: list[str]) -> dict:
    want = numbers(ref.read_text())
    res: dict = {"numbers_in_paper_table": sum(want.values())}
    have = numbers(regen.read_text()) if regen is not None and regen.is_file() else None
    if regen is not None:
        res["regenerated"] = regen.is_file()
        if have is not None:
            res["byte_identical"] = regen.read_bytes() == ref.read_bytes()
            missing = want - have
            res["found_in_regenerated"] = sum(want.values()) - sum(missing.values())
            res["missing_from_regenerated"] = sorted(missing.elements(), key=lambda s: (float(s), s))
    if bundle:
        bs = bundle_strings(bundle)
        rest = Counter(res.get("missing_from_regenerated", list(want.elements())))
        miss2 = [s for s in rest.elements() if s not in bs]
        res["missing_from_regenerated_and_bundle"] = sorted(miss2, key=lambda s: (float(s), s))
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default=str(REGEN / "check_report.json"))
    a = ap.parse_args()
    T = json.load(open(ROOT / "tools" / "targets.json"))
    report = []
    for it in T["items"]:
        ref = ROOT / it["reference"] if it.get("reference") else None
        row = {"number": it["number"], "label": it["label"], "generator": it.get("command") or "hand-built"}
        if it["kind"] == "table" and ref is not None:
            regen = REGEN / it["regenerated"] if it.get("command") else None
            row.update(check_table(ref, regen, it.get("bundle", [])))
            if regen is None and not it.get("bundle"):
                row["status"] = "hand-built text table (numbers not checked)"
            elif regen is None:
                row["status"] = ("hand-built; all numbers found in bundle" if not row.get("missing_from_regenerated_and_bundle")
                                 else "hand-built; some numbers not found in bundle")
            elif not row.get("regenerated"):
                row["status"] = "NOT REGENERATED"
            elif row.get("byte_identical"):
                row["status"] = "identical"
            elif not row["missing_from_regenerated"]:
                row["status"] = "all paper numbers reproduced"
            elif not row.get("missing_from_regenerated_and_bundle"):
                row["status"] = "all paper numbers in regenerated table or bundle"
            else:
                row["status"] = "MISMATCH"
        elif it["kind"] == "figure":
            regen_data = REGEN / it["regenerated_data"] if it.get("regenerated_data") else None
            regen_pdf = REGEN / it["regenerated"] if it.get("regenerated") else None
            if it.get("reference_data") and regen_data is not None:
                rd = ROOT / it["reference_data"]
                if not regen_data.is_file():
                    row["status"] = "NOT REGENERATED"
                else:
                    new, ref = json.load(open(regen_data)), json.load(open(rd))
                    row["status"] = ("data.json identical" if new == ref else
                                     "data.json identical (plotted values; provenance paths differ)"
                                     if plotted(new) == plotted(ref) else "MISMATCH")
            else:
                row["status"] = ("regenerated (no data.json; compare visually)" if regen_pdf is not None and regen_pdf.is_file()
                                 else "not regenerated" + (f" ({it['note']})" if it.get("note") else ""))
        else:
            row["status"] = "hand-built (text only)"
        report.append(row)
        extra = ""
        if "numbers_in_paper_table" in row:
            extra = f"  numbers {row.get('found_in_regenerated', '-')}/{row['numbers_in_paper_table']}"
            m = row.get("missing_from_regenerated_and_bundle", row.get("missing_from_regenerated"))
            if m:
                extra += f"  missing {m[:12]}"
        print(f"{it['number']:9s} {it['label']:22s} {row['status']}{extra}")
    Path(a.json).parent.mkdir(parents=True, exist_ok=True)
    json.dump(report, open(a.json, "w"), indent=1)
    bad = [r for r in report if r["status"] in ("MISMATCH", "NOT REGENERATED")]
    print(f"\n{len(report) - len(bad)}/{len(report)} items OK; report -> {a.json}")
    raise SystemExit(1 if bad else 0)


if __name__ == "__main__":
    main()
