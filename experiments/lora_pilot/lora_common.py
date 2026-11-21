"""Exp-63 shared helpers: paths, model keys of the fine-tuned checkpoints, training / evaluation item loaders.

Model keys (all start with ``qwen3vl`` so vx.make_llm keeps its Qwen3-VL handling: V1 engine, top-20 readout,
max_pixels = 1280*28*28):
  qwen3vl_8b            base (vx)
  qwen3vl_8b_e63match   merged MATCH-arm checkpoint, path from env E63_CKPT_MATCH (default /dev/shm/e63_match)
  qwen3vl_8b_e63cf      merged CF-arm checkpoint,    path from env E63_CKPT_CF    (default /dev/shm/e63_cf)

Evaluation sets (``load_eval_set``), each item in the protocol schema (question with option lines, choices, gold,
group = bootstrap cluster) plus ``cf`` (counterfactual image spec), ``base`` (stored base-model R and readouts):
  main     2,850-item main binary pool (vx.load_items; base R / Direct / CoT from outputs/e4/qwen3vl_8b.json)
  heldout  767 held-out natural items (e6x.load_heldout; base from outputs/e6x/qwen3vl_8b_{direct,rest}.jsonl)
  cues     cue-factorial occlusion twins, both phrasings (outputs/e50/occ_cues; base from e10/qwen3vl_8b_cues.jsonl)
  occ      rendered occlusion twins, both phrasings (data/items/occlusion; base from e10/qwen3vl_8b.jsonl, as e14)
  coco     COCO natural-object occlusion twins (outputs/e62/stimuli/items_final.jsonl; base from e62/qwen3vl_8b.jsonl)
  e60      mirrored CV-Bench 2D left/right, kept items of the Exp-60 filter (base R = e60_items.stored_r)
"""
from __future__ import annotations
from srww import paths as _vtp  # noqa: E402

import os

_DEV_SHARE = f"{_vtp.DATA_ROOT}"
_POD_SHARE = f"{_vtp.DATA_ROOT}"
if "SHARE" not in os.environ and not os.path.isdir(f"{_POD_SHARE}/code/vtax"):
    os.environ["VTAX_DATA_ROOT"] = _DEV_SHARE

from srww import vllm_utils as vx  # noqa: E402
from srww.vllm_utils import C  # noqa: E402

from srww.rcut import strip_answer  # noqa: E402,F401

E63 = f"{vx.OUT}/e63"
STIM = f"{E63}/train_stimuli"
SEED = 63
MAX_PIXELS = 1280 * 28 * 28
BASE = "qwen3vl_8b"
ARMS = ("match", "cf")
MODEL_OF_ARM = {"match": "qwen3vl_8b_e63match", "cf": "qwen3vl_8b_e63cf"}
EVAL_MODELS = (BASE, MODEL_OF_ARM["match"], MODEL_OF_ARM["cf"])
EVAL_SETS = ("main", "heldout", "cues", "occ", "coco", "e60")
TWIN_SETS = ("cues", "occ", "coco", "e60")

for _arm, _key in MODEL_OF_ARM.items():
    C.MODELS[_key] = os.environ.get(f"E63_CKPT_{_arm.upper()}", f"/dev/shm/e63_{_arm}")


def fix_path(p: str) -> str:
    """Map a stored data-root path to the current VTAX_DATA_ROOT when it does not exist here."""
    if os.path.exists(p):
        return p
    for a, b in ((_POD_SHARE, vx.SHARE), (_DEV_SHARE, vx.SHARE)):
        if p.startswith(a) and os.path.exists(b + p[len(a):]):
            return b + p[len(a):]
    return p


def r_cut(r: str) -> str:
    """The paper's R_cut (srww.rcut.strip_answer)."""
    return strip_answer(r)


# ----------------------------------------------------------------------------- training stimuli

def load_train_items(limit_pairs: int | None = None) -> list[dict]:
    """Training items (one per image x question); ``image_path`` resolved against the current share mount."""
    items = vx.read_jsonl(f"{STIM}/items.jsonl")
    for it in items:
        it["image_path"] = f"{STIM}/{it['image_rel']}"
    if limit_pairs:
        keep = set(sorted({it["pair"] for it in items if it["kind"] == "depth"})[:limit_pairs]
                   + sorted({it["pair"] for it in items if it["kind"] == "lr"})[:limit_pairs])
        items = [it for it in items if it["pair"] in keep]
    return items


# ----------------------------------------------------------------------------- evaluation sets

def _pred(scores: dict | None) -> str | None:
    """Argmax letter of a score dict (None / -inf ignored)."""
    import math
    fin = {k: v for k, v in (scores or {}).items() if v is not None and math.isfinite(v)}
    return max(fin, key=fin.get) if fin else None


def pe(r: dict) -> str | None:
    """Prediction of a readout record: argmax lp_sum (exact) if present else top-20 (= srww.records.pe)."""
    return _pred(r.get("lp_sum") or r.get("top20"))


def twin_of(k: str) -> str:
    """occ001 <-> occ001_cf, keeping an optional _behind suffix."""
    base, beh = (k[:-7], "_behind") if k.endswith("_behind") else (k, "")
    return (base[:-3] if base.endswith("_cf") else base + "_cf") + beh


def _base_block(r_text: str | None, d_ok: bool | None, c_ok: bool | None) -> dict:
    return {"R": r_text, "rcut": r_cut(r_text) if r_text else None, "direct_ok": d_ok, "cot_ok": c_ok}


def _load_main(limit: int | None) -> list[dict]:
    e4 = {it["uid"]: it for it in vx.legacy_e1(BASE).values()}
    items = vx.load_items(limit=limit)
    for it in items:
        s = e4.get(it["uid"])
        it["set"] = "main"
        it["cf"] = None
        it["base"] = _base_block(s["cot"] if s else None, _pred(s["scores"]["direct"]) == it["gold"] if s else None,
                                 _pred(s["scores"]["cot"]) == it["gold"] if s else None)
    return items


def _load_heldout(limit: int | None) -> list[dict]:
    from experiments.behavior.heldout_benchmarks import load_heldout
    d = {r["item_id"]: r for r in vx.read_jsonl(f"{vx.OUT}/e6x/{BASE}_direct.jsonl")}
    c = {r["item_id"]: r for r in vx.read_jsonl(f"{vx.OUT}/e6x/{BASE}_rest.jsonl") if r["condition"] == "cot"}
    items = load_heldout(limit)
    for it in items:
        it["set"], it["cf"] = "heldout", None
        dr, cr = d.get(it["uid"]), c.get(it["uid"])
        it["base"] = _base_block(cr["text"] if cr else None, pe(dr) == it["gold"] if dr else None,
                                 pe(cr) == it["gold"] if cr else None)
    return items


def _load_twins(name: str, limit: int | None) -> list[dict]:
    """Rendered occlusion twins (occ) or cue-factorial twins (cues), both phrasings; key = uid[_behind]."""
    from experiments.occlusion.twins_readout import load_occ
    items_dir = f"{vx.OUT}/e50/occ_cues" if name == "cues" else None
    stored = f"{vx.OUT}/e10/{BASE}{'_cues' if name == 'cues' else ''}.jsonl"
    items = [dict(it, phrasing="front", key=it["uid"]) for it in load_occ(limit, items_dir=items_dir)]
    items += [dict(it, phrasing="behind", key=it["uid"] + "_behind") for it in load_occ(limit, polar=True, items_dir=items_dir)]
    recs = {(r["item_id"], r["condition"]): r for r in vx.read_jsonl(stored) if r["task"].startswith("occ")}
    by_key = {it["key"]: it for it in items}
    out = []
    for it in items:
        tk = twin_of(it["key"])
        if tk not in by_key:
            continue
        dr, cr = recs.get((it["key"], "direct")), recs.get((it["key"], "cot"))
        out.append(dict(it, uid=it["key"], base_uid=it["uid"], set=name,
                        cf={"kind": "twin", "uid": tk, "image_path": by_key[tk]["image_path"], "gold": by_key[tk]["gold"]},
                        base=_base_block(cr["text"] if cr else None, pe(dr) == it["gold"] if dr else None,
                                         pe(cr) == it["gold"] if cr else None)))
    return out


def _load_coco(limit: int | None) -> list[dict]:
    items = vx.read_jsonl(f"{vx.OUT}/e62/stimuli/items_final.jsonl")
    if limit:
        keep = sorted({it["pair"] for it in items})[:limit]
        items = [it for it in items if it["pair"] in keep]
    by_uid = {it["uid"]: it for it in items}
    recs = {(r["uid"], r["arm"], r["img"]): r for r in vx.read_jsonl(f"{vx.OUT}/e62/{BASE}.jsonl")}
    out = []
    for it in items:
        it = dict(it, image_path=fix_path(it["image_path"]), set="coco", group=it["pair"])
        tw = by_uid.get(it["twin_item"])
        if tw is None:
            continue
        dr, cr = recs.get((it["uid"], "direct", "v")), recs.get((it["uid"], "cot", "v"))
        it["cf"] = {"kind": "twin", "uid": tw["uid"], "image_path": fix_path(tw["image_path"]), "gold": tw["gold"]}
        it["base"] = _base_block(cr["text"] if cr else None, pe(dr) == it["gold"] if dr else None,
                                 pe(cr) == it["gold"] if cr else None)
        if cr is not None:
            assert it["base"]["rcut"] == cr["rcut"], it["uid"]
        out.append(it)
    return out


def _load_e60(limit: int | None) -> list[dict]:
    from data_gen.mirrored_cvbench import E60, candidates, stored_r
    keep = {r["uid"] for r in vx.read_jsonl(f"{E60}/items.jsonl") if r["keep"] and r["set"] == "cvb"}
    items = [c for c in candidates(with_images=True) if c["uid"] in keep]
    if limit:
        items = items[:limit]
    rs = stored_r(BASE)
    recs = {(r["item_id"], r["img"], r["ctx"]): r for r in vx.read_jsonl(f"{E60}/{BASE}.jsonl")}
    for it in items:
        it["set"] = "e60"
        it["cf"] = {"kind": "mirror", "gold": it["gold_mirror"]}
        dr, fr = recs.get((it["uid"], "orig", "direct")), recs.get((it["uid"], "orig", "full"))
        it["base"] = _base_block(rs.get(it["uid"]), pe(dr) == it["gold"] if dr else None,
                                 pe(fr) == it["gold"] if fr else None)
    return items


def load_eval_set(name: str, limit: int | None = None) -> list[dict]:
    """One evaluation set in the common schema (see module docstring)."""
    return {"main": _load_main, "heldout": _load_heldout, "cues": lambda n: _load_twins("cues", n),
            "occ": lambda n: _load_twins("occ", n), "coco": _load_coco, "e60": _load_e60}[name](limit)


_IMG: dict[str, object] = {}
_BYTES: dict[str, bytes] = {}


def load_image(path: str):
    """PIL RGB image by path, cached (network storage reads are latency-bound; see prefetch)."""
    import io
    from PIL import Image
    if path in _BYTES:
        return Image.open(io.BytesIO(_BYTES[path])).convert("RGB")
    if path not in _IMG:
        _IMG[path] = Image.open(path).convert("RGB")
    return _IMG[path]


def prefetch(paths, workers: int = 32, as_bytes: bool = False) -> None:
    """Read many image files in parallel threads (~20x faster than serial reads on network storage); ``as_bytes`` keeps
    the encoded file bytes (decoded on use; small RAM footprint for the 5,040 training PNGs)."""
    from concurrent.futures import ThreadPoolExecutor

    def rd(p: str) -> None:
        if as_bytes:
            with open(p, "rb") as f:
                _BYTES[p] = f.read()
        else:
            load_image(p)
    todo = sorted({p for p in paths if p and p not in _IMG and p not in _BYTES})
    with ThreadPoolExecutor(workers) as ex:
        list(ex.map(rd, todo))


def item_image(it: dict):
    """Original image of an item (in-memory image or cached file)."""
    return it["image"] if "image" in it else load_image(it["image_path"])


def image_of(it: dict, img: str):
    """PIL image of an eval item under v | cf | gray (cf = twin image or horizontal mirror)."""
    from PIL import Image, ImageOps
    if img == "v":
        return item_image(it)
    if img == "gray":
        return Image.new("RGB", item_image(it).size, vx.GRAY)
    if it["cf"]["kind"] == "mirror":
        return ImageOps.mirror(item_image(it))
    return load_image(it["cf"]["image_path"])
