"""Shared utilities for the short-CoT spatial VQA experiments.

This module is importable without torch so that CPU-only analysis (E0) and
smoke tests can run on machines without a GPU stack. Model adapters import
torch / transformers lazily.

Protocol constants follow the paper's implementation and prompt specifications. The vLLM runners use
``srww.protocol`` (the protocol module of the original runs) through ``srww.vllm_utils``; items marked ``VERIFY``
were not checked against it.
"""

from __future__ import annotations

import dataclasses
import datetime as _dt
import hashlib
import io
import itertools
import json
import math
import os
import random
import re
import subprocess
import sys
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator, Sequence

import numpy as np

# ---------------------------------------------------------------------------
# Protocol constants (verbatim from the paper appendix)
# ---------------------------------------------------------------------------

COT_TRIGGER = "Let's think step by step."
ANSWER_PREFIX = "Answer: ("
TURN_PROMPT = "Now give your final answer."
RESTATE_TEMPLATE = "Question again: {q}\nNow give your final answer."
REIMAGE_TEXT = "Here is the image again.\nNow give your final answer."
# VERIFY: the paper does not give the exact combined re-image + restate text.
REIMAGE_RESTATE_TEMPLATE = "Here is the image again.\nQuestion again: {q}\nNow give your final answer."
DIRECT_RESTATE_TEMPLATE = "{q}\n\nQuestion again: {q}\nNow give your final answer."
# E2c continuation prompts (new in this study; not part of the original v2 protocol).
REIMAGE_CONTINUE_TEXT = "Here is the image again.\nContinue your reasoning."
TEXT_CONTINUE_TEXT = "Continue your reasoning."

FILLER_SENTENCES: tuple[str, ...] = (
    "The history of paper making spans many centuries and several cultures.",
    "Early writers recorded ideas on bark, cloth, clay and animal skin.",
    "Later, workers learned to soak plant fibers and press them into thin sheets.",
    "The method spread slowly through trade and the movement of skilled craftsmen.",
    "Mills powered by water made production faster and cheaper.",
    "Printed books became common, and literacy rates rose over time.",
    "Today most paper is produced from wood pulp in large industrial plants.",
    "Recycling programs recover a large share of used paper every year.",
    "Some artists still prefer handmade sheets for their texture and strength.",
    "Libraries preserve old documents by controlling humidity and temperature.",
    "Scholars study watermarks to estimate the age of historical manuscripts.",
    "The chemistry of acid-free paper helps documents last for generations.",
)

LETTERS = "ABCDEFGH"
GRAY_RGB = (128, 128, 128)
MAX_COT_TOKENS = 512
QWEN25_MAX_PIXELS = 1280 * 28 * 28

# Free-text option parsing: three ordered patterns, last match wins (appendix).
_PARSE_PATTERNS = (
    re.compile(r"answer is\s*\(?([A-H])\)?", re.IGNORECASE),
    re.compile(r"\(([A-H])\)"),
    re.compile(r"\b([A-H])(?=[.):])"),
)


# ---------------------------------------------------------------------------
# Items
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class Item:
    """One multiple-choice spatial VQA item.

    Attributes:
        item_id: Stable identifier (must match legacy outputs for joins).
        task: Task name, e.g. ``cvb_relation``, ``syn_depth``.
        image_paths: Paths of the input images (usually one).
        question: Question text without options.
        options: Option strings in letter order.
        gold: Gold option letter.
        image_id: Cluster id for image-grouped bootstrap / CV.
        prompt: Optional pre-formatted question+options string; when set it is
            used verbatim as ``q``.
        meta: Free-form metadata (bboxes, generator state, source fields).
    """

    item_id: str
    task: str
    image_paths: list[str]
    question: str
    options: list[str]
    gold: str
    image_id: str = ""
    prompt: str | None = None
    meta: dict[str, Any] = dataclasses.field(default_factory=dict)

    @property
    def letters(self) -> list[str]:
        """Option letters in order."""
        return list(LETTERS[: len(self.options)])

    @property
    def gold_index(self) -> int:
        """Index of the gold option."""
        return LETTERS.index(self.gold)

    def q_text(self) -> str:
        """Return the question text ``q`` with lettered options.

        VERIFY: the original formatting of ``q`` (CV-Bench ``prompt`` field vs.
        question + option lines) must be matched to `scripts/v2/common.py`.
        """
        if self.prompt:
            return self.prompt
        lines = [self.question] + [f"({l}) {o}" for l, o in zip(self.letters, self.options)]
        return "\n".join(lines)

    def to_json(self) -> dict[str, Any]:
        """Serialize to a JSON-compatible dict."""
        return dataclasses.asdict(self)

    @classmethod
    def from_json(cls, d: dict[str, Any]) -> "Item":
        """Build an item from a JSON dict."""
        fields = {f.name for f in dataclasses.fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in fields})


def load_items(path: str | os.PathLike, tasks: Sequence[str] | None = None,
               limit: int | None = None) -> list[Item]:
    """Load an item manifest JSONL, resolving relative image paths.

    Args:
        path: Item manifest (JSONL of ``Item`` records).
        tasks: Optional task filter.
        limit: Optional per-manifest cap (smoke runs).

    Returns:
        List of items.
    """
    base = Path(path).resolve().parent
    items: list[Item] = []
    for rec in read_jsonl(path):
        it = Item.from_json(rec)
        it.image_paths = [p if os.path.isabs(p) else str(base / p) for p in it.image_paths]
        if tasks and it.task not in tasks:
            continue
        items.append(it)
        if limit and len(items) >= limit:
            break
    return items


# ---------------------------------------------------------------------------
# JSONL I/O
# ---------------------------------------------------------------------------


def read_jsonl(path: str | os.PathLike) -> Iterator[dict[str, Any]]:
    """Yield records from a JSONL file, skipping blank lines."""
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


class JsonlWriter:
    """Append-only JSONL writer with resume support.

    Records are flushed after each write so a crashed run keeps completed
    items. ``done_keys`` lets callers skip work that is already on disk.
    """

    def __init__(self, path: str | os.PathLike, key_fields: Sequence[str]) -> None:
        """Open ``path`` for appending.

        Args:
            path: Output JSONL path (parent directories are created).
            key_fields: Fields that identify a record for resume.
        """
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.key_fields = tuple(key_fields)
        self.done_keys: set[tuple] = set()
        if self.path.exists():
            for rec in read_jsonl(self.path):
                self.done_keys.add(self.key(rec))
        self._f = open(self.path, "a", encoding="utf-8")

    def key(self, rec: dict[str, Any]) -> tuple:
        """Return the resume key of a record."""
        return tuple(json.dumps(rec.get(k), sort_keys=True) for k in self.key_fields)

    def has(self, **kw: Any) -> bool:
        """Check whether a record with these key values was written."""
        return self.key(kw) in self.done_keys

    def write(self, rec: dict[str, Any]) -> None:
        """Write and flush one record."""
        self._f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        self._f.flush()
        self.done_keys.add(self.key(rec))

    def close(self) -> None:
        """Close the file handle."""
        self._f.close()

    def __enter__(self) -> "JsonlWriter":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()


def sha256_file(path: str | os.PathLike, chunk: int = 1 << 22) -> str:
    """Compute the sha256 hex digest of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def utc_stamp() -> str:
    """Return a compact UTC timestamp, e.g. ``20261112T004500Z``."""
    return _dt.datetime.now(_dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def git_revision(path: str | os.PathLike = ".") -> str | None:
    """Return ``git rev-parse HEAD`` (plus ``-dirty``) for ``path`` if available."""
    try:
        rev = subprocess.run(["git", "-C", str(path), "rev-parse", "HEAD"], capture_output=True,
                             text=True, timeout=10).stdout.strip()
        if not rev:
            return None
        dirty = subprocess.run(["git", "-C", str(path), "status", "--porcelain"], capture_output=True,
                               text=True, timeout=10).stdout.strip()
        return rev + ("-dirty" if dirty else "")
    except (OSError, subprocess.SubprocessError):
        return None


def safe_torch_load(path: str | os.PathLike, map_location: str = "cpu") -> Any:
    """Load a ``.pt`` file with ``weights_only=True`` (never unpickle code)."""
    import torch

    return torch.load(path, map_location=map_location, weights_only=True)


# ---------------------------------------------------------------------------
# Image transforms
# ---------------------------------------------------------------------------


def load_image(path: str) -> "Image.Image":
    """Open an image as RGB."""
    from PIL import Image

    with Image.open(path) as im:
        return im.convert("RGB")


def gray_image(img: "Image.Image") -> "Image.Image":
    """Return a constant mid-gray image with the same resolution (image ablation)."""
    from PIL import Image

    return Image.new("RGB", img.size, GRAY_RGB)


def mirror_image(img: "Image.Image") -> "Image.Image":
    """Return the horizontally mirrored image."""
    from PIL import ImageOps

    return ImageOps.mirror(img)


_LR_SWAP = {"left": "right", "right": "left"}


def lr_option_kind(item: Item) -> str | None:
    """Classify how a horizontal mirror changes the item's answer.

    Returns:
        ``"options"`` when the options are {left, right} (mirror swaps gold),
        ``"question"`` when the question asks a yes/no left/right relation
        (mirror flips yes/no), or ``None`` when mirroring should not change the
        answer (depth, distance, above/below).
    """
    opts = [o.strip().lower() for o in item.options]
    if sorted(opts) == ["left", "right"]:
        return "options"
    q = item.question.lower()
    if sorted(opts) == ["no", "yes"] and re.search(r"\b(left|right)\b", q):
        return "question"
    return None


def mirrored_gold(item: Item) -> str:
    """Gold letter for the horizontally mirrored image.

    For left/right items the answer flips; for other items it is unchanged.
    """
    kind = lr_option_kind(item)
    if kind is None:
        return item.gold
    if kind == "options":
        g = item.options[item.gold_index].strip().lower()
        target = _LR_SWAP[g]
        return item.letters[[o.strip().lower() for o in item.options].index(target)]
    # yes/no question about left/right: flips
    return item.letters[1 - item.gold_index]


def variant_images(item: Item, variant: str) -> list["Image.Image"] | None:
    """Return the item's images under an image variant.

    Args:
        item: The item.
        variant: ``original``, ``gray``, ``mirror``, or ``none`` (no image).

    Returns:
        List of images, or ``None`` for the no-image condition.
    """
    if variant == "none":
        return None
    imgs = [load_image(p) for p in item.image_paths]
    if variant == "original":
        return imgs
    if variant == "gray":
        return [gray_image(im) for im in imgs]
    if variant == "mirror":
        return [mirror_image(im) for im in imgs]
    raise ValueError(f"unknown image variant {variant}")


def image_hash(img_bytes: bytes) -> str:
    """Short content hash used as image cluster id."""
    return hashlib.sha1(img_bytes).hexdigest()[:16]


# ---------------------------------------------------------------------------
# Prompt construction (model-agnostic)
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class Turn:
    """One chat turn. ``parts`` mixes PIL images and strings in order."""

    role: str
    parts: list[Any]


@dataclasses.dataclass
class Prompt:
    """A rendered-agnostic prompt: closed turns plus an open assistant prefix."""

    turns: list[Turn]
    assistant_prefix: str

    def n_images(self) -> int:
        """Number of image parts across turns."""
        return sum(1 for t in self.turns for p in t.parts if not isinstance(p, str))

    def text_signature(self) -> str:
        """Human-readable string with ``<image>`` markers (for checks/logging)."""
        out = []
        for t in self.turns:
            body = "".join("<image>" if not isinstance(p, str) else p for p in t.parts)
            out.append(f"[{t.role}] {body}")
        out.append(f"[assistant-open] {self.assistant_prefix}")
        return "\n".join(out)


def filler_text(n_tokens: int, encode: Callable[[str], list[int]],
                decode: Callable[[list[int]], str]) -> str:
    """Neutral filler truncated to exactly ``n_tokens`` under a tokenizer.

    Args:
        n_tokens: Target token count.
        encode: Tokenizer encode (no special tokens).
        decode: Tokenizer decode.

    Returns:
        Filler prose whose re-encoding has ``n_tokens`` tokens (up to BPE
        boundary effects, which are reported by callers).
    """
    if n_tokens <= 0:
        return ""
    text = " ".join(FILLER_SENTENCES)
    while len(encode(text)) < n_tokens + 8:
        text = text + " " + " ".join(FILLER_SENTENCES)
    return decode(encode(text)[:n_tokens])


def cot_content(rationale: str) -> str:
    """Assistant text for CoT: trigger followed by the generated continuation.

    VERIFY: assumes ``R`` was generated as a continuation of an assistant
    prefix equal to ``COT_TRIGGER`` and stored without the trigger.
    """
    r = rationale
    if r.startswith(COT_TRIGGER):
        r = r[len(COT_TRIGGER):]
    return (COT_TRIGGER + r).rstrip()


# Condition registry. Each condition maps to (intermediate kind, layout, second-turn kind).
#   kind: none | cot | filler_R | filler_512 | filler_1024
#   layout: same (continue assistant turn) | turn (close turn, add user turn)
#   second: turn | restate | reimage | reimage_restate
CONDITIONS: dict[str, tuple[str, str, str]] = {
    "direct": ("none", "same", ""),
    "direct_restate": ("none", "restate1", ""),
    "cot": ("cot", "same", ""),
    "cot_turn": ("cot", "turn", "turn"),
    "cot_restate": ("cot", "turn", "restate"),
    "cot_reimage": ("cot", "turn", "reimage"),
    "cot_reimage_restate": ("cot", "turn", "reimage_restate"),
    "filler": ("filler_R", "same", ""),
    "filler_turn": ("filler_R", "turn", "turn"),
    "f512": ("filler_512", "same", ""),
    "f512_turn": ("filler_512", "turn", "turn"),
    "f512_restate": ("filler_512", "turn", "restate"),
    "f512_reimage": ("filler_512", "turn", "reimage"),
    "f1024": ("filler_1024", "same", ""),
    "f1024_turn": ("filler_1024", "turn", "turn"),
}


def second_turn(kind: str, q: str, images: list[Any] | None) -> Turn:
    """Build the second user turn for extra-turn conditions."""
    if kind == "turn":
        return Turn("user", [TURN_PROMPT])
    if kind == "restate":
        return Turn("user", [RESTATE_TEMPLATE.format(q=q)])
    if kind in ("reimage", "reimage_restate"):
        if not images:
            raise ValueError("re-image requires images")
        text = REIMAGE_TEXT if kind == "reimage" else REIMAGE_RESTATE_TEMPLATE.format(q=q)
        return Turn("user", list(images) + [text])
    raise ValueError(kind)


def first_user_turn(q: str, images: list[Any] | None) -> Turn:
    """First user turn ``[<image>..., q]`` shared by all conditions."""
    return Turn("user", (list(images) if images else []) + [q])


def build_condition_prompt(condition: str, q: str, images: list[Any] | None,
                           intermediate: str = "") -> Prompt:
    """Build the prompt for a registered condition.

    Args:
        condition: Key of ``CONDITIONS``.
        q: Question text with lettered options.
        images: Images for the first user turn (``None`` = no image).
        intermediate: Assistant text ``C`` (CoT content or filler) when the
            condition uses one.

    Returns:
        Prompt ending with the open assistant prefix ``Answer: (``.
    """
    kind, layout, second = CONDITIONS[condition]
    if layout == "restate1":
        return Prompt([first_user_turn(DIRECT_RESTATE_TEMPLATE.format(q=q), images)], ANSWER_PREFIX)
    user1 = first_user_turn(q, images)
    if kind == "none":
        return Prompt([user1], ANSWER_PREFIX)
    if layout == "same":
        return Prompt([user1], intermediate.rstrip() + "\n" + ANSWER_PREFIX)
    return Prompt([user1, Turn("assistant", [intermediate.rstrip()]), second_turn(second, q, images)],
                  ANSWER_PREFIX)


def build_cot_generation_prompt(q: str, images: list[Any] | None) -> Prompt:
    """Prompt for generating ``R`` (greedy continuation of the CoT trigger)."""
    return Prompt([first_user_turn(q, images)], COT_TRIGGER)


def build_closed_readout(turns: list[Turn], assistant_text: str,
                         final_user: str = TURN_PROMPT) -> Prompt:
    """Close ``assistant_text`` as a turn, add a user turn, and open ``Answer: (``."""
    return Prompt(list(turns) + [Turn("assistant", [assistant_text.rstrip()]), Turn("user", [final_user])],
                  ANSWER_PREFIX)


# ---------------------------------------------------------------------------
# Rationale text utilities
# ---------------------------------------------------------------------------

_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z(\"'])|\n+")
_ENUM_ONLY = re.compile(r"^\s*(\d+|[a-z])[.)]\s*$")


def split_sentences(text: str) -> list[str]:
    """Split a rationale into sentences, keeping list markers attached.

    Returns sentences whose concatenation with single spaces approximates the
    original text; empty pieces are removed.
    """
    raw = [s.strip() for s in _SENT_SPLIT.split(text) if s and s.strip()]
    out: list[str] = []
    for s in raw:
        if out and _ENUM_ONLY.match(out[-1]):
            out[-1] = out[-1] + " " + s
        else:
            out.append(s)
    return out


def parse_stated_option(text: str, n_options: int) -> str | None:
    """Parse the option letter stated in free text (appendix rule).

    Applies the three ordered patterns and returns the last match of the first
    pattern that matches at all.
    """
    valid = set(LETTERS[:n_options])
    for pat in _PARSE_PATTERNS:
        hits = [m.group(1).upper() for m in pat.finditer(text) if m.group(1).upper() in valid]
        if hits:
            return hits[-1]
    return None


def option_word_present(sentence: str, options: Sequence[str]) -> bool:
    """Whether a sentence contains any option word or letter marker."""
    s = sentence.lower()
    for o in options:
        o = o.strip().lower()
        if o and re.search(r"\b" + re.escape(o) + r"\b", s):
            return True
    return bool(re.search(r"\([A-H]\)", sentence))


def cut_conclusion(rationale: str, n_options: int) -> tuple[str, str | None]:
    """Remove the final concluding sentence that states an option letter.

    Returns:
        ``(R_cut, removed_sentence)``; ``removed_sentence`` is ``None`` when no
        sentence states a parsable option.
    """
    sents = split_sentences(rationale)
    for j in range(len(sents) - 1, -1, -1):
        if parse_stated_option(sents[j], n_options) is not None:
            return " ".join(sents[:j]), sents[j]
    return " ".join(sents), None


def clean_trailing_option_words(text: str, options: Sequence[str]) -> tuple[str, int]:
    """Drop trailing sentences that still name an option word.

    Returns:
        ``(cleaned_text, n_dropped)``.
    """
    sents = split_sentences(text)
    dropped = 0
    while sents and option_word_present(sents[-1], options):
        sents.pop()
        dropped += 1
    return " ".join(sents), dropped


def conclusion_sentence(letter: str, option_text: str) -> str:
    """Appended conclusion ``Y`` used in E3 swaps.

    VERIFY: match the conclusion template used by the original swap script.
    """
    return f"So the answer is ({letter}) {option_text}."


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------


def _cluster_index(groups: Sequence[Any]) -> tuple[np.ndarray, int]:
    _, inv = np.unique(np.asarray([str(g) for g in groups]), return_inverse=True)
    return inv, int(inv.max()) + 1 if len(inv) else 0


def cluster_bootstrap_ci(values: Sequence[float], groups: Sequence[Any], n_boot: int = 2000,
                         alpha: float = 0.05, seed: int = 0) -> tuple[float, float, float]:
    """Image-cluster bootstrap CI for the mean of per-item values.

    Clusters are resampled with replacement; the statistic is the item-level
    mean of the resampled concatenation.

    Args:
        values: Per-item values (e.g. paired differences).
        groups: Cluster id per item.
        n_boot: Number of resamples.
        alpha: Two-sided level.
        seed: RNG seed.

    Returns:
        ``(mean, lo, hi)``.
    """
    v = np.asarray(values, dtype=float)
    if len(v) == 0:
        return float("nan"), float("nan"), float("nan")
    inv, c = _cluster_index(groups)
    sums = np.bincount(inv, weights=v, minlength=c)
    cnts = np.bincount(inv, minlength=c).astype(float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, c, size=(n_boot, c))
    boot = sums[idx].sum(1) / cnts[idx].sum(1)
    lo, hi = np.quantile(boot, [alpha / 2, 1 - alpha / 2])
    return float(v.mean()), float(lo), float(hi)


def cluster_signflip_p(diffs: Sequence[float], groups: Sequence[Any], n_perm: int = 5000,
                       seed: int = 0) -> float:
    """Two-sided image-cluster sign-flip permutation p-value for a paired mean."""
    d = np.asarray(diffs, dtype=float)
    if len(d) == 0:
        return float("nan")
    inv, c = _cluster_index(groups)
    sums = np.bincount(inv, weights=d, minlength=c)
    obs = abs(sums.sum())
    rng = np.random.default_rng(seed)
    signs = rng.choice([-1.0, 1.0], size=(n_perm, c))
    perm = np.abs((signs * sums).sum(1))
    return float((1 + (perm >= obs - 1e-12).sum()) / (n_perm + 1))


def holm_adjust(pvals: Sequence[float]) -> list[float]:
    """Holm step-down adjusted p-values."""
    p = np.asarray(pvals, dtype=float)
    order = np.argsort(p)
    m = len(p)
    adj = np.empty(m)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (m - rank) * p[i])
        adj[i] = min(1.0, running)
    return adj.tolist()


def _compositions(n: int, k: int) -> Iterator[tuple[int, ...]]:
    """All k-tuples of non-negative ints summing to n."""
    for cuts in itertools.combinations(range(n + k - 1), k - 1):
        prev = -1
        out = []
        for c in cuts:
            out.append(c - prev - 1)
            prev = c
        out.append(n + k - 1 - prev - 1)
        yield tuple(out)


def plurality_win_prob(p: Sequence[float], gold: int, n_draws: int = 4) -> float:
    """Exact probability that ``gold`` wins an ``n_draws`` plurality vote.

    Votes are i.i.d. categorical with probabilities ``p``; ties at the top are
    split uniformly (expected accuracy under random tie-breaking). For two
    options and four draws this equals
    ``P(Bin(4,p) >= 3) + 0.5 * P(Bin(4,p) = 2)``.

    Args:
        p: Option probabilities (renormalized to sum to 1).
        gold: Index of the gold option.
        n_draws: Number of votes.

    Returns:
        Expected plurality-vote accuracy for this item.
    """
    p = np.asarray(p, dtype=float)
    s = p.sum()
    p = p / s if s > 0 else np.full(len(p), 1.0 / len(p))
    k = len(p)
    total = 0.0
    log_nf = math.lgamma(n_draws + 1)
    with np.errstate(divide="ignore"):
        logp = np.log(p)
    for c in _compositions(n_draws, k):
        mx = max(c)
        if c[gold] != mx:
            continue
        n_top = sum(1 for x in c if x == mx)
        lw = log_nf - sum(math.lgamma(x + 1) for x in c)
        lp = sum(x * logp[i] for i, x in enumerate(c) if x > 0)
        if np.isfinite(lp):
            total += math.exp(lw + lp) / n_top
    return float(total)


def observed_vote_credit(preds: Sequence[str | None], gold: str) -> float:
    """Observed plurality credit with ties split (``None`` votes are dropped)."""
    votes = [v for v in preds if v is not None]
    if not votes:
        return 0.0
    counts: dict[str, int] = {}
    for v in votes:
        counts[v] = counts.get(v, 0) + 1
    mx = max(counts.values())
    top = [k for k, c in counts.items() if c == mx]
    return 1.0 / len(top) if gold in top else 0.0


def grouped_kfold(groups: Sequence[Any], n_splits: int = 5, seed: int = 0) -> list[np.ndarray]:
    """Image-grouped K-fold assignment (balanced by item count).

    Returns:
        List of test-index arrays, one per fold.
    """
    inv, c = _cluster_index(groups)
    sizes = np.bincount(inv, minlength=c)
    rng = np.random.default_rng(seed)
    order = rng.permutation(c)
    order = order[np.argsort(-sizes[order], kind="stable")]
    fold_of = np.empty(c, dtype=int)
    load = np.zeros(n_splits)
    for g in order:
        f = int(np.argmin(load))
        fold_of[g] = f
        load[f] += sizes[g]
    item_fold = fold_of[inv]
    return [np.where(item_fold == f)[0] for f in range(n_splits)]


def margin_top2(probs: Sequence[float]) -> float:
    """Label-free Direct margin ``m(x) = P(c1) - P(c2)`` over options."""
    p = np.sort(np.asarray(probs, dtype=float))[::-1]
    return float(p[0] - (p[1] if len(p) > 1 else 0.0))


def select_tau(margins: np.ndarray, direct_ok: np.ndarray, alt_ok: np.ndarray) -> float:
    """Pick the gate threshold maximizing accuracy (ties -> smallest tau).

    Policy: use Direct when ``m >= tau`` else the alternative (CoT / vote).
    A smaller tau routes fewer items to the alternative, so ties prefer it.
    """
    cands = np.unique(np.concatenate([[0.0], margins, [1.0 + 1e-9]]))
    best_tau, best_acc = 0.0, -1.0
    for t in cands:
        acc = np.where(margins >= t, direct_ok, alt_ok).mean()
        if acc > best_acc + 1e-12:
            best_tau, best_acc = float(t), float(acc)
    return best_tau


def logistic_irls(x: np.ndarray, y: np.ndarray, n_iter: int = 50,
                  ridge: float = 1e-6) -> tuple[np.ndarray, np.ndarray]:
    """Logistic regression by IRLS (intercept must be a column of ``x``).

    Returns:
        ``(coef, se)`` with model-based (non-clustered) standard errors.
    """
    beta = np.zeros(x.shape[1])
    for _ in range(n_iter):
        eta = np.clip(x @ beta, -30, 30)
        mu = 1 / (1 + np.exp(-eta))
        w = mu * (1 - mu) + 1e-9
        h = x.T @ (x * w[:, None]) + ridge * np.eye(x.shape[1])
        step = np.linalg.solve(h, x.T @ (y - mu))
        beta = beta + step
        if np.max(np.abs(step)) < 1e-8:
            break
    eta = np.clip(x @ beta, -30, 30)
    mu = 1 / (1 + np.exp(-eta))
    h = x.T @ (x * (mu * (1 - mu) + 1e-9)[:, None]) + ridge * np.eye(x.shape[1])
    se = np.sqrt(np.diag(np.linalg.inv(h)))
    return beta, se


# ---------------------------------------------------------------------------
# Model registry and provenance
# ---------------------------------------------------------------------------

MODEL_REGISTRY: dict[str, dict[str, Any]] = {
    "qwen25vl_3b": {"hf_id": "Qwen/Qwen2.5-VL-3B-Instruct", "family": "qwen25vl", "env": "main"},
    "qwen25vl_7b": {"hf_id": "Qwen/Qwen2.5-VL-7B-Instruct", "family": "qwen25vl", "env": "main"},
    "internvl3_8b": {"hf_id": "OpenGVLab/InternVL3-8B", "family": "internvl", "env": "main"},
    "phi35v": {"hf_id": "microsoft/Phi-3.5-vision-instruct", "family": "phi35v", "env": "main"},
    "molmo7b_d": {"hf_id": "allenai/Molmo-7B-D-0924", "family": "molmo", "env": "main", "max_images": 1},
    "qwen3vl_8b_instruct": {"hf_id": "Qwen/Qwen3-VL-8B-Instruct", "family": "qwen3vl", "env": "qwen3vl"},
    "qwen3vl_8b_thinking": {"hf_id": "Qwen/Qwen3-VL-8B-Thinking", "family": "qwen3vl", "env": "qwen3vl",
                            "thinking": True},
}


def resolve_model(model_key: str, model_path: str | None = None, revision: str | None = None,
                  provenance_dir: str | os.PathLike = "outputs/provenance",
                  verify_sha256: bool = True) -> tuple[str, dict[str, Any]]:
    """Resolve a model to a local directory and record its provenance.

    Downloads with ``huggingface_hub.snapshot_download`` when ``model_path`` is
    not given. ``HF_TOKEN`` is read from the environment only and never
    written. The sha256 of each ``*.safetensors`` shard is computed and, when
    the Hub exposes LFS metadata, compared with the expected digest.

    Args:
        model_key: Key in ``MODEL_REGISTRY``.
        model_path: Optional local checkpoint directory (skips download).
        revision: Optional Hub revision to pin.
        provenance_dir: Where ``<model_key>.json`` is written.
        verify_sha256: Whether to hash the safetensors shards.

    Returns:
        ``(local_dir, provenance_dict)``.
    """
    spec = MODEL_REGISTRY[model_key]
    token = os.environ.get("HF_TOKEN") or None
    prov: dict[str, Any] = {"model_key": model_key, "hf_id": spec["hf_id"], "requested_revision": revision,
                            "resolved_at": utc_stamp()}
    if model_path is None:
        from huggingface_hub import snapshot_download

        model_path = snapshot_download(spec["hf_id"], revision=revision, token=token)
    local = Path(model_path).resolve()
    prov["local_path"] = str(local)
    m = re.search(r"snapshots/([0-9a-f]{40})", str(local))
    prov["hf_revision"] = m.group(1) if m else revision
    expected: dict[str, str] = {}
    try:
        from huggingface_hub import HfApi

        info = HfApi().model_info(spec["hf_id"], revision=prov["hf_revision"] or revision,
                                  files_metadata=True, token=token)
        prov["hf_revision"] = prov["hf_revision"] or info.sha
        for s in info.siblings or []:
            if s.rfilename.endswith(".safetensors") and getattr(s, "lfs", None):
                lfs = s.lfs
                expected[s.rfilename] = lfs.get("sha256") if isinstance(lfs, dict) else getattr(lfs, "sha256", None)
    except Exception as e:  # network may be unavailable on the cluster
        prov["hub_metadata_error"] = f"{type(e).__name__}: {e}"[:300]
    files: dict[str, Any] = {}
    out_dir = Path(provenance_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_path = out_dir / f"{model_key}.json"
    cached = json.loads(cache_path.read_text()) if cache_path.exists() else {}
    for f in sorted(local.glob("*.safetensors")):
        entry: dict[str, Any] = {"bytes": f.stat().st_size}
        old = cached.get("files", {}).get(f.name)
        if old and old.get("bytes") == entry["bytes"] and old.get("sha256") and \
                cached.get("hf_revision") == prov["hf_revision"]:
            entry["sha256"] = old["sha256"]
        elif verify_sha256:
            entry["sha256"] = sha256_file(f)
        if f.name in expected and expected[f.name]:
            entry["expected_sha256"] = expected[f.name]
            entry["match"] = entry.get("sha256") == expected[f.name]
        files[f.name] = entry
    prov["files"] = files
    mismatches = [k for k, v in files.items() if v.get("match") is False]
    prov["checksum_status"] = "mismatch" if mismatches else (
        "verified" if any("match" in v for v in files.values()) else "computed_unverified")
    cache_path.write_text(json.dumps(prov, indent=2))
    if mismatches:
        raise RuntimeError(f"safetensors sha256 mismatch for {model_key}: {mismatches}")
    return str(local), prov


# ---------------------------------------------------------------------------
# Option readout
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class LetterTokens:
    """Token ids that decode to an option letter under different definitions."""

    primary: dict[str, int]
    upper_variants: dict[str, list[int]]
    case_variants: dict[str, list[int]]


def build_letter_tokens(tokenizer: Any, letters: Sequence[str] = LETTERS) -> LetterTokens:
    """Find primary and variant token ids for each option letter.

    Primary: the single token produced when the letter follows ``Answer: (``.
    Upper variants: every vocab token whose decoded text stripped of
    whitespace equals the uppercase letter (``A``, `` A``, ...).
    Case variants: upper variants plus lowercase forms (reported separately
    because lowercase ``a`` is also an English word).
    """
    base = tokenizer.encode(ANSWER_PREFIX, add_special_tokens=False)
    primary: dict[str, int] = {}
    for l in letters:
        ids = tokenizer.encode(ANSWER_PREFIX + l, add_special_tokens=False)
        tail = ids[len(base):] if ids[: len(base)] == base else ids[-1:]
        primary[l] = tail[0] if len(tail) == 1 else ids[-1]
    upper: dict[str, list[int]] = {l: [] for l in letters}
    case: dict[str, list[int]] = {l: [] for l in letters}
    lower_map = {l.lower(): l for l in letters}
    n_vocab = len(tokenizer)
    for tid in range(n_vocab):
        try:
            s = tokenizer.decode([tid]).strip()
        except Exception:
            continue
        if s in upper:
            upper[s].append(tid)
            case[s].append(tid)
        elif s in lower_map:
            case[lower_map[s]].append(tid)
    for l in letters:
        if primary[l] not in upper[l]:
            upper[l].append(primary[l])
            case[l].append(primary[l])
    return LetterTokens(primary, upper, case)


def _lse(x: np.ndarray) -> float:
    m = float(np.max(x))
    return m + math.log(float(np.exp(x - m).sum()))


def readout_from_logits(logits: np.ndarray, letters: Sequence[str], lt: LetterTokens,
                        decode: Callable[[int], str], top_k: int = 20) -> dict[str, Any]:
    """Compute option probabilities from full-vocabulary answer logits.

    Args:
        logits: Float32 logits at the position after ``Answer: (``, already
            restricted to ``len(tokenizer)`` entries.
        letters: Valid option letters.
        lt: Letter token table.
        decode: Single-token decoder.
        top_k: Size of the emulated vLLM top-k list.

    Returns:
        Dict with per-letter probabilities for four definitions
        (``primary``, ``sum`` (uppercase variants), ``sum_case``, ``top20``
        emulation of the original readout), predictions for each, absolute
        full-vocab option mass, and the top-k token list.
    """
    lp = logits.astype(np.float64)
    lp = lp - _lse(lp)
    out: dict[str, Any] = {}
    prim = np.array([lp[lt.primary[l]] for l in letters])
    sums = np.array([_lse(lp[lt.upper_variants[l]]) for l in letters])
    sumc = np.array([_lse(lp[lt.case_variants[l]]) for l in letters])
    for name, arr in (("primary", prim), ("sum", sums), ("sum_case", sumc)):
        p = np.exp(arr - _lse(arr))
        out[f"p_{name}"] = {l: float(v) for l, v in zip(letters, p)}
        out[f"logprob_{name}"] = {l: float(v) for l, v in zip(letters, arr)}
        out[f"pred_{name}"] = letters[int(np.argmax(arr))]
    out["option_mass_primary"] = float(np.exp(prim).sum())
    out["option_mass_sum"] = float(np.exp(sums).sum())
    top = np.argpartition(-lp, top_k)[:top_k]
    top = top[np.argsort(-lp[top])]
    toks = [(decode(int(t)), float(lp[t])) for t in top]
    out["top20"] = [[s, v, int(t)] for (s, v), t in zip(toks, top)]
    best: dict[str, float] = {}
    for s, v in toks:
        k = s.strip()
        if k in letters and (k not in best or v > best[k]):
            best[k] = v
    if best:
        arr = np.array([best.get(l, -np.inf) for l in letters])
        p = np.exp(arr - _lse(arr[np.isfinite(arr)]))
        out["p_top20"] = {l: float(v) for l, v in zip(letters, p)}
        out["pred_top20"] = letters[int(np.argmax(arr))]
    else:
        out["p_top20"] = None
        out["pred_top20"] = None
    out["top20_missing_letters"] = [l for l in letters if l not in best]
    out["top20_last_logprob"] = float(lp[top[-1]])
    return out


# ---------------------------------------------------------------------------
# Model adapters (lazy torch import)
# ---------------------------------------------------------------------------


class ModelAdapter:
    """Uniform interface over HF VLM families.

    Subclasses implement ``_encode`` (Prompt -> model inputs) and
    ``_forward_logits``; generation uses each family's HF ``generate``.
    """

    family = "base"

    def __init__(self, model_key: str, model_dir: str, device: str = "cuda",
                 dtype: str = "bfloat16") -> None:
        import torch

        self.torch = torch
        self.model_key = model_key
        self.spec = MODEL_REGISTRY[model_key]
        self.model_dir = model_dir
        self.device = device
        self.dtype = getattr(torch, dtype)
        self.max_images = self.spec.get("max_images", 4)
        self.thinking = bool(self.spec.get("thinking", False))
        self._load()
        self._letter_cache: dict[str, LetterTokens] = {}

    # -- hooks ---------------------------------------------------------------
    def _load(self) -> None:
        raise NotImplementedError

    @property
    def tokenizer(self) -> Any:
        raise NotImplementedError

    def render_text(self, prompt: Prompt) -> str:
        """Render the prompt to the model's text format (image placeholders kept)."""
        raise NotImplementedError

    def _encode(self, prompt: Prompt) -> dict[str, Any]:
        raise NotImplementedError

    def _forward_logits(self, inputs: dict[str, Any]) -> Any:
        raise NotImplementedError

    def _generate_ids(self, inputs: dict[str, Any], gen_kwargs: dict[str, Any]) -> list[int]:
        raise NotImplementedError

    # -- shared --------------------------------------------------------------
    def encode_text(self, text: str) -> list[int]:
        """Tokenize plain text without special tokens."""
        return self.tokenizer.encode(text, add_special_tokens=False)

    def decode_ids(self, ids: Sequence[int]) -> str:
        """Decode token ids, skipping special tokens."""
        return self.tokenizer.decode(list(ids), skip_special_tokens=True)

    def n_tokens(self, text: str) -> int:
        """Token count of plain text."""
        return len(self.encode_text(text))

    def filler(self, n_tokens: int) -> str:
        """Neutral filler with ``n_tokens`` tokens under this tokenizer."""
        return filler_text(n_tokens, self.encode_text, lambda ids: self.tokenizer.decode(ids))

    def letter_tokens(self) -> LetterTokens:
        """Cached letter token table."""
        if "all" not in self._letter_cache:
            self._letter_cache["all"] = build_letter_tokens(self.tokenizer)
        return self._letter_cache["all"]

    def check_images(self, prompt: Prompt) -> None:
        """Raise when the prompt exceeds the model's image slots."""
        if prompt.n_images() > self.max_images:
            raise ValueError(f"{self.model_key} accepts at most {self.max_images} image(s)")

    def readout(self, prompt: Prompt, letters: Sequence[str]) -> dict[str, Any]:
        """Forced one-token option readout at the open ``Answer: (`` prefix."""
        self.check_images(prompt)
        inputs = self._encode(prompt)
        with self.torch.inference_mode():
            logits = self._forward_logits(inputs)
        n_vocab = len(self.tokenizer)
        arr = logits.float()[:n_vocab].cpu().numpy()
        res = readout_from_logits(arr, letters, self.letter_tokens(),
                                  lambda t: self.tokenizer.decode([t]))
        res["n_prompt_tokens"] = int(inputs["_n_tokens"])
        return res

    def generate(self, prompt: Prompt, max_new_tokens: int = MAX_COT_TOKENS, temperature: float = 0.0,
                 seed: int | None = None, stop_strings: Sequence[str] | None = None) -> dict[str, Any]:
        """Generate a continuation of the open assistant prefix.

        Returns:
            Dict with ``text``, ``n_tokens`` and ``truncated``.
        """
        self.check_images(prompt)
        inputs = self._encode(prompt)
        kw: dict[str, Any] = {"max_new_tokens": max_new_tokens}
        if temperature > 0:
            kw.update(do_sample=True, temperature=temperature, top_p=1.0, top_k=0)
            if seed is not None:
                self.torch.manual_seed(seed)
        else:
            kw.update(do_sample=False, temperature=None, top_p=None, top_k=None)
        if stop_strings:
            kw.update(stop_strings=list(stop_strings), tokenizer=self.tokenizer)
        with self.torch.inference_mode():
            new_ids = self._generate_ids(inputs, kw)
        eos = set(self._eos_ids())
        trimmed = []
        for t in new_ids:
            if t in eos:
                break
            trimmed.append(t)
        text = self.decode_ids(trimmed)
        if stop_strings:
            for s in stop_strings:
                if s in text:
                    text = text[: text.index(s)]
        return {"text": text, "n_tokens": len(trimmed), "truncated": len(trimmed) >= max_new_tokens}

    def _eos_ids(self) -> list[int]:
        ids: list[int] = []
        gc = getattr(getattr(self, "model", None), "generation_config", None)
        e = getattr(gc, "eos_token_id", None)
        if isinstance(e, int):
            ids.append(e)
        elif isinstance(e, (list, tuple)):
            ids.extend(e)
        if self.tokenizer.eos_token_id is not None:
            ids.append(self.tokenizer.eos_token_id)
        return ids


def _to_device(d: dict[str, Any], device: str, dtype: Any) -> dict[str, Any]:
    import torch

    out = {}
    for k, v in d.items():
        if isinstance(v, torch.Tensor):
            v = v.to(device)
            if v.is_floating_point():
                v = v.to(dtype)
        out[k] = v
    return out


class QwenVLAdapter(ModelAdapter):
    """Qwen2.5-VL and Qwen3-VL via ``AutoProcessor.apply_chat_template``."""

    family = "qwen"

    def _load(self) -> None:
        from transformers import AutoProcessor

        fam = self.spec["family"]
        if fam == "qwen25vl":
            from transformers import Qwen2_5_VLForConditionalGeneration as Cls

            self.processor = AutoProcessor.from_pretrained(self.model_dir, max_pixels=QWEN25_MAX_PIXELS)
        else:
            from transformers import Qwen3VLForConditionalGeneration as Cls

            self.processor = AutoProcessor.from_pretrained(self.model_dir)
            ip = self.processor.image_processor
            # Align the per-image pixel budget with Qwen2.5-VL max_pixels.
            if isinstance(getattr(ip, "size", None), dict) and "longest_edge" in ip.size:
                ip.size = {**ip.size, "longest_edge": QWEN25_MAX_PIXELS}
            if hasattr(ip, "max_pixels"):
                ip.max_pixels = QWEN25_MAX_PIXELS
        self.model = Cls.from_pretrained(self.model_dir, torch_dtype=self.dtype,
                                         attn_implementation="sdpa").to(self.device).eval()

    @property
    def tokenizer(self) -> Any:
        return self.processor.tokenizer

    def _messages(self, prompt: Prompt) -> tuple[list[dict[str, Any]], list[Any]]:
        msgs, imgs = [], []
        for t in prompt.turns:
            if t.role == "assistant":
                msgs.append({"role": "assistant", "content": [{"type": "text", "text": "".join(t.parts)}]})
                continue
            content = []
            for p in t.parts:
                if isinstance(p, str):
                    content.append({"type": "text", "text": p})
                else:
                    content.append({"type": "image"})
                    imgs.append(p)
            msgs.append({"role": "user", "content": content})
        return msgs, imgs

    def render_text(self, prompt: Prompt) -> str:
        msgs, _ = self._messages(prompt)
        text = self.processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        if self.thinking:
            # Thinking template opens "<think>\n"; prefixes are written after the
            # closed think block by callers via ``assistant_prefix``.
            pass
        return text + prompt.assistant_prefix

    def _encode(self, prompt: Prompt) -> dict[str, Any]:
        _, imgs = self._messages(prompt)
        text = self.render_text(prompt)
        enc = self.processor(text=[text], images=imgs or None, return_tensors="pt")
        enc = _to_device(dict(enc), self.device, self.dtype)
        enc["_n_tokens"] = enc["input_ids"].shape[1]
        return enc

    def _forward_logits(self, inputs: dict[str, Any]) -> Any:
        kw = {k: v for k, v in inputs.items() if not k.startswith("_")}
        return self.model(**kw, use_cache=False).logits[0, -1]

    def _generate_ids(self, inputs: dict[str, Any], gen_kwargs: dict[str, Any]) -> list[int]:
        kw = {k: v for k, v in inputs.items() if not k.startswith("_")}
        out = self.model.generate(**kw, **gen_kwargs)
        return out[0, inputs["input_ids"].shape[1]:].tolist()


_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)


def internvl_dynamic_tiles(img: "Image.Image", image_size: int = 448, max_num: int = 6,
                           min_num: int = 1, use_thumbnail: bool = True) -> list["Image.Image"]:
    """InternVL dynamic tiling (reference implementation from the model card)."""
    w, h = img.size
    aspect = w / h
    ratios = sorted({(i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1)
                     for j in range(1, n + 1) if min_num <= i * j <= max_num}, key=lambda x: x[0] * x[1])
    best, best_diff = (1, 1), float("inf")
    area = w * h
    for r in ratios:
        diff = abs(aspect - r[0] / r[1])
        if diff < best_diff:
            best_diff, best = diff, r
        elif diff == best_diff and area > 0.5 * image_size * image_size * r[0] * r[1]:
            best = r
    tw, th = image_size * best[0], image_size * best[1]
    resized = img.resize((tw, th))
    tiles = []
    for i in range(best[0] * best[1]):
        box = ((i % (tw // image_size)) * image_size, (i // (tw // image_size)) * image_size,
               ((i % (tw // image_size)) + 1) * image_size, ((i // (tw // image_size)) + 1) * image_size)
        tiles.append(resized.crop(box))
    if use_thumbnail and len(tiles) != 1:
        tiles.append(img.resize((image_size, image_size)))
    return tiles


class InternVLAdapter(ModelAdapter):
    """InternVL3 (trust_remote_code) with tokenizer chat template + IMG_CONTEXT expansion."""

    family = "internvl"

    def _load(self) -> None:
        from transformers import AutoModel, AutoTokenizer

        self._tok = AutoTokenizer.from_pretrained(self.model_dir, trust_remote_code=True, use_fast=False)
        self.model = AutoModel.from_pretrained(self.model_dir, torch_dtype=self.dtype, trust_remote_code=True,
                                               use_flash_attn=False, low_cpu_mem_usage=True).to(self.device).eval()
        self.model.img_context_token_id = self._tok.convert_tokens_to_ids("<IMG_CONTEXT>")
        self.num_image_token = int(self.model.num_image_token)

    @property
    def tokenizer(self) -> Any:
        return self._tok

    def _messages(self, prompt: Prompt) -> tuple[list[dict[str, str]], list[Any]]:
        msgs, imgs = [], []
        for t in prompt.turns:
            text_parts, n_img = [], 0
            for p in t.parts:
                if isinstance(p, str):
                    text_parts.append(p)
                else:
                    n_img += 1
                    imgs.append(p)
            # vLLM's InternVL chat path places "<image>" placeholders before text.
            body = "<image>\n" * n_img + "".join(text_parts)
            msgs.append({"role": t.role, "content": body})
        return msgs, imgs

    def render_text(self, prompt: Prompt) -> str:
        msgs, _ = self._messages(prompt)
        text = self._tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        return text + prompt.assistant_prefix

    def _pixel_values(self, imgs: list[Any]) -> tuple[Any, list[int]]:
        import torch

        mean = torch.tensor(_IMAGENET_MEAN).view(3, 1, 1)
        std = torch.tensor(_IMAGENET_STD).view(3, 1, 1)
        tensors, counts = [], []
        for im in imgs:
            tiles = internvl_dynamic_tiles(im, 448, max_num=6)
            arr = [torch.from_numpy(np.asarray(t.convert("RGB").resize((448, 448)), dtype=np.float32) / 255.0)
                   .permute(2, 0, 1) for t in tiles]
            tensors.append(((torch.stack(arr) - mean) / std))
            counts.append(len(tiles))
        return torch.cat(tensors), counts

    def _encode(self, prompt: Prompt) -> dict[str, Any]:
        import torch

        _, imgs = self._messages(prompt)
        text = self.render_text(prompt)
        pv, counts = (None, [])
        if imgs:
            pv, counts = self._pixel_values(imgs)
        for c in counts:
            text = text.replace("<image>", "<img>" + "<IMG_CONTEXT>" * (self.num_image_token * c) + "</img>", 1)
        enc = self._tok(text, return_tensors="pt", add_special_tokens=False)
        d = {"input_ids": enc["input_ids"].to(self.device), "attention_mask": enc["attention_mask"].to(self.device),
             "_n_tokens": enc["input_ids"].shape[1]}
        if pv is not None:
            d["pixel_values"] = pv.to(self.device, self.dtype)
            d["image_flags"] = torch.ones(pv.shape[0], 1, dtype=torch.long, device=self.device)
        return d

    def _forward_logits(self, inputs: dict[str, Any]) -> Any:
        if "pixel_values" in inputs:
            out = self.model(pixel_values=inputs["pixel_values"], input_ids=inputs["input_ids"],
                             attention_mask=inputs["attention_mask"], image_flags=inputs["image_flags"],
                             use_cache=False)
        else:
            out = self.model.language_model(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"],
                                            use_cache=False)
        return out.logits[0, -1]

    def _generate_ids(self, inputs: dict[str, Any], gen_kwargs: dict[str, Any]) -> list[int]:
        gen_kwargs = dict(gen_kwargs)
        gen_kwargs.setdefault("eos_token_id", self._tok.convert_tokens_to_ids("<|im_end|>"))
        out = self.model.generate(pixel_values=inputs.get("pixel_values"), input_ids=inputs["input_ids"],
                                  attention_mask=inputs["attention_mask"], **gen_kwargs)
        # InternVL generate works on inputs_embeds and returns only new tokens.
        return out[0].tolist()

    def _eos_ids(self) -> list[int]:
        return super()._eos_ids() + [self._tok.convert_tokens_to_ids("<|im_end|>")]


class Phi35VAdapter(ModelAdapter):
    """Phi-3.5-Vision (trust_remote_code), ``num_crops=4``, ``<|image_i|>`` placeholders."""

    family = "phi35v"

    def _load(self) -> None:
        from transformers import AutoModelForCausalLM, AutoProcessor

        self.processor = AutoProcessor.from_pretrained(self.model_dir, trust_remote_code=True, num_crops=4)
        self.model = AutoModelForCausalLM.from_pretrained(self.model_dir, trust_remote_code=True,
                                                          torch_dtype=self.dtype,
                                                          _attn_implementation="eager").to(self.device).eval()

    @property
    def tokenizer(self) -> Any:
        return self.processor.tokenizer

    def _messages(self, prompt: Prompt) -> tuple[list[dict[str, str]], list[Any]]:
        msgs, imgs = [], []
        for t in prompt.turns:
            text_parts, ph = [], []
            for p in t.parts:
                if isinstance(p, str):
                    text_parts.append(p)
                else:
                    imgs.append(p)
                    ph.append(f"<|image_{len(imgs)}|>\n")
            msgs.append({"role": t.role, "content": "".join(ph) + "".join(text_parts)})
        return msgs, imgs

    def render_text(self, prompt: Prompt) -> str:
        msgs, _ = self._messages(prompt)
        text = self.processor.tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        return text + prompt.assistant_prefix

    def _encode(self, prompt: Prompt) -> dict[str, Any]:
        _, imgs = self._messages(prompt)
        text = self.render_text(prompt)
        enc = self.processor(text, imgs or None, return_tensors="pt")
        enc = _to_device(dict(enc), self.device, self.dtype)
        enc["_n_tokens"] = enc["input_ids"].shape[1]
        return enc

    def _forward_logits(self, inputs: dict[str, Any]) -> Any:
        kw = {k: v for k, v in inputs.items() if not k.startswith("_")}
        return self.model(**kw, use_cache=False).logits[0, -1]

    def _generate_ids(self, inputs: dict[str, Any], gen_kwargs: dict[str, Any]) -> list[int]:
        kw = {k: v for k, v in inputs.items() if not k.startswith("_")}
        gen_kwargs = dict(gen_kwargs)
        gen_kwargs.setdefault("eos_token_id", self.processor.tokenizer.eos_token_id)
        out = self.model.generate(**kw, **gen_kwargs)
        return out[0, inputs["input_ids"].shape[1]:].tolist()


class MolmoAdapter(ModelAdapter):
    """Molmo-7B-D with the plain ``User: ... Assistant: ...`` template (one image)."""

    family = "molmo"

    def _load(self) -> None:
        from transformers import AutoModelForCausalLM, AutoProcessor

        self.processor = AutoProcessor.from_pretrained(self.model_dir, trust_remote_code=True, torch_dtype="auto")
        self.model = AutoModelForCausalLM.from_pretrained(self.model_dir, trust_remote_code=True,
                                                          torch_dtype=self.dtype).to(self.device).eval()

    @property
    def tokenizer(self) -> Any:
        return self.processor.tokenizer

    def render_text(self, prompt: Prompt) -> str:
        chunks = []
        for t in prompt.turns:
            body = "".join(p for p in t.parts if isinstance(p, str))
            chunks.append(("User: " if t.role == "user" else "Assistant: ") + body)
        text = " ".join(chunks) + " Assistant:"
        if prompt.assistant_prefix:
            text += " " + prompt.assistant_prefix
        return text

    def _encode(self, prompt: Prompt) -> dict[str, Any]:
        imgs = [p for t in prompt.turns for p in t.parts if not isinstance(p, str)]
        text = self.render_text(prompt)
        tokens = self.tokenizer.encode(" " + text, add_special_tokens=False)
        try:
            enc = self.processor.process(images=imgs or None, tokens=tokens)
        except TypeError:
            enc = self.processor.process(images=imgs or None, text=text, message_format="none")
        d = {k: v.unsqueeze(0) for k, v in enc.items()}
        d = _to_device(d, self.device, self.dtype)
        d["_n_tokens"] = d["input_ids"].shape[1]
        return d

    def _forward_logits(self, inputs: dict[str, Any]) -> Any:
        kw = {k: v for k, v in inputs.items() if not k.startswith("_")}
        return self.model(**kw).logits[0, -1]

    def _generate_ids(self, inputs: dict[str, Any], gen_kwargs: dict[str, Any]) -> list[int]:
        from transformers import GenerationConfig

        kw = {k: v for k, v in inputs.items() if not k.startswith("_")}
        stop = gen_kwargs.pop("stop_strings", None)
        gen_kwargs.pop("tokenizer", None)
        gc = GenerationConfig(**{k: v for k, v in gen_kwargs.items() if v is not None},
                              stop_strings=stop or "<|endoftext|>")
        out = self.model.generate_from_batch(kw, gc, tokenizer=self.tokenizer)
        return out[0, inputs["input_ids"].shape[1]:].tolist()


def load_adapter(model_key: str, model_path: str | None = None, device: str = "cuda",
                 revision: str | None = None, provenance_dir: str = "outputs/provenance"
                 ) -> tuple[ModelAdapter, dict[str, Any]]:
    """Resolve, verify and load a model adapter.

    Returns:
        ``(adapter, provenance)``.
    """
    local, prov = resolve_model(model_key, model_path, revision, provenance_dir)
    fam = MODEL_REGISTRY[model_key]["family"]
    cls = {"qwen25vl": QwenVLAdapter, "qwen3vl": QwenVLAdapter, "internvl": InternVLAdapter,
           "phi35v": Phi35VAdapter, "molmo": MolmoAdapter}[fam]
    return cls(model_key, local, device=device), prov


# ---------------------------------------------------------------------------
# Rationale cache
# ---------------------------------------------------------------------------


class RationaleStore:
    """Per-model cache of generated texts keyed by (item, kind, variant, sample)."""

    def __init__(self, path: str | os.PathLike) -> None:
        """Load an existing cache file (JSONL) and open it for appending."""
        self.path = Path(path)
        self.data: dict[tuple, dict[str, Any]] = {}
        if self.path.exists():
            for r in read_jsonl(self.path):
                self.data[self._k(r["item_id"], r["kind"], r.get("image_variant", "original"),
                                  r.get("sample_idx", 0))] = r
        self.writer = JsonlWriter(self.path, ["item_id", "kind", "image_variant", "sample_idx"])

    @staticmethod
    def _k(item_id: str, kind: str, variant: str, sample_idx: int) -> tuple:
        return (str(item_id), kind, variant, int(sample_idx))

    def get(self, item_id: str, kind: str = "cot", variant: str = "original",
            sample_idx: int = 0) -> dict[str, Any] | None:
        """Return a cached record or ``None``."""
        return self.data.get(self._k(item_id, kind, variant, sample_idx))

    def put(self, rec: dict[str, Any]) -> None:
        """Insert and persist a record."""
        rec.setdefault("image_variant", "original")
        rec.setdefault("sample_idx", 0)
        k = self._k(rec["item_id"], rec["kind"], rec["image_variant"], rec["sample_idx"])
        if k not in self.data:
            self.data[k] = rec
            self.writer.write(rec)


def get_or_generate(store: RationaleStore, adapter: ModelAdapter | None, item: Item, kind: str = "cot",
                    variant: str = "original", sample_idx: int = 0, temperature: float = 0.0,
                    seed: int = 0) -> dict[str, Any]:
    """Return a cached rationale or generate it with the HF adapter.

    Args:
        store: Rationale cache.
        adapter: Loaded adapter (``None`` forbids generation).
        item: The item.
        kind: ``cot`` (greedy), ``cot_sample`` (temperature sampling).
        variant: Image variant used while generating.
        sample_idx: Sample index for sampled CoT.
        temperature: Sampling temperature.
        seed: Base seed; the per-sample seed is ``seed + sample_idx``.

    Returns:
        Cache record with ``text``.
    """
    rec = store.get(item.item_id, kind, variant, sample_idx)
    if rec is not None:
        return rec
    if adapter is None:
        raise KeyError(f"missing rationale {item.item_id}/{kind}/{variant}/{sample_idx}")
    images = variant_images(item, variant)
    g = adapter.generate(build_cot_generation_prompt(item.q_text(), images), MAX_COT_TOKENS,
                         temperature=temperature, seed=seed + sample_idx)
    rec = {"item_id": item.item_id, "kind": kind, "image_variant": variant, "sample_idx": sample_idx,
           "text": g["text"], "n_tokens": g["n_tokens"], "truncated": g["truncated"], "backend": "hf",
           "temperature": temperature, "seed": seed + sample_idx, "model": adapter.model_key}
    store.put(rec)
    return rec


# ---------------------------------------------------------------------------
# CLI helpers
# ---------------------------------------------------------------------------


def add_common_args(ap: Any) -> None:
    """Arguments shared by GPU experiment scripts."""
    ap.add_argument("--model", required=True, choices=sorted(MODEL_REGISTRY))
    ap.add_argument("--model-path", default=None, help="local checkpoint dir (skips Hub download)")
    ap.add_argument("--revision", default=None, help="pin a Hub revision")
    ap.add_argument("--items", required=True, nargs="+", help="item manifest JSONL(s)")
    ap.add_argument("--tasks", nargs="*", default=None)
    ap.add_argument("--limit", type=int, default=None, help="cap items per manifest (smoke runs)")
    ap.add_argument("--out-dir", default="outputs")
    ap.add_argument("--rationales", default=None,
                    help="rationale cache JSONL (default <out-dir>/rationales/<model>.jsonl)")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=0)


def load_all_items(paths: Sequence[str], tasks: Sequence[str] | None, limit: int | None) -> list[Item]:
    """Load and concatenate item manifests, checking id uniqueness."""
    items: list[Item] = []
    for p in paths:
        items.extend(load_items(p, tasks, limit))
    ids = [i.item_id for i in items]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate item_id across manifests")
    return items


def set_seed(seed: int) -> None:
    """Seed python, numpy and torch (if installed)."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def write_run_meta(out_path: str | os.PathLike, args: Any, extra: dict[str, Any] | None = None) -> None:
    """Write ``<out>.meta.json`` with argv, versions and provenance."""
    meta: dict[str, Any] = {"argv": sys.argv, "args": vars(args) if hasattr(args, "__dict__") else args,
                            "started_utc": utc_stamp(), "git_revision": git_revision(Path(__file__).parent),
                            "python": sys.version}
    for mod in ("torch", "transformers", "numpy", "PIL", "vllm"):
        try:
            meta[f"version_{mod}"] = __import__(mod).__version__
        except Exception:
            pass
    if extra:
        meta.update(extra)
    p = Path(str(out_path) + ".meta.json")
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(meta, indent=2, default=str))


def pil_to_png_bytes(img: "Image.Image") -> bytes:
    """Encode a PIL image as PNG bytes."""
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def iter_progress(xs: Iterable[Any], every: int = 50, label: str = "") -> Iterator[Any]:
    """Yield from ``xs`` while printing a progress line every ``every`` items."""
    for i, x in enumerate(xs):
        if i % every == 0:
            print(f"[{utc_stamp()}] {label} {i}", flush=True)
        yield x
