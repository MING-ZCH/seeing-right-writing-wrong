"""Sentence splitting and parsers for written spatial relations in rationales.

split_sents  plain sentence split.
stype        sentence type: conclusion / relation / non_relational.
first_claim  v1: first two-object occlusion / front claim (index, claimed-front object).
first_claim_v2 / v3: first depth-order claim, stricter rules.

v1 counts symmetric verbs ("A overlaps B", "overlapping") as "A is in front".
v2: a sentence counts only if it carries an ordered relation: active (covers, on top of, obscures, blocks, in front of)
    or passive (covered / hidden / obscured / blocked / overlapped / occluded by, behind, underneath, beneath).
v3: v2 plus negation ("A is not in front of B", "A does not cover B") -> the other object is claimed in front.
"""
from __future__ import annotations

import re


OBJ = re.compile(r"\b(red|blue|green|yellow|purple|orange|cyan|gray|grey|pink|brown|black|white)\s+"
                 r"(circle|square|triangle|diamond|hexagon|pentagon|star|rectangle)s?\b", re.I)
ACT = re.compile(r"\b(overlap\w*|cover\w*|on top of|obscur\w*|block\w*|in front of)\b", re.I)
PAS = re.compile(r"\b(covered|hidden|obscured|blocked|overlapped|occluded)\s+by\b|\bbehind\b|\bunderneath\b|\bbeneath\b", re.I)
SKIP = re.compile(r"\?|whether|if the|determine|asks", re.I)
QRX = re.compile(r"Is the (\w+ \w+) in front of the (\w+ \w+)\?")
REL = re.compile(r"\b(in front|behind|closer|farther|further|nearer|left|right|above|below|on top|under|beneath|"
                 r"foreground|background|occlud\w*|overlap\w*|hidden|blocks?|covers?)\b", re.I)
CONCL = re.compile(r"\b(answer|correct option|conclusion)\b|\([A-F]\)", re.I)
_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z(\"'*\d])|\n+")
ORD = re.compile(r"\b(cover(?:s|ing|ed)?\b(?!\s+by)|on top of|obscur\w*\b(?!\s+by)|block\w*\b(?!\s+by)|in front of|occlud(?:es|ing)\b)", re.I)
NEG = re.compile(r"\b(not|n't|never|no longer)\b", re.I)


def _objs(s: str) -> list[str]:
    return [(o.group(1) + " " + o.group(2)).lower().replace("grey", "gray") for o in OBJ.finditer(s)]


def first_claim_v2(sents: list[str], negation: bool = False) -> tuple[int, str] | None:
    """(sentence index, claimed-front object) for the first sentence with an ordered two-object relation."""
    for j, s in enumerate(sents):
        objs = _objs(s)
        if len(set(objs)) < 2 or SKIP.search(s):
            continue
        a, p = ORD.search(s), PAS.search(s)
        if not a and not p:
            continue
        use_pas = p is not None and (a is None or p.start() < a.start())
        m = p if use_pas else a
        before = [(o.group(1) + " " + o.group(2)).lower().replace("grey", "gray") for o in OBJ.finditer(s) if o.end() <= m.start()]
        after = [o for o in OBJ.finditer(s) if o.start() >= m.end()]
        subj = before[-1] if before and after else objs[0]
        other = [o for o in objs if o != subj][0]
        claimed = other if use_pas else subj
        first = subj
        if negation:
            if NEG.search(s[: m.start()][-25:]):
                claimed = other if claimed == first else first
        return j, claimed
    return None


def first_claim_v3(sents: list[str]) -> tuple[int, str] | None:
    return first_claim_v2(sents, negation=True)


def first_claim(sents: list[str]) -> tuple[int, int] | None:
    """(index of the first two-object occlusion/front claim, claimed-front object) or None."""
    for j, s in enumerate(sents):
        objs = [(o.group(1) + " " + o.group(2)).lower().replace("grey", "gray") for o in OBJ.finditer(s)]
        if len(set(objs)) < 2 or not (ACT.search(s) or PAS.search(s)) or SKIP.search(s):
            continue
        first = objs[0]
        claimed = first if not PAS.search(s) else [o for o in objs if o != first][0]
        return j, claimed
    return None


def split_sents(text: str) -> list[str]:
    """Plain sentence split (same boundary rule as e2b.split_sentences, without list-marker merging)."""
    return [s.strip() for s in _SPLIT.split(text) if s and s.strip()]


def stype(s: str) -> str:
    return "conclusion" if CONCL.search(s) else ("relation" if REL.search(s) else "non_relational")
