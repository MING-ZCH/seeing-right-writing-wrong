"""First depth-order claim parser, v2/v3 (round-5 fix of e2f_claim_edit.first_claim).

v1 (e2f_claim_edit.first_claim) counts symmetric verbs ("A overlaps B", "overlapping") as "A is in front".
v2: a sentence counts only if it carries an ordered relation: active (covers, on top of, obscures, blocks, in front of)
    or passive (covered / hidden / obscured / blocked / overlapped / occluded by, behind, underneath, beneath).
v3: v2 plus negation ("A is not in front of B", "A does not cover B") -> the other object is claimed in front.
"""
from __future__ import annotations

import re

from e2f_claim_edit import OBJ, PAS, SKIP

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
