"""Claim span helper shared by e17_probe_ctrl.py and CPU analyses (no vLLM / torch imports)."""
from __future__ import annotations

from claim_parse import NEG, ORD, first_claim_v3
from e2f_claim_edit import OBJ, PAS, split_sents

COT_PREFIX = "Let's think step by step."


def claim_spans(cot: str) -> dict | None:
    """Char offsets (in the CoT text) of the claim's first object mention and relation phrase, and the claimed front."""
    body = cot[len(COT_PREFIX):]
    sents = split_sents(body)
    fc = first_claim_v3(sents)
    if fc is None:
        return None
    j, named = fc
    pos = len(COT_PREFIX)
    for k in range(j + 1):
        pos = cot.find(sents[k], pos)
        if k < j:
            pos += len(sents[k])
    s = sents[j]
    a, p = ORD.search(s), PAS.search(s)
    m = p if (p is not None and (a is None or p.start() < a.start())) else a
    o = OBJ.search(s)
    return {"subj_start": pos + o.start(), "subj_end": pos + o.end() - 1, "rel_start": pos + m.start(), "rel_end": pos + m.end() - 1,
            "named": named, "negated": bool(NEG.search(s[: m.start()][-25:])), "sentence": s}
