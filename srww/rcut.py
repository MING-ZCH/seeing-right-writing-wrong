"""Rationale truncation rules.

strip_answer  R_cut: the model's own rationale with its concluding answer sentence(s) removed. R is cut at the
              start of the sentence holding the last answer mention, repeated while the tail still mentions one.
              Used for the fixed-rationale image swaps and the knockouts.
truncate      Conclusion removal for written-relation extraction (two-channel accounting, closed models): drops
              answer sentences, everything after the last answer sentence, a 'Conclusion' header and what follows
              it, and trailing concluding sentences or sentences that merely restate an option.
"""
from __future__ import annotations

import re
from collections import Counter

from srww import protocol as C

ANS_RX = re.compile(r"answer|\([A-D]\)|\b[A-D]\)|option [A-D]\b", re.I)


def strip_answer(r: str) -> str:
    """Cut R at the start of the sentence holding the last answer mention; repeat while the tail mentions one."""
    text = r
    for _ in range(4):
        ms = list(ANS_RX.finditer(text))
        if not ms:
            break
        k = ms[-1].start()
        cut = max(text.rfind(". ", 0, k), text.rfind("\n", 0, k), text.rfind(": ", 0, k))
        if cut <= len(C.COT_PREFIX):
            text = text[:len(C.COT_PREFIX)]
            break
        text = text[:cut + 1].rstrip()
        if len(text) < 0.3 * len(r):
            break
    return text


ANS = re.compile(r"\b(?:the|final|correct|my|right)\s+(?:answer|option|choice)\b|\banswer\s*(?:is|would be|should be|:)|"
                 r"^\W*answer\b|\b(?:option|choice)\s*\(?\s*[AB]\b|(?<!point )(?<!points )(?<!circle )(?<!side )(?<!marker )(?<!labeled )(?<!label )(?<!box )"
                 r"(?<!object )(?<!location )(?<!dot )(?<!and )\(\s*[AB]\s*\)|\*\*\s*\(?[AB]\)?\s*\*\*|^\W*[AB]\W*$", re.I)
HEADER = re.compile(r"^\W*(?:\d+\.\s*)?\**\s*(?:conclusion|final answer|answer|summary)\b\W*$", re.I)
HEADER_INLINE = re.compile(r"^\W*(?:\d+\.\s*)?\*\*\s*(?:conclusion|final answer|summary)\s*\**\s*:", re.I)
CONC = re.compile(r"^\W*(?:\d+\.\s*)?\**\s*(?:therefore|thus|so\b|hence|in conclusion|to conclude|in summary|to summarize|"
                  r"overall|consequently|ultimately|based on|given (?:this|these|that|the)|considering|"
                  r"from (?:this|these|the)|it is (?:logical|reasonable|clear|safe) to conclude|"
                  r"we can (?:conclude|determine|infer|say)|putting (?:this|it) (?:all )?together|in this case,? the|conclusion)", re.I)
CMP = r"(?:closer|nearer|closest|nearest)"


def _norm(s: str) -> str:
    s = re.sub(r"[*_`#>]", "", s).strip().lower()
    s = re.sub(r"^(?:[-\u2022]\s*|\d+\.\s*)", "", s)
    s = re.sub(r"\s*\((?:highlighted|marked|in|labeled)[^)]*\)", "", s)
    s = re.sub(r"\b(?:highlighted|marked|outlined) (?:by|with|in) (?:a |the )?\w+ box\b", "", s)
    return re.sub(r"\s+", " ", s).strip(" .!")


def restates(s: str, opts: list[str]) -> bool:
    """Sentence is only 'X is closer (to the camera) (than Y)' or 'the object closer to the camera is X' for an option X."""
    t = _norm(s)
    t = re.sub(r"^(?:therefore|thus|so|hence|clearly|overall)\b,?\s*", "", t)
    o = "|".join(re.escape(x.lower()) for x in opts)
    obj = rf"(?:the |point |circle |object )?(?:{o})(?: object| point)?"
    pats = [rf"^{obj} (?:is|appears to be|seems to be|would be|should be|must be) (?:clearly |definitely |indeed |likely |probably )?"
            rf"(?:the )?{CMP}(?: one| object| point)?(?: to the (?:camera|viewer|observer))?(?: than {obj})?$",
            rf"^the (?:object|point|one) (?:that is |which is )?{CMP} to the (?:camera|viewer) is {obj}$",
            rf"^{obj} is {CMP}$"]
    return any(re.match(p, t) for p in pats)


def truncate(sents: list[str], opts: list[str]) -> tuple[list[str], dict]:
    """Remove the conclusion so that the remaining text cannot carry the answer itself."""
    info = Counter()
    s = [x for x in sents if x.strip()]
    hits = [j for j, x in enumerate(s) if ANS.search(x)]
    if hits:
        info["post_answer_dropped"] = len(s) - 1 - hits[-1]
        s = s[: hits[-1]]
        s2 = [x for x in s if not ANS.search(x)]
        info["answer_sents_dropped"] = len(hits) - 0
        s = s2
    hdr = [j for j, x in enumerate(s) if HEADER.match(x) or HEADER_INLINE.match(x)]
    if hdr:
        info["after_header_dropped"] = len(s) - hdr[-1]
        s = s[: hdr[-1]]
    while s and (CONC.match(s[-1]) or restates(s[-1], opts) or not re.search(r"[A-Za-z]{2,}", s[-1]) or s[-1].rstrip().endswith(":")):
        info["tail_dropped"] += 1
        s = s[:-1]
    return s, dict(info)
