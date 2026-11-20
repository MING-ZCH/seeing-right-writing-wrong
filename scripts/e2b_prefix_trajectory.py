"""E2b: sentence-level early-answering trajectory of R and commit-sentence typing.

For each selected item, R is split into sentences ``s_1..s_J``. For every
prefix ``R_{1:j}`` (j = 0..J; j = 0 is the trigger only) the assistant turn is
closed, ``Now give your final answer.`` is added, and the forced readout gives
``P_j(gold)``. The commit sentence ``j*`` is the first ``j >= 1`` such that
``P_k(gold) < 0.5`` for every ``k >= j`` (``None`` if ``P_J >= 0.5``).

Sentences are typed by a keyword classifier into ``object_description``,
``relation_statement``, ``heuristic_rule`` and ``conclusion``. For the
cue-decoupled occlusion set, relation and size claims are checked against the
generator metadata. For other sets a judge-request JSONL is written for an
LLM judge plus 200-sentence human audit (kappa reported separately); this
script does not call any API.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from common import (COT_TRIGGER, JsonlWriter, Prompt, RationaleStore, Turn, add_common_args, build_closed_readout,
                    cluster_bootstrap_ci, first_user_turn, iter_progress, load_adapter, load_all_items,
                    parse_stated_option, read_jsonl, set_seed, split_sentences, variant_images, write_run_meta)
from e2_groups import e1_index, make_groups

_REL = re.compile(r"\b(left|right|above|below|under|beneath|on top|in front|behind|closer|nearer|farther|further|"
                  r"occlud\w*|overlap\w*|cover\w*|hidden|blocks?|next to|between|distance|far from|near)\b", re.I)
_HEUR = re.compile(r"(\b(larger|bigger|smaller|taller|lower|higher)\b.*\b(closer|nearer|farther|front|behind|"
                   r"further|distance)\b)|(\b(closer|nearer|farther|front)\b.*\b(larger|bigger|smaller|appear\w*)\b)|"
                   r"\b(generally|typically|usually|tend to|in general|as a rule|perspective)\b", re.I)
_CONCL = re.compile(r"\b(answer is|therefore|thus|hence|so the|in conclusion|the correct (option|answer))\b", re.I)


def classify_sentence(s: str, n_options: int) -> str:
    """Keyword classifier for sentence type (heuristic; audited by judge/human)."""
    if _CONCL.search(s) or parse_stated_option(s, n_options) is not None:
        return "conclusion"
    if _HEUR.search(s):
        return "heuristic_rule"
    if _REL.search(s):
        return "relation_statement"
    return "object_description"


def check_occlusion_claim(s: str, meta: dict[str, Any]) -> bool | None:
    """Truth of a relation/size claim on occlusion items using generator metadata.

    Returns ``True``/``False`` for a recognized claim about the two named
    objects, else ``None``.
    """
    f, b = meta.get("front_name"), meta.get("back_name")
    if not f or not b:
        return None
    low = s.lower()

    def pos(name: str) -> int:
        i = low.find(name)
        if i < 0:
            i = low.find(name.split()[-1])
        return i

    pf, pb = pos(f), pos(b)
    if pf < 0 or pb < 0:
        return None
    subj_front = pf < pb  # the first mentioned object is the grammatical subject
    if re.search(r"in front of|occlud|overlap|cover|on top of|closer|nearer|block", low):
        return subj_front
    if re.search(r"behind|hidden|farther|further|partially covered|underneath|beneath", low):
        return not subj_front
    if re.search(r"larger|bigger", low):
        a_f, a_b = meta.get("area_front_px"), meta.get("area_back_px")
        return (a_f > a_b) == subj_front if a_f and a_b else None
    if re.search(r"smaller", low):
        a_f, a_b = meta.get("area_front_px"), meta.get("area_back_px")
        return (a_f < a_b) == subj_front if a_f and a_b else None
    return None


def commit_index(p_gold: list[float]) -> int | None:
    """First j >= 1 after which P(gold) stays below 0.5."""
    if not p_gold or p_gold[-1] >= 0.5:
        return None
    j = len(p_gold) - 1
    while j - 1 >= 1 and p_gold[j - 1] < 0.5:
        j -= 1
    return j


def prefix_text(sents: list[str], j: int) -> str:
    """Assistant text for prefix ``R_{1:j}`` (trigger included)."""
    return (COT_TRIGGER + (" " + " ".join(sents[:j]) if j > 0 else "")).strip()


def summarize(path: Path) -> dict[str, Any]:
    """Commit-sentence type distribution and truth rates by group."""
    recs = list(read_jsonl(path))
    out: dict[str, Any] = {}
    by_g: dict[str, list[dict]] = defaultdict(list)
    for r in recs:
        by_g[r["group"]].append(r)
    for g, rs in by_g.items():
        types = Counter(r["commit_type"] for r in rs if r["commit_index"] is not None)
        truth = [r["commit_truth"] for r in rs if r.get("commit_truth") is not None]
        has = np.array([float(r["commit_index"] is not None) for r in rs])
        out[g] = {"n": len(rs), "commit_found": cluster_bootstrap_ci(has, [r["image_id"] for r in rs]),
                  "commit_type_counts": dict(types),
                  "commit_claim_true_frac": float(np.mean(truth)) if truth else None, "n_checked": len(truth),
                  "mean_n_sentences": float(np.mean([r["n_sentences"] for r in rs]))}
    return out


def main() -> None:
    """CLI entry point."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_args(ap)
    ap.add_argument("--e1", default=None)
    ap.add_argument("--pred-key", default="pred_sum")
    ap.add_argument("--groups", nargs="+", default=["D+C-", "D+C+"])
    ap.add_argument("--max-per-group", type=int, default=None)
    ap.add_argument("--max-sentences", type=int, default=24)
    ap.add_argument("--summarize", action="store_true")
    args = ap.parse_args()
    set_seed(args.seed)
    out_path = Path(args.out_dir) / "e2b" / f"{args.model}.jsonl"
    if args.summarize:
        s = summarize(out_path)
        Path(str(out_path).replace(".jsonl", "_summary.json")).write_text(json.dumps(s, indent=2))
        print(json.dumps(s, indent=1))
        return
    items = load_all_items(args.items, args.tasks, args.limit)
    by_id = {i.item_id: i for i in items}
    e1 = e1_index(args.e1 or Path(args.out_dir) / "e1" / f"{args.model}.jsonl", args.pred_key)
    groups = {k: g for k, g in make_groups(items, e1, args.seed, max_per_group=args.max_per_group).items()
              if g in args.groups}
    adapter, prov = load_adapter(args.model, args.model_path, args.device, args.revision,
                                 str(Path(args.out_dir) / "provenance"))
    write_run_meta(out_path, args, {"provenance": prov, "n_selected": len(groups)})
    store = RationaleStore(args.rationales or Path(args.out_dir) / "rationales" / f"{args.model}.jsonl")
    w = JsonlWriter(out_path, ["item_id"])
    judge = open(Path(args.out_dir) / "e2b" / f"{args.model}_judge_requests.jsonl", "a")
    for iid in iter_progress(sorted(groups), 25, f"E2b {args.model}"):
        if w.has(item_id=iid):
            continue
        it = by_id[iid]
        rec = store.get(iid, "cot", "original")
        text = rec["text"] if rec else (e1["cot"][iid].get("rationale") or "")
        sents = split_sentences(text)[: args.max_sentences]
        imgs = variant_images(it, "original")
        user1 = first_user_turn(it.q_text(), imgs)
        traj = []
        for j in range(len(sents) + 1):
            res = adapter.readout(build_closed_readout([user1], prefix_text(sents, j)), it.letters)
            traj.append({"j": j, "p_gold": res["p_sum"][it.gold], "pred": res["pred_sum"], "p": res["p_sum"]})
        pg = [t["p_gold"] for t in traj]
        j_star = commit_index(pg)
        types = [classify_sentence(s, len(it.options)) for s in sents]
        truths = [check_occlusion_claim(s, it.meta) for s in sents]
        w.write({"model": args.model, "item_id": iid, "task": it.task, "image_id": it.image_id,
                 "group": groups[iid], "gold": it.gold, "sentences": sents, "sentence_types": types,
                 "sentence_truth": truths, "trajectory": traj, "n_sentences": len(sents), "commit_index": j_star,
                 "commit_sentence": sents[j_star - 1] if j_star else None,
                 "commit_type": types[j_star - 1] if j_star else None,
                 "commit_truth": truths[j_star - 1] if j_star else None})
        for k, s in enumerate(sents):
            judge.write(json.dumps({"model": args.model, "item_id": iid, "sentence_index": k + 1, "sentence": s,
                                    "question": it.q_text(), "image_paths": it.image_paths,
                                    "heuristic_type": types[k], "is_commit": j_star == k + 1}) + "\n")
    judge.close()
    w.close()
    s = summarize(out_path)
    Path(str(out_path).replace(".jsonl", "_summary.json")).write_text(json.dumps(s, indent=2))


if __name__ == "__main__":
    main()
