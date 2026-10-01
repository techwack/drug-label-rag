"""Run the evaluation suite and report retrieval, answer and refusal accuracy.

Usage: python evaluate.py [--no-filter]
"""
import json
import re
import sys
import time

from config import EVAL_DIR
from rag import RAG


def is_refusal(answer):
    return "can't answer" in answer.lower() or "cannot answer" in answer.lower()


def matches(keyword, text):
    """True if any '|'-separated alternative appears in text, matched literally (so "H+/K+" works)."""
    return any(alt.lower() in text for alt in keyword.split("|"))


def check(item, result):
    """Score one result. Keywords use '|' for acceptable alternatives."""
    if item.get("refuse"):
        return {"refused": is_refusal(result["answer"])}
    want = f"{item['drug']} – {item['section']}"
    # Labels differ: some use "Warnings" instead of "Warnings and Precautions".
    alt = want.replace("Warnings and Precautions", "Warnings")
    hit = any(s in (want, alt) or s.startswith(want) for s in result["sources"])
    answer = result["answer"].lower()
    correct = all(matches(k, answer) for k in item["keywords"]) and not is_refusal(answer)
    return {"retrieval_hit": hit, "correct": correct, "cited": bool(re.search(r"\[\d+(?:\s*,\s*\d+)*\]", answer))}


def main(use_filter=True):
    items = json.loads((EVAL_DIR / "questions.json").read_text())
    rag = RAG()
    rows = []
    for item in items:
        t = time.time()
        result = rag.ask(item["q"], use_filter=use_filter)
        score = check(item, result)
        rows.append({**result, **score, "seconds": round(time.time() - t, 1)})
        mark = "PASS" if all(score.values()) else "FAIL"
        print(f"{mark}  {item['q']}")

    answer_rows = [r for r in rows if "correct" in r]
    refuse_rows = [r for r in rows if "refused" in r]
    pct = lambda xs: f"{100 * sum(xs) / len(xs):.0f}% ({sum(xs)}/{len(xs)})"
    summary = {
        "mode": "filter" if use_filter else "no-filter",
        "retrieval_hit_rate": pct([r["retrieval_hit"] for r in answer_rows]),
        "answer_accuracy": pct([r["correct"] for r in answer_rows]),
        "citation_rate": pct([r["cited"] for r in answer_rows]),
        "refusal_accuracy": pct([r["refused"] for r in refuse_rows]),
        "avg_seconds": round(sum(r["seconds"] for r in rows) / len(rows), 1),
    }
    print()
    for k, v in summary.items():
        print(f"{k:20} {v}")
    out = EVAL_DIR / "results"
    out.mkdir(exist_ok=True)
    (out / f"{summary['mode']}.json").write_text(json.dumps({"summary": summary, "rows": rows}, indent=2))
    print(f"\nDetails in eval/results/{summary['mode']}.json")


if __name__ == "__main__":
    main(use_filter="--no-filter" not in sys.argv)
