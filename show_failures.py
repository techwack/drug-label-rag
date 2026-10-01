"""Print the full answers for questions that failed the last eval run.

Usage: python show_failures.py [filter|no-filter]
"""
import json
import sys

from config import EVAL_DIR

mode = sys.argv[1] if len(sys.argv) > 1 else "filter"
rows = json.loads((EVAL_DIR / "results" / f"{mode}.json").read_text())["rows"]
for r in rows:
    failed = [k for k in ("retrieval_hit", "correct", "cited", "refused") if k in r and not r[k]]
    if failed:
        print(f"Q: {r['question']}\n   failed: {', '.join(failed)}\n   A: {r['answer']}")
        print(f"   sources: {r['sources']}\n")
