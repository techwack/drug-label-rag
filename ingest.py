"""Chunk the downloaded labels and build the FAISS index."""
from collections import Counter

from store import build_index, chunk_labels, load_labels


def main():
    labels = load_labels()
    if not labels:
        raise SystemExit("No labels found. Run `python fetch_data.py` first.")
    docs = chunk_labels(labels)
    print(f"{len(labels)} labels -> {len(docs)} chunks")
    for section, n in Counter(d.metadata["section"] for d in docs).most_common():
        print(f"  {section:30} {n}")
    print("Embedding (first run takes a few minutes)...")
    build_index(docs)
    print("Saved index/")


if __name__ == "__main__":
    main()
