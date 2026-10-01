"""Chunking, the FAISS vector store, and drug-aware retrieval."""
import json
import re
import time

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config import CHUNK_OVERLAP, CHUNK_SIZE, EMBED_MODEL, INDEX_DIR, LABEL_DIR, SECTIONS, TOP_K


def get_embeddings():
    from langchain_ollama import OllamaEmbeddings
    return OllamaEmbeddings(model=EMBED_MODEL)


def load_labels():
    return [json.loads(p.read_text()) for p in sorted(LABEL_DIR.glob("*.json"))]


def chunk_labels(labels):
    """Split each section separately so a chunk never mixes two sections.

    Every chunk starts with a header like "[warfarin | Boxed Warning]" so the
    embedding knows which drug and section the text belongs to, even when the
    passage itself never names the drug.
    """
    splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    docs = []
    for label in labels:
        for field, text in label["sections"].items():
            title = SECTIONS[field]
            for i, piece in enumerate(splitter.split_text(text)):
                docs.append(Document(
                    page_content=f"[{label['drug']} | {title}]\n{piece}",
                    metadata={"drug": label["drug"], "section": title, "chunk": i,
                              "source": f"{label['drug']} – {title}"},
                ))
    return docs


def _with_retry(fn, tries=3):
    """Ollama's model runner can restart mid-run; wait and retry a batch instead of losing all progress."""
    for attempt in range(tries):
        try:
            return fn()
        except Exception as e:
            if attempt == tries - 1:
                raise
            print(f"  retrying batch after error: {str(e)[:80]}")
            time.sleep(5)


def build_index(docs, embeddings=None, batch_size=64):
    """Embed in small batches so one failed request doesn't lose the whole run."""
    embeddings = embeddings or get_embeddings()
    store = None
    for start in range(0, len(docs), batch_size):
        batch = docs[start:start + batch_size]
        if store is None:
            store = _with_retry(lambda: FAISS.from_documents(batch, embeddings))
        else:
            _with_retry(lambda: store.add_documents(batch))
        print(f"  embedded {min(start + batch_size, len(docs))}/{len(docs)}")
    store.save_local(str(INDEX_DIR))
    return store


def load_index(embeddings=None):
    return FAISS.load_local(str(INDEX_DIR), embeddings or get_embeddings(),
                            allow_dangerous_deserialization=True)  # index is built locally by us


def name_map(labels):
    """Map every generic and brand name to its generic drug name."""
    names = {}
    for label in labels:
        names[label["drug"]] = label["drug"]
        for brand in label.get("brand_names", []):
            if len(brand) > 3:  # skip short brand names that collide with words
                names.setdefault(brand, label["drug"])
    return names


def drugs_in(question, names):
    q = question.lower()
    return sorted({drug for name, drug in names.items()
                   if re.search(rf"\b{re.escape(name)}\b", q)})


def retrieve(store, question, names, k=TOP_K, use_filter=True):
    """Return the top-k chunks, limited to the drugs the question mentions."""
    drugs = drugs_in(question, names) if use_filter else []
    if not drugs:
        return store.similarity_search(question, k=k), drugs
    if len(drugs) == 1:
        return store.similarity_search(question, k=k, filter={"drug": drugs[0]}, fetch_k=100), drugs
    # Several drugs (e.g. an interaction question): take an even share from each.
    per = max(2, k // len(drugs))
    docs = []
    for d in drugs:
        docs += store.similarity_search(question, k=per, filter={"drug": d}, fetch_k=100)
    return docs, drugs
