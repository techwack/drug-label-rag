# Drug Label RAG

Retrieval-augmented question answering over official FDA drug labels (openFDA) for 20 common drugs.
It runs fully locally with **Llama 3.1 8B** and **nomic-embed-text** through Ollama. Answers cite
their sources, and the system refuses when the answer isn't in the labels.

> Example: _paste one real question, answer and sources from `python rag.py ...` here_

## How it works

```
openFDA API → fetch_data.py → section-aware chunks → nomic-embed-text → FAISS
question → drug-name detection (generic + brand) → filtered top-k retrieval → grounded prompt → Llama 3.1 → cited answer
```

## Run it

```bash
ollama pull llama3.1:8b
ollama pull nomic-embed-text
python -m venv .venv && .venv\Scripts\activate      # macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
python fetch_data.py        # download the labels
python ingest.py            # chunk, embed and build the FAISS index
python rag.py "What is the boxed warning for warfarin?"
python evaluate.py          # add --no-filter to compare against plain vector search
```

## Evaluation

`eval/questions.json` has 19 questions: 16 answerable from the labels and 3 that the system should refuse
(off-topic, a drug that isn't in the corpus, and a prompt-injection attempt).

| Metric | With drug filter | Without filter |
|---|---|---|
| Retrieval hit rate (right drug + section in top-k) | _fill_ | _fill_ |
| Answer accuracy (expected facts present) | _fill_ | _fill_ |
| Citation rate | _fill_ | _fill_ |
| Refusal accuracy | _fill_ | _fill_ |
| Avg latency (s) | _fill_ | _fill_ |

## Design decisions

- **RAG instead of fine-tuning.** Labels change, and answers must be traceable to the source text.
  RAG updates by re-indexing and gives citations. Fine-tuning teaches style, not reliable facts,
  and it can't show where an answer came from.
- **Section-aware chunking.** Each label section is split on its own, so a chunk never mixes
  "Contraindications" with "Adverse Reactions". Each chunk also gets a `[drug | section]` header,
  so its embedding knows the context even when the passage never names the drug.
- **Metadata filtering.** Drug labels share a lot of wording ("may cause bleeding"), so plain
  similarity search often retrieves the wrong drug. Detecting the drug in the question (including
  brand names such as Plavix → clopidogrel) and filtering the search fixes this. `--no-filter`
  measures how much it helps.
- **Grounded prompt.** The model may use only the excerpts, must cite them, refuses with a fixed
  sentence when the answer isn't there, and treats the excerpts as data rather than instructions,
  which defends against prompt injection. Temperature 0 keeps evaluation runs repeatable.
- **Fully local.** No API keys and no data leaves the machine.

## Limitations

- One label per drug. Generic manufacturers' labels can differ.
- Keyword-based answer scoring is strict about wording. Failures were reviewed by hand in `eval/results/`.
- This is not medical advice.
