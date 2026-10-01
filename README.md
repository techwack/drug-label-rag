# Drug Label RAG

Retrieval-augmented question answering over official FDA drug labels (openFDA) for 20 common drugs.
It runs fully locally with **Llama 3.1 8B** and **nomic-embed-text** through Ollama. Answers cite
their sources, and the system refuses when the answer isn't in the labels.

```
$ python rag.py "What does the Plavix label say about poor metabolizers?"
Tests are available to identify patients who are CYP2C19 poor metabolizers [2]. Consider use of
another platelet P2Y 12 inhibitor in patients identified as CYP2C19 poor metabolizers [1, 2].

Sources:
  [1] clopidogrel – Boxed Warning
  [2] clopidogrel – Boxed Warning
  ...
```
"Plavix" is a brand name, so the system first maps it to clopidogrel and searches only that label.

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

Results (llama3.1:8b, temperature 0, top-k 5, on an RTX 5060 laptop):

| Metric | First run | After fixes (filter) | After fixes (no filter) |
|---|---|---|---|
| Retrieval hit rate (right drug + section in top-k) | 100% (16/16) | 100% (16/16) | 100% (16/16) |
| Answer accuracy (expected facts present) | 88% (14/16) | 100% (16/16) | 100% (16/16) |
| Citation rate | 94% (15/16) | 100% (16/16) | 100% (16/16) |
| Refusal accuracy | 100% (3/3) | 100% (3/3) | 100% (3/3) |
| Avg latency | 3.0 s | 1.6 s | 1.2 s |

### What the first run found

`python show_failures.py` prints the full answer for each failed question. The three failures had three different causes:

1. **Omeprazole was a bug in the evaluation, not the system.** The answer correctly said "H+/K+ ATPase", but the
   keyword was matched as a regex, where `+` is a quantifier, so it could never match. Keywords are now matched literally.
2. **Warfarin: the model copied the label instead of answering.** The answer was the boxed warning verbatim, with no citation.
3. **Levothyroxine: the model replied with only `[1]`.**

Fix for 2 and 3: the prompt now asks for complete sentences in the model's own words, with a citation after every
sentence, and includes one example of the format (a few-shot example). All 16 answerable questions then passed,
and the refusal tests still passed.

### What the filter comparison showed

The drug-name filter made no difference on this set. Every question names its drug and every chunk starts with a
`[drug | section]` header, so plain vector search already finds the right label. The filter is a safeguard for vaguer
questions and for labels with near-identical wording. The next step would be harder test questions to measure it properly.

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
- Keyword-based answer scoring checks that key facts are present, not full correctness. Failures are reviewed by hand with `show_failures.py`.
- 19 questions is a small test set; results show the method works, not production-level accuracy.
- This is not medical advice.
