# Python Docs Assistant (RAG)

A small question-answering assistant over the official Python documentation. You type a question, it finds the most relevant passages using hybrid (semantic + keyword) search and shows the best match along with its sources.

I originally built this as a personal project to learn how retrieval-augmented generation works under the hood. This repo is a cleaned-up rebuild of that project, written with help from AI coding tools.

## How it works

1. **Fetch**: `fetch_docs.py` downloads 18 Python tutorial and library pages and saves them as plain text, with each page's URL and title.
2. **Chunk**: the text is split into overlapping ~180-word chunks so each one holds a single idea.
3. **Embed**: each chunk (prefixed with its page title) is run through DistilBERT, and the token vectors are mean-pooled into one embedding per chunk.
4. **Index**: embeddings are L2-normalised and stored in a FAISS inner-product index, so search scores are cosine similarity.
5. **Retrieve**: the question is scored two ways, by FAISS (semantic) and by BM25 (keyword), and the two rankings are merged with weighted reciprocal rank fusion.
6. **Answer**: the Flask app returns the best chunk with its top 3 sources, links back to docs.python.org, and both scores.

The answer step is extractive for now: it returns real text from the docs instead of generating new text, so it can't hallucinate. If no keyword in the question appears in the docs, it says so instead of guessing.

## Why hybrid search

Plain DistilBERT isn't trained for sentence similarity, so its embeddings rank passages only loosely. I measured this with `evaluate.py`, which asks 14 questions where the right docs page is known:

| Retrieval            | hit@1 | hit@3 |
| -------------------- | ----- | ----- |
| FAISS only (dense)   | 43%   | 71%   |
| BM25 only (keyword)  | 86%   | 100%  |
| **Hybrid (default)** | **93%** | **100%** |

So BM25 carries most of the weight, and the semantic ranking breaks ties between close keyword matches.

## Stack

Python, Flask, Hugging Face Transformers (DistilBERT), FAISS, NumPy, BeautifulSoup

## Run it

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt

python fetch_docs.py      # download the docs
python build_index.py     # embed and index them (first run downloads DistilBERT, ~250 MB)
python app.py             # open http://127.0.0.1:5000
python evaluate.py        # optional: measure retrieval quality
```

## API

- `POST /ask` with `{"question": "..."}` returns `{"answer": "...", "sources": [{"title", "source", "text", "score", "cosine", "bm25"}]}`
- `GET /health` returns the number of indexed chunks

## Project layout

```
fetch_docs.py      download and clean docs pages -> data/docs/
build_index.py     chunk, embed and save the FAISS index -> data/
rag.py             chunking, embeddings, BM25 and hybrid retrieval
app.py             Flask web app and JSON API
evaluate.py        hit@1 / hit@3 on a small question set
templates/         single-page UI
```

## What I'd improve

- Use an LLM to write a short answer from the retrieved chunks, with citations
- Swap DistilBERT for a model trained for sentence embeddings (e.g. a sentence-transformers model), which should make the semantic side pull its weight
- Grow the evaluation set and track it in CI
- Cover more of the docs and re-index automatically when they change
