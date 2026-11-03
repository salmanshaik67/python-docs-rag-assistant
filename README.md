# Python Docs Assistant (RAG)

A small question-answering assistant over the official Python documentation. You type a question, it finds the most relevant passages using semantic search and shows the best match along with its sources.

I originally built this as a personal project to learn how retrieval-augmented generation works under the hood. This repo is a cleaned-up rebuild of that project, written with help from AI coding tools.

## How it works

1. **Fetch**: `fetch_docs.py` downloads a few Python tutorial and library pages and saves them as plain text.
2. **Chunk**: the text is split into overlapping ~180-word chunks so each one holds a single idea.
3. **Embed**: each chunk is run through DistilBERT, and the token vectors are mean-pooled into one embedding per chunk.
4. **Index**: embeddings are L2-normalised and stored in a FAISS inner-product index, so search scores are cosine similarity.
5. **Answer**: the Flask app embeds the question, pulls the top 3 chunks, and returns the best one with its sources and similarity scores.

The answer step is extractive for now: it returns real text from the docs instead of generating new text, so it can't hallucinate. Adding an LLM to summarise the retrieved passages is the next step.

## Stack

Python, Flask, Hugging Face Transformers (DistilBERT), FAISS, NumPy

## Run it

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt

python fetch_docs.py      # download the docs
python build_index.py     # embed and index them (first run downloads DistilBERT)
python app.py             # open http://127.0.0.1:5000
```

## What I'd improve

- Use an LLM to write a short answer from the retrieved chunks, with citations
- Swap DistilBERT for a model trained for sentence embeddings, for better retrieval
- Add a small evaluation set of questions to measure retrieval quality
- Cover more of the docs and re-index automatically when they change
