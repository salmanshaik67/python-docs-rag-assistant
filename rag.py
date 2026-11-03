"""Shared retrieval code: chunking, DistilBERT embeddings, BM25 and hybrid FAISS search."""

import json
import re
from collections import Counter
from pathlib import Path

import faiss
import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

MODEL_NAME = "distilbert-base-uncased"
DATA_DIR = Path(__file__).parent / "data"
INDEX_PATH = DATA_DIR / "index.faiss"
CHUNKS_PATH = DATA_DIR / "chunks.json"

CHUNK_WORDS = 180
CHUNK_OVERLAP = 40

CANDIDATES = 50   # dense hits considered before fusion
RRF_K = 60        # standard reciprocal-rank-fusion constant
# Plain DistilBERT isn't trained for sentence similarity, so its ranking is noisy;
# trust BM25 more and let the semantic side break ties and cover keyword misses.
DENSE_WEIGHT, KEYWORD_WEIGHT = 0.1, 1.0
BM25_K1, BM25_B = 1.5, 0.75

STOPWORDS = {
    "a", "an", "the", "and", "or", "of", "to", "in", "on", "for", "with", "is", "are",
    "was", "be", "do", "does", "how", "what", "why", "when", "where", "which", "who",
    "i", "you", "my", "it", "its", "can", "should", "use", "using", "python", "work",
    "works", "get", "make", "from", "by", "as", "at", "that", "this", "into", "about",
}


def stem(word: str) -> str:
    """Very small suffix stripper, so 'handling'/'handle' and 'classes'/'class' match."""
    for suffix in ("ing", "ed", "s", "e"):
        if word.endswith(suffix) and len(word) - len(suffix) >= 3 and not word.endswith("ss"):
            word = word[: -len(suffix)]
    return word


def tokenize(text: str) -> list[str]:
    """Lower-cased, stemmed content words."""
    words = re.findall(r"[a-z_][a-z0-9_]+", text.lower())
    return [stem(w) for w in words if w not in STOPWORDS]


class BM25:
    """Okapi BM25 keyword scoring over the chunk corpus."""

    def __init__(self, docs: list[list[str]]):
        self.docs = [Counter(d) for d in docs]
        self.lengths = np.array([len(d) for d in docs], dtype="float32")
        self.avg_len = float(self.lengths.mean())
        df = Counter(term for d in self.docs for term in d)
        n = len(docs)
        self.idf = {t: np.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}

    def scores(self, query: list[str]) -> np.ndarray:
        out = np.zeros(len(self.docs), dtype="float32")
        norm = BM25_K1 * (1 - BM25_B + BM25_B * self.lengths / self.avg_len)
        for term in set(query):
            idf = self.idf.get(term)
            if idf is None:
                continue
            tf = np.array([d.get(term, 0) for d in self.docs], dtype="float32")
            out += idf * tf * (BM25_K1 + 1) / (tf + norm)
        return out


def chunk_text(text: str, size: int = CHUNK_WORDS, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """Split text into overlapping windows of roughly `size` words."""
    words = text.split()
    if not words:
        return []
    step = size - overlap
    chunks = []
    for start in range(0, len(words), step):
        chunks.append(" ".join(words[start:start + size]))
        if start + size >= len(words):
            break
    return chunks


class Embedder:
    """Mean-pooled, L2-normalised DistilBERT sentence embeddings."""

    def __init__(self, model_name: str = MODEL_NAME):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.eval()

    @torch.no_grad()
    def embed(self, texts: list[str], batch_size: int = 16) -> np.ndarray:
        vectors = []
        for i in range(0, len(texts), batch_size):
            batch = self.tokenizer(
                texts[i:i + batch_size],
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt",
            )
            hidden = self.model(**batch).last_hidden_state
            # Average token vectors, ignoring padding.
            mask = batch["attention_mask"].unsqueeze(-1).float()
            pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
            vectors.append(pooled.numpy())
        out = np.vstack(vectors).astype("float32")
        faiss.normalize_L2(out)
        return out


class Retriever:
    """Loads the saved index and answers questions with the top matching chunks."""

    def __init__(self, embedder: Embedder | None = None):
        if not INDEX_PATH.exists() or not CHUNKS_PATH.exists():
            raise FileNotFoundError(
                "No index found. Run `python fetch_docs.py` then `python build_index.py` first."
            )
        self.index = faiss.read_index(str(INDEX_PATH))
        self.chunks = json.loads(CHUNKS_PATH.read_text(encoding="utf-8"))
        self.embedder = embedder or Embedder()
        self.bm25 = BM25([tokenize(f'{c["title"]} {c["text"]}') for c in self.chunks])

    def search(self, question: str, k: int = 3) -> list[dict]:
        """Hybrid search: fuse the FAISS (semantic) and BM25 (keyword) rankings with RRF."""
        query = self.embedder.embed([question])
        dense_scores, dense_ids = self.index.search(query, min(CANDIDATES, self.index.ntotal))
        cosine = {int(i): float(s) for s, i in zip(dense_scores[0], dense_ids[0]) if i != -1}

        keyword_scores = self.bm25.scores(tokenize(question))
        keyword_ids = [int(i) for i in np.argsort(-keyword_scores)[:CANDIDATES] if keyword_scores[i] > 0]

        fused: dict[int, float] = {}
        for weight, ranking in ((DENSE_WEIGHT, list(cosine)), (KEYWORD_WEIGHT, keyword_ids)):
            for rank, i in enumerate(ranking):
                fused[i] = fused.get(i, 0.0) + weight / (RRF_K + rank + 1)

        best = sorted(fused, key=fused.get, reverse=True)[:k]
        return [
            {
                **self.chunks[i],
                "score": round(fused[i], 4),
                "cosine": round(float(self.index.reconstruct(i) @ query[0]), 4),
                "bm25": round(float(keyword_scores[i]), 2),
            }
            for i in best
        ]

    def answer(self, question: str, k: int = 3) -> dict:
        """Extractive answer: the best chunk, plus all top-k chunks as sources."""
        results = self.search(question, k)
        if not results or all(r["bm25"] == 0 for r in results):
            return {"answer": "I couldn't find anything relevant in the docs.", "sources": []}
        return {"answer": results[0]["text"], "sources": results}
