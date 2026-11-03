"""Chunk the downloaded docs, embed every chunk and save a FAISS index."""

import json

import faiss

from rag import CHUNKS_PATH, DATA_DIR, INDEX_PATH, Embedder, chunk_text

DOCS_DIR = DATA_DIR / "docs"


def load_chunks() -> list[dict]:
    chunks = []
    for path in sorted(DOCS_DIR.glob("*.txt")):
        url, title, body = path.read_text(encoding="utf-8").split("\n", 2)
        title = title or path.stem.replace("_", "/")
        for text in chunk_text(body):
            chunks.append({"text": text, "source": url, "title": title})
    return chunks


def main() -> None:
    if not DOCS_DIR.exists() or not any(DOCS_DIR.glob("*.txt")):
        raise SystemExit("No docs found. Run `python fetch_docs.py` first.")

    chunks = load_chunks()
    print(f"Embedding {len(chunks)} chunks...")
    # Prefix the page title so each chunk carries its topic into the embedding.
    vectors = Embedder().embed([f'{c["title"]}: {c["text"]}' for c in chunks])

    # Vectors are L2-normalised, so inner product == cosine similarity.
    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)

    faiss.write_index(index, str(INDEX_PATH))
    CHUNKS_PATH.write_text(json.dumps(chunks, ensure_ascii=False), encoding="utf-8")
    print(f"Saved {index.ntotal} vectors to {INDEX_PATH}")


if __name__ == "__main__":
    main()
