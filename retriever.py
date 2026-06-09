import chromadb
from chromadb.utils import embedding_functions

from config import CHROMA_COLLECTION, CHROMA_PATH, EMBEDDING_MODEL, N_RESULTS

# Embedding function + ChromaDB client are initialized once at module load.
# sentence-transformers downloads all-MiniLM-L6-v2 on first use (30-60s the
# very first time); subsequent runs read it from the local cache.
_ef = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name=EMBEDDING_MODEL
)
_client = chromadb.PersistentClient(path=CHROMA_PATH)
_collection = _client.get_or_create_collection(
    name=CHROMA_COLLECTION,
    embedding_function=_ef,
    metadata={"hnsw:space": "cosine"},   # cosine distance: lower = more similar
)


def get_collection():
    """Return the ChromaDB collection (used by app.py to check/run ingestion)."""
    return _collection


def embed_and_store(chunks):
    """
    Embed a list of chunks (from chunk_document) and store them in ChromaDB.

    ChromaDB's embedding function turns each `documents` string into a vector
    automatically — we hand over text + metadata + ids and it does the math.
    Metadata travels with the vector so retrieve() can attribute each result
    back to its source document and URL.
    """
    _collection.add(
        documents=[c["text"] for c in chunks],
        metadatas=[c["metadata"] for c in chunks],
        ids=[c["chunk_id"] for c in chunks],
    )
    print(f"Stored {_collection.count()} total chunks in the vector database.")


def retrieve(query, n_results=N_RESULTS):
    """
    Semantic search: return the top-n most relevant chunks for a query.

    Returns a list of dicts, each with:
      - "text"     : the chunk text (context line + review/description)
      - "source"   : source filename (attribution key)
      - "title"    : human-readable source title
      - "url"      : source URL
      - "distance" : cosine distance (lower = more similar)
    """
    if _collection.count() == 0:
        return []

    results = _collection.query(
        query_texts=[query],
        n_results=n_results,
        include=["documents", "metadatas", "distances"],
    )

    # query() is batch-capable, so each field is a list-of-lists with one inner
    # list per query. We passed a single query, so [0] unwraps to the results.
    documents = results["documents"][0]
    metadatas = results["metadatas"][0]
    distances = results["distances"][0]

    return [
        {
            "text": text,
            "source": meta["source"],
            "title": meta["title"],
            "url": meta.get("url", ""),
            "distance": distance,
        }
        for text, meta, distance in zip(documents, metadatas, distances)
    ]


if __name__ == "__main__":
    # Manual retrieval check — run after ingesting via app.py.
    for q in [
        "Is Naomi Zweben a good professor for Software 1?",
        "Which professor gives the most useful feedback?",
        "What do students say about KT Vandergriff's exams?",
    ]:
        print(f"\n=== Query: {q} ===")
        for r in retrieve(q):
            print(f"[{r['distance']:.3f}] {r['source']}: {r['text'][:120]}...")
