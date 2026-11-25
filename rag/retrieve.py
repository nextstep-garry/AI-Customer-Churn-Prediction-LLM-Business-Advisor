"""
retrieve.py -- QUERY side of RAG.

Phase 6: question -> query vector, with a safety check that the question is
         embedded by the SAME model as the documents.
Phase 7: similarity search in ChromaDB  ->  retrieve(question, top_k)
"""
import os

import config  # noqa: F401  (loads .env)
from embedder import EMBEDDING_MODEL_NAME, embed_query
from vector_store import VectorStoreError, get_collection

# How many chunks to fetch (override with:  set TOP_K=3  in the terminal).
TOP_K = int(os.getenv("TOP_K", "5"))


def check_same_model(collection, model_name: str = EMBEDDING_MODEL_NAME):
    """
    Compare the model that built the database with the model we are about to
    use for the question. If they differ, the search results would be garbage,
    so we stop with a clear message instead.
    """
    sample = collection.get(limit=1, include=["metadatas"])
    if not sample["ids"]:
        raise VectorStoreError("The database is empty. Run: python ingest.py")

    stored_model = sample["metadatas"][0].get("embedding_model")
    if stored_model is None:
        raise VectorStoreError(
            "The database does not record its embedding model. "
            "Run: python ingest.py  (it will update the records)."
        )
    if stored_model != model_name:
        raise VectorStoreError(
            f"Embedding model mismatch: database built with '{stored_model}', "
            f"but the question would use '{model_name}'. "
            f"Use the same model, or re-run ingest.py."
        )


def prepare_query(question: str, collection=None):
    """Question text -> query vector (same model as the stored chunks)."""
    collection = collection or get_collection()
    check_same_model(collection)
    return embed_query(question)      # adds the bge prefix, normalises the vector


# ======================= PHASE 7: SIMILARITY SEARCH =========================

def retrieve(question: str, top_k: int = TOP_K, where: dict = None):
    """
    Find the top_k chunks most similar to the question.

    Returns a list (best first) of dicts:
        {"text", "source", "section", "chunk_uid", "distance", "similarity", "metadata"}
    Returns an empty list if nothing was found.

    where: optional metadata filter, for example {"document_type": "model_evaluation"}
    """
    if top_k < 1:
        raise ValueError("top_k must be at least 1")

    collection = get_collection()
    query_vector = prepare_query(question, collection)          # Phase 6

    try:
        result = collection.query(
            query_embeddings=[query_vector.tolist()],
            n_results=min(top_k, collection.count()),           # never ask for more than exist
            where=where,
            include=["documents", "metadatas", "distances"],
        )
    except Exception as err:
        raise VectorStoreError(f"ChromaDB search failed: {err}") from err

    # Chroma returns one list per query; we sent one query, so take [0].
    texts = result["documents"][0]
    metadatas = result["metadatas"][0]
    distances = result["distances"][0]

    chunks = []
    for text, meta, dist in zip(texts, metadatas, distances):
        chunks.append({
            "text": text,
            "source": meta["source"],
            "section": meta["section"],
            "chunk_uid": meta["chunk_uid"],
            "distance": float(dist),
            "similarity": 1.0 - float(dist),     # cosine: similarity = 1 - distance
            "metadata": meta,
        })
    return chunks
