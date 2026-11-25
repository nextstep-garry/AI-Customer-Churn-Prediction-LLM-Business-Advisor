"""
embedder.py  --  Phase 4: turn text into vectors (embeddings).

This file is shared by BOTH pipelines:
    * ingestion  : chunks  -> vectors  -> ChromaDB
    * querying   : question -> vector  -> search ChromaDB
Keeping it in one place guarantees that documents and questions are embedded
by the SAME model (a hard rule of RAG).

Install:  pip install sentence-transformers numpy
"""
import os
from functools import lru_cache

import numpy as np

import config  # noqa: F401  (loads .env before the settings below are read)

# ---- Settings (change with environment variables, no code edits needed) ----
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")

# bge models want a short instruction in front of the QUESTION (not the documents).
# Other models (for example all-MiniLM-L6-v2) need no prefix.
_BGE_PREFIX = "Represent this sentence for searching relevant passages: "
QUERY_PREFIX = os.getenv(
    "EMBEDDING_QUERY_PREFIX",
    _BGE_PREFIX if "bge" in EMBEDDING_MODEL_NAME.lower() else "",
)


class EmbeddingError(Exception):
    """Raised when the model cannot load or cannot embed the text."""


@lru_cache(maxsize=1)
def get_model():
    """Load the model once; later calls reuse it (loading is slow)."""
    try:
        from sentence_transformers import SentenceTransformer
        return SentenceTransformer(EMBEDDING_MODEL_NAME)
    except Exception as err:
        raise EmbeddingError(
            f"Could not load embedding model '{EMBEDDING_MODEL_NAME}'. "
            f"Is sentence-transformers installed and the internet available "
            f"for the first download? Details: {err}"
        ) from err


def embed_texts(texts, batch_size=32):
    """
    list of strings -> numpy array of shape (len(texts), dimension).
    Vectors are normalised to length 1, so cosine similarity = dot product.
    """
    if not texts:
        raise EmbeddingError("embed_texts() received an empty list.")
    model = get_model()
    try:
        vectors = model.encode(
            texts,
            batch_size=batch_size,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
    except Exception as err:
        raise EmbeddingError(f"Embedding failed: {err}") from err
    return np.asarray(vectors, dtype="float32")


def embed_query(question: str):
    """One user question -> one vector (adds the model's query prefix)."""
    if not question or not question.strip():
        raise EmbeddingError("The question is empty.")
    return embed_texts([QUERY_PREFIX + question.strip()])[0]
