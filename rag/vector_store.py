"""
vector_store.py -- Phase 5: connect to the persistent ChromaDB database.

Shared by ingestion (writes) and retrieval (reads), so both always talk to the
same folder and the same collection.

Install:  pip install chromadb
"""
import os
from pathlib import Path

import config  # noqa: F401  (loads .env)

# ---- Settings (override with environment variables if needed) --------------
CHROMA_DIR = os.getenv("CHROMA_DIR", str(Path(__file__).parent / "chroma_db"))
COLLECTION_NAME = os.getenv("CHROMA_COLLECTION", "churn_project_knowledge")


class VectorStoreError(Exception):
    """Raised when ChromaDB cannot be opened or written to."""


def get_collection():
    """Open (or create) the persistent collection and return it."""
    try:
        import chromadb
        client = chromadb.PersistentClient(path=CHROMA_DIR)   # saved on disk
        return client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},   # compare vectors by cosine similarity
            embedding_function=None,             # WE supply vectors, Chroma must not embed
        )
    except Exception as err:
        raise VectorStoreError(f"Could not open ChromaDB at '{CHROMA_DIR}': {err}") from err
