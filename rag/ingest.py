"""
ingest.py  --  Phases 2-5: the whole INGESTION pipeline.

    .md / .json files -> load_documents() -> chunk_documents() -> embed_texts()
                      -> ChromaDB (persistent)

Run from the folder that contains ingest.py:   python ingest.py
(Safe to run many times: it never creates duplicates.)
"""
import json
import os
from pathlib import Path

import config  # noqa: F401  (loads .env)
from embedder import EMBEDDING_MODEL_NAME, embed_texts
from vector_store import VectorStoreError, get_collection

BASE_DIR = Path(__file__).parent

# Three folders, three kinds of knowledge. The group name is stored in every
# chunk's metadata (kb_group), so an answer can say which kind of source it used.
DOC_FOLDERS = [
    ("notebook_docs", BASE_DIR / "documents"),                       # 12 documents written from the churn notebook
    ("knowledge_base", BASE_DIR / "knowledge_base"),                 # KB-00 .. KB-04 (sample documents + data findings)
    ("app_docs", BASE_DIR / "knowledge_base" / "project_docs"),      # the churn APP: README, CHANGES, model card ...
]
FILE_PATTERNS = ("*.md", "*.json")

# Files WITHOUT a front-matter block get these defaults (the app docs have none).
GROUP_DEFAULTS = {
    "app_docs": {"status": "FACTUAL - REAL PROJECT DOCUMENTATION (churn app)", "synthetic": "false", "doc_prefix": "PD"},
}
TITLE_OVERRIDES = {
    "README.md": "Churn App README",
    "models_README.md": "Models Folder README",
    "CHANGES.md": "Churn App Changes and Status",
    "model_card.json": "Model Card",
    "model_manifest.json": "Model Manifest",
    "calibration_report.json": "Calibration Report",
}

# ---- Chunking settings (all tunable) ---------------------------------------
CHUNK_SIZE_TOKENS = int(os.getenv("CHUNK_SIZE_TOKENS", "300"))      # a chunk may not be bigger than this
CHUNK_OVERLAP_TOKENS = int(os.getenv("CHUNK_OVERLAP_TOKENS", "40"))  # used ONLY when one section must be cut


def parse_front_matter(raw_text: str):
    """
    Split a file into (front_matter_dict, body_text).

    A file looks like:
        ---
        doc_id: PK-10
        title: Model Comparison ...
        ---
        # Heading ... body ...
    """
    if not raw_text.startswith("---"):
        return {}, raw_text.strip()          # no header block in this file

    end = raw_text.find("\n---", 3)          # find the closing '---'
    if end == -1:
        return {}, raw_text.strip()

    header_block = raw_text[3:end]
    body = raw_text[end + 4:].strip()

    front_matter = {}
    for line in header_block.strip().splitlines():
        if ":" in line:
            key, value = line.split(":", 1)  # split only on the FIRST colon
            front_matter[key.strip()] = value.strip()
    return front_matter, body


# ---------- JSON files are turned into readable Markdown, then chunked like any document ----------

# model_manifest.json repeats these sections from model_card.json; skip the copies so
# the same facts do not appear twice in the database.
JSON_DROP_SECTIONS = {"model_manifest.json": {"metrics", "training_ranges", "risk_tiers", "known_limitations"}}
JSON_HIDE_KEYS = {"sha256"}            # long hash strings are noise for search


def _render(value, path=""):
    """Yield (label, text) pairs for any JSON value. Floats are rounded to 4 decimals."""
    if isinstance(value, dict):
        for key, inner in value.items():
            if key not in JSON_HIDE_KEYS:
                yield from _render(inner, f"{path}.{key}" if path else key)
    elif isinstance(value, list):
        if value and all(isinstance(x, dict) for x in value):
            for item in value:                      # e.g. the three test tiers: one line per item
                yield path, "; ".join(f"{label}: {text}" for label, text in _render(item))
        elif all(not isinstance(x, (dict, list)) for x in value):
            items = [str(round(x, 4) if isinstance(x, float) else x) for x in value]
            if max((len(x) for x in items), default=0) <= 40:       # short items: one line
                if "range" in path.lower() and len(items) == 2:
                    yield path, f"min {items[0]}, max {items[1]}"
                else:
                    yield path, ", ".join(items)
            else:                                                    # long sentences: one line each
                for item in items:
                    yield path, item
    elif value is not None:
        yield path, round(value, 4) if isinstance(value, float) else value


def json_to_markdown(data: dict, title: str, drop=()):
    """
    One '## section' per top-level key that holds a dict or list; all simple top-level
    values (version, description, threshold ...) are collected in '## Overview'.
    Every bullet repeats its own label, so each line makes sense on its own.
    """
    overview, sections = [], []
    for key, value in data.items():
        if key in drop or key in JSON_HIDE_KEYS:
            continue
        if isinstance(value, (dict, list)):
            bullets = []
            for label, text in _render(value, key):
                relative = label[len(key) + 1:] if label.startswith(key + ".") else ""
                if relative:
                    bullets.append(f"- {relative}: {text}")
                elif len(str(text)) > 40:                # a sentence: no label needed
                    bullets.append(f"- {text}")
                else:                                    # short list or list-of-dicts line
                    bullets.append(f"- {key}: {text}")
            sections.append(f"## {key}\n" + "\n".join(bullets))
        elif value is not None:
            overview.append(f"- {key}: {round(value, 4) if isinstance(value, float) else value}")
    parts = [f"# {title}"]
    if overview:
        parts.append("## Overview\n" + "\n".join(overview))
    parts.extend(sections)
    return "\n\n".join(parts)


def load_documents(docs_dir: Path = None):
    """
    Load every .md and .json file and return a list of documents.
      docs_dir=None  -> read ALL folders in DOC_FOLDERS (a missing folder is skipped)
      docs_dir=path  -> read only that folder
    A file name that appears in two folders is loaded once (the first folder wins),
    because the file name is part of the chunk ID.
    """
    if docs_dir is not None:
        folders = [(Path(docs_dir).name, Path(docs_dir))]
    else:
        folders = [(group, folder) for group, folder in DOC_FOLDERS if folder.is_dir()]
    if not folders:
        raise FileNotFoundError(f"No documents folder found. Expected: {DOC_FOLDERS[0][1]}")

    documents, seen = [], set()
    for group, folder in folders:
        if not folder.is_dir():
            raise FileNotFoundError(f"Documents folder not found: {folder}")
        paths = sorted(p for pattern in FILE_PATTERNS for p in folder.glob(pattern))
        for path in paths:
            if path.name in seen:
                print(f"WARNING: {path.name} exists in more than one folder; "
                      f"using the copy from the first folder only.")
                continue
            try:
                raw_text = path.read_text(encoding="utf-8-sig")
                if path.suffix == ".json":
                    drop = JSON_DROP_SECTIONS.get(path.name, ())
                    raw_text = json_to_markdown(json.loads(raw_text),
                                                TITLE_OVERRIDES.get(path.name, path.stem), drop)
            except (OSError, UnicodeDecodeError, json.JSONDecodeError) as err:
                print(f"WARNING: could not read {path.name}: {err}")     # skip bad file
                continue
            seen.add(path.name)

            front_matter, body = parse_front_matter(raw_text)
            defaults = GROUP_DEFAULTS.get(group, {})
            first_heading = next((l[2:].strip() for l in body.splitlines() if l.startswith("# ")), path.stem)

            # Metadata must stay FLAT (str / int / bool only): ChromaDB accepts nothing else.
            metadata = {
                "source": path.name,                     # model_evaluation.md
                "document_type": path.stem,              # model_evaluation
                "kb_group": group,                       # notebook_docs | knowledge_base | app_docs
                "doc_id": front_matter.get("doc_id")
                          or (f"{defaults['doc_prefix']}-{path.stem}" if "doc_prefix" in defaults else "unknown"),
                "title": front_matter.get("title") or TITLE_OVERRIDES.get(path.name, first_heading),
                "status": front_matter.get("status") or defaults.get("status", "unknown"),
                "synthetic": front_matter.get("synthetic", defaults.get("synthetic", "false")).lower() == "true",
                "char_count": len(body),
            }
            documents.append({"text": body, "metadata": metadata})

    if not documents:
        raise FileNotFoundError("No .md or .json files found in: " + ", ".join(str(f) for _, f in folders))
    return documents


# ============================ PHASE 3: CHUNKING =============================

def estimate_tokens(text: str) -> int:
    """Rough token count: about 4 characters per token for English text.
    (Phase 4 will use the real tokenizer of the embedding model.)"""
    return max(1, len(text) // 4)


def split_into_sections(body: str):
    """
    Cut a document into sections. Returns a list of (heading, section_text).
      '## Heading'      starts a section
      '### Subheading'  starts a sub-section, named "Heading > Subheading"
      text before the first '##' (for example a SAMPLE disclaimer) becomes a section
      called "Overview". The '# Title' line itself is skipped: the title is in metadata.
    """
    sections, h2, h3, lines = [], "Overview", None, []

    def flush():
        text = "\n".join(lines).strip()
        if text:
            sections.append((h2 if h3 is None else f"{h2} > {h3}", text))

    for line in body.splitlines():
        if line.startswith("## "):
            flush()
            h2, h3, lines = line[3:].strip(), None, []
        elif line.startswith("### "):
            flush()
            h3, lines = line[4:].strip(), []
        elif line.startswith("# "):
            continue                                  # the document title
        else:
            lines.append(line)
    flush()
    return sections


def split_long_text(text: str, max_tokens: int, overlap_tokens: int):
    """
    Safety net: if ONE section is bigger than max_tokens, cut it line by line
    (so table rows are never cut in half). Each new piece starts with the last
    few lines of the previous piece (the overlap).
    """
    if estimate_tokens(text) <= max_tokens:
        return [text]

    pieces, current = [], []
    for line in text.splitlines():
        current.append(line)
        if estimate_tokens("\n".join(current)) > max_tokens and len(current) > 1:
            current.pop()                          # this line goes to the next piece
            pieces.append("\n".join(current))
            # build the overlap from the tail of the finished piece
            tail, size = [], 0
            for old in reversed(current):
                size += estimate_tokens(old)
                if size > overlap_tokens:
                    break
                tail.insert(0, old)
            current = tail + [line]
    if current:
        pieces.append("\n".join(current))
    return pieces


def chunk_document(doc, chunk_size=CHUNK_SIZE_TOKENS, overlap=CHUNK_OVERLAP_TOKENS):
    """Turn ONE document into chunks. Every chunk keeps the document metadata."""
    meta = doc["metadata"]
    chunks = []
    for heading, section_text in split_into_sections(doc["text"]):
        if not section_text:
            continue                                  # empty section, skip
        for piece in split_long_text(section_text, chunk_size, overlap):
            n = len(chunks)                           # 0, 1, 2 ... inside this document
            chunks.append({
                # Header line gives the chunk its own context when read alone.
                "text": f"{meta['title']} | {heading}\n\n{piece}",
                "metadata": {
                    **meta,                           # source, document_type, doc_id, status ...
                    "chunk_id": n,
                    "section": heading,
                    # deterministic id, used later as the ChromaDB id
                    "chunk_uid": f"{meta['document_type']}_{n:03d}",
                },
            })
    return chunks


def chunk_documents(documents, chunk_size=CHUNK_SIZE_TOKENS, overlap=CHUNK_OVERLAP_TOKENS):
    """Chunk every document and return one flat list of chunks."""
    all_chunks = []
    for doc in documents:
        all_chunks.extend(chunk_document(doc, chunk_size, overlap))
    return all_chunks


# ======================= PHASE 5: STORE IN CHROMADB =========================

def build_vector_store(chunk_size=CHUNK_SIZE_TOKENS, overlap=CHUNK_OVERLAP_TOKENS):
    """Load -> chunk -> embed -> save in ChromaDB. Returns a small summary dict."""
    documents = load_documents()
    chunks = chunk_documents(documents, chunk_size, overlap)

    texts = [c["text"] for c in chunks]
    ids = [c["metadata"]["chunk_uid"] for c in chunks]      # e.g. model_evaluation_006
    # Remember WHICH model made these vectors; retrieve.py checks it later.
    metadatas = [{**c["metadata"], "embedding_model": EMBEDDING_MODEL_NAME} for c in chunks]
    vectors = embed_texts(texts)                            # shape (n_chunks, 384)

    collection = get_collection()
    try:
        # upsert = insert new ids, overwrite existing ids  ->  no duplicates
        collection.upsert(
            ids=ids,
            embeddings=vectors.tolist(),
            documents=texts,
            metadatas=metadatas,
        )
        # Remove chunks that no longer exist (for example you edited a document
        # and it now has fewer chunks), so the database always mirrors the files.
        stale = list(set(collection.get(include=[])["ids"]) - set(ids))
        if stale:
            collection.delete(ids=stale)
    except Exception as err:
        raise VectorStoreError(f"Writing to ChromaDB failed: {err}") from err

    return {
        "documents": len(documents),
        "chunks_written": len(ids),
        "removed_stale": len(stale),
        "collection_count": collection.count(),
    }


if __name__ == "__main__":
    summary = build_vector_store()
    print("Ingestion finished:")
    for key, value in summary.items():
        print(f"  {key:17}: {value}")
