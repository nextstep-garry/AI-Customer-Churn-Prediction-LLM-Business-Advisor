"""
rag_pipeline.py -- Phase 11: the complete RAG pipeline in ONE function.

    ask("Why was XGBoost selected?")
        -> {"answer": "...", "sources": ["model_evaluation.md", ...]}

Steps inside ask():
    question -> query embedding -> ChromaDB search -> top-k chunks
             -> context -> prompt -> LLM -> answer + sources

This file never touches the churn ML model. RAG answers questions ABOUT the
project; the ML model (POST /predict) predicts churn. They stay separate.
"""
import os
import re
import time

import config  # noqa: F401  (loads .env)
from embedder import EmbeddingError, embed_texts
from generate import LLMError, generate_answer
from prompt import NOT_AVAILABLE_TEXT, build_context
from retrieve import TOP_K, retrieve
from vector_store import VectorStoreError

# Safety gate: if even the BEST retrieved chunk is less similar than this, the
# question is clearly off-topic, so we refuse immediately and never call the LLM.
# Measured on the Phase 12 test set: the LOWEST answerable question scored 0.553,
# so 0.50 is safe. A higher value (0.68 was tried) wrongly refuses real questions,
# because answerable and unanswerable scores overlap (0.553 .. 0.690).
# Set MIN_TOP_SIMILARITY=0 to turn the gate off.
MIN_TOP_SIMILARITY = float(os.getenv("MIN_TOP_SIMILARITY", "0.50"))

NOT_AVAILABLE = NOT_AVAILABLE_TEXT

# Words a small LLM may use when it refuses in its own words.
REFUSAL_HINTS = ("not available", "does not contain", "do not contain", "no information",
                 "cannot be determined", "not mentioned", "not specified")


def is_refusal(answer: str) -> bool:
    """True if the answer says "this is not in the knowledge base" (and gives no real content)."""
    text = answer.strip().lower()
    if text.startswith(NOT_AVAILABLE_TEXT.lower()):
        return True
    if text.startswith("not available in the current project artifacts"):
        return True
    # Paraphrased refusal: short, contains a refusal hint, and has NO digits.
    # (A real answer from this project nearly always contains a number.)
    return (len(text) < 450
            and any(hint in text for hint in REFUSAL_HINTS)
            and not any(ch.isdigit() for ch in text))


def _pick_sources(question: str, answer: str, chunks: list) -> list:
    """
    Decide which source files to show, from EVIDENCE, not from the LLM's own citations
    (small models sometimes cite the wrong file).
      1. A refusal has no sources.
      2. Otherwise embed question + answer and compare them with each retrieved chunk.
         (Very short answers such as "13" have weak vectors; the question gives context.) The chunks
         the answer is closest to are the ones it was written from. Show their files
         (the best one, plus others within 0.05 of it, at most 3).
    """
    if is_refusal(answer):
        return []
    try:
        vectors = embed_texts([f"{question}\n{answer}"] + [c["text"] for c in chunks])
    except EmbeddingError:
        return [chunks[0]["source"]]                    # fall back to the best-retrieved file
    scores = vectors[1:] @ vectors[0]                   # cosine (vectors are normalised)
    best = float(scores.max())
    ranked = sorted(zip(scores, chunks), key=lambda pair: -pair[0])
    sources = []
    for score, chunk in ranked:
        if score >= best - 0.05 and chunk["source"] not in sources:
            sources.append(chunk["source"])
        if len(sources) == 3:
            break
    return sources


def ask(question: str, top_k: int = TOP_K, debug: bool = False) -> dict:
    """
    Answer a question about the churn project.
    Returns {"answer": str, "sources": [str, ...]}.
    On a failure it returns the same keys plus "error" (it never crashes the caller).
    debug=True also returns what was retrieved (used for retrieval testing).
    """
    started = time.time()
    result = {"answer": "", "sources": []}

    if not question or not question.strip():
        result.update(answer="Please type a question.", error="empty_question")
        return result

    # ---- 1-3. embed the question and search ChromaDB -----------------------
    try:
        chunks = retrieve(question, top_k=top_k)
    except (EmbeddingError, VectorStoreError) as err:
        result.update(answer="The knowledge base could not be searched.", error=str(err))
        return result

    # ---- 4. gate + context --------------------------------------------------
    if chunks and chunks[0]["similarity"] < MIN_TOP_SIMILARITY:
        gated = chunks                       # remember them for debug output
        chunks = []
    else:
        gated = chunks
    context, _ = build_context(chunks)
    if debug:
        result["retrieved"] = [
            {"source": c["source"], "section": c["section"], "similarity": round(c["similarity"], 3)}
            for c in gated
        ]

    if not context:                          # nothing relevant: do not call the LLM
        result["answer"] = NOT_AVAILABLE
        if debug:
            result["seconds"] = round(time.time() - started, 1)
        return result

    # ---- 5-6. prompt + LLM --------------------------------------------------
    try:
        answer = generate_answer(context, question)
    except LLMError as err:
        result.update(answer="The language model could not answer right now.", error=str(err))
        return result

    result["answer"] = answer
    result["sources"] = _pick_sources(question, answer, chunks)
    if debug:
        result["seconds"] = round(time.time() - started, 1)
    return result


if __name__ == "__main__":
    import sys
    q = " ".join(sys.argv[1:]) or "Why was XGBoost selected?"
    out = ask(q, debug=True)
    print(f"Question : {q}\n")
    print("Answer   :\n" + out["answer"])
    print("\nSources  :", out["sources"])
    print("Retrieved:", [f"{r['source']} ({r['similarity']})" for r in out.get("retrieved", [])])
    print(f"Time     : {out.get('seconds', '?')} s")
    if "error" in out:
        print("ERROR    :", out["error"])
