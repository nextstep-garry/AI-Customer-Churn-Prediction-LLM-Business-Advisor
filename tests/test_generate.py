"""rag/generate.py: generate_text() sends any prompt through the chosen provider; generate_answer() still works."""
import sys
from pathlib import Path

import pytest

RAG = str(Path(__file__).resolve().parents[1] / "rag")


@pytest.fixture
def gen(monkeypatch):
    pytest.importorskip("requests")
    pytest.importorskip("dotenv")
    monkeypatch.syspath_prepend(RAG)
    import generate
    seen = []
    monkeypatch.setitem(generate.PROVIDERS, "fake", (lambda s, u, m, t: seen.append((s, u)) or "<think>x</think> Hello ", "m"))
    monkeypatch.setenv("LLM_PROVIDER", "fake")
    return generate, seen


def test_generate_text_passes_prompts_through_and_cleans_the_reply(gen):
    generate, seen = gen
    assert generate.generate_text("SYS", "USER") == "Hello"
    assert seen == [("SYS", "USER")]


def test_generate_answer_still_builds_the_rag_prompt(gen):
    generate, seen = gen
    assert generate.generate_answer("some context", "some question") == "Hello"
    system, user = seen[0]
    assert "some context" in user and "some question" in user


def test_unknown_provider_is_a_clear_error(gen, monkeypatch):
    generate, _ = gen
    monkeypatch.setenv("LLM_PROVIDER", "nope")
    with pytest.raises(generate.LLMError):
        generate.generate_text("a", "b")