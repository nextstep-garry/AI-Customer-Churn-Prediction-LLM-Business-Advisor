"""
generate.py -- Phase 10: send the prompt to an LLM and return the answer.

The rest of the RAG code only calls   generate_answer(context, question).
Which LLM runs behind it is chosen in the .env file (LLM_PROVIDER), so you can
switch between Hugging Face (local), Ollama (local), OpenAI and Gemini without
touching retrieval code.

Install:  pip install requests python-dotenv transformers torch
          (transformers + torch are only needed for LLM_PROVIDER=huggingface)
"""
import os
import re
from functools import lru_cache

import requests

import config  # noqa: F401  (loads the .env file: API keys and settings)
from prompt import build_prompt

TIMEOUT_SECONDS = 60


class LLMError(Exception):
    """Raised for any problem talking to the LLM (missing key, network, bad reply)."""


# ----------------------------- one function per provider --------------------
# Every provider function has the SAME signature and returns plain text.

def _call_openai(system_prompt, user_prompt, model, temperature):
    key = os.getenv("OPENAI_API_KEY")
    if not key:
        raise LLMError("OPENAI_API_KEY is missing. Add it to your .env file.")
    response = requests.post(
        "https://api.openai.com/v1/chat/completions",
        headers={"Authorization": f"Bearer {key}"},
        json={
            "model": model,
            "temperature": temperature,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        },
        timeout=TIMEOUT_SECONDS,
    )
    _raise_for_status(response, "OpenAI")
    return response.json()["choices"][0]["message"]["content"]


def _call_gemini(system_prompt, user_prompt, model, temperature):
    key = os.getenv("GEMINI_API_KEY")
    if not key:
        raise LLMError("GEMINI_API_KEY is missing. Add it to your .env file.")
    response = requests.post(
        f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent",
        headers={"x-goog-api-key": key},
        json={
            "systemInstruction": {"parts": [{"text": system_prompt}]},
            "contents": [{"role": "user", "parts": [{"text": user_prompt}]}],
            "generationConfig": {"temperature": temperature},
        },
        timeout=TIMEOUT_SECONDS,
    )
    _raise_for_status(response, "Gemini")
    return response.json()["candidates"][0]["content"]["parts"][0]["text"]


def _call_ollama(system_prompt, user_prompt, model, temperature):
    base_url = os.getenv("OLLAMA_URL", "http://localhost:11434")
    payload = {
            "model": model,
            "stream": False,
            # num_ctx = how many tokens the model may read. Our prompt is ~900 tokens,
            # so 4096 is safe. A too-small window would silently cut the context.
            "options": {
                "temperature": temperature,
                "num_ctx": int(os.getenv("OLLAMA_NUM_CTX", "4096")),
            },
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
    }
    # Qwen3 is a "thinking" model: it writes its reasoning before the answer.
    # We want only the answer (faster, and no reasoning text in the output).
    if "qwen3" in model.lower():
        payload["think"] = False
    response = requests.post(
        f"{base_url}/api/chat",
        json=payload,
        timeout=int(os.getenv("OLLAMA_TIMEOUT", "300")),   # local models can be slow (seconds)
    )
    _raise_for_status(response, "Ollama")
    return response.json()["message"]["content"]


@lru_cache(maxsize=1)
def _load_hf_model(model_name):
    """
    Download (first time only) and load a Hugging Face chat model.
    lru_cache keeps it in memory, so loading happens ONCE per Python process.
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = os.getenv("HF_DEVICE", "auto").lower()          # auto | cpu | cuda
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    # bfloat16 = half the memory of float32 (a 7B model: ~15 GB instead of ~30 GB)
    model = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch.bfloat16)
    model.to(device)
    model.eval()
    return tokenizer, model, device


def _call_huggingface(system_prompt, user_prompt, model, temperature):
    import torch

    tokenizer, hf_model, device = _load_hf_model(model)

    # The tokenizer knows this model's chat format (Qwen's special tokens etc.).
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    prompt_text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(prompt_text, return_tensors="pt").to(device)

    gen_kwargs = {"max_new_tokens": int(os.getenv("HF_MAX_NEW_TOKENS", "400"))}
    if temperature > 0:
        gen_kwargs.update(do_sample=True, temperature=temperature)
    else:                                   # temperature 0 = greedy = most factual
        gen_kwargs.update(do_sample=False, temperature=None, top_p=None, top_k=None)

    with torch.no_grad():
        output_ids = hf_model.generate(**inputs, **gen_kwargs)

    # generate() returns prompt + answer; keep only the NEW tokens (the answer).
    new_tokens = output_ids[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True)


def _raise_for_status(response, name):
    """Turn an HTTP error into a short, readable LLMError."""
    if response.status_code != 200:
        raise LLMError(f"{name} returned HTTP {response.status_code}: {response.text[:300]}")


# The "menu" of providers: name -> (function, default model).
# To add another LLM, write one more _call_xxx function and add a line here.
PROVIDERS = {
    "huggingface": (_call_huggingface, "Qwen/Qwen2.5-7B-Instruct"),
    "openai": (_call_openai, "gpt-4o-mini"),
    "gemini": (_call_gemini, "gemini-2.5-flash"),
    "ollama": (_call_ollama, "llama3.2"),
}


def _strip_thinking(text: str) -> str:
    """
    Remove the model's reasoning, keep only the final answer. Handles:
      1. reasoning with ONLY a closing tag  ("...reasoning...</think> answer")
         (some Qwen3 versions start thinking silently; we keep what follows the LAST </think>)
      2. complete <think>...</think> blocks
      3. an unclosed <think> block (cut to the end)
    """
    if "</think>" in text:
        text = text.rsplit("</think>", 1)[1]
    return re.sub(r"<think>.*?(</think>|$)", "", text, flags=re.DOTALL)

'''
def generate_answer(context: str, question: str) -> str:
    """Build the prompt from (context, question), call the chosen LLM, return the answer text."""
    provider = os.getenv("LLM_PROVIDER", "").strip().lower()

    if provider not in PROVIDERS:
        raise LLMError(
            f"LLM_PROVIDER is '{provider}'. Set it in .env to one of: {', '.join(PROVIDERS)}"
        )
    call_llm, default_model = PROVIDERS[provider]
    model = os.getenv("LLM_MODEL") or default_model
    temperature = float(os.getenv("LLM_TEMPERATURE", "0"))   # 0 = most factual, least random

    system_prompt, user_prompt = build_prompt(context, question)
    try:
        answer = call_llm(system_prompt, user_prompt, model, temperature)
    except LLMError:
        raise
    except requests.exceptions.RequestException as err:       # no internet, timeout, Ollama not running
        raise LLMError(f"Could not reach the {provider} LLM: {err}") from err
    except (KeyError, IndexError, ValueError) as err:         # reply had an unexpected shape
        raise LLMError(f"Unexpected reply format from {provider}: {err}") from err
    except ImportError as err:                                # transformers / torch not installed
        raise LLMError(f"A required library is missing: {err}. Run: pip install transformers torch") from err
    except (OSError, RuntimeError, MemoryError) as err:       # model not found, download failed, out of memory
        raise LLMError(
            f"The {provider} model '{model}' failed: {err}. "
            f"If this is memory, set HF_DEVICE=cpu or use a smaller model in LLM_MODEL."
        ) from err

    answer = _strip_thinking(answer or "")
    if not answer.strip():
        raise LLMError(f"The {provider} LLM returned an empty answer.")
    return answer.strip()
'''

def generate_answer(context: str, question: str) -> str:
    """Build the prompt from (context, question), call the chosen LLM, return the answer text."""
    system_prompt, user_prompt = build_prompt(context, question)
    return generate_text(system_prompt, user_prompt)


def generate_text(system_prompt: str, user_prompt: str) -> str:
    """Send ANY system + user prompt to the chosen LLM and return the cleaned reply.
    Used by generate_answer() (RAG) and by the tool planner/answer writer in the backend."""
    provider = os.getenv("LLM_PROVIDER", "").strip().lower()

    if provider not in PROVIDERS:
        raise LLMError(
            f"LLM_PROVIDER is '{provider}'. Set it in .env to one of: {', '.join(PROVIDERS)}"
        )
    call_llm, default_model = PROVIDERS[provider]
    model = os.getenv("LLM_MODEL") or default_model
    temperature = float(os.getenv("LLM_TEMPERATURE", "0"))   # 0 = most factual, least random

    try:
        answer = call_llm(system_prompt, user_prompt, model, temperature)
    except LLMError:
        raise
    except requests.exceptions.RequestException as err:       # no internet, timeout, Ollama not running
        raise LLMError(f"Could not reach the {provider} LLM: {err}") from err
    except (KeyError, IndexError, ValueError) as err:         # reply had an unexpected shape
        raise LLMError(f"Unexpected reply format from {provider}: {err}") from err
    except ImportError as err:                                # transformers / torch not installed
        raise LLMError(f"A required library is missing: {err}. Run: pip install transformers torch") from err
    except (OSError, RuntimeError, MemoryError) as err:       # model not found, download failed, out of memory
        raise LLMError(
            f"The {provider} model '{model}' failed: {err}. "
            f"If this is memory, set HF_DEVICE=cpu or use a smaller model in LLM_MODEL."
        ) from err

    answer = _strip_thinking(answer or "")
    if not answer.strip():
        raise LLMError(f"The {provider} LLM returned an empty answer.")
    return answer.strip()