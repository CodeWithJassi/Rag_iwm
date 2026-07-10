"""Provider failover chain. Lifted from the existing summariser, unchanged in behaviour.

Every LLM call in this project -- summarise, decompose, rephrase, generate, judge --
goes through llm_complete(). One place to change providers, one place to debug.
"""
import json
import logging
import re
import time

import requests

from config import (GEMINI_API_KEY, GROQ_API_KEY, LLM_BASE_URL, LLM_CYCLES,
                    LLM_MODEL, OPENAI_API_KEY, OPENROUTER_API_KEY)

logger = logging.getLogger(__name__)


def _clean_llm_text(s: str) -> str:
    """Strip markdown fences and leading/trailing whitespace."""
    if not s:
        return ""
    return re.sub(r"^```(?:json|markdown)?|```$", "", s.strip(), flags=re.M).strip()


def _try_ollama(prompt: str, max_tokens: int, model: str = "") -> str:
    effective_model = model or LLM_MODEL
    resp = requests.post(
        f"{LLM_BASE_URL}/api/generate",
        json={"model": effective_model, "prompt": prompt, "stream": False,
              "options": {"temperature": 0.2, "num_predict": max_tokens}},
        timeout=300,  # 5 min is enough for any on-prem model
    )
    resp.raise_for_status()
    body = resp.json()
    out = body.get("response", "").strip()
    if not out:
        # The model may still be loading into GPU memory on first use.
        # Log what Ollama returned so we can tell "loading" from "missing".
        logger.warning(f"Ollama({effective_model}) returned empty response. "
                       f"Full body keys: {list(body.keys())}. "
                       f"Waiting 8s then retrying once (model may be warming up).")
        time.sleep(8)
        resp = requests.post(
            f"{LLM_BASE_URL}/api/generate",
            json={"model": effective_model, "prompt": prompt, "stream": False,
                  "options": {"temperature": 0.2, "num_predict": max_tokens}},
            timeout=300,
        )
        resp.raise_for_status()
        out = resp.json().get("response", "").strip()
        if not out:
            raise ValueError(
                f"Empty Ollama response after retry — is '{effective_model}' "
                f"pulled? Run: ollama pull {effective_model}")
    return out


def _try_openai(prompt: str, max_tokens: int) -> str:
    if not OPENAI_API_KEY:
        raise ValueError("OPENAI_API_KEY not set")
    from openai import OpenAI
    client = OpenAI(api_key=OPENAI_API_KEY)
    c = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "system", "content": "You are an equity research analyst."},
                  {"role": "user", "content": prompt}],
        temperature=0.2, max_tokens=max_tokens,
    )
    return c.choices[0].message.content.strip()


def _try_gemini(prompt: str, max_tokens: int) -> str:
    if not GEMINI_API_KEY:
        raise ValueError("GEMINI_API_KEY not set")
    last = None
    for model in ("gemini-2.0-flash", "gemini-1.5-flash"):
        try:
            url = (f"https://generativelanguage.googleapis.com/v1beta/models/"
                   f"{model}:generateContent?key={GEMINI_API_KEY}")
            data = {"contents": [{"parts": [{"text": prompt}]}],
                    "generationConfig": {"temperature": 0.2, "maxOutputTokens": max_tokens}}
            r = requests.post(url, headers={"Content-Type": "application/json"},
                              json=data, timeout=120)
            if r.status_code == 429:
                last = Exception(f"{model} rate limited")
                continue
            r.raise_for_status()
            return r.json()["candidates"][0]["content"]["parts"][0]["text"].strip()
        except Exception as e:
            last = e
            continue
    raise last or Exception("All Gemini models failed")


def _try_groq(prompt: str, model: str, max_tokens: int) -> str:
    if not GROQ_API_KEY:
        raise ValueError("GROQ_API_KEY not set")
    from groq import Groq
    client = Groq(api_key=GROQ_API_KEY)
    c = client.chat.completions.create(
        model=model, messages=[{"role": "user", "content": prompt}],
        temperature=0.2, max_tokens=max_tokens, top_p=1,
    )
    return c.choices[0].message.content.strip()


def _try_openrouter(prompt: str, max_tokens: int) -> str:
    if not OPENROUTER_API_KEY:
        raise ValueError("OPENROUTER_API_KEY not set")
    from openai import OpenAI
    client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=OPENROUTER_API_KEY)
    c = client.chat.completions.create(
        model="google/gemini-2.0-flash:free",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=max_tokens, timeout=120,
    )
    return c.choices[0].message.content.strip()


def _providers(prompt: str, max_tokens: int, model: str) -> list[tuple[str, callable]]:
    """Build the failover chain from *available* providers only.

    Providers whose API key is not set are skipped entirely — no point trying
    them and logging a warning for something we already know will fail."""
    providers: list[tuple[str, callable]] = [
        (f"Ollama({model or LLM_MODEL})", lambda: _try_ollama(prompt, max_tokens, model)),
    ]
    if OPENAI_API_KEY:
        providers.append(("OpenAI gpt-4o-mini", lambda: _try_openai(prompt, max_tokens)))
    if GEMINI_API_KEY:
        providers.append(("Gemini", lambda: _try_gemini(prompt, max_tokens)))
    if GROQ_API_KEY:
        providers.append(("Groq 70b", lambda: _try_groq(prompt, "llama-3.3-70b-versatile", max_tokens)))
        providers.append(("Groq 8b", lambda: _try_groq(prompt, "llama-3.1-8b-instant", max_tokens)))
    if OPENROUTER_API_KEY:
        providers.append(("OpenRouter", lambda: _try_openrouter(prompt, max_tokens)))
    return providers


def llm_complete(prompt: str, max_tokens: int = 700, label: str = "llm",
                 model: str = "") -> str | None:
    """Run a prompt through the provider failover chain. Returns None if all fail.

    The optional ``model`` overrides the Ollama model for this call only (e.g.
    when the user selected a different model in the UI).  Other providers ignore
    it and use their own fixed model strings."""
    providers = _providers(prompt, max_tokens, model)
    for cycle in range(LLM_CYCLES):
        for name, fn in providers:
            try:
                out = _clean_llm_text(fn())
                if out:
                    logger.info(f"[{label}] success via {name} (cycle {cycle + 1})")
                    return out
            except Exception as e:
                logger.warning(f"[{label}] {name} failed: {e}")
                continue
        if cycle < LLM_CYCLES - 1:
            time.sleep(2)
    logger.error(f"[{label}] all providers failed across {LLM_CYCLES} cycles")
    return None


def llm_json(prompt: str, max_tokens: int = 400, label: str = "json", model: str = ""):
    """llm_complete + tolerant JSON parse. Returns None on failure at either stage.

    LLMs wrap JSON in prose despite instructions, so we grab the outermost
    brace/bracket span rather than trusting the whole response to parse.
    """
    out = llm_complete(prompt, max_tokens=max_tokens, label=label, model=model)
    if not out:
        return None
    for opener, closer in (("{", "}"), ("[", "]")):
        i, j = out.find(opener), out.rfind(closer)
        if i != -1 and j > i:
            try:
                return json.loads(out[i:j + 1])
            except json.JSONDecodeError:
                continue
    logger.warning(f"[{label}] response was not parseable JSON")
    return None
