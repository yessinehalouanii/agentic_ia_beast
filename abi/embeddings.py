from __future__ import annotations

import os
from typing import List, Optional, Any

from openai import OpenAI

_CLIENT: Optional[OpenAI] = None


def _client() -> OpenAI:
    global _CLIENT
    if _CLIENT is None:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("Missing OPENAI_API_KEY in environment.")
        _CLIENT = OpenAI(api_key=api_key)
    return _CLIENT


def _sanitize_texts(texts: List[Any]) -> List[str]:
    """
    OpenAI embeddings expects: input = str OR list[str]
    This makes sure we always send list[str] (non-empty).
    """
    if texts is None:
        return []

    out: List[str] = []
    for t in texts:
        if t is None:
            continue

        # If someone accidentally passes a dict/list/object -> force stringify
        if not isinstance(t, str):
            t = str(t)

        t = t.strip()
        if t:
            out.append(t)

    return out


def embed_texts(texts: List[Any]) -> List[List[float]]:
    """
    Generate embeddings for texts using OpenAI embeddings.
    """
    cleaned = _sanitize_texts(texts)

    if not cleaned:
        raise ValueError(
            "embed_texts(): input texts are empty/invalid after sanitization. "
            f"Original types: {[type(x).__name__ for x in (texts or [])]}"
        )

    model = os.getenv("EMBED_MODEL", "text-embedding-3-large")

    try:
        resp = _client().embeddings.create(
            model=model,
            input=cleaned,   # ✅ always list[str]
        )
    except Exception as e:
        # Helpful debug
        raise RuntimeError(
            "OpenAI embeddings.create() failed. "
            f"model={model} cleaned_types={[type(x).__name__ for x in cleaned]} "
            f"first_200={cleaned[0][:200]!r}"
        ) from e

    return [d.embedding for d in resp.data]
