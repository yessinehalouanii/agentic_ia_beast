# abi/chat_answer.py
from __future__ import annotations

import os
import json
from typing import Any, Dict, List, Optional

from openai import OpenAI


def _client() -> OpenAI:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("Missing OPENAI_API_KEY in environment.")
    return OpenAI(api_key=api_key)


def answer(question: str, profile: Dict[str, Any], chunks: List[Dict[str, Any]]) -> str:
    """
    chunks format expected (from pinecone_store.query):
      [{"score":..., "metadata": {"text":..., "doc_id":..., "filename":..., "section":...}}, ...]
    """
    model = os.getenv("CHAT_MODEL", "gpt-4.1-mini")
    client = _client()

    # Build compact context
    ctx_parts = []
    for m in chunks[:6]:
        md = (m.get("metadata") or {})
        txt = (md.get("text") or "").strip()
        if not txt:
            continue
        filename = md.get("filename", "document")
        section = md.get("section") or md.get("heading") or ""
        score = m.get("score", 0.0)
        header = f"[{filename}{' | ' + section if section else ''} | score={score:.3f}]"
        ctx_parts.append(header + "\n" + txt)

    context_text = "\n\n".join(ctx_parts).strip() or "(no relevant document context retrieved)"

    system = (
        "You are a business-aware assistant.\n"
        "You MUST adapt to the provided BUSINESS PROFILE.\n"
        "You MUST ground answers in the provided DOCUMENT CONTEXT.\n"
        "If the answer is not supported by the context, say exactly what is missing and ask a precise follow-up.\n"
        "When asked to write a document, follow writing_rules and doc_templates from the profile.\n"
        "Do not invent company-specific facts.\n"
    )

    user = {
        "business_profile": profile,
        "document_context": context_text,
        "question": question,
        "response_format": {
            "sections": [
                "Answer",
                "If writing is requested: Recommended document structure",
                "If app/action is requested: Suggested app actions",
                "Missing info (if any)",
            ]
        },
    }

    resp = client.responses.create(
        model=model,
        instructions=system,
        input=json.dumps(user, ensure_ascii=False),
        temperature=0.2,
        max_output_tokens=800,
    )
    return (getattr(resp, "output_text", "") or "").strip()
