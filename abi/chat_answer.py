# abi/chat_answer.py
from __future__ import annotations

import os
import json
from typing import Any, Dict, List, Optional

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


def answer(question: str, profile: Dict[str, Any], chunks: List[Dict[str, Any]]) -> str:
    """
    chunks format expected (from pinecone_store.query):
      [{"score":..., "metadata": {"text":..., "doc_id":..., "filename":..., "section":...}}, ...]
    """
    model = os.getenv("CHAT_MODEL", "gpt-4.1-mini")
    client = _client()

    # Build compact context
    ctx_parts: List[str] = []
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

    # ✅ Modified: remove writing/app/action guidance, keep it strictly docs-grounded Q/A
    system = (
        "You are a documents-only assistant.\n"
        "You MUST ground answers in the provided DOCUMENT CONTEXT.\n"
        "Do NOT invent facts not present in the document context.\n"
        "If the answer is not supported by the context, say what is missing and ask a precise follow-up question.\n"
        "Return ONLY valid JSON. Never include markdown fences.\n"
    )

    # ✅ Modified: remove the extra sections from response_format
    user = {
        "business_profile": profile,  # kept in case you still want tone/entities later
        "document_context": context_text,
        "question": question,
        "response_format": {
            "sections": [
                "Answer",
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
