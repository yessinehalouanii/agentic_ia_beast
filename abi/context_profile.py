# abi/context_profile.py
from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional

from openai import OpenAI


def _client() -> OpenAI:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("Missing OPENAI_API_KEY in environment.")
    return OpenAI(api_key=api_key)


PROFILE_SCHEMA = {
    "business_name": "",
    "domain": "",
    "summary": "",
    "audience": "",
    "tone": "",
    "glossary": [{"term": "", "definition": ""}],
    "entities": {
        "products": [],
        "systems": [],
        "teams": [],
        "people": [],
    },
    "kpis": [],
    "constraints": [],
    "writing_rules": [],
    "doc_templates": [
        {"name": "General", "headings": ["Overview", "Details", "Decisions", "Next steps"]}
    ],
    "questions_to_ask_if_missing": [],
}


def build_or_update_profile(
    excerpts: str,
    existing_profile: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Build a structured business profile from document excerpts.
    If existing_profile is provided, update/merge it.
    Returns a dict (JSON).
    """
    model = os.getenv("CHAT_MODEL", "gpt-4.1-mini")
    client = _client()

    system = (
        "You extract business context from documents and return ONLY valid JSON.\n"
        "Never include markdown fences.\n"
        "If information is missing, keep fields empty or add follow-up questions.\n"
        "Use this JSON shape (keys must exist):\n"
        + json.dumps(PROFILE_SCHEMA, ensure_ascii=False)
    )

    user = {
        "existing_profile": existing_profile or {},
        "document_excerpts": excerpts,
        "task": (
            "Create or update a Business Context Profile from these documents. "
            "Infer domain, audience, tone, glossary, entities, constraints, writing rules, and doc templates "
            "based on what you see. Keep it general and accurate."
        ),
    }

    resp = client.responses.create(
        model=model,
        instructions=system,
        input=json.dumps(user, ensure_ascii=False),
        temperature=0.2,
        max_output_tokens=900,
    )

    text = (getattr(resp, "output_text", "") or "").strip()

    # Robust JSON parsing
    try:
        profile = json.loads(text)
    except Exception:
        # Last resort: try to extract first JSON object
        start = text.find("{")
        end = text.rfind("}")
        if start >= 0 and end > start:
            profile = json.loads(text[start : end + 1])
        else:
            raise ValueError("Profile generation did not return valid JSON.")

    # Ensure required keys exist
    for k in PROFILE_SCHEMA.keys():
        if k not in profile:
            profile[k] = PROFILE_SCHEMA[k]

    return profile
