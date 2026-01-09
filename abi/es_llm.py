# abi/es_llm.py
"""
LLM → Elasticsearch DSL generator.

Used by routes/es_test.py:

    from abi.es_llm import llm_generate_es_query

It takes:
  - natural-language question
  - index name
  - ES mappings (properties)
and returns a single ES request as plain text:

    GET <index>/_search
    {
      ...
    }
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, Optional

OPENAI_AVAILABLE = False
try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except Exception as e:
    print("OpenAI import failed in abi.es_llm:", repr(e))


def _strip_code_fences(text: str) -> str:
    """
    Remove ```json ... ``` or ``` ... ``` wrappers if present.
    """
    return re.sub(
        r"^```(?:json)?\s*|\s*```$",
        "",
        text.strip(),
        flags=re.MULTILINE,
    ).strip()


def _ensure_get_search_format(
    text: str,
    index_name: str,
) -> str:
    """
    Make sure the output looks like:

        GET index/_search
        { ... }

    If the model only returns JSON, wrap it.
    """
    stripped = text.strip()
    if stripped.upper().startswith("GET "):
        return stripped

    # Try to treat the whole thing as JSON
    try:
        body = json.loads(stripped)
        body_str = json.dumps(body, indent=2, ensure_ascii=False)
        return f"GET {index_name}/_search\n{body_str}"
    except Exception:
        # Fallback: just prefix a GET line
        return f"GET {index_name}/_search\n{stripped}"


def llm_generate_es_query(
    question: str,
    index_name: str,
    mappings: Dict[str, Any],
    model: str = "gpt-4o-mini",
    api_key: Optional[str] = None,
) -> str:
    """
    Main entry point used by /es/ask/dynamic.

    Returns a **plain text** ES DSL string.
    """
    if not OPENAI_AVAILABLE:
        raise RuntimeError("OpenAI client is not available (import failed).")
    if not api_key:
        raise RuntimeError("Missing OpenAI API key for llm_generate_es_query.")

    client = OpenAI(api_key=api_key)

    # We only need properties for the prompt; keep it small-ish.
    properties = (mappings or {}).get("properties", mappings or {})
    mapping_str = json.dumps(properties, indent=2, ensure_ascii=False)

    # 🔥 STRONG SYSTEM PROMPT TO ENFORCE GOOD ES DSL
    system = (
        "You are an expert Elasticsearch engineer. Your ONLY job is to translate a "
        "natural-language analytics question into a single Elasticsearch search request.\n"
        "\n"
        "OUTPUT FORMAT (VERY IMPORTANT):\n"
        "  - Return ONLY plain text, no markdown, no backticks, no comments.\n"
        "  - First line: GET <index-name>/_search\n"
        "  - Then a newline, then a JSON object body.\n"
        "  - The JSON MUST be syntactically valid (double-quoted keys, proper commas, etc.).\n"
        "\n"
        "GENERAL QUERY RULES:\n"
        "  - Use ONLY fields that exist in the provided mappings.\n"
        "  - If a field has both `text` and `keyword` subfields, use the `keyword` subfield for\n"
        "    exact filters and aggregations (e.g. `customer_name.keyword`).\n"
        "  - Prefer `bool` queries with `must`, `filter`, `must_not`, `should` instead of `query_string`.\n"
        "  - For exact matches on keyword fields use `term` or `terms`.\n"
        "  - For full-text search on text fields use `match` or `multi_match` (sparingly).\n"
        "  - For numeric fields use `range`, `term`, or aggregations (`sum`, `avg`, etc.).\n"
        "  - DO NOT use scripts or painless; queries must be safe and performant.\n"
        "  - Always include a reasonable `size`:\n"
        "      * If the user only cares about KPIs / metrics → use `size: 0`.\n"
        "      * If the user wants example documents → use a small size like 50.\n"
        "  - When counts/percentages matter, add `track_total_hits: true`.\n"
        "\n"
        "DATE / TIME RULES:\n"
        "  - Use `range` queries on date fields with `gte`/`lte`.\n"
        "  - For relative ranges use:\n"
        "      * last 7 days   → `now-7d/d`\n"
        "      * last 30 days  → `now-30d/d`\n"
        "      * last 12 months / last year → `now-12M/M` or `now-1y/y`\n"
        "  - Choose the most appropriate date field from mappings (names containing\n"
        "    `created`, `date`, `timestamp`, `dropoff`, `pickup`, etc.).\n"
        "\n"
        "AGGREGATION RULES:\n"
        "  - If the question asks for **totals**, **sums**, **revenue**, or **amounts**, use metric\n"
        "    aggregations such as `sum`, `avg`, or `value_count` on numeric fields.\n"
        "  - If the question asks for **how many** or **count** per dimension (customer, location,\n"
        "    status, etc.), use `terms` aggregations on the appropriate keyword field.\n"
        "  - For **top N** entities (top customers, top locations, etc.), use `terms` aggregation\n"
        "    with `size` = N and order by a metric (e.g. sum of revenue).\n"
        "  - For **trends over time** (per day, per week, per month, per year, over time), use a\n"
        "    `date_histogram` on a date field with a reasonable `calendar_interval` (e.g. `day`,\n"
        "    `week`, or `month`).\n"
        "  - Nest aggregations logically (e.g. `date_histogram` → `terms` → `sum`).\n"
        "\n"
        "MULTI-INDEX / OTHER RULES:\n"
        "  - Assume the request targets exactly ONE index: the `Target index` provided.\n"
        "  - Do NOT use cross-cluster search or index patterns.\n"
        "  - Do NOT invent fields: every field you reference must exist in the mappings.\n"
        "\n"
        "FINAL ANSWER REQUIREMENTS:\n"
        "  - Your final answer MUST be exactly one Elasticsearch search request in the format:\n"
        "        GET <index-name>/_search\n"
        "        { JSON body }\n"
        "  - No markdown, no prose, no explanation, no comments, no surrounding backticks.\n"
    )

    user = (
        f"User question:\n{question}\n\n"
        f"Target index: {index_name}\n\n"
        "Elasticsearch mappings (properties):\n"
        f"{mapping_str}\n\n"
        "Write exactly ONE Elasticsearch search request that best answers the question, "
        "following ALL the rules above."
    )

    # Simple implementation using chat.completions.
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        temperature=0.1,
        max_tokens=800,
    )

    content = resp.choices[0].message.content or ""
    content = _strip_code_fences(content)
    if not content.strip():
        raise RuntimeError("llm_generate_es_query: model returned empty content.")

    return _ensure_get_search_format(content, index_name=index_name)
