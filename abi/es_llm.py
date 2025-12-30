# abi/es_llm.py
import json
from typing import Any, Dict, Optional

try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except Exception:
    OPENAI_AVAILABLE = False


def _collect_field_types(mappings: Dict[str, Any]) -> tuple[list[str], list[str]]:
    """
    Walk the mapping and collect:
      - date_fields: fields with type "date" or "date_nanos"
      - year_like_fields: numeric/keyword fields whose name includes "year"
    """
    date_fields: list[str] = []
    year_like_fields: list[str] = []

    # In our routes we usually pass {"properties": {...}}, but be defensive
    props = mappings.get("properties") if isinstance(mappings, dict) else None
    if not isinstance(props, dict):
        props = mappings or {}

    def walk(path: str, spec: Any):
        if not isinstance(spec, dict):
            return

        ftype = spec.get("type")
        fname = path

        # Date-like fields
        if ftype in ("date", "date_nanos"):
            date_fields.append(fname)

        # "year"-looking fields that are numeric or keyword-like
        if (
            ftype in ("integer", "long", "short", "byte", "float", "double", "keyword")
            and "year" in fname.lower()
        ):
            year_like_fields.append(fname)

        # Recurse nested properties
        nested = spec.get("properties")
        if isinstance(nested, dict):
            for sub_name, sub_spec in nested.items():
                child_path = f"{fname}.{sub_name}" if fname else sub_name
                walk(child_path, sub_spec)

    if isinstance(props, dict):
        for root_name, root_spec in props.items():
            walk(root_name, root_spec)

    return date_fields, year_like_fields


def llm_generate_es_query(
    question: str,
    index_name: str,
    mappings: Dict[str, Any],
    model: str,
    api_key: Optional[str],
) -> str:
    """
    Returns ONLY:
    GET <index>/_search
    { Elasticsearch JSON body }
    """

    if not OPENAI_AVAILABLE:
        raise RuntimeError("OpenAI SDK not available")
    if not api_key:
        raise RuntimeError("Missing OPENAI_API_KEY")

    client = OpenAI(api_key=api_key)

    # 🔎 Pre-compute helpful hints for the model from the mapping
    date_fields, year_like_fields = _collect_field_types(mappings)

    system = (
        "You are an Elasticsearch DSL generator.\n"
        "Your job is to produce a single Elasticsearch query that answers the question.\n\n"

        "OUTPUT FORMAT (MUST FOLLOW EXACTLY):\n"
        "GET <index>/_search\n"
        "{ valid Elasticsearch JSON body }\n\n"

        "STRICT RULES (DO NOT BREAK):\n"
        "- Output NOTHING except the DSL.\n"
        "- No explanations.\n"
        "- No markdown.\n"
        "- No comments.\n"
        "- No analysis.\n"
        "- DO NOT invent date ranges.\n"
        "- DO NOT add time filters unless explicitly requested in the question.\n"
        "- DO NOT infer years, months, or ranges not mentioned by the user.\n"
        "- Use bool.filter, NOT bool.must, for hard filters unless scoring is required.\n"
        "- Keep numeric filters as range queries where appropriate.\n"
        "- Use painless scripts ONLY when absolutely required and there is no native aggregation.\n\n"

        "MAPPING TYPE RULES (VERY IMPORTANT):\n"
        "- You are given the index MAPPING JSON and precomputed lists of DATE_FIELDS and YEAR_LIKE_FIELDS.\n"
        "- ALWAYS respect the 'type' from the mapping:\n"
        "  * If a field has type 'date' or 'date_nanos', you MAY use date_histogram and range queries on it.\n"
        "  * If a field has type 'integer', 'long', 'short', 'byte', 'float', 'double', or 'keyword',\n"
        "    you MUST NOT use date_histogram on that field.\n"
        "- For fields listed in YEAR_LIKE_FIELDS (e.g. 'import_year' with values 2024, 2025):\n"
        "  * Treat them as year dimensions (numeric or keyword), NOT as dates.\n"
        "  * Use 'terms' aggregation or 'filter' aggregations (term/terms) on that field to group/compare years.\n"
        "  * Do NOT use date_histogram on these year fields.\n\n"

        "FIELD NAME RULES:\n"
        "- If the QUESTION explicitly mentions a field name (e.g. 'use import_at field'),\n"
        "  you MUST use exactly that field for filters and/or aggregations related to that concept.\n"
        "- Do NOT silently switch to a different field name.\n\n"

        "DATE / TIME & GROUP BY RULES:\n"
        "- Only use date_histogram on fields that are actual date types in the mapping.\n"
        "- If the question asks for metrics PER YEAR, PER MONTH, PER DAY, etc.,\n"
        "  use date_histogram on the appropriate *date* field instead of scripts.\n"
        "- Example: for per-year metrics on a date field, use:\n"
        "  \"date_histogram\": { \"field\": \"<date_field>\", \"calendar_interval\": \"year\" }\n"
        "- Never use doc['<date_field>'].value.getYear() + 1900 or similar hacks.\n"
        "- Only use scripts on dates if there is truly no alternative.\n"
        "- For general group-by on non-date fields, use terms aggregation.\n\n"

        "VISITS / DISTINCT COUNTS RULES:\n"
        "- If the question refers to 'distinct visits', 'unique visits', or 'number of visits',\n"
        "  and the mapping contains a field like 'visit_id' (keyword / id field),\n"
        "  use a cardinality aggregation on that field:\n"
        "    \"cardinality\": { \"field\": \"visit_id\" }\n"
        "- If comparing visits across years (e.g. 2024 vs 2025):\n"
        "  * If you have a DATE_FIELD (e.g. 'import_at', 'updated_at') and the question says to use it,\n"
        "    you MAY use a date_histogram by year on that date field plus a cardinality sub-aggregation.\n"
        "  * If you have a YEAR_LIKE_FIELD (e.g. 'import_year' = 2024, 2025),\n"
        "    you MUST use terms or filter aggregations on that year field, NOT date_histogram.\n"
        "- Do NOT use scripts to extract the year from a date when date_histogram can do it.\n\n"

        "FILTERING RULES:\n"
        "- If the question mentions an explicit date range, use a range query on the relevant date field.\n"
        "- If the question mentions a specific year (e.g. 2024) and you are using a date field, convert that into\n"
        "  a range [2024-01-01, 2025-01-01) on that date field.\n"
        "- If the question mentions a specific year and you have a YEAR_LIKE_FIELD (e.g. 'import_year'),\n"
        "  you may instead filter using term/terms on that year field.\n"
        "- For weekday filtering, if explicitly requested, use painless with:\n"
        "  doc['<date_field>'].size()!=0 &&\n"
        "  doc['<date_field>'].value.getDayOfWeek().getValue() == <1-7>\n"
        "  where Monday = 1.\n\n"

        "AGGREGATION RULES:\n"
        "- Aggregations MUST reflect the question exactly.\n"
        "- If the user asks to compare metrics between two periods (e.g. 2024 vs 2025),\n"
        "  include aggregations that produce buckets/values for each period so that\n"
        "  the client code can compare them.\n"
        "- Do NOT compute ratios or differences in scripts unless explicitly requested;\n"
        "  just return the raw per-period metrics (e.g. per year).\n"
    )

    user = (
        f"INDEX: {index_name}\n"
        f"DATE_FIELDS (from mapping): {date_fields}\n"
        f"YEAR_LIKE_FIELDS (from mapping): {year_like_fields}\n"
        f"MAPPING:\n{json.dumps(mappings, ensure_ascii=False)[:120000]}\n\n"
        f"QUESTION:\n{question}\n"
    )

    resp = client.responses.create(
        model=model,
        instructions=system,
        input=user,
        temperature=0.0,
        max_output_tokens=900,
    )

    text = (getattr(resp, "output_text", "") or "").strip()
    if not text:
        raise RuntimeError("Empty model response")

    return text
