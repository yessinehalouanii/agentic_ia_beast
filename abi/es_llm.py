# abi/es_llm.py
import json
from typing import Any, Dict, Optional

try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except Exception:
    OPENAI_AVAILABLE = False


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

    system = (
        "You are an Elasticsearch DSL generator.\n"
        "Output MUST be EXACTLY:\n"
        "GET <index>/_search\n"
        "{ valid Elasticsearch JSON body }\n\n"

        "STRICT RULES (DO NOT BREAK):\n"
        "- Output NOTHING except the DSL.\n"
        "- No explanations.\n"
        "- No markdown.\n"
        "- No comments.\n"
        "- No analysis.\n"
        "- DO NOT invent date ranges.\n"
        "- DO NOT add time filters unless explicitly requested.\n"
        "- DO NOT infer years, months, or ranges.\n"
        "- Use bool.filter, NOT bool.must, unless scoring is required.\n"
        "- Keep numeric filters as range queries.\n"
        "- Use painless scripts ONLY when required.\n"
        "- For weekday filtering, use:\n"
        "  doc['<date_field>'].size()!=0 && "
        "doc['<date_field>'].value.getDayOfWeek().getValue() == 1\n"
        "- Monday = 1.\n"
        "- Aggregations MUST reflect the question exactly.\n"
        "- Group-by MUST use terms aggregation.\n"
    )


    user = (
        f"INDEX: {index_name}\n"
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
