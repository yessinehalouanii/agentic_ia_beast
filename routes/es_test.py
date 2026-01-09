# routes/es_test.py

from __future__ import annotations

import os
import json
from datetime import datetime, timezone  # 👈 NEW
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel
from typing import Optional, Any, Dict, List

import pandas as pd
import numpy as np
from pandas.api.types import is_datetime64_any_dtype

from abi.nested import detect_nested_schema, get_or_make_dotted_series

# ✅ LLM -> ES DSL
from abi.es_llm import llm_generate_es_query

# ✅ STATIC client (from .env)
from app.core.es_client import es

# ✅ DYNAMIC client factory (from user input)
from app.core.es_dynamic import make_es_client

# ❌ DISABLED: TABLE STORE for analytics (we are NOT loading full index into pandas)
# from app.core.table_store import TABLE_STORE

router = APIRouter(prefix="/es", tags=["Elasticsearch"])


# ------------------------------------------------------------------
# Models
# ------------------------------------------------------------------

class EsConnectRequest(BaseModel):
    base_url: str
    username: Optional[str] = None
    password: Optional[str] = None


class EsDynamicIndexRequest(BaseModel):
    base_url: str
    username: Optional[str] = None
    password: Optional[str] = None
    index_name: str


class EsAskRequest(BaseModel):
    base_url: str
    username: Optional[str] = None
    password: Optional[str] = None
    index_name: str

    question: str

    model: str = "gpt-4o-mini"
    api_key: Optional[str] = None

    size: int = 50
    flatten: bool = True

    # optional, in case later you want ask() to run too
    run_query: bool = False


class EsRunDslRequest(BaseModel):
    """
    Execute a RAW ES DSL string on the chosen Elasticsearch endpoint.

    dsl example:
      "GET invoices/_search\n{ \"size\": 0, \"query\": { ... } }"
    """
    base_url: str
    username: Optional[str] = None
    password: Optional[str] = None

    dsl: str
    flatten: bool = True


# ------------------------------------------------------------------
# ❌ DISABLED: FULL INDEX LOAD MODEL (prod-safe)
# ------------------------------------------------------------------
# class EsLoadIndexRequest(BaseModel):
#     """
#     Load a FULL ES index into pandas and register it in TABLE_STORE
#     so /docs/ask-analytics can run on it.
#     """
#     base_url: str
#     username: Optional[str] = None
#     password: Optional[str] = None
#     index_name: str
#
#     # where analytics will read tables from
#     workspace_id: str = "default"
#
#     # optionally override table name (default = index_name)
#     table_name: Optional[str] = None


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _flatten_mapping_properties(
    properties: Dict[str, Any],
    prefix: str = ""
) -> List[Dict[str, Any]]:
    """
    Flatten ES mapping "properties" into a list of:
      { "field": "a.b.c", "type": "keyword", "raw": {...} }
    """
    out: List[Dict[str, Any]] = []

    for field_name, spec in (properties or {}).items():
        path = f"{prefix}.{field_name}" if prefix else field_name
        if not isinstance(spec, dict):
            out.append({"field": path, "type": "unknown", "raw": spec})
            continue

        ftype = spec.get("type")

        # Object with nested properties
        if "properties" in spec and isinstance(spec["properties"], dict):
            out.append({"field": path, "type": ftype or "object", "raw": spec})
            out.extend(_flatten_mapping_properties(spec["properties"], prefix=path))
            continue

        # Multi-fields: text + keyword, etc.
        if "fields" in spec and isinstance(spec["fields"], dict):
            out.append({"field": path, "type": ftype or "multi", "raw": spec})
            for sub_name, sub_spec in spec["fields"].items():
                sub_path = f"{path}.{sub_name}"
                out.append(
                    {
                        "field": sub_path,
                        "type": (sub_spec or {}).get("type", "unknown"),
                        "raw": sub_spec,
                    }
                )
            continue

        out.append({"field": path, "type": ftype or "unknown", "raw": spec})

    return out


def _flatten_docs_to_rows(docs: List[dict]) -> List[dict]:
    """
    - Convert nested dict/list fields into dotted __ columns
    - Convert datetimes to strings
    - JSON-safe None for NaN/Inf
    """
    if not docs:
        return []

    df = pd.DataFrame(docs)
    if df.empty:
        return []

    dotted_paths = detect_nested_schema(df)
    nested_roots: set[str] = set()

    for dotted in dotted_paths:
        parts = dotted.split(".")
        if not parts:
            continue

        root = parts[0]
        nested_roots.add(root)

        new_col = get_or_make_dotted_series(df, dotted)
        if not new_col:
            continue

        nice_name = dotted.replace(".", "__")
        if nice_name in df.columns:
            continue

        df.rename(columns={new_col: nice_name}, inplace=True)

    # Drop original nested columns
    for root in nested_roots:
        if root in df.columns:
            df.drop(columns=[root], inplace=True)

    # Stringify any remaining dict/list cells
    for col in df.columns:
        if df[col].map(lambda x: isinstance(x, (dict, list))).any():
            df[col] = df[col].map(
                lambda x: str(x) if isinstance(x, (dict, list)) else x
            )

    # Convert datetime-like columns to strings
    for col in df.columns:
        if is_datetime64_any_dtype(df[col]):
            df[col] = df[col].astype(str)

    # Ensure JSON-safe values
    df = df.astype(object)
    df = df.replace({np.nan: None, np.inf: None, -np.inf: None})

    return df.to_dict(orient="records")


def _extract_properties_from_mapping(mapping: Dict[str, Any], index_name: str) -> Dict[str, Any]:
    """
    mapping = client.indices.get_mapping(index=...)
    returns properties dict safely.
    """
    idx_obj = mapping.get(index_name) or next(iter(mapping.values()), {})
    mappings_obj = (idx_obj or {}).get("mappings", {}) or {}
    properties = (mappings_obj or {}).get("properties", {}) or {}
    return properties


def _parse_es_dsl(dsl: str) -> tuple[Optional[str], Dict[str, Any]]:
    """
    Parse a string like:

      GET my-index/_search
      {
        "size": 0,
        "query": { ... }
      }

    → returns (index_name, body_dict)
    """

    lines = [ln for ln in dsl.splitlines() if ln.strip()]
    if not lines:
        raise ValueError("Empty DSL text")

    first = lines[0].strip()
    if not first.upper().startswith("GET "):
        raise ValueError("DSL must start with 'GET <index>/_search'")

    parts = first.split()
    if len(parts) < 2:
        raise ValueError("Invalid GET line in DSL")

    path = parts[1]  # e.g. "my-index/_search" or "/my-index/_search"
    path = path.lstrip("/")
    index_part = path.split("/")[0] if path else None

    json_str = "\n".join(lines[1:])
    try:
        body = json.loads(json_str)
    except Exception as e:
        raise ValueError(f"Failed to parse JSON body from DSL: {e}")

    return index_part, body


# ------------------------------------------------------------------
# ❌ DISABLED: FULL INDEX LOADER (prod-safe)
# ------------------------------------------------------------------
# def _load_index_as_df(
#     client,
#     index_name: str,
#     page_size: int = 5000,
# ) -> pd.DataFrame:
#     """
#     Stream the ENTIRE index into a single pandas DataFrame.
#
#     - Uses ES scroll API.
#     - No max docs limit (full index).
#     - Keeps nested fields as dicts (no flattening here).
#
#     ⚠️ DISABLED: can overload ES + crash API (RAM).
#     """
#     all_docs: list[dict] = []
#
#     resp = client.search(
#         index=index_name,
#         body={"query": {"match_all": {}}},  # no filter
#         size=page_size,
#         scroll="2m",
#     )
#
#     scroll_id = resp.get("_scroll_id")
#     hits = resp.get("hits", {}).get("hits", [])
#
#     while hits:
#         for h in hits:
#             all_docs.append(h.get("_source", {}))
#
#         resp = client.scroll(scroll_id=scroll_id, scroll="2m")
#         scroll_id = resp.get("_scroll_id")
#         hits = resp.get("hits", {}).get("hits", [])
#
#     if scroll_id:
#         try:
#             client.clear_scroll(scroll_id=scroll_id)
#         except Exception:
#             pass
#
#     if not all_docs:
#         return pd.DataFrame()
#
#     return pd.DataFrame(all_docs)


def _summarize_visits_per_year_agg(aggregations: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Detect an aggregation of the form:
      aggregations.visits_per_year.buckets[...].distinct_visits.value

    It supports both:
      - date_histogram on a date field (key_as_string + epoch key)
      - terms aggregation on a year-like field  (key = 2023, 2024, 2025, ...)

    Build a generic summary for all years:
      - visits_by_year: {"2022": 123, "2023": 150, ...}
      - start_year / end_year comparison (earliest vs latest)
      - human-readable text.

    Returns a dict or None if pattern not recognized.
    """
    if not aggregations:
        return None

    visits_per_year = aggregations.get("visits_per_year")
    if not isinstance(visits_per_year, dict):
        return None

    buckets = visits_per_year.get("buckets") or []
    if not buckets:
        return None

    visits_by_year: dict[str, int] = {}

    for b in buckets:
        year: Optional[str] = None

        # 1) Prefer key_as_string if present (e.g. "2024-01-01T00:00:00.000Z")
        kas = b.get("key_as_string")
        if isinstance(kas, str) and len(kas) >= 4:
            year = kas[:4]
        else:
            # 2) Fallback: treat small integers as literal years (e.g. 2023, 2024, 2025)
            key = b.get("key")
            if isinstance(key, (int, float)):
                k_int = int(key)
                if 1900 <= k_int <= 2100:
                    year = str(k_int)
                else:
                    # 3) Last resort: treat as epoch_millis
                    try:
                        ts = key / 1000.0
                        year = str(datetime.fromtimestamp(ts, tz=timezone.utc).year)
                    except Exception:
                        pass

        if not year:
            continue

        dv = b.get("distinct_visits", {})
        value = dv.get("value")
        if value is None:
            continue

        visits_by_year[year] = int(value)

    if not visits_by_year:
        return None

    cleaned = {
        y: v
        for y, v in visits_by_year.items()
        if y.isdigit() and 1900 <= int(y) <= 2100
    }
    if cleaned:
        visits_by_year = cleaned

    if not visits_by_year:
        return None

    years_sorted = sorted(visits_by_year.keys(), key=int)
    start_year = years_sorted[0]
    end_year = years_sorted[-1]

    start_value = visits_by_year[start_year]
    end_value = visits_by_year[end_year]

    if start_year == end_year:
        text = f"In {start_year} you had {start_value} distinct visits."
        return {
            "visits_by_year": visits_by_year,
            "start_year": start_year,
            "end_year": end_year,
            "start_value": start_value,
            "end_value": end_value,
            "delta": 0,
            "pct_change": None,
            "text": text,
        }

    delta = end_value - start_value
    pct_change = (delta / start_value * 100.0) if start_value else None

    timeline_parts = [f"{y}: {visits_by_year[y]}" for y in years_sorted]
    timeline_str = ", ".join(timeline_parts)

    text = (
        f"From {start_year} to {end_year}, distinct visits changed "
        f"from {start_value} to {end_value}"
    )
    if pct_change is not None:
        text += f" ({pct_change:+.1f}% change)."
    else:
        text += "."

    text += f" Year-by-year: {timeline_str}."

    return {
        "visits_by_year": visits_by_year,
        "start_year": start_year,
        "end_year": end_year,
        "start_value": start_value,
        "end_value": end_value,
        "delta": delta,
        "pct_change": pct_change,
        "text": text,
    }


# ------------------------------------------------------------------
# 1) Dynamic ES connection (used by EsConnectPanel)
# ------------------------------------------------------------------

@router.post("/indices/dynamic")
def list_indices_dynamic(req: EsConnectRequest):
    try:
        client = make_es_client(req.base_url, req.username, req.password)

        if not client.ping():
            raise HTTPException(status_code=400, detail=f"Could not ping Elasticsearch at {req.base_url}")

        rows = client.cat.indices(format="json")
        indices = [{"index": r.get("index")} for r in rows if r.get("index")]

        return {"indices": indices}

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ------------------------------------------------------------------
# 1.b) Dynamic mapping
# ------------------------------------------------------------------

@router.post("/mapping/dynamic")
def get_mapping_dynamic(req: EsDynamicIndexRequest):
    try:
        client = make_es_client(req.base_url, req.username, req.password)

        if not client.ping():
            raise HTTPException(status_code=400, detail=f"Could not ping Elasticsearch at {req.base_url}")

        index_name = (req.index_name or "").strip()
        if not index_name:
            raise HTTPException(status_code=400, detail="index_name is required")

        mapping = client.indices.get_mapping(index=index_name)
        properties = _extract_properties_from_mapping(mapping, index_name)
        fields = _flatten_mapping_properties(properties)

        return {
            "index": index_name,
            "fields": fields,
            "raw_mapping": mapping,
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ------------------------------------------------------------------
# 1.c) Dynamic sample
# ------------------------------------------------------------------

@router.post("/sample/dynamic")
def sample_index_dynamic(
    req: EsDynamicIndexRequest,
    size: int = Query(10, ge=1, le=200),
    flatten: bool = True,
):
    try:
        client = make_es_client(req.base_url, req.username, req.password)

        if not client.ping():
            raise HTTPException(status_code=400, detail=f"Could not ping Elasticsearch at {req.base_url}")

        index_name = (req.index_name or "").strip()
        if not index_name:
            raise HTTPException(status_code=400, detail="index_name is required")

        res = client.search(
            index=index_name,
            size=size,
            sort=[{"created_at": {"order": "desc"}}],
        )

        hits = res.get("hits", {}).get("hits", [])
        docs = [h.get("_source", {}) for h in hits]

        if not docs or not flatten:
            return docs

        return _flatten_docs_to_rows(docs)

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ------------------------------------------------------------------
# ❌ DISABLED: Load FULL index into TABLE_STORE for analytics (prod-safe)
# ------------------------------------------------------------------
# @router.post("/load-index-to-tables")
# def load_index_to_tables(req: EsLoadIndexRequest):
#     """
#     Load a FULL Elasticsearch index into pandas and register it
#     as a table in the in-memory TABLE_STORE for analytics.
#
#     Then /docs/ask-analytics can use it via workspace_id.
#
#     ⚠️ DISABLED: This can overload Elasticsearch + crash the API (RAM).
#     """
#     raise HTTPException(
#         status_code=403,
#         detail="Disabled: bulk index load into pandas/TABLE_STORE is not allowed.",
#     )


# ------------------------------------------------------------------
# 1.d) Ask ES (LLM -> DSL, NO execution here)
# ------------------------------------------------------------------

@router.post("/ask/dynamic")
def ask_es_dynamic(req: EsAskRequest):
    try:
        client = make_es_client(req.base_url, req.username, req.password)

        if not client.ping():
            raise HTTPException(status_code=400, detail=f"Could not ping Elasticsearch at {req.base_url}")

        index_name = (req.index_name or "").strip()
        if not index_name:
            raise HTTPException(status_code=400, detail="index_name is required")

        question = (req.question or "").strip()
        if not question:
            raise HTTPException(status_code=400, detail="question is required")

        api_key = (req.api_key or "").strip() or os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise HTTPException(status_code=500, detail="Missing OpenAI API key")

        mapping = client.indices.get_mapping(index=index_name)
        properties = _extract_properties_from_mapping(mapping, index_name)

        query_text = llm_generate_es_query(
            question=question,
            index_name=index_name,
            mappings={"properties": properties},
            model=req.model,
            api_key=api_key,
        )

        return {
            "ok": True,
            "query": query_text,
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ------------------------------------------------------------------
# 1.e) Run raw DSL on DB (used by second frontend button)
# ------------------------------------------------------------------

@router.post("/run-dsl/dynamic")
def run_dsl_dynamic(req: EsRunDslRequest):
    """
    Execute a RAW ES DSL string on the chosen Elasticsearch endpoint.

    Expects:
      dsl = "GET index/_search\n{ ... }"
    """
    try:
        client = make_es_client(req.base_url, req.username, req.password)

        if not client.ping():
            raise HTTPException(
                status_code=400,
                detail=f"Could not ping Elasticsearch at {req.base_url}",
            )

        dsl_text = (req.dsl or "").strip()
        if not dsl_text:
            raise HTTPException(status_code=400, detail="dsl is required")

        try:
            index_name, body = _parse_es_dsl(dsl_text)
        except Exception as e:
            raise HTTPException(
                status_code=400,
                detail=f"Failed to parse DSL: {e}",
            )

        if not index_name:
            raise HTTPException(
                status_code=400,
                detail="No index name found in DSL GET line",
            )

        try:
            res = client.search(index=index_name, body=body)
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"Error executing ES search: {e}",
            )

        hits = res.get("hits", {}).get("hits", [])
        docs = [h.get("_source", {}) for h in hits]

        if req.flatten and docs:
            docs = _flatten_docs_to_rows(docs)

        aggregations = res.get("aggregations") or {}
        summary = _summarize_visits_per_year_agg(aggregations)

        return {
            "ok": True,
            "index": index_name,
            "body": body,
            "hits": docs,
            "raw": {
                "took": res.get("took"),
                "total": res.get("hits", {}).get("total"),
                "aggregations": aggregations,
            },
            "summary": summary,
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ------------------------------------------------------------------
# 2) Static ES client endpoints (unchanged)
# ------------------------------------------------------------------

@router.get("/indices")
def list_indices():
    try:
        return es.cat.indices(format="json")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/sample/{index_name}")
def sample_index(
    index_name: str,
    size: int = Query(10, ge=1, le=200),
    flatten: bool = True,
):
    try:
        res = es.search(
            index=index_name,
            size=size,
            sort=[{"created_at": {"order": "desc"}}],
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    hits = res.get("hits", {}).get("hits", [])
    docs = [h.get("_source", {}) for h in hits]

    if not docs or not flatten:
        return docs

    return _flatten_docs_to_rows(docs)
