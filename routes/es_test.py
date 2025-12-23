# routes/es_test.py

from __future__ import annotations

import os
import json
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

        # Get mapping
        mapping = client.indices.get_mapping(index=index_name)
        properties = _extract_properties_from_mapping(mapping, index_name)

        # ✅ Generate ES query ONLY (no execution)
        query_text = llm_generate_es_query(
            question=question,
            index_name=index_name,
            mappings={"properties": properties},
            model=req.model,
            api_key=api_key,
        )

        # Return exactly what the model produced
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

        # Execute against the REAL ES index
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

        return {
            "ok": True,
            "index": index_name,
            "body": body,
            "hits": docs,
            "raw": {
                "took": res.get("took"),
                "total": res.get("hits", {}).get("total"),
                "aggregations": res.get("aggregations"),
            },
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
