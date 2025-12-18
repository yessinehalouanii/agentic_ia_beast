# routes/es_test.py

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel
from elasticsearch.exceptions import NotFoundError

import pandas as pd
import numpy as np
from pandas.api.types import is_datetime64_any_dtype

from abi.nested import detect_nested_schema, get_or_make_dotted_series

# ✅ STATIC client (from .env)
from app.core.es_client import es

# ✅ DYNAMIC client factory (from user input)
from app.core.es_dynamic import make_es_client


router = APIRouter(prefix="/es", tags=["Elasticsearch"])


# ------------------------------------------------------------------
# 1) Dynamic ES connection (used by EsConnectPanel)
# ------------------------------------------------------------------

class EsConnectRequest(BaseModel):
    base_url: str
    username: str | None = None
    password: str | None = None


@router.post("/indices/dynamic")
def list_indices_dynamic(req: EsConnectRequest):
    """
    Frontend EsConnectPanel calls this.

    ✅ Uses dynamic client created from user-provided base_url/auth
    ✅ Uses _cat/indices (fast)
    """
    try:
        client = make_es_client(req.base_url, req.username, req.password)

        # quick ping (obeys request_timeout if configured in es_dynamic)
        if not client.ping():
            raise HTTPException(
                status_code=400,
                detail=f"Could not ping Elasticsearch at {req.base_url}",
            )

        # ✅ fast list of indices
        rows = client.cat.indices(format="json")  # list[dict]
        indices = [{"index": r.get("index")} for r in rows if r.get("index")]

        print("DEBUG /es/indices/dynamic ->", indices)
        return {"indices": indices}

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ------------------------------------------------------------------
# 2) Static ES client (your existing insight ES) endpoints
# ------------------------------------------------------------------

@router.get("/indices")
def list_indices():
    """
    List indices using the default ES client (app.core.es_client.es).
    """
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
    """
    Return sample documents from any Elasticsearch index (STATIC ES),
    flattened so the frontend sees scalar columns like `customer__customer_id`
    instead of nested dicts/lists. Sorted by newest created_at.
    """
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

    # If no docs or flatten disabled → return raw
    if not docs or not flatten:
        return docs

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
