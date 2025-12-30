# app/api/tables_routes.py

from fastapi import APIRouter, HTTPException, UploadFile, File, Cookie
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
from urllib.parse import urljoin
import io
import pandas as pd
from abi.fetch import fetch_endpoints
from app.core.table_store import TABLE_STORE
router = APIRouter(prefix="/tables", tags=["Tables"])


# -------------------------------------------------------------------
# Models
# -------------------------------------------------------------------

class EndpointIn(BaseModel):
    name: str
    path_or_url: str
    limit: Optional[int] = None


class FetchRequest(BaseModel):
    workspace_id: str = "default"  # workspace / customer / environment
    base_url: str
    token: Optional[str] = None    # token optional (cookie-first)
    endpoints: List[EndpointIn]
    flatten: bool = True


class TableMeta(BaseModel):
    name: str
    rows: int
    cols: int


class FetchResponse(BaseModel):
    tables: List[TableMeta]
    errors: List[str]


# -------------------------------------------------------------------
# Fetch all endpoints → tables
# -------------------------------------------------------------------

@router.post("/fetch-all", response_model=FetchResponse)
def fetch_all(
    req: FetchRequest,
    access_token: Optional[str] = Cookie(default=None),  # ✅ READ COOKIE
):
    """
    Equivalent of Streamlit 'Fetch all':
    - UI sends endpoints
    - Auth token is taken from:
        1) request body (legacy)
        2) HttpOnly cookie (preferred)
    - Backend calls abi.fetch.fetch_endpoints
    """
    if not req.endpoints:
        raise HTTPException(status_code=400, detail="No endpoints provided.")

    # ✅ Resolve token (cookie first, fallback to body)
    token = req.token or access_token
    if not token:
        raise HTTPException(status_code=401, detail="Not authenticated")

    endpoint_dicts: List[Dict[str, Any]] = []
    for ep in req.endpoints:
        path = ep.path_or_url
        if path.lower().startswith("http"):
            full_url = path
        else:
            full_url = urljoin(req.base_url.rstrip("/") + "/", path.lstrip("/"))

        endpoint_dicts.append(
            {
                "name": ep.name.strip(),
                "url": full_url,
                "limit": ep.limit,
            }
        )

    # ✅ workspace-aware existing tables (from memory only)
    ws = (req.workspace_id or "default").strip() or "default"
    existing = TABLE_STORE.get_tables(ws)

    tables, errors = fetch_endpoints(
        endpoints=endpoint_dicts,
        token=token,
        flatten=req.flatten,
        max_endpoints=5,
        use_cache=True,
        progress_hook=None,
        existing_tables=existing,
    )

    # ✅ keep in-memory store updated per workspace
    TABLE_STORE.set_tables(ws, tables)

    meta = [
        TableMeta(name=name, rows=len(df), cols=df.shape[1])
        for name, df in tables.items()
    ]

    return FetchResponse(tables=meta, errors=errors)


# -------------------------------------------------------------------
# List tables (from TABLE_STORE only)
# -------------------------------------------------------------------

class TableListResponse(BaseModel):
    tables: List[TableMeta]


@router.get("/list", response_model=TableListResponse)
def list_tables(workspace_id: str = "default"):
    ws = (workspace_id or "default").strip() or "default"
    tables = TABLE_STORE.get_tables(ws)

    meta = [
        TableMeta(name=name, rows=len(df), cols=df.shape[1])
        for name, df in tables.items()
    ]
    return TableListResponse(tables=meta)


# -------------------------------------------------------------------
# Download CSV (from TABLE_STORE)
# -------------------------------------------------------------------

@router.get("/{table_name}/download")
def download_table_csv(
    table_name: str,
    workspace_id: str = "default",
):
    ws = (workspace_id or "default").strip() or "default"
    df = TABLE_STORE.get(ws, table_name)
    if df is None:
        raise HTTPException(status_code=404, detail="Table not found")

    csv_str = df.to_csv(index=False)
    csv_bytes = csv_str.encode("utf-8")

    return StreamingResponse(
        io.BytesIO(csv_bytes),
        media_type="text/csv",
        headers={
            "Content-Disposition": f'attachment; filename="{table_name}.csv"'
        },
    )



# -------------------------------------------------------------------
# Preview table (from TABLE_STORE)
# -------------------------------------------------------------------

class TablePreviewResponse(BaseModel):
    name: str
    rows: List[Dict[str, Any]]


@router.get("/{table_name}/preview", response_model=TablePreviewResponse)
def preview_table(
    table_name: str,
    limit: int = 10,
    workspace_id: str = "default",
):
    ws = (workspace_id or "default").strip() or "default"
    df = TABLE_STORE.get(ws, table_name)
    if df is None:
        raise HTTPException(status_code=404, detail="Table not found")

    preview_df = df.head(limit)
    rows = preview_df.to_dict(orient="records")
    return TablePreviewResponse(name=table_name, rows=rows)


# -------------------------------------------------------------------
# Upload CSV → TABLE_STORE only
# -------------------------------------------------------------------

class UploadResponse(BaseModel):
    name: str
    rows: int
    cols: int


@router.post("/upload-csv", response_model=UploadResponse)
async def upload_csv(
    file: UploadFile = File(...),
    workspace_id: str = "default",  # query param
):
    if not file.filename.lower().endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only CSV files are supported.")

    content = await file.read()
    try:
        df = pd.read_csv(io.StringIO(content.decode("utf-8")))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to parse CSV: {e}")

    name = file.filename.rsplit(".", 1)[0]

    ws = (workspace_id or "default").strip() or "default"
    tables = TABLE_STORE.get_tables(ws)
    tables[name] = df
    TABLE_STORE.set_tables(ws, tables)

    return UploadResponse(name=name, rows=len(df), cols=df.shape[1])
