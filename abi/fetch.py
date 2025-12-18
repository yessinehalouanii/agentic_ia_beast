# abi/fetch.py (FastAPI-friendly, no Streamlit)

import io
import requests
import pandas as pd
from typing import Optional, List, Dict, Tuple, Callable, Any


# =====================================================================================
# JSON → DataFrame helper (no flattening)
# =====================================================================================

def _json_to_df(obj) -> pd.DataFrame:
    """
    Very small JSON → DataFrame helper.
    Handles:
      - list of objects
      - {"rows": [...]}
      - {"data": [...]}
      - plain dict → single-row DataFrame
    """
    if isinstance(obj, list):
        return pd.DataFrame(obj)
    if isinstance(obj, dict):
        if isinstance(obj.get("rows"), list):
            return pd.DataFrame(obj["rows"])
        if isinstance(obj.get("data"), list):
            return pd.DataFrame(obj["data"])
        return pd.DataFrame([obj])
    # Fall back: wrap in list
    return pd.DataFrame([obj])


# =====================================================================================
# Single-endpoint fetch (no flatten / harmonize)
# =====================================================================================

def fetch_endpoint_as_df(
    full_url: str,
    token: Optional[str],
    flatten: bool = True,
) -> pd.DataFrame:
    """
    Single fetch (no pagination, no flattening).
    - Adds Bearer token if provided.
    - If Content-Type is JSON → parse via _json_to_df
    - Otherwise → try CSV via pandas
    NOTE: `flatten` is accepted for compatibility but ignored.
    """
    headers = {"Authorization": f"Bearer {token}"} if token else {}

    try:
        resp = requests.get(full_url, headers=headers, timeout=500)
    except requests.RequestException as e:
        raise RuntimeError(f"Fetch failed: {e}")

    if resp.status_code == 401:
        raise RuntimeError("401 Unauthorized — the token is missing/expired/invalid.")
    if resp.status_code >= 400:
        raise RuntimeError(f"Fetch failed: {resp.status_code} {resp.text}")

    ct = (resp.headers.get("Content-Type") or "").lower()

    # JSON
    if "application/json" in ct or ct.endswith("+json"):
        try:
            obj = resp.json()
        except Exception:
            raise RuntimeError("Response advertised JSON but parsing failed.")
        return _json_to_df(obj)

    # Otherwise: try CSV
    try:
        return pd.read_csv(io.StringIO(resp.text))
    except Exception as e:
        raise RuntimeError(f"Could not parse response as CSV: {e}")


# =====================================================================================
# Cached wrapper (FastAPI version → simple alias, no real cache)
# =====================================================================================

def cached_fetch_endpoint_as_df(
    full_url: str,
    token: Optional[str],
    flatten: bool = True,
) -> pd.DataFrame:
    """
    Simple wrapper kept for API compatibility with old code.
    No real caching here (FastAPI version).
    """
    return fetch_endpoint_as_df(full_url, token, flatten)


# =====================================================================================
# Helper to attach limit param to URL
# =====================================================================================

def _attach_limit_param(url: str, limit: Optional[int]) -> str:
    """
    If a positive limit is provided, append it as a `limit` query parameter.
    Otherwise return the URL unchanged.
    """
    if limit is None:
        return url

    try:
        limit_int = int(limit)
    except (TypeError, ValueError):
        return url

    if limit_int <= 0:
        return url

    sep = "&" if "?" in url else "?"
    return f"{url}{sep}limit={limit_int}"


# =====================================================================================
# Helper: guess a primary key column for incremental merge
# =====================================================================================

_PK_CANDIDATES = [
    "invoice_id",
    "customer_id",
    "client_id",
    "cust_id",
    "internal_id",
    "order_id",
    "sale_id",
    "visit_id",
]

def _guess_pk(df: pd.DataFrame) -> Optional[str]:
    """
    Try to guess a primary key column to deduplicate rows.
    """
    if df is None or df.empty:
        return None
    cols = set(df.columns)
    for c in _PK_CANDIDATES:
        if c in cols:
            return c
    return None


# =====================================================================================
# Nested JSON helpers (for dotted paths)
# =====================================================================================

def _is_dict_like(x) -> bool:
    return isinstance(x, dict)

def _is_list_of_dicts(x) -> bool:
    return isinstance(x, list) and any(isinstance(i, dict) for i in (x or []))

def _iter_dotted_paths(value, base: str = "", max_depth: int = 3):
    if max_depth <= 0 or value is None:
        return
    if isinstance(value, dict):
        for k, v in value.items():
            path = f"{base}.{k}" if base else k
            yield path
            yield from _iter_dotted_paths(v, path, max_depth - 1)
    elif isinstance(value, list):
        first = next((i for i in value if isinstance(i, (dict, list))), None)
        if first is not None:
            yield from _iter_dotted_paths(first, base, max_depth - 1)

def detect_nested_schema(df: pd.DataFrame, sample_rows: int = 200) -> list[str]:
    """
    Try to detect nested JSON-like paths inside DataFrame columns.
    Returns a list of dotted paths (e.g. 'customer.address.city').
    """
    if df is None or df.empty:
        return []
    sample = df.head(sample_rows)
    dotted = set()
    for col in df.columns:
        for v in sample[col]:
            if _is_dict_like(v) or _is_list_of_dicts(v):
                for path in _iter_dotted_paths(v, base="", max_depth=3):
                    dotted.add(
                        f"{col}.{path}" if not path.startswith(f"{col}.") else path
                    )
                break
    return sorted(dotted)

def _read_nested(obj, path_tokens: list[str]):
    cur = obj
    for tok in path_tokens:
        if cur is None:
            return None
        if isinstance(cur, list):
            cur = next((i for i in cur if isinstance(i, (dict, list))), None)
            if cur is None:
                return None
        if isinstance(cur, dict):
            cur = cur.get(tok, None)
        else:
            return None
    return cur

def get_or_make_dotted_series(df: pd.DataFrame, dotted: str) -> str | None:
    """
    Given a dotted path (e.g. 'customer.address.city'), create a new column
    extracting that nested value from a dict/list column.

    Returns the name of the new column, or None on failure.
    """
    if not dotted or "." not in dotted:
        return None
    parts = dotted.split(".")
    head, tail = parts[0], parts[1:]
    if head not in df.columns:
        return None
    new_col = "__nested__" + "__".join(parts)
    if new_col in df.columns:
        return new_col
    try:
        df[new_col] = df[head].apply(lambda x: _read_nested(x, tail))
        return new_col
    except Exception:
        return None


# =====================================================================================
# Batch fetcher — simplified but API-compatible + incremental merge
# =====================================================================================

def fetch_endpoints(
    endpoints: List[Dict[str, Any]],
    token: Optional[str],
    flatten: bool = True,
    max_endpoints: int = 5,
    use_cache: bool = True,
    progress_hook: Optional[Callable[[Dict], None]] = None,
    existing_tables: Optional[Dict[str, pd.DataFrame]] = None,
) -> Tuple[Dict[str, pd.DataFrame], List[str]]:
    """
    Batch fetcher — simplified:
    - One request per endpoint (no pagination).
    - "use_cache" just controls whether we go through cached_fetch_endpoint_as_df.
    - Optionally calls progress_hook with:
        {"label": name, "event": "endpoint_done", "rows": int}
        {"label": name, "event": "error", "error": str}
    - If existing_tables is provided and a table with the same name exists,
      we do an incremental merge:
        old + new → drop_duplicates on a guessed primary key.
      This does NOT reduce what the API sends, but keeps your local table
      as "old rows + new unique rows".
    Each endpoint dict can optionally include:
      - "limit": int  -> sent as ?limit=<int> in the URL and applied as df.head(limit)
    """
    out: Dict[str, pd.DataFrame] = {}
    errs: List[str] = []

    if not endpoints:
        return out, errs

    selected = endpoints[:max_endpoints]

    for item in selected:
        name = item.get("name") or ""
        base_url = item.get("url") or ""
        if not (name and base_url):
            errs.append("(missing) Skipped one endpoint with empty name or url")
            continue

        # Read and normalize optional limit
        raw_limit = item.get("limit", None)
        try:
            if raw_limit is not None:
                limit: Optional[int] = int(raw_limit)
                if limit <= 0:
                    limit = None
            else:
                limit = None
        except Exception:
            limit = None

        # Build URL with ?limit= if provided
        url = _attach_limit_param(base_url, limit)

        try:
            if use_cache:
                df_new = cached_fetch_endpoint_as_df(url, token, flatten=flatten)
            else:
                df_new = fetch_endpoint_as_df(url, token, flatten=flatten)

            if not isinstance(df_new, pd.DataFrame):
                raise RuntimeError("Parsed result is not a DataFrame.")

            # Clamp rows client-side if limit is set
            if limit is not None:
                df_new = df_new.head(limit)

            # 🔁 Incremental merge with existing local table if present
            df_existing = None
            if existing_tables and name in existing_tables:
                df_existing = existing_tables[name]

            if df_existing is not None and not df_existing.empty and not df_new.empty:
                # Try to guess a primary key on the NEW data first
                pk = _guess_pk(df_new) or _guess_pk(df_existing)
                if pk and pk in df_new.columns and pk in df_existing.columns:
                    combined = pd.concat(
                        [df_existing, df_new],
                        ignore_index=True,
                        sort=False,
                    )
                    combined = combined.drop_duplicates(subset=[pk], keep="last")
                    df_final = combined
                else:
                    # No PK found → just concatenate (may duplicate)
                    df_final = pd.concat(
                        [df_existing, df_new],
                        ignore_index=True,
                        sort=False,
                    )
            else:
                df_final = df_new

            out[name] = df_final

            if progress_hook:
                progress_hook({
                    "label": name,
                    "event": "endpoint_done",
                    "rows": len(df_final),
                })

        except Exception as e:
            msg = f"{name}: {e}"
            errs.append(msg)
            if progress_hook:
                progress_hook({
                    "label": name,
                    "event": "error",
                    "error": str(e),
                })

    return out, errs
