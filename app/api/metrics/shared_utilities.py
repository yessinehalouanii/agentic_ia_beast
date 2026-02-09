# app/api/metrics/shared_utilities.py
from __future__ import annotations

from datetime import datetime, timezone, timedelta
from typing import Any, Dict, List, Optional, Tuple

from fastapi import HTTPException

from abi.runtime import to_json_safe


# -------------------------------------------------------------------
# ✅ Shared response helpers (use from metrics modules)
# -------------------------------------------------------------------

def _es_cannot_answer(message: str, business_rules: Optional[str]) -> Dict[str, Any]:
    """
    Standard ES "cannot answer" payload used by metrics modules.
    """
    return {
        "insight": to_json_safe(message),
        "rows": [],
        "rules_used": business_rules or "",
        "engine": "es",
    }


# Backwards-compatible alias (if some older code still imports it)
def _es_cannot_answer_payload(message: str, business_rules: Optional[str]) -> Dict[str, Any]:
    return _es_cannot_answer(message, business_rules)


# -------------------------------------------------------------------
# ✅ Shared date window helper (move OUT of metrics_promos_coupons.py)
# -------------------------------------------------------------------

DEFAULT_WINDOW_DAYS = 365


def _date_filters_or_default(req: Any, date_field: str) -> Tuple[List[Dict[str, Any]], str]:
    """
    Build ES filter list based on req.start_date/end_date.
    If none provided, apply a default last DEFAULT_WINDOW_DAYS window.
    Returns: (filters, window_label)
    """
    filters: List[Dict[str, Any]] = []

    start = getattr(req, "start_date", None)
    end = getattr(req, "end_date", None)

    if start or end:
        range_body: Dict[str, Any] = {}
        if start:
            range_body["gte"] = start
        if end:
            range_body["lte"] = end

        filters.append({"range": {date_field: range_body}})
        window_label = f"{range_body.get('gte','')} → {range_body.get('lte','')}".strip()
        return filters, window_label or "custom window"

    today = datetime.now(timezone.utc).date()
    start_d = today - timedelta(days=DEFAULT_WINDOW_DAYS)
    filters.append({"range": {date_field: {"gte": start_d.isoformat(), "lte": today.isoformat()}}})
    return filters, f"last {DEFAULT_WINDOW_DAYS} days"


# -------------------------------------------------------------------
# Optional: customers context loader (reusable in metrics)
# -------------------------------------------------------------------

def _load_customers_ctx(
    req: Any,
    client: Any,
    business_rules: Optional[str] = None,
    *,
    existing_mappings: Optional[Dict[str, Any]] = None,
    existing_index: Optional[str] = None,
) -> Tuple[Optional[str], Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
    """
    Resolve customers index + mappings (alias-safe).

    Reads customers index from: req.es_customers_index_name

    Returns:
      (customers_index, {"properties": ...}, None) on success
      (None, None, error_payload_dict) on failure
    """
    customers_index = (getattr(req, "es_customers_index_name", None) or "").strip()
    if not customers_index:
        return (
            None,
            None,
            _es_cannot_answer(
                "Cannot compute this metric because 'es_customers_index_name' is not set "
                "(customers index/alias is required).",
                business_rules,
            ),
        )

    try:
        idx, cust_mappings = _get_customers_index_and_mappings(
            client,
            customers_index,
            existing_mappings=existing_mappings,
            existing_index=existing_index or customers_index,
        )
        return idx, cust_mappings, None
    except Exception as e:
        return (
            None,
            None,
            _es_cannot_answer(
                f"Could not fetch Elasticsearch mappings for customers index '{customers_index}': {e}",
                business_rules,
            ),
        )


# -------------------------------------------------------------------
# Mapping helpers
# -------------------------------------------------------------------

def _extract_properties_from_mapping(mapping: Dict[str, Any], index_name: str) -> Dict[str, Any]:
    """
    mapping = client.indices.get_mapping(index=...)
    returns properties dict safely.
    """
    idx_obj = mapping.get(index_name) or next(iter(mapping.values()), {})
    mappings_obj = (idx_obj or {}).get("mappings", {}) or {}
    properties = (mappings_obj or {}).get("properties", {}) or {}
    return properties


def _field_exists(mappings: Dict[str, Any], dotted: str) -> bool:
    """
    True only if 'dotted' exists in mappings (supports multi-fields like *.keyword).
    Expected mappings shape: {"properties": {...}}.
    """
    if not dotted:
        return False

    node: Any = mappings or {}
    parts = dotted.split(".")

    for part in parts:
        if not isinstance(node, dict):
            return False

        props = node.get("properties") or {}
        if part in props:
            node = props[part] or {}
            continue

        fields = node.get("fields") or {}
        if part in fields:
            node = fields[part] or {}
            continue

        return False

    return True


# -------------------------------------------------------------------
# ES-safe helpers
# -------------------------------------------------------------------

def _safe_es_search(client: Any, *, index: str, body: Dict[str, Any]) -> Dict[str, Any]:
    """
    Safety defaults to avoid long-running queries killing the cluster.
    - timeout: ES-side
    - track_total_hits: off (we don't need exact hits)
    - request_timeout: client-side
    """
    body = dict(body or {})
    body.setdefault("timeout", "10s")
    body.setdefault("track_total_hits", False)
    return client.search(index=index, body=body, request_timeout=15)


def _get_req_int(req: Any, name: str, default: int, *, min_v: int, max_v: int) -> int:
    v = getattr(req, name, default)
    try:
        v = int(v)
    except Exception:
        v = default
    return max(min_v, min(int(v), max_v))


# -------------------------------------------------------------------
# ✅ Shared: resolve invoice index + mappings, with "reuse if already fetched"
# -------------------------------------------------------------------

def _get_invoice_index_and_mappings(
    client: Any,
    es_index_name: str,
    *,
    existing_mappings: Optional[Dict[str, Any]] = None,
    existing_index: Optional[str] = None,
) -> Tuple[str, Dict[str, Any]]:
    """
    Resolve the concrete invoices index (handles aliases/wildcards) and return its mappings.

    ✅ If `existing_mappings` already look like {"properties": {...}} and non-empty,
       we reuse them and DO NOT fetch mappings again.

    Returns: (invoice_index_name, {"properties": ...})
    """
    if isinstance(existing_mappings, dict):
        props = existing_mappings.get("properties")
        if isinstance(props, dict) and props:
            return (existing_index or es_index_name), existing_mappings

    invoice_index, invoice_mapping = _select_invoice_index_from_es_mapping(client, es_index_name)
    props = _extract_properties_from_mapping(invoice_mapping, invoice_index)
    return invoice_index, {"properties": props}


def _select_invoice_index_from_es_mapping(client: Any, raw_index_name: str) -> Tuple[str, Dict[str, Any]]:
    raw_index_name = (raw_index_name or "").strip()
    if not raw_index_name:
        raise HTTPException(status_code=400, detail="No es_index_name provided for invoice metrics.")

    try:
        full_mapping = client.indices.get_mapping(index=raw_index_name)
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Could not fetch Elasticsearch mappings for '{raw_index_name}': {e}",
        )

    index_names = sorted(full_mapping.keys())
    if not index_names:
        raise HTTPException(status_code=400, detail=f"No indices found for pattern '{raw_index_name}'.")

    if len(index_names) == 1:
        chosen = index_names[0]
        return chosen, {chosen: full_mapping[chosen]}

    invoice_like = [name for name in index_names if "invoice" in name.lower() or "invoices" in name.lower()]
    chosen = sorted(invoice_like)[0] if invoice_like else index_names[0]

    return chosen, {chosen: full_mapping[chosen]}


# -------------------------------------------------------------------
# ✅ Shared: resolve customers index + mappings, with "reuse if already fetched"
# -------------------------------------------------------------------

def _get_customers_index_and_mappings(
    client: Any,
    customers_index: str,
    *,
    existing_mappings: Optional[Dict[str, Any]] = None,
    existing_index: Optional[str] = None,
) -> Tuple[str, Dict[str, Any]]:
    """
    Load customers mappings (alias-safe).

    ✅ If existing_mappings already look like {"properties": {...}} and non-empty,
       reuse them and DO NOT fetch again.

    Returns: (index_name_to_query, {"properties": ...})
    """
    if isinstance(existing_mappings, dict):
        props = existing_mappings.get("properties")
        if isinstance(props, dict) and props:
            return (existing_index or customers_index), existing_mappings

    raw = client.indices.get_mapping(index=customers_index)
    props = _extract_properties_from_mapping(raw, customers_index)
    return customers_index, {"properties": props}
