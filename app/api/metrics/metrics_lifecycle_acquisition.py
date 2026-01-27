from __future__ import annotations
from datetime import datetime, timezone, timedelta
from typing import Any, Dict, List, Optional, Iterable
from abi.runtime import to_json_safe
from routes.es_test import _extract_properties_from_mapping
from app.api.metrics.metrics_promos_coupons import _date_filters_or_default
from app.api.metrics.shared_utilities import (
    _field_exists,
    _safe_es_search,
)
    
    
from app.api.docs_analytics_routes import (
    ES_MAX_CUSTOMERS_DEFAULT,
    _ms_to_dt,
    _parse_date_str,
    _es_cannot_answer,
    _es_get_customer_stats,
    _es_get_customer_signups,
    _select_invoice_index_from_es_mapping,
)

# -------------------------------------------------------------------
# ✅ mapping-aware field existence helper (prevents silent 0 results)
# -------------------------------------------------------------------


# -------------------------------------------------------------------
# ES-safe helpers
# -------------------------------------------------------------------

def _get_customers_index_and_mappings(client, customers_index: str) -> tuple[str, Dict[str, Any]]:
    """
    Load customers mappings (works even if customers_index is an alias).
    """
    raw = client.indices.get_mapping(index=customers_index)
    props = _extract_properties_from_mapping(raw, customers_index)
    return customers_index, {"properties": props}

def _get_req_int(req, name: str, default: int, *, min_v: int, max_v: int) -> int:
    v = getattr(req, name, default)
    try:
        v = int(v)
    except Exception:
        v = default
    return max(min_v, min(int(v), max_v))


def _get_invoice_index_and_mappings(client, es_index_name: str) -> tuple[str, Dict[str, Any]]:
    """
    ✅ Always resolve the real invoices index (handles aliases) and extract its mapping.
    """
    invoice_index, invoice_mapping = _select_invoice_index_from_es_mapping(client, es_index_name)
    props = _extract_properties_from_mapping(invoice_mapping, invoice_index)
    return invoice_index, {"properties": props}


# -------------------------------------------------------------------
# Stats access (NO rollup)
# -------------------------------------------------------------------

def _get_customer_stats_invoices_only(
    req,
    client,
    invoices_index: str,
    invoices_mappings: Dict[str, Any],
) -> Optional[List[Dict[str, Any]]]:
    """
    Returns per-customer stats computed from invoices.
    Expected shape (from _es_get_customer_stats):
      {
        "customer_id": ...,
        "visit_count": int,
        "first_visit": datetime|None,
        "last_visit": datetime|None,
        "total_revenue": float|None,
        "total_pieces": float|None,
      }
    """
    # ✅ "no surprises": ensure core fields exist in the invoices mapping
    if not _field_exists(invoices_mappings, "customer_id") or not _field_exists(invoices_mappings, "dropoff_at"):
        return None
    # total/pieces/visit_id may be optional for some metrics; we still validate where needed.

    return _es_get_customer_stats(client, invoices_index, invoices_mappings)
