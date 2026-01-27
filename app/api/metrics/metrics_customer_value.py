from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict, List, Optional

from abi.runtime import to_json_safe
from routes.es_test import _extract_properties_from_mapping

from app.api.docs_analytics_routes import (
    _build_date_range_filter,
    _es_cannot_answer,
    _select_invoice_index_from_es_mapping,
)
from app.api.metrics.shared_utilities import (
    _field_exists,
    _safe_es_search,
)

# -------------------------------------------------------------------
# Core visit metrics (lifetime) - customers rollup index
# -------------------------------------------------------------------

def _es_core_visit_metrics(
    req,
    client,
    mappings: Dict[str, Any],  # ✅ added for compatibility with router (unused here)
    business_rules: Optional[str],
):
    """
    Core visit KPIs for the dashboard (LIFETIME / ALL HISTORY)
    ✅ Customers index ONLY (fast)
    """
    cust_index = (getattr(req, "es_customers_index_name", None) or "").strip()
    if not cust_index:
        return _es_cannot_answer(
            "Cannot compute core lifetime metrics: es_customers_index_name is not configured on the request.",
            business_rules,
        )

    # fetch customers mappings
    try:
        full_mapping = client.indices.get_mapping(index=cust_index)
    except Exception as e:
        return _es_cannot_answer(
            f"Cannot compute core lifetime metrics: failed to fetch mappings for customers index '{cust_index}': {e}",
            business_rules,
        )

    index_names = sorted(full_mapping.keys())
    if not index_names:
        return _es_cannot_answer(
            f"Cannot compute core lifetime metrics: no indices found for pattern '{cust_index}'.",
            business_rules,
        )

    chosen = index_names[0]
    cust_props = ((full_mapping[chosen].get("mappings") or {}).get("properties")) or {}
    cust_mappings = {"properties": cust_props}

    # required rollup fields in your customers mapping
    required = ["customer_id", "visits_lifetime", "sales_pickup_lifetime"]
    missing = [f for f in required if not _field_exists(cust_mappings, f)]
    if missing:
        return _es_cannot_answer(
            "Cannot compute core lifetime metrics from customers index because required fields "
            f"are missing from mappings: {', '.join(missing)}. "
            "Expected fields: customer_id, visits_lifetime, sales_pickup_lifetime.",
            business_rules,
        )

    body: Dict[str, Any] = {
        "size": 0,
        "aggs": {
            "unique_customers": {"value_count": {"field": "customer_id"}},
            "total_visits": {"sum": {"field": "visits_lifetime"}},
            "total_visit_amount": {"sum": {"field": "sales_pickup_lifetime"}},
        },
    }

    try:
        res = _safe_es_search(client, index=chosen, body=body)
    except Exception as e:
        return _es_cannot_answer(
            f"Error executing customers-index aggregation on '{chosen}': {e}",
            business_rules,
        )

    agg = (res.get("aggregations") or {})

    unique_customers = int((agg.get("unique_customers") or {}).get("value") or 0)
    total_visits = float((agg.get("total_visits") or {}).get("value") or 0.0)
    total_visit_amount = float((agg.get("total_visit_amount") or {}).get("value") or 0.0)

    rows: List[Dict[str, Any]] = [
        {"metric": "total_visit_amount", "label": "Total Visit Amount", "value": total_visit_amount},
        {"metric": "total_visits", "label": "Total Visits", "value": total_visits},
        {"metric": "unique_customers", "label": "Unique Customers", "value": unique_customers},
    ]

    insight = (
        f"Core lifetime metrics were computed from customers rollups on index '{chosen}' "
        f"using sum(visits_lifetime) and sum(sales_pickup_lifetime)."
    )

    return {
        "insight": to_json_safe(insight),
        "rows": to_json_safe(rows),
        "rules_used": business_rules or "",
        "engine": "es",
    }


# -------------------------------------------------------------------
# Windowed customer-value metrics (per period) - invoices index
# -------------------------------------------------------------------

def _es_customer_value_metrics(
    req,
    client,
    mappings: Dict[str, Any],  # ✅ added for compatibility (unused here)
    business_rules: Optional[str],
):
    """
    Windowed customer value metrics for the selected period (invoices index).

    ✅ Uses direct mapping fields (no resolve_es_field):
      - customer_id, dropoff_at, total, pieces, visit_id
    """
    invoice_index, invoice_mapping = _select_invoice_index_from_es_mapping(client, req.es_index_name)
    index_name = invoice_index

    properties = _extract_properties_from_mapping(invoice_mapping, invoice_index)
    invoice_mappings = {"properties": properties}

    customer_field = "customer_id"
    date_field = "dropoff_at"
    amount_field = "total"

    pieces_field = "pieces" if _field_exists(invoice_mappings, "pieces") else None
    visit_field = "visit_id" if _field_exists(invoice_mappings, "visit_id") else None

    if not (
        _field_exists(invoice_mappings, customer_field)
        and _field_exists(invoice_mappings, date_field)
        and _field_exists(invoice_mappings, amount_field)
    ):
        return _es_cannot_answer(
            "Cannot compute customer value metrics because one of the required fields "
            "is missing from the invoices index mapping: customer_id, dropoff_at, total.",
            business_rules,
        )

    filters = _build_date_range_filter(req, date_field)

    aggs: Dict[str, Any] = {
        "unique_customers": {"cardinality": {"field": customer_field}},
        "total_revenue": {"sum": {"field": amount_field}},
    }

    if pieces_field:
        aggs["total_pieces"] = {"sum": {"field": pieces_field}}

    if visit_field:
        aggs["total_visits"] = {"cardinality": {"field": visit_field}}
    else:
        aggs["total_visits"] = {"value_count": {"field": date_field}}

    body: Dict[str, Any] = {"size": 0, "aggs": aggs}
    if filters:
        body["query"] = {"bool": {"filter": filters}}

    try:
        res = _safe_es_search(client, index=index_name, body=body)
    except Exception as e:
        return _es_cannot_answer(
            f"Error executing invoices-index aggregation on '{index_name}': {e}",
            business_rules,
        )

    agg = res.get("aggregations", {}) or {}

    unique_customers = int((agg.get("unique_customers") or {}).get("value") or 0)
    total_revenue = float((agg.get("total_revenue") or {}).get("value") or 0.0)

    total_pieces = float((agg.get("total_pieces") or {}).get("value") or 0.0) if "total_pieces" in agg else 0.0
    total_visits = int((agg.get("total_visits") or {}).get("value") or 0)

    if unique_customers > 0:
        avg_visits_per_customer = total_visits / float(unique_customers)
        revenue_per_customer = total_revenue / float(unique_customers)
        pieces_per_customer = total_pieces / float(unique_customers)
    else:
        avg_visits_per_customer = 0.0
        revenue_per_customer = 0.0
        pieces_per_customer = 0.0

    avg_dollar_per_piece = (total_revenue / total_pieces) if total_pieces > 0 else 0.0

    rows: List[Dict[str, Any]] = [
        {"metric": "average_visits_per_customer", "label": "Average Visits per Customer", "value": avg_visits_per_customer},
        {"metric": "visit_pieces_per_customer", "label": "Visit Pieces per Customer", "value": pieces_per_customer},
        {"metric": "revenue_per_customer", "label": "Revenue Per Customer", "value": revenue_per_customer},
        {"metric": "avg_dollar_per_piece", "label": "Avg $ per Piece", "value": avg_dollar_per_piece},
        {"metric": "total_visits", "label": "Total Visits (window)", "value": float(total_visits)},
        {"metric": "unique_customers", "label": "Unique Customers (window)", "value": float(unique_customers)},
        {"metric": "total_revenue", "label": "Total Revenue (window)", "value": total_revenue},
        {"metric": "total_pieces", "label": "Total Pieces (window)", "value": total_pieces},
    ]

    if getattr(req, "start_date", None) or getattr(req, "end_date", None):
        parts: List[str] = []
        if getattr(req, "start_date", None):
            parts.append(f"from {req.start_date}")
        if getattr(req, "end_date", None):
            parts.append(f"to {req.end_date}")
        window_str = " ".join(parts)
    else:
        window_str = "for all available history"

    insight = (
        f"Customer value metrics were computed on index '{index_name}' {window_str}, "
        f"using '{date_field}' as the visit date, '{customer_field}' as the customer id, "
        f"and '{amount_field}' as the invoice total. A visit is treated as a distinct "
        f"visit_id when available, otherwise invoice rows are used as a proxy."
    )

    return {
        "insight": to_json_safe(insight),
        "rows": to_json_safe(rows),
        "rules_used": business_rules or "",
        "engine": "es",
    }


def _es_one_time_vs_repeat(
    req,
    client,
    mappings: Dict[str, Any],  # ✅ added for compatibility (unused here)
    business_rules: Optional[str],
):
    cust_index = (getattr(req, "es_customers_index_name", None) or "").strip()
    if not cust_index:
        return _es_cannot_answer(
            "Cannot compute one-time vs repeat: es_customers_index_name is not configured.",
            business_rules,
        )

    basis = (getattr(req, "repeat_basis", None) or "lifetime").lower()
    visits_field = "visits_365" if basis in ("365", "1y", "year", "visits_365") else "visits_lifetime"

    try:
        full_mapping = client.indices.get_mapping(index=cust_index)
    except Exception as e:
        return _es_cannot_answer(
            f"Cannot compute one-time vs repeat: failed to fetch mappings for '{cust_index}': {e}",
            business_rules,
        )

    index_names = sorted(full_mapping.keys())
    if not index_names:
        return _es_cannot_answer(
            f"Cannot compute one-time vs repeat: no indices found for pattern '{cust_index}'.",
            business_rules,
        )

    chosen = index_names[0]
    cust_props = ((full_mapping[chosen].get("mappings") or {}).get("properties")) or {}
    cust_mappings = {"properties": cust_props}

    if not _field_exists(cust_mappings, visits_field):
        return _es_cannot_answer(
            f"Cannot compute one-time vs repeat: '{visits_field}' is missing in customers mapping.",
            business_rules,
        )

    filters: List[Dict[str, Any]] = []
    company_id = getattr(req, "company_id", None)
    if company_id is not None:
        filters.append({"term": {"company_id": int(company_id)}})

    filters.append({"range": {visits_field: {"gte": 1}}})

    body: Dict[str, Any] = {
        "size": 0,
        "request_cache": True,
        "aggs": {
            "segments": {
                "filters": {
                    "filters": {
                        "one_time": {"term": {visits_field: 1}},
                        "repeat": {"range": {visits_field: {"gte": 2}}},
                    }
                }
            }
        },
    }
    if filters:
        body["query"] = {"bool": {"filter": filters}}

    try:
        res = _safe_es_search(client, index=chosen, body=body)
    except Exception as e:
        return _es_cannot_answer(
            f"Error executing customers-index aggregation on '{chosen}': {e}",
            business_rules,
        )

    seg = ((res.get("aggregations") or {}).get("segments")) or {}
    buckets = seg.get("buckets") or {}

    one_time = int((buckets.get("one_time") or {}).get("doc_count") or 0)
    repeat = int((buckets.get("repeat") or {}).get("doc_count") or 0)
    total = one_time + repeat

    if total == 0:
        return {
            "insight": to_json_safe("No customers with at least one visit were found."),
            "rows": [],
            "rules_used": business_rules or "",
            "engine": "es",
        }

    one_pct = one_time * 100.0 / total
    rep_pct = repeat * 100.0 / total

    rows = [
        {"segment": "one-time", "customer_count": one_time, "percentage_of_customers": one_pct},
        {"segment": "repeat", "customer_count": repeat, "percentage_of_customers": rep_pct},
    ]

    insight = (
        f"One-time vs repeat was computed from customers rollups on '{chosen}' "
        f"using '{visits_field}'. Out of {total} customers with >=1 visit, "
        f"{one_pct:.1f}% are one-time and {rep_pct:.1f}% are repeat."
    )

    return {
        "insight": to_json_safe(insight),
        "rows": to_json_safe(rows),
        "rules_used": business_rules or "",
        "engine": "es",
    }


def _window_customer_value_metrics(
    req,
    period,
    client,
    mappings: Dict[str, Any],
) -> Dict[str, Optional[float]]:
    """
    Dashboard adapter:
    - injects period.start_date/end_date into a copy of req
    - calls _es_customer_value_metrics()
    - returns {metric_id -> value} mapping
    """
    req2 = deepcopy(req)
    req2.start_date = getattr(period, "start_date", None)
    req2.end_date = getattr(period, "end_date", None)

    resp = _es_customer_value_metrics(req2, client, mappings, business_rules=None)

    out: Dict[str, Optional[float]] = {}
    rows = resp.get("rows") or []
    for r in rows:
        metric_id = r.get("metric") or r.get("id")
        if not metric_id:
            continue
        v = r.get("value")
        try:
            out[metric_id] = float(v) if v is not None else None
        except Exception:
            out[metric_id] = None
    return out


__all__ = [
    "_es_core_visit_metrics",
    "_es_customer_value_metrics",
    "_es_one_time_vs_repeat",
    "_window_customer_value_metrics",
]
