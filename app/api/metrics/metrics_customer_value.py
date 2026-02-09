from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict, List, Optional, Tuple

from abi.runtime import to_json_safe

from app.api.docs_analytics_routes import (
    _build_date_range_filter,
    _es_cannot_answer,
)
from app.api.metrics.shared_utilities import (
    _field_exists,
    _safe_es_search,
    _load_customers_ctx,         
    _get_invoice_index_and_mappings,
)

# -------------------------------------------------------------------
# ✅ Shared helper: invoice mapping / index selection (NO DUPLICATION)
# -------------------------------------------------------------------


def _load_invoice_index_mappings(
    req,
    client,
    mappings: Optional[Dict[str, Any]],
) -> Tuple[str, Dict[str, Any]]:
    """
    Returns (invoice_index_name, invoice_mappings).

    ✅ Reuses provided invoice mappings (dashboard path) and avoids refetching.
    ✅ Otherwise resolves the concrete invoices index (alias/wildcard-safe) and fetches once.
    """
    index_name = (getattr(req, "es_index_name", None) or "").strip()
    if not index_name:
        raise ValueError("No es_index_name provided for invoice metrics.")

    invoice_index, invoice_mappings = _get_invoice_index_and_mappings(
        client,
        index_name,
        existing_mappings=mappings,
        existing_index=index_name,
    )
    return invoice_index, invoice_mappings


# -------------------------------------------------------------------
# Core visit metrics (lifetime) - customers rollup index
# -------------------------------------------------------------------


def _es_core_visit_metrics(
    req,
    client,
    mappings: Dict[str, Any],  # kept for router compatibility (not used)
    business_rules: Optional[str],
):
    """
    Core visit KPIs for the dashboard (LIFETIME / ALL HISTORY)
    ✅ Customers index ONLY (fast)
    """
    customers_index, cust_mappings, err = _load_customers_ctx(req, client, business_rules)
    if err:
        return err

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
            # NOTE: value_count assumes 1 doc per customer_id in this rollup index.
            "unique_customers": {"value_count": {"field": "customer_id"}},
            "total_visits": {"sum": {"field": "visits_lifetime"}},
            "total_visit_amount": {"sum": {"field": "sales_pickup_lifetime"}},
        },
    }

    try:
        res = _safe_es_search(client, index=customers_index, body=body)
    except Exception as e:
        return _es_cannot_answer(
            f"Error executing customers-index aggregation on '{customers_index}': {e}",
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
        f"Core lifetime metrics were computed from customers rollups on index '{customers_index}' "
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
    mappings: Dict[str, Any],  # ✅ reused (dashboard passes invoice mappings)
    business_rules: Optional[str],
):
    """
    Windowed customer value metrics for the selected period (invoices index).

    ✅ Requires visit_id to compute total_visits (NO fallback).
    """
    try:
        index_name, invoice_mappings = _load_invoice_index_mappings(req, client, mappings)
    except Exception as e:
        return _es_cannot_answer(str(e), business_rules)

    customer_field = "customer_id"
    date_field = "dropoff_at"
    amount_field = "total"
    visit_field = "visit_id"

    pieces_field = "pieces" if _field_exists(invoice_mappings, "pieces") else None

    # ✅ Require visit_id now (no fallback)
    required = [customer_field, date_field, amount_field, visit_field]
    missing = [f for f in required if not _field_exists(invoice_mappings, f)]
    if missing:
        return _es_cannot_answer(
            "Cannot compute customer value metrics because required fields are missing "
            f"from the invoices index mapping: {', '.join(missing)}. "
            "Expected fields: customer_id, dropoff_at, total, visit_id.",
            business_rules,
        )

    filters = _build_date_range_filter(req, date_field)

    aggs: Dict[str, Any] = {
        "unique_customers": {"cardinality": {"field": customer_field}},
        "total_revenue": {"sum": {"field": amount_field}},
        "total_visits": {"cardinality": {"field": visit_field}},  # ✅ NO fallback
    }

    if pieces_field:
        aggs["total_pieces"] = {"sum": {"field": pieces_field}}

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
    total_pieces = (
        float((agg.get("total_pieces") or {}).get("value") or 0.0) if "total_pieces" in agg else 0.0
    )
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
        f"'{visit_field}'."
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
    mappings: Dict[str, Any],  # kept for compatibility (unused)
    business_rules: Optional[str],
):
    """
    One-time vs repeat computed from customers rollups.
    """
    customers_index, cust_mappings, err = _load_customers_ctx(req, client, business_rules)
    if err:
        return err

    basis = (getattr(req, "repeat_basis", None) or "lifetime").lower()
    visits_field = "visits_365" if basis in ("365", "1y", "year", "visits_365") else "visits_lifetime"

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
        res = _safe_es_search(client, index=customers_index, body=body)
    except Exception as e:
        return _es_cannot_answer(
            f"Error executing customers-index aggregation on '{customers_index}': {e}",
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
        f"One-time vs repeat was computed from customers rollups on '{customers_index}' "
        f"using '{visits_field}'. Out of {total} customers with >=1 visit, "
        f"{one_pct:.1f}% are one-time and {rep_pct:.1f}% are repeat."
    )

    return {
        "insight": to_json_safe(insight),
        "rows": to_json_safe(rows),
        "rules_used": business_rules or "",
        "engine": "es",
    }


def _es_new_customers_first_visit_in_period_value_count(
    req,
    client,
    mappings: Dict[str, Any],  # unused (kept for signature compatibility)
    business_rules: Optional[str],
):
    """
    KPI: New Customers (First Visit in Window)
    Definition: count customer docs whose customers.first_visit is within [start_date, end_date].
    Uses value_count(customer_id) assuming 1 doc per customer.
    """
    customers_index, cust_mappings, err = _load_customers_ctx(req, client, business_rules)
    if err:
        return err

    # Use first_visit (your mapping has it)
    if not _field_exists(cust_mappings, "first_visit"):
        return _es_cannot_answer(
            "Cannot compute New Customers (first visit in window): customers.first_visit is missing.",
            business_rules,
        )

    date_field = "first_visit"

    # Dashboard must provide a window
    if not getattr(req, "start_date", None) and not getattr(req, "end_date", None):
        return _es_cannot_answer(
            "New Customers (first visit) requires start_date/end_date on the request.",
            business_rules,
        )

    filters: List[Dict[str, Any]] = []
    filters.extend(_build_date_range_filter(req, date_field))
    filters.append({"exists": {"field": date_field}})
    filters.append({"exists": {"field": "customer_id"}})

    # Optional company filter (only if you pass it)
    company_id = getattr(req, "company_id", None)
    if company_id is not None:
        try:
            filters.append({"term": {"company_id": int(company_id)}})
        except Exception:
            pass

    body: Dict[str, Any] = {
        "size": 0,
        "query": {"bool": {"filter": filters}},
        "aggs": {
            "new_customers": {"value_count": {"field": "customer_id"}}
        },
    }

    try:
        res = _safe_es_search(client, index=customers_index, body=body)
    except Exception as e:
        return _es_cannot_answer(
            f"Error executing customers-index aggregation on '{customers_index}': {e}",
            business_rules,
        )

    agg = (res.get("aggregations") or {})
    new_customers = int((agg.get("new_customers") or {}).get("value") or 0)

    return {
        "insight": to_json_safe(
            f"New Customers computed from '{customers_index}' as value_count(customer_id) "
            f"filtered by customers.{date_field} within the dashboard window."
        ),
        "rows": to_json_safe({"count": new_customers}),
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
    "_es_new_customers_first_visit_in_period_value_count",
]
