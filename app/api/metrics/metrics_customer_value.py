from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict, List, Optional, Iterable, Tuple
import heapq

from abi.runtime import to_json_safe
from routes.es_test import _extract_properties_from_mapping

from app.api.docs_analytics_routes import (
    _ms_to_dt,
    _parse_date_str,
    _es_cannot_answer,
    _build_date_range_filter,
    _select_invoice_index_from_es_mapping,
    _es_get_customer_stats,
)

# -------------------------------------------------------------------
# ✅ NEW: mapping-aware field existence helper (prevents silent 0 results)
# -------------------------------------------------------------------

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
# ✅ NEW: small coercion helpers (so dashboard shows 0.0 instead of null/—)
# -------------------------------------------------------------------

def _as_float(v: Any, default: float = 0.0) -> float:
    if v is None:
        return default
    # avoid bool -> 1/0 surprises
    if isinstance(v, bool):
        return default
    if isinstance(v, (int, float)):
        return float(v)
    if isinstance(v, str):
        try:
            return float(v.strip())
        except Exception:
            return default
    return default


# -------------------------------------------------------------------
# ES-safe helpers
# -------------------------------------------------------------------

def _safe_es_search(client, *, index: str, body: Dict[str, Any]) -> Dict[str, Any]:
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


# -------------------------------------------------------------------
# Dashboard-safe helpers
# -------------------------------------------------------------------

def _metric_value_from_rows(resp: Dict[str, Any], metric_id: str) -> Optional[float]:
    """Extract value from rows like: [{"metric": "...", "value": ...}, ...]."""
    rows = (resp or {}).get("rows") or []
    for r in rows:
        if isinstance(r, dict) and r.get("metric") == metric_id:
            v = r.get("value")
            # ✅ accept numeric strings too
            if isinstance(v, (int, float)):
                return float(v)
            if isinstance(v, str):
                try:
                    return float(v.strip())
                except Exception:
                    return None
    return None


def _metric_value_from_rows_any(resp: Dict[str, Any], metric_ids: Iterable[str]) -> Optional[float]:
    """
    Try multiple possible metric IDs and return the first non-None value.
    """
    for mid in metric_ids:
        v = _metric_value_from_rows(resp, mid)
        if v is not None:
            return v
    return None


def _sum_field_from_rows(resp: Dict[str, Any], field: str) -> Optional[float]:
    """Sum a numeric field across rows like: [{"period":..., "new_customers": N}, ...]."""
    rows = (resp or {}).get("rows") or []
    total = 0.0
    any_val = False
    for r in rows:
        if not isinstance(r, dict):
            continue
        v = r.get(field)
        if isinstance(v, (int, float)):
            total += float(v)
            any_val = True
        elif isinstance(v, str):
            try:
                total += float(v.strip())
                any_val = True
            except Exception:
                pass
    return total if any_val else None


def _count_rows(resp: Dict[str, Any]) -> float:
    """Count rows (for list-type metrics that return customers)."""
    rows = (resp or {}).get("rows") or []
    return float(len([r for r in rows if isinstance(r, dict)]))


# -------------------------------------------------------------------
# Composite streaming helper for initial_visit_totals (invoices-only)
# -------------------------------------------------------------------

def _iter_customer_visit_composite_buckets(
    client,
    *,
    index: str,
    query_filters: List[Dict[str, Any]],
    customer_field: str,
    visit_field: str,
    date_field: str,
    amount_field: str,
    pieces_field: Optional[str],
    page_size: int = 300,
    max_pages: int = 2000,
) -> Iterable[Dict[str, Any]]:
    """
    Stream buckets using composite aggregation on (customer_id, visit_id).
    """
    after: Optional[Dict[str, Any]] = None
    pages = 0

    while True:
        pages += 1
        if pages > max_pages:
            break

        sources = [
            {"customer": {"terms": {"field": customer_field}}},
            {"visit": {"terms": {"field": visit_field}}},
        ]

        cv_aggs: Dict[str, Any] = {
            "composite": {
                "size": int(page_size),
                "sources": sources,
                **({"after": after} if after else {}),
            },
            "aggs": {
                "first_date": {"min": {"field": date_field}},
                "visit_amount": {"sum": {"field": amount_field}},
            },
        }

        if pieces_field:
            cv_aggs["aggs"]["visit_pieces"] = {"sum": {"field": pieces_field}}

        body: Dict[str, Any] = {"size": 0, "aggs": {"cv": cv_aggs}}
        if query_filters:
            body["query"] = {"bool": {"filter": query_filters}}

        res = _safe_es_search(client, index=index, body=body)
        cv = ((res.get("aggregations") or {}).get("cv")) or {}
        buckets = cv.get("buckets") or []

        if not buckets:
            break

        for b in buckets:
            if isinstance(b, dict):
                yield b

        after = cv.get("after_key")
        if not after:
            break


def _initial_visit_totals_from_invoices_composite(
    req,
    client,
    *,
    invoice_index: str,
    customer_field: str,
    visit_field: str,
    date_field: str,
    amount_field: str,
    pieces_field: Optional[str],
) -> Dict[str, Optional[float]]:
    """
    Compute initial visit totals (amount/pieces) using composite paging.
    """
    start_d = _parse_date_str(getattr(req, "start_date", None))
    end_d = _parse_date_str(getattr(req, "end_date", None))

    query_filters: List[Dict[str, Any]] = []
    if end_d:
        query_filters.append({"range": {date_field: {"lte": f"{end_d.isoformat()}T23:59:59"}}})

    page_size = int(getattr(req, "es_composite_page_size", 300) or 300)
    page_size = max(50, min(page_size, 1000))

    max_pages = int(getattr(req, "es_composite_max_pages", 2000) or 2000)
    max_pages = max(10, min(max_pages, 20000))

    total_amount = 0.0
    total_pieces = 0.0
    any_pieces_values = False

    current_customer = None
    best_dt = None
    best_amount = 0.0
    best_pieces = 0.0

    def flush_current_customer():
        nonlocal total_amount, total_pieces, any_pieces_values, best_dt, best_amount, best_pieces
        if best_dt is None:
            return

        vd = best_dt.date()
        if start_d and vd < start_d:
            return
        if end_d and vd > end_d:
            return

        total_amount += float(best_amount)
        if pieces_field is not None:
            total_pieces += float(best_pieces)
            any_pieces_values = True

    for b in _iter_customer_visit_composite_buckets(
        client,
        index=invoice_index,
        query_filters=query_filters,
        customer_field=customer_field,
        visit_field=visit_field,
        date_field=date_field,
        amount_field=amount_field,
        pieces_field=pieces_field,
        page_size=page_size,
        max_pages=max_pages,
    ):
        key = b.get("key") or {}
        cust = key.get("customer")

        if current_customer is None:
            current_customer = cust
        elif cust != current_customer:
            flush_current_customer()
            current_customer = cust
            best_dt = None
            best_amount = 0.0
            best_pieces = 0.0

        first_date_ms = ((b.get("first_date") or {}).get("value"))
        dt = _ms_to_dt(first_date_ms)
        if not dt:
            continue

        amt = float(((b.get("visit_amount") or {}).get("value")) or 0.0)
        pcs = 0.0
        if pieces_field is not None:
            pcs = float(((b.get("visit_pieces") or {}).get("value")) or 0.0)

        if best_dt is None or dt < best_dt:
            best_dt = dt
            best_amount = amt
            best_pieces = pcs

    if current_customer is not None:
        flush_current_customer()

    return {
        "initial_visit_amount": total_amount,
        "initial_visit_pieces": (total_pieces if any_pieces_values else None),
    }


# -------------------------------------------------------------------
# Composite streaming helper for customer_ltv (invoices-only)
# -------------------------------------------------------------------

def _customer_ltv_from_invoices_composite(
    req,
    client,
    *,
    index_name: str,
    customer_field: str,
    amount_field: str,
) -> Dict[str, Any]:
    """
    Safe-ish invoices fallback for customer LTV.
    """
    page_size = int(getattr(req, "es_composite_page_size", 300) or 300)
    page_size = max(50, min(page_size, 1000))

    max_pages = int(getattr(req, "es_composite_max_pages", 2000) or 2000)
    max_pages = max(10, min(max_pages, 20000))

    top_n = int(getattr(req, "top_n_customers", 500) or 500)
    top_n = max(1, min(top_n, 2000))

    query_filters: List[Dict[str, Any]] = []

    total_spend_pos = 0.0
    count_pos = 0
    scanned_customers = 0
    pages = 0
    truncated = False

    top_heap: List[Tuple[float, Any]] = []
    after: Optional[Dict[str, Any]] = None

    while True:
        pages += 1
        if pages > max_pages:
            truncated = True
            break

        body: Dict[str, Any] = {
            "size": 0,
            "aggs": {
                "customers": {
                    "composite": {
                        "size": page_size,
                        "sources": [{"customer": {"terms": {"field": customer_field}}}],
                        **({"after": after} if after else {}),
                    },
                    "aggs": {"ltv": {"sum": {"field": amount_field}}},
                }
            },
        }
        if query_filters:
            body["query"] = {"bool": {"filter": query_filters}}

        res = _safe_es_search(client, index=index_name, body=body)
        cust = ((res.get("aggregations") or {}).get("customers")) or {}
        buckets = cust.get("buckets") or []
        if not buckets:
            break

        for b in buckets:
            if not isinstance(b, dict):
                continue

            key = b.get("key") or {}
            cid = key.get("customer")
            ltv_val = float(((b.get("ltv") or {}).get("value")) or 0.0)

            scanned_customers += 1

            if ltv_val > 0:
                total_spend_pos += ltv_val
                count_pos += 1

            if len(top_heap) < top_n:
                heapq.heappush(top_heap, (ltv_val, cid))
            else:
                if ltv_val > top_heap[0][0]:
                    heapq.heapreplace(top_heap, (ltv_val, cid))

        after = cust.get("after_key")
        if not after:
            break

    avg_ltv = (total_spend_pos / float(count_pos)) if count_pos > 0 else None

    top_sorted = sorted(top_heap, key=lambda x: x[0], reverse=True)
    rows = [{"customer_id": cid, "ltv": ltv} for (ltv, cid) in top_sorted]

    insight = (
        f"Customer lifetime value was computed from invoices on index '{index_name}' using a "
        f"composite aggregation over '{customer_field}' (paged) with sum('{amount_field}'). "
    )
    if avg_ltv is not None:
        insight += f"Average LTV (customers with LTV > 0) is approximately {avg_ltv:.2f} across {count_pos} customers. "
    else:
        insight += "No customers with positive lifetime value were found. "

    insight += (
        f"Scanned {scanned_customers} customers in {pages} composite pages. "
        f"Rows are limited to the top {top_n} customers by LTV to keep the response small."
    )
    if truncated:
        insight += " NOTE: Scan stopped early due to max_pages limit; results may be incomplete."

    return {"insight": to_json_safe(insight), "rows": to_json_safe(rows), "avg_ltv": avg_ltv, "truncated": truncated}


# -------------------------------------------------------------------
# Core visit metrics (lifetime) - invoices only
# -------------------------------------------------------------------
def _es_core_visit_metrics(
    req,
    client,
    mappings: Dict[str, Any],
    business_rules: Optional[str],
):
    """
    Core visit KPIs for the dashboard (LIFETIME / ALL HISTORY)

    ✅ Customers index ONLY (fast)
    ❌ No invoice fallback
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
        "request_cache": True,  # good for “all-time” totals
        "aggs": {
            "unique_customers": {"cardinality": {"field": "customer_id"}},
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

    # Customers mapping does NOT have lifetime pieces (only averages) → null
    total_visit_pieces = None

    rows: List[Dict[str, Any]] = [
        {"metric": "total_visit_amount", "label": "Total Visit Amount", "value": total_visit_amount},
        {"metric": "total_visit_pieces", "label": "Total Visit Pieces", "value": total_visit_pieces},
        {"metric": "total_visits", "label": "Total Visits", "value": total_visits},
        {"metric": "unique_customers", "label": "Unique Customers", "value": unique_customers},
    ]

    insight = (
        f"Core lifetime metrics were computed from customers rollups on index '{chosen}' "
        f"using sum(visits_lifetime) and sum(sales_pickup_lifetime). "
        f"Total Visit Pieces is not available because the customers index does not store lifetime pieces totals."
    )

    return {
        "insight": to_json_safe(insight),
        "rows": to_json_safe(rows),
        "rules_used": business_rules or "",
        "engine": "es",
    }

# -------------------------------------------------------------------
# Windowed customer-value metrics (per period)
# -------------------------------------------------------------------

def _es_customer_value_metrics(
    req,
    client,
    mappings: Dict[str, Any],
    business_rules: Optional[str],
):
    """
    Windowed customer value metrics for the selected period (invoices index).

    ✅ MODIFIED: use direct mapping fields (no resolve_es_field)
      - customer_id, dropoff_at, total, pieces, visit_id
    """
    invoice_index, invoice_mapping = _select_invoice_index_from_es_mapping(client, req.es_index_name)
    index_name = invoice_index

    properties = _extract_properties_from_mapping(invoice_mapping, invoice_index)
    invoice_mappings = {"properties": properties}

    # ✅ MODIFIED PART START ---------------------------------------------
    customer_field = "customer_id"
    date_field = "dropoff_at"
    amount_field = "total"

    pieces_field = "pieces" if _field_exists(invoice_mappings, "pieces") else None
    visit_field = "visit_id" if _field_exists(invoice_mappings, "visit_id") else None

    if not (_field_exists(invoice_mappings, customer_field) and _field_exists(invoice_mappings, date_field) and _field_exists(invoice_mappings, amount_field)):
        return _es_cannot_answer(
            "Cannot compute customer value metrics because one of the required fields "
            "is missing from the invoices index mapping: customer_id, dropoff_at, total.",
            business_rules,
        )
    # ✅ MODIFIED PART END -----------------------------------------------

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

    res = _safe_es_search(client, index=index_name, body=body)
    agg = res.get("aggregations", {}) or {}

    unique_customers = int((agg.get("unique_customers") or {}).get("value") or 0)
    total_revenue = float((agg.get("total_revenue") or {}).get("value") or 0.0)

    # ✅ always provide numeric total_pieces (0.0 if missing)
    if "total_pieces" in agg:
        total_pieces = float((agg["total_pieces"] or {}).get("value") or 0.0)
    else:
        total_pieces = 0.0

    total_visits = int((agg.get("total_visits") or {}).get("value") or 0)

    # ✅ dashboard prefers 0.0 instead of null when there is no data
    if unique_customers > 0:
        avg_visits_per_customer = total_visits / float(unique_customers)
        revenue_per_customer = total_revenue / float(unique_customers)
        pieces_per_customer = total_pieces / float(unique_customers)
    else:
        avg_visits_per_customer = 0.0
        revenue_per_customer = 0.0
        pieces_per_customer = 0.0

    if total_pieces > 0:
        avg_dollar_per_piece = total_revenue / total_pieces
    else:
        avg_dollar_per_piece = 0.0

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
        window_desc = []
        if getattr(req, "start_date", None):
            window_desc.append(f"from {req.start_date}")
        if getattr(req, "end_date", None):
            window_desc.append(f"to {req.end_date}")
        window_str = " ".join(window_desc)
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


def _window_customer_value_metrics(
    base_req,
    period,
    client,
    mappings: Dict[str, Any],
) -> Dict[str, Optional[float]]:
    """
    Run windowed metrics for a specific [start_date, end_date]
    and return a dict {metric_id -> value}.

    ✅ MODIFIED:
      - Always return ALL dashboard metric keys
      - Never return None for numeric KPIs (use 0.0 instead)
      - Use invoice mappings extracted from the selected invoice index for downstream ES metrics
    """
    req = deepcopy(base_req)
    req.start_date = period.start_date
    req.end_date = period.end_date

    expected_metric_ids = [
        "total_visits",
        "unique_customers",
        "total_revenue",
        "total_pieces",
        "average_visits_per_customer",
        "visit_pieces_per_customer",
        "revenue_per_customer",
        "avg_dollar_per_piece",
        "initial_visit_amount",
        "initial_visit_pieces",
        "avg_pickup_delay_retail",
        "redo_invoices_count",
        "new_customer_acquisition_rate",
        "new_customer_30d_return_rate",
        "customers_2plus_visits",
        "customers_3plus_visits",
        "customers_4plus_visits",
        "customers_5plus_visits",
        "top20_customers_redo_issues",
        "coupon_returns_365d_since_signup",
    ]

    values: Dict[str, Optional[float]] = {k: 0.0 for k in expected_metric_ids}

    # ✅ MODIFIED: always use extracted invoice mappings for any invoice-based ES metric
    invoice_index, invoice_mapping = _select_invoice_index_from_es_mapping(client, req.es_index_name)
    invoice_props = _extract_properties_from_mapping(invoice_mapping, invoice_index)
    invoice_mappings = {"properties": invoice_props}

    # ---- core windowed metrics ----
    resp = _es_customer_value_metrics(req=req, client=client, mappings=invoice_mappings, business_rules=None)
    for r in (resp.get("rows") or []):
        if isinstance(r, dict) and r.get("metric") is not None:
            mid = str(r["metric"])
            values[mid] = _as_float(r.get("value"), 0.0)

    # ----- initial visit totals (amount / pieces) -----
    init_vals = _es_initial_visit_totals(req, client, invoice_mappings) or {}
    values["initial_visit_amount"] = _as_float(init_vals.get("initial_visit_amount"), 0.0)
    values["initial_visit_pieces"] = _as_float(init_vals.get("initial_visit_pieces"), 0.0)

    # ----- extra lifecycle / promo KPIs -----
    from app.api.metrics import metrics_lifecycle, metrics_promos_coupons

    window_req = deepcopy(req)

    # Average Pickup Delay
    try:
        window_req.question = "Average Pickup Delay (Retail)"
        delay_resp = metrics_promos_coupons._es_avg_pickup_delay_retail(
            window_req, client, invoice_mappings, business_rules=None
        )
        delay_val = _metric_value_from_rows_any(
            delay_resp,
            [
                "avg_pickup_delay_retail",
                "avg_pickup_delay_days",
                "average_pickup_delay_days",
                "value",
            ],
        )
        values["avg_pickup_delay_retail"] = _as_float(delay_val, 0.0)
    except Exception:
        values["avg_pickup_delay_retail"] = 0.0

    # Invoices with Redo Items
    try:
        window_req.question = "Invoices with Redo Items"
        redo_resp = metrics_promos_coupons._es_invoices_with_redo_items(
            window_req, client, invoice_mappings, business_rules=None
        )
        redo_val = _metric_value_from_rows_any(
            redo_resp,
            [
                "redo_invoices_count",
                "invoices_with_redo",
                "redo_invoices",
                "value",
            ],
        )
        values["redo_invoices_count"] = _as_float(redo_val, 0.0)
    except Exception:
        values["redo_invoices_count"] = 0.0

    # New Customer Acquisition
    try:
        window_req.question = "New Customer Acquisition"
        new_cust_resp = metrics_lifecycle._es_new_customer_acquisition(
            window_req, client, invoice_mappings, business_rules=None
        )
        s = _sum_field_from_rows(new_cust_resp, "new_customers")
        if s is None:
            s = _sum_field_from_rows(new_cust_resp, "new_customers_count")
        values["new_customer_acquisition_rate"] = _as_float(s, 0.0)
    except Exception:
        values["new_customer_acquisition_rate"] = 0.0

    # New Customer 30-Day Return Rate
    try:
        window_req.question = "New Customer 30-Day Return Rate"
        r30_resp = metrics_lifecycle._es_new_customer_30d_return_rate(
            window_req, client, invoice_mappings, business_rules=None
        )
        r30_val = _metric_value_from_rows_any(
            r30_resp,
            [
                "new_customer_30d_return_rate",
                "return_rate_30d",
                "thirty_day_return_rate",
                "value",
            ],
        )
        values["new_customer_30d_return_rate"] = _as_float(r30_val, 0.0)
    except Exception:
        values["new_customer_30d_return_rate"] = 0.0

    # Nth visits (2nd/3rd/4th/5th)
    def _nth_value_for(question: str) -> float:
        window_req.question = question
        resp2 = metrics_lifecycle._es_customers_nth_visit(
            window_req, client, invoice_mappings, business_rules=None
        )

        v = _metric_value_from_rows_any(
            resp2,
            [
                "customers_2plus_visits",
                "customers_3plus_visits",
                "customers_4plus_visits",
                "customers_5plus_visits",
                "customers_achieved_2nd_visit",
                "customers_achieved_3rd_visit",
                "customers_achieved_4th_visit",
                "customers_achieved_5th_visit",
                "value",
            ],
        )
        if v is not None:
            return _as_float(v, 0.0)

        rows2 = (resp2 or {}).get("rows") or []
        if rows2 and all(isinstance(r, dict) for r in rows2):
            return float(len(rows2))

        return 0.0

    try:
        values["customers_2plus_visits"] = _nth_value_for("Customers Achieving 2nd Visit")
        values["customers_3plus_visits"] = _nth_value_for("Customers Achieving 3rd Visit")
        values["customers_4plus_visits"] = _nth_value_for("Customers Achieving 4th Visit")
        values["customers_5plus_visits"] = _nth_value_for("Customers Achieving 5th Visit")
    except Exception:
        values["customers_2plus_visits"] = 0.0
        values["customers_3plus_visits"] = 0.0
        values["customers_4plus_visits"] = 0.0
        values["customers_5plus_visits"] = 0.0

    # Top 20% Customers – Redo Issues
    try:
        window_req.question = "Top 20% Customers – Redo / Courtesy Issues"
        top20_resp = metrics_promos_coupons._es_top20_customers_with_redo_courtesy(
            window_req, client, invoice_mappings, business_rules=None
        )
        values["top20_customers_redo_issues"] = _as_float(_count_rows(top20_resp), 0.0)
    except Exception:
        values["top20_customers_redo_issues"] = 0.0

    # Coupon Returns (365 Days Since Signup)
    try:
        window_req.question = "Coupon Returns – First Coupon Visit 365+ Days After Signup"
        coupon_resp = metrics_promos_coupons._es_coupon_returns_365d_since_signup(
            window_req, client, invoice_mappings, business_rules=None
        )
        values["coupon_returns_365d_since_signup"] = _as_float(_count_rows(coupon_resp), 0.0)
    except Exception:
        values["coupon_returns_365d_since_signup"] = 0.0

    # ✅ final safety: ensure expected keys exist and are numeric
    for k in expected_metric_ids:
        values[k] = _as_float(values.get(k), 0.0)

    return values


# -------------------------------------------------------------------
# Initial visit totals per window (invoices-only)
# -------------------------------------------------------------------

def _es_initial_visit_totals(
    req,
    client,
    mappings: Dict[str, Any],
) -> Dict[str, Optional[float]]:
    """
    Initial Visit – Amount / Pieces (windowed)

    ✅ MODIFIED: use direct mapping fields (no resolve_es_field)
      - customer_id, dropoff_at, total, pieces, visit_id
    """
    try:
        invoice_index, invoice_mapping = _select_invoice_index_from_es_mapping(client, req.es_index_name)
    except Exception:
        return {"initial_visit_amount": None, "initial_visit_pieces": None}

    properties = _extract_properties_from_mapping(invoice_mapping, invoice_index)
    invoice_mappings = {"properties": properties}

    # ✅ MODIFIED PART START ---------------------------------------------
    customer_field = "customer_id"
    date_field = "dropoff_at"
    amount_field = "total"
    visit_field = "visit_id"
    pieces_field = "pieces" if _field_exists(invoice_mappings, "pieces") else None

    required = [customer_field, date_field, amount_field, visit_field]
    if not all(_field_exists(invoice_mappings, f) for f in required):
        return {"initial_visit_amount": None, "initial_visit_pieces": None}
    # ✅ MODIFIED PART END -----------------------------------------------

    try:
        return _initial_visit_totals_from_invoices_composite(
            req,
            client,
            invoice_index=invoice_index,
            customer_field=customer_field,
            visit_field=visit_field,
            date_field=date_field,
            amount_field=amount_field,
            pieces_field=pieces_field,
        )
    except Exception:
        return {"initial_visit_amount": None, "initial_visit_pieces": None}


# -------------------------------------------------------------------
# Average CLV (lifetime) - invoices only
# -------------------------------------------------------------------

def _es_customer_ltv(
    req,
    client,
    mappings: Dict[str, Any],
    business_rules: Optional[str],
):
    """
    Average customer lifetime value (average customer spend).

    ✅ MODIFIED: use direct mapping fields (no resolve_es_field)
      - customer_id, total
    """
    try:
        invoice_index, invoice_mapping = _select_invoice_index_from_es_mapping(client, req.es_index_name)
    except Exception:
        return _es_cannot_answer(
            "Cannot compute 'average customer lifetime value' because invoice index could not be selected.",
            business_rules,
        )

    properties = _extract_properties_from_mapping(invoice_mapping, invoice_index)
    invoice_mappings = {"properties": properties}

    # ✅ MODIFIED PART START ---------------------------------------------
    customer_field = "customer_id"
    amount_field = "total"

    if not (_field_exists(invoice_mappings, customer_field) and _field_exists(invoice_mappings, amount_field)):
        return _es_cannot_answer(
            "Cannot compute 'average customer lifetime value' because required fields "
            "customer_id and/or total are missing from the invoices mapping.",
            business_rules,
        )
    # ✅ MODIFIED PART END -----------------------------------------------

    try:
        out = _customer_ltv_from_invoices_composite(
            req,
            client,
            index_name=invoice_index,
            customer_field=customer_field,
            amount_field=amount_field,
        )
        return {
            "insight": out.get("insight"),
            "rows": out.get("rows"),
            "rules_used": business_rules or "",
            "engine": "es",
        }
    except Exception:
        return _es_cannot_answer(
            "Cannot compute 'average customer lifetime value' safely on this Elasticsearch index "
            "due to an error during composite aggregation paging.",
            business_rules,
        )


def _es_one_time_vs_repeat(
    req,
    client,
    mappings: Dict[str, Any],
    business_rules: Optional[str],
):
    """
    One-time vs repeat customers (invoices-only).

    ✅ MODIFIED:
      - Always select the invoice index + extract invoice mappings
      - Avoid passing possibly-wrong 'mappings' from caller
    """
    # ✅ MODIFIED PART START ---------------------------------------------
    try:
        invoice_index, invoice_mapping = _select_invoice_index_from_es_mapping(client, req.es_index_name)
    except Exception:
        return _es_cannot_answer(
            "Cannot compute 'one-time vs repeat customers' because invoice index could not be selected.",
            business_rules,
        )

    properties = _extract_properties_from_mapping(invoice_mapping, invoice_index)
    invoice_mappings = {"properties": properties}

    stats = _es_get_customer_stats(client, invoice_index, invoice_mappings)
    # ✅ MODIFIED PART END -----------------------------------------------

    if stats is None:
        return _es_cannot_answer(
            "Cannot compute 'one-time vs repeat customers' because customer or date fields "
            "could not be resolved from the Elasticsearch mappings.",
            business_rules,
        )

    stats_nonzero = [s for s in stats if (s.get("visit_count") or 0) > 0]
    total = len(stats_nonzero)

    if total == 0:
        return {
            "insight": to_json_safe("No customers with at least one visit were found."),
            "rows": [],
            "rules_used": business_rules or "",
            "engine": "es",
        }

    one_time = sum(1 for s in stats_nonzero if (s.get("visit_count") or 0) == 1)
    repeat = sum(1 for s in stats_nonzero if (s.get("visit_count") or 0) > 1)

    one_pct = (one_time * 100.0 / total) if total else 0.0
    repeat_pct = (repeat * 100.0 / total) if total else 0.0

    rows = [
        {"segment": "one-time", "customer_count": one_time, "percentage_of_customers": one_pct},
        {"segment": "repeat", "customer_count": repeat, "percentage_of_customers": repeat_pct},
    ]

    insight = (
        f"Out of {total} customers with at least one visit, about {one_pct:.1f}% are one-time "
        f"and {repeat_pct:.1f}% are repeat (based on ES visit counts)."
    )

    return {
        "insight": to_json_safe(insight),
        "rows": to_json_safe(rows),
        "rules_used": business_rules or "",
        "engine": "es",
    }


__all__ = [
    "_es_core_visit_metrics",
    "_es_customer_value_metrics",
    "_window_customer_value_metrics",
    "_es_initial_visit_totals",
    "_es_customer_ltv",
    "_es_one_time_vs_repeat",
]
