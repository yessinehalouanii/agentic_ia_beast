from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional

from abi.runtime import to_json_safe
from app.api.metrics.shared_utilities import _es_cannot_answer, _date_filters_or_default
from app.api.metrics.shared_utilities import (
    _safe_es_search,
    _get_customers_index_and_mappings,  # ✅ new signature supports reuse
)

# -------------------------------------------------------------------
# Safety limits (server-friendly defaults)
# -------------------------------------------------------------------

DEFAULT_WINDOW_DAYS = 365
MAX_SCAN_HITS = 50_000
MAX_SCAN_PAGES = 200
MAX_TOP20_CUSTOMERS = 50_000
MAX_ROWS_RETURNED = 10_000
TERMS_CHUNK_SIZE = 1000


# -------------------------------------------------------------------
# Small helpers
# -------------------------------------------------------------------

def _chunks(lst: List[Any], n: int) -> Iterable[List[Any]]:
    for i in range(0, len(lst), n):
        yield lst[i : i + n]


def _scan_all_hits(
    client: Any,
    *,
    index: str,
    query: Dict[str, Any],
    source_fields: Optional[List[str]] = None,
    page_size: int = 2000,
    max_hits: int = MAX_SCAN_HITS,
    max_pages: int = MAX_SCAN_PAGES,
    sort: Optional[List[Dict[str, Any]]] = None,
) -> Iterable[Dict[str, Any]]:
    """
    Scan hits using search_after (no scroll) with hard caps.

    IMPORTANT:
    - search_after needs a deterministic sort. Always pass a real field sort.
    - We intentionally DO NOT fallback to '_doc' because it is not stable for search_after pagination.
    """
    if not sort:
        raise ValueError("scan_all_hits requires a deterministic 'sort' for search_after (e.g. [{'customer_id':'asc'}]).")

    base_body: Dict[str, Any] = {
        "size": page_size,
        "query": query or {"match_all": {}},
        "sort": sort,
        "track_total_hits": False,
    }
    if source_fields is not None:
        base_body["_source"] = source_fields

    search_after = None
    emitted = 0
    pages = 0

    while True:
        if pages >= max_pages or emitted >= max_hits:
            break

        body = dict(base_body)
        if search_after is not None:
            body["search_after"] = search_after

        res = _safe_es_search(client, index=index, body=body)
        hits = ((res.get("hits") or {}).get("hits") or []) or []
        if not hits:
            break

        for h in hits:
            yield h
            emitted += 1
            if emitted >= max_hits:
                break

        pages += 1
        search_after = hits[-1].get("sort")
        if not search_after:
            break


def _wildcards_for_token(field: str, token: str) -> List[Dict[str, Any]]:
    """
    Case-tolerant wildcard queries WITHOUT relying on ES case_insensitive.
    Uses coupon.keyword (stable schema).
    """
    t = (token or "").strip()
    if not t:
        return []
    variants = {t.lower(), t.upper(), t[:1].upper() + t[1:].lower()}
    return [{"wildcard": {field: f"*{v}*"}} for v in sorted(variants)]


def _coupon_should_clauses_stable(tokens: List[str]) -> List[Dict[str, Any]]:
    """
    Stable schema: coupon.keyword exists (text + keyword).
    We only query coupon.keyword using wildcards.
    """
    coupon_kw = "coupon.keyword"
    clauses: List[Dict[str, Any]] = []
    for tok in tokens:
        clauses.extend(_wildcards_for_token(coupon_kw, tok))
    return clauses


def _get_existing_customers_mappings(req: Any) -> Optional[Dict[str, Any]]:
    """
    Optional reuse hook:
    If the caller (dashboard/router) already loaded customers mappings, it can attach them
    on the request object under one of these names.
    """
    for attr in ("es_customers_mappings", "customers_mappings", "_customers_mappings"):
        m = getattr(req, attr, None)
        if isinstance(m, dict) and isinstance(m.get("properties"), dict) and m.get("properties"):
            return m
    return None


# -------------------------------------------------------------------
# Generic invoices SUM metric (stable schema)
# -------------------------------------------------------------------

def _es_sum_invoices_metric(
    req: Any,
    client: Any,
    business_rules: Optional[str],
    *,
    metric_id: str,
    label: str,
    date_field: str,
    value_field: str,
) -> Dict[str, Any]:
    """
    Stable schema version:
      - invoices index is req.es_index_name (concrete)
      - fields are used directly (no mapping resolver)
      - invoice_id is integer; value_count(invoice_id) is valid
      - window filters are built by shared helper _date_filters_or_default
    """
    invoices_index = (getattr(req, "es_index_name", "") or "").strip()
    if not invoices_index:
        return _es_cannot_answer(f"{label} requires invoices index (es_index_name).", business_rules)

    filters, window_label = _date_filters_or_default(req, date_field)
    filters.append({"exists": {"field": date_field}})
    filters.append({"exists": {"field": value_field}})

    body = {
        "size": 0,
        "query": {"bool": {"filter": filters}},
        "aggs": {
            metric_id: {"sum": {"field": value_field}},
            "invoice_count": {"value_count": {"field": "invoice_id"}},
        },
    }

    res = _safe_es_search(client, index=invoices_index, body=body)
    aggs = res.get("aggregations") or {}

    value = float((aggs.get(metric_id) or {}).get("value") or 0.0)
    invoice_count = int((aggs.get("invoice_count") or {}).get("value") or 0)

    rows = [
        {
            "metric": metric_id,
            "label": label,
            "value": value,
            "invoice_count": invoice_count,
            "date_field": date_field,
            "value_field": value_field,
            "window": window_label,
            "invoices_index": invoices_index,
        }
    ]

    insight = f"{label} = SUM(invoices.{value_field}) for invoices whose invoices.{date_field} is in ({window_label})."

    return {
        "insight": to_json_safe(insight),
        "rows": to_json_safe(rows),
        "rules_used": business_rules or "",
        "engine": "es",
    }


# -------------------------------------------------------------------
# Metrics (stable schema)
# -------------------------------------------------------------------

def _es_invoices_with_redo_items(
    req: Any,
    client: Any,
    mappings: Dict[str, Any],  # unused in stable version; kept for signature compatibility
    business_rules: Optional[str],
) -> Dict[str, Any]:
    """
    Stable schema:
      - Windowed by dropoff_at
      - coupon.keyword exists, query by wildcard contains "redo" (case variants)
      - Count invoices using value_count(invoice_id) (invoice_id is integer)
    """
    invoices_index = (getattr(req, "es_index_name", "") or "").strip()
    if not invoices_index:
        return _es_cannot_answer("Invoices with Redo Items requires invoices index (es_index_name).", business_rules)

    date_field = "dropoff_at"
    filters, window_label = _date_filters_or_default(req, date_field)
    filters.append({"exists": {"field": date_field}})

    coupon_filter = {
        "bool": {
            "should": _coupon_should_clauses_stable(["redo"]),
            "minimum_should_match": 1,
        }
    }
    filters.append(coupon_filter)

    body: Dict[str, Any] = {
        "size": 0,
        "track_total_hits": False,
        "query": {"bool": {"filter": filters}},
        "aggs": {"redo_invoices": {"value_count": {"field": "invoice_id"}}},
    }

    res = _safe_es_search(client, index=invoices_index, body=body)
    count = int((((res.get("aggregations") or {}).get("redo_invoices") or {}).get("value")) or 0)

    rows = [{"metric": "invoices_with_redo", "label": "Invoices with Redo Items", "value": float(count)}]
    insight = (
        f"'Invoices with Redo Items' counted invoices on index '{invoices_index}' ({window_label}) "
        f"where coupon.keyword contains 'redo', windowed by '{date_field}'."
    )

    return {
        "insight": to_json_safe(insight),
        "rows": to_json_safe(rows),
        "rules_used": business_rules or "",
        "engine": "es",
    }


def _es_avg_pickup_delay_retail(
    req: Any,
    client: Any,
    mappings: Dict[str, Any],  # unused in stable version; kept for signature compatibility
    business_rules: Optional[str],
) -> Dict[str, Any]:
    """
    Stable schema:
      - Windowed by dropoff_at
      - Delay computed from ready_at -> pickup_at
      - Filter ONLY Retail route via nested route.name.keyword == "Retail"
    """
    invoices_index = (getattr(req, "es_index_name", "") or "").strip()
    if not invoices_index:
        return _es_cannot_answer("Average Pickup Delay (Retail) requires invoices index (es_index_name).", business_rules)

    window_date_field = "dropoff_at"
    ready_field = "ready_at"
    pickup_field = "pickup_at"

    filters, window_label = _date_filters_or_default(req, window_date_field)
    filters.append({"exists": {"field": ready_field}})
    filters.append({"exists": {"field": pickup_field}})

    route_filter = {
        "nested": {
            "path": "route",
            "query": {"term": {"route.name.keyword": "Retail"}},
            "ignore_unmapped": True,
        }
    }

    script_source = (
        f"if (doc['{pickup_field}'].size() == 0 || doc['{ready_field}'].size() == 0) return null; "
        f"long diff = doc['{pickup_field}'].value.toInstant().toEpochMilli() "
        f"- doc['{ready_field}'].value.toInstant().toEpochMilli(); "
        "return diff / 86400000.0;"
    )

    body: Dict[str, Any] = {
        "size": 0,
        "track_total_hits": False,
        "query": {"bool": {"filter": filters + [route_filter]}},
        "aggs": {"pickup_delay": {"stats": {"script": {"lang": "painless", "source": script_source}}}},
    }

    res = _safe_es_search(client, index=invoices_index, body=body)
    stats = (res.get("aggregations") or {}).get("pickup_delay") or {}

    count = int(stats.get("count") or 0)
    avg_days = stats.get("avg")
    min_days = stats.get("min")
    max_days = stats.get("max")

    if count == 0 or avg_days is None:
        rows = [
            {
                "metric": "avg_pickup_delay_days",
                "label": "Average Pickup Delay (Retail, days)",
                "value": 0.0,
                "count_invoices": 0,
                "min_delay_days": None,
                "max_delay_days": None,
                "window": window_label,
                "invoices_index": invoices_index,
            }
        ]
        insight = (
            "Average pickup delay (Retail) is 0.0 because no retail invoices had both "
            f"ready_at and pickup_at in ({window_label})."
        )
    else:
        rows = [
            {
                "metric": "avg_pickup_delay_days",
                "label": "Average Pickup Delay (Retail, days)",
                "value": float(avg_days),
                "count_invoices": count,
                "min_delay_days": float(min_days) if min_days is not None else None,
                "max_delay_days": float(max_days) if max_days is not None else None,
                "window": window_label,
                "invoices_index": invoices_index,
            }
        ]
        insight = (
            f"Average pickup delay (Retail) computed on '{invoices_index}' ({window_label}) "
            f"as (pickup_at - ready_at) in days, filtered by nested route.name.keyword == 'Retail'. "
            f"Across {count} invoices, avg delay is {float(avg_days):.1f} days."
        )

    return {
        "insight": to_json_safe(insight),
        "rows": to_json_safe(rows),
        "rules_used": business_rules or "",
        "engine": "es",
    }

def _es_top20_customers_with_redo_courtesy(
    req: Any,
    client: Any,
    mappings: Dict[str, Any],  # unused in stable version; kept for signature compatibility
    business_rules: Optional[str],
) -> Dict[str, Any]:
    """
    Stable schema:
      - Customers index has nested tags with tags.name.keyword (and tags.tag_id exists but not required here)
      - req provides top20_tag_name or top20_tag_names (we do not require tag_id)
      - Invoices have customer_id (int), dropoff_at (date), coupon.keyword (keyword)
      - For those Top20 customers, count invoices where coupon.keyword contains redo/courtesy.
    """
    invoices_index = (getattr(req, "es_index_name", "") or "").strip()
    customers_index_in = (getattr(req, "es_customers_index_name", "") or "").strip()

    if not invoices_index or not customers_index_in:
        return _es_cannot_answer(
            "Top 20% Customers with Redo/Courtesy Items requires invoices and customers indexes.",
            business_rules,
        )

    # ------------------------------------------------------------------
    # ✅ CHANGED: require explicit tag NAME(s) instead of tag_id(s)
    # ------------------------------------------------------------------
    raw_names = getattr(req, "top20_tag_names", None)
    raw_one_name = getattr(req, "top20_tag_name", None)

    top20_tag_names: List[str] = []

    if raw_names:
        try:
            # raw_names might be a list/tuple OR a single string
            if isinstance(raw_names, (list, tuple, set)):
                for x in raw_names:
                    if not x:
                        continue
                    top20_tag_names.append(str(x).strip())
            else:
                top20_tag_names.append(str(raw_names).strip())
        except Exception:
            top20_tag_names = []

    if not top20_tag_names and raw_one_name:
        top20_tag_names = [str(raw_one_name).strip()]

    top20_tag_names = sorted(set([n for n in top20_tag_names if n]))
    if not top20_tag_names:
        return _es_cannot_answer(
            "Cannot compute Top 20% Customers because top20_tag_name/top20_tag_names was not provided.",
            business_rules,
        )

    # ✅ reuse-aware mappings loader (unchanged)
    existing_cust_mappings = _get_existing_customers_mappings(req)
    try:
        cust_index, _cust_mappings = _get_customers_index_and_mappings(
            client,
            customers_index_in,
            existing_mappings=existing_cust_mappings,
            existing_index=customers_index_in,
        )
    except Exception as e:
        return _es_cannot_answer(
            f"Could not load mappings for customers index '{customers_index_in}': {e}",
            business_rules,
        )

    # Stable customers fields
    cust_id_field = "customer_id"
    tags_name_field = "tags.name.keyword"  # ✅ CHANGED

    top20_filter = {
        "nested": {
            "path": "tags",
            "ignore_unmapped": True,
            "query": {"terms": {tags_name_field: top20_tag_names}},  # ✅ CHANGED
        }
    }

    # Scan Top20 customers (sorted by customer_id)
    safe_customer_sort = [{cust_id_field: "asc"}]

    top20_info_by_id: Dict[Any, Dict[str, Any]] = {}
    scan_capped = False

    for h in _scan_all_hits(
        client,
        index=cust_index,
        query={"bool": {"filter": [top20_filter]}},
        source_fields=[
            "customer_id",
            "first_name",
            "last_name",
            "sales_pickup_lifetime",
            "sales_pickup_365",
        ],
        page_size=2000,
        max_hits=MAX_TOP20_CUSTOMERS,
        sort=safe_customer_sort,
    ):
        src = h.get("_source", {}) or {}
        cid = src.get("customer_id")
        if cid is None:
            continue

        first_name = (src.get("first_name") or "").strip()
        last_name = (src.get("last_name") or "").strip()
        full_name = (f"{first_name} {last_name}").strip() or f"Customer {cid}"

        ltv_lifetime = src.get("sales_pickup_lifetime")
        ltv_365 = src.get("sales_pickup_365")
        ltv = float(ltv_lifetime) if ltv_lifetime is not None else (
            float(ltv_365) if ltv_365 is not None else None
        )

        top20_info_by_id[cid] = {
            "name": full_name,
            "lifetime_value": ltv,
            "sales_pickup_lifetime": ltv_lifetime,
            "sales_pickup_365": ltv_365,
        }

        if len(top20_info_by_id) >= MAX_TOP20_CUSTOMERS:
            scan_capped = True
            break

    if not top20_info_by_id:
        return {
            "insight": to_json_safe(
                f"No customers matched Top20 tag name(s) {top20_tag_names} in customers index."
            ),
            "rows": to_json_safe(
                [
                    {
                        "metric": "top20_customers_redo_issues",
                        "label": "Top 20% Customers – Redo Issues",
                        "value": 0.0,
                        "top20_customers_scanned": 0,
                    }
                ]
            ),
            "rules_used": business_rules or "",
            "engine": "es",
        }

    customer_ids = list(top20_info_by_id.keys())

    # Stable invoices fields
    invoice_customer_field = "customer_id"
    invoice_date_field = "dropoff_at"

    filters_base, window_label = _date_filters_or_default(req, invoice_date_field)

    should_clauses = _coupon_should_clauses_stable(["redo", "courtesy"])
    coupon_filter = {"bool": {"should": should_clauses, "minimum_should_match": 1}}

    redo_counts_by_customer: Dict[Any, int] = {}

    for cid_chunk in _chunks(customer_ids, TERMS_CHUNK_SIZE):
        filters = list(filters_base)
        filters.append({"terms": {invoice_customer_field: cid_chunk}})
        filters.append(coupon_filter)

        body_inv = {
            "size": 0,
            "query": {"bool": {"filter": filters}},
            "aggs": {
                "customers": {
                    "terms": {"field": invoice_customer_field, "size": min(len(cid_chunk), 10000)}
                }
            },
        }

        res_inv = _safe_es_search(client, index=invoices_index, body=body_inv)
        buckets = (((res_inv.get("aggregations") or {}).get("customers") or {}).get("buckets") or []) or []

        for b in buckets:
            cid = b.get("key")
            c = int(b.get("doc_count") or 0)
            if c > 0:
                redo_counts_by_customer[cid] = redo_counts_by_customer.get(cid, 0) + c

    detail_rows: List[Dict[str, Any]] = []
    for cid, redo_count in redo_counts_by_customer.items():
        info = top20_info_by_id.get(cid)
        if not info:
            continue
        detail_rows.append(
            {
                "customer_id": cid,
                "customer_name": info["name"],
                "lifetime_value": info["lifetime_value"],
                "sales_pickup_lifetime": info["sales_pickup_lifetime"],
                "sales_pickup_365": info["sales_pickup_365"],
                "redo_count": redo_count,
            }
        )

    detail_rows.sort(key=lambda r: (r.get("lifetime_value") or 0.0), reverse=True)
    if len(detail_rows) > MAX_ROWS_RETURNED:
        detail_rows = detail_rows[:MAX_ROWS_RETURNED]

    insight = (
        f"Scanned {len(top20_info_by_id)} Top20 customers using tag name(s) {top20_tag_names}. "
        f"Matched {len(detail_rows)} with redo/courtesy coupons in invoices ({window_label})."
    )
    if scan_capped:
        insight += " Note: Top20 scan capped for safety."

    metric_value = float(len(redo_counts_by_customer))
    metric_row = {
        "metric": "top20_customers_redo_issues",
        "label": "Top 20% Customers – Redo Issues",
        "value": metric_value,
        "top20_customers_scanned": len(top20_info_by_id),
        "window": window_label,
        "scan_capped": scan_capped,
        "top20_tag_names": top20_tag_names,  # ✅ CHANGED
        "customers_index": cust_index,
        "invoices_index": invoices_index,
    }

    return {
        "insight": to_json_safe(insight),
        "rows": to_json_safe([metric_row] + detail_rows),
        "rules_used": business_rules or "",
        "engine": "es",
    }


def _es_incoming_sales(req: Any, client: Any, mappings: Dict[str, Any], business_rules: Optional[str]) -> Dict[str, Any]:
    return _es_sum_invoices_metric(
        req,
        client,
        business_rules,
        metric_id="incoming_sales",
        label="Incoming Sales",
        date_field="dropoff_at",
        value_field="total",
    )


def _es_incoming_pieces(req: Any, client: Any, mappings: Dict[str, Any], business_rules: Optional[str]) -> Dict[str, Any]:
    return _es_sum_invoices_metric(
        req,
        client,
        business_rules,
        metric_id="incoming_pieces",
        label="Incoming Pieces",
        date_field="dropoff_at",
        value_field="pieces",
    )


def _es_outgoing_sales(req: Any, client: Any, mappings: Dict[str, Any], business_rules: Optional[str]) -> Dict[str, Any]:
    return _es_sum_invoices_metric(
        req,
        client,
        business_rules,
        metric_id="outgoing_sales",
        label="Outgoing Sales",
        date_field="pickup_at",
        value_field="total",
    )


__all__ = [
    "_es_invoices_with_redo_items",
    "_es_avg_pickup_delay_retail",
    "_es_top20_customers_with_redo_courtesy",
    "_es_incoming_sales",
    "_es_incoming_pieces",
    "_es_outgoing_sales",
]
