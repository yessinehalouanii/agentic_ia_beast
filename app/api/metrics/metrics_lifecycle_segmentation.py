# metrics_lifecycle_segmentation.py
from __future__ import annotations

from datetime import datetime, timezone, timedelta
from typing import Any, Dict, List, Optional, Tuple, Union

from abi.runtime import to_json_safe
from app.api.metrics.metrics_lifecycle_engagement import _require_fields
from app.api.metrics.shared_utilities import (
    _field_exists,
    _safe_es_search,
    _get_invoice_index_and_mappings,
    _get_req_int,
    _load_customers_ctx,
)
from app.api.docs_analytics_routes import (
    ES_MAX_CUSTOMERS_DEFAULT,
    _es_cannot_answer,
    _build_date_range_filter,
)

# -------------------------------------------------------------------
# Shared helpers (dedupe)
# -------------------------------------------------------------------

_VISIT_FREQUENCY_BUCKET_META: List[Tuple[str, str, int, Optional[int]]] = [
    ("1_visit", "1 Visit", 1, 1),
    ("2_5", "2–5 Visits", 2, 5),
    ("6_11", "6–11 Visits", 6, 11),
    ("12_24", "12–24 Visits", 12, 24),
    ("25_plus", "25+ Visits", 25, None),
]


def _get_customer_stats_invoices_only(
    req,
    client,
    invoices_index: str,
    invoices_mappings: Dict[str, Any],
) -> Optional[List[Dict[str, Any]]]:
    """
    Build per-customer revenue stats from invoices ONLY.

    Returns a list like:
      [{"customer_id": <id>, "total_revenue": <float>}, ...]

    NOTE:
    - No visit_count (Option A).
    - Uses ES composite aggregation to page through customers safely.
    - Windowed using req.start_date/end_date via _build_date_range_filter(req, "dropoff_at").
    """
    customer_field = "customer_id"
    date_field = "dropoff_at"
    amount_field = "total"

    # Required fields for revenue stats
    if not (
        _field_exists(invoices_mappings, customer_field)
        and _field_exists(invoices_mappings, date_field)
        and _field_exists(invoices_mappings, amount_field)
    ):
        return None

    # Date window + basic guards
    filters = _build_date_range_filter(req, date_field) or []
    filters.append({"exists": {"field": customer_field}})
    filters.append({"exists": {"field": amount_field}})

    # Safety knobs
    page_size = _get_req_int(req, "es_composite_page_size", 1000, min_v=100, max_v=5000)
    max_customers = _get_req_int(req, "es_max_customers", 50_000, min_v=1000, max_v=200_000)
    max_pages = _get_req_int(req, "es_max_pages", 200, min_v=1, max_v=2000)

    after = None
    out: List[Dict[str, Any]] = []
    pages = 0

    while True:
        if pages >= max_pages or len(out) >= max_customers:
            break

        comp: Dict[str, Any] = {
            "size": page_size,
            "sources": [{"customer_id": {"terms": {"field": customer_field}}}],
        }
        if after:
            comp["after"] = after

        body: Dict[str, Any] = {
            "size": 0,
            "query": {"bool": {"filter": filters}},
            "aggs": {
                "by_customer": {
                    "composite": comp,
                    "aggs": {
                        "total_revenue": {"sum": {"field": amount_field}},
                    },
                }
            },
        }

        res = _safe_es_search(client, index=invoices_index, body=body)
        byc = ((res.get("aggregations") or {}).get("by_customer")) or {}
        buckets = byc.get("buckets") or []
        after = byc.get("after_key")

        if not buckets:
            break

        for b in buckets:
            key = b.get("key") or {}
            cid = key.get("customer_id")
            if cid is None:
                continue

            total_rev = float(((b.get("total_revenue") or {}).get("value")) or 0.0)
            out.append({"customer_id": cid, "total_revenue": total_rev})

            if len(out) >= max_customers:
                break

        pages += 1
        if not after:
            break

    return out


def _customers_ctx_or_error(
    req,
    client,
    business_rules: Optional[str],
    *,
    metric_name: str,
    required_fields: Optional[List[str]] = None,
    existing_mappings: Optional[Dict[str, Any]] = None,
    existing_index: Optional[str] = None,
) -> Union[Tuple[str, Dict[str, Any]], Dict[str, Any]]:
    """
    Unified customers-index context loader:
      - uses _load_customers_ctx (respects req.es_customers_index_name)
      - supports reuse of mappings/index when already fetched
      - optionally validates required fields via _require_fields
    """
    customers_index, cust_mappings, err = _load_customers_ctx(
        req,
        client,
        business_rules,
        existing_mappings=existing_mappings,
        existing_index=existing_index,
    )
    if err:
        return err

    if required_fields:
        missing_err = _require_fields(
            cust_mappings,
            customers_index,
            required_fields,
            metric_name,
            business_rules,
        )
        if missing_err:
            return missing_err

    return customers_index, cust_mappings


def _customers_visits_lifetime_positive_filters(cust_mappings: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Common “active-ish customer” filter used by lifetime visit segmentations:
      - visits_lifetime > 0
      - optionally require paying customers if sales_pickup_lifetime exists
    """
    filters: List[Dict[str, Any]] = [{"range": {"visits_lifetime": {"gt": 0}}}]
    if _field_exists(cust_mappings, "sales_pickup_lifetime"):
        filters.append({"range": {"sales_pickup_lifetime": {"gt": 0}}})
    return filters


def _build_visit_frequency_bucket_filters(visits_field: str) -> Dict[str, Any]:
    filters: Dict[str, Any] = {}
    for key, _label, min_v, max_v in _VISIT_FREQUENCY_BUCKET_META:
        if min_v == 1 and max_v == 1:
            filters[key] = {"term": {visits_field: 1}}
        elif max_v is None:
            filters[key] = {"range": {visits_field: {"gte": min_v}}}
        else:
            filters[key] = {"range": {visits_field: {"gte": min_v, "lte": max_v}}}
    return filters


def _format_visit_frequency_rows(
    buckets: Dict[str, Any],
    *,
    visits_field: str,
    window_days: Optional[int] = None,
) -> List[Dict[str, Any]]:
    total_customers = sum(int((buckets.get(k) or {}).get("doc_count") or 0) for k, *_ in _VISIT_FREQUENCY_BUCKET_META)
    if total_customers <= 0:
        return []

    rows: List[Dict[str, Any]] = []
    for key, label, min_v, max_v in _VISIT_FREQUENCY_BUCKET_META:
        count = int((buckets.get(key) or {}).get("doc_count") or 0)
        pct = (count * 100.0 / float(total_customers)) if total_customers else 0.0
        row: Dict[str, Any] = {
            "segment": key,
            "label": label,
            f"min_{visits_field}": min_v,
            f"max_{visits_field}": max_v,
            "customer_count": count,
            "percentage_of_customers": pct,
        }
        if window_days is not None:
            row["window_days"] = int(window_days)
        rows.append(row)

    return rows


def _es_visit_frequency_common(
    req,
    client,
    cust_index: str,
    cust_mappings: Dict[str, Any],
    *,
    visits_field: str,
    agg_name: str,
    metric_label: str,
    business_rules: Optional[str],
    window_days: Optional[int] = None,
    min_signup_age_days: int = 180,
) -> Dict[str, Any]:
    """
    Shared implementation used by “Visit Frequency – 730 Days” (and can be reused elsewhere).
    """
    required = ["original_signup", visits_field]
    if window_days is not None:
        required.append("last_visit")

    missing = [f for f in required if not _field_exists(cust_mappings, f)]
    if missing:
        return _es_cannot_answer(
            f"Cannot compute '{metric_label}' because required fields are missing "
            f"from customers index '{cust_index}': {', '.join(missing)}.",
            business_rules,
        )

    today = datetime.now(timezone.utc).date()
    cutoff_signup_str = (today - timedelta(days=int(min_signup_age_days))).isoformat()

    base_filters: List[Dict[str, Any]] = [
        {"exists": {"field": visits_field}},
        {"range": {visits_field: {"gt": 0}}},
        {"exists": {"field": "original_signup"}},
    ]

    if window_days is not None:
        cutoff_last_visit_str = (today - timedelta(days=int(window_days))).isoformat()
        base_filters.extend(
            [
                {"exists": {"field": "last_visit"}},
                {"range": {"last_visit": {"gte": cutoff_last_visit_str}}},
            ]
        )

    exclusion_clause: Dict[str, Any] = {
        "bool": {
            "filter": [
                {"term": {visits_field: 1}},
                {"range": {"original_signup": {"gte": cutoff_signup_str}}},
            ]
        }
    }

    body: Dict[str, Any] = {
        "size": 0,
        "query": {"bool": {"filter": base_filters, "must_not": [exclusion_clause]}},
        "aggs": {agg_name: {"filters": {"filters": _build_visit_frequency_bucket_filters(visits_field)}}},
    }

    res = _safe_es_search(client, index=cust_index, body=body)
    buckets = ((res.get("aggregations") or {}).get(agg_name) or {}).get("buckets") or {}

    rows = _format_visit_frequency_rows(buckets, visits_field=visits_field, window_days=window_days)
    if not rows:
        return {
            "insight": to_json_safe(f"{metric_label} found zero customers after applying filters/exclusion."),
            "rows": [],
            "rules_used": business_rules or "",
            "engine": "es",
        }

    if window_days is None:
        insight = (
            f"{metric_label} computed from customers.{visits_field}, excluding new single-visit customers "
            f"(original_signup within last {min_signup_age_days} days AND {visits_field}=1)."
        )
    else:
        insight = (
            f"{metric_label} computed from customers.{visits_field}, restricted to customers with last_visit in the last "
            f"{window_days} days, excluding new single-visit customers "
            f"(original_signup within last {min_signup_age_days} days AND {visits_field}=1)."
        )

    return {
        "insight": to_json_safe(insight),
        "rows": to_json_safe(rows),
        "rules_used": business_rules or "",
        "engine": "es",
    }


# -------------------------------------------------------------------
# Segmentation metrics
# -------------------------------------------------------------------


def _es_visit_frequency_distribution(req, client, mappings: Dict[str, Any], business_rules: Optional[str]):
    """
    Distribution of customers by visits_lifetime bucket:
      1, 2–5, 6–11, 12+

    Uses customers.visits_lifetime (one doc per customer).
    """
    ctx = _customers_ctx_or_error(
        req,
        client,
        business_rules,
        metric_name="Visit frequency distribution",
        required_fields=["customer_id", "visits_lifetime"],
    )
    if isinstance(ctx, dict):
        return ctx
    customers_index, cust_mappings = ctx

    base_filters = _customers_visits_lifetime_positive_filters(cust_mappings)

    body = {
        "size": 0,
        "query": {"bool": {"filter": base_filters}},
        "aggs": {
            "total": {"value_count": {"field": "customer_id"}},
            "buckets": {
                "filters": {
                    "filters": {
                        "1_visit": {"term": {"visits_lifetime": 1}},
                        "2_5": {"range": {"visits_lifetime": {"gte": 2, "lte": 5}}},
                        "6_11": {"range": {"visits_lifetime": {"gte": 6, "lte": 11}}},
                        "12_plus": {"range": {"visits_lifetime": {"gte": 12}}},
                    }
                }
            },
        },
    }

    res = _safe_es_search(client, index=customers_index, body=body)
    aggs = res.get("aggregations") or {}
    total = int((aggs.get("total") or {}).get("value") or 0)
    b = (aggs.get("buckets") or {}).get("buckets") or {}

    def _row(key: str, label: str) -> Dict[str, Any]:
        c = int((b.get(key) or {}).get("doc_count") or 0)
        pct = (100.0 * c / total) if total else 0.0
        return {"frequency_bucket": label, "customer_count": c, "percentage_of_customers": pct}

    rows = [
        _row("1_visit", "1 visit"),
        _row("2_5", "2–5 visits"),
        _row("6_11", "6–11 visits"),
        _row("12_plus", "12+ visits"),
    ]

    return {
        "insight": to_json_safe("Visit frequency distribution computed from customers.visits_lifetime."),
        "rows": to_json_safe(rows),
        "rules_used": business_rules or "",
        "engine": "es",
    }


def _es_customers_nth_visit_in_period(req, client, mappings: Dict[str, Any], business_rules: Optional[str]):
    """
    Windowed: count customers with >=2/3/4/5 DISTINCT visits (visit_id)
    inside req.start_date..req.end_date, using invoices index (dropoff_at).

    ✅ Alias-safe: resolves invoices concrete index + mappings via shared_utilities.
    ✅ Reuses existing invoices mappings if caller already fetched them and passed as `mappings`.
    """
    if not (getattr(req, "es_index_name", "") or "").strip():
        return {
            "insight": to_json_safe("Missing invoices index (es_index_name)."),
            "rows": [],
            "rules_used": business_rules or "",
            "engine": "es",
        }

    customer_field = "customer_id"
    date_field = "dropoff_at"
    visit_field = "visit_id"

    invoices_index, invoices_mappings = _get_invoice_index_and_mappings(
        client,
        req.es_index_name,
        existing_mappings=mappings,
        existing_index=req.es_index_name,
    )

    missing = [f for f in (customer_field, date_field, visit_field) if not _field_exists(invoices_mappings, f)]
    if missing:
        return {
            "insight": to_json_safe(f"Missing required fields for windowed nth-visit metric: {', '.join(missing)}."),
            "rows": [],
            "rules_used": business_rules or "",
            "engine": "es",
        }

    if not getattr(req, "start_date", None) and not getattr(req, "end_date", None):
        return {
            "insight": to_json_safe("Provide start_date/end_date to compute windowed Nth-visit counts."),
            "rows": [],
            "rules_used": business_rules or "",
            "engine": "es",
        }

    rb: Dict[str, Any] = {}
    if getattr(req, "start_date", None):
        rb["gte"] = req.start_date
    if getattr(req, "end_date", None):
        rb["lte"] = req.end_date

    filters = [
        {"exists": {"field": customer_field}},
        {"exists": {"field": visit_field}},
        {"range": {date_field: rb}},
    ]

    after_key = None
    ge2 = ge3 = ge4 = ge5 = 0
    total_customers = 0

    max_pages = _get_req_int(req, "nth_visit_max_pages", 200, min_v=1, max_v=2000)
    page_size = _get_req_int(req, "nth_visit_page_size", 1000, min_v=100, max_v=5000)
    pages = 0

    while True:
        if pages >= max_pages:
            break
        pages += 1

        comp = {"size": page_size, "sources": [{"cid": {"terms": {"field": customer_field}}}]}
        if after_key:
            comp["after"] = after_key

        body = {
            "size": 0,
            "track_total_hits": False,
            "query": {"bool": {"filter": filters}},
            "aggs": {
                "customers": {
                    "composite": comp,
                    "aggs": {"vcount": {"cardinality": {"field": visit_field}}},
                }
            },
        }

        res = _safe_es_search(client, index=invoices_index, body=body)
        agg = (res.get("aggregations") or {}).get("customers") or {}
        buckets = agg.get("buckets") or []
        if not buckets:
            break

        for b in buckets:
            total_customers += 1
            v = int(((b.get("vcount") or {}).get("value")) or 0)
            if v >= 2:
                ge2 += 1
            if v >= 3:
                ge3 += 1
            if v >= 4:
                ge4 += 1
            if v >= 5:
                ge5 += 1

        after_key = agg.get("after_key")
        if not after_key:
            break

    rows = [
        {"metric": "customers_2plus_visits_period", "label": "Customers with ≥2 visits (in period)", "value": ge2},
        {"metric": "customers_3plus_visits_period", "label": "Customers with ≥3 visits (in period)", "value": ge3},
        {"metric": "customers_4plus_visits_period", "label": "Customers with ≥4 visits (in period)", "value": ge4},
        {"metric": "customers_5plus_visits_period", "label": "Customers with ≥5 visits (in period)", "value": ge5},
        {"metric": "total_customers_period", "label": "Customers with ≥1 visit (in period)", "value": total_customers},
        {"metric": "pages_scanned", "label": "Composite pages scanned", "value": pages},
        {"metric": "invoices_index_used", "label": "Invoices index used", "value": invoices_index},
    ]

    return {
        "insight": to_json_safe("Windowed Nth-visit counts computed from DISTINCT visit_id inside the selected period."),
        "rows": to_json_safe(rows),
        "rules_used": business_rules or "",
        "engine": "es",
    }


def _es_customers_nth_visit(req, client, mappings: Dict[str, Any], business_rules: Optional[str]):
    """
    Lifetime counts of customers who have reached >=2/3/4/5 visits,
    using customers.visits_lifetime.
    """
    ctx = _customers_ctx_or_error(
        req,
        client,
        business_rules,
        metric_name="Customers Achieving Nth Visit",
        required_fields=["customer_id", "visits_lifetime"],
    )
    if isinstance(ctx, dict):
        return ctx
    customers_index, cust_mappings = ctx

    base_filters = _customers_visits_lifetime_positive_filters(cust_mappings)

    body = {
        "size": 0,
        "query": {"bool": {"filter": base_filters}},
        "aggs": {
            "total": {"value_count": {"field": "customer_id"}},
            "ge2": {"filter": {"range": {"visits_lifetime": {"gte": 2}}}},
            "ge3": {"filter": {"range": {"visits_lifetime": {"gte": 3}}}},
            "ge4": {"filter": {"range": {"visits_lifetime": {"gte": 4}}}},
            "ge5": {"filter": {"range": {"visits_lifetime": {"gte": 5}}}},
        },
    }

    res = _safe_es_search(client, index=customers_index, body=body)
    aggs = res.get("aggregations") or {}

    total = int((aggs.get("total") or {}).get("value") or 0)
    c2 = int((aggs.get("ge2") or {}).get("doc_count") or 0)
    c3 = int((aggs.get("ge3") or {}).get("doc_count") or 0)
    c4 = int((aggs.get("ge4") or {}).get("doc_count") or 0)
    c5 = int((aggs.get("ge5") or {}).get("doc_count") or 0)

    rows = [
        {"metric": "customers_2plus_visits", "label": "Customers Achieving 2nd Visit (≥2 visits)", "value": c2},
        {"metric": "customers_3plus_visits", "label": "Customers Achieving 3rd Visit (≥3 visits)", "value": c3},
        {"metric": "customers_4plus_visits", "label": "Customers Achieving 4th Visit (≥4 visits)", "value": c4},
        {"metric": "customers_5plus_visits", "label": "Customers Achieving 5th Visit (≥5 visits)", "value": c5},
        {"metric": "total_customers_lifetime", "label": "Total Customers (visits_lifetime > 0)", "value": total},
    ]

    return {
        "insight": to_json_safe("Customers achieving Nth visit computed from customers.visits_lifetime."),
        "rows": to_json_safe(rows),
        "rules_used": business_rules or "",
        "engine": "es",
    }


# -------------------------------------------------------------------
# Revenue concentration using CUSTOMERS index (tag_id-based)
# -------------------------------------------------------------------


def _resolve_tag_ids_by_name(
    client: Any,
    *,
    customers_index: str,
    tags_name_kw_field: str,
    tags_id_field: str,
    name_variants: List[str],
) -> List[int]:
    """
    Resolve tag_id(s) by exact match on tags.name.keyword (nested).
    Returns sorted unique ints (may be empty).
    """
    body = {
        "size": 0,
        "aggs": {
            "tags_nested": {
                "nested": {"path": "tags"},
                "aggs": {
                    "by_name": {
                        "filter": {"terms": {tags_name_kw_field: name_variants}},
                        "aggs": {"ids": {"terms": {"field": tags_id_field, "size": 50}}},
                    }
                },
            }
        },
    }

    res = _safe_es_search(client, index=customers_index, body=body)
    buckets = (
        ((((res.get("aggregations") or {}).get("tags_nested") or {}).get("by_name") or {}).get("ids") or {})
        .get("buckets")
        or []
    )

    out: List[int] = []
    for b in buckets:
        k = b.get("key")
        if k is None:
            continue
        try:
            out.append(int(k))
        except Exception:
            continue
    return sorted(set(out))


def _get_req_tag_ids(req: Any, *, plural_attr: str, single_attr: str) -> List[int]:
    """
    Priority:
      1) req.<plural_attr> (iterable)
      2) req.<single_attr> (single)
    """
    raw_ids = getattr(req, plural_attr, None)
    if raw_ids:
        out: List[int] = []
        try:
            for x in raw_ids:
                if x is None:
                    continue
                out.append(int(x))
        except Exception:
            out = []
        if out:
            return sorted(set(out))

    raw_one = getattr(req, single_attr, None)
    if raw_one is not None:
        try:
            return [int(raw_one)]
        except Exception:
            pass

    return []


def _es_top_revenue_share_customers_index(
    req: Any,
    client: Any,
    mappings: Dict[str, Any],  # kept for router signature consistency (often invoices mappings)
    business_rules: Optional[str],
    *,
    which: str,  # "top5" or "top20"
) -> Dict[str, Any]:
    """
    Revenue concentration using CUSTOMERS index only.
    """
    ctx = _customers_ctx_or_error(
        req,
        client,
        business_rules,
        metric_name="Top revenue share",
        required_fields=["customer_id"],
    )
    if isinstance(ctx, dict):
        return ctx
    customers_index, cust_mappings = ctx

    revenue_field = None
    if _field_exists(cust_mappings, "sales_pickup_lifetime"):
        revenue_field = "sales_pickup_lifetime"
    elif _field_exists(cust_mappings, "sales_pickup_365"):
        revenue_field = "sales_pickup_365"

    if not revenue_field:
        return _es_cannot_answer(
            f"Cannot compute revenue share because neither 'sales_pickup_lifetime' nor 'sales_pickup_365' exists in '{customers_index}'.",
            business_rules,
        )

    if not (_field_exists(cust_mappings, "tags") and _field_exists(cust_mappings, "tags.tag_id")):
        return _es_cannot_answer(
            f"Cannot compute revenue share because 'tags.tag_id' (nested) is missing in customers index '{customers_index}'.",
            business_rules,
        )

    tags_id_field = "tags.tag_id"
    tags_name_kw_field = "tags.name.keyword"

    if which == "top5":
        tag_ids = _get_req_tag_ids(req, plural_attr="top5_tag_ids", single_attr="top5_tag_id")
        name_variants = ["Top 5%", "TOP 5%", "top 5%"]
        metric_key = "top5_revenue_share_pct"
        label = "Top 5% revenue share (%)"
    else:
        tag_ids = _get_req_tag_ids(req, plural_attr="top20_tag_ids", single_attr="top20_tag_id")
        name_variants = ["Top 20%", "TOP 20%", "top 20%"]
        metric_key = "top20_revenue_share_pct"
        label = "Top 20% revenue share (%)"

    if not tag_ids and _field_exists(cust_mappings, tags_name_kw_field):
        tag_ids = _resolve_tag_ids_by_name(
            client,
            customers_index=customers_index,
            tags_name_kw_field=tags_name_kw_field,
            tags_id_field=tags_id_field,
            name_variants=name_variants,
        )

    if not tag_ids:
        return _es_cannot_answer(
            f"Cannot compute {label} because no tag_id was provided on the request and it could not be resolved from tags.name.keyword.",
            business_rules,
        )

    base_filters: List[Dict[str, Any]] = [
        {"exists": {"field": "customer_id"}},
        {"range": {revenue_field: {"gt": 0}}},
    ]

    top_filter = {
        "nested": {
            "path": "tags",
            "ignore_unmapped": True,
            "query": {"terms": {tags_id_field: tag_ids}},
        }
    }

    body: Dict[str, Any] = {
        "size": 0,
        "query": {"bool": {"filter": base_filters}},
        "aggs": {
            "total_revenue": {"sum": {"field": revenue_field}},
            "total_customers": {"value_count": {"field": "customer_id"}},
            "top_group": {
                "filter": top_filter,
                "aggs": {
                    "revenue": {"sum": {"field": revenue_field}},
                    "customers": {"value_count": {"field": "customer_id"}},
                },
            },
        },
    }

    res = _safe_es_search(client, index=customers_index, body=body)
    aggs = res.get("aggregations") or {}

    total_rev = float((aggs.get("total_revenue") or {}).get("value") or 0.0)
    total_n = int((aggs.get("total_customers") or {}).get("value") or 0)

    tg = aggs.get("top_group") or {}
    top_rev = float(((tg.get("revenue") or {}).get("value")) or 0.0)
    top_n = int(((tg.get("customers") or {}).get("value")) or 0)

    if total_rev <= 0:
        return _es_cannot_answer(f"Cannot compute {label} because total revenue is 0 after base filters.", business_rules)

    share_pct = (top_rev * 100.0) / total_rev

    rows = [
        {
            "metric": metric_key,
            "label": label,
            "value": float(share_pct),
            "top_customers_tagged": top_n,
            "total_paying_customers": total_n,
            "top_revenue": float(top_rev),
            "total_revenue": float(total_rev),
            "revenue_field_used": revenue_field,
            "customers_index": customers_index,
            "tag_ids_used": tag_ids,
        }
    ]

    insight = (
        f"{label} computed from customers index '{customers_index}' using '{revenue_field}' over paying customers "
        f"(revenue>0), filtered by nested tags.tag_id."
    )
    return {"insight": to_json_safe(insight), "rows": to_json_safe(rows), "rules_used": business_rules or "", "engine": "es"}


def _es_top5_revenue_share(req, client, mappings: Dict[str, Any], business_rules: Optional[str]) -> Dict[str, Any]:
    return _es_top_revenue_share_customers_index(req, client, mappings, business_rules, which="top5")


def _es_top20_revenue_share(req, client, mappings: Dict[str, Any], business_rules: Optional[str]) -> Dict[str, Any]:
    return _es_top_revenue_share_customers_index(req, client, mappings, business_rules, which="top20")


# -------------------------------------------------------------------
# Pareto + Top customers (legacy / invoices-derived)
# -------------------------------------------------------------------


def _es_pareto_80_20(req, client, mappings: Dict[str, Any], business_rules: Optional[str]):
    """
    80/20 Rule (Pareto) for revenue concentration (customers index).
    """
    ctx = _customers_ctx_or_error(
        req,
        client,
        business_rules,
        metric_name="Pareto 80/20 Rule",
        required_fields=["customer_id", "sales_pickup_lifetime"],
    )
    if isinstance(ctx, dict):
        return ctx
    customers_index, cust_mappings = ctx

    pareto_target_pct = _get_req_int(req, "pareto_target_share", 80, min_v=1, max_v=99)
    pareto_target_fraction = pareto_target_pct / 100.0

    max_customers_for_pareto = _get_req_int(
        req,
        "max_customers_for_pareto",
        ES_MAX_CUSTOMERS_DEFAULT,
        min_v=100,
        max_v=200_000,
    )

    base_filter = [
        {"exists": {"field": "sales_pickup_lifetime"}},
        {"range": {"sales_pickup_lifetime": {"gt": 0}}},
    ]

    totals_body: Dict[str, Any] = {
        "size": 0,
        "query": {"bool": {"filter": base_filter}},
        "aggs": {
            "total_revenue": {"sum": {"field": "sales_pickup_lifetime"}},
            "total_customers": {"value_count": {"field": "customer_id"}},
        },
    }

    totals_res = _safe_es_search(client, index=customers_index, body=totals_body)
    totals_aggs = totals_res.get("aggregations") or {}

    total_revenue = float((totals_aggs.get("total_revenue") or {}).get("value") or 0.0)
    total_customers = int((totals_aggs.get("total_customers") or {}).get("value") or 0)

    if total_customers == 0 or total_revenue <= 0:
        return {
            "insight": to_json_safe(
                "Pareto could not be computed because no customers with sales_pickup_lifetime > 0 were found."
            ),
            "rows": [],
            "rules_used": business_rules or "",
            "engine": "es",
        }

    hard_cap = 10_000
    limit = min(total_customers, max_customers_for_pareto, hard_cap)
    approximate_due_to_cap = (max_customers_for_pareto > hard_cap) or (total_customers > hard_cap)

    top_body: Dict[str, Any] = {
        "size": int(limit),
        "query": {"bool": {"filter": base_filter}},
        "sort": [{"sales_pickup_lifetime": {"order": "desc"}}],
        "_source": ["customer_id", "sales_pickup_lifetime"],
    }

    top_res = _safe_es_search(client, index=customers_index, body=top_body)
    hits = (top_res.get("hits") or {}).get("hits") or []

    if not hits:
        return {
            "insight": to_json_safe("Pareto could not be computed because top customers could not be fetched."),
            "rows": [],
            "rules_used": business_rules or "",
            "engine": "es",
        }

    target_revenue = total_revenue * pareto_target_fraction
    cumulative = 0.0
    customers_used = 0

    for h in hits:
        src = h.get("_source") or {}
        rev = src.get("sales_pickup_lifetime")
        try:
            rev_val = float(rev)
        except Exception:
            continue
        if rev_val <= 0:
            continue
        cumulative += rev_val
        customers_used += 1
        if cumulative >= target_revenue:
            break

    reached_target = cumulative >= target_revenue
    approximate = (not reached_target) or approximate_due_to_cap

    if customers_used == 0:
        return {
            "insight": to_json_safe("Pareto could not be computed because no valid customer revenues were found."),
            "rows": [],
            "rules_used": business_rules or "",
            "engine": "es",
        }

    pareto_pct = (customers_used / total_customers) * 100.0

    insight = (
        f"Pareto {pareto_target_pct}/100: ~{pareto_pct:.1f}% of customers (top {customers_used} of {total_customers}) "
        f"reach {pareto_target_pct}% of lifetime revenue."
    )
    if approximate:
        insight += " Result is approximate (target not fully reached or capped fetch size)."

    rows = [
        {
            "metric": "pareto_customers_pct",
            "label": f"Customers Share to Reach {pareto_target_pct}% Revenue",
            "value": pareto_pct,
            "pareto_target_pct": float(pareto_target_pct),
            "customers_with_revenue": int(total_customers),
            "customers_to_reach_target": int(customers_used),
            "total_revenue": float(total_revenue),
            "max_customers_for_pareto": int(max_customers_for_pareto),
            "approximate": bool(approximate),
            "customers_index": customers_index,
        }
    ]

    return {
        "insight": to_json_safe(insight),
        "rows": to_json_safe(rows),
        "rules_used": business_rules or "",
        "engine": "es",
    }


def _es_top_customers_by_revenue(req, client, mappings: Dict[str, Any], business_rules: Optional[str]):
    """
    Top 5% / Top 20% customers by revenue (invoice-derived stats).
    NOTE: Requires full per-customer revenue list in Python (can be heavy).

    ✅ Uses shared_utilities invoice resolver (alias-safe + reuse).
    ✅ Option A: no visit_count anywhere.
    """
    if not (getattr(req, "es_index_name", "") or "").strip():
        return _es_cannot_answer("Missing invoices index (es_index_name).", business_rules)

    invoices_index, invoices_mappings = _get_invoice_index_and_mappings(
        client,
        req.es_index_name,
        existing_mappings=mappings,
        existing_index=req.es_index_name,
    )
    max_rows = _get_req_int(req, "es_max_rows", 500, min_v=50, max_v=5000)

    stats = _get_customer_stats_invoices_only(req, client, invoices_index, invoices_mappings)
    if stats is None:
        return _es_cannot_answer(
            "Cannot compute 'Top 5% / Top 20% customers by revenue' because required invoices fields "
            "(customer_id, dropoff_at, total) are missing or could not be derived from the invoices mappings.",
            business_rules,
        )

    revenue_stats = [s for s in stats if (s.get("total_revenue") is not None and float(s.get("total_revenue") or 0) > 0)]
    if not revenue_stats:
        return {
            "insight": to_json_safe("No customers with positive total revenue were found."),
            "rows": [],
            "rules_used": business_rules or "",
            "engine": "es",
        }

    revenue_stats.sort(key=lambda s: float(s.get("total_revenue") or 0.0), reverse=True)
    n = len(revenue_stats)
    top5_n = max(1, int(round(0.05 * n)))
    top20_n = max(top5_n, int(round(0.20 * n)))

    top5 = revenue_stats[:top5_n]
    top20 = revenue_stats[:top20_n]

    total_revenue_all = sum(float(s.get("total_revenue") or 0.0) for s in revenue_stats)
    top5_revenue = sum(float(s.get("total_revenue") or 0.0) for s in top5)
    top20_revenue = sum(float(s.get("total_revenue") or 0.0) for s in top20)

    share5 = (100.0 * top5_revenue / total_revenue_all) if total_revenue_all > 0 else 0.0
    share20 = (100.0 * top20_revenue / total_revenue_all) if total_revenue_all > 0 else 0.0

    rows: List[Dict[str, Any]] = []
    shown = min(top20_n, max_rows)
    for idx, s in enumerate(top20[:shown]):
        group = "top_5_percent" if idx < top5_n else "top_20_percent"
        rows.append(
            {
                "customer_id": s.get("customer_id"),
                "total_revenue": float(s.get("total_revenue") or 0.0),
                "group": group,
                "rank": idx + 1,
            }
        )

    insight = (
        f"Top 5% / Top 20% customers by revenue computed from invoice-derived stats. "
        f"Top 5% account for ~{share5:.1f}% of revenue; top 20% account for ~{share20:.1f}% of revenue. "
        f"Rows show {shown} customers for safety."
    )
    return {
        "insight": to_json_safe(insight),
        "rows": to_json_safe(rows),
        "rules_used": business_rules or "",
        "engine": "es",
    }


# -------------------------------------------------------------------
# Visit Frequency – 365/730
# -------------------------------------------------------------------


def _es_visit_frequency_365(req, client, mappings: Dict[str, Any], business_rules: Optional[str]):
    """
    Visit Frequency – 365 Days (customers index), based on customers.visits_365:
      Buckets: 1, 2–5, 6–11, 12+

    Excludes new single-visit customers:
      original_signup within last N days (default 180) AND visits_365 == 1

    ✅ Uses shared customers ctx loader (alias-safe + reuse).
    """
    ctx = _customers_ctx_or_error(
        req,
        client,
        business_rules,
        metric_name="Visit Frequency – 365 Days",
        required_fields=["customer_id", "original_signup", "visits_365"],
    )
    if isinstance(ctx, dict):
        return ctx
    customers_index, cust_mappings = ctx

    min_signup_age_days = _get_req_int(req, "min_signup_age_days_for_visit_frequency_365", 180, min_v=1, max_v=3650)
    today = datetime.now(timezone.utc).date()
    cutoff_signup_str = (today - timedelta(days=int(min_signup_age_days))).isoformat()

    base_filters: List[Dict[str, Any]] = [
        {"exists": {"field": "customer_id"}},
        {"exists": {"field": "original_signup"}},
        {"exists": {"field": "visits_365"}},
        {"range": {"visits_365": {"gt": 0}}},
    ]

    exclusion = {
        "bool": {
            "filter": [
                {"term": {"visits_365": 1}},
                {"range": {"original_signup": {"gte": cutoff_signup_str}}},
            ]
        }
    }

    bucket_filters = {
        "1": {"term": {"visits_365": 1}},
        "2_5": {"range": {"visits_365": {"gte": 2, "lte": 5}}},
        "6_11": {"range": {"visits_365": {"gte": 6, "lte": 11}}},
        "12_plus": {"range": {"visits_365": {"gte": 12}}},
    }

    body = {
        "size": 0,
        "query": {"bool": {"filter": base_filters, "must_not": [exclusion]}},
        "aggs": {
            "total_customers": {"value_count": {"field": "customer_id"}},
            "visit_frequency": {"filters": {"filters": bucket_filters}},
        },
    }

    res = _safe_es_search(client, index=customers_index, body=body)
    aggs = res.get("aggregations") or {}
    total = int((aggs.get("total_customers") or {}).get("value") or 0)
    buckets = ((aggs.get("visit_frequency") or {}).get("buckets")) or {}

    if total == 0:
        return {
            "insight": to_json_safe("No customers matched visits_365 > 0 (after exclusions)."),
            "rows": [],
            "rules_used": business_rules or "",
            "engine": "es",
        }

    def _row(key: str, label: str) -> Dict[str, Any]:
        c = int((buckets.get(key) or {}).get("doc_count") or 0)
        pct = (100.0 * c / total) if total else 0.0
        return {"frequency_bucket": label, "customer_count": c, "percentage_of_customers": pct}

    rows = [
        _row("1", "1 visit (365d)"),
        _row("2_5", "2–5 visits (365d)"),
        _row("6_11", "6–11 visits (365d)"),
        _row("12_plus", "12+ visits (365d)"),
    ]

    insight = (
        f"Visit frequency (365d) computed from customers.visits_365, excluding new single-visit customers "
        f"(original_signup within last {min_signup_age_days} days AND visits_365=1)."
    )

    return {"insight": to_json_safe(insight), "rows": to_json_safe(rows), "rules_used": business_rules or "", "engine": "es"}


def _es_visit_frequency_730(req, client, mappings: Dict[str, Any], business_rules: Optional[str]):
    """
    Visit Frequency – 730 Days (customers index).
    """
    ctx = _customers_ctx_or_error(req, client, business_rules, metric_name="Visit Frequency – 730 Days")
    if isinstance(ctx, dict):
        return ctx
    customers_index, cust_mappings = ctx

    window_days = _get_req_int(req, "visit_frequency_730_window_days", 730, min_v=30, max_v=3650)
    min_signup_age_days = _get_req_int(req, "min_signup_age_days_for_visit_frequency_730", 180, min_v=1, max_v=3650)

    return _es_visit_frequency_common(
        req,
        client,
        customers_index,
        cust_mappings,
        visits_field="visits_lifetime",
        agg_name="visit_frequency_730",
        metric_label="Visit Frequency – 730 Days",
        business_rules=business_rules,
        window_days=window_days,
        min_signup_age_days=min_signup_age_days,
    )


# -------------------------------------------------------------------
# Route vs retail + retail target segment
# -------------------------------------------------------------------


def _es_route_vs_retail_comparison(req, client, mappings: Dict[str, Any], business_rules: Optional[str]):
    """
    Route vs Retail Comparison (customers index).
    """
    ctx = _customers_ctx_or_error(req, client, business_rules, metric_name="Route vs Retail Comparison")
    if isinstance(ctx, dict):
        return ctx
    customers_index, cust_mappings = ctx

    signup_field: Optional[str] = None
    for cand in ("original_sign_up", "original_signup", "signup_date"):
        if _field_exists(cust_mappings, cand):
            signup_field = cand
            break

    pieces_field: Optional[str] = None
    for cand in ("pieces_lifetime", "pieces_pickup_lifetime", "total_pieces_lifetime", "total_pieces"):
        if _field_exists(cust_mappings, cand):
            pieces_field = cand
            break

    required_fields = ["customer_id", "sales_pickup_lifetime", "visits_lifetime", "last_visit"]
    missing_required = [f for f in required_fields if not _field_exists(cust_mappings, f)]
    if missing_required or not signup_field:
        msg_parts = []
        if missing_required:
            msg_parts.append(f"missing required fields: {', '.join(missing_required)}")
        if not signup_field:
            msg_parts.append("missing signup field (original_signup/original_sign_up/signup_date)")
        return _es_cannot_answer(
            "Cannot compute 'Route vs Retail Comparison' because " + "; ".join(msg_parts) + ".",
            business_rules,
        )

    base_filter: List[Dict[str, Any]] = [
        {"exists": {"field": "sales_pickup_lifetime"}},
        {"range": {"sales_pickup_lifetime": {"gt": 0}}},
        {"exists": {"field": "visits_lifetime"}},
        {"range": {"visits_lifetime": {"gt": 0}}},
        {"exists": {"field": signup_field}},
        {"exists": {"field": "last_visit"}},
    ]

    ROUTE_PATH = "route"
    route_name_field = "route.name.keyword"
    route_id_field = "route.route_id"

    retail_names = ["Retail", "retail", "Unassigned", "unassigned"]
    retail_name_query = {"terms": {route_name_field: retail_names}}

    retail_filter: Dict[str, Any] = {
        "bool": {
            "should": [
                {"nested": {"path": ROUTE_PATH, "query": retail_name_query}},
                {
                    "bool": {
                        "must_not": [
                            {"nested": {"path": ROUTE_PATH, "query": {"exists": {"field": route_id_field}}}}
                        ]
                    }
                },
            ],
            "minimum_should_match": 1,
        }
    }

    route_filter: Dict[str, Any] = {
        "nested": {
            "path": ROUTE_PATH,
            "query": {
                "bool": {
                    "filter": [{"exists": {"field": route_id_field}}],
                    "must_not": [retail_name_query],
                }
            },
        }
    }

    seg_aggs: Dict[str, Any] = {
        "customer_count": {"value_count": {"field": "customer_id"}},
        "segment_revenue": {"sum": {"field": "sales_pickup_lifetime"}},
        "total_visits": {"sum": {"field": "visits_lifetime"}},
        "avg_last_visit": {"avg": {"field": "last_visit"}},
        "avg_signup": {"avg": {"field": signup_field}},
    }
    if pieces_field:
        seg_aggs["total_pieces"] = {"sum": {"field": pieces_field}}

    body: Dict[str, Any] = {
        "size": 0,
        "query": {"bool": {"filter": base_filter}},
        "aggs": {
            "segments": {"filters": {"filters": {"retail": retail_filter, "route": route_filter}}, "aggs": seg_aggs},
            "total_revenue": {"sum": {"field": "sales_pickup_lifetime"}},
        },
    }

    res = _safe_es_search(client, index=customers_index, body=body)
    aggs = res.get("aggregations") or {}
    seg_buckets = (aggs.get("segments") or {}).get("buckets") or {}
    total_revenue = float((aggs.get("total_revenue") or {}).get("value") or 0.0)

    ms_per_day = 1000.0 * 60 * 60 * 24

    def summarize(segment_key: str, label: str) -> Dict[str, Any]:
        b = seg_buckets.get(segment_key) or {}
        cust_count = float((b.get("customer_count") or {}).get("value") or 0.0)
        seg_revenue = float((b.get("segment_revenue") or {}).get("value") or 0.0)
        total_visits = float((b.get("total_visits") or {}).get("value") or 0.0)
        avg_last_ms = (b.get("avg_last_visit") or {}).get("value")
        avg_signup_ms = (b.get("avg_signup") or {}).get("value")

        total_pieces = None
        if pieces_field:
            total_pieces = float((b.get("total_pieces") or {}).get("value") or 0.0)

        avg_ltv = (seg_revenue / cust_count) if cust_count > 0 else 0.0
        avg_visit_value = (seg_revenue / total_visits) if total_visits > 0 else 0.0
        revenue_share_pct = (seg_revenue * 100.0 / total_revenue) if total_revenue > 0 else 0.0

        avg_pieces_per_visit = None
        if pieces_field and total_pieces is not None and total_visits > 0:
            avg_pieces_per_visit = total_pieces / total_visits

        visits_per_year = 0.0
        if cust_count > 0 and total_visits > 0 and (avg_last_ms is not None) and (avg_signup_ms is not None):
            avg_lifespan_days = max(1.0, (avg_last_ms - avg_signup_ms) / ms_per_day)
            years = max(0.01, avg_lifespan_days / 365.0)
            avg_visits_per_customer = total_visits / cust_count
            visits_per_year = avg_visits_per_customer / years

        return {
            "segment_id": segment_key,
            "segment_label": label,
            "customer_count": int(cust_count),
            "revenue": seg_revenue,
            "revenue_share_pct": revenue_share_pct,
            "avg_ltv": avg_ltv,
            "avg_visit_value": avg_visit_value,
            "avg_pieces_per_visit": avg_pieces_per_visit,
            "visits_per_year": visits_per_year,
        }

    rows: List[Dict[str, Any]] = [summarize("retail", "Retail"), summarize("route", "Route Customers")]

    return {
        "insight": to_json_safe(
            f"Route vs Retail comparison computed on customers index '{customers_index}' using nested route segmentation."
        ),
        "rows": to_json_safe(rows),
        "rules_used": business_rules or "",
        "engine": "es",
    }


def _es_high_value_retail_targets(req, client, mappings: Dict[str, Any], business_rules: Optional[str]):
    """
    High-Value Retail Targets.
    """
    ctx = _customers_ctx_or_error(
        req,
        client,
        business_rules,
        metric_name="High-Value Retail Targets",
        required_fields=["customer_id", "visit_average_sales", "route.name"],
    )
    if isinstance(ctx, dict):
        return ctx
    customers_index, cust_mappings = ctx

    route_name_field = "route.name.keyword" if _field_exists(cust_mappings, "route.name.keyword") else "route.name"

    retail_name_query = (
        {"term": {route_name_field: "Retail"}}
        if route_name_field.endswith(".keyword")
        else {"match_phrase": {"route.name": "Retail"}}
    )

    retail_nested = {"nested": {"path": "route", "query": retail_name_query}}
    no_route = {"bool": {"must_not": [{"nested": {"path": "route", "query": {"match_all": {}}}}]}}
    retail_filter = {"bool": {"should": [retail_nested, no_route], "minimum_should_match": 1}}

    base_filters = [
        {"exists": {"field": "customer_id"}},
        {"exists": {"field": "visit_average_sales"}},
        retail_filter,
    ]

    body = {
        "size": 0,
        "query": {"bool": {"filter": base_filters}},
        "aggs": {
            "total_retail_customers": {"value_count": {"field": "customer_id"}},
            "high_value_retail": {
                "filter": {"range": {"visit_average_sales": {"gte": 75}}},
                "aggs": {"customers": {"value_count": {"field": "customer_id"}}},
            },
        },
    }

    res = _safe_es_search(client, index=customers_index, body=body)
    aggs = res.get("aggregations") or {}

    total_retail = int((aggs.get("total_retail_customers") or {}).get("value") or 0)
    hv_bucket = aggs.get("high_value_retail") or {}
    high_value_count = int((hv_bucket.get("customers") or {}).get("value") or 0)

    if total_retail == 0:
        return {
            "insight": to_json_safe("High-Value Retail Targets could not be computed because no retail customers were found."),
            "rows": [],
            "rules_used": business_rules or "",
            "engine": "es",
        }

    pct = (high_value_count * 100.0 / float(total_retail))
    rows = [
        {
            "metric": "high_value_retail_targets",
            "label": "High-Value Retail Customers (visit_average_sales ≥ 75)",
            "value": high_value_count,
            "high_value_retail_pct": pct,
            "total_retail_customers": total_retail,
        }
    ]

    return {
        "insight": to_json_safe(
            f"High-Value Retail Targets computed on customers index '{customers_index}': {high_value_count} customers (~{pct:.1f}%)."
        ),
        "rows": to_json_safe(rows),
        "rules_used": business_rules or "",
        "engine": "es",
    }


__all__ = [
    "_es_visit_frequency_distribution",
    "_es_customers_nth_visit",
    "_es_customers_nth_visit_in_period",
    "_es_top5_revenue_share",
    "_es_top20_revenue_share",
    "_es_pareto_80_20",
    "_es_top_customers_by_revenue",
    "_es_visit_frequency_365",
    "_es_visit_frequency_730",
    "_es_route_vs_retail_comparison",
    "_es_high_value_retail_targets",
]
