from __future__ import annotations

from datetime import datetime, timezone, timedelta
from typing import Any, Dict, List, Optional, Iterable

from abi.runtime import to_json_safe
from routes.es_test import _extract_properties_from_mapping
from app.api.docs_analytics_routes import (
    resolve_es_field,
    _es_cannot_answer,
    _build_date_range_filter,
)
from app.api.metrics.shared_utilities import (
    _field_exists,
    _safe_es_search,
)

# -------------------------------------------------------------------
# Safety limits (server-friendly defaults)
# -------------------------------------------------------------------

DEFAULT_WINDOW_DAYS = 365
MAX_SCAN_HITS = 50_000
MAX_SCAN_PAGES = 200
MAX_COMPOSITE_PAGES = 500
MAX_TOP20_CUSTOMERS = 50_000
MAX_ROWS_RETURNED = 10_000
TERMS_CHUNK_SIZE = 1000


# -------------------------------------------------------------------
# Small ES-safe helpers (performance + robustness)
# -------------------------------------------------------------------

def _chunks(lst: List[Any], n: int) -> Iterable[List[Any]]:
    for i in range(0, len(lst), n):
        yield lst[i: i + n]


def _date_filters_or_default(req, date_field: str) -> tuple[List[Dict[str, Any]], str]:
    """
    If req.start_date/end_date missing => default last DEFAULT_WINDOW_DAYS.
    Returns (filters, window_label).
    """
    filters = _build_date_range_filter(req, date_field) or []
    if filters:
        parts: List[str] = []
        if getattr(req, "start_date", None):
            parts.append(f"from {req.start_date}")
        if getattr(req, "end_date", None):
            parts.append(f"to {req.end_date}")
        return filters, (" ".join(parts) if parts else "custom window")

    end = datetime.now(timezone.utc).date()
    start = end - timedelta(days=DEFAULT_WINDOW_DAYS)
    return (
        [
            {
                "range": {
                    date_field: {
                        "gte": start.isoformat(),
                        "lte": f"{end.isoformat()}T23:59:59",
                    }
                }
            }
        ],
        f"default last {DEFAULT_WINDOW_DAYS} days",
    )


def _scan_all_hits(
    client,
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

    ✅ IMPORTANT:
    - DO NOT sort by '_id' (ES blocks fielddata on _id by default).
    - Provide a deterministic sort on a real field when possible (recommended).
    - If sort is not provided, uses '_doc' as a best-effort fallback.
      (For large scans, prefer passing a stable business key like customer_id.)
    """
    effective_sort = sort if sort else [{"_doc": "asc"}]

    base_body: Dict[str, Any] = {
        "size": page_size,
        "query": query or {"match_all": {}},
        "sort": effective_sort,
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


def _composite_by_customer(
    client,
    *,
    index: str,
    filters: List[Dict[str, Any]],
    customer_field: str,
    sub_aggs: Dict[str, Any],
    page_size: int = 500,
    max_pages: int = MAX_COMPOSITE_PAGES,
) -> Iterable[Dict[str, Any]]:
    """
    Page through customers using composite aggregation with hard max_pages cap.
    """
    after_key = None
    pages = 0

    while True:
        if pages >= max_pages:
            break

        comp: Dict[str, Any] = {
            "size": page_size,
            "sources": [{"cid": {"terms": {"field": customer_field}}}],
        }
        if after_key:
            comp["after"] = after_key

        body = {
            "size": 0,
            "query": {"bool": {"filter": filters}},
            "aggs": {
                "by_customer": {
                    "composite": comp,
                    "aggs": sub_aggs,
                }
            },
        }

        res = _safe_es_search(client, index=index, body=body)
        agg = (res.get("aggregations") or {}).get("by_customer") or {}
        buckets = agg.get("buckets") or []
        if not buckets:
            break

        for b in buckets:
            yield b

        pages += 1
        after_key = agg.get("after_key")
        if not after_key:
            break


# -------------------------------------------------------------------
# ✅ mapping-aware field existence + safe field resolver
# -------------------------------------------------------------------
def _resolve_existing_field(
    mappings: Dict[str, Any],
    *,
    primary: str,
    fallback: Optional[str] = None,
    alias_family: Optional[str] = None,
) -> Optional[str]:
    """
    Resolve a field with resolve_es_field, but return it ONLY if it exists in mappings.
    """
    f = resolve_es_field(mappings, user_term=primary, alias_family=alias_family)
    if f and _field_exists(mappings, f):
        return f

    if fallback:
        f2 = resolve_es_field(mappings, user_term=fallback, alias_family=alias_family)
        if f2 and _field_exists(mappings, f2):
            return f2

    return None


def _maybe_keyword(mappings: Dict[str, Any], field: str) -> Optional[str]:
    """
    Return field.keyword if it exists in mapping, else None.
    Does NOT call resolve_es_field.
    """
    if not field:
        return None
    if field.endswith(".keyword"):
        return field if _field_exists(mappings, field) else None
    kw = f"{field}.keyword"
    return kw if _field_exists(mappings, kw) else None


def _wildcards_for_token(field: str, token: str) -> List[Dict[str, Any]]:
    """
    Case-tolerant wildcard queries WITHOUT relying on ES case_insensitive (compat-safe).
    """
    t = (token or "").strip()
    if not t:
        return []
    variants = {t.lower(), t.upper(), t[:1].upper() + t[1:].lower()}
    out: List[Dict[str, Any]] = []
    for v in sorted(variants):
        out.append({"wildcard": {field: f"*{v}*"}})
    return out


def _coupon_should_clauses(mappings: Dict[str, Any], coupon_field: str, tokens: List[str]) -> List[Dict[str, Any]]:
    """
    Build robust "coupon contains token" clauses:
      - match_phrase on coupon_field (works well for text)
      - wildcard on coupon.keyword if available (works for keyword)
    """
    clauses: List[Dict[str, Any]] = []
    for tok in tokens:
        clauses.append({"match_phrase": {coupon_field: tok}})

    coupon_kw = _maybe_keyword(mappings, coupon_field)
    if not coupon_kw and coupon_field and coupon_field.endswith(".keyword") and _field_exists(mappings, coupon_field):
        coupon_kw = coupon_field

    if coupon_kw:
        for tok in tokens:
            clauses.extend(_wildcards_for_token(coupon_kw, tok))

    return clauses


def _route_terms_filter(field: str, value: str) -> Dict[str, Any]:
    """
    Robust route filter:
      - if field is keyword-ish => terms with common case variants
      - else => match_phrase
    """
    if field.endswith(".keyword"):
        variants = [value, value.lower(), value.upper(), value[:1].upper() + value[1:].lower()]
        uniq: List[str] = []
        for v in variants:
            if v not in uniq:
                uniq.append(v)
        return {"terms": {field: uniq}}
    return {"match_phrase": {field: value}}


# -------------------------------------------------------------------
# Metrics
# -------------------------------------------------------------------

def _es_invoices_with_redo_items(
    req,
    client,
    mappings: Dict[str, Any],
    business_rules: Optional[str],
):
    """
    Invoices with Redo Items
    ✅ Always returns a metric row (value can be 0)

    Definition:
      - Count distinct invoice_id where coupon.keyword is exactly one of: REDO/Redo/redo
      - Windowed by dropoff_at
    """
    index_name = req.es_index_name.strip()

    # --- direct fields from your mapping ---
    date_field = "dropoff_at"
    invoice_id_field = "invoice_id"
    coupon_text_field = "coupon"
    coupon_kw_field = "coupon.keyword"

    # resolve only if the direct field isn't present (keeps it robust)
    if not _field_exists(mappings, date_field):
        date_field = resolve_es_field(mappings, user_term="dropoff_at", alias_family="date") or date_field

    if not _field_exists(mappings, coupon_text_field):
        coupon_text_field = resolve_es_field(mappings, user_term="coupon") or coupon_text_field

    # prefer coupon.keyword; fall back to coupon if keyword doesn't exist
    if not _field_exists(mappings, coupon_kw_field):
        # if coupon_text_field is already "coupon.keyword", keep it
        if coupon_text_field.endswith(".keyword") and _field_exists(mappings, coupon_text_field):
            coupon_kw_field = coupon_text_field
        else:
            coupon_kw_field = ""  # force fallback below

    if not (_field_exists(mappings, date_field) and _field_exists(mappings, invoice_id_field) and _field_exists(mappings, coupon_text_field)):
        return _es_cannot_answer(
            "Cannot compute 'Invoices with Redo Items' because dropoff_at, invoice_id, "
            "or coupon fields are missing from the invoices Elasticsearch mappings.",
            business_rules,
        )

    filters, window_label = _date_filters_or_default(req, date_field)

    # ✅ MODIFIED PART START ------------------------------------------------
    # Cheapest: exact match on coupon.keyword with common case variants.
    # Fallback: match on coupon (analyzed text) if keyword isn't available.
    if coupon_kw_field and _field_exists(mappings, coupon_kw_field):
        filters.append({"terms": {coupon_kw_field: ["REDO", "Redo", "redo"]}})
        coupon_rule_desc = f"{coupon_kw_field} IN ['REDO','Redo','redo']"
    else:
        filters.append({"match": {coupon_text_field: "redo"}})
        coupon_rule_desc = f"match({coupon_text_field}, 'redo')"
    # ✅ MODIFIED PART END --------------------------------------------------

    body: Dict[str, Any] = {
        "size": 0,
        "query": {"bool": {"filter": filters}},
        "aggs": {"invoices_with_redo": {"cardinality": {"field": invoice_id_field}}},
    }

    res = _safe_es_search(client, index=index_name, body=body)
    agg = (res.get("aggregations") or {}).get("invoices_with_redo") or {}
    count = int(agg.get("value") or 0)

    rows: List[Dict[str, Any]] = [
        {"metric": "invoices_with_redo", "label": "Invoices with Redo Items", "value": float(count)}
    ]

    insight = (
        f"'Invoices with Redo Items' is computed on index '{index_name}' ({window_label}) "
        f"as the count of distinct invoices where {coupon_rule_desc}, "
        f"using '{date_field}' as the invoice date and '{invoice_id_field}' as the invoice identifier."
    )

    return {
        "insight": to_json_safe(insight),
        "rows": to_json_safe(rows),
        "rules_used": business_rules or "",
        "engine": "es",
    }


def _es_avg_pickup_delay_retail(
    req,
    client,
    mappings: Dict[str, Any],
    business_rules: Optional[str],
):
    """
    Average Pickup Delay (Retail)
    ✅ Always returns a metric row (value 0.0 when no data)
    ✅ Robust retail route filter (nested OR non-nested) using ignore_unmapped
    ✅ Window filters based on dropoff_at (invoice date), delay computed from ready→pickup.
    """
    index_name = req.es_index_name.strip()

    ready_field = resolve_es_field(mappings, user_term="ready_at", alias_family="date")
    pickup_field = (
        resolve_es_field(mappings, user_term="pickup_at", alias_family="date")
        or resolve_es_field(mappings, user_term="picked_up_at", alias_family="date")
        or resolve_es_field(mappings, user_term="pickup_date", alias_family="date")
    )

    if not (ready_field and pickup_field):
        return _es_cannot_answer(
            "Cannot compute 'Average Pickup Delay (Retail)' because ready_at "
            "and pickup_at date fields could not be resolved from the "
            "Elasticsearch mappings.",
            business_rules,
        )

    # ✅ MODIFIED PART START ---------------------------------------------
    window_date_field = (
        resolve_es_field(mappings, user_term="dropoff_at", alias_family="date")
        or "dropoff_at"
        or ready_field
    )
    filters, window_label = _date_filters_or_default(req, window_date_field)

    filters.append({"exists": {"field": ready_field}})
    filters.append({"exists": {"field": pickup_field}})
    # ✅ MODIFIED PART END -----------------------------------------------

    route_name_field = _resolve_existing_field(
        mappings,
        primary="route.name.keyword",
        fallback="route.name",
    )
    if not route_name_field:
        return _es_cannot_answer(
            "Cannot compute 'Average Pickup Delay (Retail)' because route.name "
            "could not be resolved from the invoices mapping.",
            business_rules,
        )

    route_clause = _route_terms_filter(route_name_field, "Retail")

    route_filter = {
        "bool": {
            "should": [
                {
                    "nested": {
                        "path": "route",
                        "ignore_unmapped": True,
                        "query": route_clause,
                    }
                },
                route_clause,
            ],
            "minimum_should_match": 1,
        }
    }

    script_source = (
        f"if (doc['{pickup_field}'].size() == 0 || doc['{ready_field}'].size() == 0) "
        "return null; "
        f"long diff = doc['{pickup_field}'].value.toInstant().toEpochMilli() "
        f"- doc['{ready_field}'].value.toInstant().toEpochMilli(); "
        "return diff / 86400000.0;"
    )

    body: Dict[str, Any] = {
        "size": 0,
        "query": {"bool": {"filter": filters + [route_filter]}},
        "aggs": {
            "pickup_delay": {
                "stats": {"script": {"lang": "painless", "source": script_source}}
            }
        },
    }

    res = _safe_es_search(client, index=index_name, body=body)

    stats = (res.get("aggregations") or {}).get("pickup_delay") or {}
    count = int(stats.get("count") or 0)
    avg_days = stats.get("avg")
    min_days = stats.get("min")
    max_days = stats.get("max")

    if count == 0 or avg_days is None:
        rows: List[Dict[str, Any]] = [
            {
                "metric": "avg_pickup_delay_days",
                "label": "Average Pickup Delay (Retail, days)",
                "value": 0.0,
                "count_invoices": 0,
                "min_delay_days": None,
                "max_delay_days": None,
            }
        ]
        insight = (
            "Average pickup delay (Retail) is 0.0 because no retail invoices had both "
            f"ready_at and pickup_at dates in the selected window ({window_label})."
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
            }
        ]
        insight = (
            f"Average pickup delay (Retail) was computed on index '{index_name}' "
            f"({window_label}), using '{ready_field}' as the ready date and "
            f"'{pickup_field}' as the pickup date, filtering invoices where "
            "route.name is 'Retail' (nested/non-nested tolerant). Across "
            f"{count} invoices with both dates, the average delay is "
            f"{float(avg_days):.1f} days."
        )

    return {
        "insight": to_json_safe(insight),
        "rows": to_json_safe(rows),
        "rules_used": business_rules or "",
        "engine": "es",
    }


def _es_top20_customers_with_redo_courtesy(
    req,
    client,
    mappings: Dict[str, Any],  # invoices mappings
    business_rules: Optional[str],
):
    invoices_index = (req.es_index_name or "").strip()
    customers_index = (req.es_customers_index_name or "").strip()

    if not invoices_index or not customers_index:
        return _es_cannot_answer(
            "Top 20% Customers with Redo/Courtesy Items requires invoices and customers indexes.",
            business_rules,
        )

    # ✅ DIRECT invoices fields (from your mapping)
    invoice_customer_field = "customer_id"      # integer
    invoice_date_field = "dropoff_at"           # date
    invoice_id_field = "invoice_id"             # integer
    coupon_field = "coupon"                     # text (+ coupon.keyword exists)

    for f in (invoice_customer_field, invoice_date_field, invoice_id_field, coupon_field):
        if not _field_exists(mappings, f):
            return _es_cannot_answer(
                f"Cannot compute Top 20% Customers with Redo/Courtesy Items because invoices field '{f}' "
                "does not exist in invoices mapping.",
                business_rules,
            )

    # ✅ build customers mappings object
    cust_mapping_raw = client.indices.get_mapping(index=customers_index)
    cust_props = _extract_properties_from_mapping(cust_mapping_raw, customers_index)
    cust_mappings = {"properties": cust_props}

    # ✅ DIRECT customers fields (from your mapping)
    cust_id_field = "customer_id"          # integer

    # ✅ MODIFIED: tags.name is text; for terms/wildcard use tags.name.keyword
    tags_name_field = "tags.name.keyword"  # keyword (nested multi-field)

    if not _field_exists(cust_mappings, cust_id_field):
        return _es_cannot_answer(
            "Cannot compute Top 20% Customers because 'customer_id' does not exist in customers mapping.",
            business_rules,
        )

    if not _field_exists(cust_mappings, "tags") or not _field_exists(cust_mappings, tags_name_field):
        return _es_cannot_answer(
            "Cannot compute Top 20% Customers because 'tags' / 'tags.name.keyword' does not exist in customers mapping.",
            business_rules,
        )

    # ✅ tags is nested in your mapping -> MUST use nested query
    tag_clause = {
        "bool": {
            "should": [
                {"terms": {tags_name_field: ["Top 20%", "TOP 20%", "top 20%"]}},
                {"wildcard": {tags_name_field: "Top 20*"}},
                {"wildcard": {tags_name_field: "top 20*"}},
                {"wildcard": {tags_name_field: "TOP 20*"}},
            ],
            "minimum_should_match": 1,
        }
    }

    top20_filter = {
        "nested": {
            "path": "tags",
            "ignore_unmapped": True,
            "query": tag_clause,
        }
    }

    # ---- scan Top20 customers
    top20_info_by_id: Dict[Any, Dict[str, Any]] = {}
    scan_capped = False

    # ✅ MODIFIED PART START ------------------------------------------------
    # CRITICAL: do NOT sort by _id (ES blocks _id fielddata).
    # Sort by a real deterministic field that exists: customer_id.
    safe_customer_sort = [{cust_id_field: "asc"}]
    # ✅ MODIFIED PART END --------------------------------------------------

    for h in _scan_all_hits(
        client,
        index=customers_index,
        query={"bool": {"filter": [top20_filter]}},
        source_fields=["customer_id", "first_name", "last_name", "sales_pickup_lifetime", "sales_pickup_365"],
        page_size=2000,
        max_hits=MAX_TOP20_CUSTOMERS,
        sort=safe_customer_sort,  # ✅ MODIFIED
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

    # ✅ always return a metric row (value=0) instead of rows=[]
    if not top20_info_by_id:
        return {
            "insight": to_json_safe("No customers matched the Top 20% tag in customers index (nested tags)."),
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

    # ---- invoices: count redo/courtesy coupons for those customers in the selected window
    customer_ids = list(top20_info_by_id.keys())
    filters_base, window_label = _date_filters_or_default(req, invoice_date_field)

    should_clauses = _coupon_should_clauses(mappings, coupon_field, ["redo", "courtesy"])
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
                    "terms": {"field": invoice_customer_field, "size": min(len(cid_chunk), 10000)},
                    "aggs": {"redo_invoices": {"cardinality": {"field": invoice_id_field}}},
                }
            },
        }

        res_inv = _safe_es_search(client, index=invoices_index, body=body_inv)
        buckets = (((res_inv.get("aggregations") or {}).get("customers") or {}).get("buckets") or [])

        for b in buckets:
            cid = b.get("key")
            c = int(((b.get("redo_invoices") or {}).get("value")) or 0)
            if c > 0:
                redo_counts_by_customer[cid] = redo_counts_by_customer.get(cid, 0) + c

    rows: List[Dict[str, Any]] = []
    for cid, redo_count in redo_counts_by_customer.items():
        info = top20_info_by_id.get(cid)
        if not info:
            continue
        rows.append(
            {
                "customer_id": cid,
                "customer_name": info["name"],
                "lifetime_value": info["lifetime_value"],
                "sales_pickup_lifetime": info["sales_pickup_lifetime"],
                "sales_pickup_365": info["sales_pickup_365"],
                "redo_count": redo_count,
            }
        )

    rows.sort(key=lambda r: (r.get("lifetime_value") or 0.0), reverse=True)
    if len(rows) > MAX_ROWS_RETURNED:
        rows = rows[:MAX_ROWS_RETURNED]

    insight = (
        f"Scanned {len(top20_info_by_id)} Top20-tagged customers (nested tags). "
        f"Matched {len(rows)} with redo/courtesy coupons in invoices ({window_label})."
    )
    if scan_capped:
        insight += " Note: Top20 scan capped for safety."

    # ✅ prepend a metric row so the dashboard can read rows[0].value
    metric_value = float(len(redo_counts_by_customer))  # unique customers with redo/courtesy in window

    metric_row = {
        "metric": "top20_customers_redo_issues",
        "label": "Top 20% Customers – Redo Issues",
        "value": metric_value,
        "top20_customers_scanned": len(top20_info_by_id),
        "window": window_label,
        "scan_capped": scan_capped,
    }

    rows_out: List[Dict[str, Any]] = [metric_row] + rows

    return {
        "insight": to_json_safe(insight),
        "rows": to_json_safe(rows_out),
        "rules_used": business_rules or "",
        "engine": "es",
    }


def _parse_iso_date_to_date(value: Any) -> Optional[datetime.date]:
    if not value:
        return None
    if isinstance(value, str):
        s = value.strip()
        if s.endswith("Z"):
            s = s[:-1] + "+00:00"
        try:
            return datetime.fromisoformat(s).date()
        except Exception:
            return None
    return None


def _es_incoming_sales(
    req,
    client,
    mappings: Dict[str, Any],  # invoices mapping: {"properties": ...}
    business_rules: Optional[str],
):
    """
    Incoming Sales (Invoices)
    - Windowed by: dropoff_at
    - Value: SUM(total)
    - Extra: invoice_count (value_count(invoice_id))
    - Always returns 1 metric row (0.0 if no data)
    """
    index_name = (getattr(req, "es_index_name", "") or "").strip()
    if not index_name:
        return _es_cannot_answer("Incoming Sales requires invoices index (es_index_name).", business_rules)

    date_field = "dropoff_at"
    sales_field = "total"
    invoice_id_field = "invoice_id"

    # ✅ Direct mapping checks (no resolve, no guessing)
    required = [date_field, sales_field, invoice_id_field]
    missing = [f for f in required if not _field_exists(mappings, f)]
    if missing:
        return _es_cannot_answer(
            "Cannot compute Incoming Sales because required invoices fields are missing: "
            + ", ".join(missing),
            business_rules,
        )

    filters, window_label = _date_filters_or_default(req, date_field)
    filters.append({"exists": {"field": date_field}})
    filters.append({"exists": {"field": sales_field}})

    body = {
        "size": 0,
        "query": {"bool": {"filter": filters}},
        "aggs": {
            "incoming_sales": {"sum": {"field": sales_field}},
            "invoice_count": {"value_count": {"field": invoice_id_field}},
        },
    }

    res = _safe_es_search(client, index=index_name, body=body)
    aggs = res.get("aggregations") or {}

    value = float((aggs.get("incoming_sales") or {}).get("value") or 0.0)
    invoice_count = int((aggs.get("invoice_count") or {}).get("value") or 0)

    rows = [
        {
            "metric": "incoming_sales",
            "label": "Incoming Sales",
            "value": value,
            "invoice_count": invoice_count,
            "date_field": date_field,
            "sales_field": sales_field,
            "window": window_label,
        }
    ]

    insight = (
        f"Incoming Sales = SUM(invoices.{sales_field}) for invoices whose invoices.{date_field} is in the window "
        f"({window_label})."
    )

    return {
        "insight": to_json_safe(insight),
        "rows": to_json_safe(rows),
        "rules_used": business_rules or "",
        "engine": "es",
    }


def _es_incoming_pieces(
    req,
    client,
    mappings: Dict[str, Any],  # invoices mapping: {"properties": ...}
    business_rules: Optional[str],
):
    """
    Incoming Pieces (Invoices)
    - Windowed by: dropoff_at
    - Value: SUM(pieces)
    - Extra: invoice_count (value_count(invoice_id))
    - Always returns 1 metric row (0.0 if no data)
    """
    index_name = (getattr(req, "es_index_name", "") or "").strip()
    if not index_name:
        return _es_cannot_answer("Incoming Pieces requires invoices index (es_index_name).", business_rules)

    date_field = "dropoff_at"
    pieces_field = "pieces"
    invoice_id_field = "invoice_id"

    required = [date_field, pieces_field, invoice_id_field]
    missing = [f for f in required if not _field_exists(mappings, f)]
    if missing:
        return _es_cannot_answer(
            "Cannot compute Incoming Pieces because required invoices fields are missing: "
            + ", ".join(missing),
            business_rules,
        )

    filters, window_label = _date_filters_or_default(req, date_field)
    filters.append({"exists": {"field": date_field}})
    filters.append({"exists": {"field": pieces_field}})

    body = {
        "size": 0,
        "query": {"bool": {"filter": filters}},
        "aggs": {
            "incoming_pieces": {"sum": {"field": pieces_field}},
            "invoice_count": {"value_count": {"field": invoice_id_field}},
        },
    }

    res = _safe_es_search(client, index=index_name, body=body)
    aggs = res.get("aggregations") or {}

    value = float((aggs.get("incoming_pieces") or {}).get("value") or 0.0)
    invoice_count = int((aggs.get("invoice_count") or {}).get("value") or 0)

    rows = [
        {
            "metric": "incoming_pieces",
            "label": "Incoming Pieces",
            "value": value,
            "invoice_count": invoice_count,
            "date_field": date_field,
            "pieces_field": pieces_field,
            "window": window_label,
        }
    ]

    insight = (
        f"Incoming Pieces = SUM(invoices.{pieces_field}) for invoices whose invoices.{date_field} is in the window "
        f"({window_label})."
    )

    return {
        "insight": to_json_safe(insight),
        "rows": to_json_safe(rows),
        "rules_used": business_rules or "",
        "engine": "es",
    }


def _es_outgoing_sales(
    req,
    client,
    mappings: Dict[str, Any],  # invoices mapping: {"properties": ...}
    business_rules: Optional[str],
):
    """
    Outgoing Sales (Invoices)
    - Windowed by: pickup_at
    - Value: SUM(total)
    - Extra: invoice_count (value_count(invoice_id))
    - Always returns 1 metric row (0.0 if no data)
    """
    index_name = (getattr(req, "es_index_name", "") or "").strip()
    if not index_name:
        return _es_cannot_answer("Outgoing Sales requires invoices index (es_index_name).", business_rules)

    date_field = "pickup_at"
    sales_field = "total"
    invoice_id_field = "invoice_id"

    required = [date_field, sales_field, invoice_id_field]
    missing = [f for f in required if not _field_exists(mappings, f)]
    if missing:
        return _es_cannot_answer(
            "Cannot compute Outgoing Sales because required invoices fields are missing: "
            + ", ".join(missing),
            business_rules,
        )

    filters, window_label = _date_filters_or_default(req, date_field)
    filters.append({"exists": {"field": date_field}})
    filters.append({"exists": {"field": sales_field}})

    body = {
        "size": 0,
        "query": {"bool": {"filter": filters}},
        "aggs": {
            "outgoing_sales": {"sum": {"field": sales_field}},
            "invoice_count": {"value_count": {"field": invoice_id_field}},
        },
    }

    res = _safe_es_search(client, index=index_name, body=body)
    aggs = res.get("aggregations") or {}

    value = float((aggs.get("outgoing_sales") or {}).get("value") or 0.0)
    invoice_count = int((aggs.get("invoice_count") or {}).get("value") or 0)

    rows = [
        {
            "metric": "outgoing_sales",
            "label": "Outgoing Sales",
            "value": value,
            "invoice_count": invoice_count,
            "date_field": date_field,
            "sales_field": sales_field,
            "window": window_label,
        }
    ]

    insight = (
        f"Outgoing Sales = SUM(invoices.{sales_field}) for invoices whose invoices.{date_field} is in the window "
        f"({window_label})."
    )

    return {
        "insight": to_json_safe(insight),
        "rows": to_json_safe(rows),
        "rules_used": business_rules or "",
        "engine": "es",
    }


__all__ = [
    "_es_invoices_with_redo_items",
    "_es_avg_pickup_delay_retail",
    "_es_top20_customers_with_redo_courtesy",
    "_es_incoming_sales",
    "_es_incoming_pieces",
    "_es_outgoing_sales",
]
