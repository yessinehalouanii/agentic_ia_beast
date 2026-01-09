from __future__ import annotations

from datetime import datetime, timezone, timedelta
from typing import Any, Dict, List, Optional, Iterable

from abi.runtime import to_json_safe
from routes.es_test import _extract_properties_from_mapping
from app.api.docs_analytics_routes import (
    resolve_es_field,
    _ms_to_dt,
    _es_cannot_answer,
    _build_date_range_filter,
    _es_get_customer_signups,
)

# -------------------------------------------------------------------
# Safety limits (server-friendly defaults)
# -------------------------------------------------------------------

DEFAULT_WINDOW_DAYS = 365          # if user does not pass start/end
MAX_SCAN_HITS = 50_000             # max docs scanned via search_after
MAX_SCAN_PAGES = 200               # max pages via search_after
MAX_COMPOSITE_PAGES = 500          # max composite pages
MAX_TOP20_CUSTOMERS = 50_000       # cap customers scanned for "Top 20%" tag
MAX_ROWS_RETURNED = 10_000         # cap returned rows
TERMS_CHUNK_SIZE = 1000            # safer than 2000 in many clusters


# -------------------------------------------------------------------
# Small ES-safe helpers (performance + robustness)
# -------------------------------------------------------------------

def _safe_es_search(client, *, index: str, body: Dict[str, Any]) -> Dict[str, Any]:
    body = dict(body or {})
    body.setdefault("timeout", "10s")
    body.setdefault("track_total_hits", False)
    return client.search(index=index, body=body, request_timeout=20)


def _chunks(lst: List[Any], n: int) -> Iterable[List[Any]]:
    for i in range(0, len(lst), n):
        yield lst[i : i + n]


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
) -> Iterable[Dict[str, Any]]:
    """
    Scan hits using search_after (no scroll) with hard caps.
    """
    body: Dict[str, Any] = {
        "size": page_size,
        "query": query or {"match_all": {}},
        "sort": [{"_id": "asc"}],
    }
    if source_fields is not None:
        body["_source"] = source_fields

    search_after = None
    emitted = 0
    pages = 0

    while True:
        if pages >= max_pages or emitted >= max_hits:
            break

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
# ✅ NEW: mapping-aware field existence + safe field resolver
# -------------------------------------------------------------------

def _field_exists(mappings: Dict[str, Any], dotted: str) -> bool:
    """
    True only if 'dotted' exists in mappings (supports multi-fields like *.keyword).
    Prevents silent 0-results when querying non-existing fields.
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
    Try to resolve a .keyword variant (only if it exists).
    ✅ if field already ends with .keyword, only return it if it exists.
    ✅ don't produce coupon.keyword.keyword
    """
    if not field:
        return None

    if field.endswith(".keyword"):
        return field if _field_exists(mappings, field) else None

    kw = resolve_es_field(mappings, user_term=f"{field}.keyword")
    return kw if (kw and _field_exists(mappings, kw)) else None


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
    """
    index_name = req.es_index_name.strip()

    date_field = resolve_es_field(mappings, user_term="dropoff_at", alias_family="date")
    invoice_id_field = _resolve_existing_field(
        mappings,
        primary="invoice_id.keyword",
        fallback="invoice_id",
        alias_family="invoice",
    )
    coupon_field = resolve_es_field(mappings, user_term="coupon")

    if not (date_field and invoice_id_field and coupon_field):
        return _es_cannot_answer(
            "Cannot compute 'Invoices with Redo Items' because date, invoice_id "
            "or coupon fields could not be resolved from the Elasticsearch mappings.",
            business_rules,
        )

    filters, window_label = _date_filters_or_default(req, date_field)

    should_clauses = _coupon_should_clauses(mappings, coupon_field, ["redo"])
    filters.append({"bool": {"should": should_clauses, "minimum_should_match": 1}})

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
        f"as the count of distinct invoices where the coupon field contains the word "
        f"'redo', using '{date_field}' as the invoice date and '{invoice_id_field}' as "
        "the invoice identifier."
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
    ✅ FIX: window filters should be based on dropoff_at (or a date alias) instead of ready_at
            so the metric doesn’t become 0 when ready_at drifts outside the window.
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
    # Filter the *window* by dropoff_at (invoice date), but compute delay from ready→pickup.
    window_date_field = (
        resolve_es_field(mappings, user_term="dropoff_at", alias_family="date")
        or ready_field
    )

    filters, window_label = _date_filters_or_default(req, window_date_field)

    # Ensure both timestamps exist (otherwise script returns null anyway, but this speeds up)
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

    # ✅ one filter that works for both nested and non-nested mappings
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
    mappings: Dict[str, Any],  # mappings for the *invoices* index
    business_rules: Optional[str],
):
    """
    Top 20% Customers with Redo/Courtesy Items.
    (List metric: rows are customers; dashboard counts rows.)
    ✅ More tolerant Top 20% tag matching (Top 20% / TOP 20% / top 20%)
    ✅ Coupon matching more tolerant (Redo/REDO/Courtesy/COURTESY)
    """
    invoices_index = (req.es_index_name or "").strip()
    customers_index = (req.es_customers_index_name or "").strip()

    if not invoices_index or not customers_index:
        return _es_cannot_answer(
            "Top 20% Customers with Redo/Courtesy Items requires both an invoices "
            "index (es_index_name) and a customers index (es_customers_index_name).",
            business_rules,
        )

    invoice_customer_field = _resolve_existing_field(
        mappings,
        primary="customer_id.keyword",
        fallback="customer_id",
        alias_family="customer",
    )
    invoice_date_field = resolve_es_field(mappings, user_term="dropoff_at", alias_family="date")
    invoice_id_field = _resolve_existing_field(
        mappings,
        primary="invoice_id.keyword",
        fallback="invoice_id",
        alias_family="invoice",
    )
    coupon_field = resolve_es_field(mappings, user_term="coupon")

    if not (invoice_customer_field and invoice_date_field and invoice_id_field and coupon_field):
        return _es_cannot_answer(
            "Cannot compute Top 20% Customers with Redo/Courtesy Items because "
            "customer, date, invoice_id or coupon fields could not be resolved "
            "from the invoices index mappings.",
            business_rules,
        )

    cust_mapping_raw = client.indices.get_mapping(index=customers_index)
    cust_props = _extract_properties_from_mapping(cust_mapping_raw, customers_index)
    cust_mappings = {"properties": cust_props}

    cust_id_field = resolve_es_field(cust_mappings, user_term="customer_id", alias_family="customer")
    tags_name_field = _resolve_existing_field(
        cust_mappings,
        primary="tags.name.keyword",
        fallback="tags.name",
    )

    if not (cust_id_field and tags_name_field):
        return _es_cannot_answer(
            "Cannot compute Top 20% Customers with Redo/Courtesy Items because "
            "customer_id or tags.name could not be resolved from the customers index.",
            business_rules,
        )

    if tags_name_field.endswith(".keyword"):
        tag_clause = {"terms": {tags_name_field: ["Top 20%", "TOP 20%", "top 20%"]}}
    else:
        tag_clause = {
            "bool": {
                "should": [
                    {"match_phrase": {tags_name_field: "Top 20%"}},
                    {"match_phrase": {tags_name_field: "top 20%"}},
                ],
                "minimum_should_match": 1,
            }
        }

    nested_top20_filter = {"nested": {"path": "tags", "query": tag_clause}}
    plain_top20_filter = tag_clause

    top20_info_by_id: Dict[Any, Dict[str, Any]] = {}
    scan_capped = False

    try:
        hits_iter = _scan_all_hits(
            client,
            index=customers_index,
            query={"bool": {"filter": [nested_top20_filter]}},
            source_fields=["customer_id", "first_name", "last_name", "sales_pickup_lifetime", "sales_pickup_365"],
            page_size=2000,
            max_hits=MAX_TOP20_CUSTOMERS,
        )
        for h in hits_iter:
            src = h.get("_source", {}) or {}
            cid = src.get("customer_id")
            if cid is None:
                cid = src.get(cust_id_field.split(".")[-1])
            if cid is None:
                continue

            first_name = (src.get("first_name") or "").strip()
            last_name = (src.get("last_name") or "").strip()
            full_name = (f"{first_name} {last_name}").strip() or f"Customer {cid}"

            ltv_lifetime = src.get("sales_pickup_lifetime")
            ltv_365 = src.get("sales_pickup_365")
            if ltv_lifetime is not None:
                ltv = float(ltv_lifetime)
            elif ltv_365 is not None:
                ltv = float(ltv_365)
            else:
                ltv = None

            top20_info_by_id[cid] = {
                "name": full_name,
                "lifetime_value": ltv,
                "sales_pickup_lifetime": ltv_lifetime,
                "sales_pickup_365": ltv_365,
            }

            if len(top20_info_by_id) >= MAX_TOP20_CUSTOMERS:
                scan_capped = True
                break
    except Exception:
        for h in _scan_all_hits(
            client,
            index=customers_index,
            query={"bool": {"filter": [plain_top20_filter]}},
            source_fields=["customer_id", "first_name", "last_name", "sales_pickup_lifetime", "sales_pickup_365"],
            page_size=2000,
            max_hits=MAX_TOP20_CUSTOMERS,
        ):
            src = h.get("_source", {}) or {}
            cid = src.get("customer_id")
            if cid is None:
                cid = src.get(cust_id_field.split(".")[-1])
            if cid is None:
                continue

            first_name = (src.get("first_name") or "").strip()
            last_name = (src.get("last_name") or "").strip()
            full_name = (f"{first_name} {last_name}").strip() or f"Customer {cid}"

            ltv_lifetime = src.get("sales_pickup_lifetime")
            ltv_365 = src.get("sales_pickup_365")
            if ltv_lifetime is not None:
                ltv = float(ltv_lifetime)
            elif ltv_365 is not None:
                ltv = float(ltv_365)
            else:
                ltv = None

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
        insight = (
            "No customers with a 'Top 20%' tag were found in the customers index, "
            "so Top 20% Customers with Redo/Courtesy Items cannot be computed."
        )
        return {
            "insight": to_json_safe(insight),
            "rows": [],
            "rules_used": business_rules or "",
            "engine": "es",
        }

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
                    "terms": {
                        "field": invoice_customer_field,
                        "size": min(len(cid_chunk), 10000),
                    },
                    "aggs": {"redo_invoices": {"cardinality": {"field": invoice_id_field}}},
                }
            },
        }

        res_inv = _safe_es_search(client, index=invoices_index, body=body_inv)
        buckets = (res_inv.get("aggregations", {}) or {}).get("customers", {}).get("buckets", []) or []

        for b in buckets:
            cid = b.get("key")
            c = int((b.get("redo_invoices") or {}).get("value") or 0)
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
                "details": "Has redo/courtesy coupon invoices in selected period",
            }
        )

    if not rows:
        insight = (
            "No customers tagged 'Top 20%' have invoices with redo or courtesy coupons "
            f"({window_label})."
        )
        if scan_capped:
            insight += " Note: Top20 customer scan was capped for safety."
        return {
            "insight": to_json_safe(insight),
            "rows": [],
            "rules_used": business_rules or "",
            "engine": "es",
        }

    rows.sort(key=lambda r: (r.get("lifetime_value") or 0.0), reverse=True)

    if len(rows) > MAX_ROWS_RETURNED:
        rows = rows[:MAX_ROWS_RETURNED]

    insight = (
        "Top 20% Customers with Redo/Courtesy Items was computed by selecting customers "
        "from the customers index that have a 'Top 20%' tag (case-variant tolerant), "
        "then counting how many of their invoices in the invoices index have a coupon "
        "containing 'redo' or 'courtesy' "
        f"({window_label})."
    )
    if scan_capped:
        insight += " Note: Top20 customer scan was capped for safety."

    return {
        "insight": to_json_safe(insight),
        "rows": to_json_safe(rows),
        "rules_used": business_rules or "",
        "engine": "es",
    }


def _es_coupon_returns_365d_since_signup(
    req,
    client,
    mappings: Dict[str, Any],  # mappings for the *invoices* index
    business_rules: Optional[str],
):
    """
    Coupon Returns (365+ Days Since Signup)
    (List metric: rows are customers; dashboard counts rows.)
    ✅ FIX: do NOT treat coupon="" as coupon-bearing
    ✅ FIX: use existing-field resolver for customer_id / invoice_id
    """
    invoices_index = (req.es_index_name or "").strip()
    customers_index = (req.es_customers_index_name or "").strip()

    if not invoices_index or not customers_index:
        return _es_cannot_answer(
            "Coupon Returns (365+ Days Since Signup) requires both an invoices index "
            "(es_index_name) and a customers index (es_customers_index_name).",
            business_rules,
        )

    invoice_customer_field = _resolve_existing_field(
        mappings,
        primary="customer_id.keyword",
        fallback="customer_id",
        alias_family="customer",
    )
    invoice_date_field = resolve_es_field(mappings, user_term="dropoff_at", alias_family="date")
    coupon_field = resolve_es_field(mappings, user_term="coupon")
    coupon_total_field = resolve_es_field(mappings, user_term="coupon_total")

    if not (invoice_customer_field and invoice_date_field):
        return _es_cannot_answer(
            "Cannot compute Coupon Returns (365+ Days Since Signup) because customer_id "
            "or dropoff_at could not be resolved from the invoices index.",
            business_rules,
        )

    if not (coupon_field or coupon_total_field):
        return _es_cannot_answer(
            "Cannot compute Coupon Returns (365+ Days Since Signup) because no coupon "
            "field (coupon or coupon_total) could be resolved from the invoices index.",
            business_rules,
        )

    # 1) signup dates from CUSTOMERS (original_signup only)
    cust_mapping_raw = client.indices.get_mapping(index=customers_index)
    cust_props = _extract_properties_from_mapping(cust_mapping_raw, customers_index)
    cust_mappings = {"properties": cust_props}

    signup_by_customer = _es_get_customer_signups(client, customers_index, cust_mappings)
    if not signup_by_customer:
        return _es_cannot_answer(
            "Cannot compute Coupon Returns (365+ Days Since Signup) because no "
            "original_signup dates could be found in the customers index.",
            business_rules,
        )

    cust_id_field = resolve_es_field(cust_mappings, user_term="customer_id", alias_family="customer")
    first_name_field = resolve_es_field(cust_mappings, user_term="first_name") or "first_name"
    last_name_field = resolve_es_field(cust_mappings, user_term="last_name") or "last_name"

    # 2) invoices: first coupon date per customer
    filters, window_label = _date_filters_or_default(req, invoice_date_field)

    # ✅ FIX: exists(coupon) includes coupon="" so we exclude empty string using keyword (if available)
    coupon_should: List[Dict[str, Any]] = []
    coupon_kw = _maybe_keyword(mappings, coupon_field) if coupon_field else None

    if coupon_kw:
        coupon_should.append(
            {
                "bool": {
                    "must": [{"exists": {"field": coupon_kw}}],
                    "must_not": [{"term": {coupon_kw: ""}}],
                }
            }
        )
    elif coupon_field:
        # fallback (can't reliably exclude empty without keyword)
        coupon_should.append({"exists": {"field": coupon_field}})

    if coupon_total_field:
        coupon_should.append({"range": {coupon_total_field: {"lt": 0}}})
        coupon_should.append({"range": {coupon_total_field: {"gt": 0}}})

    filters.append({"bool": {"should": coupon_should, "minimum_should_match": 1}})

    invoice_id_field = _resolve_existing_field(
        mappings,
        primary="invoice_id.keyword",
        fallback="invoice_id",
        alias_family="invoice",
    )
    count_field = invoice_id_field or invoice_date_field

    rows: List[Dict[str, Any]] = []
    matched_customer_ids: List[Any] = []

    sub_aggs = {
        "first_coupon_date": {"min": {"field": invoice_date_field}},
        "coupon_invoice_count": {"value_count": {"field": count_field}},
    }

    for b in _composite_by_customer(
        client,
        index=invoices_index,
        filters=filters,
        customer_field=invoice_customer_field,
        sub_aggs=sub_aggs,
        page_size=500,
        max_pages=MAX_COMPOSITE_PAGES,
    ):
        cid = (b.get("key") or {}).get("cid")
        if cid is None:
            continue

        signup_date = signup_by_customer.get(cid)
        if not signup_date:
            continue

        first_coupon_ms = (b.get("first_coupon_date") or {}).get("value")
        dt = _ms_to_dt(first_coupon_ms)
        if not dt:
            continue

        coupon_date = dt.date()
        diff_days = (coupon_date - signup_date).days
        if diff_days < 365:
            continue

        coupon_count = int((b.get("coupon_invoice_count") or {}).get("value") or 0)

        rows.append(
            {
                "customer_id": cid,
                "customer_name": f"Customer {cid}",  # filled later
                "original_signup": signup_date.isoformat(),
                "first_coupon_visit": coupon_date.isoformat(),
                "days_from_signup_to_first_coupon": diff_days,
                "coupon_invoice_count": coupon_count,
            }
        )
        matched_customer_ids.append(cid)

        if len(rows) >= MAX_ROWS_RETURNED:
            break

    if not rows:
        insight = (
            "No customers were found whose first coupon-bearing invoice occurred at least "
            "365 days after original signup "
            f"({window_label})."
        )
        return {
            "insight": to_json_safe(insight),
            "rows": [],
            "rules_used": business_rules or "",
            "engine": "es",
        }

    # 3) Fetch names ONLY for matched customers
    name_by_id: Dict[Any, str] = {}
    if cust_id_field and matched_customer_ids:
        for chunk in _chunks(matched_customer_ids, TERMS_CHUNK_SIZE):
            body_names = {
                "size": 10000,
                "query": {"bool": {"filter": [{"terms": {cust_id_field: chunk}}]}},
                "_source": [
                    cust_id_field,
                    first_name_field,
                    last_name_field,
                    "customer_id",
                    "first_name",
                    "last_name",
                ],
            }
            res_names = _safe_es_search(client, index=customers_index, body=body_names)
            hits_names = (res_names.get("hits") or {}).get("hits") or []
            for h in hits_names:
                src = h.get("_source", {}) or {}
                cid2 = src.get("customer_id")
                if cid2 is None:
                    cid2 = src.get(cust_id_field.split(".")[-1])
                if cid2 is None:
                    continue
                first = (src.get(first_name_field) or src.get("first_name") or "").strip()
                last = (src.get(last_name_field) or src.get("last_name") or "").strip()
                name_by_id[cid2] = (f"{first} {last}").strip() or f"Customer {cid2}"

    for r in rows:
        cid = r.get("customer_id")
        r["customer_name"] = name_by_id.get(cid, r.get("customer_name") or f"Customer {cid}")

    rows.sort(key=lambda r: r.get("days_from_signup_to_first_coupon", 0), reverse=True)

    insight = (
        "Coupon Returns (365+ Days Since Signup) was computed by joining the invoices index "
        f"'{invoices_index}' with the customers index '{customers_index}'. For each customer we "
        "look at their first coupon-bearing invoice and keep only those where that visit occurs "
        f"at least 365 days after original signup ({window_label})."
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
    "_es_coupon_returns_365d_since_signup",
]
