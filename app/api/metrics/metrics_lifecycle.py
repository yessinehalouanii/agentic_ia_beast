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


# -------------------------------------------------------------------
# Metrics (NO rollup)
# -------------------------------------------------------------------
def _es_avg_days_between_visits_active(req, client, mappings: Dict[str, Any], business_rules: Optional[str]):
    """
    Avg days between visits for active customers (last_visit in last 365d),
    using customers.visits_interval_avg (already precomputed avg gap in days).

    Filters:
      - visits_lifetime >= 2
      - visits_interval_avg > 0
      - last_visit >= today-365
      - OPTIONAL (recommended): sales_pickup_lifetime > 0 (paying customers)
    """
    customers_index = (getattr(req, "es_customers_index_name", "") or "").strip()
    if not customers_index:
        return _es_cannot_answer("Missing customers index (es_customers_index_name).", business_rules)

    customers_index, cust_mappings = _get_customers_index_and_mappings(client, customers_index)

    required = ["customer_id", "last_visit", "visits_lifetime", "visits_interval_avg"]
    missing = [f for f in required if not _field_exists(cust_mappings, f)]
    if missing:
        return _es_cannot_answer(
            f"Cannot compute avg days between visits from customers index '{customers_index}' "
            f"because required fields are missing: {', '.join(missing)}.",
            business_rules,
        )

    today = datetime.now(timezone.utc).date()
    cutoff = (today - timedelta(days=365)).isoformat()

    # Optional paying filter (keeps consistency with most other KPIs)
    paying_filter = []
    if _field_exists(cust_mappings, "sales_pickup_lifetime"):
        paying_filter = [{"range": {"sales_pickup_lifetime": {"gt": 0}}}]

    base_filters: List[Dict[str, Any]] = [
        {"range": {"last_visit": {"gte": cutoff}}},
        {"range": {"visits_lifetime": {"gte": 2}}},
        {"range": {"visits_interval_avg": {"gt": 0}}},
        *paying_filter,
    ]

    body = {
        "size": 0,
        "query": {"bool": {"filter": base_filters}},
        "aggs": {
            "avg_gap_days": {"avg": {"field": "visits_interval_avg"}},
            "customers_counted": {"value_count": {"field": "customer_id"}},
        },
    }

    res = _safe_es_search(client, index=customers_index, body=body)
    aggs = res.get("aggregations") or {}
    avg_gap = (aggs.get("avg_gap_days") or {}).get("value")
    count = int((aggs.get("customers_counted") or {}).get("value") or 0)

    if avg_gap is None or count == 0:
        return {
            "insight": to_json_safe(
                "No active repeat customers matched the filters (last_visit in 365d and visits_lifetime ≥ 2), "
                "so the average gap could not be computed."
            ),
            "rows": [],
            "rules_used": business_rules or "",
            "engine": "es",
        }

    rows = [{
        "metric": "avg_days_between_visits_active",
        "label": "Avg Days Between Visits (Active, via visits_interval_avg)",
        "value": float(avg_gap),
        "customers_counted": count,
        "active_window_days": 365,
        "customers_index": customers_index,
    }]

    insight = (
        f"Avg days between visits for active repeat customers was computed from customers.visits_interval_avg. "
        f"Filters: last_visit within 365d, visits_lifetime ≥ 2, visits_interval_avg > 0"
        f"{' and sales_pickup_lifetime > 0' if paying_filter else ''}. "
        f"Result: ~{float(avg_gap):.1f} days across {count} customers."
    )

    return {
        "insight": to_json_safe(insight),
        "rows": to_json_safe(rows),
        "rules_used": business_rules or "",
        "engine": "es",
    }


def _es_active_customers(
    req,
    client,
    mappings: Dict[str, Any],
    business_rules: Optional[str],
    days_threshold: int = 180,
):
    """
    Active Customers (canonical definition).

    Preferred path (customers index):
      - Use the customers index (es_customers_index_name)
      - customer has sales_pickup_lifetime > 0
      - last_visit is within the last `days_threshold` days (default: 180)

    Fallback path (invoices index):
      - If customers index is missing or does not have the required fields,
        fall back to invoice-derived stats (previous behavior).
    """
    # ------------------------------------------------------------
    # ✅ Preferred: use customers index directly
    # ------------------------------------------------------------
    customers_index = (getattr(req, "es_customers_index_name", "") or "").strip()

    today = datetime.now(timezone.utc).date()
    cutoff_date = today - timedelta(days=int(days_threshold))
    cutoff_str = cutoff_date.isoformat()  # e.g. '2026-01-13'

    if customers_index:
        try:
            cust_mapping_raw = client.indices.get_mapping(index=customers_index)
        except Exception:
            cust_mapping_raw = None

        if cust_mapping_raw is not None:
            cust_props = _extract_properties_from_mapping(cust_mapping_raw, customers_index)
            cust_mappings = {"properties": cust_props}

            # We only use last_visit + sales_pickup_lifetime if they truly exist
            if _field_exists(cust_mappings, "last_visit") and _field_exists(
                cust_mappings, "sales_pickup_lifetime"
            ):
                # One doc per customer in customers index:
                #   - sales_pickup_lifetime > 0
                #   - last_visit >= cutoff_str
                body = {
                    "size": 0,
                    "query": {
                        "bool": {
                            "filter": [
                                {"range": {"sales_pickup_lifetime": {"gt": 0}}},
                                {"range": {"last_visit": {"gte": cutoff_str}}},
                            ]
                        }
                    },
                    "aggs": {
                        "active_customers": {
                            "value_count": {"field": "customer_id"}
                        }
                    },
                }

                res = _safe_es_search(client, index=customers_index, body=body)
                agg = (res.get("aggregations") or {}).get("active_customers") or {}
                active_count = float(agg.get("value") or 0.0)

                rows = [
                    {
                        "metric": "active_customers",
                        "label": "Active Customers",
                        "value": active_count,
                    }
                ]

                insight = (
                    f"'Active Customers' is defined as customers with sales_pickup_lifetime > 0 and a last_visit "
                    f"within the last {days_threshold} days, using the customers index '{customers_index}'. "
                    f"This yields {int(active_count)} active customers."
                )

                return {
                    "insight": to_json_safe(insight),
                    "rows": to_json_safe(rows),
                    "rules_used": business_rules or "",
                    "engine": "es",
                }

    # ------------------------------------------------------------
    # 🟡 Fallback: previous invoices-only behavior
    # ------------------------------------------------------------
    if not (req.es_index_name or "").strip():
        return _es_cannot_answer(
            "Missing invoices index (es_index_name), and customers index "
            "could not be used for 'Active Customers'.",
            business_rules,
        )

    invoices_index, invoices_mappings = _get_invoice_index_and_mappings(client, req.es_index_name)

    stats = _get_customer_stats_invoices_only(req, client, invoices_index, invoices_mappings)
    if stats is None:
        return _es_cannot_answer(
            "Cannot compute 'Active Customers' because required fields are missing in both "
            "customers and invoices indices.",
            business_rules,
        )

    active_count = 0

    for s in stats:
        last = s.get("last_visit")
        if not last:
            continue

        # Has made at least one purchase (lifetime revenue > 0)
        if (s.get("total_revenue") or 0.0) <= 0.0:
            continue

        days_since = (today - last.date()).days
        if days_since <= int(days_threshold):
            active_count += 1

    rows = [
        {
            "metric": "active_customers",
            "label": "Active Customers",
            "value": float(active_count),
        }
    ]

    insight = (
        f"'Active Customers' is defined as customers with >0 lifetime spend and a last visit "
        f"within the last {days_threshold} days. This was computed from invoice-derived stats "
        f"as a fallback and yields {active_count} active customers."
    )

    return {
        "insight": to_json_safe(insight),
        "rows": to_json_safe(rows),
        "rules_used": business_rules or "",
        "engine": "es",
    } 


def _es_overdue_customers(req, client, mappings: Dict[str, Any], business_rules: Optional[str]):
    """
    Customers overdue for their next visit:
      typical interval ≈ (last - first)/(visits-1)
      overdue if days_since_last_visit > 1.5 * interval

    Invoice-derived stats.
    """
    if not (req.es_index_name or "").strip():
        return _es_cannot_answer("Missing invoices index (es_index_name).", business_rules)

    invoices_index, invoices_mappings = _get_invoice_index_and_mappings(client, req.es_index_name)

    max_rows = _get_req_int(req, "es_max_rows", 2000, min_v=100, max_v=20_000)
    today = datetime.now(timezone.utc).date()

    stats = _get_customer_stats_invoices_only(req, client, invoices_index, invoices_mappings)
    if stats is None:
        return _es_cannot_answer(
            "Cannot compute 'overdue customers' because required invoices fields "
            "(customer_id, dropoff_at) are missing or could not be derived from the invoices mappings.",
            business_rules,
        )

    overdue: List[Dict[str, Any]] = []
    overdue_count = 0

    for s in stats:
        vc = int(s.get("visit_count") or 0)
        first = s.get("first_visit")
        last = s.get("last_visit")
        if vc <= 1 or not first or not last:
            continue

        days_between = (last.date() - first.date()).days
        interval = days_between / float(vc - 1) if vc > 1 else float(days_between)
        if interval <= 0:
            continue

        days_since = (today - last.date()).days
        if days_since > 1.5 * interval:
            overdue_count += 1
            if len(overdue) < max_rows:
                overdue.append(
                    {
                        "customer_id": s.get("customer_id"),
                        "last_visit": last.isoformat(),
                        "days_since_last_visit": days_since,
                        "expected_interval_days": interval,
                    }
                )

    total = len(stats)
    pct = (overdue_count * 100.0 / total) if total else 0.0

    insight = (
        f"There are {overdue_count} customers who appear overdue for their next visit "
        f"(days since last visit > 1.5× typical interval), about {pct:.1f}% of all customers. "
        f"Rows are limited to {max_rows} for safety."
    )

    return {
        "insight": to_json_safe(insight),
        "rows": to_json_safe(overdue),
        "rules_used": business_rules or "",
        "engine": "es",
    }


def _es_visit_frequency_distribution(req, client, mappings: Dict[str, Any], business_rules: Optional[str]):
    """
    Distribution of customers by visits_lifetime bucket:
      1, 2–5, 6–11, 12+

    Uses customers.visits_lifetime (one doc per customer).
    """
    customers_index = (getattr(req, "es_customers_index_name", "") or "").strip()
    if not customers_index:
        return _es_cannot_answer("Missing customers index (es_customers_index_name).", business_rules)

    customers_index, cust_mappings = _get_customers_index_and_mappings(client, customers_index)

    if not _field_exists(cust_mappings, "customer_id") or not _field_exists(cust_mappings, "visits_lifetime"):
        return _es_cannot_answer(
            f"Cannot compute visit frequency distribution because customers index '{customers_index}' "
            "is missing customer_id and/or visits_lifetime.",
            business_rules,
        )

    # Optional paying filter (comment out if you want *all* customers regardless of spend)
    base_filters: List[Dict[str, Any]] = [
        {"range": {"visits_lifetime": {"gt": 0}}},
    ]
    if _field_exists(cust_mappings, "sales_pickup_lifetime"):
        base_filters.append({"range": {"sales_pickup_lifetime": {"gt": 0}}})

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

    insight = (
        "Visit frequency distribution computed from customers.visits_lifetime"
        f"{' (paying customers only)' if any('sales_pickup_lifetime' in f.get('range', {}) for f in base_filters) else ''}."
    )

    return {
        "insight": to_json_safe(insight),
        "rows": to_json_safe(rows),
        "rules_used": business_rules or "",
        "engine": "es",
    }



def _es_customers_nth_visit(req, client, mappings: Dict[str, Any], business_rules: Optional[str]):
    """
    Lifetime counts of customers who have reached >=2/3/4/5 visits,
    using customers.visits_lifetime.
    """
    customers_index = (getattr(req, "es_customers_index_name", "") or "").strip()
    if not customers_index:
        return _es_cannot_answer("Missing customers index (es_customers_index_name).", business_rules)

    customers_index, cust_mappings = _get_customers_index_and_mappings(client, customers_index)

    required = ["customer_id", "visits_lifetime"]
    missing = [f for f in required if not _field_exists(cust_mappings, f)]
    if missing:
        return _es_cannot_answer(
            f"Cannot compute Customers Achieving Nth Visit because customers index '{customers_index}' "
            f"is missing: {', '.join(missing)}.",
            business_rules,
        )

    base_filters: List[Dict[str, Any]] = [
        {"range": {"visits_lifetime": {"gt": 0}}},
    ]
    if _field_exists(cust_mappings, "sales_pickup_lifetime"):
        base_filters.append({"range": {"sales_pickup_lifetime": {"gt": 0}}})

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
def _es_top_customers_by_revenue(req, client, mappings: Dict[str, Any], business_rules: Optional[str]):
    """
    Top 5% / Top 20% customers by revenue (invoice-derived stats).
    NOTE: Requires full per-customer revenue list in Python (can be heavy).
    """
    if not (req.es_index_name or "").strip():
        return _es_cannot_answer("Missing invoices index (es_index_name).", business_rules)

    invoices_index, invoices_mappings = _get_invoice_index_and_mappings(client, req.es_index_name)

    max_rows = _get_req_int(req, "es_max_rows", 500, min_v=50, max_v=5000)

    stats = _get_customer_stats_invoices_only(req, client, invoices_index, invoices_mappings)
    if stats is None:
        return _es_cannot_answer(
            "Cannot compute 'Top 5% / Top 20% customers by revenue' because required invoices fields "
            "(customer_id, dropoff_at) are missing or could not be derived from the invoices mappings.",
            business_rules,
        )

    revenue_stats = [
        s for s in stats
        if (s.get("total_revenue") is not None and float(s.get("total_revenue") or 0) > 0)
    ]
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
                "visit_count": int(s.get("visit_count") or 0),
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

def _es_new_customer_acquisition_from_customers(
    req,
    client,
    cust_mappings: Dict[str, Any],
    business_rules: Optional[str],
):
    """
    New Customer Acquisition (cheap)
    Definition: count customers by original_signup (or created_at fallback), grouped by month or quarter.
    """
    customers_index = (req.es_customers_index_name or "").strip()
    if not customers_index:
        return _es_cannot_answer("New Customer Acquisition requires customers index (es_customers_index_name).", business_rules)

    # pick the best date field available
    date_field = "original_signup" if _field_exists(cust_mappings, "original_signup") else None
    if not date_field and _field_exists(cust_mappings, "created_at"):
        date_field = "created_at"

    if not date_field:
        return _es_cannot_answer(
            "Cannot compute New Customer Acquisition because neither customers.original_signup nor customers.created_at exists.",
            business_rules,
        )

    filters, window_label = _date_filters_or_default(req, date_field)
    filters.append({"exists": {"field": date_field}})

    ql = (getattr(req, "question", "") or "").lower()
    use_quarter = any(p in ql for p in ["quarter", "q1", "q2", "q3", "q4"])

    # month is standard; quarter you can either:
    # - still use month and convert in python, OR
    # - use a runtime field (not cheap), so do python conversion.
    body = {
        "size": 0,
        "query": {"bool": {"filter": filters}},
        "aggs": {
            "by_month": {
                "date_histogram": {
                    "field": date_field,
                    "calendar_interval": "month",
                    "min_doc_count": 0,
                }
            }
        },
    }

    res = _safe_es_search(client, index=customers_index, body=body)
    buckets = (((res.get("aggregations") or {}).get("by_month") or {}).get("buckets") or [])

    # build rows
    counts = {}
    for b in buckets:
        key_as_string = b.get("key_as_string")  # ISO datetime
        doc_count = int(b.get("doc_count") or 0)
        if not key_as_string:
            continue
        # key_as_string like "2026-01-01T00:00:00.000Z" -> take YYYY-MM
        ym = key_as_string[:7]
        if use_quarter:
            y = int(ym[:4]); m = int(ym[5:7])
            q = (m - 1) // 3 + 1
            label = f"{y}-Q{q}"
        else:
            label = ym
        counts[label] = counts.get(label, 0) + doc_count

    rows = [{"period": p, "new_customers": c} for p, c in sorted(counts.items())]

    insight = (
        f"New Customer Acquisition computed from customers.{date_field} ({window_label}), "
        f"grouped by {'quarter' if use_quarter else 'month'}."
    )

    return {
        "insight": to_json_safe(insight),
        "rows": to_json_safe(rows),
        "rules_used": business_rules or "",
        "engine": "es",
    }

def _es_new_customer_30d_return_rate(req, client, mappings: Dict[str, Any], business_rules: Optional[str]):
    """
    New Customer 30-Day Return Rate (NO rollup):
      - Find "new customers" using same rule as acquisition (first_visit window + signup diff<=30)
      - Then check whether they have a 2nd visit within 30 days after first

    ✅ MODIFIED: use direct invoices fields (no resolve_es_field):
      - customer_id, dropoff_at, visit_id (optional)
    """
    invoices_index_in = (req.es_index_name or "").strip()
    customers_index = (req.es_customers_index_name or "").strip()

    if not invoices_index_in or not customers_index:
        return _es_cannot_answer(
            "New Customer 30-Day Return Rate requires both an invoices index (es_index_name) and a customers index (es_customers_index_name).",
            business_rules,
        )

    invoices_index, invoices_mappings = _get_invoice_index_and_mappings(client, invoices_index_in)

    # ✅ Direct fields (from your invoices mapping)
    customer_field = "customer_id"
    date_field = "dropoff_at"
    visit_field: Optional[str] = "visit_id" if _field_exists(invoices_mappings, "visit_id") else None

    if not (_field_exists(invoices_mappings, customer_field) and _field_exists(invoices_mappings, date_field)):
        return _es_cannot_answer(
            "Cannot compute 'New Customer 30-Day Return Rate' because required invoices fields "
            "customer_id and/or dropoff_at are missing from the invoices mapping.",
            business_rules,
        )

    # customers mapping (read-only OK)
    cust_mapping_raw = client.indices.get_mapping(index=customers_index)
    cust_props = _extract_properties_from_mapping(cust_mapping_raw, customers_index)
    cust_mappings = {"properties": cust_props}
    signup_by_customer = _es_get_customer_signups(client, customers_index, cust_mappings)

    start_d = _parse_date_str(getattr(req, "start_date", None))
    end_d = _parse_date_str(getattr(req, "end_date", None))
    max_diff_days = 30

    # 1) Build new customer list from invoice-derived stats
    stats = _get_customer_stats_invoices_only(req, client, invoices_index, invoices_mappings)
    if stats is None:
        return _es_cannot_answer(
            "Cannot compute 'New Customer 30-Day Return Rate' because required invoices fields "
            "(customer_id, dropoff_at) are missing or could not be derived from the invoices mappings.",
            business_rules,
        )

    new_customers: List[Any] = []
    first_dt_by_customer: Dict[Any, datetime] = {}

    for s in stats:
        cid = s.get("customer_id")
        first = s.get("first_visit")
        if not cid or not first:
            continue

        fd = first.date()
        if start_d and fd < start_d:
            continue
        if end_d and fd > end_d:
            continue

        signup_date = signup_by_customer.get(cid)
        if signup_date is not None:
            diff_days = (fd - signup_date).days
            if diff_days < 0 or diff_days > max_diff_days:
                continue

        new_customers.append(cid)
        first_dt_by_customer[cid] = first

    if not new_customers:
        return {
            "insight": to_json_safe(
                "No new customers were found in the specified date window, so the 30-day return rate cannot be computed."
            ),
            "rows": [],
            "rules_used": business_rules or "",
            "engine": "es",
        }

    # Guardrail: cap how many new customers we check for 2nd visit (to avoid huge ES work)
    max_check = _get_req_int(req, "es_max_new_customers_check", 50_000, min_v=1_000, max_v=200_000)
    truncated = False
    if len(new_customers) > max_check:
        new_customers = new_customers[:max_check]
        truncated = True

    # 2) Check second visit within 30 days using ES queries in chunks
    def _chunks(lst: List[Any], n: int) -> Iterable[List[Any]]:
        for i in range(0, len(lst), n):
            yield lst[i : i + n]

    chunk_size = _get_req_int(req, "es_chunk_size", 500, min_v=100, max_v=2000)
    returned_within_30 = 0
    total_new = len(new_customers)

    if visit_field:
        # Using visit_id: per customer -> take first 2 visits ordered by first_date
        for chunk in _chunks(new_customers, chunk_size):
            body = {
                "size": 0,
                "query": {"bool": {"filter": [{"terms": {customer_field: chunk}}]}},
                "aggs": {
                    "customers": {
                        "terms": {"field": customer_field, "size": len(chunk)},
                        "aggs": {
                            "visits": {
                                "terms": {"field": visit_field, "size": 2, "order": {"first_date": "asc"}},
                                "aggs": {"first_date": {"min": {"field": date_field}}},
                            }
                        },
                    }
                },
            }
            res = _safe_es_search(client, index=invoices_index, body=body)
            cust_buckets = (res.get("aggregations", {}).get("customers", {}).get("buckets", [])) or []
            for cb in cust_buckets:
                vb = (cb.get("visits") or {}).get("buckets") or []
                if len(vb) < 2:
                    continue
                dt0 = _ms_to_dt(((vb[0].get("first_date") or {}).get("value")))
                dt1 = _ms_to_dt(((vb[1].get("first_date") or {}).get("value")))
                if not dt0 or not dt1:
                    continue
                if (dt1.date() - dt0.date()).days <= 30:
                    returned_within_30 += 1
    else:
        # No visit_id: use earliest 2 invoices (top_hits size=2 sorted by date)
        for chunk in _chunks(new_customers, chunk_size):
            body = {
                "size": 0,
                "query": {"bool": {"filter": [{"terms": {customer_field: chunk}}]}},
                "aggs": {
                    "customers": {
                        "terms": {"field": customer_field, "size": len(chunk)},
                        "aggs": {
                            "first_two": {
                                "top_hits": {
                                    "size": 2,
                                    "sort": [{date_field: {"order": "asc"}}],
                                    "_source": False,
                                    "track_scores": False,
                                }
                            }
                        },
                    }
                },
            }
            res = _safe_es_search(client, index=invoices_index, body=body)
            cust_buckets = (res.get("aggregations", {}).get("customers", {}).get("buckets", [])) or []
            for cb in cust_buckets:
                hits = (((cb.get("first_two") or {}).get("hits", {}) or {}).get("hits", [])) or []
                if len(hits) < 2:
                    continue
                sort0 = hits[0].get("sort") or []
                sort1 = hits[1].get("sort") or []
                if not sort0 or not sort1:
                    continue
                dt0 = _ms_to_dt(sort0[0])
                dt1 = _ms_to_dt(sort1[0])
                if not dt0 or not dt1:
                    continue
                if (dt1.date() - dt0.date()).days <= 30:
                    returned_within_30 += 1

    rate = (returned_within_30 * 100.0 / total_new) if total_new else 0.0

    rows = [
        {"metric": "new_customers_window", "label": "New Customers in Window", "value": total_new},
        {"metric": "new_customers_returned_30d", "label": "New Customers Returning within 30 Days (lifetime visits)", "value": returned_within_30},
        {"metric": "new_customer_30d_return_rate", "label": "New Customer 30-Day Return Rate (%)", "value": rate},
    ]

    window_desc = []
    if getattr(req, "start_date", None):
        window_desc.append(f"from {req.start_date}")
    if getattr(req, "end_date", None):
        window_desc.append(f"to {req.end_date}")
    window_str = " ".join(window_desc) if window_desc else "for the full dataset"

    insight = (
        f"New Customer 30-Day Return Rate was computed {window_str} using invoice-derived first_visit joined with "
        f"customers index '{customers_index}' (original_signup). Return means the second visit occurred within "
        f"30 days of the first."
    )
    if truncated:
        insight += " NOTE: evaluation was capped (es_max_new_customers_check) so results may be approximate."

    return {
        "insight": to_json_safe(insight),
        "rows": to_json_safe(rows),
        "rules_used": business_rules or "",
        "engine": "es",
    }
def _es_customer_retention_rate_730_180(
    req,
    client,
    mappings: Dict[str, Any],
    business_rules: Optional[str],
):
    """
    Customer Retention Rate (730d cohort → 180d active).

    Definition:
      - Customers730     = customers with >0 lifetime sales AND last_visit in last 730 days
      - Active180From730 = subset of Customers730 with last_visit in last 180 days
      - Retention Rate   = (Active180From730 / Customers730) × 100
    """

    outer_days = 730
    inner_days = 180

    today = datetime.now(timezone.utc).date()
    cutoff_outer = today - timedelta(days=outer_days)
    cutoff_inner = today - timedelta(days=inner_days)
    cutoff_outer_str = cutoff_outer.isoformat()
    cutoff_inner_str = cutoff_inner.isoformat()

    # ------------------------------------------------------------
    # ✅ Preferred: compute directly from customers index
    # ------------------------------------------------------------
    customers_index = (getattr(req, "es_customers_index_name", "") or "").strip()

    if customers_index:
        try:
            cust_mapping_raw = client.indices.get_mapping(index=customers_index)
        except Exception:
            cust_mapping_raw = None

        if cust_mapping_raw is not None:
            cust_props = _extract_properties_from_mapping(cust_mapping_raw, customers_index)
            cust_mappings = {"properties": cust_props}

            # Need last_visit + sales_pickup_lifetime to apply canonical rules
            if _field_exists(cust_mappings, "last_visit") and _field_exists(
                cust_mappings, "sales_pickup_lifetime"
            ):
                # Base cohort: sales_pickup_lifetime > 0 AND last_visit within 730d
                body = {
                    "size": 0,
                    "query": {
                        "bool": {
                            "filter": [
                                {"range": {"sales_pickup_lifetime": {"gt": 0}}},
                                {"range": {"last_visit": {"gte": cutoff_outer_str}}},
                            ]
                        }
                    },
                    "aggs": {
                        "customers_730d": {
                            "value_count": {"field": "customer_id"}
                        },
                        "active_180_from_730": {
                            "filter": {
                                "range": {"last_visit": {"gte": cutoff_inner_str}}
                            },
                            "aggs": {
                                "customers_180d": {
                                    "value_count": {"field": "customer_id"}
                                }
                            },
                        },
                    },
                }

                res = _safe_es_search(client, index=customers_index, body=body)
                aggs = res.get("aggregations") or {}

                customers_730 = float((aggs.get("customers_730d") or {}).get("value") or 0.0)
                active_agg = (aggs.get("active_180_from_730") or {}).get("customers_180d") or {}
                active_180 = float(active_agg.get("value") or 0.0)

                rate = (active_180 * 100.0 / customers_730) if customers_730 > 0 else 0.0

                rows = [
                    {
                        "metric": "customers_730d",
                        "label": "Customers with visit in last 730 days",
                        "value": customers_730,
                    },
                    {
                        "metric": "active_180_from_730",
                        "label": "Still active (visited in last 180 days)",
                        "value": active_180,
                    },
                    {
                        "metric": "customer_retention_rate_730_to_180",
                        "label": "Customer Retention Rate 730d→180d (%)",
                        "value": rate,
                    },
                ]

                insight = (
                    "Customer Retention Rate (730d cohort) was computed using the customers index "
                    f"'{customers_index}'. Among customers with >0 lifetime sales and a last_visit in the last "
                    f"{outer_days} days, {int(active_180)} also visited in the last {inner_days} days, "
                    f"yielding a retention rate of {rate:.1f}%."
                )

                return {
                    "insight": to_json_safe(insight),
                    "rows": to_json_safe(rows),
                    "rules_used": business_rules or "",
                    "engine": "es",
                }

    # ------------------------------------------------------------
    # 🟡 Fallback: invoice-derived stats
    # ------------------------------------------------------------
    invoices_index_name = (getattr(req, "es_index_name", "") or "").strip()
    if not invoices_index_name:
        return _es_cannot_answer(
            "Customer Retention Rate requires either a customers index (es_customers_index_name) "
            "or an invoices index (es_index_name).",
            business_rules,
        )

    invoices_index, invoices_mappings = _get_invoice_index_and_mappings(client, invoices_index_name)

    stats = _get_customer_stats_invoices_only(req, client, invoices_index, invoices_mappings)
    if stats is None:
        return _es_cannot_answer(
            "Cannot compute 'Customer Retention Rate' because required fields are missing in both "
            "customers and invoices indices.",
            business_rules,
        )

    customers_730 = 0
    active_180 = 0

    for s in stats:
        last = s.get("last_visit")
        if not last:
            continue

        # Only customers who have generated revenue
        if (s.get("total_revenue") or 0.0) <= 0.0:
            continue

        days_since = (today - last.date()).days
        if days_since <= outer_days:
            customers_730 += 1
            if days_since <= inner_days:
                active_180 += 1

    customers_730_f = float(customers_730)
    active_180_f = float(active_180)
    rate = (active_180_f * 100.0 / customers_730_f) if customers_730_f > 0 else 0.0

    rows = [
        {
            "metric": "customers_730d",
            "label": "Customers with visit in last 730 days",
            "value": customers_730_f,
        },
        {
            "metric": "active_180_from_730",
            "label": "Still active (visited in last 180 days)",
            "value": active_180_f,
        },
        {
            "metric": "customer_retention_rate_730_to_180",
            "label": "Customer Retention Rate 730d→180d (%)",
            "value": rate,
        },
    ]

    insight = (
        "Customer Retention Rate (730d cohort) was computed from invoice-derived stats. "
        f"{customers_730} customers have >0 sales_pick_lifetime and a visit in the last {outer_days} days; "
        f"{active_180} of them also visited in the last {inner_days} days, "
        f"yielding a retention rate of {rate:.1f}%."
    )

    return {
        "insight": to_json_safe(insight),
        "rows": to_json_safe(rows),
        "rules_used": business_rules or "",
        "engine": "es",
    }
def _es_avg_customer_lifespan(
    req,
    client,
    mappings: Dict[str, Any],
    business_rules: Optional[str],
):
    """
    Average Customer Lifespan (in days).

    Definition:
      - Universe: customers in es_customers_index_name with BOTH:
          * original signup date
          * last_visit
      - For each customer:
          lifespan_days = (last_visit - original_signup) in days
      - Metric:
          Avg Lifespan = average(lifespan_days) across all included customers.

    Optional:
      - If start_date / end_date are provided on the request, we restrict to
        customers whose last_visit is within that window.
    """

    customers_index = (getattr(req, "es_customers_index_name", "") or "").strip()
    if not customers_index:
        return _es_cannot_answer(
            "Average Customer Lifespan requires a customers index (es_customers_index_name).",
            business_rules,
        )

    # Load customers mapping
    try:
        cust_mapping_raw = client.indices.get_mapping(index=customers_index)
    except Exception:
        cust_mapping_raw = None

    if cust_mapping_raw is None:
        return _es_cannot_answer(
            f"Could not load mappings for customers index '{customers_index}' "
            "when computing Average Customer Lifespan.",
            business_rules,
        )

    cust_props = _extract_properties_from_mapping(cust_mapping_raw, customers_index)
    cust_mappings = {"properties": cust_props}

    # Resolve original signup field name (we support a few variants)
    signup_field: Optional[str] = None
    for cand in ("original_signup", "signup_date"):
        if _field_exists(cust_mappings, cand):
            signup_field = cand
            break

    if not signup_field or not _field_exists(cust_mappings, "last_visit"):
        return _es_cannot_answer(
            "Cannot compute 'Average Customer Lifespan' because required fields "
            f"(original signup date and last_visit) are missing from customers index '{customers_index}'.",
            business_rules,
        )

    # Base filters: both dates must exist
    filters: List[Dict[str, Any]] = [
        {"exists": {"field": signup_field}},
        {"exists": {"field": "last_visit"}},
    ]

    # Optional window on last_visit if the question included dates
    start_d = _parse_date_str(getattr(req, "start_date", None))
    end_d = _parse_date_str(getattr(req, "end_date", None))

    if start_d:
        filters.append({"range": {"last_visit": {"gte": start_d.isoformat()}}})
    if end_d:
        filters.append({"range": {"last_visit": {"lte": end_d.isoformat()}}})

    body: Dict[str, Any] = {
        "size": 0,
        "query": {
            "bool": {
                "filter": filters,
            }
        },
        "aggs": {
            "avg_last_visit": {"avg": {"field": "last_visit"}},
            "avg_signup": {"avg": {"field": signup_field}},
            "count_customers": {"value_count": {"field": "customer_id"}},
        },
    }

    res = _safe_es_search(client, index=customers_index, body=body)
    aggs = res.get("aggregations") or {}

    avg_last_ms = (aggs.get("avg_last_visit") or {}).get("value")
    avg_signup_ms = (aggs.get("avg_signup") or {}).get("value")
    count = (aggs.get("count_customers") or {}).get("value") or 0

    if avg_last_ms is None or avg_signup_ms is None or count == 0:
        return {
            "insight": to_json_safe(
                "Average Customer Lifespan could not be computed because there were no customers with both "
                "an original signup date and a last_visit in the selected window."
            ),
            "rows": [],
            "rules_used": business_rules or "",
            "engine": "es",
        }

    ms_per_day = 1000.0 * 60 * 60 * 24
    avg_lifespan_days = (avg_last_ms - avg_signup_ms) / ms_per_day

    window_desc: List[str] = []
    if getattr(req, "start_date", None):
        window_desc.append(f"from {req.start_date}")
    if getattr(req, "end_date", None):
        window_desc.append(f"to {req.end_date}")
    window_str = " ".join(window_desc) if window_desc else "for all customers with data"

    insight = (
        "Average Customer Lifespan is defined as the average number of days between original signup "
        f"and last_visit for customers in '{customers_index}' who have both dates set. "
        f"Computed {window_str}, the average lifespan is approximately {avg_lifespan_days:.1f} days "
        f"across {int(count)} customers."
    )

    rows = [
        {
            "metric": "avg_customer_lifespan_days",
            "label": "Average Customer Lifespan (days)",
            "value": float(avg_lifespan_days),
            "customers_counted": int(count),
            "customers_index": customers_index,
            "signup_field": signup_field,
        }
    ]

    return {
        "insight": to_json_safe(insight),
        "rows": to_json_safe(rows),
        "rules_used": business_rules or "",
        "engine": "es",
    }
def _es_active_customer_rate(
    req,
    client,
    mappings: Dict[str, Any],
    business_rules: Optional[str],
    days_threshold: int = 180,
):
    """
    Active Customer Rate (% of paying customers who are currently active).

    Definition:
      - Total Customers with Sales:
          customers with sales_pickup_lifetime > 0
      - Active Customers:
          customers with sales_pickup_lifetime > 0
          AND last_visit within the last `days_threshold` days (default: 180)

      Active Customer Rate = (Active Customers / Total Customers with Sales) * 100

    Preferred path:
      - Use customers index (es_customers_index_name) with sales_pickup_lifetime + last_visit.

    Fallback:
      - Use invoice-derived stats (total_revenue, last_visit).
    """

    customers_index = (getattr(req, "es_customers_index_name", "") or "").strip()
    today = datetime.now(timezone.utc).date()
    cutoff_date = today - timedelta(days=int(days_threshold))
    cutoff_str = cutoff_date.isoformat()  # e.g. '2026-01-13'

    # ------------------------------------------------------------
    # ✅ Preferred: customers index
    # ------------------------------------------------------------
    if customers_index:
        try:
            cust_mapping_raw = client.indices.get_mapping(index=customers_index)
        except Exception:
            cust_mapping_raw = None

        if cust_mapping_raw is not None:
            cust_props = _extract_properties_from_mapping(cust_mapping_raw, customers_index)
            cust_mappings = {"properties": cust_props}

            if _field_exists(cust_mappings, "sales_pickup_lifetime") and _field_exists(
                cust_mappings, "last_visit"
            ):
                # Base query: paying customers (sales > 0)
                body = {
                    "size": 0,
                    "query": {
                        "bool": {
                            "filter": [
                                {"range": {"sales_pickup_lifetime": {"gt": 0}}},
                            ]
                        }
                    },
                    "aggs": {
                        # total paying customers (one doc per customer)
                        "total_customers_with_sales": {
                            "value_count": {"field": "customer_id"}
                        },
                        # active customers: paying + last_visit >= cutoff
                        "active_customers": {
                            "filter": {
                                "range": {"last_visit": {"gte": cutoff_str}}
                            }
                        },
                    },
                }

                res = _safe_es_search(client, index=customers_index, body=body)
                aggs = res.get("aggregations") or {}

                total_with_sales = float(
                    (aggs.get("total_customers_with_sales") or {}).get("value") or 0.0
                )
                active_customers = float(
                    (aggs.get("active_customers") or {}).get("doc_count") or 0.0
                )

                if total_with_sales <= 0:
                    return {
                        "insight": to_json_safe(
                            "Active Customer Rate could not be computed because there are no customers with "
                            "sales_pickup_lifetime > 0 in the customers index."
                        ),
                        "rows": [],
                        "rules_used": business_rules or "",
                        "engine": "es",
                    }

                rate = (active_customers * 100.0) / total_with_sales

                insight = (
                    "Active Customer Rate is defined as the percentage of paying customers "
                    f"(sales_pickup_lifetime > 0) whose last_visit is within the last {days_threshold} days. "
                    f"Using customers index '{customers_index}', there are {int(active_customers)} active customers "
                    f"out of {int(total_with_sales)} paying customers, giving an Active Customer Rate of "
                    f"approximately {rate:.1f}%."
                )

                rows = [
                    {
                        "metric": "active_customer_rate",
                        "label": "Active Customer Rate (%)",
                        "value": rate,
                        "active_customers": active_customers,
                        "total_customers_with_sales": total_with_sales,
                        "days_threshold": int(days_threshold),
                        "customers_index": customers_index,
                    }
                ]

                return {
                    "insight": to_json_safe(insight),
                    "rows": to_json_safe(rows),
                    "rules_used": business_rules or "",
                    "engine": "es",
                }

    # ------------------------------------------------------------
    # 🟡 Fallback: invoices-only behavior
    # ------------------------------------------------------------
    if not (req.es_index_name or "").strip():
        return _es_cannot_answer(
            "Missing invoices index (es_index_name), and customers index "
            "could not be used for 'Active Customer Rate'.",
            business_rules,
        )

    invoices_index, invoices_mappings = _get_invoice_index_and_mappings(client, req.es_index_name)

    stats = _get_customer_stats_invoices_only(req, client, invoices_index, invoices_mappings)
    if stats is None:
        return _es_cannot_answer(
            "Cannot compute 'Active Customer Rate' because required fields are missing in both "
            "customers and invoices indices.",
            business_rules,
        )

    total_with_sales = 0
    active_customers = 0

    for s in stats:
        last = s.get("last_visit")
        total_rev = float(s.get("total_revenue") or 0.0)

        if total_rev <= 0.0:
            # not a paying customer
            continue

        total_with_sales += 1

        if not last:
            continue

        days_since = (today - last.date()).days
        if days_since <= int(days_threshold):
            active_customers += 1

    if total_with_sales <= 0:
        return {
            "insight": to_json_safe(
                "Active Customer Rate could not be computed from invoice-derived stats because no customers "
                "with positive lifetime revenue were found."
            ),
            "rows": [],
            "rules_used": business_rules or "",
            "engine": "es",
        }

    rate = (active_customers * 100.0) / float(total_with_sales)

    insight = (
        "Active Customer Rate is defined as the percentage of paying customers "
        f"(lifetime revenue > 0) whose last visit is within the last {days_threshold} days. "
        "This was computed from invoice-derived stats as a fallback and yields "
        f"{active_customers} active customers out of {total_with_sales} paying customers "
        f"(~{rate:.1f}%)."
    )

    rows = [
        {
            "metric": "active_customer_rate",
            "label": "Active Customer Rate (%)",
            "value": rate,
            "active_customers": float(active_customers),
            "total_customers_with_sales": float(total_with_sales),
            "days_threshold": int(days_threshold),
            "source": "invoices_fallback",
        }
    ]

    return {
        "insight": to_json_safe(insight),
        "rows": to_json_safe(rows),
        "rules_used": business_rules or "",
        "engine": "es",
    }
def _es_30d_activity_rate(
    req,
    client,
    mappings: Dict[str, Any],
    business_rules: Optional[str],
):
    """
    30-Day Activity Rate = (Customers with sales_pickup_30 > 0 / Total Customers with Sales) × 100
    """
    customers_index = (getattr(req, "es_customers_index_name", "") or "").strip()
    if not customers_index:
        return _es_cannot_answer(
            "30-Day Activity Rate requires a customers index (es_customers_index_name).",
            business_rules,
        )

    # load mapping
    cust_mapping_raw = client.indices.get_mapping(index=customers_index)
    cust_props = _extract_properties_from_mapping(cust_mapping_raw, customers_index)
    cust_mappings = {"properties": cust_props}

    # need sales_pickup_lifetime + sales_pickup_30
    if not (
        _field_exists(cust_mappings, "sales_pickup_lifetime")
        and _field_exists(cust_mappings, "sales_pickup_30")
    ):
        return _es_cannot_answer(
            "Cannot compute 30-Day Activity Rate because sales_pickup_lifetime and/or "
            "sales_pickup_30 are missing from the customers mapping.",
            business_rules,
        )

    body = {
        "size": 0,
        "query": {
            "bool": {
                "filter": [
                    {"range": {"sales_pickup_lifetime": {"gt": 0}}},  # paying customers
                ]
            }
        },
        "aggs": {
            "total_customers_with_sales": {
                "value_count": {"field": "customer_id"}
            },
            "active_30d": {
                "filter": {
                    "range": {"sales_pickup_30": {"gt": 0}}
                },
                "aggs": {
                    "count": {"value_count": {"field": "customer_id"}}
                },
            },
        },
    }

    res = _safe_es_search(client, index=customers_index, body=body)
    aggs = res.get("aggregations") or {}

    total_with_sales = float(
        (aggs.get("total_customers_with_sales") or {}).get("value") or 0.0
    )
    active_30d = float(
        ((aggs.get("active_30d") or {}).get("count") or {}).get("value") or 0.0
    )

    if total_with_sales <= 0:
        return {
            "insight": to_json_safe(
                "30-Day Activity Rate could not be computed because no paying customers were found."
            ),
            "rows": [],
            "rules_used": business_rules or "",
            "engine": "es",
        }

    rate = (active_30d * 100.0) / total_with_sales

    rows = [
        {
            "metric": "activity_30d_rate",
            "label": "30-Day Activity Rate (%)",
            "value": rate,
            "customers_30d": active_30d,
            "total_customers_with_sales": total_with_sales,
        }
    ]

    insight = (
        "30-Day Activity Rate is defined as the percentage of paying customers "
        "(sales_pickup_lifetime > 0) who have made a purchase in the last 30 days "
        "(sales_pickup_30 > 0)."
    )

    return {
        "insight": to_json_safe(insight),
        "rows": to_json_safe(rows),
        "rules_used": business_rules or "",
        "engine": "es",
    }
def _es_avg_visit_interval(
    req,
    client,
    mappings: Dict[str, Any],
    business_rules: Optional[str],
):
    """
    Average Visit Interval (in days).

    Definition (customers index only):
      - Avg Visit Interval = AVERAGE(visits_interval_avg)

    Qualifying customers:
      - visits_lifetime >= min_visits_lifetime (default 2)
      - visits_interval_avg > min_interval_days (default 7 days)
      - original_signup <= today - min_signup_age_days (default 90 days ago)

    Notes:
      - Excludes very new customers (signup < 90 days) so they have time to form a pattern.
      - Excludes single-visit customers (no interval).
      - Excludes customers with extremely small intervals (<= 7 days) to avoid anomalies.

    You can override thresholds on the request:
      - req.min_visits_lifetime_for_interval
      - req.min_interval_days_for_avg
      - req.min_signup_age_days_for_interval
    """

    customers_index = (getattr(req, "es_customers_index_name", "") or "").strip()
    if not customers_index:
        return _es_cannot_answer(
            "Average Visit Interval requires a customers index (es_customers_index_name).",
            business_rules,
        )

    # Load customers mapping
    try:
        cust_mapping_raw = client.indices.get_mapping(index=customers_index)
    except Exception:
        cust_mapping_raw = None

    if cust_mapping_raw is None:
        return _es_cannot_answer(
            f"Could not load mappings for customers index '{customers_index}' "
            "when computing Average Visit Interval.",
            business_rules,
        )

    cust_props = _extract_properties_from_mapping(cust_mapping_raw, customers_index)
    cust_mappings = {"properties": cust_props}

    # Required fields
    required_fields = ["visits_interval_avg", "visits_lifetime", "original_signup", "customer_id"]
    missing = [f for f in required_fields if not _field_exists(cust_mappings, f)]
    if missing:
        return _es_cannot_answer(
            "Cannot compute 'Average Visit Interval' because required fields are missing "
            f"from customers index '{customers_index}': {', '.join(missing)}.",
            business_rules,
        )

    # Thresholds (configurable via req, with sane defaults)
    min_visits_lifetime = _get_req_int(
        req,
        "min_visits_lifetime_for_interval",
        2,
        min_v=2,
        max_v=10_000,
    )
    min_interval_days = _get_req_int(
        req,
        "min_interval_days_for_avg",
        7,
        min_v=1,
        max_v=10_000,
    )
    min_signup_age_days = _get_req_int(
        req,
        "min_signup_age_days_for_interval",
        90,
        min_v=1,
        max_v=10_000,
    )

    today = datetime.now(timezone.utc).date()
    cutoff_signup = today - timedelta(days=min_signup_age_days)
    cutoff_signup_str = cutoff_signup.isoformat()

    # Filters implementing the business rules
    filters: List[Dict[str, Any]] = [
        {"exists": {"field": "visits_interval_avg"}},
        {"range": {"visits_lifetime": {"gte": min_visits_lifetime}}},
        {"range": {"visits_interval_avg": {"gt": min_interval_days}}},
        {"range": {"original_signup": {"lte": cutoff_signup_str}}},
    ]

    body: Dict[str, Any] = {
        "size": 0,
        "query": {
            "bool": {
                "filter": filters,
            }
        },
        "aggs": {
            "avg_interval": {"avg": {"field": "visits_interval_avg"}},
            "count_customers": {"value_count": {"field": "customer_id"}},
        },
    }

    res = _safe_es_search(client, index=customers_index, body=body)
    aggs = res.get("aggregations") or {}

    avg_interval_val = (aggs.get("avg_interval") or {}).get("value")
    count = (aggs.get("count_customers") or {}).get("value") or 0

    if avg_interval_val is None or count == 0:
        return {
            "insight": to_json_safe(
                "Average Visit Interval could not be computed because no customers matched the filters "
                f"(visits_lifetime ≥ {min_visits_lifetime}, visits_interval_avg > {min_interval_days} days, "
                f"original_signup ≥ {min_signup_age_days} days ago)."
            ),
            "rows": [],
            "rules_used": business_rules or "",
            "engine": "es",
        }

    avg_interval_days = float(avg_interval_val)

    insight = (
        "Average Visit Interval is defined as the average of visits_interval_avg (in days) "
        "for established repeat customers. "
        f"Filters used: visits_lifetime ≥ {min_visits_lifetime}, visits_interval_avg > {min_interval_days} days, "
        f"original_signup at least {min_signup_age_days} days ago. "
        f"On customers index '{customers_index}', this yields an average interval of "
        f"approximately {avg_interval_days:.1f} days across {int(count)} customers."
    )

    rows = [
        {
            "metric": "avg_visit_interval_days",
            "label": "Average Visit Interval (days)",
            "value": avg_interval_days,
            "customers_counted": int(count),
            "customers_index": customers_index,
            "min_visits_lifetime": int(min_visits_lifetime),
            "min_interval_days": int(min_interval_days),
            "min_signup_age_days": int(min_signup_age_days),
        }
    ]

    return {
        "insight": to_json_safe(insight),
        "rows": to_json_safe(rows),
        "rules_used": business_rules or "",
        "engine": "es",
    }
def _es_pareto_80_20(
    req,
    client,
    mappings: Dict[str, Any],
    business_rules: Optional[str],
):
    """
    80/20 Rule (Pareto) for revenue concentration.

    Visual metric:
      - "What percentage of customers generate 80% of lifetime revenue?"

    Definition (customers index only):
      1. Consider only customers with sales_pickup_lifetime > 0
      2. total_revenue = SUM(sales_pickup_lifetime)
      3. total_customers = COUNT(customer_id) over those customers
      4. Sort customers by sales_pickup_lifetime DESC
      5. Walk down the list accumulating revenue until cumulative >= 80% of total_revenue
      6. pareto_pct = (customers_used / total_customers) * 100

    Notes:
      - Lower percentages indicate higher revenue concentration
        (e.g. 15% of customers generating 80% of revenue).
      - We cap how many top customers we inspect in Python for safety:
          req.max_customers_for_pareto (default ES_MAX_CUSTOMERS_DEFAULT).
        If 80% is not reached within that cap, result is flagged as approximate.
    """

    customers_index = (getattr(req, "es_customers_index_name", "") or "").strip()
    if not customers_index:
        return _es_cannot_answer(
            "Pareto 80/20 Rule requires a customers index (es_customers_index_name).",
            business_rules,
        )

    # Load customers mapping
    try:
        cust_mapping_raw = client.indices.get_mapping(index=customers_index)
    except Exception:
        cust_mapping_raw = None

    if cust_mapping_raw is None:
        return _es_cannot_answer(
            f"Could not load mappings for customers index '{customers_index}' "
            "when computing Pareto 80/20 Rule.",
            business_rules,
        )

    cust_props = _extract_properties_from_mapping(cust_mapping_raw, customers_index)
    cust_mappings = {"properties": cust_props}

    # Required fields
    required_fields = ["customer_id", "sales_pickup_lifetime"]
    missing = [f for f in required_fields if not _field_exists(cust_mappings, f)]
    if missing:
        return _es_cannot_answer(
            "Cannot compute 'Pareto 80/20 Rule' because required fields are missing "
            f"from customers index '{customers_index}': {', '.join(missing)}.",
            business_rules,
        )

    # Pareto target (normally 80%, but configurable if needed)
    pareto_target_pct = _get_req_int(
        req,
        "pareto_target_share",
        80,
        min_v=1,
        max_v=99,
    )
    pareto_target_fraction = pareto_target_pct / 100.0

    # Cap for how many top customers to inspect
    max_customers_for_pareto = _get_req_int(
        req,
        "max_customers_for_pareto",
        ES_MAX_CUSTOMERS_DEFAULT,
        min_v=100,
        max_v=200_000,
    )

    # --- 1) Global totals (all customers with revenue) via aggregations ---

    base_filter = [
        {"exists": {"field": "sales_pickup_lifetime"}},
        {"range": {"sales_pickup_lifetime": {"gt": 0}}},
    ]

    totals_body: Dict[str, Any] = {
        "size": 0,
        "query": {
            "bool": {
                "filter": base_filter,
            }
        },
        "aggs": {
            "total_revenue": {"sum": {"field": "sales_pickup_lifetime"}},
            "total_customers": {"value_count": {"field": "customer_id"}},
        },
    }

    totals_res = _safe_es_search(client, index=customers_index, body=totals_body)
    totals_aggs = totals_res.get("aggregations") or {}

    total_revenue = (totals_aggs.get("total_revenue") or {}).get("value") or 0.0
    total_customers = int((totals_aggs.get("total_customers") or {}).get("value") or 0)

    if total_customers == 0 or total_revenue <= 0:
        return {
            "insight": to_json_safe(
                "Pareto 80/20 Rule could not be computed because no customers with "
                "sales_pickup_lifetime > 0 were found in the customers index "
                f"'{customers_index}'."
            ),
            "rows": [],
            "rules_used": business_rules or "",
            "engine": "es",
        }

    # --- 2) Fetch top-N customers by lifetime revenue ---

    # We only inspect up to this many richest customers in Python
    limit = min(total_customers, max_customers_for_pareto)

    top_body: Dict[str, Any] = {
        "size": int(limit),
        "query": {
            "bool": {
                "filter": base_filter,
            }
        },
        "sort": [
            {"sales_pickup_lifetime": {"order": "desc"}},
        ],
        "_source": ["customer_id", "sales_pickup_lifetime"],
    }

    top_res = _safe_es_search(client, index=customers_index, body=top_body)
    hits = (top_res.get("hits") or {}).get("hits") or []

    if not hits:
        return {
            "insight": to_json_safe(
                "Pareto 80/20 Rule could not be computed because no top customers "
                "could be fetched from the customers index "
                f"'{customers_index}'."
            ),
            "rows": [],
            "rules_used": business_rules or "",
            "engine": "es",
        }

    # --- 3) Walk top-N until we reach the Pareto target ---

    target_revenue = total_revenue * pareto_target_fraction
    cumulative = 0.0
    customers_used = 0

    for h in hits:
        src = h.get("_source") or {}
        rev = src.get("sales_pickup_lifetime")
        if rev is None:
            continue
        try:
            rev_val = float(rev)
        except (TypeError, ValueError):
            continue
        if rev_val <= 0:
            continue

        cumulative += rev_val
        customers_used += 1

        if cumulative >= target_revenue:
            break

    # If we never reached the target within our safety cap,
    # then this metric is a lower bound / approximation.
    reached_target = cumulative >= target_revenue
    approximate = not reached_target

    if customers_used == 0:
        return {
            "insight": to_json_safe(
                "Pareto 80/20 Rule could not be computed because no valid customer "
                "revenues were found in the top segment."
            ),
            "rows": [],
            "rules_used": business_rules or "",
            "engine": "es",
        }

    pareto_pct = (customers_used / total_customers) * 100.0

    if approximate:
        insight = (
            f"Pareto {pareto_target_pct}/100 Rule (revenue concentration): "
            f"we inspected the top {customers_used} customers out of {total_customers} "
            f"and did not fully reach {pareto_target_pct}% of total revenue within the "
            f"safety cap of {max_customers_for_pareto} customers. "
            f"The computed value ({pareto_pct:.1f}% of customers) is therefore a "
            "lower-bound approximation; in reality, it likely takes a higher share "
            "of your customers to generate that revenue fraction."
        )
    else:
        insight = (
            f"Pareto {pareto_target_pct}/100 Rule (revenue concentration): "
            f"on customers index '{customers_index}', approximately {pareto_pct:.1f}% "
            f"of customers (top {customers_used} by lifetime revenue out of "
            f"{total_customers}) are responsible for {pareto_target_pct}% of "
            "total sales_pickup_lifetime. Lower percentages indicate a more "
            "concentrated revenue base."
        )

    rows = [
        {
            "metric": "pareto_80_20_customers_pct",
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
def _es_single_visit_lifetime(
    req,
    client,
    mappings: Dict[str, Any],
    business_rules: Optional[str],
):
    """
    Single Visit (Lifetime) %

    Definition (customers index only):

      Single Visit Lifetime % =
        (COUNT of customers with visits_lifetime = 1
         / Total Customers with Sales) × 100

      Where:
        - Total Customers with Sales = customers with sales_pickup_lifetime > 0

    Notes:
      - Tells you what % of paying customers only ever visited once.
      - Industry benchmark is around 37% (higher means weaker retention).
    """

    customers_index = (getattr(req, "es_customers_index_name", "") or "").strip()
    if not customers_index:
        return _es_cannot_answer(
            "Single Visit (Lifetime) requires a customers index (es_customers_index_name).",
            business_rules,
        )

    # Load customers mapping
    try:
        cust_mapping_raw = client.indices.get_mapping(index=customers_index)
    except Exception:
        cust_mapping_raw = None

    if cust_mapping_raw is None:
        return _es_cannot_answer(
            f"Could not load mappings for customers index '{customers_index}' "
            "when computing Single Visit (Lifetime).",
            business_rules,
        )

    cust_props = _extract_properties_from_mapping(cust_mapping_raw, customers_index)
    cust_mappings = {"properties": cust_props}

    # Required fields
    required_fields = ["customer_id", "visits_lifetime", "sales_pickup_lifetime"]
    missing = [f for f in required_fields if not _field_exists(cust_mappings, f)]
    if missing:
        return _es_cannot_answer(
            "Cannot compute 'Single Visit (Lifetime)' because required fields are missing "
            f"from customers index '{customers_index}': {', '.join(missing)}.",
            business_rules,
        )

    # Base filter: "customers with sales" (denominator)
    base_filter: List[Dict[str, Any]] = [
        {"exists": {"field": "sales_pickup_lifetime"}},
        {"range": {"sales_pickup_lifetime": {"gt": 0}}},
    ]

    body: Dict[str, Any] = {
        "size": 0,
        "query": {
            "bool": {
                "filter": base_filter,
            }
        },
        "aggs": {
            # Total customers with sales (denominator)
            "customers_with_sales": {
                "value_count": {"field": "customer_id"}
            },
            # Single-visit customers (numerator)
            "single_visit": {
                "filter": {"term": {"visits_lifetime": 1}},
                "aggs": {
                    "customers": {
                        "value_count": {"field": "customer_id"}
                    }
                },
            },
        },
    }

    res = _safe_es_search(client, index=customers_index, body=body)
    aggs = res.get("aggregations") or {}

    total_with_sales = (aggs.get("customers_with_sales") or {}).get("value") or 0
    single_bucket = (aggs.get("single_visit") or {})
    single_count = (single_bucket.get("customers") or {}).get("value") or 0

    total_with_sales = int(total_with_sales)
    single_count = int(single_count)

    if total_with_sales == 0:
        return {
            "insight": to_json_safe(
                "Single Visit (Lifetime) % could not be computed because no customers "
                f"with sales_pickup_lifetime > 0 were found in customers index "
                f"'{customers_index}'."
            ),
            "rows": [],
            "rules_used": business_rules or "",
            "engine": "es",
        }

    single_pct = (single_count / total_with_sales) * 100.0

    insight = (
        "Single Visit (Lifetime) % measures what share of all paying customers have "
        "only ever visited once. It is calculated as:\n"
        "  (customers with visits_lifetime = 1) / (customers with sales_pickup_lifetime > 0) × 100.\n"
        f"On customers index '{customers_index}', approximately {single_pct:.1f}% of paying "
        f"customers ({single_count} out of {total_with_sales}) are single-visit customers. "
        "Higher values indicate weaker retention; many industries see around 37% as a typical benchmark."
    )

    rows = [
        {
            "metric": "single_visit_lifetime_pct",
            "label": "Single Visit (Lifetime) %",
            "value": single_pct,
            "customers_single_visit": single_count,
            "customers_with_sales": total_with_sales,
            "customers_index": customers_index,
        }
    ]

    return {
        "insight": to_json_safe(insight),
        "rows": to_json_safe(rows),
        "rules_used": business_rules or "",
        "engine": "es",
    }
def _es_single_visit_365(
    req,
    client,
    mappings: Dict[str, Any],
    business_rules: Optional[str],
):
    """
    Single Visit (365 Days)

    Visual Example:
      ┌─────────────────────┐
      │ Single Visit (365d) │
      │       42.3%         │
      │   18,234 customers  │
      └─────────────────────┘

    Calculation:
      1. Filter customers with visits_365 > 0
      2. EXCLUDE customers where:
           - original_signup < min_signup_age_days (default 180 days ago)
           - AND visits_365 = 1
      3. Single Visit 365 % = (remaining 1-visit customers / filtered total) × 100

    Notes:
      - This excludes very new 1-visit customers who haven't had time to come back.
      - "New" here means: original_signup > today - min_signup_age_days.
      - You can override the 180 days with:
          req.min_signup_age_days_for_single_365
    """

    customers_index = (getattr(req, "es_customers_index_name", "") or "").strip()
    if not customers_index:
        return _es_cannot_answer(
            "Single Visit 365% requires a customers index (es_customers_index_name).",
            business_rules,
        )

    # Load customers mapping
    try:
        cust_mapping_raw = client.indices.get_mapping(index=customers_index)
    except Exception:
        cust_mapping_raw = None

    if cust_mapping_raw is None:
        return _es_cannot_answer(
            f"Could not load mappings for customers index '{customers_index}' "
            "when computing Single Visit 365%.",
            business_rules,
        )

    cust_props = _extract_properties_from_mapping(cust_mapping_raw, customers_index)
    cust_mappings = {"properties": cust_props}

    # Required fields
    required_fields = ["visits_365", "original_signup", "customer_id"]
    missing = [f for f in required_fields if not _field_exists(cust_mappings, f)]
    if missing:
        return _es_cannot_answer(
            "Cannot compute 'Single Visit 365%' because required fields are missing "
            f"from customers index '{customers_index}': {', '.join(missing)}.",
            business_rules,
        )

    # Threshold (what we call "new" vs established)
    min_signup_age_days = _get_req_int(
        req,
        "min_signup_age_days_for_single_365",
        180,          # default: 180 days
        min_v=1,
        max_v=10_000,
    )

    today = datetime.now(timezone.utc).date()
    cutoff_signup = today - timedelta(days=min_signup_age_days)
    cutoff_signup_str = cutoff_signup.isoformat()

    # Base filter: at least one visit in last 365 days
    base_filters: List[Dict[str, Any]] = [
        {"exists": {"field": "visits_365"}},
        {"range": {"visits_365": {"gt": 0}}},
    ]

    # Exclude "new, one-visit" customers:
    #   original_signup > cutoff_signup  AND  visits_365 = 1
    must_not = [
        {
            "bool": {
                "filter": [
                    {"range": {"original_signup": {"gt": cutoff_signup_str}}},
                    {"term": {"visits_365": 1}},
                ]
            }
        }
    ]

    body: Dict[str, Any] = {
        "size": 0,
        "query": {
            "bool": {
                "filter": base_filters,
                "must_not": must_not,
            }
        },
        "aggs": {
            # All eligible customers after filters/must_not
            "total_customers": {"value_count": {"field": "customer_id"}},

            # Among those, how many have exactly 1 visit in last 365d
            "single_visit_customers": {
                "filter": {"term": {"visits_365": 1}},
                "aggs": {
                    "count": {"value_count": {"field": "customer_id"}}
                },
            },
        },
    }

    res = _safe_es_search(client, index=customers_index, body=body)
    aggs = res.get("aggregations") or {}

    total = (aggs.get("total_customers") or {}).get("value") or 0
    single_visit_bucket = aggs.get("single_visit_customers") or {}
    single_visit_count = (single_visit_bucket.get("count") or {}).get("value") or 0

    if total == 0:
        return {
            "insight": to_json_safe(
                "Single Visit 365% could not be computed because no customers matched "
                "the filters (visits_365 > 0 after excluding very new one-visit customers)."
            ),
            "rows": [],
            "rules_used": business_rules or "",
            "engine": "es",
        }

    pct_single = (single_visit_count / total) * 100.0

    insight = (
        "Single Visit 365% measures the percentage of customers who had exactly one visit "
        "in the last 365 days, after excluding very new one-visit customers. "
        f"We treat customers as 'new' if their signup is within the last {min_signup_age_days} days. "
        f"On index '{customers_index}', {int(single_visit_count)} out of {int(total)} "
        f"eligible customers are single-visit in 365 days, giving approximately "
        f"{pct_single:.1f}%."
    )

    rows = [
        {
            "metric": "single_visit_365_pct",
            "label": "Single Visit 365% (established customers)",
            "value": pct_single,
            "single_visit_customers": int(single_visit_count),
            "total_customers": int(total),
            "min_signup_age_days": int(min_signup_age_days),
            "customers_index": customers_index,
        }
    ]

    return {
        "insight": to_json_safe(insight),
        "rows": to_json_safe(rows),
        "rules_used": business_rules or "",
        "engine": "es",
    }
def _es_top_5pct_revenue_from_tags(
    req,
    client,
    mappings: Dict[str, Any],
    business_rules: Optional[str],
):
    """
    Top 5% Revenue Contribution (using customer tag 'Top 5%').

    Logic:
      - Revenue basis: sales_pickup_365 (can be overridden).
      - 'Top 5%' customers are identified by nested tag: tags.name = "Top 5%".
      - Only customers with revenue > 0 are considered.

    Calculation:
      1) totalRevenue = SUM(revenue_field) for all customers with revenue_field > 0
      2) top5Revenue = SUM(revenue_field) for customers with tag 'Top 5%'
      3) Top 5% Revenue % = (top5Revenue / totalRevenue) * 100
    """

    customers_index = (getattr(req, "es_customers_index_name", "") or "").strip()
    if not customers_index:
        return _es_cannot_answer(
            "Top 5% Revenue Contribution requires a customers index (es_customers_index_name).",
            business_rules,
        )

    # Choose which field to use for revenue (default: sales_pickup_365)
    revenue_field = getattr(req, "top5_revenue_field", None) or "sales_pickup_365"

    # Load mapping
    try:
        cust_mapping_raw = client.indices.get_mapping(index=customers_index)
    except Exception:
        cust_mapping_raw = None

    if cust_mapping_raw is None:
        return _es_cannot_answer(
            f"Could not load mappings for customers index '{customers_index}' "
            "when computing Top 5% Revenue Contribution.",
            business_rules,
        )

    cust_props = _extract_properties_from_mapping(cust_mapping_raw, customers_index)
    cust_mappings = {"properties": cust_props}

    required_fields = [revenue_field, "customer_id", "tags"]
    missing = [f for f in required_fields if not _field_exists(cust_mappings, f)]
    if missing:
        return _es_cannot_answer(
            "Cannot compute 'Top 5% Revenue Contribution' because required fields are missing "
            f"from customers index '{customers_index}': {', '.join(missing)}.",
            business_rules,
        )

    # Base query = match_all, filters happen in aggs
    body: Dict[str, Any] = {
        "size": 0,
        "query": {"match_all": {}},
        "aggs": {
            # All customers with revenue > 0
            "total_revenue": {
                "filter": {
                    "range": {revenue_field: {"gt": 0}}
                },
                "aggs": {
                    "customers": {"value_count": {"field": "customer_id"}},
                    "revenue": {"sum": {"field": revenue_field}},
                },
            },
            # Customers tagged "Top 5%" with revenue > 0
            "top5_revenue": {
                "filter": {
                    "bool": {
                        "filter": [
                            {"range": {revenue_field: {"gt": 0}}},
                            {
                                "nested": {
                                    "path": "tags",
                                    "query": {
                                        "term": {
                                            "tags.name.keyword": "Top 5%"
                                        }
                                    },
                                }
                            },
                        ]
                    }
                },
                "aggs": {
                    "customers": {"value_count": {"field": "customer_id"}},
                    "revenue": {"sum": {"field": revenue_field}},
                },
            },
        },
    }

    res = _safe_es_search(client, index=customers_index, body=body)
    aggs = res.get("aggregations") or {}

    total_bucket = aggs.get("total_revenue") or {}
    top5_bucket = aggs.get("top5_revenue") or {}

    total_revenue = (total_bucket.get("revenue") or {}).get("value") or 0.0
    total_customers = (total_bucket.get("customers") or {}).get("value") or 0

    top5_revenue = (top5_bucket.get("revenue") or {}).get("value") or 0.0
    top5_customers = (top5_bucket.get("customers") or {}).get("value") or 0

    if total_revenue <= 0:
        return {
            "insight": to_json_safe(
                "Top 5% Revenue Contribution could not be computed because no customers "
                f"had positive {revenue_field}."
            ),
            "rows": [],
            "rules_used": business_rules or "",
            "engine": "es",
        }

    pct = (top5_revenue / total_revenue) * 100.0

    insight = (
        "Top 5% Revenue Contribution measures how much of your total revenue is generated "
        "by customers tagged as 'Top 5%' (based on last 365 days spending). "
        f"Using revenue field '{revenue_field}' on index '{customers_index}', "
        f"{int(top5_customers)} out of {int(total_customers)} revenue-generating customers "
        f"contribute approximately {pct:.1f}% of total revenue."
    )

    rows = [
        {
            "metric": "top_5pct_revenue_pct",
            "label": "Top 5% Revenue Contribution (%)",
            "value": pct,
            "top5_customers": int(top5_customers),
            "total_customers": int(total_customers),
            "top5_revenue": float(top5_revenue),
            "total_revenue": float(total_revenue),
            "revenue_field": revenue_field,
            "customers_index": customers_index,
        }
    ]

    return {
        "insight": to_json_safe(insight),
        "rows": to_json_safe(rows),
        "rules_used": business_rules or "",
        "engine": "es",
    }
def _es_visit_frequency_365(
    req,
    client,
    mappings: Dict[str, Any],
    business_rules: Optional[str],
):
    """
    Visit Frequency – 365 Days (customers index).

    Buckets:
      - 1 Visit
      - 2–5 Visits
      - 6–11 Visits
      - 12–24 Visits
      - 25+ Visits

    Definition (customers index):
      - visits_365: number of visits in last 365 days
      - original_signup: customer signup date

    Base population:
      - customers with visits_365 > 0
      - original_signup is present

    EXCLUSION (important):
      - Exclude customers where:
          • original_signup is within the last N days (default 180)
          • AND visits_365 = 1
      -> This prevents very new one-visit customers from inflating the single-visit bucket.

    Optional override:
      - req.min_signup_age_days_for_visit_frequency_365  (default 180)
    """

    # ---------------------------------------------
    # 1) Resolve customers index & load mappings
    # ---------------------------------------------
    customers_index = (getattr(req, "es_customers_index_name", "") or "").strip()
    if not customers_index:
        return _es_cannot_answer(
            "Visit Frequency – 365 Days requires a customers index (es_customers_index_name).",
            business_rules,
        )

    try:
        cust_mapping_raw = client.indices.get_mapping(index=customers_index)
    except Exception:
        cust_mapping_raw = None

    if cust_mapping_raw is None:
        return _es_cannot_answer(
            f"Could not load mappings for customers index '{customers_index}' "
            "when computing Visit Frequency – 365 Days.",
            business_rules,
        )

    cust_props = _extract_properties_from_mapping(cust_mapping_raw, customers_index)
    cust_mappings = {"properties": cust_props}

    # Required fields
    required_fields = ["visits_365", "original_signup"]
    missing = [f for f in required_fields if not _field_exists(cust_mappings, f)]
    if missing:
        return _es_cannot_answer(
            "Cannot compute 'Visit Frequency – 365 Days' because required fields are missing "
            f"from customers index '{customers_index}': {', '.join(missing)}.",
            business_rules,
        )

    # ---------------------------------------------
    # 2) Signup age threshold (for exclusion)
    # ---------------------------------------------
    raw_days = getattr(req, "min_signup_age_days_for_visit_frequency_365", 180)
    try:
        min_signup_age_days = int(raw_days or 180)
    except Exception:
        min_signup_age_days = 180

    # Clamp to sane range
    min_signup_age_days = max(1, min(min_signup_age_days, 3650))

    today = datetime.now(timezone.utc).date()
    cutoff_signup = today - timedelta(days=min_signup_age_days)
    cutoff_signup_str = cutoff_signup.isoformat()

    # ---------------------------------------------
    # 3) Base query + exclusion rule
    # ---------------------------------------------
    # Base population: customers with visits_365 > 0 and a known signup date
    base_filters: List[Dict[str, Any]] = [
        {"exists": {"field": "visits_365"}},
        {"range": {"visits_365": {"gt": 0}}},
        {"exists": {"field": "original_signup"}},
    ]

    # Exclusion: new customers with exactly 1 visit (within last N days)
    exclusion_clause: Dict[str, Any] = {
        "bool": {
            "filter": [
                {"range": {"visits_365": {"gte": 1, "lte": 1}}},
                {"range": {"original_signup": {"gte": cutoff_signup_str}}},
            ]
        }
    }

    query: Dict[str, Any] = {
        "bool": {
            "filter": base_filters,
            "must_not": [exclusion_clause],
        }
    }

    # ---------------------------------------------
    # 4) Bucket definitions on visits_365
    # ---------------------------------------------
    bucket_filters: Dict[str, Any] = {
        "1_visit": {"range": {"visits_365": {"gte": 1, "lte": 1}}},
        "2_5": {"range": {"visits_365": {"gte": 2, "lte": 5}}},
        "6_11": {"range": {"visits_365": {"gte": 6, "lte": 11}}},
        "12_24": {"range": {"visits_365": {"gte": 12, "lte": 24}}},
        "25_plus": {"range": {"visits_365": {"gte": 25}}},
    }

    body: Dict[str, Any] = {
        "size": 0,
        "query": query,
        "aggs": {
            "visit_frequency": {
                "filters": {
                    "filters": bucket_filters
                }
            }
        },
    }

    # ---------------------------------------------
    # 5) Run ES query
    # ---------------------------------------------
    res = _safe_es_search(client, index=customers_index, body=body)
    aggs = res.get("aggregations") or {}
    vf = (aggs.get("visit_frequency") or {})
    buckets = vf.get("buckets") or {}

    if not buckets:
        return {
            "insight": to_json_safe(
                "Visit Frequency – 365 Days could not be computed because no customers matched "
                "the filters (visits_365 > 0, original_signup present, excluding new single-visit customers)."
            ),
            "rows": [],
            "rules_used": business_rules or "",
            "engine": "es",
        }

    # ---------------------------------------------
    # 6) Convert buckets -> rows with percentages
    # ---------------------------------------------
    # Keep a fixed order & labels for the dashboard
    bucket_meta = [
        ("1_visit", "1 Visit", 1, 1),
        ("2_5", "2–5 Visits", 2, 5),
        ("6_11", "6–11 Visits", 6, 11),
        ("12_24", "12–24 Visits", 12, 24),
        ("25_plus", "25+ Visits", 25, None),
    ]

    # total customers in our base population
    total_customers = 0
    for key, _, _, _ in bucket_meta:
        b = buckets.get(key) or {}
        total_customers += int(b.get("doc_count") or 0)

    if total_customers == 0:
        return {
            "insight": to_json_safe(
                "Visit Frequency – 365 Days found zero customers after applying the exclusion rule "
                f"(removing customers with original_signup within the last {min_signup_age_days} days "
                "and exactly 1 visit in 365 days)."
            ),
            "rows": [],
            "rules_used": business_rules or "",
            "engine": "es",
        }

    rows: List[Dict[str, Any]] = []
    for key, label, min_v, max_v in bucket_meta:
        b = buckets.get(key) or {}
        count = int(b.get("doc_count") or 0)
        pct = (count * 100.0 / float(total_customers)) if total_customers > 0 else 0.0

        rows.append(
            {
                "segment": key,  # stable ID for charts ("1_visit", "2_5", etc.)
                "label": label,  # human label ("1 Visit", "2–5 Visits", ...)
                "min_visits_365": min_v,
                "max_visits_365": max_v,
                "customer_count": count,
                "percentage_of_customers": pct,
            }
        )

    # ---------------------------------------------
    # 7) Insight text for the dashboard
    # ---------------------------------------------
    insight = (
        "Visit Frequency – 365 Days is computed on the customers index "
        f"'{customers_index}' using 'visits_365' as the number of visits in the last 365 days "
        "and 'original_signup' as the signup date. "
        "We first select all customers with visits_365 > 0 and a known signup date, "
        f"then exclude new customers who signed up within the last {min_signup_age_days} days "
        "and have exactly 1 visit in 365 days to avoid artificially inflating the single-visit bucket. "
        "Remaining customers are grouped into frequency buckets (1, 2–5, 6–11, 12–24, 25+ visits), "
        "and percentages are computed relative to this filtered base population."
    )

    return {
        "insight": to_json_safe(insight),
        "rows": to_json_safe(rows),
        "rules_used": business_rules or "",
        "engine": "es",
    }
def _es_visit_frequency_730(
    req,
    client,
    mappings: Dict[str, Any],
    business_rules: Optional[str],
):
    """
    Visit Frequency – 730 Days (customers index).

    Uses visits_lifetime for customers whose last_visit is within a window
    (default 730 days). New 1-visit customers are excluded using the same
    rule as the 365-day chart.

    Buckets (on visits_lifetime):
      - 1 Visit
      - 2–5 Visits
      - 6–11 Visits
      - 12–24 Visits
      - 25+ Visits

    Base population:
      - customers with visits_lifetime > 0
      - original_signup present
      - last_visit within the last N days (default 730)

    EXCLUSION:
      - Exclude customers where:
          • visits_lifetime = 1
          • AND original_signup < N2 days old (default 180 days, i.e. very new)
        This is the same “new single-visit” exclusion as the 365-day metric.

    Optional overrides:
      - req.visit_frequency_730_window_days (default 730)
      - req.min_signup_age_days_for_visit_frequency_730 (default 180)
    """

    # ---------------------------------------------
    # 1) Resolve customers index & load mappings
    # ---------------------------------------------
    customers_index = (getattr(req, "es_customers_index_name", "") or "").strip()
    if not customers_index:
        return _es_cannot_answer(
            "Visit Frequency – 730 Days requires a customers index (es_customers_index_name).",
            business_rules,
        )

    try:
        cust_mapping_raw = client.indices.get_mapping(index=customers_index)
    except Exception:
        cust_mapping_raw = None

    if cust_mapping_raw is None:
        return _es_cannot_answer(
            f"Could not load mappings for customers index '{customers_index}' "
            "when computing Visit Frequency – 730 Days.",
            business_rules,
        )

    cust_props = _extract_properties_from_mapping(cust_mapping_raw, customers_index)
    cust_mappings = {"properties": cust_props}

    # Required fields for this metric
    required_fields = ["visits_lifetime", "original_signup", "last_visit"]
    missing = [f for f in required_fields if not _field_exists(cust_mappings, f)]
    if missing:
        return _es_cannot_answer(
            "Cannot compute 'Visit Frequency – 730 Days' because required fields are missing "
            f"from customers index '{customers_index}': {', '.join(missing)}.",
            business_rules,
        )

    # ---------------------------------------------
    # 2) Window length & signup-age thresholds
    # ---------------------------------------------
    # Window length: how far back we look on last_visit
    raw_window_days = getattr(req, "visit_frequency_730_window_days", 730)
    try:
        window_days = int(raw_window_days or 730)
    except Exception:
        window_days = 730
    window_days = max(30, min(window_days, 3650))  # clamp to [30, 10 years]

    # New-customer signup age threshold (for exclusion)
    raw_signup_days = getattr(req, "min_signup_age_days_for_visit_frequency_730", 180)
    try:
        min_signup_age_days = int(raw_signup_days or 180)
    except Exception:
        min_signup_age_days = 180
    min_signup_age_days = max(1, min(min_signup_age_days, 3650))

    today = datetime.now(timezone.utc).date()
    cutoff_last_visit = today - timedelta(days=window_days)
    cutoff_signup = today - timedelta(days=min_signup_age_days)

    cutoff_last_visit_str = cutoff_last_visit.isoformat()
    cutoff_signup_str = cutoff_signup.isoformat()

    # ---------------------------------------------
    # 3) Base query + exclusion rule
    # ---------------------------------------------
    # Base population:
    #   - visits_lifetime > 0
    #   - original_signup exists
    #   - last_visit exists AND last_visit within window (>= cutoff_last_visit)
    base_filters: List[Dict[str, Any]] = [
        {"exists": {"field": "visits_lifetime"}},
        {"range": {"visits_lifetime": {"gt": 0}}},
        {"exists": {"field": "original_signup"}},
        {"exists": {"field": "last_visit"}},
        {"range": {"last_visit": {"gte": cutoff_last_visit_str}}},
    ]

    # Exclusion (same idea as 365-day):
    # remove "very new" one-visit customers:
    #   visits_lifetime = 1 AND original_signup within last min_signup_age_days
    exclusion_clause: Dict[str, Any] = {
        "bool": {
            "filter": [
                {"range": {"visits_lifetime": {"gte": 1, "lte": 1}}},
                {"range": {"original_signup": {"gte": cutoff_signup_str}}},
            ]
        }
    }

    query: Dict[str, Any] = {
        "bool": {
            "filter": base_filters,
            "must_not": [exclusion_clause],
        }
    }

    # ---------------------------------------------
    # 4) Bucket definitions on visits_lifetime
    # ---------------------------------------------
    bucket_filters: Dict[str, Any] = {
        "1_visit": {"range": {"visits_lifetime": {"gte": 1, "lte": 1}}},
        "2_5": {"range": {"visits_lifetime": {"gte": 2, "lte": 5}}},
        "6_11": {"range": {"visits_lifetime": {"gte": 6, "lte": 11}}},
        "12_24": {"range": {"visits_lifetime": {"gte": 12, "lte": 24}}},
        "25_plus": {"range": {"visits_lifetime": {"gte": 25}}},
    }

    body: Dict[str, Any] = {
        "size": 0,
        "query": query,
        "aggs": {
            "visit_frequency_730": {
                "filters": {
                    "filters": bucket_filters
                }
            }
        },
    }

    # ---------------------------------------------
    # 5) Run ES query
    # ---------------------------------------------
    res = _safe_es_search(client, index=customers_index, body=body)
    aggs = res.get("aggregations") or {}
    vf = (aggs.get("visit_frequency_730") or {})
    buckets = vf.get("buckets") or {}

    if not buckets:
        return {
            "insight": to_json_safe(
                "Visit Frequency – 730 Days could not be computed because no customers matched "
                "the filters (visits_lifetime > 0, last_visit within window, original_signup present, "
                "excluding new single-visit customers)."
            ),
            "rows": [],
            "rules_used": business_rules or "",
            "engine": "es",
        }

    # ---------------------------------------------
    # 6) Convert buckets -> rows (counts + %)
    # ---------------------------------------------
    bucket_meta = [
        ("1_visit", "1 Visit", 1, 1),
        ("2_5", "2–5 Visits", 2, 5),
        ("6_11", "6–11 Visits", 6, 11),
        ("12_24", "12–24 Visits", 12, 24),
        ("25_plus", "25+ Visits", 25, None),
    ]

    total_customers = 0
    for key, _, _, _ in bucket_meta:
        b = buckets.get(key) or {}
        total_customers += int(b.get("doc_count") or 0)

    if total_customers == 0:
        return {
            "insight": to_json_safe(
                "Visit Frequency – 730 Days found zero customers after applying the filters and "
                f"the exclusion rule (removing customers with original_signup within the last "
                f"{min_signup_age_days} days and exactly 1 lifetime visit)."
            ),
            "rows": [],
            "rules_used": business_rules or "",
            "engine": "es",
        }

    rows: List[Dict[str, Any]] = []
    for key, label, min_v, max_v in bucket_meta:
        b = buckets.get(key) or {}
        count = int(b.get("doc_count") or 0)
        pct = (count * 100.0 / float(total_customers)) if total_customers > 0 else 0.0

        rows.append(
            {
                "segment": key,  # stable ID for chart
                "label": label,  # “1 Visit”, “2–5 Visits”, etc.
                "min_visits_lifetime": min_v,
                "max_visits_lifetime": max_v,
                "customer_count": count,
                "percentage_of_customers": pct,
                "window_days": window_days,
            }
        )

    # ---------------------------------------------
    # 7) Insight text
    # ---------------------------------------------
    insight = (
        "Visit Frequency – 730 Days is computed on the customers index "
        f"'{customers_index}' using 'visits_lifetime' as the total number of visits and "
        "'last_visit' to restrict customers to those active within the last "
        f"{window_days} days. "
        "We start from all customers with visits_lifetime > 0, a known signup date, and "
        "last_visit within the window. "
        f"We then exclude very new single-visit customers (original_signup in the last "
        f"{min_signup_age_days} days with visits_lifetime = 1) to avoid artificially "
        "inflating the single-visit bucket. "
        "The remaining customers are grouped into lifetime frequency buckets "
        "(1, 2–5, 6–11, 12–24, 25+ visits), and percentages are computed relative to "
        "this filtered base population to give a two-year view of visit patterns."
    )

    return {
        "insight": to_json_safe(insight),
        "rows": to_json_safe(rows),
        "rules_used": business_rules or "",
        "engine": "es",
    }
def _es_route_vs_retail_comparison(
    req,
    client,
    mappings: Dict[str, Any],
    business_rules: Optional[str],
):
    """
    Route vs Retail Comparison (customers index).

    ✅ FIXED for your mapping:
      - route is NESTED
      - use route.name.keyword and route.route_id (NOT route.keyword)

    Segmentation:
      - Retail = route missing OR route_id missing OR route.name in {"Retail","Unassigned"}
      - Route  = route_id exists AND route.name not in {"Retail","Unassigned"}

    Metrics per segment:
      - customer_count
      - revenue (sales_pickup_lifetime)
      - revenue_share_pct (segment_revenue / total_revenue)
      - avg_ltv = segment_revenue / customer_count
      - avg_visit_value ≈ segment_revenue / total_visits_lifetime
      - avg_pieces_per_visit ≈ total_pieces_lifetime / total_visits_lifetime (if pieces field exists)
      - visits_per_year ≈ (avg visits per customer) / (avg lifespan in years)
    """

    customers_index = (getattr(req, "es_customers_index_name", "") or "").strip()
    if not customers_index:
        return _es_cannot_answer(
            "Route vs Retail Comparison requires a customers index (es_customers_index_name).",
            business_rules,
        )

    # Load customers mapping
    try:
        cust_mapping_raw = client.indices.get_mapping(index=customers_index)
    except Exception:
        cust_mapping_raw = None

    if cust_mapping_raw is None:
        return _es_cannot_answer(
            f"Could not load mappings for customers index '{customers_index}' "
            "when computing Route vs Retail Comparison.",
            business_rules,
        )

    cust_props = _extract_properties_from_mapping(cust_mapping_raw, customers_index)
    cust_mappings = {"properties": cust_props}

    # Detect signup field (same logic as avg_customer_lifespan)
    signup_field: Optional[str] = None
    for cand in ("original_sign_up", "original_signup", "signup_date"):
        if _field_exists(cust_mappings, cand):
            signup_field = cand
            break

    # Optional pieces field (if present, we use it; otherwise we skip avg_pieces_per_visit)
    pieces_field: Optional[str] = None
    for cand in ("pieces_lifetime", "pieces_pickup_lifetime", "total_pieces_lifetime", "total_pieces"):
        if _field_exists(cust_mappings, cand):
            pieces_field = cand
            break

    # ✅ Important: route is nested; don't require "route" as a top-level field existence check
    required_fields = ["customer_id", "sales_pickup_lifetime", "visits_lifetime", "last_visit"]
    missing_required = [f for f in required_fields if not _field_exists(cust_mappings, f)]
    if missing_required or not signup_field:
        msg_parts = []
        if missing_required:
            msg_parts.append(
                "missing required fields in customers index "
                f"'{customers_index}': {', '.join(missing_required)}"
            )
        if not signup_field:
            msg_parts.append("could not find an original signup field (original_signup/original_sign_up/signup_date)")
        return _es_cannot_answer(
            "Cannot compute 'Route vs Retail Comparison' because " + "; ".join(msg_parts) + ".",
            business_rules,
        )

    # Base filters: paying customers with visits and both dates
    base_filter: List[Dict[str, Any]] = [
        {"exists": {"field": "sales_pickup_lifetime"}},
        {"range": {"sales_pickup_lifetime": {"gt": 0}}},
        {"exists": {"field": "visits_lifetime"}},
        {"range": {"visits_lifetime": {"gt": 0}}},
        {"exists": {"field": signup_field}},
        {"exists": {"field": "last_visit"}},
    ]

    # ------------------------------------------------------------
    # ✅ FIXED SEGMENTATION FOR YOUR NESTED route
    # ------------------------------------------------------------
    ROUTE_PATH = "route"
    ROUTE_NAME_KW = "route.name.keyword"
    ROUTE_ID_FIELD = "route.route_id"

    retail_names = ["Retail", "retail", "Unassigned", "unassigned"]

    # Retail = (route.name in retail_names) OR (route.route_id missing)
    retail_filter: Dict[str, Any] = {
        "bool": {
            "should": [
                {
                    "nested": {
                        "path": ROUTE_PATH,
                        "query": {"terms": {ROUTE_NAME_KW: retail_names}},
                    }
                },
                # route_id is null -> not indexed -> treated as "missing"
                {
                    "bool": {
                        "must_not": [
                            {
                                "nested": {
                                    "path": ROUTE_PATH,
                                    "query": {"exists": {"field": ROUTE_ID_FIELD}},
                                }
                            }
                        ]
                    }
                },
            ],
            "minimum_should_match": 1,
        }
    }

    # Route = route_id exists AND route.name NOT in retail_names
    route_filter: Dict[str, Any] = {
        "nested": {
            "path": ROUTE_PATH,
            "query": {
                "bool": {
                    "filter": [{"exists": {"field": ROUTE_ID_FIELD}}],
                    "must_not": [{"terms": {ROUTE_NAME_KW: retail_names}}],
                }
            },
        }
    }

    # Aggregation body
    tiers_aggs: Dict[str, Any] = {
        "customer_count": {"value_count": {"field": "customer_id"}},
        "segment_revenue": {"sum": {"field": "sales_pickup_lifetime"}},
        "total_visits": {"sum": {"field": "visits_lifetime"}},
        "avg_last_visit": {"avg": {"field": "last_visit"}},
        "avg_signup": {"avg": {"field": signup_field}},
    }
    if pieces_field:
        tiers_aggs["total_pieces"] = {"sum": {"field": pieces_field}}

    body: Dict[str, Any] = {
        "size": 0,
        "query": {"bool": {"filter": base_filter}},
        "aggs": {
            "segments": {
                "filters": {"filters": {"retail": retail_filter, "route": route_filter}},
                "aggs": tiers_aggs,
            },
            "total_revenue": {"sum": {"field": "sales_pickup_lifetime"}},
        },
    }

    res = _safe_es_search(client, index=customers_index, body=body)
    aggs = res.get("aggregations") or {}

    seg_buckets = (aggs.get("segments") or {}).get("buckets") or {}
    total_revenue = float((aggs.get("total_revenue") or {}).get("value") or 0.0)

    if not seg_buckets:
        return {
            "insight": to_json_safe(
                "Route vs Retail Comparison could not be computed because no customers matched the filters "
                "(paying customers with visits, signup date, and last_visit)."
            ),
            "rows": [],
            "rules_used": business_rules or "",
            "engine": "es",
        }

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

        if cust_count <= 0:
            avg_ltv = 0.0
            visits_per_year = 0.0
        else:
            avg_ltv = seg_revenue / cust_count if seg_revenue > 0 else 0.0
            visits_per_year = 0.0
            if avg_last_ms is not None and avg_signup_ms is not None and total_visits > 0:
                avg_lifespan_days = max(1.0, (avg_last_ms - avg_signup_ms) / ms_per_day)
                avg_visits_per_customer = total_visits / cust_count
                years = max(0.01, avg_lifespan_days / 365.0)
                visits_per_year = avg_visits_per_customer / years

        avg_visit_value = seg_revenue / total_visits if total_visits > 0 else 0.0

        avg_pieces_per_visit = None
        if pieces_field and total_pieces is not None and total_visits > 0:
            avg_pieces_per_visit = total_pieces / total_visits

        revenue_share_pct = (seg_revenue * 100.0 / total_revenue) if total_revenue > 0 else 0.0

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

    rows: List[Dict[str, Any]] = [
        summarize("retail", "Retail"),
        summarize("route", "Route Customers"),
    ]

    if all(r["customer_count"] == 0 for r in rows):
        return {
            "insight": to_json_safe(
                "Route vs Retail Comparison could not be computed because both segments have zero eligible customers "
                "(no paying customers with visits, signup, and last_visit)."
            ),
            "rows": [],
            "rules_used": business_rules or "",
            "engine": "es",
        }

    insight = (
        "Route vs Retail Comparison is computed on the customers index "
        f"'{customers_index}' using nested route segmentation:\n"
        "  - Retail = route missing OR route_id missing OR route.name in {'Retail','Unassigned'}\n"
        "  - Route = route_id exists AND route.name not in {'Retail','Unassigned'}.\n"
        "For each segment we calculate customer count, total lifetime revenue "
        "(sales_pickup_lifetime), revenue share, average lifetime value per customer, "
        "average revenue per visit, and an approximate visits-per-year using visits_lifetime, "
        "signup date, and last_visit."
    )

    return {
        "insight": to_json_safe(insight),
        "rows": to_json_safe(rows),
        "rules_used": business_rules or "",
        "engine": "es",
    }
def _es_churn_rate(
    req,
    client,
    mappings: Dict[str, Any],
    business_rules: Optional[str],
):
    """
    Churn Rate

    avgInterval = Average Visit Interval (days)

    Eligible for Churn:
      - visits_lifetime >= 1
      - original_signup is older than avgInterval days
        (strict: original_signup < now-avgInterval days)

    Churned:
      - Eligible customers where last_visit is older than avgInterval days
        (strict: last_visit < now-avgInterval days)

    Churn Rate = (Churned / Eligible) × 100

    NOTE:
      - Assumes 1 document per customer (customers index uses _id = customer_id),
        so we use value_count(customer_id) instead of cardinality(customer_id).
    """

    customers_index = (getattr(req, "es_customers_index_name", "") or "").strip()
    if not customers_index:
        return _es_cannot_answer(
            "Churn Rate requires a customers index (es_customers_index_name).",
            business_rules,
        )

    # Load mapping
    try:
        cust_mapping_raw = client.indices.get_mapping(index=customers_index)
    except Exception:
        cust_mapping_raw = None

    if cust_mapping_raw is None:
        return _es_cannot_answer(
            f"Could not load mappings for customers index '{customers_index}' when computing Churn Rate.",
            business_rules,
        )

    cust_props = _extract_properties_from_mapping(cust_mapping_raw, customers_index)
    cust_mappings = {"properties": cust_props}

    required_fields = [
        "visits_interval_avg",
        "visits_lifetime",
        "original_signup",
        "last_visit",
        "customer_id",
    ]
    missing = [f for f in required_fields if not _field_exists(cust_mappings, f)]
    if missing:
        return _es_cannot_answer(
            "Cannot compute 'Churn Rate' because required fields are missing from customers index "
            f"'{customers_index}': {', '.join(missing)}.",
            business_rules,
        )

    # 1) Global avgInterval from visits_interval_avg (use only real repeat customers)
    avg_body = {
        "size": 0,
        "query": {
            "bool": {
                "filter": [
                    {"exists": {"field": "visits_interval_avg"}},
                    {"range": {"visits_interval_avg": {"gt": 0}}},
                    {"range": {"visits_lifetime": {"gte": 2}}},
                ]
            }
        },
        "aggs": {"avg_interval": {"avg": {"field": "visits_interval_avg"}}},
    }

    avg_res = _safe_es_search(client, index=customers_index, body=avg_body)
    avg_val = (avg_res.get("aggregations") or {}).get("avg_interval", {}).get("value")

    if avg_val is None or float(avg_val) <= 0:
        return _es_cannot_answer(
            "Cannot compute 'Churn Rate' because Average Visit Interval could not be derived "
            "from visits_interval_avg.",
            business_rules,
        )

    avg_interval_days = float(avg_val)
    avg_d_int = max(1, int(round(avg_interval_days)))

    # ES date-math cutoff (strict older-than)
    cutoff_expr = f"now-{avg_d_int}d/d"

    # 2) Eligible + Churned aggregations
    base_filters = [
        {"exists": {"field": "customer_id"}},
        {"exists": {"field": "visits_lifetime"}},
        {"range": {"visits_lifetime": {"gte": 1}}},
        {"exists": {"field": "original_signup"}},
        {"exists": {"field": "last_visit"}},
    ]

    body = {
        "size": 0,
        "query": {"bool": {"filter": base_filters}},
        "aggs": {
            "eligible": {
                "filter": {"range": {"original_signup": {"lt": cutoff_expr}}},
                "aggs": {
                    # ✅ switched to value_count
                    "customers": {"value_count": {"field": "customer_id"}}
                },
            },
            "churned": {
                "filter": {
                    "bool": {
                        "filter": [
                            {"range": {"original_signup": {"lt": cutoff_expr}}},
                            {"range": {"last_visit": {"lt": cutoff_expr}}},
                        ]
                    }
                },
                "aggs": {
                    # ✅ switched to value_count
                    "customers": {"value_count": {"field": "customer_id"}}
                },
            },
        },
    }

    res = _safe_es_search(client, index=customers_index, body=body)
    aggs = res.get("aggregations") or {}

    eligible_count = int(((aggs.get("eligible") or {}).get("customers") or {}).get("value") or 0)
    churned_count = int(((aggs.get("churned") or {}).get("customers") or {}).get("value") or 0)

    if eligible_count == 0:
        return {
            "insight": to_json_safe(
                "Churn Rate could not be computed because no customers were eligible "
                f"(signup older than ~{avg_d_int} days with at least one visit)."
            ),
            "rows": [],
            "rules_used": business_rules or "",
            "engine": "es",
        }

    churn_rate = (churned_count * 100.0) / float(eligible_count)

    rows = [
        {
            "metric": "churn_rate_pct",
            "label": "Churn Rate (%)",
            "value": churn_rate,
            "eligible_customers": eligible_count,
            "churned_customers": churned_count,
            "avg_interval_days": avg_interval_days,
            "avg_interval_days_rounded": avg_d_int,
        }
    ]

    insight = (
        "Churn Rate uses the global Average Visit Interval (from visits_interval_avg) as the threshold. "
        f"Eligible customers are those with >=1 visit and an original_signup older than ~{avg_d_int} days; "
        "churned customers are eligible customers whose last_visit is also older than that interval. "
        f"On customers index '{customers_index}', this yields ~{churn_rate:.1f}% churn "
        f"({churned_count} churned out of {eligible_count} eligible)."
    )

    return {
        "insight": to_json_safe(insight),
        "rows": to_json_safe(rows),
        "rules_used": business_rules or "",
        "engine": "es",
    }
def _es_days_since_last_visit_distribution(
    req,
    client,
    mappings: Dict[str, Any],
    business_rules: Optional[str],
):
    """
    Days Since Last Visit Distribution (customers index).

    Buckets (non-overlapping):
      - 0–30 days
      - 31–60 days
      - 61–90 days
      - 91–180 days
      - 181–365 days
      - 365+ days

    Optimized version:
      - Uses value_count(customer_id) instead of cardinality(customer_id)
      - Assumes 1 doc per customer (e.g., _id == customer_id)
    """

    customers_index = (getattr(req, "es_customers_index_name", "") or "").strip()
    if not customers_index:
        return _es_cannot_answer(
            "Days Since Last Visit Distribution requires a customers index (es_customers_index_name).",
            business_rules,
        )

    # mapping
    try:
        cust_mapping_raw = client.indices.get_mapping(index=customers_index)
    except Exception:
        cust_mapping_raw = None

    if cust_mapping_raw is None:
        return _es_cannot_answer(
            f"Could not load mappings for customers index '{customers_index}' when computing "
            "Days Since Last Visit Distribution.",
            business_rules,
        )

    cust_props = _extract_properties_from_mapping(cust_mapping_raw, customers_index)
    cust_mappings = {"properties": cust_props}

    required_fields = ["last_visit", "customer_id"]
    missing = [f for f in required_fields if not _field_exists(cust_mappings, f)]
    if missing:
        return _es_cannot_answer(
            "Cannot compute 'Days Since Last Visit Distribution' because required fields are missing "
            f"from customers index '{customers_index}': {', '.join(missing)}.",
            business_rules,
        )

    # Use ES date math to avoid timezone/off-by-one issues.
    # Day-rounded boundaries (/d) + "now+1d/d" to include today.
    bucket_filters = {
        "0_30": {
            "range": {"last_visit": {"gte": "now-30d/d", "lt": "now+1d/d"}}
        },
        "31_60": {
            "range": {"last_visit": {"gte": "now-60d/d", "lt": "now-30d/d"}}
        },
        "61_90": {
            "range": {"last_visit": {"gte": "now-90d/d", "lt": "now-60d/d"}}
        },
        "91_180": {
            "range": {"last_visit": {"gte": "now-180d/d", "lt": "now-90d/d"}}
        },
        "181_365": {
            "range": {"last_visit": {"gte": "now-365d/d", "lt": "now-180d/d"}}
        },
        "365_plus": {
            "range": {"last_visit": {"lt": "now-365d/d"}}
        },
    }

    body = {
        "size": 0,
        "query": {
            "bool": {
                "filter": [
                    {"exists": {"field": "last_visit"}},
                    {"exists": {"field": "customer_id"}},
                ]
            }
        },
        "aggs": {
            # total customers with last_visit (fast exact under 1-doc-per-customer)
            "total_customers": {"value_count": {"field": "customer_id"}},
            "days_since_last": {
                "filters": {"filters": bucket_filters},
                "aggs": {
                    # count customers inside each bucket (fast exact under 1-doc-per-customer)
                    "customers": {"value_count": {"field": "customer_id"}}
                },
            },
        },
    }

    res = _safe_es_search(client, index=customers_index, body=body)
    aggs = res.get("aggregations") or {}

    total_customers = int((aggs.get("total_customers") or {}).get("value") or 0)
    if total_customers == 0:
        return {
            "insight": to_json_safe(
                "Days Since Last Visit Distribution could not be computed because no customers with last_visit were found."
            ),
            "rows": [],
            "rules_used": business_rules or "",
            "engine": "es",
        }

    buckets = ((aggs.get("days_since_last") or {}).get("buckets")) or {}

    bucket_meta = [
        ("0_30", "0–30 Days", 0, 30, "active"),
        ("31_60", "31–60 Days", 31, 60, "active"),
        ("61_90", "61–90 Days", 61, 90, "normal"),
        ("91_180", "91–180 Days", 91, 180, "at_risk"),
        ("181_365", "181–365 Days", 181, 365, "lapsed"),
        ("365_plus", "365+ Days", 366, None, "lost"),
    ]

    rows: List[Dict[str, Any]] = []
    for key, label, min_days, max_days, risk in bucket_meta:
        b = buckets.get(key) or {}
        count = int((b.get("customers") or {}).get("value") or 0)
        pct = (count * 100.0 / float(total_customers)) if total_customers > 0 else 0.0
        rows.append(
            {
                "bucket_id": key,
                "label": label,
                "min_days_since_last_visit": min_days,
                "max_days_since_last_visit": max_days,
                "customer_count": count,
                "percentage_of_customers": pct,
                "risk_level": risk,  # active / normal / at_risk / lapsed / lost
            }
        )

    insight = (
        "Days Since Last Visit Distribution groups customers by how long it has been since their last_visit "
        "using 0–30, 31–60, 61–90, 91–180, 181–365 and 365+ day buckets, and counts customers per bucket. "
        f"Computed on customers index '{customers_index}' for customers with a known last_visit. "
        "This optimized version uses value_count(customer_id) and assumes one document per customer."
    )

    return {
        "insight": to_json_safe(insight),
        "rows": to_json_safe(rows),
        "rules_used": business_rules or "",
        "engine": "es",
    }
def _es_daily_acquisition_rate_by_period_customers(
    req,
    client,
    mappings: Dict[str, Any],
    business_rules: Optional[str],
):
    """
    Daily Acquisition Rate by 30-day periods over the last 180 days (customers index).

    Buckets (non-overlapping, most recent first):
      - 0–30 days:     first_visit >= now-30d AND < now
      - 30–60 days:    first_visit >= now-60d AND < now-30d
      - 60–90 days:    first_visit >= now-90d AND < now-60d
      - 90–120 days:   first_visit >= now-120d AND < now-90d
      - 120–150 days:  first_visit >= now-150d AND < now-120d
      - 150–180 days:  first_visit >= now-180d AND < now-150d

    For each bucket:
      - Count = customers whose first_visit is inside that window
      - Daily Rate % = (count / 30) / total_customers × 100

    Denominator:
      - total_customers = customers that have first_visit (exists)

    OPTIMIZATION:
      - Uses value_count(customer_id) instead of cardinality(customer_id)
        Assumes 1 doc per customer in customers index.
    """

    customers_index = (getattr(req, "es_customers_index_name", "") or "").strip()
    if not customers_index:
        return _es_cannot_answer(
            "Daily Acquisition Rate by Period requires a customers index (es_customers_index_name).",
            business_rules,
        )

    # Load customers mappings
    try:
        cust_mapping_raw = client.indices.get_mapping(index=customers_index)
        cust_props = _extract_properties_from_mapping(cust_mapping_raw, customers_index)
        cust_mappings = {"properties": cust_props}
    except Exception:
        return _es_cannot_answer(
            f"Could not load mappings for customers index '{customers_index}' "
            "when computing Daily Acquisition Rate by Period.",
            business_rules,
        )

    required_fields = ["customer_id", "first_visit"]
    missing = [f for f in required_fields if not _field_exists(cust_mappings, f)]
    if missing:
        return _es_cannot_answer(
            "Cannot compute 'Daily Acquisition Rate by Period' because required fields are missing "
            f"from customers index '{customers_index}': {', '.join(missing)}.",
            business_rules,
        )

    bucket_meta = [
        ("0_30", "0–30 Days", "now-30d/d", "now/d"),
        ("30_60", "30–60 Days", "now-60d/d", "now-30d/d"),
        ("60_90", "60–90 Days", "now-90d/d", "now-60d/d"),
        ("90_120", "90–120 Days", "now-120d/d", "now-90d/d"),
        ("120_150", "120–150 Days", "now-150d/d", "now-120d/d"),
        ("150_180", "150–180 Days", "now-180d/d", "now-150d/d"),
    ]

    # Build filters aggregation
    filters_obj: Dict[str, Any] = {}
    for key, _label, gte_expr, lt_expr in bucket_meta:
        filters_obj[key] = {
            "range": {
                "first_visit": {
                    "gte": gte_expr,
                    "lt": lt_expr,
                }
            }
        }

    body = {
        "size": 0,
        "query": {
            "bool": {
                "filter": [
                    {"exists": {"field": "first_visit"}},
                    {"exists": {"field": "customer_id"}},
                ]
            }
        },
        "aggs": {
            # ✅ OPTIMIZED: exact + faster when 1 doc per customer
            "total_customers": {"value_count": {"field": "customer_id"}},
            "periods": {
                "filters": {"filters": filters_obj},
                "aggs": {
                    # ✅ OPTIMIZED: exact + faster when 1 doc per customer
                    "customers": {"value_count": {"field": "customer_id"}}
                },
            },
        },
    }

    res = _safe_es_search(client, index=customers_index, body=body)
    aggs = res.get("aggregations") or {}

    total_customers = int((aggs.get("total_customers") or {}).get("value") or 0)
    if total_customers == 0:
        return {
            "insight": to_json_safe(
                "Daily Acquisition Rate by Period could not be computed because no customers with a first_visit were found."
            ),
            "rows": [],
            "rules_used": business_rules or "",
            "engine": "es",
        }

    periods = (aggs.get("periods") or {}).get("buckets") or {}

    rows: List[Dict[str, Any]] = []
    for key, label, _gte, _lt in bucket_meta:
        b = periods.get(key) or {}
        count = int((b.get("customers") or {}).get("value") or 0)
        daily_rate_pct = ((count / 30.0) / float(total_customers)) * 100.0 if count > 0 else 0.0

        rows.append(
            {
                "period_id": key,
                "label": label,
                "customer_count": count,
                "daily_rate_pct": daily_rate_pct,
            }
        )

    insight = (
        "Daily Acquisition Rate by Period groups customers by when their first_visit happened in the last 180 days, "
        "using non-overlapping 30-day windows. For each window, it computes a normalized daily rate: "
        "(count / 30) / total_customers × 100."
    )

    return {
        "insight": to_json_safe(insight),
        "rows": to_json_safe(rows),
        "rules_used": business_rules or "",
        "engine": "es",
    }
def _es_yoy_new_customers_customers_index(
    req,
    client,
    mappings: Dict[str, Any],
    business_rules: Optional[str],
):
    """
    Year-over-Year New Customers (from customers.first_visit).

    For each year:
      - New Customers = count of customers with first_visit in that year
      - YoY Change % = ((current_year - previous_year) / previous_year) × 100
        where previous_year is the previous CALENDAR year (y-1), even if not displayed.
      - Display filter: only show years with >= 50 new customers

    Notes:
      - Based on first_visit date, not signup date.
      - Latest year may be partial depending on data coverage.

    OPTIMIZATION:
      - Uses value_count(customer_id) instead of cardinality(customer_id)
        Assumes 1 doc per customer in customers index.
    """

    customers_index = (getattr(req, "es_customers_index_name", "") or "").strip()
    if not customers_index:
        return _es_cannot_answer(
            "Year-over-Year New Customers requires a customers index (es_customers_index_name).",
            business_rules,
        )

    # Fetch customers mappings
    try:
        cust_mapping_raw = client.indices.get_mapping(index=customers_index)
        cust_props = _extract_properties_from_mapping(cust_mapping_raw, customers_index)
        cust_mappings = {"properties": cust_props}
    except Exception:
        return _es_cannot_answer(
            f"Could not load mappings for customers index '{customers_index}' "
            "when computing Year-over-Year New Customers.",
            business_rules,
        )

    required_fields = ["first_visit", "customer_id"]
    missing = [f for f in required_fields if not _field_exists(cust_mappings, f)]
    if missing:
        return _es_cannot_answer(
            "Cannot compute 'Year-over-Year New Customers' because required fields are missing "
            f"from customers index '{customers_index}': {', '.join(missing)}.",
            business_rules,
        )

    body = {
        "size": 0,
        "query": {
            "bool": {
                "filter": [
                    {"exists": {"field": "first_visit"}},
                    {"exists": {"field": "customer_id"}},
                ]
            }
        },
        "aggs": {
            "by_year": {
                "date_histogram": {
                    "field": "first_visit",
                    "calendar_interval": "year",
                    "min_doc_count": 1,
                },
                "aggs": {
                    # ✅ OPTIMIZED: exact + faster when 1 doc per customer
                    "new_customers": {"value_count": {"field": "customer_id"}}
                },
            }
        },
    }

    res = _safe_es_search(client, index=customers_index, body=body)
    buckets = (((res.get("aggregations") or {}).get("by_year") or {}).get("buckets")) or []

    if not buckets:
        return {
            "insight": to_json_safe(
                "Year-over-Year New Customers could not be computed because no customers had a first_visit date."
            ),
            "rows": [],
            "rules_used": business_rules or "",
            "engine": "es",
        }

    # Build year -> count
    per_year: Dict[int, int] = {}
    for b in buckets:
        key_as_string = b.get("key_as_string") or ""
        year = None

        if len(key_as_string) >= 4 and key_as_string[:4].isdigit():
            year = int(key_as_string[:4])
        else:
            try:
                year = datetime.fromtimestamp((b.get("key") or 0) / 1000, tz=timezone.utc).year
            except Exception:
                continue

        cnt = int(((b.get("new_customers") or {}).get("value")) or 0)
        per_year[year] = cnt

    if not per_year:
        return {
            "insight": to_json_safe(
                "Year-over-Year New Customers could not be computed because yearly buckets were empty."
            ),
            "rows": [],
            "rules_used": business_rules or "",
            "engine": "es",
        }

    # Display only years with >= 50 customers
    years_display = sorted([y for y, c in per_year.items() if c >= 50])
    if not years_display:
        return {
            "insight": to_json_safe(
                "Year-over-Year New Customers could not be computed because no year had at least 50 new customers."
            ),
            "rows": [],
            "rules_used": business_rules or "",
            "engine": "es",
        }

    rows: List[Dict[str, Any]] = []
    for y in years_display:
        count = int(per_year.get(y, 0))
        prev = per_year.get(y - 1)  # previous CALENDAR year
        yoy = None if (prev is None or prev == 0) else ((count - prev) * 100.0 / float(prev))

        rows.append(
            {
                "year": int(y),
                "new_customers": count,
                "yoy_change_pct": yoy,
            }
        )

    insight = (
        "Year-over-Year New Customers counts customers by the year of their first_visit (customers index). "
        "Only years with at least 50 new customers are displayed. YoY change is computed versus the previous "
        "calendar year (year-1), even if that prior year is not displayed."
    )

    return {
        "insight": to_json_safe(insight),
        "rows": to_json_safe(rows),
        "rules_used": business_rules or "",
        "engine": "es",
    }
def _es_return_rate_by_cohort_year_customers(
    req,
    client,
    mappings: Dict[str, Any],
    business_rules: Optional[str],
):
    customers_index = (getattr(req, "es_customers_index_name", "") or "").strip()
    if not customers_index:
        return _es_cannot_answer(
            "Return Rate by Cohort Year requires a customers index (es_customers_index_name).",
            business_rules,
        )

    # ---- mappings (robust) ----
    try:
        cust_mapping_raw = client.indices.get_mapping(index=customers_index)
    except Exception:
        cust_mapping_raw = None

    if cust_mapping_raw is None:
        return _es_cannot_answer(
            f"Could not load mappings for customers index '{customers_index}'.",
            business_rules,
        )

    cust_props = _extract_properties_from_mapping(cust_mapping_raw, customers_index)
    cust_mappings = {"properties": cust_props}

    required_fields = ["customer_id", "first_visit", "visits_lifetime", "visits_interval_avg"]
    missing = [f for f in required_fields if not _field_exists(cust_mappings, f)]
    if missing:
        return _es_cannot_answer(
            "Cannot compute 'Return Rate by Cohort Year' because required fields are missing "
            f"from customers index '{customers_index}': {', '.join(missing)}.",
            business_rules,
        )

    # ---- 1) global avgIntervalDays (repeat customers only) ----
    avg_body = {
        "size": 0,
        "query": {
            "bool": {
                "filter": [
                    {"exists": {"field": "visits_interval_avg"}},
                    {"range": {"visits_interval_avg": {"gt": 0}}},
                    {"range": {"visits_lifetime": {"gte": 2}}},
                ]
            }
        },
        "aggs": {"avg_interval_days": {"avg": {"field": "visits_interval_avg"}}},
    }

    avg_res = _safe_es_search(client, index=customers_index, body=avg_body)
    avg_val = ((avg_res.get("aggregations") or {}).get("avg_interval_days") or {}).get("value")

    if avg_val is None:
        return _es_cannot_answer(
            "Cannot compute 'Return Rate by Cohort Year' because Average Visit Interval could not be derived.",
            business_rules,
        )

    try:
        avg_interval_days = float(avg_val)
    except Exception:
        avg_interval_days = 0.0

    if avg_interval_days <= 0:
        return _es_cannot_answer(
            "Cannot compute 'Return Rate by Cohort Year' because Average Visit Interval is <= 0.",
            business_rules,
        )

    avg_d_int = max(1, int(round(avg_interval_days)))
    cutoff_expr = f"now-{avg_d_int}d/d"  # ES date math

    # ---- 2) cohort year + eligible + returned ----
    cohort_body = {
        "size": 0,
        "query": {"bool": {"filter": [{"exists": {"field": "first_visit"}}]}},
        "aggs": {
            "by_year": {
                "date_histogram": {
                    "field": "first_visit",
                    "calendar_interval": "year",
                    "min_doc_count": 1,
                },
                "aggs": {
                    # use value_count since you said 1 doc per customer
                    "original_customers": {"value_count": {"field": "customer_id"}},
                    "eligible": {
                        "filter": {"range": {"first_visit": {"lt": cutoff_expr}}},
                        "aggs": {
                            "eligible_customers": {"value_count": {"field": "customer_id"}},
                            "returned": {
                                "filter": {"range": {"visits_lifetime": {"gte": 2}}},
                                "aggs": {
                                    "returned_customers": {"value_count": {"field": "customer_id"}}
                                },
                            },
                        },
                    },
                },
            }
        },
    }

    resp = _safe_es_search(client, index=customers_index, body=cohort_body)
    buckets = (((resp.get("aggregations") or {}).get("by_year") or {}).get("buckets")) or []

    if not buckets:
        return {
            "insight": to_json_safe(
                "Return Rate by Cohort Year could not be computed because no customers had a first_visit date."
            ),
            "rows": [],
            "rules_used": business_rules or "",
            "engine": "es",
        }

    raw_rows: List[Dict[str, Any]] = []
    for b in buckets:
        key_as_string = b.get("key_as_string") or ""
        try:
            year = int(key_as_string[:4])
        except Exception:
            try:
                year = datetime.fromtimestamp((b.get("key") or 0) / 1000, tz=timezone.utc).year
            except Exception:
                continue

        original = int(((b.get("original_customers") or {}).get("value")) or 0)
        eligible_bucket = (b.get("eligible") or {})
        eligible = int(((eligible_bucket.get("eligible_customers") or {}).get("value")) or 0)
        returned_bucket = (eligible_bucket.get("returned") or {})
        returned = int(((returned_bucket.get("returned_customers") or {}).get("value")) or 0)

        raw_rows.append(
            {
                "year": year,
                "original_cohort_size": original,
                "eligible_customers": eligible,
                "returned_customers": returned,
            }
        )

    if not raw_rows:
        return {
            "insight": to_json_safe("Return Rate by Cohort Year could not be computed because cohort buckets were empty."),
            "rows": [],
            "rules_used": business_rules or "",
            "engine": "es",
        }

    # ---- 3) year filtering ----
    max_year_count = max(r["original_cohort_size"] for r in raw_rows)
    threshold = max_year_count * 0.50

    filtered = [
        r for r in sorted(raw_rows, key=lambda x: x["year"])
        if r["original_cohort_size"] >= threshold and r["eligible_customers"] >= 50
    ]

    if not filtered:
        return {
            "insight": to_json_safe(
                "Return Rate by Cohort Year could not be computed because no year met the volume threshold "
                "(>=50% of peak cohort size and at least 50 eligible customers)."
            ),
            "rows": [],
            "rules_used": business_rules or "",
            "engine": "es",
        }

    rows: List[Dict[str, Any]] = []
    for r in filtered:
        eligible = r["eligible_customers"]
        returned = r["returned_customers"]
        rate = (returned * 100.0 / float(eligible)) if eligible > 0 else 0.0
        rows.append(
            {
                "year": int(r["year"]),
                "eligible_customers": int(eligible),
                "returned_customers": int(returned),
                "return_rate_pct": rate,
                "original_cohort_size": int(r["original_cohort_size"]),
                "avg_interval_days_rounded": int(avg_d_int),
            }
        )

    insight = (
        f"Return Rate by Cohort Year groups customers by first_visit year and counts only customers whose "
        f"first_visit is older than the global average visit interval (~{avg_d_int} days). "
        "For each included year, it reports the share of eligible customers who reached at least 2 lifetime visits."
    )

    return {
        "insight": to_json_safe(insight),
        "rows": to_json_safe(rows),
        "rules_used": business_rules or "",
        "engine": "es",
    }

def _es_repeat_customers_365(
    req,
    client,
    mappings: Dict[str, Any],
    business_rules: Optional[str],
):
    """
    Repeat Customers (365 Days) — customers index

    Definition:
      - Base population: customers with visits_365 > 0
      - Optional exclusion (same as other 365 metrics):
          exclude customers where original_signup is within last N days (default 180)
          AND visits_365 = 1
      - Repeat customers: visits_365 >= 2
      - Repeat Rate = repeat / base * 100

    OPTIMIZATION:
      - Uses value_count(customer_id) instead of cardinality(customer_id)
      - Assumes 1 doc per customer in customers index (_id == customer_id)
        If that assumption might be false, switch back to cardinality.
    """

    customers_index = (getattr(req, "es_customers_index_name", "") or "").strip()
    if not customers_index:
        return _es_cannot_answer(
            "Repeat Customers 365 requires a customers index (es_customers_index_name).",
            business_rules,
        )

    # Load customers mapping
    try:
        cust_mapping_raw = client.indices.get_mapping(index=customers_index)
    except Exception:
        cust_mapping_raw = None

    if cust_mapping_raw is None:
        return _es_cannot_answer(
            f"Could not load mappings for customers index '{customers_index}' "
            "when computing Repeat Customers 365.",
            business_rules,
        )

    cust_props = _extract_properties_from_mapping(cust_mapping_raw, customers_index)
    cust_mappings = {"properties": cust_props}

    # Required fields
    required_fields = ["customer_id", "visits_365"]
    missing = [f for f in required_fields if not _field_exists(cust_mappings, f)]
    if missing:
        return _es_cannot_answer(
            "Cannot compute 'Repeat Customers 365' because required fields are missing "
            f"from customers index '{customers_index}': {', '.join(missing)}.",
            business_rules,
        )

    # Optional exclusion needs original_signup
    has_signup = _field_exists(cust_mappings, "original_signup")

    min_signup_age_days = _get_req_int(
        req,
        "min_signup_age_days_for_repeat_365",
        180,
        min_v=1,
        max_v=10_000,
    )

    today = datetime.now(timezone.utc).date()
    cutoff_signup = today - timedelta(days=int(min_signup_age_days))
    cutoff_signup_str = cutoff_signup.isoformat()

    # Base population: customers with visits_365 > 0
    base_filters: List[Dict[str, Any]] = [
        {"exists": {"field": "customer_id"}},
        {"exists": {"field": "visits_365"}},
        {"range": {"visits_365": {"gt": 0}}},
    ]

    # Exclude very new one-visit customers (optional, if original_signup exists)
    must_not: List[Dict[str, Any]] = []
    if has_signup:
        must_not.append(
            {
                "bool": {
                    "filter": [
                        {"range": {"original_signup": {"gte": cutoff_signup_str}}},
                        {"term": {"visits_365": 1}},
                    ]
                }
            }
        )

    body: Dict[str, Any] = {
        "size": 0,
        "query": {"bool": {"filter": base_filters, "must_not": must_not}},
        "aggs": {
            # ✅ CHANGED: exact + cheaper if 1 doc per customer
            "base_customers": {"value_count": {"field": "customer_id"}},

            "repeat_customers": {
                "filter": {"range": {"visits_365": {"gte": 2}}},
                "aggs": {
                    # ✅ CHANGED: exact + cheaper if 1 doc per customer
                    "customers": {"value_count": {"field": "customer_id"}}
                },
            },
        },
    }

    res = _safe_es_search(client, index=customers_index, body=body)
    aggs = res.get("aggregations") or {}

    base_n = int((aggs.get("base_customers") or {}).get("value") or 0)
    repeat_bucket = aggs.get("repeat_customers") or {}
    repeat_n = int(((repeat_bucket.get("customers") or {}).get("value") or 0))

    if base_n == 0:
        return {
            "insight": to_json_safe(
                "Repeat Customers 365 could not be computed because no customers matched the base population "
                "(visits_365 > 0, after exclusions)."
            ),
            "rows": [],
            "rules_used": business_rules or "",
            "engine": "es",
        }

    repeat_pct = (repeat_n * 100.0) / float(base_n)

    insight = (
        "Repeat Customers (365 Days) measures the share of customers who visited at least twice in the last 365 days "
        f"(visits_365 ≥ 2). Base population is customers with visits_365 > 0"
    )
    if has_signup:
        insight += (
            f", excluding very new single-visit customers (original_signup within last {min_signup_age_days} days "
            "AND visits_365 = 1)."
        )
    else:
        insight += "; original_signup was not available so the 'new single-visit' exclusion was not applied."

    insight += f" Result: {repeat_n} repeat customers out of {base_n} = {repeat_pct:.1f}%."

    rows = [
        {"metric": "repeat_customers_365", "label": "Repeat Customers (365d)", "value": repeat_n},
        {"metric": "base_customers_365", "label": "Customers with visits_365 > 0 (filtered base)", "value": base_n},
        {"metric": "repeat_customers_365_pct", "label": "Repeat Customers 365 (%)", "value": repeat_pct},
        {
            "metric": "repeat_customers_365_params",
            "label": "Params",
            "value": {
                "min_signup_age_days_for_repeat_365": int(min_signup_age_days),
                "applied_new_single_visit_exclusion": bool(has_signup),
                "customers_index": customers_index,
            },
        },
    ]

    return {
        "insight": to_json_safe(insight),
        "rows": to_json_safe(rows),
        "rules_used": business_rules or "",
        "engine": "es",
    }

def _es_high_value_retail_targets(
    req,
    client,
    mappings: Dict[str, Any],
    business_rules: Optional[str],
):
    """
    High-Value Retail Targets (Route Conversion Opportunity).

    Retail customer:
      - route.name == "Retail"  (route is nested)
      - OR route missing/empty (no nested route docs)

    High-Value Retail:
      - Retail customers with visit_average_sales >= 75

    Metrics:
      - high_value_retail_count
      - total_retail_customers
      - high_value_retail_pct
    """

    customers_index = (getattr(req, "es_customers_index_name", "") or "").strip()
    if not customers_index:
        return _es_cannot_answer(
            "High-Value Retail Targets requires a customers index (es_customers_index_name).",
            business_rules,
        )

    # Load customers mappings
    try:
        cust_mapping_raw = client.indices.get_mapping(index=customers_index)
        cust_props = _extract_properties_from_mapping(cust_mapping_raw, customers_index)
        cust_mappings = {"properties": cust_props}
    except Exception:
        return _es_cannot_answer(
            f"Could not load mappings for customers index '{customers_index}' "
            "when computing High-Value Retail Targets.",
            business_rules,
        )

    # We need: customer_id, visit_average_sales, and route.name(.keyword) nested
    if not _field_exists(cust_mappings, "customer_id"):
        return _es_cannot_answer(
            f"Cannot compute High-Value Retail Targets: missing customer_id in '{customers_index}'.",
            business_rules,
        )
    if not _field_exists(cust_mappings, "visit_average_sales"):
        return _es_cannot_answer(
            f"Cannot compute High-Value Retail Targets: missing visit_average_sales in '{customers_index}'.",
            business_rules,
        )

    # Prefer route.name.keyword if present, otherwise fall back to route.name
    route_name_field = "route.name.keyword" if _field_exists(cust_mappings, "route.name.keyword") else "route.name"
    if not _field_exists(cust_mappings, "route.name"):
        return _es_cannot_answer(
            f"Cannot compute High-Value Retail Targets: missing route.name (nested) in '{customers_index}'.",
            business_rules,
        )

    # --- Retail filter ---
    # A) route.name == "Retail" (nested)
    if route_name_field.endswith(".keyword"):
        retail_name_query = {"term": {route_name_field: "Retail"}}
    else:
        # fallback if only text exists (less ideal than keyword, but works)
        retail_name_query = {"match_phrase": {"route.name": "Retail"}}

    retail_nested = {
        "nested": {
            "path": "route",
            "query": retail_name_query,
        }
    }

    # B) route missing/empty => no nested route docs exist
    no_route = {
        "bool": {
            "must_not": [
                {
                    "nested": {
                        "path": "route",
                        "query": {"match_all": {}},
                    }
                }
            ]
        }
    }

    retail_filter = {
        "bool": {
            "should": [retail_nested, no_route],
            "minimum_should_match": 1,
        }
    }

    # Base query: retail customers (optionally require visit_average_sales exists)
    base_filters = [
        {"exists": {"field": "customer_id"}},
        {"exists": {"field": "visit_average_sales"}},
        retail_filter,
    ]

    body = {
        "size": 0,
        "query": {"bool": {"filter": base_filters}},
        "aggs": {
            # If 1 doc per customer => value_count is exact + faster
            "total_retail_customers": {"value_count": {"field": "customer_id"}},

            "high_value_retail": {
                "filter": {"range": {"visit_average_sales": {"gte": 75}}},
                "aggs": {
                    "customers": {"value_count": {"field": "customer_id"}}
                },
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
            "insight": to_json_safe(
                "High-Value Retail Targets could not be computed because no retail customers were found "
                "(route.name='Retail' or missing/empty route)."
            ),
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

    insight = (
        "High-Value Retail Targets counts retail customers (route.name='Retail' or no route) whose "
        "visit_average_sales is at least 75. These are strong candidates for route conversion. "
        f"On customers index '{customers_index}', there are {high_value_count} such customers "
        f"({pct:.1f}% of retail customers)."
    )

    return {
        "insight": to_json_safe(insight),
        "rows": to_json_safe(rows),
        "rules_used": business_rules or "",
        "engine": "es",
    }


__all__ = [
    "_es_avg_days_between_visits_active",
    "_es_lapsed_customers",
    "_es_overdue_customers",
    "_es_visit_frequency_distribution",
    "_es_customers_nth_visit",
    "_es_new_customer_acquisition_from_customers",
    "_es_new_customer_30d_return_rate",
    "_es_top_customers_by_revenue",
    "_es_active_customers",
    "_es_customer_retention_rate_730_180",
    "_es_avg_customer_lifespan",
    "_es_active_customer_rate",
    "_es_30d_activity_rate",
    "_es_avg_visit_interval",
    "_es_repeat_customers_365",
    "_es_pareto_80_20",
    "_es_single_visit_lifetime",
    "_es_single_visit_365",
    "_es_top_5pct_revenue_from_tags",
    "_es_visit_frequency_365",
    "_es_visit_frequency_730",
    "_es_route_vs_retail_comparison",
    "_es_churn_rate",
    "_es_days_since_last_visit_distribution",
    "_es_daily_acquisition_rate_by_period_customers",
    "_es_yoy_new_customers_customers_index",
    "_es_return_rate_by_cohort_year_customers",
    "_es_high_value_retail_targets",
]
