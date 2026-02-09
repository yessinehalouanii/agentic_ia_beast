from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from abi.runtime import to_json_safe
from app.api.metrics.metrics_promos_coupons import _date_filters_or_default
from app.api.metrics.shared_utilities import (
    _field_exists,
    _safe_es_search,
    _load_customers_ctx,  # ✅ NEW
)
from app.api.docs_analytics_routes import _es_cannot_answer

def _missing_fields(mappings: Dict[str, Any], fields: List[str]) -> List[str]:
    return [f for f in fields if not _field_exists(mappings, f)]


# -------------------------------------------------------------------
# ✅ Acquisition metrics (customers index)
# -------------------------------------------------------------------

def _es_new_customer_acquisition(
    req,
    client,
    mappings: Dict[str, Any],  # kept for signature compatibility (unused)
    business_rules: Optional[str],
):
    """
    New Customer Acquisition:
      - Count customers by signup date (original_signup else created_at)
      - Group by month, or quarter if question mentions quarter/Q1..Q4
    """
    customers_index, cust_mappings, err = _load_customers_ctx(req, client, business_rules)
    if err:
        return err

    # pick the best date field available
    date_field: Optional[str] = "original_signup" if _field_exists(cust_mappings, "original_signup") else None
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

    counts: Dict[str, int] = {}
    for b in buckets:
        key_as_string = b.get("key_as_string")
        doc_count = int(b.get("doc_count") or 0)
        if not key_as_string:
            continue

        ym = key_as_string[:7]  # YYYY-MM

        if use_quarter:
            y = int(ym[:4])
            m = int(ym[5:7])
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


def _es_daily_acquisition_rate_by_period_customers(
    req,
    client,
    mappings: Dict[str, Any],  # kept for signature compatibility (unused)
    business_rules: Optional[str],
):
    """
    Daily Acquisition Rate by 30-day windows over last 180 days (customers index).

    For each window:
      daily_rate_pct = (count / 30) / total_customers * 100
    """
    customers_index, cust_mappings, err = _load_customers_ctx(req, client, business_rules)
    if err:
        return err

    required_fields = ["customer_id", "first_visit"]
    missing = _missing_fields(cust_mappings, required_fields)
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

    filters_obj: Dict[str, Any] = {
        key: {"range": {"first_visit": {"gte": gte_expr, "lt": lt_expr}}}
        for key, _label, gte_expr, lt_expr in bucket_meta
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
            "total_customers": {"value_count": {"field": "customer_id"}},
            "periods": {
                "filters": {"filters": filters_obj},
                "aggs": {"customers": {"value_count": {"field": "customer_id"}}},
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
        rows.append({"period_id": key, "label": label, "customer_count": count, "daily_rate_pct": daily_rate_pct})

    insight = (
        "Daily Acquisition Rate by Period groups customers by when their first_visit happened in the last 180 days, "
        "using non-overlapping 30-day windows. For each window: (count / 30) / total_customers × 100."
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
    mappings: Dict[str, Any],  # kept for signature compatibility (unused)
    business_rules: Optional[str],
):
    """
    Year-over-Year New Customers (customers.first_visit), with display filter >= 50.
    """
    customers_index, cust_mappings, err = _load_customers_ctx(req, client, business_rules)
    if err:
        return err

    required_fields = ["first_visit", "customer_id"]
    missing = _missing_fields(cust_mappings, required_fields)
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
                "aggs": {"new_customers": {"value_count": {"field": "customer_id"}}},
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

    per_year: Dict[int, int] = {}
    for b in buckets:
        key_as_string = b.get("key_as_string") or ""
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
            "insight": to_json_safe("Year-over-Year New Customers could not be computed because yearly buckets were empty."),
            "rows": [],
            "rules_used": business_rules or "",
            "engine": "es",
        }

    years_display = sorted([y for y, c in per_year.items() if c >= 50])
    if not years_display:
        return {
            "insight": to_json_safe("No year had at least 50 new customers, so nothing is displayed."),
            "rows": [],
            "rules_used": business_rules or "",
            "engine": "es",
        }

    rows: List[Dict[str, Any]] = []
    for y in years_display:
        count = int(per_year.get(y, 0))
        prev = per_year.get(y - 1)
        yoy = None if (prev is None or prev == 0) else ((count - prev) * 100.0 / float(prev))
        rows.append({"year": int(y), "new_customers": count, "yoy_change_pct": yoy})

    insight = (
        "Year-over-Year New Customers counts customers by the year of their first_visit (customers index). "
        "Only years with at least 50 new customers are displayed. YoY change is computed versus year-1."
    )

    return {
        "insight": to_json_safe(insight),
        "rows": to_json_safe(rows),
        "rules_used": business_rules or "",
        "engine": "es",
    }


__all__ = [
    "_es_new_customer_acquisition",
    "_es_daily_acquisition_rate_by_period_customers",
    "_es_yoy_new_customers_customers_index",
]
