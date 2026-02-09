from __future__ import annotations

from datetime import datetime, timezone, timedelta
from typing import Any, Dict, List, Optional

from abi.runtime import to_json_safe
from app.api.docs_analytics_routes import (
    _parse_date_str,
    _es_cannot_answer,
)
from app.api.metrics.shared_utilities import (
    _field_exists,
    _safe_es_search,
    _get_req_int,
    _load_customers_ctx,  # ✅ NEW
)

# -------------------------------------------------------------------
# Shared helpers (field guards + common filters)
# -------------------------------------------------------------------


def _require_fields(
    mappings: Dict[str, Any],
    index_name: str,
    fields: List[str],
    metric_label: str,
    business_rules: Optional[str],
):
    missing = [f for f in fields if not _field_exists(mappings, f)]
    if missing:
        return _es_cannot_answer(
            f"Cannot compute '{metric_label}' because fields are missing from '{index_name}': {', '.join(missing)}.",
            business_rules,
        )
    return None


def _today_utc_date():
    return datetime.now(timezone.utc).date()


def _cutoff_iso(days: int) -> str:
    return (_today_utc_date() - timedelta(days=int(days))).isoformat()


def _base_customer_filters(cust_mappings: Dict[str, Any]) -> tuple[List[Dict[str, Any]], bool]:
    """
    Base population = customers with customer_id.
    If sales_pickup_lifetime exists, restrict to paying customers (gt 0).
    Returns (filters, paying_only_applied).
    """
    filters: List[Dict[str, Any]] = [{"exists": {"field": "customer_id"}}]
    paying_only_applied = False

    if _field_exists(cust_mappings, "sales_pickup_lifetime"):
        filters.append({"range": {"sales_pickup_lifetime": {"gt": 0}}})
        paying_only_applied = True

    return filters, paying_only_applied


def _as_of_utc_date(req):
    """
    Anchor date for dashboard comparisons:
    - Prefer req.end_date if parseable
    - Else use today (UTC)
    """
    d = _parse_date_str(getattr(req, "end_date", None))
    return d or datetime.now(timezone.utc).date()


# -------------------------------------------------------------------
# Metrics
# -------------------------------------------------------------------


def _es_active_customers(
    req,
    client,
    mappings: Dict[str, Any],
    business_rules: Optional[str],
    days_threshold: int = 180,
):
    customers_index, cust_mappings, err = _load_customers_ctx(req, client, business_rules)
    if err:
        return err

    end_d = _as_of_utc_date(req)
    cutoff_d = end_d - timedelta(days=int(days_threshold))
    end_str = end_d.isoformat()
    cutoff_str = cutoff_d.isoformat()

    need = ["customer_id", "last_visit", "sales_pickup_lifetime"]
    missing_err = _require_fields(cust_mappings, customers_index, need, "Active Customers", business_rules)
    if missing_err is not None:
        return missing_err

    body = {
        "size": 0,
        "query": {
            "bool": {
                "filter": [
                    {"exists": {"field": "customer_id"}},
                    {"range": {"sales_pickup_lifetime": {"gt": 0}}},
                    {"range": {"last_visit": {"gte": cutoff_str, "lte": end_str}}},
                ]
            }
        },
        "aggs": {"active_customers": {"value_count": {"field": "customer_id"}}},
    }

    res = _safe_es_search(client, index=customers_index, body=body)
    active_count = float(((res.get("aggregations") or {}).get("active_customers") or {}).get("value") or 0.0)

    rows = [{"metric": "active_customers", "label": "Active Customers", "value": active_count}]
    insight = (
        f"'Active Customers' (as-of {end_str}) = sales_pickup_lifetime > 0 AND "
        f"last_visit between {cutoff_str} and {end_str} on customers index '{customers_index}'. "
        f"Result: {int(active_count)}."
    )

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
    customers_index, cust_mappings, err = _load_customers_ctx(req, client, business_rules)
    if err:
        return err

    end_d = _as_of_utc_date(req)
    cutoff_d = end_d - timedelta(days=int(days_threshold))
    end_str = end_d.isoformat()
    cutoff_str = cutoff_d.isoformat()

    need = ["customer_id", "sales_pickup_lifetime", "last_visit"]
    missing_err = _require_fields(cust_mappings, customers_index, need, "Active Customer Rate", business_rules)
    if missing_err is not None:
        return missing_err

    body = {
        "size": 0,
        "query": {"bool": {"filter": [{"range": {"sales_pickup_lifetime": {"gt": 0}}}]}},
        "aggs": {
            "total_paying": {"value_count": {"field": "customer_id"}},
            "active": {
                "filter": {"range": {"last_visit": {"gte": cutoff_str, "lte": end_str}}},
                "aggs": {"customers": {"value_count": {"field": "customer_id"}}},
            },
        },
    }

    res = _safe_es_search(client, index=customers_index, body=body)
    aggs = res.get("aggregations") or {}

    total_paying = float((aggs.get("total_paying") or {}).get("value") or 0.0)
    active = float((((aggs.get("active") or {}).get("customers") or {}).get("value")) or 0.0)

    if total_paying <= 0:
        return {
            "insight": to_json_safe(
                "Active Customer Rate could not be computed because there are no customers with sales_pickup_lifetime > 0."
            ),
            "rows": [],
            "rules_used": business_rules or "",
            "engine": "es",
        }

    rate = (active * 100.0) / total_paying

    rows = [
        {
            "metric": "active_customer_rate",
            "label": "Active Customer Rate (%)",
            "value": rate,
            "active_customers": active,
            "total_paying_customers": total_paying,
            "days_threshold": int(days_threshold),
            "as_of_end_date": end_str,
            "cutoff_date": cutoff_str,
            "customers_index": customers_index,
        }
    ]

    insight = (
        f"Active Customer Rate (as-of {end_str}) = active paying / paying * 100, "
        f"where paying is sales_pickup_lifetime > 0 and active means last_visit between {cutoff_str} and {end_str}. "
        f"On '{customers_index}': {int(active)} active out of {int(total_paying)} paying → ~{rate:.1f}%."
    )

    return {
        "insight": to_json_safe(insight),
        "rows": to_json_safe(rows),
        "rules_used": business_rules or "",
        "engine": "es",
    }


def _es_30d_activity_rate(req, client, mappings: Dict[str, Any], business_rules: Optional[str]):
    customers_index, cust_mappings, err = _load_customers_ctx(req, client, business_rules)
    if err:
        return err

    need = ["customer_id", "sales_pickup_lifetime", "sales_pickup_30"]
    missing_err = _require_fields(cust_mappings, customers_index, need, "30-Day Activity Rate", business_rules)
    if missing_err:
        return missing_err

    body = {
        "size": 0,
        "query": {"bool": {"filter": [{"range": {"sales_pickup_lifetime": {"gt": 0}}}]}},
        "aggs": {
            "total_paying": {"value_count": {"field": "customer_id"}},
            "active_30d": {
                "filter": {"range": {"sales_pickup_30": {"gt": 0}}},
                "aggs": {"customers": {"value_count": {"field": "customer_id"}}},
            },
        },
    }

    res = _safe_es_search(client, index=customers_index, body=body)
    aggs = res.get("aggregations") or {}

    total_paying = float((aggs.get("total_paying") or {}).get("value") or 0.0)
    active_30d = float((((aggs.get("active_30d") or {}).get("customers") or {}).get("value")) or 0.0)

    if total_paying <= 0:
        return {
            "insight": to_json_safe("30-Day Activity Rate could not be computed because no paying customers were found."),
            "rows": [],
            "rules_used": business_rules or "",
            "engine": "es",
        }

    rate = (active_30d * 100.0) / total_paying
    rows = [
        {
            "metric": "activity_30d_rate",
            "label": "30-Day Activity Rate (%)",
            "value": rate,
            "customers_30d": active_30d,
            "total_paying_customers": total_paying,
            "customers_index": customers_index,
        }
    ]
    insight = "30-Day Activity Rate = (paying customers with sales_pickup_30 > 0) / (paying customers) × 100."
    return {"insight": to_json_safe(insight), "rows": to_json_safe(rows), "rules_used": business_rules or "", "engine": "es"}


def _es_days_since_last_visit_distribution(req, client, mappings: Dict[str, Any], business_rules: Optional[str]):
    customers_index, cust_mappings, err = _load_customers_ctx(req, client, business_rules)
    if err:
        return err

    missing_err = _require_fields(
        cust_mappings,
        customers_index,
        ["customer_id", "last_visit"],
        "Days Since Last Visit Distribution",
        business_rules,
    )
    if missing_err:
        return missing_err

    bucket_filters = {
        "0_30": {"range": {"last_visit": {"gte": "now-30d/d", "lt": "now+1d/d"}}},
        "31_60": {"range": {"last_visit": {"gte": "now-60d/d", "lt": "now-30d/d"}}},
        "61_90": {"range": {"last_visit": {"gte": "now-90d/d", "lt": "now-60d/d"}}},
        "91_180": {"range": {"last_visit": {"gte": "now-180d/d", "lt": "now-90d/d"}}},
        "181_365": {"range": {"last_visit": {"gte": "now-365d/d", "lt": "now-180d/d"}}},
        "365_plus": {"range": {"last_visit": {"lt": "now-365d/d"}}},
    }

    body = {
        "size": 0,
        "query": {"bool": {"filter": [{"exists": {"field": "customer_id"}}, {"exists": {"field": "last_visit"}}]}},
        "aggs": {
            "total_customers": {"value_count": {"field": "customer_id"}},
            "days_since_last": {
                "filters": {"filters": bucket_filters},
                "aggs": {"customers": {"value_count": {"field": "customer_id"}}},
            },
        },
    }

    res = _safe_es_search(client, index=customers_index, body=body)
    aggs = res.get("aggregations") or {}

    total_customers = int((aggs.get("total_customers") or {}).get("value") or 0)
    if total_customers == 0:
        return {
            "insight": to_json_safe("No customers with last_visit were found, so distribution cannot be computed."),
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
        count = int(((b.get("customers") or {}).get("value")) or 0)
        pct = (count * 100.0 / float(total_customers)) if total_customers > 0 else 0.0
        rows.append(
            {
                "bucket_id": key,
                "label": label,
                "min_days_since_last_visit": min_days,
                "max_days_since_last_visit": max_days,
                "customer_count": count,
                "percentage_of_customers": pct,
                "risk_level": risk,
            }
        )

    insight = (
        "Days Since Last Visit Distribution groups customers by recency of last_visit into 0–30, 31–60, 61–90, "
        "91–180, 181–365, and 365+ day buckets."
    )
    return {"insight": to_json_safe(insight), "rows": to_json_safe(rows), "rules_used": business_rules or "", "engine": "es"}


def _es_avg_visit_interval(req, client, mappings: Dict[str, Any], business_rules: Optional[str]):
    customers_index, cust_mappings, err = _load_customers_ctx(req, client, business_rules)
    if err:
        return err

    missing_err = _require_fields(
        cust_mappings,
        customers_index,
        ["customer_id", "visits_interval_avg", "visits_lifetime", "original_signup", "last_visit"],
        "Average Visit Interval",
        business_rules,
    )
    if missing_err:
        return missing_err

    min_visits_lifetime = _get_req_int(req, "min_visits_lifetime_for_interval", 2, min_v=2, max_v=10_000)
    min_interval_days = _get_req_int(req, "min_interval_days_for_avg", 7, min_v=1, max_v=10_000)
    min_signup_age_days = _get_req_int(req, "min_signup_age_days_for_interval", 90, min_v=1, max_v=10_000)

    as_of = _as_of_utc_date(req)
    cutoff_signup_str = (as_of - timedelta(days=min_signup_age_days)).isoformat()

    filters: List[Dict[str, Any]] = [
        {"exists": {"field": "visits_interval_avg"}},
        {"range": {"visits_lifetime": {"gte": min_visits_lifetime}}},
        {"range": {"visits_interval_avg": {"gt": min_interval_days}}},
        {"range": {"original_signup": {"lte": cutoff_signup_str}}},
    ]

    start = getattr(req, "start_date", None)
    end = getattr(req, "end_date", None)
    if start or end:
        r: Dict[str, Any] = {}
        if start:
            r["gte"] = start
        if end:
            r["lte"] = end
        filters.append({"range": {"last_visit": r}})

    body: Dict[str, Any] = {
        "size": 0,
        "query": {"bool": {"filter": filters}},
        "aggs": {
            "avg_interval": {"avg": {"field": "visits_interval_avg"}},
            "count_customers": {"cardinality": {"field": "customer_id"}},
        },
    }

    res = _safe_es_search(client, index=customers_index, body=body)
    aggs = res.get("aggregations") or {}

    avg_interval_val = (aggs.get("avg_interval") or {}).get("value")
    count = int((aggs.get("count_customers") or {}).get("value") or 0)

    if avg_interval_val is None or count == 0:
        return {
            "insight": to_json_safe("Average Visit Interval could not be computed because no customers matched the filters."),
            "rows": [],
            "rules_used": business_rules or "",
            "engine": "es",
        }

    rows = [
        {
            "metric": "avg_visit_interval_days",
            "label": "Average Visit Interval (days)",
            "value": float(avg_interval_val),
            "customers_counted": count,
            "customers_index": customers_index,
            "min_visits_lifetime": int(min_visits_lifetime),
            "min_interval_days": int(min_interval_days),
            "min_signup_age_days": int(min_signup_age_days),
            "as_of": as_of.isoformat(),
            "window_start": start,
            "window_end": end,
        }
    ]

    insight = f"Average Visit Interval computed from visits_interval_avg on '{customers_index}' (as of {as_of.isoformat()})."
    return {"insight": to_json_safe(insight), "rows": to_json_safe(rows), "rules_used": business_rules or "", "engine": "es"}

def _es_avg_days_between_visits_active(req, client, mappings: Dict[str, Any], business_rules: Optional[str]):
    customers_index, cust_mappings, err = _load_customers_ctx(req, client, business_rules)
    if err:
        return err

    missing_err = _require_fields(
        cust_mappings,
        customers_index,
        ["customer_id", "last_visit", "visits_lifetime", "visits_interval_avg"],
        "Avg Days Between Visits (Active)",
        business_rules,
    )
    if missing_err:
        return missing_err

    start = getattr(req, "start_date", None)
    end = getattr(req, "end_date", None)

    filters: List[Dict[str, Any]] = [
        {"exists": {"field": "customer_id"}},
        {"range": {"visits_lifetime": {"gte": 2}}},
        {"range": {"visits_interval_avg": {"gt": 0}}},
    ]

    # ✅ CHANGED: "active" window comes from dashboard (req.start_date/end_date) when provided
    if start or end:
        r: Dict[str, Any] = {}
        if start:
            r["gte"] = start
        if end:
            r["lte"] = end
        filters.append({"range": {"last_visit": r}})
        active_window_days = None  # window is explicit dates
    else:
        cutoff = _cutoff_iso(365)
        filters.append({"range": {"last_visit": {"gte": cutoff}}})
        active_window_days = 365

    paying_filter_applied = False
    if _field_exists(cust_mappings, "sales_pickup_lifetime"):
        filters.append({"range": {"sales_pickup_lifetime": {"gt": 0}}})
        paying_filter_applied = True

    body = {
        "size": 0,
        "query": {"bool": {"filter": filters}},
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
                "No active repeat customers matched the filters, so the average gap could not be computed."
            ),
            "rows": [],
            "rules_used": business_rules or "",
            "engine": "es",
        }

    rows = [
        {
            "metric": "avg_days_between_visits_active",
            "label": "Avg Days Between Visits (Active, via visits_interval_avg)",
            "value": float(avg_gap),
            "customers_counted": count,
            "customers_index": customers_index,
            "paying_filter_applied": paying_filter_applied,
            # ✅ helpful debug fields for dashboard
            "window_start": start,
            "window_end": end,
            "active_window_days": active_window_days,
        }
    ]

    if start or end:
        insight = (
            f"Avg days between visits for repeat customers active in the selected window "
            f"({start or '-∞'} → {end or '+∞'}) computed from visits_interval_avg on '{customers_index}'. "
            f"Result: ~{float(avg_gap):.1f} days across {count} customers."
        )
    else:
        insight = (
            f"Avg days between visits for active repeat customers (last 365 days) computed from visits_interval_avg "
            f"on '{customers_index}'. Result: ~{float(avg_gap):.1f} days across {count} customers."
        )

    return {"insight": to_json_safe(insight), "rows": to_json_safe(rows), "rules_used": business_rules or "", "engine": "es"}


def _es_avg_customer_lifespan(req, client, mappings: Dict[str, Any], business_rules: Optional[str]):
    customers_index, cust_mappings, err = _load_customers_ctx(req, client, business_rules)
    if err:
        return err

    signup_field: Optional[str] = None
    for cand in ("original_signup", "signup_date"):
        if _field_exists(cust_mappings, cand):
            signup_field = cand
            break

    if not signup_field:
        return _es_cannot_answer(
            "Cannot compute 'Average Customer Lifespan' because original_signup/signup_date is missing.",
            business_rules,
        )

    missing_err = _require_fields(
        cust_mappings,
        customers_index,
        ["customer_id", "last_visit", signup_field],
        "Average Customer Lifespan",
        business_rules,
    )
    if missing_err:
        return missing_err

    filters: List[Dict[str, Any]] = [
        {"exists": {"field": signup_field}},
        {"exists": {"field": "last_visit"}},
        {"exists": {"field": "customer_id"}},
    ]

    start_d = _parse_date_str(getattr(req, "start_date", None))
    end_d = _parse_date_str(getattr(req, "end_date", None))
    if start_d:
        filters.append({"range": {"last_visit": {"gte": start_d.isoformat()}}})
    if end_d:
        filters.append({"range": {"last_visit": {"lte": end_d.isoformat()}}})

    ms_per_day = 1000.0 * 60.0 * 60.0 * 24.0
    body: Dict[str, Any] = {
        "size": 0,
        "query": {"bool": {"filter": filters}},
        "aggs": {
            "customers_counted": {"value_count": {"field": "customer_id"}},
            "avg_lifespan_ms": {
                "avg": {
                    "script": {
                        "lang": "painless",
                        "source": (
                            "doc['last_visit'].value.toInstant().toEpochMilli() - "
                            f"doc['{signup_field}'].value.toInstant().toEpochMilli()"
                        ),
                    }
                }
            },
        },
    }

    res = _safe_es_search(client, index=customers_index, body=body)
    aggs = res.get("aggregations") or {}

    count = int((aggs.get("customers_counted") or {}).get("value") or 0)
    avg_ms = (aggs.get("avg_lifespan_ms") or {}).get("value")

    if count == 0 or avg_ms is None:
        return {
            "insight": to_json_safe("Average Customer Lifespan could not be computed because no customers matched the filters."),
            "rows": [],
            "rules_used": business_rules or "",
            "engine": "es",
        }

    avg_days = float(avg_ms) / ms_per_day

    rows = [
        {
            "metric": "avg_customer_lifespan_days",
            "label": "Average Customer Lifespan (days)",
            "value": avg_days,
            "customers_counted": count,
            "customers_index": customers_index,
            "signup_field": signup_field,
        }
    ]

    insight = (
        f"Average Customer Lifespan computed as avg(last_visit - {signup_field}) in days on '{customers_index}'. "
        f"Result: ~{avg_days:.1f} days across {count} customers."
    )
    return {"insight": to_json_safe(insight), "rows": to_json_safe(rows), "rules_used": business_rules or "", "engine": "es"}


def _es_lifecycle_engagement(req, client, mappings: Dict[str, Any], business_rules: Optional[str]):
    customers_index, cust_mappings, err = _load_customers_ctx(req, client, business_rules)
    if err:
        return err

    missing_err = _require_fields(
        cust_mappings,
        customers_index,
        ["customer_id", "last_visit"],
        "Lifecycle Engagement",
        business_rules,
    )
    if missing_err:
        return missing_err

    has_signup = _field_exists(cust_mappings, "original_signup")

    base_filters, paying_only_applied = _base_customer_filters(cust_mappings)

    new_days = 30
    new_range_q = {"range": {"original_signup": {"gte": f"now-{new_days}d/d"}}}

    def _not_new_clause() -> List[Dict[str, Any]]:
        return [new_range_q] if has_signup else []

    filters: Dict[str, Any] = {}

    if has_signup:
        filters["new"] = {"bool": {"filter": [{"exists": {"field": "original_signup"}}, new_range_q]}}

    filters["unknown"] = {"bool": {"must_not": [{"exists": {"field": "last_visit"}}, *(_not_new_clause())]}}

    def _stage_range(gte_expr: Optional[str], lt_expr: Optional[str]) -> Dict[str, Any]:
        r: Dict[str, Any] = {}
        if gte_expr is not None:
            r["gte"] = gte_expr
        if lt_expr is not None:
            r["lt"] = lt_expr
        return {"range": {"last_visit": r}}

    def _stage_bool(range_q: Dict[str, Any]) -> Dict[str, Any]:
        q: Dict[str, Any] = {"bool": {"filter": [range_q]}}
        if has_signup:
            q["bool"]["must_not"] = [new_range_q]
        return q

    filters["active"] = _stage_bool(_stage_range("now-30d/d", "now+1d/d"))
    filters["warm"] = _stage_bool(_stage_range("now-60d/d", "now-30d/d"))
    filters["cooling"] = _stage_bool(_stage_range("now-90d/d", "now-60d/d"))
    filters["at_risk"] = _stage_bool(_stage_range("now-180d/d", "now-90d/d"))
    filters["lapsed"] = _stage_bool(_stage_range("now-365d/d", "now-180d/d"))
    filters["lost"] = _stage_bool(_stage_range(None, "now-365d/d"))

    body = {
        "size": 0,
        "query": {"bool": {"filter": base_filters}},
        "aggs": {
            "total_customers": {"value_count": {"field": "customer_id"}},
            "lifecycle": {
                "filters": {"filters": filters},
                "aggs": {"customers": {"value_count": {"field": "customer_id"}}},
            },
        },
    }

    res = _safe_es_search(client, index=customers_index, body=body)
    aggs = res.get("aggregations") or {}

    total = int((aggs.get("total_customers") or {}).get("value") or 0)
    if total == 0:
        return {
            "insight": to_json_safe("Lifecycle Engagement could not be computed because no customers matched the base population."),
            "rows": [],
            "rules_used": business_rules or "",
            "engine": "es",
        }

    buckets = (((aggs.get("lifecycle") or {}).get("buckets")) or {})

    stage_order = []
    if has_signup:
        stage_order.append(("new", "New (signed up ≤30d)", "Welcome + onboarding sequence"))
    stage_order += [
        ("unknown", "Unknown (no last_visit)", "Fix data + encourage first visit"),
        ("active", "Active (0–30d)", "Maintain engagement + upsell"),
        ("warm", "Warm (31–60d)", "Nudge with offers / reminders"),
        ("cooling", "Cooling (61–90d)", "Win-back campaign (soft)"),
        ("at_risk", "At Risk (91–180d)", "Win-back campaign (strong)"),
        ("lapsed", "Lapsed (181–365d)", "Reactivation offer"),
        ("lost", "Lost (365d+)", "Long-lapse reactivation / suppress"),
    ]

    rows: List[Dict[str, Any]] = []
    for key, label, action in stage_order:
        b = buckets.get(key) or {}
        count = int(((b.get("customers") or {}).get("value")) or 0)
        pct = (count * 100.0 / float(total)) if total > 0 else 0.0
        rows.append(
            {
                "stage_id": key,
                "label": label,
                "customer_count": count,
                "percentage_of_base": pct,
                "recommended_action": action,
            }
        )

    note = ""
    if not has_signup:
        note = " NOTE: 'original_signup' is missing, so the NEW stage was not computed."
    if not paying_only_applied:
        note += " NOTE: sales_pickup_lifetime is missing, so lifecycle was computed on all customers (not only paying customers)."

    insight = (
        f"Lifecycle Engagement segments customers on '{customers_index}' into engagement stages based on recency of last_visit"
        f"{' and NEW based on original_signup' if has_signup else ''}. Base population is customers with customer_id"
        f"{' and sales_pickup_lifetime > 0' if paying_only_applied else ''}."
        f"{note}"
    )
    return {"insight": to_json_safe(insight), "rows": to_json_safe(rows), "rules_used": business_rules or "", "engine": "es"}


__all__ = [
    "_es_active_customers",
    "_es_active_customer_rate",
    "_es_30d_activity_rate",
    "_es_days_since_last_visit_distribution",
    "_es_avg_visit_interval",
    "_es_avg_days_between_visits_active",
    "_es_avg_customer_lifespan",
    "_es_lifecycle_engagement",
]
