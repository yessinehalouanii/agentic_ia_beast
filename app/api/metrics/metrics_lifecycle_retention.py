from __future__ import annotations

from datetime import datetime, timezone, timedelta
from typing import Any, Dict, List, Optional, Iterable, Tuple

from abi.runtime import to_json_safe
from app.api.metrics.metrics_lifecycle_engagement import _as_of_utc_date, _require_fields
from app.api.metrics.shared_utilities import (
    _field_exists,
    _safe_es_search,
    _get_req_int,
    _load_customers_ctx,  # ✅ NEW: now comes from shared_utilities
)

from app.api.docs_analytics_routes import (
    _parse_date_str,
    _es_cannot_answer,
)

# -------------------------------------------------------------------
# ✅ NEW: reuse customers ctx passed from dashboard/orchestrator
# -------------------------------------------------------------------

def _load_customers_ctx_reuse(
    req,
    client,
    mappings: Optional[Dict[str, Any]],
    business_rules: Optional[str],
) -> Tuple[Optional[str], Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
    """
    Dashboard path:
      - orchestrator resolves customers_index + cust_mappings once
      - passes cust_mappings as `mappings` into each customers-metric

    Direct/single-metric path:
      - `mappings` may be empty -> fallback to _load_customers_ctx(req, client, business_rules)

    Returns: (customers_index, cust_mappings, err)
    """
    customers_index = (getattr(req, "es_customers_index_name", "") or "").strip()

    if isinstance(mappings, dict):
        props = mappings.get("properties")
        if isinstance(props, dict) and props and customers_index:
            return customers_index, mappings, None

    # fallback: load inside (older behavior)
    return _load_customers_ctx(req, client, business_rules)


# -------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------

def _missing_required_fields(mappings: Dict[str, Any], required_fields: List[str]) -> List[str]:
    return [f for f in required_fields if not _field_exists(mappings, f)]


def _get_global_avg_interval_days_repeat_customers(client, customers_index: str) -> Optional[float]:
    """
    Global avg interval in days, computed from visits_interval_avg on repeat customers only:
      - visits_interval_avg exists and > 0
      - visits_lifetime >= 2
    """
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

    if avg_val is None:
        return None

    try:
        v = float(avg_val)
    except Exception:
        return None

    return v if v > 0 else None


def _coerce_to_date(value) -> Optional[Any]:
    """
    Best-effort conversion to a date-like object without importing datetime.date.
    - datetime -> .date()
    - str -> _parse_date_str()
    - already date-like -> returned as-is
    """
    if value is None:
        return None
    from datetime import datetime as _dt  # local import

    if isinstance(value, _dt):
        return value.date()
    if isinstance(value, str):
        return _parse_date_str(value)
    try:
        _ = value.year
        _ = value.month
        _ = value.day
        return value
    except Exception:
        return None


def _chunks(lst: List[Any], n: int) -> Iterable[List[Any]]:
    for i in range(0, len(lst), n):
        yield lst[i : i + n]


# -------------------------------------------------------------------
# Metrics
# -------------------------------------------------------------------
def _es_customer_retention_rate_730_180(
    req,
    client,
    mappings: Dict[str, Any],
    business_rules: Optional[str],
):
    """
    Customer Retention Rate (730d cohort → 180d active).

    ✅ Period-anchored ("as of" req.end_date):
      - Anchor date = req.end_date (if provided), else today (UTC)
      - Customers730     = customers with >0 lifetime sales AND last_visit in [anchor-730, anchor]
      - Active180From730 = subset of Customers730 with last_visit in [anchor-180, anchor]
      - Retention Rate   = (Active180From730 / Customers730) × 100

    NOTE: invoices fallback removed — customers index is required.
    """
    outer_days = 730
    inner_days = 180

    # ----------------------------
    # ✅ MODIFIED: anchor to period end_date (dashboard passes this)
    # ----------------------------
    anchor = _parse_date_str(getattr(req, "end_date", None)) or datetime.now(timezone.utc).date()
    cutoff_outer_str = (anchor - timedelta(days=outer_days)).isoformat()
    cutoff_inner_str = (anchor - timedelta(days=inner_days)).isoformat()
    anchor_str = anchor.isoformat()

    customers_index, cust_mappings, err = _load_customers_ctx_reuse(req, client, mappings, business_rules)
    if err:
        return err

    required = ["customer_id", "last_visit", "sales_pickup_lifetime"]
    missing = [f for f in required if not _field_exists(cust_mappings, f)]
    if missing:
        return _es_cannot_answer(
            "Cannot compute Customer Retention Rate (730→180) because required fields are missing from "
            f"customers index '{customers_index}': {', '.join(missing)}.",
            business_rules,
        )

    # ----------------------------
    # ✅ MODIFIED: add lte anchor_str so the cohort is truly "as of end_date"
    # ----------------------------
    body = {
        "size": 0,
        "query": {
            "bool": {
                "filter": [
                    {"exists": {"field": "customer_id"}},
                    {"range": {"sales_pickup_lifetime": {"gt": 0}}},
                    {"range": {"last_visit": {"gte": cutoff_outer_str, "lte": anchor_str}}},
                ]
            }
        },
        "aggs": {
            "customers_730d": {"value_count": {"field": "customer_id"}},
            "active_180_from_730": {
                "filter": {"range": {"last_visit": {"gte": cutoff_inner_str, "lte": anchor_str}}},
                "aggs": {"customers_180d": {"value_count": {"field": "customer_id"}}},
            },
        },
    }

    res = _safe_es_search(client, index=customers_index, body=body)
    aggs = res.get("aggregations") or {}

    customers_730 = float((aggs.get("customers_730d") or {}).get("value") or 0.0)
    active_agg = (aggs.get("active_180_from_730") or {}).get("customers_180d") or {}
    active_180 = float(active_agg.get("value") or 0.0)

    if customers_730 <= 0:
        return {
            "insight": to_json_safe(
                "Customer Retention Rate (730→180) could not be computed because no paying customers "
                f"had a last_visit in the 730-day window ending on {anchor_str}."
            ),
            "rows": [],
            "rules_used": business_rules or "",
            "engine": "es",
        }

    rate = (active_180 * 100.0) / customers_730

    rows = [
        {
            "metric": "customers_730d",
            "label": f"Customers with visit in last 730 days (ending {anchor_str})",
            "value": customers_730,
        },
        {
            "metric": "active_180_from_730",
            "label": f"Still active (visited in last 180 days, ending {anchor_str})",
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
        f"'{customers_index}' as of {anchor_str}. Among customers with >0 lifetime sales and a last_visit "
        f"in the last {outer_days} days, {int(active_180)} also visited in the last {inner_days} days, "
        f"yielding a retention rate of {rate:.1f}%."
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
    """
    customers_index, cust_mappings, err = _load_customers_ctx_reuse(req, client, mappings, business_rules)
    if err:
        return err

    required_fields = ["customer_id", "visits_365"]
    missing = _missing_required_fields(cust_mappings, required_fields)
    if missing:
        return _es_cannot_answer(
            "Cannot compute 'Repeat Customers 365' because required fields are missing "
            f"from customers index '{customers_index}': {', '.join(missing)}.",
            business_rules,
        )

    has_signup = _field_exists(cust_mappings, "original_signup")

    min_signup_age_days = _get_req_int(req, "min_signup_age_days_for_repeat_365", 180, min_v=1, max_v=10_000)
    today = datetime.now(timezone.utc).date()
    cutoff_signup_str = (today - timedelta(days=int(min_signup_age_days))).isoformat()

    base_filters: List[Dict[str, Any]] = [
        {"exists": {"field": "customer_id"}},
        {"exists": {"field": "visits_365"}},
        {"range": {"visits_365": {"gt": 0}}},
    ]

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
            "base_customers": {"value_count": {"field": "customer_id"}},
            "repeat_customers": {
                "filter": {"range": {"visits_365": {"gte": 2}}},
                "aggs": {"customers": {"value_count": {"field": "customer_id"}}},
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

    return {"insight": to_json_safe(insight), "rows": to_json_safe(rows), "rules_used": business_rules or "", "engine": "es"}


def _es_single_visit_lifetime(
    req,
    client,
    mappings: Dict[str, Any],
    business_rules: Optional[str],
):
    """
    Single Visit (Lifetime) %
    """
    customers_index, cust_mappings, err = _load_customers_ctx_reuse(req, client, mappings, business_rules)
    if err:
        return err

    required_fields = ["customer_id", "visits_lifetime", "sales_pickup_lifetime"]
    missing = _missing_required_fields(cust_mappings, required_fields)
    if missing:
        return _es_cannot_answer(
            "Cannot compute 'Single Visit (Lifetime)' because required fields are missing "
            f"from customers index '{customers_index}': {', '.join(missing)}.",
            business_rules,
        )

    base_filter: List[Dict[str, Any]] = [
        {"exists": {"field": "customer_id"}},
        {"exists": {"field": "sales_pickup_lifetime"}},
        {"range": {"sales_pickup_lifetime": {"gt": 0}}},
    ]

    body: Dict[str, Any] = {
        "size": 0,
        "query": {"bool": {"filter": base_filter}},
        "aggs": {
            "customers_with_sales": {"value_count": {"field": "customer_id"}},
            "single_visit": {
                "filter": {"term": {"visits_lifetime": 1}},
                "aggs": {"customers": {"value_count": {"field": "customer_id"}}},
            },
        },
    }

    res = _safe_es_search(client, index=customers_index, body=body)
    aggs = res.get("aggregations") or {}

    total_with_sales = int((aggs.get("customers_with_sales") or {}).get("value") or 0)
    single_bucket = aggs.get("single_visit") or {}
    single_count = int(((single_bucket.get("customers") or {}).get("value")) or 0)

    if total_with_sales == 0:
        return {
            "insight": to_json_safe(
                "Single Visit (Lifetime) % could not be computed because no customers "
                f"with sales_pickup_lifetime > 0 were found in customers index '{customers_index}'."
            ),
            "rows": [],
            "rules_used": business_rules or "",
            "engine": "es",
        }

    single_pct = (single_count / total_with_sales) * 100.0

    insight = (
        "Single Visit (Lifetime) % measures what share of all paying customers have only ever visited once. "
        f"On customers index '{customers_index}', ~{single_pct:.1f}% of paying customers "
        f"({single_count} out of {total_with_sales}) are single-visit customers."
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

    return {"insight": to_json_safe(insight), "rows": to_json_safe(rows), "rules_used": business_rules or "", "engine": "es"}


def _es_single_visit_365(
    req,
    client,
    mappings: Dict[str, Any],
    business_rules: Optional[str],
):
    """
    Single Visit (365 Days)
    """
    customers_index, cust_mappings, err = _load_customers_ctx_reuse(req, client, mappings, business_rules)
    if err:
        return err

    required_fields = ["visits_365", "original_signup", "customer_id"]
    missing = _missing_required_fields(cust_mappings, required_fields)
    if missing:
        return _es_cannot_answer(
            "Cannot compute 'Single Visit 365%' because required fields are missing "
            f"from customers index '{customers_index}': {', '.join(missing)}.",
            business_rules,
        )

    min_signup_age_days = _get_req_int(req, "min_signup_age_days_for_single_365", 180, min_v=1, max_v=10_000)
    today = datetime.now(timezone.utc).date()
    cutoff_signup_str = (today - timedelta(days=min_signup_age_days)).isoformat()

    base_filters: List[Dict[str, Any]] = [
        {"exists": {"field": "customer_id"}},
        {"exists": {"field": "visits_365"}},
        {"range": {"visits_365": {"gt": 0}}},
    ]

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
        "query": {"bool": {"filter": base_filters, "must_not": must_not}},
        "aggs": {
            "total_customers": {"value_count": {"field": "customer_id"}},
            "single_visit_customers": {
                "filter": {"term": {"visits_365": 1}},
                "aggs": {"count": {"value_count": {"field": "customer_id"}}},
            },
        },
    }

    res = _safe_es_search(client, index=customers_index, body=body)
    aggs = res.get("aggregations") or {}

    total = int((aggs.get("total_customers") or {}).get("value") or 0)
    single_visit_bucket = aggs.get("single_visit_customers") or {}
    single_visit_count = int(((single_visit_bucket.get("count") or {}).get("value")) or 0)

    if total == 0:
        return {
            "insight": to_json_safe(
                "Single Visit 365% could not be computed because no customers matched the filters "
                "(visits_365 > 0 after excluding very new one-visit customers)."
            ),
            "rows": [],
            "rules_used": business_rules or "",
            "engine": "es",
        }

    pct_single = (single_visit_count / total) * 100.0

    insight = (
        "Single Visit 365% measures the percentage of customers who had exactly one visit in the last 365 days, "
        f"after excluding very new one-visit customers (signup within last {min_signup_age_days} days). "
        f"On index '{customers_index}', {single_visit_count} out of {total} eligible customers are single-visit "
        f"→ ~{pct_single:.1f}%."
    )

    rows = [
        {
            "metric": "single_visit_365_pct",
            "label": "Single Visit 365% (established customers)",
            "value": pct_single,
            "single_visit_customers": single_visit_count,
            "total_customers": total,
            "min_signup_age_days": int(min_signup_age_days),
            "customers_index": customers_index,
        }
    ]

    return {"insight": to_json_safe(insight), "rows": to_json_safe(rows), "rules_used": business_rules or "", "engine": "es"}

def _es_churn_rate(
    req,
    client,
    mappings: Dict[str, Any],
    business_rules: Optional[str],
):
    """
    Churn Rate (snapshot as-of req.end_date)
    """
    customers_index, cust_mappings, err = _load_customers_ctx_reuse(req, client, mappings, business_rules)
    if err:
        return err

    required_fields = ["visits_interval_avg", "visits_lifetime", "original_signup", "last_visit", "customer_id"]
    missing = _missing_required_fields(cust_mappings, required_fields)
    if missing:
        return _es_cannot_answer(
            "Cannot compute 'Churn Rate' because required fields are missing from customers index "
            f"'{customers_index}': {', '.join(missing)}.",
            business_rules,
        )

    avg_interval_days = _get_global_avg_interval_days_repeat_customers(client, customers_index)
    if avg_interval_days is None:
        return _es_cannot_answer(
            "Cannot compute 'Churn Rate' because Average Visit Interval could not be derived from visits_interval_avg.",
            business_rules,
        )

    avg_d_int = max(1, int(round(avg_interval_days)))

    # ✅ MODIFIED: anchor churn to req.end_date (dashboard passes it)
    # - If req.end_date is parseable, use it
    # - Else, fall back to "today" (UTC)
    end_d = _as_of_utc_date(req)
    end_str = end_d.isoformat()

    # ES date math anchored to end_str (NOT "now")
    cutoff_expr = f"{end_str}||-{avg_d_int}d/d"
    cutoff_date_str = (end_d - timedelta(days=avg_d_int)).isoformat()

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
                "aggs": {"customers": {"value_count": {"field": "customer_id"}}},
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
                "aggs": {"customers": {"value_count": {"field": "customer_id"}}},
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
            # ✅ extra debug context (safe to keep, helps dashboard validation)
            "as_of_end_date": end_str,
            "cutoff_date": cutoff_date_str,
            "cutoff_expr": cutoff_expr,
            "customers_index": customers_index,
        }
    ]

    insight = (
        "Churn Rate uses the global Average Visit Interval (from visits_interval_avg) as the threshold. "
        f"Snapshot is anchored to end_date={end_str} with cutoff={cutoff_date_str} (~{avg_d_int} days). "
        "Eligible customers are those with >=1 visit and an original_signup older than the cutoff; "
        "churned customers are eligible customers whose last_visit is also older than the cutoff. "
        f"On customers index '{customers_index}', this yields ~{churn_rate:.1f}% churn "
        f"({churned_count} churned out of {eligible_count} eligible)."
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
    customers_index, cust_mappings, err = _load_customers_ctx_reuse(req, client, mappings, business_rules)
    if err:
        return err

    required_fields = ["customer_id", "first_visit", "visits_lifetime", "visits_interval_avg"]
    missing = _missing_required_fields(cust_mappings, required_fields)
    if missing:
        return _es_cannot_answer(
            "Cannot compute 'Return Rate by Cohort Year' because required fields are missing "
            f"from customers index '{customers_index}': {', '.join(missing)}.",
            business_rules,
        )

    avg_interval_days = _get_global_avg_interval_days_repeat_customers(client, customers_index)
    if avg_interval_days is None:
        return _es_cannot_answer(
            "Cannot compute 'Return Rate by Cohort Year' because Average Visit Interval could not be derived.",
            business_rules,
        )

    avg_d_int = max(1, int(round(avg_interval_days)))
    cutoff_expr = f"now-{avg_d_int}d/d"

    cohort_body = {
        "size": 0,
        "query": {"bool": {"filter": [{"exists": {"field": "first_visit"}}, {"exists": {"field": "customer_id"}}]}},
        "aggs": {
            "by_year": {
                "date_histogram": {"field": "first_visit", "calendar_interval": "year", "min_doc_count": 1},
                "aggs": {
                    "original_customers": {"value_count": {"field": "customer_id"}},
                    "eligible": {
                        "filter": {"range": {"first_visit": {"lt": cutoff_expr}}},
                        "aggs": {
                            "eligible_customers": {"value_count": {"field": "customer_id"}},
                            "returned": {
                                "filter": {"range": {"visits_lifetime": {"gte": 2}}},
                                "aggs": {"returned_customers": {"value_count": {"field": "customer_id"}}},
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
            {"year": year, "original_cohort_size": original, "eligible_customers": eligible, "returned_customers": returned}
        )

    if not raw_rows:
        return {
            "insight": to_json_safe("Return Rate by Cohort Year could not be computed because cohort buckets were empty."),
            "rows": [],
            "rules_used": business_rules or "",
            "engine": "es",
        }

    max_year_count = max(r["original_cohort_size"] for r in raw_rows)
    threshold = max_year_count * 0.50

    filtered = [
        r
        for r in sorted(raw_rows, key=lambda x: x["year"])
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

    return {"insight": to_json_safe(insight), "rows": to_json_safe(rows), "rules_used": business_rules or "", "engine": "es"}


def _es_overdue_customers(req, client, mappings, business_rules):
    customers_index, cust_mappings, err = _load_customers_ctx_reuse(req, client, mappings, business_rules)
    if err:
        return err

    missing_err = _require_fields(
        cust_mappings,
        customers_index,
        ["customer_id", "last_visit", "visits_lifetime", "visits_interval_avg"],
        "Overdue Customers",
        business_rules,
    )
    if missing_err:
        return missing_err

    max_rows = _get_req_int(req, "es_max_rows", 2000, min_v=100, max_v=20_000)

    now_ms = int(datetime.now(timezone.utc).timestamp() * 1000)

    filters = [
        {"range": {"visits_lifetime": {"gte": 2}}},
        {"range": {"visits_interval_avg": {"gt": 0}}},
        {"exists": {"field": "last_visit"}},
        {
            "script": {
                "script": {
                    "lang": "painless",
                    "params": {"now_ms": now_ms, "mult": 1.5},
                    "source": """
                      long lv = doc['last_visit'].value.toInstant().toEpochMilli();
                      double days_since = (params.now_ms - lv) / 86400000.0;
                      double interval = doc['visits_interval_avg'].value;
                      return days_since > params.mult * interval;
                    """,
                }
            }
        },
    ]

    if _field_exists(cust_mappings, "sales_pickup_lifetime"):
        filters.insert(0, {"range": {"sales_pickup_lifetime": {"gt": 0}}})

    body = {
        "size": max_rows,
        "track_total_hits": True,
        "_source": ["customer_id", "last_visit", "visits_interval_avg", "visits_lifetime"],
        "query": {"bool": {"filter": filters}},
        "sort": [{"last_visit": "asc"}],
    }

    res = _safe_es_search(client, index=customers_index, body=body)

    total_overdue = int((res.get("hits", {}).get("total") or {}).get("value") or 0)
    hits = res.get("hits", {}).get("hits", []) or []

    rows = []
    for h in hits:
        src = h.get("_source") or {}
        rows.append(
            {
                "customer_id": src.get("customer_id"),
                "last_visit": src.get("last_visit"),
                "visits_interval_avg": src.get("visits_interval_avg"),
                "visits_lifetime": src.get("visits_lifetime"),
            }
        )

    insight = f"Found {total_overdue} overdue customers (customers index='{customers_index}'). Rows capped at {max_rows}."
    return {"insight": to_json_safe(insight), "rows": to_json_safe(rows), "rules_used": business_rules or "", "engine": "es"}


__all__ = [
    "_es_customer_retention_rate_730_180",
    "_es_repeat_customers_365",
    "_es_single_visit_lifetime",
    "_es_single_visit_365",
    "_es_churn_rate",
    "_es_return_rate_by_cohort_year_customers",
    "_es_overdue_customers",
]
