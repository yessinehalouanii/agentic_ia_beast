from __future__ import annotations

import os
from datetime import datetime, timezone, timedelta
from typing import List, Optional, Dict, Any, Literal
from copy import deepcopy

from fastapi import APIRouter, HTTPException    
from pydantic import BaseModel, Field

from helpers.analytics_helpers import ALIASES_UNIVERSAL
from abi.docs_rules import get_business_rules
from abi.llm import llm_codegen
from abi.runtime import run_generated_code, to_json_safe
from app.core.table_store import TABLE_STORE

# 🔹 ES helpers
from abi.es_llm import llm_generate_es_query
from app.core.es_dynamic import make_es_client
from routes.es_test import (
    _parse_es_dsl,
    _flatten_docs_to_rows,
)

from app.api.metrics.shared_utilities import (
    _get_invoice_index_and_mappings,   # ✅ NEW: central resolver
    _field_exists,
    _safe_es_search,
)

router = APIRouter(prefix="/docs", tags=["Docs Analytics"])


# -------------------------------------------------------------------
# ✅ SAFETY KNOBS (prod-safe defaults; override via env)
# -------------------------------------------------------------------

ES_MAX_CUSTOMERS_DEFAULT = int(os.getenv("ES_MAX_CUSTOMERS", "10000"))  # ✅ lowered from 50k
ES_COMPOSITE_MAX_PAGES = int(os.getenv("ES_COMPOSITE_MAX_PAGES", "200"))
ES_COMPOSITE_MAX_BUCKETS = int(os.getenv("ES_COMPOSITE_MAX_BUCKETS", "200000"))
ES_DSL_MAX_SIZE = int(os.getenv("ES_DSL_MAX_SIZE", "500"))             # cap hits
ES_DSL_DEFAULT_DAYS = int(os.getenv("ES_DSL_DEFAULT_DAYS", "365"))      # default window
ES_AGG_MAX_TERMS_SIZE = int(os.getenv("ES_AGG_MAX_TERMS_SIZE", "1000"))
ES_AGG_MAX_COMPOSITE_SIZE = int(os.getenv("ES_AGG_MAX_COMPOSITE_SIZE", "1000"))


# -------------------------------------------------------------------
# ✅ DIRECT FIELD MAP (uses your known mappings; avoids resolver surprises)
# -------------------------------------------------------------------

INVOICE_FIELDS: Dict[str, str] = {
    "customer_id": "customer_id",
    "date": "dropoff_at",
    "amount": "total",
    "pieces": "pieces",
    "visit_id": "visit_id",
    "location_id": "location_id",
}

CUSTOMER_FIELDS: Dict[str, str] = {
    "customer_id": "customer_id",
    "signup": "original_signup",
}


# -------------------------------------------------------------------
# Request models
# -------------------------------------------------------------------

class DocsAnalyticsRequest(BaseModel):
    workspace_id: str
    question: str

    mode: str = "predefined"
    doc_ids: Optional[List[str]] = None

    model: str = "gpt-4o-mini"
    api_key: str | None = None

    start_date: Optional[str] = None
    end_date: Optional[str] = None

    es_base_url: Optional[str] = None
    es_username: Optional[str] = None
    es_password: Optional[str] = None

    # primary (invoices) index
    es_index_name: Optional[str] = None

    # customers index (for signup dates, tags, etc.)
    es_customers_index_name: Optional[str] = None
    es_customer_stats_index_name: Optional[str] = None

    # ✅ used by some metrics (eg one-time vs repeat) via getattr(req, "repeat_basis", ...)
    repeat_basis: Optional[str] = None


class Period(BaseModel):
    start_date: str
    end_date: str


class MetricsDashboardRequest(BaseModel):
    es_base_url: str
    es_index_name: str
    es_username: Optional[str] = None
    es_password: Optional[str] = None

    # customers index to support lifecycle / signup metrics
    es_customers_index_name: Optional[str] = None
    es_customer_stats_index_name: Optional[str] = None

    workspace_id: str = "default"

    current: Period
    previous: Optional[Period] = None

    dashboard_id: Literal["performance", "ops", "lifecycle", "growth"] = "performance"

    # ✅ datasets-only period selector (optional)
    datasets_period: Optional[Period] = None

    # ✅ safety cap (hard max 3 months)
    datasets_max_months: int = Field(default=3, ge=1, le=3)


class MetricsDashboardMetric(BaseModel):
    id: str
    label: str
    current: Optional[float]
    previous: Optional[float]
    change_pct: Optional[float]
    es_customer_stats_index_name: Optional[str] = None


class DashboardDataset(BaseModel):
    """
    Non-KPI outputs: charts/tables.
    """
    id: str
    label: str
    rows: Any
    insight: str
    engine: str = "es"


def _cap_agg_sizes(obj: Any) -> None:
    """
    ✅ Safety: cap common agg sizes in arbitrary LLM-generated DSL.
    Mutates obj in place.
    """
    if isinstance(obj, dict):
        # terms agg size
        if "terms" in obj and isinstance(obj["terms"], dict):
            t = obj["terms"]
            if isinstance(t.get("size"), int):
                t["size"] = min(t["size"], ES_AGG_MAX_TERMS_SIZE)

        # composite agg size
        if "composite" in obj and isinstance(obj["composite"], dict):
            c = obj["composite"]
            if isinstance(c.get("size"), int):
                c["size"] = min(c["size"], ES_AGG_MAX_COMPOSITE_SIZE)

        for v in obj.values():
            _cap_agg_sizes(v)

    elif isinstance(obj, list):
        for v in obj:
            _cap_agg_sizes(v)


def _wrap_query_with_filter(existing_query: Optional[Dict[str, Any]], extra_filter: Dict[str, Any]) -> Dict[str, Any]:
    """
    Wrap query safely:
      - if no query: bool.filter=[extra]
      - else: bool.must=[existing], bool.filter=[extra]
    """
    if not existing_query:
        return {"bool": {"filter": [extra_filter]}}
    return {"bool": {"must": [existing_query], "filter": [extra_filter]}}


def _apply_date_window_to_body(
    req: DocsAnalyticsRequest,
    body: Dict[str, Any],
    *,
    date_field: Optional[str],
) -> Dict[str, Any]:
    """
    ✅ Safety: ensure ES DSL queries are windowed.
    - If req.start_date/end_date provided: enforce that range.
    - Else: apply default last ES_DSL_DEFAULT_DAYS days window (only if we know date_field).
    """
    if not date_field:
        return body

    range_body: Dict[str, Any] = {}
    if req.start_date:
        range_body["gte"] = req.start_date
    if req.end_date:
        range_body["lte"] = req.end_date

    if not range_body:
        today = datetime.now(timezone.utc).date()
        start = today - timedelta(days=ES_DSL_DEFAULT_DAYS)
        range_body = {"gte": start.isoformat(), "lte": today.isoformat()}

    date_filter = {"range": {date_field: range_body}}

    body = deepcopy(body or {})
    body["query"] = _wrap_query_with_filter(body.get("query"), date_filter)
    return body


def _sanitize_es_body_for_prod(
    req: DocsAnalyticsRequest,
    body: Dict[str, Any],
    *,
    date_field: Optional[str],
) -> Dict[str, Any]:
    """
    ✅ Safety: cap hit size, cap agg sizes, and enforce date window.
    """
    body = deepcopy(body or {})

    if isinstance(body.get("size"), int):
        body["size"] = min(body["size"], ES_DSL_MAX_SIZE)
    else:
        body.setdefault("size", min(10, ES_DSL_MAX_SIZE))

    _cap_agg_sizes(body)
    body = _apply_date_window_to_body(req, body, date_field=date_field)
    return body


def _maybe_prefer_keyword(
    resolved: Optional[str],
    lower_map: Dict[str, str],
    *,
    user_term: Optional[str],
    alias_family: Optional[str],
) -> Optional[str]:
    """
    If we resolved a field like 'customer_id' and 'customer_id.keyword' exists,
    prefer the keyword variant for terms/cardinality robustness.
    """
    if not resolved:
        return resolved

    u = (user_term or "").lower()
    af = (alias_family or "").lower()

    want_kw = (
        any(k in u for k in ["id", "name", "tag", "coupon", "location", "channel", "visit", "invoice"])
        or af in {"customer", "visit", "invoice", "location", "channel", "tag", "tags"}
    )

    if not want_kw:
        return resolved

    if resolved.endswith(".keyword"):
        return resolved

    kw_key = (resolved + ".keyword").lower()
    return lower_map.get(kw_key) or resolved


# -------------------------------------------------------------------
# ES mapping + date helpers (compat)
# -------------------------------------------------------------------

def resolve_es_field(
    mappings: Dict[str, Any],
    user_term: Optional[str] = None,
    alias_family: Optional[str] = None,
) -> Optional[str]:
    """
    Resolve an ES field path from index mappings.
    (Kept for backwards compatibility with any modules that still import it.)
    """
    props = mappings.get("properties", {}) or {}
    flat_fields: List[str] = []

    def _walk(prefix: str, node: Dict[str, Any]):
        if not isinstance(node, dict):
            return

        # ROOT CASE: some code may pass flattened "properties"
        if (
            prefix == ""
            and "type" not in node
            and "properties" not in node
            and "fields" not in node
        ):
            for sub_name, spec in node.items():
                path = sub_name if not prefix else f"{prefix}.{sub_name}"
                _walk(path, spec)
            return

        ftype = node.get("type")
        has_props = isinstance(node.get("properties"), dict)
        has_fields = isinstance(node.get("fields"), dict)

        if ftype in ("nested", "object") and has_props:
            for sub_name, spec in node["properties"].items():
                path = f"{prefix}.{sub_name}" if prefix else sub_name
                _walk(path, spec)

            if has_fields:
                for sub_name, spec in node["fields"].items():
                    path = f"{prefix}.{sub_name}"
                    _walk(path, spec)
            return

        if ftype or not (has_props or has_fields):
            if prefix:
                flat_fields.append(prefix)
            return

        if has_props:
            for sub_name, spec in node["properties"].items():
                path = f"{prefix}.{sub_name}" if prefix else sub_name
                _walk(path, spec)

        if has_fields:
            for sub_name, spec in node["fields"].items():
                path = f"{prefix}.{sub_name}"
                _walk(path, spec)

    _walk("", props)

    lower = {f.lower(): f for f in flat_fields}

    if not user_term:
        user_term = alias_family or "date"

    variants = [user_term, user_term.replace(" ", "_"), user_term.replace("_", " ")]
    for v in variants:
        key = v.lower()
        if key in lower:
            resolved = lower[key]
            return _maybe_prefer_keyword(resolved, lower, user_term=user_term, alias_family=alias_family)

    def _normalize_field_key(field_key: str) -> str:
        fk = field_key
        for suffix in (".keyword", ".raw", ".exact"):
            if fk.endswith(suffix):
                fk = fk[: -len(suffix)]
                break
        return fk

    if alias_family in ALIASES_UNIVERSAL:
        for c in ALIASES_UNIVERSAL[alias_family]:
            for v in [c, c.replace("_", " "), c.replace(" ", "_")]:
                key = v.lower()
                for field_key_lower, real in lower.items():
                    norm = _normalize_field_key(field_key_lower)
                    if norm == key or norm.endswith("." + key):
                        return _maybe_prefer_keyword(real, lower, user_term=user_term, alias_family=alias_family)

    return None


def _ms_to_dt(ms: Optional[float]) -> Optional[datetime]:
    if ms is None:
        return None
    try:
        return datetime.fromtimestamp(ms / 1000.0, tz=timezone.utc)
    except Exception:
        return None


def _parse_date_str(s: Optional[str]):
    if not s:
        return None
    s = s.strip()
    if s.endswith("Z"):
        s = s[:-1] + "+00:00"
    try:
        return datetime.fromisoformat(s).date()
    except Exception:
        return None


def _es_cannot_answer(
    insight: str,
    business_rules: Optional[str],
) -> Dict[str, Any]:
    return {
        "insight": to_json_safe(insight),
        "rows": [],
        "rules_used": business_rules or "",
        "engine": "es",
    }


def _build_date_range_filter(
    req: DocsAnalyticsRequest,
    date_field: str,
) -> List[Dict[str, Any]]:
    filters: List[Dict[str, Any]] = []
    range_body: Dict[str, Any] = {}

    if req.start_date:
        range_body["gte"] = req.start_date
    if req.end_date:
        range_body["lte"] = req.end_date

    if range_body:
        filters.append({"range": {date_field: range_body}})

    return filters


# -------------------------------------------------------------------
# ES special router: known questions → custom ES logic
# -------------------------------------------------------------------

def _route_es_special(
    req: DocsAnalyticsRequest,
    client,
    mappings: Dict[str, Any],
    business_rules: Optional[str],
):
    from app.api.metrics import (
        metrics_customer_value,
        metrics_promos_coupons,
        metrics_time_series,
        metrics_lifecycle_acquisition as m_acq,
        metrics_lifecycle_engagement as m_eng,
        metrics_lifecycle_retention as m_ret,
        metrics_lifecycle_segmentation as m_seg,
    )

    q_lower = (req.question or "").lower()

    # --- Customer Value
    if (
        "total visit amount" in q_lower
        or "total visit pieces" in q_lower
        or ("total visits" in q_lower and "unique customers" in q_lower)
        or "core visit metrics" in q_lower
        or "core 4 metrics" in q_lower
        or "core four metrics" in q_lower
    ):
        return metrics_customer_value._es_core_visit_metrics(req, client, mappings, business_rules)

    if "average customer lifetime value" in q_lower or "average clv" in q_lower:
        return metrics_customer_value._es_customer_ltv(req, client, mappings, business_rules)

    if (
        "average visits per customer" in q_lower
        or "avg visits per customer" in q_lower
        or "visit pieces per customer" in q_lower
        or "revenue per customer" in q_lower
        or "avg $ per piece" in q_lower
        or "average $ per piece" in q_lower
        or "average dollars per piece" in q_lower
    ):
        return metrics_customer_value._es_customer_value_metrics(req, client, mappings, business_rules)

    if "one-time" in q_lower and "repeat" in q_lower and "customer" in q_lower:
        return metrics_customer_value._es_one_time_vs_repeat(req, client, mappings, business_rules)

    # --- Promos / Coupons
    if "average pickup delay" in q_lower:
        return metrics_promos_coupons._es_avg_pickup_delay_retail(req, client, mappings, business_rules)

    if ("top 20%" in q_lower and ("redo" in q_lower or "courtesy" in q_lower) and "customer" in q_lower):
        return metrics_promos_coupons._es_top20_customers_with_redo_courtesy(req, client, mappings, business_rules)

    if "redo" in q_lower and "invoice" in q_lower:
        return metrics_promos_coupons._es_invoices_with_redo_items(req, client, mappings, business_rules)

    # --- Segmentation / Pricing / Route vs Retail / Tiers
    if (
        "route vs retail" in q_lower
        or "retail vs route" in q_lower
        or ("route" in q_lower and "retail" in q_lower and ("compare" in q_lower or "comparison" in q_lower))
        or ("route customers" in q_lower and "retail" in q_lower)
    ):
        return m_seg._es_route_vs_retail_comparison(req, client, mappings, business_rules)

    if (
        "high-value retail" in q_lower
        or "high value retail" in q_lower
        or ("retail" in q_lower and "targets" in q_lower)
        or ("route conversion" in q_lower and "retail" in q_lower)
        or ("high" in q_lower and "value" in q_lower and "retail" in q_lower and "targets" in q_lower)
    ):
        return m_seg._es_high_value_retail_targets(req, client, mappings, business_rules)

    # --- Retention / Churn
    if (
        "churn rate" in q_lower
        or ("churn" in q_lower and "rate" in q_lower)
        or ("churned" in q_lower and "customers" in q_lower)
    ):
        return m_ret._es_churn_rate(req, client, mappings, business_rules)

    # --- Engagement / Recency distribution
    if (
        "days since last visit" in q_lower
        or ("last visit" in q_lower and "distribution" in q_lower)
        or ("recency" in q_lower and "distribution" in q_lower)
        or ("0–30" in q_lower and "31–60" in q_lower and "91–180" in q_lower)
        or ("0-30" in q_lower and "31-60" in q_lower and "91-180" in q_lower)
    ):
        return m_eng._es_days_since_last_visit_distribution(req, client, mappings, business_rules)

    # --- lifecycle rate / interval / repeat metrics
    if (("active customer rate" in q_lower or ("active rate" in q_lower and "customer" in q_lower)) and "30" not in q_lower):
        return m_eng._es_active_customer_rate(req, client, mappings, business_rules)

    if (
        "30-day activity rate" in q_lower
        or "30 day activity rate" in q_lower
        or ("30" in q_lower and "activity rate" in q_lower)
        or ("activity rate" in q_lower and "30" in q_lower)
    ):
        return m_eng._es_30d_activity_rate(req, client, mappings, business_rules)

    if (
        "average visit interval" in q_lower
        or "avg visit interval" in q_lower
        or ("visit interval" in q_lower and ("average" in q_lower or "avg" in q_lower))
    ):
        return m_eng._es_avg_visit_interval(req, client, mappings, business_rules)

    if (
        "repeat customers 365" in q_lower
        or "repeat customer 365" in q_lower
        or ("repeat customers" in q_lower and "365" in q_lower)
        or ("repeat rate" in q_lower and "365" in q_lower)
    ):
        return m_eng._es_repeat_customers_365(req, client, mappings, business_rules)

    # --- Visit Frequency charts
    if (
        "visit frequency 365" in q_lower
        or "visit frequency – 365" in q_lower
        or "visit frequency - 365" in q_lower
        or ("visit frequency" in q_lower and "365" in q_lower)
        or ("distribution" in q_lower and "visits_365" in q_lower)
    ):
        return m_seg._es_visit_frequency_365(req, client, mappings, business_rules)

    if (
        "visit frequency 730" in q_lower
        or "visit frequency – 730" in q_lower
        or "visit frequency - 730" in q_lower
        or ("visit frequency" in q_lower and "730" in q_lower)
        or ("distribution" in q_lower and "visits_lifetime" in q_lower and ("730" in q_lower or "2 year" in q_lower))
    ):
        return m_seg._es_visit_frequency_730(req, client, mappings, business_rules)

    # --- Pareto + Single-Visit metrics
    if (
        "pareto" in q_lower
        or "80/20" in q_lower
        or ("80" in q_lower and "20" in q_lower and "rule" in q_lower)
        or ("percentage of customers" in q_lower and "80" in q_lower and "revenue" in q_lower)
    ):
        return m_seg._es_pareto_80_20(req, client, mappings, business_rules)

    if (
        "single visit lifetime" in q_lower
        or "single-visit lifetime" in q_lower
        or ("single visit" in q_lower and "lifetime" in q_lower)
        or ("one visit" in q_lower and "lifetime" in q_lower)
    ):
        return m_ret._es_single_visit_lifetime(req, client, mappings, business_rules)

    if (
        "single visit 365" in q_lower
        or "single-visit 365" in q_lower
        or ("single visit" in q_lower and "365" in q_lower)
        or ("one visit" in q_lower and "365" in q_lower)
        or ("single visit" in q_lower and "last year" in q_lower)
    ):
        return m_ret._es_single_visit_365(req, client, mappings, business_rules)

    # --- Acquisition / YoY / Cohort
    if (
        "daily acquisition rate" in q_lower
        or ("acquisition rate" in q_lower and "daily" in q_lower)
        or ("acquisition" in q_lower and "0–30" in q_lower and "30–60" in q_lower)
        or ("acquisition" in q_lower and "0-30" in q_lower and "30-60" in q_lower)
        or ("first_visit" in q_lower and "180" in q_lower and "days" in q_lower)
    ):
        return m_acq._es_daily_acquisition_rate_by_period_customers(req, client, mappings, business_rules)

    if (
        ("year-over-year" in q_lower or "year over year" in q_lower or "yoy" in q_lower)
        and ("new customers" in q_lower or ("new" in q_lower and "customers" in q_lower))
    ):
        return m_acq._es_yoy_new_customers_customers_index(req, client, mappings, business_rules)

    if (
        "return rate by cohort" in q_lower
        or "cohort return rate" in q_lower
        or ("return rate" in q_lower and "cohort" in q_lower)
        or ("cohort" in q_lower and "year" in q_lower and "return" in q_lower)
        or ("cohort" in q_lower and "year" in q_lower)
    ):
        return m_ret._es_return_rate_by_cohort_year_customers(req, client, mappings, business_rules)

    # --- Existing lifecycle/time-series routes
    if ("top 20%" in q_lower and "overdue" in q_lower and "customer" in q_lower):
        return m_eng._es_top20_customers_overdue_14d(req, client, mappings, business_rules)

    if (
        "active customers" in q_lower
        and "active customer rate" not in q_lower
        and "average days between visits" not in q_lower
        and "avg days between visits" not in q_lower
    ):
        return m_eng._es_active_customers(req, client, mappings, business_rules)

    if (
        ("retention rate" in q_lower or "customer retention" in q_lower)
        and ("730" in q_lower or "2 year" in q_lower or "two year" in q_lower)
        and ("180" in q_lower or "6 month" in q_lower or "six month" in q_lower)
    ):
        return m_ret._es_customer_retention_rate_730_180(req, client, mappings, business_rules)

    if ("retention" in q_lower) and ("730" in q_lower) and ("180" in q_lower):
        return m_ret._es_customer_retention_rate_730_180(req, client, mappings, business_rules)

    if "average days between visits" in q_lower and "active customers" in q_lower:
        return m_eng._es_avg_days_between_visits_active(req, client, mappings, business_rules)

    if "overdue for their next visit" in q_lower or "overdue for next visit" in q_lower:
        return m_ret._es_overdue_customers(req, client, mappings, business_rules)

    if "distribution of customers by visit frequency" in q_lower or (
        "visit frequency" in q_lower and "1, 2–5, 6–11, 12+" in q_lower
    ):
        return m_eng._es_visit_frequency_distribution(req, client, mappings, business_rules)

    if (
        ("top 5%" in q_lower and "top 20%" in q_lower and "revenue" in q_lower)
        or ("top 5 percent" in q_lower and "top 20 percent" in q_lower and "revenue" in q_lower)
        or ("which customers fall into the top 5%" in q_lower)
        or ("top 20%" in q_lower and "revenue" in q_lower)
        or ("top 20 percent" in q_lower and "revenue" in q_lower)
        or ("percentage of revenue comes from the top 20%" in q_lower)
        or ("percentage of revenue comes from the top 20 percent" in q_lower)
    ):
        return m_seg._es_top_customers_by_revenue(req, client, mappings, business_rules)

    if ("month-over-month" in q_lower or "month over month" in q_lower) and "visit" in q_lower:
        return metrics_time_series._es_month_over_month_visits(req, client, mappings, business_rules)

    if "seasonal patterns" in q_lower or ("seasonal" in q_lower and "last year" in q_lower):
        return metrics_time_series._es_seasonal_revenue_patterns(req, client, mappings, business_rules)

    if "average ticket size" in q_lower and (
        "day of week" in q_lower
        or "day-of-week" in q_lower
        or "dow" in q_lower
        or "month of year" in q_lower
        or "month" in q_lower
        or "day" in q_lower
    ):
        return metrics_time_series._es_avg_ticket_size(req, client, mappings, business_rules)

    if "new customer acquisition rate" in q_lower or ("new customer" in q_lower and "acquisition" in q_lower):
        return m_acq._es_new_customer_acquisition(req, client, mappings, business_rules)

    if (
        "customers achieving 2nd visit" in q_lower
        or "customers achieving second visit" in q_lower
        or "customers achieving 3rd visit" in q_lower
        or "customers achieving third visit" in q_lower
        or "customers achieving 4th visit" in q_lower
        or "customers achieving fourth visit" in q_lower
        or "customers achieving 5th visit" in q_lower
        or "customers achieving fifth visit" in q_lower
    ):
        return m_seg._es_customers_nth_visit(req, client, mappings, business_rules)

    if (
        ("year-over-year" in q_lower or "year over year" in q_lower or "yoy" in q_lower)
        and "revenue" in q_lower
        and "location" in q_lower
    ):
        return metrics_time_series._es_yoy_revenue_by_location(req, client, mappings, business_rules)

    return None


# -------------------------------------------------------------------
# ES path: question -> (special ES or DSL) -> ES -> rows
# -------------------------------------------------------------------

def _ask_via_es(req: DocsAnalyticsRequest):
    if not req.es_base_url or not req.es_index_name:
        raise HTTPException(
            status_code=400,
            detail="ES mode requires es_base_url and es_index_name on the request.",
        )

    client = make_es_client(req.es_base_url, req.es_username, req.es_password)

    if not client.ping():
        raise HTTPException(status_code=400, detail=f"Could not ping Elasticsearch at {req.es_base_url}")

    # ✅ MODIFIED: use shared_utilities (select concrete invoices index + mappings)
    chosen_index, mappings = _get_invoice_index_and_mappings(client, req.es_index_name)

    # optional business rules
    business_rules: Optional[str] = None
    if req.mode == "documents":
        workspace_id = (req.workspace_id or "default").strip() or "default"
        business_rules = get_business_rules(
            workspace_id=workspace_id,
            question=req.question,
            doc_ids=req.doc_ids,
        )
        if not (business_rules or "").strip():
            raise HTTPException(status_code=400, detail="No business rules found in documents for this question.")

    # ✅ Try ES-special handlers FIRST (no OpenAI key needed)
    req2 = req.model_copy(update={"es_index_name": chosen_index})
    special_resp = _route_es_special(req2, client, mappings, business_rules)
    if special_resp is not None:
        return special_resp

    # ✅ Only now require API key for LLM-generated DSL
    api_key = (req.api_key or "").strip() or os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise HTTPException(status_code=500, detail="No OpenAI API key configured for ES DSL mode.")

    dsl_text = llm_generate_es_query(
        question=req.question,
        index_name=chosen_index,
        mappings=mappings,
        model=req.model,
        api_key=api_key,
    )

    try:
        index_from_dsl, body = _parse_es_dsl(dsl_text)
    except Exception:
        insight = (
            "I tried to generate an Elasticsearch query for your question, "
            "but the generated DSL could not be parsed or was invalid. "
            "Please rephrase your question or be more specific."
        )
        return {
            "insight": to_json_safe(insight),
            "rows": [],
            "rules_used": business_rules or "",
            "engine": "es",
        }

    index_to_use = index_from_dsl or chosen_index

    # ✅ SAFETY: sanitize + enforce window on LLM DSL queries (DIRECT FIELD)
    date_field = INVOICE_FIELDS["date"] if _field_exists(mappings, INVOICE_FIELDS["date"]) else None
    body = _sanitize_es_body_for_prod(req, body, date_field=date_field)

    try:
        res = _safe_es_search(client, index=index_to_use, body=body)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error executing ES search: {e}")

    hits = res.get("hits", {}).get("hits", [])
    docs = [h.get("_source", {}) for h in hits]
    rows = _flatten_docs_to_rows(docs) if docs else []

    insight = f"Answer computed directly on Elasticsearch index '{index_to_use}' using an ES query."

    return {
        "insight": to_json_safe(insight),
        "rows": to_json_safe(rows),
        "rules_used": business_rules or "",
        "engine": "es",
    }


# -------------------------------------------------------------------
# Python path: current logic (mdf + llm.codegen + Pandas)
# -------------------------------------------------------------------

def _ask_via_python(req: DocsAnalyticsRequest):
    workspace_id = (req.workspace_id or "default").strip() or "default"

    tables = TABLE_STORE.get_tables(workspace_id)
    if not tables:
        raise HTTPException(status_code=400, detail="No tables loaded for this workspace_id.")

    if req.mode == "predefined":
        business_rules = None
    elif req.mode == "documents":
        business_rules = get_business_rules(
            workspace_id=workspace_id,
            question=req.question,
            doc_ids=req.doc_ids,
        )
        if not (business_rules or "").strip():
            raise HTTPException(status_code=400, detail="No business rules found in documents for this question.")
    else:
        raise HTTPException(status_code=400, detail="Invalid mode. Use 'predefined' or 'documents'.")

    api_key = (req.api_key or "").strip() or os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise HTTPException(
            status_code=500,
            detail="No OpenAI API key configured (set OPENAI_API_KEY on the server or pass api_key).",
        )

    code = llm_codegen(
        question=req.question,
        tables=tables,
        model=req.model,
        api_key=api_key,
        business_rules=business_rules,
    )
    if not code:
        raise HTTPException(status_code=500, detail="Failed to generate analytics code.")

    result_df, insight = run_generated_code(code, tables)
    safe_rows = to_json_safe(result_df)

    return {
        "insight": to_json_safe(insight),
        "rows": safe_rows,
        "rules_used": business_rules or "",
        "engine": "python",
    }


# -------------------------------------------------------------------
# ✅ Dashboard helpers (now supports: performance, ops, lifecycle, growth)
# -------------------------------------------------------------------

def _pct_change(cur: Optional[float], prev: Optional[float]) -> Optional[float]:
    if cur is None or prev in (None, 0):
        return None
    try:
        return (float(cur) - float(prev)) * 100.0 / float(prev)
    except Exception:
        return None


PERFORMANCE_LABEL_MAP: Dict[str, str] = {
    "total_visits": "Total Visits",
    "unique_customers": "Unique Customers",
    "total_revenue": "Total Visit Amount",
    "total_pieces": "Total Visit Pieces",
    "average_visits_per_customer": "Average Visits per Customer",
    "visit_pieces_per_customer": "Visit Pieces per Customer",
    "revenue_per_customer": "Revenue Per Customer",
    "avg_dollar_per_piece": "Avg $ per Piece",
    "new_customers_first_visit_in_period": "New Customers (First Visit)",
    "customers_2plus_visits": "Customers Achieving 2nd Visit",
    "customers_3plus_visits": "Customers Achieving 3rd Visit",
    "customers_4plus_visits": "Customers Achieving 4th Visit",
    "customers_5plus_visits": "Customers Achieving 5th Visit",
}

OPS_LABEL_MAP: Dict[str, str] = {
    "avg_pickup_delay_retail": "Average Pickup Delay (Retail)",
    "redo_invoices_count": "Invoices with Redo Items",
    "top20_customers_redo_issues": "Top 20% Customers – Redo Issues",
}

LIFECYCLE_LABEL_MAP: Dict[str, str] = {
    "active_customers": "Active Customers",
    "active_customer_rate": "Active Customer Rate (%)",
    "activity_rate_30d": "30-Day Activity Rate (%)",
    "churn_rate": "Churn Rate (%)",
    "customer_retention_rate_730_180": "Retention Rate (730 → 180)",
    "avg_visit_interval": "Average Visit Interval (days)",
    "repeat_customers_365": "Repeat Customers (365 days)",
    "single_visit_lifetime": "Single-Visit Customers (Lifetime) (%)",
    "single_visit_365": "Single-Visit Customers (365 days) (%)",
    "avg_days_between_visits_active": "Avg Days Between Visits (Active)",
    "overdue_customers": "Overdue Customers",
}

GROWTH_LABEL_MAP: Dict[str, str] = {
    "pareto_80_20": "Pareto 80/20 (Customer share for 80% revenue)",
}

LABEL_MAPS: Dict[str, Dict[str, str]] = {
    "performance": PERFORMANCE_LABEL_MAP,
    "ops": OPS_LABEL_MAP,
    "lifecycle": LIFECYCLE_LABEL_MAP,
    "growth": GROWTH_LABEL_MAP,
}


def _extract_scalar(resp: Any) -> Optional[float]:
    if not isinstance(resp, dict):
        return None

    rows = resp.get("rows")

    if isinstance(rows, (int, float)):
        return float(rows)

    if isinstance(rows, dict):
        for k in ("value", "count", "rate", "pct", "percentage"):
            v = rows.get(k)
            if isinstance(v, (int, float)):
                return float(v)
        nums = [v for v in rows.values() if isinstance(v, (int, float))]
        return float(nums[0]) if len(nums) == 1 else None

    if isinstance(rows, list) and rows:
        r0 = rows[0]
        if isinstance(r0, (int, float)):
            return float(r0)
        if isinstance(r0, dict):
            for k in ("value", "count", "rate", "pct", "percentage"):
                v = r0.get(k)
                if isinstance(v, (int, float)):
                    return float(v)
            nums = [v for v in r0.values() if isinstance(v, (int, float))]
            return float(nums[0]) if len(nums) == 1 else None

    return None

def _window_kpis_for_dashboard(
    dashboard_id: str,
    *,
    base_docs_req: DocsAnalyticsRequest,
    period: Period,
    client,
    mappings: Dict[str, Any],
) -> Dict[str, float]:
    from app.api.metrics import (
        metrics_customer_value,
        metrics_lifecycle_acquisition as m_acq,  # kept for parity
        metrics_lifecycle_engagement as m_eng,
        metrics_lifecycle_retention as m_ret,
        metrics_lifecycle_segmentation as m_seg,
    )

    # ------------------------------------------------------------
    # Performance / Ops KPIs
    # ------------------------------------------------------------
    if dashboard_id in ("performance", "ops"):
        vals = metrics_customer_value._window_customer_value_metrics(base_docs_req, period, client, mappings)

        out: Dict[str, float] = {}
        for k, v in (vals or {}).items():
            if isinstance(v, (int, float)):
                out[k] = float(v)

        # ✅ PERFORMANCE: override Nth-visit KPIs to be "in-period DISTINCT visit_id"
        if dashboard_id == "performance":
            reqp = base_docs_req.model_copy(update={"start_date": period.start_date, "end_date": period.end_date})

            # ✅ REPLACEMENT KPI: New Customers (First Visit in Window)
            # This KPI is computed from the CUSTOMERS index, not invoices.
            newc = metrics_customer_value._es_new_customers_first_visit_in_period_value_count(
                reqp, client, mappings, None
            )
            rows_newc = newc.get("rows") or {}
            if isinstance(rows_newc, dict) and isinstance(rows_newc.get("count"), (int, float)):
                # Use the KPI id you put in PERFORMANCE_LABEL_MAP
                out["new_customers_first_visit_in_period"] = float(rows_newc["count"])

            nth = m_seg._es_customers_nth_visit_in_period(reqp, client, mappings, None)

            mm: Dict[str, float] = {}
            for r in (nth.get("rows") or []):
                if (
                    isinstance(r, dict)
                    and isinstance(r.get("metric"), str)
                    and isinstance(r.get("value"), (int, float))
                ):
                    mm[r["metric"]] = float(r["value"])

            if "customers_2plus_visits_period" in mm:
                out["customers_2plus_visits"] = mm["customers_2plus_visits_period"]
            if "customers_3plus_visits_period" in mm:
                out["customers_3plus_visits"] = mm["customers_3plus_visits_period"]
            if "customers_4plus_visits_period" in mm:
                out["customers_4plus_visits"] = mm["customers_4plus_visits_period"]
            if "customers_5plus_visits_period" in mm:
                out["customers_5plus_visits"] = mm["customers_5plus_visits_period"]

        return out

    # ------------------------------------------------------------
    # Other dashboards (already windowed via reqp)
    # ------------------------------------------------------------
    reqp = base_docs_req.model_copy(update={"start_date": period.start_date, "end_date": period.end_date})
    out: Dict[str, float] = {}

    if dashboard_id == "lifecycle":
        from app.api.metrics import (
            metrics_lifecycle_engagement as m_eng,
            metrics_lifecycle_retention as m_ret,
        )
        candidates = {
            "active_customers": m_eng._es_active_customers(reqp, client, mappings, None),
            "active_customer_rate": m_eng._es_active_customer_rate(reqp, client, mappings, None),
            "activity_rate_30d": m_eng._es_30d_activity_rate(reqp, client, mappings, None),
            "churn_rate": m_ret._es_churn_rate(reqp, client, mappings, None),
            "customer_retention_rate_730_180": m_ret._es_customer_retention_rate_730_180(reqp, client, mappings, None),
            "avg_visit_interval": m_eng._es_avg_visit_interval(reqp, client, mappings, None),
            "repeat_customers_365": m_ret._es_repeat_customers_365(reqp, client, mappings, None),
            "single_visit_lifetime": m_ret._es_single_visit_lifetime(reqp, client, mappings, None),
            "single_visit_365": m_ret._es_single_visit_365(reqp, client, mappings, None),
            "avg_days_between_visits_active": m_eng._es_avg_days_between_visits_active(reqp, client, mappings, None),
            "overdue_customers": m_ret._es_overdue_customers(reqp, client, mappings, None),
        }
        for k, resp in candidates.items():
            v = _extract_scalar(resp)
            if isinstance(v, (int, float)):
                out[k] = float(v)

    elif dashboard_id == "growth":
        from app.api.metrics import metrics_lifecycle_segmentation as m_seg
        v = _extract_scalar(m_seg._es_pareto_80_20(reqp, client, mappings, None))
        if isinstance(v, (int, float)):
            out["pareto_80_20"] = float(v)

    return out



def _add_months(d: datetime.date, months: int) -> datetime.date:
    y = d.year + (d.month - 1 + months) // 12
    m = (d.month - 1 + months) % 12 + 1
    return datetime(y, m, 1).date()


def _normalize_dataset_period(p: Period, *, max_months: int) -> Period:
    end_d = _parse_date_str(p.end_date) or datetime.now(timezone.utc).date()
    end_iso = end_d.isoformat()

    end_month_start = end_d.replace(day=1)
    clamp_start = _add_months(end_month_start, -(max_months - 1))

    user_start = _parse_date_str(p.start_date)
    start_d = user_start if (user_start and user_start > clamp_start) else clamp_start

    return Period(start_date=start_d.isoformat(), end_date=end_iso)


def _build_datasets(
    dashboard_id: str,
    *,
    base_docs_req: DocsAnalyticsRequest,
    current_period: Period,
    client,
    mappings: Dict[str, Any],
) -> List[Dict[str, Any]]:
    from app.api.metrics import (
        metrics_time_series,
        metrics_promos_coupons,
        metrics_customer_value,
        metrics_lifecycle_acquisition as m_acq,
        metrics_lifecycle_engagement as m_eng,
        metrics_lifecycle_retention as m_ret,
        metrics_lifecycle_segmentation as m_seg,
    )

    req_current = base_docs_req.model_copy(update={"start_date": current_period.start_date, "end_date": current_period.end_date})

    datasets: List[Dict[str, Any]] = []

    if dashboard_id == "performance":
        mom = metrics_time_series._es_month_over_month_visits(req_current, client, mappings, None)
        datasets.append(DashboardDataset(
            id="month_over_month_visits",
            label="Month-over-Month Visits",
            rows=to_json_safe(mom.get("rows", [])),
            insight=str(mom.get("insight", "")),
            engine=str(mom.get("engine", "es")),
        ).model_dump())

        seasonal = metrics_time_series._es_seasonal_revenue_patterns(req_current, client, mappings, None)
        datasets.append(DashboardDataset(
            id="seasonal_revenue_patterns",
            label="Seasonal Revenue Patterns",
            rows=to_json_safe(seasonal.get("rows", [])),
            insight=str(seasonal.get("insight", "")),
            engine=str(seasonal.get("engine", "es")),
        ).model_dump())

        seg_req = req_current.model_copy(update={"repeat_basis": "365"})
        one_time = metrics_customer_value._es_one_time_vs_repeat(seg_req, client, mappings, None)
        datasets.append(DashboardDataset(
            id="one_time_vs_repeat",
            label="One-Time vs Repeat Customers",
            rows=to_json_safe(one_time.get("rows", [])),
            insight=str(one_time.get("insight", "")),
            engine=str(one_time.get("engine", "es")),
        ).model_dump())

    elif dashboard_id == "ops":
        top = metrics_promos_coupons._es_top20_customers_with_redo_courtesy(req_current, client, mappings, None)
        datasets.append(DashboardDataset(
            id="top20_customers_redo_issues",
            label="Top Customers – Redo/Courtesy Issues",
            rows=to_json_safe(top.get("rows", [])),
            insight=str(top.get("insight", "")),
            engine=str(top.get("engine", "es")),
        ).model_dump())

        redo = metrics_promos_coupons._es_invoices_with_redo_items(req_current, client, mappings, None)
        datasets.append(DashboardDataset(
            id="redo_invoices",
            label="Invoices with Redo Items",
            rows=to_json_safe(redo.get("rows", [])),
            insight=str(redo.get("insight", "")),
            engine=str(redo.get("engine", "es")),
        ).model_dump())

        delay = metrics_promos_coupons._es_avg_pickup_delay_retail(req_current, client, mappings, None)
        datasets.append(DashboardDataset(
            id="avg_pickup_delay_retail",
            label="Average Pickup Delay (Retail)",
            rows=to_json_safe(delay.get("rows", [])),
            insight=str(delay.get("insight", "")),
            engine=str(delay.get("engine", "es")),
        ).model_dump())

    elif dashboard_id == "lifecycle":
        dist = m_eng._es_days_since_last_visit_distribution(req_current, client, mappings, None)
        datasets.append(DashboardDataset(
            id="days_since_last_visit_distribution",
            label="Days Since Last Visit Distribution",
            rows=to_json_safe(dist.get("rows", [])),
            insight=str(dist.get("insight", "")),
            engine=str(dist.get("engine", "es")),
        ).model_dump())

        vf = m_seg._es_visit_frequency_distribution(req_current, client, mappings, None)
        datasets.append(DashboardDataset(
            id="visit_frequency_distribution",
            label="Visit Frequency Distribution",
            rows=to_json_safe(vf.get("rows", [])),
            insight=str(vf.get("insight", "")),
            engine=str(vf.get("engine", "es")),
        ).model_dump())

        vf365 = m_seg._es_visit_frequency_365(req_current, client, mappings, None)
        datasets.append(DashboardDataset(
            id="visit_frequency_365",
            label="Visit Frequency (365 Days)",
            rows=to_json_safe(vf365.get("rows", [])),
            insight=str(vf365.get("insight", "")),
            engine=str(vf365.get("engine", "es")),
        ).model_dump())

        vf730 = m_seg._es_visit_frequency_730(req_current, client, mappings, None)
        datasets.append(DashboardDataset(
            id="visit_frequency_730",
            label="Visit Frequency (730 Days)",
            rows=to_json_safe(vf730.get("rows", [])),
            insight=str(vf730.get("insight", "")),
            engine=str(vf730.get("engine", "es")),
        ).model_dump())

        nth = m_seg._es_customers_nth_visit(req_current, client, mappings, None)
        datasets.append(DashboardDataset(
            id="customers_nth_visit",
            label="Customers Achieving Nth Visit",
            rows=to_json_safe(nth.get("rows", [])),
            insight=str(nth.get("insight", "")),
            engine=str(nth.get("engine", "es")),
        ).model_dump())

        top_rev = m_seg._es_top_customers_by_revenue(req_current, client, mappings, None)
        datasets.append(DashboardDataset(
            id="top_customers_by_revenue",
            label="Top Customers by Revenue",
            rows=to_json_safe(top_rev.get("rows", [])),
            insight=str(top_rev.get("insight", "")),
            engine=str(top_rev.get("engine", "es")),
        ).model_dump())

    elif dashboard_id == "growth":
        acq = m_acq._es_daily_acquisition_rate_by_period_customers(req_current, client, mappings, None)
        datasets.append(DashboardDataset(
            id="daily_acquisition_rate_by_period",
            label="Daily Acquisition Rate (by Period)",
            rows=to_json_safe(acq.get("rows", [])),
            insight=str(acq.get("insight", "")),
            engine=str(acq.get("engine", "es")),
        ).model_dump())

        yoy = m_acq._es_yoy_new_customers_customers_index(req_current, client, mappings, None)
        datasets.append(DashboardDataset(
            id="yoy_new_customers",
            label="YoY New Customers",
            rows=to_json_safe(yoy.get("rows", [])),
            insight=str(yoy.get("insight", "")),
            engine=str(yoy.get("engine", "es")),
        ).model_dump())

        cohort = m_ret._es_return_rate_by_cohort_year_customers(req_current, client, mappings, None)
        datasets.append(DashboardDataset(
            id="return_rate_by_cohort_year",
            label="Return Rate by Cohort Year",
            rows=to_json_safe(cohort.get("rows", [])),
            insight=str(cohort.get("insight", "")),
            engine=str(cohort.get("engine", "es")),
        ).model_dump())

        rv = m_seg._es_route_vs_retail_comparison(req_current, client, mappings, None)
        datasets.append(DashboardDataset(
            id="route_vs_retail_comparison",
            label="Route vs Retail Comparison",
            rows=to_json_safe(rv.get("rows", [])),
            insight=str(rv.get("insight", "")),
            engine=str(rv.get("engine", "es")),
        ).model_dump())

        targets = m_seg._es_high_value_retail_targets(req_current, client, mappings, None)
        datasets.append(DashboardDataset(
            id="high_value_retail_targets",
            label="High-Value Retail Targets",
            rows=to_json_safe(targets.get("rows", [])),
            insight=str(targets.get("insight", "")),
            engine=str(targets.get("engine", "es")),
        ).model_dump())

    return datasets


@router.post("/dashboard")
def es_dashboard(req: MetricsDashboardRequest):
    if not req.es_base_url or not req.es_index_name:
        raise HTTPException(status_code=400, detail="es_base_url and es_index_name are required")

    dashboard_id = (req.dashboard_id or "performance").strip().lower()
    if dashboard_id not in ("performance", "ops", "lifecycle", "growth"):
        raise HTTPException(
            status_code=400,
            detail="dashboard_id must be 'performance', 'ops', 'lifecycle', or 'growth'",
        )

    client = make_es_client(req.es_base_url, req.es_username, req.es_password)
    if not client.ping():
        raise HTTPException(status_code=400, detail=f"Could not ping Elasticsearch at {req.es_base_url}")

    # ✅ MODIFIED: use shared_utilities (select concrete invoices index + mappings)
    chosen_index, mappings = _get_invoice_index_and_mappings(client, req.es_index_name)

    base_docs_req = DocsAnalyticsRequest(
        workspace_id=req.workspace_id or "default",
        question="dashboard metrics",
        es_base_url=req.es_base_url,
        es_username=req.es_username,
        es_password=req.es_password,
        es_index_name=chosen_index,  # ✅ concrete index
        es_customers_index_name=req.es_customers_index_name,
        es_customer_stats_index_name=req.es_customer_stats_index_name,
    )

    current_vals = _window_kpis_for_dashboard(
        dashboard_id,
        base_docs_req=base_docs_req,
        period=req.current,
        client=client,
        mappings=mappings,
    )

    previous_vals: Dict[str, float] = {}
    if req.previous:
        previous_vals = _window_kpis_for_dashboard(
            dashboard_id,
            base_docs_req=base_docs_req,
            period=req.previous,
            client=client,
            mappings=mappings,
        )

    label_map = LABEL_MAPS.get(dashboard_id) or {}

    metrics: List[MetricsDashboardMetric] = []
    for metric_id, label in label_map.items():
        cur = current_vals.get(metric_id)
        prev = previous_vals.get(metric_id) if req.previous else None

        metrics.append(
            MetricsDashboardMetric(
                id=metric_id,
                label=label,
                current=float(cur) if cur is not None else None,
                previous=float(prev) if prev is not None else None,
                change_pct=_pct_change(cur, prev),
            )
        )

    max_months = int(getattr(req, "datasets_max_months", 3) or 3)
    max_months = min(max(max_months, 1), 3)

    ds_period = getattr(req, "datasets_period", None) or req.current
    ds_period = _normalize_dataset_period(ds_period, max_months=max_months)

    datasets = _build_datasets(
        dashboard_id,
        base_docs_req=base_docs_req,
        current_period=ds_period,
        client=client,
        mappings=mappings,
    )

    return {
        "dashboard_id": dashboard_id,
        "current_period": {"start_date": req.current.start_date, "end_date": req.current.end_date},
        "previous_period": (
            {"start_date": req.previous.start_date, "end_date": req.previous.end_date} if req.previous else None
        ),
        "datasets_period": {"start_date": ds_period.start_date, "end_date": ds_period.end_date},
        "datasets_max_months": max_months,
        "metrics": [m.model_dump() for m in metrics],
        "datasets": datasets,
    }


@router.post("/dashboard/performance")
def es_dashboard_performance(req: MetricsDashboardRequest):
    req2 = req.model_copy(update={"dashboard_id": "performance"})
    return es_dashboard(req2)


@router.post("/dashboard/ops")
def es_dashboard_ops(req: MetricsDashboardRequest):
    req2 = req.model_copy(update={"dashboard_id": "ops"})
    return es_dashboard(req2)


@router.post("/dashboard/lifecycle")
def es_dashboard_lifecycle(req: MetricsDashboardRequest):
    req2 = req.model_copy(update={"dashboard_id": "lifecycle"})
    return es_dashboard(req2)


@router.post("/dashboard/growth")
def es_dashboard_growth(req: MetricsDashboardRequest):
    req2 = req.model_copy(update={"dashboard_id": "growth"})
    return es_dashboard(req2)


# -------------------------------------------------------------------
# Analytics endpoint (router)
# -------------------------------------------------------------------

@router.post("/ask-analytics")
def ask_docs_analytics(req: DocsAnalyticsRequest):
    """
    Router:
      - If ES connection info is present (es_base_url + es_index_name),
        ALWAYS go through the ES engine.
      - If ES is NOT configured, fall back to Python engine.
    """
    if req.es_base_url and req.es_index_name:
        return _ask_via_es(req)

    return _ask_via_python(req)
