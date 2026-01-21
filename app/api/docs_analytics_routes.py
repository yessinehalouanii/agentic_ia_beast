from __future__ import annotations

import os
from datetime import datetime, timezone, timedelta
from typing import List, Optional, Dict, Any, Iterable
from copy import deepcopy

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from helpers.analytics_helpers import ALIASES_UNIVERSAL
from abi.docs_rules import get_business_rules
from abi.llm import llm_codegen
from abi.runtime import run_generated_code, to_json_safe
from app.core.table_store import TABLE_STORE

# 🔹 ES helpers
from abi.es_llm import llm_generate_es_query
from app.core.es_dynamic import make_es_client
from routes.es_test import (
    _extract_properties_from_mapping,
    _parse_es_dsl,
    _flatten_docs_to_rows,
)

router = APIRouter(prefix="/docs", tags=["Docs Analytics"])


# -------------------------------------------------------------------
# ✅ SAFETY KNOBS (prod-safe defaults; override via env)
# -------------------------------------------------------------------

ES_MAX_CUSTOMERS_DEFAULT = int(os.getenv("ES_MAX_CUSTOMERS", "10000"))  # ✅ lowered from 50k
ES_COMPOSITE_MAX_PAGES = int(os.getenv("ES_COMPOSITE_MAX_PAGES", "200"))
ES_COMPOSITE_MAX_BUCKETS = int(os.getenv("ES_COMPOSITE_MAX_BUCKETS", "200000"))

ES_DSL_MAX_SIZE = int(os.getenv("ES_DSL_MAX_SIZE", "500"))            # cap hits
ES_DSL_DEFAULT_DAYS = int(os.getenv("ES_DSL_DEFAULT_DAYS", "365"))     # default window
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


def _field_exists(mappings: Dict[str, Any], field: str) -> bool:
    """
    Works with:
      - flat dict from _extract_properties_from_mapping (keys are field paths), OR
      - nested ES mapping style dict.
    """
    props = (mappings or {}).get("properties") or {}
    if not isinstance(props, dict) or not field:
        return False

    # flat case
    if field in props:
        return True

    # nested traversal case for "a.b.c"
    node: Any = props
    parts = field.split(".")
    for part in parts:
        if not isinstance(node, dict):
            return False

        # typical nested mapping: node[part] is field spec
        if part in node:
            spec = node[part]
        # sometimes node is a spec with "properties"
        elif "properties" in node and isinstance(node["properties"], dict) and part in node["properties"]:
            spec = node["properties"][part]
        else:
            return False

        node = spec

    return True


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


class MetricsDashboardMetric(BaseModel):
    id: str
    label: str
    current: Optional[float]
    previous: Optional[float]
    change_pct: Optional[float]
    es_customer_stats_index_name: Optional[str] = None


# -------------------------------------------------------------------
# ES-safe helpers (timeouts + composite pagination)
# -------------------------------------------------------------------

def _safe_es_search(client, *, index: str, body: Dict[str, Any]) -> Dict[str, Any]:
    """
    ES search wrapper with conservative timeouts (prod-safe).
    """
    body = dict(body or {})
    body.setdefault("timeout", "10s")
    body.setdefault("track_total_hits", False)
    return client.search(index=index, body=body, request_timeout=20)


def _composite_terms(
    client,
    *,
    index: str,
    query_filters: List[Dict[str, Any]],
    field: str,
    sub_aggs: Dict[str, Any],
    page_size: int = 1000,
    agg_name: str = "groups",
    source_key: str = "k",
    # ✅ SAFETY CAPS
    max_pages: int = ES_COMPOSITE_MAX_PAGES,
    max_buckets: int = ES_COMPOSITE_MAX_BUCKETS,
    state: Optional[Dict[str, Any]] = None,
) -> Iterable[Dict[str, Any]]:
    """
    Page through ALL unique terms using composite aggregation, WITH HARD CAPS.
    Yields each bucket.

    state (optional) will be populated with:
      - truncated: bool
      - pages: int
      - buckets: int
    """
    after_key = None
    pages = 0
    buckets_yielded = 0
    if state is not None:
        state.setdefault("truncated", False)
        state.setdefault("pages", 0)
        state.setdefault("buckets", 0)

    # ✅ cap composite page size (even if caller passes bigger)
    page_size = min(int(page_size or 1000), ES_AGG_MAX_COMPOSITE_SIZE)

    while True:
        if pages >= max_pages:
            if state is not None:
                state["truncated"] = True
                state["pages"] = pages
                state["buckets"] = buckets_yielded
            break

        comp: Dict[str, Any] = {
            "size": page_size,
            "sources": [{source_key: {"terms": {"field": field}}}],
        }
        if after_key:
            comp["after"] = after_key

        body = {
            "size": 0,
            "query": {"bool": {"filter": query_filters or []}},
            "aggs": {
                agg_name: {
                    "composite": comp,
                    "aggs": sub_aggs,
                }
            },
        }

        res = _safe_es_search(client, index=index, body=body)
        pages += 1

        agg = (res.get("aggregations") or {}).get(agg_name) or {}
        buckets = agg.get("buckets") or []
        if not buckets:
            break

        for b in buckets:
            if buckets_yielded >= max_buckets:
                if state is not None:
                    state["truncated"] = True
                    state["pages"] = pages
                    state["buckets"] = buckets_yielded
                return
            buckets_yielded += 1
            yield b

        after_key = agg.get("after_key")
        if not after_key:
            break

    if state is not None:
        state["pages"] = pages
        state["buckets"] = buckets_yielded


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
        # default rolling window
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

    # cap hits size
    if isinstance(body.get("size"), int):
        body["size"] = min(body["size"], ES_DSL_MAX_SIZE)
    else:
        body.setdefault("size", min(10, ES_DSL_MAX_SIZE))

    # cap agg sizes recursively
    _cap_agg_sizes(body)

    # enforce window
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
# ES mapping + date helpers (shared by metric modules)
# (kept for compatibility; your updated metrics can ignore it)
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

        # ROOT CASE: _extract_properties_from_mapping already produced a flat dict
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

        # "nested" and "object" are containers — keep walking into their properties
        if ftype in ("nested", "object") and has_props:
            for sub_name, spec in node["properties"].items():
                path = f"{prefix}.{sub_name}" if prefix else sub_name
                _walk(path, spec)

            if has_fields:
                for sub_name, spec in node["fields"].items():
                    path = f"{prefix}.{sub_name}"
                    _walk(path, spec)
            return

        # leaf field OR non-container without properties/fields
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

    # 1) direct match
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

    # 2) alias family matching
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


def _select_invoice_index_from_es_mapping(
    client,
    raw_index_name: str,
) -> tuple[str, Dict[str, Any]]:
    raw_index_name = (raw_index_name or "").strip()
    if not raw_index_name:
        raise HTTPException(status_code=400, detail="No es_index_name provided for invoice metrics.")

    try:
        full_mapping = client.indices.get_mapping(index=raw_index_name)
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Could not fetch Elasticsearch mappings for '{raw_index_name}': {e}",
        )

    index_names = sorted(full_mapping.keys())
    if not index_names:
        raise HTTPException(status_code=400, detail=f"No indices found for pattern '{raw_index_name}'.")

    if len(index_names) == 1:
        chosen = index_names[0]
        return chosen, {chosen: full_mapping[chosen]}

    invoice_like = [name for name in index_names if "invoice" in name.lower() or "invoices" in name.lower()]
    if invoice_like:
        invoice_like.sort()
        chosen = invoice_like[0]
    else:
        chosen = index_names[0]

    return chosen, {chosen: full_mapping[chosen]}


# -------------------------------------------------------------------
# ✅ UPDATED: Direct-field versions (no resolve_es_field)
# -------------------------------------------------------------------

def _es_get_customer_signups(
    client,
    index_name: str,
    mappings: Dict[str, Any],
) -> Dict[Any, datetime.date]:
    """
    From the customers index, return:
        { customer_id -> original_signup_date }

    ✅ Uses COMPOSITE pagination (no 10k cap), with HARD CAPS + lower default max.
    ✅ Uses direct fields from CUSTOMER_FIELDS.
    """
    customer_field = CUSTOMER_FIELDS["customer_id"]
    signup_field = CUSTOMER_FIELDS["signup"]

    if not (_field_exists(mappings, customer_field) and _field_exists(mappings, signup_field)):
        return {}

    signups: Dict[Any, datetime.date] = {}

    max_customers = int(os.getenv("ES_MAX_CUSTOMERS", str(ES_MAX_CUSTOMERS_DEFAULT)))
    seen = 0

    state: Dict[str, Any] = {}
    for b in _composite_terms(
        client,
        index=index_name,
        query_filters=[],
        field=customer_field,
        sub_aggs={"signup": {"min": {"field": signup_field}}},
        page_size=1000,
        agg_name="customers",
        source_key="cid",
        state=state,
    ):
        cid = (b.get("key") or {}).get("cid")
        ms = (b.get("signup") or {}).get("value")
        dt = _ms_to_dt(ms)
        if cid is None or not dt:
            continue

        signups[cid] = dt.date()
        seen += 1
        if seen >= max_customers:
            break

    return signups


def _es_get_customer_stats(
    client,
    index_name: str,
    mappings: Dict[str, Any],
) -> Optional[List[Dict[str, Any]]]:
    """
    Aggregate per-customer stats in ES.

    ✅ Uses COMPOSITE pagination (no 10k cap), with HARD CAPS + lower default max.
    ✅ Uses direct fields from INVOICE_FIELDS.
    """
    customer_field = INVOICE_FIELDS["customer_id"]
    date_field = INVOICE_FIELDS["date"]
    amount_field = INVOICE_FIELDS["amount"]
    visit_field = INVOICE_FIELDS["visit_id"]
    pieces_field = INVOICE_FIELDS["pieces"]

    if not (_field_exists(mappings, customer_field) and _field_exists(mappings, date_field)):
        return None

    has_visit_field = _field_exists(mappings, visit_field)
    has_amount_field = _field_exists(mappings, amount_field)
    has_pieces_field = _field_exists(mappings, pieces_field)

    if has_visit_field:
        per_customer_aggs: Dict[str, Any] = {
            "first_visit": {"min": {"field": date_field}},
            "last_visit": {"max": {"field": date_field}},
            "visit_count": {"cardinality": {"field": visit_field}},
            "visits_365": {
                "filter": {"range": {date_field: {"gte": "now-365d/d"}}},
                "aggs": {"visits_365_distinct": {"cardinality": {"field": visit_field}}},
            },
        }
    else:
        per_customer_aggs = {
            "first_visit": {"min": {"field": date_field}},
            "last_visit": {"max": {"field": date_field}},
            "visit_count": {"value_count": {"field": date_field}},
            "visits_365": {"filter": {"range": {date_field: {"gte": "now-365d/d"}}}},
        }

    if has_amount_field:
        per_customer_aggs["total_revenue"] = {"sum": {"field": amount_field}}
    if has_pieces_field:
        per_customer_aggs["total_pieces"] = {"sum": {"field": pieces_field}}

    stats: List[Dict[str, Any]] = []

    max_customers = int(os.getenv("ES_MAX_CUSTOMERS", str(ES_MAX_CUSTOMERS_DEFAULT)))
    seen = 0

    state: Dict[str, Any] = {}
    for b in _composite_terms(
        client,
        index=index_name,
        query_filters=[],
        field=customer_field,
        sub_aggs=per_customer_aggs,
        page_size=1000,
        agg_name="customers",
        source_key="cid",
        state=state,
    ):
        cid = (b.get("key") or {}).get("cid")

        fv = (b.get("first_visit") or {}).get("value")
        lv = (b.get("last_visit") or {}).get("value")

        if has_visit_field:
            vc = int(((b.get("visit_count") or {}).get("value")) or 0)
            v365 = int(
                ((((b.get("visits_365") or {}).get("visits_365_distinct") or {}).get("value")) or 0)
            )
        else:
            vc = int(((b.get("visit_count") or {}).get("value")) or 0)
            v365 = int(((b.get("visits_365") or {}).get("doc_count")) or 0)

        tr = None
        if has_amount_field and "total_revenue" in b:
            tr_val = (b.get("total_revenue") or {}).get("value")
            tr = float(tr_val) if tr_val is not None else 0.0

        tp = None
        if has_pieces_field and "total_pieces" in b:
            tp_val = (b.get("total_pieces") or {}).get("value")
            tp = float(tp_val) if tp_val is not None else 0.0

        stats.append(
            {
                "customer_id": cid,
                "first_visit": _ms_to_dt(fv),
                "last_visit": _ms_to_dt(lv),
                "visit_count": vc,
                "visits_365": v365,
                "total_revenue": tr,
                "total_pieces": tp,
            }
        )

        seen += 1
        if seen >= max_customers:
            break

    return stats


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
        metrics_lifecycle,
        metrics_promos_coupons,
        metrics_time_series,
    )

    q_lower = (req.question or "").lower()

    # ------------------------------------------------------------
    # Customer Value (existing)
    # ------------------------------------------------------------
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

    # ------------------------------------------------------------
    # Promos / Coupons (existing)
    # ------------------------------------------------------------
    if "average pickup delay" in q_lower:
        return metrics_promos_coupons._es_avg_pickup_delay_retail(req, client, mappings, business_rules)

    if ("top 20%" in q_lower and ("redo" in q_lower or "courtesy" in q_lower) and "customer" in q_lower):
        return metrics_promos_coupons._es_top20_customers_with_redo_courtesy(req, client, mappings, business_rules)

    if "redo" in q_lower and "invoice" in q_lower:
        return metrics_promos_coupons._es_invoices_with_redo_items(req, client, mappings, business_rules)

    if (
        "coupon" in q_lower
        and ("return" in q_lower or "returns" in q_lower)
        and "365" in q_lower
        and "signup" in q_lower
    ):
        return metrics_promos_coupons._es_coupon_returns_365d_since_signup(req, client, mappings, business_rules)

    # ============================================================
    # ✅ NEW: Segmentation / Pricing / Route vs Retail / Tiers
    # ============================================================

    # Customer Value Tiers
    if (
        "customer value tiers" in q_lower
        or ("value tiers" in q_lower and "customer" in q_lower)
        or ("tier 1" in q_lower and "tier 2" in q_lower and "tier 3" in q_lower)
        or ("top 5%" in q_lower and "next 15%" in q_lower and "tier" in q_lower)
        or ("tiers" in q_lower and "p80" in q_lower and "p95" in q_lower)
    ):
        return metrics_lifecycle._es_customer_value_tiers(req, client, mappings, business_rules)

    # Route vs Retail Comparison
    if (
        "route vs retail" in q_lower
        or "retail vs route" in q_lower
        or ("route" in q_lower and "retail" in q_lower and ("compare" in q_lower or "comparison" in q_lower))
        or ("route customers" in q_lower and "retail" in q_lower)
    ):
        return metrics_lifecycle._es_route_vs_retail_comparison(req, client, mappings, business_rules)

    # Price Segments by Average Visit Value
    if (
        "price segments" in q_lower
        or "pricing segments" in q_lower
        or ("average visit value" in q_lower and ("segments" in q_lower or "segment" in q_lower))
        or ("visit_average_sales" in q_lower and ("segments" in q_lower or "segment" in q_lower))
        or ("under $25" in q_lower and "over $75" in q_lower)
    ):
        return metrics_lifecycle._es_price_segments_by_avg_visit_value(req, client, mappings, business_rules)

    # ✅ NEW: High-Value Retail Targets (moved here)
    if (
        "high-value retail" in q_lower
        or "high value retail" in q_lower
        or ("retail" in q_lower and "targets" in q_lower)
        or ("route conversion" in q_lower and "retail" in q_lower)
        or ("high" in q_lower and "value" in q_lower and "retail" in q_lower and "targets" in q_lower)
    ):
        return metrics_lifecycle._es_high_value_retail_targets(req, client, mappings, business_rules)

    # ============================================================
    # ✅ NEW: Churn / Recency distribution
    # ============================================================

    # Churn Rate
    if (
        "churn rate" in q_lower
        or ("churn" in q_lower and "rate" in q_lower)
        or ("churned" in q_lower and "customers" in q_lower)
    ):
        return metrics_lifecycle._es_churn_rate(req, client, mappings, business_rules)

    # Days Since Last Visit Distribution
    if (
        "days since last visit" in q_lower
        or ("last visit" in q_lower and "distribution" in q_lower)
        or ("recency" in q_lower and "distribution" in q_lower)
        or ("0–30" in q_lower and "31–60" in q_lower and "91–180" in q_lower)
        or ("0-30" in q_lower and "31-60" in q_lower and "91-180" in q_lower)
    ):
        return metrics_lifecycle._es_days_since_last_visit_distribution(req, client, mappings, business_rules)

    # ============================================================
    # ✅ NEW: lifecycle rate / interval / repeat metrics
    # ============================================================

    # Active Customer Rate (%)
    if (
        ("active customer rate" in q_lower or ("active rate" in q_lower and "customer" in q_lower))
        and "30" not in q_lower  # avoid confusion with 30d activity rate
    ):
        return metrics_lifecycle._es_active_customer_rate(req, client, mappings, business_rules)

    # 30-Day Activity Rate (%)
    if (
        "30-day activity rate" in q_lower
        or "30 day activity rate" in q_lower
        or ("30" in q_lower and "activity rate" in q_lower)
        or ("activity rate" in q_lower and "30" in q_lower)
    ):
        return metrics_lifecycle._es_30d_activity_rate(req, client, mappings, business_rules)

    # Average Visit Interval (days)
    if (
        "average visit interval" in q_lower
        or "avg visit interval" in q_lower
        or ("visit interval" in q_lower and ("average" in q_lower or "avg" in q_lower))
    ):
        return metrics_lifecycle._es_avg_visit_interval(req, client, mappings, business_rules)

    # Repeat Customers (365 days)
    if (
        "repeat customers 365" in q_lower
        or "repeat customer 365" in q_lower
        or ("repeat customers" in q_lower and "365" in q_lower)
        or ("repeat rate" in q_lower and "365" in q_lower)
    ):
        return metrics_lifecycle._es_repeat_customers_365(req, client, mappings, business_rules)

    # ============================================================
    # ✅ NEW: Visit Frequency charts
    # ============================================================

    # Visit Frequency – 365 Days
    if (
        "visit frequency 365" in q_lower
        or "visit frequency – 365" in q_lower
        or "visit frequency - 365" in q_lower
        or ("visit frequency" in q_lower and "365" in q_lower)
        or ("distribution" in q_lower and "visits_365" in q_lower)
    ):
        return metrics_lifecycle._es_visit_frequency_365(req, client, mappings, business_rules)

    # Visit Frequency – 730 Days
    if (
        "visit frequency 730" in q_lower
        or "visit frequency – 730" in q_lower
        or "visit frequency - 730" in q_lower
        or ("visit frequency" in q_lower and "730" in q_lower)
        or ("distribution" in q_lower and "visits_lifetime" in q_lower and ("730" in q_lower or "2 year" in q_lower))
    ):
        return metrics_lifecycle._es_visit_frequency_730(req, client, mappings, business_rules)

    # ============================================================
    # ✅ NEW: Pareto + Single-Visit metrics
    # ============================================================

    # Pareto 80/20
    if (
        "pareto" in q_lower
        or "80/20" in q_lower
        or ("80" in q_lower and "20" in q_lower and "rule" in q_lower)
        or ("percentage of customers" in q_lower and "80" in q_lower and "revenue" in q_lower)
    ):
        return metrics_lifecycle._es_pareto_80_20(req, client, mappings, business_rules)

    # Single Visit (Lifetime) %
    if (
        "single visit lifetime" in q_lower
        or "single-visit lifetime" in q_lower
        or ("single visit" in q_lower and "lifetime" in q_lower)
        or ("one visit" in q_lower and "lifetime" in q_lower)
    ):
        return metrics_lifecycle._es_single_visit_lifetime(req, client, mappings, business_rules)

    # Single Visit (365 days) %
    if (
        "single visit 365" in q_lower
        or "single-visit 365" in q_lower
        or ("single visit" in q_lower and "365" in q_lower)
        or ("one visit" in q_lower and "365" in q_lower)
        or ("single visit" in q_lower and "last year" in q_lower)
    ):
        return metrics_lifecycle._es_single_visit_365(req, client, mappings, business_rules)

    # ============================================================
    # ✅ NEW: Acquisition / YoY / Cohort (customers index)
    # ============================================================

    # Daily Acquisition Rate by Period (last 180 days)
    if (
        "daily acquisition rate" in q_lower
        or ("acquisition rate" in q_lower and "daily" in q_lower)
        or ("acquisition" in q_lower and "0–30" in q_lower and "30–60" in q_lower)
        or ("acquisition" in q_lower and "0-30" in q_lower and "30-60" in q_lower)
        or ("first_visit" in q_lower and "180" in q_lower and "days" in q_lower)
    ):
        return metrics_lifecycle._es_daily_acquisition_rate_by_period_customers(
            req, client, mappings, business_rules
        )

    # YoY New Customers (customers index)
    if (
        ("year-over-year" in q_lower or "year over year" in q_lower or "yoy" in q_lower)
        and ("new customers" in q_lower or ("new" in q_lower and "customers" in q_lower))
    ):
        return metrics_lifecycle._es_yoy_new_customers_customers_index(req, client, mappings, business_rules)

    # ✅ NEW: Return Rate by Cohort Year (moved here + stronger triggers)
    if (
        "return rate by cohort" in q_lower
        or "cohort return rate" in q_lower
        or ("return rate" in q_lower and "cohort" in q_lower)
        or ("cohort" in q_lower and "year" in q_lower and "return" in q_lower)
        or ("cohort" in q_lower and "year" in q_lower)
    ):
        return metrics_lifecycle._es_return_rate_by_cohort_year_customers(
            req, client, mappings, business_rules
        )

    # ============================================================
    # Existing lifecycle/time-series routes (existing)
    # ============================================================

    if ("top 20%" in q_lower and "overdue" in q_lower and "customer" in q_lower):
        return metrics_lifecycle._es_top20_customers_overdue_14d(req, client, mappings, business_rules)

    # Active Customers (count) — keep this AFTER "active customer rate" rule
    if (
        "active customers" in q_lower
        and "active customer rate" not in q_lower
        and "average days between visits" not in q_lower
        and "avg days between visits" not in q_lower
    ):
        return metrics_lifecycle._es_active_customers(req, client, mappings, business_rules)

    if (
        ("retention rate" in q_lower or "customer retention" in q_lower)
        and ("730" in q_lower or "2 year" in q_lower or "two year" in q_lower)
        and ("180" in q_lower or "6 month" in q_lower or "six month" in q_lower)
    ):
        return metrics_lifecycle._es_customer_retention_rate_730_180(req, client, mappings, business_rules)

    if ("retention" in q_lower) and ("730" in q_lower) and ("180" in q_lower):
        return metrics_lifecycle._es_customer_retention_rate_730_180(req, client, mappings, business_rules)

    if (
        "average customer lifespan" in q_lower
        or "avg customer lifespan" in q_lower
        or ("customer lifespan" in q_lower and ("average" in q_lower or "avg" in q_lower))
    ):
        return metrics_lifecycle._es_avg_customer_lifespan(req, client, mappings, business_rules)

    if "average days between visits" in q_lower and "active customers" in q_lower:
        return metrics_lifecycle._es_avg_days_between_visits_active(req, client, mappings, business_rules)

    if "overdue for their next visit" in q_lower or "overdue for next visit" in q_lower:
        return metrics_lifecycle._es_overdue_customers(req, client, mappings, business_rules)

    if "lapsed customers" in q_lower or (">180 days" in q_lower and "last visit" in q_lower):
        return metrics_lifecycle._es_lapsed_customers(req, client, mappings, business_rules)

    # Keep generic visit-frequency handler AFTER the explicit 365/730 ones
    if "distribution of customers by visit frequency" in q_lower or (
        "visit frequency" in q_lower and "1, 2–5, 6–11, 12+" in q_lower
    ):
        return metrics_lifecycle._es_visit_frequency_distribution(req, client, mappings, business_rules)

    # Keep top-customers-by-revenue AFTER Pareto
    if (
        ("top 5%" in q_lower and "top 20%" in q_lower and "revenue" in q_lower)
        or ("top 5 percent" in q_lower and "top 20 percent" in q_lower and "revenue" in q_lower)
        or ("which customers fall into the top 5%" in q_lower)
        or ("top 20%" in q_lower and "revenue" in q_lower)
        or ("top 20 percent" in q_lower and "revenue" in q_lower)
        or ("percentage of revenue comes from the top 20%" in q_lower)
        or ("percentage of revenue comes from the top 20 percent" in q_lower)
    ):
        return metrics_lifecycle._es_top_customers_by_revenue(req, client, mappings, business_rules)

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
        return metrics_lifecycle._es_new_customer_acquisition(req, client, mappings, business_rules)

    if (
        "new customer" in q_lower
        and "30" in q_lower
        and "day" in q_lower
        and ("return" in q_lower or "retention" in q_lower)
    ):
        return metrics_lifecycle._es_new_customer_30d_return_rate(req, client, mappings, business_rules)

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
        return metrics_lifecycle._es_customers_nth_visit(req, client, mappings, business_rules)

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

    # ✅ Choose ONE concrete invoices index even if alias/wildcard/multi-index passed
    chosen_index, chosen_mapping = _select_invoice_index_from_es_mapping(client, req.es_index_name)
    properties = _extract_properties_from_mapping(chosen_mapping, chosen_index)
    mappings = {"properties": properties}

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
    special_resp = _route_es_special(req, client, mappings, business_rules)
    if special_resp is not None:
        return special_resp

    # ✅ Only now require API key for LLM-generated DSL
    api_key = (req.api_key or "").strip() or os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise HTTPException(status_code=500, detail="No OpenAI API key configured for ES DSL mode.")

    # LLM-generated DSL path
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
# Dashboard endpoint
# -------------------------------------------------------------------
@router.post("/dashboard")
def es_dashboard(req: MetricsDashboardRequest):
    if not req.es_base_url or not req.es_index_name:
        raise HTTPException(status_code=400, detail="es_base_url and es_index_name are required")

    from app.api.metrics import metrics_customer_value

    client = make_es_client(req.es_base_url, req.es_username, req.es_password)
    if not client.ping():
        raise HTTPException(status_code=400, detail=f"Could not ping Elasticsearch at {req.es_base_url}")

    # ✅ FIX: choose ONE concrete invoices index + mapping (handles alias/wildcards safely)
    chosen_index, chosen_mapping = _select_invoice_index_from_es_mapping(client, req.es_index_name)
    properties = _extract_properties_from_mapping(chosen_mapping, chosen_index)
    mappings = {"properties": properties}

    # ✅ IMPORTANT: use chosen_index everywhere downstream
    base_docs_req = DocsAnalyticsRequest(
        workspace_id=req.workspace_id or "default",
        question="dashboard metrics",
        es_base_url=req.es_base_url,
        es_username=req.es_username,
        es_password=req.es_password,
        es_index_name=chosen_index,  # ✅ FIX
        es_customers_index_name=req.es_customers_index_name,
        es_customer_stats_index_name=req.es_customer_stats_index_name,
    )

    current_vals = metrics_customer_value._window_customer_value_metrics(
        base_docs_req, req.current, client, mappings
    )

    previous_vals: Dict[str, float] = {}
    if req.previous:
        previous_vals = metrics_customer_value._window_customer_value_metrics(
            base_docs_req, req.previous, client, mappings
        )

    label_map = {
        "total_visits": "Total Visits",
        "unique_customers": "Unique Customers",
        "total_revenue": "Total Visit Amount",
        "total_pieces": "Total Visit Pieces",
        "average_visits_per_customer": "Average Visits per Customer",
        "visit_pieces_per_customer": "Visit Pieces per Customer",
        "revenue_per_customer": "Revenue Per Customer",
        "avg_dollar_per_piece": "Avg $ per Piece",
        "initial_visit_amount": "Initial Visit – Amount",
        "initial_visit_pieces": "Initial Visit – Pieces",
        "avg_pickup_delay_retail": "Average Pickup Delay (Retail)",
        "redo_invoices_count": "Invoices with Redo Items",
        "new_customer_acquisition_rate": "New Customer Acquisition",
        "new_customer_30d_return_rate": "New Customer 30-Day Return Rate",
        "customers_2plus_visits": "Customers Achieving 2nd Visit",
        "customers_3plus_visits": "Customers Achieving 3rd Visit",
        "customers_4plus_visits": "Customers Achieving 4th Visit",
        "customers_5plus_visits": "Customers Achieving 5th Visit",
        "top20_customers_redo_issues": "Top 20% Customers – Redo Issues",
        "coupon_returns_365d_since_signup": "Coupon Returns (365 Days Since Signup)",
    }

    metrics: List[MetricsDashboardMetric] = []

    for metric_id, label in label_map.items():
        cur = current_vals.get(metric_id)
        prev = previous_vals.get(metric_id)
        change = None
        if cur is not None and prev not in (None, 0):
            try:
                change = (cur - prev) * 100.0 / float(prev)
            except Exception:
                change = None

        metrics.append(
            MetricsDashboardMetric(
                id=metric_id,
                label=label,
                current=cur,
                previous=prev,
                change_pct=change,
            )
        )

    return {
        "current_period": {"start_date": req.current.start_date, "end_date": req.current.end_date},
        "previous_period": (
            {"start_date": req.previous.start_date, "end_date": req.previous.end_date} if req.previous else None
        ),
        "metrics": [m.model_dump() for m in metrics],
    }


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
