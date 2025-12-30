from __future__ import annotations

import os
from datetime import datetime, timezone
from typing import List, Optional, Dict, Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from copy import deepcopy

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
# Request model
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

    # 👇 existing: primary index (we'll treat this as the invoices index)
    es_index_name: Optional[str] = None

    # 👇 NEW: customers index (for signup dates, membership, etc)
    es_customers_index_name: Optional[str] = None


class Period(BaseModel):
    start_date: str  # "2025-12-13"
    end_date: str    # "2025-12-19"


class MetricsDashboardRequest(BaseModel):
    es_base_url: str
    es_index_name: str
    es_username: Optional[str] = None
    es_password: Optional[str] = None

    # just reuse a workspace; can be "default"
    workspace_id: str = "default"

    current: Period
    previous: Optional[Period] = None  # optional comparison window


class MetricsDashboardMetric(BaseModel):
    id: str
    label: str
    current: Optional[float]
    previous: Optional[float]
    change_pct: Optional[float]


# -------------------------------------------------------------------
# Helper: decide if question is ES-friendly (currently unused)
# -------------------------------------------------------------------

def is_es_friendly_question(q: str) -> bool:
    """
    Decide if a question should go through ES (instead of Python engine).
    We include both generic patterns and all the special business questions.

    NOTE: Router now prefers ES whenever ES config is present, so this
    helper is not used in routing anymore, but kept here if you ever
    want to reintroduce selective ES usage.
    """
    q = (q or "").lower()

    # generic/simple analytics
    simple_patterns = [
        "total revenue",
        "sum of total",
        "revenue by location",
        "revenue by day",
        "revenue by month",
        "grouped by location",
        "grouped by channel",
        "count of invoices",
        "number of invoices",
        "average ticket size",
        "avg ticket size",
    ]

    # special questions (mirror what you had in llm.py)
    special_patterns = [
        "average customer lifetime value",
        "average clv",
        "one-time vs repeat customers",
        "one time vs repeat customers",
        "average days between visits",
        "overdue for their next visit",
        "overdue for next visit",
        "lapsed customers",
        "distribution of customers by visit frequency",
        "top 5%",
        "top 5 percent",
        "top 20%",
        "top 20 percent",
        "month-over-month visit volume trend",
        "month over month visit volume",
        "seasonal patterns",
        "seasonal revenue patterns",
        "seasonal patterns vs last year",
        "new customer acquisition rate",
        "year-over-year revenue growth by location",
        "year over year revenue growth by location",
        "yoy revenue by location",
    ]

    return any(p in q for p in (simple_patterns + special_patterns))


# -------------------------------------------------------------------
# ES mapping resolver (mirror of resolve_column for DataFrames)
# -------------------------------------------------------------------

def resolve_es_field(
    mappings: Dict[str, Any],
    user_term: Optional[str] = None,
    alias_family: Optional[str] = None,
) -> Optional[str]:
    """
    Resolve an ES field path from index mappings, similar to resolve_column()
    but using ES mapping instead of DataFrame columns.

    Matching behaviour:
      - Walk mappings to collect full field paths (e.g. "invoice.customer_id", "total.keyword").
      - Try direct match on user_term (if provided).
      - If alias_family is provided, try to match against ALIASES_UNIVERSAL[alias_family]
        using the LEAF name, being smart about suffixes like .keyword/.raw/.exact.
      - Returns the REAL ES field path to use in queries (not the shortened leaf).
    """
    props = mappings.get("properties", {}) or {}
    flat_fields: List[str] = []

    def _walk(prefix: str, node: Dict[str, Any]):
        if not isinstance(node, dict):
            return

        # ⭐ ROOT CASE: _extract_properties_from_mapping already produced a flat dict
        # like {"customer_id": {...}, "total": {...}} without "type"/"properties"/"fields".
        # If we treated this as a leaf, we would collect nothing, so instead iterate keys.
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

        # leaf-like field
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

    # keys are LOWERCASE field paths, values are the REAL field paths
    lower = {f.lower(): f for f in flat_fields}

    if not user_term:
        user_term = alias_family or "date"

    # 1) direct match on user_term
    variants = [user_term, user_term.replace(" ", "_"), user_term.replace("_", " ")]
    for v in variants:
        key = v.lower()
        if key in lower:
            return lower[key]

    # 🔹 helper: normalize a mapping field key for comparison
    def _normalize_field_key(field_key: str) -> str:
        """
        Lowercase key and strip common suffixes like .keyword/.raw/.exact.
        Keep prefixes (e.g. 'invoice.'), because we still want to support
        matching on leaf via endswith('.customer_id').
        """
        fk = field_key
        for suffix in (".keyword", ".raw", ".exact"):
            if fk.endswith(suffix):
                fk = fk[: -len(suffix)]
                break
        return fk

    # 2) alias family (location, amount, customer...)
    if alias_family in ALIASES_UNIVERSAL:
        for c in ALIASES_UNIVERSAL[alias_family]:
            for v in [c, c.replace("_", " "), c.replace(" ", "_")]:
                key = v.lower()
                for field_key_lower, real in lower.items():
                    norm = _normalize_field_key(field_key_lower)
                    if norm == key or norm.endswith("." + key):
                        return real

    return None


# -------------------------------------------------------------------
# ES helpers: time + "cannot answer" wrapper
# -------------------------------------------------------------------

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
    # Handle trailing 'Z' from ISO 8601 (UTC)
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
    """
    Helper: when ES cannot compute something (missing fields / bad mappings),
    we return an ES-engine response with empty rows instead of falling back
    to the Python/TABLE_STORE path.
    """
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
    """
    Build a bool filter for start_date / end_date on the given date_field.
    Dates are interpreted as inclusive bounds (gte / lte).
    """
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
    """
    Given a potentially multi-index selection (comma-separated, alias, wildcard),
    resolve it to a *single* concrete index that looks like the invoices index.

    Returns (selected_index_name, mapping_for_selected_only).

    Strategy:
      - Ask ES for the mapping behind `raw_index_name`.
      - If it yields a single index, use that.
      - If it yields many indices, prefer an index whose name contains
        'invoice' or 'invoices' (case-insensitive).
      - If none match, fall back to the first index name (sorted).
    """
    raw_index_name = (raw_index_name or "").strip()
    if not raw_index_name:
        raise HTTPException(
            status_code=400,
            detail="No es_index_name provided for invoice metrics.",
        )

    try:
        full_mapping = client.indices.get_mapping(index=raw_index_name)
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Could not fetch Elasticsearch mappings for '{raw_index_name}': {e}",
        )

    index_names = sorted(full_mapping.keys())
    if not index_names:
        raise HTTPException(
            status_code=400,
            detail=f"No indices found for pattern '{raw_index_name}'.",
        )

    # Only one index → nothing to choose
    if len(index_names) == 1:
        chosen = index_names[0]
        return chosen, {chosen: full_mapping[chosen]}

    # Prefer any index that looks like an invoices index
    invoice_like = [
        name for name in index_names
        if "invoice" in name.lower() or "invoices" in name.lower()
    ]
    if invoice_like:
        invoice_like.sort()
        chosen = invoice_like[0]
    else:
        # Fallback: first index alphabetically
        chosen = index_names[0]

    return chosen, {chosen: full_mapping[chosen]}
def _es_get_customer_signups(
    client,
    index_name: str,
    mappings: Dict[str, Any],
) -> Dict[Any, datetime.date]:
    """
    From the *customers* index, return:
        { customer_id -> original_signup_date }

    Uses:
      - customer_id       (segment key)
      - original_signup   (min date per customer)
    """
    customer_field = resolve_es_field(
        mappings,
        user_term="customer_id",
        alias_family="customer",
    )
    signup_field = resolve_es_field(
        mappings,
        user_term="original_signup",
        alias_family="date",
    )

    if not (customer_field and signup_field):
        # No signup data → return empty dict; metric can decide how to handle
        return {}

    body = {
        "size": 0,
        "aggs": {
            "customers": {
                "terms": {
                    "field": customer_field,
                    "size": 10000,  # tune if needed
                },
                "aggs": {
                    "signup": {
                        "min": {"field": signup_field}
                    }
                },
            }
        },
    }

    res = client.search(index=index_name, body=body)
    buckets = (
        res.get("aggregations", {})
           .get("customers", {})
           .get("buckets", [])
    )

    signups: Dict[Any, datetime.date] = {}
    for b in buckets:
        ms = (b.get("signup") or {}).get("value")
        dt = _ms_to_dt(ms)
        if not dt:
            continue
        signups[b.get("key")] = dt.date()

    return signups

def _es_get_customer_stats(
    client,
    index_name: str,
    mappings: Dict[str, Any],
) -> Optional[List[Dict[str, Any]]]:
    """
    Aggregate per-customer stats in ES:

      - first_visit (min date)
      - last_visit  (max date)
      - visit_count (all time, DISTINCT visit_id when available)
      - visits_365  (last 365 days, DISTINCT visit_id when available)
      - total_revenue (sum of amount, if available)
      - total_pieces  (sum of pieces, if available)

    NOTE: this is intentionally "all history", it does not use start_date/end_date.
    Windowed metrics that need a date filter build their own queries.
    """
    # 🔧 Prefer concrete fields for your invoices index,
    # but still keep alias logic as fallback.
    customer_field = resolve_es_field(
        mappings,
        user_term="customer_id",
        alias_family="customer",
    )
    date_field = resolve_es_field(
        mappings,
        user_term="dropoff_at",
        alias_family="date",
    )
    amount_field = resolve_es_field(
        mappings,
        user_term="total",
        alias_family="amount",
    )
    # 🔄 NEW: try to resolve visit_id field so a visit = distinct visit_id
    visit_field = resolve_es_field(
        mappings,
        user_term="visit_id",
        alias_family="visit",
    )
    # 🔄 NEW: try to resolve a "pieces" field (for Total Visit Pieces)
    pieces_field = (
        resolve_es_field(
            mappings,
            user_term="total_pieces",
            alias_family="pieces",
        )
        or resolve_es_field(
            mappings,
            user_term="pieces",
            alias_family="pieces",
        )
    )

    if not customer_field or not date_field:
        return None

    # Build per-customer aggregations depending on whether visit_id is available
    if visit_field:
        customer_aggs: Dict[str, Any] = {
            "first_visit": {"min": {"field": date_field}},
            "last_visit": {"max": {"field": date_field}},
            # DISTINCT visits per customer = cardinality of visit_id
            "visit_count": {"cardinality": {"field": visit_field}},
            # DISTINCT visits in last 365 days
            "visits_365": {
                "filter": {
                    "range": {
                        date_field: {"gte": "now-365d/d"}
                    }
                },
                "aggs": {
                    "visits_365_distinct": {
                        "cardinality": {"field": visit_field}
                    }
                },
            },
        }
    else:
        # Fallback: no visit_id → approximate visits by counting rows (date values)
        customer_aggs = {
            "first_visit": {"min": {"field": date_field}},
            "last_visit": {"max": {"field": date_field}},
            "visit_count": {"value_count": {"field": date_field}},
            "visits_365": {
                "filter": {
                    "range": {
                        date_field: {"gte": "now-365d/d"}
                    }
                },
            },
        }

    aggs: Dict[str, Any] = {
        "customers": {
            "terms": {
                "field": customer_field,
                "size": 10000,  # adjust if needed
            },
            "aggs": customer_aggs,
        }
    }

    if amount_field:
        aggs["customers"]["aggs"]["total_revenue"] = {"sum": {"field": amount_field}}

    # NEW: total pieces per customer if a pieces field exists
    if pieces_field:
        aggs["customers"]["aggs"]["total_pieces"] = {"sum": {"field": pieces_field}}

    body = {
        "size": 0,
        "aggs": aggs,
    }

    res = client.search(index=index_name, body=body)
    buckets = (
        res.get("aggregations", {})
           .get("customers", {})
           .get("buckets", [])
    )

    has_visit_field = bool(visit_field)

    stats: List[Dict[str, Any]] = []
    for b in buckets:
        fv = b.get("first_visit", {}).get("value")
        lv = b.get("last_visit", {}).get("value")

        if has_visit_field:
            vc_raw = (b.get("visit_count") or {}).get("value")
            vc = int(vc_raw or 0)
            v365_raw = (
                (b.get("visits_365") or {})
                .get("visits_365_distinct", {})
                .get("value")
            )
            v365 = int(v365_raw or 0)
        else:
            vc_raw = (b.get("visit_count") or {}).get("value")
            vc = int(vc_raw or 0)
            v365_raw = (b.get("visits_365") or {}).get("doc_count")
            v365 = int(v365_raw or 0)

        tr = None
        if amount_field and "total_revenue" in b:
            tr_val = b["total_revenue"].get("value")
            tr = float(tr_val) if tr_val is not None else 0.0

        tp = None
        if pieces_field and "total_pieces" in b:
            tp_val = b["total_pieces"].get("value")
            tp = float(tp_val) if tp_val is not None else 0.0

        stats.append(
            {
                "customer_id": b.get("key"),
                "first_visit": _ms_to_dt(fv),
                "last_visit": _ms_to_dt(lv),
                "visit_count": vc,
                "visits_365": v365,
                "total_revenue": tr,
                "total_pieces": tp,
            }
        )

    return stats


# -------------------------------------------------------------------
# ES special implementations
# -------------------------------------------------------------------

def _es_core_visit_metrics(
    req: DocsAnalyticsRequest,
    client,
    mappings: Dict[str, Any],
    business_rules: Optional[str],
):
    """
    Core visit KPIs for the dashboard:

      - Total Visit Amount   (sum of all customer revenue)
      - Total Visit Pieces   (sum of all pieces, if a pieces field exists)
      - Total Visits         (sum of per-customer visit_count)
      - Unique Customers     (number of distinct customers with visits)

    NOTE: this is lifetime/all-history, not windowed.

    👉 Even if multiple indices are selected, we resolve to a single
       'invoices' index and run the metrics there.
    """
    # 🔹 Always resolve to a single invoices index (if multiple selected)
    invoice_index, invoice_mapping = _select_invoice_index_from_es_mapping(
        client,
        req.es_index_name,
    )

    # Flatten mappings for the chosen invoices index only
    properties = _extract_properties_from_mapping(invoice_mapping, invoice_index)
    invoice_mappings = {"properties": properties}

    stats = _es_get_customer_stats(client, invoice_index, invoice_mappings)

    if stats is None or not stats:
        return _es_cannot_answer(
            "Cannot compute core visit metrics because customer/date fields "
            "could not be resolved from the Elasticsearch mappings or no customers "
            "with visits were found.",
            business_rules,
        )

    total_visit_amount = 0.0
    total_visit_pieces = 0.0
    total_visits = 0
    unique_customers = 0

    any_pieces = False

    for s in stats:
        unique_customers += 1

        vc = s.get("visit_count") or 0
        total_visits += vc

        tr = s.get("total_revenue")
        if tr is not None:
            total_visit_amount += float(tr)

        tp = s.get("total_pieces")
        if tp is not None:
            total_visit_pieces += float(tp)
            any_pieces = True

    pieces_value = total_visit_pieces if any_pieces else None

    rows: List[Dict[str, Any]] = [
        {
            "metric": "total_visit_amount",
            "label": "Total Visit Amount",
            "value": total_visit_amount,
        },
        {
            "metric": "total_visit_pieces",
            "label": "Total Visit Pieces",
            "value": pieces_value,
        },
        {
            "metric": "total_visits",
            "label": "Total Visits",
            "value": total_visits,
        },
        {
            "metric": "unique_customers",
            "label": "Unique Customers",
            "value": unique_customers,
        },
    ]

    insight = (
        f"Core visit metrics were computed directly on Elasticsearch index '{invoice_index}' "
        f"using per-customer visit statistics. "
        f"Total Visit Amount is the sum of all customer revenue, Total Visits is the sum of per-customer "
        f"visit counts, and Unique Customers is the number of customers with at least one visit. "
    )
    if pieces_value is None:
        insight += (
            "No suitable 'pieces' field could be found in the index mappings, so "
            "Total Visit Pieces is not available (value is null)."
        )
    else:
        insight += (
            "Total Visit Pieces is the sum of per-customer piece counts based on the resolved pieces field."
        )

    return {
        "insight": to_json_safe(insight),
        "rows": to_json_safe(rows),
        "rules_used": business_rules or "",
        "engine": "es",
    }



# NEW: Average Visits / Pieces / Revenue per Customer + Avg $ per Piece
def _es_customer_value_metrics(
    req: DocsAnalyticsRequest,
    client,
    mappings: Dict[str, Any],
    business_rules: Optional[str],
):
    """
    Windowed customer value metrics for the selected period (if start_date/end_date set):

      - Average Visits per Customer
      - Visit Pieces per Customer
      - Revenue Per Customer
      - Avg $ per Piece

    Window is applied on dropoff_at. A "visit" is a distinct visit_id if available,
    otherwise we approximate using invoice rows.

    👉 Even if multiple indices are selected, these metrics are always computed
       on a single invoices index.
    """
    # 🔹 Resolve the invoices index (even if `es_index_name` is multi/pattern)
    invoice_index, invoice_mapping = _select_invoice_index_from_es_mapping(
        client,
        req.es_index_name,
    )
    index_name = invoice_index

    # Flatten mappings for that index only
    properties = _extract_properties_from_mapping(invoice_mapping, invoice_index)
    invoice_mappings = {"properties": properties}

    customer_field = resolve_es_field(
        invoice_mappings,
        user_term="customer_id",
        alias_family="customer",
    )
    date_field = resolve_es_field(
        invoice_mappings,
        user_term="dropoff_at",
        alias_family="date",
    )
    amount_field = resolve_es_field(
        invoice_mappings,
        user_term="total",
        alias_family="amount",
    )
    pieces_field = resolve_es_field(
        invoice_mappings,
        user_term="pieces",
        alias_family="pieces",
    )
    visit_field = resolve_es_field(
        invoice_mappings,
        user_term="visit_id",
        alias_family="visit",
    )

    if not (customer_field and date_field and amount_field):
        return _es_cannot_answer(
            "Cannot compute customer value metrics because customer, date or amount "
            "fields could not be resolved from the Elasticsearch mappings.",
            business_rules,
        )

    filters = _build_date_range_filter(req, date_field)

    aggs: Dict[str, Any] = {
        "unique_customers": {"cardinality": {"field": customer_field}},
        "total_revenue": {"sum": {"field": amount_field}},
    }

    if pieces_field:
        aggs["total_pieces"] = {"sum": {"field": pieces_field}}

    if visit_field:
        aggs["total_visits"] = {"cardinality": {"field": visit_field}}
    else:
        aggs["total_visits"] = {"value_count": {"field": date_field}}

    body: Dict[str, Any] = {
        "size": 0,
        "aggs": aggs,
    }
    if filters:
        body["query"] = {"bool": {"filter": filters}}

    res = client.search(index=index_name, body=body)
    agg = res.get("aggregations", {}) or {}

    unique_customers = int((agg.get("unique_customers") or {}).get("value") or 0)
    total_revenue = float((agg.get("total_revenue") or {}).get("value") or 0.0)

    total_pieces = None
    if "total_pieces" in agg:
        total_pieces = float((agg["total_pieces"] or {}).get("value") or 0.0)

    total_visits = int((agg.get("total_visits") or {}).get("value") or 0)

    if unique_customers > 0:
        avg_visits_per_customer = (
            total_visits / float(unique_customers)
            if total_visits is not None
            else None
        )
        revenue_per_customer = total_revenue / float(unique_customers)
        pieces_per_customer = (
            total_pieces / float(unique_customers)
            if (total_pieces is not None and unique_customers > 0)
            else None
        )
    else:
        avg_visits_per_customer = None
        revenue_per_customer = None
        pieces_per_customer = None

    if total_pieces and total_pieces > 0:
        avg_dollar_per_piece = total_revenue / total_pieces
    else:
        avg_dollar_per_piece = None

    rows: List[Dict[str, Any]] = [
        {
            "metric": "average_visits_per_customer",
            "label": "Average Visits per Customer",
            "value": avg_visits_per_customer,
        },
        {
            "metric": "visit_pieces_per_customer",
            "label": "Visit Pieces per Customer",
            "value": pieces_per_customer,
        },
        {
            "metric": "revenue_per_customer",
            "label": "Revenue Per Customer",
            "value": revenue_per_customer,
        },
        {
            "metric": "avg_dollar_per_piece",
            "label": "Avg $ per Piece",
            "value": avg_dollar_per_piece,
        },
        {
            "metric": "total_visits",
            "label": "Total Visits (window)",
            "value": total_visits,
        },
        {
            "metric": "unique_customers",
            "label": "Unique Customers (window)",
            "value": unique_customers,
        },
        {
            "metric": "total_revenue",
            "label": "Total Revenue (window)",
            "value": total_revenue,
        },
    ]

    if total_pieces is not None:
        rows.append(
            {
                "metric": "total_pieces",
                "label": "Total Pieces (window)",
                "value": total_pieces,
            }
        )

    if req.start_date or req.end_date:
        window_desc = []
        if req.start_date:
            window_desc.append(f"from {req.start_date}")
        if req.end_date:
            window_desc.append(f"to {req.end_date}")
        window_str = " ".join(window_desc)
    else:
        window_str = "for all available history"

    insight = (
        f"Customer value metrics were computed on index '{index_name}' {window_str}, "
        f"using '{date_field}' as the visit date, '{customer_field}' as the customer id, "
        f"and '{amount_field}' as the invoice total. A visit is treated as a distinct "
        f"visit_id when available, otherwise invoice rows are used as a proxy."
    )

    return {
        "insight": to_json_safe(insight),
        "rows": to_json_safe(rows),
        "rules_used": business_rules or "",
        "engine": "es",
    }
def _window_customer_value_metrics(
    base_req: DocsAnalyticsRequest,
    period: Period,
    client,
    mappings: Dict[str, Any],
) -> Dict[str, float]:
    """
    Run _es_customer_value_metrics for a specific [start_date, end_date]
    and return a dict {metric_id -> value}.

    Also enrich with:
      - initial_visit_amount
      - initial_visit_pieces
    """
    req = deepcopy(base_req)
    req.start_date = period.start_date
    req.end_date = period.end_date

    # Core customer value metrics (visits, revenue, pieces, etc.)
    resp = _es_customer_value_metrics(
        req=req,
        client=client,
        mappings=mappings,
        business_rules=None,
    )

    rows = resp.get("rows") or []
    values: Dict[str, float] = {}
    for r in rows:
        mid = r.get("metric")
        if mid is not None:
            values[mid] = r.get("value")

    # NEW: initial visit metrics for this window
    init_vals = _es_initial_visit_totals(req, client, mappings) or {}
    for mid, val in init_vals.items():
        values[mid] = val

    return values


def _es_avg_pickup_delay_retail(
    req: DocsAnalyticsRequest,
    client,
    mappings: Dict[str, Any],
    business_rules: Optional[str],
):
    """
    Average pickup delay for **Retail route only** based on pickup_at - ready_at.

    Logic:
      - Filter invoices to route.name = "Retail" (nested route object)
      - Optional date window on dropoff_at if start_date/end_date are provided
      - Compute avg( pickup_at - ready_at ) in milliseconds, then expose as days + hours
    """
    index_name = req.es_index_name.strip()

    # main date field for the window
    date_field = resolve_es_field(
        mappings,
        user_term="dropoff_at",
        alias_family="date",
    )
    ready_field = resolve_es_field(
        mappings,
        user_term="ready_at",
        alias_family="date",
    )
    pickup_field = resolve_es_field(
        mappings,
        user_term="pickup_at",
        alias_family="date",
    )

    if not (date_field and ready_field and pickup_field):
        return _es_cannot_answer(
            "Cannot compute average pickup delay because dropoff_at, ready_at or "
            "pickup_at could not be resolved from the Elasticsearch mappings.",
            business_rules,
        )

    # 🔍 resolve route name field (nested)
    route_name_field = (
        resolve_es_field(mappings, user_term="route.name.keyword")
        or resolve_es_field(mappings, user_term="route.name")
    )

    if not route_name_field:
        return _es_cannot_answer(
            "Cannot compute average pickup delay (Retail) because no route.name field "
            "could be resolved from the Elasticsearch mappings.",
            business_rules,
        )

    # base filters: optional date range on dropoff_at
    filters = _build_date_range_filter(req, date_field)

    # ✅ Retail only: route.name = "Retail"
    filters.append(
        {
            "nested": {
                "path": "route",
                "query": {
                    "term": {
                        route_name_field: "Retail"  # exact match on Retail route
                    }
                },
            }
        }
    )

    script_source = (
        f"if (doc['{pickup_field}'].size() == 0 || doc['{ready_field}'].size() == 0) "
        f"{{ return null; }} "
        f"return doc['{pickup_field}'].value.toInstant().toEpochMilli() - "
        f"       doc['{ready_field}'].value.toInstant().toEpochMilli();"
    )

    body: Dict[str, Any] = {
        "size": 0,
        "aggs": {
            "avg_delay_ms": {
                "avg": {
                    "script": {
                        "lang": "painless",
                        "source": script_source,
                    }
                }
            }
        },
    }
    if filters:
        body["query"] = {"bool": {"filter": filters}}

    res = client.search(index=index_name, body=body)
    avg_obj = (res.get("aggregations") or {}).get("avg_delay_ms") or {}
    value_ms = avg_obj.get("value")

    if value_ms is None:
        rows: List[Dict[str, Any]] = []
        insight = (
            "Average pickup delay (Retail) could not be computed because no retail invoice "
            "had both ready_at and pickup_at populated in the selected period."
        )
    else:
        ms = float(value_ms)
        seconds = ms / 1000.0
        hours = seconds / 3600.0
        days = seconds / 86400.0

        rows = [
            {
                "metric": "avg_pickup_delay_days",
                "label": "Average Pickup Delay (Days, Retail only)",
                "value": days,
            },
            {
                "metric": "avg_pickup_delay_hours",
                "label": "Average Pickup Delay (Hours, Retail only)",
                "value": hours,
            },
        ]
        insight = (
            f"Average pickup delay (Retail only) was computed on index '{index_name}' using "
            f"pickup_at - ready_at for invoices where route.name = 'Retail', restricted by the "
            f"optional date range on {date_field}. The result is reported in days and hours."
        )

    return {
        "insight": to_json_safe(insight),
        "rows": to_json_safe(rows),
        "rules_used": business_rules or "",
        "engine": "es",
    }
def _es_initial_visit_totals(
    req: DocsAnalyticsRequest,
    client,
    mappings: Dict[str, Any],
) -> Dict[str, Optional[float]]:
    """
    Initial Visit – Amount / Initial Visit – Pieces (windowed):

      - We work on the invoices index only.
      - For each customer:
          * group documents by visit_id
          * for each visit:
                first_date  = min(dropoff_at)
                visit_amount  = sum(total)
                visit_pieces  = sum(pieces)  (if pieces field exists)
          * order visits by first_date ascending
          * first visit for that customer = first bucket

      - Initial Visit metrics for the window:
          * include customers where first_visit_date is in [start_date, end_date]
          * sum their visit_amount (Initial Visit – Amount)
          * sum their visit_pieces (Initial Visit – Pieces)

    NOTE:
      - Requires a visit_id field to be resolved; if not available,
        we return None for both metrics (cannot do it correctly
        without visit-level grouping).
    """
    # 🔹 Resolve a single invoices index (if pattern/alias used)
    try:
        invoice_index, invoice_mapping = _select_invoice_index_from_es_mapping(
            client,
            req.es_index_name,
        )
    except Exception:
        return {
            "initial_visit_amount": None,
            "initial_visit_pieces": None,
        }

    # Flatten mappings for the chosen invoices index only
    properties = _extract_properties_from_mapping(invoice_mapping, invoice_index)
    invoice_mappings = {"properties": properties}

    customer_field = resolve_es_field(
        invoice_mappings,
        user_term="customer_id",
        alias_family="customer",
    )
    date_field = resolve_es_field(
        invoice_mappings,
        user_term="dropoff_at",
        alias_family="date",
    )
    amount_field = resolve_es_field(
        invoice_mappings,
        user_term="total",
        alias_family="amount",
    )
    pieces_field = resolve_es_field(
        invoice_mappings,
        user_term="pieces",
        alias_family="pieces",
    )
    visit_field = resolve_es_field(
        invoice_mappings,
        user_term="visit_id",
        alias_family="visit",
    )

    # Must have customer, date, amount and visit_id to compute visit-level first visits
    if not (customer_field and date_field and amount_field and visit_field):
        return {
            "initial_visit_amount": None,
            "initial_visit_pieces": None,
        }

    has_pieces = bool(pieces_field)

    # Per-customer → per-visit (visit_id) aggregations
    visits_aggs: Dict[str, Any] = {
        "first_date": {"min": {"field": date_field}},
        "visit_amount": {"sum": {"field": amount_field}},
    }
    if has_pieces:
        visits_aggs["visit_pieces"] = {"sum": {"field": pieces_field}}

    body: Dict[str, Any] = {
        "size": 0,
        "aggs": {
            "customers": {
                "terms": {
                    "field": customer_field,
                    "size": 10000,  # max distinct customers
                },
                "aggs": {
                    "visits": {
                        "terms": {
                            "field": visit_field,
                            # we only need the *first* visit per customer,
                            # but set a reasonable cap on visits per customer
                            "size": 100,
                            "order": {"first_date": "asc"},
                        },
                        "aggs": visits_aggs,
                    }
                },
            }
        },
    }

    try:
        res = client.search(index=invoice_index, body=body)
    except Exception:
        return {
            "initial_visit_amount": None,
            "initial_visit_pieces": None,
        }

    cust_buckets = (
        res.get("aggregations", {})
           .get("customers", {})
           .get("buckets", [])
    )

    start_d = _parse_date_str(req.start_date)
    end_d = _parse_date_str(req.end_date)

    total_amount = 0.0
    total_pieces = 0.0
    any_pieces_values = False

    for cb in cust_buckets:
        visits = (cb.get("visits") or {}).get("buckets") or []
        if not visits:
            continue

        first_visit_bucket = visits[0]
        first_date_ms = (first_visit_bucket.get("first_date") or {}).get("value")
        dt = _ms_to_dt(first_date_ms)
        if not dt:
            continue

        visit_date = dt.date()
        if start_d and visit_date < start_d:
            continue
        if end_d and visit_date > end_d:
            continue

        amt_val = (first_visit_bucket.get("visit_amount") or {}).get("value")
        if amt_val is not None:
            total_amount += float(amt_val)

        if has_pieces:
            pieces_val = (first_visit_bucket.get("visit_pieces") or {}).get("value")
            if pieces_val is not None:
                total_pieces += float(pieces_val)
                any_pieces_values = True

    return {
        "initial_visit_amount": total_amount,
        "initial_visit_pieces": (total_pieces if any_pieces_values else None),
    }
    
def _es_invoices_with_redo_items(
    req: DocsAnalyticsRequest,
    client,
    mappings: Dict[str, Any],
    business_rules: Optional[str],
):
    """
    Invoices with Redo Items

    Definition:
      - Any invoice where the coupon field contains the word "redo"
        (case-insensitive, anywhere in the coupon value).
      - Count distinct invoices in the selected date window.

    Window:
      - Uses dropoff_at as the invoice date.
      - Optional: req.start_date / req.end_date (inclusive).
    """
    index_name = req.es_index_name.strip()

    # Prefer invoices fields: dropoff_at + invoice_id + coupon
    date_field = resolve_es_field(
        mappings,
        user_term="dropoff_at",
        alias_family="date",
    )
    invoice_id_field = resolve_es_field(
        mappings,
        user_term="invoice_id",
        alias_family="invoice",
    )
    coupon_field = resolve_es_field(
        mappings,
        user_term="coupon",
    )

    if not (date_field and invoice_id_field and coupon_field):
        return _es_cannot_answer(
            "Cannot compute 'Invoices with Redo Items' because date, invoice_id "
            "or coupon fields could not be resolved from the Elasticsearch mappings.",
            business_rules,
        )

    # Base filters: optional date range on dropoff_at
    filters = _build_date_range_filter(req, date_field)

    # We treat an invoice as 'redo' if coupon contains the word 'redo'
    # (case-insensitive). We combine a match on the analyzed field and
    # a wildcard on the keyword field for robustness.
    should_clauses = [
        {"match_phrase": {coupon_field: "redo"}},
    ]

    # also try coupon.keyword if it exists
    keyword_field = (
        coupon_field if coupon_field.endswith(".keyword") else f"{coupon_field}.keyword"
    )
    should_clauses.append(
        {
            "wildcard": {
                keyword_field: "*redo*"
            }
        }
    )

    coupon_filter = {
        "bool": {
            "should": should_clauses,
            "minimum_should_match": 1,
        }
    }

    filters.append(coupon_filter)

    body: Dict[str, Any] = {
        "size": 0,
        "query": {
            "bool": {
                "filter": filters,
            }
        },
        "aggs": {
            "invoices_with_redo": {
                "cardinality": {
                    "field": invoice_id_field,
                }
            }
        },
    }

    res = client.search(index=index_name, body=body)
    agg = (res.get("aggregations") or {}).get("invoices_with_redo") or {}
    count = int(agg.get("value") or 0)

    rows: List[Dict[str, Any]] = [
        {
            "metric": "invoices_with_redo",
            "label": "Invoices with Redo Items",
            "value": count,
        }
    ]

    # Describe the window for the insight text
    window_desc = []
    if req.start_date:
        window_desc.append(f"from {req.start_date}")
    if req.end_date:
        window_desc.append(f"to {req.end_date}")
    window_str = " ".join(window_desc) if window_desc else "for all available history"

    insight = (
        f"'Invoices with Redo Items' is computed on index '{index_name}' {window_str} "
        f"as the count of distinct invoices where the coupon field contains the word "
        f"'redo' (case-insensitive), using '{date_field}' as the invoice date and "
        f"'{invoice_id_field}' as the invoice identifier."
    )


    return {
        "insight": to_json_safe(insight),
        "rows": to_json_safe(rows),
        "rules_used": business_rules or "",
        "engine": "es",
    }
def _es_top20_customers_with_redo_courtesy(
    req: DocsAnalyticsRequest,
    client,
    mappings: Dict[str, Any],          # mappings for the *invoices* index
    business_rules: Optional[str],
):
    """
    Top 20% Customers with Redo/Courtesy Items.

    Steps:

      1. From the *customers* index, find all customers that have a tag
         with name "Top 20%". These are pre-computed top 20% spenders
         (last 365 days).

      2. For those customers, look in the invoices index and count how
         many invoices in the selected date window have a coupon
         containing "redo" or "courtesy".

      3. Return only customers that:
           - are tagged "Top 20%", AND
           - have at least one redo/courtesy invoice in the window.
    """

    invoices_index = (req.es_index_name or "").strip()
    customers_index = (req.es_customers_index_name or "").strip()

    if not invoices_index or not customers_index:
        return _es_cannot_answer(
            "Top 20% Customers with Redo/Courtesy Items requires both an invoices "
            "index (es_index_name) and a customers index (es_customers_index_name).",
            business_rules,
        )

    # ---------------------------------------------------------
    # 1) Invoice-side fields (invoices index)
    # ---------------------------------------------------------
    invoice_customer_field = resolve_es_field(
        mappings,
        user_term="customer_id",
        alias_family="customer",
    )
    invoice_date_field = resolve_es_field(
        mappings,
        user_term="dropoff_at",
        alias_family="date",
    )
    invoice_id_field = resolve_es_field(
        mappings,
        user_term="invoice_id",
        alias_family="invoice",
    )
    coupon_field = resolve_es_field(mappings, user_term="coupon")

    if not (
        invoice_customer_field
        and invoice_date_field
        and invoice_id_field
        and coupon_field
    ):
        return _es_cannot_answer(
            "Cannot compute Top 20% Customers with Redo/Courtesy Items because "
            "customer, date, invoice_id or coupon fields could not be resolved "
            "from the invoices index mappings.",
            business_rules,
        )

    # ---------------------------------------------------------
    # 2) Customer-side fields (customers index)
    # ---------------------------------------------------------
    cust_mapping_raw = client.indices.get_mapping(index=customers_index)
    cust_props = _extract_properties_from_mapping(cust_mapping_raw, customers_index)
    cust_mappings = {"properties": cust_props}

    cust_id_field = resolve_es_field(
        cust_mappings,
        user_term="customer_id",
        alias_family="customer",
    )
    tags_name_field = (
        resolve_es_field(cust_mappings, user_term="tags.name")
        or resolve_es_field(cust_mappings, user_term="tags.name.keyword")
    )

    if not (cust_id_field and tags_name_field):
        return _es_cannot_answer(
            "Cannot compute Top 20% Customers with Redo/Courtesy Items because "
            "customer_id or tags.name could not be resolved from the customers index.",
            business_rules,
        )

    # ---------------------------------------------------------
    # 3) Fetch all customers tagged "Top 20%" from customers index
    # ---------------------------------------------------------
    top20_filter = {
        "nested": {
            "path": "tags",
            "query": {
                "term": {
                    tags_name_field: "Top 20%"
                }
            },
        }
    }

    body_cust = {
        "size": 10000,
        "query": {"bool": {"filter": [top20_filter]}},
    }

    res_cust = client.search(index=customers_index, body=body_cust)
    cust_hits = (res_cust.get("hits") or {}).get("hits") or []

    if not cust_hits:
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

    # Build a dict: customer_id -> basic info (name + spend)
    top20_info_by_id: Dict[Any, Dict[str, Any]] = {}

    for h in cust_hits:
        src = h.get("_source", {}) or {}

        # Prefer explicit field names from your sample doc
        cid = src.get("customer_id")
        if cid is None:
            # fallback if mapping uses a different leaf name
            cid = src.get(cust_id_field.split(".")[-1])
        if cid is None:
            continue

        first_name = (src.get("first_name") or "").strip()
        last_name = (src.get("last_name") or "").strip()
        full_name = (f"{first_name} {last_name}").strip() or f"Customer {cid}"

        # Use lifetime if available, otherwise 365-day spend
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

    if not top20_info_by_id:
        insight = (
            "Customers with a 'Top 20%' tag were found, but no usable customer_id "
            "values could be extracted."
        )
        return {
            "insight": to_json_safe(insight),
            "rows": [],
            "rules_used": business_rules or "",
            "engine": "es",
        }

    customer_ids = list(top20_info_by_id.keys())

    # ---------------------------------------------------------
    # 4) Invoices: redo/courtesy counts for those customers
    # ---------------------------------------------------------
    filters = _build_date_range_filter(req, invoice_date_field)

    # Only invoices from Top-20 customers
    filters.append({"terms": {invoice_customer_field: customer_ids}})

    # Coupon filter: contains "redo" or "courtesy"
    should_clauses = [
        {"match_phrase": {coupon_field: "redo"}},
        {"match_phrase": {coupon_field: "courtesy"}},
    ]
    keyword_field = (
        coupon_field if coupon_field.endswith(".keyword") else f"{coupon_field}.keyword"
    )
    should_clauses.extend(
        [
            {"wildcard": {keyword_field: "*redo*"}},
            {"wildcard": {keyword_field: "*courtesy*"}},
        ]
    )

    filters.append(
        {
            "bool": {
                "should": should_clauses,
                "minimum_should_match": 1,
            }
        }
    )

    body_inv = {
        "size": 0,
        "query": {"bool": {"filter": filters}},
        "aggs": {
            "customers": {
                "terms": {
                    "field": invoice_customer_field,
                    "size": len(customer_ids),
                },
                "aggs": {
                    # Distinct invoices with redo/courtesy coupons
                    "redo_invoices": {
                        "cardinality": {"field": invoice_id_field}
                    }
                },
            }
        },
    }

    res_inv = client.search(index=invoices_index, body=body_inv)
    buckets = (
        res_inv.get("aggregations", {})
        .get("customers", {})
        .get("buckets", [])
    )

    rows: List[Dict[str, Any]] = []
    for b in buckets:
        cid = b.get("key")
        info = top20_info_by_id.get(cid)
        if not info:
            continue

        redo_count = int((b.get("redo_invoices") or {}).get("value") or 0)
        if redo_count <= 0:
            continue

        rows.append(
            {
                "customer_id": cid,
                "customer_name": info["name"],
                # you can bind this to the "Lifetime Value" column in the UI
                "lifetime_value": info["lifetime_value"],
                "sales_pickup_lifetime": info["sales_pickup_lifetime"],
                "sales_pickup_365": info["sales_pickup_365"],
                "redo_count": redo_count,
                "details": "Has redo/courtesy coupon invoices in selected period",
            }
        )

    if not rows:
        window_desc = []
        if req.start_date:
            window_desc.append(f"from {req.start_date}")
        if req.end_date:
            window_desc.append(f"to {req.end_date}")
        window_str = " ".join(window_desc) if window_desc else "for the full dataset"

        insight = (
            "No customers tagged 'Top 20%' have invoices with redo or courtesy coupons "
            f"{window_str}."
        )
        return {
            "insight": to_json_safe(insight),
            "rows": [],
            "rules_used": business_rules or "",
            "engine": "es",
        }

    # Sort by lifetime_value (highest first)
    rows.sort(key=lambda r: (r.get("lifetime_value") or 0.0), reverse=True)

    window_desc = []
    if req.start_date:
        window_desc.append(f"from {req.start_date}")
    if req.end_date:
        window_desc.append(f"to {req.end_date}")
    window_str = " ".join(window_desc) if window_desc else "for the full dataset"

    insight = (
        "Top 20% Customers with Redo/Courtesy Items was computed by first selecting "
        "customers from the customers index that have a 'Top 20%' customer tag "
        "(top 20% spenders over the last 365 days), then counting how many of their "
        f"invoices in the invoices index have a coupon containing 'redo' or 'courtesy' "
        f"{window_str}. Only tagged customers with at least one such invoice are returned."
    )

    return {
        "insight": to_json_safe(insight),
        "rows": to_json_safe(rows),
        "rules_used": business_rules or "",
        "engine": "es",
    }

def _es_top20_customers_overdue_14d(
    req: DocsAnalyticsRequest,
    client,
    mappings: Dict[str, Any],          # mappings for the *invoices* index
    business_rules: Optional[str],
):
    """
    Top 20% Customers – Overdue (14+ Days Past Interval)

    Rules (per John + your update):

      - Use ONLY the 'Top 20%' tag from the *customers* index.
      - All timing metrics (first/last visit, interval, days overdue)
        come from the *invoices* index via _es_get_customer_stats.

    Steps:

      1. From invoices index: lifetime stats per customer
         (first_visit, last_visit, visit_count, total_revenue).

      2. From customers index:
           - Find customers tagged 'Top 20%'.
           - Get first_name / last_name to display in the table.

      3. For each tagged customer that has stats:
           interval ≈ (last_visit - first_visit) / (visit_count - 1)
           days_since_last = today - last_visit
           days_overdue = days_since_last - interval

         Keep only rows where days_overdue ≥ 14.
    """

    invoices_index = (req.es_index_name or "").strip()
    customers_index = (req.es_customers_index_name or "").strip()

    if not invoices_index or not customers_index:
        return _es_cannot_answer(
            "Top 20% Customers – Overdue (14+ Days Past Interval) requires both "
            "an invoices index (es_index_name) and a customers index (es_customers_index_name).",
            business_rules,
        )

    # 1) lifetime stats from invoices index
    stats = _es_get_customer_stats(client, invoices_index, mappings)
    if stats is None:
        return _es_cannot_answer(
            "Cannot compute Top 20% Customers – Overdue because customer/date fields "
            "could not be resolved from the invoices index mappings.",
            business_rules,
        )

    stats_by_id: Dict[Any, Dict[str, Any]] = {
        s["customer_id"]: s for s in stats if s.get("customer_id") is not None
    }

    # 2) resolve customers mapping + tag + names
    cust_mapping_raw = client.indices.get_mapping(index=customers_index)
    cust_props = _extract_properties_from_mapping(cust_mapping_raw, customers_index)
    cust_mappings = {"properties": cust_props}

    cust_id_field = resolve_es_field(
        cust_mappings,
        user_term="customer_id",
        alias_family="customer",
    )
    tags_name_field = (
        resolve_es_field(cust_mappings, user_term="tags.name")
        or resolve_es_field(cust_mappings, user_term="tags.name.keyword")
    )

    if not (cust_id_field and tags_name_field):
        return _es_cannot_answer(
            "Cannot compute Top 20% Customers – Overdue because customer_id or tags.name "
            "could not be resolved from the customers index.",
            business_rules,
        )

    # pull customers tagged "Top 20%"
    top20_filter = {
        "nested": {
            "path": "tags",
            "query": {
                "term": {
                    tags_name_field: "Top 20%"
                }
            },
        }
    }

    body_cust = {
        "size": 10000,
        "query": {"bool": {"filter": [top20_filter]}},
    }

    res_cust = client.search(index=customers_index, body=body_cust)
    cust_hits = (res_cust.get("hits") or {}).get("hits") or []

    today = datetime.now(timezone.utc).date()
    rows: List[Dict[str, Any]] = []

    for h in cust_hits:
        src = h.get("_source", {}) or {}

        cid = src.get("customer_id")
        if cid is None:
            # fallback: use the leaf from cust_id_field
            cid = src.get(cust_id_field.split(".")[-1])
        if cid is None:
            continue

        stat = stats_by_id.get(cid)
        if not stat:
            # tagged as Top 20% but no invoice stats; skip
            continue

        first = stat.get("first_visit")
        last = stat.get("last_visit")
        vc = stat.get("visit_count") or 0

        if not first or not last or vc <= 1:
            # need at least 2 visits and valid dates
            continue

        first_d = first.date()
        last_d = last.date()
        days_between = (last_d - first_d).days
        if days_between <= 0:
            continue

        intervals = vc - 1
        if intervals <= 0:
            continue

        typical_interval = days_between / float(intervals)
        days_since_last = (today - last_d).days
        days_overdue = days_since_last - typical_interval

        if days_overdue < 14:
            continue

        first_name = (src.get("first_name") or "").strip()
        last_name = (src.get("last_name") or "").strip()
        full_name = (f"{first_name} {last_name}").strip() or f"Customer {cid}"

        rows.append(
            {
                "customer_id": cid,
                "customer_name": full_name,
                "first_visit": first.isoformat(),
                "last_visit": last.isoformat(),
                "visit_count": vc,
                "typical_interval_days": typical_interval,
                "days_since_last_visit": days_since_last,
                "days_overdue": days_overdue,
                "lifetime_revenue": stat.get("total_revenue"),
            }
        )

    if not rows:
        insight = (
            "No customers tagged 'Top 20%' are currently 14 or more days beyond their "
            "typical visit interval (based on invoice visit history)."
        )
        return {
            "insight": to_json_safe(insight),
            "rows": [],
            "rules_used": business_rules or "",
            "engine": "es",
        }

    # Sort by how overdue they are (most overdue first)
    rows.sort(key=lambda r: r["days_overdue"], reverse=True)

    insight = (
        "Top 20% Customers – Overdue (14+ Days Past Interval) was computed by taking all "
        "customers tagged 'Top 20%' from the customers index, joining them to invoice-based "
        "visit stats, and keeping only those whose days since last visit are at least "
        "14 days beyond their typical interval "
        "((last–first)/(visits–1))."
    )

    return {
        "insight": to_json_safe(insight),
        "rows": to_json_safe(rows),
        "rules_used": business_rules or "",
        "engine": "es",
    }
def _es_coupon_returns_365d_since_signup(
    req: DocsAnalyticsRequest,
    client,
    mappings: Dict[str, Any],          # mappings for the *invoices* index
    business_rules: Optional[str],
):
    """
    Coupon Returns (365+ Days Since Signup)

    Definition (as implemented here):

      - Take original_signup from the customers index (per customer_id).
      - Find customers whose FIRST coupon-bearing invoice occurs
        at least 365 days after original_signup.

      Notes:
        - Only customer_id + original_signup come from the customers index.
          We do NOT use visits_365.
        - Coupon invoices are detected using:
            * coupon field present (string), and/or
            * coupon_total != 0 (if such a numeric field exists).
        - Optional start_date / end_date on dropoff_at restrict the window
          of coupon invoices considered.
    """

    invoices_index = (req.es_index_name or "").strip()
    customers_index = (req.es_customers_index_name or "").strip()

    if not invoices_index or not customers_index:
        return _es_cannot_answer(
            "Coupon Returns (365+ Days Since Signup) requires both an invoices index "
            "(es_index_name) and a customers index (es_customers_index_name).",
            business_rules,
        )

    # -------- 1) resolve invoice-side fields --------
    invoice_customer_field = resolve_es_field(
        mappings,
        user_term="customer_id",
        alias_family="customer",
    )
    invoice_date_field = resolve_es_field(
        mappings,
        user_term="dropoff_at",
        alias_family="date",
    )
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

    # -------- 2) signup dates from CUSTOMERS (original_signup only) --------
    cust_mapping_raw = client.indices.get_mapping(index=customers_index)
    cust_props = _extract_properties_from_mapping(cust_mapping_raw, customers_index)
    cust_mappings = {"properties": cust_props}

    signup_by_customer = _es_get_customer_signups(
        client,
        customers_index,
        cust_mappings,
    )
    if not signup_by_customer:
        return _es_cannot_answer(
            "Cannot compute Coupon Returns (365+ Days Since Signup) because no "
            "original_signup dates could be found in the customers index.",
            business_rules,
        )

    cust_id_field = resolve_es_field(
        cust_mappings,
        user_term="customer_id",
        alias_family="customer",
    )
    first_name_field = resolve_es_field(cust_mappings, user_term="first_name") or "first_name"
    last_name_field = resolve_es_field(cust_mappings, user_term="last_name") or "last_name"

    # Fetch names for customers we have signup dates for
    cust_ids = list(signup_by_customer.keys())
    body_names = {
        "size": 10000,
        "query": {
            "bool": {
                "filter": [
                    {"terms": {cust_id_field: cust_ids}}
                ]
            }
        }
    }
    res_names = client.search(index=customers_index, body=body_names)
    hits_names = (res_names.get("hits") or {}).get("hits") or []

    name_by_id: Dict[Any, str] = {}
    for h in hits_names:
        src = h.get("_source", {}) or {}
        cid = src.get("customer_id")
        if cid is None:
            cid = src.get(cust_id_field.split(".")[-1])
        if cid is None:
            continue

        first = (src.get("first_name") or "").strip()
        last = (src.get("last_name") or "").strip()
        full_name = (f"{first} {last}").strip() or f"Customer {cid}"
        name_by_id[cid] = full_name

    # -------- 3) invoices: first coupon visit per customer --------
    filters = _build_date_range_filter(req, invoice_date_field)
    filters.append({"terms": {invoice_customer_field: cust_ids}})

    coupon_should: List[Dict[str, Any]] = []
    if coupon_field:
        coupon_should.append({"exists": {"field": coupon_field}})
    if coupon_total_field:
        coupon_should.append({"range": {coupon_total_field: {"lt": 0}}})
        coupon_should.append({"range": {coupon_total_field: {"gt": 0}}})

    filters.append(
        {
            "bool": {
                "should": coupon_should,
                "minimum_should_match": 1,
            }
        }
    )

    body_inv = {
        "size": 0,
        "query": {"bool": {"filter": filters}},
        "aggs": {
            "customers": {
                "terms": {
                    "field": invoice_customer_field,
                    "size": 10000,
                },
                "aggs": {
                    "first_coupon_date": {"min": {"field": invoice_date_field}},
                    "coupon_invoice_count": {"value_count": {"field": invoice_date_field}},
                },
            }
        },
    }

    res_inv = client.search(index=invoices_index, body=body_inv)
    buckets = (
        res_inv.get("aggregations", {})
        .get("customers", {})
        .get("buckets", [])
    )

    rows: List[Dict[str, Any]] = []

    for b in buckets:
        cid = b.get("key")
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
            # we only want first coupon visits at least 365 days post-signup
            continue

        coupon_count = int((b.get("coupon_invoice_count") or {}).get("value") or 0)

        rows.append(
            {
                "customer_id": cid,
                "customer_name": name_by_id.get(cid, f"Customer {cid}"),
                "original_signup": signup_date.isoformat(),
                "first_coupon_visit": coupon_date.isoformat(),
                "days_from_signup_to_first_coupon": diff_days,
                "coupon_invoice_count": coupon_count,
            }
        )

    if not rows:
        window_desc = []
        if req.start_date:
            window_desc.append(f"from {req.start_date}")
        if req.end_date:
            window_desc.append(f"to {req.end_date}")
        window_str = " ".join(window_desc) if window_desc else "for the full dataset"

        insight = (
            "No customers were found whose first coupon-bearing invoice occurred at least "
            "365 days after original signup "
            f"{window_str}."
        )
        return {
            "insight": to_json_safe(insight),
            "rows": [],
            "rules_used": business_rules or "",
            "engine": "es",
        }

    # Most extreme long-gap cases first
    rows.sort(
        key=lambda r: r.get("days_from_signup_to_first_coupon", 0),
        reverse=True,
    )

    insight = (
        "Coupon Returns (365+ Days Since Signup) was computed by joining the invoices index "
        f"'{invoices_index}' with the customers index '{customers_index}'. For each customer we "
        "look at their first coupon-bearing invoice and keep only those where that visit occurs "
        "at least 365 days after original signup. Optional start_date/end_date restrict which "
        "coupon visits are considered."
    )

    return {
        "insight": to_json_safe(insight),
        "rows": to_json_safe(rows),
        "rules_used": business_rules or "",
        "engine": "es",
    }


def _es_customer_ltv(
    req: DocsAnalyticsRequest,
    client,
    mappings: Dict[str, Any],
    business_rules: Optional[str],
):
    """
    ES implementation of 'average customer lifetime value' (average customer spend):

      - Group invoices by customer_id
      - Compute LTV per customer = sum(total)
      - Compute:
          total_spend = sum(LTV for customers with LTV > 0)
          customer_count = number of customers with LTV > 0
          avg_ltv = total_spend / customer_count

      This matches:
          Average Customer Spend = Total customer spend ÷ Number of customers with spend (> 0).
    """
    index_name = req.es_index_name.strip()

    # Prefer invoices fields: customer_id + total
    customer_field = resolve_es_field(
        mappings,
        user_term="customer_id",
        alias_family="customer",
    )
    amount_field = resolve_es_field(
        mappings,
        user_term="total",
        alias_family="amount",
    )

    if not (customer_field and amount_field):
        return _es_cannot_answer(
            "Cannot compute 'average customer lifetime value' on this Elasticsearch index "
            "because customer or amount fields could not be found in the mappings.",
            business_rules,
        )

    body = {
        "size": 0,
        "aggs": {
            "customers": {
                "terms": {
                    "field": customer_field,
                    # number of distinct customers to aggregate; can be tuned
                    "size": 10000,
                },
                "aggs": {
                    "ltv": {
                        "sum": {
                            "field": amount_field,
                        }
                    }
                },
            },
        },
    }

    res = client.search(index=index_name, body=body)

    cust_buckets = (
        res.get("aggregations", {})
           .get("customers", {})
           .get("buckets", [])
    )

    rows: List[Dict[str, Any]] = []

    total_spend = 0.0
    customer_count = 0

    for b in cust_buckets:
        ltv_obj = b.get("ltv") or {}
        ltv_val = ltv_obj.get("value")
        if ltv_val is None:
            ltv_val = 0.0

        # table row per customer
        rows.append(
            {
                "customer_id": b.get("key"),
                "ltv": ltv_val,
            }
        )

        # include only customers with LTV > 0 in the average
        if ltv_val > 0:
            total_spend += ltv_val
            customer_count += 1

    avg_ltv = None
    if customer_count > 0:
        avg_ltv = total_spend / float(customer_count)

    insight = (
        f"Customer lifetime value was computed as the sum of '{amount_field}' "
        f"per '{customer_field}' on index '{index_name}'. "
    )
    if avg_ltv is not None:
        insight += (
            f"Total customer spend (LTV > 0) is {total_spend:.2f} across "
            f"{customer_count} customers, giving an average customer spend of "
            f"approximately {avg_ltv:.2f}."
        )
    else:
        insight += (
            "No customers with positive lifetime value were found, so average "
            "customer spend could not be computed."
        )

    return {
        "insight": to_json_safe(insight),
        "rows": to_json_safe(rows),
        "rules_used": business_rules or "",
        "code": body,    # ES body; hide in prod if you like
        "engine": "es",
        "raw": {
            "took": res.get("took"),
            "total": res.get("hits", {}).get("total"),
            "aggregations": res.get("aggregations"),
            "total_spend": total_spend,
            "customer_count_with_spend": customer_count,
            "average_customer_spend": avg_ltv,
        },
    }


def _es_one_time_vs_repeat(
    req: DocsAnalyticsRequest,
    client,
    mappings: Dict[str, Any],
    business_rules: Optional[str],
):
    """One-time vs repeat customers using ES per-customer visit_count.

    Rules:
      - Customers with 0 visits are ignored.
      - One-time  = exactly 1 visit.
      - Repeat    = more than 1 visit (in the time range / dataset).
    """
    index_name = req.es_index_name.strip()
    stats = _es_get_customer_stats(client, index_name, mappings)
    if stats is None:
        return _es_cannot_answer(
            "Cannot compute 'one-time vs repeat customers' because customer or date fields "
            "could not be resolved from the Elasticsearch mappings.",
            business_rules,
        )

    # 🔎 Ignore customers with 0 visits (no usable data)
    stats_nonzero = [
        s for s in stats
        if (s.get("visit_count") or 0) > 0
    ]

    total = len(stats_nonzero)
    if total == 0:
        rows: List[Dict[str, Any]] = []
        insight = (
            "No customers with at least one visit were found, so one-time vs repeat "
            "customers could not be computed."
        )
    else:
        one_time = sum(
            1 for s in stats_nonzero
            if (s.get("visit_count") or 0) == 1
        )
        repeat = sum(
            1 for s in stats_nonzero
            if (s.get("visit_count") or 0) > 1
        )

        one_pct = (one_time * 100.0 / total) if total else 0.0
        repeat_pct = (repeat * 100.0 / total) if total else 0.0

        rows = [
            {
                "segment": "one-time",
                "customer_count": one_time,
                "percentage_of_customers": one_pct,
            },
            {
                "segment": "repeat",
                "customer_count": repeat,
                "percentage_of_customers": repeat_pct,
            },
        ]
        insight = (
            f"Out of {total} customers with at least one visit, about {one_pct:.1f}% are one-time "
            f"and {repeat_pct:.1f}% are repeat (based on ES visit counts). "
            "Customers with zero visits were excluded from this calculation."
        )

    return {
        "insight": to_json_safe(insight),
        "rows": to_json_safe(rows),
        "rules_used": business_rules or "",
        "engine": "es",
    }


def _es_avg_days_between_visits_active(
    req: DocsAnalyticsRequest,
    client,
    mappings: Dict[str, Any],
    business_rules: Optional[str],
):
    """
    Average days between visits for *active* customers, computed at
    company level (no "average of averages").

    Active customer:
      - visit_count >= 2 (at least one interval)
      - last_visit within last 365 days

    Formula:

      For each active customer:
        days_between = (last_visit_date - first_visit_date)
        intervals    = visit_count - 1

      total_days_between = sum(days_between over all active customers)
      total_intervals    = sum(intervals    over all active customers)

      avg_days_between_visits = total_days_between / total_intervals

    This is equivalent to computing the average gap between *all* visits
    across all active customers.
    """
    index_name = req.es_index_name.strip()
    stats = _es_get_customer_stats(client, index_name, mappings)
    if stats is None:
        return _es_cannot_answer(
            "Cannot compute 'average days between visits for active customers' "
            "because customer or date fields could not be resolved from the mappings.",
            business_rules,
        )

    today = datetime.now(timezone.utc).date()

    active_rows: List[Dict[str, Any]] = []
    total_days_between = 0.0
    total_intervals = 0

    for s in stats:
        vc = s["visit_count"] or 0
        first = s["first_visit"]
        last = s["last_visit"]
        if vc <= 1 or not first or not last:
            # need at least 2 visits and valid dates
            continue

        # Only keep active customers (last visit in last 365 days)
        if (today - last.date()).days > 365:
            continue

        days_between = (last.date() - first.date()).days
        intervals = vc - 1

        if intervals <= 0:
            continue

        total_days_between += float(days_between)
        total_intervals += intervals

        # keep per-customer info for the table
        avg_for_customer = days_between / float(intervals)
        active_rows.append(
            {
                "customer_id": s["customer_id"],
                "visits": vc,
                "avg_days_between_visits_customer": avg_for_customer,
                "first_visit": first.isoformat(),
                "last_visit": last.isoformat(),
                "intervals": intervals,
            }
        )

    if total_intervals == 0:
        rows: List[Dict[str, Any]] = []
        insight = (
            "Cannot compute average days between visits for active customers "
            "because there are no customers with at least two visits in the last 365 days."
        )
    else:
        avg_days = total_days_between / float(total_intervals)
        rows = active_rows
        insight = (
            f"For active customers (last visit in the last 365 days), the average gap "
            f"between visits across the whole company is approximately {avg_days:.1f} days. "
            "This is computed by summing all days between the first and last visit for each "
            "active customer and dividing by the total number of visit intervals, not by "
            "taking an average of per-customer averages."
        )

    return {
        "insight": to_json_safe(insight),
        "rows": to_json_safe(rows),
        "rules_used": business_rules or "",
        "engine": "es",
    }


def _es_lapsed_customers(
    req: DocsAnalyticsRequest,
    client,
    mappings: Dict[str, Any],
    business_rules: Optional[str],
    days_threshold: int = 180,
):
    """Customers with no visit in the last N days (default 180)."""
    index_name = req.es_index_name.strip()
    stats = _es_get_customer_stats(client, index_name, mappings)
    if stats is None:
        return _es_cannot_answer(
            "Cannot compute 'lapsed customers' because customer or date fields "
            "could not be resolved from the mappings.",
            business_rules,
        )

    today = datetime.now(timezone.utc).date()
    lapsed: List[Dict[str, Any]] = []

    for s in stats:
        last = s["last_visit"]
        if not last:
            continue
        days_since = (today - last.date()).days
        if days_since > days_threshold:
            lapsed.append(
                {
                    "customer_id": s["customer_id"],
                    "last_visit": last.isoformat(),
                    "days_since_last_visit": days_since,
                }
            )

    total = len(stats)
    lapsed_count = len(lapsed)
    pct = (lapsed_count * 100.0 / total) if total else 0.0

    rows = lapsed
    insight = (
        f"There are {lapsed_count} lapsed customers (no visit in the last {days_threshold} days), "
        f"which is about {pct:.1f}% of all customers (based on ES visit dates)."
    )

    return {
        "insight": to_json_safe(insight),
        "rows": to_json_safe(rows),
        "rules_used": business_rules or "",
        "engine": "es",
    }


def _es_overdue_customers(
    req: DocsAnalyticsRequest,
    client,
    mappings: Dict[str, Any],
    business_rules: Optional[str],
):
    """
    Customers overdue for their next visit:
    - approximate typical interval = (last - first)/(visits-1)
    - overdue if days_since_last_visit > 1.5 * interval.
    """
    index_name = req.es_index_name.strip()
    stats = _es_get_customer_stats(client, index_name, mappings)
    if stats is None:
        return _es_cannot_answer(
            "Cannot compute 'overdue customers' because customer or date fields "
            "could not be resolved from the mappings.",
            business_rules,
        )

    today = datetime.now(timezone.utc).date()
    overdue: List[Dict[str, Any]] = []

    for s in stats:
        vc = s["visit_count"] or 0
        first = s["first_visit"]
        last = s["last_visit"]
        if vc <= 1 or not first or not last:
            continue

        days_between = (last.date() - first.date()).days
        if vc > 1:
            interval = days_between / float(vc - 1)
        else:
            interval = float(days_between)
        if interval <= 0:
            continue

        days_since = (today - last.date()).days
        if days_since > 1.5 * interval:
            overdue.append(
                {
                    "customer_id": s["customer_id"],
                    "last_visit": last.isoformat(),
                    "days_since_last_visit": days_since,
                    "expected_interval_days": interval,
                }
            )

    overdue_count = len(overdue)
    total = len(stats)
    pct = (overdue_count * 100.0 / total) if total else 0.0

    rows = overdue
    insight = (
        f"There are {overdue_count} customers who appear overdue for their next visit "
        f"(days since last visit > 1.5× typical interval), about {pct:.1f}% of all customers."
    )

    return {
        "insight": to_json_safe(insight),
        "rows": to_json_safe(rows),
        "rules_used": business_rules or "",
        "engine": "es",
    }


def _es_visit_frequency_distribution(
    req: DocsAnalyticsRequest,
    client,
    mappings: Dict[str, Any],
    business_rules: Optional[str],
):
    """
    Distribution of customers by visit frequency:
    Buckets: 1 visit, 2–5, 6–11, 12+.
    """
    index_name = req.es_index_name.strip()
    stats = _es_get_customer_stats(client, index_name, mappings)
    if stats is None:
        return _es_cannot_answer(
            "Cannot compute 'distribution of customers by visit frequency' because "
            "customer or date fields could not be resolved from the mappings.",
            business_rules,
        )

    buckets = {
        "1 visit": 0,
        "2–5 visits": 0,
        "6–11 visits": 0,
        "12+ visits": 0,
    }

    for s in stats:
        v = s["visit_count"] or 0
        if v <= 0:
            continue
        if v == 1:
            buckets["1 visit"] += 1
        elif 2 <= v <= 5:
            buckets["2–5 visits"] += 1
        elif 6 <= v <= 11:
            buckets["6–11 visits"] += 1
        else:
            buckets["12+ visits"] += 1

    total = sum(buckets.values())
    rows: List[Dict[str, Any]] = []
    for label in ["1 visit", "2–5 visits", "6–11 visits", "12+ visits"]:
        count = buckets[label]
        pct = (count * 100.0 / total) if total else 0.0
        rows.append(
            {
                "frequency_bucket": label,
                "customer_count": count,
                "percentage_of_customers": pct,
            }
        )

    insight = (
        "Distribution of customers by visit frequency has been computed using ES visit counts "
        "into buckets: 1, 2–5, 6–11, and 12+ visits."
    )

    return {
        "insight": to_json_safe(insight),
        "rows": to_json_safe(rows),
        "rules_used": business_rules or "",
        "engine": "es",
    }


def _es_top_customers_by_revenue(
    req: DocsAnalyticsRequest,
    client,
    mappings: Dict[str, Any],
    business_rules: Optional[str],
):
    """
    Top 5% / Top 20% customers by revenue + how much revenue they represent.
    """
    index_name = req.es_index_name.strip()
    stats = _es_get_customer_stats(client, index_name, mappings)
    if stats is None:
        return _es_cannot_answer(
            "Cannot compute 'top 5% / top 20% customers by revenue' because customer, "
            "date, or amount fields could not be resolved from the mappings.",
            business_rules,
        )

    # Filter to customers with some revenue
    filtered = [s for s in stats if s["total_revenue"] is not None and s["total_revenue"] > 0]
    if not filtered:
        rows: List[Dict[str, Any]] = []
        insight = (
            "Cannot compute top customers by revenue because ES did not return any revenue values."
        )
    else:
        # Sort by revenue desc
        filtered.sort(key=lambda s: s["total_revenue"], reverse=True)
        n = len(filtered)
        top20_n = max(1, int(round(0.20 * n)))
        top5_n = max(1, int(round(0.05 * n)))

        total_revenue = sum(s["total_revenue"] for s in filtered)
        top20_revenue = sum(s["total_revenue"] for s in filtered[:top20_n])
        top5_revenue = sum(s["total_revenue"] for s in filtered[:top5_n])

        if total_revenue <= 0:
            top20_share = 0.0
            top5_share = 0.0
        else:
            top20_share = 100.0 * top20_revenue / total_revenue
            top5_share = 100.0 * top5_revenue / total_revenue

        # label each customer
        rows = []
        for idx, s in enumerate(filtered):
            if idx < top5_n:
                segment = "Top 5%"
            elif idx < top20_n:
                segment = "Top 20%"
            else:
                segment = "Bottom 80%"
            rows.append(
                {
                    "customer_id": s["customer_id"],
                    "segment": segment,
                    "revenue": s["total_revenue"],
                }
            )

        insight = (
            f"Top 20% of customers by revenue contribute about {top20_share:.1f}% of total revenue, "
            f"and the Top 5% contribute about {top5_share:.1f}%. "
            "The result table labels each customer as 'Top 5%', 'Top 20%', or 'Bottom 80%'."
        )

    return {
        "insight": to_json_safe(insight),
        "rows": to_json_safe(rows),
        "rules_used": business_rules or "",
        "engine": "es",
    }


def _es_month_over_month_visits(
    req: DocsAnalyticsRequest,
    client,
    mappings: Dict[str, Any],
    business_rules: Optional[str],
):
    """
    Month-over-month visit volume trend based on ES date_histogram by month.

    When visit_id exists, we count DISTINCT visit_id per month so that:
      - 4 invoices with the same visit_id / dropoff_at = 1 visit
      - 4 invoices with 4 different visit_id / dropoff_at = 4 visits

    If visit_id is not available, we fall back to counting invoice rows.
    """
    index_name = req.es_index_name.strip()
    # 🔧 Prefer dropoff_at as main date for invoices
    date_field = resolve_es_field(
        mappings,
        user_term="dropoff_at",
        alias_family="date",
    )
    if not date_field:
        return _es_cannot_answer(
            "Cannot compute month-over-month visit volume trend because no date field "
            "could be resolved from the Elasticsearch mappings.",
            business_rules,
        )

    # try to resolve visit_id so we can count distinct visits
    visit_field = resolve_es_field(
        mappings,
        user_term="visit_id",
        alias_family="visit",
    )
    has_visit_field = bool(visit_field)

    if has_visit_field:
        body = {
            "size": 0,
            "aggs": {
                "months": {
                    "date_histogram": {
                        "field": date_field,
                        "calendar_interval": "month",
                    },
                    "aggs": {
                        # DISTINCT visits per month
                        "distinct_visits": {
                            "cardinality": {"field": visit_field}
                        }
                    },
                }
            }
        }
    else:
        # fallback: count invoice rows as visits
        body = {
            "size": 0,
            "aggs": {
                "months": {
                    "date_histogram": {
                        "field": date_field,
                        "calendar_interval": "month",
                    }
                }
            }
        }

    res = client.search(index=index_name, body=body)
    buckets = (
        res.get("aggregations", {})
           .get("months", {})
           .get("buckets", [])
    )

    rows: List[Dict[str, Any]] = []
    for b in buckets:
        dt = _ms_to_dt(b.get("key"))
        if not dt:
            continue

        if has_visit_field:
            visit_count = int(
                (b.get("distinct_visits") or {}).get("value") or 0
            )
        else:
            # fallback to document count
            visit_count = int(b.get("doc_count", 0))

        rows.append(
            {
                "month": dt.date().isoformat(),
                "visit_count": visit_count,
            }
        )

    rows.sort(key=lambda r: r["month"])

    if len(rows) >= 2:
        last = rows[-1]
        prev = rows[-2]
        last_visits = last["visit_count"]
        prev_visits = prev["visit_count"]
        if prev_visits == 0:
            change_str = "previous month had zero visits, so change is not comparable."
        else:
            delta = last_visits - prev_visits
            pct = 100.0 * delta / prev_visits
            sign = "increase" if delta >= 0 else "decrease"
            change_str = (
                f"{sign} of {abs(delta)} visits ({pct:+.1f}% vs previous month)."
            )
        insight = (
            "Month-over-month visit volume trend has been computed from ES. "
            f"Last month ({last['month']}) had {last_visits} visits vs {prev_visits} in {prev['month']}; "
            f"{change_str}"
        )
    elif len(rows) == 1:
        insight = (
            "Month-over-month visit volume trend has been computed, "
            "but only a single month of data is available."
        )
    else:
        insight = "No visits found to compute a month-over-month trend."

    return {
        "insight": to_json_safe(insight),
        "rows": to_json_safe(rows),
        "rules_used": business_rules or "",
        "engine": "es",
    }


def _es_seasonal_revenue_patterns(
    req: DocsAnalyticsRequest,
    client,
    mappings: Dict[str, Any],
    business_rules: Optional[str],
):
    """
    Seasonal revenue patterns vs last year using ES date_histogram (monthly) + sum(amount).
    """
    index_name = req.es_index_name.strip()
    # 🔧 Prefer dropoff_at + total for invoices
    date_field = resolve_es_field(
        mappings,
        user_term="dropoff_at",
        alias_family="date",
    )
    amount_field = resolve_es_field(
        mappings,
        user_term="total",
        alias_family="amount",
    )
    if not date_field or not amount_field:
        return _es_cannot_answer(
            "Cannot compute seasonal revenue patterns because date or amount fields "
            "could not be resolved from the Elasticsearch mappings.",
            business_rules,
        )

    body = {
        "size": 0,
        "aggs": {
            "months": {
                "date_histogram": {
                    "field": date_field,
                    "calendar_interval": "month",
                },
                "aggs": {
                    "revenue": {"sum": {"field": amount_field}}
                },
            }
        },
    }

    res = client.search(index=index_name, body=body)
    buckets = (
        res.get("aggregations", {})
           .get("months", {})
           .get("buckets", [])
    )

    rows_raw: List[Dict[str, Any]] = []
    for b in buckets:
        dt = _ms_to_dt(b.get("key"))
        if not dt:
            continue
        year = dt.year
        month = dt.month
        revenue = b.get("revenue", {}).get("value") or 0.0
        rows_raw.append(
            {
                "year": year,
                "month": month,
                "revenue": float(revenue),
            }
        )

    if not rows_raw:
        return {
            "insight": to_json_safe(
                "Cannot compute seasonal revenue patterns because ES did not return any data."
            ),
            "rows": [],
            "rules_used": business_rules or "",
            "engine": "es",
        }

    # focus on last 2 years present in data
    all_years = sorted({r["year"] for r in rows_raw})
    current_year = all_years[-1]
    prev_year = current_year - 1
    rows_two = [r for r in rows_raw if r["year"] in (prev_year, current_year)]

    if not rows_two:
        return {
            "insight": to_json_safe(
                "Cannot compute seasonal revenue patterns because there is not enough data for last two years."
            ),
            "rows": [],
            "rules_used": business_rules or "",
            "engine": "es",
        }

    # build month table
    month_names = {i: datetime(2000, i, 1).strftime("%b") for i in range(1, 13)}
    table: Dict[int, Dict[str, Any]] = {}
    for r in rows_two:
        m = r["month"]
        if m not in table:
            table[m] = {
                "month_num": m,
                "month_name": month_names[m],
                f"revenue_{prev_year}": 0.0,
                f"revenue_{current_year}": 0.0,
            }
        key = f"revenue_{r['year']}"
        table[m][key] += r["revenue"]

    rows = []
    for m in sorted(table.keys()):
        row = table[m]
        ly = row.get(f"revenue_{prev_year}", 0.0)
        cy = row.get(f"revenue_{current_year}", 0.0)
        delta = cy - ly
        pct = (delta * 100.0 / ly) if ly > 0 else None
        rows.append(
            {
                "month_num": m,
                "month_name": row["month_name"],
                f"revenue_{prev_year}": ly,
                f"revenue_{current_year}": cy,
                "yoy_change": delta,
                "yoy_change_pct": pct,
            }
        )

    # overall totals
    total_prev = sum(r[f"revenue_{prev_year}"] for r in rows)
    total_curr = sum(r[f"revenue_{current_year}"] for r in rows)
    if total_prev == 0:
        yoy_text = (
            f"Total revenue in {current_year} was {total_curr:.2f}, "
            f"while {prev_year} had no revenue (cannot compute percentage change)."
        )
    else:
        delta_total = total_curr - total_prev
        pct_total = 100.0 * delta_total / total_prev
        sign_total = "higher" if delta_total >= 0 else "lower"
        yoy_text = (
            f"Total revenue in {current_year} was {total_curr:.2f} vs {total_prev:.2f} in {prev_year}, "
            f"{sign_total} by {abs(delta_total):.2f} ({pct_total:+.1f}% YoY)."
        )

    insight = (
        "Seasonal revenue patterns have been compared month-by-month vs last year using ES aggregates. "
        + yoy_text
    )

    return {
        "insight": to_json_safe(insight),
        "rows": to_json_safe(rows),
        "rules_used": business_rules or "",
        "engine": "es",
    }


def _es_avg_ticket_size(
    req: DocsAnalyticsRequest,
    client,
    mappings: Dict[str, Any],
    business_rules: Optional[str],
):
    """
    average $ per visit by day of week / month of year.

    we treat one visit as one distinct visit_id. if visit_id is not
    available in the index, we fall back to counting invoice rows.
    """
    index_name = req.es_index_name.strip()

    # prefer dropoff_at + total for your invoices index
    date_field = resolve_es_field(
        mappings,
        user_term="dropoff_at",
        alias_family="date",
    )
    if not date_field:
        date_field = resolve_es_field(mappings, alias_family="date")

    amount_field = resolve_es_field(
        mappings,
        user_term="total",
        alias_family="amount",
    )

    # new: resolve visit_id if available
    visit_field = resolve_es_field(
        mappings,
        user_term="visit_id",
        alias_family="visit",
    )

    if not (date_field and amount_field):
        return _es_cannot_answer(
            "cannot compute average $ per visit by day-of-week/month-of-year "
            "because date or amount fields could not be resolved from the "
            "elasticsearch mappings.",
            business_rules,
        )

    # if we have visit_id, use cardinality(visit_id) as visit count.
    # otherwise, fall back to counting documents.
    if visit_field:
        by_dow_aggs = {
            "total_revenue": {"sum": {"field": amount_field}},
            "visit_count": {"cardinality": {"field": visit_field}},
        }
        by_month_aggs = {
            "total_revenue": {"sum": {"field": amount_field}},
            "visit_count": {"cardinality": {"field": visit_field}},
        }
    else:
        by_dow_aggs = {
            "total_revenue": {"sum": {"field": amount_field}},
            "visit_count": {"value_count": {"field": date_field}},
        }
        by_month_aggs = {
            "total_revenue": {"sum": {"field": amount_field}},
            "visit_count": {"value_count": {"field": date_field}},
        }

    body = {
        "size": 0,
        "aggs": {
            "by_dow": {
                "terms": {
                    # 1 (monday) .. 7 (sunday)
                    "script": {
                        "source": f"doc['{date_field}'].value.dayOfWeek",
                        "lang": "painless",
                    },
                    "size": 7,
                    "order": {"_key": "asc"},
                },
                "aggs": by_dow_aggs,
            },
            "by_month": {
                "terms": {
                    # 1..12
                    "script": {
                        "source": f"doc['{date_field}'].value.monthOfYear",
                        "lang": "painless",
                    },
                    "size": 12,
                    "order": {"_key": "asc"},
                },
                "aggs": by_month_aggs,
            },
        },
    }

    res = client.search(index=index_name, body=body)
    aggs = res.get("aggregations", {}) or {}

    dow_buckets = aggs.get("by_dow", {}).get("buckets", [])
    month_buckets = aggs.get("by_month", {}).get("buckets", [])

    DOW_NAMES = {
        1: "monday",
        2: "tuesday",
        3: "wednesday",
        4: "thursday",
        5: "friday",
        6: "saturday",
        7: "sunday",
    }
    MONTH_NAMES = {
        1: "jan", 2: "feb", 3: "mar", 4: "apr",
        5: "may", 6: "jun", 7: "jul", 8: "aug",
        9: "sep", 10: "oct", 11: "nov", 12: "dec",
    }

    rows: List[Dict[str, Any]] = []

    # day-of-week rows (average $ per visit)
    for b in dow_buckets:
        key = int(b.get("key", 0))
        total_rev = (b.get("total_revenue") or {}).get("value") or 0.0
        visits = (b.get("visit_count") or {}).get("value") or 0.0
        avg_per_visit = total_rev / visits if visits > 0 else None

        rows.append(
            {
                "dimension": "day_of_week",
                "label": DOW_NAMES.get(key, str(key)),
                "total_revenue": total_rev,
                "visit_count": visits,
                "avg_value_per_visit": avg_per_visit,
            }
        )

    # month-of-year rows (average $ per visit)
    for b in month_buckets:
        key = int(b.get("key", 0))
        total_rev = (b.get("total_revenue") or {}).get("value") or 0.0
        visits = (b.get("visit_count") or {}).get("value") or 0.0
        avg_per_visit = total_rev / visits if visits > 0 else None

        rows.append(
            {
                "dimension": "month_of_year",
                "label": MONTH_NAMES.get(key, str(key)),
                "total_revenue": total_rev,
                "visit_count": visits,
                "avg_value_per_visit": avg_per_visit,
            }
        )

    insight = (
        f"average $ per visit was computed directly on index '{index_name}' "
        f"by day-of-week and by month-of-year using field '{date_field}' "
        f"for dates and '{amount_field}' for amounts. one visit is treated "
        f"as one distinct visit_id{' (falling back to invoice rows where visit_id is missing)' if not visit_field else ''}."
    )

    return {
        "insight": to_json_safe(insight),
        "rows": to_json_safe(rows),
        "rules_used": business_rules or "",
        "code": body,
        "engine": "es",
        "raw": {
            "took": res.get("took"),
            "total": res.get("hits", {}).get("total"),
            "aggregations": aggs,
        },
    }

def _es_new_customer_acquisition(
    req: DocsAnalyticsRequest,
    client,
    mappings: Dict[str, Any],          # mappings for the *invoices* index
    business_rules: Optional[str],
):
    """
    New Customer Acquisition (per month / quarter) using BOTH indexes:

      - Invoices index:
          * first_visit  (via _es_get_customer_stats)
      - Customers index:
          * original_signup (via _es_get_customer_signups)

    Rules:
      - "New" customers = first_visit is in [start_date, end_date] (if given)
      - 30-day "stale" rule:
            diff_days = first_visit - original_signup
            if original_signup exists AND diff_days > 30  → EXCLUDE customer
            if original_signup is missing                 → INCLUDE customer

      Result:
        rows = [{ "period": "YYYY-MM" or "YYYY-Qn", "new_customers": N }, ...]
    """
    # -----------------------------
    # 0) basic sanity
    # -----------------------------
    invoices_index = (req.es_index_name or "").strip()
    customers_index = (req.es_customers_index_name or "").strip()

    if not invoices_index or not customers_index:
        return _es_cannot_answer(
            "New Customer Acquisition requires both an invoices index "
            "(es_index_name) and a customers index (es_customers_index_name).",
            business_rules,
        )

    # -----------------------------
    # 1) lifetime per-customer stats from INVOICES
    #     → gives us first_visit (and more)
    # -----------------------------
    # mappings passed in here are already the flattened invoices mapping
    invoice_stats = _es_get_customer_stats(client, invoices_index, mappings)
    if invoice_stats is None:
        return _es_cannot_answer(
            "Cannot compute 'New Customer Acquisition' because customer/date fields "
            "could not be resolved from the invoices index mappings.",
            business_rules,
        )

    # -----------------------------
    # 2) signup dates from CUSTOMERS
    # -----------------------------
    # fetch + flatten mapping for the customers index
    cust_mapping_raw = client.indices.get_mapping(index=customers_index)
    cust_props = _extract_properties_from_mapping(cust_mapping_raw, customers_index)
    cust_mappings = {"properties": cust_props}

    signup_by_customer = _es_get_customer_signups(
        client,
        customers_index,
        cust_mappings,
    )

    # -----------------------------
    # 3) date window + 30-day rule
    # -----------------------------
    start_d = _parse_date_str(req.start_date)
    end_d = _parse_date_str(req.end_date)
    max_diff_days = 30

    # decide if user wants month or quarter
    ql = (req.question or "").lower()
    use_quarter = any(p in ql for p in ["quarter", "q1", "q2", "q3", "q4"])

    counts: Dict[str, int] = {}

    for s in invoice_stats:
        cid = s["customer_id"]
        first = s["first_visit"]
        if not first:
            continue

        fd = first.date()

        # window on first_visit
        if start_d and fd < start_d:
            continue
        if end_d and fd > end_d:
            continue

        signup_date = signup_by_customer.get(cid)

        # 30-day "stale" rule
        if signup_date is not None:
            diff_days = (fd - signup_date).days
            # if signup is *after* first_visit, also treat as stale
            if diff_days < 0 or diff_days > max_diff_days:
                continue
        # if signup_date is missing → include (per spec)

        # bucket into month or quarter
        if use_quarter:
            q = (fd.month - 1) // 3 + 1
            label = f"{fd.year}-Q{q}"
        else:
            label = f"{fd.year}-{fd.month:02d}"

        counts[label] = counts.get(label, 0) + 1

    rows = [
        {"period": period, "new_customers": count}
        for period, count in sorted(counts.items())
    ]

    # -----------------------------
    # 4) build insight text
    # -----------------------------
    if req.start_date or req.end_date:
        window_desc = []
        if req.start_date:
            window_desc.append(f"from {req.start_date}")
        if req.end_date:
            window_desc.append(f"to {req.end_date}")
        window_str = " ".join(window_desc)
    else:
        window_str = "for all available history"

    freq_label = "quarter" if use_quarter else "month"

    if not rows:
        insight = (
            f"New Customer Acquisition by {freq_label} could not be computed {window_str} "
            "because no customers met the criteria (first_visit in the period and signup "
            "no more than 30 days earlier when available)."
        )
    else:
        last = rows[-1]
        last_period = last["period"]
        last_count = last["new_customers"]
        if len(rows) >= 2:
            prev = rows[-2]
            prev_count = prev["new_customers"]
            if prev_count == 0:
                change_str = (
                    "the previous period had zero new customers, so percentage "
                    "change is not comparable."
                )
            else:
                delta = last_count - prev_count
                pct = 100.0 * delta / prev_count
                sign = "increase" if delta >= 0 else "decrease"
                change_str = (
                    f"{sign} of {abs(delta)} new customers "
                    f"({pct:+.1f}% vs previous {freq_label})."
                )
        else:
            change_str = "there is only one period of data."

        insight = (
            f"New Customer Acquisition by {freq_label} was computed by joining invoices "
            f"index '{invoices_index}' (first_visit) with customers index '{customers_index}' "
            f"(original_signup). Customers are counted as 'new' only if their first_visit falls "
            f"in the selected window ({window_str}) and, when a signup date exists, it is no more "
            f"than {max_diff_days} days earlier. The most recent {freq_label} ({last_period}) has "
            f"{last_count} new customers; {change_str}"
        )

    return {
        "insight": to_json_safe(insight),
        "rows": to_json_safe(rows),
        "rules_used": business_rules or "",
        "engine": "es",
    }

# NEW: New Customer 30-Day Return Rate
# NEW: New Customer 30-Day Return Rate
def _es_new_customer_30d_return_rate(
    req: DocsAnalyticsRequest,
    client,
    mappings: Dict[str, Any],          # mappings for the *invoices* index
    business_rules: Optional[str],
):
    """
    New Customer 30-Day Return Rate:

      - "New customers" = same definition as _es_new_customer_acquisition:
          * first_visit (lifetime) is in [start_date, end_date] if provided
          * AND, if original_signup exists in the customers index,
            first_visit - original_signup <= 30 days (and not negative).
          * If signup is missing → still treated as new.

      - For each such new customer, look at their 2nd visit (lifetime):
          * If 2nd visit date - 1st visit date <= 30 days → counted as returned.

      - Rate = (customers who returned within 30 days) / (new customers) * 100
    """

    # -----------------------------
    # 0) basic sanity: need *both* indexes
    # -----------------------------
    invoices_index = (req.es_index_name or "").strip()
    customers_index = (req.es_customers_index_name or "").strip()

    if not invoices_index or not customers_index:
        return _es_cannot_answer(
            "New Customer 30-Day Return Rate requires both an invoices index "
            "(es_index_name) and a customers index (es_customers_index_name).",
            business_rules,
        )

    # ES fields for invoices index
    customer_field = resolve_es_field(
        mappings,
        user_term="customer_id",
        alias_family="customer",
    )
    date_field = resolve_es_field(
        mappings,
        user_term="dropoff_at",
        alias_family="date",
    )
    visit_field = resolve_es_field(
        mappings,
        user_term="visit_id",
        alias_family="visit",
    )

    if not (customer_field and date_field):
        return _es_cannot_answer(
            "Cannot compute 'New Customer 30-Day Return Rate' because customer_id "
            "or dropoff_at could not be resolved from the Elasticsearch mappings.",
            business_rules,
        )

    # -----------------------------
    # 1) lifetime per-customer stats from INVOICES
    # -----------------------------
    invoice_stats = _es_get_customer_stats(client, invoices_index, mappings)
    if invoice_stats is None:
        return _es_cannot_answer(
            "Cannot compute 'New Customer 30-Day Return Rate' because customer/date "
            "fields could not be resolved from the invoices index mappings.",
            business_rules,
        )

    # -----------------------------
    # 2) signup dates from CUSTOMERS (original_signup)
    # -----------------------------
    cust_mapping_raw = client.indices.get_mapping(index=customers_index)
    cust_props = _extract_properties_from_mapping(cust_mapping_raw, customers_index)
    cust_mappings = {"properties": cust_props}

    signup_by_customer = _es_get_customer_signups(
        client,
        customers_index,
        cust_mappings,
    )

    # -----------------------------
    # 3) identify "new customers" using SAME rules as _es_new_customer_acquisition
    # -----------------------------
    start_d = _parse_date_str(req.start_date)
    end_d = _parse_date_str(req.end_date)
    max_diff_days = 30

    new_customers: List[Any] = []

    for s in invoice_stats:
        cid = s["customer_id"]
        first = s["first_visit"]
        if not first:
            continue

        fd = first.date()

        # Restrict on first_visit window
        if start_d and fd < start_d:
            continue
        if end_d and fd > end_d:
            continue

        signup_date = signup_by_customer.get(cid)

        # 30-day "stale" rule from spec
        if signup_date is not None:
            diff_days = (fd - signup_date).days
            # if signup is after first_visit OR too far before → not considered "new"
            if diff_days < 0 or diff_days > max_diff_days:
                continue

        new_customers.append(cid)

    if not new_customers:
        insight = (
            "No new customers were found in the specified date window (using the same "
            "definition as New Customer Acquisition), so the 30-day return rate cannot "
            "be computed."
        )
        return {
            "insight": to_json_safe(insight),
            "rows": [],
            "rules_used": business_rules or "",
            "engine": "es",
        }

    # -----------------------------
    # 4) for each new customer, find 1st & 2nd visit dates (lifetime)
    #     - use visit_id if available so we count VISITS, not invoices
    # -----------------------------
    returned_within_30 = 0
    total_new = len(new_customers)

    for cid in new_customers:
        if visit_field:
            # ✅ proper "visit" semantics: distinct visit_id, ordered by first date
            body = {
                "size": 0,
                "query": {
                    "bool": {
                        "filter": [
                            {"term": {customer_field: cid}},
                        ]
                    }
                },
                "aggs": {
                    "visits": {
                        "terms": {
                            "field": visit_field,
                            "size": 2,  # we only need first 2 visits
                            "order": {"first_date": "asc"},
                        },
                        "aggs": {
                            "first_date": {"min": {"field": date_field}}
                        },
                    }
                },
            }

            res = client.search(index=invoices_index, body=body)
            buckets = (
                res.get("aggregations", {})
                   .get("visits", {})
                   .get("buckets", [])
            ) or []

            if len(buckets) < 2:
                # fewer than 2 visits → cannot be counted as "returned"
                continue

            first_ms = (buckets[0].get("first_date") or {}).get("value")
            second_ms = (buckets[1].get("first_date") or {}).get("value")

            dt0 = _ms_to_dt(first_ms)
            dt1 = _ms_to_dt(second_ms)

        else:
            # ❗ fallback: treat each invoice row as a visit (no visit_id available)
            body = {
                "size": 2,
                "query": {
                    "bool": {
                        "filter": [
                            {"term": {customer_field: cid}},
                        ]
                    }
                },
                "sort": [
                    {date_field: {"order": "asc"}}
                ],
                "track_total_hits": False,
                "_source": False,
            }
            res = client.search(index=invoices_index, body=body)
            hits = (res.get("hits") or {}).get("hits") or []
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

        diff_days = (dt1.date() - dt0.date()).days
        if diff_days <= 30:
            returned_within_30 += 1

    rate = (returned_within_30 * 100.0 / total_new) if total_new else 0.0

    rows = [
        {
            "metric": "new_customers_window",
            "label": "New Customers in Window",
            "value": total_new,
        },
        {
            "metric": "new_customers_returned_30d",
            "label": "New Customers Returning within 30 Days (lifetime visits)",
            "value": returned_within_30,
        },
        {
            "metric": "new_customer_30d_return_rate",
            "label": "New Customer 30-Day Return Rate (%)",
            "value": rate,
        },
    ]

    window_desc = []
    if req.start_date:
        window_desc.append(f"from {req.start_date}")
    if req.end_date:
        window_desc.append(f"to {req.end_date}")
    window_str = " ".join(window_desc) if window_desc else "for the full dataset"

    insight = (
        f"New Customer 30-Day Return Rate was computed {window_str} using the same "
        f'\"new customer\" definition as New Customer Acquisition (first_visit in the '
        f"window and, when available, signup no more than {max_diff_days} days earlier). "
        f"For those new customers, we then checked whether their second visit occurred "
        f"within 30 days of the first. The rate is the percentage of such new customers "
        f"who returned within 30 days."
    )

    return {
        "insight": to_json_safe(insight),
        "rows": to_json_safe(rows),
        "rules_used": business_rules or "",
        "engine": "es",
    }


# NEW: Customers Achieving 2nd / 3rd / 4th / 5th Visit (lifetime)
def _es_customers_nth_visit(
    req: DocsAnalyticsRequest,
    client,
    mappings: Dict[str, Any],
    business_rules: Optional[str],
):
    """
    Lifetime counts of customers who have reached at least their
    2nd / 3rd / 4th / 5th visit.

    NOTE:
      - Uses lifetime visit_count from _es_get_customer_stats.
      - Not windowed; it's "as of all available data".
    """
    index_name = req.es_index_name.strip()
    stats = _es_get_customer_stats(client, index_name, mappings)
    if stats is None:
        return _es_cannot_answer(
            "Cannot compute 'Customers Achieving Nth Visit' because customer or date "
            "fields could not be resolved from the mappings.",
            business_rules,
        )

    total_customers = len(stats)
    c2 = c3 = c4 = c5 = 0

    for s in stats:
        vc = s.get("visit_count") or 0
        if vc >= 2:
            c2 += 1
        if vc >= 3:
            c3 += 1
        if vc >= 4:
            c4 += 1
        if vc >= 5:
            c5 += 1

    rows = [
        {
            "metric": "customers_2plus_visits",
            "label": "Customers Achieving 2nd Visit (≥2 visits)",
            "value": c2,
        },
        {
            "metric": "customers_3plus_visits",
            "label": "Customers Achieving 3rd Visit (≥3 visits)",
            "value": c3,
        },
        {
            "metric": "customers_4plus_visits",
            "label": "Customers Achieving 4th Visit (≥4 visits)",
            "value": c4,
        },
        {
            "metric": "customers_5plus_visits",
            "label": "Customers Achieving 5th Visit (≥5 visits)",
            "value": c5,
        },
        {
            "metric": "total_customers_lifetime",
            "label": "Total Customers (lifetime)",
            "value": total_customers,
        },
    ]

    insight = (
        f"Customers achieving their 2nd, 3rd, 4th and 5th visit were computed on index "
        f"'{index_name}' using lifetime visit_count per customer. This is not restricted "
        f"to a date window; it is based on all available historical data."
    )

    return {
        "insight": to_json_safe(insight),
        "rows": to_json_safe(rows),
        "rules_used": business_rules or "",
        "engine": "es",
    }


def _es_yoy_revenue_by_location(
    req: DocsAnalyticsRequest,
    client,
    mappings: Dict[str, Any],
    business_rules: Optional[str],
):
    """
    Year-over-year revenue growth by location using:
      - terms on location_id
      - date_histogram (year) + sum(amount)
    """
    index_name = req.es_index_name.strip()

    # 🔧 Prefer invoices: dropoff_at + total + location_id
    date_field = resolve_es_field(
        mappings,
        user_term="dropoff_at",
        alias_family="date",
    )
    amount_field = resolve_es_field(
        mappings,
        user_term="total",
        alias_family="amount",
    )
    loc_field = (
        resolve_es_field(
            mappings,
            user_term="location_id",
            alias_family="location_id",
        )
        or resolve_es_field(mappings, alias_family="location")
    )

    if not date_field or not amount_field or not loc_field:
        return _es_cannot_answer(
            "Cannot compute year-over-year revenue growth by location because date, amount, "
            "or location fields could not be resolved from the Elasticsearch mappings.",
            business_rules,
        )

    body = {
        "size": 0,
        "aggs": {
            "locations": {
                "terms": {
                    "field": loc_field,
                    "size": 10000,
                },
                "aggs": {
                    "by_year": {
                        "date_histogram": {
                            "field": date_field,
                            "calendar_interval": "year",
                        },
                        "aggs": {
                            "revenue": {"sum": {"field": amount_field}}
                        },
                    }
                },
            }
        },
    }

    res = client.search(index=index_name, body=body)
    loc_buckets = (
        res.get("aggregations", {})
           .get("locations", {})
           .get("buckets", [])
    )

    rows: List[Dict[str, Any]] = []

    for lb in loc_buckets:
        loc_key = lb.get("key")
        year_buckets = (
            lb.get("by_year", {})
              .get("buckets", [])
        )
        # sort by year
        year_buckets = sorted(year_buckets, key=lambda b: b.get("key", 0))
        prev_rev = None
        prev_year = None
        for b in year_buckets:
            dt = _ms_to_dt(b.get("key"))
            if not dt:
                continue
            year = dt.year
            rev = float(b.get("revenue", {}).get("value") or 0.0)
            if prev_rev is not None and prev_year is not None:
                delta = rev - prev_rev
                pct = (delta * 100.0 / prev_rev) if prev_rev > 0 else None
                rows.append(
                    {
                        "location_id": loc_key,
                        "year": year,
                        "revenue": rev,
                        "prev_year": prev_year,
                        "prev_year_revenue": prev_rev,
                        "yoy_change": delta,
                        "yoy_change_pct": pct,
                    }
                )
            prev_rev = rev
            prev_year = year

    if not rows:
        insight = (
            "Year-over-year revenue growth by location could not be computed because "
            "ES did not return enough data."
        )
    else:
        # overall mean YoY for latest year in the data
        latest_year = max(r["year"] for r in rows)
        latest_rows = [r for r in rows if r["year"] == latest_year and r["yoy_change_pct"] is not None]
        if latest_rows:
            mean_yoy = sum(r["yoy_change_pct"] for r in latest_rows) / len(latest_rows)
            insight = (
                f"Year-over-year revenue growth by location has been computed using ES. "
                f"For the most recent year ({latest_year}), average YoY change across locations "
                f"is approximately {mean_yoy:+.1f}%."
            )
        else:
            insight = (
                "Year-over-year revenue by location has been computed, but no previous-year baseline is "
                "available to calculate YoY change."
            )

    return {
        "insight": to_json_safe(insight),
        "rows": to_json_safe(rows),
        "rules_used": business_rules or "",
        "engine": "es",
    }


# -------------------------------------------------------------------
# ES special router: known questions → custom ES logic
# -------------------------------------------------------------------
def _route_es_special(
    req: DocsAnalyticsRequest,
    client,
    mappings: Dict[str, Any],
    business_rules: Optional[str],
):
    """
    Look at req.question and, if it matches a known pattern,
    run a dedicated ES implementation.

    Return:
      - a response dict OR
      - None if no ES special handler should be used.
    """
    q_lower = (req.question or "").lower()

    # 0) Core visit metrics (Total Visit Amount / Pieces / Visits / Unique Customers)
    if (
        "total visit amount" in q_lower
        or "total visit pieces" in q_lower
        or ("total visits" in q_lower and "unique customers" in q_lower)
        or "core visit metrics" in q_lower
        or "core 4 metrics" in q_lower
        or "core four metrics" in q_lower
    ):
        return _es_core_visit_metrics(req, client, mappings, business_rules)

    # 1) Average customer lifetime value (CLV)
    if "average customer lifetime value" in q_lower or "average clv" in q_lower:
        return _es_customer_ltv(req, client, mappings, business_rules)

    # 1b) NEW: per-customer value metrics
    if (
        "average visits per customer" in q_lower
        or "avg visits per customer" in q_lower
        or "visit pieces per customer" in q_lower
        or "revenue per customer" in q_lower
        or "avg $ per piece" in q_lower
        or "average $ per piece" in q_lower
        or "average dollars per piece" in q_lower
    ):
        return _es_customer_value_metrics(req, client, mappings, business_rules)

    # 1c) NEW: Average Pickup Delay (Retail)
    if "average pickup delay" in q_lower:
        return _es_avg_pickup_delay_retail(req, client, mappings, business_rules)

    # 1d) NEW: Top 20% Customers with Redo/Courtesy Items (uses customer tags)
    if (
        "top 20%" in q_lower
        and ("redo" in q_lower or "courtesy" in q_lower)
        and "customer" in q_lower
    ):
        return _es_top20_customers_with_redo_courtesy(
            req,
            client,
            mappings,
            business_rules,
        )

    # 1d-bis) NEW: Top 20% Customers – Overdue (14+ Days Past Interval)
    if (
        "top 20%" in q_lower
        and "overdue" in q_lower
        and "customer" in q_lower
    ):
        return _es_top20_customers_overdue_14d(
            req,
            client,
            mappings,
            business_rules,
        )

    # 1e) Invoices with Redo Items (simple count)
    if "redo" in q_lower and "invoice" in q_lower:
        return _es_invoices_with_redo_items(req, client, mappings, business_rules)

    # 2) One-time vs repeat customers
    if "one-time" in q_lower and "repeat" in q_lower and "customer" in q_lower:
        return _es_one_time_vs_repeat(req, client, mappings, business_rules)

    # 3) Average days between visits (active customers)
    if "average days between visits" in q_lower and "active customers" in q_lower:
        return _es_avg_days_between_visits_active(req, client, mappings, business_rules)

    # 4) Overdue for next visit
    if "overdue for their next visit" in q_lower or "overdue for next visit" in q_lower:
        return _es_overdue_customers(req, client, mappings, business_rules)

    # 5) Lapsed customers
    if "lapsed customers" in q_lower or (">180 days" in q_lower and "last visit" in q_lower):
        return _es_lapsed_customers(req, client, mappings, business_rules)

    # 6) Distribution by visit frequency
    if "distribution of customers by visit frequency" in q_lower or (
        "visit frequency" in q_lower and "1, 2–5, 6–11, 12+" in q_lower
    ):
        return _es_visit_frequency_distribution(req, client, mappings, business_rules)

    # 7) Top 5% / Top 20% customers by revenue
    if (
        ("top 5%" in q_lower and "top 20%" in q_lower and "revenue" in q_lower)
        or ("top 5 percent" in q_lower and "top 20 percent" in q_lower and "revenue" in q_lower)
        or ("which customers fall into the top 5%" in q_lower)
        or ("top 20%" in q_lower and "revenue" in q_lower)
        or ("top 20 percent" in q_lower and "revenue" in q_lower)
        or ("percentage of revenue comes from the top 20%" in q_lower)
    ):
        return _es_top_customers_by_revenue(req, client, mappings, business_rules)

    # 8) Month-over-month visit volume trend
    if ("month-over-month" in q_lower or "month over month" in q_lower) and "visit" in q_lower:
        return _es_month_over_month_visits(req, client, mappings, business_rules)

    # 9) Seasonal patterns vs last year (revenue)
    if "seasonal patterns" in q_lower or ("seasonal" in q_lower and "last year" in q_lower):
        return _es_seasonal_revenue_patterns(req, client, mappings, business_rules)

    # 10) Average ticket size by day-of-week / month-of-year
    if "average ticket size" in q_lower and (
        "day of week" in q_lower
        or "day-of-week" in q_lower
        or "dow" in q_lower
        or "month of year" in q_lower
        or "month" in q_lower
        or "day" in q_lower
    ):
        return _es_avg_ticket_size(req, client, mappings, business_rules)

    # 11) New customer acquisition rate
    if "new customer acquisition rate" in q_lower or (
        "new customer" in q_lower and "acquisition" in q_lower
    ):
        return _es_new_customer_acquisition(req, client, mappings, business_rules)

    # 11b) NEW: New Customer 30-Day Return Rate
    if (
        "new customer" in q_lower
        and "30" in q_lower
        and "day" in q_lower
        and ("return" in q_lower or "retention" in q_lower)
    ):
        return _es_new_customer_30d_return_rate(req, client, mappings, business_rules)

    # 11c) NEW: Customers Achieving 2nd / 3rd / 4th / 5th Visit
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
        return _es_customers_nth_visit(req, client, mappings, business_rules)

    # 11d) NEW: Coupon Returns (365+ Days Since Signup)
    if (
        "coupon" in q_lower
        and ("return" in q_lower or "returns" in q_lower)
        and "365" in q_lower
        and "signup" in q_lower
    ):
        return _es_coupon_returns_365d_since_signup(
            req,
            client,
            mappings,
            business_rules,
        )

    # 12) Year-over-year revenue growth by location
    if (
        ("year-over-year" in q_lower or "year over year" in q_lower or "yoy" in q_lower)
        and "revenue" in q_lower
        and "location" in q_lower
    ):
        return _es_yoy_revenue_by_location(req, client, mappings, business_rules)

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

    client = make_es_client(
        req.es_base_url,
        req.es_username,
        req.es_password,
    )

    if not client.ping():
        raise HTTPException(
            status_code=400,
            detail=f"Could not ping Elasticsearch at {req.es_base_url}",
        )

    api_key = (req.api_key or "").strip() or os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise HTTPException(
            status_code=500,
            detail="No OpenAI API key configured for ES mode.",
        )

    index_name = req.es_index_name.strip()

    # 1) Get mapping & build simplified properties dict
    mapping = client.indices.get_mapping(index=index_name)
    properties = _extract_properties_from_mapping(mapping, index_name)
    mappings = {"properties": properties}

    # 2) optional business rules when mode='documents'
    business_rules: Optional[str] = None
    if req.mode == "documents":
        workspace_id = (req.workspace_id or "default").strip() or "default"
        business_rules = get_business_rules(
            workspace_id=workspace_id,
            question=req.question,
            doc_ids=req.doc_ids,
        )
        if not (business_rules or "").strip():
            raise HTTPException(
                status_code=400,
                detail="No business rules found in documents for this question.",
            )

    # 3) try ES-special handlers (fast path for known questions)
    special_resp = _route_es_special(req, client, mappings, business_rules)
    if special_resp is not None:
        return special_resp

    # 4) generic ES path via LLM-generated DSL
    dsl_text = llm_generate_es_query(
        question=req.question,
        index_name=index_name,
        mappings=mappings,
        model=req.model,
        api_key=api_key,
    )

    # 5) Parse DSL into (index, body)
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
            "code": dsl_text,
            "engine": "es",
        }

    index_to_use = index_from_dsl or index_name

    # 6) Execute query on ES
    try:
        res = client.search(index=index_to_use, body=body)
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error executing ES search: {e}",
        )

    hits = res.get("hits", {}).get("hits", [])
    docs = [h.get("_source", {}) for h in hits]

    # Optional: flatten hits into rows
    rows = _flatten_docs_to_rows(docs) if docs else []

    insight = (
        f"Answer computed directly on Elasticsearch index '{index_to_use}' "
        f"using an ES aggregation/search query."
    )

    return {
        "insight": to_json_safe(insight),
        "rows": to_json_safe(rows),
        "rules_used": business_rules or "",
        "code": dsl_text,    # currently expose the DSL; hide later if you want
        "engine": "es",
        "raw": {
            "took": res.get("took"),
            "total": res.get("hits", {}).get("total"),
            "aggregations": res.get("aggregations"),
        },
    }


# -------------------------------------------------------------------
# Python path: current logic (mdf + llm.codegen + Pandas)
# -------------------------------------------------------------------

def _ask_via_python(req: DocsAnalyticsRequest):
    # ------------------------------------------------------------
    # 1) Workspace + tables
    # ------------------------------------------------------------
    workspace_id = (req.workspace_id or "default").strip() or "default"

    tables = TABLE_STORE.get_tables(workspace_id)
    if not tables:
        raise HTTPException(
            status_code=400,
            detail="No tables loaded for this workspace_id.",
        )

    # ------------------------------------------------------------
    # 2) Select rules STRICTLY based on mode
    # ------------------------------------------------------------
    if req.mode == "predefined":
        # 🚫 Ignore documents completely
        business_rules = None

    elif req.mode == "documents":
        # ✅ Documents are REQUIRED
        business_rules = get_business_rules(
            workspace_id=workspace_id,
            question=req.question,
            doc_ids=req.doc_ids,
        )
        if not (business_rules or "").strip():
            raise HTTPException(
                status_code=400,
                detail="No business rules found in documents for this question.",
            )

    else:
        raise HTTPException(
            status_code=400,
            detail="Invalid mode. Use 'predefined' or 'documents'.",
        )

    # ------------------------------------------------------------
    # 3) Resolve API key
    # ------------------------------------------------------------
    api_key = (req.api_key or "").strip() or os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise HTTPException(
            status_code=500,
            detail="No OpenAI API key configured (set OPENAI_API_KEY on the server or pass api_key).",
        )

    # ------------------------------------------------------------
    # 4) Generate analytics code
    # ------------------------------------------------------------
    code = llm_codegen(
        question=req.question,
        tables=tables,
        model=req.model,
        api_key=api_key,
        business_rules=business_rules,
    )

    if not code:
        raise HTTPException(
            status_code=500,
            detail="Failed to generate analytics code.",
        )

    # ------------------------------------------------------------
    # 5) Execute generated code
    # ------------------------------------------------------------
    result_df, insight = run_generated_code(code, tables)

    # ✅ Make rows JSON-safe (fix NaN/Inf crash)
    safe_rows = to_json_safe(result_df)

    # ------------------------------------------------------------
    # 6) Return response
    # ------------------------------------------------------------
    return {
        "insight": to_json_safe(insight),
        "rows": safe_rows,
        "rules_used": business_rules or "",
        "code": code,      # ⚠️ remove in production
        "engine": "python",
    }


# -------------------------------------------------------------------
# Dashboard endpoint
# -------------------------------------------------------------------

@router.post("/dashboard")
def es_dashboard(req: MetricsDashboardRequest):
    """
    Returns dashboard metrics (current + optional previous window)
    for a given ES index, using _es_customer_value_metrics under the hood.
    """
    if not req.es_base_url or not req.es_index_name:
        raise HTTPException(
            status_code=400,
            detail="es_base_url and es_index_name are required",
        )

    # 1) ES client + mappings
    client = make_es_client(
        req.es_base_url,
        req.es_username,
        req.es_password,
    )
    if not client.ping():
        raise HTTPException(
            status_code=400,
            detail=f"Could not ping Elasticsearch at {req.es_base_url}",
        )

    index_name = req.es_index_name.strip()
    mapping = client.indices.get_mapping(index=index_name)
    properties = _extract_properties_from_mapping(mapping, index_name)
    mappings = {"properties": properties}

    # 2) base DocsAnalyticsRequest (we only use it for ES helpers)
    base_docs_req = DocsAnalyticsRequest(
        workspace_id=req.workspace_id or "default",
        question="dashboard metrics",
        es_base_url=req.es_base_url,
        es_username=req.es_username,
        es_password=req.es_password,
        es_index_name=req.es_index_name,
    )

    # 3) metrics for current + previous windows
    current_vals = _window_customer_value_metrics(
        base_docs_req, req.current, client, mappings
    )
    previous_vals: Dict[str, float] = {}
    if req.previous:
        previous_vals = _window_customer_value_metrics(
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

        # ✅ NEW:
        "initial_visit_amount": "Initial Visit – Amount",
        "initial_visit_pieces": "Initial Visit – Pieces",
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
        "current_period": {
            "start_date": req.current.start_date,
            "end_date": req.current.end_date,
        },
        "previous_period": (
            {
                "start_date": req.previous.start_date,
                "end_date": req.previous.end_date,
            }
            if req.previous
            else None
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
        ALWAYS go through the ES engine:

          * _ask_via_es() will:
              - try ES special handlers in _route_es_special()
              - if no special handler matches, it will call
                llm_generate_es_query() to build a generic ES DSL query
                using the index mappings, then execute it on ES.

      - If ES is NOT configured on the request, fall back to the
        existing Python + Pandas engine (_ask_via_python).
    """

    # 👇 prefer ES whenever we have an ES index configured
    if req.es_base_url and req.es_index_name:
        return _ask_via_es(req)

    # 👇 fallback: old behavior (TABLE_STORE + llm_codegen + pandas)
    return _ask_via_python(req)
