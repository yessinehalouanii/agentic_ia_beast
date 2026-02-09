from __future__ import annotations

from datetime import datetime, timezone, timedelta
from typing import Any, Dict, List, Optional, Tuple

from abi.runtime import to_json_safe
from app.api.metrics.shared_utilities import (
    _field_exists,
    _safe_es_search,
    _get_invoice_index_and_mappings,  # ✅ reuse-aware helper
)
from app.api.docs_analytics_routes import (
    _ms_to_dt,
    _es_cannot_answer,
    _build_date_range_filter,
)


def _invoice_index_and_mappings(
    client,
    index_in: str,
    mappings: Optional[Dict[str, Any]],
) -> Tuple[str, Dict[str, Any]]:
    """
    ✅ NEW: Always use shared utility so we:
      - reuse provided mappings if already fetched (dashboard path)
      - otherwise resolve alias/wildcard to a concrete index + fetch mappings once (ask-analytics path)
    """
    return _get_invoice_index_and_mappings(
        client,
        index_in,
        existing_mappings=mappings,
        existing_index=index_in,
    )


# -------------------------------------------------------------------
# Window helpers (single approach used by all metrics here)
# -------------------------------------------------------------------
class _WindowReq:
    """
    Wrapper that injects start_date/end_date without mutating the original req.
    Anything else is delegated to the original req.
    """
    def __init__(self, base: Any, start_date: Optional[str], end_date: Optional[str]):
        self._base = base
        self.start_date = start_date
        self.end_date = end_date

    def __getattr__(self, name: str) -> Any:
        return getattr(self._base, name)


def _with_default_window(req: Any, *, default_days: int) -> Any:
    """
    If req has neither start_date nor end_date, apply:
      [today-default_days, today]
    """
    has_start = bool(getattr(req, "start_date", None))
    has_end = bool(getattr(req, "end_date", None))
    if has_start or has_end:
        return req

    today = datetime.now(timezone.utc).date()
    start = today - timedelta(days=default_days)
    return _WindowReq(req, start_date=start.isoformat(), end_date=today.isoformat())


def _window_label(reqw: Any, *, default_days: int) -> str:
    s = getattr(reqw, "start_date", None)
    e = getattr(reqw, "end_date", None)
    if s and e:
        return f"{s} → {e}"
    if s:
        return f"since {s}"
    if e:
        return f"until {e}"
    return f"last {default_days} days"


# -------------------------------------------------------------------
# Metrics
# -------------------------------------------------------------------
def _es_month_over_month_visits(
    req,
    client,
    mappings: Dict[str, Any],
    business_rules: Optional[str],
):
    """
    Month-over-month visit volume trend based on ES date_histogram by month.

    Uses:
      - dropoff_at
      - cardinality(visit_id) per month
    """
    index_in = (getattr(req, "es_index_name", "") or "").strip()
    if not index_in:
        return _es_cannot_answer("Missing es_index_name.", business_rules)

    index_name, idx_mappings = _invoice_index_and_mappings(client, index_in, mappings)

    date_field = "dropoff_at"
    if not _field_exists(idx_mappings, date_field):
        return _es_cannot_answer(
            "Cannot compute month-over-month visits because required field 'dropoff_at' "
            "is missing from the invoices mapping.",
            business_rules,
        )

    visit_field = "visit_id"
    if not _field_exists(idx_mappings, visit_field):
        return _es_cannot_answer(
            "Cannot compute month-over-month visits because required field 'visit_id' "
            "is missing from the invoices mapping.",
            business_rules,
        )

    # default window (~13 months)
    reqw = _with_default_window(req, default_days=400)

    filters = _build_date_range_filter(reqw, date_field) or []
    query = {"bool": {"filter": filters}} if filters else None

    body: Dict[str, Any] = {
        "size": 0,
        "aggs": {
            "months": {
                "date_histogram": {"field": date_field, "calendar_interval": "month"},
                "aggs": {"distinct_visits": {"cardinality": {"field": visit_field}}},
            }
        },
    }
    if query:
        body["query"] = query

    res = _safe_es_search(client, index=index_name, body=body)
    buckets = ((res.get("aggregations") or {}).get("months") or {}).get("buckets", []) or []

    rows: List[Dict[str, Any]] = []
    for b in buckets:
        dt = _ms_to_dt(b.get("key"))
        if not dt:
            continue
        visit_count = int(((b.get("distinct_visits") or {}).get("value")) or 0)
        rows.append({"month": dt.date().isoformat(), "visit_count": visit_count})

    rows.sort(key=lambda r: r["month"])

    if len(rows) >= 2:
        last, prev = rows[-1], rows[-2]
        last_visits, prev_visits = last["visit_count"], prev["visit_count"]
        if prev_visits == 0:
            change_str = "previous month had zero visits, so change is not comparable."
        else:
            delta = last_visits - prev_visits
            pct = 100.0 * delta / prev_visits
            sign = "increase" if delta >= 0 else "decrease"
            change_str = f"{sign} of {abs(delta)} visits ({pct:+.1f}% vs previous month)."
        insight = (
            "Month-over-month visit trend computed from ES. "
            f"Last month ({last['month']}) had {last_visits} visits vs {prev_visits} in {prev['month']}; {change_str}"
        )
    elif len(rows) == 1:
        insight = "Month-over-month visit trend computed, but only one month of data is available."
    else:
        insight = "No visits found to compute a month-over-month trend."

    return {
        "insight": to_json_safe(insight),
        "rows": to_json_safe(rows),
        "rules_used": business_rules or "",
        "engine": "es",
    }


def _es_seasonal_revenue_patterns(
    req,
    client,
    mappings: Dict[str, Any],
    business_rules: Optional[str],
):
    """
    Seasonal revenue patterns vs last year using monthly date_histogram + sum(total).

    Uses:
      - dropoff_at
      - total
    """
    index_in = (getattr(req, "es_index_name", "") or "").strip()
    if not index_in:
        return _es_cannot_answer("Missing es_index_name.", business_rules)

    index_name, idx_mappings = _invoice_index_and_mappings(client, index_in, mappings)

    date_field = "dropoff_at"
    amount_field = "total"
    if not _field_exists(idx_mappings, date_field) or not _field_exists(idx_mappings, amount_field):
        return _es_cannot_answer(
            "Cannot compute seasonal revenue patterns because required fields 'dropoff_at' "
            "and/or 'total' are missing from the invoices mapping.",
            business_rules,
        )

    # default window (~26 months)
    reqw = _with_default_window(req, default_days=800)

    filters = _build_date_range_filter(reqw, date_field) or []
    query = {"bool": {"filter": filters}} if filters else None

    body: Dict[str, Any] = {
        "size": 0,
        "aggs": {
            "months": {
                "date_histogram": {"field": date_field, "calendar_interval": "month"},
                "aggs": {"revenue": {"sum": {"field": amount_field}}},
            }
        },
    }
    if query:
        body["query"] = query

    res = _safe_es_search(client, index=index_name, body=body)
    buckets = ((res.get("aggregations") or {}).get("months") or {}).get("buckets", []) or []

    rows_raw: List[Dict[str, Any]] = []
    for b in buckets:
        dt = _ms_to_dt(b.get("key"))
        if not dt:
            continue
        revenue = ((b.get("revenue") or {}).get("value")) or 0.0
        rows_raw.append({"year": dt.year, "month": dt.month, "revenue": float(revenue)})

    if not rows_raw:
        return {
            "insight": to_json_safe("Cannot compute seasonal revenue patterns because ES returned no data."),
            "rows": [],
            "rules_used": business_rules or "",
            "engine": "es",
        }

    all_years = sorted({r["year"] for r in rows_raw})
    current_year = all_years[-1]
    prev_year = current_year - 1

    rows_two = [r for r in rows_raw if r["year"] in (prev_year, current_year)]
    if not rows_two:
        return {
            "insight": to_json_safe("Not enough data to compare the last two years."),
            "rows": [],
            "rules_used": business_rules or "",
            "engine": "es",
        }

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
        table[m][f"revenue_{r['year']}"] += r["revenue"]

    rows: List[Dict[str, Any]] = []
    for m in sorted(table.keys()):
        row = table[m]
        ly = float(row.get(f"revenue_{prev_year}", 0.0) or 0.0)
        cy = float(row.get(f"revenue_{current_year}", 0.0) or 0.0)
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

    total_prev = sum(r[f"revenue_{prev_year}"] for r in rows)
    total_curr = sum(r[f"revenue_{current_year}"] for r in rows)
    if total_prev == 0:
        yoy_text = f"Total revenue in {current_year} was {total_curr:.2f}, while {prev_year} had no revenue."
    else:
        delta_total = total_curr - total_prev
        pct_total = 100.0 * delta_total / total_prev
        sign_total = "higher" if delta_total >= 0 else "lower"
        yoy_text = (
            f"Total revenue in {current_year} was {total_curr:.2f} vs {total_prev:.2f} in {prev_year}, "
            f"{sign_total} by {abs(delta_total):.2f} ({pct_total:+.1f}% YoY)."
        )

    insight = "Seasonal revenue patterns compared month-by-month vs last year using ES aggregates. " + yoy_text

    return {
        "insight": to_json_safe(insight),
        "rows": to_json_safe(rows),
        "rules_used": business_rules or "",
        "engine": "es",
    }


def _es_dropoff_visits(req, client, mappings: Dict[str, Any], business_rules: Optional[str]):
    index_in = (getattr(req, "es_index_name", "") or "").strip()
    if not index_in:
        return _es_cannot_answer("Dropoff Visits requires invoices index (es_index_name).", business_rules)

    index_name, idx_mappings = _invoice_index_and_mappings(client, index_in, mappings)

    date_field = "dropoff_at"
    visit_id_field = "visit_id"
    invoice_id_field = "invoice_id"
    customer_id_field = "customer_id"

    missing: List[str] = []
    if not _field_exists(idx_mappings, date_field):
        missing.append(date_field)
    if not _field_exists(idx_mappings, visit_id_field):
        missing.append(visit_id_field)
    if not _field_exists(idx_mappings, invoice_id_field):
        missing.append(invoice_id_field)
    if not _field_exists(idx_mappings, customer_id_field):
        missing.append(customer_id_field)

    if missing:
        return _es_cannot_answer(
            "Cannot compute Dropoff Visits because required invoices fields are missing: " + ", ".join(missing),
            business_rules,
        )

    reqw = _with_default_window(req, default_days=365)
    filters = _build_date_range_filter(reqw, date_field) or []
    window_label = _window_label(reqw, default_days=365)

    filters.append({"exists": {"field": date_field}})
    filters.append({"exists": {"field": visit_id_field}})

    body = {
        "size": 0,
        "query": {"bool": {"filter": filters}},
        "aggs": {
            "dropoff_visits": {"cardinality": {"field": visit_id_field}},
            "invoice_count": {"value_count": {"field": invoice_id_field}},
            "unique_customers": {"cardinality": {"field": customer_id_field}},
        },
    }

    res = _safe_es_search(client, index=index_name, body=body)
    aggs = res.get("aggregations") or {}

    visits = int((aggs.get("dropoff_visits") or {}).get("value") or 0)
    invoices = int((aggs.get("invoice_count") or {}).get("value") or 0)
    customers = int((aggs.get("unique_customers") or {}).get("value") or 0)

    rows = [
        {
            "metric": "dropoff_visits",
            "label": "Dropoff Visits",
            "value": float(visits),
            "invoice_count": invoices,
            "unique_customers": customers,
            "window": window_label,
            "date_field": date_field,
            "visit_id_field": visit_id_field,
        }
    ]

    insight = (
        f"Dropoff Visits is computed as DISTINCT invoices.{visit_id_field} "
        f"over invoices.{date_field} in ({window_label}). "
        "This uses Elasticsearch cardinality (approximate) for stable performance at scale."
    )

    return {
        "insight": to_json_safe(insight),
        "rows": to_json_safe(rows),
        "rules_used": business_rules or "",
        "engine": "es",
    }


__all__ = [
    "_es_month_over_month_visits",
    "_es_seasonal_revenue_patterns",
    "_es_dropoff_visits",
]
