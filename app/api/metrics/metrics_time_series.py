from __future__ import annotations

from datetime import datetime, timezone, timedelta
from typing import Any, Dict, List, Optional, Iterable

from abi.runtime import to_json_safe
from app.api.docs_analytics_routes import (
    resolve_es_field,
    _ms_to_dt,
    _es_cannot_answer,
    _build_date_range_filter,
)


# -------------------------------------------------------------------
# ES-safe helpers (timeout + pagination + safety caps)
# -------------------------------------------------------------------

def _safe_es_search(client, *, index: str, body: Dict[str, Any]) -> Dict[str, Any]:
    body = dict(body or {})
    body.setdefault("timeout", "10s")
    body.setdefault("track_total_hits", False)
    return client.search(index=index, body=body, request_timeout=20)


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
    If req has neither start_date nor end_date, apply a safe default window:
      [today-default_days, today]
    Returns either req (unchanged) or a wrapper with injected dates.
    """
    has_start = bool(getattr(req, "start_date", None))
    has_end = bool(getattr(req, "end_date", None))
    if has_start or has_end:
        return req

    today = datetime.now(timezone.utc).date()
    start = today - timedelta(days=default_days)
    return _WindowReq(req, start_date=start.isoformat(), end_date=today.isoformat())


def _composite_buckets(
    client,
    *,
    index: str,
    query_filters: List[Dict[str, Any]],
    sources: List[Dict[str, Any]],
    sub_aggs: Dict[str, Any],
    page_size: int = 1000,
    agg_name: str = "groups",
    state: Optional[Dict[str, Any]] = None,
    max_pages: int = 200,
    max_buckets: int = 200_000,
) -> Iterable[Dict[str, Any]]:
    """
    Composite aggregation paginator with safety caps.

    - max_pages: hard cap on number of composite requests
    - max_buckets: hard cap on total buckets yielded

    If capped, sets:
      state["truncated"] = True
      state["pages"] = <int>
      state["buckets"] = <int>
    """
    if state is None:
        state = {}
    state.setdefault("pages", 0)
    state.setdefault("buckets", 0)
    state.setdefault("truncated", False)

    after_key = None

    while True:
        if state["pages"] >= max_pages:
            state["truncated"] = True
            break

        comp: Dict[str, Any] = {"size": page_size, "sources": sources}
        if after_key:
            comp["after"] = after_key

        body = {
            "size": 0,
            "query": {"bool": {"filter": query_filters}},
            "aggs": {
                agg_name: {
                    "composite": comp,
                    "aggs": sub_aggs,
                }
            },
        }

        res = _safe_es_search(client, index=index, body=body)
        state["pages"] += 1

        agg = (res.get("aggregations") or {}).get(agg_name) or {}
        buckets = agg.get("buckets") or []
        if not buckets:
            break

        for b in buckets:
            if state["buckets"] >= max_buckets:
                state["truncated"] = True
                return
            state["buckets"] += 1
            yield b

        after_key = agg.get("after_key")
        if not after_key:
            break


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

    Safety changes:
      - If no window is provided, defaults to last ~13 months.
      - Uses visit_id.keyword if available.
      - Adds low-ish cardinality precision_threshold to reduce memory.
    """
    index_name = (req.es_index_name or "").strip()
    if not index_name:
        return _es_cannot_answer("Missing es_index_name.", business_rules)

    date_field = resolve_es_field(mappings, user_term="dropoff_at", alias_family="date")
    if not date_field:
        return _es_cannot_answer(
            "Cannot compute month-over-month visits because no date field could be resolved.",
            business_rules,
        )

    visit_field = (
        resolve_es_field(mappings, user_term="visit_id.keyword", alias_family="visit")
        or resolve_es_field(mappings, user_term="visit_id", alias_family="visit")
    )
    has_visit_field = bool(visit_field)

    # ✅ default window (prevents full-history scans)
    reqw = _with_default_window(req, default_days=400)  # ~13 months

    filters = _build_date_range_filter(reqw, date_field) or []
    query = {"bool": {"filter": filters}} if filters else None

    months_agg: Dict[str, Any] = {
        "date_histogram": {
            "field": date_field,
            "calendar_interval": "month",
        }
    }

    if has_visit_field:
        precision = int(getattr(req, "cardinality_precision", 4000) or 4000)
        months_agg["aggs"] = {
            "distinct_visits": {
                "cardinality": {
                    "field": visit_field,
                    "precision_threshold": precision,
                }
            }
        }

    body: Dict[str, Any] = {"size": 0, "aggs": {"months": months_agg}}
    if query:
        body["query"] = query

    res = _safe_es_search(client, index=index_name, body=body)
    buckets = ((res.get("aggregations") or {}).get("months") or {}).get("buckets", []) or []

    rows: List[Dict[str, Any]] = []
    for b in buckets:
        dt = _ms_to_dt(b.get("key"))
        if not dt:
            continue

        if has_visit_field:
            visit_count = int(((b.get("distinct_visits") or {}).get("value")) or 0)
        else:
            visit_count = int(b.get("doc_count") or 0)

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
    Seasonal revenue patterns vs last year using monthly date_histogram + sum(amount).

    Safety changes:
      - If no window is provided, defaults to last ~26 months (enough for YoY seasonality).
    """
    index_name = (req.es_index_name or "").strip()
    if not index_name:
        return _es_cannot_answer("Missing es_index_name.", business_rules)

    date_field = resolve_es_field(mappings, user_term="dropoff_at", alias_family="date")
    amount_field = resolve_es_field(mappings, user_term="total", alias_family="amount")
    if not date_field or not amount_field:
        return _es_cannot_answer(
            "Cannot compute seasonal revenue patterns because date or amount fields could not be resolved.",
            business_rules,
        )

    # ✅ default window (~26 months)
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
        yoy_text = (
            f"Total revenue in {current_year} was {total_curr:.2f}, while {prev_year} had no revenue."
        )
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


def _es_avg_ticket_size(
    req,
    client,
    mappings: Dict[str, Any],
    business_rules: Optional[str],
):
    """
    Average $ per visit by day of week / month of year.

    Safety changes:
      - If no window is provided, defaults to last 365 days (scripts can be CPU-heavy).
      - Uses visit_id.keyword if available.
    """
    index_name = (req.es_index_name or "").strip()
    if not index_name:
        return _es_cannot_answer("Missing es_index_name.", business_rules)

    date_field = resolve_es_field(mappings, user_term="dropoff_at", alias_family="date") or resolve_es_field(
        mappings, alias_family="date"
    )
    amount_field = resolve_es_field(mappings, user_term="total", alias_family="amount")
    visit_field = (
        resolve_es_field(mappings, user_term="visit_id.keyword", alias_family="visit")
        or resolve_es_field(mappings, user_term="visit_id", alias_family="visit")
    )

    if not (date_field and amount_field):
        return _es_cannot_answer(
            "Cannot compute avg $ per visit because date or amount fields could not be resolved.",
            business_rules,
        )

    # ✅ default window (scripts run per-doc; avoid full-history)
    reqw = _with_default_window(req, default_days=365)

    filters = _build_date_range_filter(reqw, date_field) or []
    query = {"bool": {"filter": filters}} if filters else None

    if visit_field:
        by_dow_aggs = {
            "total_revenue": {"sum": {"field": amount_field}},
            "visit_count": {"cardinality": {"field": visit_field, "precision_threshold": 4000}},
        }
        by_month_aggs = {
            "total_revenue": {"sum": {"field": amount_field}},
            "visit_count": {"cardinality": {"field": visit_field, "precision_threshold": 4000}},
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

    body: Dict[str, Any] = {
        "size": 0,
        "aggs": {
            "by_dow": {
                "terms": {
                    "script": {"source": f"doc['{date_field}'].value.dayOfWeek", "lang": "painless"},
                    "size": 7,
                    "order": {"_key": "asc"},
                },
                "aggs": by_dow_aggs,
            },
            "by_month": {
                "terms": {
                    "script": {"source": f"doc['{date_field}'].value.monthOfYear", "lang": "painless"},
                    "size": 12,
                    "order": {"_key": "asc"},
                },
                "aggs": by_month_aggs,
            },
        },
    }
    if query:
        body["query"] = query

    res = _safe_es_search(client, index=index_name, body=body)
    aggs = res.get("aggregations", {}) or {}

    dow_buckets = aggs.get("by_dow", {}).get("buckets", []) or []
    month_buckets = aggs.get("by_month", {}).get("buckets", []) or []

    DOW_NAMES = {1: "monday", 2: "tuesday", 3: "wednesday", 4: "thursday", 5: "friday", 6: "saturday", 7: "sunday"}
    MONTH_NAMES = {1: "jan", 2: "feb", 3: "mar", 4: "apr", 5: "may", 6: "jun", 7: "jul", 8: "aug", 9: "sep", 10: "oct", 11: "nov", 12: "dec"}

    rows: List[Dict[str, Any]] = []

    for b in dow_buckets:
        key = int(b.get("key", 0))
        total_rev = float(((b.get("total_revenue") or {}).get("value")) or 0.0)
        visits = float(((b.get("visit_count") or {}).get("value")) or 0.0)
        avg_per_visit = (total_rev / visits) if visits > 0 else None
        rows.append(
            {
                "dimension": "day_of_week",
                "label": DOW_NAMES.get(key, str(key)),
                "total_revenue": total_rev,
                "visit_count": visits,
                "avg_value_per_visit": avg_per_visit,
            }
        )

    for b in month_buckets:
        key = int(b.get("key", 0))
        total_rev = float(((b.get("total_revenue") or {}).get("value")) or 0.0)
        visits = float(((b.get("visit_count") or {}).get("value")) or 0.0)
        avg_per_visit = (total_rev / visits) if visits > 0 else None
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
        f"Average $ per visit computed on '{index_name}' by day-of-week and month-of-year. "
        f"Uses '{date_field}' for dates and '{amount_field}' for amounts. "
        f"One visit is treated as one distinct visit_id{' (fallback to invoice rows if visit_id missing)' if not visit_field else ''}."
    )

    return {
        "insight": to_json_safe(insight),
        "rows": to_json_safe(rows),
        "rules_used": business_rules or "",
        "engine": "es",
    }


def _es_yoy_revenue_by_location(
    req,
    client,
    mappings: Dict[str, Any],
    business_rules: Optional[str],
):
    """
    Year-over-year revenue growth by location.

    ✅ Major safety change:
      - Uses composite over (location, year) instead of (location -> date_histogram),
        which is typically much cheaper and streams well.

    ✅ Additional safety:
      - If no window is provided, defaults to ~26 months.
      - Composite caps (max_pages/max_buckets) to prevent overload.
    """
    index_name = (req.es_index_name or "").strip()
    if not index_name:
        return _es_cannot_answer("Missing es_index_name.", business_rules)

    date_field = resolve_es_field(mappings, user_term="dropoff_at", alias_family="date")
    amount_field = resolve_es_field(mappings, user_term="total", alias_family="amount")

    loc_field = (
        resolve_es_field(mappings, user_term="location_id.keyword", alias_family="location_id")
        or resolve_es_field(mappings, user_term="location_id", alias_family="location_id")
        or resolve_es_field(mappings, alias_family="location")
    )

    if not date_field or not amount_field or not loc_field:
        return _es_cannot_answer(
            "Cannot compute YoY revenue by location because date, amount, or location fields could not be resolved.",
            business_rules,
        )

    # ✅ default window (~26 months)
    reqw = _with_default_window(req, default_days=800)
    filters = _build_date_range_filter(reqw, date_field) or []

    # composite sources: location + year (date_histogram)
    sources = [
        {"loc": {"terms": {"field": loc_field}}},
        {"year": {"date_histogram": {"field": date_field, "calendar_interval": "year"}}},
    ]
    sub_aggs = {"revenue": {"sum": {"field": amount_field}}}

    # safety caps (allow overriding via req.* if you want)
    page_size = int(getattr(req, "composite_page_size", 800) or 800)
    max_pages = int(getattr(req, "composite_max_pages", 200) or 200)
    max_buckets = int(getattr(req, "composite_max_buckets", 200_000) or 200_000)

    state: Dict[str, Any] = {}

    rows: List[Dict[str, Any]] = []

    current_loc = None
    prev_year = None
    prev_rev = None

    for b in _composite_buckets(
        client,
        index=index_name,
        query_filters=filters,
        sources=sources,
        sub_aggs=sub_aggs,
        page_size=page_size,
        agg_name="loc_year",
        state=state,
        max_pages=max_pages,
        max_buckets=max_buckets,
    ):
        key = b.get("key") or {}
        loc = key.get("loc")
        year_ms = key.get("year")

        dt = _ms_to_dt(year_ms)
        if not dt or loc is None:
            continue

        year = dt.year
        rev = float(((b.get("revenue") or {}).get("value")) or 0.0)

        if loc != current_loc:
            # reset per-location streaming state
            current_loc = loc
            prev_year = year
            prev_rev = rev
            continue

        # same location, next year bucket
        if prev_year is not None and prev_rev is not None:
            delta = rev - prev_rev
            pct = (delta * 100.0 / prev_rev) if prev_rev > 0 else None
            rows.append(
                {
                    "location_id": loc,
                    "year": year,
                    "revenue": rev,
                    "prev_year": prev_year,
                    "prev_year_revenue": prev_rev,
                    "yoy_change": delta,
                    "yoy_change_pct": pct,
                }
            )

        prev_year = year
        prev_rev = rev

    if not rows:
        insight = "YoY revenue by location could not be computed because ES did not return enough data."
    else:
        latest_year = max(r["year"] for r in rows)
        latest_rows = [r for r in rows if r["year"] == latest_year and r["yoy_change_pct"] is not None]
        if latest_rows:
            mean_yoy = sum(r["yoy_change_pct"] for r in latest_rows) / len(latest_rows)
            insight = (
                "YoY revenue by location computed from ES. "
                f"For the most recent year ({latest_year}), average YoY change across locations is ~{mean_yoy:+.1f}%."
            )
        else:
            insight = "YoY revenue by location computed, but no previous-year baseline is available for % change."

    if state.get("truncated"):
        insight += (
            f" (NOTE: results were truncated for safety after {state.get('pages', 0)} pages / "
            f"{state.get('buckets', 0)} buckets.)"
        )

    return {
        "insight": to_json_safe(insight),
        "rows": to_json_safe(rows),
        "rules_used": business_rules or "",
        "engine": "es",
        "meta": {
            "composite_pages": state.get("pages"),
            "composite_buckets": state.get("buckets"),
            "truncated": state.get("truncated"),
        },
    }


__all__ = [
    "_es_month_over_month_visits",
    "_es_seasonal_revenue_patterns",
    "_es_avg_ticket_size",
    "_es_yoy_revenue_by_location",
]
