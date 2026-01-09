# metrics_lifecycle_no_rollup.py
# ------------------------------------------------------------
# NO-ROLLUP (read-only) version:
# - Removes ALL customer_stats / rollup logic.
# - Computes everything directly from the invoices index using:
#   - _es_get_customer_stats() (which should already be composite-paged)
#   - additional ES aggregations when needed (read-only safe)
# ------------------------------------------------------------

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Iterable

from abi.runtime import to_json_safe
from routes.es_test import _extract_properties_from_mapping

from app.api.docs_analytics_routes import (
    resolve_es_field,
    _ms_to_dt,
    _parse_date_str,
    _es_cannot_answer,
    _es_get_customer_stats,
    _es_get_customer_signups,
)

# -------------------------------------------------------------------
# ES-safe helpers
# -------------------------------------------------------------------

def _safe_es_search(client, *, index: str, body: Dict[str, Any]) -> Dict[str, Any]:
    """
    Safety defaults to avoid long-running queries.
    - timeout: ES-side
    - track_total_hits: off
    - request_timeout: client-side
    """
    body = dict(body or {})
    body.setdefault("timeout", "10s")
    body.setdefault("track_total_hits", False)
    return client.search(index=index, body=body, request_timeout=20)


def _get_req_int(req, name: str, default: int, *, min_v: int, max_v: int) -> int:
    v = getattr(req, name, default)
    try:
        v = int(v)
    except Exception:
        v = default
    return max(min_v, min(int(v), max_v))


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

    NOTE:
    - This can still be heavy if there are millions of customers.
    - _es_get_customer_stats should be implemented with composite paging + caps.
    """
    return _es_get_customer_stats(client, invoices_index, invoices_mappings)


# -------------------------------------------------------------------
# Metrics (NO rollup)
# -------------------------------------------------------------------

def _es_avg_days_between_visits_active(req, client, mappings: Dict[str, Any], business_rules: Optional[str]):
    """
    Average days between visits for *active* customers (last visit in last 365 days),
    using invoice-derived per-customer stats.
    """
    invoices_index = (req.es_index_name or "").strip()
    if not invoices_index:
        return _es_cannot_answer("Missing invoices index (es_index_name).", business_rules)

    today = datetime.now(timezone.utc).date()
    max_rows = _get_req_int(req, "es_max_rows", 2000, min_v=100, max_v=20_000)

    stats = _get_customer_stats_invoices_only(req, client, invoices_index, mappings)
    if stats is None:
        return _es_cannot_answer(
            "Cannot compute 'average days between visits for active customers' because customer/date fields "
            "could not be resolved from the mappings.",
            business_rules,
        )

    total_days_between = 0.0
    total_intervals = 0
    active_rows: List[Dict[str, Any]] = []

    for s in stats:
        vc = int(s.get("visit_count") or 0)
        first = s.get("first_visit")
        last = s.get("last_visit")
        if vc <= 1 or not first or not last:
            continue
        if (today - last.date()).days > 365:
            continue

        days_between = (last.date() - first.date()).days
        intervals = vc - 1
        if intervals <= 0:
            continue

        total_days_between += float(days_between)
        total_intervals += intervals

        if len(active_rows) < max_rows:
            active_rows.append(
                {
                    "customer_id": s.get("customer_id"),
                    "visits": vc,
                    "avg_days_between_visits_customer": days_between / float(intervals),
                    "first_visit": first.isoformat(),
                    "last_visit": last.isoformat(),
                    "intervals": intervals,
                }
            )

    if total_intervals == 0:
        return {
            "insight": to_json_safe(
                "Cannot compute average days between visits for active customers because there are no customers "
                "with at least two visits in the last 365 days."
            ),
            "rows": [],
            "rules_used": business_rules or "",
            "engine": "es",
        }

    avg_days = total_days_between / float(total_intervals)
    insight = (
        f"For active customers (last visit in the last 365 days), the average gap between visits across the "
        f"whole company is approximately {avg_days:.1f} days. Rows are limited to {max_rows} for safety."
    )
    return {
        "insight": to_json_safe(insight),
        "rows": to_json_safe(active_rows),
        "rules_used": business_rules or "",
        "engine": "es",
    }


def _es_lapsed_customers(req, client, mappings: Dict[str, Any], business_rules: Optional[str], days_threshold: int = 180):
    """
    Customers with no visit in the last N days (invoice-derived stats).
    """
    invoices_index = (req.es_index_name or "").strip()
    if not invoices_index:
        return _es_cannot_answer("Missing invoices index (es_index_name).", business_rules)

    max_rows = _get_req_int(req, "es_max_rows", 2000, min_v=100, max_v=20_000)
    today = datetime.now(timezone.utc).date()

    stats = _get_customer_stats_invoices_only(req, client, invoices_index, mappings)
    if stats is None:
        return _es_cannot_answer(
            "Cannot compute 'lapsed customers' because customer/date fields could not be resolved from the mappings.",
            business_rules,
        )

    lapsed_rows: List[Dict[str, Any]] = []
    lapsed_count = 0

    for s in stats:
        last = s.get("last_visit")
        if not last:
            continue
        days_since = (today - last.date()).days
        if days_since > int(days_threshold):
            lapsed_count += 1
            if len(lapsed_rows) < max_rows:
                lapsed_rows.append(
                    {
                        "customer_id": s.get("customer_id"),
                        "last_visit": last.isoformat(),
                        "days_since_last_visit": days_since,
                    }
                )

    total_customers = len(stats)
    pct = (lapsed_count * 100.0 / total_customers) if total_customers else 0.0

    insight = (
        f"There are {lapsed_count} lapsed customers (no visit in the last {days_threshold} days), "
        f"about {pct:.1f}% of all customers. Rows are limited to {max_rows} for safety."
    )

    return {
        "insight": to_json_safe(insight),
        "rows": to_json_safe(lapsed_rows),
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
    invoices_index = (req.es_index_name or "").strip()
    if not invoices_index:
        return _es_cannot_answer("Missing invoices index (es_index_name).", business_rules)

    max_rows = _get_req_int(req, "es_max_rows", 2000, min_v=100, max_v=20_000)
    today = datetime.now(timezone.utc).date()

    stats = _get_customer_stats_invoices_only(req, client, invoices_index, mappings)
    if stats is None:
        return _es_cannot_answer(
            "Cannot compute 'overdue customers' because customer/date fields could not be resolved from the mappings.",
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
    Distribution of customers by visit frequency (invoice-derived stats).
    """
    invoices_index = (req.es_index_name or "").strip()
    if not invoices_index:
        return _es_cannot_answer("Missing invoices index (es_index_name).", business_rules)

    stats = _get_customer_stats_invoices_only(req, client, invoices_index, mappings)
    if stats is None:
        return _es_cannot_answer(
            "Cannot compute 'distribution of customers by visit frequency' because customer/date fields could not be resolved.",
            business_rules,
        )

    buckets = {"1 visit": 0, "2–5 visits": 0, "6–11 visits": 0, "12+ visits": 0}
    for s in stats:
        v = int(s.get("visit_count") or 0)
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
    rows = []
    for label in ["1 visit", "2–5 visits", "6–11 visits", "12+ visits"]:
        count = buckets[label]
        pct = (count * 100.0 / total) if total else 0.0
        rows.append({"frequency_bucket": label, "customer_count": count, "percentage_of_customers": pct})

    return {
        "insight": to_json_safe("Distribution of customers by visit frequency computed from invoice-derived stats."),
        "rows": to_json_safe(rows),
        "rules_used": business_rules or "",
        "engine": "es",
    }


def _es_customers_nth_visit(req, client, mappings: Dict[str, Any], business_rules: Optional[str]):
    """
    Lifetime counts of customers who have reached >=2/3/4/5 visits (invoice-derived stats).
    """
    invoices_index = (req.es_index_name or "").strip()
    if not invoices_index:
        return _es_cannot_answer("Missing invoices index (es_index_name).", business_rules)

    stats = _get_customer_stats_invoices_only(req, client, invoices_index, mappings)
    if stats is None:
        return _es_cannot_answer(
            "Cannot compute 'Customers Achieving Nth Visit' because customer/date fields could not be resolved.",
            business_rules,
        )

    total_customers = 0
    c2 = c3 = c4 = c5 = 0

    for s in stats:
        vc = int(s.get("visit_count") or 0)
        if vc <= 0:
            continue
        total_customers += 1
        if vc >= 2:
            c2 += 1
        if vc >= 3:
            c3 += 1
        if vc >= 4:
            c4 += 1
        if vc >= 5:
            c5 += 1

    rows = [
        {"metric": "customers_2plus_visits", "label": "Customers Achieving 2nd Visit (≥2 visits)", "value": c2},
        {"metric": "customers_3plus_visits", "label": "Customers Achieving 3rd Visit (≥3 visits)", "value": c3},
        {"metric": "customers_4plus_visits", "label": "Customers Achieving 4th Visit (≥4 visits)", "value": c4},
        {"metric": "customers_5plus_visits", "label": "Customers Achieving 5th Visit (≥5 visits)", "value": c5},
        {"metric": "total_customers_lifetime", "label": "Total Customers (lifetime)", "value": total_customers},
    ]

    return {
        "insight": to_json_safe("Customers achieving Nth visit computed from invoice-derived stats."),
        "rows": to_json_safe(rows),
        "rules_used": business_rules or "",
        "engine": "es",
    }


def _es_top_customers_by_revenue(req, client, mappings: Dict[str, Any], business_rules: Optional[str]):
    """
    Top 5% / Top 20% customers by revenue (invoice-derived stats).
    NOTE: Requires full per-customer revenue list in Python (can be heavy).
    """
    invoices_index = (req.es_index_name or "").strip()
    if not invoices_index:
        return _es_cannot_answer("Missing invoices index (es_index_name).", business_rules)

    max_rows = _get_req_int(req, "es_max_rows", 500, min_v=50, max_v=5000)

    stats = _get_customer_stats_invoices_only(req, client, invoices_index, mappings)
    if stats is None:
        return _es_cannot_answer(
            "Cannot compute 'Top 5% / Top 20% customers by revenue' because customer/amount fields could not be resolved.",
            business_rules,
        )

    revenue_stats = [s for s in stats if (s.get("total_revenue") is not None and float(s.get("total_revenue") or 0) > 0)]
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


def _es_new_customer_acquisition(req, client, mappings: Dict[str, Any], business_rules: Optional[str]):
    """
    New customers over time.
    Definition:
      - Customer's first_visit date is within [start_date, end_date] if provided
      - Optional join with customers_index (original_signup) to require diff <= 30 days

    NO rollup: uses invoice-derived stats.
    """
    invoices_index = (req.es_index_name or "").strip()
    customers_index = (req.es_customers_index_name or "").strip()

    if not invoices_index or not customers_index:
        return _es_cannot_answer(
            "New Customer Acquisition requires both an invoices index (es_index_name) and a customers index (es_customers_index_name).",
            business_rules,
        )

    # Load signup dates from customers index (read-only OK)
    cust_mapping_raw = client.indices.get_mapping(index=customers_index)
    cust_props = _extract_properties_from_mapping(cust_mapping_raw, customers_index)
    cust_mappings = {"properties": cust_props}
    signup_by_customer = _es_get_customer_signups(client, customers_index, cust_mappings)

    start_d = _parse_date_str(getattr(req, "start_date", None))
    end_d = _parse_date_str(getattr(req, "end_date", None))
    max_diff_days = 30

    ql = (getattr(req, "question", "") or "").lower()
    use_quarter = any(p in ql for p in ["quarter", "q1", "q2", "q3", "q4"])

    counts: Dict[str, int] = {}

    stats = _get_customer_stats_invoices_only(req, client, invoices_index, mappings)
    if stats is None:
        return _es_cannot_answer(
            "Cannot compute 'New Customer Acquisition' because customer/date fields could not be resolved from the invoices index mappings.",
            business_rules,
        )

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

        if use_quarter:
            q = (fd.month - 1) // 3 + 1
            label = f"{fd.year}-Q{q}"
        else:
            label = f"{fd.year}-{fd.month:02d}"

        counts[label] = counts.get(label, 0) + 1

    rows = [{"period": p, "new_customers": c} for p, c in sorted(counts.items())]

    window_desc = []
    if getattr(req, "start_date", None):
        window_desc.append(f"from {req.start_date}")
    if getattr(req, "end_date", None):
        window_desc.append(f"to {req.end_date}")
    window_str = " ".join(window_desc) if window_desc else "for all available history"

    freq_label = "quarter" if use_quarter else "month"

    if not rows:
        insight = f"New Customer Acquisition by {freq_label} could not be computed {window_str} because no customers met the criteria."
    else:
        last = rows[-1]
        insight = (
            f"New Customer Acquisition by {freq_label} was computed from invoice-derived first_visit, joined with "
            f"customers index '{customers_index}' (original_signup). Window: {window_str}. "
            f"Most recent {freq_label} ({last['period']}) has {last['new_customers']} new customers."
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

    Implementation:
      - Uses customer list from invoice-derived stats (first_visit)
      - For second visit, runs ES queries on invoices for those customer IDs in chunks
    """
    invoices_index = (req.es_index_name or "").strip()
    customers_index = (req.es_customers_index_name or "").strip()

    if not invoices_index or not customers_index:
        return _es_cannot_answer(
            "New Customer 30-Day Return Rate requires both an invoices index (es_index_name) and a customers index (es_customers_index_name).",
            business_rules,
        )

    customer_field = resolve_es_field(mappings, user_term="customer_id", alias_family="customer")
    date_field = resolve_es_field(mappings, user_term="dropoff_at", alias_family="date")
    visit_field = resolve_es_field(mappings, user_term="visit_id", alias_family="visit")

    if not (customer_field and date_field):
        return _es_cannot_answer(
            "Cannot compute 'New Customer 30-Day Return Rate' because customer_id or dropoff_at could not be resolved.",
            business_rules,
        )

    cust_mapping_raw = client.indices.get_mapping(index=customers_index)
    cust_props = _extract_properties_from_mapping(cust_mapping_raw, customers_index)
    cust_mappings = {"properties": cust_props}
    signup_by_customer = _es_get_customer_signups(client, customers_index, cust_mappings)

    start_d = _parse_date_str(getattr(req, "start_date", None))
    end_d = _parse_date_str(getattr(req, "end_date", None))
    max_diff_days = 30

    # 1) Build new customer list from invoice-derived stats
    stats = _get_customer_stats_invoices_only(req, client, invoices_index, mappings)
    if stats is None:
        return _es_cannot_answer(
            "Cannot compute 'New Customer 30-Day Return Rate' because customer/date fields could not be resolved from invoices.",
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


__all__ = [
    "_es_avg_days_between_visits_active",
    "_es_lapsed_customers",
    "_es_overdue_customers",
    "_es_visit_frequency_distribution",
    "_es_customers_nth_visit",
    "_es_new_customer_acquisition",
    "_es_new_customer_30d_return_rate",
    "_es_top_customers_by_revenue",
]
