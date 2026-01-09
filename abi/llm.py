# abi/llm.py
import re
import os
import json
import hashlib
from typing import Dict, Optional

from helpers.analytics_helpers import (
    resolve_column,
    compute_customer_ltv,
    compute_visit_frequency,
    add_service_channel_visits,
    build_customer_channel_map_from_visits,
    merge_customer_channel_into_snapshot,
    _expand_es_nested_columns,
)

import pandas as pd
_ENV_QUESTION_PATTERNS = [
    ".env",
    "env file",
    "environment file",
    "environment variables",
    "env variables",
    "api key",
    "api_key",
    "apikey",
    "secret key",
    "secret_key",
    "db password",
    "database password",
    "jwt secret",
    "jwt_secret",
    "openai key",
    "openai_api_key",
]


def is_question_about_env(question: str) -> bool:
    """
    Very simple content filter: detect questions trying to access
    environment files / secrets / API keys, etc.
    """
    if not question:
        return False
    q = question.lower()
    return any(pat in q for pat in _ENV_QUESTION_PATTERNS)
OPENAI_AVAILABLE = False
try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except Exception as e:
    print("OpenAI import failed in abi.llm:", repr(e))


def _strip_code_fences(text: str) -> str:
    """
    Remove ```python ... ``` or ``` ... ``` wrappers if present.
    """
    return re.sub(
        r"^```(?:python)?\s*|\s*```$",
        "",
        text.strip(),
        flags=re.MULTILINE,
    ).strip()


# ------------------------------------------------------------
# Schema builder with truncation + caching
# ------------------------------------------------------------

_SCHEMA_CACHE: dict[str, str] = {}


def _schema_cache_key(tables: Dict) -> str:
    parts = []
    for tname, df in tables.items():
        try:
            if isinstance(df, pd.DataFrame):
                df = _expand_es_nested_columns(df)  # ✅ expand nested ES objs
            cols = list(df.columns)
        except Exception:
            cols = []
        parts.append(f"{tname}:{','.join(map(str, cols))}")
    return "|".join(sorted(parts))



def _build_schema_and_hints(tables: Dict) -> str:
    """
    Build a human-readable schema string + domain hints.
    Only uses column names, no data.

    Performance tweaks:
    - Cache based on table/column signature.
    - Truncate long column lists per table to keep prompt small.
    """
    if not tables:
        return "(no tables loaded)"

    cache_key = _schema_cache_key(tables)
    cached = _SCHEMA_CACHE.get(cache_key)
    if cached is not None:
        return cached

    lines = []
    MAX_COLS_PER_TABLE = 40  # cap for speed; logic unchanged, just less verbosity

    for tname, df in tables.items():
        try:
            cols = list(df.columns)
        except Exception:
            cols = []

        # Truncate column list but note how many were hidden
        if len(cols) > MAX_COLS_PER_TABLE:
            shown = cols[:MAX_COLS_PER_TABLE]
            hidden = len(cols) - MAX_COLS_PER_TABLE
            col_desc = f"{shown} ... (+{hidden} more)"
        else:
            col_desc = f"{cols}"

        lines.append(f"- {tname}: {col_desc}")

        lower_name = str(tname).lower()

        # Invoice-like tables
        if "invoice" in lower_name:
            if "total" in cols:
                lines.append(
                    f"  *Hint for table '{tname}':* 'total' is main monetary amount per order."
                )
            if "customer_id" in cols:
                lines.append(
                    f"  *Hint for table '{tname}':* 'customer_id' links to customers."
                )
            date_cols = [
                c
                for c in cols
                if any(
                    k in str(c).lower()
                    for k in [
                        "created_at",
                        "updated_at",
                        "dropoff_at",
                        "ready_at",
                        "pickup_at",
                        "date",
                    ]
                )
            ]
            if date_cols:
                lines.append(
                    f"  *Hint for table '{tname}':* date/time columns: {date_cols}."
                )

        # Customer-like tables
        if "customer" in lower_name:
            if "customer_id" in cols:
                lines.append(
                    f"  *Hint for table '{tname}':* 'customer_id' is primary key."
                )
            name_cols = [
                c
                for c in cols
                if any(k in str(c).lower() for k in ["first_name", "last_name", "name"])
            ]
            if name_cols:
                lines.append(
                    f"  *Hint for table '{tname}':* name columns: {name_cols}."
                )
            loc_cols = [
                c
                for c in cols
                if any(k in str(c).lower() for k in ["city", "state", "zip", "address"])
            ]
            if loc_cols:
                lines.append(
                    f"  *Hint for table '{tname}':* location columns: {loc_cols}."
                )

    if not lines:
        schema = "(no tables loaded)"
    else:
        schema = "\n".join(lines)

    _SCHEMA_CACHE[cache_key] = schema
    return schema


# ------------------------------------------------------------
# Question variants loader (from question_variants.json)
# ------------------------------------------------------------

def _load_question_variants() -> list:
    """
    Load question variants from abi/questions/question_variants.json.

    Expected structure:
    [
      {
        "question": "...",
        "variants": ["...", "..."]
      },
      ...
    ]
    """
    try:
        base_dir = os.path.dirname(__file__)
        path = os.path.join(base_dir, "questions", "question_variants.json")
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
            if isinstance(data, list):
                return data
    except Exception as e:
        print("llm: failed to load question_variants.json:", repr(e))
    return []


QUESTION_VARIANTS = _load_question_variants()

# ------------------------------------------------------------
# LLM code cache (per question + model + schema + rules)
# ------------------------------------------------------------

def _rules_hash(business_rules: Optional[str]) -> str:
    """
    Stable hash for business_rules so cache invalidates when rules change.
    """
    if not business_rules:
        return "no_rules"
    norm = business_rules.strip().encode("utf-8", errors="ignore")
    return hashlib.sha256(norm).hexdigest()[:16]


# ✅ modified: include rules_sig in cache key
_LLM_CODE_CACHE: dict[tuple[str, str, str, str], str] = {}


def llm_codegen(
    question: str,
    tables: Dict,
    model: str,
    api_key: Optional[str],
    business_rules: Optional[str] = None,  # ✅ NEW
) -> Optional[str]:
    """
    Generate Python code using OpenAI.

    - Uses the Responses API as primary.
    - Fallback: chat.completions.
    - Uses schema + business hints.
    - Caches code per (question, model, schema_signature, rules_signature) to speed up repeats.
    """
    if is_question_about_env(question):
        print("llm_codegen: blocked env-related question:", repr(question))
        return None
    if not OPENAI_AVAILABLE:
        print("llm_codegen: OPENAI_AVAILABLE is False")
        return None
    if not api_key:
        print("llm_codegen: api_key is empty or None")
        return None

    # Build schema (cached) and cache key for LLM result
    schema_str = _build_schema_and_hints(tables)
    schema_sig = _schema_cache_key(tables)
    q_norm_for_cache = question.strip()

    # ✅ NEW: include rules hash in cache key
    rules_sig = _rules_hash(business_rules)

    cache_key = (q_norm_for_cache, model, schema_sig, rules_sig)

    # 👉 Fast path: if we've already generated code for this question/model/schema/rules
    cached_code = _LLM_CODE_CACHE.get(cache_key)
    if cached_code is not None:
        return cached_code

    q_lower = question.strip().lower()

    # ------------------------------------------------------------
    # SPECIAL HANDLER: Average customer lifetime value (CLV)
    #  - Supports "last N months" dynamically (3, 6, 12, ...)
    # ------------------------------------------------------------
    if "average customer lifetime value" in q_lower or "average clv" in q_lower:
        code = f"""
import re
import pandas as pd
from helpers.analytics_helpers import compute_customer_ltv, resolve_column

# Original natural-language question
question_text = {repr(question)}
ql = question_text.lower()

# --------------------------------------------
# 1) Try to detect "last N months" in question
# --------------------------------------------
months = None
m = re.search(r"(last|past)\\s+(\\d+)\\s+month", ql)
if m:
    try:
        months = int(m.group(2))
    except Exception:
        months = None

# --------------------------------------------
# Helper: CLV from customer_snapshot (lifetime)
# --------------------------------------------
def _avg_clv_lifetime(customer_snapshot):
    cs = compute_customer_ltv(customer_snapshot)

    if cs is None or not isinstance(cs, pd.DataFrame) or cs.empty:
        return pd.DataFrame(), "Customer lifetime value could not be computed because there is no per-customer snapshot available."

    # Prefer realized pickup-based lifetime revenue if present
    values = None
    metric_label = ""

    if "sales_pickup_lifetime" in cs.columns:
        values = pd.to_numeric(cs["sales_pickup_lifetime"], errors="coerce")
        metric_label = "pickup-based lifetime revenue per customer"
    elif "total" in cs.columns:
        values = pd.to_numeric(cs["total"], errors="coerce")
        metric_label = "total lifetime revenue per customer (using 'total')"
    else:
        numeric_cols = cs.select_dtypes(include="number")
        if numeric_cols.empty:
            return cs.copy(), "Customer lifetime value could not be computed because no numeric revenue columns are available."
        values = numeric_cols.sum(axis=1)
        metric_label = "sum of numeric fields per customer"

    values = values.fillna(0)
    if len(values) == 0:
        avg_ltv = 0.0
    else:
        avg_ltv = float(values.mean())

    out = cs.copy()
    out["ltv_value"] = values
    insight = f"The average customer lifetime value (based on {{metric_label}}) is {{avg_ltv:.2f}}."
    return out, insight

# --------------------------------------------
# Helper: CLV over last N months using mdf
# --------------------------------------------
def _avg_clv_last_n_months(mdf, months):
    if months is None or months <= 0:
        return None, "Invalid months window; falling back to lifetime CLV."

    if "mdf" not in globals() or mdf is None or not isinstance(mdf, pd.DataFrame) or mdf.empty:
        return None, (
            f"Cannot compute CLV for the last {{months}} months because the main fact table (mdf) is not available."
        )

    df = mdf.copy()

    # Resolve columns: customer, date (pickup_at preferred), amount
    cust_col = resolve_column(df, alias_family="customer")
    date_col = resolve_column(df, "pickup_at", alias_family="date")
    if date_col is None:
        date_col = resolve_column(df, alias_family="date")
    amount_col = resolve_column(df, alias_family="amount")

    if cust_col is None or date_col is None or amount_col is None:
        return None, (
            "Cannot compute CLV for the last {{months}} months because customer, date, "
            "or amount columns could not be resolved."
        )

    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df[amount_col] = pd.to_numeric(df[amount_col], errors="coerce")
    df = df.dropna(subset=[date_col, amount_col, cust_col])

    if df.empty:
        return None, (
            "Cannot compute CLV for the last {{months}} months because all dates or amounts are invalid."
        )

    # Build time window [today - months, today) in Florida local time (tz-naive)
    today = pd.Timestamp.now(tz="America/New_York").normalize().tz_localize(None)
    start = today - pd.DateOffset(months=months)

    mask = (df[date_col] >= start) & (df[date_col] < today)
    window_df = df[mask]

    if window_df.empty:
        return None, (
            f"No realized revenue in the last {{months}} months; cannot compute CLV for that window."
        )

    # Group by customer and sum revenue in that window
    per_customer = (
        window_df.groupby(cust_col)[amount_col]
        .sum()
        .reset_index(name="ltv_value")
    )

    if per_customer.empty:
        return None, (
            f"No customers with revenue in the last {{months}} months; cannot compute CLV for that window."
        )

    # Normalize customer id column name if possible
    if cust_col in per_customer.columns and cust_col != "customer_id":
        per_customer = per_customer.rename(columns={{cust_col: "customer_id"}})

    avg_ltv = float(per_customer["ltv_value"].mean())
    insight = (
        f"The average customer lifetime value based on realized revenue in the last {{months}} months "
        f"is {{avg_ltv:.2f}}."
    )
    return per_customer, insight

# --------------------------------------------
# Decide which path to use
# --------------------------------------------
if months is not None:
    # Try N-month window first
    df_window, insight_window = _avg_clv_last_n_months(mdf, months)
    if df_window is not None:
        result_df = df_window
        insight = insight_window
    else:
        # Fallback to lifetime CLV
        result_df, insight = _avg_clv_lifetime(customer_snapshot)
else:
    # No explicit "last N months" → lifetime CLV
    result_df, insight = _avg_clv_lifetime(customer_snapshot)
"""
        _LLM_CODE_CACHE[cache_key] = code
        return code

    # ------------------------------------------------------------
    # SPECIAL HANDLER: One-time vs repeat customers
    # ------------------------------------------------------------
    if "one-time" in q_lower and "repeat" in q_lower and "customer" in q_lower:
        # Use an f-string so we can embed the original question text
        code = f"""
import pandas as pd
from helpers.analytics_helpers import compute_visit_frequency

vf = compute_visit_frequency(customer_snapshot)

if vf is None or not isinstance(vf, pd.DataFrame) or vf.empty:
    result_df = pd.DataFrame()
    insight = "Cannot compute one-time vs repeat customers because visit frequency data is not available."
else:
    # Read the original natural-language question to detect if the user
    # explicitly asked for a time window like "last 365 days".
    question_text = {repr(question)}
    ql = question_text.lower()

    # Decide whether the user wants a RECENT window vs ALL history.
    # If they mention an explicit timeframe (e.g. "last 365 days",
    # "last year", "last 12 months"), we try to use visits_365.
    wants_recent = any(
        phrase in ql
        for phrase in [
            "last 365 days",
            "last 365",
            "past 365 days",
            "past 365",
            "last year",
            "past year",
            "last 12 months",
            "past 12 months",
        ]
    )

    visit_col = None

    if wants_recent and "visits_365" in vf.columns:
        # User explicitly asked for a recent window → use visits_365
        visit_col = "visits_365"
    elif "visits_lifetime" in vf.columns:
        # DEFAULT: use ALL HISTORY when no timeframe is specified
        visit_col = "visits_lifetime"
    elif "visits_365" in vf.columns:
        # Fallback: if lifetime isn’t available but 365 is, use it
        visit_col = "visits_365"
    else:
        # Last-resort fallback: any numeric column
        numeric_cols = vf.select_dtypes(include="number")
        if not numeric_cols.empty:
            visit_col = numeric_cols.columns[0]

    if visit_col is None:
        result_df = vf.copy()
        insight = (
            "Cannot compute one-time vs repeat customers because no visit count "
            "column (e.g. visits_lifetime or visits_365) is available."
        )
    else:
        counts = pd.to_numeric(vf[visit_col], errors="coerce").fillna(0)

        # Business rule: 1 or 0 visits = one-time, >1 = repeat
        one_time_mask = counts <= 1
        repeat_mask = counts > 1

        one_time = int(one_time_mask.sum())
        repeat = int(repeat_mask.sum())
        total = int(len(vf))

        if total == 0:
            one_pct = repeat_pct = 0.0
        else:
            one_pct = 100.0 * one_time / total
            repeat_pct = 100.0 * repeat / total

        result_df = pd.DataFrame(
            {{
                "segment": ["one-time", "repeat"],
                "customer_count": [one_time, repeat],
                "percentage_of_customers": [one_pct, repeat_pct],
            }}
        )

        if wants_recent and visit_col == "visits_365":
            window_desc = "in the last 365 days"
        else:
            window_desc = "over full available history"

        insight = (
            f"Out of {{total}} customers, about {{one_pct:.1f}}% are one-time customers "
            f"and {{repeat_pct:.1f}}% are repeat customers, based on {{visit_col}} "
            f"({{window_desc}})."
        )
"""
        _LLM_CODE_CACHE[cache_key] = code
        return code

    # ------------------------------------------------------------
    # SPECIAL HANDLER: Average days between visits for active customers
    # ------------------------------------------------------------
    if "average days between visits" in q_lower and "active customers" in q_lower:
        code = """
import pandas as pd

cs = customer_snapshot

if cs is None or not isinstance(cs, pd.DataFrame) or cs.empty:
    result_df = pd.DataFrame()
    insight = "Cannot compute average days between visits because customer snapshot is not available."
else:
    df = cs.copy()
    if "last_visit" in df.columns:
        df["last_visit"] = pd.to_datetime(df["last_visit"], errors="coerce")
    if "visits_interval_avg" not in df.columns:
        result_df = df.copy()
        insight = "Cannot compute average days between visits because 'visits_interval_avg' is not available."
    else:
        today = pd.Timestamp.now(tz="America/New_York").normalize().tz_localize(None)
        one_year_ago = today - pd.Timedelta(days=365)

        # Active customers: at least one visit in last 365 days
        if "last_visit" in df.columns:
            active_mask = df["last_visit"] >= one_year_ago
        else:
            active_mask = pd.Series(True, index=df.index)

        intervals = pd.to_numeric(df.loc[active_mask, "visits_interval_avg"], errors="coerce")
        intervals = intervals.dropna()
        if intervals.empty:
            avg_days = 0.0
        else:
            avg_days = float(intervals.mean())

        result_df = df.loc[active_mask, ["customer_id", "customer_name", "visits_interval_avg"]].copy()
        insight = (
            f"For active customers (visited in the last 365 days), "
            f"the average days between visits is approximately {avg_days:.1f} days."
        )
"""
        _LLM_CODE_CACHE[cache_key] = code
        return code

    # ------------------------------------------------------------
    # SPECIAL HANDLER: Customers overdue for next visit
    # ------------------------------------------------------------
    if "overdue for their next visit" in q_lower or "overdue for next visit" in q_lower:
        code = """
import pandas as pd

cs = customer_snapshot

if cs is None or not isinstance(cs, pd.DataFrame) or cs.empty:
    result_df = pd.DataFrame()
    insight = "Cannot determine overdue customers because customer snapshot is not available."
else:
    df = cs.copy()

    if "last_visit" not in df.columns or "visits_interval_avg" not in df.columns:
        result_df = df.copy()
        insight = "Cannot determine overdue customers because 'last_visit' or 'visits_interval_avg' is missing."
    else:
        df["last_visit"] = pd.to_datetime(df["last_visit"], errors="coerce")
        intervals = pd.to_numeric(df["visits_interval_avg"], errors="coerce")

        today = pd.Timestamp.now(tz="America/New_York").normalize().tz_localize(None)
        # Threshold: last_visit + 1.5 * average interval
        expected = df["last_visit"] + pd.to_timedelta(intervals * 1.5, unit="D")

        mask = (
            df["last_visit"].notna()
            & intervals.notna()
            & (intervals > 0)
            & expected.notna()
            & (expected < today)
        )

        overdue_df = df.loc[mask, ["customer_id", "customer_name", "last_visit", "visits_interval_avg"]].copy()
        overdue_count = int(len(overdue_df))

        result_df = overdue_df
        insight = (
            f"There are {overdue_count} customers who appear overdue for their next visit "
            f"based on their typical visit interval."
        )
"""
        _LLM_CODE_CACHE[cache_key] = code
        return code

    # ------------------------------------------------------------
    # NEW SPECIAL HANDLER: Lapsed customers (>180 days since last visit)
    # ------------------------------------------------------------
    if "lapsed customers" in q_lower or (
        ">180 days" in q_lower and "last visit" in q_lower
    ):
        code = """
import pandas as pd
from helpers.analytics_helpers import resolve_column

cs = customer_snapshot

if cs is None or not isinstance(cs, pd.DataFrame) or cs.empty:
    result_df = pd.DataFrame()
    insight = "Cannot compute lapsed customers because customer_snapshot is not available."
else:
    df = cs.copy()

    # If last_visit is missing or all NaT, try to rebuild it from mdf
    need_rebuild = ("last_visit" not in df.columns) or df["last_visit"].isna().all()

    if need_rebuild:
        if "mdf" in globals() and isinstance(mdf, pd.DataFrame) and not mdf.empty:
            base = mdf.copy()
            cust_col = resolve_column(base, alias_family="customer")
            date_col = resolve_column(base, "dropoff_at", alias_family="date")
            if date_col is None:
                date_col = resolve_column(base, alias_family="date")

            if cust_col is None or date_col is None or cust_col not in base.columns or date_col not in base.columns:
                result_df = df.copy()
                insight = (
                    "Cannot compute lapsed customers because neither 'last_visit' "
                    "nor a usable visit date column is available."
                )
            else:
                base[date_col] = pd.to_datetime(base[date_col], errors="coerce")
                base = base.dropna(subset=[date_col, cust_col])

                if base.empty:
                    result_df = df.copy()
                    insight = (
                        "Cannot compute lapsed customers because all visit dates are invalid."
                    )
                else:
                    last_visit_map = (
                        base.groupby(cust_col)[date_col]
                        .max()
                    )

                    if "customer_id" in df.columns:
                        df["last_visit"] = df["customer_id"].map(last_visit_map)
                    else:
                        result_df = df.copy()
                        insight = (
                            "Cannot compute lapsed customers because 'customer_id' is not available "
                            "to join with visit dates."
                        )
        else:
            result_df = df.copy()
            insight = (
                "Cannot compute lapsed customers because 'last_visit' is missing and mdf is not usable."
            )

    # If we have last_visit now, compute lapsed customers
    if "last_visit" in df.columns:
        df["last_visit"] = pd.to_datetime(df["last_visit"], errors="coerce")
        today = pd.Timestamp.now(tz="America/New_York").normalize().tz_localize(None)
        cutoff = today - pd.Timedelta(days=180)

        mask = df["last_visit"].notna() & (df["last_visit"] < cutoff)
        lapsed_df = df.loc[mask].copy()

        lapsed_count = int(len(lapsed_df))
        total_customers = int(len(df))

        if "customer_id" in lapsed_df.columns and "customer_name" in lapsed_df.columns:
            result_df = lapsed_df[["customer_id", "customer_name", "last_visit"]]
        else:
            result_df = lapsed_df

        if total_customers > 0:
            pct = 100.0 * lapsed_count / total_customers
        else:
            pct = 0.0

        insight = (
            f"There are {lapsed_count} lapsed customers (no visit in the last 180 days), "
            f"which is about {pct:.1f}% of all customers."
        )
"""
        _LLM_CODE_CACHE[cache_key] = code
        return code

    # ------------------------------------------------------------
    # SPECIAL HANDLER: Distribution by visit frequency
    # ------------------------------------------------------------
    if "distribution of customers by visit frequency" in q_lower or (
        "visit frequency" in q_lower and "1, 2–5, 6–11, 12+" in q_lower
    ):
        code = """
import pandas as pd

cs = customer_snapshot

if cs is None or not isinstance(cs, pd.DataFrame) or cs.empty:
    result_df = pd.DataFrame()
    insight = "Cannot compute visit frequency distribution because customer snapshot is not available."
else:
    df = cs.copy()
    # Prefer visits_365 if present, else visits_lifetime
    freq_col = None
    if "visits_365" in df.columns:
        freq_col = "visits_365"
    elif "visits_lifetime" in df.columns:
        freq_col = "visits_lifetime"

    if freq_col is None:
        numeric_cols = df.select_dtypes(include="number")
        if not numeric_cols.empty:
            freq_col = numeric_cols.columns[0]

    if freq_col is None:
        result_df = df.copy()
        insight = "Cannot compute visit frequency distribution because no visit count column is available."
    else:
        visits = pd.to_numeric(df[freq_col], errors="coerce").fillna(0)

        # We only care about customers with at least 1 visit
        mask_nonzero = visits > 0
        visits = visits[mask_nonzero]

        bins = [0, 1, 5, 11, 1e9]
        labels = ["1 visit", "2–5 visits", "6–11 visits", "12+ visits"]

        categories = pd.cut(
            visits,
            bins=bins,
            labels=labels,
            right=True,
            include_lowest=False,
        )

        counts = categories.value_counts().reindex(labels, fill_value=0)
        total = int(counts.sum())
        if total == 0:
            percents = [0.0] * len(labels)
        else:
            percents = [float(c) * 100.0 / total for c in counts]

        result_df = pd.DataFrame(
            {
                "frequency_bucket": labels,
                "customer_count": [int(c) for c in counts],
                "percentage_of_customers": percents,
            }
        )
        insight = (
            "Distribution of customers by visit frequency has been computed "
            "into buckets: 1, 2–5, 6–11, and 12+ visits."
        )
"""
        _LLM_CODE_CACHE[cache_key] = code
        return code

    # ------------------------------------------------------------
    # SPECIAL HANDLER: Top 5% / Top 20% customers by revenue
    # and percentage of revenue from Top 20%
    # ------------------------------------------------------------
    if (
        ("top 5%" in q_lower and "top 20%" in q_lower and "revenue" in q_lower)
        or ("top 5 percent" in q_lower and "top 20 percent" in q_lower and "revenue" in q_lower)
        or ("which customers fall into the top 5%" in q_lower)
        or ("top 20%" in q_lower and "revenue" in q_lower)
        or ("top 20 percent" in q_lower and "revenue" in q_lower)
        or ("percentage of revenue comes from the top 20%" in q_lower)
    ):
        code = """
import pandas as pd
from helpers.analytics_helpers import compute_customer_ltv

cs = compute_customer_ltv(customer_snapshot)

if cs is None or not isinstance(cs, pd.DataFrame) or cs.empty:
    result_df = pd.DataFrame()
    insight = "Cannot compute top customers by revenue because customer snapshot is not available."
else:
    df = cs.copy()

    # Choose a revenue column: prefer sales_pickup_lifetime, else total, else numeric sum
    if "sales_pickup_lifetime" in df.columns:
        revenue = pd.to_numeric(df["sales_pickup_lifetime"], errors="coerce")
        revenue_label = "pickup-based lifetime revenue"
    elif "total" in df.columns:
        revenue = pd.to_numeric(df["total"], errors="coerce")
        revenue_label = "lifetime revenue from 'total'"
    else:
        numeric_cols = df.select_dtypes(include="number")
        if numeric_cols.empty:
            result_df = df.copy()
            insight = "Cannot compute top customers by revenue because no numeric revenue columns are available."
        else:
            revenue = numeric_cols.sum(axis=1)
            revenue_label = "sum of numeric fields"

    if 'revenue' not in locals():
        pass
    else:
        revenue = revenue.fillna(0)
        base = pd.DataFrame({
            "customer_id": df.get("customer_id", pd.Series(range(len(df)))),
            "customer_name": df.get("customer_name", pd.Series([""] * len(df))),
            "revenue": revenue,
        })

        base = base.sort_values("revenue", ascending=False).reset_index(drop=True)
        n = len(base)
        if n == 0:
            result_df = base
            insight = "No customers found to compute top revenue segments."
        else:
            top20_n = max(1, int(round(0.20 * n)))
            top5_n = max(1, int(round(0.05 * n)))

            total_revenue = float(base["revenue"].sum())
            top20_revenue = float(base.iloc[:top20_n]["revenue"].sum())
            top5_revenue = float(base.iloc[:top5_n]["revenue"].sum())

            if total_revenue <= 0:
                top20_share = 0.0
                top5_share = 0.0
            else:
                top20_share = 100.0 * top20_revenue / total_revenue
                top5_share = 100.0 * top5_revenue / total_revenue

            base["segment"] = "Bottom 80%"
            base.loc[base.index < top20_n, "segment"] = "Top 20%"
            base.loc[base.index < top5_n, "segment"] = "Top 5%"

            result_df = base

            insight = (
                f"Top 20% of customers by {revenue_label} contribute about {top20_share:.1f}% "
                f"of total revenue, and the Top 5% contribute about {top5_share:.1f}%. "
                f"The result table labels customers as 'Top 5%', 'Top 20%', or 'Bottom 80%'."
            )
"""
        _LLM_CODE_CACHE[cache_key] = code
        return code
      # ------------------------------------------------------------
    # SPECIAL HANDLER: Month-over-month visit volume trend
    # ------------------------------------------------------------
    if ("month-over-month" in q_lower or "month over month" in q_lower) and "visit" in q_lower:
        code = """
import pandas as pd
from helpers.analytics_helpers import resolve_column, resolve_visit_col

df = mdf

if df is None or not isinstance(df, pd.DataFrame) or df.empty:
    result_df = pd.DataFrame()
    insight = "Cannot compute month-over-month visit volume because the main fact table (mdf) is not available."
else:
    df = df.copy()

    # Prefer dropoff_at as the visit signal; fallback to any date column
    date_col = resolve_column(df, "dropoff_at", alias_family="date")
    if date_col is None:
        date_col = resolve_column(df, alias_family="date")

    if date_col is None or date_col not in df.columns:
        result_df = pd.DataFrame()
        insight = "Cannot compute month-over-month visits because no usable visit date column (e.g., dropoff_at) could be resolved."
    else:
        # Parse dates safely (assume raw data is UTC and normalize to Florida local, tz-naive)
        df[date_col] = (
            pd.to_datetime(df[date_col], utc=True, errors="coerce")
              .dt.tz_convert("America/New_York")
              .dt.tz_localize(None)
        )
        df = df.dropna(subset=[date_col])
        if df.empty:
            result_df = pd.DataFrame()
            insight = "No valid visit dates found to compute month-over-month trends."
        else:
            # --- Use the same visit logic as _build_customer_snapshot ---
            cust_col = resolve_column(df, alias_family="customer")
            visit_col = resolve_visit_col(df)

            # Normalize to visit_day (calendar day)
            df["visit_day"] = df[date_col].dt.normalize()

            if visit_col is not None and visit_col in df.columns:
                # Case 1: visit_id present
                if cust_col is not None and cust_col in df.columns:
                    # One visit per (customer, visit_id), using earliest day
                    tmp = df[[cust_col, visit_col, "visit_day"]].dropna(
                        subset=[cust_col, visit_col, "visit_day"]
                    ).copy()
                    tmp["visit_day"] = (
                        tmp.groupby([cust_col, visit_col])["visit_day"]
                           .transform("min")
                    )
                    visits = tmp.drop_duplicates(subset=[cust_col, visit_col])
                else:
                    # No customer_id, one visit per visit_id (earliest day)
                    tmp = df[[visit_col, "visit_day"]].dropna(
                        subset=[visit_col, "visit_day"]
                    ).copy()
                    tmp["visit_day"] = (
                        tmp.groupby(visit_col)["visit_day"]
                           .transform("min")
                    )
                    visits = tmp.drop_duplicates(subset=[visit_col])
            else:
                # Case 2: no visit_id
                if cust_col is not None and cust_col in df.columns:
                    # One visit per (customer, day)
                    tmp = df[[cust_col, "visit_day"]].dropna(
                        subset=[cust_col, "visit_day"]
                    ).copy()
                    visits = tmp.drop_duplicates(subset=[cust_col, "visit_day"])
                else:
                    # Case 3: last resort — one visit per day
                    visits = df[["visit_day"]].dropna().drop_duplicates()

            if visits.empty:
                result_df = pd.DataFrame()
                insight = "No visits found to compute a month-over-month trend."
            else:
                # Convert to month periods, then to timestamps (month start)
                visits["month"] = visits["visit_day"].dt.to_period("M").dt.to_timestamp()

                # Limit to last 12 months for a clean trend in Florida local time
                today = pd.Timestamp.now(tz="America/New_York").normalize().tz_localize(None)
                twelve_months_ago = (today - pd.DateOffset(months=12)).replace(day=1)
                visits = visits[visits["month"] >= twelve_months_ago]

                if visits.empty:
                    result_df = pd.DataFrame()
                    insight = "No visits in the last 12 months to compute a month-over-month trend."
                else:
                    monthly = (
                        visits.groupby("month")
                        .size()
                        .reset_index(name="visit_count")
                        .sort_values("month")
                    )

                    result_df = monthly

                    # Build a short insight comparing last vs previous month if possible
                    if len(monthly) >= 2:
                        last_row = monthly.iloc[-1]
                        prev_row = monthly.iloc[-2]
                        last_month = last_row["month"]
                        prev_month = prev_row["month"]
                        last_visits = int(last_row["visit_count"])
                        prev_visits = int(prev_row["visit_count"])

                        if prev_visits == 0:
                            change_str = "previous month had zero visits, so change is not comparable."
                        else:
                            delta = last_visits - prev_visits
                            pct = 100.0 * delta / prev_visits
                            sign = "increase" if delta >= 0 else "decrease"
                            change_str = (
                                f"{sign} of {abs(delta)} visits "
                                f"({pct:+.1f}% vs previous month)."
                            )

                        insight = (
                            "Month-over-month visit volume trend has been computed. "
                            f"Last month ({last_month.date()}) had {last_visits} visits compared to {prev_visits} "
                            f"in the prior month ({prev_month.date()}); {change_str}"
                        )
                    else:
                        insight = (
                            "Month-over-month visit volume trend has been computed, "
                            "but only a single month of data is available."
                        )
"""
        _LLM_CODE_CACHE[cache_key] = code
        return code

    # ------------------------------------------------------------
    # NEW SPECIAL HANDLER: Seasonal patterns vs last year (revenue)
    # ------------------------------------------------------------
    if "seasonal patterns" in q_lower or (
        "seasonal" in q_lower and "last year" in q_lower
    ):
        code = """
import pandas as pd
from helpers.analytics_helpers import resolve_column

df = mdf

if df is None or not isinstance(df, pd.DataFrame) or df.empty:
    result_df = pd.DataFrame()
    insight = "Cannot compute seasonal revenue patterns because the main fact table (mdf) is not available."
else:
    df = df.copy()

    # Use pickup_at as realized revenue date; fallback to dropoff_at or any date
    date_col = resolve_column(df, "pickup_at", alias_family="date")
    if date_col is None:
        date_col = resolve_column(df, "dropoff_at", alias_family="date")
    if date_col is None:
        date_col = resolve_column(df, alias_family="date")

    amount_col = resolve_column(df, alias_family="amount")

    if date_col is None or date_col not in df.columns or amount_col is None or amount_col not in df.columns:
        result_df = pd.DataFrame()
        insight = (
            "Cannot compute seasonal revenue patterns because either the revenue date column "
            "or amount column could not be resolved."
        )
    else:
        df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
        df[amount_col] = pd.to_numeric(df[amount_col], errors="coerce")
        df = df.dropna(subset=[date_col, amount_col])

        if df.empty:
            result_df = pd.DataFrame()
            insight = "Cannot compute seasonal revenue patterns because all dates or amounts are invalid."
        else:
            # Focus on last 2 full years based on max date
            max_date = df[date_col].max().normalize()
            current_year = int(max_date.year)
            prev_year = current_year - 1

            df["year"] = df[date_col].dt.year
            df["month_num"] = df[date_col].dt.month

            # Keep only those 2 years
            df_two = df[df["year"].isin([prev_year, current_year])].copy()
            if df_two.empty:
                result_df = pd.DataFrame()
                insight = (
                    "Cannot compute seasonal revenue patterns because there is not enough data "
                    "for the last two years."
                )
            else:
                # Group by year+month and sum revenue
                grouped = (
                    df_two.groupby(["year", "month_num"])[amount_col]
                    .sum()
                    .reset_index(name="revenue")
                )

                # Pivot: one row per month, columns per year
                pivot = grouped.pivot(index="month_num", columns="year", values="revenue")

                # Sort by calendar month
                pivot = pivot.sort_index()

                # Add month name for readability
                month_names = {i: pd.Timestamp(2000, i, 1).strftime("%b") for i in range(1, 13)}
                pivot["month_name"] = pivot.index.map(month_names)

                # Ensure both year columns exist
                if current_year not in pivot.columns:
                    pivot[current_year] = 0.0
                if prev_year not in pivot.columns:
                    pivot[prev_year] = 0.0

                # Reorder columns: month_name, last year, this year
                pivot = pivot[["month_name", prev_year, current_year]]

                pivot = pivot.rename(
                    columns={
                        prev_year: f"revenue_{prev_year}",
                        current_year: f"revenue_{current_year}",
                    }
                )

                # Compute YoY change per month
                ly_col = f"revenue_{prev_year}"
                cy_col = f"revenue_{current_year}"
                pivot["yoy_change"] = pivot[cy_col] - pivot[ly_col]
                with pd.option_context("mode.use_inf_as_na", True):
                    pivot["yoy_change_pct"] = (
                        (pivot["yoy_change"] / pivot[ly_col].replace(0, pd.NA)) * 100.0
                    )

                result_df = pivot.reset_index(drop=True)

                # Build summary insight for the whole year
                total_prev = float(df_two[df_two["year"] == prev_year][amount_col].sum())
                total_curr = float(df_two[df_two["year"] == current_year][amount_col].sum())

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

                # Find month with largest positive YoY change (if any)
                best_row = pivot.sort_values("yoy_change", ascending=False).iloc[0]
                best_month = best_row["month_name"]
                best_delta = float(best_row["yoy_change"])
                best_prev = float(best_row[ly_col])
                best_curr = float(best_row[cy_col])

                if best_prev == 0:
                    best_text = (
                        f"The strongest month vs last year was {best_month}, which had {best_curr:.2f} this year "
                        f"compared to {best_prev:.2f} last year."
                    )
                else:
                    best_pct = 100.0 * best_delta / best_prev
                    sign_best = "increase" if best_delta >= 0 else "decrease"
                    best_text = (
                        f"The strongest month vs last year was {best_month}, with a {sign_best} of "
                        f"{abs(best_delta):.2f} ({best_pct:+.1f}% YoY)."
                    )

                insight = (
                    "Seasonal revenue patterns have been compared month-by-month vs last year. "
                    + yoy_text + " " + best_text
                )
"""
        _LLM_CODE_CACHE[cache_key] = code
        return code

    # ------------------------------------------------------------
    # SPECIAL HANDLER: Average ticket size by day of week and month of year
    # ------------------------------------------------------------
    if "average ticket size" in q_lower and (
        "day of week" in q_lower
        or "day-of-week" in q_lower
        or "dow" in q_lower
        or "month of year" in q_lower
        or "month" in q_lower
    ):
        code = """
import pandas as pd
from helpers.analytics_helpers import resolve_column

df = mdf

if df is None or not isinstance(df, pd.DataFrame) or df.empty:
    result_df = pd.DataFrame()
    insight = "Cannot compute average ticket size because the main fact table (mdf) is not available."
else:
    df = df.copy()

    # Use dropoff_at as realized incoming revenue date; fallback to any date
    date_col = resolve_column(df, "dropoff_at", alias_family="date")
    if date_col is None:
        date_col = resolve_column(df, alias_family="date")

    amount_col = resolve_column(df, alias_family="amount")

    if date_col is None or date_col not in df.columns or amount_col is None or amount_col not in df.columns:
        result_df = pd.DataFrame()
        insight = "Cannot compute average ticket size because either the ticket date column or amount column could not be resolved."
    else:
        # Normalize raw UTC timestamps to Florida local time, tz-naive
        df[date_col] = (
            pd.to_datetime(df[date_col], utc=True, errors="coerce")
              .dt.tz_convert("America/New_York")
              .dt.tz_localize(None)
        )
        df[amount_col] = pd.to_numeric(df[amount_col], errors="coerce")

        df = df.dropna(subset=[date_col, amount_col])
        if df.empty:
            result_df = pd.DataFrame()
            insight = "Cannot compute average ticket size because all dates or amounts are invalid."
        else:
            # Day of week and month-of-year buckets
            df["day_of_week"] = df[date_col].dt.day_name()
            df["month_num"] = df[date_col].dt.month
            df["month_of_year"] = df[date_col].dt.month_name()

            # Order days Monday→Sunday
            dow_order = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
            df["day_of_week"] = pd.Categorical(df["day_of_week"], categories=dow_order, ordered=True)

            # Average ticket size by day of week
            avg_dow = (
                df.groupby("day_of_week")[amount_col]
                  .mean()
                  .reset_index(name="avg_ticket_size")
                  .sort_values("day_of_week")
            )

            # Average ticket size by month of year
            avg_month = (
                df.groupby(["month_num","month_of_year"])[amount_col]
                  .mean()
                  .reset_index()
                  .sort_values("month_num")
            )
            avg_month = avg_month.rename(columns={amount_col: "avg_ticket_size"})

            # Put both in a single long-format table: dimension, label, avg_ticket_size
            avg_dow["dimension"] = "day_of_week"
            avg_dow = avg_dow.rename(columns={"day_of_week": "label"})
            avg_dow = avg_dow[["dimension", "label", "avg_ticket_size"]]

            avg_month["dimension"] = "month_of_year"
            avg_month = avg_month.rename(columns={"month_of_year": "label"})
            avg_month = avg_month[["dimension", "label", "avg_ticket_size"]]

            result_df = pd.concat([avg_dow, avg_month], ignore_index=True)

            insight = (
                "Average ticket size has been computed by day of week and by month of year, "
                "using dropoff_at as the realized incoming revenue date."
            )
"""
        _LLM_CODE_CACHE[cache_key] = code
        return code
    # ------------------------------------------------------------
    # SPECIAL HANDLER: Percentile rank for visit frequency / CLV / retention
    # ------------------------------------------------------------
    if (
        "percentile" in q_lower
        and (
            "visit" in q_lower
            or "frequency" in q_lower
            or "clv" in q_lower
            or "lifetime value" in q_lower
            or "retention" in q_lower
            or "rank" in q_lower
        )
    ):
        code = f"""
import pandas as pd
from helpers.analytics_helpers import compute_customer_ltv, compute_visit_frequency

# Use defensive wrapper around customer_snapshot
cs = compute_customer_ltv(customer_snapshot)

if cs is None or not isinstance(cs, pd.DataFrame) or cs.empty:
    result_df = pd.DataFrame()
    insight = (
        "Cannot compute percentile ranks because customer_snapshot is not available."
    )
else:
    df = cs.copy()

    question_text = {repr(question)}
    ql = question_text.lower()

    metric_type = "clv"
    metric_label = "customer lifetime value"
    metric = None

    # Decide which metric the user is asking about
    if "visit" in ql or "frequency" in ql:
        metric_type = "visit_frequency"
    elif "retention" in ql:
        metric_type = "retention"
    else:
        # Default = CLV
        metric_type = "clv"

    # --------------------------------------------------------
    # 1) Build the metric series
    # --------------------------------------------------------
    if metric_type == "clv":
        # Prefer pickup-based lifetime revenue
        if "sales_pickup_lifetime" in df.columns:
            metric = pd.to_numeric(df["sales_pickup_lifetime"], errors="coerce")
            metric_label = "pickup-based lifetime revenue"
        elif "total" in df.columns:
            metric = pd.to_numeric(df["total"], errors="coerce")
            metric_label = "lifetime revenue from 'total'"
        else:
            # Fallback: sum all numeric columns as a crude revenue proxy
            numeric_cols = df.select_dtypes(include="number")
            if numeric_cols.empty:
                metric = None
            else:
                metric = numeric_cols.sum(axis=1)
                metric_label = "sum of numeric fields (proxy for CLV)"

    elif metric_type == "visit_frequency":
        # Prefer visits_lifetime, else visits_365
        if "visits_lifetime" in df.columns:
            metric = pd.to_numeric(df["visits_lifetime"], errors="coerce")
            metric_label = "lifetime visit count"
        elif "visits_365" in df.columns:
            metric = pd.to_numeric(df["visits_365"], errors="coerce")
            metric_label = "visits in last 365 days"
        else:
            metric = None

    elif metric_type == "retention":
        # Simple retention proxy: visits_365 (more visits in last year = higher retention)
        if "visits_365" in df.columns:
            metric = pd.to_numeric(df["visits_365"], errors="coerce")
            metric_label = "visits in last 365 days (retention proxy)"
        else:
            metric = None

    if metric is None:
        result_df = df.copy()
        insight = (
            "Cannot compute percentile ranks because the requested metric "
            f"({{metric_type}}) is not available in customer_snapshot."
        )
    else:
        metric = metric.fillna(0.0)

        if metric.empty:
            result_df = pd.DataFrame()
            insight = (
                "Cannot compute percentile ranks because the metric series is empty."
            )
        else:
            # --------------------------------------------------------
            # 2) Compute percentile rank (0–100, higher value = higher percentile)
            # --------------------------------------------------------
            pct_rank = metric.rank(method="average", pct=True) * 100.0

            # Build output table
            cust_id = df.get("customer_id", pd.Series(range(len(df)), index=df.index))
            cust_name = df.get(
                "customer_name",
                pd.Series([""] * len(df), index=df.index),
            )

            out = pd.DataFrame(
                {{
                    "customer_id": cust_id,
                    "customer_name": cust_name,
                    "metric_type": metric_type,
                    "metric_label": metric_label,
                    "metric_value": metric,
                    "percentile_rank": pct_rank,
                }}
            )

            # Sort from highest metric (and percentile) to lowest
            result_df = out.sort_values("metric_value", ascending=False).reset_index(drop=True)

            insight = (
                f"Computed percentile ranks (0–100) for {{metric_label}} across all customers. "
                f"Higher values correspond to higher percentiles."
            )
"""
        _LLM_CODE_CACHE[cache_key] = code
        return code
    # ------------------------------------------------------------
    # SPECIAL HANDLER: Compare Q1–Q4 this year vs last year (revenue by quarter)
    # ------------------------------------------------------------
    if (
        (
            "q1" in q_lower and "q4" in q_lower
            and "this year" in q_lower
            and "last year" in q_lower
        )
        or (
            "quarter" in q_lower
            and "this year" in q_lower
            and "last year" in q_lower
        )
    ):
        code = """
import pandas as pd
from helpers.analytics_helpers import resolve_column

df = mdf

if df is None or not isinstance(df, pd.DataFrame) or df.empty:
    result_df = pd.DataFrame()
    insight = "Cannot compare Q1–Q4 this year vs last year because the main fact table (mdf) is not available."
else:
    df = df.copy()

    # Use dropoff_at as realized incoming revenue date; fallback to any date
    date_col = resolve_column(df, "dropoff_at", alias_family="date")
    if date_col is None:
        date_col = resolve_column(df, alias_family="date")

    # Revenue / amount column
    amount_col = resolve_column(df, alias_family="amount")

    if (
        date_col is None or date_col not in df.columns
        or amount_col is None or amount_col not in df.columns
    ):
        result_df = pd.DataFrame()
        insight = (
            "Cannot compare Q1–Q4 this year vs last year because "
            "a usable date or amount column could not be resolved."
        )
    else:
        # Normalize raw UTC timestamps to Florida local time, tz-naive
        df[date_col] = (
            pd.to_datetime(df[date_col], utc=True, errors="coerce")
              .dt.tz_convert("America/New_York")
              .dt.tz_localize(None)
        )
        df[amount_col] = pd.to_numeric(df[amount_col], errors="coerce")

        df = df.dropna(subset=[date_col, amount_col])
        if df.empty:
            result_df = pd.DataFrame()
            insight = (
                "Cannot compare Q1–Q4 this year vs last year because "
                "all dates or amounts are invalid after cleaning."
            )
        else:
            # Extract year and quarter
            df["year"] = df[date_col].dt.year
            df["quarter"] = df[date_col].dt.quarter

            # Determine 'this year' and 'last year' from data
            max_year = int(df["year"].max())
            current_year = max_year
            prev_year = current_year - 1

            df_two = df[df["year"].isin([prev_year, current_year])].copy()
            if df_two.empty:
                result_df = pd.DataFrame()
                insight = (
                    "Cannot compare Q1–Q4 this year vs last year because "
                    "there is not enough data for the most recent two years."
                )
            else:
                # Aggregate revenue by (year, quarter)
                grouped = (
                    df_two.groupby(["year", "quarter"])[amount_col]
                          .sum()
                          .reset_index(name="revenue")
                )

                if grouped.empty:
                    result_df = pd.DataFrame()
                    insight = (
                        "Cannot compare Q1–Q4 this year vs last year because "
                        "aggregated quarterly revenue is empty."
                    )
                else:
                    # Pivot to compare quarters side-by-side: index=quarter, columns=year
                    pivot = grouped.pivot(index="quarter", columns="year", values="revenue")
                    pivot = pivot.sort_index()

                    # Ensure both year columns exist
                    if prev_year not in pivot.columns:
                        pivot[prev_year] = 0.0
                    if current_year not in pivot.columns:
                        pivot[current_year] = 0.0

                    # Reorder for readability: Q1..Q4, previous year then current year
                    pivot = pivot[[prev_year, current_year]]

                    # Build a clean result DataFrame
                    res = pd.DataFrame({"quarter": pivot.index})
                    res["quarter_label"] = res["quarter"].apply(lambda q: f"Q{int(q)}")
                    res[f"revenue_{prev_year}"] = pivot[prev_year].astype(float).values
                    res[f"revenue_{current_year}"] = pivot[current_year].astype(float).values

                    # YoY change and percentage
                    res["yoy_change"] = res[f"revenue_{current_year}"] - res[f"revenue_{prev_year}"]
                    with pd.option_context("mode.use_inf_as_na", True):
                        res["yoy_change_pct"] = (
                            res["yoy_change"]
                            / res[f"revenue_{prev_year}"].replace(0, pd.NA)
                        ) * 100.0

                    result_df = res[[
                        "quarter",
                        "quarter_label",
                        f"revenue_{prev_year}",
                        f"revenue_{current_year}",
                        "yoy_change",
                        "yoy_change_pct",
                    ]]

                    # Build an overall insight summarizing all four quarters
                    total_prev = float(result_df[f"revenue_{prev_year}"].sum())
                    total_curr = float(result_df[f"revenue_{current_year}"].sum())

                    if total_prev == 0:
                        yoy_text = (
                            f"Total revenue in {current_year} across Q1–Q4 was {total_curr:.2f}, "
                            f"while {prev_year} had no revenue (cannot compute percentage change)."
                        )
                    else:
                        delta_total = total_curr - total_prev
                        pct_total = 100.0 * delta_total / total_prev
                        sign_total = "higher" if delta_total >= 0 else "lower"
                        yoy_text = (
                            f"Total revenue in {current_year} across Q1–Q4 was {total_curr:.2f} "
                            f"vs {total_prev:.2f} in {prev_year}, "
                            f"{sign_total} by {abs(delta_total):.2f} ({pct_total:+.1f}% YoY)."
                        )

                    insight = (
                        "Quarterly revenue has been compared for Q1–Q4 this year vs last year. "
                        + yoy_text
                    )
"""
        _LLM_CODE_CACHE[cache_key] = code
        return code
    # ------------------------------------------------------------
    # SPECIAL HANDLER: Top/bottom quartile locations for key metrics
    #   - Uses location_id
    #   - Key metrics:
    #       * total_revenue (sum of amount)
    #       * avg_ticket_size (total_revenue / ticket_count)
    # ------------------------------------------------------------
    if (
        "quartile" in q_lower
        and ("location" in q_lower or "store" in q_lower or "branch" in q_lower or "market" in q_lower)
        and ("top" in q_lower or "bottom" in q_lower)
    ):
        code = """
import pandas as pd
from helpers.analytics_helpers import resolve_column

df = mdf

if df is None or not isinstance(df, pd.DataFrame) or df.empty:
    result_df = pd.DataFrame()
    insight = (
        "Cannot compute top/bottom quartile locations because the main fact table (mdf) is not available."
    )
else:
    df = df.copy()

    # Revenue / amount column
    amount_col = resolve_column(df, alias_family="amount")

    # Location dimension --> prefer location_id
    loc_col = resolve_column(df, alias_family="location_id")
    if loc_col is None:
        loc_col = resolve_column(df, alias_family="location")

    if amount_col is None or amount_col not in df.columns or loc_col is None or loc_col not in df.columns:
        result_df = pd.DataFrame()
        insight = (
            "Cannot compute top/bottom quartile locations because amount or location_id column "
            "could not be resolved."
        )
    else:
        df[amount_col] = pd.to_numeric(df[amount_col], errors="coerce")
        df = df.dropna(subset=[amount_col, loc_col])

        if df.empty:
            result_df = pd.DataFrame()
            insight = (
                "Cannot compute top/bottom quartile locations because all amounts or locations are invalid."
            )
        else:
            # Aggregate by location_id:
            #   - total_revenue = sum(amount)
            #   - ticket_count  = number of rows (tickets)
            grouped = (
                df.groupby(loc_col)[amount_col]
                  .agg(total_revenue="sum", ticket_count="size")
                  .reset_index()
            )

            if grouped.empty:
                result_df = pd.DataFrame()
                insight = (
                    "Cannot compute top/bottom quartile locations because aggregated metrics are empty."
                )
            else:
                # Average ticket size per location
                grouped["avg_ticket_size"] = grouped["total_revenue"] / grouped["ticket_count"].where(
                    grouped["ticket_count"] > 0
                )

                # Standardize location column name to 'location_id'
                if loc_col != "location_id":
                    grouped = grouped.rename(columns={loc_col: "location_id"})

                # Helper to compute quartile labels for a metric
                def _quartile_labels(series: pd.Series, metric_name: str) -> pd.Series:
                    s = pd.to_numeric(series, errors="coerce")
                    s = s.dropna()
                    if s.empty:
                        # If we cannot compute quartiles, everything is 'unknown'
                        return pd.Series(["unknown"] * len(series), index=series.index)

                    q1 = float(s.quantile(0.25))
                    q3 = float(s.quantile(0.75))

                    def label(x):
                        try:
                            vx = float(x)
                        except Exception:
                            return "unknown"
                        if vx >= q3:
                            return "top_quartile"
                        elif vx <= q1:
                            return "bottom_quartile"
                        else:
                            return "middle"
                    return series.apply(label)

                # Quartile labels for key metrics
                grouped["total_revenue_quartile"] = _quartile_labels(grouped["total_revenue"], "total_revenue")
                grouped["avg_ticket_size_quartile"] = _quartile_labels(grouped["avg_ticket_size"], "avg_ticket_size")

                result_df = grouped[[
                    "location_id",
                    "total_revenue",
                    "ticket_count",
                    "avg_ticket_size",
                    "total_revenue_quartile",
                    "avg_ticket_size_quartile",
                ]]

                # Build a short insight summarizing how many locations are in each quartile for revenue
                counts = result_df["total_revenue_quartile"].value_counts().to_dict()
                top_count = int(counts.get("top_quartile", 0))
                bottom_count = int(counts.get("bottom_quartile", 0))
                middle_count = int(counts.get("middle", 0))
                total_locations = len(result_df)

                insight = (
                    "Locations have been segmented into top/middle/bottom quartiles for total revenue and "
                    "average ticket size. "
                    f"Out of {total_locations} locations, about {top_count} are in the top revenue quartile, "
                    f"{bottom_count} are in the bottom revenue quartile, and {middle_count} are in the middle."
                )
"""
        _LLM_CODE_CACHE[cache_key] = code
        return code
    # ------------------------------------------------------------
    # SPECIAL HANDLER: Network median for average visit value
    # ------------------------------------------------------------
    if (
        "median" in q_lower
        and (
            "average visit value" in q_lower
            or "avg visit value" in q_lower
            or ("average" in q_lower and "visit" in q_lower and "value" in q_lower)
        )
    ):
        code = """
import pandas as pd
from helpers.analytics_helpers import compute_customer_ltv

cs = customer_snapshot

if cs is None or not isinstance(cs, pd.DataFrame) or cs.empty:
    result_df = pd.DataFrame()
    insight = (
        "Cannot compute network median for average visit value because customer_snapshot is not available."
    )
else:
    df = cs.copy()

    # Use compute_customer_ltv so we get a consistent revenue field
    clv_df = compute_customer_ltv(df)
    if clv_df is None or not isinstance(clv_df, pd.DataFrame) or clv_df.empty:
        result_df = pd.DataFrame()
        insight = (
            "Cannot compute network median for average visit value because customer lifetime value "
            "data is not available."
        )
    else:
        base = clv_df.copy()

        # Choose a revenue column: prefer sales_pickup_lifetime, else total, else numeric sum
        if "sales_pickup_lifetime" in base.columns:
            revenue = pd.to_numeric(base["sales_pickup_lifetime"], errors="coerce")
            revenue_label = "pickup-based lifetime revenue"
        elif "total" in base.columns:
            revenue = pd.to_numeric(base["total"], errors="coerce")
            revenue_label = "lifetime revenue from 'total'"
        else:
            numeric_cols = base.select_dtypes(include="number")
            if numeric_cols.empty:
                result_df = base.copy()
                insight = (
                    "Cannot compute network median for average visit value because no numeric revenue "
                    "columns are available."
                )
            else:
                revenue = numeric_cols.sum(axis=1)
                revenue_label = "sum of numeric fields"

        if 'revenue' not in locals():
            # We failed to pick a revenue column above
            result_df = base.copy()
            insight = (
                "Cannot compute network median for average visit value because no revenue measure "
                "could be derived."
            )
        else:
            revenue = revenue.fillna(0)

            # Find a visits column: prefer visits_lifetime, else visits_365, else any 'visits_' numeric
            visits_col = None
            if "visits_lifetime" in base.columns:
                visits_col = "visits_lifetime"
            elif "visits_365" in base.columns:
                visits_col = "visits_365"
            else:
                # Any numeric column whose name starts with 'visits_'
                visit_candidates = [
                    c for c in base.columns
                    if isinstance(c, str) and c.startswith("visits_")
                ]
                if visit_candidates:
                    visits_col = visit_candidates[0]

            if visits_col is None or visits_col not in base.columns:
                result_df = base.copy()
                insight = (
                    "Cannot compute network median for average visit value because no visit count "
                    "column (e.g. visits_lifetime) is available."
                )
            else:
                visits = pd.to_numeric(base[visits_col], errors="coerce").fillna(0)

                # Avoid division by zero; customers with 0 visits get NaN
                avg_visit_value = revenue / visits.replace(0, pd.NA)

                base_out = base.copy()
                base_out["avg_visit_value"] = avg_visit_value

                # Compute network median across customers with valid avg_visit_value
                valid = avg_visit_value.dropna()
                if valid.empty:
                    network_median = 0.0
                else:
                    network_median = float(valid.median())

                base_out["network_median_avg_visit_value"] = network_median

                # Keep the most relevant columns
                cols = []
                for c in ["customer_id", "customer_name"]:
                    if c in base_out.columns:
                        cols.append(c)
                cols += [revenue.name if hasattr(revenue, "name") else None]
                cols = [c for c in cols if c is not None]
                cols += ["avg_visit_value", "network_median_avg_visit_value"]

                result_df = base_out[cols]

                insight = (
                    f"The network median for average visit value (based on {revenue_label} divided by "
                    f"{visits_col}) is approximately {network_median:.2f} per visit."
                )
"""
        _LLM_CODE_CACHE[cache_key] = code
        return code
    # ------------------------------------------------------------
    # SPECIAL HANDLER: Customers in each value tier (Platinum/Gold/Silver/Bronze)
    # ------------------------------------------------------------
    if (
        "value tier" in q_lower
        or (
            ("platinum" in q_lower or "gold" in q_lower or "silver" in q_lower or "bronze" in q_lower)
            and "customer" in q_lower
        )
    ):
        code = """
import pandas as pd
from helpers.analytics_helpers import compute_customer_ltv

# Start from the customer snapshot, but run it through the defensive wrapper
cs = compute_customer_ltv(customer_snapshot)

if cs is None or not isinstance(cs, pd.DataFrame) or cs.empty:
    result_df = pd.DataFrame()
    insight = (
        "Cannot compute value tiers because customer_snapshot is not available."
    )
else:
    base = cs.copy()

    # ---------------------------------------------
    # 1) Choose a revenue column for tiering
    # ---------------------------------------------
    if "sales_pickup_lifetime" in base.columns:
        revenue = pd.to_numeric(base["sales_pickup_lifetime"], errors="coerce")
        revenue_label = "pickup-based lifetime revenue"
    elif "total" in base.columns:
        revenue = pd.to_numeric(base["total"], errors="coerce")
        revenue_label = "lifetime revenue from 'total'"
    else:
        numeric_cols = base.select_dtypes(include="number")
        if numeric_cols.empty:
            result_df = base.copy()
            insight = (
                "Cannot compute value tiers because no numeric revenue columns are available."
            )
        else:
            revenue = numeric_cols.sum(axis=1)
            revenue_label = "sum of numeric fields"

    if 'revenue' not in locals():
        result_df = base.copy()
        insight = (
            "Cannot compute value tiers because no revenue measure could be derived."
        )
    else:
        # Clean revenue series
        revenue = revenue.fillna(0.0)

        # If there are no customers, bail out early
        if revenue.empty:
            result_df = pd.DataFrame()
            insight = "Cannot compute value tiers because revenue series is empty."
        else:
            # ---------------------------------------------
            # 2) Rank-based tiers (Platinum/Gold/Silver/Bronze)
            #    Platinum  = top 5%
            #    Gold      = next 15%  (up to 20%)
            #    Silver    = next 30%  (up to 50%)
            #    Bronze    = bottom 50%
            # ---------------------------------------------
            tmp = base.copy()
            tmp["revenue_for_tier"] = revenue

            # Sort by revenue descending (highest spenders first)
            tmp = tmp.sort_values("revenue_for_tier", ascending=False).reset_index(drop=True)

            n = len(tmp)
            if n == 0:
                result_df = pd.DataFrame()
                insight = "Cannot compute value tiers because there are no customers."
            else:
                # rank_idx: 0,1,2,...,n-1  → rank_pct in [0,1)
                tmp["rank_idx"] = tmp.index
                tmp["rank_pct"] = tmp["rank_idx"] / max(n, 1)

                def assign_tier(p):
                    # p is rank_pct in [0,1)
                    try:
                        v = float(p)
                    except Exception:
                        return "Bronze"
                    if v < 0.05:
                        return "Platinum"
                    elif v < 0.20:
                        return "Gold"
                    elif v < 0.50:
                        return "Silver"
                    else:
                        return "Bronze"

                tmp["value_tier"] = tmp["rank_pct"].apply(assign_tier)

                # ---------------------------------------------
                # 3) Build summary: how many customers in each tier
                # ---------------------------------------------
                counts = tmp["value_tier"].value_counts().reindex(
                    ["Platinum", "Gold", "Silver", "Bronze"], fill_value=0
                )
                total = int(len(tmp))

                if total == 0:
                    result_df = pd.DataFrame()
                    insight = "Cannot compute value tiers because there are no customers."
                else:
                    pct = [float(c) * 100.0 / total for c in counts]

                    summary = pd.DataFrame(
                        {
                            "value_tier": ["Platinum", "Gold", "Silver", "Bronze"],
                            "customer_count": [int(c) for c in counts],
                            "percentage_of_customers": pct,
                        }
                    )

                    result_df = summary

                    insight = (
                        f"Customers have been segmented into Platinum/Gold/Silver/Bronze based on {revenue_label}, "
                        f"using rank-based tiers (top 5% Platinum, next 15% Gold, next 30% Silver, bottom 50% Bronze). "
                        f"Out of {total} customers, there are "
                        f"{counts.get('Platinum', 0)} Platinum, "
                        f"{counts.get('Gold', 0)} Gold, "
                        f"{counts.get('Silver', 0)} Silver, and "
                        f"{counts.get('Bronze', 0)} Bronze customers."
                    )
"""
        _LLM_CODE_CACHE[cache_key] = code
        return code

    # ------------------------------------------------------------
    # SPECIAL HANDLER: Customers "at risk" based on visit interval patterns
    # ------------------------------------------------------------
    if (
        "at risk" in q_lower
        and "visit" in q_lower
        and ("interval" in q_lower or "pattern" in q_lower or "patterns" in q_lower)
    ):
        code = """
import pandas as pd

cs = customer_snapshot

if cs is None or not isinstance(cs, pd.DataFrame) or cs.empty:
    result_df = pd.DataFrame()
    insight = (
        "Cannot determine at-risk customers because customer_snapshot is not available."
    )
else:
    df = cs.copy()

    if "last_visit" not in df.columns or "visits_interval_avg" not in df.columns:
        result_df = df.copy()
        insight = (
            "Cannot determine at-risk customers because 'last_visit' or 'visits_interval_avg' "
            "is missing from the snapshot."
        )
    else:
        # Parse dates and intervals
        df["last_visit"] = pd.to_datetime(df["last_visit"], errors="coerce")
        intervals = pd.to_numeric(df["visits_interval_avg"], errors="coerce")

        # Today in Florida, tz-naive
        today = pd.Timestamp.now(tz="America/New_York").normalize().tz_localize(None)

        # Days since last visit
        days_since = (today - df["last_visit"]).dt.days

        # Define "at risk" based on visit interval patterns:
        # - Has valid last_visit and interval > 0
        # - Already beyond their typical interval
        # - But not extremely overdue (we leave those to lapsed/overdue logic)
        mask_valid = (
            df["last_visit"].notna()
            & intervals.notna()
            & (intervals > 0)
            & days_since.notna()
        )

        # Business rule:
        #   normal        : days_since <= interval
        #   at risk       : interval < days_since <= 1.5 * interval
        #   more severe   : days_since > 1.5 * interval (handled elsewhere as overdue/lapsed)
        at_risk_mask = mask_valid & (days_since > intervals) & (days_since <= 1.5 * intervals)

        at_risk_df = df.loc[at_risk_mask].copy()

        # Add helper columns for inspection
        at_risk_df["days_since_last_visit"] = days_since[at_risk_mask]
        at_risk_df["interval_avg_days"] = intervals[at_risk_mask]

        at_risk_count = int(len(at_risk_df))
        total_customers = int(len(df))

        if "customer_id" in at_risk_df.columns and "customer_name" in at_risk_df.columns:
            cols = ["customer_id", "customer_name", "last_visit", "days_since_last_visit", "interval_avg_days"]
            result_df = at_risk_df[cols]
        else:
            result_df = at_risk_df

        if total_customers > 0:
            pct = 100.0 * at_risk_count / total_customers
        else:
            pct = 0.0

        insight = (
            f"There are {at_risk_count} customers who appear 'at risk' based on visit interval patterns "
            f"(they are slightly overdue relative to their typical interval), representing about {pct:.1f}% "
            f"of all customers."
        )
"""
        _LLM_CODE_CACHE[cache_key] = code
        return code


    # ------------------------------------------------------------
    # NEW SPECIAL HANDLER: New customer acquisition rate by month/quarter
    # ------------------------------------------------------------
    if "new customer acquisition rate" in q_lower or (
        "new customer" in q_lower and "acquisition" in q_lower
    ):
        code = f"""
import pandas as pd

cs = customer_snapshot

if cs is None or not isinstance(cs, pd.DataFrame) or cs.empty:
    result_df = pd.DataFrame()
    insight = "Cannot compute new customer acquisition rate because customer_snapshot is not available."
else:
    df = cs.copy()

    if "first_visit" not in df.columns:
        result_df = df.copy()
        insight = "Cannot compute new customer acquisition rate because 'first_visit' is not available on the snapshot."
    else:
        df["first_visit"] = pd.to_datetime(df["first_visit"], errors="coerce")
        df = df.dropna(subset=["first_visit"])
        if df.empty:
            result_df = pd.DataFrame()
            insight = "Cannot compute new customer acquisition rate because all 'first_visit' values are invalid."
        else:
            # Decide aggregation based on question text (month vs quarter)
            question_text = {repr(question)}
            ql = question_text.lower()
            use_quarter = ("quarter" in ql or "q1" in ql or "q2" in ql or "q3" in ql or "q4" in ql)

            if use_quarter:
                df["period"] = df["first_visit"].dt.to_period("Q").dt.to_timestamp()
                freq_label = "quarter"
            else:
                df["period"] = df["first_visit"].dt.to_period("M").dt.to_timestamp()
                freq_label = "month"

            # New customers = count of distinct customer_ids whose first_visit falls in that period
            if "customer_id" in df.columns:
                grouped = (
                    df.groupby("period")["customer_id"]
                    .nunique()
                    .reset_index(name="new_customers")
                )
            else:
                # Fallback: one row per customer already => just count rows
                grouped = (
                    df.groupby("period")
                    .size()
                    .reset_index(name="new_customers")
                )

            grouped = grouped.sort_values("period")

            result_df = grouped

            if len(grouped) == 0:
                insight = (
                    "New customer acquisition rate could not be computed because no valid first_visit dates were found."
                )
            else:
                # Build a small textual summary for the most recent period
                last_row = grouped.iloc[-1]
                last_period = last_row["period"]
                last_count = int(last_row["new_customers"])

                # Compare to previous period if exists
                if len(grouped) >= 2:
                    prev_row = grouped.iloc[-2]
                    prev_count = int(prev_row["new_customers"])
                    if prev_count == 0:
                        change_str = "previous period had zero new customers, so change is not comparable."
                    else:
                        delta = last_count - prev_count
                        pct = 100.0 * delta / prev_count
                        sign = "increase" if delta >= 0 else "decrease"
                        change_str = (
                            f"{{sign}} of {{abs(delta)}} new customers "
                            f"({{pct:+.1f}}% vs previous {{freq_label}})."
                        )
                else:
                    change_str = "there is only one period of data."

                insight = (
                    f"New customer acquisition by {{freq_label}} has been computed. "
                    f"The most recent {{freq_label}} ({{last_period}}) had {{last_count}} new customers; "
                    f"{{change_str}}"
                )
"""
        _LLM_CODE_CACHE[cache_key] = code
        return code
    # ------------------------------------------------------------
    # SPECIAL HANDLER: Year-over-year revenue growth by location (location_id)
    # ------------------------------------------------------------
    if (
        ("year-over-year" in q_lower or "year over year" in q_lower or "yoy" in q_lower)
        and "revenue" in q_lower
        and "location" in q_lower
    ):
        code = """
import pandas as pd
from helpers.analytics_helpers import resolve_column

df = mdf

if df is None or not isinstance(df, pd.DataFrame) or df.empty:
    result_df = pd.DataFrame()
    insight = "Cannot compute year-over-year revenue growth by location because the main fact table (mdf) is not available."
else:
    df = df.copy()

    # Use dropoff_at as realized incoming revenue date; fallback to any date
    date_col = resolve_column(df, "dropoff_at", alias_family="date")
    if date_col is None:
        date_col = resolve_column(df, alias_family="date")

    # Revenue / amount column
    amount_col = resolve_column(df, alias_family="amount")

    # Location dimension --> we only have location_id in this data
    loc_col = resolve_column(df, alias_family="location_id")
    if loc_col is None:
        # fallback: try the generic "location" alias family, just in case
        loc_col = resolve_column(df, alias_family="location")

    if (
        date_col is None or date_col not in df.columns
        or amount_col is None or amount_col not in df.columns
        or loc_col is None or loc_col not in df.columns
    ):
        result_df = pd.DataFrame()
        insight = (
            "Cannot compute year-over-year revenue growth by location because "
            "date, amount, or location_id columns could not be resolved."
        )
    else:
        # Normalize raw UTC timestamps to Florida local time, tz-naive
        df[date_col] = (
            pd.to_datetime(df[date_col], utc=True, errors="coerce")
              .dt.tz_convert("America/New_York")
              .dt.tz_localize(None)
        )
        df[amount_col] = pd.to_numeric(df[amount_col], errors="coerce")

        df = df.dropna(subset=[date_col, amount_col, loc_col])
        if df.empty:
            result_df = pd.DataFrame()
            insight = (
                "Cannot compute year-over-year revenue growth by location because "
                "all dates, amounts, or locations are invalid."
            )
        else:
            # Extract year and keep only rows with a valid year
            df["year"] = df[date_col].dt.year
            df = df.dropna(subset=["year"])

            # Optional: focus on the last few years based on data
            max_year = int(df["year"].max())
            min_year = max_year - 3  # last ~4 years
            df = df[df["year"] >= min_year]

            if df.empty:
                result_df = pd.DataFrame()
                insight = (
                    "Cannot compute year-over-year revenue growth by location because "
                    "there is not enough data in recent years."
                )
            else:
                # Aggregate revenue per (location_id, year)
                grouped = (
                    df.groupby([loc_col, "year"])[amount_col]
                      .sum()
                      .reset_index(name="revenue")
                )

                if grouped.empty:
                    result_df = pd.DataFrame()
                    insight = (
                        "Cannot compute year-over-year revenue growth by location because "
                        "aggregated revenue is empty."
                    )
                else:
                    # Standardize the column name to 'location_id'
                    if loc_col != "location_id":
                        grouped = grouped.rename(columns={loc_col: "location_id"})

                    # Sort and compute previous-year revenue per location_id
                    grouped = grouped.sort_values(["location_id", "year"])
                    grouped["prev_year_revenue"] = grouped.groupby("location_id")["revenue"].shift(1)

                    # YoY change and percentage
                    grouped["yoy_change"] = grouped["revenue"] - grouped["prev_year_revenue"]
                    with pd.option_context("mode.use_inf_as_na", True):
                        grouped["yoy_change_pct"] = (
                            grouped["yoy_change"] / grouped["prev_year_revenue"].replace(0, pd.NA)
                        ) * 100.0

                    result_df = grouped[[
                        "location_id",
                        "year",
                        "revenue",
                        "prev_year_revenue",
                        "yoy_change",
                        "yoy_change_pct",
                    ]]

                    # Build a short overall insight for the most recent year
                    latest_year = result_df["year"].max()
                    latest_rows = result_df[result_df["year"] == latest_year].copy()
                    latest_rows = latest_rows.dropna(subset=["prev_year_revenue"])

                    if latest_rows.empty:
                        insight = (
                            f"Year-over-year revenue by location_id has been computed up to {latest_year}, "
                            "but no previous-year baseline is available to calculate growth."
                        )
                    else:
                        mean_yoy = float(latest_rows["yoy_change_pct"].mean())
                        insight = (
                            f"Year-over-year revenue growth by location_id has been computed. "
                            f"For the most recent year ({latest_year}), average YoY change across locations "
                            f"is approximately {mean_yoy:+.1f}%."
                        )
"""
        _LLM_CODE_CACHE[cache_key] = code
        return code

    # ------------------------------------------------------------
    # (More special handlers can be added here later)
    # ------------------------------------------------------------
    client = OpenAI(api_key=api_key)

    # ------------------------------------------------------------
    # Optionally inject known variants for this question
    # ------------------------------------------------------------
    variants_text = ""
    q_norm = question.strip().lower()
    if q_norm:
        for entry in QUESTION_VARIANTS:
            base_q = (entry.get("question") or "").strip().lower()
            if base_q and base_q == q_norm:
                variants = entry.get("variants") or []
                if variants:
                    variants_text = (
                        "\n\nVariants for this question (same intent, different wording):\n"
                        + "\n".join(f"- {v}" for v in variants)
                    )
                break

    # ------------------------------------------------------------
    # SYSTEM PROMPT
    # ------------------------------------------------------------
    system = (
        "You are an analytics agent that writes ONLY Python code (no prose). "
        "Assume pandas is imported as pd and environment has: "
        "tables, mdf, customer_snapshot, "
        "runtime_normalize_mdf_env, resolve_column.\n\n"
        "BUSINESS CONTEXT: dry cleaning, visits, revenue, customer value, route vs retail.\n"
        "Use visit-based metrics when question mentions visits.\n"
        "Use dropoff for incoming sales and pickup for outgoing sales.\n"
        "Use customer_snapshot if available for customer-level metrics.\n"
        "Always resolve columns via resolve_column, never hard-code.\n"
        "CRITICAL RULES FOR resolve_column:\n"
        "  - Always assign its result to a local variable, e.g. 'col = resolve_column(df, ... )'.\n"
        "  - Then check 'if col is None or col not in df.columns:' and handle that case by\n"
        "    either skipping that metric or returning an empty result with a clear insight.\n"
        "  - NEVER do df[col] without first verifying that col is not None and exists in df.\n\n"
        "IMPORTANT:\n"
        "  - Treat 'mdf' and 'customer_snapshot' as existing pandas DataFrames.\n"
        "  - Never reassign 'mdf' or 'customer_snapshot' to strings, lists, or other types.\n"
        "  - Do NOT use the '.empty' attribute on arbitrary objects. If you need to check\n"
        "    if a DataFrame is empty, use 'len(df) == 0' or only call '.empty' on a known DataFrame.\n"
        "  - Never call .groupby() without a 'by' or 'level' argument.\n"
        "Produce only:\n"
        "    result_df = <DataFrame>\n"
        "    insight = <string>\n"
        "assume mdf is already prepared from the appropriate fact table.\n"
        "TIMEZONE RULES (IMPORTANT):\n"
        "  • All timestamps in the raw data (e.g. '2025-12-01T22:15:00.000Z') should be treated as UTC.\n"
        "  • When you parse a datetime column from mdf, normalize it to Florida local, tz-naive, e.g.:\n"
        "        df[col] = pd.to_datetime(df[col], utc=True, errors='coerce')\\\n"
        "                    .dt.tz_convert('America/New_York')\\\n"
        "                    .dt.tz_localize(None)\n"
        "    After this, df[col] is in Florida local time with NO timezone attached.\n"
        "  • When you build explicit date boundaries from strings (like '2024-09-09'), create tz-naive timestamps, e.g.:\n"
        "        start = pd.to_datetime('2024-09-09', errors='coerce')\n"
        "        end = pd.to_datetime('2024-10-10', errors='coerce')\n"
        "    Do NOT pass a tz= argument to pd.Timestamp or pd.to_datetime for boundaries.\n"
        "  • For relative 'today', use Florida local and then drop tz:\n"
        "        today = pd.Timestamp.now(tz='America/New_York').normalize().tz_localize(None)\n"
        "  • NEVER write pd.Timestamp('2024-09-09', tz='America/New_York') or similar tz-aware boundaries,\n"
        "    because mdf date columns must be tz-naive after normalization.\n"
        "  • 'America/New_York' must ONLY appear either in tz='America/New_York' in .now(),\n"
        "    or in .dt.tz_convert('America/New_York') when normalizing from UTC.\n"
        "  • Do not parse timezone strings as dates (never pass 'America/New_York' as the date value).\n"
        "CHANNEL RULES:\n"
        "  - Route vs Retail is derived from location.name (contains 'route' => ROUTE else RETAIL).\n"
        "  - If customer_snapshot has channel_segment/dominant_channel/route_visits/retail_visits, use them instead of recomputing.\n"
        "  - Never rely on route.name to classify channel.\n"
    )

    # ------------------------------------------------------------
    # USER MESSAGE
    # ------------------------------------------------------------
    rules_block = ""
    if business_rules:
        rules_block = (
            "\n\nDOCUMENTED BUSINESS RULES (MUST BE APPLIED EXACTLY):\n"
            f"{business_rules}\n"
            "These rules define HOW metrics are calculated. "
            "DO NOT explain them. EXECUTE them using the tables.\n"
        )

    user = (
        f"Question: {question}"
        f"{variants_text}\n\n"
        f"Available tables: {list(tables.keys())}\n\n"
        f"Schema and hints:\n{schema_str}"
        f"{rules_block}\n\n"
        "CRITICAL EXECUTION INSTRUCTION:\n"
        "- Use the DOCUMENTED BUSINESS RULES above to COMPUTE results.\n"
        "- Do NOT explain formulas.\n"
        "- Do NOT restate rules.\n"
        "- Produce executable pandas code.\n\n"
        "Use mdf as main DataFrame. Resolve date/amount/customer columns via resolve_column. "
        "Output result_df and insight. Prefer customer_snapshot for customer-level metrics when available."
    )

    # ===========================================================
    # 1) Try Responses API
    # ===========================================================
    try:
        resp = client.responses.create(
            model=model,
            instructions=system,
            input=user,
            temperature=0.1,
            max_output_tokens=800,
        )

        text = getattr(resp, "output_text", None)

        if not text and getattr(resp, "output", None):
            parts = []
            for item in resp.output:
                for block in getattr(item, "content", []) or []:
                    text_obj = getattr(block, "text", None)
                    if text_obj:
                        value = getattr(text_obj, "value", None)
                        if value:
                            parts.append(value)
            text = "\n".join(parts).strip()

        if not text:
            raise ValueError("Empty response from Responses API")

        code = _strip_code_fences(text)
        if not code:
            raise ValueError("Empty code after stripping fences")

        _LLM_CODE_CACHE[cache_key] = code
        return code

    except Exception as e:
        print("llm_codegen: Responses API failed → fallback to chat.completions:", repr(e))

    # ===========================================================
    # 2) Fallback → Chat Completions
    # ===========================================================
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=0.1,
            max_tokens=800,
        )

        content = resp.choices[0].message.content or ""
        code = _strip_code_fences(content)

        if code:
            _LLM_CODE_CACHE[cache_key] = code
        return code or None

    except Exception as e2:
        print("llm_codegen: chat.completions failed:", repr(e2))
        return None