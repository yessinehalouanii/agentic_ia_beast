# helpers/analytics_helpers.py

import pandas as pd
from difflib import get_close_matches
from typing import Any
import numpy as np

# ----------------------------
# Column alias map
# ----------------------------
ALIASES_UNIVERSAL = {
    "location": ["location", "location_id", "store", "branch", "market", "location_name"],
    # Allow alias_family="location_id" to behave the same as "location"
    "location_id": ["location_id", "store", "branch", "market", "location_name"],
    "customer": ["customer_id", "client_id", "cust_id"],
    "customer_name": ["customer_name", "name", "full_name", "first_name", "last_name"],
    "date": ["pickup_at", "dropoff_at", "created_at", "updated_at", "order_date", "timestamp"],
    "amount": ["total", "sale_amount", "revenue", "price", "amount"],
    "pieces": ["pieces", "items", "quantity", "qty"],
    "route": ["route_id", "route", "route_name", "line", "line_id"],
    "visit": ["visit_id", "visit", "visit_no"],
    "channel": ["channel", "customer_type", "type_name"],
}

# ============================================================
# Column resolver (used everywhere, also imported by runtime)
# ============================================================
def resolve_column(df: Any, user_term=None, alias_family=None):
    """
    Safely resolve a column name in a DataFrame.

    - If df is not a DataFrame, or is empty, return None.
    - This defends against LLM mistakes like resolve_column("mdf", ...).
    """
    if not isinstance(df, pd.DataFrame):
        return None

    if df.empty:
        return None

    if not user_term:
        user_term = alias_family or "date"

    lower = {str(c).lower(): c for c in df.columns}
    user_term_l = str(user_term).lower()

    # 1) Direct match on the column name
    variants = [user_term, user_term.replace(" ", "_"), user_term.replace("_", " ")]
    for v in variants:
        key = str(v).lower()
        if key in lower:
            return lower[key]

    # 2) Alias family match
    if alias_family in ALIASES_UNIVERSAL:
        for c in ALIASES_UNIVERSAL[alias_family]:
            for v in [c, c.replace("_", " "), c.replace(" ", "_")]:
                key = v.lower()
                if key in lower:
                    return lower[key]

    # 3) Fuzzy match on user_term
    col_names = list(lower.keys())
    matches = get_close_matches(user_term_l, col_names, n=1, cutoff=0.75)
    if matches:
        return lower[matches[0]]

    # 4) Extra heuristics for LOCATION / LOCATION_ID
    #    This protects questions like:
    #    "What is the total revenue for location 1 this month?"
    if alias_family in ("location", "location_id") or ("location" in user_term_l) or ("loc_id" in user_term_l):
        # First prefer anything that looks like location_id
        for col in df.columns:
            name = str(col).lower()
            if "location_id" in name:
                return col
        # Then any column with "location" in it
        for col in df.columns:
            name = str(col).lower()
            if "location" in name:
                return col

    # Final: log once and return None
    print(
        f"resolve_column: could not resolve user_term='{user_term}' alias_family='{alias_family}' "
        f"against columns={list(df.columns)}"
    )
    return None


# ============================================================
# Visits helpers
# ============================================================
def resolve_visit_col(mdf: pd.DataFrame):
    return resolve_column(mdf, alias_family="visit")


# ============================================================
# Customer-level simple helpers
# ============================================================
def compute_customer_ltv(customer_snapshot: pd.DataFrame) -> pd.DataFrame:
    """
    Defensive: only treat it as a snapshot if it's really a DataFrame.
    This avoids 'str' object has no attribute 'empty' when LLM passes
    the wrong thing.
    """
    if not isinstance(customer_snapshot, pd.DataFrame):
        return pd.DataFrame()
    if customer_snapshot.empty:
        return pd.DataFrame()
    return customer_snapshot


def compute_visit_frequency(customer_snapshot: pd.DataFrame) -> pd.DataFrame:
    """
    Same defensive pattern as compute_customer_ltv.
    """
    if not isinstance(customer_snapshot, pd.DataFrame):
        return pd.DataFrame()
    if customer_snapshot.empty:
        return pd.DataFrame()

    cols = [
        c
        for c in ["customer_id", "visits_lifetime", "visits_365", "visits_interval_avg"]
        if c in customer_snapshot.columns
    ]
    return customer_snapshot[cols]


# ============================================================
# Column cleaning helper (local, for snapshot)
# ============================================================
def _clean_columns_local(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure columns are:
      - single-level (flatten MultiIndex if needed)
      - without duplicate names (keep first occurrence).
    """
    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        return df

    df = df.copy()

    # 1) Flatten MultiIndex columns like ('pickup_at','x') -> 'pickup_at__x'
    if isinstance(df.columns, pd.MultiIndex):
        flat_cols = []
        for col in df.columns:
            parts = [str(p) for p in col if p is not None]
            flat_cols.append("__".join(parts))
        df.columns = flat_cols

    # 2) Drop exact duplicate column names
    if df.columns.duplicated().any():
        dupes = list(df.columns[df.columns.duplicated()])
        print(f"_clean_columns_local: dropping duplicate columns: {dupes}")
        df = df.loc[:, ~df.columns.duplicated()]

    return df


# ============================================================
# ES nested columns expander
# ============================================================
def _expand_es_nested_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Expand common nested ES objects (customer, location, route, items)
    into simple scalar columns so that resolve_column() can find them.

    Works safely on non-ES data too (it just does nothing if those cols
    aren't dict/list).
    """
    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        return df

    df = df.copy()

    # --------- customer { first_name, last_name } ----------
    if "customer" in df.columns:
        is_dict = df["customer"].map(lambda x: isinstance(x, dict)).any()
        if is_dict:
            df["customer_first_name"] = df["customer"].map(
                lambda x: (x or {}).get("first_name")
            )
            df["customer_last_name"] = df["customer"].map(
                lambda x: (x or {}).get("last_name")
            )

            # Combined name
            df["customer_name"] = (
                df["customer_first_name"].fillna("").astype(str)
                + " "
                + df["customer_last_name"].fillna("").astype(str)
            ).str.strip()

    # --------- location { name } ----------
    if "location" in df.columns:
        is_dict = df["location"].map(lambda x: isinstance(x, dict)).any()
        if is_dict:
            df["location_name"] = df["location"].map(
                lambda x: (x or {}).get("name")
            )

    # --------- route { name } ----------
    if "route" in df.columns:
        is_dict = df["route"].map(lambda x: isinstance(x, dict)).any()
        if is_dict:
            df["route_name"] = df["route"].map(
                lambda x: (x or {}).get("name")
            )

    # --------- items [ { ... } ] ----------
    if "items" in df.columns:
        is_list = df["items"].map(lambda x: isinstance(x, list)).any()
        if is_list:
            df["items_count"] = df["items"].map(lambda x: len(x or []))

    return df


# ============================================================
# Customer snapshot builder (used by runtime)
# ============================================================
def _find_pickup_amount_col(df: pd.DataFrame, resolver):
    """
    Try to find a 'realized revenue' column first (sales_pickup / pickup_sales / outgoing_sales),
    and if nothing is found, fall back to a generic monetary column (amount/total/etc.).

    IMPORTANT: here we **avoid fuzzy matching** for these synthetic names,
    and only look for exact (case-insensitive) matches. This prevents
    accidentally matching 'pickup_sales' -> 'pickup_at'.
    """
    if df is None or df.empty:
        return None

    # Case-insensitive lookup
    cols_lower = {str(c).lower(): c for c in df.columns}

    # 1) Exact search for known revenue columns
    explicit_terms = ["sales_pickup", "pickup_sales", "outgoing_sales", "realized_sales"]
    for term in explicit_terms:
        for v in (term, term.replace(" ", "_"), term.replace("_", " ")):
            key = v.lower()
            if key in cols_lower:
                return cols_lower[key]

    # 2) Fallback: generic amount family (total, amount, price, etc.)
    return resolver(df, alias_family="amount")


def _build_customer_snapshot(mdf: pd.DataFrame, resolver) -> pd.DataFrame:
    """
    Build a per-customer snapshot DataFrame with:
      - customer_id
      - customer_name (if available)
      - original_signup  (if column exists; else proxy = first_visit)
      - first_visit, last_visit
      - visits_lifetime  (distinct visits across all available data)
      - visits_365       (distinct visits in last 365 days)
      - visits_interval_avg (avg days between visits)
      - sales_pickup_lifetime
      - sales_pickup_30 / 60 / 90 / 365
    """
    # --- Guard: must be a non-empty DataFrame -------------------------
    if mdf is None or not isinstance(mdf, pd.DataFrame) or mdf.empty:
        return pd.DataFrame()

    # Always clean columns first (flatten+dedupe)
    df = _clean_columns_local(mdf)

    # ✅ Expand common ES nested objects (customer/location/route/items)
    df = _expand_es_nested_columns(df)

    # Core identifiers
    cust_src_col = resolver(df, alias_family="customer")
    if cust_src_col is None:
        return pd.DataFrame()

    name_col = resolver(df, alias_family="customer_name")

    # Signup / relationship start
    signup_col = (
        resolver(df, "original_signup", alias_family=None)
        or resolver(df, "signup_date", alias_family=None)
    )

    # Dates: dropoff vs pickup
    dropoff_col = resolver(df, "dropoff_at", alias_family="date")
    pickup_col = resolver(df, "pickup_at", alias_family="date")

    # Fallback: if neither is found, try any generic date column
    if dropoff_col is None and pickup_col is None:
        dropoff_col = resolver(df, alias_family="date")

    # Main date used for defining a "visit" (detail date preferred)
    main_date_col = dropoff_col or pickup_col

    # Revenue column for realized revenue (pickup-based)
    pickup_amt_col = _find_pickup_amount_col(df, resolver)

    # 🔒 Safety: don't allow amount column to be the same as the pickup date column
    if pickup_col and pickup_amt_col and pickup_col == pickup_amt_col:
        print(
            f"_build_customer_snapshot: pickup_amt_col '{pickup_amt_col}' "
            f"is the same as pickup_col; skipping revenue stats for safety."
        )
        pickup_amt_col = None

    # --- Safe datetime parsing helper ---------------------------------
    def _safe_to_datetime(df_: pd.DataFrame, col_name: str) -> None:
        if not col_name or col_name not in df_.columns:
            return
        try:
            col_data = df_[col_name]

            # If somehow this is a DataFrame (duplicate names), use the first column
            if isinstance(col_data, pd.DataFrame):
                print(
                    f"_build_customer_snapshot: column '{col_name}' "
                    f"is a DataFrame with subcolumns {list(col_data.columns)}; "
                    "using the first one."
                )
                col_data = col_data.iloc[:, 0]

            df_[col_name] = pd.to_datetime(col_data, errors="coerce")
        except Exception as e:
            print(f"_build_customer_snapshot: failed to parse '{col_name}' as datetime: {e}")
            df_[col_name] = pd.NaT

    # Ensure the relevant columns are datetime
    for c in [dropoff_col, pickup_col, main_date_col, signup_col]:
        _safe_to_datetime(df, c)

    # Business "today" in naive local time
    today = pd.Timestamp.now().normalize()
    d30 = today - pd.Timedelta(days=30)
    d60 = today - pd.Timedelta(days=60)
    d90 = today - pd.Timedelta(days=90)
    d365 = today - pd.Timedelta(days=365)

    # --------------------------------------------------
    # VISIT-BASED METRICS  (prefer visit_id when present)
    # --------------------------------------------------
    visit_col = resolve_visit_col(df)

    if main_date_col and main_date_col in df.columns:
        if visit_col is not None and visit_col in df.columns:
            # --- Use visit_id as the atomic unit of a visit ---
            tmp = df[[cust_src_col, visit_col, main_date_col]].dropna(
                subset=[cust_src_col, visit_col, main_date_col]
            ).copy()

            # visit_day = earliest date for that visit (per customer, per visit_id)
            tmp["visit_day"] = (
                tmp.groupby([cust_src_col, visit_col])[main_date_col]
                   .transform("min")
                   .dt.normalize()
            )

            # One row per (customer, visit_id)
            tmp_unique = tmp[[cust_src_col, visit_col, "visit_day"]].drop_duplicates(
                subset=[cust_src_col, visit_col]
            )

            grp = tmp_unique.groupby(cust_src_col)["visit_day"]

            first_visit = grp.min()
            last_visit = grp.max()
            visits_lifetime = grp.nunique()

            # Visits in last 365 days = distinct visit_ids whose visit_day is recent
            recent_mask = tmp_unique["visit_day"] >= d365
            visits_365 = (
                tmp_unique[recent_mask]
                .groupby(cust_src_col)["visit_day"]
                .nunique()
            )

        else:
            # --- Fallback: no visit_id, use one visit per day per customer ---
            tmp = df[[cust_src_col, main_date_col]].dropna(
                subset=[cust_src_col, main_date_col]
            ).copy()
            tmp["visit_day"] = tmp[main_date_col].dt.normalize()

            grp = tmp.groupby(cust_src_col)["visit_day"]

            first_visit = grp.min()
            last_visit = grp.max()
            visits_lifetime = grp.nunique()

            # Visits in last 365 days
            recent_mask = tmp["visit_day"] >= d365
            visits_365 = (
                tmp[recent_mask]
                .groupby(cust_src_col)["visit_day"]
                .nunique()
            )

        # Average interval between visits (days) – based on visit_day
        def _avg_interval(s: pd.Series):
            s = s.sort_values().drop_duplicates()
            if len(s) < 2:
                return pd.NA
            diffs = s.diff().dt.days.dropna()
            return diffs.mean() if len(diffs) > 0 else pd.NA

        visits_interval_avg = grp.apply(_avg_interval)

        visit_stats = pd.DataFrame(
            {
                "first_visit": first_visit,
                "last_visit": last_visit,
                "visits_lifetime": visits_lifetime,
                "visits_365": visits_365,
                "visits_interval_avg": visits_interval_avg,
            }
        )
    else:
        visit_stats = pd.DataFrame(index=df[cust_src_col].dropna().unique())

    # --------------------------------------------------
    # REVENUE METRICS (pickup / realized)
    # --------------------------------------------------
    try:
        if (
            pickup_col
            and pickup_col in df.columns
            and pickup_amt_col
            and pickup_amt_col in df.columns
        ):
            rev_df = df[[cust_src_col, pickup_col, pickup_amt_col]].dropna().copy()

            # Clean columns again in case of weirdness
            rev_df = _clean_columns_local(rev_df)

            _safe_to_datetime(rev_df, pickup_col)
            rev_df = rev_df.dropna(subset=[pickup_col])

            # Force amount to numeric so we never sum datetimes
            rev_df[pickup_amt_col] = pd.to_numeric(
                rev_df[pickup_amt_col],
                errors="coerce",
            )

            # If everything became NaN → skip revenue stats
            if rev_df[pickup_amt_col].dropna().empty:
                print(
                    f"_build_customer_snapshot: pickup_amt_col '{pickup_amt_col}' "
                    "is non-numeric; skipping revenue stats"
                )
                revenue_stats = pd.DataFrame(index=visit_stats.index)
            else:
                def _sum_in_window(start, end):
                    mask = (rev_df[pickup_col] >= start) & (rev_df[pickup_col] < end)
                    if not mask.any():
                        return pd.Series(dtype=float)
                    return (
                        rev_df.loc[mask]
                            .groupby(cust_src_col)[pickup_amt_col]
                            .sum()
                    )

                lifetime = rev_df.groupby(cust_src_col)[pickup_amt_col].sum()
                r30 = _sum_in_window(d30, today)
                r60 = _sum_in_window(d60, today)
                r90 = _sum_in_window(d90, today)
                r365 = _sum_in_window(d365, today)

                revenue_stats = pd.DataFrame(
                    {
                        "sales_pickup_lifetime": lifetime,
                        "sales_pickup_30": r30,
                        "sales_pickup_60": r60,
                        "sales_pickup_90": r90,
                        "sales_pickup_365": r365,
                    }
                )
        else:
            revenue_stats = pd.DataFrame(index=visit_stats.index)
    except Exception as e:
        print(f"_build_customer_snapshot: failed to build revenue stats: {e}")
        revenue_stats = pd.DataFrame(index=visit_stats.index)

    # --------------------------------------------------
    # COMBINE INTO SNAPSHOT
    # --------------------------------------------------
    base_index = visit_stats.index.union(revenue_stats.index)
    snapshot = pd.DataFrame(index=base_index)
    snapshot["customer_id"] = snapshot.index

    # Attach visit metrics
    for col in [
        "first_visit",
        "last_visit",
        "visits_lifetime",
        "visits_365",
        "visits_interval_avg",
    ]:
        if col in visit_stats.columns:
            snapshot[col] = visit_stats.reindex(base_index)[col]

    # Attach revenue metrics
    for col in [
        "sales_pickup_lifetime",
        "sales_pickup_30",
        "sales_pickup_60",
        "sales_pickup_90",
        "sales_pickup_365",
    ]:
        if col in revenue_stats.columns:
            snapshot[col] = revenue_stats.reindex(base_index)[col]

    # Original signup date if explicitly available; otherwise proxy = first_visit
    if signup_col and signup_col in df.columns:
        signup = df.groupby(cust_src_col)[signup_col].min()
        snapshot["original_signup"] = signup.reindex(base_index)
    else:
        snapshot["original_signup"] = snapshot.get("first_visit")

    # Customer name (if we can resolve it)
    if name_col and name_col in df.columns:
        name_series = df.groupby(cust_src_col)[name_col].first()
        snapshot["customer_name"] = name_series.reindex(base_index)

    # Nice column order (extras appended at the end)
    cols_order = [
        "customer_id",
        "customer_name",
        "original_signup",
        "first_visit",
        "last_visit",
        "visits_lifetime",
        "visits_365",
        "visits_interval_avg",
        "sales_pickup_30",
        "sales_pickup_60",
        "sales_pickup_90",
        "sales_pickup_365",
        "sales_pickup_lifetime",
    ]
    existing = [c for c in cols_order if c in snapshot.columns]
    others = [c for c in snapshot.columns if c not in existing]
    snapshot = snapshot[existing + others]

    return snapshot.reset_index(drop=True)


def add_service_channel_visits(mdf: pd.DataFrame) -> pd.DataFrame:
    """
    Adds a visit-level service_channel column: ROUTE vs RETAIL.
    Uses location.name (best signal from your examples).
    """
    if mdf is None or not isinstance(mdf, pd.DataFrame) or mdf.empty:
        return mdf

    df = mdf.copy()

    # ✅ Expand nested ES objects so location_name exists
    df = _expand_es_nested_columns(df)

    loc_name_col = resolve_column(df, "location.name", alias_family="location")

    if loc_name_col is None or loc_name_col not in df.columns:
        df["service_channel"] = "RETAIL"
        return df

    loc_name = df[loc_name_col].astype(str).str.lower()
    is_route = loc_name.str.contains("route", na=False)

    df["service_channel"] = np.where(is_route, "ROUTE", "RETAIL")
    return df


def build_customer_channel_map_from_visits(visits_df: pd.DataFrame) -> pd.DataFrame:
    """
    Build per-customer channel labels from ACTUAL visits behavior:
      - ROUTE_ONLY / RETAIL_ONLY / BOTH
      - plus dominant_channel (ROUTE or RETAIL) if you need a single label
    """
    out_cols = ["customer_id", "route_visits", "retail_visits", "channel_segment", "dominant_channel"]

    if visits_df is None or not isinstance(visits_df, pd.DataFrame) or visits_df.empty:
        return pd.DataFrame(columns=out_cols)

    df = visits_df.copy()

    # ✅ Expand nested ES objects once
    df = _expand_es_nested_columns(df)

    cust_col = resolve_column(df, alias_family="customer")
    if cust_col is None or cust_col not in df.columns:
        return pd.DataFrame(columns=out_cols)

    # Ensure service_channel exists
    if "service_channel" not in df.columns:
        df = add_service_channel_visits(df)

    # ✅ Try to count DISTINCT visits using visit_id (or alias)
    visit_col = resolve_visit_col(df)

    if visit_col is not None and visit_col in df.columns:
        # Count UNIQUE visit IDs per customer per channel
        counts = (
            df.dropna(subset=[cust_col, "service_channel", visit_col])
              .groupby([cust_col, "service_channel"])[visit_col]
              .nunique()
              .unstack(fill_value=0)
              .reset_index()
        )
    else:
        # Fallback: row count (assumes 1 row = 1 visit)
        counts = (
            df.dropna(subset=[cust_col, "service_channel"])
              .groupby([cust_col, "service_channel"])
              .size()
              .unstack(fill_value=0)
              .reset_index()
        )

    # Normalize channel columns
    if "ROUTE" not in counts.columns:
        counts["ROUTE"] = 0
    if "RETAIL" not in counts.columns:
        counts["RETAIL"] = 0

    counts = counts.rename(columns={cust_col: "customer_id", "ROUTE": "route_visits", "RETAIL": "retail_visits"})

    # Segment
    def seg(row):
        r = int(row["route_visits"])
        t = int(row["retail_visits"])
        if r > 0 and t > 0:
            return "BOTH"
        if r > 0 and t == 0:
            return "ROUTE_ONLY"
        if t > 0 and r == 0:
            return "RETAIL_ONLY"
        return "UNKNOWN"

    counts["channel_segment"] = counts.apply(seg, axis=1)

    # Dominant single label (ties -> RETAIL, same as before)
    counts["dominant_channel"] = np.where(counts["route_visits"] > counts["retail_visits"], "ROUTE", "RETAIL")

    return counts[out_cols]


def merge_customer_channel_into_snapshot(customer_snapshot: pd.DataFrame, channel_map: pd.DataFrame) -> pd.DataFrame:
    """
    Adds channel_segment + dominant_channel into your customer_snapshot.
    """
    if customer_snapshot is None or not isinstance(customer_snapshot, pd.DataFrame) or customer_snapshot.empty:
        return customer_snapshot

    if channel_map is None or not isinstance(channel_map, pd.DataFrame) or channel_map.empty:
        out = customer_snapshot.copy()
        if "channel_segment" not in out.columns:
            out["channel_segment"] = "UNKNOWN"
        if "dominant_channel" not in out.columns:
            out["dominant_channel"] = "UNKNOWN"
        return out

    out = customer_snapshot.copy()

    if "customer_id" not in out.columns:
        return out

    out = out.merge(channel_map, on="customer_id", how="left")

    out["channel_segment"] = out["channel_segment"].fillna("UNKNOWN")
    out["dominant_channel"] = out["dominant_channel"].fillna("UNKNOWN")
    out["route_visits"] = out.get("route_visits", 0).fillna(0)
    out["retail_visits"] = out.get("retail_visits", 0).fillna(0)

    return out
