# abi/runtime.py

import pandas as pd
import numpy as np  # ✅ For NaN/Inf cleanup
from typing import Any, Dict, Tuple

# Only this at top to avoid circular imports
from helpers.analytics_helpers import resolve_column

# Business timezone (Florida)
BUSINESS_TZ = "America/New_York"

# ---------------------------------------------------------------------
# Date hints for normalization
# ---------------------------------------------------------------------

_DATE_HINTS = [
    "created_at",
    "updated_at",
    "dropoff_at",
    "ready_at",
    "pickup_at",
    "created",
    "order_date",
    "sale_date",
    "purchased_at",
    "timestamp",
    "datetime",
    "date",
]


def _normalize_dates_inplace(df: pd.DataFrame) -> None:
    """
    Normalize date-like columns to tz-naive Florida-local timestamps.

    NOTE:
    - If your raw timestamps are strings like "...Z", pandas often parses them as tz-aware UTC.
    - We convert tz-aware timestamps to BUSINESS_TZ and then drop tz info (tz-naive).
    """
    if df is None or df.empty:
        return

    for col in df.columns:
        name = str(col).lower()
        if any(h in name for h in _DATE_HINTS):
            try:
                parsed = pd.to_datetime(df[col], errors="coerce")

                # If tz-aware -> convert to Florida local, then drop tz (tz-naive)
                if getattr(parsed.dt, "tz", None) is not None:
                    parsed = parsed.dt.tz_convert(BUSINESS_TZ).dt.tz_localize(None)

                df[col] = parsed

            except Exception as e:
                print(f"_normalize_dates_inplace: failed on column {col}: {e}")


# ---------------------------------------------------------------------
# Column cleaning helper
# ---------------------------------------------------------------------

def _clean_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure columns are:
      - single-level (flatten MultiIndex if needed)
      - without duplicate names (keep first occurrence).
    """
    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        return df

    df = df.copy()

    # 1) Flatten MultiIndex columns: ('pickup_at', 'x') -> 'pickup_at__x'
    if isinstance(df.columns, pd.MultiIndex):
        flat_cols = []
        for col in df.columns:
            parts = [str(p) for p in col if p is not None]
            flat_cols.append("__".join(parts))
        df.columns = flat_cols

    # 2) Drop exact duplicate column names
    if df.columns.duplicated().any():
        dupes = list(df.columns[df.columns.duplicated()])
        print(f"_clean_columns: dropping duplicate columns: {dupes}")
        df = df.loc[:, ~df.columns.duplicated()]

    return df


# =====================================================================
# ✅ JSON-safe conversion helper (fixes NaN/Inf FastAPI 500)
# =====================================================================

def to_json_safe(obj: Any) -> Any:
    """
    Convert pandas/numpy objects to JSON-safe Python types:
      - NaN/Inf -> None
      - numpy scalars -> python scalars
      - Timestamps -> isoformat strings
      - DataFrame -> list[dict]
    """
    # DataFrame -> list[dict]
    if isinstance(obj, pd.DataFrame):
        df = obj.copy()
        df = df.replace([np.inf, -np.inf], np.nan)
        df = df.where(pd.notnull(df), None)

        # Convert datetime columns to strings to avoid non-serializable types
        for c in df.columns:
            try:
                if pd.api.types.is_datetime64_any_dtype(df[c]):
                    df[c] = df[c].dt.strftime("%Y-%m-%dT%H:%M:%S")
            except Exception:
                # If something weird happens, leave as-is after NaN cleanup
                pass

        return df.to_dict(orient="records")

    # Series -> list
    if isinstance(obj, pd.Series):
        s = obj.replace([np.inf, -np.inf], np.nan)
        s = s.where(pd.notnull(s), None)
        return s.tolist()

    # numpy scalar -> python scalar (and clean NaN/Inf)
    if isinstance(obj, (np.integer, np.floating, np.bool_)):
        v = obj.item()
        if isinstance(v, float) and (np.isnan(v) or np.isinf(v)):
            return None
        return v

    # float NaN/Inf
    if isinstance(obj, float):
        if np.isnan(obj) or np.isinf(obj):
            return None
        return obj

    # pandas Timestamp
    if isinstance(obj, pd.Timestamp):
        if pd.isna(obj):
            return None
        return obj.isoformat()

    # dict/list recurse
    if isinstance(obj, dict):
        return {k: to_json_safe(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [to_json_safe(v) for v in obj]

    return obj


# ---------------------------------------------------------------------
# Main runtime environment builder
# ---------------------------------------------------------------------

def runtime_normalize_mdf_env(env: dict[str, Any]) -> None:
    """
    Build a working 'mdf' DataFrame and normalize date-like columns.
    Also builds a per-customer 'customer_snapshot' DataFrame.

    NEW LOGIC:
    - No auto-merge across all tables.
    - Choose a single fact-ish table as mdf:
        * Prefer a table whose name contains 'invoice'
        * Otherwise use the first table in env['tables']
    - Normalize dates in each table and in mdf.
    - Clean columns (flatten MultiIndex + drop duplicates) everywhere.

    CHANNEL ENRICHMENT (NEW):
    - Adds visit-level service_channel (ROUTE vs RETAIL)
    - Builds a per-customer channel map (ROUTE_ONLY / RETAIL_ONLY / BOTH)
    - Merges channel fields into customer_snapshot
    """
    # 👇 Import inside runtime to avoid circular imports
    from helpers.analytics_helpers import (
        _build_customer_snapshot,
        add_service_channel_visits,
        build_customer_channel_map_from_visits,
        merge_customer_channel_into_snapshot,
        _expand_es_nested_columns,
    )

    pd_ = env["pd"]
    tables_ = env.get("tables", {}) or {}

    # 1) Normalize dates in each individual table and clean columns
    norm_tables: dict[str, pd.DataFrame] = {}
    for name, df in tables_.items():
        if isinstance(df, pd_.DataFrame):
            df2 = _clean_columns(df)         # flatten + dedupe columns
            _normalize_dates_inplace(df2)    # normalize dates
            df2 = _expand_es_nested_columns(df2)  # ✅ add customer_name, location_name, etc.
            norm_tables[name] = df2
        else:
            norm_tables[name] = df

    tables_ = norm_tables
    env["tables"] = tables_

    # 2) Choose a fact table for mdf (no more global merging)
    mdf = pd_.DataFrame()
    if tables_:
        # Try to find an "invoice" table by name
        fact_name = None
        for name in tables_.keys():
            if "invoice" in str(name).lower():
                fact_name = name
                break

        # Fallback: just use the first table
        if fact_name is None:
            fact_name = list(tables_.keys())[0]

        base_df = tables_.get(fact_name)
        if isinstance(base_df, pd_.DataFrame):
            base_df = _clean_columns(base_df)  # defensive
            mdf = base_df.copy()
        else:
            mdf = pd_.DataFrame()

    # 3) Final safety: normalize dates in mdf
    if not mdf.empty:
        _normalize_dates_inplace(mdf)

    # 4) Build a simple full-name / name column if possible (for customers)
    if not mdf.empty:
        first_col = resolve_column(mdf, "first_name", alias_family="customer_name")
        last_col = resolve_column(mdf, "last_name", alias_family="customer_name")

        if first_col or last_col:
            if first_col and first_col in mdf.columns:
                first = mdf[first_col].astype(str).fillna("")
            else:
                first = pd_.Series([""] * len(mdf), index=mdf.index)

            if last_col and last_col in mdf.columns:
                last = mdf[last_col].astype(str).fillna("")
            else:
                last = pd_.Series([""] * len(mdf), index=mdf.index)

            full = (first.str.strip() + " " + last.str.strip()).str.strip()
            mdf["full_name"] = full
            mdf["name"] = full
            mdf["customer_name"] = full

    # 4.5) CHANNEL ENRICHMENT ON VISITS (mdf level)
    # - Adds: mdf["service_channel"] = ROUTE / RETAIL
    mdf_ch = mdf
    channel_map = pd_.DataFrame()
    if isinstance(mdf, pd_.DataFrame) and not mdf.empty:
        try:
            mdf_ch = add_service_channel_visits(mdf)
            channel_map = build_customer_channel_map_from_visits(mdf_ch)
        except Exception as e:
            print(f"runtime_normalize_mdf_env: channel enrichment failed: {e}")
            mdf_ch = mdf
            channel_map = pd_.DataFrame()

    # Persist mdf (use enriched one if available)
    env["mdf"] = mdf_ch
    # Optional but useful for debugging / direct queries
    env["channel_map"] = channel_map

    # 5) Build customer_snapshot from mdf only + merge channel info
    try:
        customer_snapshot = _build_customer_snapshot(mdf_ch, resolve_column)

        # Merge per-customer channel info (ROUTE_ONLY / RETAIL_ONLY / BOTH)
        try:
            customer_snapshot = merge_customer_channel_into_snapshot(customer_snapshot, channel_map)
        except Exception as e2:
            print(f"runtime_normalize_mdf_env: failed to merge channel info into snapshot: {e2}")

        env["customer_snapshot"] = customer_snapshot

    except Exception as e:
        print(f"runtime_normalize_mdf_env: failed to build customer_snapshot: {e}")
        env["customer_snapshot"] = pd_.DataFrame()


# =====================================================================
# ✅ Execute LLM-generated analytics code
# =====================================================================

def run_generated_code(code: str, tables: Dict[str, pd.DataFrame]) -> Tuple[pd.DataFrame, str]:
    """
    Execute LLM-generated analytics code.

    The generated code must assign:
        result_df = <pd.DataFrame>
        insight   = <str>

    We also call runtime_normalize_mdf_env(env) first so that:
      - env["mdf"] exists
      - env["customer_snapshot"] exists
    """
    if not code or not isinstance(code, str):
        return pd.DataFrame(), "No code to execute."

    # 👇 Import helpers here so we can expose them to the exec env
    from helpers.analytics_helpers import (
        compute_customer_ltv,
        compute_visit_frequency,
        add_service_channel_visits,
        build_customer_channel_map_from_visits,
        merge_customer_channel_into_snapshot,
    )

    env: Dict[str, Any] = {
        "pd": pd,
        "tables": tables or {},
        "mdf": pd.DataFrame(),
        "customer_snapshot": pd.DataFrame(),
        "resolve_column": resolve_column,
        "runtime_normalize_mdf_env": runtime_normalize_mdf_env,

        # ✅ Expose helper functions so generated code can call them
        "compute_customer_ltv": compute_customer_ltv,
        "compute_visit_frequency": compute_visit_frequency,
        "add_service_channel_visits": add_service_channel_visits,
        "build_customer_channel_map_from_visits": build_customer_channel_map_from_visits,
        "merge_customer_channel_into_snapshot": merge_customer_channel_into_snapshot,

        # ✅ Expose JSON sanitizer for your API layer
        "to_json_safe": to_json_safe,
    }

    # Prepare mdf + customer_snapshot
    try:
        runtime_normalize_mdf_env(env)
    except Exception as e:
        return pd.DataFrame(), f"Failed to prepare runtime env: {type(e).__name__}: {e}"

    local_vars: Dict[str, Any] = {}
    try:
        exec(code, env, local_vars)
    except Exception as e:
        return pd.DataFrame(), f"Error executing generated code: {type(e).__name__}: {e}"

    result_df = local_vars.get("result_df")
    insight = local_vars.get("insight", "")

    if not isinstance(result_df, pd.DataFrame):
        result_df = pd.DataFrame()

    return result_df, str(insight or "")
