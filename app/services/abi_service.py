# app/services/abi_service.py

from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional
import os
import json

import pandas as pd
from dotenv import load_dotenv

from abi.llm import llm_codegen
from abi.docs_rules import get_business_rules

# ✅ Use ONE runtime executor (single source of truth)
from abi.runtime import run_generated_code

load_dotenv()

CACHE_DIR = ".abi_cache"

# ============================================================
# 1) LOAD / SAVE TABLES  (CSV instead of parquet)
# ============================================================

def load_tables_from_disk() -> Dict[str, pd.DataFrame]:
    tables: Dict[str, pd.DataFrame] = {}
    manifest_path = Path(CACHE_DIR) / "manifest.json"
    if not manifest_path.exists():
        return tables

    try:
        with manifest_path.open("r", encoding="utf-8") as f:
            manifest = json.load(f)
    except Exception as e:
        print("load_tables_from_disk (API): failed to read manifest:", e)
        return tables

    for name, path_str in manifest.items():
        path = Path(path_str)
        try:
            if path.exists():
                df = pd.read_csv(path)

                # Defensive: drop duplicate column names
                if df.columns.duplicated().any():
                    dupes = list(df.columns[df.columns.duplicated()])
                    print(f"[ABI API] {name}: dropping dup columns: {dupes}")
                    df = df.loc[:, ~df.columns.duplicated()]

                tables[name] = df
        except Exception as e:
            print(f"load_tables_from_disk (API): failed for {name}: {e}")

    return tables


def save_tables_to_disk(tables: Dict[str, pd.DataFrame]) -> None:
    if not tables:
        return

    Path(CACHE_DIR).mkdir(exist_ok=True)
    manifest: Dict[str, str] = {}

    for name, df in tables.items():
        safe_name = name.replace("/", "_").replace("\\", "_")
        path = Path(CACHE_DIR) / f"{safe_name}.csv"
        try:
            df.to_csv(path, index=False)
            manifest[name] = str(path)
        except Exception as e:
            print(f"save_tables_to_disk (API): failed for {name}: {e}")

    manifest_path = Path(CACHE_DIR) / "manifest.json"
    try:
        with manifest_path.open("w", encoding="utf-8") as f:
            json.dump(manifest, f)
    except Exception as e:
        print("save_tables_to_disk (API): failed to write manifest:", e)


def list_tables_meta() -> List[Dict[str, Any]]:
    tables = load_tables_from_disk()
    return [
        {"name": name, "rows": len(df), "cols": df.shape[1]}
        for name, df in tables.items()
    ]


def get_table_csv(name: str) -> str:
    tables = load_tables_from_disk()
    if name not in tables:
        raise KeyError(f"Table '{name}' not found in cache.")
    return tables[name].to_csv(index=False)