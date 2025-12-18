# app/core/table_store.py
from __future__ import annotations

from dataclasses import dataclass, field
from threading import RLock
from typing import Dict, Optional

import pandas as pd


@dataclass
class TableStore:
    # workspace_id -> { table_name -> df }
    _by_workspace: Dict[str, Dict[str, pd.DataFrame]] = field(default_factory=dict)
    _lock: RLock = field(default_factory=RLock)

    def set_tables(self, workspace_id: str, tables: Dict[str, pd.DataFrame]) -> None:
        ws = (workspace_id or "default").strip() or "default"
        with self._lock:
            self._by_workspace[ws] = dict(tables or {})

    def get_tables(self, workspace_id: str) -> Dict[str, pd.DataFrame]:
        ws = (workspace_id or "default").strip() or "default"
        with self._lock:
            return dict(self._by_workspace.get(ws, {}) or {})

    def get(self, workspace_id: str, name: str) -> Optional[pd.DataFrame]:
        ws = (workspace_id or "default").strip() or "default"
        with self._lock:
            return (self._by_workspace.get(ws, {}) or {}).get(name)

    def clear(self, workspace_id: Optional[str] = None) -> None:
        with self._lock:
            if workspace_id:
                ws = (workspace_id or "default").strip() or "default"
                self._by_workspace.pop(ws, None)
            else:
                self._by_workspace.clear()


TABLE_STORE = TableStore()
