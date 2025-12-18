from __future__ import annotations

import os
from typing import List, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from abi.docs_rules import get_business_rules
from abi.llm import llm_codegen
from abi.runtime import run_generated_code, to_json_safe  # ✅ ADDED
from app.core.table_store import TABLE_STORE

router = APIRouter(prefix="/docs", tags=["Docs Analytics"])


# -------------------------------------------------------------------
# Request model
# -------------------------------------------------------------------

class DocsAnalyticsRequest(BaseModel):
    workspace_id: str
    question: str

    # 👇 STRICT MODES ONLY
    # "predefined" = ignore documents
    # "documents"  = require documents
    mode: str = "predefined"

    # ✅ NEW: optionally restrict business rules to specific uploaded documents
    doc_ids: Optional[List[str]] = None

    model: str = "gpt-4o-mini"
    api_key: str | None = None


# -------------------------------------------------------------------
# Analytics endpoint
# -------------------------------------------------------------------

@router.post("/ask-analytics")
def ask_docs_analytics(req: DocsAnalyticsRequest):
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
        "rows": safe_rows,  # ✅ CHANGED
        "rules_used": business_rules or "",
        "code": code,  # ⚠️ remove in production
    }
