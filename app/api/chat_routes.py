from __future__ import annotations

from typing import Optional, List

from fastapi import APIRouter
from pydantic import BaseModel

from abi.docs_qa import answer_from_docs

router = APIRouter(tags=["Chat"])


class ChatRequest(BaseModel):
    workspace_id: str
    message: str
    doc_ids: Optional[List[str]] = None
    model_name: Optional[str] = None
    api_key: Optional[str] = None


@router.post("/chat")
def chat(req: ChatRequest):
    answer, used = answer_from_docs(
        workspace_id=req.workspace_id,
        question=req.message,
        doc_ids=req.doc_ids,
        existing_profile=None,
    )
    return {"answer": answer, "used_chunks": used}
