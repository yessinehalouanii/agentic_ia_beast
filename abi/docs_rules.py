from __future__ import annotations

from typing import List, Optional

from abi.pinecone_store import query
from abi.embeddings import embed_texts


def get_business_rules(
    workspace_id: str,
    question: str,
    doc_ids: Optional[List[str]] = None,
) -> str:
    # ---- basic safety ----
    if not workspace_id or not str(workspace_id).strip():
        raise ValueError("workspace_id is required")

    q = "" if question is None else str(question).strip()
    if not q:
        return ""

    # ---- embed question ----
    qvecs = embed_texts([q])
    qvec = qvecs[0] if qvecs else None
    if not qvec:
        return ""

    # ---- optional Pinecone filter by doc_ids ----
    pinecone_filter = None
    if doc_ids:
        clean = [str(d).strip() for d in doc_ids if str(d).strip()]
        if clean:
            pinecone_filter = {"doc_id": {"$in": clean}}

    matches = query(
        workspace_id=workspace_id,
        query_embedding=qvec,
        top_k=5,
        filter=pinecone_filter,
    )

    # ---- collect rule text (sorted + deduped) ----
    matches = sorted(matches or [], key=lambda m: float(m.get("score") or 0.0), reverse=True)

    rules: List[str] = []
    seen = set()
    for m in matches:
        text = ((m.get("metadata") or {}).get("text") or "").strip()
        if not text:
            continue
        if text in seen:
            continue
        seen.add(text)
        rules.append(text)

    return "\n".join(rules)
