# abi/docs_qa.py
from __future__ import annotations

from typing import Any, Dict, Optional, List, Tuple

from abi.embeddings import embed_texts
from abi.pinecone_store import query as pc_query
from abi.chat_answer import answer as grounded_answer
from abi.context_profile import build_or_update_profile


def answer_from_docs(
    workspace_id: str,
    question: str,
    doc_ids: Optional[List[str]] = None,
    existing_profile: Optional[Dict[str, Any]] = None,
) -> Tuple[str, List[dict]]:
    """
    Returns:
      (answer_text, matches)

    - Embeds the question
    - Queries Pinecone (namespace=workspace_id)
    - Builds/updates business profile from retrieved excerpts
    - Produces grounded answer
    """

    # -----------------------------
    # ✅ HARD VALIDATION (NEW)
    # -----------------------------
    if not workspace_id or not str(workspace_id).strip():
        raise ValueError("workspace_id is required")

    # Make sure question is a clean string (prevents OpenAI '$.input' invalid)
    question_str = "" if question is None else str(question).strip()
    if not question_str:
        raise ValueError("question is required")

    # -----------------------------
    # ✅ EMBEDDING (SAFE) (NEW)
    # -----------------------------
    try:
        qvecs = embed_texts([question_str])
        if not qvecs or not qvecs[0]:
            raise RuntimeError("Embedding returned empty vector.")
        qvec = qvecs[0]
    except Exception as e:
        # Return a clean error message to the API layer (or re-raise)
        raise RuntimeError(f"Failed to embed question: {type(e).__name__}: {e}")

    # -----------------------------
    # Pinecone filter
    # -----------------------------
    pinecone_filter = None
    if doc_ids:
        # keep only truthy doc ids
        clean_doc_ids = [d for d in doc_ids if d]
        if clean_doc_ids:
            pinecone_filter = {"doc_id": {"$in": clean_doc_ids}}

    # -----------------------------
    # ✅ QUERY PINECONE (SAFE) (NEW)
    # -----------------------------
    try:
        matches = pc_query(
            workspace_id=workspace_id,
            query_embedding=qvec,
            top_k=6,
            filter=pinecone_filter,
        )
        if matches is None:
            matches = []
    except Exception as e:
        raise RuntimeError(f"Failed to query Pinecone: {type(e).__name__}: {e}")

    # -----------------------------
    # Build excerpts for profile
    # -----------------------------
    excerpts = "\n\n".join(
        [((m.get("metadata") or {}).get("text", "") or "") for m in matches[:6]]
    ).strip()

    # -----------------------------
    # Profile extraction (safe)
    # -----------------------------
    try:
        profile = build_or_update_profile(
            excerpts=excerpts,
            existing_profile=existing_profile,
        )
    except Exception:
        # fallback: still answer, but without enriched profile
        profile = existing_profile or {}

    # -----------------------------
    # Final grounded answer
    # -----------------------------
    ans = grounded_answer(
        question=question_str,
        profile=profile,
        chunks=matches,
    )
    return ans, matches
