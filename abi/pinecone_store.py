# abi/pinecone_store.py
from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

from pinecone import Pinecone


# Cached singletons (avoid reconnecting per request)
_PC: Optional[Pinecone] = None
_INDEX = None


def _get_index():
    global _PC, _INDEX

    if _INDEX is not None:
        return _INDEX

    api_key = os.getenv("PINECONE_API_KEY")
    index_name = os.getenv("PINECONE_INDEX_NAME")

    if not api_key:
        raise RuntimeError("Missing PINECONE_API_KEY in environment.")
    if not index_name:
        raise RuntimeError("Missing PINECONE_INDEX_NAME in environment.")

    _PC = Pinecone(api_key=api_key)
    _INDEX = _PC.Index(index_name)
    return _INDEX


def upsert_chunks(
    workspace_id: str,
    doc_id: str,
    chunks: List[str],
    embeddings: List[List[float]],
    metadata: Optional[Dict[str, Any]] = None,
) -> int:
    """
    Upsert chunk vectors into Pinecone under namespace=workspace_id.

    - workspace_id: tenant/workspace identifier (namespace)
    - doc_id: document identifier
    - chunks: list of chunk texts
    - embeddings: list of vectors aligned with chunks
    - metadata: extra metadata applied to every chunk (merged)
    """
    if not workspace_id:
        raise ValueError("workspace_id is required")
    if not doc_id:
        raise ValueError("doc_id is required")
    if len(chunks) != len(embeddings):
        raise ValueError("chunks and embeddings must be the same length")

    index = _get_index()

    base_meta = metadata.copy() if metadata else {}
    base_meta["workspace_id"] = workspace_id
    base_meta["doc_id"] = doc_id

    vectors = []
    for i, (text, vec) in enumerate(zip(chunks, embeddings)):
        chunk_id = f"{doc_id}::chunk::{i}"
        m = dict(base_meta)
        # Keep chunk text in metadata (handy), but you may truncate if large
        m["chunk_index"] = i
        m["text"] = text
        vectors.append({"id": chunk_id, "values": vec, "metadata": m})

    # upsert in batches to avoid payload limits
    BATCH = 100
    for start in range(0, len(vectors), BATCH):
        index.upsert(
            vectors=vectors[start : start + BATCH],
            namespace=workspace_id,
        )

    return len(vectors)


def query(
    workspace_id: str,
    query_embedding: List[float],
    top_k: int = 6,
    filter: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    """
    Query Pinecone for top_k matches in namespace=workspace_id.
    Returns list of matches with metadata.
    """
    if not workspace_id:
        raise ValueError("workspace_id is required")
    if not query_embedding:
        return []

    index = _get_index()

    res = index.query(
        namespace=workspace_id,
        vector=query_embedding,
        top_k=int(top_k),
        include_metadata=True,
        filter=filter,
    )

    matches = []
    for m in (res.get("matches") or []):
        matches.append(
            {
                "id": m.get("id"),
                "score": float(m.get("score") or 0.0),
                "metadata": m.get("metadata") or {},
            }
        )
    return matches
