from __future__ import annotations

import os
import uuid
import tempfile
from typing import Optional

from fastapi import APIRouter, UploadFile, File, Form, HTTPException

from abi.extract import extract_text
from abi.chunking import chunk_text
from abi.embeddings import embed_texts
from abi.pinecone_store import upsert_chunks

router = APIRouter(prefix="/documents", tags=["Documents"])


@router.post("/upload")
async def upload_document(
    file: UploadFile = File(...),
    workspace_id: str = Form(...),
    doc_id: Optional[str] = Form(None),
):
    # ------------------------------------------------------------
    # 1) Basic validation
    # ------------------------------------------------------------
    if not workspace_id.strip():
        raise HTTPException(status_code=400, detail="workspace_id is required")

    filename = file.filename or "document"
    ext = os.path.splitext(filename.lower())[1]

    if ext not in (".pdf", ".docx", ".txt", ".md"):
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: {ext}",
        )

    tmp_path = None

    try:
        # ------------------------------------------------------------
        # 2) Persist upload to temp file (READ FILE ONCE)
        # ------------------------------------------------------------
        with tempfile.NamedTemporaryFile(delete=False, suffix=ext or ".bin") as tmp:
            tmp_path = tmp.name
            content = await file.read()
            tmp.write(content)

        # ------------------------------------------------------------
        # 3) Extract text
        # ------------------------------------------------------------
        text = extract_text(tmp_path)
        if not text.strip():
            raise HTTPException(
                status_code=400,
                detail="No text extracted from document",
            )

        # ✅ DEBUG PROOF
        print("EXTRACT LEN:", len(text))
        print("HAS BANANA:", "BANANA-XYZ-123" in text)

        # ------------------------------------------------------------
        # 4) Chunk text WITH headings embedded
        # ------------------------------------------------------------
        pairs = chunk_text(text)  # [(heading, chunk)]

        chunks: list[str] = []
        for heading, chunk in pairs:
            chunk = (chunk or "").strip()
            if not chunk:
                continue

            # ✅ SIMPLE + EFFECTIVE:
            # put heading directly into the chunk text
            if heading:
                chunks.append(f"{heading}\n{chunk}")
            else:
                chunks.append(chunk)

        if not chunks:
            raise HTTPException(
                status_code=400,
                detail="No chunks produced from document",
            )

        # ------------------------------------------------------------
        # 5) Embed chunks
        # ------------------------------------------------------------
        vectors = embed_texts(chunks)

        print("CHUNKS:", len(chunks))
        print("EMBED_DIM:", len(vectors[0]) if vectors else None)

        # ------------------------------------------------------------
        # 6) Upsert to Pinecone (namespace = workspace_id)
        # ------------------------------------------------------------
        doc_id_final = doc_id or uuid.uuid4().hex

        meta = {
            "filename": filename,
        }

        n = upsert_chunks(
            workspace_id=workspace_id,
            doc_id=doc_id_final,
            chunks=chunks,
            embeddings=vectors,
            metadata=meta,
        )

        return {
            "doc_id": doc_id_final,
            "filename": filename,
            "chunks_indexed": n,
        }

    finally:
        # ------------------------------------------------------------
        # 7) Cleanup temp file
        # ------------------------------------------------------------
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except Exception:
                pass
