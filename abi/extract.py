# abi/extract.py
from __future__ import annotations

from pathlib import Path
from typing import Optional

from pypdf import PdfReader
import docx


def extract_text(path: str) -> str:
    """
    Extract text from PDF / DOCX / TXT / MD.
    Best-effort extraction; returns "" if empty.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(str(p))

    ext = p.suffix.lower()

    if ext in (".txt", ".md"):
        return p.read_text(encoding="utf-8", errors="ignore").strip()

    if ext == ".pdf":
        reader = PdfReader(str(p))
        parts = []
        for page in reader.pages:
            parts.append(page.extract_text() or "")
        return "\n".join(parts).strip()

    if ext == ".docx":
        d = docx.Document(str(p))
        return "\n".join([para.text for para in d.paragraphs if para.text]).strip()

    raise ValueError(f"Unsupported extension: {ext}")
