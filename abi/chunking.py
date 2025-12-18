# abi/chunking.py
from __future__ import annotations

import re
from typing import List, Tuple


_HEADING_RE = re.compile(
    r"""
    ^(
        \#.+                         |   # Markdown heading
        [A-Z][A-Z0-9 \-/]{6,}$        |   ALL CAPS-ish headings
        .{1,80}:$                         # short lines ending with colon
    )$
    """,
    re.VERBOSE,
)


def _is_heading(line: str) -> bool:
    line = (line or "").strip()
    if not line:
        return False
    if _HEADING_RE.match(line):
        return True
    return False


def chunk_text(
    text: str,
    chunk_size: int = 1100,
    overlap: int = 160,
) -> List[Tuple[str, str]]:
    """
    Returns list of (heading, chunk_text).
    heading may be "" if unknown.

    chunk_size/overlap are chars (simple and robust).
    """
    t = (text or "").strip()
    if not t:
        return []

    lines = [ln.rstrip() for ln in t.splitlines()]
    sections: List[Tuple[str, str]] = []

    cur_heading = ""
    buf: List[str] = []

    for ln in lines:
        if _is_heading(ln):
            # flush previous section
            if buf:
                sections.append((cur_heading, "\n".join(buf).strip()))
                buf = []
            cur_heading = ln.strip()
        else:
            buf.append(ln)

    if buf:
        sections.append((cur_heading, "\n".join(buf).strip()))

    # Now chunk each section into chunk_size with overlap
    out: List[Tuple[str, str]] = []
    for heading, body in sections:
        body = body.strip()
        if not body:
            continue

        start = 0
        n = len(body)
        while start < n:
            end = min(start + chunk_size, n)
            chunk = body[start:end].strip()
            if chunk:
                out.append((heading, chunk))
            if end >= n:
                break
            start = max(0, end - overlap)

    return out
