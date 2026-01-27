from typing import Any, Dict

def _field_exists(mappings: Dict[str, Any], dotted: str) -> bool:
    """
    True only if 'dotted' exists in mappings (supports multi-fields like *.keyword).
    Expected mappings shape: {"properties": {...}}.
    """
    if not dotted:
        return False

    node: Any = mappings or {}
    parts = dotted.split(".")

    for part in parts:
        if not isinstance(node, dict):
            return False

        props = node.get("properties") or {}
        if part in props:
            node = props[part] or {}
            continue

        fields = node.get("fields") or {}
        if part in fields:
            node = fields[part] or {}
            continue

        return False

    return True
# -------------------------------------------------------------------
# ES-safe helpers
# -------------------------------------------------------------------

def _safe_es_search(client, *, index: str, body: Dict[str, Any]) -> Dict[str, Any]:
    """
    Safety defaults to avoid long-running queries killing the cluster.
    - timeout: ES-side
    - track_total_hits: off (we don't need exact hits)
    - request_timeout: client-side
    """
    body = dict(body or {})
    body.setdefault("timeout", "10s")
    body.setdefault("track_total_hits", False)
    return client.search(index=index, body=body, request_timeout=15)
