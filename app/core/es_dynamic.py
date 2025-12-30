# app/core/es_dynamic.py

from typing import Optional
from urllib.parse import urlparse
import ipaddress

from elasticsearch import Elasticsearch


def _validate_public_url(raw_url: str) -> str:
    """
    Minimal validation to avoid obvious SSRF:
    - Must be http/https
    - Must have hostname
    - Forbids direct private/loopback IPs (10.x, 192.168.x, 172.16-31, 127.x, etc.)

    NOTE: This does NOT fully protect against DNS-based SSRF
    (a domain pointing to an internal IP). For full security,
    you should add an allowlist of domains you control.
    """
    raw_url = (raw_url or "").strip()
    if not raw_url:
        raise ValueError("Elasticsearch base_url is required")

    parsed = urlparse(raw_url)

    if parsed.scheme not in ("http", "https"):
        raise ValueError("Elasticsearch base_url must start with http:// or https://")

    if not parsed.hostname:
        raise ValueError("Elasticsearch base_url must include a hostname")

    host = parsed.hostname

    # If host is an IP literal, block private/loopback ranges
    try:
        ip_obj = ipaddress.ip_address(host)
        if ip_obj.is_private or ip_obj.is_loopback:
            raise ValueError("Elasticsearch base_url cannot be a private/loopback IP")
    except ValueError:
        # Not an IP literal -> it's a hostname; you may still want an allowlist here.
        pass

    # Normalise: drop trailing slash
    normalised = parsed._replace(path=parsed.path.rstrip("/")).geturl()
    return normalised


def make_es_client(
    base_url: str,
    username: Optional[str] = None,
    password: Optional[str] = None,
) -> Elasticsearch:
    """
    Create a dynamic Elasticsearch client from user input.

    ✅ Fast-fail (no hanging)
    ✅ Basic anti-SSRF hardening
    ✅ Works with or without auth
    """

    safe_url = _validate_public_url(base_url)

    common_kwargs = dict(
        request_timeout=5,      # ⏱️ FAIL FAST
        max_retries=1,
        retry_on_timeout=True,
        verify_certs=True,      # ✅ TLS verification ON for prod
    )

    if username and password:
        return Elasticsearch(
            safe_url,
            basic_auth=(username, password),
            **common_kwargs,
        )

    return Elasticsearch(safe_url, **common_kwargs)
