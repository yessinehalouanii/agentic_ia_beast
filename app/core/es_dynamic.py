# app/core/es_dynamic.py
from typing import Optional
from urllib.parse import urlparse
import ipaddress

from elasticsearch import Elasticsearch


def _validate_public_url(raw_url: str) -> str:
    """
    Minimal validation (still okay for local):
    - Must be http/https
    - Must have hostname
    - Blocks private/loopback ONLY if the hostname is an IP literal (e.g. 127.0.0.1)
      but allows hostnames like "localhost".
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

    # Block private/loopback only if it's an IP literal (127.0.0.1, 10.x, 192.168.x...)
    try:
        ip_obj = ipaddress.ip_address(host)
        if ip_obj.is_private or ip_obj.is_loopback:
            raise ValueError("Elasticsearch base_url cannot be a private/loopback IP literal")
    except ValueError:
        # Not an IP literal (hostname like localhost) -> allow
        pass

    # normalize (remove trailing slash)
    normalised = parsed._replace(path=parsed.path.rstrip("/")).geturl()
    return normalised


def make_es_client(
    base_url: str,
    username: Optional[str] = None,
    password: Optional[str] = None,
) -> Elasticsearch:
    """
    LOCAL TESTING DYNAMIC CLIENT
    - Works with http://... and https://...
    - TLS options only applied for https:// (avoid ValueError)
    """
    safe_url = _validate_public_url(base_url)
    scheme = urlparse(safe_url).scheme.lower()

    common_kwargs = dict(
        request_timeout=30,
        max_retries=2,
        retry_on_timeout=True,
    )

    # ✅ ONLY apply TLS options when using https://
    if scheme == "https":
        common_kwargs.update(
            verify_certs=False,         # LOCAL DEV
            ssl_assert_hostname=False,  # LOCAL DEV
            ssl_show_warn=False,        # optional
        )

    if username and password:
        return Elasticsearch(
            safe_url,
            basic_auth=(username, password),
            **common_kwargs,
        )

    return Elasticsearch(safe_url, **common_kwargs)
