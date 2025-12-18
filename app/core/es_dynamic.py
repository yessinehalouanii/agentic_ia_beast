# app/core/es_dynamic.py

from typing import Optional
from elasticsearch import Elasticsearch


def make_es_client(
    base_url: str,
    username: Optional[str] = None,
    password: Optional[str] = None,
) -> Elasticsearch:
    """
    Create a dynamic Elasticsearch client from user input.

    ✅ Fast-fail (no hanging)
    ✅ Safe for UI-triggered requests
    ✅ Works with or without auth
    """

    if not base_url:
        raise ValueError("Elasticsearch base_url is required")

    # Normalize URL
    base_url = base_url.strip().rstrip("/")

    common_kwargs = dict(
        request_timeout=5,     # ⏱️ FAIL FAST
        max_retries=1,
        retry_on_timeout=True,
        verify_certs=False,    # 🔒 set True when SSL is valid
    )

    if username and password:
        return Elasticsearch(
            base_url,
            basic_auth=(username, password),
            **common_kwargs,
        )

    return Elasticsearch(base_url, **common_kwargs)
