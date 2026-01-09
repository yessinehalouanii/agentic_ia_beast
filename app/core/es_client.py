# app/core/es_client.py
from urllib.parse import urlparse

from elasticsearch import Elasticsearch
from .config import settings


def get_es_client() -> Elasticsearch:
    """
    LOCAL TESTING CLIENT
    - Works with http://... and https://...
    - If https://... -> TLS verification is disabled (local only)
    - If http://...  -> NO TLS options are passed (avoids ValueError)
    """

    headers = {}

    # Priority of auth: API KEY > Bearer > Basic
    if getattr(settings, "es_api_key", None):
        headers["Authorization"] = f"ApiKey {settings.es_api_key}"
    elif getattr(settings, "es_bearer_token", None):
        headers["Authorization"] = f"Bearer {settings.es_bearer_token}"

    es_url = (settings.es_url or "").strip()
    if not es_url:
        raise ValueError("ES_URL is missing. Set it in your .env")

    scheme = urlparse(es_url).scheme.lower()

    # Common settings (safe for both http and https)
    common_kwargs = dict(
        headers=headers,
        request_timeout=30,
        max_retries=3,
        retry_on_timeout=True,
    )

    # ✅ ONLY apply TLS options when using https://
    if scheme == "https":
        common_kwargs.update(
            verify_certs=False,         # LOCAL DEV
            ssl_assert_hostname=False,  # LOCAL DEV
            ssl_show_warn=False,        # optional
        )

    # Basic auth if provided
    if getattr(settings, "es_username", None) and getattr(settings, "es_password", None):
        return Elasticsearch(
            es_url,
            basic_auth=(settings.es_username, settings.es_password),
            **common_kwargs,
        )

    return Elasticsearch(es_url, **common_kwargs)


# Global instance (created at startup)
es = get_es_client()
