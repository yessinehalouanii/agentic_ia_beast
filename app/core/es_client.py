# app/core/es_client.py
from elasticsearch import Elasticsearch
from .config import settings


def get_es_client() -> Elasticsearch:
    """
    Creates an ES client using credentials provided in .env.

    Works with:
    - API key (es_api_key)
    - Bearer token (es_bearer_token)
    - Basic auth (es_username + es_password)

    ⚠️ For production, this uses verify_certs=True so your ES URL
    MUST have a valid TLS certificate (public CA or proper internal CA).
    """
    headers = {}

    # Priority of auth: API KEY > Bearer > Basic
    if settings.es_api_key:
        headers["Authorization"] = f"ApiKey {settings.es_api_key}"
    elif settings.es_bearer_token:
        headers["Authorization"] = f"Bearer {settings.es_bearer_token}"

    common_kwargs = dict(
        headers=headers,
        verify_certs=True,      # ✅ DO NOT disable in production
        request_timeout=10,     # reasonable timeout
        max_retries=2,
        retry_on_timeout=True,
    )

    if settings.es_username and settings.es_password:
        # Basic Authentication
        es = Elasticsearch(
            settings.es_url,
            basic_auth=(settings.es_username, settings.es_password),
            **common_kwargs,
        )
    else:
        es = Elasticsearch(
            settings.es_url,
            **common_kwargs,
        )

    return es


# Global instance, created once at startup
es = get_es_client()
