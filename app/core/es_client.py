# app/core/es_client.py
from elasticsearch import Elasticsearch
from .config import settings

def get_es_client() -> Elasticsearch:
    """
    Creates an ES client using credentials provided in .env.
    Works with Basic Auth, Bearer Token, or API Key.
    """
    headers = {}

    # Priority of auth: API KEY > Bearer > Basic
    if settings.es_api_key:
        headers["Authorization"] = f"ApiKey {settings.es_api_key}"
    elif settings.es_bearer_token:
        headers["Authorization"] = f"Bearer {settings.es_bearer_token}"

    if settings.es_username and settings.es_password:
        # Basic Authentication
        es = Elasticsearch(
            settings.es_url,
            basic_auth=(settings.es_username, settings.es_password),
            headers=headers,
            verify_certs=False,   # set True once SSL is valid
        )
    else:
        es = Elasticsearch(
            settings.es_url,
            headers=headers,
            verify_certs=False,
        )

    return es


es = get_es_client()  # global instance
