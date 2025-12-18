# app/core/config.py
from typing import Optional
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # Existing fields
    openai_api_key: str
    default_model: str = "gpt-4o-mini"

    # Elasticsearch settings
    es_url: str
    es_username: Optional[str] = None
    es_password: Optional[str] = None
    es_api_key: Optional[str] = None
    es_bearer_token: Optional[str] = None

    # 🔹 Pinecone (optional, boss will configure later)
    pinecone_api_key: Optional[str] = None
    pinecone_index_name: Optional[str] = None
    pinecone_cloud: Optional[str] = None
    pinecone_region: Optional[str] = None
    pinecone_namespace_default: str = "default"

    # Pydantic config
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",  # ✅ CRITICAL: ignore unknown env vars
    )


settings = Settings()
