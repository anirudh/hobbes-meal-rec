"""
Centralised settings loader.

Works with:
• Pydantic-v1  → BaseSettings from pydantic
• Pydantic-v2  → BaseSettings from pydantic-settings
"""

from __future__ import annotations
from functools import lru_cache

# ------------------------------------------------------------------ #
#  Import BaseSettings depending on pydantic version
# ------------------------------------------------------------------ #
try:  # Pydantic v1
    from pydantic import BaseSettings, Field
except ImportError:  # Pydantic v2
    from pydantic_settings import BaseSettings  # type: ignore
    from pydantic import Field


# ------------------------------------------------------------------ #
#  Master settings model
# ------------------------------------------------------------------ #
class _Settings(BaseSettings):
    # ─── runtime / DB / API keys (unchanged) ─────────────────────────
    env_name: str = Field("local", env="ENV_NAME")
    database_url: str | None = Field(None, env="DATABASE_URL")
    cloud_sql_instance: str | None = Field(None, env="CLOUD_SQL_CONNECTION_NAME")
    db_user: str | None = Field(None, env="DB_USER")
    db_pass: str | None = Field(None, env="DB_PASS")
    db_name: str | None = Field(None, env="DB_NAME")
    edamam_app_id: str = Field("dummy", env="EDAMAM_APP_ID")
    edamam_app_key: str = Field("dummy", env="EDAMAM_APP_KEY")
    jwt_secret: str = Field("changeme", env="JWT_SECRET")

    # ─── NEW: Gemini / PaLM etc. ────────────────────────────────────
    gemini_api_key: str | None = Field(None, env="GEMINI_API_KEY")

    # allow other teammates’ env-vars without crashing
    model_config = {"extra": "ignore", "env_file": ".env", "env_file_encoding": "utf-8"}


# ------------------------------------------------------------------ #
#  Cached singleton accessor
# ------------------------------------------------------------------ #
@lru_cache
def _cached() -> _Settings:  # pragma: no cover
    return _Settings()  # type: ignore[call-arg]


settings: _Settings = _cached()



