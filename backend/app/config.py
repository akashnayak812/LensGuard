"""Application configuration loaded from environment variables."""
from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # Database
    db_path: str = "data/lensguard.db"

    # Model
    model_path: str = "ml/models"
    model_version: str = "v1"

    # Upload limits
    max_upload_mb: int = 20

    # CORS
    cors_origins: str = "http://localhost,http://localhost:80,http://localhost:3000"

    # Storage directories
    upload_dir: str = "data/uploads"
    heatmap_dir: str = "data/heatmaps"

    # Logging
    log_level: str = "INFO"

    # Server
    host: str = "0.0.0.0"
    port: int = 8000

    @property
    def cors_origins_list(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",")]

    @property
    def max_upload_bytes(self) -> int:
        return self.max_upload_mb * 1024 * 1024


settings = Settings()
