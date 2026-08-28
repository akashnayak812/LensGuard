"""SQLAlchemy ORM models."""
from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy import DateTime, Float, Integer, String, Text, func
from sqlalchemy.orm import Mapped, mapped_column

from app.database import Base


class Analysis(Base):
    """Persisted result of one image quality analysis."""

    __tablename__ = "analyses"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    filename: Mapped[str] = mapped_column(String(512), nullable=False)
    upload_time: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        nullable=False,
    )
    quality_score: Mapped[int] = mapped_column(Integer, nullable=False)
    quality_label: Mapped[str] = mapped_column(String(20), nullable=False, index=True)

    # Stored as JSON strings
    issues_json: Mapped[str] = mapped_column(Text, nullable=False, default="[]")
    image_stats_json: Mapped[str] = mapped_column(Text, nullable=False, default="{}")

    model_version: Mapped[str] = mapped_column(String(32), nullable=False)
    thumbnail_path: Mapped[str | None] = mapped_column(String(512), nullable=True)
    heatmap_path: Mapped[str | None] = mapped_column(String(512), nullable=True)
    original_path: Mapped[str | None] = mapped_column(String(512), nullable=True)
