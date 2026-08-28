"""Storage helpers — persist and retrieve Analysis records."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Optional

from sqlalchemy.orm import Session

from app.models import Analysis
from app.schemas import AnalysisResponse, ImageStats, Issue


def _build_url(base: str, id: int) -> str:
    return f"/api/v1/analyses/{id}/heatmap"


def save_analysis(db: Session, filename: str, pipeline_result: dict) -> Analysis:
    """Persist a pipeline result to the database and return the ORM object."""
    record = Analysis(
        filename=filename,
        upload_time=datetime.now(timezone.utc),
        quality_score=pipeline_result["quality_score"],
        quality_label=pipeline_result["quality_label"],
        issues_json=json.dumps(pipeline_result["issues"]),
        image_stats_json=json.dumps(pipeline_result["image_stats"]),
        model_version=pipeline_result["model_version"],
        thumbnail_path=pipeline_result.get("thumbnail_path"),
        heatmap_path=pipeline_result.get("heatmap_path"),
        original_path=pipeline_result.get("original_path"),
    )
    db.add(record)
    db.commit()
    db.refresh(record)
    return record


def orm_to_response(record: Analysis) -> AnalysisResponse:
    """Convert an ORM Analysis record to the API response schema."""
    issues_raw = json.loads(record.issues_json or "[]")
    stats_raw = json.loads(record.image_stats_json or "{}")

    issues = [Issue(**i) for i in issues_raw]
    stats = ImageStats(**stats_raw)

    heatmap_url = _build_url("/api/v1", record.id) if record.heatmap_path else None
    thumbnail_url = f"/api/v1/analyses/{record.id}/thumbnail" if record.thumbnail_path else None

    return AnalysisResponse(
        id=record.id,
        filename=record.filename,
        quality_score=record.quality_score,
        quality_label=record.quality_label,
        issues=issues,
        image_stats=stats,
        model_version=record.model_version,
        heatmap_url=heatmap_url,
        thumbnail_url=thumbnail_url,
        created_at=record.upload_time,
    )
