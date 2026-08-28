"""
/api/v1/analyses — history retrieval endpoints.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import FileResponse
from sqlalchemy import select, func
from sqlalchemy.orm import Session

from app.database import get_db
from app.models import Analysis
from app.schemas import AnalysisListResponse, AnalysisResponse
from app.services.storage import orm_to_response

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/analyses", tags=["analyses"])


@router.get("", response_model=AnalysisListResponse)
def list_analyses(
    limit: int = Query(default=20, ge=1, le=100, description="Max results to return"),
    offset: int = Query(default=0, ge=0, description="Pagination offset"),
    label: Optional[str] = Query(default=None, description="Filter by quality_label: ACCEPTABLE | DEGRADED | DEFECTIVE"),
    sort_by: str = Query(default="created_at", description="Sort field: created_at | quality_score"),
    order: str = Query(default="desc", description="Sort order: asc | desc"),
    db: Session = Depends(get_db),
):
    """Retrieve paginated history of past image analyses.

    Supports optional filtering by quality label and sorting by date or score.
    """
    stmt = select(Analysis)

    if label:
        label_upper = label.upper()
        if label_upper not in ("ACCEPTABLE", "DEGRADED", "DEFECTIVE"):
            raise HTTPException(status_code=400, detail={"error": {"code": "INVALID_LABEL", "message": "label must be one of: ACCEPTABLE, DEGRADED, DEFECTIVE"}})
        stmt = stmt.where(Analysis.quality_label == label_upper)

    # Count total (before pagination)
    count_stmt = select(func.count()).select_from(stmt.subquery())
    total = db.scalar(count_stmt) or 0

    # Sorting
    sort_col = Analysis.upload_time if sort_by == "created_at" else Analysis.quality_score
    stmt = stmt.order_by(sort_col.desc() if order == "desc" else sort_col.asc())

    # Pagination
    stmt = stmt.limit(limit).offset(offset)
    records = db.scalars(stmt).all()

    return AnalysisListResponse(
        items=[orm_to_response(r) for r in records],
        total=total,
        limit=limit,
        offset=offset,
    )


@router.get("/{analysis_id}", response_model=AnalysisResponse)
def get_analysis(analysis_id: int, db: Session = Depends(get_db)):
    """Retrieve a single analysis result by ID."""
    record = db.get(Analysis, analysis_id)
    if record is None:
        raise HTTPException(status_code=404, detail={"error": {"code": "NOT_FOUND", "message": f"Analysis {analysis_id} not found."}})
    return orm_to_response(record)


@router.get("/{analysis_id}/heatmap")
def get_heatmap(analysis_id: int, db: Session = Depends(get_db)):
    """Serve the Grad-CAM heatmap PNG for a given analysis."""
    record = db.get(Analysis, analysis_id)
    if record is None:
        raise HTTPException(status_code=404, detail={"error": {"code": "NOT_FOUND", "message": f"Analysis {analysis_id} not found."}})
    if not record.heatmap_path:
        raise HTTPException(status_code=404, detail={"error": {"code": "HEATMAP_UNAVAILABLE", "message": "Grad-CAM heatmap was not generated for this analysis."}})
    path = Path(record.heatmap_path)
    if not path.exists():
        raise HTTPException(status_code=404, detail={"error": {"code": "HEATMAP_FILE_MISSING", "message": "Heatmap file not found on disk."}})
    return FileResponse(str(path), media_type="image/png")


@router.get("/{analysis_id}/thumbnail")
def get_thumbnail(analysis_id: int, db: Session = Depends(get_db)):
    """Serve the thumbnail image for a given analysis."""
    record = db.get(Analysis, analysis_id)
    if record is None:
        raise HTTPException(status_code=404, detail={"error": {"code": "NOT_FOUND", "message": f"Analysis {analysis_id} not found."}})
    if not record.thumbnail_path:
        raise HTTPException(status_code=404, detail={"error": {"code": "THUMBNAIL_UNAVAILABLE", "message": "Thumbnail not available for this analysis."}})
    path = Path(record.thumbnail_path)
    if not path.exists():
        raise HTTPException(status_code=404, detail={"error": {"code": "THUMBNAIL_FILE_MISSING", "message": "Thumbnail file not found on disk."}})
    return FileResponse(str(path), media_type="image/jpeg")
