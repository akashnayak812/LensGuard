"""
/api/v1/analyze — single and batch image upload endpoints.
"""
from __future__ import annotations

import logging
import traceback
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, UploadFile, File
from fastapi.responses import JSONResponse
from sqlalchemy.orm import Session

from app.config import settings
from app.database import get_db
from app.schemas import (
    AnalysisResponse,
    BatchAnalysisResponse,
    ErrorResponse,
    ErrorDetail,
)
from app.services import pipeline as pipeline_svc
from app.services.storage import orm_to_response, save_analysis

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/analyze", tags=["analyze"])

# Allowed MIME types (magic-byte check happens in validate_image)
ALLOWED_CONTENT_TYPES = {
    "image/jpeg",
    "image/png",
    "image/webp",
    "image/bmp",
    "image/tiff",
}

# Magic bytes for common image formats
MAGIC_SIGNATURES: list[tuple[bytes, str]] = [
    (b"\xff\xd8\xff", "JPEG"),
    (b"\x89PNG\r\n\x1a\n", "PNG"),
    (b"RIFF", "WEBP"),  # RIFF....WEBP
    (b"BM", "BMP"),
    (b"II*\x00", "TIFF"),
    (b"MM\x00*", "TIFF"),
    (b"GIF87a", "GIF"),
    (b"GIF89a", "GIF"),
]


def _check_magic(data: bytes) -> bool:
    """Return True if the file starts with a recognised image magic signature."""
    for sig, _ in MAGIC_SIGNATURES:
        if data[: len(sig)] == sig:
            return True
    # WEBP special case: RIFF....WEBP
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return True
    return False


async def validate_and_read(upload: UploadFile) -> bytes:
    """Validate file size + type, read bytes. Raises HTTPException on failure."""
    # Content-type header check (basic)
    content_type = upload.content_type or ""
    if content_type and content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(
            status_code=415,
            detail={"error": {"code": "UNSUPPORTED_MEDIA_TYPE", "message": f"File type '{content_type}' is not supported. Upload a JPEG, PNG, WebP, BMP, or TIFF image."}},
        )

    data = await upload.read()

    # Size check
    if len(data) > settings.max_upload_bytes:
        raise HTTPException(
            status_code=413,
            detail={"error": {"code": "FILE_TOO_LARGE", "message": f"File exceeds maximum allowed size of {settings.max_upload_mb} MB."}},
        )

    if len(data) == 0:
        raise HTTPException(
            status_code=400,
            detail={"error": {"code": "EMPTY_FILE", "message": "Uploaded file is empty."}},
        )

    # Magic byte check
    if not _check_magic(data):
        raise HTTPException(
            status_code=400,
            detail={"error": {"code": "INVALID_IMAGE", "message": "File does not appear to be a valid image (magic byte check failed)."}},
        )

    return data


def _error_response(code: str, message: str, status: int) -> JSONResponse:
    return JSONResponse(
        status_code=status,
        content={"error": {"code": code, "message": message}},
    )


@router.post("", response_model=AnalysisResponse, status_code=200)
async def analyze_single(
    file: UploadFile = File(..., description="Image file to analyse"),
    db: Session = Depends(get_db),
):
    """Analyse a single uploaded image and return quality assessment.

    Accepts JPEG, PNG, WebP, BMP, or TIFF images up to MAX_UPLOAD_MB in size.
    Runs the full CV/ML pipeline and persists the result.
    """
    if not pipeline_svc.is_model_loaded():
        return _error_response("MODEL_NOT_READY", "Model is not loaded yet. Please retry shortly.", 503)

    try:
        data = await validate_and_read(file)
    except HTTPException:
        raise
    except Exception as e:
        logger.warning(f"File validation error: {e}")
        return _error_response("VALIDATION_ERROR", str(e), 400)

    try:
        result = pipeline_svc.run_pipeline(
            file_bytes=data,
            filename=file.filename or "upload.jpg",
            upload_dir=settings.upload_dir,
            heatmap_dir=settings.heatmap_dir,
        )
    except Exception as e:
        logger.error(f"Pipeline error for {file.filename}: {e}\n{traceback.format_exc()}")
        return _error_response("PIPELINE_ERROR", f"Image analysis failed: {e}", 500)

    try:
        record = save_analysis(db, file.filename or "upload.jpg", result)
        return orm_to_response(record)
    except Exception as e:
        logger.error(f"DB persistence error: {e}\n{traceback.format_exc()}")
        return _error_response("DATABASE_ERROR", f"Analysis completed but could not be saved: {e}", 500)


@router.post("/batch", response_model=BatchAnalysisResponse, status_code=200)
async def analyze_batch(
    files: list[UploadFile] = File(..., description="Image files to analyse (multi-file)"),
    db: Session = Depends(get_db),
):
    """Analyse multiple uploaded images in one request.

    Returns per-image results. Images that fail validation or pipeline
    execution are reported in the ``failed`` list; others succeed normally.
    """
    if not pipeline_svc.is_model_loaded():
        return _error_response("MODEL_NOT_READY", "Model is not loaded yet. Please retry shortly.", 503)

    if len(files) > 20:
        return _error_response("TOO_MANY_FILES", "Maximum 20 files per batch request.", 400)

    results: list[AnalysisResponse] = []
    failed: list[dict[str, Any]] = []

    for upload in files:
        fname = upload.filename or "upload.jpg"
        try:
            data = await validate_and_read(upload)
            pipeline_result = pipeline_svc.run_pipeline(
                file_bytes=data,
                filename=fname,
                upload_dir=settings.upload_dir,
                heatmap_dir=settings.heatmap_dir,
            )
            record = save_analysis(db, fname, pipeline_result)
            results.append(orm_to_response(record))
        except HTTPException as e:
            failed.append({"filename": fname, "error": e.detail})
        except Exception as e:
            logger.error(f"Batch pipeline error for {fname}: {e}\n{traceback.format_exc()}")
            failed.append({"filename": fname, "error": {"code": "PIPELINE_ERROR", "message": "Analysis failed for this image."}})

    return BatchAnalysisResponse(results=results, failed=failed)
