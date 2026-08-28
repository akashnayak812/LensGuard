"""Pydantic v2 schemas for request/response validation."""
from __future__ import annotations

from datetime import datetime
from typing import Any, Optional

from pydantic import BaseModel, ConfigDict, Field


class Issue(BaseModel):
    type: str
    severity: str
    confidence: float = Field(ge=0.0, le=1.0)


class ImageStats(BaseModel):
    laplacian_variance: float = 0.0
    mean_luminance: float = 0.0
    pct_near_black: float = 0.0
    pct_near_white: float = 0.0
    luminance_std: float = 0.0
    noise_estimate: float = 0.0
    contrast_rms: float = 0.0
    block_artifact_score: float = 0.0
    mean_saturation: float = 0.0
    std_saturation: float = 0.0
    pct_low_saturation: float = 0.0

    model_config = ConfigDict(extra="allow")


class AnalysisResponse(BaseModel):
    id: int
    filename: str
    quality_score: int = Field(ge=0, le=100)
    quality_label: str
    issues: list[Issue]
    image_stats: ImageStats
    model_version: str
    heatmap_url: Optional[str] = None
    thumbnail_url: Optional[str] = None
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)


class AnalysisListResponse(BaseModel):
    items: list[AnalysisResponse]
    total: int
    limit: int
    offset: int


class BatchAnalysisResponse(BaseModel):
    results: list[AnalysisResponse]
    failed: list[dict[str, Any]]


class ErrorDetail(BaseModel):
    code: str
    message: str


class ErrorResponse(BaseModel):
    error: ErrorDetail


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_version: str
    db_reachable: bool
    uptime_seconds: float


class MetricsResponse(BaseModel):
    total_requests: int
    total_analyses: int
    avg_inference_latency_ms: float
    error_count: int
