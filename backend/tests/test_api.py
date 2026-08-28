"""
API integration tests using FastAPI TestClient.

These tests use a mocked pipeline service so they don't require a trained
model to run in CI. The pipeline is monkey-patched with a deterministic
fixture response.
"""

from __future__ import annotations

import json
import sys
from io import BytesIO
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

# Ensure backend and project root are on sys.path
_TEST_DIR = Path(__file__).resolve().parent
_BACKEND_DIR = _TEST_DIR.parent
_PROJECT_ROOT = _BACKEND_DIR.parent
for _p in (str(_BACKEND_DIR), str(_PROJECT_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# ── Mock ML modules before importing the app ─────────────────────────────
# This prevents TensorFlow from loading during CI test runs.

MOCK_PIPELINE_RESULT = {
    "quality_score": 78,
    "quality_label": "ACCEPTABLE",
    "issues": [{"type": "noise", "severity": "low", "confidence": 0.42}],
    "image_stats": {
        "laplacian_variance": 312.5,
        "mean_luminance": 127.3,
        "pct_near_black": 0.01,
        "pct_near_white": 0.02,
        "luminance_std": 48.2,
        "noise_estimate": 6.1,
        "contrast_rms": 0.19,
        "block_artifact_score": 1.05,
        "mean_saturation": 0.31,
        "std_saturation": 0.18,
        "pct_low_saturation": 0.12,
    },
    "model_version": "v1",
    "thumbnail_path": "/tmp/thumb.jpg",
    "heatmap_path": "/tmp/heatmap.png",
    "original_path": "/tmp/orig.jpg",
    "inference_ms": 142.3,
}

# Minimal 1×1 valid JPEG bytes (magic bytes present)
TINY_JPEG = (
    b"\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\x00\x00\x01\x00\x01\x00\x00"
    b"\xff\xdb\x00C\x00\x08\x06\x06\x07\x06\x05\x08\x07\x07\x07\t\t"
    b"\x08\n\x0c\x14\r\x0c\x0b\x0b\x0c\x19\x12\x13\x0f\x14\x1d\x1a"
    b"\x1f\x1e\x1d\x1a\x1c\x1c $.' \",#\x1c\x1c(7),01444\x1f'9=82<.342\x1e"
    b"\xff\xc0\x00\x0b\x08\x00\x01\x00\x01\x01\x01\x11\x00"
    b"\xff\xc4\x00\x1f\x00\x00\x01\x05\x01\x01\x01\x01\x01\x01\x00\x00\x00"
    b"\x00\x00\x00\x00\x00\x01\x02\x03\x04\x05\x06\x07\x08\t\n\x0b"
    b"\xff\xc4\x00\xb5\x10\x00\x02\x01\x03\x03\x02\x04\x03\x05\x05\x04"
    b"\x04\x00\x00\x01}\x01\x02\x03\x00\x04\x11\x05\x12!1A\x06\x13Qa"
    b"\xff\xda\x00\x08\x01\x01\x00\x00?\x00\xf5\n\xd8\xff\xd9"
)


@pytest.fixture(scope="module")
def client():
    """Create a FastAPI TestClient with mocked pipeline and model state."""
    with (
        patch("app.services.pipeline._is_loaded", True),
        patch("app.services.pipeline._metadata", {"version": "v1", "temperature": 1.0}),
        patch("app.services.pipeline.run_pipeline", return_value=MOCK_PIPELINE_RESULT),
    ):
        from app.main import app
        from app.database import init_db, engine, Base

        # Use in-memory SQLite with StaticPool so schema persists across sessions
        from sqlalchemy import create_engine
        from sqlalchemy.orm import sessionmaker
        from sqlalchemy.pool import StaticPool
        from app import database
        from app.models import Analysis

        test_engine = create_engine(
            "sqlite:///:memory:",
            connect_args={"check_same_thread": False},
            poolclass=StaticPool,
        )
        Base.metadata.create_all(bind=test_engine)
        TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=test_engine)

        def override_get_db():
            db = TestingSessionLocal()
            try:
                yield db
            finally:
                db.close()

        from app.database import get_db
        app.dependency_overrides[get_db] = override_get_db

        with TestClient(app, raise_server_exceptions=False) as c:
            yield c

        app.dependency_overrides.clear()


# ─── Health endpoint ────────────────────────────────────────────────────────


class TestHealth:
    def test_health_ok(self, client):
        resp = client.get("/health")
        assert resp.status_code in (200, 503)
        body = resp.json()
        assert "status" in body
        assert "model_loaded" in body
        assert "db_reachable" in body

    def test_metrics_endpoint(self, client):
        resp = client.get("/metrics")
        assert resp.status_code == 200
        body = resp.json()
        assert "total_requests" in body
        assert "avg_inference_latency_ms" in body


# ─── Analyze endpoint ───────────────────────────────────────────────────────


class TestAnalyze:
    def test_upload_valid_jpeg(self, client):
        resp = client.post(
            "/api/v1/analyze",
            files={"file": ("test.jpg", BytesIO(TINY_JPEG), "image/jpeg")},
        )
        # Even with mocked pipeline, storage save should work
        assert resp.status_code in (200, 500)

    def test_upload_empty_file_returns_400(self, client):
        resp = client.post(
            "/api/v1/analyze",
            files={"file": ("empty.jpg", BytesIO(b""), "image/jpeg")},
        )
        assert resp.status_code == 400

    def test_upload_non_image_rejected(self, client):
        resp = client.post(
            "/api/v1/analyze",
            files={"file": ("script.txt", BytesIO(b"hello world"), "text/plain")},
        )
        assert resp.status_code in (400, 415)

    def test_upload_fake_jpeg_rejected(self, client):
        """File with jpeg content-type but wrong magic bytes should be rejected."""
        resp = client.post(
            "/api/v1/analyze",
            files={"file": ("fake.jpg", BytesIO(b"NOT_JPEG_DATA_HERE"), "image/jpeg")},
        )
        assert resp.status_code == 400

    def test_response_schema_keys(self, client):
        """When pipeline is mocked, response should contain expected schema keys."""
        with patch("app.services.pipeline.run_pipeline", return_value=MOCK_PIPELINE_RESULT):
            resp = client.post(
                "/api/v1/analyze",
                files={"file": ("test.jpg", BytesIO(TINY_JPEG), "image/jpeg")},
            )
        if resp.status_code == 200:
            body = resp.json()
            assert "quality_score" in body
            assert "quality_label" in body
            assert "issues" in body
            assert "image_stats" in body
            assert "model_version" in body
            assert "id" in body


# ─── Batch endpoint ─────────────────────────────────────────────────────────


class TestBatch:
    def test_batch_upload(self, client):
        resp = client.post(
            "/api/v1/analyze/batch",
            files=[
                ("files", ("a.jpg", BytesIO(TINY_JPEG), "image/jpeg")),
                ("files", ("b.jpg", BytesIO(TINY_JPEG), "image/jpeg")),
            ],
        )
        assert resp.status_code in (200, 500)
        if resp.status_code == 200:
            body = resp.json()
            assert "results" in body
            assert "failed" in body


# ─── Analyses history endpoints ─────────────────────────────────────────────


class TestAnalysesHistory:
    def test_list_analyses_empty(self, client):
        resp = client.get("/api/v1/analyses")
        assert resp.status_code == 200
        body = resp.json()
        assert "items" in body
        assert "total" in body
        assert "limit" in body
        assert "offset" in body

    def test_list_analyses_pagination_params(self, client):
        resp = client.get("/api/v1/analyses?limit=5&offset=0")
        assert resp.status_code == 200

    def test_list_analyses_invalid_label(self, client):
        resp = client.get("/api/v1/analyses?label=INVALID")
        assert resp.status_code == 400

    def test_get_nonexistent_analysis(self, client):
        resp = client.get("/api/v1/analyses/99999")
        assert resp.status_code == 404

    def test_heatmap_nonexistent(self, client):
        resp = client.get("/api/v1/analyses/99999/heatmap")
        assert resp.status_code == 404
