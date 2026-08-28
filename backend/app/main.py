"""
LensGuard FastAPI Application
================================
Entry point. Configures:
  - Structured JSON logging (python-json-logger)
  - CORS middleware
  - Application startup (model load + DB init)
  - /health and /metrics endpoints
  - All API routers
"""

from __future__ import annotations

import logging
import os
import sys
import time
import traceback
from collections import defaultdict
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pythonjsonlogger import jsonlogger

# ─── Configure structured JSON logging before anything else ────────────────

def _setup_logging(level: str = "INFO") -> None:
    handler = logging.StreamHandler(sys.stdout)
    formatter = jsonlogger.JsonFormatter(
        fmt="%(asctime)s %(levelname)s %(name)s %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )
    handler.setFormatter(formatter)
    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(getattr(logging, level.upper(), logging.INFO))


# ─── Add backend & project root to sys.path ────────────────────────────────
_BACKEND_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_PROJECT_ROOT = os.path.dirname(_BACKEND_DIR)
for _p in (_BACKEND_DIR, _PROJECT_ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# Allow settings to be loaded before other imports
from app.config import settings

_setup_logging(settings.log_level)
logger = logging.getLogger(__name__)

# ─── Metrics state ─────────────────────────────────────────────────────────

_metrics: dict[str, Any] = {
    "total_requests": 0,
    "total_analyses": 0,
    "inference_latencies_ms": [],
    "error_count": 0,
    "start_time": time.time(),
}


# ─── Lifespan ──────────────────────────────────────────────────────────────


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup: initialise DB and load ML model. Shutdown: cleanup."""
    logger.info("LensGuard starting up...")

    # Initialise database
    try:
        from app.database import init_db
        init_db()
        logger.info("Database initialised")
    except Exception as e:
        logger.error(f"Database init failed: {e}")

    # Load ML model
    try:
        from app.services.pipeline import load_model
        load_model(settings.model_path, settings.model_version)
        logger.info("ML model loaded successfully")
    except Exception as e:
        logger.error(f"Model load failed: {e}\n{traceback.format_exc()}")

    _metrics["start_time"] = time.time()
    yield

    logger.info("LensGuard shutting down.")


# ─── FastAPI application ────────────────────────────────────────────────────

app = FastAPI(
    title="LensGuard API",
    description="AI-powered image quality and defect detection. All inference is local — no external APIs.",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─── Metrics middleware ─────────────────────────────────────────────────────


@app.middleware("http")
async def metrics_middleware(request: Request, call_next):
    _metrics["total_requests"] += 1
    t0 = time.perf_counter()
    try:
        response = await call_next(request)
        return response
    except Exception:
        _metrics["error_count"] += 1
        raise
    finally:
        elapsed_ms = (time.perf_counter() - t0) * 1000
        if "/analyze" in request.url.path and request.method == "POST":
            _metrics["total_analyses"] += 1
            _metrics["inference_latencies_ms"].append(elapsed_ms)
            # Keep only last 1000 latency samples
            if len(_metrics["inference_latencies_ms"]) > 1000:
                _metrics["inference_latencies_ms"] = _metrics["inference_latencies_ms"][-1000:]


# ─── System endpoints ───────────────────────────────────────────────────────


@app.get("/health", tags=["system"])
def health():
    """Liveness + readiness check. Returns 200 only when model is loaded and DB is reachable."""
    from app.services.pipeline import is_model_loaded, get_model_version, get_load_error
    from app.database import engine
    import sqlalchemy

    model_loaded = is_model_loaded()
    load_error = get_load_error()

    db_ok = False
    try:
        with engine.connect() as conn:
            conn.execute(sqlalchemy.text("SELECT 1"))
        db_ok = True
    except Exception as e:
        logger.warning(f"DB health check failed: {e}")

    uptime = time.time() - _metrics["start_time"]
    status = "ready" if (model_loaded and db_ok) else "degraded"

    return JSONResponse(
        status_code=200 if status == "ready" else 503,
        content={
            "status": status,
            "model_loaded": model_loaded,
            "model_version": get_model_version(),
            "db_reachable": db_ok,
            "uptime_seconds": round(uptime, 1),
            "load_error": load_error,
        },
    )


@app.get("/metrics", tags=["system"])
def metrics():
    """Lightweight metrics endpoint: request counts and inference latency stats."""
    latencies = _metrics["inference_latencies_ms"]
    avg_latency = sum(latencies) / max(len(latencies), 1)

    return {
        "total_requests": _metrics["total_requests"],
        "total_analyses": _metrics["total_analyses"],
        "avg_inference_latency_ms": round(avg_latency, 2),
        "p95_inference_latency_ms": round(sorted(latencies)[int(0.95 * len(latencies))] if latencies else 0, 2),
        "error_count": _metrics["error_count"],
        "uptime_seconds": round(time.time() - _metrics["start_time"], 1),
    }


# ─── Routers ───────────────────────────────────────────────────────────────

from app.routers import analyze, analyses  # noqa: E402

app.include_router(analyze.router)
app.include_router(analyses.router)


# ─── Global exception handler ───────────────────────────────────────────────


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    _metrics["error_count"] += 1
    logger.error(f"Unhandled exception: {exc}\n{traceback.format_exc()}")
    return JSONResponse(
        status_code=500,
        content={"error": {"code": "INTERNAL_ERROR", "message": "An unexpected error occurred. Please try again."}},
    )
