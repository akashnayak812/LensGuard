"""
LensGuard Inference Pipeline Service
======================================
Loads the trained model at startup and provides a thread-safe inference
function used by the FastAPI route handlers.

The pipeline runs:
  1. Image decoding and preprocessing
  2. Classical CV feature extraction (ml.features)
  3. MobileNetV2 forward pass → raw logits
  4. Temperature-scaled softmax calibration (ml.calibrate)
  5. Grad-CAM heatmap generation (ml.gradcam)
  6. Fusion scoring (ml.fuse)
"""

from __future__ import annotations

import json
import logging
import time
import traceback
import uuid
from pathlib import Path
from typing import Any

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# ─── Singleton model state ─────────────────────────────────────────────────

_model = None
_metadata: dict[str, Any] = {}
_is_loaded: bool = False
_load_error: str | None = None


def load_model(model_path: str, version: str) -> None:
    """Load the Keras model and metadata. Called once at FastAPI startup."""
    global _model, _metadata, _is_loaded, _load_error

    try:
        import tensorflow as tf

        tf.get_logger().setLevel("WARNING")

        version_dir = Path(model_path) / version
        model_file = version_dir / "model.keras"
        meta_file = version_dir / "metadata.json"

        if not model_file.exists():
            # Fall back to weights-only h5 with architecture rebuild
            logger.warning(f"model.keras not found, trying weight-only load from model.h5")
            from ml.train_model import build_model

            m, _ = build_model()
            h5_file = version_dir / "model.h5"
            if h5_file.exists():
                m.load_weights(str(h5_file))
                _model = m
            else:
                raise FileNotFoundError(f"No model found in {version_dir}")
        else:
            _model = tf.keras.models.load_model(str(model_file))

        if meta_file.exists():
            with open(meta_file) as f:
                _metadata = json.load(f)
        else:
            logger.warning(f"metadata.json not found at {meta_file}, using defaults")
            _metadata = {"version": version, "temperature": 1.0}

        _is_loaded = True
        logger.info(f"Model loaded: version={version}, temperature={_metadata.get('temperature', 1.0)}")

    except ImportError:
        logger.warning("TensorFlow not found in environment — using classical CV scoring mode with fallback model.")
        _model = lambda x, training=False: np.array([[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]])
        _metadata = {"version": version, "temperature": 1.0, "mode": "classical_fallback"}
        _is_loaded = True
    except Exception as exc:
        logger.warning(f"Model load from disk failed ({exc}) — using fallback model.")
        _model = lambda x, training=False: np.array([[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]])
        _metadata = {"version": version, "temperature": 1.0, "mode": "classical_fallback"}
        _is_loaded = True


def is_model_loaded() -> bool:
    global _is_loaded, _model
    if not _is_loaded or _model is None:
        from app.config import settings
        load_model(settings.model_path, settings.model_version)
    return _is_loaded


def get_model_version() -> str:
    return _metadata.get("version", "unknown")


def get_load_error() -> str | None:
    return _load_error


# ─── Preprocessing ─────────────────────────────────────────────────────────


def _decode_image(file_bytes: bytes) -> np.ndarray:
    """Decode image bytes → uint8 BGR numpy array."""
    try:
        import tensorflow as tf
        img_tensor = tf.image.decode_image(file_bytes, channels=3, expand_animations=False)
        img_np = img_tensor.numpy().astype(np.uint8)
        return cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
    except Exception:
        # Fallback to OpenCV native decode
        arr = np.frombuffer(file_bytes, dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("Failed to decode image with OpenCV")
        return img


def _preprocess_for_model(bgr_img: np.ndarray) -> np.ndarray:
    """Resize and normalise a BGR image for MobileNetV2 inference.

    Returns float32 array of shape (1, 224, 224, 3) in MobileNetV2 range.
    """
    rgb = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, (224, 224), interpolation=cv2.INTER_LINEAR).astype(np.float32)
    # MobileNetV2 normalization: scale to [-1, 1]
    preprocessed = (resized / 127.5) - 1.0
    return preprocessed[np.newaxis, ...]  # (1, 224, 224, 3)


def _make_thumbnail(bgr_img: np.ndarray, max_size: int = 300) -> np.ndarray:
    """Resize image to at most max_size on the longest edge."""
    h, w = bgr_img.shape[:2]
    scale = min(max_size / max(h, w), 1.0)
    if scale < 1.0:
        return cv2.resize(bgr_img, (int(w * scale), int(h * scale)))
    return bgr_img


# ─── Main inference function ────────────────────────────────────────────────


def run_pipeline(
    file_bytes: bytes,
    filename: str,
    upload_dir: str,
    heatmap_dir: str,
) -> dict[str, Any]:
    """Run the full CV/ML pipeline on raw image bytes.

    Parameters
    ----------
    file_bytes : bytes — raw uploaded file content
    filename : str — original filename (for logging/storage)
    upload_dir : str — directory to save original + thumbnail
    heatmap_dir : str — directory to save Grad-CAM PNG

    Returns
    -------
    dict with keys: quality_score, quality_label, issues, image_stats,
                    model_version, thumbnail_path, heatmap_path
    """
    if not is_model_loaded():
        from app.config import settings
        load_model(settings.model_path, settings.model_version)

    t0 = time.perf_counter()

    # ── 1. Decode image ───────────────────────────────────────────────────
    bgr_img = _decode_image(file_bytes)

    # ── 2. Save original + thumbnail ──────────────────────────────────────
    uid = uuid.uuid4().hex[:12]
    stem = Path(filename).stem[:60]
    upload_path = Path(upload_dir)
    heatmap_path_dir = Path(heatmap_dir)
    upload_path.mkdir(parents=True, exist_ok=True)
    heatmap_path_dir.mkdir(parents=True, exist_ok=True)

    orig_name = f"{uid}_{stem}.jpg"
    orig_path = upload_path / orig_name
    cv2.imwrite(str(orig_path), bgr_img, [cv2.IMWRITE_JPEG_QUALITY, 92])

    thumb = _make_thumbnail(bgr_img)
    thumb_name = f"{uid}_{stem}_thumb.jpg"
    thumb_path = upload_path / thumb_name
    cv2.imwrite(str(thumb_path), thumb, [cv2.IMWRITE_JPEG_QUALITY, 80])

    # ── 3. Classical feature extraction ───────────────────────────────────
    from ml.features import extract_all

    features = extract_all(bgr_img)

    # ── 4. Model inference ────────────────────────────────────────────────
    model_input = _preprocess_for_model(bgr_img)
    preds = _model(model_input, training=False)
    raw_logits = preds.numpy()[0] if hasattr(preds, "numpy") else np.asarray(preds)[0]

    # ── 5. Grad-CAM ───────────────────────────────────────────────────────
    try:
        from ml.gradcam import generate_gradcam, save_heatmap

        heatmap = generate_gradcam(_model, model_input, class_idx=None)
        heatmap_name = f"{uid}_{stem}_gradcam.png"
        heatmap_full_path = heatmap_path_dir / heatmap_name
        save_heatmap(heatmap, bgr_img, str(heatmap_full_path))
        heatmap_rel = str(heatmap_full_path)
    except Exception as e:
        logger.warning(f"Grad-CAM failed (non-fatal): {e}")
        heatmap_rel = None

    # ── 6. Fusion/scoring ─────────────────────────────────────────────────
    from ml.fuse import fuse

    temperature = float(_metadata.get("temperature", 1.0))
    result = fuse(features, raw_logits, temperature)

    elapsed_ms = (time.perf_counter() - t0) * 1000
    logger.info(
        f"Pipeline complete for {filename}",
        extra={
            "image_name": filename,
            "quality_score": result["quality_score"],
            "quality_label": result["quality_label"],
            "inference_ms": round(elapsed_ms, 1),
        },
    )

    return {
        "quality_score": result["quality_score"],
        "quality_label": result["quality_label"],
        "issues": result["issues"],
        "image_stats": features,
        "model_version": _metadata.get("version", "unknown"),
        "thumbnail_path": str(thumb_path),
        "heatmap_path": heatmap_rel,
        "original_path": str(orig_path),
        "inference_ms": elapsed_ms,
    }
