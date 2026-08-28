#!/usr/bin/env python3
"""
LensGuard Confidence Calibration — Temperature Scaling
=======================================================
After training, raw softmax outputs from neural networks are often
overconfident: the model assigns high probability to its top class even when
it is uncertain. Temperature scaling (Guo et al., 2017) is a simple,
post-hoc calibration method that divides the logits by a learned scalar T
before applying softmax:

    p_calibrated(y | x) = softmax(z / T)

where z are the raw logits from the model and T > 1 softens the distribution,
reducing overconfidence. T is fitted by minimising Negative Log-Likelihood
(NLL) on the validation set — a one-dimensional scalar optimisation.

This script:
1. Loads the trained model and the validation split.
2. Collects raw logits for all validation images.
3. Minimises NLL over T using scipy.optimize.minimize_scalar.
4. Updates `temperature` in the model's metadata.json.

The temperature is then applied at inference time in ml/fuse.py via
``apply_temperature_scaling``.

Usage
-----
    python ml/calibrate.py --data-dir data/ --model-dir ml/models/v1
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import minimize_scalar


# ─────────────────────────────────────────────────────────────────────────────
# Logit collection
# ─────────────────────────────────────────────────────────────────────────────


def collect_logits(model, val_rows: list[dict], data_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    """Run the model on all validation images and return raw logits + true labels.

    Parameters
    ----------
    model : tf.keras.Model
    val_rows : list of manifest row dicts (split == "val")
    data_dir : Path to dataset root

    Returns
    -------
    logits : np.ndarray of shape (N, num_classes)
    labels : np.ndarray of shape (N,) — integer class indices
    """
    import tensorflow as tf
    from ml.train_model import CLASS_TO_IDX, IMG_SIZE

    all_logits: list[np.ndarray] = []
    all_labels: list[int] = []

    for row in val_rows:
        img_path = str(data_dir / row["filename"])
        raw = tf.io.read_file(img_path)
        img = tf.image.decode_jpeg(raw, channels=3)
        img = tf.image.resize(img, IMG_SIZE)
        img = tf.keras.applications.mobilenet_v2.preprocess_input(img)
        img = tf.expand_dims(img, 0)

        logits = model(img, training=False).numpy()  # (1, num_classes)
        all_logits.append(logits[0])
        all_labels.append(CLASS_TO_IDX[row["issue_type"]])

    return np.array(all_logits, dtype=np.float32), np.array(all_labels, dtype=np.int32)


# ─────────────────────────────────────────────────────────────────────────────
# Calibration
# ─────────────────────────────────────────────────────────────────────────────


def softmax_with_temperature(logits: np.ndarray, temperature: float) -> np.ndarray:
    """Apply temperature scaling and softmax to a batch of logits.

    Parameters
    ----------
    logits : np.ndarray of shape (N, C)
    temperature : float — must be > 0

    Returns
    -------
    np.ndarray of shape (N, C) — calibrated probabilities
    """
    scaled = logits / max(temperature, 1e-6)
    # Numerically stable softmax
    exp = np.exp(scaled - scaled.max(axis=1, keepdims=True))
    return exp / exp.sum(axis=1, keepdims=True)


def nll_loss(temperature: float, logits: np.ndarray, labels: np.ndarray) -> float:
    """Negative log-likelihood for the temperature parameter.

    This is the objective minimised during calibration.

    Parameters
    ----------
    temperature : float
    logits : np.ndarray (N, C)
    labels : np.ndarray (N,) — integer class indices

    Returns
    -------
    float — mean NLL across the validation set
    """
    probs = softmax_with_temperature(logits, temperature)
    n = len(labels)
    # Gather true-class probabilities; clip to avoid log(0)
    true_probs = probs[np.arange(n), labels]
    true_probs = np.clip(true_probs, 1e-9, 1.0)
    return float(-np.mean(np.log(true_probs)))


def fit_temperature(logits: np.ndarray, labels: np.ndarray) -> float:
    """Find the optimal temperature T* by minimising NLL on the validation set.

    Uses scipy's bounded scalar minimisation (Brent's method). Search range
    is [0.1, 10.0] — temperatures outside this range are pathological.

    Returns
    -------
    float — optimal temperature (T* ≈ 1 means already well-calibrated;
                                  T* > 1 reduces overconfidence)
    """
    result = minimize_scalar(
        nll_loss,
        bounds=(0.1, 10.0),
        method="bounded",
        args=(logits, labels),
        options={"xatol": 1e-4},
    )
    return float(result.x)


def apply_temperature_scaling(logits: np.ndarray, temperature: float) -> np.ndarray:
    """Public function used at inference time to apply stored temperature.

    Parameters
    ----------
    logits : np.ndarray of shape (C,) or (N, C) — raw model output
    temperature : float — from metadata.json

    Returns
    -------
    np.ndarray — calibrated probabilities, same shape as logits
    """
    scalar = logits.ndim == 1
    if scalar:
        logits = logits[np.newaxis, :]
    probs = softmax_with_temperature(logits, temperature)
    return probs[0] if scalar else probs


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────


def main(args: argparse.Namespace) -> None:
    import tensorflow as tf

    model_dir = Path(args.model_dir).resolve()
    data_dir = Path(args.data_dir).resolve()

    # Load model
    model_path = model_dir / "model.keras"
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found at {model_path}")
    print(f"Loading model from {model_path}...")
    model = tf.keras.models.load_model(str(model_path))

    # Load val rows from manifest
    with open(data_dir / "manifest.json") as f:
        all_rows = json.load(f)
    val_rows = [r for r in all_rows if r["split"] == "val"]
    print(f"Calibrating on {len(val_rows)} validation images...")

    # Collect logits
    logits, labels = collect_logits(model, val_rows, data_dir)
    print(f"Logits shape: {logits.shape}, Labels shape: {labels.shape}")

    # Pre-calibration NLL
    pre_nll = nll_loss(1.0, logits, labels)
    print(f"NLL before calibration (T=1.0): {pre_nll:.4f}")

    # Fit temperature
    T = fit_temperature(logits, labels)
    post_nll = nll_loss(T, logits, labels)
    print(f"Optimal temperature T* = {T:.4f}")
    print(f"NLL after calibration (T={T:.4f}): {post_nll:.4f}")

    # Update metadata.json
    meta_path = model_dir / "metadata.json"
    with open(meta_path) as f:
        metadata = json.load(f)
    metadata["temperature"] = T
    metadata["calibration_pre_nll"] = pre_nll
    metadata["calibration_post_nll"] = post_nll
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"Updated temperature in {meta_path}")
    print("✓ Calibration complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Temperature-scale calibrate the LensGuard model")
    parser.add_argument("--data-dir", default="data/", help="Dataset directory")
    parser.add_argument("--model-dir", default="ml/models/v1", help="Model version directory")
    main(parser.parse_args())
