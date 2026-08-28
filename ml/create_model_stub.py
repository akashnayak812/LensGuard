#!/usr/bin/env python3
"""
create_model_stub.py
====================
Creates a minimal untrained MobileNetV2 model stub with default metadata.
This lets the Docker container start and serve the API without requiring
a full training run.

The model will return near-random predictions but the full pipeline will
run successfully, allowing UI/API testing without GPU access.

Run during Docker build (in Dockerfile.backend) before starting the server.
"""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path


def create_stub(model_dir: str = "/app/ml/models", version: str = "v1") -> None:
    """Create a stub model with random weights if no trained model exists."""
    import tensorflow as tf

    version_dir = Path(model_dir) / version
    model_path = version_dir / "model.keras"
    meta_path = version_dir / "metadata.json"

    if model_path.exists() and meta_path.exists():
        print(f"Model already exists at {model_path}, skipping stub creation.")
        return

    version_dir.mkdir(parents=True, exist_ok=True)
    print(f"Creating stub model at {model_path}...")

    CLASSES = ["clean", "blur", "underexposed", "overexposed", "noisy", "corrupted", "defective"]
    IMG_SIZE = (224, 224)

    base_model = tf.keras.applications.MobileNetV2(
        input_shape=(*IMG_SIZE, 3),
        include_top=False,
        weights="imagenet",
        pooling="avg",
    )
    base_model.trainable = False

    inputs = tf.keras.Input(shape=(*IMG_SIZE, 3), name="image_input")
    x = base_model(inputs, training=False)
    x = tf.keras.layers.Dropout(0.3)(x)
    x = tf.keras.layers.Dense(256, activation="relu", name="dense_head")(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    outputs = tf.keras.layers.Dense(len(CLASSES), name="logits")(x)

    model = tf.keras.Model(inputs, outputs, name="LensGuard_v1_stub")
    model.save(str(model_path))
    model.save_weights(str(version_dir / "model.h5"))

    metadata = {
        "version": version,
        "training_date": datetime.now(timezone.utc).isoformat(),
        "classes": CLASSES,
        "img_size": list(IMG_SIZE),
        "phase1_epochs": 0,
        "phase2_epochs": 0,
        "val_loss": None,
        "val_accuracy": None,
        "temperature": 1.0,
        "architecture": "MobileNetV2 + 2-layer dense head (STUB — not trained)",
        "note": "This is a stub model with ImageNet backbone but untrained head. Run ml/train_model.py for production use.",
    }
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"✓ Stub model created: {model_path}")
    print(f"✓ Metadata written:   {meta_path}")
    print("  NOTE: This is an untrained stub. Predictions will be near-random.")
    print("  Run `python ml/train_model.py` to train a production model.")


if __name__ == "__main__":
    model_dir = os.environ.get("MODEL_PATH", "ml/models")
    version = os.environ.get("MODEL_VERSION", "v1")
    create_stub(model_dir, version)
