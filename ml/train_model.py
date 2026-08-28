#!/usr/bin/env python3
"""
LensGuard Model Training Script
================================
Fine-tunes a MobileNetV2 backbone on the synthetic dataset produced by
generate_dataset.py to classify image quality defects.

Architecture decision
---------------------
We use a **single-label primary classifier** over 7 classes:
  0=clean, 1=blur, 2=underexposed, 3=overexposed, 4=noisy, 5=corrupted, 6=defective

This is simpler to calibrate (temperature scaling on 7-way softmax) and more
interpretable than a multi-label head. The primary class predicted drives the
issue type reported; severity is derived from classical CV features (see fuse.py).

Training strategy
-----------------
Phase 1 (10 epochs): Backbone frozen, only the dense head trained. This lets
the randomly initialised head converge before the backbone weights are altered.

Phase 2 (5 epochs): Top 30 layers of MobileNetV2 unfrozen for fine-tuning at
a reduced learning rate (10× smaller). This adapts high-level ImageNet features
to the domain of quality-degraded images.

Usage
-----
    python ml/train_model.py --data-dir data/ --output-dir ml/models --version v1
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# Class definitions
# ─────────────────────────────────────────────────────────────────────────────

CLASSES = ["clean", "blur", "underexposed", "overexposed", "noisy", "corrupted", "defective"]
CLASS_TO_IDX: dict[str, int] = {c: i for i, c in enumerate(CLASSES)}
NUM_CLASSES = len(CLASSES)

IMG_SIZE = (224, 224)
BATCH_SIZE = 32
PHASE1_EPOCHS = 10
PHASE2_EPOCHS = 5
PHASE1_LR = 1e-3
PHASE2_LR = 1e-4


# ─────────────────────────────────────────────────────────────────────────────
# Data loading
# ─────────────────────────────────────────────────────────────────────────────


def load_manifest(data_dir: Path) -> dict[str, list[dict]]:
    """Load dataset manifest and group by split."""
    manifest_path = data_dir / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"Manifest not found at {manifest_path}. "
            "Run `python ml/generate_dataset.py` first."
        )
    with open(manifest_path) as f:
        rows = json.load(f)

    splits: dict[str, list[dict]] = {"train": [], "val": [], "test": []}
    for row in rows:
        splits[row["split"]].append(row)
    return splits


def make_dataset(rows: list[dict], data_dir: Path, augment: bool = False):
    """Create a tf.data.Dataset from manifest rows."""
    import tensorflow as tf

    file_paths = [str(data_dir / row["filename"]) for row in rows]
    labels = [CLASS_TO_IDX[row["issue_type"]] for row in rows]

    ds = tf.data.Dataset.from_tensor_slices((file_paths, labels))

    def load_image(path, label):
        raw = tf.io.read_file(path)
        img = tf.image.decode_jpeg(raw, channels=3)
        img = tf.image.resize(img, IMG_SIZE)
        img = tf.keras.applications.mobilenet_v2.preprocess_input(img)
        return img, tf.one_hot(label, NUM_CLASSES)

    def augment_fn(img, label):
        img = tf.image.random_flip_left_right(img)
        img = tf.image.random_brightness(img, 0.1)
        img = tf.image.random_contrast(img, 0.9, 1.1)
        return img, label

    ds = ds.map(load_image, num_parallel_calls=tf.data.AUTOTUNE)
    if augment:
        ds = ds.map(augment_fn, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.cache().shuffle(buffer_size=min(len(rows), 5000)).batch(BATCH_SIZE)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds


# ─────────────────────────────────────────────────────────────────────────────
# Model definition
# ─────────────────────────────────────────────────────────────────────────────


def build_model(num_classes: int = NUM_CLASSES):
    """Build the MobileNetV2 + custom head model.

    Returns the compiled model with backbone frozen (Phase 1 config).
    """
    import tensorflow as tf

    base_model = tf.keras.applications.MobileNetV2(
        input_shape=(*IMG_SIZE, 3),
        include_top=False,
        weights="imagenet",
        pooling="avg",
    )
    base_model.trainable = False

    inputs = tf.keras.Input(shape=(*IMG_SIZE, 3), name="image_input")
    x = base_model(inputs, training=False)
    x = tf.keras.layers.Dropout(0.3, name="dropout")(x)
    x = tf.keras.layers.Dense(256, activation="relu", name="dense_head")(x)
    x = tf.keras.layers.Dropout(0.2, name="dropout2")(x)
    outputs = tf.keras.layers.Dense(num_classes, name="logits")(x)  # raw logits

    model = tf.keras.Model(inputs, outputs, name="LensGuard_v1")
    return model, base_model


def compile_model(model, lr: float):
    import tensorflow as tf

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
        metrics=[tf.keras.metrics.CategoricalAccuracy(name="accuracy")],
    )


# ─────────────────────────────────────────────────────────────────────────────
# Training
# ─────────────────────────────────────────────────────────────────────────────


def compute_class_weights(rows: list[dict]) -> dict[int, float]:
    """Compute balanced class weights to handle synthetic imbalance."""
    from sklearn.utils.class_weight import compute_class_weight

    labels = [CLASS_TO_IDX[row["issue_type"]] for row in rows]
    unique = sorted(set(labels))
    weights = compute_class_weight("balanced", classes=np.array(unique), y=np.array(labels))
    return {cls: float(w) for cls, w in zip(unique, weights)}


class TrainingLogger:
    """Accumulate per-epoch metrics into a JSON-serialisable list."""

    def __init__(self):
        self.log: list[dict[str, Any]] = []

    def record(self, phase: str, epoch: int, metrics: dict[str, float]) -> None:
        self.log.append({"phase": phase, "epoch": epoch, **metrics})

    def save(self, path: Path) -> None:
        with open(path, "w") as f:
            json.dump(self.log, f, indent=2)
        print(f"Training log saved to {path}")


def train(
    train_ds,
    val_ds,
    class_weights: dict[int, float],
    model,
    base_model,
    logger: TrainingLogger,
):
    """Two-phase training. Logs epoch metrics to logger. Returns val metrics."""
    import tensorflow as tf

    # ── Phase 1: frozen backbone ──────────────────────────────────────────
    print("\n=== Phase 1: Training head (backbone frozen) ===")
    compile_model(model, PHASE1_LR)

    for epoch in range(PHASE1_EPOCHS):
        history = model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=epoch + 1,
            initial_epoch=epoch,
            class_weight=class_weights,
            verbose=1,
        )
        metrics = {
            "loss": float(history.history["loss"][-1]),
            "accuracy": float(history.history["accuracy"][-1]),
            "val_loss": float(history.history["val_loss"][-1]),
            "val_accuracy": float(history.history["val_accuracy"][-1]),
        }
        logger.record("phase1", epoch + 1, metrics)

    # ── Phase 2: fine-tune top layers ────────────────────────────────────
    print("\n=== Phase 2: Fine-tuning top 30 layers ===")
    base_model.trainable = True
    for layer in base_model.layers[:-30]:
        layer.trainable = False
    compile_model(model, PHASE2_LR)

    for epoch in range(PHASE2_EPOCHS):
        global_epoch = PHASE1_EPOCHS + epoch
        history = model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=global_epoch + 1,
            initial_epoch=global_epoch,
            class_weight=class_weights,
            verbose=1,
        )
        metrics = {
            "loss": float(history.history["loss"][-1]),
            "accuracy": float(history.history["accuracy"][-1]),
            "val_loss": float(history.history["val_loss"][-1]),
            "val_accuracy": float(history.history["val_accuracy"][-1]),
        }
        logger.record("phase2", epoch + 1, metrics)

    # Final val metrics
    val_results = model.evaluate(val_ds, verbose=0)
    return {"val_loss": float(val_results[0]), "val_accuracy": float(val_results[1])}


# ─────────────────────────────────────────────────────────────────────────────
# Model persistence
# ─────────────────────────────────────────────────────────────────────────────


def save_model(model, output_dir: Path, version: str, val_metrics: dict, logger: TrainingLogger) -> None:
    """Save weights and metadata under output_dir/version/."""
    version_dir = output_dir / version
    version_dir.mkdir(parents=True, exist_ok=True)

    # Save weights in TF SavedModel format (h5 deprecated in TF2.16+)
    model_path = version_dir / "model.keras"
    model.save(str(model_path))
    print(f"Model saved to {model_path}")

    # Also export as h5 for backward compat (spec requires model.h5)
    h5_path = version_dir / "model.h5"
    model.save_weights(str(h5_path))
    print(f"Weights saved to {h5_path}")

    metadata = {
        "version": version,
        "training_date": datetime.now(timezone.utc).isoformat(),
        "classes": CLASSES,
        "img_size": list(IMG_SIZE),
        "phase1_epochs": PHASE1_EPOCHS,
        "phase2_epochs": PHASE2_EPOCHS,
        "val_loss": val_metrics["val_loss"],
        "val_accuracy": val_metrics["val_accuracy"],
        "temperature": 1.0,  # updated by calibrate.py after training
        "architecture": "MobileNetV2 + 2-layer dense head (single-label, 7 classes)",
    }
    meta_path = version_dir / "metadata.json"
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"Metadata saved to {meta_path}")

    # Save training log alongside model
    logger.save(output_dir.parent / "training_log.json")


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────


def main(args: argparse.Namespace) -> None:
    import tensorflow as tf

    tf.get_logger().setLevel("WARNING")
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

    data_dir = Path(args.data_dir).resolve()
    output_dir = Path(args.output_dir).resolve()

    print(f"Data dir  : {data_dir}")
    print(f"Output dir: {output_dir}")
    print(f"Version   : {args.version}")

    splits = load_manifest(data_dir)
    print(f"Train: {len(splits['train'])} | Val: {len(splits['val'])} | Test: {len(splits['test'])}")

    train_ds = make_dataset(splits["train"], data_dir, augment=True)
    val_ds = make_dataset(splits["val"], data_dir, augment=False)

    class_weights = compute_class_weights(splits["train"])
    print(f"Class weights: {class_weights}")

    model, base_model = build_model()
    model.summary()

    logger = TrainingLogger()
    val_metrics = train(train_ds, val_ds, class_weights, model, base_model, logger)
    print(f"\nFinal val metrics: {val_metrics}")

    save_model(model, output_dir, args.version, val_metrics, logger)
    print("\n✓ Training complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the LensGuard quality classifier")
    parser.add_argument("--data-dir", default="data/", help="Dataset directory")
    parser.add_argument("--output-dir", default="ml/models", help="Model output directory")
    parser.add_argument("--version", default="v1", help="Version tag for saved model")
    main(parser.parse_args())
