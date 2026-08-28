#!/usr/bin/env python3
"""
LensGuard Synthetic Dataset Generator
======================================
Generates a labelled image dataset for training the quality-defect classifier.

Source images: CIFAR-100 test split (10,000 images, 32×32) upscaled to 224×224.
For each source image we generate degraded variants covering 6 defect types at
3 severity levels (low / medium / high), keeping a proportion of images clean.

Split strategy: the train/val/test split is performed on SOURCE image indices
**before** generating degraded versions. This prevents leakage where the same
underlying scene appears in both training and test sets.

Usage
-----
    python ml/generate_dataset.py --output-dir data/ --seed 42
"""

import argparse
import csv
import json
import random
import sys
import os
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
from PIL import Image

# ── CIFAR-100 is fetched via TensorFlow; keep TF import local to avoid slow
#    startup when other modules import from this file. ─────────────────────


DEFECT_TYPES = [
    "blur",
    "underexposed",
    "overexposed",
    "noisy",
    "corrupted",
    "defective",
    "clean",
]

SEVERITIES = ["low", "medium", "high"]

# Fraction of the source images to leave completely unmodified (clean samples).
CLEAN_FRACTION = 0.15

# How many degraded variants to generate per source image per split.
# Keeping this at 1 per (defect, severity) gives 6 × 3 = 18 variants per source.
VARIANTS_PER_COMBO = 1

# Train / val / test split ratios (applied to source image indices).
SPLIT_RATIOS = {"train": 0.70, "val": 0.15, "test": 0.15}

# Maximum source images to use (set lower to speed up quick runs).
MAX_SOURCE_IMAGES = 2000  # set to None to use full 10k


# ─────────────────────────────────────────────────────────────────────────────
# Degradation functions
# Each function accepts a uint8 HxWxC numpy array and returns uint8 HxWxC.
# ─────────────────────────────────────────────────────────────────────────────


def apply_blur(img: np.ndarray, severity: str) -> np.ndarray:
    """Apply Gaussian blur. Kernel size scales with severity.

    Theory: Gaussian smoothing attenuates high-spatial-frequency components.
    The Laplacian variance metric drops dramatically as kernel size grows,
    making blur directly measurable via the CV pipeline.
    """
    params = {"low": (5, 1.5), "medium": (15, 5.0), "high": (31, 10.0)}
    ksize, sigma = params[severity]
    return cv2.GaussianBlur(img, (ksize, ksize), sigma)


def apply_underexposure(img: np.ndarray, severity: str) -> np.ndarray:
    """Darken image by applying gamma > 1 (power-law transform).

    Theory: gamma > 1 compresses bright pixels and pushes the luminance
    histogram toward the low end. The exposure_stats feature captures this
    as elevated near-zero pixel fraction.
    """
    gammas = {"low": 1.8, "medium": 2.8, "high": 4.5}
    gamma = gammas[severity]
    lut = np.array([((i / 255.0) ** gamma) * 255 for i in range(256)], dtype=np.uint8)
    return cv2.LUT(img, lut)


def apply_overexposure(img: np.ndarray, severity: str) -> np.ndarray:
    """Brighten image by applying gamma < 1, then clip highlights.

    Theory: gamma < 1 lifts shadow regions and causes highlight clipping.
    exposure_stats captures the near-255 pixel fraction spike.
    """
    gammas = {"low": 0.6, "medium": 0.4, "high": 0.2}
    gamma = gammas[severity]
    lut = np.array([min(255, int(((i / 255.0) ** gamma) * 255)) for i in range(256)], dtype=np.uint8)
    return cv2.LUT(img, lut)


def apply_noise(img: np.ndarray, severity: str, rng: np.random.Generator) -> np.ndarray:
    """Add Gaussian and salt-and-pepper noise at increasing intensity.

    Theory: Gaussian noise elevates high-frequency image energy and inflates
    the noise_estimate metric. Salt-and-pepper creates isolated extreme pixels
    detectable via local variance in smooth regions.
    """
    result = img.astype(np.float32)
    std_devs = {"low": 15.0, "medium": 40.0, "high": 80.0}
    sp_fractions = {"low": 0.02, "medium": 0.06, "high": 0.15}

    # Gaussian noise
    std = std_devs[severity]
    gaussian = rng.normal(0, std, result.shape).astype(np.float32)
    result += gaussian

    # Salt-and-pepper
    sp = sp_fractions[severity]
    h, w = result.shape[:2]
    n_salt = int(sp * h * w / 2)
    n_pepper = int(sp * h * w / 2)
    salt_y = rng.integers(0, h, n_salt)
    salt_x = rng.integers(0, w, n_salt)
    pepper_y = rng.integers(0, h, n_pepper)
    pepper_x = rng.integers(0, w, n_pepper)
    result[salt_y, salt_x] = 255
    result[pepper_y, pepper_x] = 0

    return np.clip(result, 0, 255).astype(np.uint8)


def apply_corruption(img: np.ndarray, severity: str, rng: np.random.Generator) -> np.ndarray:
    """Simulate JPEG re-compression artifacts at decreasing quality.

    Theory: JPEG uses DCT block quantisation; low quality settings produce
    8×8 block-boundary discontinuities detectable by block_artifact_score.
    """
    qualities = {"low": 30, "medium": 12, "high": 5}
    quality = qualities[severity]
    encode_params = [cv2.IMWRITE_JPEG_QUALITY, quality]
    _, encoded = cv2.imencode(".jpg", img, encode_params)
    corrupted = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    if corrupted is None:
        return img
    return corrupted


def apply_defect(img: np.ndarray, severity: str, rng: np.random.Generator) -> np.ndarray:
    """Overlay synthetic visual defects: scratches, blobs, and dark occlusions.

    Theory: Real-world visual defects (scratches, lens dirt, sensor damage)
    appear as high-contrast localised anomalies. We simulate these with
    random-geometry overlays at varying coverage fractions.
    """
    result = img.copy()
    h, w = result.shape[:2]

    n_scratches = {"low": 2, "medium": 6, "high": 15}[severity]
    n_blobs = {"low": 1, "medium": 3, "high": 7}[severity]
    blob_max_r = {"low": 8, "medium": 20, "high": 45}[severity]

    # Scratches: thin random lines
    for _ in range(n_scratches):
        x1, y1 = rng.integers(0, w), rng.integers(0, h)
        dx, dy = rng.integers(-w // 2, w // 2), rng.integers(-h // 2, h // 2)
        x2, y2 = np.clip(x1 + dx, 0, w - 1), np.clip(y1 + dy, 0, h - 1)
        color = (int(rng.integers(200, 255)),) * 3  # bright scratch
        cv2.line(result, (x1, y1), (x2, y2), color, thickness=1)

    # Blobs: dark occlusions (e.g., lens dirt)
    for _ in range(n_blobs):
        cx, cy = rng.integers(0, w), rng.integers(0, h)
        r = rng.integers(3, blob_max_r)
        alpha = rng.uniform(0.5, 1.0)
        overlay = result.copy()
        cv2.circle(overlay, (cx, cy), r, (0, 0, 0), -1)
        result = cv2.addWeighted(overlay, alpha, result, 1 - alpha, 0)

    return result


# ─────────────────────────────────────────────────────────────────────────────
# Dataset generation
# ─────────────────────────────────────────────────────────────────────────────


def load_source_images(max_images: int | None) -> np.ndarray:
    """Download CIFAR-100 via Keras and return upscaled RGB images (224×224)."""
    print("Loading CIFAR-100 source images (downloading if needed)...")
    # Lazy import to keep startup fast when only feature functions are imported
    import tensorflow as tf  # noqa: PLC0415

    (x_train, _), (x_test, _) = tf.keras.datasets.cifar100.load_data()
    source = np.concatenate([x_train, x_test], axis=0)  # 60k images

    if max_images is not None:
        source = source[:max_images]

    print(f"  Loaded {len(source)} source images (32×32 RGB). Upscaling to 224×224...")
    upscaled = np.array(
        [cv2.resize(img, (224, 224), interpolation=cv2.INTER_CUBIC) for img in source],
        dtype=np.uint8,
    )
    print("  Upscaling complete.")
    return upscaled


def split_indices(n: int, ratios: Dict[str, float], seed: int) -> Dict[str, List[int]]:
    """Return train/val/test index lists via non-overlapping random partition."""
    rng = random.Random(seed)
    indices = list(range(n))
    rng.shuffle(indices)
    n_train = int(n * ratios["train"])
    n_val = int(n * ratios["val"])
    return {
        "train": indices[:n_train],
        "val": indices[n_train : n_train + n_val],
        "test": indices[n_train + n_val :],
    }


def generate_split(
    images: np.ndarray,
    indices: List[int],
    split_name: str,
    output_dir: Path,
    seed: int,
) -> List[Dict]:
    """Generate all degraded + clean variants for a given split, return manifest rows."""
    rng = np.random.default_rng(seed)
    manifest_rows: List[Dict] = []

    img_dir = output_dir / "images" / split_name
    img_dir.mkdir(parents=True, exist_ok=True)

    for src_idx in indices:
        img = images[src_idx]
        is_clean = rng.random() < CLEAN_FRACTION

        if is_clean:
            # Save a single clean copy
            fname = f"{split_name}_{src_idx:05d}_clean.jpg"
            fpath = img_dir / fname
            cv2.imwrite(str(fpath), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
            manifest_rows.append(
                {
                    "filename": str(fpath.relative_to(output_dir)),
                    "split": split_name,
                    "source_id": src_idx,
                    "issue_type": "clean",
                    "severity": "none",
                }
            )
        else:
            for defect_type in [t for t in DEFECT_TYPES if t != "clean"]:
                for severity in SEVERITIES:
                    for _ in range(VARIANTS_PER_COMBO):
                        degraded = _degrade(img, defect_type, severity, rng)
                        fname = f"{split_name}_{src_idx:05d}_{defect_type}_{severity}.jpg"
                        fpath = img_dir / fname
                        cv2.imwrite(str(fpath), cv2.cvtColor(degraded, cv2.COLOR_RGB2BGR))
                        manifest_rows.append(
                            {
                                "filename": str(fpath.relative_to(output_dir)),
                                "split": split_name,
                                "source_id": src_idx,
                                "issue_type": defect_type,
                                "severity": severity,
                            }
                        )
    return manifest_rows


def _degrade(
    img: np.ndarray, defect_type: str, severity: str, rng: np.random.Generator
) -> np.ndarray:
    """Dispatch to the appropriate degradation function."""
    if defect_type == "blur":
        return apply_blur(img, severity)
    if defect_type == "underexposed":
        return apply_underexposure(img, severity)
    if defect_type == "overexposed":
        return apply_overexposure(img, severity)
    if defect_type == "noisy":
        return apply_noise(img, severity, rng)
    if defect_type == "corrupted":
        return apply_corruption(img, severity, rng)
    if defect_type == "defective":
        return apply_defect(img, severity, rng)
    return img  # fallback: clean


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────


def main(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    seed = args.seed
    max_images = args.max_images

    source_images = load_source_images(max_images)
    n = len(source_images)

    splits = split_indices(n, SPLIT_RATIOS, seed)
    print(f"Split sizes — train: {len(splits['train'])}, val: {len(splits['val'])}, test: {len(splits['test'])}")

    all_manifest_rows: List[Dict] = []
    for split_name, indices in splits.items():
        print(f"Generating {split_name} split ({len(indices)} source images)...")
        rows = generate_split(source_images, indices, split_name, output_dir, seed + hash(split_name) % 1000)
        all_manifest_rows.extend(rows)
        print(f"  {len(rows)} images generated for {split_name}.")

    # Save manifest as both JSON and CSV
    manifest_path = output_dir / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(all_manifest_rows, f, indent=2)

    csv_path = output_dir / "manifest.csv"
    fieldnames = ["filename", "split", "source_id", "issue_type", "severity"]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_manifest_rows)

    # Dataset statistics
    stats = {
        "total_images": len(all_manifest_rows),
        "split_counts": {s: len(i) for s, i in splits.items()},
        "class_distribution": {},
    }
    for row in all_manifest_rows:
        key = f"{row['issue_type']}_{row['severity']}"
        stats["class_distribution"][key] = stats["class_distribution"].get(key, 0) + 1

    stats_path = output_dir / "dataset_stats.json"
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)

    print(f"\nDataset generation complete.")
    print(f"  Total images : {len(all_manifest_rows)}")
    print(f"  Manifest     : {manifest_path}")
    print(f"  Stats        : {stats_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate LensGuard synthetic dataset")
    parser.add_argument("--output-dir", default="data/", help="Output directory for dataset")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--max-images",
        type=int,
        default=MAX_SOURCE_IMAGES,
        help="Max source images to use (None = all 60k)",
    )
    main(parser.parse_args())
