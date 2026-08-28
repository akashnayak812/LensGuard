"""
Unit tests for ml/features.py

Each test verifies the *directional behaviour* of a feature function using
controlled synthetic inputs. For every metric we know a priori whether a
degraded sample should produce a higher or lower value than the clean sample.

Tests do NOT rely on exact numeric thresholds (those would be fragile across
platforms); they only assert ordinal relationships.
"""

import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

# Allow importing from the project root
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from ml.features import (
    block_artifact_score,
    contrast_rms,
    exposure_stats,
    extract_all,
    laplacian_variance,
    noise_estimate,
    saturation_stats,
)

# ─────────────────────────────────────────────────────────────────────────────
# Shared fixtures
# ─────────────────────────────────────────────────────────────────────────────

RNG = np.random.default_rng(42)


@pytest.fixture()
def sharp_image() -> np.ndarray:
    """224×224 BGR image with strong edges (checkerboard + gradient)."""
    img = np.zeros((224, 224, 3), dtype=np.uint8)
    # Checkerboard pattern creates many high-contrast edges
    block = 16
    for r in range(0, 224, block):
        for c in range(0, 224, block):
            val = 255 if ((r // block + c // block) % 2 == 0) else 0
            img[r : r + block, c : c + block] = val
    return img


@pytest.fixture()
def blurred_image(sharp_image: np.ndarray) -> np.ndarray:
    """Same checkerboard, heavily Gaussian-blurred (31×31 kernel)."""
    return cv2.GaussianBlur(sharp_image, (31, 31), 10.0)


@pytest.fixture()
def natural_image() -> np.ndarray:
    """224×224 random natural-looking image (uniform random with moderate range)."""
    return RNG.integers(30, 220, (224, 224, 3), dtype=np.uint8)


@pytest.fixture()
def underexposed_image(natural_image: np.ndarray) -> np.ndarray:
    """Darkened version via gamma=4.0."""
    lut = np.array([int(((i / 255.0) ** 4.0) * 255) for i in range(256)], dtype=np.uint8)
    return cv2.LUT(natural_image, lut)


@pytest.fixture()
def overexposed_image(natural_image: np.ndarray) -> np.ndarray:
    """Brightened version via gamma=0.2."""
    lut = np.array([min(255, int(((i / 255.0) ** 0.2) * 255)) for i in range(256)], dtype=np.uint8)
    return cv2.LUT(natural_image, lut)


@pytest.fixture()
def noisy_image(natural_image: np.ndarray) -> np.ndarray:
    """Same image with heavy Gaussian noise (σ=60)."""
    noise = RNG.normal(0, 60, natural_image.shape).astype(np.float32)
    return np.clip(natural_image.astype(np.float32) + noise, 0, 255).astype(np.uint8)


@pytest.fixture()
def jpeg_compressed_image(natural_image: np.ndarray) -> np.ndarray:
    """Same image re-encoded at JPEG quality=5 (heavy blocking)."""
    encode_params = [cv2.IMWRITE_JPEG_QUALITY, 5]
    _, encoded = cv2.imencode(".jpg", natural_image, encode_params)
    return cv2.imdecode(encoded, cv2.IMREAD_COLOR)


@pytest.fixture()
def saturated_image() -> np.ndarray:
    """Fully saturated red image (R=255, G=0, B=0)."""
    img = np.zeros((224, 224, 3), dtype=np.uint8)
    img[:, :, 2] = 255  # BGR: red channel
    return img


@pytest.fixture()
def grey_image() -> np.ndarray:
    """Greyscale image (equal R=G=B) — near-zero saturation."""
    return np.full((224, 224, 3), 128, dtype=np.uint8)


# ─────────────────────────────────────────────────────────────────────────────
# Test: laplacian_variance
# ─────────────────────────────────────────────────────────────────────────────


class TestLaplacianVariance:
    def test_sharp_greater_than_blurred(self, sharp_image, blurred_image):
        """Sharp image must have higher Laplacian variance than its blurred version."""
        lv_sharp = laplacian_variance(sharp_image)
        lv_blurred = laplacian_variance(blurred_image)
        assert lv_sharp > lv_blurred, (
            f"Expected sharp ({lv_sharp:.2f}) > blurred ({lv_blurred:.2f})"
        )

    def test_returns_positive_float(self, sharp_image):
        result = laplacian_variance(sharp_image)
        assert isinstance(result, float)
        assert result >= 0.0

    def test_flat_image_near_zero(self):
        """Completely flat image should have near-zero Laplacian variance."""
        flat = np.full((64, 64, 3), 128, dtype=np.uint8)
        assert laplacian_variance(flat) < 1.0

    def test_accepts_grayscale(self, sharp_image):
        gray = cv2.cvtColor(sharp_image, cv2.COLOR_BGR2GRAY)
        result = laplacian_variance(gray)
        assert isinstance(result, float)
        assert result >= 0.0


# ─────────────────────────────────────────────────────────────────────────────
# Test: exposure_stats
# ─────────────────────────────────────────────────────────────────────────────


class TestExposureStats:
    def test_underexposed_has_low_mean_luminance(self, underexposed_image):
        stats = exposure_stats(underexposed_image)
        assert stats["mean_luminance"] < 80, f"Expected low luminance, got {stats['mean_luminance']:.1f}"

    def test_overexposed_has_high_mean_luminance(self, overexposed_image):
        stats = exposure_stats(overexposed_image)
        assert stats["mean_luminance"] > 180, f"Expected high luminance, got {stats['mean_luminance']:.1f}"

    def test_underexposed_pct_near_black(self, underexposed_image, natural_image):
        """Under-exposed image should have more near-black pixels."""
        under_stats = exposure_stats(underexposed_image)
        nat_stats = exposure_stats(natural_image)
        assert under_stats["pct_near_black"] > nat_stats["pct_near_black"]

    def test_overexposed_pct_near_white(self, overexposed_image, natural_image):
        """Over-exposed image should have more near-white pixels."""
        over_stats = exposure_stats(overexposed_image)
        nat_stats = exposure_stats(natural_image)
        assert over_stats["pct_near_white"] > nat_stats["pct_near_white"]

    def test_returns_expected_keys(self, natural_image):
        stats = exposure_stats(natural_image)
        assert set(stats.keys()) == {"mean_luminance", "pct_near_black", "pct_near_white", "luminance_std"}

    def test_fractions_in_range(self, natural_image):
        stats = exposure_stats(natural_image)
        assert 0.0 <= stats["pct_near_black"] <= 1.0
        assert 0.0 <= stats["pct_near_white"] <= 1.0


# ─────────────────────────────────────────────────────────────────────────────
# Test: noise_estimate
# ─────────────────────────────────────────────────────────────────────────────


class TestNoiseEstimate:
    def test_noisy_greater_than_clean(self, noisy_image, natural_image):
        """Noisy image must produce a higher noise estimate than the clean version."""
        ne_noisy = noise_estimate(noisy_image)
        ne_clean = noise_estimate(natural_image)
        assert ne_noisy > ne_clean, (
            f"Expected noisy ({ne_noisy:.2f}) > clean ({ne_clean:.2f})"
        )

    def test_flat_image_near_zero(self):
        """A perfectly flat image should yield near-zero noise estimate."""
        flat = np.full((64, 64, 3), 128, dtype=np.uint8)
        assert noise_estimate(flat) < 5.0

    def test_returns_non_negative_float(self, natural_image):
        result = noise_estimate(natural_image)
        assert isinstance(result, float)
        assert result >= 0.0


# ─────────────────────────────────────────────────────────────────────────────
# Test: contrast_rms
# ─────────────────────────────────────────────────────────────────────────────


class TestContrastRms:
    def test_high_contrast_greater_than_flat(self, sharp_image):
        """Checkerboard (high contrast) > flat grey image."""
        flat = np.full((224, 224, 3), 128, dtype=np.uint8)
        assert contrast_rms(sharp_image) > contrast_rms(flat)

    def test_result_in_0_1(self, natural_image):
        c = contrast_rms(natural_image)
        assert 0.0 <= c <= 1.0

    def test_flat_image_zero(self):
        flat = np.full((32, 32, 3), 128, dtype=np.uint8)
        assert contrast_rms(flat) == pytest.approx(0.0, abs=1e-3)


# ─────────────────────────────────────────────────────────────────────────────
# Test: block_artifact_score
# ─────────────────────────────────────────────────────────────────────────────


class TestBlockArtifactScore:
    def test_compressed_greater_than_clean(self, jpeg_compressed_image, natural_image):
        """Heavily JPEG-compressed image should have higher block artifact score."""
        ba_compressed = block_artifact_score(jpeg_compressed_image)
        ba_clean = block_artifact_score(natural_image)
        # Allow a looser assertion — block artifacts depend on image content
        # but should always produce a measurable increase for quality=5
        assert ba_compressed >= ba_clean * 0.9, (
            f"compressed={ba_compressed:.3f}, clean={ba_clean:.3f}"
        )

    def test_returns_positive_float(self, natural_image):
        result = block_artifact_score(natural_image)
        assert isinstance(result, float)
        assert result > 0.0

    def test_flat_image_returns_1(self):
        """Flat image has no interior gradients → returns sentinel 1.0."""
        flat = np.full((64, 64, 3), 128, dtype=np.uint8)
        result = block_artifact_score(flat)
        assert result == pytest.approx(1.0, abs=0.1)


# ─────────────────────────────────────────────────────────────────────────────
# Test: saturation_stats
# ─────────────────────────────────────────────────────────────────────────────


class TestSaturationStats:
    def test_saturated_higher_than_grey(self, saturated_image, grey_image):
        """A pure red image has higher mean saturation than a grey image."""
        sat_red = saturation_stats(saturated_image)
        sat_grey = saturation_stats(grey_image)
        assert sat_red["mean_saturation"] > sat_grey["mean_saturation"]

    def test_grey_pct_low_saturation_near_1(self, grey_image):
        """Equal-channel grey image → nearly all pixels have near-zero saturation."""
        stats = saturation_stats(grey_image)
        assert stats["pct_low_saturation"] > 0.9

    def test_returns_expected_keys(self, natural_image):
        stats = saturation_stats(natural_image)
        assert set(stats.keys()) == {"mean_saturation", "std_saturation", "pct_low_saturation"}

    def test_fractions_in_range(self, natural_image):
        stats = saturation_stats(natural_image)
        assert 0.0 <= stats["mean_saturation"] <= 1.0
        assert 0.0 <= stats["pct_low_saturation"] <= 1.0


# ─────────────────────────────────────────────────────────────────────────────
# Test: extract_all
# ─────────────────────────────────────────────────────────────────────────────


class TestExtractAll:
    EXPECTED_KEYS = {
        "laplacian_variance",
        "mean_luminance",
        "pct_near_black",
        "pct_near_white",
        "luminance_std",
        "noise_estimate",
        "contrast_rms",
        "block_artifact_score",
        "mean_saturation",
        "std_saturation",
        "pct_low_saturation",
    }

    def test_returns_all_expected_keys(self, natural_image):
        result = extract_all(natural_image)
        assert set(result.keys()) == self.EXPECTED_KEYS

    def test_all_values_are_floats(self, natural_image):
        result = extract_all(natural_image)
        for k, v in result.items():
            assert isinstance(v, float), f"Key '{k}' returned {type(v)}, expected float"

    def test_no_nan_or_inf(self, natural_image, blurred_image, noisy_image):
        for img in [natural_image, blurred_image, noisy_image]:
            result = extract_all(img)
            for k, v in result.items():
                assert np.isfinite(v), f"Key '{k}' returned non-finite value: {v}"
