"""
LensGuard Classical Feature Extraction
=======================================
Pure-function CV feature extractors used both during training (to generate
feature vectors) and at inference time (to provide interpretable defect signals).

Every function:
  - Accepts a uint8 or float32 HxW or HxWxC numpy array.
  - Returns a float or dict of floats (no side effects).
  - Includes a docstring explaining the CV theory behind the metric.

These classical features feed the fusion layer (ml/fuse.py) alongside the
calibrated model output, giving the quality score an interpretable, auditable
component that does not depend on opaque neural activations.
"""

from __future__ import annotations

import cv2
import numpy as np
import pywt


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────


def _to_gray_float(img: np.ndarray) -> np.ndarray:
    """Convert any input image to a float32 grayscale array in [0, 255]."""
    if img.ndim == 3:
        gray = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_BGR2GRAY).astype(np.float32)
    else:
        gray = img.astype(np.float32)
    return gray


def _to_gray_uint8(img: np.ndarray) -> np.ndarray:
    """Convert any input image to a uint8 grayscale array."""
    if img.ndim == 3:
        return cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_BGR2GRAY)
    return np.clip(img, 0, 255).astype(np.uint8)


def _to_bgr_uint8(img: np.ndarray) -> np.ndarray:
    """Ensure a 3-channel uint8 BGR image for colour-space conversions."""
    if img.ndim == 2:
        return cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_GRAY2BGR)
    return img.astype(np.uint8)


# ─────────────────────────────────────────────────────────────────────────────
# Feature functions
# ─────────────────────────────────────────────────────────────────────────────


def laplacian_variance(img: np.ndarray) -> float:
    """Measure image sharpness via the variance of the Laplacian operator.

    Theory
    ------
    The Laplacian is the second spatial derivative of image intensity. Sharp
    images contain strong, localised high-frequency transitions (edges); blurred
    images have these transitions smoothed out. The *variance* of the Laplacian
    response summarises global sharpness in a single scalar: a perfectly
    homogeneous (or perfectly blurred) image yields near-zero variance, while a
    sharp, detail-rich image yields high variance.

    Reference: Pech-Pacheco et al., "Diatom autofocusing in brightfield
    microscopy", ICPR 2000.

    Parameters
    ----------
    img : np.ndarray
        Input image (uint8 or float32), grayscale or BGR.

    Returns
    -------
    float
        Variance of the Laplacian. Higher → sharper. Typical values:
        sharp photo ~100–2000, moderate blur ~10–100, heavy blur <10.
    """
    # OpenCV 5+ requires uint8 source when using CV_64F destination.
    # We convert to uint8 first and cast the Laplacian result to float64.
    gray_u8 = _to_gray_uint8(img)
    laplacian = cv2.Laplacian(gray_u8, cv2.CV_64F)
    return float(np.var(laplacian))


def exposure_stats(img: np.ndarray) -> dict[str, float]:
    """Compute luminance statistics to detect under- and over-exposure.

    Theory
    ------
    A well-exposed image has a luminance histogram that is roughly centred and
    does not heavily clip at either extreme. We characterise exposure via:

    - ``mean_luminance``: mean pixel value of the grayscale image (0–255).
      Strongly under-exposed images have mean < 40; over-exposed > 210.

    - ``pct_near_black``: fraction of pixels with luminance ≤ 10/255. A spike
      here indicates shadow clipping / under-exposure.

    - ``pct_near_white``: fraction of pixels with luminance ≥ 245/255. A spike
      here indicates highlight clipping / over-exposure.

    - ``luminance_std``: standard deviation of the luminance channel; very low
      values combined with extreme mean_luminance indicate flat over/under
      exposure rather than natural scene variation.

    Parameters
    ----------
    img : np.ndarray
        Input image (uint8 or float32), grayscale or BGR.

    Returns
    -------
    dict with keys: mean_luminance, pct_near_black, pct_near_white,
                    luminance_std
    """
    gray = _to_gray_float(img)
    n_pixels = gray.size
    mean_lum = float(np.mean(gray))
    std_lum = float(np.std(gray))
    pct_near_black = float(np.sum(gray <= 10) / n_pixels)
    pct_near_white = float(np.sum(gray >= 245) / n_pixels)
    return {
        "mean_luminance": mean_lum,
        "pct_near_black": pct_near_black,
        "pct_near_white": pct_near_white,
        "luminance_std": std_lum,
    }


def noise_estimate(img: np.ndarray) -> float:
    """Estimate image noise level via high-frequency wavelet energy in smooth regions.

    Theory
    ------
    Pure noise manifests as high-frequency energy broadly distributed across the
    image. We use a two-stage approach:

    1. **Discrete Wavelet Transform**: decompose the grayscale image with a
       Haar wavelet (one level). The diagonal detail sub-band (HH) captures
       high-frequency energy in both spatial directions. Noise inflates this
       sub-band.

    2. **Flat-region masking**: to avoid mistaking genuine high-contrast edges
       for noise, we restrict our estimate to *smooth* regions (Laplacian
       magnitude below the 30th percentile). The standard deviation of the HH
       sub-band in these regions isolates noise from texture.

    The returned value is the MAD-normalised noise estimate, analogous to
    Donoho's universal threshold σ = median(|HH|) / 0.6745.

    Parameters
    ----------
    img : np.ndarray
        Input image (uint8 or float32), grayscale or BGR.

    Returns
    -------
    float
        Estimated noise standard deviation. Higher → noisier.
        Clean images: ~1–8. Noisy images: 10–60+.
    """
    gray = _to_gray_float(img)

    # Wavelet decomposition
    _, (cH, cV, cD) = pywt.dwt2(gray, "haar")  # type: ignore[misc]
    hh = cD  # diagonal detail sub-band

    # Donoho estimator
    sigma = np.median(np.abs(hh)) / 0.6745
    return float(sigma)


def contrast_rms(img: np.ndarray) -> float:
    """Compute RMS (root-mean-square) contrast of the image.

    Theory
    ------
    RMS contrast is defined as the standard deviation of the normalised pixel
    intensities. Unlike Michelson or Weber contrast (which are ratio-based and
    sensitive to global min/max), RMS contrast reflects the *distribution* of
    intensity variation across the entire image and is robust to isolated
    outlier pixels.

    Formula:  C_rms = std(I) / 255
    where I is the grayscale image in [0, 255].

    A low RMS contrast indicates a low-dynamic-range, flat image (typical of
    foggy, over-exposed, or washed-out captures). A very high RMS contrast
    may indicate noise or artificial posterisation.

    Parameters
    ----------
    img : np.ndarray
        Input image (uint8 or float32), grayscale or BGR.

    Returns
    -------
    float
        RMS contrast in [0, 1]. Well-exposed natural image: ~0.15–0.40.
    """
    gray = _to_gray_float(img)
    return float(np.std(gray) / 255.0)


def block_artifact_score(img: np.ndarray) -> float:
    """Detect JPEG block-boundary discontinuities as a compression-artifact proxy.

    Theory
    ------
    JPEG compression divides the image into 8×8 DCT blocks and quantises
    coefficients independently per block. At low quality settings this
    introduces step discontinuities at block boundaries that do not occur in
    natural images.

    We measure this by comparing, for every pair of adjacent 8×8-block rows
    and columns, the mean absolute difference *across* the boundary versus the
    mean absolute difference *within* the block interior. The ratio
    (boundary / interior) > 1 indicates blocking; extreme ratios indicate
    heavy compression.

    Parameters
    ----------
    img : np.ndarray
        Input image (uint8 or float32), grayscale or BGR.

    Returns
    -------
    float
        Block artifact ratio. Clean or lossless image: ~0.8–1.2.
        Heavily JPEG-compressed: can exceed 3.0.
    """
    gray = _to_gray_float(img)
    h, w = gray.shape

    boundary_diffs: list[float] = []
    interior_diffs: list[float] = []

    # Horizontal block boundaries (every 8 rows)
    for row in range(8, h, 8):
        if row >= h:
            break
        bdiff = np.mean(np.abs(gray[row, :].astype(float) - gray[row - 1, :].astype(float)))
        if row + 1 < h and row - 2 >= 0:
            idiff = np.mean(np.abs(gray[row - 1, :].astype(float) - gray[row - 2, :].astype(float)))
            boundary_diffs.append(float(bdiff))
            interior_diffs.append(float(idiff))

    # Vertical block boundaries (every 8 columns)
    for col in range(8, w, 8):
        if col >= w:
            break
        bdiff = np.mean(np.abs(gray[:, col].astype(float) - gray[:, col - 1].astype(float)))
        if col + 1 < w and col - 2 >= 0:
            idiff = np.mean(np.abs(gray[:, col - 1].astype(float) - gray[:, col - 2].astype(float)))
            boundary_diffs.append(float(bdiff))
            interior_diffs.append(float(idiff))

    if not interior_diffs or np.mean(interior_diffs) < 1e-6:
        return 1.0  # avoid divide-by-zero on flat images

    return float(np.mean(boundary_diffs) / np.mean(interior_diffs))


def saturation_stats(img: np.ndarray) -> dict[str, float]:
    """Compute HSV saturation statistics as a secondary quality signal.

    Theory
    ------
    Saturation measures the colourfulness of an image in the HSV colour space.
    Healthy, well-rendered images typically have a broad saturation distribution.
    Pathological conditions affect saturation:
    - Over-exposed images tend toward washed-out, near-zero saturation.
    - Heavily JPEG-compressed images show irregular saturation artefacts.
    - Defective regions (scratches, blobs) often have anomalous saturation.

    Returns
    -------
    dict with keys: mean_saturation, std_saturation, pct_low_saturation
        (fraction of pixels with saturation < 0.1, i.e., near-greyscale)
    """
    bgr = _to_bgr_uint8(img)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV).astype(np.float32)
    sat = hsv[:, :, 1] / 255.0  # normalise to [0, 1]

    return {
        "mean_saturation": float(np.mean(sat)),
        "std_saturation": float(np.std(sat)),
        "pct_low_saturation": float(np.mean(sat < 0.1)),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Convenience: extract all features at once
# ─────────────────────────────────────────────────────────────────────────────


def extract_all(img: np.ndarray) -> dict[str, float]:
    """Run all feature extractors on an image and return a flat dict of floats.

    This is the primary entry point called by the inference pipeline
    (backend/app/services/pipeline.py). The returned dict maps cleanly onto
    the ``image_stats`` field in the API response schema.

    Parameters
    ----------
    img : np.ndarray
        Input image, uint8, HxWxC (BGR or RGB accepted — we internally
        convert as needed).

    Returns
    -------
    dict[str, float]
        Flat mapping of feature name → float value. Keys:
        laplacian_variance, mean_luminance, pct_near_black, pct_near_white,
        luminance_std, noise_estimate, contrast_rms, block_artifact_score,
        mean_saturation, std_saturation, pct_low_saturation
    """
    lv = laplacian_variance(img)
    exp = exposure_stats(img)
    ne = noise_estimate(img)
    cr = contrast_rms(img)
    ba = block_artifact_score(img)
    ss = saturation_stats(img)

    return {
        "laplacian_variance": lv,
        "mean_luminance": exp["mean_luminance"],
        "pct_near_black": exp["pct_near_black"],
        "pct_near_white": exp["pct_near_white"],
        "luminance_std": exp["luminance_std"],
        "noise_estimate": ne,
        "contrast_rms": cr,
        "block_artifact_score": ba,
        "mean_saturation": ss["mean_saturation"],
        "std_saturation": ss["std_saturation"],
        "pct_low_saturation": ss["pct_low_saturation"],
    }
