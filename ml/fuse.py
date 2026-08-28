"""
LensGuard Quality Score Fusion
================================
Combines classical CV feature signals with calibrated model probabilities to
produce a single quality score (0–100), a quality label, and a ranked list of
detected issues.

See ml/FUSION.md for a detailed description of the weighting formula and
design rationale.
"""

from __future__ import annotations

import numpy as np

from ml.calibrate import apply_temperature_scaling


# ─────────────────────────────────────────────────────────────────────────────
# Thresholds & constants
# ─────────────────────────────────────────────────────────────────────────────

LABEL_THRESHOLDS = {
    "ACCEPTABLE": 75,   # score ≥ 75
    "DEGRADED": 45,     # 45 ≤ score < 75
    # "DEFECTIVE"         # score < 45
}

CLASSES = ["clean", "blur", "underexposed", "overexposed", "noisy", "corrupted", "defective"]
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}

# Classical feature penalty weights — how much each feature contributes to the
# overall penalty signal. Must sum to 1.0 across all feature dimensions used.
FEATURE_WEIGHTS = {
    "blur":         0.30,  # laplacian_variance is the dominant blur signal
    "underexposed": 0.15,  # mean_luminance + pct_near_black
    "overexposed":  0.15,  # mean_luminance + pct_near_white
    "noisy":        0.20,  # noise_estimate
    "corrupted":    0.10,  # block_artifact_score
    "defective":    0.10,  # saturation anomaly + contrast anomaly
}

# Blend ratio: classical features vs. model probabilities
CLASSICAL_WEIGHT = 0.60
MODEL_WEIGHT = 0.40


# ─────────────────────────────────────────────────────────────────────────────
# Classical feature → per-defect severity maps
# ─────────────────────────────────────────────────────────────────────────────


def _blur_severity(laplacian_variance: float) -> tuple[float, str]:
    """Map Laplacian variance to blur penalty [0,1] and severity label.

    Calibrated against synthetic dataset ranges:
      sharp (224×224 natural photo): LV ~200–2000+
      low blur  (kernel 5):  LV ~50–200
      med blur  (kernel 15): LV ~5–50
      high blur (kernel 31): LV ~0.5–5
    """
    if laplacian_variance >= 150:
        return 0.0, "none"
    elif laplacian_variance >= 50:
        return 0.20, "low"
    elif laplacian_variance >= 10:
        return 0.55, "medium"
    else:
        return 0.90, "high"


def _underexposure_severity(mean_lum: float, pct_near_black: float) -> tuple[float, str]:
    """Map luminance stats to underexposure penalty [0,1]."""
    score = 0.0
    if mean_lum < 40:
        score += 0.6
    elif mean_lum < 80:
        score += 0.3
    if pct_near_black > 0.20:
        score += 0.3
    elif pct_near_black > 0.08:
        score += 0.15
    score = min(score, 1.0)
    if score < 0.15:
        return 0.0, "none"
    elif score < 0.40:
        return score, "low"
    elif score < 0.70:
        return score, "medium"
    else:
        return score, "high"


def _overexposure_severity(mean_lum: float, pct_near_white: float) -> tuple[float, str]:
    """Map luminance stats to overexposure penalty [0,1]."""
    score = 0.0
    if mean_lum > 210:
        score += 0.6
    elif mean_lum > 175:
        score += 0.3
    if pct_near_white > 0.20:
        score += 0.3
    elif pct_near_white > 0.08:
        score += 0.15
    score = min(score, 1.0)
    if score < 0.15:
        return 0.0, "none"
    elif score < 0.40:
        return score, "low"
    elif score < 0.70:
        return score, "medium"
    else:
        return score, "high"


def _noise_severity(noise_estimate: float) -> tuple[float, str]:
    """Map noise estimate to penalty [0,1]."""
    if noise_estimate < 5:
        return 0.0, "none"
    elif noise_estimate < 15:
        return 0.20, "low"
    elif noise_estimate < 35:
        return 0.55, "medium"
    else:
        return 0.90, "high"


def _corruption_severity(block_artifact_score: float) -> tuple[float, str]:
    """Map block artifact score to penalty [0,1]."""
    if block_artifact_score < 1.3:
        return 0.0, "none"
    elif block_artifact_score < 1.8:
        return 0.25, "low"
    elif block_artifact_score < 2.5:
        return 0.60, "medium"
    else:
        return 0.90, "high"


def _defect_severity(contrast_rms: float, pct_low_sat: float) -> tuple[float, str]:
    """Heuristic for visible defect severity from contrast/saturation anomalies."""
    # Synthetic defects (scratches, blobs) tend to create localised saturation
    # drops and contrast increases. This heuristic is intentionally weak —
    # the model probability carries most of the defect signal.
    score = 0.0
    if contrast_rms > 0.45:
        score += 0.2
    if pct_low_sat > 0.60:
        score += 0.2
    if score < 0.15:
        return 0.0, "none"
    elif score < 0.30:
        return score, "low"
    elif score < 0.55:
        return score, "medium"
    else:
        return score, "high"


# ─────────────────────────────────────────────────────────────────────────────
# Main fusion function
# ─────────────────────────────────────────────────────────────────────────────


def fuse(
    features: dict[str, float],
    logits: np.ndarray,
    temperature: float = 1.0,
) -> dict:
    """Combine classical features and model logits into a quality assessment.

    Parameters
    ----------
    features : dict[str, float]
        Output of ml.features.extract_all (flat dict of CV metrics).
    logits : np.ndarray of shape (num_classes,)
        Raw model output logits (before softmax).
    temperature : float
        Calibration temperature from metadata.json. Applied before softmax.

    Returns
    -------
    dict with keys:
        quality_score  : int (0–100)
        quality_label  : str ("ACCEPTABLE" | "DEGRADED" | "DEFECTIVE")
        issues         : list of {type, severity, confidence}
        primary_class  : str — top model class
        class_probs    : dict[str, float] — calibrated class probabilities
    """
    # ── Step 1: Calibrated model probabilities ────────────────────────────
    probs = apply_temperature_scaling(logits, temperature)  # shape (num_classes,)
    class_probs = {cls: float(probs[i]) for i, cls in enumerate(CLASSES)}
    primary_class = CLASSES[int(np.argmax(probs))]

    # ── Step 2: Classical feature penalties ──────────────────────────────
    blur_pen, blur_sev = _blur_severity(features.get("laplacian_variance", 999))
    under_pen, under_sev = _underexposure_severity(
        features.get("mean_luminance", 128),
        features.get("pct_near_black", 0),
    )
    over_pen, over_sev = _overexposure_severity(
        features.get("mean_luminance", 128),
        features.get("pct_near_white", 0),
    )
    noise_pen, noise_sev = _noise_severity(features.get("noise_estimate", 0))
    corrupt_pen, corrupt_sev = _corruption_severity(features.get("block_artifact_score", 1.0))
    defect_pen, defect_sev = _defect_severity(
        features.get("contrast_rms", 0),
        features.get("pct_low_saturation", 0),
    )

    # Weighted sum of classical penalties (each in [0,1])
    classical_penalty = (
        FEATURE_WEIGHTS["blur"]         * blur_pen
        + FEATURE_WEIGHTS["underexposed"] * under_pen
        + FEATURE_WEIGHTS["overexposed"]  * over_pen
        + FEATURE_WEIGHTS["noisy"]        * noise_pen
        + FEATURE_WEIGHTS["corrupted"]    * corrupt_pen
        + FEATURE_WEIGHTS["defective"]    * defect_pen
    )  # range [0, 1]

    # ── Step 3: Model penalty — prob mass on non-clean classes ───────────
    model_penalty = float(1.0 - class_probs.get("clean", 0.0))  # range [0, 1]

    # ── Step 4: Blended penalty ───────────────────────────────────────────
    raw_penalty = CLASSICAL_WEIGHT * classical_penalty + MODEL_WEIGHT * model_penalty
    raw_penalty = float(np.clip(raw_penalty, 0.0, 1.0))

    # ── Step 5: Quality score (0–100, higher = better) ───────────────────
    quality_score = int(round(100 * (1.0 - raw_penalty)))

    # ── Step 6: Quality label ─────────────────────────────────────────────
    if quality_score >= LABEL_THRESHOLDS["ACCEPTABLE"]:
        quality_label = "ACCEPTABLE"
    elif quality_score >= LABEL_THRESHOLDS["DEGRADED"]:
        quality_label = "DEGRADED"
    else:
        quality_label = "DEFECTIVE"

    # ── Step 7: Issue list — only report non-trivial issues ───────────────
    issue_map = [
        ("blur",        blur_sev,    blur_pen,    class_probs.get("blur", 0)),
        ("underexposed",under_sev,   under_pen,   class_probs.get("underexposed", 0)),
        ("overexposed", over_sev,    over_pen,    class_probs.get("overexposed", 0)),
        ("noise",       noise_sev,   noise_pen,   class_probs.get("noisy", 0)),
        ("corruption",  corrupt_sev, corrupt_pen, class_probs.get("corrupted", 0)),
        ("defect",      defect_sev,  defect_pen,  class_probs.get("defective", 0)),
    ]

    issues = []
    for issue_type, severity, classical_conf, model_conf in issue_map:
        if severity == "none":
            continue
        # Confidence = blend of classical feature signal strength and model prob
        confidence = float(0.5 * classical_conf + 0.5 * model_conf)
        issues.append(
            {
                "type": issue_type,
                "severity": severity,
                "confidence": round(confidence, 4),
            }
        )

    # Sort issues by confidence descending
    issues.sort(key=lambda x: x["confidence"], reverse=True)

    return {
        "quality_score": quality_score,
        "quality_label": quality_label,
        "issues": issues,
        "primary_class": primary_class,
        "class_probs": class_probs,
    }


def score_to_label(score: int) -> str:
    """Map a quality score to its label string."""
    if score >= LABEL_THRESHOLDS["ACCEPTABLE"]:
        return "ACCEPTABLE"
    elif score >= LABEL_THRESHOLDS["DEGRADED"]:
        return "DEGRADED"
    return "DEFECTIVE"
