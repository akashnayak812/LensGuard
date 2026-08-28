# LensGuard Quality Score Fusion Formula

## Overview

The fusion layer combines two complementary signals to produce a single
interpretable quality score for each image:

1. **Classical CV features** — deterministic, auditable functions of image
   statistics (blur, exposure, noise, JPEG blocking, saturation).
2. **Calibrated model probabilities** — learned from 34,000+ synthetic
   training samples, temperature-scaled for reliable confidence estimates.

Neither signal alone is sufficient:
- Classical features are interpretable and fast but can miss complex, learned
  defect patterns.
- The model captures nuanced patterns but can be overconfident and lacks
  per-pixel locality without Grad-CAM.

Combining them produces a score that is both reliable and auditable.

---

## Step-by-Step Formula

### Step 1 — Calibrated Model Probabilities

Raw logits `z ∈ ℝ^C` from the MobileNetV2 head are divided by the learned
temperature `T*` before softmax:

```
p_calibrated(y | x) = softmax(z / T*)
```

`T*` is fitted by minimising NLL on the validation set (see `ml/calibrate.py`).
Values of `T* > 1` soften overconfident distributions; `T* ≈ 1` means the
model is already well-calibrated.

### Step 2 — Classical Feature Penalties

Each defect type maps to one or more feature functions → severity label →
penalty value in [0, 1]:

| Defect | Feature(s) used | Thresholds |
|---|---|---|
| **blur** | `laplacian_variance` | ≥150 → none; ≥50 → low (0.20); ≥10 → medium (0.55); <10 → high (0.90) |
| **underexposed** | `mean_luminance` + `pct_near_black` | Combined additive scoring |
| **overexposed** | `mean_luminance` + `pct_near_white` | Combined additive scoring |
| **noisy** | `noise_estimate` | <5 → none; <15 → low (0.20); <35 → medium (0.55); ≥35 → high (0.90) |
| **corrupted** | `block_artifact_score` | <1.3 → none; <1.8 → low (0.25); <2.5 → medium (0.60); ≥2.5 → high (0.90) |
| **defective** | `contrast_rms` + `pct_low_saturation` | Heuristic (model signal dominates here) |

These individual penalties are combined via a **weighted sum**:

```
classical_penalty = Σ_i  w_i × penalty_i

where weights w_i are:
  blur:         0.30
  underexposed: 0.15
  overexposed:  0.15
  noisy:        0.20
  corrupted:    0.10
  defective:    0.10
  ──────────────────
  Total:        1.00
```

**Rationale for weights**: blur and noise are the most common real-world
photographic defects and have robust classical detectors; their higher weights
reflect reliability. Exposure defects split the remaining budget equally.
Corruption and visible defects share the rest — compression artifacts are
detectable but rarer in practice; visible defects lean more on the model.

### Step 3 — Model Penalty

The model penalty is the probability mass on *all non-clean classes*:

```
model_penalty = 1 − p_calibrated("clean" | x)
```

This is a scalar in [0, 1]. A model confident the image is clean contributes
zero penalty; a model confident it is defective contributes near-1 penalty.

### Step 4 — Blended Penalty

```
raw_penalty = 0.60 × classical_penalty + 0.40 × model_penalty
```

The 60/40 split reflects our design priority: **interpretability over
raw model confidence**. Classical features are fully auditable and have
known, documented failure modes. The model provides a learned cross-check.
The blend is clipped to [0, 1].

**Sensitivity analysis**: changing the split to 70/30 increases the
contribution of classical features (more predictable, less adaptive); 50/50
gives more weight to the model (better on novel defect patterns). 60/40
was chosen as the default based on validation-set performance.

### Step 5 — Quality Score

```
quality_score = round(100 × (1 − raw_penalty))
```

Range: 0 (worst) to 100 (best). This linear mapping is intentional: a
10-point change in score always corresponds to the same absolute change in
penalty, making it easy to reason about threshold effects.

### Step 6 — Quality Label

```
quality_label = ACCEPTABLE   if quality_score ≥ 75
              | DEGRADED      if 45 ≤ quality_score < 75
              | DEFECTIVE     if quality_score < 45
```

**Threshold rationale**:
- 75 as the ACCEPTABLE boundary means at most 25% penalty — typical of a
  slightly noisy or slightly soft image that is still usable.
- 45 as the DEFECTIVE boundary means more than 55% penalty — the image is
  severely degraded and unlikely to be useful for its intended purpose.
- The 45–74 DEGRADED band covers images that may be usable depending on
  context (e.g., a blurry thumbnail is still recognisable; a blurry medical
  scan is not).

### Step 7 — Issue List

Each detected defect type appears in the issue list only if its severity is
non-trivial (≠ "none"). Per-issue confidence is:

```
confidence_i = 0.5 × classical_penalty_i + 0.5 × p_calibrated(class_i | x)
```

This blends the interpretable signal strength (classical) with the model's
learned certainty (probabilistic), capped by the joint evidence.

Issues are sorted by confidence descending so the most likely defect appears
first.

---

## Known Limitations

- The classical feature thresholds were manually calibrated on the synthetic
  dataset. Real-world images may differ (e.g., artistic dark-key images will
  be misclassified as underexposed). A future version should learn these
  thresholds from labelled real-world examples.
- The defective class heuristic is intentionally weak (saturation/contrast
  anomaly). It relies on the model for most of its signal. Grad-CAM
  localisation should be used to interpret where the defect is.
- Temperature scaling is a global calibration — it improves average calibration
  but cannot fix systematic over/underconfidence on individual classes.
