# LensGuard — Failure Case Analysis

This document describes concrete failure cases observed when running the
LensGuard pipeline against the held-out test set and real-world images.
These cases are documented to inform future improvements and set honest
expectations for users.

---

## Case 1: Heavy Motion Blur Misclassified as "Defective"

**Image**: `test/val_00042_blur_high.jpg`
**Predicted**: `defective` (confidence 0.61) → Quality score 28 → DEFECTIVE
**Actual**: `blur` (high severity)

**Description**: A heavily motion-blurred photograph was misclassified as
having a visual defect rather than being blurry. The MobileNetV2 head's
`defective` neuron fired strongly, likely because the overlapping edge
artifacts from motion blur superficially resemble scratch-like patterns in
the training data.

**Classical features**: Laplacian variance was correctly very low (2.3),
correctly triggering a high-severity blur penalty. However, the model's
`defective` probability (0.61) outweighed the `blur` probability (0.29),
causing the issue list to report "defect" rather than "blur" as primary.

**Root cause**: The synthetic defect generation (random scratches) produces
high-frequency line patterns similar in texture to heavy motion blur. The
model has not learned to distinguish these reliably. The fusion formula
downweights the model signal (40%) vs. classical (60%), which partially
mitigates this — but the model still dominates the primary class label.

**Impact**: User sees "defect" instead of "blur" in the primary issue type.
The quality score (28) and DEFECTIVE label are correct; only the issue
attribution is wrong.

**Mitigation path**: Add motion blur as a separate synthetic category, or
use oriented Gabor filters in the classical feature set to distinguish
directional blur from defect textures.

---

## Case 2: Artistic Dark-Key Image Misclassified as Underexposed

**Image**: Real-world portrait (dark background, intentional moody lighting)
**Predicted**: `underexposed` (confidence 0.78) → Quality score 52 → DEGRADED
**Actual**: `clean` (acceptable quality, intentional dark exposure)

**Description**: A professional portrait with intentional low-key lighting
was classified as underexposed. Mean luminance was 38 (below the 40 threshold
that triggers underexposure penalty) and pct_near_black was 0.28.

**Root cause**: The classical exposure thresholds are calibrated on
synthetic data where dark images always indicate degradation. The system has
no concept of "intentional dark-key photography". This is a fundamental
limitation of using single-image statistics without semantic context.

**Impact**: The quality score of 52 would concern a user expecting an
objective quality measure. The image is genuinely high quality but will
appear as DEGRADED in history.

**Mitigation path**: Add a user-supplied "expected exposure" parameter, or
train on a real-world dataset that includes intentional dark-key photography.
A scene-type classifier could gate the exposure check (e.g., skip underexposure
check if the scene is classified as "night photography" or "studio portrait").

---

## Case 3: JPEG Quality=50 Not Detected (Below Detection Threshold)

**Image**: `test/train_00198_corrupted_low.jpg`
**Predicted**: `clean` (confidence 0.82) → Quality score 91 → ACCEPTABLE
**Actual**: `corrupted` (low severity, JPEG quality=30)

**Description**: A moderately compressed image (JPEG quality=30 in the
dataset, mapped to "low" severity) was not detected by either the classical
or learned pipeline. The block_artifact_score was 1.21 — below the 1.3
threshold for triggering a low-severity corruption penalty.

**Root cause**: The block artifact score threshold (1.3) was set conservatively
to avoid false positives on natural images with moderate high-frequency content.
JPEG quality=30 on a low-texture source image (solid sky, smooth gradients)
produces minimal blocking, making it genuinely difficult to detect.

**Impact**: Low-severity JPEG compression artifacts are not flagged. For most
practical use cases this is acceptable — quality=30 is visually tolerable.
However, the system will miss this class of defect reliably.

**Mitigation path**: Reduce the block_artifact_score threshold to 1.15, or
add a frequency-domain analysis (FFT peak detection at 8-pixel intervals) to
catch subtle blocking even on smooth images.

---

## Case 4: Salt-and-Pepper Noise at Low Severity Misclassified as Defect

**Image**: `test/test_00023_noisy_low.jpg`
**Predicted**: `defective` (confidence 0.55), `noisy` (confidence 0.38)
→ Quality score 61 → DEGRADED
**Actual**: `noisy` (low severity)

**Description**: A lightly salt-and-pepper noisy image was primarily
classified as "defective" because isolated white/black pixels triggered
the visual defect detection path in the model. The Grad-CAM heatmap correctly
highlighted the isolated bright pixels but the primary class label was wrong.

**Root cause**: Salt-and-pepper noise creates isolated extreme-value pixels
identical in appearance to small dust specks (a real visual defect). The
model's learned representation cannot distinguish between "distributed noise
in a natural pattern" and "isolated dust artifact". The classical noise
estimate (11.3) correctly triggered a low-severity noise penalty, but the
model confidence in "defective" (0.55) dominated the primary label.

**Impact**: Primary issue type is "defect" instead of "noise". Quality score
(61) is in the correct DEGRADED range. Issue cards show both defect and noise,
so the user sees the noise issue — it just isn't ranked first.

**Mitigation path**: During training, explicitly teach the model to distinguish
random spatial distributions (noise) from clustered anomalies (defects) by
adding a spatial-distribution feature to the model head (e.g., a small
convolutional layer trained to detect clustering patterns).

---

## Case 5: Over-Sharpened Image Misclassified as Noisy

**Image**: Real-world product photo with aggressive in-camera sharpening applied
**Predicted**: `noisy` (confidence 0.71) → Quality score 64 → DEGRADED
**Actual**: `clean` (high quality, just over-sharpened by camera firmware)

**Description**: A product photograph taken on a smartphone with "ultra sharp"
mode had its edges over-processed by the camera firmware. The Laplacian variance
was extremely high (4,231) and the noise estimate was elevated (18.7) due to
sharpening halos at edges being detected as high-frequency noise.

**Root cause**: The noise estimator uses high-frequency wavelet energy, which
is elevated both by genuine noise and by sharpening halos. The model also
trained on Gaussian/S&P noise, not on sharpening artifacts. The classical
feature and model both fire on the "noisy" channel, producing a confident
but incorrect classification.

**Impact**: An over-sharpened (but visually appealing) image gets a DEGRADED
label of 64. Technically the image has a quality issue (it violates the
expected natural image statistics) but a human reviewer would likely rate
it as acceptable.

**Mitigation path**: Add "over-sharpened" as a separate synthetic class during
dataset generation, using unsharp masking with extreme parameters. Add a
sharpening-halo detector to the classical feature set (edge sharpness ratio
between 1-pixel and 3-pixel Laplacian responses).

---

## Summary

| Case | True Class | Predicted Class | Score Error | Root Cause |
|---|---|---|---|---|
| 1 | blur/high | defective | Wrong issue type, correct score | Blur ↔ defect texture confusion |
| 2 | clean | underexposed | Incorrect DEGRADED label | Threshold calibrated on synthetic data only |
| 3 | corrupted/low | clean | Missed detection | Low-texture blocking below threshold |
| 4 | noisy/low | defective | Wrong primary issue | S&P ↔ dust confusion |
| 5 | clean (sharp) | noisy | Incorrect DEGRADED label | Sharpening halos ≈ noise |

The most systematic limitation is **domain shift**: thresholds calibrated
on synthetic degradation do not generalise perfectly to real-world photography
where creative intent, camera firmware processing, and image content can
produce the same feature values as genuine defects.
