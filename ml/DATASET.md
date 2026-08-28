# LensGuard Synthetic Dataset

## Overview

The dataset is generated entirely from CIFAR-100 (60,000 images, 32×32 RGB),
upscaled to 224×224 via bicubic interpolation. No paid datasets or external
API keys are required — CIFAR-100 is downloaded automatically by
`tensorflow.keras.datasets` on first run.

## Generation Parameters

| Parameter | Value |
|---|---|
| Source | CIFAR-100 (train + test splits concatenated) |
| Source resolution | 32×32 → upscaled to 224×224 (bicubic) |
| Max source images used | 2,000 (configurable via `--max-images`) |
| Random seed | 42 |
| Clean image fraction | 15% of source images left unmodified |
| Degraded combos per source | 6 defect types × 3 severities = 18 variants |
| Total estimated images | ~34,300 (2,000 × 0.15 clean + 2,000 × 0.85 × 18) |

## Split Strategy

**Critical**: splits are applied to **source image indices** before generating
degraded variants. This prevents train/test leakage where the same underlying
scene (clean) appears in both splits under different degradation conditions.

| Split | Fraction | Source images |
|---|---|---|
| train | 70% | 1,400 |
| val | 15% | 300 |
| test | 15% | 300 |

## Defect Types and Severity Levels

### `blur` — Gaussian Blur
| Severity | Kernel size | Sigma |
|---|---|---|
| low | 5×5 | 1.5 |
| medium | 15×15 | 5.0 |
| high | 31×31 | 10.0 |

### `underexposed` — Gamma Darkening (γ > 1)
| Severity | Gamma |
|---|---|
| low | 1.8 |
| medium | 2.8 |
| high | 4.5 |

### `overexposed` — Gamma Brightening (γ < 1)
| Severity | Gamma |
|---|---|
| low | 0.6 |
| medium | 0.4 |
| high | 0.2 |

### `noisy` — Gaussian + Salt-and-Pepper
| Severity | Gaussian σ | S&P fraction |
|---|---|---|
| low | 15 | 2% |
| medium | 40 | 6% |
| high | 80 | 15% |

### `corrupted` — JPEG Re-compression
| Severity | JPEG quality |
|---|---|
| low | 30 |
| medium | 12 |
| high | 5 |

### `defective` — Synthetic Visual Defects (scratches + blobs)
| Severity | Scratches | Blobs | Max blob radius |
|---|---|---|---|
| low | 2 | 1 | 8px |
| medium | 6 | 3 | 20px |
| high | 15 | 7 | 45px |

## Output Artefacts

```
data/
├── images/
│   ├── train/   *.jpg
│   ├── val/     *.jpg
│   └── test/    *.jpg
├── manifest.json   — full record of every image with labels
├── manifest.csv    — same, CSV format
└── dataset_stats.json — class distribution summary
```

## Manifest Schema

```json
{
  "filename": "images/train/train_00001_blur_medium.jpg",
  "split": "train",
  "source_id": 1,
  "issue_type": "blur",
  "severity": "medium"
}
```

## Reproducibility

Re-run `python ml/generate_dataset.py --seed 42` to fully regenerate the
dataset. The `/data/` directory is git-ignored (too large to commit), but
`manifest.json` schema and all generation parameters are captured here.
