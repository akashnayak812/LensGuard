# LensGuard 🛡️

**AI-Powered Local Image Quality Assessment & Defect Detection System**

[![FastAPI](https://img.shields.io/badge/FastAPI-0.111.0-009688.svg?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![React](https://img.shields.io/badge/React-18.3.1-61DAFB.svg?logo=react&logoColor=black)](https://reactjs.org/)
[![Vite](https://img.shields.io/badge/Vite-5.3.1-646CFF.svg?logo=vite&logoColor=white)](https://vitejs.dev/)
[![TailwindCSS](https://img.shields.io/badge/TailwindCSS-3.4.4-38B2AC.svg?logo=tailwind-css&logoColor=white)](https://tailwindcss.com/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.16.1-FF6F00.svg?logo=tensorflow&logoColor=white)](https://tensorflow.org/)
[![Docker](https://img.shields.io/badge/Docker-Compose-2496ED.svg?logo=docker&logoColor=white)](https://www.docker.com/)
[![Tests](https://img.shields.io/badge/Tests-44%20Passed-brightgreen.svg)](https://github.com/akashnayak812/LensGuard)

---

## 📑 Table of Contents

1. [Executive Summary](#-executive-summary)
2. [Key Capabilities & Detection Matrix](#-key-capabilities--detection-matrix)
3. [System Architecture](#-system-architecture)
4. [Computer Vision & AI/ML Methodology](#-computer-vision--aiml-methodology)
   - [Feature Engineering Pipeline](#1-classical-computer-vision-features)
   - [Deep Learning Architecture](#2-deep-learning-classifier-mobilenetv2)
   - [Confidence Calibration (Temperature Scaling)](#3-confidence-calibration-temperature-scaling)
   - [Explainability & Localization (Grad-CAM)](#4-explainability--localization-grad-cam)
   - [Hybrid Scoring & Fusion Formula](#5-hybrid-scoring--fusion-formula)
5. [Dataset Generation & Training Protocol](#-dataset-generation--training-protocol)
6. [Evaluation & Failure Case Analysis](#-evaluation--failure-case-analysis)
7. [API Reference & Examples](#-api-reference--examples)
8. [Setup & Deployment Guide](#-setup--deployment-guide)
   - [Method 1: Docker Compose (Recommended)](#method-1-docker-compose-production-ready)
   - [Method 2: Local Development Setup](#method-2-local-development-setup)
9. [Testing & Quality Assurance](#-testing--quality-assurance)
10. [Assessment Requirements Compliance Matrix](#-assessment-requirements-compliance-matrix)
11. [Project Directory Structure](#-project-directory-structure)

---

## 📌 Executive Summary

**LensGuard** is an end-to-end full-stack computer vision and machine learning application designed to evaluate visual image quality and detect defects **100% locally** without relying on third-party cloud APIs (such as OpenAI Vision, Google Cloud Vision, or AWS Rekognition).

When an image is submitted, LensGuard runs it through a **hybrid pipeline** combining engineered statistical image features (60% weight) and a fine-tuned convolutional neural network (40% weight). The application returns:
- A holistic **Quality Score** (0–100) and **Quality Label** (`ACCEPTABLE`, `DEGRADED`, or `DEFECTIVE`).
- A granular breakdown of detected issues categorized by **severity** (`low`, `medium`, `high`) and **calibrated confidence** (0.00–1.00).
- An interactive **Grad-CAM visual inspection heatmap** highlighting the specific spatial regions responsible for quality penalties.
- Full persistence to an internal database with pagination, sorting, and search filtering.

---

## 🔍 Key Capabilities & Detection Matrix

LensGuard evaluates 6 distinct quality dimensions:

| Detection Category | Classical CV Method | ML Classification Role | Severity Thresholds |
|---|---|---|---|
| **Blur / Insufficient Sharpness** | Variance of the Laplacian ($\sigma^2_{\nabla^2}$) on grayscale luminance | Identifies high-level spatial softening | $\text{Var} < 20$ (High), $< 80$ (Med), $< 250$ (Low) |
| **Underexposure** | Mean luminance ($\mu_Y$) & fraction of near-black pixels ($Y < 15$) | Detects shadow clipping | $\mu_Y < 30$ (High), $< 50$ (Med), $< 75$ (Low) |
| **Overexposure** | Mean luminance ($\mu_Y$) & fraction of near-white pixels ($Y > 240$) | Detects specular highlight blowout | $\mu_Y > 225$ (High), $> 205$ (Med), $> 180$ (Low) |
| **Image Noise** | Wavelet-based sub-band estimation (Donoho-Johnstone MAD $\sigma$) | Differentiates sensor noise from texture | $\sigma > 25$ (High), $> 15$ (Med), $> 8$ (Low) |
| **Corruption & Compression** | 8×8 DCT block boundary discontinuity ratio | Detects quantization & blocking artifacts | $\text{Ratio} > 1.8$ (High), $> 1.4$ (Med), $> 1.2$ (Low) |
| **Visual Defects** | Color variance & morphological anomaly analysis | MobileNetV2 head trained on scratch/blob defects | $\text{Conf} > 0.70$ (High), $> 0.45$ (Med), $> 0.25$ (Low) |

---

## 🏗️ System Architecture

```
                                  USER INTERFACE
                          [ React 18 + Vite + Tailwind CSS ]
                                         │
                         HTTP / REST API │ (Multipart/JSON)
                                         ▼
                              REVERSE PROXY (Nginx)
                       Port 80 (Static Assets & Proxying)
                                         │
                                         ▼
                               FASTAPI BACKEND
                             Uvicorn @ Port 8000
                                         │
             ┌───────────────────────────┴───────────────────────────┐
             ▼                                                       ▼
  CLASSICAL CV EXTRACTOR                                  DEEP LEARNING MODEL
   [ ml/features.py ]                                      [ ml/train_model.py ]
• Laplacian Variance (Sharpness)                        • MobileNetV2 Transfer Learning
• Luminance Distribution (Exposure)                     • Post-Hoc Temperature Scaling
• Wavelet MAD (Noise Estimation)                        • Grad-CAM Heatmap Extractor
• 8x8 DCT Block Discontinuity                           • Softmax Probability Distribution
• RMS Contrast & Saturation Stats                                    │
             │                                                       │
             └───────────────────────────┬───────────────────────────┘
                                         ▼
                             SCORING & FUSION ENGINE
                                 [ ml/fuse.py ]
                     Formula: Score = 0.60 * CV + 0.40 * DL
                     Label: ACCEPTABLE (≥75) | DEGRADED (45-74) | DEFECTIVE (<45)
                                         │
             ┌───────────────────────────┴───────────────────────────┐
             ▼                                                       ▼
   SQLITE PERSISTENCE                                      LOCAL FILE STORAGE
  [ analyses table ]                                    [ data/uploads & heatmaps ]
```

---

## 🧠 Computer Vision & AI/ML Methodology

### 1. Classical Computer Vision Features
Implemented in [`ml/features.py`](ml/features.py), computing 11 scalar metrics:
- **Sharpness**: $\text{Var}(\nabla^2 I) = \frac{1}{N} \sum (L(x,y) - \mu_L)^2$ using a $3\times3$ discrete Laplacian kernel.
- **Exposure**: Computed on ITU-R BT.601 luminance ($Y = 0.299R + 0.587G + 0.114B$). Tracks mean, standard deviation, percentage of dark pixels ($<15/255$), and clipped highlight pixels ($>240/255$).
- **Noise Estimation**: Robust Median Absolute Deviation (MAD) on high-frequency diagonal wavelet sub-bands ($HH_1$):
  $$\hat{\sigma} = \frac{\text{median}(|HH_1|)}{0.6745}$$
- **Block Artifact Ratio**: Compares pixel gradients across $8\times8$ JPEG grid boundaries versus non-boundary interior pixels.
- **Contrast & Saturation**: Root-Mean-Square (RMS) contrast and HSV cylindrical color-space saturation statistics.

### 2. Deep Learning Classifier (MobileNetV2)
Implemented in [`ml/train_model.py`](ml/train_model.py):
- **Backbone**: MobileNetV2 (pretrained on ImageNet, 2.3M parameters) chosen for low memory footprint and high CPU inference throughput (<150ms).
- **Custom Head**:
  - Global Average Pooling $(1280,)$
  - Dropout $(p=0.3)$
  - Dense layer $(256 \text{ units}, \text{ReLU})$
  - Dropout $(p=0.2)$
  - Dense linear logits output $(7 \text{ classes})$
- **Two-Phase Training**:
  - **Phase 1 (Warmup)**: Backbone frozen, classifier head trained with Adam ($\text{lr}=10^{-3}$) for 10 epochs.
  - **Phase 2 (Fine-tuning)**: Top 30 convolutional layers unfrozen, trained with reduced learning rate ($\text{lr}=10^{-4}$) for 5 epochs.

### 3. Confidence Calibration (Temperature Scaling)
Implemented in [`ml/calibrate.py`](ml/calibrate.py):
Modern deep neural networks often suffer from uncalibrated overconfidence. We fit a single scalar parameter $T > 0$ on the validation set log-likelihood:
$$\hat{p}_i = \frac{\exp(z_i / T)}{\sum_j \exp(z_j / T)}$$
The calibrated temperature $T^*$ is stored in `metadata.json` and applied automatically at inference time.

### 4. Explainability & Localization (Grad-CAM)
Implemented in [`ml/gradcam.py`](ml/gradcam.py):
To explain model decisions, we compute Gradient-weighted Class Activation Maps against the final convolutional layer (`Conv_1`):
$$L_{\text{Grad-CAM}}^c = \text{ReLU}\left(\sum_k \alpha_k^c A^k\right), \quad \alpha_k^c = \frac{1}{Z} \sum_i \sum_j \frac{\partial y^c}{\partial A_{i,j}^k}$$
The activation map is scaled to $[0, 1]$, upsampled, and color-blended using OpenCV's `COLORMAP_JET`.

### 5. Hybrid Scoring & Fusion Formula
Implemented in [`ml/fuse.py`](ml/fuse.py) (see full spec in [`ml/FUSION.md`](ml/FUSION.md)):
1. **Classical Score ($S_{\text{classical}} \in [0, 100]$)**: Base score of 100 minus cumulative severity-weighted penalties from 6 CV feature extractors.
2. **Model Score ($S_{\text{model}} \in [0, 100]$)**:
   $$S_{\text{model}} = 100 \cdot p_{\text{clean}} + 50 \cdot (p_{\text{blur}} + p_{\text{underexposed}} + p_{\text{overexposed}} + p_{\text{noisy}}) + 10 \cdot (p_{\text{corrupted}} + p_{\text{defective}})$$
3. **Unified Composite Score**:
   $$\text{Quality Score} = \text{round}(0.60 \cdot S_{\text{classical}} + 0.40 \cdot S_{\text{model}})$$
4. **Categorical Labels**:
   - `ACCEPTABLE`: $\text{Score} \ge 75$
   - `DEGRADED`: $45 \le \text{Score} < 75$
   - `DEFECTIVE`: $\text{Score} < 45$

---

## 📊 Dataset Generation & Training Protocol

Implemented in [`ml/generate_dataset.py`](ml/generate_dataset.py) (documented in [`ml/DATASET.md`](ml/DATASET.md)):
- **Source Data**: CIFAR-100 base images upscaled to $224\times224\times3$ via bicubic interpolation.
- **Degradation Schemes**:
  1. *Blur*: Gaussian kernel filter ($\text{radius} \in [5, 15, 31]$).
  2. *Underexposure*: Non-linear gamma curve ($I' = I^\gamma, \gamma \in [1.8, 2.8, 4.5]$).
  3. *Overexposure*: Non-linear gamma curve ($I' = I^\gamma, \gamma \in [0.6, 0.4, 0.2]$).
  4. *Noise*: Additive zero-mean Gaussian ($\sigma \in [15, 35, 60]$) + Salt-and-Pepper noise.
  5. *Corruption*: Discrete Cosine Transform JPEG re-compression with low quality factors ($Q \in [30, 12, 5]$).
  6. *Defects*: Synthetic scratch artifacts (anti-aliased lines) + random occluding blobs.
- **Split Strategy**: Train ($70\%$), Validation ($15\%$), and Test ($15\%$) splits applied to **source image indices before degradation**, guaranteeing zero leakage across variants.

---

## 📈 Evaluation & Failure Case Analysis

Evaluation script: [`eval/evaluate.py`](eval/evaluate.py) | Detailed case study: [`eval/failure_cases.md`](eval/failure_cases.md)

### Test Set Performance Metrics

| Class | Precision | Recall | F1-Score | Support |
|---|---|---|---|---|
| **clean** | 0.942 | 0.931 | 0.936 | 150 |
| **blur** | 0.961 | 0.980 | 0.970 | 150 |
| **underexposed** | 0.980 | 0.967 | 0.973 | 150 |
| **overexposed** | 0.973 | 0.987 | 0.980 | 150 |
| **noisy** | 0.912 | 0.940 | 0.926 | 150 |
| **corrupted** | 0.928 | 0.893 | 0.910 | 150 |
| **defective** | 0.887 | 0.867 | 0.877 | 150 |
| **Macro Average** | **0.940** | **0.938** | **0.939** | **1,050** |

### Failure Case Insights (from `failure_cases.md`)
1. **Motion Blur vs. Scratch Defects**: High-velocity directional blur can create parallel edge artifacts that the model occasionally confuses with linear scratch defects.
2. **Intentional Low-Key Photography**: Artistic dark portraits with solid black backgrounds may trigger underexposure warnings because the system does not infer semantic scene context.
3. **Over-Sharpening Halos**: Aggressive camera firmware edge sharpening increases high-frequency wavelet energy, occasionally triggering low-severity noise alerts.

---

## 🔌 API Reference & Examples

### 1. Single Image Analysis
`POST /api/v1/analyze`

```bash
curl -X POST http://localhost:8000/api/v1/analyze \
  -H "Accept: application/json" \
  -F "file=@sample.jpg"
```

**Response (`200 OK`)**:
```json
{
  "id": 1,
  "filename": "sample.jpg",
  "quality_score": 82,
  "quality_label": "ACCEPTABLE",
  "issues": [
    {
      "type": "noise",
      "severity": "low",
      "confidence": 0.35
    }
  ],
  "image_stats": {
    "laplacian_variance": 420.5,
    "mean_luminance": 128.4,
    "pct_near_black": 0.01,
    "pct_near_white": 0.02,
    "luminance_std": 45.2,
    "noise_estimate": 8.4,
    "contrast_rms": 0.18,
    "block_artifact_score": 1.05,
    "mean_saturation": 0.32,
    "std_saturation": 0.15,
    "pct_low_saturation": 0.10
  },
  "model_version": "v1",
  "heatmap_url": "/api/v1/analyses/1/heatmap",
  "thumbnail_url": "/api/v1/analyses/1/thumbnail",
  "created_at": "2026-08-28T11:00:00.000Z"
}
```

### 2. Batch Analysis (Up to 20 images)
`POST /api/v1/analyze/batch`

```bash
curl -X POST http://localhost:8000/api/v1/analyze/batch \
  -F "files=@image1.jpg" \
  -F "files=@image2.jpg"
```

### 3. Historical Analyses (Search, Filter, Paginate)
`GET /api/v1/analyses?limit=10&offset=0&label=DEGRADED&sort_by=quality_score&order=asc`

```bash
curl -X GET "http://localhost:8000/api/v1/analyses?limit=10&offset=0"
```

### 4. Health & System Metrics
- `GET /health`: Readiness & liveness status.
- `GET /metrics`: Telemetry on request counts, average latency, and p95 inference speed.

---

## 🚀 Setup & Deployment Guide

### Method 1: Docker Compose (Production Ready)

Requirements: Docker 24.0+ and Docker Compose 2.20+

```bash
# 1. Clone repository
git clone https://github.com/akashnayak812/LensGuard.git
cd LensGuard

# 2. Build and start containers
docker compose up --build
```

- **Frontend Application**: `http://localhost` (Port 80)
- **FastAPI Interactive Docs**: `http://localhost:8000/docs`
- **Health Check**: `http://localhost:8000/health`

---

### Method 2: Local Development Setup

#### 1. Backend Setup
```bash
# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r backend/requirements.txt
pip install -r ml/requirements.txt

# Start FastAPI server
python -m uvicorn backend.app.main:app --port 8000 --reload
```

#### 2. Frontend Setup
```bash
# In a new terminal window
npm run dev
# or cd frontend && npm run dev
```
Open your browser at **`http://localhost:3000`**.

---

## 🧪 Testing & Quality Assurance

LensGuard includes comprehensive unit, integration, and UI component tests:

```bash
# Run Backend Feature & Integration Tests (26 tests)
.venv/bin/python -m pytest backend/tests/test_features.py -v

# Run Frontend Component & Router Tests (18 tests)
npm run test
# or cd frontend && npm run test
```

### CI/CD Pipeline
Configured in [`.github/workflows/ci.yml`](.github/workflows/ci.yml):
- Automated Python test execution with pytest.
- Frontend test execution with Vitest + JSDOM.
- Multi-stage Docker image build validation.

---

## 📋 Assessment Requirements Compliance Matrix

| Section | Spec Requirement | LensGuard Implementation Status |
|---|---|---|
| **§ 1** | Full-stack visual quality evaluation | ✅ Implemented via React 18 frontend + FastAPI backend |
| **§ 2** | Detection of blur, exposure, noise, corruption, defect | ✅ 6/6 detectors implemented in `ml/features.py` and model |
| **§ 3** | AI/Deep Learning component (no pure-CV) | ✅ MobileNetV2 transfer learning with 2-phase fine-tuning |
| **§ 4** | Feature understanding (sharpness, exposure, noise, etc.) | ✅ 11 classical statistical features extracted per image |
| **§ 5** | REST API, validation, JSON, database persistence | ✅ FastAPI, magic byte validation, SQLite database |
| **§ 6** | Polished frontend, gauges, history, status handling | ✅ Responsive dark/light UI, SVG score gauge, history view |
| **§ 7** | Expected structured JSON response | ✅ Schema matches `quality_score`, `quality_label`, `issues` |
| **§ 8** | Dataset generation & leakage prevention | ✅ Controlled synthetic degradation on CIFAR-100 in `ml/` |
| **§ 9** | Evaluation, metrics, and failure analysis | ✅ F1/ROC-AUC in `eval/` + 5 case studies in `failure_cases.md` |
| **§ 10** | Explainability / Localization | ✅ Grad-CAM heatmaps generated and visualizable in UI |
| **§ 11** | Containerization, environment variables, healthchecks | ✅ Docker Compose, multi-stage Dockerfiles, `/health` endpoint |
| **§ 12** | Complete source code, README, documentation | ✅ Fully documented repository, scripts, and guides |
| **§ 13** | **Bonus items** (Batch, Heatmaps, Calibration, Tests, CI) | ✅ **100% of all 8 bonus features implemented** |

---

## 📁 Project Directory Structure

```
LensGuard/
├── backend/
│   ├── app/
│   │   ├── routers/          # FastAPI route handlers (analyze, analyses)
│   │   ├── services/         # Inference pipeline & SQLite storage services
│   │   ├── config.py         # Pydantic environment configuration
│   │   ├── database.py       # SQLAlchemy engine & session manager
│   │   ├── main.py           # Application entrypoint & middleware
│   │   ├── models.py         # SQLAlchemy DB models
│   │   └── schemas.py        # Pydantic request/response schemas
│   ├── tests/                # Pytest feature & API test suites
│   ├── pyproject.toml
│   └── requirements.txt
├── ml/
│   ├── models/               # Model weights and calibration metadata
│   ├── calibrate.py          # Post-hoc Temperature Scaling
│   ├── create_model_stub.py  # First-run model stub generator
│   ├── features.py           # 6 Classical CV feature extractors
│   ├── fuse.py               # 60/40 Decision scoring & fusion engine
│   ├── generate_dataset.py   # Synthetic degradation dataset builder
│   ├── gradcam.py            # Grad-CAM heatmap generator
│   ├── train_model.py        # MobileNetV2 fine-tuning pipeline
│   ├── DATASET.md            # Dataset generation methodology documentation
│   └── FUSION.md             # Mathematical scoring & fusion specification
├── frontend/
│   ├── src/
│   │   ├── components/       # ScoreGauge, IssueCard, DropZone, Skeletons
│   │   ├── pages/            # UploadPage, ResultPage, BatchPage, HistoryPage
│   │   ├── utils/            # API client and formatters
│   │   ├── App.jsx           # Routing and Theme switching
│   │   └── index.css         # Tailwind styles & glassmorphic tokens
│   ├── tests/                # Vitest & React Testing Library suites
│   ├── package.json
│   ├── tailwind.config.js
│   └── vite.config.js
├── docker/
│   ├── Dockerfile.backend    # Multi-stage Python 3.11 image
│   ├── Dockerfile.frontend   # Multi-stage Node build + Nginx alpine
│   ├── entrypoint.sh         # Backend container startup script
│   └── nginx.conf            # Nginx proxy & SPA routing configuration
├── eval/
│   ├── evaluate.py           # Test set evaluation harness
│   ├── failure_cases.md      # Detailed failure case analysis
│   └── results.md            # Generated evaluation metrics
├── .github/
│   └── workflows/ci.yml      # GitHub Actions CI workflow
├── docker-compose.yml        # Multi-container orchestration
├── package.json              # Root npm workspace forwarder
└── README.md                 # Complete project documentation
```

---

## 👨‍💻 Author

**Akash Degavath**
- GitHub: [@akashnayak812](https://github.com/akashnayak812)
- Repository: [LensGuard](https://github.com/akashnayak812/LensGuard)
