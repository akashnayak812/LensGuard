# LensGuard

**AI-powered image quality and defect detection — all inference runs locally, no external APIs.**

Upload an image and get an instant quality score (0–100), ACCEPTABLE/DEGRADED/DEFECTIVE label, per-defect breakdown with severity and confidence, and a Grad-CAM heatmap showing exactly where the defect was detected.

---

## Table of Contents

1. [Quick Start](#quick-start)
2. [Architecture](#architecture)
3. [Dataset & Training](#dataset--training)
4. [Model Architecture](#model-architecture)
5. [API Reference](#api-reference)
6. [Inference Pipeline](#inference-pipeline)
7. [Deployment](#deployment)
8. [Known Limitations](#known-limitations)
9. [Development](#development)

---

## Quick Start

### Requirements
- Docker ≥ 24.0
- Docker Compose ≥ 2.20

### Run
```bash
git clone https://github.com/yourname/lensguard
cd lensguard
docker compose up --build
```

- **Frontend**: http://localhost
- **API docs**: http://localhost:8000/docs
- **Health**: http://localhost:8000/health

> **First run note**: The backend downloads MobileNetV2 ImageNet weights (~14MB) on startup and creates a stub model. This takes ~2–3 minutes. Watch `docker compose logs backend` for progress.

> **Training a real model**: run `python ml/generate_dataset.py` then `python ml/train_model.py`. This takes 5–30 minutes depending on hardware. The stub model serves the API with near-random predictions.

---

## Architecture

```
Browser (React) ──HTTP──▶ nginx ──▶ static frontend build
       │
       └────HTTP (REST)────▶ FastAPI backend (port 8000)
                                  │
                        ┌─────────┴─────────┐
                        ▼                   ▼
                Classical CV features   TensorFlow model
                (OpenCV, NumPy,         (MobileNetV2 head,
                 PyWavelets)             7-class classifier)
                        │                   │
                        └─────────┬─────────┘
                                  ▼
                          Fusion / scoring
                          (ml/fuse.py)
                                  │
                                  ▼
                         SQLite (analyses table)
                         + Grad-CAM PNG storage
```

### Directory Layout
```
LensGuard/
├── backend/          FastAPI app, DB models, API routes, tests
├── ml/               Dataset generation, feature extraction, model training,
│                     Grad-CAM, calibration, fusion
├── frontend/         React (Vite) app with Tailwind CSS
├── docker/           Dockerfiles, docker-compose.yml, nginx.conf
├── eval/             Evaluation scripts and results
├── data/             Generated dataset (gitignored — regeneratable)
├── .github/          CI workflow (GitHub Actions)
└── README.md
```

---

## Dataset & Training

### Dataset Generation

Source images: **CIFAR-100** (freely available via `tensorflow.keras.datasets`), upscaled from 32×32 to 224×224 via bicubic interpolation.

For each source image, 6 defect types × 3 severity levels = 18 degraded variants are generated:

| Defect | Method |
|---|---|
| Blur | Gaussian blur (kernel 5/15/31) |
| Underexposed | Gamma darkening (γ=1.8/2.8/4.5) |
| Overexposed | Gamma brightening (γ=0.6/0.4/0.2) |
| Noisy | Gaussian + salt-and-pepper noise |
| Corrupted | JPEG re-compression (quality 30/12/5) |
| Defective | Synthetic scratches + blob overlays |

**Split strategy**: Train/val/test split applied to **source image indices** before degradation — prevents leakage.

```bash
# Regenerate dataset
python ml/generate_dataset.py --output-dir data/ --seed 42 --max-images 2000
```

See [`ml/DATASET.md`](ml/DATASET.md) for full parameter documentation.

### Training

```bash
# Train the model (requires dataset)
python ml/train_model.py --data-dir data/ --output-dir ml/models --version v1

# Calibrate confidence (temperature scaling)
python ml/calibrate.py --data-dir data/ --model-dir ml/models/v1
```

Training logs written to `ml/training_log.json`. Model saved to `ml/models/v1/`.

---

## Model Architecture

```
Input (224×224×3 RGB)
       │
       ▼
MobileNetV2 backbone (ImageNet pretrained, 2.3M params)
  Phase 1: frozen, only head trained (10 epochs, LR=1e-3)
  Phase 2: top 30 layers fine-tuned (5 epochs, LR=1e-4)
       │
       ▼ GlobalAveragePooling → (1280,)
       │
       ▼ Dense(256, relu) + Dropout(0.2)
       │
       ▼ Dense(7, no activation) → logits (7 classes)
       │
       ▼ Temperature scaling (T = calibrated)
       │
       ▼ Softmax → calibrated probabilities
```

**Classes**: clean, blur, underexposed, overexposed, noisy, corrupted, defective

**Confidence calibration**: Temperature scaling (Guo et al., 2017) minimises NLL on the validation set. Temperature `T*` stored in `metadata.json` and applied at inference. See [`ml/calibrate.py`](ml/calibrate.py).

**Explainability**: Grad-CAM against the last MobileNetV2 convolutional layer. See [`ml/gradcam.py`](ml/gradcam.py).

**Scoring**: See [`ml/FUSION.md`](ml/FUSION.md) for the complete formula.

---

## API Reference

Base URL: `http://localhost:8000` (direct) or `http://localhost` (through nginx)

### `POST /api/v1/analyze`

Analyse a single image.

```bash
curl -X POST http://localhost:8000/api/v1/analyze \
  -F "file=@photo.jpg"
```

**Response:**
```json
{
  "id": 42,
  "filename": "photo.jpg",
  "quality_score": 82,
  "quality_label": "ACCEPTABLE",
  "issues": [
    {"type": "noise", "severity": "low", "confidence": 0.71}
  ],
  "image_stats": {
    "laplacian_variance": 340.2,
    "mean_luminance": 128.4,
    "contrast_rms": 0.192
  },
  "model_version": "v1",
  "heatmap_url": "/api/v1/analyses/42/heatmap",
  "thumbnail_url": "/api/v1/analyses/42/thumbnail",
  "created_at": "2026-08-28T10:00:00Z"
}
```

**Error response (400/413/422/500):**
```json
{"error": {"code": "FILE_TOO_LARGE", "message": "File exceeds maximum allowed size of 20 MB."}}
```

---

### `POST /api/v1/analyze/batch`

Analyse multiple images.

```bash
curl -X POST http://localhost:8000/api/v1/analyze/batch \
  -F "files=@photo1.jpg" \
  -F "files=@photo2.jpg"
```

**Response:**
```json
{
  "results": [...],
  "failed": [{"filename": "bad.txt", "error": {"code": "INVALID_IMAGE", "message": "..."}}]
}
```

---

### `GET /api/v1/analyses`

Paginated history.

```bash
# All analyses, most recent first
curl "http://localhost:8000/api/v1/analyses?limit=20&offset=0"

# Filter by label
curl "http://localhost:8000/api/v1/analyses?label=DEFECTIVE"

# Sort by score ascending
curl "http://localhost:8000/api/v1/analyses?sort_by=quality_score&order=asc"
```

---

### `GET /api/v1/analyses/{id}`

Single analysis by ID.

```bash
curl http://localhost:8000/api/v1/analyses/42
```

---

### `GET /api/v1/analyses/{id}/heatmap`

Serve Grad-CAM heatmap PNG.

```bash
curl http://localhost:8000/api/v1/analyses/42/heatmap -o heatmap.png
```

---

### `GET /health`

Liveness + readiness check.

```bash
curl http://localhost:8000/health
# {"status": "ready", "model_loaded": true, "model_version": "v1", "db_reachable": true, "uptime_seconds": 42.1}
```

---

### `GET /metrics`

Lightweight metrics.

```bash
curl http://localhost:8000/metrics
# {"total_requests": 150, "total_analyses": 48, "avg_inference_latency_ms": 187.3, "error_count": 2}
```

---

## Inference Pipeline

When a `POST /api/v1/analyze` request arrives:

1. **File validation**: MIME type check + magic byte verification + size limit
2. **Decode**: `tf.image.decode_image` → BGR numpy array (consistent with training)
3. **Save**: original + thumbnail written to `UPLOAD_DIR`
4. **Classical features**: `ml.features.extract_all` — 6 functions → 11 scalar features
5. **Preprocess**: resize to 224×224, MobileNetV2 normalisation
6. **Forward pass**: model logits → raw shape (7,)
7. **Calibration**: `ml.calibrate.apply_temperature_scaling(logits, T*)` → calibrated probs
8. **Grad-CAM**: `ml.gradcam.generate_gradcam` → heatmap PNG saved to `HEATMAP_DIR`
9. **Fusion**: `ml.fuse.fuse(features, logits, T*)` → `quality_score`, `quality_label`, `issues[]`
10. **Persist**: SQLAlchemy writes `Analysis` record to SQLite
11. **Response**: Pydantic schema serialised to JSON

**Typical latency**: 100–400ms (CPU, MobileNetV2). Grad-CAM adds ~50ms.

---

## Deployment

### Environment Variables

Copy `.env.example` → `.env` and adjust:

| Variable | Default | Description |
|---|---|---|
| `DB_PATH` | `/app/data/lensguard.db` | SQLite database file path |
| `MODEL_PATH` | `/app/ml/models` | Root directory for model versions |
| `MODEL_VERSION` | `v1` | Active model version folder |
| `MAX_UPLOAD_MB` | `20` | Max upload size in MB |
| `CORS_ORIGINS` | `http://localhost,...` | Comma-separated allowed origins |
| `UPLOAD_DIR` | `/app/data/uploads` | Image + thumbnail storage |
| `HEATMAP_DIR` | `/app/data/heatmaps` | Grad-CAM PNG storage |
| `LOG_LEVEL` | `INFO` | Python log level |

### Data Persistence

All data (DB, uploads, heatmaps) lives in the `lensguard_data` Docker named volume. Data survives `docker compose restart` but is wiped by `docker compose down -v`.

### Model Versioning

Each trained model lives in `ml/models/vN/`. To deploy a new model:
1. Train: `python ml/train_model.py --version v2`
2. Calibrate: `python ml/calibrate.py --model-dir ml/models/v2`
3. Set `MODEL_VERSION=v2` in `.env`
4. Restart backend: `docker compose restart backend`

---

## Known Limitations

1. **Synthetic training data**: The model was trained on programmatic degradations of CIFAR-100 images. Real-world photos with intentional artistic choices (dark-key, high-key, film grain) may be misclassified. See `eval/failure_cases.md`.

2. **No multi-defect ground truth**: The single-label classifier predicts one primary defect class. Images with multiple simultaneous defects may have the secondary defect missed or the primary class mis-assigned.

3. **CPU-only inference**: TensorFlow on CPU yields 100–400ms per image. For production throughput, GPU passthrough or ONNX export is recommended.

4. **SQLite**: Not suitable for high-concurrency writes. For production, replace with PostgreSQL via `DATABASE_URL` env var.

5. **No authentication**: The API is open. Add an API key middleware or reverse-proxy authentication before public deployment.

6. **CIFAR-100 source resolution**: 32×32 images upscaled to 224×224 are blurrier than real photographs, which may under-represent sharpness-dependent features in training.

---

## Development

### Backend

```bash
cd backend
pip install -r requirements.txt
pip install -r ../ml/requirements.txt
uvicorn app.main:app --reload
```

### Frontend

```bash
cd frontend
npm install
npm run dev   # http://localhost:3000
```

### Tests

```bash
# Backend
cd backend && pytest tests/ -v

# Frontend
cd frontend && npm test
```

### Regenerate Dataset + Retrain

```bash
python ml/generate_dataset.py --max-images 2000
python ml/train_model.py
python ml/calibrate.py
python eval/evaluate.py
```
# LensGuard
