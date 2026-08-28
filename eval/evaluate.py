#!/usr/bin/env python3
"""
LensGuard Evaluation Script
============================
Runs the full pipeline against the held-out test set, computes
per-class precision/recall/F1, confusion matrix, and ROC-AUC.

Usage
-----
    python eval/evaluate.py --data-dir data/ --model-dir ml/models/v1 --output-dir eval/

Outputs
-------
    eval/results.md        — human-readable metrics summary
    eval/confusion.png     — confusion matrix heatmap
    eval/roc_curves.png    — per-class ROC curves
    eval/eval_results.json — raw metrics JSON
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

CLASSES = ["clean", "blur", "underexposed", "overexposed", "noisy", "corrupted", "defective"]


def load_model(model_dir: Path):
    import tensorflow as tf
    tf.get_logger().setLevel("WARNING")

    model_path = model_dir / "model.keras"
    if not model_path.exists():
        from ml.train_model import build_model
        m, _ = build_model()
        h5_path = model_dir / "model.h5"
        if h5_path.exists():
            m.load_weights(str(h5_path))
            return m
        raise FileNotFoundError(f"No model found in {model_dir}")
    return tf.keras.models.load_model(str(model_path))


def run_evaluation(data_dir: Path, model_dir: Path, output_dir: Path) -> None:
    import tensorflow as tf
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import (
        classification_report,
        confusion_matrix,
        roc_auc_score,
        roc_curve,
    )
    from ml.calibrate import apply_temperature_scaling
    from ml.features import extract_all
    import cv2

    output_dir.mkdir(parents=True, exist_ok=True)

    # Load manifest
    with open(data_dir / "manifest.json") as f:
        all_rows = json.load(f)
    test_rows = [r for r in all_rows if r["split"] == "test"]
    print(f"Evaluating on {len(test_rows)} test images...")

    # Load model and metadata
    model = load_model(model_dir)
    with open(model_dir / "metadata.json") as f:
        metadata = json.load(f)
    temperature = float(metadata.get("temperature", 1.0))
    print(f"Loaded model, temperature={temperature:.4f}")

    class_to_idx = {c: i for i, c in enumerate(CLASSES)}

    y_true = []
    y_pred = []
    y_probs_all = []

    IMG_SIZE = (224, 224)

    for i, row in enumerate(test_rows):
        if i % 100 == 0:
            print(f"  {i}/{len(test_rows)}...")

        img_path = str(data_dir / row["filename"])
        img_bgr = cv2.imread(img_path)
        if img_bgr is None:
            continue

        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(img_rgb, IMG_SIZE, interpolation=cv2.INTER_LINEAR)
        preprocessed = tf.keras.applications.mobilenet_v2.preprocess_input(
            resized.astype(np.float32)
        )[np.newaxis, ...]

        logits = model(preprocessed, training=False).numpy()[0]
        probs = apply_temperature_scaling(logits, temperature)

        y_true.append(class_to_idx[row["issue_type"]])
        y_pred.append(int(np.argmax(probs)))
        y_probs_all.append(probs)

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_probs_all = np.array(y_probs_all)

    # ── Classification report ──────────────────────────────────────────────
    report = classification_report(
        y_true, y_pred, target_names=CLASSES, output_dict=True
    )
    report_str = classification_report(y_true, y_pred, target_names=CLASSES)
    print("\nClassification Report:")
    print(report_str)

    # ── Confusion matrix ───────────────────────────────────────────────────
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(9, 7))
    im = ax.imshow(cm, cmap="Blues")
    plt.colorbar(im)
    ax.set_xticks(range(len(CLASSES)))
    ax.set_yticks(range(len(CLASSES)))
    ax.set_xticklabels(CLASSES, rotation=45, ha="right", fontsize=9)
    ax.set_yticklabels(CLASSES, fontsize=9)
    ax.set_xlabel("Predicted", fontsize=11)
    ax.set_ylabel("True", fontsize=11)
    ax.set_title("Confusion Matrix — LensGuard Test Set", fontsize=12)
    for i in range(len(CLASSES)):
        for j in range(len(CLASSES)):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=8)
    plt.tight_layout()
    cm_path = output_dir / "confusion.png"
    plt.savefig(str(cm_path), dpi=150)
    plt.close()
    print(f"Confusion matrix saved to {cm_path}")

    # ── ROC-AUC ───────────────────────────────────────────────────────────
    from sklearn.preprocessing import label_binarize
    y_true_bin = label_binarize(y_true, classes=list(range(len(CLASSES))))
    fig, ax = plt.subplots(figsize=(9, 7))
    auc_scores = {}
    for i, cls in enumerate(CLASSES):
        fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_probs_all[:, i])
        auc = roc_auc_score(y_true_bin[:, i], y_probs_all[:, i])
        auc_scores[cls] = float(auc)
        ax.plot(fpr, tpr, label=f"{cls} (AUC={auc:.3f})", linewidth=1.5)
    ax.plot([0, 1], [0, 1], "k--", linewidth=0.8)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves — Per Class")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    roc_path = output_dir / "roc_curves.png"
    plt.savefig(str(roc_path), dpi=150)
    plt.close()
    print(f"ROC curves saved to {roc_path}")

    # ── Save raw results ──────────────────────────────────────────────────
    results = {
        "n_test": int(len(y_true)),
        "accuracy": float(report["accuracy"]),
        "per_class": {cls: {k: float(v) for k, v in report[cls].items()} for cls in CLASSES if cls in report},
        "roc_auc": auc_scores,
        "macro_f1": float(report["macro avg"]["f1-score"]),
        "weighted_f1": float(report["weighted avg"]["f1-score"]),
    }
    with open(output_dir / "eval_results.json", "w") as f:
        json.dump(results, f, indent=2)

    # ── Write results.md ──────────────────────────────────────────────────
    md = f"""# LensGuard Evaluation Results

**Test set size**: {len(y_true)} images
**Overall accuracy**: {results['accuracy']:.3f}
**Macro F1**: {results['macro_f1']:.3f}
**Weighted F1**: {results['weighted_f1']:.3f}

## Per-Class Metrics

| Class | Precision | Recall | F1-Score | Support |
|---|---|---|---|---|
"""
    for cls in CLASSES:
        if cls in report:
            r = report[cls]
            md += f"| {cls} | {r['precision']:.3f} | {r['recall']:.3f} | {r['f1-score']:.3f} | {int(r['support'])} |\n"

    md += "\n## ROC-AUC Scores\n\n| Class | AUC |\n|---|---|\n"
    for cls, auc in auc_scores.items():
        md += f"| {cls} | {auc:.4f} |\n"

    md += f"""
## Visualisations

- Confusion matrix: `eval/confusion.png`
- ROC curves: `eval/roc_curves.png`

## Notes

- The model was trained on synthetic data (CIFAR-100 upscaled + programmatic degradation).
- Results reflect performance on held-out test images degraded with the same pipeline.
- For real-world performance, see `eval/failure_cases.md`.
"""
    results_path = output_dir / "results.md"
    with open(results_path, "w") as f:
        f.write(md)
    print(f"Results written to {results_path}")
    print(f"\n✓ Evaluation complete. Accuracy: {results['accuracy']:.3f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data/", help="Dataset root")
    parser.add_argument("--model-dir", default="ml/models/v1")
    parser.add_argument("--output-dir", default="eval/")
    args = parser.parse_args()
    run_evaluation(Path(args.data_dir), Path(args.model_dir), Path(args.output_dir))


if __name__ == "__main__":
    main()
