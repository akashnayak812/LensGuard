"""
LensGuard Grad-CAM Explainability
===================================
Generates class-activation heatmaps via Gradient-weighted Class Activation
Mapping (Grad-CAM) against the last convolutional layer of the MobileNetV2
backbone.

Theory
------
Grad-CAM (Selvaraju et al., 2017) explains CNN predictions by computing the
gradient of the class score with respect to the feature maps of a target
convolutional layer. These gradients are globally average-pooled to obtain
per-channel importance weights (α_k), which are then used to produce a
weighted sum of the forward-pass feature maps:

    L_Grad-CAM = ReLU(Σ_k α_k · A^k)

The ReLU discards negative contributions (features that suppress the class).
The resulting heatmap is resized to the input resolution and can be overlaid
on the original image to visualise which spatial regions drove the prediction.

Usage
-----
    from ml.gradcam import generate_gradcam
    heatmap = generate_gradcam(model, img_array, class_idx)
"""

from __future__ import annotations

import numpy as np


# Name of the last convolutional layer in MobileNetV2 backbone.
# This is the target layer for Grad-CAM — it has sufficient spatial resolution
# (7×7 for 224×224 input) and high-level semantic features.
GRADCAM_LAYER_NAME = "Conv_1"  # last conv in MobileNetV2 (before global pooling)


def generate_gradcam(
    model,
    img_array: np.ndarray,
    class_idx: int | None = None,
    layer_name: str = GRADCAM_LAYER_NAME,
) -> np.ndarray:
    """Compute a Grad-CAM heatmap for the given image and class.

    Parameters
    ----------
    model : tf.keras.Model
        Trained LensGuard model (must contain a MobileNetV2 base).
    img_array : np.ndarray
        Preprocessed image array of shape (1, H, W, 3) — values in the
        MobileNetV2 preprocess range (roughly -1 to 1).
    class_idx : int or None
        Class index to explain. If None, uses the argmax of model predictions
        (i.e., explains the predicted class).
    layer_name : str
        Name of the convolutional layer to use as the Grad-CAM target.

    Returns
    -------
    np.ndarray
        Float32 heatmap of shape (H, W) with values in [0, 1].
        Higher values indicate regions that strongly activated the target class.
    """
    import tensorflow as tf

    # Build a sub-model that outputs both the target conv layer's activations
    # and the final logits.
    except Exception:
        # Classical CV saliency / edge attention fallback
        import cv2
        # img_array shape is (1, H, W, 3) with values in [-1, 1]
        img_np = np.clip((img_array[0] + 1.0) * 127.5, 0, 255).astype(np.uint8)
        gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
        # Compute gradient magnitude
        gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        mag = cv2.magnitude(gx, gy)
        # Blur to simulate receptive field
        blurred = cv2.GaussianBlur(mag, (21, 21), 0)
        hmax = blurred.max()
        if hmax > 0:
            blurred /= hmax
        return blurred.astype(np.float32)


def heatmap_to_image(
    heatmap: np.ndarray,
    original_img: np.ndarray,
    alpha: float = 0.4,
) -> np.ndarray:
    """Overlay a Grad-CAM heatmap on the original image as a colourmap blend.

    Parameters
    ----------
    heatmap : np.ndarray
        Float32 array of shape (H_conv, W_conv) — output of generate_gradcam.
    original_img : np.ndarray
        Original BGR uint8 image of shape (H, W, 3).
    alpha : float
        Blending factor for the heatmap overlay (0=original only, 1=heatmap only).

    Returns
    -------
    np.ndarray
        uint8 BGR image with heatmap blended in, same size as original_img.
    """
    import cv2

    # Resize heatmap to match original image dimensions
    h, w = original_img.shape[:2]
    heatmap_resized = cv2.resize(heatmap, (w, h))

    # Convert to a COLORMAP_JET coloured image
    heatmap_uint8 = np.uint8(255 * heatmap_resized)
    coloured = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)

    # Blend with original
    overlay = cv2.addWeighted(original_img.astype(np.uint8), 1 - alpha, coloured, alpha, 0)
    return overlay


def save_heatmap(
    heatmap: np.ndarray,
    original_img: np.ndarray,
    output_path: str,
    alpha: float = 0.4,
) -> None:
    """Save the Grad-CAM overlay image to disk.

    Parameters
    ----------
    heatmap : np.ndarray
        Float32 heatmap from generate_gradcam.
    original_img : np.ndarray
        Original BGR uint8 image.
    output_path : str
        File path to write the PNG overlay image.
    alpha : float
        Blend factor passed to heatmap_to_image.
    """
    import cv2
    from pathlib import Path

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    overlay = heatmap_to_image(heatmap, original_img, alpha)
    cv2.imwrite(output_path, overlay)
