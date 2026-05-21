"""
src/evaluate/metrics.py
Segmentation evaluation metrics.
"""

import numpy as np
from sklearn.metrics import (
    jaccard_score, f1_score,
    precision_score, recall_score
)


def evaluate(pred_mask, true_mask):
    """
    Compute segmentation metrics against ground truth.

    Args:
        pred_mask: (H, W) uint8 predicted binary mask
        true_mask: (H, W) uint8 ground truth binary mask

    Returns:
        dict with IoU, F1, Precision, Recall, PixelAcc
    """
    p = pred_mask.flatten().astype(int)
    t = true_mask.flatten().astype(int)

    return {
        "IoU":       round(jaccard_score(t, p, zero_division=0), 3),
        "F1":        round(f1_score(t, p, zero_division=0), 3),
        "Precision": round(precision_score(t, p, zero_division=0), 3),
        "Recall":    round(recall_score(t, p, zero_division=0), 3),
        "PixelAcc":  round(float((p == t).mean()), 3),
    }


def moran_i(residuals, weight_matrix):
    """
    Compute Moran's I spatial autocorrelation on model residuals.
    High positive value = errors cluster spatially (bad).
    Near zero = errors random in space (good — model captured spatial patterns).

    Args:
        residuals:      1D array of prediction errors
        weight_matrix:  Spatial weight matrix (e.g. from libpysal)

    Returns:
        moran_i_value: float
        p_value:        float
    """
    try:
        from esda.moran import Moran
        moran = Moran(residuals, weight_matrix)
        return moran.I, moran.p_sim
    except ImportError:
        print("Install esda for Moran's I: pip install esda libpysal")
        return None, None


def compute_boundary_f1(pred, true, tolerance=3):
    """
    Boundary F1 score — penalizes rough or inaccurate mask edges.
    More meaningful than pixel accuracy for evaluating segmentation quality.

    Args:
        pred:      (H, W) binary predicted mask
        true:      (H, W) binary ground truth mask
        tolerance: Boundary tolerance in pixels

    Returns:
        boundary_f1: float
    """
    import cv2

    pred_boundary = cv2.Canny(pred.astype(np.uint8) * 255, 50, 150)
    true_boundary = cv2.Canny(true.astype(np.uint8) * 255, 50, 150)

    kernel       = np.ones((tolerance * 2 + 1, tolerance * 2 + 1), np.uint8)
    true_dilated = cv2.dilate(true_boundary, kernel)
    pred_dilated = cv2.dilate(pred_boundary, kernel)

    tp = np.logical_and(pred_boundary > 0, true_dilated > 0).sum()
    fp = np.logical_and(pred_boundary > 0, true_dilated == 0).sum()
    fn = np.logical_and(true_boundary > 0, pred_dilated == 0).sum()

    precision = tp / (tp + fp + 1e-10)
    recall    = tp / (tp + fn + 1e-10)

    return round(2 * precision * recall / (precision + recall + 1e-10), 3)


def summarize_results(all_metrics):
    """Compute mean and std of metrics across multiple images."""
    keys = ["IoU", "F1", "Precision", "Recall"]
    summary = {}
    for k in keys:
        vals = [m[k] for m in all_metrics if k in m]
        summary[k] = {
            "mean": round(np.mean(vals), 3),
            "std":  round(np.std(vals), 3),
            "min":  round(np.min(vals), 3),
            "max":  round(np.max(vals), 3),
        }
    return summary
