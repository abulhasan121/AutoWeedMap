"""
src/segmentation/sam_segmenter.py
Patch-based SAM inference with spectral validation.
"""

import numpy as np
import torch
from segment_anything import sam_model_registry, SamPredictor


def load_sam(checkpoint, model_type="vit_b", device=None):
    """Load and return a SAM predictor."""
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    sam    = sam_model_registry[model_type](checkpoint=checkpoint)
    sam.to(device)
    return SamPredictor(sam)


def run_sam_patches(predictor, input_image, point_coords,
                    point_labels, weed_score,
                    patch_size=192, iou_threshold=0.6):
    """
    Run SAM on patches centered around each weed prompt.

    Why patch-based?
    Whole-image SAM tends to segment large coherent regions
    (e.g., the entire side of a field). Patch-based inference
    forces SAM to focus on local weed patches, dramatically
    improving precision.

    Args:
        predictor:     SAM SamPredictor instance
        input_image:   (H, W, 3) uint8 — MCARI composite or RGB
        point_coords:  (N, 2) prompt coordinates (x, y)
        point_labels:  (N,) — 1=weed, 0=background
        weed_score:    (H, W) float weed probability map
        patch_size:    Square patch size in pixels
        iou_threshold: Minimum SAM confidence to accept a mask

    Returns:
        final_mask: (H, W) uint8 binary mask — 1=weedy rice
    """
    H, W       = input_image.shape[:2]
    final_mask = np.zeros((H, W), dtype=np.uint8)
    half       = patch_size // 2
    weed_pts   = point_coords[point_labels == 1]

    for px, py in weed_pts:
        px, py = int(px), int(py)

        x1 = max(0, px - half); x2 = min(W, px + half)
        y1 = max(0, py - half); y2 = min(H, py + half)

        patch = input_image[y1:y2, x1:x2]
        if patch.shape[0] < 32 or patch.shape[1] < 32:
            continue

        local_x = px - x1
        local_y = py - y1

        predictor.set_image(patch)
        masks, scores, _ = predictor.predict(
            point_coords=np.array([[local_x, local_y]]),
            point_labels=np.array([1]),
            multimask_output=True
        )

        best_idx = np.argmax(scores)
        if scores[best_idx] < iou_threshold:
            continue

        patch_mask = masks[best_idx].astype(np.uint8)

        # Spectral validation: mask must overlap with high weed score
        ws_patch = weed_score[y1:y2, x1:x2]
        ws_norm  = (ws_patch - ws_patch.min()) / \
                    (ws_patch.max() - ws_patch.min() + 1e-10)

        if ws_norm[patch_mask == 1].mean() < 0.15:
            continue  # skip masks in low-score regions

        final_mask[y1:y2, x1:x2] = np.maximum(
            final_mask[y1:y2, x1:x2], patch_mask
        )

    return final_mask


def run_sam_box(predictor, image, boxes, weed_score,
                ndvi_anomaly, iou_threshold=0.6):
    """
    Run SAM with bounding box prompts (for Grounded SAM integration).

    Args:
        predictor:    SAM predictor
        image:        (H, W, 3) uint8 input image
        boxes:        List of [x1, y1, x2, y2] boxes
        weed_score:   (H, W) spectral weed score
        ndvi_anomaly: (H, W) NDVI anomaly map
        iou_threshold: Minimum SAM confidence

    Returns:
        final_mask: (H, W) uint8 binary mask
    """
    H, W       = image.shape[:2]
    final_mask = np.zeros((H, W), dtype=np.uint8)

    predictor.set_image(image)

    valid          = weed_score[weed_score > 0]
    spectral_thresh = np.percentile(valid, 60) if len(valid) > 0 else 0.3

    for x1, y1, x2, y2 in boxes:
        sam_box = np.array([x1, y1, x2, y2])
        masks, scores, _ = predictor.predict(
            box=sam_box,
            multimask_output=True
        )
        best_idx = np.argmax(scores)
        if scores[best_idx] < iou_threshold:
            continue

        box_mask = masks[best_idx].astype(np.uint8)

        # Spectral check
        if box_mask.sum() > 0:
            anom = ndvi_anomaly[box_mask == 1].mean()
            if anom < spectral_thresh * 0.5:
                continue

        final_mask = np.maximum(final_mask, box_mask)

    return final_mask
