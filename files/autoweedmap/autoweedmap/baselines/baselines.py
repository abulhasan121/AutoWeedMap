"""
baselines/baselines.py
Baseline methods for comparison against AutoWeedMap.

Three baselines:
1. NDVI Threshold — pure spectral thresholding, no deep learning
2. SAM + Grid Prompts — SAM with uniform grid, no spectral guidance
3. Grounded SAM — language-guided detection on RGB (proves RGB insufficient)
"""

import numpy as np
import cv2
from scipy.ndimage import uniform_filter

from src.indices.vegetation import normalize_to_uint8
from src.segmentation.sam_segmenter import run_sam_patches


# ── Baseline 1: NDVI Threshold ────────────────────────────────

def ndvi_threshold_baseline(bands, percentile_threshold=75):
    """
    Simplest possible approach — threshold NDVI directly.
    No SAM, no deep learning, no prompts.
    Low NDVI within vegetation = weedy rice.

    This is the minimum bar AutoWeedMap must beat.

    Args:
        bands:                MS band dict
        percentile_threshold: Percentage of vegetation pixels to flag as weed

    Returns:
        weed_mask: (H, W) uint8 binary mask
    """
    R   = bands["R"]
    NIR = bands["NIR"]
    eps = 1e-10

    ndvi    = (NIR - R) / (NIR + R + eps)
    nir_n   = (NIR - NIR.min()) / (NIR.max() - NIR.min() + eps)
    not_veg = nir_n < np.percentile(nir_n, 8)

    ndvi_veg             = ndvi.copy()
    ndvi_veg[not_veg]    = ndvi.max()  # suppress non-veg

    threshold = np.percentile(
        ndvi_veg[~not_veg], 100 - percentile_threshold
    )
    weed_mask = (ndvi_veg < threshold) & (~not_veg)

    return weed_mask.astype(np.uint8)


# ── Baseline 2: SAM + Grid Prompts ───────────────────────────

def grid_prompt_baseline(predictor, input_image, weed_score,
                          grid_size=6, patch_size=192,
                          iou_threshold=0.6):
    """
    SAM with uniform grid prompts — no spectral guidance.
    Regular grid regardless of where weeds are.

    Demonstrates that NDVI guidance improves over random prompting.

    Args:
        predictor:    SAM predictor
        input_image:  (H, W, 3) input image
        weed_score:   (H, W) weed score (used for spectral validation only)
        grid_size:    Number of grid divisions per axis
        patch_size:   SAM patch size
        iou_threshold: SAM confidence threshold

    Returns:
        weed_mask: (H, W) uint8 binary mask
    """
    H, W = input_image.shape[:2]

    rows = np.linspace(H // 8, 7 * H // 8, grid_size).astype(int)
    cols = np.linspace(W // 8, 7 * W // 8, grid_size).astype(int)

    grid_pts = []
    for r in rows:
        for c in cols:
            grid_pts.append([c, r])  # (x, y)

    point_coords = np.array(grid_pts)
    point_labels = np.ones(len(point_coords), dtype=int)

    return run_sam_patches(
        predictor, input_image,
        point_coords, point_labels, weed_score,
        patch_size=patch_size,
        iou_threshold=iou_threshold
    )


# ── Baseline 3: Grounded SAM (RGB only) ──────────────────────

def grounded_sam_baseline(gdino_model, predictor, rgb, bands,
                           text_prompt="weedy rice . weed patch . grass weed",
                           box_threshold=0.25,
                           text_threshold=0.20):
    """
    Language-guided detection (Grounding DINO) + SAM segmentation.

    Key finding: Achieves near-zero IoU on weedy rice because
    weedy and cultivated rice are visually identical in RGB.
    This proves that multispectral input is necessary.

    Requires: pip install groundingdino-py

    Args:
        gdino_model:    Loaded Grounding DINO model
        predictor:      SAM predictor
        rgb:            (H, W, 3) uint8 RGB image
        bands:          MS band dict (for spectral validation)
        text_prompt:    Natural language description of target
        box_threshold:  Detection confidence threshold
        text_threshold: Text similarity threshold

    Returns:
        final_mask:  (H, W) uint8 binary mask
        n_total:     Total boxes detected
        n_validated: Boxes passing spectral filter
    """
    from PIL import Image as PILImage
    from groundingdino.util.inference import load_image, predict

    H, W = rgb.shape[:2]
    eps  = 1e-10

    # Compute NDVI anomaly for spectral validation
    NIR          = bands["NIR"]
    R            = bands["R"]
    ndvi         = (NIR - R) / (NIR + R + eps)
    ndvi_n       = (ndvi - ndvi.min()) / (ndvi.max() - ndvi.min() + eps)
    local_mean   = uniform_filter(ndvi_n, size=31)
    ndvi_anomaly = np.clip(local_mean - ndvi_n, 0, None)
    ndvi_anomaly = (ndvi_anomaly - ndvi_anomaly.min()) / \
                    (ndvi_anomaly.max() - ndvi_anomaly.min() + eps)

    # Save temp RGB for Grounding DINO
    import tempfile
    import os
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
        tmp_path = tmp.name
    PILImage.fromarray(rgb).save(tmp_path)

    image_source, image_tensor = load_image(tmp_path)
    os.unlink(tmp_path)

    boxes, logits, phrases = predict(
        model=gdino_model,
        image=image_tensor,
        caption=text_prompt,
        box_threshold=box_threshold,
        text_threshold=text_threshold,
        device="cuda" if __import__("torch").cuda.is_available() else "cpu"
    )

    if len(boxes) == 0:
        return np.zeros((H, W), dtype=np.uint8), 0, 0

    # Convert to pixel coordinates
    boxes_xyxy = []
    for box in boxes:
        cx, cy, bw, bh = box.numpy()
        x1 = max(0, int((cx - bw/2) * W))
        y1 = max(0, int((cy - bh/2) * H))
        x2 = min(W, int((cx + bw/2) * W))
        y2 = min(H, int((cy + bh/2) * H))
        boxes_xyxy.append([x1, y1, x2, y2])

    # Spectral filter — relative threshold
    valid_anom       = ndvi_anomaly[ndvi_anomaly > 0]
    spectral_thresh  = np.percentile(valid_anom, 60) if len(valid_anom) > 0 else 0.3

    validated = []
    for box in boxes_xyxy:
        x1, y1, x2, y2 = box
        if x2 - x1 < 20 or y2 - y1 < 20:
            continue
        box_anom = ndvi_anomaly[y1:y2, x1:x2].mean()
        box_max  = ndvi_anomaly[y1:y2, x1:x2].max()
        if box_anom >= spectral_thresh or box_max >= 0.7:
            validated.append(box)

    # Fallback: keep all if none pass
    if not validated:
        validated = boxes_xyxy

    # SAM with box prompts
    from src.segmentation.sam_segmenter import run_sam_box
    weed_score_proxy = ndvi_anomaly  # use as weed score

    final_mask = run_sam_box(
        predictor, rgb, validated,
        weed_score_proxy, ndvi_anomaly
    )

    return final_mask, len(boxes_xyxy), len(validated)
