"""
src/data/loader.py
Dataset loading utilities for WeedyRice-RGBMS-DB
"""

import numpy as np
import os
import rasterio
from PIL import Image


def load_ids(root_dir, split="train"):
    """Load image IDs from train/val/test split files."""
    txt_path = os.path.join(root_dir, f"{split}_list.txt")
    with open(txt_path) as f:
        return [line.strip().replace(".JPG", "") for line in f.readlines()]


def load_sample(rgb_path, ms_dir, image_id):
    """
    Load one complete RGB + MS band sample.

    Args:
        rgb_path:  Path to RGB image
        ms_dir:    Directory containing MS TIF files
        image_id:  Base filename without extension

    Returns:
        rgb:   (H, W, 3) uint8 numpy array
        bands: dict with keys G, R, RE, NIR — (H, W) float32 arrays
    """
    rgb = np.array(Image.open(rgb_path).convert("RGB"))

    bands = {}
    for band in ["G", "R", "RE", "NIR"]:
        band_path = os.path.join(ms_dir, f"{image_id}_{band}.TIF")
        bands[band] = load_band_file(band_path)

    # Ensure bands match RGB resolution
    H, W = rgb.shape[:2]
    import cv2
    for k in bands:
        if bands[k].shape != (H, W):
            bands[k] = cv2.resize(bands[k], (W, H)).astype(np.float32)

    return rgb, bands


def load_band_file(path):
    """Load a single band TIF file."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Band file not found: {path}")
    with rasterio.open(path) as src:
        return src.read(1).astype(np.float32)


def load_band(file_obj, fallback_arr):
    """
    Load band from file object (for Gradio) or return fallback array.
    Used in the Gradio app when MS bands are not provided.
    """
    if file_obj is None:
        return fallback_arr.astype(np.float32)
    try:
        with rasterio.open(file_obj.name) as src:
            return src.read(1).astype(np.float32)
    except Exception:
        return np.array(
            Image.open(file_obj.name).convert("L")
        ).astype(np.float32)


def load_mask(mask_path):
    """Load binary ground truth mask. Returns (H, W) uint8 array."""
    mask = np.array(Image.open(mask_path))
    return (mask == 255).astype(np.uint8)


def scan_valid_images(root_dir, split="test", min_weed_pct=10.0):
    """
    Scan dataset split and return IDs with sufficient weed coverage.

    Args:
        root_dir:      Dataset root directory
        split:         Which split to scan
        min_weed_pct:  Minimum weed coverage threshold

    Returns:
        valid_ids:   List of image IDs above threshold
        all_stats:   List of dicts with image_id and weed_pct
    """
    ids       = load_ids(root_dir, split)
    mask_dir  = os.path.join(root_dir, "Masks")
    valid_ids = []
    all_stats = []

    for img_id in ids:
        mask_path = os.path.join(mask_dir, f"{img_id}.png")
        if not os.path.exists(mask_path):
            continue
        mask     = load_mask(mask_path)
        weed_pct = mask.mean() * 100
        all_stats.append({"image_id": img_id, "weed_pct": weed_pct})
        if weed_pct >= min_weed_pct:
            valid_ids.append(img_id)

    return valid_ids, all_stats
