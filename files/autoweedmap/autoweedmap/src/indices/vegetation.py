"""
src/indices/vegetation.py
Vegetation index computation from multispectral bands
"""

import numpy as np
import cv2


def compute_indices(bands):
    """
    Compute all vegetation indices from MS bands.

    Args:
        bands: dict with G, R, RE, NIR keys (float32 arrays)

    Returns:
        dict of vegetation indices
    """
    R, G, RE, NIR = bands["R"], bands["G"], bands["RE"], bands["NIR"]
    eps = 1e-10

    return {
        # Standard NDVI — most common
        "NDVI":  (NIR - R)  / (NIR + R  + eps),

        # Red-Edge NDVI — sensitive to early stress
        "NDRE":  (NIR - RE) / (NIR + RE + eps),

        # Green NDVI — chlorophyll sensitive
        "GNDVI": (NIR - G)  / (NIR + G  + eps),

        # Modified Chlorophyll Absorption Ratio
        # Best for separating weedy from cultivated rice
        "MCARI": ((RE - R) - 0.2 * (RE - G)) * (RE / (R + eps)),

        # Ratio Vegetation Index
        "RVI":   NIR / (R + eps),

        # Normalized Difference Water Index
        "NDWI":  (G - NIR) / (G + NIR + eps),
    }


def normalize_to_uint8(arr):
    """Normalize float array to 0-255 uint8, clipping outliers."""
    p2, p98 = np.percentile(arr, [2, 98])
    arr     = np.clip(arr, p2, p98)
    return ((arr - p2) / (p98 - p2 + 1e-10) * 255).astype(np.uint8)


def compose_mcari(rgb, bands):
    """
    Create MCARI-NDVI-NDRE false color composite for SAM input.
    This combination outperforms RGB for weedy rice segmentation.

    Returns:
        (H, W, 3) uint8 array — SAM-compatible input
    """
    NIR, R, RE, G = bands["NIR"], bands["R"], bands["RE"], bands["G"]
    eps = 1e-10

    mcari = ((RE - R) - 0.2 * (RE - G)) * (RE / (R + eps))
    ndvi  = (NIR - R)  / (NIR + R  + eps)
    ndre  = (NIR - RE) / (NIR + RE + eps)

    return np.stack([
        normalize_to_uint8(mcari),
        normalize_to_uint8(ndvi),
        normalize_to_uint8(ndre)
    ], axis=-1)


def compose_false_color(bands):
    """NIR-Red-Green false color composite."""
    return np.stack([
        normalize_to_uint8(bands["NIR"]),
        normalize_to_uint8(bands["R"]),
        normalize_to_uint8(bands["G"])
    ], axis=-1)


def compose_color_infrared(bands):
    """NIR-RedEdge-Red composite — maximizes chlorophyll differences."""
    return np.stack([
        normalize_to_uint8(bands["NIR"]),
        normalize_to_uint8(bands["RE"]),
        normalize_to_uint8(bands["R"])
    ], axis=-1)


def ndvi_to_rgb(ndvi_8bit):
    """
    Convert NDVI grayscale to red-green colormap.
    Compatible with all OpenCV versions (no COLORMAP_RdYlGn needed).
    """
    lut = np.zeros((256, 1, 3), dtype=np.uint8)
    for i in range(256):
        t = i / 255.0
        if t < 0.5:
            lut[i, 0] = [0, int(t * 2 * 255), 220]     # BGR: red
        else:
            lut[i, 0] = [0, 220, int((1 - t) * 2 * 255)]  # BGR: green
    gray3   = cv2.cvtColor(ndvi_8bit, cv2.COLOR_GRAY2BGR)
    colored = cv2.LUT(gray3, lut)
    return cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)


BAND_MODES = {
    "rgb":            lambda rgb, bands: rgb,
    "false_color":    lambda rgb, bands: compose_false_color(bands),
    "color_infrared": lambda rgb, bands: compose_color_infrared(bands),
    "mcari_composite":lambda rgb, bands: compose_mcari(rgb, bands),
    "ndvi":           lambda rgb, bands: np.stack(
        [normalize_to_uint8(((bands["NIR"]-bands["R"])/
         (bands["NIR"]+bands["R"]+1e-10)))]*3, axis=-1),
    "ndre":           lambda rgb, bands: np.stack(
        [normalize_to_uint8(((bands["NIR"]-bands["RE"])/
         (bands["NIR"]+bands["RE"]+1e-10)))]*3, axis=-1),
}
