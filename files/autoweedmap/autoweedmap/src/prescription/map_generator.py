"""
src/prescription/map_generator.py
Variable-rate herbicide prescription map generation.
"""

import numpy as np
import cv2


# Dose levels in liters per hectare
DOSE_LEVELS = {
    "none":   0,
    "low":    150,
    "medium": 300,
    "high":   450,
}

# Weed density thresholds
THRESHOLDS = [0.05, 0.25, 0.50]

# Colors (RGB)
ZONE_COLORS = {
    "none":   [200, 230, 200],  # green
    "low":    [255, 255, 100],  # yellow
    "medium": [255, 165,   0],  # orange
    "high":   [220,  50,  50],  # red
}


def generate_prescription(pred_mask,
                           cell_size=64,
                           pixel_size_m=0.05,
                           smooth=True):
    """
    Convert weed detection mask to variable-rate prescription map.

    Args:
        pred_mask:     (H, W) uint8 binary weed mask
        cell_size:     Grid cell size in pixels
        pixel_size_m:  Ground resolution in meters/pixel (DJI Mavic 3M default)
        smooth:        Whether to apply Gaussian smoothing to final map

    Returns:
        prescription_rgb: (H, W, 3) uint8 color-coded prescription map
        savings_pct:      Float — herbicide saved vs uniform application
        weed_coverage:    Float — % of image covered by weeds
        zone_stats:       Dict with zone areas and doses
    """
    H, W      = pred_mask.shape
    dose_map  = np.zeros((H, W), dtype=np.uint8)
    label_map = np.empty((H, W), dtype=object)

    zone_counts = {"none": 0, "low": 0, "medium": 0, "high": 0}

    # ── Compute density per grid cell ─────────────────────────
    for r in range(0, H, cell_size):
        for c in range(0, W, cell_size):
            cell    = pred_mask[r:r+cell_size, c:c+cell_size]
            density = cell.mean()

            if density < THRESHOLDS[0]:
                zone = "none"
            elif density < THRESHOLDS[1]:
                zone = "low"
            elif density < THRESHOLDS[2]:
                zone = "medium"
            else:
                zone = "high"

            dose = DOSE_LEVELS[zone]
            dose_map[r:r+cell_size, c:c+cell_size]  = dose
            label_map[r:r+cell_size, c:c+cell_size] = zone
            zone_counts[zone] += cell.size

    # ── Color map ─────────────────────────────────────────────
    colors = np.zeros((H, W, 3), dtype=np.uint8)
    for zone, color in ZONE_COLORS.items():
        colors[label_map == zone] = color

    # ── Optional smoothing ────────────────────────────────────
    if smooth:
        colors = cv2.GaussianBlur(colors, (15, 15), 0)

    # ── Compute savings ───────────────────────────────────────
    total_pixels = H * W
    uniform_dose = DOSE_LEVELS["high"] * total_pixels
    variable_dose = dose_map.sum()
    savings_pct   = (1 - variable_dose / max(uniform_dose, 1)) * 100
    weed_coverage = pred_mask.mean() * 100

    # ── Field area calculations ───────────────────────────────
    cell_area_m2   = (cell_size * pixel_size_m) ** 2
    field_area_ha  = (total_pixels * pixel_size_m ** 2) / 10000

    zone_stats = {}
    for zone in ["none", "low", "medium", "high"]:
        area_m2  = zone_counts[zone] * pixel_size_m ** 2
        area_ha  = area_m2 / 10000
        dose_lha = DOSE_LEVELS[zone]
        volume_L = dose_lha * area_ha
        zone_stats[zone] = {
            "area_ha":   round(area_ha, 4),
            "dose_lha":  dose_lha,
            "volume_L":  round(volume_L, 2),
            "pct_field": round(zone_counts[zone] / total_pixels * 100, 1)
        }

    # Summary
    uniform_volume   = DOSE_LEVELS["high"] * field_area_ha
    variable_volume  = sum(z["volume_L"] for z in zone_stats.values())
    herbicide_saved_L = uniform_volume - variable_volume

    zone_stats["summary"] = {
        "field_area_ha":      round(field_area_ha, 4),
        "weed_coverage_pct":  round(weed_coverage, 1),
        "uniform_volume_L":   round(uniform_volume, 1),
        "variable_volume_L":  round(variable_volume, 1),
        "herbicide_saved_L":  round(herbicide_saved_L, 1),
        "herbicide_saved_pct": round(savings_pct, 1),
    }

    return colors, savings_pct, weed_coverage, zone_stats


def prescription_to_geotiff(prescription_rgb, output_path,
                              transform=None, crs=None):
    """
    Save prescription map as georeferenced GeoTIFF.

    Args:
        prescription_rgb: (H, W, 3) uint8 prescription map
        output_path:      Output file path (.tif)
        transform:        Rasterio transform (if available)
        crs:              Coordinate reference system string
    """
    import rasterio
    from rasterio.transform import from_bounds

    H, W = prescription_rgb.shape[:2]

    if transform is None:
        # Default transform if no GPS available
        transform = from_bounds(0, 0, W, H, W, H)

    with rasterio.open(
        output_path, mode="w",
        driver="GTiff",
        height=H, width=W,
        count=3,
        dtype=np.uint8,
        crs=crs or "EPSG:32648",
        transform=transform,
    ) as dst:
        for i in range(3):
            dst.write(prescription_rgb[:, :, i], i + 1)

    print(f"Prescription map saved: {output_path}")
