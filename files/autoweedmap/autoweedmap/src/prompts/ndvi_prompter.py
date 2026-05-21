"""
src/prompts/ndvi_prompter.py
NDVI-guided automatic SAM prompt generation.

Key insight: Weedy rice has locally lower NDVI than surrounding
cultivated rice. We detect these anomalous pixels automatically
and use them as SAM point prompts — zero human clicks required.
"""

import numpy as np
import cv2
from scipy import ndimage
from scipy.ndimage import binary_erosion, uniform_filter
from sklearn.cluster import DBSCAN


def generate_prompts(indices, bands,
                     n_weed_prompts=30,
                     n_bg_prompts=15):
    """
    Generate SAM point prompts automatically using spectral analysis.

    Strategy:
        1. Suppress non-vegetation (irrigation channels, soil)
        2. Estimate infestation level → choose scoring strategy
        3. Compute local NDVI + NDRE anomaly maps
        4. Cluster high-score regions → one prompt per weed patch
        5. Sample background prompts from low-score vegetation

    Args:
        indices:        dict from compute_indices()
        bands:          dict of MS band arrays
        n_weed_prompts: max number of foreground prompts
        n_bg_prompts:   max number of background prompts

    Returns:
        point_coords:  (N, 2) array of (x, y) prompt coordinates
        point_labels:  (N,) array — 1=weed, 0=background
        weed_score:    (H, W) float array — weed probability map
        high_infest:   bool — whether high infestation detected
    """
    ndvi = indices["NDVI"]
    ndre = indices["NDRE"]
    NIR  = bands["NIR"]

    def norm(arr):
        mn, mx = np.percentile(arr, [2, 98])
        return np.clip((arr - mn) / (mx - mn + 1e-10), 0, 1)

    ndvi_n = norm(ndvi)
    ndre_n = norm(ndre)
    nir_n  = norm(NIR)

    # ── Suppress non-vegetation ───────────────────────────────
    not_veg  = nir_n < np.percentile(nir_n, 8)
    veg_mask = binary_erosion(~not_veg,
                               structure=np.ones((3, 3)),
                               iterations=2)

    # ── Estimate infestation level ────────────────────────────
    veg_ndvi    = ndvi_n[veg_mask]
    high_infest = (veg_ndvi < np.percentile(veg_ndvi, 40)).mean() > 0.35

    # ── Local NDVI anomaly ────────────────────────────────────
    local_mean   = uniform_filter(ndvi_n, size=31)
    local_std    = np.sqrt(np.abs(
        uniform_filter(ndvi_n**2, size=31) - local_mean**2) + 1e-10)
    ndvi_anomaly = np.clip((local_mean - ndvi_n) /
                            (local_std + 1e-10), 0, None)

    # ── Local NDRE anomaly ────────────────────────────────────
    local_mean_re = uniform_filter(ndre_n, size=31)
    local_std_re  = np.sqrt(np.abs(
        uniform_filter(ndre_n**2, size=31) - local_mean_re**2) + 1e-10)
    ndre_anomaly  = np.clip((local_mean_re - ndre_n) /
                             (local_std_re + 1e-10), 0, None)

    # ── Global low-NDVI score ─────────────────────────────────
    global_low = norm(1 - ndvi_n)

    # ── Combine signals ───────────────────────────────────────
    if high_infest:
        # High infestation: global NDVI dominates
        weed_score = (0.55 * global_low +
                      0.25 * norm(ndvi_anomaly) +
                      0.20 * norm(ndre_anomaly))
    else:
        # Moderate infestation: local anomaly dominates
        weed_score = (0.20 * global_low +
                      0.45 * norm(ndvi_anomaly) +
                      0.35 * norm(ndre_anomaly))

    weed_score[~veg_mask] = 0

    # ── Build candidate mask ──────────────────────────────────
    valid     = weed_score[weed_score > 0]
    if len(valid) == 0:
        return None, None, weed_score, high_infest

    threshold = np.percentile(valid, 70)
    cand_mask = (weed_score > threshold).astype(np.uint8)
    kernel    = np.ones((7, 7), np.uint8)
    cand_mask = cv2.morphologyEx(cand_mask, cv2.MORPH_OPEN,   kernel)
    cand_mask = cv2.morphologyEx(cand_mask, cv2.MORPH_DILATE,
                                  np.ones((5, 5), np.uint8))
    cand_coords = np.column_stack(np.where(cand_mask > 0))

    # ── Weed prompts via DBSCAN clustering ────────────────────
    weed_prompts = []
    if len(cand_coords) > 10:
        labels = DBSCAN(eps=20, min_samples=8).fit_predict(cand_coords)
        info   = {}
        for label in set(labels) - {-1}:
            cluster    = cand_coords[labels == label]
            mean_score = weed_score[cluster[:, 0], cluster[:, 1]].mean()
            info[label] = {
                "priority": len(cluster) * mean_score,
                "coords":   cluster
            }
        for label in sorted(info,
                             key=lambda l: info[l]["priority"],
                             reverse=True):
            cluster  = info[label]["coords"]
            scores_c = weed_score[cluster[:, 0], cluster[:, 1]]
            best_px  = cluster[scores_c.argmax()]
            weed_prompts.append([best_px[1], best_px[0]])  # (x, y)

    # Fallback: top scoring pixels
    if not weed_prompts:
        idx  = np.argsort(weed_score.flatten())[-n_weed_prompts:]
        r, c = np.unravel_index(idx, weed_score.shape)
        weed_prompts = [[c_, r_] for r_, c_ in zip(r, c)]

    weed_prompts = np.array(weed_prompts[:n_weed_prompts])

    # ── Background prompts ────────────────────────────────────
    low_weed  = weed_score < np.percentile(valid, 25)
    bg_valid  = veg_mask & low_weed
    bg_coords = np.column_stack(np.where(bg_valid))

    filtered = []
    for coord in bg_coords[::10]:
        if len(weed_prompts) > 0:
            dists    = np.sqrt(
                ((coord[0] - weed_prompts[:, 1])**2 +
                 (coord[1] - weed_prompts[:, 0])**2))
            min_dist = dists.min()
        else:
            min_dist = 999
        if min_dist > 60:
            filtered.append(coord)

    filtered = np.array(filtered) if filtered else bg_coords

    if len(filtered) >= n_bg_prompts:
        idx      = np.random.choice(len(filtered), n_bg_prompts, replace=False)
        bg_pts   = filtered[idx][:, [1, 0]]
    else:
        bg_pts   = filtered[:, [1, 0]]

    point_coords = np.vstack([weed_prompts, bg_pts])
    point_labels = np.array([1] * len(weed_prompts) +
                             [0] * len(bg_pts))

    return point_coords, point_labels, weed_score, high_infest
