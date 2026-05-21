"""
AutoWeedMap — Gradio Deployment
Zero-Click Weedy Rice Detection & Herbicide Prescription Mapping
"""

import gradio as gr
import numpy as np
import cv2
import rasterio
import torch
import os
import tempfile
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from PIL import Image
from scipy import ndimage
from scipy.ndimage import binary_erosion, uniform_filter
from sklearn.cluster import DBSCAN
from sklearn.metrics import jaccard_score
from segment_anything import sam_model_registry, SamPredictor

# ── Model loading ──────────────────────────────────────────────
SAM_CHECKPOINT = "sam_vit_b_01ec64.pth"
DEVICE         = "cuda" if torch.cuda.is_available() else "cpu"

def load_sam_model():
    sam = sam_model_registry["vit_b"](checkpoint=SAM_CHECKPOINT)
    sam.to(DEVICE)
    return SamPredictor(sam)

predictor = load_sam_model()

# ── Core functions ─────────────────────────────────────────────
def normalize_to_uint8(arr):
    p2, p98 = np.percentile(arr, [2, 98])
    arr = np.clip(arr, p2, p98)
    return ((arr - p2) / (p98 - p2 + 1e-10) * 255).astype(np.uint8)

def compute_indices(bands):
    R, G, RE, NIR = bands["R"], bands["G"], bands["RE"], bands["NIR"]
    eps = 1e-10
    return {
        "NDVI":  (NIR - R)  / (NIR + R  + eps),
        "NDRE":  (NIR - RE) / (NIR + RE + eps),
        "GNDVI": (NIR - G)  / (NIR + G  + eps),
        "MCARI": ((RE - R) - 0.2*(RE - G)) * (RE / (R + eps)),
    }

def compose_mcari(rgb, bands):
    NIR, R, RE, G = bands["NIR"], bands["R"], bands["RE"], bands["G"]
    eps = 1e-10
    mcari = ((RE - R) - 0.2*(RE - G)) * (RE / (R + eps))
    ndvi  = (NIR - R)  / (NIR + R  + eps)
    ndre  = (NIR - RE) / (NIR + RE + eps)
    return np.stack([normalize_to_uint8(mcari),
                     normalize_to_uint8(ndvi),
                     normalize_to_uint8(ndre)], axis=-1)

def generate_prompts(indices, bands, n_weed=30, n_bg=15):
    ndvi = indices["NDVI"]
    ndre = indices["NDRE"]
    NIR  = bands["NIR"]

    def norm(arr):
        mn, mx = np.percentile(arr, [2, 98])
        return np.clip((arr - mn) / (mx - mn + 1e-10), 0, 1)

    ndvi_n = norm(ndvi)
    ndre_n = norm(ndre)
    nir_n  = norm(NIR)

    not_veg  = nir_n < np.percentile(nir_n, 8)
    veg_mask = binary_erosion(~not_veg, structure=np.ones((3,3)), iterations=2)

    veg_ndvi     = ndvi_n[veg_mask]
    high_infest  = (veg_ndvi < np.percentile(veg_ndvi, 40)).mean() > 0.35

    local_mean   = uniform_filter(ndvi_n, size=31)
    local_std    = np.sqrt(np.abs(uniform_filter(ndvi_n**2, size=31) - local_mean**2) + 1e-10)
    ndvi_anomaly = np.clip((local_mean - ndvi_n) / (local_std + 1e-10), 0, None)

    local_mean_re = uniform_filter(ndre_n, size=31)
    local_std_re  = np.sqrt(np.abs(uniform_filter(ndre_n**2, size=31) - local_mean_re**2) + 1e-10)
    ndre_anomaly  = np.clip((local_mean_re - ndre_n) / (local_std_re + 1e-10), 0, None)

    global_low = norm(1 - ndvi_n)

    if high_infest:
        weed_score = 0.55*global_low + 0.25*norm(ndvi_anomaly) + 0.20*norm(ndre_anomaly)
    else:
        weed_score = 0.20*global_low + 0.45*norm(ndvi_anomaly) + 0.35*norm(ndre_anomaly)

    weed_score[~veg_mask] = 0

    valid     = weed_score[weed_score > 0]
    if len(valid) == 0:
        return None, None, weed_score, high_infest

    threshold = np.percentile(valid, 70)
    cand      = (weed_score > threshold).astype(np.uint8)
    kernel    = np.ones((7,7), np.uint8)
    cand      = cv2.morphologyEx(cand, cv2.MORPH_OPEN,   kernel)
    cand      = cv2.morphologyEx(cand, cv2.MORPH_DILATE, np.ones((5,5), np.uint8))
    coords    = np.column_stack(np.where(cand > 0))

    weed_pts = []
    if len(coords) > 10:
        labels = DBSCAN(eps=20, min_samples=8).fit_predict(coords)
        info   = {}
        for l in set(labels) - {-1}:
            cl = coords[labels==l]
            ms = weed_score[cl[:,0], cl[:,1]].mean()
            info[l] = {"priority": len(cl)*ms, "coords": cl}
        for l in sorted(info, key=lambda x: info[x]["priority"], reverse=True):
            cl = info[l]["coords"]
            sc = weed_score[cl[:,0], cl[:,1]]
            bp = cl[sc.argmax()]
            weed_pts.append([bp[1], bp[0]])

    if not weed_pts:
        idx  = np.argsort(weed_score.flatten())[-n_weed:]
        r, c = np.unravel_index(idx, weed_score.shape)
        weed_pts = [[c_, r_] for r_, c_ in zip(r, c)]

    weed_pts = np.array(weed_pts[:n_weed])

    low_weed  = weed_score < np.percentile(valid, 25)
    bg_valid  = veg_mask & low_weed
    bg_coords = np.column_stack(np.where(bg_valid))

    filtered = []
    for coord in bg_coords[::10]:
        if len(weed_pts) > 0:
            d = np.sqrt(((coord[0]-weed_pts[:,1])**2 + (coord[1]-weed_pts[:,0])**2))
            if d.min() > 60:
                filtered.append(coord)
    filtered = np.array(filtered) if filtered else bg_coords

    if len(filtered) >= n_bg:
        idx      = np.random.choice(len(filtered), n_bg, replace=False)
        bg_pts   = filtered[idx][:, [1,0]]
    else:
        bg_pts   = filtered[:, [1,0]]

    pt_coords = np.vstack([weed_pts, bg_pts])
    pt_labels = np.array([1]*len(weed_pts) + [0]*len(bg_pts))

    return pt_coords, pt_labels, weed_score, high_infest

def run_sam_patches(predictor, image, pt_coords, pt_labels,
                    weed_score, patch_size=192, iou_thresh=0.6):
    H, W       = image.shape[:2]
    final_mask = np.zeros((H, W), dtype=np.uint8)
    half       = patch_size // 2
    weed_pts   = pt_coords[pt_labels == 1]

    for px, py in weed_pts:
        px, py = int(px), int(py)
        x1 = max(0, px-half); x2 = min(W, px+half)
        y1 = max(0, py-half); y2 = min(H, py+half)
        patch = image[y1:y2, x1:x2]
        if patch.shape[0] < 32 or patch.shape[1] < 32:
            continue
        predictor.set_image(patch)
        masks, scores, _ = predictor.predict(
            point_coords=np.array([[px-x1, py-y1]]),
            point_labels=np.array([1]),
            multimask_output=True
        )
        best = np.argmax(scores)
        if scores[best] < iou_thresh:
            continue
        pm     = masks[best].astype(np.uint8)
        ws_p   = weed_score[y1:y2, x1:x2]
        ws_n   = (ws_p - ws_p.min()) / (ws_p.max() - ws_p.min() + 1e-10)
        if ws_n[pm==1].mean() < 0.15:
            continue
        final_mask[y1:y2, x1:x2] = np.maximum(final_mask[y1:y2, x1:x2], pm)

    return final_mask

def generate_prescription(pred_mask, cell_size=64):
    H, W      = pred_mask.shape
    dose_map  = np.zeros((H, W), dtype=np.uint8)
    colors    = np.zeros((H, W, 3), dtype=np.uint8)

    for r in range(0, H, cell_size):
        for c in range(0, W, cell_size):
            cell    = pred_mask[r:r+cell_size, c:c+cell_size]
            density = cell.mean()
            if density < 0.05:
                dose, col = 0,   [200, 230, 200]
            elif density < 0.25:
                dose, col = 85,  [255, 255, 100]
            elif density < 0.50:
                dose, col = 170, [255, 165,   0]
            else:
                dose, col = 255, [220,  50,  50]
            dose_map[r:r+cell_size, c:c+cell_size] = dose
            colors[r:r+cell_size,   c:c+cell_size] = col

    uniform   = 255 * H * W
    variable  = dose_map.sum()
    savings   = (1 - variable / uniform) * 100
    coverage  = pred_mask.mean() * 100

    return colors, savings, coverage


# ── Main inference function ────────────────────────────────────
def run_pipeline(rgb_file, g_file, r_file, re_file, nir_file, progress=gr.Progress()):
    """Full AutoWeedMap inference pipeline."""

    if rgb_file is None:
        return [None]*6 + ["⚠️ Please upload an RGB image."]

    progress(0.1, desc="Loading images...")

    # Load RGB
    rgb = np.array(Image.open(rgb_file.name).convert("RGB"))

    # Load MS bands — use RGB as fallback if not provided
    def load_band(f, rgb_channel):
        if f is None:
            return rgb[:, :, rgb_channel].astype(np.float32)
        try:
            with rasterio.open(f.name) as src:
                return src.read(1).astype(np.float32)
        except:
            img = np.array(Image.open(f.name).convert("L"))
            return img.astype(np.float32)

    bands = {
        "G":   load_band(g_file,   1),
        "R":   load_band(r_file,   0),
        "RE":  load_band(re_file,  0),
        "NIR": load_band(nir_file, 0),
    }

    # Resize bands to match RGB if needed
    H, W = rgb.shape[:2]
    for k in bands:
        if bands[k].shape != (H, W):
            bands[k] = cv2.resize(bands[k], (W, H)).astype(np.float32)

    progress(0.25, desc="Computing vegetation indices...")

    indices    = compute_indices(bands)
    ndvi       = indices["NDVI"]
    ndvi_8bit  = normalize_to_uint8(ndvi)
    ndvi_color = cv2.applyColorMap(ndvi_8bit, cv2.COLORMAP_RdYlGn)
    ndvi_rgb   = cv2.cvtColor(ndvi_color, cv2.COLOR_BGR2RGB)

    progress(0.40, desc="Generating NDVI-guided prompts...")

    pt_coords, pt_labels, weed_score, high_infest = generate_prompts(indices, bands)

    if pt_coords is None:
        return [rgb, ndvi_rgb, None, None, None, None,
                "⚠️ Could not generate prompts. Check that MS bands are valid."]

    # Visualize prompts on RGB
    prompt_vis = rgb.copy()
    weed_pts   = pt_coords[pt_labels == 1]
    bg_pts     = pt_coords[pt_labels == 0]
    for px, py in weed_pts:
        cv2.drawMarker(prompt_vis, (int(px), int(py)), (255, 50, 50),
                       cv2.MARKER_STAR, 15, 2)
    for px, py in bg_pts:
        cv2.circle(prompt_vis, (int(px), int(py)), 6, (50, 100, 255), -1)

    progress(0.55, desc="Running SAM segmentation...")

    mcari_img  = compose_mcari(rgb, bands)
    pred_mask  = run_sam_patches(predictor, mcari_img,
                                  pt_coords, pt_labels, weed_score)

    progress(0.75, desc="Generating prescription map...")

    presc_map, savings, coverage = generate_prescription(pred_mask)

    # Detection overlay
    overlay      = rgb.copy()
    overlay[pred_mask == 1] = (overlay[pred_mask == 1] * 0.4 +
                                np.array([220, 50, 50]) * 0.6).astype(np.uint8)

    progress(0.90, desc="Preparing results...")

    # Stats string
    infest_level = "High" if high_infest else "Moderate"
    weed_pct     = coverage

    stats = f"""
## 📊 Analysis Results

| Metric | Value |
|--------|-------|
| 🌾 Weed Coverage | {weed_pct:.1f}% |
| 💊 Infestation Level | {infest_level} |
| 🧪 Herbicide Saved | **{savings:.1f}%** vs uniform |
| 📍 Weed Prompts | {(pt_labels==1).sum()} |
| 🔵 Background Prompts | {(pt_labels==0).sum()} |

## 🗺️ Prescription Map Legend
- 🟢 **Green** — No treatment (< 5% weed density)
- 🟡 **Yellow** — Low dose 150 L/ha (5–25%)
- 🟠 **Orange** — Medium dose 300 L/ha (25–50%)
- 🔴 **Red** — High dose 450 L/ha (> 50%)

## 💡 Agronomic Note
{"⚠️ High infestation detected (>35% of vegetation). Global NDVI scoring used for prompt generation." if high_infest else "✅ Moderate infestation. Local NDVI anomaly detection used for prompt generation."}
Pipeline performs best at **20–60% weed coverage**.
"""

    progress(1.0, desc="Done!")

    return [
        rgb,
        ndvi_rgb,
        prompt_vis,
        overlay,
        presc_map,
        stats
    ]


# ── Gradio UI ──────────────────────────────────────────────────
css = """
@import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=Syne:wght@400;600;800&display=swap');

:root {
    --green-dark:  #1a3a2a;
    --green-mid:   #2d6a4f;
    --green-light: #52b788;
    --green-pale:  #b7e4c7;
    --cream:       #f4f1e8;
    --amber:       #d4a017;
    --red-weed:    #c1440e;
    --text-dark:   #1a1a1a;
    --mono:        'DM Mono', monospace;
    --display:     'Syne', sans-serif;
}

body, .gradio-container {
    background: var(--cream) !important;
    font-family: var(--display) !important;
}

.gradio-container {
    max-width: 1400px !important;
    margin: 0 auto !important;
}

/* Header */
.app-header {
    background: var(--green-dark);
    padding: 2.5rem 3rem;
    margin-bottom: 2rem;
    border-bottom: 4px solid var(--green-light);
    position: relative;
    overflow: hidden;
}

.app-header::before {
    content: '';
    position: absolute;
    top: -50%;
    right: -10%;
    width: 400px;
    height: 400px;
    border-radius: 50%;
    background: radial-gradient(circle, rgba(82,183,136,0.15) 0%, transparent 70%);
}

.app-title {
    font-family: var(--display) !important;
    font-weight: 800 !important;
    font-size: 2.2rem !important;
    color: var(--green-pale) !important;
    letter-spacing: -0.02em;
    margin: 0 !important;
    line-height: 1.1 !important;
}

.app-subtitle {
    font-family: var(--mono) !important;
    font-size: 0.85rem !important;
    color: var(--green-light) !important;
    margin-top: 0.5rem !important;
    letter-spacing: 0.05em;
}

.app-badge {
    display: inline-block;
    background: var(--green-mid);
    color: var(--green-pale);
    font-family: var(--mono);
    font-size: 0.7rem;
    padding: 0.2rem 0.6rem;
    border-radius: 2px;
    margin-right: 0.4rem;
    letter-spacing: 0.08em;
    text-transform: uppercase;
}

/* Panels */
.panel-label {
    font-family: var(--mono) !important;
    font-size: 0.75rem !important;
    text-transform: uppercase !important;
    letter-spacing: 0.12em !important;
    color: var(--green-mid) !important;
    margin-bottom: 0.4rem !important;
}

/* Upload area */
.upload-section {
    background: white;
    border: 2px dashed var(--green-light);
    border-radius: 8px;
    padding: 1.5rem;
    margin-bottom: 1rem;
}

/* Run button */
#run-btn {
    background: var(--green-dark) !important;
    color: var(--green-pale) !important;
    font-family: var(--display) !important;
    font-weight: 600 !important;
    font-size: 1rem !important;
    border: none !important;
    border-radius: 4px !important;
    padding: 0.8rem 2rem !important;
    letter-spacing: 0.05em !important;
    cursor: pointer !important;
    transition: background 0.2s ease !important;
    width: 100% !important;
}

#run-btn:hover {
    background: var(--green-mid) !important;
}

/* Result images */
.result-img {
    border: 1px solid var(--green-pale);
    border-radius: 6px;
    overflow: hidden;
}

/* Stats panel */
.stats-panel {
    background: var(--green-dark);
    color: var(--cream);
    border-radius: 8px;
    padding: 1.5rem;
    font-family: var(--mono);
}

/* Info box */
.info-box {
    background: white;
    border-left: 4px solid var(--green-light);
    padding: 1rem 1.2rem;
    border-radius: 0 6px 6px 0;
    margin-bottom: 1rem;
    font-size: 0.875rem;
    color: var(--text-dark);
    font-family: var(--mono);
}

/* Tabs */
.tab-nav button {
    font-family: var(--display) !important;
    font-weight: 600 !important;
}

/* Section headers */
h2, h3 {
    font-family: var(--display) !important;
    font-weight: 600 !important;
    color: var(--green-dark) !important;
}
"""

with gr.Blocks(css=css, title="AutoWeedMap") as demo:

    # ── Header ────────────────────────────────────────────────
    gr.HTML("""
    <div class="app-header">
        <div class="app-title">🌾 AutoWeedMap</div>
        <div class="app-subtitle">
            Zero-Click Weedy Rice Detection · Multispectral UAV · SAM + NDVI Guidance
        </div>
        <div style="margin-top: 1rem;">
            <span class="app-badge">Zero-Shot</span>
            <span class="app-badge">No Training Data</span>
            <span class="app-badge">Multispectral</span>
            <span class="app-badge">SAM</span>
        </div>
    </div>
    """)

    with gr.Tabs():

        # ── Tab 1: Run Pipeline ───────────────────────────────
        with gr.Tab("🚀 Run Pipeline"):

            gr.HTML("""
            <div class="info-box">
                Upload your multispectral UAV image bands below. RGB is required.
                Multispectral bands (G, R, RE, NIR) are optional but strongly recommended
                — the pipeline degrades significantly on RGB-only input.
                Optimal for images with <b>20–60% weed coverage</b>.
            </div>
            """)

            with gr.Row():
                # ── Left: Inputs ──────────────────────────────
                with gr.Column(scale=1):
                    gr.HTML('<div class="panel-label">📁 Input Images</div>')

                    rgb_input = gr.File(
                        label="RGB Image (required) — .JPG or .PNG",
                        file_types=[".jpg", ".jpeg", ".png"],
                    )

                    with gr.Accordion("🌈 Multispectral Bands (optional)", open=True):
                        g_input   = gr.File(label="Green band — _G.TIF",   file_types=[".tif", ".tiff"])
                        r_input   = gr.File(label="Red band — _R.TIF",     file_types=[".tif", ".tiff"])
                        re_input  = gr.File(label="RedEdge band — _RE.TIF",file_types=[".tif", ".tiff"])
                        nir_input = gr.File(label="NIR band — _NIR.TIF",   file_types=[".tif", ".tiff"])

                    run_btn = gr.Button("▶ Run AutoWeedMap Pipeline", elem_id="run-btn")

                    gr.HTML("""
                    <div class="info-box" style="margin-top: 1rem;">
                        <b>Processing time:</b> ~30 seconds on GPU · ~2 min on CPU<br>
                        <b>Best results:</b> DJI Mavic 3M or similar multispectral UAV<br>
                        <b>Image size:</b> Works on any resolution
                    </div>
                    """)

                # ── Right: Outputs ────────────────────────────
                with gr.Column(scale=2):
                    gr.HTML('<div class="panel-label">📊 Results</div>')

                    with gr.Row():
                        out_rgb    = gr.Image(label="Input RGB",          elem_classes=["result-img"])
                        out_ndvi   = gr.Image(label="NDVI Map",           elem_classes=["result-img"])

                    with gr.Row():
                        out_prompts = gr.Image(label="Auto-Generated Prompts (red=weed, blue=background)",
                                               elem_classes=["result-img"])
                        out_detect  = gr.Image(label="Weed Detection Overlay",
                                               elem_classes=["result-img"])

                    with gr.Row():
                        out_presc  = gr.Image(label="🗺️ Herbicide Prescription Map",
                                              elem_classes=["result-img"])
                        out_stats  = gr.Markdown(label="Analysis Statistics")

            # ── Wire up ───────────────────────────────────────
            run_btn.click(
                fn=run_pipeline,
                inputs=[rgb_input, g_input, r_input, re_input, nir_input],
                outputs=[out_rgb, out_ndvi, out_prompts, out_detect,
                         out_presc, out_stats]
            )

        # ── Tab 2: How It Works ───────────────────────────────
        with gr.Tab("📖 How It Works"):
            gr.Markdown("""
## AutoWeedMap Pipeline

AutoWeedMap is a **zero-click** weedy rice detection system. It requires no human
interaction between image upload and prescription map output.

---

### The Problem

Weedy rice (*Oryza sativa* f. *spontanea*) is visually **identical** to cultivated
rice in standard RGB photographs at 55–60 days after sowing. Human scouts cannot
distinguish them efficiently at field scale. Blanket herbicide application wastes
chemical, increases costs, and accelerates resistance.

---

### The Pipeline

```
Multispectral UAV Image
        │
        ▼
Vegetation Indices (NDVI · NDRE · MCARI)
        │
        ▼
NDVI-Guided Automatic Prompting
  Local anomaly detection
  DBSCAN cluster → one prompt per weed patch
  Zero human clicks
        │
        ▼
SAM Segmentation (patch-based, 192px)
  MCARI composite input
  Spectral validation per patch
        │
        ▼
Prescription Map
  4-level dose zones (none / low / medium / high)
  GeoTIFF export ready
  Herbicide savings calculation
```

---

### Key Finding: RGB is Not Enough

We tested Grounding DINO (language-guided detection on RGB) as an alternative
prompting strategy. It achieved **IoU = 0.031** across 20 test images — essentially
random. The spectral information from NIR and RedEdge bands is **necessary**, not
optional, for weedy rice detection.

---

### Performance

| Method | Mean IoU | Requires Training? |
|--------|----------|--------------------|
| NDVI Threshold only | 0.175 | No |
| SAM + Grid Prompts | 0.353 | No |
| Grounded SAM (RGB) | 0.031 | No |
| **AutoWeedMap (ours)** | **0.223** | **No** |
| AutoWeedMap (20–60% infestation) | **0.354** | **No** |

**Herbicide savings: 52–67%** vs uniform application on best-performing images.

---

### Limitations

- Performs best at **20–60% weed coverage**
- Degrades below 20% (signal too weak) and above 70% (weeds become the norm)
- Requires multispectral sensor — RGB-only input significantly reduces performance
- Validated on Mekong Delta, Vietnam rice fields (55–60 days after sowing)
- Patch-based SAM produces block-shaped mask boundaries

---

### Dataset

**WeedyRice-RGBMS-DB** — 734 aligned RGB + multispectral UAV image pairs
from Vietnam's Mekong Delta across 3 cropping seasons (2024–2025).

[Mendeley Data](https://data.mendeley.com/datasets/vt4s83pxx6/1) ·
[Paper](https://doi.org/10.1016/j.dib.2025.112237)

---

### Citation

```bibtex
@misc{autoweedmap2025,
  title  = {AutoWeedMap: Zero-Click Weedy Rice Detection},
  author = {Your Name},
  year   = {2025},
  url    = {https://github.com/yourusername/autoweedmap}
}
```
            """)

        # ── Tab 3: Example Results ────────────────────────────
        with gr.Tab("🖼️ Example Results"):
            gr.Markdown("""
## Best Performing Results on WeedyRice-RGBMS-DB Test Set

These results were generated automatically — zero human clicks.

| Image | Weed Coverage | IoU | Herbicide Saved |
|-------|---------------|-----|-----------------|
| Sample 1 | 39.4% | 0.565 | 52.2% |
| Sample 2 | 42.1% | 0.530 | 66.8% |
| Sample 3 | 27.3% | 0.509 | 62.4% |

### Performance by Infestation Level

| Range | Mean IoU | Rating |
|-------|----------|--------|
| 10–20% | 0.173 | 🔴 Poor |
| 20–30% | 0.317 | 🟡 Moderate |
| 30–40% | 0.369 | 🟡 Moderate |
| 40–50% | 0.354 | 🟡 Moderate |
| 50–60% | 0.335 | 🟡 Moderate |
| 70–90% | ~0.19 | 🔴 Poor |

Upload your own multispectral UAV images in the **Run Pipeline** tab to try it on your data.
            """)

    # ── Footer ────────────────────────────────────────────────
    gr.HTML("""
    <div style="
        margin-top: 2rem;
        padding: 1rem 2rem;
        background: var(--green-dark);
        color: var(--green-light);
        font-family: 'DM Mono', monospace;
        font-size: 0.75rem;
        text-align: center;
        border-top: 2px solid var(--green-mid);
    ">
        AutoWeedMap · Built with Meta SAM + NDVI Guidance ·
        Dataset: WeedyRice-RGBMS-DB (Nguyen et al. 2025) ·
        <a href="https://github.com/yourusername/autoweedmap"
           style="color: var(--green-pale); text-decoration: none;">
           GitHub ↗
        </a>
    </div>
    """)


if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=True,          # creates public gradio.live link
        show_error=True,
    )
