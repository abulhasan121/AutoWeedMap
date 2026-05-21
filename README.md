# AutoWeedMap
Zero-Click Weedy Rice Detection from Multispectral UAV Imagery
# 🌾 AutoWeedMap: Zero-Click Weedy Rice Detection and Herbicide Prescription Mapping from Multispectral UAV Imagery

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-orange.svg)](https://pytorch.org)
[![SAM](https://img.shields.io/badge/Meta-SAM-purple.svg)](https://github.com/facebookresearch/segment-anything)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## Overview

AutoWeedMap is a fully automated pipeline that takes multispectral UAV imagery as input and produces a **variable-rate herbicide prescription map** as output — with zero human interaction between those two steps.

Weedy rice (*Oryza sativa* f. *spontanea*) is one of the most damaging threats to cultivated rice production in Southeast Asia. It is visually nearly identical to cultivated rice in standard RGB photography, making automated detection extremely difficult. This project addresses that challenge using spectral information from multispectral UAV sensors combined with the Segment Anything Model (SAM).

**Key contribution:** We replace the manual click-based prompting used in prior SAM-based agricultural segmentation work with a fully automatic NDVI-guided prompting strategy — requiring zero human interaction at inference time.

---

## Pipeline

```
Multispectral UAV Image (RGB + G + R + RE + NIR)
                │
                ▼
    ┌───────────────────────┐
    │   Vegetation Indices   │
    │  NDVI · NDRE · MCARI  │
    └───────────┬───────────┘
                │
                ▼
    ┌───────────────────────┐
    │   NDVI-Guided Auto    │
    │      Prompting        │
    │                       │
    │  Local anomaly maps   │
    │  DBSCAN clustering    │
    │  Zero human clicks    │
    └───────────┬───────────┘
                │
                ▼
    ┌───────────────────────┐
    │    SAM Segmentation   │
    │                       │
    │  Patch-based (192px)  │
    │  Spectral validation  │
    │  MCARI composite input│
    └───────────┬───────────┘
                │
                ▼
    ┌───────────────────────┐
    │   Prescription Map    │
    │                       │
    │  Grid density mapping │
    │  4-level dose zones   │
    │  GeoTIFF export       │
    │  Herbicide savings    │
    └───────────────────────┘
```

---

## Results

### Main Comparison Table

| Method | Mean IoU | Zero-Shot | Training Data | Notes |
|---|---|---|---|---|
| NDVI Threshold (baseline) | 0.175 | ✅ | None | Pure spectral thresholding |
| Grounded SAM — RGB only | 0.031 | ✅ | None | Language detection fails on weedy rice |
| SAM + Grid Prompts | 0.353 | ✅ | None | No spectral guidance |
| **SAM + NDVI Prompts (ours)** | **0.223** | ✅ | **None** | **Proposed method** |
| SAM + NDVI Prompts (20–60% infestation) | **0.354** | ✅ | None | Optimal operating range |

### Performance by Infestation Level

| Weed Coverage | Mean IoU | Rating | N Images |
|---|---|---|---|
| 10–20% | 0.173 | 🔴 Poor | 31 |
| 20–30% | 0.317 | 🟡 Moderate | 15 |
| 30–40% | 0.369 | 🟡 Moderate | 14 |
| 40–50% | 0.354 | 🟡 Moderate | 21 |
| 50–60% | 0.335 | 🟡 Moderate | 11 |
| 70–90% | 0.172–0.207 | 🔴 Poor | 20 |

**The pipeline performs best at 20–60% weed coverage** — the agronomically critical range where intervention decisions are most needed and most actionable.

### Prescription Map Output

| Image | Weed Coverage | IoU | Herbicide Saved |
|---|---|---|---|
| Sample 1 | 39.4% | 0.565 | **52.2%** |
| Sample 2 | 42.1% | 0.530 | **66.8%** |
| Sample 3 | 27.3% | 0.509 | **62.4%** |

Average herbicide savings of **52–67%** compared to uniform field application across best-performing images.

---

## Key Findings

**1. Multispectral input is necessary, not optional.**
We demonstrate this experimentally: Grounded SAM (language-guided detection on RGB) achieves only IoU = 0.031 across 20 test images. Weedy rice and cultivated rice are visually indistinguishable in RGB at 55–60 days after sowing. Spectral bands — particularly RedEdge and NIR — contain the discriminative information.

**2. NDVI-guided prompting beats grid prompting in the 20–60% range.**
Random or grid-based SAM prompting does not target spectrally anomalous regions. NDVI local anomaly detection directs SAM toward pixels that deviate from their neighborhood — exactly where weed-crop boundaries exist.

**3. MCARI composite input outperforms RGB for SAM.**
Among six tested input modes (RGB, false color, color infrared, NDVI, NDRE, MCARI composite), the MCARI-NDVI-NDRE composite consistently achieved the highest IoU, confirming that spectral information improves SAM segmentation quality.

**4. The pipeline fails at extreme infestation levels.**
Below 20% coverage: spectral anomaly signal too weak. Above 70% coverage: weedy rice becomes the spectral norm — local anomaly detection inverts. Both failure modes are reported honestly and represent open research challenges.

---

## Dataset

**Weedy Rice RGB-MS Database (WeedyRice-RGBMS-DB)**

- 734 aligned RGB + multispectral UAV image pairs
- Collected using DJI Mavic 3 Multispectral UAV
- 3 cropping seasons, Mekong Delta, Vietnam (2024–2025)
- 4 spectral bands: Green (560nm), Red (650nm), RedEdge (730nm), NIR (860nm)
- Expert-verified polygon masks for weedy rice
- Infestation levels: 0.4% to 89.4% per image

**Download:** [Mendeley Data](https://data.mendeley.com/datasets/vt4s83pxx6/1)

**Citation:**
```
Nguyen et al. (2025). A dataset of aligned RGB and multispectral UAV imagery 
for semantic segmentation of weedy rice. Data in Brief, 63, 112237.
```

---

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/autoweedmap.git
cd autoweedmap

# Create environment
conda create -n autoweedmap python=3.10 -y
conda activate autoweedmap

# Install dependencies
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install git+https://github.com/facebookresearch/segment-anything.git
pip install groundingdino-py transformers==4.38.2
pip install rasterio geopandas scipy scikit-learn opencv-python matplotlib pillow pandas

# Download SAM weights
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth
```

---

## Usage

```python
from pipeline import AutoWeedMap

# Initialize pipeline
pipeline = AutoWeedMap(
    sam_checkpoint="sam_vit_b_01ec64.pth",
    device="cuda"
)

# Run on one image
result = pipeline.run(
    rgb_path    = "data/RGB/image_001.JPG",
    ms_dir      = "data/Multispectral/",
    image_id    = "image_001",
    output_dir  = "results/"
)

print(f"Weed coverage:    {result['weed_pct']:.1f}%")
print(f"Herbicide saved:  {result['savings_pct']:.1f}%")
print(f"Prescription map: {result['prescription_path']}")
```

---

## Repository Structure

```
files/autoweedmap/
├── README.md
├── requirements.txt
│
├── src/
│   ├── data/
│   │   └── loader.py              ← dataset loading utilities
│   ├── indices/
│   │   └── vegetation.py          ← NDVI, NDRE, MCARI computation
│   ├── prompts/
│   │   └── ndvi_prompter.py       ← automatic SAM prompt generation
│   ├── segmentation/
│   │   └── sam_segmenter.py       ← patch-based SAM inference
│   ├── prescription/
│   │   └── map_generator.py       ← prescription map generation
│   └── evaluate/
│       └── metrics.py             ← IoU, F1, precision, recall
│
├── baselines/
│   ├── ndvi_threshold.py          ← spectral threshold baseline
│   ├── grid_prompt_sam.py         ← grid-prompted SAM baseline
│   └── grounded_sam.py            ← Grounded DINO + SAM baseline
│
├── experiments/
│   ├── band_ablation.py           ← compare 6 input band modes
│   ├── infestation_analysis.py    ← performance by weed coverage
│   └── full_evaluation.py         ← run on all 117 valid test images
│
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_vegetation_indices.ipynb
│   ├── 03_prompting_strategy.ipynb
│   ├── 04_sam_band_ablation.ipynb
│   └── 05_results_analysis.ipynb
│
└── results/
    └── figures/
        ├── vegetation_indices.png
        ├── band_modes.png
        ├── ndvi_prompts.png
        ├── performance_by_weed_level.png
        ├── final_result_iou0.565.png
        ├── final_result_iou0.530.png
        └── final_result_iou0.509.png
```

---

## Prescription Map Legend

| Color | Dose | Weed Density | Action |
|---|---|---|---|
| 🟢 Green | 0 L/ha | < 5% | No treatment |
| 🟡 Yellow | 150 L/ha | 5–25% | Low dose |
| 🟠 Orange | 300 L/ha | 25–50% | Medium dose |
| 🔴 Red | 450 L/ha | > 50% | High dose |

---

## Limitations

- **Sparse infestations (< 20% coverage):** NDVI anomaly signal too weak for reliable detection. Prompts fail to localize weed patches.
- **Severe infestations (> 70% coverage):** When most of the field is weedy, the local anomaly approach inverts — cultivated rice becomes the anomaly.
- **RGB-only sensors:** Multispectral input is required. The failure of Grounded SAM (RGB-only, IoU = 0.031) confirms this experimentally.
- **Growth stage dependency:** Optimized for 55–60 days after sowing when visual differentiation peaks. Performance at other stages untested.
- **Block-shaped mask artifacts:** Patch-based SAM produces rectangular boundaries. Pixel-level precision requires further post-processing.
- **Single geography:** Validated only on Mekong Delta, Vietnam. Generalization to other rice-growing regions requires validation.

---

## Experimental Setup

All experiments run on Google Colab with NVIDIA T4 GPU (16GB VRAM).

- SAM model: ViT-B (357 MB)
- Patch size: 192 × 192 pixels
- IoU threshold: 0.60
- Test set: 117 images with ≥ 10% weed coverage (out of 148 total)
- Evaluation metrics: IoU, F1, Precision, Recall

---

## How to Cite

```bibtex
@misc{autoweedmap2025,
  title   = {AutoWeedMap: Zero-Click Weedy Rice Detection and 
              Herbicide Prescription Mapping from Multispectral UAV Imagery},
  author  = {Your Name},
  year    = {2025},
  url     = {https://github.com/yourusername/autoweedmap}
}
```

---

## Acknowledgements

- [Meta AI — Segment Anything Model](https://github.com/facebookresearch/segment-anything)
- [IDEA Research — Grounding DINO](https://github.com/IDEA-Research/GroundingDINO)
- [Nguyen et al. 2025 — WeedyRice-RGBMS-DB Dataset](https://data.mendeley.com/datasets/vt4s83pxx6/1)

---

## License

MIT License. See [LICENSE](LICENSE) for details.
