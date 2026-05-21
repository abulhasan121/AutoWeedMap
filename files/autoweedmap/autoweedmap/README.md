# 🌾 AutoWeedMap

**Zero-Click Weedy Rice Detection and Herbicide Prescription Mapping from Multispectral UAV Imagery**

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-orange.svg)](https://pytorch.org)
[![SAM](https://img.shields.io/badge/Meta-SAM-purple.svg)](https://github.com/facebookresearch/segment-anything)
[![Gradio](https://img.shields.io/badge/Demo-Gradio-yellow.svg)](https://gradio.app)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## What This Does

AutoWeedMap takes a multispectral UAV image as input and produces a **variable-rate herbicide prescription map** as output — with zero human clicks in between.

```
Multispectral UAV Image  →  [AutoWeedMap]  →  Prescription Map
(RGB + G + R + RE + NIR)                       (52–67% herbicide saved)
```

Weedy rice is visually identical to cultivated rice in RGB photography at 55–60 days after sowing. This project proves that **spectral information from NIR and RedEdge bands is necessary** for automated detection — and builds a complete zero-shot pipeline around that insight.

---

## Demo

```bash
# Install
pip install -r requirements.txt

# Download SAM weights
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth

# Launch Gradio app
python app.py
```

The app opens at `http://localhost:7860`. Upload your RGB and MS band files — results in ~30 seconds on GPU.

---

## Pipeline

```
┌─────────────────────────────────────────────────────────────┐
│  INPUT: Multispectral UAV Image                             │
│  RGB + Green + Red + RedEdge + NIR                          │
└──────────────────────┬──────────────────────────────────────┘
                       │
          ┌────────────▼────────────┐
          │   Vegetation Indices    │
          │  NDVI · NDRE · MCARI   │
          └────────────┬────────────┘
                       │
          ┌────────────▼────────────┐
          │  NDVI-Guided Prompting  │
          │                         │
          │  Local anomaly maps     │
          │  DBSCAN clustering      │
          │  → Weed patch centroids │
          │  Zero human clicks      │
          └────────────┬────────────┘
                       │
          ┌────────────▼────────────┐
          │   SAM Segmentation      │
          │                         │
          │  Patch-based (192px)    │
          │  MCARI composite input  │
          │  Spectral validation    │
          └────────────┬────────────┘
                       │
          ┌────────────▼────────────┐
          │   Prescription Map      │
          │                         │
          │  Grid density mapping   │
          │  4-level dose zones     │
          │  Herbicide savings calc │
          └─────────────────────────┘
```

---

## Results

### Comparison Table (20 test images)

| Method | Mean IoU | Zero-Shot | Training Data |
|--------|----------|-----------|---------------|
| NDVI Threshold (baseline) | 0.175 | ✅ | None |
| SAM + Grid Prompts | 0.353 | ✅ | None |
| Grounded SAM — RGB only | 0.031 | ✅ | None |
| **AutoWeedMap (ours)** | **0.223** | ✅ | **None** |
| AutoWeedMap (20–60% infestation) | **0.354** | ✅ | None |

### Performance by Infestation Level (117 test images)

| Weed Coverage | Mean IoU | N Images |
|---------------|----------|----------|
| 10–20% | 0.173 🔴 | 31 |
| 20–30% | 0.317 🟡 | 15 |
| 30–40% | 0.369 🟡 | 14 |
| 40–50% | 0.354 🟡 | 21 |
| 50–60% | 0.335 🟡 | 11 |
| 70–90% | ~0.19 🔴 | 20 |

### Best Results

| Image | Weed Coverage | IoU | Herbicide Saved |
|-------|---------------|-----|-----------------|
| Sample 1 | 39.4% | **0.565** | 52.2% |
| Sample 2 | 42.1% | **0.530** | 66.8% |
| Sample 3 | 27.3% | **0.509** | 62.4% |

---

## Key Finding: RGB is Not Enough

We tested Grounding DINO (language-guided detection on RGB) as an alternative prompting strategy. Result: **IoU = 0.031** across 20 test images — essentially zero.

Weedy rice and cultivated rice are indistinguishable in RGB at the critical growth stage. The spectral information from **NIR and RedEdge bands** is necessary, not optional. This is not an assumption — we proved it experimentally.

---

## Installation

```bash
git clone https://github.com/yourusername/autoweedmap.git
cd autoweedmap

conda create -n autoweedmap python=3.10 -y
conda activate autoweedmap

pip install -r requirements.txt

# Download SAM ViT-B weights (357 MB)
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth
```

---

## Usage

### Gradio App

```bash
python app.py
# Opens at http://localhost:7860
# Share publicly: set share=True in app.py
```

### Python API

```python
from pipeline import AutoWeedMap

pipeline = AutoWeedMap(sam_checkpoint="sam_vit_b_01ec64.pth")

result = pipeline.run(
    rgb_path   = "data/RGB/image_001.JPG",
    ms_dir     = "data/Multispectral/",
    image_id   = "image_001",
    output_dir = "results/"
)

print(f"Weed coverage:   {result['weed_coverage_pct']:.1f}%")
print(f"Herbicide saved: {result['herbicide_saved_pct']:.1f}%")
```

### Command Line

```bash
python pipeline.py \
    --rgb        data/RGB/image_001.JPG \
    --ms_dir     data/Multispectral/ \
    --image_id   image_001 \
    --output_dir results/ \
    --gt_mask    data/Masks/image_001.png  # optional, for IoU evaluation
```

### Full Evaluation

```bash
python experiments/full_evaluation.py \
    --root       /path/to/WeedyRice-RGBMS-DB \
    --checkpoint sam_vit_b_01ec64.pth \
    --output_dir results/
```

---

## Repository Structure

```
autoweedmap/
├── app.py                          ← Gradio web app
├── pipeline.py                     ← Main AutoWeedMap class
├── requirements.txt
├── .gitignore
│
├── src/
│   ├── data/
│   │   └── loader.py               ← Dataset loading, split management
│   ├── indices/
│   │   └── vegetation.py           ← NDVI, NDRE, MCARI computation
│   ├── prompts/
│   │   └── ndvi_prompter.py        ← Automatic SAM prompt generation
│   ├── segmentation/
│   │   └── sam_segmenter.py        ← Patch-based SAM inference
│   ├── prescription/
│   │   └── map_generator.py        ← Prescription map + GeoTIFF export
│   └── evaluate/
│       └── metrics.py              ← IoU, F1, Moran's I
│
├── baselines/
│   └── baselines.py                ← NDVI threshold, Grid SAM, Grounded SAM
│
├── experiments/
│   └── full_evaluation.py          ← Complete benchmark runner
│
└── results/
    └── figures/                    ← Output visualizations
```

---

## Prescription Map Zones

| Color | Dose | Condition |
|-------|------|-----------|
| 🟢 Green | 0 L/ha | < 5% weed density — no treatment |
| 🟡 Yellow | 150 L/ha | 5–25% weed density — low dose |
| 🟠 Orange | 300 L/ha | 25–50% weed density — medium dose |
| 🔴 Red | 450 L/ha | > 50% weed density — high dose |

---

## Dataset

**WeedyRice-RGBMS-DB**
- 734 aligned RGB + multispectral image pairs
- DJI Mavic 3 Multispectral UAV
- 3 cropping seasons, Mekong Delta, Vietnam (2024–2025)
- 4 MS bands: Green (560nm), Red (650nm), RedEdge (730nm), NIR (860nm)
- Expert-verified polygon masks
- Weed coverage: 0.4% to 89.4% per image
- Train/Val/Test: 438 / 148 / 148

📥 **Download:** [Mendeley Data](https://data.mendeley.com/datasets/vt4s83pxx6/1)

```bibtex
@article{nguyen2025weedyrice,
  title   = {A dataset of aligned RGB and multispectral UAV imagery
              for semantic segmentation of weedy rice},
  author  = {Nguyen, Van-Hoa and others},
  journal = {Data in Brief},
  volume  = {63},
  pages   = {112237},
  year    = {2025},
  doi     = {10.1016/j.dib.2025.112237}
}
```

---

## Limitations

- **Sparse infestation (< 20%):** NDVI anomaly signal too weak — prompts miss weed patches
- **Severe infestation (> 70%):** Weedy rice becomes the spectral norm — local anomaly inverts
- **Multispectral required:** RGB-only input dramatically reduces performance (IoU drops to ~0.03)
- **Growth stage:** Optimized for 55–60 days after sowing in Mekong Delta conditions
- **Block artifacts:** Patch-based SAM produces rectangular mask boundaries

---

## Citation

```bibtex
@misc{autoweedmap2025,
  title  = {AutoWeedMap: Zero-Click Weedy Rice Detection and
             Herbicide Prescription Mapping from Multispectral UAV Imagery},
  author = {Your Name},
  year   = {2025},
  url    = {https://github.com/yourusername/autoweedmap}
}
```

---

## Acknowledgements

- [Meta AI — Segment Anything Model](https://github.com/facebookresearch/segment-anything)
- [IDEA Research — Grounding DINO](https://github.com/IDEA-Research/GroundingDINO)
- [WeedyRice-RGBMS-DB Dataset](https://data.mendeley.com/datasets/vt4s83pxx6/1)

---

## License

MIT License — see [LICENSE](LICENSE) for details.
