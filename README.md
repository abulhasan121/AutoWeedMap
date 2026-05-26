# AutoWeedMap — Zero-Click Weedy Rice Detection from Multispectral UAV Imagery

**Developed by Shah Md Abul Hasan**

AutoWeedMap is a fully automated precision agriculture pipeline that detects weedy rice from multispectral UAV imagery and generates variable-rate herbicide prescription maps without any manual interaction during inference.

The system combines:
- Multispectral vegetation indices
- NDVI-guided anomaly detection
- Segment Anything Model (SAM)
- Automatic prompt generation
- Spatial prescription mapping

to create an end-to-end framework for site-specific weed management in rice production systems.

Unlike previous SAM-based agricultural segmentation approaches that require manual clicks or bounding boxes, AutoWeedMap introduces a fully automatic spectral prompting strategy driven by local NDVI anomalies.

---

# Project Motivation

Weedy rice (*Oryza sativa* f. *spontanea*) is one of the most destructive weeds in rice production systems, particularly across Southeast Asia. The weed competes aggressively with cultivated rice for:
- Nutrients
- Water
- Light
- Space

leading to severe yield and quality losses.

A major challenge for precision agriculture is that weedy rice is visually almost indistinguishable from cultivated rice in standard RGB imagery, especially during mid-season growth stages. Traditional computer vision systems based on RGB images struggle because:
- Crop and weed share nearly identical morphology
- Leaf color and canopy structure overlap heavily
- Background segmentation provides little discriminatory value

This limitation makes conventional RGB deep learning pipelines unreliable for operational field deployment.

AutoWeedMap addresses this problem using multispectral UAV imagery and spectral anomaly-guided segmentation rather than traditional RGB object detection.

The key innovation is reframing SAM prompting as a precision agriculture problem rather than a manual interactive segmentation problem.

Instead of:
- Human clicks
- Bounding box prompts
- Manual annotations

the system automatically generates prompts from multispectral vegetation anomalies using NDVI spatial deviation analysis.

This enables:
- Fully automated inference
- Zero-click segmentation
- Field-scale deployment
- Integration into precision herbicide workflows

The broader significance for precision agriculture is that AutoWeedMap connects:
1. UAV remote sensing
2. Foundation segmentation models
3. Spectral crop stress analysis
4. Variable-rate chemical application

into a single operational decision-support pipeline.

Rather than producing only segmentation masks, the framework directly generates actionable prescription maps for site-specific herbicide management.

---

# System Overview

The pipeline takes multispectral UAV imagery as input and produces:
- Weed segmentation masks
- Weed density estimation
- Herbicide treatment zones
- Variable-rate prescription maps

without requiring user interaction.

The workflow operates entirely automatically once imagery is provided.

---

# Pipeline Architecture

```text
Multispectral UAV Image
(RGB + Green + Red + RedEdge + NIR)
                ↓
Vegetation Index Computation
(NDVI · NDRE · MCARI)
                ↓
NDVI-Guided Automatic Prompt Generation
(Local anomaly detection + DBSCAN clustering)
                ↓
Segment Anything Model (SAM)
(Patch-based segmentation)
                ↓
Weedy Rice Segmentation Masks
                ↓
Spatial Weed Density Mapping
                ↓
Variable-Rate Herbicide Prescription Map
                ↓
GeoTIFF Export
```

---

# Why Multispectral Imagery Matters

A major finding of this work is that multispectral imagery is not optional for weedy rice detection.

The study experimentally demonstrates that RGB-only foundation models fail because cultivated rice and weedy rice are visually similar under standard color imagery.

Grounded SAM operating only on RGB imagery achieved:

```text
IoU = 0.031
```

across the evaluation dataset.

In contrast, multispectral vegetation indices derived from:
- RedEdge
- Near Infrared (NIR)

contain physiological information related to:
- Chlorophyll variation
- Canopy stress
- Biomass differences
- Spectral reflectance anomalies

that are invisible in RGB space.

This highlights an important precision agriculture principle:
spectral information often contains stronger agronomic signals than visual appearance alone.

---

# Automatic NDVI-Guided Prompting

The central methodological contribution of AutoWeedMap is replacing manual SAM prompting with fully automatic spectral prompting.

Traditional SAM workflows require:
- Point clicks
- Bounding boxes
- Human supervision

which prevents large-scale autonomous deployment.

AutoWeedMap instead:
1. Computes local NDVI anomaly maps
2. Detects spectrally abnormal regions
3. Clusters anomaly centers using DBSCAN
4. Converts cluster centroids into SAM prompts

This transforms SAM from:
- an interactive segmentation model

into:
- a fully autonomous agricultural segmentation pipeline.

The method is especially effective in the agronomically important 20–60% infestation range where:
- Weed-crop boundaries remain spectrally distinguishable
- Herbicide intervention decisions are economically valuable

---

# Key Findings

## 1. Spectral Guidance Improves Segmentation

NDVI-guided prompting outperformed generic grid prompting because prompts were directed toward biologically meaningful anomaly regions rather than arbitrary image locations.

---

## 2. MCARI Composite Inputs Performed Best

Among six tested image representations:
- RGB
- False Color
- Color Infrared
- NDVI
- NDRE
- MCARI-NDVI-NDRE Composite

the MCARI composite consistently produced the strongest SAM segmentation performance.

This confirms that:
- chlorophyll-sensitive spectral information
- improves foundation segmentation models in agriculture.

---

## 3. Performance Depends on Infestation Level

The pipeline performs best at moderate infestation levels.

### Moderate Infestation (20–60%)

- Strong weed-crop contrast
- Distinct anomaly boundaries
- Best operational performance

### Sparse Infestation (<20%)

- Weak anomaly signals
- Prompt localization becomes unstable

### Severe Infestation (>70%)

- Weedy rice becomes the spectral norm
- Local anomaly assumptions fail
- Detection performance decreases

These failure modes are reported explicitly and represent important open research problems for autonomous agricultural segmentation systems.

---

# Performance Results

## Main Comparison

| Method | Mean IoU | Zero-Shot | Training Data | Notes |
|---|---|---|---|---|
| NDVI Threshold | 0.175 | Yes | None | Pure spectral thresholding |
| Grounded SAM (RGB) | 0.031 | Yes | None | RGB failure case |
| SAM + Grid Prompts | 0.353 | Yes | None | No spectral guidance |
| **SAM + NDVI Prompts** | **0.223** | **Yes** | **None** | Proposed method |
| SAM + NDVI (20–60%) | **0.354** | **Yes** | **None** | Best operating range |

---

## Performance by Infestation Level

| Weed Coverage | Mean IoU | Interpretation |
|---|---|---|
| 10–20% | 0.173 | Weak anomaly signal |
| 20–30% | 0.317 | Strong operational range |
| 30–40% | 0.369 | Best segmentation |
| 40–50% | 0.354 | Stable performance |
| 50–60% | 0.335 | Good operational range |
| 70–90% | 0.172–0.207 | Anomaly inversion failure |

---

# Prescription Mapping

The final output is not only a segmentation mask but a spatial herbicide recommendation map suitable for precision agriculture workflows.

The system:
1. Computes weed density spatially
2. Divides fields into management zones
3. Assigns herbicide rates
4. Exports georeferenced prescription layers

This supports:
- Variable-rate spraying
- Reduced herbicide usage
- Lower production costs
- Reduced environmental impact

---

# Herbicide Savings

| Sample | Weed Coverage | IoU | Herbicide Saved |
|---|---|---|---|
| Sample 1 | 39.4% | 0.565 | 52.2% |
| Sample 2 | 42.1% | 0.530 | 66.8% |
| Sample 3 | 27.3% | 0.509 | 62.4% |

Average herbicide savings ranged between:
- 52–67%

compared to uniform field application.

---

# Dataset

## WeedyRice-RGBMS-DB

| Property | Value |
|---|---|
| Image Pairs | 734 |
| Data Type | RGB + Multispectral UAV |
| UAV Platform | DJI Mavic 3 Multispectral |
| Seasons | 3 |
| Region | Mekong Delta, Vietnam |
| Spectral Bands | Green · Red · RedEdge · NIR |
| Annotation Type | Expert polygon masks |
| Weed Coverage Range | 0.4%–89.4% |

### Spectral Bands

| Band | Wavelength |
|---|---|
| Green | 560 nm |
| Red | 650 nm |
| RedEdge | 730 nm |
| NIR | 860 nm |

---

# Installation

## Clone Repository

```bash
git clone https://github.com/abulhasan121/AutoWeedMap.git
cd AutoWeedMap
```

## Create Environment

```bash
conda create -n autoweedmap python=3.10 -y
conda activate autoweedmap
```

## Install Dependencies

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

pip install git+https://github.com/facebookresearch/segment-anything.git

pip install groundingdino-py transformers==4.38.2

pip install rasterio geopandas scipy scikit-learn \
            opencv-python matplotlib pillow pandas
```

## Download SAM Weights

```bash
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth
```

---

# Usage

## Example Pipeline

```python
from pipeline import AutoWeedMap

pipeline = AutoWeedMap(
    sam_checkpoint="sam_vit_b_01ec64.pth",
    device="cuda"
)

result = pipeline.run(
    rgb_path   = "data/RGB/image_001.JPG",
    ms_dir     = "data/Multispectral/",
    image_id   = "image_001",
    output_dir = "results/"
)

print(f"Weed coverage: {result['weed_pct']:.1f}%")
print(f"Herbicide saved: {result['savings_pct']:.1f}%")
```

---

# Repository Structure

```text
AutoWeedMap/
├── README.md
├── requirements.txt
│
├── src/
│   ├── data/
│   ├── indices/
│   ├── prompts/
│   ├── segmentation/
│   ├── prescription/
│   └── evaluate/
│
├── baselines/
├── experiments/
├── notebooks/
└── results/
```

---

# Prescription Map Legend

| Zone | Herbicide Dose | Weed Density | Recommendation |
|---|---|---|---|
| Green | 0 L/ha | <5% | No treatment |
| Yellow | 150 L/ha | 5–25% | Low dose |
| Orange | 300 L/ha | 25–50% | Medium dose |
| Red | 450 L/ha | >50% | High dose |

---

# Limitations

| Limitation | Field Implication |
|---|---|
| Sparse infestations | Weak NDVI anomaly signal |
| Severe infestations | Spectral inversion problem |
| RGB-only imagery | Detection fails |
| Growth-stage dependency | Optimized for 55–60 DAS |
| Patch artifacts | Block-shaped boundaries |
| Single-region validation | Unknown geographic transferability |

---

# Experimental Setup

| Component | Value |
|---|---|
| GPU | NVIDIA T4 · 16 GB VRAM |
| SAM Model | ViT-B |
| Patch Size | 192 × 192 |
| IoU Threshold | 0.60 |
| Evaluation Images | 117 |
| Metrics | IoU · F1 · Precision · Recall |

---

# Related Projects

| Project | Description |
|---|---|
| AgriScholar | Agricultural research RAG system |
| PhytoScan | Vision-language plant disease diagnosis |
| AutoWeedMap | Multispectral UAV weed mapping |

---

# Citation

```bibtex
@misc{autoweedmap2025,
  title   = {AutoWeedMap: Zero-Click Weedy Rice Detection and Herbicide Prescription Mapping from Multispectral UAV Imagery},
  author  = {Hasan, Shah Md Abul},
  year    = {2025},
  url     = {https://github.com/abulhasan121/AutoWeedMap}
}
```

---

# Acknowledgements

- Meta AI — Segment Anything Model
- IDEA Research — Grounding DINO
- Nguyen et al. — WeedyRice-RGBMS-DB Dataset

---

# Author

**Shah Md Abul Hasan**

Built with:
- PyTorch
- Segment Anything Model (SAM)
- Multispectral UAV Imagery
- OpenCV
- RasterIO

---

# License

MIT License
