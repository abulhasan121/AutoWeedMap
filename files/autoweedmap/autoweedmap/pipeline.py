"""
AutoWeedMap — Main Pipeline
Zero-Click Weedy Rice Detection & Herbicide Prescription Mapping
"""

import numpy as np
import cv2
import torch
import os
from PIL import Image
from segment_anything import sam_model_registry, SamPredictor

from src.data.loader import load_sample, load_band
from src.indices.vegetation import compute_indices, normalize_to_uint8, compose_mcari
from src.prompts.ndvi_prompter import generate_prompts
from src.segmentation.sam_segmenter import run_sam_patches
from src.prescription.map_generator import generate_prescription
from src.evaluate.metrics import evaluate


class AutoWeedMap:
    """
    End-to-end pipeline from multispectral UAV imagery
    to herbicide prescription map.
    """

    def __init__(self,
                 sam_checkpoint="sam_vit_b_01ec64.pth",
                 model_type="vit_b",
                 device=None):

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Loading SAM on {self.device}...")

        sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
        sam.to(self.device)
        self.predictor = SamPredictor(sam)

        print("✅ AutoWeedMap ready.")

    def run(self,
            rgb_path,
            ms_dir,
            image_id,
            output_dir="results/",
            patch_size=192,
            iou_threshold=0.6,
            cell_size=64,
            ground_truth_path=None):
        """
        Run the full AutoWeedMap pipeline on one image.

        Args:
            rgb_path:           Path to RGB image (.JPG or .PNG)
            ms_dir:             Directory containing MS band TIFs
            image_id:           Base filename (without extension)
            output_dir:         Where to save outputs
            patch_size:         SAM patch size in pixels
            iou_threshold:      SAM confidence threshold
            cell_size:          Prescription map grid cell size (pixels)
            ground_truth_path:  Optional mask path for evaluation

        Returns:
            dict with all results and output paths
        """
        os.makedirs(output_dir, exist_ok=True)

        # ── Step 1: Load data ─────────────────────────────────
        rgb, bands = load_sample(rgb_path, ms_dir, image_id)

        # ── Step 2: Compute indices ───────────────────────────
        indices   = compute_indices(bands)
        ndvi_8bit = normalize_to_uint8(indices["NDVI"])
        ndvi_rgb  = self._ndvi_to_rgb(ndvi_8bit)

        # ── Step 3: Generate prompts ──────────────────────────
        pt_coords, pt_labels, weed_score, high_infest = generate_prompts(
            indices, bands
        )

        # ── Step 4: SAM segmentation ──────────────────────────
        mcari_img = compose_mcari(rgb, bands)
        pred_mask = run_sam_patches(
            self.predictor, mcari_img,
            pt_coords, pt_labels, weed_score,
            patch_size=patch_size,
            iou_threshold=iou_threshold
        )

        # ── Step 5: Prescription map ──────────────────────────
        presc_rgb, savings_pct, weed_coverage = generate_prescription(
            pred_mask, cell_size=cell_size
        )

        # ── Step 6: Evaluate if ground truth provided ─────────
        metrics = None
        if ground_truth_path:
            gt_mask = np.array(Image.open(ground_truth_path))
            gt_mask = (gt_mask == 255).astype(np.uint8)
            metrics = evaluate(pred_mask, gt_mask)

        # ── Step 7: Save outputs ──────────────────────────────
        prefix = os.path.join(output_dir, image_id)
        Image.fromarray(ndvi_rgb).save(f"{prefix}_ndvi.png")
        Image.fromarray(pred_mask * 255).save(f"{prefix}_mask.png")
        Image.fromarray(presc_rgb).save(f"{prefix}_prescription.png")

        # Save overlay
        overlay = rgb.copy()
        overlay[pred_mask == 1] = (
            overlay[pred_mask == 1] * 0.4 +
            np.array([220, 50, 50]) * 0.6
        ).astype(np.uint8)
        Image.fromarray(overlay).save(f"{prefix}_overlay.png")

        result = {
            "image_id":         image_id,
            "weed_coverage_pct": weed_coverage,
            "herbicide_saved_pct": savings_pct,
            "high_infestation": high_infest,
            "n_weed_prompts":   int((pt_labels == 1).sum()),
            "n_bg_prompts":     int((pt_labels == 0).sum()),
            "metrics":          metrics,
            "outputs": {
                "ndvi":         f"{prefix}_ndvi.png",
                "mask":         f"{prefix}_mask.png",
                "prescription": f"{prefix}_prescription.png",
                "overlay":      f"{prefix}_overlay.png",
            }
        }

        self._print_summary(result)
        return result

    def _ndvi_to_rgb(self, ndvi_8bit):
        lut = np.zeros((256, 1, 3), dtype=np.uint8)
        for i in range(256):
            t = i / 255.0
            if t < 0.5:
                lut[i, 0] = [0, int(t * 2 * 255), 220]
            else:
                lut[i, 0] = [0, 220, int((1 - t) * 2 * 255)]
        gray3 = cv2.cvtColor(ndvi_8bit, cv2.COLOR_GRAY2BGR)
        return cv2.cvtColor(cv2.LUT(gray3, lut), cv2.COLOR_BGR2RGB)

    def _print_summary(self, result):
        print(f"\n{'='*45}")
        print(f"AutoWeedMap Results — {result['image_id']}")
        print(f"{'='*45}")
        print(f"  Weed coverage:     {result['weed_coverage_pct']:.1f}%")
        print(f"  Herbicide saved:   {result['herbicide_saved_pct']:.1f}%")
        print(f"  Infestation level: {'High' if result['high_infestation'] else 'Moderate'}")
        print(f"  Weed prompts:      {result['n_weed_prompts']}")
        if result["metrics"]:
            m = result["metrics"]
            print(f"  IoU:               {m['IoU']:.3f}")
            print(f"  F1:                {m['F1']:.3f}")
        print(f"{'='*45}\n")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="AutoWeedMap Pipeline")
    parser.add_argument("--rgb",        required=True, help="Path to RGB image")
    parser.add_argument("--ms_dir",     required=True, help="MS bands directory")
    parser.add_argument("--image_id",   required=True, help="Base image ID")
    parser.add_argument("--output_dir", default="results/", help="Output directory")
    parser.add_argument("--checkpoint", default="sam_vit_b_01ec64.pth")
    parser.add_argument("--gt_mask",    default=None, help="Ground truth mask")
    args = parser.parse_args()

    pipeline = AutoWeedMap(sam_checkpoint=args.checkpoint)
    pipeline.run(
        rgb_path=args.rgb,
        ms_dir=args.ms_dir,
        image_id=args.image_id,
        output_dir=args.output_dir,
        ground_truth_path=args.gt_mask
    )
