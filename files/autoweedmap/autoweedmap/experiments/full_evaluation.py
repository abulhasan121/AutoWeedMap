"""
experiments/full_evaluation.py
Run the complete benchmark evaluation on the test set.

Compares:
1. NDVI Threshold baseline
2. SAM + Grid Prompts baseline
3. Grounded SAM baseline (RGB-only)
4. AutoWeedMap (NDVI-guided SAM)

Usage:
    python experiments/full_evaluation.py \
        --root /path/to/WeedyRice-RGBMS-DB \
        --checkpoint sam_vit_b_01ec64.pth \
        --output results/
"""

import argparse
import os
import json
import numpy as np
import pandas as pd
from tqdm import tqdm

from src.data.loader import load_ids, scan_valid_images
from src.indices.vegetation import compute_indices, compose_mcari
from src.prompts.ndvi_prompter import generate_prompts
from src.segmentation.sam_segmenter import load_sam, run_sam_patches
from src.prescription.map_generator import generate_prescription
from src.evaluate.metrics import evaluate, summarize_results
from baselines.baselines import (
    ndvi_threshold_baseline,
    grid_prompt_baseline,
)


def load_image_data(root, image_id):
    from src.data.loader import load_sample, load_mask
    import os
    rgb_path  = os.path.join(root, "RGB",   f"{image_id}.JPG")
    ms_dir    = os.path.join(root, "Multispectral")
    mask_path = os.path.join(root, "Masks", f"{image_id}.png")
    rgb, bands = load_sample(rgb_path, ms_dir, image_id)
    mask       = load_mask(mask_path)
    return rgb, bands, mask


def run_evaluation(root, checkpoint, output_dir,
                   min_weed_pct=10.0, n_samples=None):

    os.makedirs(output_dir, exist_ok=True)

    # Load model
    predictor = load_sam(checkpoint)

    # Get valid test images
    valid_ids, all_stats = scan_valid_images(root, "test", min_weed_pct)
    if n_samples:
        valid_ids = valid_ids[:n_samples]

    print(f"Evaluating on {len(valid_ids)} images (≥{min_weed_pct}% weed)")

    results = []

    for img_id in tqdm(valid_ids, desc="Evaluating"):
        try:
            rgb, bands, mask = load_image_data(root, img_id)
            indices          = compute_indices(bands)
            input_img        = compose_mcari(rgb, bands)
            weed_pct         = mask.mean() * 100

            row = {"image_id": img_id, "weed_pct": round(weed_pct, 1)}

            # 1. NDVI Threshold
            pred_t = ndvi_threshold_baseline(bands)
            m_t    = evaluate(pred_t, mask)
            row["ndvi_threshold_iou"] = m_t["IoU"]

            # 2. Grid SAM
            pt, pl, ws, _ = generate_prompts(indices, bands)
            pred_g = grid_prompt_baseline(predictor, input_img, ws)
            m_g    = evaluate(pred_g, mask)
            row["grid_sam_iou"] = m_g["IoU"]

            # 3. AutoWeedMap
            pred_a = run_sam_patches(
                predictor, input_img, pt, pl, ws,
                patch_size=192, iou_threshold=0.6
            )
            m_a    = evaluate(pred_a, mask)
            row["autoweedmap_iou"]       = m_a["IoU"]
            row["autoweedmap_f1"]        = m_a["F1"]
            row["autoweedmap_precision"] = m_a["Precision"]
            row["autoweedmap_recall"]    = m_a["Recall"]

            # Prescription stats
            _, savings, coverage, zone_stats = generate_prescription(pred_a)
            row["herbicide_saved_pct"] = round(savings, 1)

            results.append(row)

        except Exception as e:
            print(f"  Error on {img_id}: {e}")

    # Save results
    df = pd.DataFrame(results)
    df.to_csv(os.path.join(output_dir, "full_results.csv"), index=False)

    # Print summary
    print(f"\n{'='*55}")
    print("FINAL RESULTS")
    print(f"{'='*55}")
    print(f"Images evaluated: {len(results)}")
    print(f"\n{'Method':<30} | {'Mean IoU':>8} | {'Std':>6}")
    print(f"{'-'*50}")

    for col, name in [
        ("ndvi_threshold_iou", "NDVI Threshold"),
        ("grid_sam_iou",       "SAM + Grid Prompts"),
        ("autoweedmap_iou",    "AutoWeedMap (ours)"),
    ]:
        vals = df[col].dropna()
        print(f"{name:<30} | {vals.mean():>8.3f} | {vals.std():>6.3f}")

    print(f"\nMean herbicide saved: {df['herbicide_saved_pct'].mean():.1f}%")
    print(f"Results saved to: {output_dir}")

    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root",       required=True)
    parser.add_argument("--checkpoint", default="sam_vit_b_01ec64.pth")
    parser.add_argument("--output_dir", default="results/")
    parser.add_argument("--min_weed",   type=float, default=10.0)
    parser.add_argument("--n_samples",  type=int,   default=None)
    args = parser.parse_args()

    run_evaluation(
        root=args.root,
        checkpoint=args.checkpoint,
        output_dir=args.output_dir,
        min_weed_pct=args.min_weed,
        n_samples=args.n_samples
    )
