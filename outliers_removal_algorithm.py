#!/usr/bin/env python3
"""
Outlier removal algorithm for video frame embeddings using DBSCAN.

Reads embeddings, detects outliers, and exports predictions to CSV files.
GPU acceleration is automatically detected and used if available.

Usage:
  # Process CLIP embeddings from outlier_artifacts
  python outliers_removal_algorithm.py --embeddings-dir ./outlier_artifacts/embeddings --output-dir ./outlier_artifacts/cleaned_CSVs --model-type clip

  # Process DINOv2 embeddings
  python outliers_removal_algorithm.py --embeddings-dir ./outlier_artifacts/embeddings --output-dir ./outlier_artifacts/cleaned_CSVs --model-type dinov2

  # Custom DBSCAN parameters with CLIP embeddings
  python outliers_removal_algorithm.py --embeddings-dir ./outlier_artifacts/embeddings --output-dir ./outlier_artifacts/cleaned_CSVs --model-type clip --eps 0.45 --min-samples 50

  # Filter to specific action category
  python outliers_removal_algorithm.py --embeddings-dir ./outlier_artifacts/embeddings --output-dir ./outlier_artifacts/cleaned_CSVs --model-type clip --action-filter Crawling

  # Limit processing to first 10 videos
  python outliers_removal_algorithm.py --embeddings-dir ./outlier_artifacts/embeddings --output-dir ./outlier_artifacts/cleaned_CSVs --model-type clip --max-videos 10

Note: To generate cleaned videos from predictions, use generate_cleaned_videos_from_predictions.py
"""

import os
import glob
import csv
import argparse
import numpy as np
import torch
from pathlib import Path

try:
    import cupy as cp
    from cuml.cluster import DBSCAN as cuDBSCAN
    CUML_AVAILABLE = True
except ImportError:
    CUML_AVAILABLE = False

from sklearn.cluster import DBSCAN as skDBSCAN

# Automatically detect GPU availability
USE_GPU = CUML_AVAILABLE and torch.cuda.is_available()


def to_numpy(x):
    """Convert tensor or array to numpy float32."""
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    return np.asarray(x, dtype=np.float32)


def dbscan_outliers(X, eps=0.55, min_samples=10):
    """
    Detect outliers using DBSCAN (noise points).

    Args:
        X: Feature matrix (N, D)
        eps: DBSCAN epsilon parameter
        min_samples: DBSCAN minimum samples parameter

    Returns:
        Boolean array of shape (N,) where True = outlier
    """
    X = to_numpy(X)
    if USE_GPU:
        labels = cuDBSCAN(eps=eps, min_samples=min_samples).fit_predict(cp.asarray(X)).get()
    else:
        labels = skDBSCAN(eps=eps, min_samples=min_samples, n_jobs=-1).fit_predict(X)
    return labels == -1


def extract_action_name(filename, model_type):
    """Extract action category from embedding filename based on model type."""
    name = os.path.basename(filename)
    suffix = f'_{model_type}_embeddings'
    name = name.replace(suffix + '.pt', '').replace(suffix + '.pth', '')
    return name


def process_all_embeddings(emb_dir, eps, min_samples, output_dir, model_type='clip',
                           max_videos=None, action_filter=None):
    """
    Process all embeddings and export predictions to CSV files.

    Args:
        emb_dir: Directory containing embedding .pt files
        eps: DBSCAN epsilon parameter
        min_samples: DBSCAN minimum samples parameter
        output_dir: Directory to save CSV predictions
        model_type: Model type to load ('clip' or 'dinov2')
        max_videos: Limit processing to first N videos
        action_filter: Filter to specific action category
    """
    # Filter files by model type (e.g., *_clip_embeddings.pt or *_dinov2_embeddings.pt)
    pattern = f"*_{model_type}_embeddings.pt"
    pt_files = sorted(glob.glob(os.path.join(emb_dir, pattern)))

    if action_filter:
        pt_files = [f for f in pt_files if action_filter.lower() in os.path.basename(f).lower()]
        print(f"Filtering to action: {action_filter}")
        print(f"Found {len(pt_files)} matching file(s)")

    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True, parents=True)

    print("=" * 80)
    print("OUTLIER REMOVAL ALGORITHM - DBSCAN")
    print("=" * 80)
    print(f"Model type: {model_type.upper()}")
    print(f"GPU Acceleration: {'Enabled (cuML)' if USE_GPU else 'Disabled (CPU/sklearn)'}")
    print(f"Embeddings dir: {emb_dir}")
    print(f"Output dir: {output_dir}")
    print(f"DBSCAN parameters: eps={eps}, min_samples={min_samples}")
    print(f"Total embedding files: {len(pt_files)}")
    print("=" * 80)

    total_videos = 0

    for pt_path in pt_files:
        data = torch.load(pt_path, map_location="cpu")
        action_name = extract_action_name(pt_path, model_type)
        print(f"\nProcessing action: {action_name}")

        # Create CSV for this action
        csv_path = output_path / f"{action_name}.csv"

        with open(csv_path, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['video_id', 'predicted_outliers_list'])

            for video_name, video_data in data.items():
                if max_videos and total_videos >= max_videos:
                    break

                total_videos += 1
                embeddings = video_data["embeddings"]

                # Run DBSCAN outlier detection
                predictions = dbscan_outliers(embeddings, eps=eps, min_samples=min_samples)

                # Convert boolean array to list of outlier indices
                outlier_indices = np.where(predictions)[0].tolist()
                outliers_str = ",".join(map(str, outlier_indices))

                # Write to CSV
                writer.writerow([video_name, outliers_str])

                num_outliers = predictions.sum()
                num_frames = len(embeddings)
                print(f"  [{total_videos}] {video_name} ({num_frames} frames): {num_outliers} outliers detected")

        print(f"  Saved: {csv_path}")

        if max_videos and total_videos >= max_videos:
            break

    print("\n" + "=" * 80)
    print("PROCESSING COMPLETE")
    print("=" * 80)
    print(f"Total videos processed: {total_videos}")
    print(f"CSV files saved to: {output_path.absolute()}")
    print("\nNext step: Use generate_cleaned_videos_from_predictions.py to create cleaned videos")
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(
        description="Outlier removal algorithm using DBSCAN: detect outliers and export predictions to CSV"
    )

    parser.add_argument("--embeddings-dir", required=True,
                       help="Directory containing embedding .pt files")
    parser.add_argument("--output-dir", default="./outlier_artifacts/cleaned_CSVs",
                       help="Directory to save prediction CSV files")
    parser.add_argument("--model-type", type=str, choices=['clip', 'dinov2'], default='clip',
                       help="Model type to load: 'clip' or 'dinov2' (default: clip)")
    parser.add_argument("--max-videos", type=int,
                       help="Limit processing to first N videos")
    parser.add_argument("--action-filter",
                       help="Filter to specific action category (e.g., 'Crawling')")

    # DBSCAN parameters
    parser.add_argument("--eps", type=float, default=0.5,
                       help="DBSCAN: Epsilon parameter")
    parser.add_argument("--min-samples", type=int, default=40,
                       help="DBSCAN: Minimum samples parameter")

    args = parser.parse_args()

    process_all_embeddings(
        emb_dir=args.embeddings_dir,
        eps=args.eps,
        min_samples=args.min_samples,
        output_dir=args.output_dir,
        model_type=args.model_type,
        max_videos=args.max_videos,
        action_filter=args.action_filter
    )


if __name__ == "__main__":
    main()
