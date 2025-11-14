#!/usr/bin/env python3
"""
Evaluate outlier detection predictions against ground truth.

Reads prediction CSV files and embedding files (with ground truth labels),
calculates precision, recall, and F1 score for each video.

CSV Format:
  Prediction CSVs (model output):
    Directory: ./outliers_predictions/
    Files: ApplyEyeMakeup.csv, Basketball.csv, ...
    Columns: video_id, predicted_outliers_list

  Embedding files (reference data with ground truth):
    Directory: ./outliers_embeddings/
    Files: ApplyEyeMakeup_clip_embeddings.pt, Basketball_clip_embeddings.pt, ...
    Contents: dict[video_id] = {"embeddings": tensor, "is_outlier": tensor}

Usage:

  python evaluate_outliers_predictions.py --embeddings-dir ./outlier_artifacts/embeddings --predictions ./outlier_artifacts/cleaned_CSVs --model-type clip
  python evaluate_outliers_predictions.py --embeddings-dir ./outlier_artifacts/embeddings --predictions ./outlier_artifacts/cleaned_CSVs --model-type dinov2
  python evaluate_outliers_predictions.py --embeddings-dir ./outlier_artifacts/embeddings --predictions ./outlier_artifacts/cleaned_CSVs --model-type clip --action-filter Crawling
  python evaluate_outliers_predictions.py --embeddings-dir ./outlier_artifacts/embeddings --predictions ./outlier_artifacts/cleaned_CSVs --model-type clip --max-videos 100
"""

import argparse
import os
import glob
import csv
import numpy as np
import torch
from pathlib import Path


def parse_outlier_list(s: str):
    """Parse comma-separated outlier indices from CSV."""
    if not s or s.strip() == "":
        return []
    return [int(x) for x in str(s).split(",") if x.strip() != ""]


def to_bool(x, length=None):
    """Convert to boolean array with optional length adjustment."""
    if x is None:
        return None
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    x = np.asarray(x, dtype=bool)
    if length is not None and len(x) != length:
        if len(x) == 1:
            x = np.repeat(x.item(), length)
        else:
            x = np.pad(x[:length], (0, max(0, length - len(x))), constant_values=False)
    return x


def indices_to_boolean(indices, length):
    """Convert list of indices to boolean array."""
    boolean_array = np.zeros(length, dtype=bool)
    for idx in indices:
        if 0 <= idx < length:
            boolean_array[idx] = True
    return boolean_array


def compute_metrics(pred, gt):
    """
    Calculate precision, recall, and F1 score.

    Args:
        pred: Boolean array of predictions
        gt: Boolean array of ground truth

    Returns:
        Tuple of (precision, recall, f1)
    """
    if gt is None:
        return None, None, None

    pred, gt = np.asarray(pred, bool), np.asarray(gt, bool)

    tp = np.logical_and(pred, gt).sum()
    fp = np.logical_and(pred, ~gt).sum()
    fn = np.logical_and(~pred, gt).sum()

    precision = tp / (tp + fp + 1e-12)
    recall = tp / (tp + fn + 1e-12)
    f1 = 2 * precision * recall / (precision + recall + 1e-12)

    return precision, recall, f1


def extract_action_name(filename, model_type):
    """Extract action category from embedding filename based on model type."""
    name = os.path.basename(filename)
    suffix = f'_{model_type}_embeddings'
    name = name.replace(suffix + '.pt', '').replace(suffix + '.pth', '')
    return name


def evaluate_all(embeddings_dir, predictions_dir, model_type='clip', max_videos=None, action_filter=None):
    """
    Evaluate all predictions against ground truth.

    Args:
        embeddings_dir: Directory containing embedding .pt files (with ground truth)
        predictions_dir: Directory containing prediction CSV files
        model_type: Model type to load ('clip' or 'dinov2')
        max_videos: Limit evaluation to first N videos
        action_filter: Filter to specific action category
    """
    # Get embedding files filtered by model type
    pattern = f"*_{model_type}_embeddings.pt"
    emb_files = sorted(glob.glob(os.path.join(embeddings_dir, pattern)))

    if action_filter:
        emb_files = [f for f in emb_files if action_filter.lower() in os.path.basename(f).lower()]
        print(f"Filtering to action: {action_filter}")
        print(f"Found {len(emb_files)} matching file(s)")

    print("OUTLIER DETECTION EVALUATION")

    precision_list, recall_list, f1_list = [], [], []
    total_videos = 0
    missing_predictions = 0

    for emb_path in emb_files:
        action_name = extract_action_name(emb_path, model_type)
        pred_csv_path = os.path.join(predictions_dir, f"{action_name}.csv")

        print(f"\nProcessing action: {action_name}")

        # Check if prediction CSV exists
        if not os.path.exists(pred_csv_path):
            print(f"  [ERROR] Prediction CSV not found: {pred_csv_path}")
            continue

        # Load embeddings (with ground truth)
        emb_data = torch.load(emb_path, map_location="cpu")

        # Read predictions CSV
        predictions = {}
        with open(pred_csv_path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                video_id = row['video_id']
                outlier_indices = parse_outlier_list(row['predicted_outliers_list'])
                predictions[video_id] = outlier_indices

        # Evaluate each video
        for video_name, video_data in emb_data.items():
            if max_videos and total_videos >= max_videos:
                break

            total_videos += 1
            embeddings = video_data["embeddings"]
            num_frames = len(embeddings)

            # Get ground truth
            ground_truth = to_bool(video_data.get("is_outlier"), num_frames)

            if ground_truth is None:
                print(f"  [{total_videos}] {video_name}: No ground truth available - skipping")
                continue

            # Get predictions
            if video_name not in predictions:
                print(f"  [{total_videos}] {video_name}: Not found in predictions CSV - skipping")
                missing_predictions += 1
                continue

            # Convert prediction indices to boolean array
            pred_indices = predictions[video_name]
            pred_boolean = indices_to_boolean(pred_indices, num_frames)

            # Compute metrics
            precision, recall, f1 = compute_metrics(pred_boolean, ground_truth)

            precision_list.append(precision)
            recall_list.append(recall)
            f1_list.append(f1)

            num_pred = len(pred_indices)
            num_gt = ground_truth.sum()
            print(f"  [{total_videos}] {video_name} ({num_frames} frames):  "
                  f"Pred={num_pred} GT={num_gt}  "
                  f"P={precision:.3f} R={recall:.3f} F1={f1:.3f}")

        if max_videos and total_videos >= max_videos:
            break

    # Print summary
    print("\n" + "=" * 80)
    print("EVALUATION RESULTS")
    print("=" * 80)

    if precision_list:
        print(f"Precision: {np.mean(precision_list):.4f}")
        print(f"Recall:    {np.mean(recall_list):.4f}")
        print(f"F1 Score:  {np.mean(f1_list):.4f}")
        print(f"Videos:    {total_videos}")
        if missing_predictions > 0:
            print(f"Missing:   {missing_predictions} videos not found in predictions")
    else:
        print("No metrics computed (missing ground-truth labels or predictions).")

    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate outlier detection predictions against ground truth"
    )

    parser.add_argument("--embeddings-dir", required=True,
                       help="Directory containing embedding .pt files (with ground truth)")
    parser.add_argument("--predictions", required=True,
                       help="Directory containing prediction CSV files")
    parser.add_argument("--model-type", type=str, choices=['clip', 'dinov2'], default='clip',
                       help="Model type to load: 'clip' or 'dinov2' (default: clip)")
    parser.add_argument("--action-filter",
                       help="Filter to specific action category")
    parser.add_argument("--max-videos", type=int,
                       help="Limit evaluation to first N videos")

    args = parser.parse_args()

    evaluate_all(
        embeddings_dir=args.embeddings_dir,
        predictions_dir=args.predictions,
        model_type=args.model_type,
        max_videos=args.max_videos,
        action_filter=args.action_filter
    )


if __name__ == "__main__":
    main()
