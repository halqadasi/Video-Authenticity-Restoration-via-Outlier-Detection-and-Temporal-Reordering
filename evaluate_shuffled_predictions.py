#!/usr/bin/env python3
"""
Evaluate frame order reconstruction using Kendall's Tau and Edit Distance.

Evaluation Logic (Bidirectional):
  1. Ground truth: [0, 1, 2, ..., N-1] (original frame order)
  2. Model receives: Shuffled video with frames in order [45, 12, 89, 3, ...]
  3. Model predicts: Attempt to reconstruct original order
  4. Evaluation: Compare against BOTH forward [0,1,2,...,N-1] AND backward [N-1,...,2,1,0]
  5. Select the direction with higher Kendall's Tau (algorithm can't distinguish direction)

CSV Format (one CSV per action, videos as rows):
  Shuffled CSVs (reference data - for video list and frame counts):
    Directory: ./shuffled_CSVs/
    Files: ApplyEyeMakeup.csv, Basketball.csv, ...
    Columns: video_id, shuffled_frames_list
    Note: shuffled_frames_list is NOT used for evaluation, only to get frame count

  Predictions CSVs (model output):
    Directory: ./predictions_csvs/
    Files: ApplyEyeMakeup.csv, Basketball.csv, ...
    Columns: video_id, predicted_frames_list

Usage:
  python evaluate_shuffled_predictions.py --csv-dir ./shuffled_artifacts/shuffled_CSVs --predictions ./shuffled_artifacts/ordered_CSVs
  python evaluate_shuffled_predictions.py --csv-dir ./shuffled_artifacts/shuffled_CSVs --predictions ./shuffled_artifacts/ordered_CSVs --action-filter ApplyEyeMakeup
"""

import argparse
import csv
import numpy as np
from pathlib import Path
from tqdm import tqdm
from scipy.stats import kendalltau


def compute_kendall_tau(true_order, pred_order):
    """
    Compute Kendall's Tau correlation coefficient.
    Measures pairwise rank correlation.

    Returns:
        tau: float in [-1, 1]
            +1 = perfect order
             0 = random order
            -1 = completely reversed
    """
    tau, _ = kendalltau(true_order, pred_order)
    return tau


def levenshtein_distance(seq1, seq2):
    """
    Compute Levenshtein distance (edit distance) between two sequences.
    Pure Python implementation.

    Args:
        seq1, seq2: Lists or sequences to compare

    Returns:
        int: Minimum number of single-element edits (insertions, deletions, substitutions)
    """
    if len(seq1) < len(seq2):
        return levenshtein_distance(seq2, seq1)

    if len(seq2) == 0:
        return len(seq1)

    # Use only two rows for space efficiency
    previous_row = range(len(seq2) + 1)

    for i, c1 in enumerate(seq1):
        current_row = [i + 1]
        for j, c2 in enumerate(seq2):
            # Cost of insertions, deletions, or substitutions
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row

    return previous_row[-1]


def compute_normalized_edit_distance(true_order, pred_order):
    """
    Compute normalized edit distance (Levenshtein distance).
    Measures sequence similarity with local penalties.

    Returns:
        score: float in [0, 1]
            1 = perfect order
            0 = completely disordered
    """
    d_edit = levenshtein_distance(true_order, pred_order)
    score = 1.0 - (d_edit / len(true_order))
    return score


def evaluate_video(true_order, pred_order):
    """
    Evaluate frame order reconstruction for a single video.

    Since the algorithm cannot distinguish forward vs backward temporal direction,
    we evaluate against both natural order and reversed order, taking the best score.

    Args:
        true_order: Ground truth order [0, 1, 2, ..., N-1]
        pred_order: Predicted order from model

    Returns:
        dict with kendall_tau, edit_distance_score, and direction ('forward' or 'backward')
    """
    # Evaluate against natural order (forward)
    tau_forward = compute_kendall_tau(true_order, pred_order)
    edit_forward = compute_normalized_edit_distance(true_order, pred_order)

    # Evaluate against reversed order (backward)
    reversed_order = list(reversed(true_order))
    tau_backward = compute_kendall_tau(reversed_order, pred_order)
    edit_backward = compute_normalized_edit_distance(reversed_order, pred_order)

    # Choose the direction with better Kendall's Tau
    if tau_forward >= tau_backward:
        return {
            'kendall_tau': tau_forward,
            'edit_distance_score': edit_forward,
            'direction': 'forward'
        }
    else:
        return {
            'kendall_tau': tau_backward,
            'edit_distance_score': edit_backward,
            'direction': 'backward'
        }


def load_predictions_dir(predictions_dir):
    """
    Load predicted orderings from directory of CSV files (one per action).

    Assumes CSV format:
        Column 0: video_id
        Column 1: predicted frames list (comma-separated)
    """
    predictions_path = Path(predictions_dir)
    predictions = {}

    csv_files = list(predictions_path.glob("*.csv"))

    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {predictions_path}")

    for csv_file in csv_files:
        with open(csv_file, 'r') as f:
            reader = csv.reader(f)

            # Skip header row
            next(reader, None)

            # Load predictions: column 0 = video_id, column 1 = predictions
            for row in reader:
                if len(row) < 2:
                    continue

                video_id = row[0]
                pred_order = list(map(int, row[1].split(',')))
                predictions[video_id] = pred_order

    return predictions


def evaluate_all(csv_dir, predictions_dir, action_filter=None, max_videos=None):
    """
    Evaluate frame order reconstruction across all videos.

    Args:
        csv_dir: Directory containing shuffled CSV files (to get video list and frame counts)
        predictions_dir: Directory with prediction CSVs (one per action)
        action_filter: Optional action name to filter
        max_videos: Optional limit on number of videos
    """
    csv_path = Path(csv_dir)
    csv_files = sorted(list(csv_path.glob("*.csv")))

    if action_filter:
        csv_files = [f for f in csv_files if action_filter.lower() in f.stem.lower()]
        print(f"Filtering to action: {action_filter}")
        print(f"Found {len(csv_files)} matching file(s)")

    predictions = load_predictions_dir(predictions_dir)
    print(f"Loaded predictions for {len(predictions)} videos")

    tau_list = []
    edit_list = []
    direction_counts = {'forward': 0, 'backward': 0}
    total_videos = 0

    print("\n" + "=" * 80)
    print("FRAME ORDER RECONSTRUCTION EVALUATION")
    print("=" * 80)

    for csv_file in tqdm(csv_files, desc="Processing actions", unit="action"):
        action_name = csv_file.stem

        with open(csv_file, 'r') as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        for row in rows:
            if max_videos and total_videos >= max_videos:
                break

            video_id = row['video_id']
            shuffled_order = list(map(int, row['shuffled_frames_list'].split(',')))
            n_frames = len(shuffled_order)

            if video_id not in predictions:
                continue

            pred_order = predictions[video_id]
            if len(pred_order) != n_frames:
                print(f"Warning: Prediction length mismatch for {video_id} (expected {n_frames}, got {len(pred_order)})")
                continue

            true_order = list(range(n_frames))
            metrics = evaluate_video(true_order, pred_order)

            tau_list.append(metrics['kendall_tau'])
            edit_list.append(metrics['edit_distance_score'])
            direction_counts[metrics['direction']] += 1

            total_videos += 1

        if max_videos and total_videos >= max_videos:
            break

    if tau_list:
        print("\n" + "=" * 80)
        print("EVALUATION RESULTS")
        print("=" * 80)
        print(f"Videos evaluated: {total_videos}")

        print(f"\nTemporal Direction:")
        print(f"  Forward:  {direction_counts['forward']} ({100*direction_counts['forward']/total_videos:.1f}%)")
        print(f"  Backward: {direction_counts['backward']} ({100*direction_counts['backward']/total_videos:.1f}%)")

        print(f"\nKendall's Tau (τ):")
        print(f"  Mean:   {np.mean(tau_list):.4f}")
        print(f"  Std:    {np.std(tau_list):.4f}")
        print(f"  Median: {np.median(tau_list):.4f}")
        print(f"  Range:  [{np.min(tau_list):.4f}, {np.max(tau_list):.4f}]")

        print(f"\nNormalized Edit Distance Score:")
        print(f"  Mean:   {np.mean(edit_list):.4f}")
        print(f"  Std:    {np.std(edit_list):.4f}")
        print(f"  Median: {np.median(edit_list):.4f}")
        print(f"  Range:  [{np.min(edit_list):.4f}, {np.max(edit_list):.4f}]")

        print("\n" + "=" * 80)
        print("INTERPRETATION:")
        print("=" * 80)

        tau_mean = np.mean(tau_list)
        edit_mean = np.mean(edit_list)

        if tau_mean > 0.8 and edit_mean > 0.8:
            print("  Excellent reconstruction (>0.8 on both metrics)")
        elif tau_mean > 0.5 and edit_mean > 0.5:
            print("  Moderate reconstruction (0.5-0.8 range)")
        else:
            print("  Poor reconstruction (<0.5 on metrics)")

        print("=" * 80)
    else:
        print("No videos evaluated.")



def main():
    parser = argparse.ArgumentParser(
        description="Evaluate frame order reconstruction (bidirectional)"
    )

    parser.add_argument("--csv-dir", required=True,
                       help="Directory containing shuffled CSV files (reference data)")
    parser.add_argument("--predictions", required=True,
                       help="Directory containing prediction CSV files (one per action)")
    parser.add_argument("--action-filter", help="Filter to specific action category")
    parser.add_argument("--max-videos", type=int, help="Limit evaluation to first N videos")

    args = parser.parse_args()

    evaluate_all(
        csv_dir=args.csv_dir,
        predictions_dir=args.predictions,
        action_filter=args.action_filter,
        max_videos=args.max_videos
    )


if __name__ == "__main__":
    main()
