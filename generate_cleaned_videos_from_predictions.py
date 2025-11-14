#!/usr/bin/env python3
"""
Generate cleaned videos from outlier prediction CSV files.

This script reads prediction CSVs containing predicted outlier indices,
loads the corresponding videos with outliers, removes the outlier frames,
and saves them as cleaned videos.

Usage:
    # Generate cleaned videos from predictions
    python generate_cleaned_videos_from_predictions.py --predictions-dir ./outlier_artifacts/cleaned_CSVs --videos-dir ./outlier_artifacts/outlier_videos --output-dir ./outlier_artifacts/cleaned_videos

    # Filter to specific action category
    python generate_cleaned_videos_from_predictions.py --predictions-dir ./outlier_artifacts/cleaned_CSVs --videos-dir ./outlier_artifacts/outlier_videos --output-dir ./outlier_artifacts/cleaned_videos --action-filter Crawling
"""

import os
import csv
import cv2
import glob
import argparse
from tqdm import tqdm
from pathlib import Path


def parse_outlier_list(s: str):
    """Parse comma-separated outlier indices from CSV."""
    if not s or s.strip() == "":
        return []
    return [int(x) for x in str(s).split(",") if x.strip() != ""]


def find_video_path(video_id: str, videos_dir: str, extensions=(".avi", ".mp4", ".mov", ".mkv")):
    """Find the full path of a video file."""
    # Try direct path
    direct = os.path.join(videos_dir, video_id)
    if os.path.isfile(direct):
        return direct

    # Try with extensions
    for ext in extensions:
        if not video_id.endswith(ext):
            path = direct + ext
            if os.path.isfile(path):
                return path

    # Try wildcard match
    pattern = os.path.join(videos_dir, f"{video_id}*")
    candidates = [p for p in glob.glob(pattern)
                  if os.path.splitext(p)[1].lower() in extensions]

    if candidates:
        candidates.sort(key=lambda x: (len(os.path.basename(x)), x))
        return candidates[0]

    raise FileNotFoundError(f"Video not found: {video_id} in {videos_dir}")


def load_video_frames(video_path: str):
    """Load all frames from a video."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video: {video_path}")

    frames = []
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)

    cap.release()

    if len(frames) == 0:
        raise ValueError(f"No frames read from {video_path}")

    return frames, fps, width, height


def save_cleaned_video(frames, outlier_indices, output_path: str, fps: float, width: int, height: int):
    """
    Save video with outlier frames removed.

    Args:
        frames: List of all frames
        outlier_indices: List of frame indices to remove
        output_path: Output video path
        fps: Frame rate
        width: Frame width
        height: Frame height
    """
    if len(frames) == 0:
        raise ValueError("No frames to save")

    # Create set for faster lookup
    outlier_set = set(outlier_indices)

    # Determine codec
    ext = os.path.splitext(output_path)[1].lower()
    if ext == '.mp4':
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    else:
        fourcc = cv2.VideoWriter_fourcc(*'XVID')

    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    if not out.isOpened():
        raise IOError(f"Cannot create video writer: {output_path}")

    frames_kept = 0
    frames_removed = 0

    # Write frames, skipping outliers
    for frame_idx, frame in enumerate(frames):
        if frame_idx not in outlier_set:
            out.write(frame)
            frames_kept += 1
        else:
            frames_removed += 1

    out.release()

    return frames_kept, frames_removed


def process_all_videos(predictions_dir: str, videos_dir: str, output_dir: str, action_filter: str = None):
    """
    Process all prediction CSVs and generate cleaned videos.

    Args:
        predictions_dir: Directory containing prediction CSV files
        videos_dir: Directory containing videos with outliers
        output_dir: Directory to save cleaned videos
        action_filter: Optional filter for specific action
    """
    os.makedirs(output_dir, exist_ok=True)

    # Get all prediction CSV files
    pred_csv_paths = sorted(glob.glob(os.path.join(predictions_dir, "*.csv")))

    if action_filter:
        pred_csv_paths = [p for p in pred_csv_paths if action_filter.lower() in os.path.basename(p).lower()]
        print(f"Filtering to action: {action_filter}")

    if not pred_csv_paths:
        raise FileNotFoundError(f"No CSV files found in {predictions_dir}")

    print("=" * 80)
    print("BATCH VIDEO CLEANING (OUTLIER REMOVAL)")
    print("=" * 80)

    total_videos = 0
    total_frames_kept = 0
    total_frames_removed = 0

    for pred_csv_path in pred_csv_paths:
        action_name = os.path.splitext(os.path.basename(pred_csv_path))[0]

        print(f"\n[{action_name}] Processing action...")

        # Read predictions CSV
        pred_data = {}
        with open(pred_csv_path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                video_id = row['video_id']
                outlier_indices = parse_outlier_list(row['predicted_outliers_list'])
                pred_data[video_id] = outlier_indices

        # Process each video
        videos_in_action = len(pred_data)
        print(f"  Videos in action: {videos_in_action}")

        for video_id, outlier_indices in tqdm(pred_data.items(),
                                               desc=f"  [{action_name}]",
                                               unit="video"):
            total_videos += 1

            try:
                # Find video path
                video_path = find_video_path(video_id, videos_dir)

                # Load video frames
                frames, fps, width, height = load_video_frames(video_path)

                # Generate output filename
                output_filename = f"{os.path.splitext(video_id)[0]}.avi"
                output_path = os.path.join(output_dir, output_filename)

                # Save cleaned video
                frames_kept, frames_removed = save_cleaned_video(
                    frames, outlier_indices, output_path, fps, width, height
                )

                total_frames_kept += frames_kept
                total_frames_removed += frames_removed

            except Exception as e:
                print(f"\n  [ERROR] {video_id}: {str(e)}")
                continue

        print(f"  [OK] {action_name} completed")

    # Final summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Total videos processed: {total_videos}")
    print(f"Total frames kept: {total_frames_kept}")
    print(f"Total frames removed: {total_frames_removed}")
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(
        description="Generate cleaned videos from outlier prediction CSV files"
    )
    parser.add_argument(
        "--predictions-dir",
        type=str,
        required=True,
        help="Directory containing prediction CSV files"
    )
    parser.add_argument(
        "--videos-dir",
        type=str,
        required=True,
        help="Directory containing videos with outliers"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./outlier_artifacts/cleaned_videos",
        help="Directory to save cleaned videos (default: cleaned_videos)"
    )
    parser.add_argument(
        "--action-filter",
        type=str,
        help="Filter to specific action category (e.g., 'Crawling')"
    )

    args = parser.parse_args()

    process_all_videos(
        args.predictions_dir,
        args.videos_dir,
        args.output_dir,
        args.action_filter
    )


if __name__ == "__main__":
    main()
