#!/usr/bin/env python3
"""
Generate ordered videos from predictions CSV files.

This script reads predictions CSVs containing predicted_frames_list,
loads the corresponding shuffled videos, reorders frames according
to the predictions, and saves them as ordered videos.

Usage:
    # Generate ordered videos from predictions
    python generate_ordered_videos_from_predictions.py --predictions-dir ./shuffled_artifacts/predictions --shuffled-csvs ./shuffled_artifacts/shuffled_CSVs --shuffled-videos ./shuffled_artifacts/shuffled_videos --output-dir ./shuffled_artifacts/ordered_videos
"""

import os
import csv
import cv2
import glob
import argparse
from tqdm import tqdm


def parse_frame_list(s: str):
    """Parse comma-separated frame list from CSV."""
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
    """Load all frames from a video as BGR (color)."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video: {video_path}")

    frames = []
    fps = cap.get(cv2.CAP_PROP_FPS)

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)

    cap.release()

    if len(frames) == 0:
        raise ValueError(f"No frames read from {video_path}")

    return frames, fps


def save_ordered_video(frames, frame_order, output_path: str, fps: float = 25.0):
    """
    Save frames in specified order.

    Args:
        frames: List of BGR frames
        frame_order: List of frame indices in desired order
        output_path: Output video path
        fps: Frame rate
    """
    if len(frames) == 0:
        raise ValueError("No frames to save")

    height, width = frames[0].shape[:2]

    # Use XVID codec for .avi, or mp4v for .mp4
    ext = os.path.splitext(output_path)[1].lower()
    if ext == '.mp4':
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    else:
        fourcc = cv2.VideoWriter_fourcc(*'XVID')

    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    if not out.isOpened():
        raise IOError(f"Cannot create video writer: {output_path}")

    # Write frames in specified order
    for frame_idx in frame_order:
        if 0 <= frame_idx < len(frames):
            out.write(frames[frame_idx])
        else:
            print(f"[WARN] Frame index {frame_idx} out of range [0, {len(frames)-1})")

    out.release()


def convert_prediction_to_order(predicted_frames, shuffled_frames):
    """
    Convert predicted_frames_list to an index order for the shuffled video.

    Args:
        predicted_frames: List of original frame numbers in predicted order
                         e.g., [0, 1, 2, 3, ...] if prediction is perfect
        shuffled_frames: List showing which original frame is at each position
                        e.g., [4, 1, 2, 0, 3] means position 0 has frame 4

    Returns:
        idx_order: List of indices to read from shuffled video
                  e.g., if predicted=[0,1,2,3,4] and shuffled=[4,1,2,0,3]
                  then idx_order=[3,1,2,4,0] to get frames in 0,1,2,3,4 order
    """
    # Create a mapping: original_frame_number -> position_in_shuffled_video
    frame_to_position = {frame_num: pos for pos, frame_num in enumerate(shuffled_frames)}

    # For each predicted frame number, find its position in the shuffled video
    idx_order = []
    for frame_num in predicted_frames:
        if frame_num in frame_to_position:
            idx_order.append(frame_to_position[frame_num])
        else:
            print(f"[WARN] Frame {frame_num} not found in shuffled_frames")

    return idx_order


def process_all_videos(predictions_dir: str, shuffled_csvs_dir: str,
                      shuffled_videos_dir: str, output_dir: str):
    """
    Process all predictions CSVs and generate ordered videos.

    Args:
        predictions_dir: Directory containing prediction CSV files
        shuffled_csvs_dir: Directory containing shuffled CSV files (ground truth)
        shuffled_videos_dir: Directory containing shuffled videos
        output_dir: Directory to save ordered videos
    """
    os.makedirs(output_dir, exist_ok=True)

    # Get all prediction CSV files
    pred_csv_paths = sorted(glob.glob(os.path.join(predictions_dir, "*.csv")))

    if not pred_csv_paths:
        raise FileNotFoundError(f"No CSV files found in {predictions_dir}")


    total_videos = 0
    successful = 0
    failed = 0

    for pred_csv_path in pred_csv_paths:
        action_name = os.path.splitext(os.path.basename(pred_csv_path))[0]
        shuffled_csv_path = os.path.join(shuffled_csvs_dir, f"{action_name}.csv")

        print(f"\n[{action_name}] Processing action...")

        # Check if shuffled CSV exists
        if not os.path.exists(shuffled_csv_path):
            print(f"  [ERROR] Shuffled CSV not found: {shuffled_csv_path}")
            continue

        # Read predictions CSV
        pred_data = {}
        with open(pred_csv_path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                video_id = row['video_id']
                predicted_frames = parse_frame_list(row['predicted_frames_list'])
                pred_data[video_id] = predicted_frames

        # Read shuffled CSV (ground truth)
        shuffled_data = {}
        with open(shuffled_csv_path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                video_id = row['video_id']
                shuffled_frames = parse_frame_list(row['shuffled_frames_list'])
                shuffled_data[video_id] = shuffled_frames

        # Process each video
        videos_in_action = len(pred_data)
        print(f"  Videos in action: {videos_in_action}")

        for video_id, predicted_frames in tqdm(pred_data.items(),
                                               desc=f"  [{action_name}]",
                                               unit="video"):
            total_videos += 1

            try:
                # Get shuffled order for this video
                if video_id not in shuffled_data:
                    print(f"\n  [ERROR] {video_id}: Not found in shuffled CSV")
                    failed += 1
                    continue

                shuffled_frames = shuffled_data[video_id]

                # Verify lengths match
                if len(predicted_frames) != len(shuffled_frames):
                    print(f"\n  [WARN] {video_id}: Length mismatch - "
                          f"predicted={len(predicted_frames)}, shuffled={len(shuffled_frames)}")

                # Find shuffled video
                video_path = find_video_path(video_id, shuffled_videos_dir)

                # Load shuffled video frames
                frames, fps = load_video_frames(video_path)

                # Verify frame count
                if len(frames) != len(shuffled_frames):
                    print(f"\n  [WARN] {video_id}: Video frame count mismatch - "
                          f"video={len(frames)}, csv={len(shuffled_frames)}")

                # Convert predicted frames to index order
                idx_order = convert_prediction_to_order(predicted_frames, shuffled_frames)

                # Generate output filename
                output_filename = f"{os.path.splitext(video_id)[0]}.avi"
                output_path = os.path.join(output_dir, output_filename)

                # Save ordered video
                save_ordered_video(frames, idx_order, output_path, fps=fps)

                successful += 1

            except Exception as e:
                print(f"\n  [ERROR] {video_id}: {str(e)}")
                failed += 1
                continue

        print(f"  [OK] {action_name} completed")

    # Final summary
    print("=" * 80)
    print(f"Total videos processed: {total_videos}")
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(
        description="Generate ordered videos from predictions CSV files"
    )
    parser.add_argument(
        "--predictions-dir",
        type=str,
        required=True,
        help="Directory containing prediction CSV files"
    )
    parser.add_argument(
        "--shuffled-csvs",
        type=str,
        required=True,
        help="Directory containing shuffled CSV files (ground truth)"
    )
    parser.add_argument(
        "--shuffled-videos",
        type=str,
        required=True,
        help="Directory containing shuffled videos"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="ordered_videos",
        help="Directory to save ordered videos (default: ordered_videos)"
    )

    args = parser.parse_args()

    process_all_videos(
        args.predictions_dir,
        args.shuffled_csvs,
        args.shuffled_videos,
        args.output_dir
    )


if __name__ == "__main__":
    main()
