#!/usr/bin/env python3
"""
Generate shuffled frame dataset for temporal order detection.

Creates videos with completely shuffled frames where NO frame remains in its original position.
Generates CSV files and shuffled videos for the first 100 action categories.

CSV Format: video_id, shuffled_frames_list
Example: v_ApplyEyeMakeup_g01_c01.avi, [45,12,89,3,...]

Usage:
  # Generate both CSVs and videos for 40 action categories
  python ./dataset_preprocessing/generate_shuffled_videos.py --videos-dir ./UCF101_videos --output-csv-dir ./shuffled_artifacts/shuffled_CSVs --output-videos-dir ./shuffled_artifacts/shuffled_videos --num-actions 40

  # Direct shuffle: shuffle videos without CSVs (for inference/testing)
  python ./dataset_preprocessing/generate_shuffled_videos.py --videos-dir ./inference --output-videos-dir ./inference_shuffled --direct-shuffle

  # Direct shuffle with CSV generation
  python ./dataset_preprocessing/generate_shuffled_videos.py --videos-dir ./inference --output-videos-dir ./inference_shuffled --output-csv-dir ./inference_csvs --direct-shuffle

  # Generate only CSVs (videos already exist)
  python ./dataset_preprocessing/generate_shuffled_videos.py --videos-dir ./UCF101_videos --output-csv-dir ./shuffled_artifacts/shuffled_CSVs --output-videos-dir ./shuffled_artifacts/shuffled_videos --csv-only

  # Generate only videos (CSVs already exist)
  python ./dataset_preprocessing/generate_shuffled_videos.py --output-csv-dir ./shuffled_artifacts/shuffled_CSVs --output-videos-dir ./shuffled_artifacts/shuffled_videos --videos-only

  # Process all 100 action categories
  python ./dataset_preprocessing/generate_shuffled_videos.py --videos-dir ./UCF101_videos --output-csv-dir ./shuffled_artifacts/shuffled_CSVs --output-videos-dir ./shuffled_artifacts/shuffled_videos --num-actions 100
"""

import csv
import cv2
import random
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm
import argparse
import os


def get_action_categories(videos_dir, num_actions=100):
    """Get first N action categories sorted alphabetically."""
    videos_path = Path(videos_dir)

    actions = set()
    for video_file in videos_path.glob("*.avi"):
        action = video_file.stem.split('_')[1]
        actions.add(action)

    sorted_actions = sorted(list(actions))[:num_actions]
    print(f"Selected {len(sorted_actions)} action categories")

    return sorted_actions


def get_videos_by_action(videos_dir, selected_actions):
    """Group videos by action category."""
    videos_path = Path(videos_dir)
    videos_by_action = defaultdict(list)

    for video_file in videos_path.glob("*.avi"):
        action = video_file.stem.split('_')[1]
        if action in selected_actions:
            videos_by_action[action].append(video_file)

    for action in videos_by_action:
        videos_by_action[action].sort()

    return videos_by_action


def get_frame_count(video_path):
    """Get total frame count by reading all frames sequentially."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return 0

    frame_count = 0
    while True:
        ret, _ = cap.read()
        if not ret:
            break
        frame_count += 1

    cap.release()
    return frame_count


def generate_derangement(n):
    """
    Generate a derangement: a permutation where no element appears in its original position.
    Uses rejection sampling for simplicity.
    """
    if n <= 1:
        return list(range(n))

    max_attempts = 10000
    for _ in range(max_attempts):
        perm = list(range(n))
        random.shuffle(perm)

        if all(perm[i] != i for i in range(n)):
            return perm

    raise ValueError(f"Failed to generate derangement for n={n} after {max_attempts} attempts")


def create_shuffled_csvs(videos_dir, selected_actions, output_csv_dir):
    """Create CSV files with shuffled frame orders for each action."""
    videos_by_action = get_videos_by_action(videos_dir, selected_actions)

    csv_output_path = Path(output_csv_dir)
    csv_output_path.mkdir(exist_ok=True, parents=True)

    print(f"\nGenerating shuffled frame orders...")
    print(f"Output: {csv_output_path}")
    print("=" * 80)

    total_videos = 0

    for action in tqdm(sorted(videos_by_action.keys()), desc="Processing actions", unit="action"):
        videos = videos_by_action[action]
        csv_file = csv_output_path / f"{action}.csv"

        with open(csv_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['video_id', 'shuffled_frames_list'])

            for video_path in videos:
                frame_count = get_frame_count(video_path)

                if frame_count == 0:
                    continue

                shuffled_order = generate_derangement(frame_count)
                shuffled_str = ','.join(map(str, shuffled_order))

                writer.writerow([video_path.name, shuffled_str])
                total_videos += 1

        tqdm.write(f"  {action}: {len(videos)} videos")

    print("\n" + "=" * 80)
    print(f"CSV generation complete!")
    print(f"Total videos: {total_videos}")
    print(f"CSVs saved to: {csv_output_path.absolute()}")
    print("=" * 80)


def create_shuffled_videos(videos_dir, csv_dir, output_videos_dir):
    """Create shuffled videos based on CSV specifications."""
    videos_path = Path(videos_dir)
    csv_path = Path(csv_dir)
    output_path = Path(output_videos_dir)
    output_path.mkdir(exist_ok=True, parents=True)

    csv_files = sorted(list(csv_path.glob("*.csv")))

    print(f"\nGenerating shuffled videos...")
    print(f"Output: {output_path}")
    print("=" * 80)

    total_videos = 0
    skipped_videos = 0

    for csv_file in tqdm(csv_files, desc="Processing actions", unit="action"):
        action_name = csv_file.stem

        with open(csv_file, 'r') as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        for row in rows:
            video_name = row['video_id']
            shuffled_order = list(map(int, row['shuffled_frames_list'].split(',')))

            video_path = videos_path / video_name
            output_video_path = output_path / video_name

            if not video_path.exists():
                skipped_videos += 1
                continue

            cap = cv2.VideoCapture(str(video_path))
            if not cap.isOpened():
                skipped_videos += 1
                continue

            frames = []
            first_frame = None
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                if first_frame is None:
                    first_frame = frame
                frames.append(frame)

            cap.release()

            if len(frames) == 0 or first_frame is None:
                skipped_videos += 1
                continue

            height, width = first_frame.shape[:2]
            fps = 25.0
            fourcc = cv2.VideoWriter_fourcc(*'XVID')

            out = cv2.VideoWriter(str(output_video_path), fourcc, fps, (width, height))

            for frame_idx in shuffled_order:
                if frame_idx < len(frames):
                    out.write(frames[frame_idx])

            out.release()
            total_videos += 1

        tqdm.write(f"  {action_name}: {len(rows)} videos")

    print("\n" + "=" * 80)
    print(f"Video generation complete!")
    print(f"Total videos created: {total_videos}")
    print(f"Skipped videos: {skipped_videos}")
    print(f"Videos saved to: {output_path.absolute()}")
    print("=" * 80)


def shuffle_videos_directly(videos_dir, output_videos_dir, output_csv_dir=None, video_extensions=None):
    """
    Shuffle videos directly without requiring pre-existing CSVs.
    Useful for inference and testing.

    Args:
        videos_dir: Directory containing videos to shuffle
        output_videos_dir: Directory to save shuffled videos
        output_csv_dir: Optional directory to save CSV with shuffling info
        video_extensions: List of video extensions to process
    """
    if video_extensions is None:
        video_extensions = ['.avi', '.mp4', '.mov', '.mkv']

    videos_path = Path(videos_dir)
    output_path = Path(output_videos_dir)
    output_path.mkdir(exist_ok=True, parents=True)

    # Find all videos
    video_files = []
    for ext in video_extensions:
        video_files.extend(videos_path.glob(f"*{ext}"))

    video_files = sorted(video_files)

    if not video_files:
        print(f"No videos found in {videos_dir}")
        return

    print(f"\nDirect shuffle mode: Processing {len(video_files)} videos...")
    print(f"Input: {videos_path}")
    print(f"Output: {output_path}")
    if output_csv_dir:
        print(f"CSV output: {output_csv_dir}")
    print("=" * 80)

    # Prepare CSV file if requested
    csv_writer = None
    csv_file_handle = None
    if output_csv_dir:
        csv_out_path = Path(output_csv_dir)
        csv_out_path.mkdir(exist_ok=True, parents=True)
        csv_file = csv_out_path / "shuffled_order.csv"
        csv_file_handle = open(csv_file, 'w', newline='')
        csv_writer = csv.writer(csv_file_handle)
        csv_writer.writerow(['video_id', 'shuffled_frames_list'])

    total_videos = 0
    skipped_videos = 0

    for video_path in tqdm(video_files, desc="Shuffling videos", unit="video"):
        try:
            # Read video
            cap = cv2.VideoCapture(str(video_path))
            if not cap.isOpened():
                skipped_videos += 1
                continue

            # Load all frames
            frames = []
            first_frame = None
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                if first_frame is None:
                    first_frame = frame
                frames.append(frame)

            cap.release()

            if len(frames) == 0:
                skipped_videos += 1
                continue

            # Generate shuffled order (derangement)
            shuffled_order = generate_derangement(len(frames))

            # Write CSV if requested
            if csv_writer:
                shuffled_str = ','.join(map(str, shuffled_order))
                csv_writer.writerow([video_path.name, shuffled_str])

            # Create shuffled video
            height, width = first_frame.shape[:2]
            fps = cap.get(cv2.CAP_PROP_FPS) or 25.0  # Default to 25 if can't read
            fourcc = cv2.VideoWriter_fourcc(*'XVID')

            output_video_path = output_path / video_path.name
            out = cv2.VideoWriter(str(output_video_path), fourcc, fps, (width, height))

            # Write shuffled frames
            for frame_idx in shuffled_order:
                if frame_idx < len(frames):
                    out.write(frames[frame_idx])

            out.release()
            total_videos += 1

        except Exception as e:
            tqdm.write(f"  Error processing {video_path.name}: {e}")
            skipped_videos += 1
            continue

    # Close CSV file if opened
    if csv_file_handle:
        csv_file_handle.close()
        print(f"\nCSV saved: {csv_file}")

    print("\n" + "=" * 80)
    print(f"Direct shuffle complete!")
    print(f"Total videos shuffled: {total_videos}")
    print(f"Skipped videos: {skipped_videos}")
    print(f"Videos saved to: {output_path.absolute()}")
    print("=" * 80)


def verify_shuffling(csv_dir):
    """Verify that no frame is in its original position."""
    csv_path = Path(csv_dir)
    csv_files = list(csv_path.glob("*.csv"))

    print(f"\nVerifying shuffling constraint...")
    print("=" * 80)

    total_videos = 0
    total_frames = 0
    violations = 0

    for csv_file in csv_files:
        with open(csv_file, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                total_videos += 1
                shuffled_order = list(map(int, row['shuffled_frames_list'].split(',')))

                for original_pos, shuffled_pos in enumerate(shuffled_order):
                    total_frames += 1
                    if original_pos == shuffled_pos:
                        violations += 1
                        print(f"WARNING: Frame {original_pos} stayed in place in {row['video_id']}")

    print(f"\nVerification Results:")
    print(f"  Total videos: {total_videos}")
    print(f"  Total frames: {total_frames:,}")
    print(f"  Violations (frames in original position): {violations}")

    if violations == 0:
        print(f"  ✓ SUCCESS: All frames have been moved from their original positions!")
    else:
        print(f"  ✗ FAILED: {violations} frames remained in their original positions")

    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(
        description="Generate shuffled frame dataset for temporal order detection"
    )

    parser.add_argument("--videos-dir", default="./UCF101_videos",
                       help="Directory containing original UCF101 videos")
    parser.add_argument("--output-csv-dir", default=None,
                       help="Directory to save CSV files with shuffled orders (optional for --direct-shuffle)")
    parser.add_argument("--output-videos-dir", default="./shuffled_videos",
                       help="Directory to save shuffled videos")
    parser.add_argument("--num-actions", type=int, default=40,
                       help="Number of action categories to use (default: 40)")
    parser.add_argument("--csv-only", action="store_true",
                       help="Only generate CSV files, skip video creation")
    parser.add_argument("--videos-only", action="store_true",
                       help="Only generate videos from existing CSVs")
    parser.add_argument("--verify-only", action="store_true",
                       help="Only verify existing CSV files")
    parser.add_argument("--direct-shuffle", action="store_true",
                       help="Directly shuffle videos without CSVs (for inference/testing)")
    parser.add_argument("--video-extensions", nargs='+', default=['.avi', '.mp4', '.mov', '.mkv'],
                       help="Video file extensions to process (default: .avi .mp4 .mov .mkv)")

    args = parser.parse_args()

    # Validation: --output-csv-dir is required for non-direct-shuffle modes
    if not args.direct_shuffle and not args.output_csv_dir:
        print("Error: --output-csv-dir is required when not using --direct-shuffle mode")
        return

    print("=" * 80)
    print("SHUFFLED FRAME DATASET GENERATION")
    print("=" * 80)

    if args.verify_only:
        verify_shuffling(args.output_csv_dir)
        return

    if args.direct_shuffle:
        # Direct shuffle mode: process videos without requiring CSVs
        if not args.videos_dir:
            print("Error: --videos-dir is required for --direct-shuffle mode")
            return

        # Optional: also generate CSV with shuffling information
        csv_dir = args.output_csv_dir if hasattr(args, 'output_csv_dir') else None

        shuffle_videos_directly(
            videos_dir=args.videos_dir,
            output_videos_dir=args.output_videos_dir,
            output_csv_dir=csv_dir,
            video_extensions=args.video_extensions
        )
        return

    if not args.videos_only:
        selected_actions = get_action_categories(args.videos_dir, args.num_actions)
        create_shuffled_csvs(args.videos_dir, selected_actions, args.output_csv_dir)
        verify_shuffling(args.output_csv_dir)

    if not args.csv_only:
        create_shuffled_videos(args.videos_dir, args.output_csv_dir, args.output_videos_dir)


if __name__ == "__main__":
    main()
