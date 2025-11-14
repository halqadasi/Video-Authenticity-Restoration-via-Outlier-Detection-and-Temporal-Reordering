#!/usr/bin/env python3
"""
Regenerate outlier dataset with ADAPTIVE CROSS-ACTION SEGMENT strategy using INSERTION.

INSERTION STRATEGY: Outlier frames are INSERTED (not replaced) so that:
- Original frames are preserved
- After cleaning outliers, you get back the EXACT original video
- Video length increases by the number of outliers

NEW STRATEGY: Outliers adapt to video length to maintain reasonable outlier ratio:

SHORT videos (< 100 frames):
  - 1 segment of 10 consecutive frames
  - Total: ~10 outliers (~10%)

MEDIUM videos (100-199 frames):
  - 2 segments of 5-10 consecutive frames
  - 2 segments of 3-5 scattered frames
  - 3-4 single isolated frames
  - Total: ~25-35 outliers (~15-20%)

LONG videos (200+ frames):
  - 2 segments of 10-25 consecutive frames
  - 3 segments of 5-10 scattered frames
  - 4-5 single isolated frames
  - Total: ~39-60 outliers (~15-20%)

Usage:
  # Generate outliers for first 40 action categories
  python generate_outliers_videos.py --videos-dir ./UCF101_videos --csvs-dir ./outlier_artifacts/outlier_CSVs --output-dir ./outlier_artifacts/outlier_videos

  # Process only 10 action categories
  python generate_outliers_videos.py --videos-dir ./UCF101_videos --csvs-dir ./outlier_artifacts/outlier_CSVs --output-dir ./outlier_artifacts/outlier_videos --num-actions 10

  # Use different seed for randomization
  python generate_outliers_videos.py --videos-dir ./UCF101_videos --csvs-dir ./outlier_artifacts/outlier_CSVs --output-dir ./outlier_artifacts/outlier_videos --seed 123

Generates both CSV annotations and videos with inserted outliers.
"""
import cv2
import csv
import random
import argparse
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm

def extract_action_name(video_name):
    """Extract action category from video name."""
    parts = video_name.replace('.avi', '').split('_')
    if len(parts) >= 2:
        return '_'.join(parts[1:-2])
    return None

def get_frame_count(video_path):
    """
    Get accurate frame count by actually reading all frames.
    """
    cap = cv2.VideoCapture(str(video_path))
    count = 0
    while True:
        ret, _ = cap.read()
        if not ret:
            break
        count += 1
    cap.release()
    return count

def get_consecutive_frames(video_path, start_frame, num_frames):
    """
    Extract consecutive frames from a video.

    Args:
        video_path: Path to source video
        start_frame: Starting frame index
        num_frames: Number of consecutive frames to extract

    Returns:
        List of frames (numpy arrays)
    """
    cap = cv2.VideoCapture(str(video_path))
    frames = []

    # Skip to start frame
    for _ in range(start_frame):
        ret, _ = cap.read()
        if not ret:
            break

    # Read consecutive frames
    for _ in range(num_frames):
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)

    cap.release()
    return frames

def create_cross_action_outlier_csvs_insertion(videos_dir, output_dir, num_actions=40):
    """
    Create CSV annotations with cross-action INSERTION outlier strategy.

    Outliers are INSERTED at specific positions, not replacing original frames.
    """
    videos_path = Path(videos_dir)
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    # Get all video files
    video_files = list(videos_path.glob("*.avi"))

    # Group videos by action category
    videos_by_action = defaultdict(list)
    for video in video_files:
        action = extract_action_name(video.name)
        if action:
            videos_by_action[action].append(video)

    all_actions = sorted(videos_by_action.keys())[:num_actions]

    print(f"\nFound {len(videos_by_action)} total action categories")
    print(f"Processing first {len(all_actions)} action categories")
    print(f"Total videos: {sum(len(videos_by_action[a]) for a in all_actions)}\n")

    # Process each action category
    for current_action in tqdm(all_actions, desc="Creating CSVs", unit="action"):
        videos = sorted(videos_by_action[current_action])
        other_actions = [a for a in all_actions if a != current_action]
        csv_path = output_path / f"{current_action}.csv"

        with open(csv_path, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['number', 'video_id', 'frame_id', 'is_outlier',
                           'outlier_source', 'source_frame_id', 'original_frame_id'])

            row_number = 1
            total_frames = 0
            total_outliers = 0

            for current_video in videos:
                # Get accurate frame count
                frame_count = get_frame_count(current_video)

                if frame_count < 20:
                    print(f"  Skipping {current_video.name} - too short ({frame_count} frames)")
                    continue

                # Plan insertions: list of (insert_before_frame, num_outliers, source_info)
                insertions = []

                # ADAPTIVE STRATEGY based on video length
                if frame_count < 100:
                    # SHORT videos: 1 segment of 10 consecutive frames
                    insert_pos = random.randint(0, frame_count)
                    source_action = random.choice(other_actions)
                    source_video = random.choice(videos_by_action[source_action])
                    source_start = random.randint(0, max(0, get_frame_count(videos_path / source_video.name) - 10))
                    insertions.append((insert_pos, 10, source_video.name, source_start))

                elif frame_count < 200:
                    # MEDIUM videos
                    # 2 segments of 5-10 consecutive
                    for _ in range(2):
                        seg_length = random.randint(5, 10)
                        insert_pos = random.randint(0, frame_count)
                        source_action = random.choice(other_actions)
                        source_video = random.choice(videos_by_action[source_action])
                        source_start = random.randint(0, max(0, get_frame_count(videos_path / source_video.name) - seg_length))
                        insertions.append((insert_pos, seg_length, source_video.name, source_start))

                    # 2 segments of 3-5 scattered (as consecutive for simplicity)
                    for _ in range(2):
                        seg_length = random.randint(3, 5)
                        insert_pos = random.randint(0, frame_count)
                        source_action = random.choice(other_actions)
                        source_video = random.choice(videos_by_action[source_action])
                        source_start = random.randint(0, max(0, get_frame_count(videos_path / source_video.name) - seg_length))
                        insertions.append((insert_pos, seg_length, source_video.name, source_start))

                    # 3-4 single frames
                    for _ in range(random.randint(3, 4)):
                        insert_pos = random.randint(0, frame_count)
                        source_action = random.choice(other_actions)
                        source_video = random.choice(videos_by_action[source_action])
                        source_start = random.randint(0, get_frame_count(videos_path / source_video.name) - 1)
                        insertions.append((insert_pos, 1, source_video.name, source_start))

                else:
                    # LONG videos
                    # 2 segments of 10-25 consecutive
                    for _ in range(2):
                        seg_length = random.randint(10, 25)
                        insert_pos = random.randint(0, frame_count)
                        source_action = random.choice(other_actions)
                        source_video = random.choice(videos_by_action[source_action])
                        source_start = random.randint(0, max(0, get_frame_count(videos_path / source_video.name) - seg_length))
                        insertions.append((insert_pos, seg_length, source_video.name, source_start))

                    # 3 segments of 5-10
                    for _ in range(3):
                        seg_length = random.randint(5, 10)
                        insert_pos = random.randint(0, frame_count)
                        source_action = random.choice(other_actions)
                        source_video = random.choice(videos_by_action[source_action])
                        source_start = random.randint(0, max(0, get_frame_count(videos_path / source_video.name) - seg_length))
                        insertions.append((insert_pos, seg_length, source_video.name, source_start))

                    # 4-5 single frames
                    for _ in range(random.randint(4, 5)):
                        insert_pos = random.randint(0, frame_count)
                        source_action = random.choice(other_actions)
                        source_video = random.choice(videos_by_action[source_action])
                        source_start = random.randint(0, get_frame_count(videos_path / source_video.name) - 1)
                        insertions.append((insert_pos, 1, source_video.name, source_start))

                # Sort insertions by position (important for correct insertion)
                insertions.sort(key=lambda x: x[0])

                # Build frame list with insertions
                # Track: (is_outlier, source_video, source_frame_id, original_frame_id)
                frame_list = []
                original_frame_id = 0

                for insert_pos, num_outliers, source_video, source_start in insertions:
                    # Add original frames up to insertion point
                    while original_frame_id < min(insert_pos, frame_count):
                        frame_list.append((False, '', '', original_frame_id))
                        original_frame_id += 1

                    # Add outlier frames
                    for i in range(num_outliers):
                        frame_list.append((True, source_video, source_start + i, ''))
                        total_outliers += 1

                # Add remaining original frames
                while original_frame_id < frame_count:
                    frame_list.append((False, '', '', original_frame_id))
                    original_frame_id += 1

                # Write CSV rows
                for frame_id, (is_outlier, source_video, source_frame_id, orig_frame_id) in enumerate(frame_list):
                    writer.writerow([
                        row_number,
                        current_video.name,
                        frame_id,
                        1 if is_outlier else 0,
                        source_video if is_outlier else '',
                        source_frame_id if is_outlier else '',
                        orig_frame_id if not is_outlier else ''
                    ])
                    row_number += 1
                    total_frames += 1

        outlier_pct = 100 * total_outliers / total_frames if total_frames > 0 else 0
        print(f"  {current_action}: {total_frames} frames, {total_outliers} outliers ({outlier_pct:.1f}%)")

    print(f"\nCSV files saved to: {output_path.absolute()}")

def inject_cross_action_outliers_insertion(videos_dir, csvs_dir, output_dir):
    """
    Create videos with outliers INSERTED at specified positions.
    """
    videos_path = Path(videos_dir)
    csvs_path = Path(csvs_dir)
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    csv_files = sorted(list(csvs_path.glob("*.csv")))
    print(f"\nProcessing {len(csv_files)} action categories...")

    for csv_file in csv_files:
        action_name = csv_file.stem
        print(f"\nProcessing: {action_name}")

        with open(csv_file, 'r') as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        # Group rows by video
        videos_data = defaultdict(list)
        for row in rows:
            videos_data[row['video_id']].append(row)

        for video_name, frame_rows in videos_data.items():
            video_path = videos_path / video_name
            output_video_path = output_path / video_name

            if not video_path.exists():
                print(f"  Warning: {video_name} not found")
                continue

            # Open source video
            cap = cv2.VideoCapture(str(video_path))
            if not cap.isOpened():
                print(f"  Warning: Could not open {video_name}")
                continue

            fps = cap.get(cv2.CAP_PROP_FPS)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fourcc = cv2.VideoWriter_fourcc(*'XVID')

            # Open output video
            out = cv2.VideoWriter(str(output_video_path), fourcc, fps, (width, height))

            # Cache for outlier frames
            outlier_cache = {}

            # Process frames according to CSV
            original_frame_id = 0
            for row in frame_rows:
                is_outlier = int(row['is_outlier'])

                if is_outlier:
                    # Insert outlier frame
                    source_video_name = row['outlier_source']
                    source_frame_id = int(row['source_frame_id'])
                    cache_key = (source_video_name, source_frame_id)

                    if cache_key not in outlier_cache:
                        source_video_path = videos_path / source_video_name
                        source_cap = cv2.VideoCapture(str(source_video_path))

                        # Skip to source frame
                        for _ in range(source_frame_id):
                            source_cap.read()

                        ret, source_frame = source_cap.read()
                        source_cap.release()

                        if ret and source_frame is not None:
                            source_frame = cv2.resize(source_frame, (width, height))
                            outlier_cache[cache_key] = source_frame

                    if cache_key in outlier_cache:
                        out.write(outlier_cache[cache_key])
                else:
                    # Write original frame
                    ret, frame = cap.read()
                    if ret:
                        out.write(frame)
                        original_frame_id += 1

            cap.release()
            out.release()

            outlier_count = sum(1 for row in frame_rows if row['is_outlier'] == '1')
            print(f"  {video_name}: {len(frame_rows)} frames ({outlier_count} outliers inserted)")

    print(f"\nModified videos saved to: {output_path.absolute()}")

def main():
    """Main function to regenerate dataset with INSERTION strategy."""
    parser = argparse.ArgumentParser(
        description="Generate outlier dataset with adaptive cross-action insertion strategy"
    )

    parser.add_argument("--videos-dir", required=True,
                       help="Directory containing source UCF101 videos")
    parser.add_argument("--csvs-dir", required=True,
                       help="Output directory for CSV annotations")
    parser.add_argument("--output-dir", required=True,
                       help="Output directory for videos with inserted outliers")
    parser.add_argument("--num-actions", type=int, default=40,
                       help="Number of action categories to process (default: 40)")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed for reproducibility (default: 42)")

    args = parser.parse_args()

    random.seed(args.seed)

    print("\nSTEP 1: Creating CSV annotations...")
    create_cross_action_outlier_csvs_insertion(args.videos_dir, args.csvs_dir, num_actions=args.num_actions)

    print("\n" + "=" * 80)
    print("STEP 2: Injecting outliers into videos...")
    inject_cross_action_outliers_insertion(args.videos_dir, args.csvs_dir, args.output_dir)

    print("\n" + "=" * 80)
    print("COMPLETE!")
    print(f"Processed {args.num_actions} action categories successfully.")
    print("=" * 80)

if __name__ == "__main__":
    main()
