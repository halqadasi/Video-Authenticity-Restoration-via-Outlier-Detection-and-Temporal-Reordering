#!/usr/bin/env python3
"""
Extract videos that have outliers at the first or last frame.
Copies both outlier and cleaned versions to a new folder for comparison.

Usage:
    python extract_first_last_outlier_videos.py
"""

import os
import csv
import shutil
from pathlib import Path
from collections import defaultdict

def find_videos_with_first_last_outliers(csv_dir):
    """
    Find all videos that have outliers at the first or last frame.

    Returns:
        dict: {video_id: {'has_first': bool, 'has_last': bool, 'action': str}}
    """
    csv_files = sorted(Path(csv_dir).glob('*.csv'))
    videos_with_first_last = {}

    print("Scanning CSV files for videos with first/last frame outliers...")
    print("=" * 80)
    print(f"Found {len(csv_files)} CSV files to process")

    total_videos_scanned = 0

    for csv_file in csv_files:
        action_name = csv_file.stem

        with open(csv_file, 'r') as f:
            reader = csv.DictReader(f)

            # Group by video
            video_frames = defaultdict(list)
            for row in reader:
                video_id = row['video_id']
                frame_id = int(row['frame_id'])
                is_outlier = int(row['is_outlier'])

                video_frames[video_id].append({
                    'frame_id': frame_id,
                    'is_outlier': is_outlier
                })

            # Check each video for first/last frame outliers
            for video_id, frames in video_frames.items():
                total_videos_scanned += 1

                # No need to sort - we just check frame_id == 0 and max

                # Get first and last frame IDs
                frame_ids = [f['frame_id'] for f in frames]
                min_frame = min(frame_ids)
                max_frame = max(frame_ids)

                # Check if first frame (frame_id = 0 or min) is an outlier
                first_frame_outlier = any(
                    f['frame_id'] == 0 and f['is_outlier'] == 1
                    for f in frames
                )

                # Check if last frame (max frame_id) is an outlier
                last_frame_outlier = any(
                    f['frame_id'] == max_frame and f['is_outlier'] == 1
                    for f in frames
                )

                if first_frame_outlier or last_frame_outlier:
                    videos_with_first_last[video_id] = {
                        'has_first': first_frame_outlier,
                        'has_last': last_frame_outlier,
                        'action': action_name,
                        'max_frame': max_frame
                    }

                    position_str = []
                    if first_frame_outlier:
                        position_str.append("FIRST")
                    if last_frame_outlier:
                        position_str.append("LAST")

                    print(f"  Found: {video_id:50s} [{'+'.join(position_str):12s}] Action: {action_name}")

    print(f"\nTotal videos scanned: {total_videos_scanned}")
    return videos_with_first_last


def copy_video_pairs(videos_info, outlier_videos_dir, cleaned_videos_dir, output_dir):
    """
    Copy both outlier and cleaned versions of videos to output directory.

    Args:
        videos_info: dict with video information
        outlier_videos_dir: directory with outlier videos
        cleaned_videos_dir: directory with cleaned videos
        output_dir: output directory
    """
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True, parents=True)

    print(f"\nCopying video pairs to: {output_path.absolute()}")
    print("=" * 80)

    stats = {
        'first_only': 0,
        'last_only': 0,
        'both': 0,
        'copied': 0,
        'missing_outlier': 0,
        'missing_cleaned': 0
    }

    for video_id, info in sorted(videos_info.items()):
        action = info['action']
        has_first = info['has_first']
        has_last = info['has_last']

        # Determine position label
        if has_first and has_last:
            position = "FIRST_LAST"
            stats['both'] += 1
        elif has_first:
            position = "FIRST"
            stats['first_only'] += 1
        else:
            position = "LAST"
            stats['last_only'] += 1

        # Construct file paths
        outlier_video_path = Path(outlier_videos_dir) / video_id
        cleaned_video_path = Path(cleaned_videos_dir) / video_id

        # Check if files exist
        if not outlier_video_path.exists():
            print(f"  [MISSING OUTLIER] {video_id}")
            stats['missing_outlier'] += 1
            continue

        if not cleaned_video_path.exists():
            print(f"  [MISSING CLEANED] {video_id}")
            stats['missing_cleaned'] += 1
            continue

        # Create output filenames with position prefix
        base_name = video_id.replace('.avi', '')
        outlier_output = output_path / f"{position}_{base_name}_OUTLIER.avi"
        cleaned_output = output_path / f"{position}_{base_name}_CLEANED.avi"

        # Copy files
        shutil.copy2(outlier_video_path, outlier_output)
        shutil.copy2(cleaned_video_path, cleaned_output)

        stats['copied'] += 1
        print(f"  [{stats['copied']:3d}] {position:12s} | {action:25s} | {video_id}")

    return stats


def main():
    # Paths (absolute)
    base_dir = Path(__file__).parent  # Directory where script is located
    csv_dir = base_dir / "outlier_artifacts" / "outlier_CSVs"
    outlier_videos_dir = base_dir / "outlier_artifacts" / "outlier_videos"
    cleaned_videos_dir = base_dir / "outlier_artifacts" / "cleaned_videos"
    output_dir = base_dir / "outlier_artifacts" / "first_last_outlier_videos"

    print("\n" + "=" * 80)
    print("EXTRACT VIDEOS WITH FIRST/LAST FRAME OUTLIERS")
    print("=" * 80)
    print(f"CSV directory:          {csv_dir}")
    print(f"Outlier videos:         {outlier_videos_dir}")
    print(f"Cleaned videos:         {cleaned_videos_dir}")
    print(f"Output directory:       {output_dir}")
    print("=" * 80 + "\n")

    # Step 1: Find videos with first/last frame outliers
    videos_info = find_videos_with_first_last_outliers(csv_dir)

    print(f"\n{'=' * 80}")
    print(f"Found {len(videos_info)} videos with first/last frame outliers:")
    first_count = sum(1 for v in videos_info.values() if v['has_first'] and not v['has_last'])
    last_count = sum(1 for v in videos_info.values() if v['has_last'] and not v['has_first'])
    both_count = sum(1 for v in videos_info.values() if v['has_first'] and v['has_last'])

    print(f"  - First frame only:  {first_count} videos")
    print(f"  - Last frame only:   {last_count} videos")
    print(f"  - Both first & last: {both_count} videos")

    if len(videos_info) == 0:
        print("\nNo videos found with first/last frame outliers. Exiting.")
        return

    # Step 2: Copy video pairs
    stats = copy_video_pairs(videos_info, outlier_videos_dir, cleaned_videos_dir, output_dir)

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Videos with first frame only:  {stats['first_only']}")
    print(f"Videos with last frame only:   {stats['last_only']}")
    print(f"Videos with both:              {stats['both']}")
    print(f"Total videos found:            {len(videos_info)}")
    print(f"\nSuccessfully copied pairs:     {stats['copied']}")
    print(f"Missing outlier videos:        {stats['missing_outlier']}")
    print(f"Missing cleaned videos:        {stats['missing_cleaned']}")
    print(f"\nTotal files in output folder:  {stats['copied'] * 2}")
    print("=" * 80)
    print(f"\nOutput saved to: {Path(output_dir).absolute()}")
    print("\nFile naming convention:")
    print("  - FIRST_<video_name>_OUTLIER.avi   (outlier version)")
    print("  - FIRST_<video_name>_CLEANED.avi   (cleaned version)")
    print("  - LAST_<video_name>_OUTLIER.avi    (outlier version)")
    print("  - LAST_<video_name>_CLEANED.avi    (cleaned version)")
    print("  - FIRST_LAST_<video_name>_OUTLIER.avi / CLEANED.avi (both)")
    print("=" * 80)


if __name__ == "__main__":
    main()
