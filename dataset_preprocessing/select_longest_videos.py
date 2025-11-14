#!/usr/bin/env python3
"""
Select longest video from each group for each action category.

Processes UCF101 dataset and selects the longest video (by file size)
from each group (g01-g25) for each action category.

Usage:
  python select_longest_videos.py

Note: Modify ucf_directory and output_directory in main() if needed.
"""

import os
import re
import shutil
from pathlib import Path
from collections import defaultdict

def extract_group_number(filename):
    """Extract group number from video filename."""
    match = re.search(r'_g(\d+)_c', filename)
    return int(match.group(1)) if match else None

def select_longest_from_groups(ucf_dir, output_dir, max_groups=25):
    """Select longest video from each group for each action."""
    ucf_path = Path(ucf_dir)
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    action_folders = sorted([f for f in ucf_path.iterdir() if f.is_dir()])
    total_videos = 0

    for action_folder in action_folders:
        print(f"Processing: {action_folder.name}")

        videos_by_group = defaultdict(list)

        for video_file in action_folder.glob("*.avi"):
            group_num = extract_group_number(video_file.name)
            if group_num and group_num <= max_groups:
                videos_by_group[group_num].append(video_file)

        selected_count = 0
        for group_num in range(1, max_groups + 1):
            if group_num not in videos_by_group:
                continue

            group_videos = videos_by_group[group_num]
            if not group_videos:
                continue

            longest_video = max(group_videos, key=lambda v: v.stat().st_size)

            dest = output_path / longest_video.name
            shutil.copy2(longest_video, dest)
            total_videos += 1
            selected_count += 1

        print(f"  Selected {selected_count} videos")

    print(f"\nTotal videos copied: {total_videos}")
    print(f"Output: {output_path.absolute()}")

if __name__ == "__main__":
    ucf_directory = r"C:\Users\hamza\Desktop\test\UCF-101"
    output_directory = r"C:\Users\hamza\Desktop\test\UCF101_videos"

    select_longest_from_groups(ucf_directory, output_directory, max_groups=25)
