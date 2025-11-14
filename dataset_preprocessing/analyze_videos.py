#!/usr/bin/env python3
"""
Analyze videos and print statistics.

Analyzes all videos in a directory and prints statistics including
frame counts, resolutions, and FPS.

Usage:
  # Analyze videos in UCF101_videos directory
  python analyze_videos.py --videos-dir ./UCF101_videos

  # Analyze videos in inference directory
  python analyze_videos.py --videos-dir ./inference

  # Analyze outlier videos
  python analyze_videos.py --videos-dir ./outlier_artifacts/outlier_videos
"""

import cv2
import argparse
from pathlib import Path

def analyze_videos(videos_dir):
    """
    Analyze videos in the given directory and print statistics.
    
    Args:
        videos_dir (str): Directory containing UCF101 videos.
    """
    videos_path = Path(videos_dir)
    video_files = list(videos_path.glob("*.avi"))

    print(f"Analyzing {len(video_files)} videos...")

    min_frames = float('inf')
    max_frames = 0
    shortest_video = None
    longest_video = None
    total_frames = 0
    failed = 0

    for i, video_file in enumerate(video_files):
        try:
            cap = cv2.VideoCapture(str(video_file))
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            cap.release()

            if frame_count < min_frames:
                min_frames = frame_count
                shortest_video = video_file.name
                shortest_fps = fps

            if frame_count > max_frames:
                max_frames = frame_count
                longest_video = video_file.name
                longest_fps = fps

            total_frames += frame_count

            if (i + 1) % 100 == 0:
                print(f"  Processed {i + 1}/{len(video_files)}...")
        except:
            failed += 1


    print(f"\n{'='*60}")
    print("VIDEO ANALYSIS RESULTS")
    print(f"{'='*60}")
    print(f"Total videos analyzed: {len(video_files) - failed}")
    print(f"Total failed to analyze: {failed}")
    print(f"Total frames across all videos: {total_frames}")

    print(f"\nSHORTEST VIDEO:")
    print(f"  Name: {shortest_video}")
    print(f"  Frames: {min_frames}")
    print(f"  FPS: {shortest_fps}")

    print(f"\nLONGEST VIDEO:")
    print(f"  Name: {longest_video}")
    print(f"  Frames: {max_frames}")
    print(f"  FPS: {longest_fps}")




if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Analyze videos and print statistics"
    )

    parser.add_argument("--videos-dir", required=True,
                       help="Directory containing videos to analyze")

    args = parser.parse_args()

    analyze_videos(args.videos_dir)
