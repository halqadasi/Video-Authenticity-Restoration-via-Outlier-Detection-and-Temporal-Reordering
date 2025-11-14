#!/usr/bin/env python3
"""
Download and extract UCF101 dataset.

Downloads the UCF101.rar archive (~6.5GB) from the official source,
then extracts the first N videos for processing.

Usage:
  python ucf101_download.py

Note: Modify num_videos and output_dir in main() if needed.
Requires 7-Zip installed at C:\\Program Files\\7-Zip\\7z.exe
"""

import os
import requests
import warnings
import subprocess
import shutil
from tqdm import tqdm
from pathlib import Path
import urllib3

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
warnings.filterwarnings('ignore')

def download_file(url, output_path):
    """Download file from URL with progress bar."""
    print(f"Downloading from {url}...")

    try:
        response = requests.get(url, stream=True, verify=False, timeout=30)
        response.raise_for_status()
        total_size = int(response.headers.get('content-length', 0))

        with open(output_path, 'wb') as f, tqdm(
            desc=os.path.basename(output_path),
            total=total_size,
            unit='iB',
            unit_scale=True,
            unit_divisor=1024
        ) as bar:
            for data in response.iter_content(chunk_size=1024*1024):
                f.write(data)
                bar.update(len(data))
        return True
    except Exception as e:
        print(f"Error: {e}")
        return False

def extract_first_n_videos(rar_path, output_dir, n=100):
    """Extract first N videos from RAR archive."""
    print(f"\nExtracting first {n} videos from {rar_path}...")

    extract_dir = os.path.join(output_dir, "extracted")
    os.makedirs(extract_dir, exist_ok=True)

    try:
        seven_zip = r"C:\Program Files\7-Zip\7z.exe"

        print("Extracting RAR archive...")
        subprocess.run([seven_zip, 'x', rar_path, f'-o{extract_dir}', '-y'],
                      check=True, capture_output=True)

        videos_dir = Path(extract_dir)
        video_files = list(videos_dir.rglob("*.avi"))

        print(f"Found {len(video_files)} videos")

        final_dir = os.path.join(output_dir, "videos")
        os.makedirs(final_dir, exist_ok=True)

        for i, video in enumerate(video_files[:n]):
            dest = os.path.join(final_dir, f"video_{i+1:03d}.avi")
            shutil.copy2(video, dest)
            if (i + 1) % 10 == 0:
                print(f"Copied {i+1}/{n} videos...")

        print(f"\n Extracted {min(n, len(video_files))} videos")

        shutil.rmtree(extract_dir)
        return True

    except Exception as e:
        print(f"Error during extraction: {e}")
        return False

def download_ucf101(num_videos=100, output_dir="ucf101_videos"):

    os.makedirs(output_dir, exist_ok=True)

    url = "https://www.crcv.ucf.edu/data/UCF101/UCF101.rar"
    rar_path = os.path.join(output_dir, "UCF101.rar")

    if os.path.exists(rar_path):
        file_size = os.path.getsize(rar_path)
        print(f"Archive exists ({file_size / (1024**3):.2f} GB)")

        if file_size < 6_000_000_000:
            print("Incomplete download. Removing...")
            os.remove(rar_path)

    if not os.path.exists(rar_path):
        print(f"\nDataset size: ~6.5GB (RAR format)")
        print(f"Destination: {rar_path}\n")

        if not download_file(url, rar_path):
            print("\n Download failed")
            print("\n Alternatives:")
            print("1. Browser: https://www.crcv.ucf.edu/data/UCF101/UCF101.rar")
            print("2. Kaggle: https://www.kaggle.com/datasets/pevogam/ucf101")
            return

        print(f"\n Download complete ({os.path.getsize(rar_path) / (1024**3):.2f} GB)")

    extract_first_n_videos(rar_path, output_dir, num_videos)

if __name__ == "__main__":
    download_ucf101(num_videos=100, output_dir="ucf101_videos")
