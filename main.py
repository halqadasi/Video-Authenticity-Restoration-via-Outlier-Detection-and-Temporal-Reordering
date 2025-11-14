#!/usr/bin/env python3
"""
Main script for video processing: outlier detection and/or frame reordering.

Place your videos in the './inference' folder and run this script to process them.
Processed videos will be saved with '_fixed' suffix.

This script can perform:
1. Outlier detection only (--task outliers)
2. Frame reordering only (--task reorder)
3. Both operations (--task both): outlier detection first, then reordering

Uses DBSCAN for outlier detection.

Usage:
  # Process all videos in ./inference folder
  python main.py --input-dir ./inference --task both

  # Process a single video from inference folder
  python main.py --video ./inference/my_video.avi --task both

  # Custom output directory (save to outlier_artifacts)
  python main.py --input-dir ./inference --task outliers --output-dir ./outlier_artifacts/cleaned_videos

  # Custom DBSCAN parameters
  python main.py --input-dir ./inference --task both --eps 0.5 --min-samples 40

  # Process videos from UCF101_videos with custom model
  python main.py --input-dir ./UCF101_videos --task outliers --model-type dinov2

Output:
  - Default: Videos saved in same directory as input with '_fixed' suffix
  - With --output-dir: Videos saved in specified directory with '_fixed' suffix
  - Outlier detection: video_fixed.avi (outliers removed)
  - Frame reordering: video_fixed.avi (frames reordered)
  - Both: video_fixed.avi (outliers removed AND frames reordered, no intermediate files)
"""

import os
import argparse
import glob
from pathlib import Path
import cv2
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from outliers_removal_algorithm import dbscan_outliers, USE_GPU
from reorder_frames_algorithm import load_video_gray, compute_blurred_mse_matrix, build_best_path

# Device configuration
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Supported video extensions
VIDEO_EXTS = ('.avi', '.mp4', '.mov', '.mkv')

# ==========================================
# EMBEDDING EXTRACTION (Outlier Detection)
# ==========================================

def load_embedding_model(model_type='clip', model_path=None, device='cuda'):
    """Load CLIP or DINOv2/v3 model for embedding extraction."""
    print(f"Loading {model_type.upper()} model...")

    if model_type == 'clip':
        import clip
        model, preprocess = clip.load("ViT-B/32", device=device, jit=False)
        model.eval()
        torch.set_grad_enabled(False)
        embedding_dim = 512

        def extract_fn(image_batch):
            with torch.no_grad():
                feats = model.encode_image(image_batch)
                feats = torch.nn.functional.normalize(feats, dim=-1)
            return feats

        print(f"CLIP model loaded: ViT-B/32 ({embedding_dim}-dim)")
        return extract_fn, preprocess, embedding_dim

    elif model_type == 'dinov2':
        from transformers import pipeline
        from torchvision import transforms

        if model_path is None:
            model_path = "facebook/dinov2-base"

        feature_extractor = pipeline(
            model=model_path,
            task="image-feature-extraction",
            device=0 if (device == 'cuda' and torch.cuda.is_available()) else -1
        )

        test_img = Image.new('RGB', (224, 224))
        test_emb = feature_extractor(test_img)
        embedding_dim = len(test_emb[0])

        preprocess = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        def extract_fn(image_batch):
            images = []
            for i in range(image_batch.shape[0]):
                img_tensor = image_batch[i]
                img_np = img_tensor.cpu().permute(1, 2, 0).numpy()
                img_np = img_np * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
                img_np = (img_np * 255).clip(0, 255).astype(np.uint8)
                images.append(Image.fromarray(img_np))

            features = feature_extractor(images)
            feats = torch.tensor(features, device=device).squeeze(1)
            feats = torch.nn.functional.normalize(feats, dim=-1)
            return feats

        print(f"DINOv2 model loaded: {model_path} ({embedding_dim}-dim)")
        return extract_fn, preprocess, embedding_dim

    else:
        raise ValueError(f"Unknown model type: {model_type}")


def extract_video_embeddings(video_path, extract_fn, preprocess, device='cuda', batch_size=128):
    """Extract embeddings for all frames in a video."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"Video: {Path(video_path).name}")
    print(f"Properties: {width}x{height}, {fps:.2f} fps, {total_frames} frames")
    print(f"Extracting embeddings with batch_size={batch_size}...")

    frame_batch = []
    all_embeddings = []

    with tqdm(total=total_frames, desc="Extracting", unit="frame") as pbar:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(frame_rgb)
            frame_tensor = preprocess(pil_image)
            frame_batch.append(frame_tensor)

            if len(frame_batch) >= batch_size:
                batch = torch.stack(frame_batch, dim=0)
                if device == 'cuda':
                    batch = batch.pin_memory().to(device, non_blocking=True)
                else:
                    batch = batch.to(device)

                feats = extract_fn(batch)
                all_embeddings.append(feats.cpu())
                frame_batch.clear()
                pbar.update(batch_size)

        if frame_batch:
            batch = torch.stack(frame_batch, dim=0)
            if device == 'cuda':
                batch = batch.pin_memory().to(device, non_blocking=True)
            else:
                batch = batch.to(device)

            feats = extract_fn(batch)
            all_embeddings.append(feats.cpu())
            pbar.update(len(frame_batch))

    cap.release()

    embeddings = torch.cat(all_embeddings, dim=0)
    print(f"Extracted {len(embeddings)} embeddings")

    return embeddings, fps, width, height


# ==========================================
# VIDEO SAVING
# ==========================================

def save_cleaned_video(video_path, predictions, output_path, fps, width, height):
    """Create cleaned video with outliers removed."""
    num_outliers = predictions.sum()
    num_inliers = len(predictions) - num_outliers

    print(f"\nOutlier Detection Results:")
    print(f"  Total frames: {len(predictions)}")
    print(f"  Inliers: {num_inliers} ({100*num_inliers/len(predictions):.1f}%)")
    print(f"  Outliers: {num_outliers} ({100*num_outliers/len(predictions):.1f}%)")

    cap = cv2.VideoCapture(str(video_path))
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

    frame_id = 0
    kept = 0

    print(f"\nGenerating cleaned video: {Path(output_path).name}")
    with tqdm(total=len(predictions), desc="Writing", unit="frame") as pbar:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_id < len(predictions) and not predictions[frame_id]:
                out.write(frame)
                kept += 1

            frame_id += 1
            pbar.update(1)

    cap.release()
    out.release()

    print(f"Cleaned video saved: {output_path}")
    return output_path


def save_reordered_video(video_path, frame_order, output_path):
    """Create reordered video using predicted frame order."""
    # Load all frames
    cap = cv2.VideoCapture(str(video_path))
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)

    fps = cap.get(cv2.CAP_PROP_FPS)
    height, width = frames[0].shape[:2]
    cap.release()

    print(f"\nFrame Reordering Results:")
    print(f"  Total frames: {len(frames)}")
    print(f"  Reconstructed order: {len(frame_order)} frames")

    # Write reordered video
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

    print(f"\nGenerating reordered video: {Path(output_path).name}")
    for idx in tqdm(frame_order, desc="Writing", unit="frame"):
        if 0 <= idx < len(frames):
            out.write(frames[idx])

    out.release()

    print(f"Reordered video saved: {output_path}")
    return output_path


def save_cleaned_and_reordered_video(video_path, outlier_predictions, frame_order, output_path):
    """Create video with outliers removed and frames reordered in one pass."""
    # Load all frames
    cap = cv2.VideoCapture(str(video_path))
    all_frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        all_frames.append(frame)

    fps = cap.get(cv2.CAP_PROP_FPS)
    height, width = all_frames[0].shape[:2]
    cap.release()

    # Filter out outliers
    inlier_frames = [all_frames[i] for i in range(len(all_frames))
                     if i < len(outlier_predictions) and not outlier_predictions[i]]

    num_outliers = outlier_predictions.sum()
    print(f"\nCombined Processing Results:")
    print(f"  Original frames: {len(all_frames)}")
    print(f"  Outliers removed: {num_outliers} ({100*num_outliers/len(all_frames):.1f}%)")
    print(f"  Inlier frames: {len(inlier_frames)} ({100*len(inlier_frames)/len(all_frames):.1f}%)")
    print(f"  Reordered frames: {len(frame_order)}")

    # Write reordered video with only inlier frames
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

    print(f"\nGenerating final video: {Path(output_path).name}")
    for idx in tqdm(frame_order, desc="Writing", unit="frame"):
        if 0 <= idx < len(inlier_frames):
            out.write(inlier_frames[idx])

    out.release()

    print(f"Final video saved: {output_path}")
    return output_path


# ==========================================
# MAIN PIPELINE
# ==========================================

def run_outlier_detection(video_path, output_path, args):
    """Run outlier detection pipeline using imported functions."""
    print("OUTLIER DETECTION")
    print(f"GPU Acceleration: {'Enabled (cuML)' if USE_GPU else 'Disabled (CPU/sklearn)'}")

    # Load embedding model
    extract_fn, preprocess, embedding_dim = load_embedding_model(
        model_type=args.model_type,
        model_path=args.model_path,
        device=DEVICE
    )

    # Extract embeddings
    embeddings, fps, width, height = extract_video_embeddings(
        video_path, extract_fn, preprocess, DEVICE, args.batch_size
    )

    # Detect outliers using DBSCAN
    print(f"\nRunning DBSCAN outlier detection...")
    predictions = dbscan_outliers(
        embeddings,
        eps=args.eps,
        min_samples=args.min_samples
    )

    # Save cleaned video
    cleaned_path = save_cleaned_video(video_path, predictions, output_path, fps, width, height)
    return cleaned_path


def run_frame_reordering(video_path, output_path):
    """Run frame reordering pipeline."""
    print("\n" + "=" * 80)
    print("FRAME REORDERING")
    print("=" * 80)

    print(f"Loading video: {Path(video_path).name}")
    frames = load_video_gray(str(video_path))
    print(f"Loaded {len(frames)} frames")

    print("Computing MSE matrix...")
    mse = compute_blurred_mse_matrix(frames)

    print("Building temporal path...")
    path = build_best_path(mse)

    # Save reordered video
    reordered_path = save_reordered_video(video_path, path, output_path)
    return reordered_path


def run_both_tasks(video_path, output_path, args):
    """Run both outlier detection and frame reordering without saving intermediate video."""
    print("\n" + "=" * 80)
    print("STEP 1: OUTLIER DETECTION")
    print("=" * 80)
    print(f"GPU Acceleration: {'Enabled (cuML)' if USE_GPU else 'Disabled (CPU/sklearn)'}")

    # Load embedding model and extract embeddings
    extract_fn, preprocess, embedding_dim = load_embedding_model(
        model_type=args.model_type,
        model_path=args.model_path,
        device=DEVICE
    )

    embeddings, fps, width, height = extract_video_embeddings(
        video_path, extract_fn, preprocess, DEVICE, args.batch_size
    )

    # Detect outliers using DBSCAN
    print(f"\nRunning DBSCAN outlier detection...")
    outlier_predictions = dbscan_outliers(
        embeddings,
        eps=args.eps,
        min_samples=args.min_samples
    )

    num_outliers = outlier_predictions.sum()
    num_inliers = len(outlier_predictions) - num_outliers
    print(f"\nOutlier Detection Results:")
    print(f"  Total frames: {len(outlier_predictions)}")
    print(f"  Inliers: {num_inliers} ({100*num_inliers/len(outlier_predictions):.1f}%)")
    print(f"  Outliers: {num_outliers} ({100*num_outliers/len(outlier_predictions):.1f}%)")

    # Step 2: Frame reordering on inlier frames
    print("\n" + "=" * 80)
    print("STEP 2: FRAME REORDERING (on inlier frames)")
    print("=" * 80)

    all_frames = load_video_gray(str(video_path))

    # Filter to only inlier frames
    inlier_frames = []
    for i in range(len(all_frames)):
        if i < len(outlier_predictions) and not outlier_predictions[i]:
            inlier_frames.append(all_frames[i])

    inlier_frames = torch.stack(inlier_frames, dim=0)
    mse = compute_blurred_mse_matrix(inlier_frames)
    path = build_best_path(mse)

    # Save final video (cleaned and reordered)
    final_path = save_cleaned_and_reordered_video(video_path, outlier_predictions, path, output_path)
    return final_path


def get_output_path(input_path, output_dir, suffix="_fixed"):
    """Determine the output path based on input path and output directory."""
    input_path = Path(input_path)

    if output_dir:
        # Use specified output directory
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True, parents=True)
        output_name = f"{input_path.stem}{suffix}{input_path.suffix}"
        return output_dir / output_name
    else:
        # Save in same directory as input
        output_name = f"{input_path.stem}{suffix}{input_path.suffix}"
        return input_path.parent / output_name


def process_single_video(video_path, args):
    """Process a single video file."""
    video_path = Path(video_path)

    if not video_path.exists():
        print(f"Error: Video not found: {video_path}")
        return

    print("=" * 80)
    print(f"Processing: {video_path.name}")
    print("=" * 80)
    print(f"Task: {args.task.upper()}")
    print("=" * 80)

    # Determine output path
    output_path = get_output_path(video_path, args.output_dir)

    # Execute tasks
    if args.task == "outliers":
        run_outlier_detection(str(video_path), str(output_path), args)

    elif args.task == "reorder":
        run_frame_reordering(str(video_path), str(output_path))

    elif args.task == "both":
        # Run both tasks without saving intermediate video
        run_both_tasks(str(video_path), str(output_path), args)

    print("\n" + "=" * 80)
    print("PROCESSING COMPLETE")
    print("=" * 80)
    print(f"Output: {output_path}")


def process_directory(input_dir, args):
    """Process all videos in a directory."""
    input_dir = Path(input_dir)

    if not input_dir.exists():
        print(f"Error: Directory not found: {input_dir}")
        return

    # Find all video files
    video_files = []
    for ext in VIDEO_EXTS:
        video_files.extend(input_dir.glob(f"*{ext}"))

    video_files = sorted(video_files)

    if not video_files:
        print(f"No video files found in {input_dir}")
        print(f"Supported extensions: {VIDEO_EXTS}")
        return

    print("=" * 80)
    print(f"Found {len(video_files)} video(s) in {input_dir}")
    print("=" * 80)

    # Process each video
    for i, video_path in enumerate(video_files, 1):
        print(f"\n[{i}/{len(video_files)}] Processing: {video_path.name}")

        # Determine output path
        output_path = get_output_path(video_path, args.output_dir)

        try:
            # Execute tasks
            if args.task == "outliers":
                run_outlier_detection(str(video_path), str(output_path), args)

            elif args.task == "reorder":
                run_frame_reordering(str(video_path), str(output_path))

            elif args.task == "both":
                # Run both tasks without saving intermediate video
                run_both_tasks(str(video_path), str(output_path), args)

            print(f"  ✓ Saved: {output_path}")

        except Exception as e:
            print(f"  ✗ Error processing {video_path.name}: {e}")
            continue

    print("\n" + "=" * 80)
    print("BATCH PROCESSING COMPLETE")
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(
        description="Main script for video processing: outlier detection (DBSCAN) and/or frame reordering"
    )

    # Input arguments (mutually exclusive)
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--video",
                           help="Process a single video file")
    input_group.add_argument("--input-dir",
                           help="Process all videos in a directory (default: ./inference)")

    # Task selection
    parser.add_argument("--task", required=True, choices=["outliers", "reorder", "both"],
                       help="Task to perform: outliers, reorder, or both")

    # Output directory (optional)
    parser.add_argument("--output-dir",
                       help="Output directory (default: same as input directory)")

    # Outlier detection parameters
    parser.add_argument("--model-type", default="clip", choices=["clip", "dinov2"],
                       help="Embedding model type for outlier detection")
    parser.add_argument("--model-path", help="Path to DINOv2 model (optional)")
    parser.add_argument("--batch-size", type=int, default=128,
                       help="Batch size for embedding extraction")

    # DBSCAN parameters
    parser.add_argument("--eps", type=float, default=0.5,
                       help="DBSCAN: Epsilon parameter")
    parser.add_argument("--min-samples", type=int, default=40,
                       help="DBSCAN: Minimum samples parameter")

    args = parser.parse_args()

    # Default to ./inference if neither --video nor --input-dir specified
    # (This won't happen due to required=True, but keeping for clarity)

    if args.task in ["outliers", "both"]:
        print(f"DBSCAN parameters: eps={args.eps}, min_samples={args.min_samples}")

    # Process based on input mode
    if args.video:
        process_single_video(args.video, args)
    elif args.input_dir:
        process_directory(args.input_dir, args)


if __name__ == "__main__":
    main()
