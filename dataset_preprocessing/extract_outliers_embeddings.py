"""
Universal Embedding Extraction with CLIP and DINOv2 Support

Extracts embeddings for all video frames using GPU acceleration.
Supports multiple foundation models:
- CLIP (ViT-B/32): 512-dim embeddings
- DINOv2 "facebook/dinov2-base": 768-dim embeddings

Saves one .pt file per action category for efficient storage and loading.

Usage:
  python ./dataset_preprocessing/extract_outliers_embeddings.py --videos-dir ./outlier_artifacts/outlier_videos --csvs-dir ./outlier_artifacts/outlier_CSVs --output-dir ./outlier_artifacts/embeddings --model-type clip
  python ./dataset_preprocessing/extract_outliers_embeddings.py --videos-dir ./outlier_artifacts/outlier_videos --csvs-dir ./outlier_artifacts/outlier_CSVs --output-dir ./outlier_artifacts/embeddings --model-type dinov2 --batch-size 64
  python ./dataset_preprocessing/extract_outliers_embeddings.py --videos-dir ./outlier_artifacts/outlier_videos --csvs-dir ./outlier_artifacts/outlier_CSVs --output-dir ./outlier_artifacts/embeddings --model-type clip --device cpu
"""
import torch
import cv2
import csv
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm
import numpy as np
from PIL import Image
import argparse
from torchvision import transforms
import clip
from transformers import AutoModel, AutoImageProcessor

torch.backends.cudnn.benchmark = False

def extract_embeddings_per_action(videos_dir, csvs_dir, output_dir,
                                   model_type='clip', model_path=None,
                                   device='cuda', batch_size=128):
    """
    Extract embeddings for all videos, organized by action category.

    Args:
        videos_dir: Directory containing cross-action videos
        csvs_dir: Directory containing CSV annotations
        output_dir: Directory to save embedding tensors (.pt files)
        model_type: 'clip' or 'dinov2'
        model_path: Path to model (for DINOv2, can be local or HuggingFace)
        device: 'cuda' or 'cpu'
        batch_size: Number of frames to process in one batch
    """
    # Setup
    videos_path = Path(videos_dir)
    csvs_path = Path(csvs_dir)
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    # Check GPU availability
    if device == 'cuda' and not torch.cuda.is_available():
        print("Warning: CUDA not available, falling back to CPU")
        device = 'cpu'

    print(f"Using device: {device}")

    # Load model based on type
    print(f"\nLoading {model_type.upper()} model...")

    if model_type == 'clip':

        model, preprocess = clip.load("ViT-B/32", device=device, jit=False)
        model.eval()
        torch.set_grad_enabled(False)
        embedding_dim = 512
        print(f"CLIP model loaded: ViT-B/32 ({embedding_dim}-dim embeddings)")

        def extract_embedding(image_batch):
            """Extract CLIP embeddings."""
            with torch.no_grad():
                feats = model.encode_image(image_batch)
                # L2 normalize (CLIP standard practice)
                feats = torch.nn.functional.normalize(feats, dim=-1)
            return feats

    elif model_type == 'dinov2':

        if model_path is None:
            model_path = "facebook/dinov2-base"

        # Load model and processor
        model = AutoModel.from_pretrained(model_path).to(device)
        model.eval()
        torch.set_grad_enabled(False)

        embedding_dim = model.config.hidden_size
        print(f"DINOv2 model loaded: {model_path} ({embedding_dim}-dim embeddings)")

        preprocess = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        def extract_embedding(image_batch):
            """Extract DINOv2 embeddings using direct model inference."""
            with torch.no_grad():
                # image_batch is already a tensor [B, C, H, W] on device
                outputs = model(pixel_values=image_batch)

                # Get CLS token embedding (first token)
                feats = outputs.last_hidden_state[:, 0]

                feats = torch.nn.functional.normalize(feats, dim=-1)
            return feats

    else:
        raise ValueError(f"Unknown model type: {model_type}. Choose 'clip' or 'dinov2'")

    # Get all CSV files
    csv_files = sorted(list(csvs_path.glob("*.csv")))
    print(f"\nFound {len(csv_files)} action categories")

    total_videos = 0
    total_frames = 0
    skipped_videos = 0

    # Process each action category
    for csv_file in tqdm(csv_files, desc="Processing actions", unit="action"):
        action_name = csv_file.stem

        # Read CSV file
        with open(csv_file, 'r') as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        # Group rows by video
        videos_data = defaultdict(list)
        for row in rows:
            videos_data[row['video_id']].append(row)

        # Dictionary to store embeddings for this action
        action_embeddings = {}

        # Process each video in this action
        for video_name, frame_rows in videos_data.items():
            video_path = videos_path / video_name

            if not video_path.exists():
                skipped_videos += 1
                continue

            # Open video
            cap = cv2.VideoCapture(str(video_path))
            if not cap.isOpened():
                skipped_videos += 1
                continue

            # Prepare storage for this video
            frame_ids = []
            is_outlier = []
            outlier_sources = []

            # Create lookup dictionary for CSV data
            csv_lookup = {int(row['frame_id']): row for row in frame_rows}
            max_frame_id = max(csv_lookup.keys())

            # Batch processing buffers
            frame_batch_cpu = []
            frame_indices_batch = []
            all_embeddings = {}

            def flush_batch():
                """Process accumulated batch and extract embeddings."""
                if not frame_batch_cpu:
                    return

                # Stack batch and move to GPU
                batch = torch.stack(frame_batch_cpu, dim=0)
                if device == 'cuda':
                    batch = batch.pin_memory().to(device, non_blocking=True)
                else:
                    batch = batch.to(device)

                # Extract features
                feats = extract_embedding(batch)

                # Move back to CPU once
                feats_cpu = feats.cpu()

                # Store embeddings
                for i, fid in enumerate(frame_indices_batch):
                    all_embeddings[fid] = feats_cpu[i]

                # Clear batch buffers
                frame_batch_cpu.clear()
                frame_indices_batch.clear()

            # Read all frames sequentially
            current_frame_id = 0
            while current_frame_id <= max_frame_id:
                ret, frame = cap.read()

                if not ret:
                    break

                # Only process frames that are in the CSV
                if current_frame_id in csv_lookup:
                    row = csv_lookup[current_frame_id]

                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    pil_image = Image.fromarray(frame_rgb)

                    # Preprocess and add to batch
                    frame_tensor = preprocess(pil_image)
                    frame_batch_cpu.append(frame_tensor)
                    frame_indices_batch.append(current_frame_id)

                    # Flush batch when it reaches batch_size
                    if len(frame_batch_cpu) >= batch_size:
                        flush_batch()

                    # Track metadata
                    frame_ids.append(current_frame_id)
                    is_outlier.append(int(row['is_outlier']))
                    outlier_sources.append(row.get('outlier_source', ''))

                current_frame_id += 1

            # Flush remaining frames
            flush_batch()

            cap.release()

            # Convert embeddings dict to list in correct order
            frame_embeddings = [all_embeddings[fid] for fid in frame_ids]

            # Stack embeddings into tensor
            if len(frame_embeddings) > 0:
                action_embeddings[video_name] = {
                    'embeddings': torch.stack(frame_embeddings),
                    'frame_ids': frame_ids,
                    'is_outlier': torch.tensor(is_outlier, dtype=torch.bool),
                    'outlier_sources': outlier_sources,
                    'model_type': model_type,
                    'embedding_dim': embedding_dim
                }

                total_videos += 1
                total_frames += len(frame_ids)

        # Save embeddings for this action category
        if action_embeddings:
            output_file = output_path / f"{action_name}_{model_type}_embeddings.pt"
            torch.save(action_embeddings, output_file)

            # Calculate file size
            file_size_mb = output_file.stat().st_size / (1024 * 1024)
            num_videos = len(action_embeddings)
            num_frames_action = sum(len(v['frame_ids']) for v in action_embeddings.values())

            tqdm.write(f"  {action_name}: {num_videos} videos, {num_frames_action} frames, {file_size_mb:.2f} MB")

    # Summary
    print("\n" + "=" * 80)
    print("EXTRACTION COMPLETE!")
    print("=" * 80)
    print(f"Model: {model_type.upper()}")
    print(f"Embedding dimension: {embedding_dim}")
    print(f"Action categories processed: {len(csv_files)}")
    print(f"Total videos processed: {total_videos}")
    print(f"Total videos skipped: {skipped_videos}")
    print(f"Total frames extracted: {total_frames:,}")
    print("=" * 80)

def main():
    """Main function to extract embeddings."""
    parser = argparse.ArgumentParser(description="Extract embeddings using CLIP or DINOv2")
    parser.add_argument("--videos-dir", type=str, default="./outlier_artifacts/outlier_videos",
                        help="Directory containing videos")
    parser.add_argument("--csvs-dir", type=str, default="./outlier_artifacts/outlier_CSVs",
                        help="Directory containing CSV annotations")
    parser.add_argument("--output-dir", type=str, default="./outlier_artifacts/embeddings",
                        help="Directory to save embeddings")
    parser.add_argument("--model-type", type=str, choices=['clip', 'dinov2'], default='clip',
                        help="Model type: 'clip' or 'dinov2'")
    parser.add_argument("--model-path", type=str, default=None,
                        help="Model path for DINOv2 (HuggingFace or local path)")
    parser.add_argument("--batch-size", type=int, default=128,
                        help="Batch size for processing")
    parser.add_argument("--device", type=str, default='cuda', choices=['cuda', 'cpu'],
                        help="Device to use")

    args = parser.parse_args()

    print("=" * 80)
    print("UNIVERSAL EMBEDDING EXTRACTION")
    print("=" * 80)
    print(f"\nConfiguration:")
    print(f"  Model type: {args.model_type.upper()}")
    if args.model_type == 'dinov2' and args.model_path:
        print(f"  Model path: {args.model_path}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Device: {args.device}")
    print(f"  Videos: {args.videos_dir}")
    print(f"  CSVs: {args.csvs_dir}")
    print(f"  Output: {args.output_dir}")
    print("  Normalization: L2 (applied to all embeddings)")
    print("=" * 80)

    extract_embeddings_per_action(
        videos_dir=args.videos_dir,
        csvs_dir=args.csvs_dir,
        output_dir=args.output_dir,
        model_type=args.model_type,
        model_path=args.model_path,
        device=args.device,
        batch_size=args.batch_size
    )

if __name__ == "__main__":
    main()
