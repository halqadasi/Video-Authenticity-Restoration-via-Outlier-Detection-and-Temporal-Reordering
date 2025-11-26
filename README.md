# Video Frame Outlier Detection & Temporal Reordering

A deep learning-based system for detecting and removing outlier frames from videos, and reordering shuffled video frames back to their correct temporal sequence.

## Overview

This project addresses two fundamental video processing challenges:

1. **Outlier Frame Detection**: Identifies and removes frames that don't belong to the video's main action (e.g., frames from different actions inserted into a video)
2. **Temporal Frame Reordering**: Reconstructs the correct temporal order of shuffled video frames

## Key Techniques

### Outlier Detection
- **Feature Extraction**: Uses foundation models (CLIP ViT-B/32 or DINOv2) to extract frame embeddings
  - CLIP: 512-dimensional embeddings
  - DINOv2: 768-dimensional embeddings
- **Clustering**: DBSCAN (Density-Based Spatial Clustering) to identify outlier frames
  - GPU-accelerated using cuML/RAPIDS when available
  - Automatic fallback to scikit-learn CPU implementation
- **Cross-Action Strategy**: Outliers are frames from different action categories inserted into videos

### Frame Reordering
- **MSE Matrix**: Computes pairwise Mean Squared Error between all frame pairs
- **Graph Construction**: Builds a graph where edges represent frame similarity
- **Path Finding**: Uses greedy algorithm to find optimal frame sequence
  - Starts from MST (Minimum Spanning Tree) diameter endpoints
  - Constructs path by selecting nearest unvisited neighbors

## Project Structure

```
test/
├── main.py                              # Main inference script (unified interface)
├── outliers_removal_algorithm.py       # Outlier detection using DBSCAN
├── reorder_frames_algorithm.py         # Frame reordering using MSE
├── generate_cleaned_videos_from_predictions.py
├── generate_ordered_videos_from_predictions.py
│
├── dataset_preprocessing/               # Dataset generation scripts
│   ├── generate_outliers_videos.py     # Generate outlier dataset
│   ├── extract_outliers_embeddings.py  # Extract embeddings for outlier detection
│   ├── generate_shuffled_videos.py     # Generate shuffled frame dataset
│   └── analyze_videos.py               # Analyze video statistics
│
├── outlier_artifacts/                   # Outlier detection artifacts
│   ├── outlier_CSVs/                   # CSV annotations
│   ├── outlier_videos/                 # Videos with inserted outliers
│   ├── outlier_embeddings/             # Pre-extracted embeddings
│   ├── cleaned_CSVs/                   # CSV with predictions
│   └── cleaned_videos/                 # Videos after outlier removal
│
├── shuffled_artifacts/                  # Frame reordering artifacts
│   ├── shuffled_CSVs/                  # CSV with shuffled orders
│   ├── shuffled_videos/                # Videos with shuffled frames
│   ├── ordered_CSVs/                   # CSV with predicted orders
│   └── ordered_videos/                 # Videos after frame reordering
│
├── UCF101_videos/                       # Original UCF101 subset (40 out of 101 actions)
│
└── inference/                           # Place your test videos here
```

## Installation

### Requirements
- Python 3.8+
- PyTorch (with CUDA support recommended)
- OpenCV
- CLIP
- Transformers (for DINOv2)


### Install Dependencies

```bash
# Install core dependencies
pip install opencv-python
pip install ftfy regex tqdm
pip install git+https://github.com/openai/CLIP.git
pip install transformers
```

## Quick Start - Inference


### Single Video Processing

Place your video in the `inference/` folder and run:

```bash
# Detect and remove outliers
python main.py --video ./inference/your_video.avi --task outliers

# Reorder shuffled frames
python main.py --video ./inference/shuffled_video.avi --task reorder

# Both tasks (outlier removal → reordering)
python main.py --video ./inference/your_video.avi --task both
```

Output will be saved in `inference/` with `_fixed` suffix by default.

### Batch Processing

Process all videos in a directory:

```bash
# Process all videos in inference folder
python main.py --input-dir ./inference --task both --output-dir ./output
```

### Custom Parameters

```bash
# Outlier detection with custom DBSCAN parameters
python main.py --video ./inference/video.avi --task outliers --eps 0.5 --min-samples 40

# Use DINOv2 instead of CLIP
python main.py --video ./inference/video.avi --task outliers --model-type dinov2 --eps 0.75 --min-samples 30

# Specify output location
python main.py --video ./inference/video.avi --task both --output-dir ./results
```

## Dataset Generation (Evaluation Data)

### Option 1: Download Pre-Generated Datasets (Recommended)

**Save time by downloading pre-generated datasets and CLIP embeddings from DropBox:**

**DropBox Links:**
- **UCF101_videos**: [Download UCF101 Dataset (40 actions)](https://www.dropbox.com/scl/fo/tebpz0x7f3osgymc4n97o/ANS_by-ITosBfD6sWIjWw10?rlkey=0wmfn9xvy4gk2i2fr4bjgf4eh&st=ig5izaxd&dl=0)
- **outlier_artifacts**: [Download Outlier Artifacts (videos, CSVs, CLIP embeddings)](https://www.dropbox.com/scl/fo/hw1w8teuq317hp5o5o8wp/ACaz-rY868mRntwQQxYRd2s?rlkey=rzv7jwo17ru4pkmflbpkp5kia&st=r0mv3oht&dl=0)
- **shuffled_artifacts**: [Download Shuffled Artifacts (videos, CSVs)](https://www.dropbox.com/scl/fo/vcy4o5uwhzhysoyc48pvp/APnmXuwNorNFF91PXo4rTEc?rlkey=5tgzlaedh6ksg4gobpa4kw0ww&st=g86upyis&dl=0)

**What's included:**
- **UCF101_videos/**: First 40 action categories (1000 videos total) from UCF101 dataset
- **outlier_artifacts/**: Pre-generated outlier videos, CSVs, and CLIP embeddings
- **shuffled_artifacts/**: Pre-generated shuffled videos and CSVs

**Setup Instructions:**
1. Download all folders from the DropBox links above
2. Extract and place `UCF101_videos/` in the project root directory
3. Copy the contents of downloaded artifact folders to the corresponding empty folders in your project:
   - `outlier_artifacts/` → `./outlier_artifacts/`
   - `shuffled_artifacts/` → `./shuffled_artifacts/`

**After downloading, you can skip ahead:**
- For **outlier detection**: Jump to **Step 4: Run Outlier Detection** (embeddings already extracted)
- For **frame reordering**: Jump to **Step 7: Run Frame Reordering** (shuffled videos already generated)

**Note**: We use only 40 action categories (out of 101) for space and time efficiency. This reduces dataset size while maintaining sufficient diversity for evaluation.

---

### Option 2: Generate Dataset from Scratch

If you want to generate everything from scratch, follow these steps:

### Step 1: Prepare UCF101 Dataset

Download and extract UCF101 videos (you can download from the official source or use the 40-action subset from DropBox):

```bash
# Your UCF101 videos should be in:
./UCF101_videos/
```

### Step 2: Generate Outlier Dataset

Create videos with inserted outlier frames from different action categories:

```bash
python ./dataset_preprocessing/generate_outliers_videos.py \
  --videos-dir ./UCF101_videos \
  --csvs-dir ./outlier_artifacts/outlier_CSVs \
  --output-dir ./outlier_artifacts/outlier_videos \
  --num-actions 40
```

**What this does:**
- Processes first 40 action categories (sorted alphabetically)
- Inserts cross-action outlier frames adaptively based on video length:
  - SHORT videos (<100 frames): ~10 outliers
  - MEDIUM videos (100-199 frames): ~25-35 outliers
  - LONG videos (200+ frames): ~39-60 outliers
- Generates CSV annotations with outlier positions
- Creates videos with inserted outliers

### Step 3: Extract Embeddings

Extract frame embeddings using CLIP or DINOv2:

```bash
# Using CLIP (faster, 512-dim)
python ./dataset_preprocessing/extract_outliers_embeddings.py \
  --videos-dir ./outlier_artifacts/outlier_videos \
  --csvs-dir ./outlier_artifacts/outlier_CSVs \
  --output-dir ./outlier_artifacts/embeddings \
  --model-type clip \
  --batch-size 128

# Using DINOv2 (more accurate, 768-dim)
python ./dataset_preprocessing/extract_outliers_embeddings.py \
  --videos-dir ./outlier_artifacts/outlier_videos \
  --csvs-dir ./outlier_artifacts/outlier_CSVs \
  --output-dir ./outlier_artifacts/embeddings \
  --model-type dinov2 \
  --batch-size 64
```

**Note:** This step uses GPU acceleration and can take several hours depending on dataset size.

### Step 4: Run Outlier Detection

Detect outliers using DBSCAN on the extracted embeddings:

```bash
python outliers_removal_algorithm.py \
  --embeddings-dir ./outlier_artifacts/outlier_embeddings \
  --output-dir ./outlier_artifacts/outlier_predictions \
  --eps 0.12 \
  --min-samples 5
```

### Step 5: Generate Cleaned Videos

Remove detected outliers to create cleaned videos:

```bash
python generate_cleaned_videos_from_predictions.py \
  --predictions-dir ./outlier_artifacts/outlier_predictions \
  --videos-dir ./outlier_artifacts/outlier_videos \
  --output-dir ./outlier_artifacts/cleaned_videos
```

### Step 6: Generate Shuffled Frame Dataset

Create videos with shuffled frames (derangement - no frame in original position):

```bash
# Generate both CSVs and shuffled videos
python ./dataset_preprocessing/generate_shuffled_videos.py \
  --videos-dir ./UCF101_videos \
  --output-csv-dir ./shuffled_artifacts/shuffled_CSVs \
  --output-videos-dir ./shuffled_artifacts/shuffled_videos \
  --num-actions 40

# Or use direct shuffle mode (for inference testing)
python ./dataset_preprocessing/generate_shuffled_videos.py \
  --videos-dir ./inference \
  --output-videos-dir ./inference_shuffled \
  --direct-shuffle
```

### Step 7: Run Frame Reordering

Reorder shuffled frames back to correct temporal sequence:

```bash
python reorder_frames_algorithm.py \
  --csv_dir ./shuffled_artifacts/shuffled_CSVs \
  --videos_dir ./shuffled_artifacts/shuffled_videos \
  --out_dir ./shuffled_artifacts/predictions
```

### Step 8: Generate Reordered Videos

Create videos with frames in predicted order:

```bash
python generate_ordered_videos_from_predictions.py \
  --predictions-dir ./shuffled_artifacts/predictions \
  --videos-dir ./shuffled_artifacts/shuffled_videos \
  --output-dir ./shuffled_artifacts/reordered_videos
```
