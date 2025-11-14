#!/usr/bin/env python3
"""
Frame order reconstruction algorithm using MSE and greedy path construction.

Reconstructs temporal frame order from shuffled videos using grayscale MSE matrix,
MST diameter endpoints, and double-ended greedy path building with local refinement.

Usage:
  # Process shuffled videos and CSVs from shuffled_artifacts
  python reorder_frames_algorithm.py --csv_dir ./shuffled_artifacts/shuffled_CSVs --videos_dir ./shuffled_artifacts/shuffled_videos --out_dir ./shuffled_artifacts/ordered_CSVs

Note: To generate reordered videos from predictions, use generate_ordered_videos_from_predictions.py
"""

import argparse
import os
import glob

import cv2
import numpy as np
import pandas as pd
import torch


# =========================
# Config
# =========================

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMG_SIZE = 64
VIDEO_EXTS = (".avi", ".mp4", ".mov", ".mkv")

# =========================
# Pairwise MSE on GPU
# =========================

def compute_mse_matrix(frames: torch.Tensor) -> torch.Tensor:
    """
    frames: [N, 1, H, W] on DEVICE
    Returns:
        mse[i,j]: mean squared error between frame i and j
    """
    N = frames.shape[0]
    flat = frames.view(N, -1)                   # [N, D]

    sq = (flat ** 2).sum(dim=1, keepdim=True)   # [N,1]
    # dist2[i,j] = ||flat[i] - flat[j]||^2
    dist2 = sq + sq.t() - 2.0 * (flat @ flat.t())
    dist2 = torch.clamp(dist2, min=0.0)

    D = flat.shape[1]
    mse = dist2 / D
    mse.fill_diagonal_(0.0)
    return mse

def compute_blurred_mse_matrix(frames: torch.Tensor, ksize: int = 9, sigma: float = 1.0):
    """
    Compute MSE on lightly blurred frames.
    """
    N, _, _, _ = frames.shape

    arr = (frames.clamp(0,1).mul(255).squeeze(1).byte().cpu().numpy())  # [N,H,W] uint8
    blurred = []
    for i in range(N):
        b = cv2.GaussianBlur(arr[i], (ksize, ksize), sigma)
        blurred.append(b)
    b_arr = np.stack(blurred, axis=0).astype(np.float32) / 255.0  # [N,H,W]
    b_t = torch.from_numpy(b_arr).to(frames.device).unsqueeze(1)  # [N,1,H,W]

    mse_blur = compute_mse_matrix(b_t)
    mse_blur.fill_diagonal_(0.0)
    return mse_blur

# =========================
# Utils
# =========================

def _mst_endpoints_via_diameter(mse: torch.Tensor):
    """
    Build an MST on the dense MSE matrix (edge weights = mse).
    Return (u, v) = endpoints of the MST diameter (longest weighted path).
    """
    N = mse.shape[0]
    if N <= 1:
        return (0, 0)

    device = mse.device
    used = torch.zeros(N, dtype=torch.bool, device=device)
    dist = torch.full((N,), float('inf'), device=device)
    parent = torch.full((N,), -1, dtype=torch.long, device=device)

    # start Prim from node 0
    used[0] = True
    dist = mse[0].clone()
    dist[0] = float('inf')

    for _ in range(N - 1):
        masked = dist.clone()
        masked[used] = float('inf')
        j = int(torch.argmin(masked).item())
        used[j] = True

        # relax edges to unused nodes
        w = mse[j]
        update_mask = (~used) & (w < dist)
        dist[update_mask] = w[update_mask]
        parent[update_mask] = j

    # build adjacency list of the MST
    adj = [[] for _ in range(N)]
    for v in range(1, N):
        u = int(parent[v].item())
        if u >= 0:
            w = float(mse[u, v].item())
            adj[u].append((v, w))
            adj[v].append((u, w))

    def _farthest(src: int):
        # single-source longest distances on a tree via DFS
        distv = [-1.0] * N
        distv[src] = 0.0
        stack = [src]
        while stack:
            x = stack.pop()
            for y, w in adj[x]:
                if distv[y] < 0.0:
                    distv[y] = distv[x] + w
                    stack.append(y)
        far = max(range(N), key=lambda k: distv[k])
        return far, distv[far]

    a, _ = _farthest(0)
    b, _ = _farthest(a)
    return a, b

def double_ended_greedy_from_pair(left: int, right: int, mse: torch.Tensor):
    """
    Maintain a path [left ... right]. At each step, attach the unused frame
    with minimal MSE to either end (choose the cheaper side).
    """
    N = mse.shape[0]
    used = torch.zeros(N, dtype=torch.bool, device=mse.device)
    used[left] = True
    used[right] = True

    path = [left, right]
    inf = float('inf')

    for _ in range(N - 2):
        # best to left
        candL = mse[:, left].clone()
        candL[used] = inf
        kL = int(torch.argmin(candL).item())
        dL = float(candL[kL])

        # best to right
        candR = mse[:, right].clone()
        candR[used] = inf
        kR = int(torch.argmin(candR).item())
        dR = float(candR[kR])

        if dL <= dR:
            path.insert(0, kL)
            used[kL] = True
            left = kL
        else:
            path.append(kR)
            used[kR] = True
            right = kR

    return path


def parse_shuffled_list(s: str):
    """
    Parse 'shuffled_frames_list' column.
    Example cell:
        "130,288,254,17,63,..."
    """
    return [int(x) for x in str(s).split(",") if x.strip() != ""]
    
    
def candidate_starts(mse: torch.Tensor, top_k: int = 10):
    """Get top-k candidate start nodes with largest average distance (likely endpoints)."""
    avg = mse.mean(dim=1)
    vals, idxs = torch.topk(avg, k=min(top_k, mse.shape[0]))
    return idxs.tolist()



def find_video_path(video_id: str, videos_dir: str) -> str:
    """
    Resolve the video path for a given video_id.

    Tries:
      - videos_dir / "<video_id>"
      - videos_dir / "<video_id>.avi"
      - videos_dir / "<video_id>.*" where extension in VIDEO_EXTS
    """
    # direct exact path (some CSVs store full filename)
    direct = os.path.join(videos_dir, video_id)
    if os.path.isfile(direct):
        return direct

    # try with .avi extension
    direct_avi = direct + ".avi"
    if os.path.isfile(direct_avi):
        return direct_avi

    # fallback: any file that starts with video_id
    pattern = os.path.join(videos_dir, f"{video_id}*")
    candidates = [
        p for p in glob.glob(pattern)
        if os.path.splitext(p)[1].lower() in VIDEO_EXTS
    ]

    if not candidates:
        raise FileNotFoundError(
            f"No video file found for video_id={video_id} in {videos_dir}"
        )

    # deterministic choice
    candidates.sort(key=lambda x: (len(os.path.basename(x)), x))
    return candidates[0]


# =========================
# Video loading (grayscale)
# =========================

def load_video_gray(video_path: str, expected_num_frames: int = None) -> torch.Tensor:
    """
    Load frames from a shuffled video as grayscale,
    resize to IMG_SIZE, and send to DEVICE.

    Returns:
        frames: [N, 1, H, W] float32 in [0,1] on DEVICE
    """
    if not os.path.isfile(video_path):
        raise FileNotFoundError(f"Video not found: {video_path}")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video: {video_path}")

    frames = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)
        frames.append(gray)
    cap.release()

    if len(frames) == 0:
        raise ValueError(f"No frames read from {video_path}")

    if expected_num_frames is not None and len(frames) != expected_num_frames:
        print(
            f"[WARN] {os.path.basename(video_path)}: "
            f"expected_num_frames={expected_num_frames}, read={len(frames)}"
        )

    arr = np.stack(frames, axis=0)          # [N, H, W]
    t = torch.from_numpy(arr).float()       # [N, H, W]
    t = t.unsqueeze(1) / 255.0              # [N, 1, H, W] in [0,1]
    return t.to(DEVICE)

def load_video_yuv(video_path: str, expected_num_frames: int = None) -> torch.Tensor:
    """
    Load frames from a shuffled video in YUV color space,
    resize to IMG_SIZE, and send to DEVICE.

    Returns:
        frames: [N, 3, H, W] float32 in [0,1] on DEVICE
    """
    if not os.path.isfile(video_path):
        raise FileNotFoundError(f"Video not found: {video_path}")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video: {video_path}")

    frames = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        # Convert to YUV color space
        yuv = cv2.cvtColor(frame, cv2.COLOR_BGR2YUV)
        # Resize
        yuv = cv2.resize(yuv, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)
        frames.append(yuv)
    cap.release()

    if len(frames) == 0:
        raise ValueError(f"No frames read from {video_path}")

    if expected_num_frames is not None and len(frames) != expected_num_frames:
        print(
            f"[WARN] {os.path.basename(video_path)}: "
            f"expected_num_frames={expected_num_frames}, read={len(frames)}"
        )

    # Stack into tensor and normalize
    arr = np.stack(frames, axis=0)  # [N, H, W, 3]
    t = torch.from_numpy(arr).permute(0, 3, 1, 2).float() / 255.0  # [N, 3, H, W]
    return t.to(DEVICE)

# =========================
# Path construction
# =========================
    

def build_best_path(mse: torch.Tensor):
    """Build temporal path using MST diameter endpoints and double-ended greedy growth."""
    N = mse.shape[0]
    if N <= 2:
        return list(range(N))

    # smart seed via MST diameter
    a, b = _mst_endpoints_via_diameter(mse)

    # grow from both ends
    path = double_ended_greedy_from_pair(a, b, mse)

    # keep your local refinement
    path = refine_path(path, mse)
    return path

# =========================
# Local refinement
# =========================

def refine_path(path, mse: torch.Tensor, max_passes: int = 10):
    """
    Local search with adjacent swaps.
    Only apply swap if it reduces total path cost.
    """
    N = len(path)
    if N <= 2:
        return path

    for _ in range(max_passes):
        improved = False

        for k in range(N - 1):
            i = path[k]
            j = path[k + 1]

            left = path[k - 1] if k > 0 else None
            right = path[k + 2] if (k + 2) < N else None

            old_cost = 0.0
            new_cost = 0.0

            if left is not None:
                old_cost += float(mse[left, i])
                new_cost += float(mse[left, j])

            old_cost += float(mse[i, j])

            if right is not None:
                old_cost += float(mse[j, right])
                new_cost += float(mse[i, right])

            if new_cost + 1e-9 < old_cost:
                path[k], path[k + 1] = path[k + 1], path[k]
                improved = True

        if not improved:
            break

    return path


# =========================
# Per-video prediction
# =========================

def predict_order_for_video(video_id: str,
                            shuffled_order,
                            videos_dir: str):
    """
    Pipeline for a single video_id:
      - load shuffled video frames
      - compute MSE matrix
      - build best greedy path
      - refine path
      - map positions to original frame indices
    """
    shuffled_order = list(shuffled_order)
    expected_num_frames = len(shuffled_order)

    video_path = find_video_path(video_id, videos_dir)
    # frames = load_video_gray(video_path, expected_num_frames=expected_num_frames)
    frames = load_video_yuv(video_path, expected_num_frames=expected_num_frames)
    frames = frames[:, 0:1, :, :] 
    N = frames.shape[0]

    if N != expected_num_frames:
        print(
            f"[WARN] {video_id}: csv_frames={expected_num_frames}, "
            f"video_frames={N}. Using min of both."
        )
        m = min(expected_num_frames, N)
        shuffled_order = shuffled_order[:m]
        frames = frames[:m]
        N = m

    if N <= 1:
        return [int(x) for x in shuffled_order]

    mse = compute_blurred_mse_matrix(frames) 
    path = build_best_path(mse)
    
    predicted = [int(shuffled_order[idx]) for idx in path]
    return predicted



# =========================
# Process all CSVs
# =========================

def process_all_csvs(csv_dir: str, videos_dir: str, out_dir: str):
    """
    For each CSV in csv_dir:
      - read video_id, shuffled_frames_list
      - compute predicted order for each video
      - write a prediction CSV with same filename into out_dir
    """
    os.makedirs(out_dir, exist_ok=True)

    csv_paths = sorted(glob.glob(os.path.join(csv_dir, "*.csv")))
    if not csv_paths:
        raise FileNotFoundError(f"No CSV files found in {csv_dir}")

    for csv_path in csv_paths:
        df = pd.read_csv(csv_path)
        rows = []

        if "video_id" not in df.columns or "shuffled_frames_list" not in df.columns:
            raise ValueError(
                f"CSV {csv_path} must contain 'video_id' and 'shuffled_frames_list' columns."
            )

        for _, row in df.iterrows():
            video_id = str(row["video_id"]).strip()
            shuffled_order = parse_shuffled_list(row["shuffled_frames_list"])
            pred = predict_order_for_video(video_id, shuffled_order, videos_dir)
            pred_str = ",".join(str(x) for x in pred)
            rows.append({"video_id": video_id, "predicted_frames_list": pred_str})

        out_csv = os.path.join(out_dir, os.path.basename(csv_path))
        pd.DataFrame(rows).to_csv(out_csv, index=False)
        print(f"[OK] {os.path.basename(csv_path)} -> {os.path.basename(out_csv)}")


# =========================
# CLI
# =========================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Reconstruct frame order from shuffled videos "
                    "using grayscale MSE and CSV metadata."
    )
    parser.add_argument(
        "--csv_dir",
        type=str,
        required=True,
        help="Directory with shuffled CSV files (e.g. shuffled_csvs).",
    )
    parser.add_argument(
        "--videos_dir",
        type=str,
        required=True,
        help="Directory with shuffled videos (e.g. UCF101_videos_shuffled).",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="./shuffled_artifacts/ordered_CSVs",
        help="Output directory for prediction CSVs.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    process_all_csvs(args.csv_dir, args.videos_dir, args.out_dir)


if __name__ == "__main__":
    main()


