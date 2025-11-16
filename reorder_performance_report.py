#!/usr/bin/env python3
"""
Visualize the top-K worst actions for frame order reconstruction.

This script reuses the evaluation logic (Kendall's Tau and normalized
edit distance) and aggregates metrics per action category. It then
plots a bar chart of the actions with the lowest average performance.

Example usage:

  python visualize_worst_actions.py \
      --csv-dir ./shuffled_artifacts/shuffled_CSVs \
      --predictions ./shuffled_artifacts/ordered_CSVs \
      --metric kendall_tau \
      --top-k 10 \
      --output reorder_performance_report.png
"""

import argparse
import csv
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.stats import kendalltau


def compute_kendall_tau(true_order, pred_order):
    """Compute Kendall's Tau correlation coefficient."""
    tau, _ = kendalltau(true_order, pred_order)
    return tau


def levenshtein_distance(seq1, seq2):
    """Compute Levenshtein distance (edit distance) between two sequences."""
    if len(seq1) < len(seq2):
        return levenshtein_distance(seq2, seq1)

    if len(seq2) == 0:
        return len(seq1)

    previous_row = range(len(seq2) + 1)
    for i, c1 in enumerate(seq1):
        current_row = [i + 1]
        for j, c2 in enumerate(seq2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row

    return previous_row[-1]


def compute_normalized_edit_distance(true_order, pred_order):
    """
    Compute normalized edit distance (Levenshtein distance).
    Returns a similarity score in [0, 1].
    """
    d_edit = levenshtein_distance(true_order, pred_order)
    return 1.0 - (d_edit / len(true_order))


def evaluate_video(true_order, pred_order):
    """
    Bidirectional evaluation (same as in evaluate_shuffled_predictions.py).
    """
    tau_forward = compute_kendall_tau(true_order, pred_order)
    edit_forward = compute_normalized_edit_distance(true_order, pred_order)

    reversed_order = list(reversed(true_order))
    tau_backward = compute_kendall_tau(reversed_order, pred_order)
    edit_backward = compute_normalized_edit_distance(reversed_order, pred_order)

    if tau_forward >= tau_backward:
        return {
            "kendall_tau": tau_forward,
            "edit_distance_score": edit_forward,
            "direction": "forward",
        }
    else:
        return {
            "kendall_tau": tau_backward,
            "edit_distance_score": edit_backward,
            "direction": "backward",
        }


def load_predictions_dir(predictions_dir: str):
    """
    Load predicted orderings from directory of CSV files (one per action).
    """
    predictions_path = Path(predictions_dir)
    predictions = {}

    csv_files = list(predictions_path.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {predictions_path}")

    for csv_file in csv_files:
        with open(csv_file, "r") as f:
            reader = csv.reader(f)
            next(reader, None)  # skip header
            for row in reader:
                if len(row) < 2:
                    continue
                video_id = row[0]
                pred_order = list(map(int, row[1].split(",")))
                predictions[video_id] = pred_order

    return predictions


def aggregate_per_action(csv_dir: str, predictions_dir: str, max_videos=None):
    """
    Compute per-action average metrics (Kendall's Tau and edit distance).

    Returns:
        per_action: dict[action_name] = {
            "kendall_tau": mean_tau,
            "edit_distance_score": mean_edit,
            "count": num_videos
        }
        all_taus: list of (video_id, tau) for each evaluated video
    """
    csv_path = Path(csv_dir)
    csv_files = sorted(list(csv_path.glob("*.csv")))

    predictions = load_predictions_dir(predictions_dir)

    per_action = {}
    total_videos = 0
    all_taus = []

    for csv_file in tqdm(csv_files, desc="Aggregating actions", unit="action"):
        action_name = csv_file.stem

        with open(csv_file, "r") as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        tau_vals = []
        edit_vals = []

        for row in rows:
            if max_videos and total_videos >= max_videos:
                break

            video_id = row["video_id"]
            shuffled_order = list(map(int, row["shuffled_frames_list"].split(",")))
            n_frames = len(shuffled_order)

            if video_id not in predictions:
                continue

            pred_order = predictions[video_id]
            if len(pred_order) != n_frames:
                # skip mismatched predictions
                continue

            true_order = list(range(n_frames))
            metrics = evaluate_video(true_order, pred_order)

            tau_vals.append(metrics["kendall_tau"])
            edit_vals.append(metrics["edit_distance_score"])
            total_videos += 1
            all_taus.append((video_id, metrics["kendall_tau"]))

        if tau_vals:
            per_action[action_name] = {
                "kendall_tau": float(np.mean(tau_vals)),
                "edit_distance_score": float(np.mean(edit_vals)),
                "count": len(tau_vals),
            }

        if max_videos and total_videos >= max_videos:
            break

    return per_action, all_taus


def plot_tau_histogram(ax, all_taus):
    """
    Plot a histogram of video-level Kendall's Tau with 10% bins.
    Each bar is annotated with the percentage of videos in that bin.
    """
    if not all_taus:
        ax.text(0.5, 0.5, "No data", ha="center", va="center")
        ax.set_axis_off()
        return

    taus = np.array([t for _, t in all_taus], dtype=float)
    bins = np.linspace(0.0, 1.0, 11)  # 0.0,0.1,...,1.0
    counts, edges = np.histogram(taus, bins=bins)
    total = counts.sum()

    centers = (edges[:-1] + edges[1:]) / 2.0
    widths = np.diff(edges)

    ax.bar(
        centers,
        counts,
        width=widths,
        align="center",
        color="steelblue",
        alpha=0.8,
        edgecolor="black",
        linewidth=0.8,
    )
    ax.set_xlabel("Kendall's Tau")
    ax.set_ylabel("Video count")
    ax.set_title("Distribution of video-level Kendall's Tau")
    ax.set_xlim(0.0, 1.0)
    # Tick labels at every 0.1
    ax.set_xticks(np.linspace(0.0, 1.0, 11))

    # Annotate each bar with percentage
    for c, x in zip(counts, centers):
        if c == 0 or total == 0:
            continue
        pct = 100.0 * c / total
        ax.text(
            x,
            c,
            f"{pct:.1f}%",
            ha="center",
            va="bottom",
            fontsize=8,
        )


def plot_worst_actions_on_ax(ax, per_action, metric: str, top_k: int):
    """
    Plot a bar chart of the worst (lowest) actions according to the chosen metric on a given Axes.
    Shows mean tau above bars, without frame counts.
    """
    if not per_action:
        ax.text(0.5, 0.5, "No per-action metrics", ha="center", va="center")
        ax.set_axis_off()
        return

    if metric not in {"kendall_tau", "edit_distance_score"}:
        raise ValueError("metric must be 'kendall_tau' or 'edit_distance_score'")

    items = sorted(
        per_action.items(),
        key=lambda kv: kv[1][metric],
    )

    worst = items[:top_k]
    actions = [k for k, _ in worst]
    values = [v[metric] for _, v in worst]

    bars = ax.bar(range(len(actions)), values, color="tomato", alpha=0.8)

    ax.set_xticks(range(len(actions)))
    ax.set_xticklabels(actions, rotation=45, ha="right")
    ax.set_ylabel(metric.replace("_", " ").title())
    ax.set_title(f"Worst {len(actions)} actions by {metric}")
    ax.set_ylim(min(values) - 0.05, min(1.0, max(values) + 0.05))

    # Annotate bars with mean score only
    for bar, val in zip(bars, values):
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            height,
            f"{val:.3f}",
            ha="center",
            va="bottom",
            fontsize=8,
        )


def plot_chunk_density(ax, all_taus, chunk_size: int = 1000):
    """
    Plot a "density-like" curve of mean tau over chunks of videos.

    - Sort all per-video taus ascending.
    - Group into consecutive chunks of `chunk_size` videos.
    - For each chunk, compute the mean tau.
    - Plot mean tau vs. chunk index (or percentile).
    - Show global mean, median, and std as reference.
    """
    if not all_taus:
        ax.text(0.5, 0.5, "No data", ha="center", va="center")
        ax.set_axis_off()
        return

    # Sort videos by tau ascending, but plot each video individually
    taus = np.array([t for _, t in all_taus], dtype=float)
    sort_idx = np.argsort(taus)
    sorted_taus = taus[sort_idx]

    x = np.arange(sorted_taus.size)  # video index (sorted by tau)

    ax.plot(x, sorted_taus, linestyle="-", color="purple", alpha=0.8)
    ax.set_xlabel("Videos (sorted by Kendall's Tau)")
    ax.set_ylabel("Kendall's Tau")
    ax.set_title("Per-video Tau curve (ascending order)")
    ax.set_xlim(0, sorted_taus.size - 1)
    ax.set_ylim(0.0, 1.0)
    ax.grid(True, alpha=0.3)

    # Global statistics
    global_mean = float(np.mean(sorted_taus))
    global_median = float(np.median(sorted_taus))
    global_std = float(np.std(sorted_taus))

    ax.axhline(global_mean, color="green", linestyle="--", linewidth=1, label=f"Mean={global_mean:.3f}")
    ax.axhline(global_median, color="orange", linestyle=":", linewidth=1, label=f"Median={global_median:.3f}")
    ax.axhline(global_mean + global_std, color="gray", linestyle="--", linewidth=0.8, alpha=0.7)
    ax.axhline(global_mean - global_std, color="gray", linestyle="--", linewidth=0.8, alpha=0.7)

    ax.legend(fontsize=8, loc="lower right")


def main():
    parser = argparse.ArgumentParser(
        description="Visualize worst-performing actions for frame order reconstruction."
    )
    parser.add_argument(
        "--csv-dir",
        required=True,
        help="Directory containing shuffled CSV files (reference data).",
    )
    parser.add_argument(
        "--predictions",
        required=True,
        help="Directory containing prediction CSV files (one per action).",
    )
    parser.add_argument(
        "--metric",
        choices=["kendall_tau", "edit_distance_score"],
        default="kendall_tau",
        help="Metric to rank actions by (default: kendall_tau).",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="Number of worst actions to visualize (default: 10).",
    )
    parser.add_argument(
        "--max-videos",
        type=int,
        help="Optional limit on total number of videos to evaluate.",
    )
    parser.add_argument(
        "--output",
        help="Optional path to save the plot (e.g., worst_actions.png). "
        "If not set, the plot is shown interactively.",
    )

    args = parser.parse_args()

    per_action, all_taus = aggregate_per_action(
        csv_dir=args.csv_dir,
        predictions_dir=args.predictions,
        max_videos=args.max_videos,
    )

    print(f"Computed per-action metrics for {len(per_action)} actions.")

    # Global statistic: percentage of videos with Kendall's Tau >= 0.9
    if all_taus:
        all_taus_arr = np.array([t for _, t in all_taus], dtype=float)
        total = all_taus_arr.size
        high = (all_taus_arr >= 0.9).sum()
        pct = 100.0 * high / total
        print(
            f"Videos with Kendall's Tau >= 0.9: "
            f"{high}/{total} ({pct:.2f}%)"
        )

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Subplot 1: histogram of video-level tau with 10% bins
    plot_tau_histogram(axes[0], all_taus)

    # Subplot 2: worst mean-tau actions (or metric chosen)
    plot_worst_actions_on_ax(axes[1], per_action, args.metric, args.top_k)

    # Subplot 3: chunked density-like profile of tau across dataset
    plot_chunk_density(axes[2], all_taus, chunk_size=1000)

    fig.tight_layout()

    if args.output:
        fig.savefig(args.output, dpi=150)
        print(f"Saved combined figure to {args.output}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
