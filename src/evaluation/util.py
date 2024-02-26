import os
from typing import List

import imageio.v3 as iio
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from matplotlib.patches import Circle, Rectangle


def load_video(path: str):
    if path.endswith(".mp4"):
        return iio.imread(path, plugin="pyav")
    elif path.endswith(".avi"):
        return iio.imread(path, format="FFMPEG")


def get_video_batch(video_paths: list[str]) -> list[torch.Tensor]:
    return [torch.from_numpy(load_video(p)) for p in video_paths]


def make_heatmap(
    similarity_matrix: np.ndarray,
    groups: List[str],
    trajectories_names: List[str],
    labels: List[str],
    result_dir: str,
    experiment_id: str,
):
    fig, ax = plt.subplots(figsize=(25, 25))

    new_similarity_matrix = similarity_matrix.copy()
    new_labels = labels.copy()
    new_trajectory_names = trajectories_names.copy()
    shift = 0
    for i in range(1, len(groups)):
        if groups[i] != groups[i - 1]:
            new_similarity_matrix = np.insert(
                new_similarity_matrix, i + shift, np.nan, axis=0
            )
            new_similarity_matrix = np.insert(
                new_similarity_matrix, i + shift, np.nan, axis=1
            )
            new_labels.insert(i + shift, "")
            new_trajectory_names.insert(i + shift, "")
            shift += 1

    sns.heatmap(
        new_similarity_matrix,
        annot=True,
        fmt=".3f",
        cmap="viridis",
        xticklabels=new_labels,
        yticklabels=new_trajectory_names,
        cbar=False,
    )

    for i in range(similarity_matrix.shape[0] + len(groups)):
        ax.add_patch(Rectangle((i, i), 1, 1, fill=False, edgecolor="red", lw=3))

    # Add borders around the cells with the highest values
    for i in range(new_similarity_matrix.shape[0]):
        row = new_similarity_matrix[i, :]
        if not np.isnan(row).all():
            mask = np.isfinite(row)
            # Use the mask to ignore nan values when sorting
            n = 5
            top_n_indices = np.argsort(row[mask])[-n:]
            # Adjust the indices to account for the mask
            top_n_indices = np.arange(len(row))[mask][top_n_indices]
            for j, index in enumerate(top_n_indices):
                ax.add_patch(
                    Circle(
                        (index + 0.5, i + 0.5),
                        0.5,
                        fill=False,
                        edgecolor="violet",
                        lw=2 * (j + 1),
                    )
                )

    plt.title(f"{experiment_id}")
    plt.xticks(rotation=30, ha="right")
    plt.yticks(rotation=0)
    if not os.path.exists(result_dir):
        os.mkdir(result_dir)
    plt.savefig(f"{result_dir}/{experiment_id}.pdf", dpi=350)
