import os
from typing import Callable, List

import imageio.v3 as iio
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch


def load_video(path: str):
    if path.endswith(".mp4"):
        return iio.imread(path, plugin="pyav")
    elif path.endswith(".avi"):
        return iio.imread(path, format="FFMPEG")


def get_video_batch(
    video_paths: list[str],
) -> tuple[list[torch.Tensor], list[str]]:
    """Reads a list of video paths, loads videos, preprocess them with `prepare_video` function and arranges them in a batch."""

    # Preprocessing can be more efficient if done for the whole batch simultaniously
    videos = [torch.from_numpy(load_video(p)) for p in video_paths]

    return videos, video_paths


def make_heatmap(
    similarity_matrix: np.ndarray,
    trajectories_names: List[str],
    labels: List[str],
    result_dir: str,
    experiment_id: str,
):
    plt.figure(figsize=(30, 30))
    sns.heatmap(
        similarity_matrix,
        annot=True,
        fmt=".3f",
        cmap="crest",
        xticklabels=labels,
        yticklabels=trajectories_names,
    )
    plt.title(experiment_id)
    plt.xticks(rotation=30, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    if not os.path.exists(result_dir):
        os.mkdir(result_dir)
    plt.savefig(f"{result_dir}/{experiment_id}.png", dpi=350)


def strip_directories_and_extension(path: str):
    return path.split("/")[-1].split(".")[0]
