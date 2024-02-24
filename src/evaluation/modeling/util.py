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


def load_prompts(path: str, verbose: bool) -> List[str]:
    prompts = []

    with open(path, "r") as f:
        for line in f.readlines():
            prompts.append(line.rstrip("\n"))

    if verbose:
        print("Loaded promts:")
        for i, p in enumerate(prompts):
            print(f"{i:2d}: {p}")

    return prompts


def get_video_batch(
    trajectories_path: str,
    prepare_video: Callable,
    n_frames: int,
    verbose: bool = False,
) -> torch.Tensor:
    """Reads a list of video paths, loads videos, preprocess them with `prepare_video` function and arranges them in a batch."""
    with open(trajectories_path, "r") as f:
        video_paths = [line.rstrip("\n") for line in f.readlines()]

    # Preprocessing can be more efficient if done for the whole batch simultaniously
    videos = torch.cat(
        [
            prepare_video(load_video(p), n_frames=n_frames, verbose=verbose)
            for p in video_paths
        ],
        dim=0,
    )

    return videos, video_paths


def make_heatmap(
    similarity_matrix: np.ndarray,
    trajectories_names: List[str],
    labels: List[str],
    result_dir: str,
    experiment_id: str,
):
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
