import argparse
from pathlib import Path
from typing import Callable

import pandas as pd
import torch
import vlmrm.reward.rewards as rewards
from einops import rearrange
from evaluation import util
from torch import Tensor
from vlmrm.reward.encoders import CLIP, S3D, Encoder, ViCLIP


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compare trajectory with given descriptions"
    )
    parser.add_argument(
        "-t",
        "--table-path",
        help="Path to a csv table containing video paths and their descriptions.",
        required=True,
    )
    parser.add_argument(
        "-m",
        "--model",
        help="Name of the model to evaluate (ViCLIP, S3D, CLIP)",
        required=True,
    )
    parser.add_argument(
        "-r",
        "--rewards",
        help="Name of the reward to calculate (logit, projection)",
        required=True,
    )
    parser.add_argument(
        "-a",
        "--alphas",
        help="If using projection reward, the value of alpha to use.",
        required=False,
        default=None,
    )
    parser.add_argument(
        "-e",
        "--experiment-id",
        help="Name of current experiment (used to save the results)",
        required=True,
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        help="Directory to save evaluation results.",
        default="out",
    )
    parser.add_argument("--cache-dir", default=".cache")

    args = parser.parse_args()
    return args


def evaluate(
    encoder: Encoder,
    videos: list[Tensor],
    descriptions: list[str],
    reward: Callable[[Tensor, Tensor], Tensor],
):
    subsampled_videos = torch.stack([encoder.subsample(video) for video in videos])
    # The encoder expects the input to be (frames, windows, episodes, c h w)
    subsampled_videos = rearrange(subsampled_videos, "b f c h w -> f 1 b c h w")
    # (f w e c h w) -> (w e d)
    video_encodings = encoder.encode_video(subsampled_videos)
    video_encodings = rearrange(video_encodings, "1 b d -> b d")

    description_encodings = encoder.encode_text(descriptions)

    return reward(video_encodings, description_encodings)


def logit_reward(video_encodings: Tensor, description_encodings: Tensor):
    return rewards.logit_reward(
        video_encodings, description_encodings, torch.arange(len(description_encodings))
    )


def mk_projection_reward(alpha: float, baselines: Tensor):
    def reward(video_encodings: Tensor, description_encodings: Tensor) -> Tensor:
        reward_cols = [
            rewards.projection_reward(video_encodings, b, t.unsqueeze(0), alpha)
            for t, b in zip(description_encodings, baselines)
        ]
        return torch.stack(reward_cols, dim=1)

    return reward


def main():
    args = parse_args()
    assert isinstance(args.model, str)
    if args.model.lower() == "viclip":
        encoder = ViCLIP(args.cache_dir)
    elif args.model.lower() == "s3d":
        encoder = S3D(args.cache_dir)
    elif args.model.lower() == "clip":
        model = "ViT-bigG-14/laion2b_s39b_b160k"
        model_name_prefix, pretrained = model.split("/")
        encoder = CLIP(model_name_prefix, pretrained, args.cache_dir)

    data = pd.read_csv(args.table_path)
    video_paths = data["path"].to_list()
    video_names = [Path(p).stem for p in video_paths]
    videos = util.get_video_batch(video_paths)
    descriptions = data["label"].to_list()

    rewards = []
    for reward_name in args.rewards.split(","):
        if reward_name == "logit":
            rewards.append((logit_reward, f"{args.model}_logit_{args.experiment_id}"))
        elif reward_name == "projection":
            baselines = encoder.encode_text(data["baseline"].to_list())
            if args.alphas is None:
                raise ValueError("Alpha must be provided when using projection reward.")
            for alpha in [float(a) for a in args.alphas.split(",")]:
                reward_fun = mk_projection_reward(alpha, baselines)
                title = f"{args.model}_projection_{alpha}_{args.experiment_id}"
                rewards.append((reward_fun, title))

    for reward_fun, title in rewards:
        reward_matrix = evaluate(encoder, videos, descriptions, reward_fun)
        util.make_heatmap(
            reward_matrix.cpu().numpy(),
            groups=data["group"].to_list(),
            trajectories_names=video_names,
            labels=descriptions,
            result_dir=args.output_dir,
            experiment_id=title,
        )


if __name__ == "__main__":
    main()
