import argparse

import evaluation.modeling.util as util
import pandas as pd
import torch
import torch.nn.functional as F
from einops import rearrange
from vlmrm.contrib.viclip import get_viclip
from vlmrm.reward.encoders import ViCLIP
from vlmrm.reward.rewards import logit_reward, projection_reward


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compare trajectory with given prompts"
    )
    parser.add_argument(
        "-t",
        "--table-path",
        help="Path to table containing video paths and their labels in csv format.",
        required=True,
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
        default="evaluation_results",
    )
    parser.add_argument("--n-frames", type=int, default=8)
    parser.add_argument("-v", "--verbose", action="store_true")
    parser.add_argument("--cache-dir", default=".cache")

    args = parser.parse_args()
    return args


def load_model_and_tokenizer(path: str):
    return get_viclip(pretrain=path)


@torch.inference_mode()
def main():
    args = parse_args()
    if args.verbose:
        print(f"Running ViCLIP evaluator with following args:\n{args}")

    data = pd.read_csv(args.table_path)

    videos, video_paths = util.get_video_batch(data["path"].to_list())
    prompts = data["label"].to_list()
    encoder = ViCLIP(args.cache_dir)

    videos = torch.stack([encoder.subsample(video) for video in videos])

    with torch.no_grad():
        text_encodings = encoder.encode_text(prompts)
        # The encoder expects the input to be in the format (frames, windows, episodes, c h w)
        videos = rearrange(videos, "b f c h w -> f 1 b c h w")
        # (f w e c h w) -> (w e d)
        video_encodings = encoder.encode_video(videos)
        video_encodings = rearrange(video_encodings, "1 b d -> b d")

        # similarity = logit_reward(
        #     video_encodings, text_encodings, torch.arange(len(text_encodings))
        # )

        baseline = encoder.encode_text(["a red car"])

        similarity = torch.stack(
            [
                projection_reward(video_encodings, baseline, target.unsqueeze(0), 0.65)
                for target in text_encodings
            ],
            dim=1,
        )

    trajectory_names = [util.strip_directories_and_extension(p) for p in video_paths]

    util.make_heatmap(
        similarity.cpu().numpy(),
        trajectory_names,
        prompts,
        args.output_dir,
        args.experiment_id,
    )


if __name__ == "__main__":
    main()
