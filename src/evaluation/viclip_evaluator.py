import argparse

import evaluation.modeling.util as util
import numpy as np
import torch
import torch.nn.functional as F
from einops import rearrange
from vlmrm.contrib.viclip import get_viclip
from vlmrm.reward_model import ViCLIP, ViCLIPEmbed


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compare trajectory with given prompts"
    )

    parser.add_argument(
        "-t",
        "--trajectories-path",
        help="Path to trajectory in mp4 format.",
        required=True,
    )
    parser.add_argument(
        "-p",
        "--prompts-path",
        help="Path to prompts in txt format. Expected to have one prompt per line.",
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


def prepare_video(video: np.ndarray, n_frames: int, verbose: bool):
    if verbose:
        print("Initial video shape:", video.shape, " dtype:", video.dtype)

    # Probably not most accurate frame sampling -- might be improved
    length = video.shape[0]
    step_size = length // n_frames
    video = video[::step_size][:n_frames]

    video = torch.from_numpy(video)

    if video.dtype not in (torch.float16, torch.float32, torch.float64):
        v_mean = torch.Tensor([0.485, 0.456, 0.406]).reshape(1, 1, 1, 3)
        v_std = torch.Tensor([0.229, 0.224, 0.225]).reshape(1, 1, 1, 3)
        video = (video.float() / 255 - v_mean) / v_std

    video = F.interpolate(
        video.permute(0, 3, 1, 2),
        mode="bicubic",
        size=(224, 224),
    )

    if verbose:
        print("Min and max values:", video.min(), video.max())

    video = video.unsqueeze(0)

    if verbose:
        print("Final video shape:", video.shape, " dtype:", video.dtype)

    return video


def load_model_and_tokenizer(path: str):
    return get_viclip(pretrain=path)


@torch.inference_mode()
def main():
    args = parse_args()
    if args.verbose:
        print(f"Running ViCLIP evaluator with following args:\n{args}")

    videos, video_paths = util.get_video_batch(
        args.trajectories_path, prepare_video, args.n_frames, args.verbose
    )

    prompts = util.load_prompts(args.prompts_path, verbose=args.verbose)

    text_encoder = ViCLIP(args.cache_dir)
    video_encoder = ViCLIPEmbed(text_encoder)

    with torch.no_grad():
        text_encoding = text_encoder.encode_text(prompts)
        videos = rearrange(videos, "b f c h w -> f 1 b h w c")
        video_encoding = video_encoder(videos)
        video_encoding = rearrange(video_encoding, "1 b d -> b d")
        similarity = F.cosine_similarity(text_encoding, video_encoding)
    trajectory_names = [util.strip_directories_and_extension(p) for p in video_paths]
    print(f"{similarity.shape=}, {len(trajectory_names)=}")
    util.make_heatmap(
        similarity.unsqueeze(0).cpu().numpy(),
        trajectory_names,
        prompts,
        args.output_dir,
        args.experiment_id,
    )


if __name__ == "__main__":
    main()
