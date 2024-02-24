import argparse

import numpy as np
import torch
import torch.nn.functional as F
from evaluation.modeling import util
from evaluation.modeling.s3dg import S3D


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compare trajectory with given prompts"
    )

    parser.add_argument(
        "-t",
        "--trajectories-path",
        help="Path to a file, containing paths to trajectories in mp4 format.",
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
    parser.add_argument("--n-frames", type=int, default=32)
    parser.add_argument("-v", "--verbose", action="store_true")
    parser.add_argument("--model-checkpoint-path", default=".cache/s3d_howto100m.pth")

    args = parser.parse_args()
    return args


def prepare_video(video: np.ndarray, n_frames: int, verbose: bool) -> torch.Tensor:
    if verbose:
        print("Initial video shape:", video.shape, " dtype:", video.dtype)

    # Probably not most accurate frame sampling -- might be improved
    length = video.shape[0]
    step_size = length // n_frames
    video = video[::step_size][:n_frames]

    video = torch.from_numpy(video)

    if video.dtype not in (torch.float16, torch.float32, torch.float64):
        video = video.float() / 255

    video = F.interpolate(
        video.permute(0, 3, 1, 2),
        mode="bicubic",
        scale_factor=0.5,
    ).transpose(0, 1)

    if verbose:
        print("Min and max before clipping:", video.min(), video.max())

    video = video.clamp(0, 1).unsqueeze(0)

    if verbose:
        print("Final video shape:", video.shape, " dtype:", video.dtype)

    return video


def load_model(model_checkpoint_path):
    # Instantiate the model
    embedding_dim = 512
    net = S3D(".cache/s3d_dict.npy", embedding_dim)
    # Load the model weights
    net.load_state_dict(torch.load(model_checkpoint_path))
    # Evaluation mode
    net = net.eval()
    return net


@torch.inference_mode()
def main():
    args = parse_args()
    if args.verbose:
        print(f"Running S3D evaluator with following args:\n{args}")

    # Video input should be of size Batch x 3 x T x H x W and normalized to [0, 1]
    # Also, afaik expects either 32 or 16 frames
    videos, video_paths = util.get_video_batch(
        args.trajectories_path, prepare_video, args.n_frames, args.verbose
    )
    prompts = util.load_prompts(args.prompts_path, verbose=args.verbose)

    net = load_model(args.model_checkpoint_path)

    # Video inference
    if args.verbose:
        print("Embedding videos...")
    video_output = net(videos)

    # Text inference
    if args.verbose:
        print("Embedding text...")
    text_output = net.text_module(prompts)

    v_embed = video_output["video_embedding"] / video_output["video_embedding"].norm(
        p=2, dim=-1, keepdim=True
    )
    p_embeds = text_output["text_embedding"] / text_output["text_embedding"].norm(
        p=2, dim=-1, keepdim=True
    )

    similarities = v_embed @ p_embeds.T

    if args.verbose:
        print("similarities.shape:", similarities.shape)

    trajectory_names = [util.strip_directories_and_extension(p) for p in video_paths]
    util.make_heatmap(
        similarities.cpu().numpy(),
        trajectory_names,
        prompts,
        args.output_dir,
        args.experiment_id,
    )


if __name__ == "__main__":
    main()
