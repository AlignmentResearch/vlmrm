import argparse
import base64
import json
import re
from pathlib import Path
from typing import Callable

import cv2
import dotenv
import numpy as np
import openai
import pandas as pd
import torch
import vlmrm.reward.rewards as rewards
from einops import rearrange
from evaluation import util
from torch import Tensor
from torch.amp.autocast_mode import autocast
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
        "-n",
        "--n_frames",
        help="How many frames to use for the video encoding. Only used in CLIP.",
    )
    parser.add_argument(
        "-r",
        "--rewards",
        help="Name of the reward to calculate (logit, projection)",
    )
    parser.add_argument(
        "-a",
        "--alphas",
        help="If using projection reward, the value of alpha to use.",
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
    parser.add_argument(
        "--standardize",
        help="Directory to save evaluation results.",
        default=False,
        action="store_true",
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
    description_encodings.to(video_encodings.device)

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


def subsample(x: torch.Tensor, frames: int) -> torch.Tensor:
    n_frames, *_ = x.shape
    step = n_frames // frames
    x = x[::step, ...][:frames, ...]
    return x


def load_video(path: str, n_frames=5):
    video = cv2.VideoCapture(path)

    b64_frames = []
    while video.isOpened():
        success, frame = video.read()
        if not success:
            break
        _, buffer = cv2.imencode(".jpg", frame)
        b64_frames.append(base64.b64encode(buffer).decode("utf-8"))

    video.release()

    total_frames = len(b64_frames)
    step = total_frames // n_frames
    b64_frames = b64_frames[::step][:n_frames]

    return b64_frames


def gpt4(video_paths, descriptions):
    prompt = """
You will be given five frames from a video depicting a red car. Important notes:
- the frames are given in chronological order
- the camera is fixed and doesn't move or rotate throughout the video
- the car sometimes doesn't follow any roads at all and just rides on grass
- the car sometimes DOES follow roads, so be sure to check this specifically
- the car doesn't necessarily start at the bottom and go up; to observe the car's movement, you need to compare position of the car between the frames

Your task is to describe what you see. Focus on the relative positions of the depicted objects (car, roads, potential crossroads, etc), their orientation, and the movement of the car between the frames. Be precise, but brief. Describe EACH OF THE FIVE FRAMES. The frames are NOT static and they DO change, although sometimes the change is small frame to frame.

# EXAMPLE

Input: [five frames]

Assistant:
0. The car is approaching a roundabout
1. The car is now closer to the roundabout
2. The car is at the roundabout, rode by the first exit
3. The car appears to be taking the second exit
4. The car continues along the road after the roundabout

The car has moved through the roundabout and took the second exit.
    """

    # Subsample the frames from the videos
    videos = [load_video(p) for p in video_paths]

    matrix = np.zeros((len(videos), len(descriptions)))

    for i, video in enumerate(videos):
        print("===========================")
        print(video_paths[i])
        print(descriptions[i])

        # save the frames as separate images
        for j, frame in enumerate(video):
            with open(f"frames/{video_paths[i].split('/')[-1]}_{j}.jpg", "wb") as f:
                f.write(base64.b64decode(frame))

        dotenv.load_dotenv()
        client = openai.OpenAI()  # API KEY should be loaded automatically by line above
        messages = [
            {"role": "system", "content": [{"type": "text", "text": prompt}]},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "# TASK"},
                    *[
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{f}",
                                "detail": "low",
                                "resize": 512,
                            },
                        }
                        for f in video
                    ],
                ],
            },
        ]
        response = client.chat.completions.create(
            model="gpt-4-vision-preview", messages=messages, max_tokens=900
        )
        # Add response.choices[0].message.content to the messages list
        messages.append(
            {"role": "assistant", "content": response.choices[0].message.content}
        )
        print(response.choices[0].message.content)
        messages.append(
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": 'Now, given this description, score the following potential video descriptions from 0 to 1 based on how well they fit the video/frames you\'ve seen. Feel free to use values between 0 and 1, too. Multiple labels can have the same score. The directions (e.g. "car turning left") are written from the POV of the driver of the car unless specified otherwise.\n\nFormat: \n- description: score\n\n'
                        + "\n".join([f"- {d}" for d in set(descriptions)]),
                    }
                ],
            }
        )
        response = client.chat.completions.create(
            model="gpt-4-vision-preview", messages=messages, max_tokens=900
        )
        answer = response.choices[0].message.content

        print(answer)

        # Parse out the scores which are in the format "- description: score"
        scores = {
            m.group(1): float(m.group(2))
            for m in re.finditer(r"- (.+): ([\d.]+)", answer)
        }

        matrix[i, :] = [scores[d] for d in descriptions]

        # with open("gpt4_scores.json", "w") as f:
        #     json.dump(matrix.tolist(), f)

    return matrix


@autocast("cuda", enabled=torch.cuda.is_available())
def main():
    args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = pd.read_csv(args.table_path)
    video_paths = data["path"].to_list()
    video_names = [Path(p).stem for p in video_paths]
    videos = util.get_video_batch(video_paths, device)
    descriptions = data["label"].to_list()

    if args.model.lower() == "gpt4":
        reward_matrix = gpt4(video_paths, descriptions)
        title = f"gpt4_{args.experiment_id}"
        util.make_heatmap(
            reward_matrix,
            groups=data["group"].to_list(),
            trajectories_names=video_names,
            labels=descriptions,
            result_dir=args.output_dir,
            experiment_id=title,
        )
        return

    assert isinstance(args.model, str)
    if args.model.lower() == "viclip":
        encoder = ViCLIP(args.cache_dir)
    elif args.model.lower() == "s3d":
        encoder = S3D(args.cache_dir)
    elif args.model.lower() == "clip":
        if args.n_frames is None:
            raise ValueError("Number of frames must be provided when using CLIP.")
        model = "ViT-bigG-14/laion2b_s39b_b160k"
        model_name_prefix, pretrained = model.split("/")
        encoder = CLIP(
            model_name_prefix,
            pretrained,
            args.cache_dir,
            expected_n_frames=int(args.n_frames),
        )
    encoder = encoder.to(device)

    rewards = []
    for reward_name in args.rewards.split(","):
        if reward_name == "logit":
            rewards.append((logit_reward, f"{args.model}_logit_{args.experiment_id}"))
        elif reward_name == "projection":
            baselines = encoder.encode_text(data["baseline"].to_list())
            if args.alphas is None:
                raise ValueError("Alpha must be provided when using projection reward.")
            for alpha in args.alphas.split(","):
                reward_fun = mk_projection_reward(float(alpha), baselines)
                title = f"{args.model}_projection_{alpha}_{args.experiment_id}"
                rewards.append((reward_fun, title))

    for reward_fun, title in rewards:
        reward_matrix = evaluate(encoder, videos, descriptions, reward_fun)
        if args.standardize:
            reward_matrix = (
                reward_matrix - reward_matrix.mean(dim=0, keepdim=True)
            ) / (reward_matrix.std(dim=0, keepdim=True) + 1e-6)
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
