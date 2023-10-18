"""
Generate an unlabelled dataset from a model checkpoint.
"""

import csv
import json
import pathlib
import sys
from typing import Any, Dict

import imageio
import numpy as np
import torch
import typer
import yaml
from loguru import logger
from PIL import Image
from pydantic import BaseModel, computed_field, model_validator

from vlmrm.contrib.sb3.base import get_clip_rewarded_rl_algorithm_class
from vlmrm.envs.base import get_clip_rewarded_env_name, get_make_env
from vlmrm.util import util


class GenerateDatasetConfig(BaseModel):
    env_name: str
    model_checkpoint: str
    model_base_path: pathlib.Path
    base_path: pathlib.Path
    camera_config: Dict
    n_rollouts: int
    episode_length: int
    seed: int

    # Auto-injected properties
    run_hash: str
    commit_hash: str

    def save(self) -> None:
        with open(self.dump_path, "w") as f:
            json.dump(
                self.model_dump(), f, indent=2, cls=util.PathlibCompatibleJSONEncoder
            )

    @computed_field
    @property
    def dump_path(self) -> pathlib.Path:
        return (self.run_path / "run_config.json").resolve()

    @model_validator(mode="before")
    def configure_properties(cls, data: Any) -> Any:
        assert isinstance(data, dict)
        data["run_hash"] = util.get_run_hash()
        data["commit_hash"] = util.get_git_commit_hash()
        if "base_path" not in data:
            data["base_path"] = pathlib.Path.cwd() / "runs/dataset"
        return data

    @computed_field
    @property
    def run_path(self) -> pathlib.Path:
        return (self.base_path / self.run_hash).resolve()

    @computed_field
    @property
    def img_base_path(self) -> pathlib.Path:
        return (self.run_path / "img").resolve()

    @computed_field
    @property
    def video_base_path(self) -> pathlib.Path:
        return (self.run_path / "video").resolve()


def generate_dataset(config_str: str):
    command = " ".join(sys.argv)
    logger.info(f"Command called: {command}")
    assert torch.cuda.is_available()
    util.set_egl_env_vars()

    config_dict = yaml.load(config_str, Loader=yaml.FullLoader)
    config = GenerateDatasetConfig(**config_dict)

    config.run_path.mkdir(parents=True, exist_ok=True)
    logger.info("Logging experiment metadata")
    config.save()

    checkpoint_name = config.model_checkpoint[:-4]
    checkpoint_path = (config.model_base_path / config.model_checkpoint).resolve()
    logger.info(f"Checkpoint: {config.model_checkpoint}.")
    checkpoint_prefix = checkpoint_name.replace("_steps", "")

    config.img_base_path.mkdir(parents=True, exist_ok=True)
    config.video_base_path.mkdir(parents=True, exist_ok=True)

    torch.cuda.manual_seed(config.seed)

    csv_file_name = f"{config.run_path}/unlabelled_data.csv"

    with open(csv_file_name, mode="w", newline="") as csv_file:
        csv_writer = csv.writer(csv_file)
        header = [
            "checkpoint",
            "rollout",
            "step",
            "image_file_name",
        ]

        logger.info(f"Generating dataset in {config.run_path} ...")

        make_env = get_make_env(
            env_name=get_clip_rewarded_env_name(config.env_name),
            episode_length=config.episode_length,
            camera_config=config.camera_config,
        )
        env = make_env()
        rl_algorithm_class = get_clip_rewarded_rl_algorithm_class(config.env_name)
        algo = rl_algorithm_class.load(
            path=str(checkpoint_path),
            env=env,
            load_clip=False,
            device="cuda:0",
        )
        for episode_idx in range(config.n_rollouts):
            video_file_name = f"{checkpoint_prefix}_{episode_idx}.mp4"
            video_path = (config.video_base_path / video_file_name).resolve()
            video_writer = imageio.get_writer(video_path, fps=30)
            obs = env.reset(seed=config.seed + episode_idx)[0]
            for step_idx in range(config.episode_length):
                action = algo.predict(obs)[0]
                obs, _, _, _, _ = env.step(action)
                image = env.render()
                image = np.uint8(image)
                pil_image = Image.fromarray(image)
                image_file_name = f"{checkpoint_prefix}_{episode_idx}_{step_idx}.png"
                image_path = str((config.img_base_path / image_file_name).resolve())
                pil_image.save(image_path)
                if header:
                    header += [f"obs_{obs_idx}" for obs_idx in range(obs.shape[0])]
                    csv_writer.writerow(header)
                    header = None
                csv_writer.writerow(
                    [
                        checkpoint_name,
                        episode_idx,
                        step_idx,
                        image_file_name,
                    ]
                    + obs.tolist()  # + [
                    #     "gt_reward",  # From Gymnasium env
                    #     "clip_reward",
                    # ]
                )
                video_writer.append_data(image)
            video_writer.close()
            logger.info(f"Rollout {episode_idx+1} saved at {video_file_name}.")

        env.close()

    logger.info("... Done.")


if __name__ == "__main__":
    typer.run(generate_dataset)
