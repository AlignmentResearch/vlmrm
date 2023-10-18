"""
Get the model checkpoint with the highest CLIP reward for a set of runs in wandb.
"""

import os
import pathlib
import re
import shutil
import sys
from typing import Any

import typer
import yaml
from loguru import logger
from pydantic import BaseModel, computed_field, model_validator
from tensorboard.backend.event_processing import tag_types
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

import wandb


class GetBestModelConfig(BaseModel):
    wandb_run_ids: list[str]
    base_path: pathlib.Path

    @model_validator(mode="before")
    def configure_properties(cls, data: Any) -> Any:
        assert isinstance(data, dict)
        if "base_path" not in data:
            data["base_path"] = pathlib.Path.cwd() / "runs/model"
        return data

    @computed_field
    @property
    def tmp_path(self) -> pathlib.Path:
        return (self.base_path / "tmp").resolve()


def get_best_model(config_str: str):
    command = " ".join(sys.argv)
    logger.info(f"Command called: {command}")

    config_dict = yaml.load(config_str, Loader=yaml.FullLoader)
    config = GetBestModelConfig(**config_dict)

    config.base_path.mkdir(parents=True, exist_ok=True)
    config.tmp_path.mkdir(parents=True, exist_ok=True)

    api = wandb.Api()

    best_run_id = ""
    best_checkpoint_global_steps = -1
    best_reward = float("-inf")

    logger.info("Analyzing runs ...")

    for run_id in config.wandb_run_ids:
        logger.info(f"- {run_id}")
        run = api.run(f"vlmrm/{run_id}")
        checkpoint_global_steps = []
        event_file = ""
        for file in run.files():
            if re.match(r"^rl_model_[0-9]+_steps.zip", file.name):
                checkpoint_global_steps.append(
                    int(file.name.replace("rl_model_", "").replace("_steps.zip", ""))
                )
            elif re.match(r"^events.out.tfevents.*", file.name):
                event_path = config.tmp_path / file.name
                event_path.mkdir(parents=True, exist_ok=True)
                file.download(root=str(event_path), exist_ok=True)
                event_file = f"{event_path}/{file.name}"

        # TODO Check: It seems EventAccumulator isn't loading all the data
        event_acc = EventAccumulator(event_file, size_guidance={tag_types.SCALARS: 0})
        event_acc.Reload()

        for event_scalar in event_acc.Scalars("rollout/ep_clip_rew_mean"):
            if (
                event_scalar.value > best_reward
                and event_scalar.step in checkpoint_global_steps
            ):
                best_run_id = run_id
                best_checkpoint_global_steps = event_scalar.step
                best_reward = event_scalar.value

    logger.info("... Done")

    best_checkpoint_name = f"rl_model_{best_checkpoint_global_steps}_steps.zip"
    run = api.run(f"vlmrm/{best_run_id}")
    for file in run.files():
        if re.match(best_checkpoint_name, file.name):
            file.download(root=str(config.base_path), exist_ok=True)
            break

    old_name = f"{config.base_path}/{best_checkpoint_name}"
    new_name = f"{config.base_path}/{best_run_id}_{best_checkpoint_name}"
    os.rename(old_name, new_name)

    shutil.rmtree(config.tmp_path)

    logger.info(f"Model {new_name} saved.")


if __name__ == "__main__":
    typer.run(get_best_model)
