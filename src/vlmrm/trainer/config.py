from __future__ import annotations

import json
import pathlib
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

import open_clip
import torch
from open_clip.factory import get_model_config as get_clip_model_config
from pydantic import BaseModel, computed_field, field_validator, model_validator

from vlmrm import util
from vlmrm.envs.base import RENDER_DIM


class Config(BaseModel):
    env_name: Literal["CartPole-v1", "Humanoid-v4", "MountainCarContinuous-v0", "ObstacleCourse-v0"]
    base_path: pathlib.Path
    seed: int
    description: str
    tags: List[str]
    reward: Union[GroundTruthConfig, CLIPRewardConfig]
    rl: RLConfig
    logging: LoggingConfig

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
        return self.run_path / "run_config.json"

    @computed_field
    @property
    def checkpoints_path(self) -> pathlib.Path:
        return self.run_path / "checkpoints"

    @computed_field
    @property
    def render_dim(self) -> Tuple[int, int]:
        return (*RENDER_DIM[self.env_name], 3)

    @computed_field
    @property
    def is_clip_rewarded(self) -> bool:
        return isinstance(self.reward, CLIPRewardConfig)

    @computed_field
    @property
    def run_name(self) -> str:
        reward_str = "CLIP" if self.is_clip_rewarded else "GT"
        return f"{self.env_name[:-3]}_{reward_str}_{self.run_hash}"

    @computed_field
    @property
    def run_path(self) -> pathlib.Path:
        return (self.base_path / self.run_name).resolve()

    @computed_field
    @property
    def log_file(self) -> pathlib.Path:
        return self.run_path / "info.log"

    @computed_field
    @property
    def tb_dir(self) -> pathlib.Path:
        return self.run_path / "tensorboard"

    @model_validator(mode="before")
    def configure_properties(cls, data: Any) -> Any:
        assert isinstance(data, dict)
        data["run_hash"] = util.get_run_hash()
        data["commit_hash"] = util.get_git_commit_hash()
        if "base_path" not in data:
            data["base_path"] = pathlib.Path.cwd() / "runs/training"
        return data

    @model_validator(mode="after")
    def check_model(self) -> "Config":
        if self.logging.checkpoint_freq % self.rl.train_freq != 0:
            raise ValueError(
                f"({self.logging.checkpoint_freq=}) must be divisible by "
                f"({self.rl.train_freq=}). Otherwise duplicated checkpoints "
                "are created."
            )
        if self.logging.video_freq % self.rl.n_envs != 0:
            raise ValueError(
                f"({self.logging.video_freq=}) must be divisible by "
                f"({self.rl.n_envs=})"
            )
        if self.logging.checkpoint_freq % self.rl.n_envs != 0:
            raise ValueError(
                f"({self.logging.checkpoint_freq=}) must be divisible by "
                f"({self.rl.n_envs=})"
            )

        if self.is_clip_rewarded:
            assert isinstance(self.reward, CLIPRewardConfig)
            if self.logging.tensorboard_freq is not None:
                raise ValueError(
                    "When doing CLIP-rewarded training, a tensorboard logging "
                    "frequency does not need to be specified."
                )
            if len(self.reward.target_prompts) != len(self.reward.baseline_prompts):
                raise ValueError(
                    f"({self.reward.target_prompts=}) and "
                    f"({self.reward.baseline_prompts=}) must have the same length."
                )

            if len(self.reward.target_prompts) == 0:
                raise ValueError(f"({self.reward.target_prompts=}) must not be empty.")
            if self.rl.train_freq % self.rl.episode_length != 0:
                raise ValueError(
                    f"({self.rl.train_freq=}) must be divisible by "
                    f"({self.rl.episode_length=}), so that training happens after "
                    "full episodes are completed."
                )
            if self.reward.batch_size % self.rl.n_workers != 0:
                raise ValueError(
                    f"({self.reward.batch_size=}) corresponds to the total size of the "
                    " batch do be distributed among workers and therefore must be "
                    f"divisible by ({self.rl.n_workers=})"
                )
            if self.rl.n_envs * self.rl.episode_length % self.reward.batch_size != 0:
                raise ValueError(
                    f"({self.rl.n_envs=}) * ({self.rl.episode_length=}) must be "
                    f"divisible by ({self.reward.batch_size=}) so that all batches"
                    "are of the same size."
                )
        else:
            if self.logging.tensorboard_freq is None:
                raise ValueError(
                    "You must specify a tensorboard logging frequency when"
                    " training on ground-truth rewards."
                )
        return self


class GroundTruthConfig(BaseModel):
    name: Literal["ground_truth"]
    camera_config: Optional[Dict[str, Any]] = None


class CLIPRewardConfig(BaseModel):
    name: Literal["clip"]
    pretrained_model: str
    batch_size: int
    alpha: float
    target_prompts: List[str]
    baseline_prompts: List[str]
    cache_dir: str
    camera_config: Optional[Dict[str, Any]] = None
    textured: bool = True

    @computed_field
    @property
    def pretrained_config_dict(self) -> Dict[str, Any]:
        if not hasattr(self, "_pretrained_config_dict"):
            self._pretrained_config_dict = self.get_pretrained_config_dict()
        return self._pretrained_config_dict

    def get_pretrained_config_dict(self):
        return get_clip_model_config(self.pretrained_model.split("/")[0])

    @computed_field
    @property
    def embed_dim(self) -> int:
        return self.pretrained_config_dict["embed_dim"]

    @field_validator("pretrained_model")
    def pretrained_model_must_be_valid(cls, v: str) -> str:
        try:
            name, tag = v.split("/")
        except ValueError:
            raise ValueError(
                f"({v=}) is not a valid model name. "
                "It must be in the form of `name/tag`."
            )
        tags = open_clip.list_pretrained_tags_by_model(name)
        if tag not in tags:
            raise ValueError(
                f"({v=}) is not a valid model name. "
                f"Available tags for {name} are {tags}."
            )
        return v


class RLConfig(BaseModel):
    policy_name: str = "MlpPolicy"
    n_steps: int
    n_envs_per_worker: int
    episode_length: int
    learning_starts: int
    train_freq: int
    batch_size: int
    gradient_steps: int
    action_noise: Optional[
        Union[NormalActionNoiseConfig, OrnsteinUhlenbeckActionNoiseConfig]
    ] = None
    tau: float = 0.005
    gamma: float = 0.99
    learning_rate: float = 3e-4
    buffer_size: int = 1_000_000
    ent_coef: Union[str, float] = "auto"
    use_sde: bool = False
    target_update_interval: int = 1
    policy_kwargs: Optional[Dict[str, Any]] = None
    rl_kwargs: Optional[Dict[str, Any]] = None

    @computed_field
    @property
    def n_workers(self) -> int:
        n_workers = torch.cuda.device_count()
        if n_workers == 0:
            raise RuntimeError("No CUDA device is found.")
        return n_workers

    @computed_field
    @property
    def device_ids(self) -> List[int]:
        return list(range(self.n_workers))

    @computed_field
    @property
    def n_envs(self) -> int:
        return self.n_workers * self.n_envs_per_worker

    @model_validator(mode="after")
    def check_model(self) -> "RLConfig":
        if self.train_freq > self.n_steps:
            raise ValueError(
                f"({self.train_freq=}) cannot be greater than "
                f"({self.n_steps=}), or no training would be performed."
            )
        return self


class LoggingConfig(BaseModel):
    checkpoint_freq: int
    video_freq: int
    tensorboard_freq: Optional[int] = None


class OrnsteinUhlenbeckActionNoiseConfig(BaseModel):
    name: Literal["OrnsteinUhlenbeckActionNoise"]
    mean: float
    sigma: float
    theta: float
    dt: float


class NormalActionNoiseConfig(BaseModel):
    name: Literal["NormalActionNoise"]
    mean: float
    sigma: float
