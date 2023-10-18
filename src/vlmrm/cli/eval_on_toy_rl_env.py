"""
Generate an unlabelled dataset from a model checkpoint.
"""

import pathlib
import sys
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import torch
import typer
import yaml
from loguru import logger
from pydantic import BaseModel
from vlmrm.contrib.sb3.base import get_clip_rewarded_rl_algorithm_class
from vlmrm.envs.base import get_clip_rewarded_env_name, get_make_env
from vlmrm.util import util


class GymEnvEvaluator:
    def __init__(self, init_success: bool) -> None:
        self.cumulative_reward = 0.0
        self.init_success = init_success
        self.success = init_success

    def step(self, obs, reward, terminated, truncated, info) -> None:
        self.cumulative_reward += reward
        if (self.init_success and self.success) or (
            not self.init_success and not self.success
        ):
            self.success = info["success"]

    def get_eval(self) -> Tuple[bool, float]:
        return self.success, self.cumulative_reward


class CartPoleEvaluator(GymEnvEvaluator):
    def __init__(self, episode_length: int) -> None:
        super().__init__(init_success=True)
        self.episode_length = episode_length

    def get_eval(self) -> Tuple[bool, float]:
        return self.success, self.cumulative_reward / self.episode_length


class MountainCarEvaluator(GymEnvEvaluator):
    def __init__(self) -> None:
        super().__init__(init_success=False)


EVAL_CLASS = {
    "CartPole-v1": CartPoleEvaluator,
    "MountainCarContinuous-v0": MountainCarEvaluator,
}


class EvalOnToyRLEnvConfig(BaseModel):
    env_name: str
    model_checkpoints: Dict[float, str]
    model_base_path: pathlib.Path
    n_rollouts: int
    episode_length: int
    seed: int


def eval_on_toy_rl_env(config_str: str):
    command = " ".join(sys.argv)
    logger.info(f"Command called: {command}")
    assert torch.cuda.is_available()
    util.set_egl_env_vars()

    config_dict = yaml.load(config_str, Loader=yaml.FullLoader)
    config = EvalOnToyRLEnvConfig(**config_dict)

    torch.cuda.manual_seed(config.seed)

    eval_class = EVAL_CLASS[config.env_name]

    logger.info("Evaluating models ...")

    n_alphas = len(config.model_checkpoints)
    models_avg_reward_pc = np.zeros(n_alphas)
    models_avg_accuracy = np.zeros(n_alphas)
    rollouts_reward_pc = np.zeros(config.n_rollouts)
    rollouts_accuracy = np.zeros(config.n_rollouts)

    for alpha_idx, (alpha, checkpoint_name) in enumerate(
        config.model_checkpoints.items()
    ):
        logger.info(f"- Alpha: {alpha}. Checkpoint: {checkpoint_name}.")
        checkpoint_path = (config.model_base_path / checkpoint_name).resolve()
        make_env = get_make_env(
            env_name=get_clip_rewarded_env_name(config.env_name),
            episode_length=config.episode_length,
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
            model_evaluator = eval_class(
                *([config.episode_length] if config.env_name == "CartPole-v1" else [])
            )
            obs = env.reset(seed=config.seed + episode_idx)[0]
            for _ in range(config.episode_length):
                action = algo.predict(obs)[0]
                obs, reward, terminated, truncated, info = env.step(action)
                model_evaluator.step(obs, reward, terminated, truncated, info)
            success, cumulative_reward = model_evaluator.get_eval()
            rollouts_reward_pc[episode_idx] = cumulative_reward
            rollouts_accuracy[episode_idx] = int(success)
        models_avg_reward_pc[alpha_idx] = rollouts_reward_pc.mean()
        models_avg_accuracy[alpha_idx] = rollouts_accuracy.mean()
        env.close()

    env_name = config.env_name[:-3]
    df_results = pd.DataFrame(
        {
            "Alpha": config.model_checkpoints.keys(),
            f"{env_name} (Reward%)": models_avg_reward_pc,
            f"{env_name} (Success Rate)": models_avg_accuracy,
        }
    )
    print(df_results)

    logger.info("... Done.")


if __name__ == "__main__":
    typer.run(eval_on_toy_rl_env)
