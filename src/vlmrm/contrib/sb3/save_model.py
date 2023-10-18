import logging
import os

from stable_baselines3.common.base_class import BaseAlgorithm

import wandb

logger = logging.getLogger(__name__)

CHECKPOINT_PREFIX = "rl_model"


def save_model(checkpoint_dir: str, model: BaseAlgorithm) -> None:
    model_path = os.path.join(
        checkpoint_dir, f"{CHECKPOINT_PREFIX}_{model.num_timesteps}_steps.zip"
    )
    model.save(model_path)
    wandb.save(model_path, base_path=checkpoint_dir)
    logger.info(f"Saving model checkpoint to {model_path}")


def get_replay_buffer_path(checkpoint_dir: str) -> str:
    return os.path.join(checkpoint_dir, f"{CHECKPOINT_PREFIX}_replay_buffer.pkl")


def save_replay_buffer(checkpoint_dir: str, model: BaseAlgorithm) -> None:
    replay_buffer_path = get_replay_buffer_path(checkpoint_dir)
    model.save_replay_buffer(replay_buffer_path)
    wandb.save(replay_buffer_path, base_path=checkpoint_dir)
    logger.info(f"Saving model replay buffer checkpoint to {replay_buffer_path}")
