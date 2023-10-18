import logging
import os
from typing import Any, Dict, Optional

import gymnasium
import torch as th
from numpy import array
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.callbacks import (
    CheckpointCallback as SB3CheckpointCallback,
)
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.logger import Video
from wandb.integration.sb3 import WandbCallback as SB3WandbCallback

from vlmrm.contrib.sb3.save_model import save_model, save_replay_buffer

logger = logging.getLogger(__name__)


class CheckpointCallback(SB3CheckpointCallback):
    def __init__(
        self,
        save_freq: int,
        save_path: str,
        save_replay_buffer: bool,
        save_vecnormalize: bool,
    ):
        super().__init__(
            save_freq,
            save_path,
            save_replay_buffer=save_replay_buffer,
            save_vecnormalize=save_vecnormalize,
        )
        self.current_replay_buffer_path: Optional[str] = None

    def _on_step(self) -> bool:
        has_replay_buffer = (
            self.save_replay_buffer
            and hasattr(self.model, "replay_buffer")
            and self.model.replay_buffer is not None
        )
        previous_replay_buffer_path: Optional[str] = None
        if has_replay_buffer and self.n_calls % self.save_freq == 0:
            previous_replay_buffer_path = self.current_replay_buffer_path
            self.current_replay_buffer_path = self._checkpoint_path(
                "replay_buffer_", extension="pkl"
            )
        ret = super()._on_step()
        if (
            has_replay_buffer
            and previous_replay_buffer_path
            and self.n_calls % self.save_freq == 0
        ):
            if self.verbose >= 2:
                print(
                    "Removing previous replay buffer checkpoint at"
                    f"{previous_replay_buffer_path}"
                )
            os.remove(previous_replay_buffer_path)
        return ret


class VideoRecorderCallback(BaseCallback):
    def __init__(
        self,
        eval_env: gymnasium.Env,
        render_freq: int,
        n_eval_episodes: int = 1,
        deterministic: bool = True,
    ):
        """
        Records a video of an agent's trajectory traversing ``eval_env`` and logs it to
        TensorBoard

        :param eval_env: A gym environment from which the trajectory is recorded
        :param render_freq: Render the agent's trajectory every eval_freq call of the
         callback.
        :param n_eval_episodes: Number of episodes to render
        :param deterministic: Whether to use deterministic or stochastic policy
        """
        super().__init__()
        self._eval_env = eval_env
        self._render_freq = render_freq
        self._n_eval_episodes = n_eval_episodes
        self._deterministic = deterministic

    def _on_step(self) -> bool:
        if self.n_calls % self._render_freq == 0:
            screens = []

            def grab_screens(_locals: Dict[str, Any], _globals: Dict[str, Any]) -> None:
                """
                Renders the environment in its current state, recording the screen in
                the captured `screens` list

                :param _locals: A dictionary containing all local variables of the
                 callback's scope
                :param _globals: A dictionary containing all global variables of the
                 callback's scope
                """
                screen = self._eval_env.render()
                # PyTorch uses CxHxW vs HxWxC gym (and tensorflow) image convention
                screens.append(screen.transpose(2, 0, 1))

            evaluate_policy(
                self.model,
                self._eval_env,
                callback=grab_screens,
                n_eval_episodes=self._n_eval_episodes,
                deterministic=self._deterministic,
            )
            self.logger.record(
                "trajectory/video",
                Video(th.ByteTensor(array([screens])), fps=40),
                exclude=("stdout", "log", "json", "csv"),
            )
        return True


class WandbCallback(SB3WandbCallback):
    def __init__(
        self,
        model_save_path: str,
        model_save_freq: int,
        **kwargs,
    ):
        super().__init__(
            model_save_path=model_save_path,
            model_save_freq=model_save_freq,
            **kwargs,
        )

    def _on_training_end(self) -> None:
        super()._on_training_end()
        if (
            hasattr(self.model, "replay_buffer")
            and self.model.replay_buffer is not None
        ):
            save_replay_buffer(self.model_save_path, self.model)

    def save_model(self) -> None:
        save_model(self.model_save_path, self.model)
