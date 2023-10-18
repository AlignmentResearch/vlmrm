from typing import Tuple

import numpy as np
from stable_baselines3.common.vec_env.subproc_vec_env import (
    SubprocVecEnv as StableBaselines3SubprocVecEnv,
)


class SubprocVecEnv(StableBaselines3SubprocVecEnv):
    def __init__(self, *args, render_dim: Tuple[int, int], **kwargs):
        super().__init__(*args, **kwargs)
        self.render_dim = render_dim

    def get_images(self) -> np.ndarray:
        if self.render_mode != "rgb_array":
            raise ValueError("Can only call on render_mode=rgb_array")

        # Send render requests to all subprocesses
        for pipe in self.remotes:
            pipe.send(("render", None))

        renders = np.zeros((self.num_envs, *self.render_dim), dtype=np.uint8)
        completed_indices = set()
        all_indices = set(range(self.num_envs))
        while completed_indices != all_indices:
            for index in all_indices:
                if index not in completed_indices:
                    pipe = self.remotes[index]
                    if pipe.poll():
                        renders[index] = pipe.recv()
                        completed_indices.add(index)
        return renders

    def step_wait(self):
        obs, rew, done, infos = super().step_wait()
        render_array = self.get_images()
        infos[0]["render_array"] = render_array
        return obs, rew, done, infos
