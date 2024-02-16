from typing import Callable

import gymnasium


def get_clip_rewarded_env_name(env_name: str) -> str:
    return "vlmrm/CLIPRewarded" + env_name


RENDER_DIM = {
    "CartPole-v1": (400, 600),
    "MountainCarContinuous-v0": (400, 600),
    "Humanoid-v4": (480, 480),
    "ObstacleCourse-v0": (400, 600),
}


def get_make_env(
    env_name: str,
    *,
    render_mode: str = "rgb_array",
    **kwargs,
) -> Callable:
    def make_env_wrapper() -> gymnasium.Env:
        env: gymnasium.Env
        env = gymnasium.make(
            env_name,
            render_mode=render_mode,
            **kwargs,
        )
        return env

    return make_env_wrapper


def is_3d_env(env_name: str) -> bool:
    return env_name == "Humanoid-v4"
