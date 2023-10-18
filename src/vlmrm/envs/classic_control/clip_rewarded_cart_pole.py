import math
from typing import Dict, Optional, Tuple

import numpy as np
from gymnasium.envs.classic_control.cartpole import CartPoleEnv as GymCartPoleEnv
from gymnasium.spaces import Box
from numpy.typing import NDArray


class CLIPRewardedCartPoleEnv(GymCartPoleEnv):
    def __init__(
        self,
        *,
        episode_length: int,
        render_mode: str = "rgb_array",
    ) -> None:
        super().__init__(render_mode)
        high = np.array(
            [
                self.x_threshold * 2,  # Position
                np.finfo(np.float32).max,  # Velocity
                math.pi,  # Angle
                np.finfo(np.float32).max,  # Angular velocity
            ],
            dtype=np.float32,
        )
        self.observation_space = Box(-high, high, dtype=np.float32)
        self.episode_length = episode_length
        self.num_steps = 0

    def step(self, action) -> Tuple[NDArray, float, bool, bool, Dict]:
        _, reward, terminated, truncated, info = super().step(action)
        # Update state
        x, x_dot, theta, theta_dot = self.state  # type: ignore[misc]
        # - Ensure theta in [-pi, pi] (if the angle goes over)
        theta = (theta + np.pi) % (2 * np.pi) - np.pi  # type: ignore[has-type]
        # - Ensure position in [-x_threshold, x_threshold]
        x = np.clip(x, -self.x_threshold, self.x_threshold)  # type: ignore[has-type]
        self.state = np.stack(  # type: ignore[assignment]
            (
                x,
                x_dot,  # type: ignore[has-type]
                theta,
                theta_dot,  # type: ignore[has-type]
            )
        )
        info["success"] = not terminated
        self.num_steps += 1
        terminated = self.num_steps >= self.episode_length
        return (
            np.array(self.state, dtype=np.float32),  # obs
            reward,
            terminated,
            truncated,
            info,
        )

    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict] = None):
        self.num_steps = 0
        return super().reset(seed=seed, options=options)
