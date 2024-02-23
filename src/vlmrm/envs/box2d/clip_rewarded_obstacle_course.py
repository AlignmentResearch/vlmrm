import math
import pathlib
from typing import Dict, Optional, Tuple

import numpy as np
from vlmrm.envs.box2d.obstacle_course import (
    ObstacleCourse,
    VIDEO_W,
    VIDEO_H,

)

# from gymnasium.envs.box2d.car_racing import (
#     CarRacing as ObstacleCourse,
# )
from gymnasium.error import DependencyNotInstalled
from numpy.typing import NDArray


class CLIPRewardedObstacleCourseEnv(ObstacleCourse):
    def __init__(
        self,
        *,
        episode_length: int,
        render_mode: str = "rgb_array",
    ) -> None:
        super().__init__(render_mode)
        self.episode_length = episode_length
        self.num_steps = 0
        self.success = False
        self.background_img = None
        self.car_img = None

        self.screen_width = VIDEO_W
        self.screen_height = VIDEO_H

    def step(self, action) -> Tuple[NDArray, float, bool, bool, Dict]:
        _, reward, _, truncated, info = super().step(action)
        # Update state
        # the state is a 96 x 96 x 3 RGB image

        # x, x_dot = self.state
        # if x >= 0.45:
        #     self.success = True
        # if self.success:
        #     x = np.clip(x, None, 0.45)
        #     self.state = np.stack((x, x_dot))

        self.num_steps += 1
        terminated = self.num_steps >= self.episode_length


        self.success = not terminated

        info["success"] = self.success
        return (
            np.array(self.state, dtype=np.float32), # deliberately broken (should be uint8) as a hack for debugging print output # dtype=np.uint8), # dtype=np.float32),  # obs
            reward,
            terminated,
            truncated,
            info,
        )

    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict] = None):
        self.num_steps = 0
        return super().reset(seed=seed, options=options)

    def render(self):
        assert self.render_mode == "rgb_array"
        try:
            import pygame
        except ImportError as e:
            raise DependencyNotInstalled(
                "pygame is not installed, run `pip install gymnasium[classic-control]`"
            ) from e

        return super().render()

        if self.screen is None:
            pygame.init()
            self.screen = pygame.Surface((self.screen_width, self.screen_height))

        if self.background_img is None or self.car_img is None:
            ctx_folder = pathlib.Path(__file__).parent
            self.background_img = pygame.image.load(
                str(ctx_folder / "obstacle_course_background.png")
            )
            self.background_img = pygame.transform.flip(
                self.background_img, False, True
            )
            self.car_img = pygame.image.load(str(ctx_folder / "sprite.png"))
            self.car_img = pygame.transform.flip(self.car_img, False, True)

        world_width = self.max_position - self.min_position
        scale = self.screen_width / world_width

        self.surf = pygame.Surface((self.screen_width, self.screen_height))
        self.surf.fill((255, 255, 255))
        self.surf.blit(self.background_img, (0, 0))

        pos = self.state[0]

        clearance = 15
        car_img = pygame.transform.rotate(
            self.car_img, -math.degrees(self._rotation_angle(pos))
        )
        car_rect = car_img.get_rect()
        car_rect.center = (
            (pos - self.min_position) * scale,
            clearance + self._height(pos) * scale,
        )
        self.surf.blit(car_img, car_rect)

        self.surf = pygame.transform.flip(self.surf, False, True)
        self.screen.blit(self.surf, (0, 0))

        return np.transpose(
            np.array(pygame.surfarray.pixels3d(self.screen)), axes=(1, 0, 2)
        )

    def _rotation_angle(self, xs):
        return np.arctan(np.cos(3 * xs) * 0.45 * 3)
