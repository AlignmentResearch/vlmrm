import multiprocessing
import time

import typer
from numpy import array
from stable_baselines3.common.vec_env import DummyVecEnv
import gymnasium

from vlmrm.contrib.sb3.make_vec_env import make_vec_env
from vlmrm.contrib.sb3.subproc_vec_env import SubprocVecEnv
from vlmrm import envs

def main(n_envs: int = 0):
    if n_envs == 0:
        n_envs = multiprocessing.cpu_count()

    def make_env():
        env = gymnasium.make(
            "vlmrm/CLIPRewardedHumanoid-v4", 
            episode_length=200,
            render_mode="rgb_array"
        )
        return env
         

    venv = make_vec_env(
        make_env,
        vec_env_cls=SubprocVecEnv if n_envs > 1 else DummyVecEnv,
        vec_env_kwargs=dict(render_dim=(480, 480)),
        n_envs=n_envs,
        verbose=True,
    )
    print(f"Benchmarking FPS with {n_envs} environments. {venv}")

    venv.reset()

    base_time = time.time()
    frames = 0

    while True:
        venv.step(array([venv.action_space.sample() for _ in range(n_envs)]))
        image_array = venv.get_images()
        frames += 1
        if frames % 10 == 0:
            print(
                f"fps: {frames * n_envs / (time.time() - base_time)}. Total frames "
                f"{frames * n_envs}. Type {type(image_array)}"
            )


if __name__ == "__main__":
    typer.run(main)
