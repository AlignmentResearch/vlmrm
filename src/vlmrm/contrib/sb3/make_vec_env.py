import itertools
import os
import sys
from typing import Any, Callable, Dict, List, Optional, Type, Union

import gymnasium
import torch
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecEnv

from vlmrm import util


def make_vec_env(
    make_env_fn: Callable[..., gymnasium.Env],
    *,
    vec_env_cls: Type[Union[DummyVecEnv, SubprocVecEnv]],
    n_envs: int,
    seed: Optional[int] = None,
    start_index: int = 0,
    monitor_dir: Optional[str] = None,
    env_kwargs: Optional[Dict[str, Any]] = None,
    vec_env_kwargs: Optional[Dict[str, Any]] = None,
    monitor_kwargs: Optional[Dict[str, Any]] = None,
    use_gpu_ids: Optional[List[int]] = None,
    verbose=False,
) -> VecEnv:
    """
    Create a wrapped, monitored ``VecEnv``.
    By default it uses a ``DummyVecEnv`` which is usually faster
    than a ``SubprocVecEnv``.

    :param env_id: either the env ID, the env class or a callable returning an env
    :param n_envs: the number of environments you wish to have in parallel
    :param seed: the initial seed for the random number generator
    :param start_index: start rank index
    :param monitor_dir: Path to a folder where the monitor files will be saved.
        If None, no file will be written, however, the env will still be wrapped
        in a Monitor wrapper to provide additional information about training.
    :param env_kwargs: Optional keyword argument to pass to the env constructor
    :param vec_env_cls: A custom ``VecEnv`` class constructor. Default: None.
    :param vec_env_kwargs: Keyword arguments to pass to the ``VecEnv`` class
     constructor.
    :param monitor_kwargs: Keyword arguments to pass to the ``Monitor`` class
     constructor.
    :param wrapper_kwargs: Keyword arguments to pass to the ``Wrapper`` class
     constructor.
    :return: The wrapped environment
    """
    env_kwargs = env_kwargs or {}
    vec_env_kwargs = vec_env_kwargs or {}
    monitor_kwargs = monitor_kwargs or {}
    use_gpu_ids = use_gpu_ids or [i for i in range(torch.cuda.device_count())]
    gpu_ids_for_envs = list(itertools.islice(itertools.cycle(use_gpu_ids), n_envs))
    if verbose:
        print(f"gpu_ids_for_envs={gpu_ids_for_envs}")
    assert vec_env_kwargs is not None  # for mypy

    def get_make_env_fn(rank: int, gpu_id: int) -> Callable[[], gymnasium.Env]:
        def _init() -> gymnasium.Env:
            # For type checker:
            assert monitor_kwargs is not None
            assert env_kwargs is not None
            util.set_egl_env_vars()
            os.environ["MUJOCO_EGL_DEVICE_ID"] = str(gpu_id)
            env = make_env_fn(**env_kwargs)

            if seed is not None:
                # Note: here we only seed the action space
                # We will seed the env at the next reset
                env.action_space.seed(seed + rank)
            # Wrap the env in a Monitor wrapper
            # to have additional training information
            monitor_path = (
                os.path.join(monitor_dir, str(rank))
                if monitor_dir is not None
                else None
            )
            # Create the monitor folder if needed
            if monitor_path is not None and monitor_dir is not None:
                os.makedirs(monitor_dir, exist_ok=True)
            env = Monitor(env, filename=monitor_path, **monitor_kwargs)
            return env

        return _init

    make_env_fns = [
        get_make_env_fn(i + start_index, gpu_id=gpu_id)
        for i, gpu_id in enumerate(gpu_ids_for_envs)
    ]

    if issubclass(vec_env_cls, SubprocVecEnv) and "mujoco" in sys.modules:
        raise RuntimeError(
            "Do NOT import Mujoco or any module that imports mujoco"
            "before calling make_vec_env when attempting to parallelize"
            "over GPUs, since this will load the EGL context and set the"
            "same device to all the subprocesses."
        )

    vec_env = vec_env_cls(make_env_fns, **vec_env_kwargs)
    # Prepare the seeds for the first reset
    vec_env.seed(seed)
    return vec_env
