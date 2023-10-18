from __future__ import annotations

import os
import signal
import sys
import traceback
import warnings
from typing import Any, Dict, Optional

import torch
import torch.distributed as dist
import yaml
from loguru import logger
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import CallbackList

import vlmrm.contrib.sb3.signal_handler as signal_handler
import wandb
from vlmrm import multiprocessing, util
from vlmrm.contrib.sb3.base import get_clip_rewarded_rl_algorithm_class
from vlmrm.contrib.sb3.callbacks import VideoRecorderCallback, WandbCallback
from vlmrm.contrib.sb3.make_vec_env import make_vec_env
from vlmrm.contrib.sb3.signal_handler import end_signal_handler
from vlmrm.contrib.sb3.subproc_vec_env import SubprocVecEnv
from vlmrm.envs.base import get_clip_rewarded_env_name, get_make_env, is_3d_env
from vlmrm.reward_model import dist_worker_compute_reward, load_reward_model_from_config
from vlmrm.trainer.config import CLIPRewardConfig, Config

signal.signal(signal.SIGINT, end_signal_handler)
signal.signal(signal.SIGTERM, end_signal_handler)


def showwarning_with_traceback(
    message, category, filename, lineno, file=None, line=None
):
    """Show warning with full traceback."""
    log = file if hasattr(file, "write") else sys.stderr
    log.write(warnings.formatwarning(message, category, filename, lineno, line))
    traceback.print_stack(file=log)


# warnings.showwarning = showwarning_with_traceback


def primary_worker(
    config: Config,
    config_dump: Dict[str, Any],
    stop_event: Optional[multiprocessing.Event] = None,
):
    # logger.add(config.log_file, enqueue=True)
    torch.cuda.manual_seed(config.seed)

    make_env_kwargs = (
        dict(
            camera_config=config.reward.camera_config,
            textured=config.reward.textured,
        )
        if is_3d_env(config.env_name)
        else {}
    )
    if config.is_clip_rewarded:
        make_env_kwargs["episode_length"] = config.rl.episode_length
        env_name = get_clip_rewarded_env_name(config.env_name)
    else:
        make_env_kwargs["max_episode_steps"] = config.rl.episode_length
        env_name = config.env_name
    make_env_fn = get_make_env(env_name, **make_env_kwargs)

    logger.info("Creating environment instance")
    vec_env = make_vec_env(
        make_env_fn,
        n_envs=config.rl.n_envs,
        seed=config.seed,
        vec_env_cls=SubprocVecEnv,
        use_gpu_ids=config.rl.device_ids,
        vec_env_kwargs=dict(render_dim=config.render_dim),
    )

    run = wandb.init(
        config=config_dump,
        project="vlmrm",
        tags=config.tags,
        notes=config.description,
        name=config.run_name,
        id=config.run_name,
        sync_tensorboard=True,
        # TODO Add support to resume a run: https://docs.wandb.ai/guides/runs/resuming
    )
    assert run is not None

    wandb.define_metric("global_step")
    global_step_metrics = ["rollout/*", "train/*", "time/*", "trajectory/*"]
    for metric in global_step_metrics:
        wandb.define_metric(metric, step_metric="global_step")

    logger.info("Setting up RL algorithm")
    if config.is_clip_rewarded:
        rl_algorithm_class = get_clip_rewarded_rl_algorithm_class(config.env_name)
        algo = rl_algorithm_class(env=vec_env, config=config)
    else:
        algo = SAC(
            config.rl.policy_name,
            vec_env,
            tensorboard_log=str(config.tb_dir),
            seed=config.seed,
            device="cuda:0",
        )

    signal_handler.model = algo
    signal_handler.checkpoint_dir = str(config.checkpoints_path)

    video_callback = VideoRecorderCallback(
        eval_env=make_env_fn(),
        render_freq=config.logging.video_freq // config.rl.n_envs,
    )
    wandb_callback = WandbCallback(
        model_save_path=str(config.checkpoints_path),
        model_save_freq=config.logging.checkpoint_freq // config.rl.n_envs,
        verbose=2,
    )
    callback = CallbackList([video_callback, wandb_callback])

    logger.info("Training RL algorithm")
    model = algo.learn(
        total_timesteps=config.rl.n_steps,
        callback=callback,
        **(
            dict(log_interval=config.logging.tensorboard_freq // config.rl.n_envs)
            if config.logging.tensorboard_freq
            else dict()
        ),
    )

    if stop_event is not None:
        stop_event.set()

    logger.info("Saving final model")
    model.save(str(config.checkpoints_path / "final_model"))

    logger.info("Done.")
    run.finish()


def train(config: str):
    command = " ".join(sys.argv)
    logger.info(f"Command called: {command}")
    assert torch.cuda.is_available()
    util.set_egl_env_vars()

    config_dict = yaml.load(config, Loader=yaml.FullLoader)
    config_obj = Config(**config_dict)

    # When pickling the object, the __pydantic_serializer__ attribute gets lost.
    # We could consider using alternative pickling libraries.
    # However, it is easier to move up the dumping and saving to the main process.
    # It's worth noting that this isn't identical to the original config string,
    # as it includes properties and other default values.

    logger.info(f"Started run with id {config_obj.run_name}")
    config_obj.run_path.mkdir(parents=True, exist_ok=True)
    config_dump = config_obj.model_dump()
    logger.info("Logging experiment metadata")
    config_obj.save()

    # logger.add(config_obj.log_file, enqueue=True)

    @logger.catch
    def _train():
        if config_obj.is_clip_rewarded:
            logger.info("Running CLIP-rewarded SAC. Spawning workers.")
            args = ("nccl", config_obj, config_dump)
            multiprocessing.spawn(
                fn=init_process,
                args=args,
                nprocs=config_obj.rl.n_workers,
                join=True,
                daemon=False,
                start_method="spawn",
            )
        else:
            logger.info("Running SAC for ground truth.")
            primary_worker(config_obj, config_dump)

    _train()


def init_process(
    rank: int,
    stop_event: multiprocessing.Event,
    /,
    backend: str,
    config: Config,
    config_dump: Dict[str, Any],
) -> None:
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "29500"
    # if backend == "nccl":
    # TODO: come back to this after fixing the kube setup
    # os.environ["NCCL_SHM_DISABLE"] = "1"
    dist.init_process_group(backend, rank=rank, world_size=config.rl.n_workers)
    if rank == 0:
        primary_worker(config, config_dump, stop_event)
    else:
        clip_inference_worker(rank, config, stop_event)


def clip_inference_worker(rank: int, config: Config, stop_event: multiprocessing.Event):
    # logger.add(config.log_file, enqueue=True)
    assert isinstance(config.reward, CLIPRewardConfig)
    assert config.reward.batch_size % config.rl.n_workers == 0
    logger.info(f"[Worker {rank}] Loading CLIP model....")
    reward_model = load_reward_model_from_config(config.reward).eval().cuda(rank)
    worker_frames_tensor = torch.zeros(
        (config.reward.batch_size // config.rl.n_workers, *config.render_dim),
        dtype=torch.uint8,
    ).cuda(rank)
    while not stop_event.is_set():
        logger.info(f"[Worker {rank}] Entering wait for compute_embeddings_dist...")
        dist_worker_compute_reward(
            rank,
            reward_model=reward_model,
            render_dim=config.render_dim,
            batch_size=config.reward.batch_size // config.rl.n_workers,
            num_workers=config.rl.n_workers,
            worker_frames_tensor=worker_frames_tensor,
        )
    logger.info(f"[Worker {rank}] Received stop event. Exiting worker")
