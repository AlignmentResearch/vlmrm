from typing import Optional, Tuple

import torch
import torch.distributed as dist
import torch.nn as nn
from einops import rearrange
from vlmrm.reward.encoders import CLIP, ViCLIP, VideoEncoder
from vlmrm.reward.rewards import LogitReward, ProjectionReward, Reward
from vlmrm.trainer.config import CLIPRewardConfig


class RewardModel(nn.Module):
    def __init__(
        self,
        encoder: VideoEncoder,
        reward: Reward,
        window_size: int,
        window_step: int,
        episode_length: int,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.reward = reward

        self.episode_length = episode_length
        self.window_size = window_size
        self.window_step = window_step

    @staticmethod
    def from_config(config: CLIPRewardConfig, episode_length: int) -> "RewardModel":
        if config.embed_type == "avg_frame":
            # TODO These fields are required by the config although they are not used sometimes
            model_name_prefix, pretrained = config.pretrained_model.split("/")
            encoder = CLIP(model_name_prefix, pretrained, config.cache_dir)

        elif config.embed_type == "viclip":
            assert config.frames_per_video is not None
            encoder = ViCLIP(config.cache_dir, frames_per_video=config.frames_per_video)

        else:
            raise ValueError(f"Unknown embed_type: {config.embed_type}")

        if config.reward_type == "projection":
            reward = ProjectionReward.from_model(
                target_prompts=config.target_prompts,
                baseline_prompts=config.baseline_prompts,
                model=encoder,
                alpha=config.alpha,
            )
        elif config.reward_type == "logit":
            reward = LogitReward.from_model(
                target_prompts=config.target_prompts,
                baseline_prompts=config.baseline_prompts,
                model=encoder,
            )
        else:
            raise ValueError(f"Unknown reward_type: {config.reward_type}")

        return RewardModel(
            encoder, reward, config.window_size, config.window_step, episode_length
        )

    @torch.inference_mode()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute the reward for a flat batch of frames.

        Args:
            x (Tensor): Tensor of shape (batch_size, channels, height, width) or (batch_size, height, width, channels).

        Returns:
            Tensor: Tensor of shape (batch_size,).
        """
        batch_size = x.shape[0]
        n_episodes = x.shape[0] // self.episode_length
        n_windows = 1 + (self.episode_length - self.window_size) // self.window_step

        if x.shape[1] != 3:
            x = rearrange(x, "b h w c -> b c h w")

        # Un-flatten the batch into episodes
        x = rearrange(
            x,
            "(n_steps n_episodes) ... -> n_steps n_episodes ...",
            n_steps=self.episode_length,
            n_episodes=n_episodes,
        )

        # Unfold each episode into (potentially overlapping) windows, each containing window_size frames
        # -> (n_windows, n_episodes, c, h, w, window_size)
        x = x.unfold(0, size=self.window_size, step=self.window_step)

        # Rearrange the dimensions to match the expected input shape of the embed model
        x = rearrange(
            x,
            "n_windows n_episodes ... n_frames -> n_frames n_windows n_episodes ...",
            n_frames=self.window_size,
            n_windows=n_windows,
            n_episodes=n_episodes,
        )

        # Embed the windows
        # (n_frames, n_windows, n_episodes, c, h, w) -> (n_windows, n_episodes, embedding_dim)
        x = self.encoder.encode_video(x)

        # Compute the reward for each window
        # (n_windows, n_episodes, embedding_dim) -> (n_windows, n_episodes)
        x = rearrange(x, "n_w n_e d -> (n_w n_e) d", n_w=n_windows, n_e=n_episodes)
        window_rewards = self.reward(x)

        rewards = torch.zeros(batch_size, device=x.device)

        # Calculate the end indices for each window
        indices = (
            torch.arange(n_windows * n_episodes, device=x.device) * self.window_step
            + self.window_size
            - 1
        )

        assert len(indices) == len(window_rewards)

        # Assign window rewards to the last frame of each window
        rewards[indices] = window_rewards

        return rewards


def compute_rewards(
    model: RewardModel,
    frames: torch.Tensor,
    batch_size: int,
    num_workers: int,
    worker_frames_tensor: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Compute rewards for all frames using the provided reward model.
    Handles splitting into batches and distributing each batch across multiple workers.

    Args:
        model (CLIPReward): reward model
        frames (torch.Tensor): frames to compute rewards for
        batch_size (int): frames will be split into batch_size sized windows
        num_workers (int): each batch will be split into num_workers windows
        worker_frames_tensor (Optional[torch.Tensor], optional): no idea what these do, maybe for logging?. Defaults to None.

    Returns:
        torch.Tensor: _description_
    """
    assert frames.device == torch.device("cpu")
    assert batch_size % num_workers == 0
    n_samples = len(frames)
    rewards = torch.zeros(n_samples, device=torch.device("cpu"))
    model = model.eval()
    with torch.no_grad():
        for i in range(0, n_samples, batch_size):
            frames_batch = frames[i : i + batch_size]
            rewards_batch = dist_worker_compute_reward(
                rank=0,
                reward_model=model,
                render_dim=frames_batch.shape[1:],  # type: ignore
                batch_size=batch_size // num_workers,
                num_workers=num_workers,
                frames=frames_batch,
                worker_frames_tensor=worker_frames_tensor,
            )
            rewards_batch = rewards_batch.cpu()  # type: ignore
            rewards[i : i + batch_size] = rewards_batch
    return rewards


def dist_worker_compute_reward(
    rank: int,
    reward_model: RewardModel,
    render_dim: Tuple[int, int, int],
    batch_size: int,
    num_workers: int,
    frames: Optional[torch.Tensor] = None,
    worker_frames_tensor: Optional[torch.Tensor] = None,
) -> Optional[torch.Tensor]:
    """Compute rewards for a batch of frames using the provided reward model in parallel.

    Args:
        rank (int): the identifier of the current process
        batch_size (int): the batch size here describes the number of frames one process gets

    Returns:
        Optional[torch.Tensor]: the computed rewards, only returned by the master process (rank 0)
    """
    if rank == 0:
        if frames is None:
            raise ValueError("Must pass render result on rank=0")
        if len(frames) != num_workers * batch_size:
            raise ValueError("Must pass render result with correct batch size")
        scatter_list = [t.cuda(rank) for t in torch.chunk(frames, num_workers, dim=0)]
    else:
        # TODO: Check wheter this should be None or []
        scatter_list = None

    worker_frames = (
        worker_frames_tensor
        if worker_frames_tensor is not None
        else torch.zeros((batch_size, *render_dim), dtype=torch.uint8).cuda(rank)
    )
    dist.scatter(worker_frames, scatter_list=scatter_list, src=0)

    with torch.no_grad():
        # worker_frames :: (batch_size, channels, height, width)
        # rewards :: (batch_size,)
        rewards = reward_model(worker_frames)

    def zero_t():
        return torch.zeros_like(rewards)

    # TODO: Check wheter this should be None or []
    recv_rewards = [zero_t() for _ in range(num_workers)] if rank == 0 else None
    dist.gather(rewards, gather_list=recv_rewards, dst=0)

    if rank == 0:
        assert recv_rewards is not None
        return torch.cat(recv_rewards, dim=0).cuda(rank)
