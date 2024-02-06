from typing import List, Literal, Optional, Tuple, overload

import open_clip
import torch
import torch.distributed as dist
import torch.nn as nn
from einops import rearrange, reduce, repeat
from torch.amp.autocast_mode import autocast

from vlmrm.contrib.open_clip.transform import image_transform
from vlmrm.trainer.config import CLIPRewardConfig


class Embed(nn.Module):
    def __init__(self, embed_model):
        super().__init__()
        self.embed_model = embed_model

    @torch.inference_mode()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Embed a batch of image chunks.

        Args:
            x (Tensor): Tensor of shape (n_frames, n_chunks, n_episodes, channels, height, width).

        Returns:
            Tensor: Tensor of shape (n_chunks, n_episodes, embedding_dim).
        """
        raise NotImplementedError


class AvgFrameEmbed(Embed):
    embed_model: open_clip.model.CLIP

    def __init__(self, clip_model: open_clip.model.CLIP):
        """Generate embeddings for a batch of image chunks
        by averaging the embeddings of all frames in a given chunk.
        """
        self.embed_model = clip_model
        size = clip_model.visual.image_size
        image_size: int = size if isinstance(size, int) else size[0]  # type: ignore
        self.transform = image_transform(image_size)

    @torch.inference_mode()
    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        if frames.shape[3] != 3:
            frames = frames.permute(0, 1, 2, 5, 3, 4)
        with torch.no_grad(), autocast("cuda", enabled=torch.cuda.is_available()):
            n_frames, n_chunks, n_episodes, *_ = frames.shape
            frames = rearrange(frames, "n_f n_ch n_e c h w -> (n_f n_ch n_e) c h w")
            # Embed every frame using CLIP
            frame_embed = self.embed_model.encode_image(frames, normalize=True)
            # Calculate a per-chunk embedding by averaging all frame embeddings of a chunk
            chunk_embed = reduce(
                frame_embed,
                "(n_f n_ch n_e) d -> n_ch n_e d",
                reduction="mean",
                n_f=n_frames,
                n_ch=n_chunks,
                n_e=n_episodes,
            )
        return chunk_embed


class Reward(nn.Module):
    @torch.inference_mode()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute the reward for a batch of chunks of embeddings.

        Args:
            x (Tensor): Tensor of shape (n_chunks, n_episodes, channels, height, width).

        Returns:
            Tensor: Tensor of shape (n_chunks, n_episodes).
        """
        raise NotImplementedError


class ProjectionReward(Reward):
    def __init__(self, baseline, target, direction, projection, alpha):
        self.register_buffer("target", target)
        self.register_buffer("baseline", baseline)
        self.register_buffer("direction", direction)
        self.register_buffer("projection", projection)
        self.alpha = alpha

    @torch.inference_mode()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x / torch.norm(x, dim=-1, keepdim=True)
        y = 1 - (torch.norm((x - self.target) @ self.projection, dim=-1) ** 2) / 2
        return y

    @staticmethod
    def from_clip(target_prompts, baseline_prompts, clip, alpha):
        # TODO: Make this work with general tokenizers and encoders
        # ...probably by splitting both into a separate class
        target_prompts = ProjectionReward._tokenize_prompts(target_prompts)
        baseline_prompts = ProjectionReward._tokenize_prompts(baseline_prompts)
        target = ProjectionReward._embed_prompts(clip, target_prompts).mean(
            dim=0, keepdim=True
        )
        baseline = ProjectionReward._embed_prompts(clip, baseline_prompts).mean(
            dim=0, keepdim=True
        )
        direction = target - baseline
        projection = ProjectionReward._compute_projection(direction, alpha)

        return ProjectionReward(baseline, target, direction, projection, alpha)

    @staticmethod
    def _compute_projection(direction: torch.Tensor, alpha: float) -> torch.Tensor:
        projection = direction.T @ direction / torch.norm(direction) ** 2
        identity = torch.diag(torch.ones(projection.shape[0])).to(projection.device)
        projection = alpha * projection + (1 - alpha) * identity
        return projection

    @staticmethod
    def _tokenize_prompts(x: List[str]) -> torch.Tensor:
        """Tokenize a list of prompts."""
        return open_clip.tokenize(x)

    @staticmethod
    def _embed_prompts(embed_model, x) -> torch.Tensor:
        """Embed a list of prompts."""
        with torch.no_grad():
            x = embed_model.encode_text(x).float()
        x = x / x.norm(dim=-1, keepdim=True)
        return x


class RewardModel(nn.Module):
    def __init__(
        self,
        embed: Embed,
        reward: Reward,
        window_size: int,
        stride: int,
        episode_length: int,
    ) -> None:
        super().__init__()
        self.embed = embed
        self.reward = reward

        self.episode_length = episode_length
        self.window_size = window_size
        self.stride = stride

    @staticmethod
    def from_config(config: CLIPRewardConfig) -> "RewardModel":
        model_name_prefix, pretrained = config.pretrained_model.split("/")
        model = open_clip.create_model(
            model_name=model_name_prefix,
            pretrained=pretrained,
            cache_dir=config.cache_dir,
        )
        reward = ProjectionReward.from_clip(
            target_prompts=config.target_prompts,
            baseline_prompts=config.baseline_prompts,
            clip=model,
            alpha=config.alpha,
        )
        embed = AvgFrameEmbed(model)

        assert config.stride == config.window_size

        return RewardModel(
            embed, reward, config.window_size, config.stride, config.episode_length
        )

    @torch.inference_mode()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute the reward for a batch of episodes.

        Args:
            x (torch.Tensor): Tensor of shape (batch_size, channels, height, width).

        Returns:
            torch.Tensor: Tensor of shape (batch_size,).
        """
        batch_size = x.shape[0]
        n_episodes = x.shape[0] // self.episode_length
        # TODO: This assumes the chunks are not overlapping
        n_chunks = self.episode_length // self.window_size
        n_frames = self.window_size

        x = rearrange(
            x,
            "(n_f n_ch n_e) ... -> n_f n_ch n_e ...",
            n_f=n_frames,
            n_ch=n_chunks,
            n_e=n_episodes,
        )

        x = self.embed(x)
        # We get a tensor of (n_chunks, n_episodes) rewards
        chunk_rewards = self.reward(x)

        # We want to return a tensor of (n_frames * n_chunks * n_episodes) rewards
        rewards = torch.zeros(batch_size, device=x.device)

        # TODO: Fix the following
        # Create an index tensor
        indices = torch.arange(n_chunks * n_frames, device=x.device).view(-1, n_frames)[
            :, -1
        ]
        indices = (
            indices.repeat(n_episodes, 1).view(-1)
            + torch.arange(n_episodes, device=x.device).view(-1, 1)
            * n_chunks
            * n_frames
        )
        # Assign the chunk rewards to the frames corresponding to the last frame of each chunk
        rewards[indices] = chunk_rewards.t().contiguous().view(-1)

        return x


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
        batch_size (int): frames will be split into batch_size sized chunks
        num_workers (int): each batch will be split into num_workers chunks
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
            render_dim = tuple(frames_batch.shape[1:])
            assert len(render_dim) == 3
            rewards_batch = dist_worker_compute_reward(
                rank=0,
                reward_model=model,
                render_dim=render_dim,
                batch_size=batch_size // num_workers,
                num_workers=num_workers,
                frames=frames_batch,
                worker_frames_tensor=worker_frames_tensor,
            )
            assert rewards_batch is not None
            rewards_batch = rewards_batch.cpu()
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
        rewards = reward_model(worker_frames)

    def zero_t():
        return torch.zeros_like(rewards)

    # TODO: Check wheter this should be None or []
    recv_rewards = [zero_t() for _ in range(num_workers)] if rank == 0 else None
    dist.gather(rewards, gather_list=recv_rewards, dst=0)

    if rank == 0:
        assert recv_rewards is not None
        return torch.cat(recv_rewards, dim=0).cuda(rank)
