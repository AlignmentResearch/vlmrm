from pathlib import Path
from typing import List, Optional, Protocol, Tuple

import open_clip
import torch
import torch.distributed as dist
import torch.nn as nn
from einops import rearrange, reduce
from loguru import logger
from torch.amp.autocast_mode import autocast

from vlmrm.contrib.open_clip.transform import VICLIP_MEAN, VICLIP_STD, image_transform
from vlmrm.contrib.viclip import get_viclip
from vlmrm.trainer.config import CLIPRewardConfig


class BaseModel(Protocol):
    def embed_text(self, x) -> torch.Tensor:
        ...


class CLIP(nn.Module):
    def __init__(self, model_name: str, pretrained: str, cache_dir: str):
        super().__init__()
        self._model: open_clip.model.CLIP = open_clip.create_model(
            model_name=model_name,
            pretrained=pretrained,
            cache_dir=cache_dir,
        )  # type: ignore

    @torch.inference_mode()
    def embed_text(self, x: List[str]) -> torch.Tensor:
        tokens = open_clip.tokenize(x)
        encoded = self._model.encode_text(tokens).float()
        encoded = encoded / encoded.norm(dim=-1, keepdim=True)
        print(f"{encoded.shape=}")
        return encoded

    @torch.inference_mode()
    def embed_image(self, x: torch.Tensor) -> torch.Tensor:
        encoded = self._model.encode_image(x, normalize=True)
        return encoded


class ViCLIP(nn.Module):
    def __init__(self, cache_dir: str, frames_per_video: int) -> None:
        super().__init__()
        model_name = "ViCLIP-L_InternVid-FLT-10M.pth"
        path = Path(cache_dir) / model_name
        self._model, self._tokenizer = get_viclip(
            "l", path.absolute().as_posix(), frames_per_video=frames_per_video
        )

    @torch.inference_mode()
    def embed_text(self, x: List[str]) -> torch.Tensor:
        result = [self._model.get_text_features(t, self._tokenizer) for t in x]
        result = torch.cat(result)
        return result


class Embed(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        ...


class ViCLIPEmbed(Embed):
    _base_model: ViCLIP

    def __init__(self, base_model: ViCLIP) -> None:
        super().__init__()
        self._base_model = base_model
        size = base_model._model.inputs_image_res
        self.transform = image_transform(size, mean=VICLIP_MEAN, std=VICLIP_STD)
        # This is a preset number in the model (8)
        self.expected_n_frames = base_model._model.video_input_num_frames

    @torch.inference_mode()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[3] != 3:
            x = x.permute(0, 1, 2, 5, 3, 4)
        with torch.no_grad(), autocast("cuda", enabled=torch.cuda.is_available()):
            n_frames, n_windows, n_episodes, *_ = x.shape

            assert n_frames >= self.expected_n_frames

            # Take only n_frames frames, evenly spaced
            step = n_frames // self.expected_n_frames
            x = x[::step, ...][: self.expected_n_frames, ...]

            x = rearrange(x, "n_f n_w n_e c h w -> (n_f n_w n_e) c h w")
            x = self.transform(x)
            x = rearrange(
                x,
                "(n_f n_w n_e) c h w -> (n_w n_e) n_f c h w",
                n_f=self.expected_n_frames,
                n_w=n_windows,
                n_e=n_episodes,
            )

            # The episodes are the different samples in a batch
            # The window, i.e. the frames, are the one video
            window_embed = self._base_model._model.get_vid_features(x)
            window_embed = rearrange(
                window_embed, "(n_w n_e) d -> n_w n_e d", n_w=n_windows, n_e=n_episodes
            )

        return window_embed


class AvgCLIPEmbed(Embed):
    _base_model: CLIP

    def __init__(self, base_model: CLIP):
        """Generate embeddings for a batch of image windows
        by averaging the embeddings of all frames in a given chunk.
        """
        super().__init__()
        self._base_model = base_model
        size = base_model._model.visual.image_size
        image_size: int = size if isinstance(size, int) else size[0]  # type: ignore
        self.transform = image_transform(image_size)

    @torch.inference_mode()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[3] != 3:
            x = x.permute(0, 1, 2, 5, 3, 4)
        with torch.no_grad(), autocast("cuda", enabled=torch.cuda.is_available()):
            n_frames, n_windows, n_episodes, *_ = x.shape
            x = rearrange(x, "n_f n_w n_e c h w -> (n_f n_w n_e) c h w")
            # Embed every frame using CLIP
            x = self.transform(x)
            frame_embed = self._base_model._model.encode_image(x, normalize=True)
            # Calculate a per-window embedding by averaging all frame embeddings in the window
            window_embed = reduce(
                frame_embed,
                "(n_f n_w n_e) d -> n_w n_e d",
                reduction="mean",
                n_f=n_frames,
                n_w=n_windows,
                n_e=n_episodes,
            )
        return window_embed


class Reward(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        ...


class ProjectionReward(Reward):
    def __init__(self, baseline, target, direction, projection, alpha):
        super().__init__()
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
    def from_embed(
        target_prompts: list[str],
        baseline_prompts: list[str],
        embed_base: BaseModel,
        alpha: float,
    ) -> "ProjectionReward":
        target = embed_base.embed_text(target_prompts).mean(dim=0, keepdim=True)
        baseline = embed_base.embed_text(baseline_prompts).mean(dim=0, keepdim=True)
        direction = target - baseline
        projection = ProjectionReward._compute_projection(direction, alpha)

        return ProjectionReward(baseline, target, direction, projection, alpha)

    @staticmethod
    def _compute_projection(direction: torch.Tensor, alpha: float) -> torch.Tensor:
        projection = direction.T @ direction / torch.norm(direction) ** 2
        identity = torch.diag(torch.ones(projection.shape[0])).to(projection.device)
        projection = alpha * projection + (1 - alpha) * identity
        return projection


class LogitReward(Reward):
    def __init__(self, baselines, target):
        super().__init__()
        self.register_buffer("options", torch.cat([target, baselines]))

    @torch.inference_mode()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.options = self.options.to(x.device)
        y = (x @ self.options.T).softmax(dim=-1)[:, 0]
        return y

    @staticmethod
    def from_embed(
        target_prompts: list[str],
        baseline_prompts: list[str],
        embed_base: BaseModel,
    ) -> "LogitReward":
        target = embed_base.embed_text(target_prompts).mean(dim=0, keepdim=True)
        baselines = embed_base.embed_text(baseline_prompts)

        return LogitReward(baselines, target)


class RewardModel(nn.Module):
    def __init__(
        self,
        embed: Embed,
        reward: Reward,
        window_size: int,
        window_step: int,
        episode_length: int,
    ) -> None:
        super().__init__()
        self.embed = embed
        self.reward = reward

        self.episode_length = episode_length
        self.window_size = window_size
        self.window_step = window_step

    @staticmethod
    def from_config(config: CLIPRewardConfig, episode_length: int) -> "RewardModel":
        if config.embed_type == "avg_frame":
            # TODO These fields are required by the config although they are not used sometimes
            model_name_prefix, pretrained = config.pretrained_model.split("/")
            base_model = CLIP(model_name_prefix, pretrained, config.cache_dir)
            embed = AvgCLIPEmbed(base_model)
        elif config.embed_type == "viclip":
            assert config.frames_per_video is not None
            base_model = ViCLIP(
                config.cache_dir, frames_per_video=config.frames_per_video
            )
            embed = ViCLIPEmbed(base_model)
        else:
            raise ValueError(f"Unknown embed_type: {config.embed_type}")

        if config.reward_type == "projection":
            reward = ProjectionReward.from_embed(
                target_prompts=config.target_prompts,
                baseline_prompts=config.baseline_prompts,
                embed_base=base_model,
                alpha=config.alpha,
            )
        elif config.reward_type == "logit":
            reward = LogitReward.from_embed(
                target_prompts=config.target_prompts,
                baseline_prompts=config.baseline_prompts,
                embed_base=base_model,
            )
        else:
            raise ValueError(f"Unknown reward_type: {config.reward_type}")

        return RewardModel(
            embed, reward, config.window_size, config.window_step, episode_length
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

        logger.debug(f"{x.shape=}, n_episodes: {n_episodes}, n_windows: {n_windows}")

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
        x = self.embed(x)

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
