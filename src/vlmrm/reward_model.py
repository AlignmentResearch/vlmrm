from typing import List, Optional, Tuple, overload

import open_clip
import torch
import torch.distributed as dist
import torch.nn as nn

from vlmrm.contrib.open_clip.transform import image_transform
from vlmrm.trainer.config import CLIPRewardConfig


class CLIPEmbed(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.clip_model = clip_model
        if isinstance(clip_model.visual.image_size, int):
            image_size = clip_model.visual.image_size
        else:
            image_size = clip_model.visual.image_size[0]
        self.transform = image_transform(image_size)

    @torch.inference_mode()
    def forward(self, x):
        if x.shape[1] != 3:
            x = x.permute(0, 3, 1, 2)

        with torch.no_grad(), torch.autocast("cuda", enabled=torch.cuda.is_available()):
            x = self.transform(x)
            x = self.clip_model.encode_image(x, normalize=True)
        return x


class CLIPReward(nn.Module):
    def __init__(
        self,
        *,
        model: CLIPEmbed,
        alpha: float,
        target_prompts: torch.Tensor,
        baseline_prompts: torch.Tensor,
    ) -> None:
        """CLIP Reward function that modifies the CLIP vector space by
        projecting all vectors onto the line spanned by the prompt and
        a baseline prompt. The alpha parameter controls the degree of
        projection. A value of 0.0 means that the reward function is
        equivalent to the CLIP reward function. A value of 1.0 means
        that the vector space is completely projected onto the line
        and becomes a 1D space. Any value in between is a linear
        interpolation between the two.

        Args:
            model (str): CLIP model.
            device (str): Device to use.
            alpha (float, optional): Coeefficient of projection.
            target_prompts (torch.Tensor): Tokenized prompts describing
                the target state.
            baseline_prompts (torch.Tensor): Tokenized prompts describing
                the baseline state.
        """
        super().__init__()
        self.embed_module = model
        target = self.embed_prompts(target_prompts).mean(dim=0, keepdim=True)
        baseline = self.embed_prompts(baseline_prompts).mean(dim=0, keepdim=True)
        direction = target - baseline
        # Register them as buffers so they are automatically moved around.
        self.register_buffer("target", target)
        self.register_buffer("baseline", baseline)
        self.register_buffer("direction", direction)

        self.alpha = alpha
        projection = self.compute_projection(alpha)
        self.register_buffer("projection", projection)
    
    def compute_projection(self, alpha: float) -> torch.Tensor:
        projection = self.direction.T @ self.direction / torch.norm(self.direction) ** 2
        identity = torch.diag(torch.ones(projection.shape[0])).to(projection.device)
        projection = alpha * projection + (1 - alpha) * identity
        return projection

    def update_alpha(self, alpha: float) -> None:
        self.alpha = alpha
        self.projection = self.compute_projection(alpha)

    @torch.inference_mode()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x / torch.norm(x, dim=-1, keepdim=True)
        y = 1 - (torch.norm((x - self.target) @ self.projection, dim=-1) ** 2) / 2
        return y

    @staticmethod
    def tokenize_prompts(x: List[str]) -> torch.Tensor:
        """Tokenize a list of prompts."""
        return open_clip.tokenize(x)

    def embed_prompts(self, x) -> torch.Tensor:
        """Embed a list of prompts."""
        with torch.no_grad():
            x = self.embed_module.clip_model.encode_text(x).float()
        x = x / x.norm(dim=-1, keepdim=True)
        return x

    def embed_images(self, x):
        return self.embed_module.forward(x)


def load_reward_model(
    model_name, target_prompts, baseline_prompts, alpha, cache_dir: Optional[str] = None
):
    model_name_prefix, pretrained = model_name.split("/")
    model = open_clip.create_model(
        model_name=model_name_prefix, pretrained=pretrained, cache_dir=cache_dir
    )
    target_prompts = CLIPReward.tokenize_prompts(target_prompts)
    baseline_prompts = CLIPReward.tokenize_prompts(baseline_prompts)
    model = CLIPEmbed(model)
    model = CLIPReward(
        model=model,
        alpha=alpha,
        target_prompts=target_prompts,
        baseline_prompts=baseline_prompts,
    )
    return model.eval()


def load_reward_model_from_config(config: CLIPRewardConfig) -> CLIPReward:
    return load_reward_model(
        model_name=config.pretrained_model,
        target_prompts=config.target_prompts,
        baseline_prompts=config.baseline_prompts,
        alpha=config.alpha,
        cache_dir=config.cache_dir,
    )


def compute_rewards(
    model: CLIPEmbed,
    frames: torch.Tensor,
    batch_size: int,
    num_workers: int,
    worker_frames_tensor=None,
) -> torch.Tensor:
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
                render_dim=frames_batch.shape[1:],
                batch_size=batch_size // num_workers,
                num_workers=num_workers,
                frames=frames_batch,
                worker_frames_tensor=worker_frames_tensor,
            )
            rewards_batch = rewards_batch.cpu()
            rewards[i : i + batch_size] = rewards_batch 
    return rewards


@overload
def dist_worker_compute_reward(
    rank: int,
    reward_model: CLIPReward,
    render_dim: Tuple[int, int, int],
    batch_size: int,
    num_workers: int,
    frames: torch.Tensor,
) -> torch.Tensor:
    ...


@overload
def dist_worker_compute_reward(
    rank: int,
    reward_model: CLIPReward,
    render_dim: Tuple[int, int, int],
    batch_size: int,
    num_workers: int,
    frames: None = None,
) -> None:
    ...


def dist_worker_compute_reward(
    rank: int,
    reward_model: CLIPReward,
    render_dim: Tuple[int, int, int],
    batch_size: int,
    num_workers: int,
    frames=None,
    worker_frames_tensor=None,
) -> Optional[torch.Tensor]:
    if rank == 0:
        if frames is None:
            raise ValueError("Must pass render result on rank=0")
        if len(frames) != num_workers * batch_size:
            raise ValueError("Must pass render result with correct batch size")
        scatter_list = [t.cuda(rank) for t in torch.chunk(frames, num_workers, dim=0)]
    else:
        scatter_list = []

    worker_frames = worker_frames_tensor if worker_frames_tensor is not None else torch.zeros((batch_size, *render_dim), dtype=torch.uint8).cuda(rank)
    dist.scatter(worker_frames, scatter_list=scatter_list, src=0)
    with torch.no_grad():
        embeddings = reward_model.embed_module(worker_frames)
        rewards = reward_model(embeddings)

    def zero_t():
        return torch.zeros_like(rewards)

    recv_rewards = [zero_t() for _ in range(num_workers)] if rank == 0 else []
    dist.gather(rewards, gather_list=recv_rewards, dst=0)

    if rank == 0:
        return torch.cat(recv_rewards, dim=0).cuda(rank)
