import torch
import torch.nn as nn
from torch import Tensor
from vlmrm.reward.encoders import TextEncoder

Reward = nn.Module


def projection_reward(
    x: Tensor, baseline: Tensor, target: Tensor, alpha: float
) -> Tensor:
    direction = target - baseline
    projection = _compute_projection(direction, alpha)

    x = x / torch.norm(x, dim=-1, keepdim=True)
    y = 1 - (torch.norm((x - target) @ projection, dim=-1) ** 2) / 2

    return y


def _compute_projection(direction: Tensor, alpha: float) -> Tensor:
    projection = direction.T @ direction / torch.norm(direction) ** 2
    identity = torch.diag(torch.ones(projection.shape[0])).to(projection.device)
    projection = alpha * projection + (1 - alpha) * identity
    return projection


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
    def from_model(
        target_prompts: list[str],
        baseline_prompts: list[str],
        model: TextEncoder,
        alpha: float,
    ) -> "ProjectionReward":
        target = model.encode_text(target_prompts).mean(dim=0, keepdim=True)
        baseline = model.encode_text(baseline_prompts).mean(dim=0, keepdim=True)
        direction = target - baseline
        projection = _compute_projection(direction, alpha)

        return ProjectionReward(baseline, target, direction, projection, alpha)


def logit_reward(x: Tensor, labels: Tensor, target: Tensor) -> Tensor:
    return (x @ labels.T).softmax(dim=-1)[:, target]


class LogitReward(Reward):
    def __init__(self, baselines, target):
        super().__init__()
        self.register_buffer("options", torch.cat([target, baselines]))

    @torch.inference_mode()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.options = self.options.to(x.device)
        return logit_reward(x, self.options, target=torch.tensor(0).to(x.device))

    @staticmethod
    def from_model(
        target_prompts: list[str],
        baseline_prompts: list[str],
        model: TextEncoder,
    ) -> "LogitReward":
        target = model.encode_text(target_prompts).mean(dim=0, keepdim=True)
        baselines = model.encode_text(baseline_prompts)

        return LogitReward(baselines, target)
