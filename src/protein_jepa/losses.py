from __future__ import annotations

import torch
from torch.nn import functional as F


def normalized_latent_loss(predicted: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    if predicted.numel() == 0:
        raise ValueError("Cannot compute JEPA loss with no target positions.")
    predicted = F.normalize(predicted, dim=-1)
    target = F.normalize(target.detach(), dim=-1)
    return F.smooth_l1_loss(predicted, target)


def variance_loss(values: torch.Tensor, *, target_std: float = 1.0, eps: float = 1e-4) -> torch.Tensor:
    if values.shape[0] < 2:
        return values.new_tensor(0.0)
    std = torch.sqrt(values.var(dim=0, unbiased=False) + eps)
    return torch.relu(target_std - std).mean()

