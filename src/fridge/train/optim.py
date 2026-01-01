from __future__ import annotations

from typing import Iterable, Optional

import torch
import torch.nn as nn


def build_optimizer(
    model: nn.Module,
    lr_head: float,
    weight_decay: float,
    lr_backbone: Optional[float] = None,
) -> torch.optim.Optimizer:
    head_params: list[nn.Parameter] = []
    backbone_params: list[nn.Parameter] = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if name.startswith("beats."):
            backbone_params.append(param)
        else:
            head_params.append(param)
    if lr_backbone is None:
        if not head_params:
            raise RuntimeError("No trainable head parameters")
        return torch.optim.AdamW(head_params, lr=lr_head, weight_decay=weight_decay)
    if not backbone_params:
        raise RuntimeError("No trainable backbone parameters")
    return torch.optim.AdamW(
        [
            {"params": backbone_params, "lr": lr_backbone},
            {"params": head_params, "lr": lr_head},
        ],
        weight_decay=weight_decay,
    )
