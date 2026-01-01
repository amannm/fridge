from __future__ import annotations

from typing import Iterable, List

import torch
import torch.nn as nn


class ModelError(RuntimeError):
    pass


class FridgeClassifier(nn.Module):
    def __init__(
        self,
        backbone: nn.Module,
        feature_dim: int,
        head_hidden: int,
        dropout: float,
        pooling: str,
        fbank_mean: float,
        fbank_std: float,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.pooling = pooling
        self.fbank_mean = fbank_mean
        self.fbank_std = fbank_std
        self.head = nn.Sequential(
            nn.Linear(feature_dim, head_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(head_hidden, 1),
        )

    def forward(self, waveforms: torch.Tensor, padding_mask: torch.Tensor | None = None) -> torch.Tensor:
        features, _ = self.backbone.extract_features(
            waveforms,
            padding_mask=padding_mask,
            fbank_mean=self.fbank_mean,
            fbank_std=self.fbank_std,
        )
        if self.pooling == "mean":
            pooled = features.mean(dim=1)
        else:
            raise ModelError(f"Unsupported pooling: {self.pooling}")
        logits = self.head(pooled).squeeze(-1)
        return logits

    def freeze_backbone(self) -> None:
        for param in self.backbone.parameters():
            param.requires_grad = False

    def unfreeze_last_blocks(self, num_blocks: int) -> None:
        if num_blocks <= 0:
            return
        if not hasattr(self.backbone, "encoder"):
            raise ModelError("Backbone has no encoder layers")
        layers = self.backbone.encoder.layers
        total = len(layers)
        if num_blocks > total:
            raise ModelError(
                f"Requested {num_blocks} blocks, but backbone has {total} layers"
            )
        for idx, layer in enumerate(layers):
            requires = idx >= total - num_blocks
            for param in layer.parameters():
                param.requires_grad = requires

    def backbone_trainable_params(self) -> List[torch.nn.Parameter]:
        return [p for p in self.backbone.parameters() if p.requires_grad]

    def head_trainable_params(self) -> List[torch.nn.Parameter]:
        return [p for p in self.head.parameters() if p.requires_grad]
