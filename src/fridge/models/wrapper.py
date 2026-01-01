from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from ..data.augment import SpecAugmentSpec, apply_specaugment
from .heads import MLPHead


class FridgeModel(nn.Module):
    def __init__(
        self,
        beats: nn.Module,
        head_fridge: MLPHead,
        head_ac: Optional[MLPHead],
        specaugment: SpecAugmentSpec,
    ) -> None:
        super().__init__()
        self.beats = beats
        self.head_fridge = head_fridge
        self.head_ac = head_ac
        self.specaugment = specaugment

    def forward(self, waveforms: torch.Tensor, padding_mask: Optional[torch.Tensor] = None):
        if waveforms.ndim != 2:
            raise RuntimeError("Expected waveforms with shape [batch, samples]")

        fbank = self.beats.preprocess(waveforms)
        if self.training and self.specaugment.enabled:
            fbank = apply_specaugment(fbank, self.specaugment)

        if padding_mask is not None:
            padding_mask = self.beats.forward_padding_mask(fbank, padding_mask)

        fbank = fbank.unsqueeze(1)
        features = self.beats.patch_embedding(fbank)
        features = features.reshape(features.shape[0], features.shape[1], -1)
        features = features.transpose(1, 2)
        features = self.beats.layer_norm(features)

        if padding_mask is not None:
            padding_mask = self.beats.forward_padding_mask(features, padding_mask)

        if self.beats.post_extract_proj is not None:
            features = self.beats.post_extract_proj(features)

        x = self.beats.dropout_input(features)
        x, _ = self.beats.encoder(x, padding_mask=padding_mask)
        pooled = masked_mean(x, padding_mask)

        fridge_logit = self.head_fridge(pooled)
        ac_logit = self.head_ac(pooled) if self.head_ac is not None else None
        return fridge_logit, ac_logit


def masked_mean(x: torch.Tensor, padding_mask: Optional[torch.Tensor]) -> torch.Tensor:
    if padding_mask is None or not padding_mask.any():
        return x.mean(dim=1)
    mask = (~padding_mask).float().unsqueeze(-1)
    summed = (x * mask).sum(dim=1)
    denom = mask.sum(dim=1).clamp(min=1.0)
    return summed / denom
