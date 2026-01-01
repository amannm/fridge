from __future__ import annotations

from pathlib import Path

import torch

from ..config import Config
from .beats_backbone import load_beats_model
from .heads import MLPHead
from .wrapper import FridgeModel
from ..data.augment import SpecAugmentSpec


def build_model(config: Config, beats_root: Path, device: torch.device) -> FridgeModel:
    beats = load_beats_model(config.model.backbone, beats_root)
    beats.to(device)

    head = MLPHead(
        input_dim=beats.cfg.encoder_embed_dim,
        hidden_dim=config.model.head.mlp_hidden,
        dropout=config.model.head.dropout,
    )
    ac_head = None
    if config.model.multitask.enabled:
        ac_head = MLPHead(
            input_dim=beats.cfg.encoder_embed_dim,
            hidden_dim=config.model.head.mlp_hidden,
            dropout=config.model.head.dropout,
        )

    specaugment = SpecAugmentSpec(
        enabled=config.augment.specaugment.enabled,
        time_masks=config.augment.specaugment.time_masks,
        time_mask_frames=config.augment.specaugment.time_mask_frames,
        freq_masks=config.augment.specaugment.freq_masks,
        freq_mask_bins=config.augment.specaugment.freq_mask_bins,
    )

    model = FridgeModel(beats=beats, head_fridge=head, head_ac=ac_head, specaugment=specaugment)
    model.to(device)
    return model


def load_checkpoint(model: torch.nn.Module, checkpoint_path: Path) -> dict:
    if not checkpoint_path.exists():
        raise RuntimeError(f"Checkpoint not found: {checkpoint_path}")
    checkpoint = torch.load(str(checkpoint_path), map_location="cpu")
    if "model" not in checkpoint:
        raise RuntimeError("Checkpoint missing model state")
    model.load_state_dict(checkpoint["model"], strict=True)
    return checkpoint.get("metadata", {})
