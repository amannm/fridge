from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn


class BeatsLoadError(RuntimeError):
    pass


def _load_beats_modules(beats_root: Path):
    beats_root = beats_root.resolve()
    if not beats_root.exists():
        raise BeatsLoadError(f"BEATs source not found at {beats_root}")
    if str(beats_root) not in sys.path:
        sys.path.insert(0, str(beats_root))
    from BEATs import BEATs, BEATsConfig

    return BEATs, BEATsConfig


def load_beats_model(checkpoint_path: Path, beats_root: Path) -> nn.Module:
    if not checkpoint_path.exists():
        raise BeatsLoadError(f"BEATs checkpoint not found: {checkpoint_path}")
    BEATs, BEATsConfig = _load_beats_modules(beats_root)
    checkpoint = torch.load(str(checkpoint_path), map_location="cpu")
    cfg = BEATsConfig(checkpoint["cfg"])
    model = BEATs(cfg)
    model.load_state_dict(checkpoint["model"], strict=True)
    return model


def freeze_all(model: nn.Module) -> None:
    for param in model.parameters():
        param.requires_grad = False


def unfreeze_last_blocks(model: nn.Module, num_blocks: int) -> None:
    if num_blocks <= 0:
        return
    if not hasattr(model, "encoder"):
        raise BeatsLoadError("Model has no encoder attribute")
    encoder = model.encoder
    if not hasattr(encoder, "layers"):
        raise BeatsLoadError("Encoder has no layers attribute")
    layers = list(encoder.layers)
    if num_blocks > len(layers):
        raise BeatsLoadError("num_blocks exceeds number of encoder layers")
    for layer in layers[-num_blocks:]:
        for param in layer.parameters():
            param.requires_grad = True
    if hasattr(model, "layer_norm"):
        for param in model.layer_norm.parameters():
            param.requires_grad = True
