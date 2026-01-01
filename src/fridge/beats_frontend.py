from __future__ import annotations

import sys
from pathlib import Path
from typing import Tuple

import torch


class BeatsError(RuntimeError):
    pass


def _ensure_beats_importable() -> None:
    root = Path(__file__).resolve().parents[2] / "reference" / "unilm" / "beats"
    if not root.exists():
        raise BeatsError(f"BEATs source not found at {root}")
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))


def load_beats(checkpoint_path: str | Path, device: torch.device) -> "torch.nn.Module":
    _ensure_beats_importable()
    from BEATs import BEATs, BEATsConfig  # type: ignore

    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise BeatsError(f"BEATs checkpoint not found: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if "cfg" not in checkpoint or "model" not in checkpoint:
        raise BeatsError("Invalid BEATs checkpoint format")
    cfg = BEATsConfig(checkpoint["cfg"])
    model = BEATs(cfg)
    model.load_state_dict(checkpoint["model"])
    model.to(device)
    model.eval()
    return model


def beats_feature_dim(beats_model: "torch.nn.Module") -> int:
    if not hasattr(beats_model, "cfg"):
        raise BeatsError("BEATs model missing config")
    return int(beats_model.cfg.encoder_embed_dim)
