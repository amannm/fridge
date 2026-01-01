from __future__ import annotations

import torch


def compute_fbank(beats_model, waveforms: torch.Tensor) -> torch.Tensor:
    if waveforms.ndim != 2:
        raise RuntimeError("Expected waveforms with shape [batch, samples]")
    return beats_model.preprocess(waveforms)
