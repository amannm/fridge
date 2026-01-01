from __future__ import annotations

from pathlib import Path

import soundfile as sf
import torch
import torchaudio


class AudioError(RuntimeError):
    pass


def load_audio(path: Path, target_sr: int, mono: bool) -> torch.Tensor:
    if not path.exists():
        raise AudioError(f"Audio file not found: {path}")
    data, sr = sf.read(str(path), dtype="float32", always_2d=True)
    waveform = torch.from_numpy(data).transpose(0, 1)
    if sr != target_sr:
        waveform = torchaudio.functional.resample(waveform, sr, target_sr)
    if mono:
        if waveform.ndim != 2:
            raise AudioError("Expected waveform with shape [channels, samples]")
        if waveform.size(0) > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
    return waveform


def trim_audio(
    waveform: torch.Tensor,
    sample_rate: int,
    trim_start_s: float,
    trim_end_s: float,
) -> torch.Tensor:
    total_samples = waveform.size(-1)
    start_samples = int(trim_start_s * sample_rate)
    end_samples = int(trim_end_s * sample_rate)
    if start_samples + end_samples >= total_samples:
        raise AudioError("Trim exceeds audio length")
    if start_samples > 0:
        waveform = waveform[..., start_samples:]
    if end_samples > 0:
        waveform = waveform[..., :-end_samples]
    return waveform
