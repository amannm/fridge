from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch

from .noise_pool import NoisePool


@dataclass(frozen=True)
class MixSpec:
    p: float
    snr_db: tuple[float, float]


@dataclass(frozen=True)
class SpecAugmentSpec:
    enabled: bool
    time_masks: int
    time_mask_frames: int
    freq_masks: int
    freq_mask_bins: int


@dataclass(frozen=True)
class AugmentSpec:
    gain_db: tuple[float, float]
    mix_noise: MixSpec
    hard_negative_mix: MixSpec
    specaugment: SpecAugmentSpec


class AudioAugmenter:
    def __init__(self, spec: AugmentSpec, noise_pool: NoisePool):
        self.spec = spec
        self.noise_pool = noise_pool

    def __call__(self, waveform: torch.Tensor, fridge_on: int) -> torch.Tensor:
        waveform = apply_gain(waveform, self.spec.gain_db)
        waveform = maybe_mix_noise(waveform, self.spec.mix_noise, self.noise_pool.sample_off)
        if fridge_on == 0:
            waveform = maybe_mix_noise(waveform, self.spec.hard_negative_mix, self.noise_pool.sample_on)
        return waveform


def apply_gain(waveform: torch.Tensor, gain_db: tuple[float, float]) -> torch.Tensor:
    low, high = gain_db
    gain = torch.empty(1).uniform_(low, high).item()
    scale = 10 ** (gain / 20)
    return waveform * scale


def maybe_mix_noise(
    clean: torch.Tensor,
    mix_spec: MixSpec,
    sampler,
) -> torch.Tensor:
    if torch.rand(1).item() > mix_spec.p:
        return clean
    noise = sampler()
    if noise.size(-1) < clean.size(-1):
        raise RuntimeError("Noise segment shorter than clean segment")
    noise = random_crop(noise, clean.size(-1))
    snr_db = torch.empty(1).uniform_(mix_spec.snr_db[0], mix_spec.snr_db[1]).item()
    return mix_at_snr(clean, noise, snr_db)


def random_crop(waveform: torch.Tensor, length: int) -> torch.Tensor:
    if waveform.size(-1) < length:
        raise RuntimeError("Cannot crop: waveform shorter than requested length")
    if waveform.size(-1) == length:
        return waveform
    max_start = waveform.size(-1) - length
    start = torch.randint(0, max_start + 1, (1,)).item()
    return waveform[..., start : start + length]


def mix_at_snr(clean: torch.Tensor, noise: torch.Tensor, snr_db: float) -> torch.Tensor:
    clean_rms = torch.sqrt(torch.mean(clean ** 2))
    noise_rms = torch.sqrt(torch.mean(noise ** 2))
    if clean_rms.item() == 0 or noise_rms.item() == 0:
        raise RuntimeError("Zero RMS encountered during noise mixing")
    desired_noise_rms = clean_rms / (10 ** (snr_db / 20))
    scale = desired_noise_rms / noise_rms
    return clean + noise * scale


def apply_specaugment(
    fbank: torch.Tensor,
    spec: SpecAugmentSpec,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    if not spec.enabled:
        return fbank
    if fbank.ndim != 3:
        raise RuntimeError("Expected fbank shape [batch, frames, bins]")
    batch, frames, bins = fbank.shape
    out = fbank.clone()
    g = generator
    for _ in range(spec.time_masks):
        if frames <= 1:
            break
        mask_len = torch.randint(0, min(spec.time_mask_frames, frames) + 1, (1,), generator=g).item()
        if mask_len == 0:
            continue
        start = torch.randint(0, frames - mask_len + 1, (1,), generator=g).item()
        out[:, start : start + mask_len, :] = 0
    for _ in range(spec.freq_masks):
        if bins <= 1:
            break
        mask_len = torch.randint(0, min(spec.freq_mask_bins, bins) + 1, (1,), generator=g).item()
        if mask_len == 0:
            continue
        start = torch.randint(0, bins - mask_len + 1, (1,), generator=g).item()
        out[:, :, start : start + mask_len] = 0
    return out
