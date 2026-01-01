from __future__ import annotations

import argparse
import yaml
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fridge.config import load_config
from fridge.io.audio import load_audio
from fridge.models.factory import build_model, load_checkpoint
from fridge.streaming.buffer import RollingBuffer
from fridge.streaming.hysteresis import HysteresisState
from fridge.utils.logging import setup_logging
from fridge.utils.paths import find_project_root


def _device_from_config(device_name: str) -> torch.device:
    if device_name == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but not available")
        return torch.device("cuda")
    if device_name == "mps":
        if not torch.backends.mps.is_available():
            raise RuntimeError("MPS requested but not available")
        return torch.device("mps")
    return torch.device("cpu")


def _load_thresholds(path: Path | None, config) -> tuple[float, float]:
    if path is None:
        return config.streaming.hysteresis_on, config.streaming.hysteresis_off
    if not path.exists():
        raise RuntimeError(f"Threshold file not found: {path}")
    data = yaml.safe_load(path.read_text())
    if "hysteresis_on" not in data or "hysteresis_off" not in data:
        raise RuntimeError("Threshold file missing hysteresis values")
    return float(data["hysteresis_on"]), float(data["hysteresis_off"])


def main() -> None:
    setup_logging()
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/inference.yaml")
    parser.add_argument("--checkpoint", default="checkpoints/best.pt")
    parser.add_argument("--thresholds", default=None)
    parser.add_argument("--audio", required=True)
    args = parser.parse_args()

    config = load_config(args.config)
    root = find_project_root(Path(args.config))
    device = _device_from_config(config.run.device)

    thresholds_path = Path(args.thresholds) if args.thresholds else None
    hysteresis_on, hysteresis_off = _load_thresholds(thresholds_path, config)

    beats_root = root / "reference" / "unilm" / "beats"
    model = build_model(config, beats_root, device)
    load_checkpoint(model, Path(args.checkpoint))
    model.eval()

    audio_path = Path(args.audio)
    waveform = load_audio(audio_path, config.audio.sample_rate, config.audio.mono)
    waveform = waveform.squeeze(0)

    sample_rate = config.audio.sample_rate
    buffer_samples = int(config.streaming.buffer_s * sample_rate)
    update_samples = int(config.streaming.update_every_s * sample_rate)
    if update_samples <= 0:
        raise RuntimeError("update_every_s too small")

    buffer = RollingBuffer(buffer_samples)
    hysteresis = HysteresisState(
        ema_alpha=config.streaming.ema_alpha,
        hysteresis_on=hysteresis_on,
        hysteresis_off=hysteresis_off,
    )

    total_samples = waveform.size(0)
    idx = 0
    step = 0
    while idx < total_samples:
        chunk = waveform[idx : idx + update_samples]
        if chunk.numel() < update_samples:
            break
        buffer.append(chunk.numpy())
        idx += update_samples
        step += 1
        if not buffer.is_full():
            continue
        window = buffer.get()
        window_tensor = torch.from_numpy(window).unsqueeze(0).to(device)
        padding_mask = torch.zeros_like(window_tensor, dtype=torch.bool)
        with torch.no_grad():
            logit, _ = model(window_tensor, padding_mask=padding_mask)
            prob = torch.sigmoid(logit).item()
        state = hysteresis.update(prob)
        timestamp = step * config.streaming.update_every_s
        print(f"t={timestamp:.2f}s prob={prob:.3f} ema={hysteresis.p_ema:.3f} state={state}")


if __name__ == "__main__":
    main()
