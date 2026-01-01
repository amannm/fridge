from __future__ import annotations

import argparse
import yaml
import sys
from pathlib import Path

import numpy as np
import sounddevice as sd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fridge.config import load_config
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

    sample_rate = config.audio.sample_rate
    buffer_samples = int(config.streaming.buffer_s * sample_rate)
    update_samples = int(config.streaming.update_every_s * sample_rate)
    if update_samples <= 0:
        raise RuntimeError("update_every_s too small")

    block_size = config.streaming.block_size or update_samples

    buffer = RollingBuffer(buffer_samples)
    hysteresis = HysteresisState(
        ema_alpha=config.streaming.ema_alpha,
        hysteresis_on=hysteresis_on,
        hysteresis_off=hysteresis_off,
    )

    stream_kwargs = {
        "samplerate": sample_rate,
        "channels": 1,
        "dtype": "float32",
        "blocksize": block_size,
    }
    if config.streaming.audio_device_index is not None:
        stream_kwargs["device"] = config.streaming.audio_device_index
    if config.streaming.latency is not None:
        stream_kwargs["latency"] = config.streaming.latency

    with sd.InputStream(**stream_kwargs) as stream:
        while True:
            data, _ = stream.read(update_samples)
            samples = np.asarray(data, dtype=np.float32).reshape(-1)
            buffer.append(samples)
            if not buffer.is_full():
                continue
            window = buffer.get()
            waveform = torch.from_numpy(window).unsqueeze(0).to(device)
            padding_mask = torch.zeros_like(waveform, dtype=torch.bool)
            with torch.no_grad():
                logit, _ = model(waveform, padding_mask=padding_mask)
                prob = torch.sigmoid(logit).item()
            state = hysteresis.update(prob)
            print(f"prob={prob:.3f} ema={hysteresis.p_ema:.3f} state={state}")


if __name__ == "__main__":
    main()
