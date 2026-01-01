from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch

from .config import Config
from .model import FridgeClassifier
from .postprocess import PostProcessor
from .train import load_checkpoint, resolve_device
from .windowing import RingBuffer


class StreamError(RuntimeError):
    pass


def load_threshold(path: str | Path) -> dict:
    threshold_path = Path(path)
    if not threshold_path.exists():
        raise StreamError(f"Threshold file not found: {threshold_path}")
    data = json.loads(threshold_path.read_text())
    if "threshold" not in data:
        raise StreamError("Threshold file missing 'threshold' field")
    return data


def stream_from_mic(
    cfg: Config,
    checkpoint_path: str | Path,
    threshold_path: str | Path,
) -> None:
    import sounddevice as sd

    device = resolve_device(cfg.train.device)
    model = load_checkpoint(Path(checkpoint_path), cfg, device)
    model.eval()

    threshold_data = load_threshold(threshold_path)
    threshold = float(threshold_data["threshold"])

    hop_samples = int(cfg.window.stream_hop_sec * cfg.audio.sample_rate)
    window_samples = int(cfg.window.stream_window_sec * cfg.audio.sample_rate)
    ring = RingBuffer(window_samples)
    post = PostProcessor(
        ema_alpha=cfg.postprocess.ema_alpha,
        on_threshold=float(threshold_data.get("on_threshold", cfg.postprocess.on_threshold)),
        on_frames=int(threshold_data.get("on_frames", cfg.postprocess.on_frames)),
        off_threshold=float(threshold_data.get("off_threshold", cfg.postprocess.off_threshold)),
        off_frames=int(threshold_data.get("off_frames", cfg.postprocess.off_frames)),
    )

    if hop_samples <= 0:
        raise StreamError("Invalid hop size")

    stream = sd.InputStream(
        samplerate=cfg.audio.sample_rate,
        channels=1,
        dtype="float32",
        blocksize=hop_samples,
        device=cfg.stream.device,
    )

    with stream:
        print("{\"status\": \"ready\"}")
        while True:
            samples, overflowed = stream.read(hop_samples)
            if overflowed:
                raise StreamError("Audio input overflowed")
            mono = samples[:, 0].copy()
            ring.append(mono)
            if not ring.filled:
                continue
            window = ring.get()
            window_tensor = torch.from_numpy(window.astype(np.float32)).unsqueeze(0).to(device)
            with torch.no_grad():
                logit = model(window_tensor).item()
                prob = float(1.0 / (1.0 + math.exp(-logit)))
            p_hat, state = post.update(prob)
            payload = {
                "timestamp": time.time(),
                "prob": prob,
                "prob_smoothed": p_hat,
                "state": "on" if state else "off",
                "threshold": threshold,
            }
            print(json.dumps(payload))
