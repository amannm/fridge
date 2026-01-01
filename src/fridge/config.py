from __future__ import annotations

from dataclasses import dataclass, field, asdict, is_dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import yaml


@dataclass
class AudioConfig:
    sample_rate: int = 16000
    mono: bool = True
    dtype: str = "float32"
    value_range_min: float = -1.0
    value_range_max: float = 1.0


@dataclass
class FbankConfig:
    num_mel_bins: int = 128
    frame_length_ms: int = 25
    frame_shift_ms: int = 10
    mean: float = 15.41663
    std: float = 6.55582


@dataclass
class WindowConfig:
    train_window_sec: float = 2.0
    train_hop_sec: float = 0.5
    stream_window_sec: float = 2.0
    stream_hop_sec: float = 0.25


@dataclass
class ModelConfig:
    beats_checkpoint: str = "models/BEATs_iter3_plus_AS2M.pt"
    pooling: str = "mean"
    head_hidden: int = 256
    dropout: float = 0.2


@dataclass
class TrainConfig:
    batch_size: int = 64
    epochs_max: int = 20
    early_stop_patience: int = 5
    lr_head: float = 1e-3
    weight_decay: float = 1e-2
    device: str = "mps"
    seed: int = 1337
    num_workers: int = 0


@dataclass
class FineTuneConfig:
    enabled: bool = False
    unfreeze_last_blocks: int = 2
    lr_backbone: float = 1e-5
    lr_head: float = 3e-4
    warmup_ratio: float = 0.05
    epochs: int = 15


@dataclass
class AugmentConfig:
    gain_db_min: float = -6.0
    gain_db_max: float = 6.0
    time_shift: bool = True
    eq_tilt: bool = False
    noise_dir: Optional[str] = None
    noise_snr_min: float = 10.0
    noise_snr_max: float = 30.0


@dataclass
class PostProcessConfig:
    ema_alpha: float = 0.8
    on_threshold: float = 0.7
    on_frames: int = 3
    off_threshold: float = 0.3
    off_frames: int = 6


@dataclass
class StreamConfig:
    device: Optional[int] = None


@dataclass
class DataConfig:
    recordings_manifest: str = "manifests/recordings.csv"
    windows_manifest: str = "manifests/windows.csv"
    decoded_dir: str = "artifacts/decoded"
    artifacts_root: str = "artifacts"


@dataclass
class InferenceConfig:
    checkpoint_path: Optional[str] = None
    threshold_path: Optional[str] = None


@dataclass
class Config:
    audio: AudioConfig = field(default_factory=AudioConfig)
    fbank: FbankConfig = field(default_factory=FbankConfig)
    window: WindowConfig = field(default_factory=WindowConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    finetune: FineTuneConfig = field(default_factory=FineTuneConfig)
    augment: AugmentConfig = field(default_factory=AugmentConfig)
    postprocess: PostProcessConfig = field(default_factory=PostProcessConfig)
    stream: StreamConfig = field(default_factory=StreamConfig)
    data: DataConfig = field(default_factory=DataConfig)
    inference: InferenceConfig = field(default_factory=InferenceConfig)


class ConfigError(RuntimeError):
    pass


def _apply_overrides(target: Any, overrides: Dict[str, Any]) -> None:
    for key, value in overrides.items():
        if not hasattr(target, key):
            raise ConfigError(f"Unknown config key: {key}")
        current = getattr(target, key)
        if is_dataclass(current):
            if value is None:
                continue
            if not isinstance(value, dict):
                raise ConfigError(f"Config section '{key}' must be a mapping")
            _apply_overrides(current, value)
        else:
            setattr(target, key, value)


def _load_raw_config(path: Path) -> Dict[str, Any]:
    data = yaml.safe_load(path.read_text()) or {}
    if not isinstance(data, dict):
        raise ConfigError("Config file must contain a top-level mapping")
    return data


def _resolve_config_chain(path: Path, visited: set[Path]) -> list[Dict[str, Any]]:
    if path in visited:
        raise ConfigError(f"Cyclic config inheritance detected at {path}")
    visited.add(path)
    data = _load_raw_config(path)
    extends = data.pop("extends", None)
    if extends is None:
        return [data]
    base_path = (path.parent / extends).resolve()
    if not base_path.exists():
        raise ConfigError(f"Base config not found: {base_path}")
    chain = _resolve_config_chain(base_path, visited)
    chain.append(data)
    return chain


def load_config(path: str | Path) -> Config:
    config_path = Path(path).resolve()
    if not config_path.exists():
        raise ConfigError(f"Config file not found: {config_path}")
    chain = _resolve_config_chain(config_path, set())
    cfg = Config()
    for data in chain:
        _apply_overrides(cfg, data)
    return cfg


def config_to_dict(cfg: Config) -> Dict[str, Any]:
    return asdict(cfg)


def save_config(cfg: Config, path: str | Path) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    data = config_to_dict(cfg)
    output_path.write_text(yaml.safe_dump(data, sort_keys=False))
