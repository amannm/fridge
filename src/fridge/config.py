from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Sequence

import yaml

from .utils.paths import find_project_root, resolve_path


class ConfigError(ValueError):
    pass


def _ensure_keys(obj: dict, required: set[str], optional: set[str], name: str) -> None:
    if not isinstance(obj, dict):
        raise ConfigError(f"{name} must be a mapping")
    extra = set(obj.keys()) - required - optional
    missing = required - set(obj.keys())
    if missing or extra:
        parts: list[str] = []
        if missing:
            parts.append(f"missing keys: {sorted(missing)}")
        if extra:
            parts.append(f"unexpected keys: {sorted(extra)}")
        raise ConfigError(f"{name} invalid: {'; '.join(parts)}")


def _as_float_pair(value: Any, name: str) -> tuple[float, float]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ConfigError(f"{name} must be a 2-item list")
    if len(value) != 2:
        raise ConfigError(f"{name} must have exactly 2 items")
    lo, hi = float(value[0]), float(value[1])
    if lo > hi:
        raise ConfigError(f"{name} lower bound > upper bound")
    return lo, hi


@dataclass(frozen=True)
class PathsConfig:
    samples_dir: Path
    processed_wav_dir: Path
    manifests_dir: Path
    checkpoints_dir: Path
    metrics_dir: Path
    artifacts_dir: Path
    thresholds_path: Path


@dataclass(frozen=True)
class RunConfig:
    seed: int
    device: str
    num_workers: int
    pin_memory: bool
    deterministic: bool
    log_every_steps: int


@dataclass(frozen=True)
class AudioConfig:
    sample_rate: int
    mono: bool
    trim_start_s: float
    trim_end_s: float


@dataclass(frozen=True)
class WindowConfig:
    train_win_s: float
    train_hop_s: float
    train_random_crop_s: tuple[float, float]
    eval_win_s: float
    eval_hop_s: float


@dataclass(frozen=True)
class SplitConfig:
    strategy: str
    fold: int


@dataclass(frozen=True)
class ModelHeadConfig:
    mlp_hidden: int
    dropout: float


@dataclass(frozen=True)
class ModelMultitaskConfig:
    enabled: bool
    aux_ac_weight: float


@dataclass(frozen=True)
class ModelConfig:
    backbone: Path
    pooling: str
    head: ModelHeadConfig
    multitask: ModelMultitaskConfig


@dataclass(frozen=True)
class TrainStageConfig:
    freeze_backbone: bool
    epochs: int
    lr_head: float
    lr_backbone: Optional[float] = None
    unfreeze_last_blocks: Optional[int] = None
    early_stopping_patience: Optional[int] = None


@dataclass(frozen=True)
class TrainConfig:
    batch_size: int
    grad_accum_steps: int
    optimizer: str
    weight_decay: float
    grad_clip: float
    schedule: str
    warmup_frac: float
    stage1: TrainStageConfig
    stage2: TrainStageConfig


@dataclass(frozen=True)
class MixConfig:
    p: float
    snr_db: tuple[float, float]


@dataclass(frozen=True)
class SpecAugmentConfig:
    enabled: bool
    time_masks: int
    time_mask_frames: int
    freq_masks: int
    freq_mask_bins: int


@dataclass(frozen=True)
class AugmentConfig:
    gain_db: tuple[float, float]
    time_shift_s: float
    mix_noise: MixConfig
    hard_negative_mix: MixConfig
    specaugment: SpecAugmentConfig


@dataclass(frozen=True)
class EvalConfig:
    threshold_sweep_on: list[float]
    threshold_sweep_off: list[float]


@dataclass(frozen=True)
class StreamingConfig:
    buffer_s: float
    update_every_s: float
    ema_alpha: float
    hysteresis_on: float
    hysteresis_off: float
    audio_device_index: Optional[int]
    block_size: int
    latency: Optional[float]


@dataclass(frozen=True)
class Config:
    paths: PathsConfig
    run: RunConfig
    audio: AudioConfig
    windows: WindowConfig
    split: SplitConfig
    model: ModelConfig
    train: TrainConfig
    augment: AugmentConfig
    eval: EvalConfig
    streaming: StreamingConfig


def load_config(path: str | Path) -> Config:
    config_path = Path(path)
    if not config_path.exists():
        raise ConfigError(f"Config not found: {config_path}")
    with config_path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise ConfigError("Config root must be a mapping")

    required_top = {
        "paths",
        "run",
        "audio",
        "windows",
        "split",
        "model",
        "train",
        "augment",
        "eval",
        "streaming",
    }
    _ensure_keys(data, required_top, set(), "config")
    root = find_project_root(config_path)

    paths_cfg = _parse_paths(data["paths"], root)
    run_cfg = _parse_run(data["run"])
    audio_cfg = _parse_audio(data["audio"])
    window_cfg = _parse_windows(data["windows"])
    split_cfg = _parse_split(data["split"])
    model_cfg = _parse_model(data["model"], root)
    train_cfg = _parse_train(data["train"])
    augment_cfg = _parse_augment(data["augment"])
    eval_cfg = _parse_eval(data["eval"])
    streaming_cfg = _parse_streaming(data["streaming"])

    _validate_consistency(audio_cfg, window_cfg, streaming_cfg, train_cfg)

    return Config(
        paths=paths_cfg,
        run=run_cfg,
        audio=audio_cfg,
        windows=window_cfg,
        split=split_cfg,
        model=model_cfg,
        train=train_cfg,
        augment=augment_cfg,
        eval=eval_cfg,
        streaming=streaming_cfg,
    )


def _parse_paths(data: dict, root: Path) -> PathsConfig:
    required = {
        "samples_dir",
        "processed_wav_dir",
        "manifests_dir",
        "checkpoints_dir",
        "metrics_dir",
        "artifacts_dir",
        "thresholds_path",
    }
    _ensure_keys(data, required, set(), "paths")
    return PathsConfig(
        samples_dir=resolve_path(root, data["samples_dir"]),
        processed_wav_dir=resolve_path(root, data["processed_wav_dir"]),
        manifests_dir=resolve_path(root, data["manifests_dir"]),
        checkpoints_dir=resolve_path(root, data["checkpoints_dir"]),
        metrics_dir=resolve_path(root, data["metrics_dir"]),
        artifacts_dir=resolve_path(root, data["artifacts_dir"]),
        thresholds_path=resolve_path(root, data["thresholds_path"]),
    )


def _parse_run(data: dict) -> RunConfig:
    required = {"seed", "device", "num_workers", "pin_memory", "deterministic", "log_every_steps"}
    _ensure_keys(data, required, set(), "run")
    device = str(data["device"]).lower()
    if device not in {"cpu", "cuda", "mps"}:
        raise ConfigError("run.device must be 'cpu', 'cuda', or 'mps'")
    return RunConfig(
        seed=int(data["seed"]),
        device=device,
        num_workers=int(data["num_workers"]),
        pin_memory=bool(data["pin_memory"]),
        deterministic=bool(data["deterministic"]),
        log_every_steps=int(data["log_every_steps"]),
    )


def _parse_audio(data: dict) -> AudioConfig:
    required = {"sample_rate", "mono", "trim_start_s", "trim_end_s"}
    _ensure_keys(data, required, set(), "audio")
    sample_rate = int(data["sample_rate"])
    trim_start = float(data["trim_start_s"])
    trim_end = float(data["trim_end_s"])
    if sample_rate <= 0:
        raise ConfigError("audio.sample_rate must be > 0")
    if trim_start < 0 or trim_end < 0:
        raise ConfigError("audio trim values must be >= 0")
    return AudioConfig(
        sample_rate=sample_rate,
        mono=bool(data["mono"]),
        trim_start_s=trim_start,
        trim_end_s=trim_end,
    )


def _parse_windows(data: dict) -> WindowConfig:
    required = {"train_win_s", "train_hop_s", "train_random_crop_s", "eval_win_s", "eval_hop_s"}
    _ensure_keys(data, required, set(), "windows")
    train_win = float(data["train_win_s"])
    train_hop = float(data["train_hop_s"])
    eval_win = float(data["eval_win_s"])
    eval_hop = float(data["eval_hop_s"])
    if train_win <= 0 or train_hop <= 0 or eval_win <= 0 or eval_hop <= 0:
        raise ConfigError("window sizes and hops must be > 0")
    crop_range = _as_float_pair(data["train_random_crop_s"], "windows.train_random_crop_s")
    return WindowConfig(
        train_win_s=train_win,
        train_hop_s=train_hop,
        train_random_crop_s=crop_range,
        eval_win_s=eval_win,
        eval_hop_s=eval_hop,
    )


def _parse_split(data: dict) -> SplitConfig:
    required = {"strategy", "fold"}
    _ensure_keys(data, required, set(), "split")
    strategy = str(data["strategy"])
    fold = int(data["fold"])
    if strategy not in {"ac_on_night_holdout", "environment_holdout"}:
        raise ConfigError("split.strategy must be 'ac_on_night_holdout' or 'environment_holdout'")
    if fold < 0:
        raise ConfigError("split.fold must be >= 0")
    return SplitConfig(strategy=strategy, fold=fold)


def _parse_model(data: dict, root: Path) -> ModelConfig:
    required = {"backbone", "pooling", "head", "multitask"}
    _ensure_keys(data, required, set(), "model")
    pooling = str(data["pooling"]).lower()
    if pooling != "mean":
        raise ConfigError("model.pooling must be 'mean'")

    head_data = data["head"]
    _ensure_keys(head_data, {"mlp_hidden", "dropout"}, set(), "model.head")
    head_cfg = ModelHeadConfig(
        mlp_hidden=int(head_data["mlp_hidden"]),
        dropout=float(head_data["dropout"]),
    )

    mt_data = data["multitask"]
    _ensure_keys(mt_data, {"enabled", "aux_ac_weight"}, set(), "model.multitask")
    multitask_cfg = ModelMultitaskConfig(
        enabled=bool(mt_data["enabled"]),
        aux_ac_weight=float(mt_data["aux_ac_weight"]),
    )
    if multitask_cfg.aux_ac_weight < 0:
        raise ConfigError("model.multitask.aux_ac_weight must be >= 0")

    backbone = resolve_path(root, data["backbone"])
    return ModelConfig(
        backbone=backbone,
        pooling=pooling,
        head=head_cfg,
        multitask=multitask_cfg,
    )


def _parse_train(data: dict) -> TrainConfig:
    required = {
        "batch_size",
        "grad_accum_steps",
        "optimizer",
        "weight_decay",
        "grad_clip",
        "schedule",
        "warmup_frac",
        "stage1",
        "stage2",
    }
    _ensure_keys(data, required, set(), "train")
    optimizer = str(data["optimizer"]).lower()
    if optimizer != "adamw":
        raise ConfigError("train.optimizer must be 'adamw'")
    schedule = str(data["schedule"]).lower()
    if schedule != "warmup_cosine":
        raise ConfigError("train.schedule must be 'warmup_cosine'")

    stage1 = _parse_train_stage(data["stage1"], name="train.stage1", is_stage2=False)
    stage2 = _parse_train_stage(data["stage2"], name="train.stage2", is_stage2=True)

    return TrainConfig(
        batch_size=int(data["batch_size"]),
        grad_accum_steps=int(data["grad_accum_steps"]),
        optimizer=optimizer,
        weight_decay=float(data["weight_decay"]),
        grad_clip=float(data["grad_clip"]),
        schedule=schedule,
        warmup_frac=float(data["warmup_frac"]),
        stage1=stage1,
        stage2=stage2,
    )


def _parse_train_stage(data: dict, name: str, is_stage2: bool) -> TrainStageConfig:
    if is_stage2:
        required = {"freeze_backbone", "epochs", "lr_head", "lr_backbone", "unfreeze_last_blocks", "early_stopping_patience"}
    else:
        required = {"freeze_backbone", "epochs", "lr_head"}
    _ensure_keys(data, required, set(), name)
    if is_stage2:
        return TrainStageConfig(
            freeze_backbone=bool(data["freeze_backbone"]),
            epochs=int(data["epochs"]),
            lr_head=float(data["lr_head"]),
            lr_backbone=float(data["lr_backbone"]),
            unfreeze_last_blocks=int(data["unfreeze_last_blocks"]),
            early_stopping_patience=int(data["early_stopping_patience"]),
        )
    return TrainStageConfig(
        freeze_backbone=bool(data["freeze_backbone"]),
        epochs=int(data["epochs"]),
        lr_head=float(data["lr_head"]),
    )


def _parse_augment(data: dict) -> AugmentConfig:
    required = {"gain_db", "time_shift_s", "mix_noise", "hard_negative_mix", "specaugment"}
    _ensure_keys(data, required, set(), "augment")
    mix_noise = _parse_mix(data["mix_noise"], "augment.mix_noise")
    hard_mix = _parse_mix(data["hard_negative_mix"], "augment.hard_negative_mix")
    spec = _parse_specaugment(data["specaugment"])
    gain_db = _as_float_pair(data["gain_db"], "augment.gain_db")
    return AugmentConfig(
        gain_db=gain_db,
        time_shift_s=float(data["time_shift_s"]),
        mix_noise=mix_noise,
        hard_negative_mix=hard_mix,
        specaugment=spec,
    )


def _parse_mix(data: dict, name: str) -> MixConfig:
    _ensure_keys(data, {"p", "snr_db"}, set(), name)
    p = float(data["p"])
    if not 0 <= p <= 1:
        raise ConfigError(f"{name}.p must be in [0, 1]")
    snr_db = _as_float_pair(data["snr_db"], f"{name}.snr_db")
    return MixConfig(p=p, snr_db=snr_db)


def _parse_specaugment(data: dict) -> SpecAugmentConfig:
    _ensure_keys(data, {"enabled", "time_masks", "time_mask_frames", "freq_masks", "freq_mask_bins"}, set(), "augment.specaugment")
    return SpecAugmentConfig(
        enabled=bool(data["enabled"]),
        time_masks=int(data["time_masks"]),
        time_mask_frames=int(data["time_mask_frames"]),
        freq_masks=int(data["freq_masks"]),
        freq_mask_bins=int(data["freq_mask_bins"]),
    )


def _parse_eval(data: dict) -> EvalConfig:
    _ensure_keys(data, {"threshold_sweep"}, set(), "eval")
    sweep = data["threshold_sweep"]
    _ensure_keys(sweep, {"on_values", "off_values"}, set(), "eval.threshold_sweep")
    on_values = [float(x) for x in sweep["on_values"]]
    off_values = [float(x) for x in sweep["off_values"]]
    if not on_values or not off_values:
        raise ConfigError("eval.threshold_sweep values must be non-empty")
    return EvalConfig(threshold_sweep_on=on_values, threshold_sweep_off=off_values)


def _parse_streaming(data: dict) -> StreamingConfig:
    required = {
        "buffer_s",
        "update_every_s",
        "ema_alpha",
        "hysteresis_on",
        "hysteresis_off",
        "audio_device_index",
        "block_size",
        "latency",
    }
    _ensure_keys(data, required, set(), "streaming")
    audio_device_index = data["audio_device_index"]
    if audio_device_index is not None:
        audio_device_index = int(audio_device_index)
    latency = data["latency"]
    if latency is not None:
        latency = float(latency)
    return StreamingConfig(
        buffer_s=float(data["buffer_s"]),
        update_every_s=float(data["update_every_s"]),
        ema_alpha=float(data["ema_alpha"]),
        hysteresis_on=float(data["hysteresis_on"]),
        hysteresis_off=float(data["hysteresis_off"]),
        audio_device_index=audio_device_index,
        block_size=int(data["block_size"]),
        latency=latency,
    )


def _validate_consistency(
    audio: AudioConfig,
    windows: WindowConfig,
    streaming: StreamingConfig,
    train: TrainConfig,
) -> None:
    if audio.sample_rate != 16000:
        raise ConfigError("BEATs requires 16 kHz audio; set audio.sample_rate=16000")
    if windows.eval_win_s <= 0 or windows.train_win_s <= 0:
        raise ConfigError("window sizes must be > 0")
    if streaming.buffer_s < windows.eval_win_s:
        raise ConfigError("streaming.buffer_s must be >= windows.eval_win_s")
    if not 0.0 < streaming.ema_alpha <= 1.0:
        raise ConfigError("streaming.ema_alpha must be in (0, 1]")
    if streaming.hysteresis_on <= streaming.hysteresis_off:
        raise ConfigError("streaming.hysteresis_on must be > streaming.hysteresis_off")
    if train.grad_accum_steps < 1:
        raise ConfigError("train.grad_accum_steps must be >= 1")
    if train.batch_size < 1:
        raise ConfigError("train.batch_size must be >= 1")
