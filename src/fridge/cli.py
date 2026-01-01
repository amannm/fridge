from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from .config import Config, load_config, save_config
from .dataset import (
    generate_windows_manifest,
    load_recordings_manifest,
    load_windows_manifest,
    save_recordings_manifest,
    split_by_group,
)
from .eval import collect_predictions, compute_metrics, find_best_threshold
from .io_audio import decode_to_wav, get_duration_seconds, load_audio
from .train import load_checkpoint, resolve_device, train_final, train_fold
from .windowing import window_indices
from .streaming import stream_from_mic


def _make_run_dir(artifacts_root: str | Path, command: str, override: Optional[str]) -> Path:
    if override:
        run_dir = Path(override)
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = Path(artifacts_root) / command / timestamp
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def _write_threshold(path: Path, threshold: float, cfg: Config) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "threshold": float(threshold),
        "on_threshold": cfg.postprocess.on_threshold,
        "on_frames": cfg.postprocess.on_frames,
        "off_threshold": cfg.postprocess.off_threshold,
        "off_frames": cfg.postprocess.off_frames,
    }
    path.write_text(json.dumps(payload, indent=2))


def cmd_prep(args: argparse.Namespace) -> None:
    cfg = load_config(args.config)
    run_dir = _make_run_dir(cfg.data.artifacts_root, "prep", args.run_dir)
    save_config(cfg, run_dir / "config.yaml")

    recordings_df = load_recordings_manifest(cfg.data.recordings_manifest)
    decoded_dir = Path(cfg.data.decoded_dir)
    decoded_dir.mkdir(parents=True, exist_ok=True)

    durations = []
    for _, rec in recordings_df.iterrows():
        recording_id = rec["recording_id"]
        input_path = Path(rec["path"])
        if not input_path.exists():
            raise FileNotFoundError(f"Recording not found: {input_path}")
        output_path = decoded_dir / f"{recording_id}.wav"
        decode_to_wav(input_path, output_path, cfg.audio.sample_rate, force=args.force_decode)
        waveform = load_audio(
            output_path,
            expected_sample_rate=cfg.audio.sample_rate,
            expected_mono=True,
            expected_dtype=cfg.audio.dtype,
            value_range=(cfg.audio.value_range_min, cfg.audio.value_range_max),
        )
        durations.append(get_duration_seconds(waveform, cfg.audio.sample_rate))

    recordings_df["duration_sec"] = durations
    save_recordings_manifest(recordings_df, cfg.data.recordings_manifest)

    windows_df = generate_windows_manifest(
        recordings_df,
        decoded_dir=decoded_dir,
        sample_rate=cfg.audio.sample_rate,
        window_sec=cfg.window.train_window_sec,
        hop_sec=cfg.window.train_hop_sec,
        output_path=cfg.data.windows_manifest,
        value_range=(cfg.audio.value_range_min, cfg.audio.value_range_max),
    )
    print(f"Generated {len(windows_df)} windows")



def _build_holdout_windows(
    recordings_df: pd.DataFrame,
    decoded_dir: Path,
    sample_rate: int,
    window_sec: float,
    hop_sec: float,
    holdout_seconds: float,
    value_range: tuple[float, float],
) -> pd.DataFrame:
    rows = []
    window_size = int(window_sec * sample_rate)
    hop_size = int(hop_sec * sample_rate)
    holdout_samples = int(holdout_seconds * sample_rate)

    for _, rec in recordings_df.iterrows():
        recording_id = rec["recording_id"]
        label = int(rec["label"])
        group = rec["group"]
        decoded_path = decoded_dir / f"{recording_id}.wav"
        waveform = load_audio(
            decoded_path,
            expected_sample_rate=sample_rate,
            expected_mono=True,
            expected_dtype="float32",
            value_range=value_range,
        )
        if waveform.shape[0] < holdout_samples:
            raise ValueError(f"Recording too short for holdout: {recording_id}")
        start_base = waveform.shape[0] - holdout_samples
        indices = window_indices(holdout_samples, window_size, hop_size)
        for idx, (start, end) in enumerate(indices):
            rows.append(
                {
                    "window_id": f"{recording_id}_holdout_{idx:05d}",
                    "recording_id": recording_id,
                    "start_sample": int(start_base + start),
                    "end_sample": int(start_base + end),
                    "label": label,
                    "group": group,
                }
            )
    if not rows:
        raise ValueError("No holdout windows generated")
    return pd.DataFrame(rows)


def cmd_train(args: argparse.Namespace) -> None:
    cfg = load_config(args.config)
    run_dir = _make_run_dir(cfg.data.artifacts_root, "train", args.run_dir)
    save_config(cfg, run_dir / "config.yaml")

    windows_df = load_windows_manifest(cfg.data.windows_manifest)

    fold_a_train, fold_a_val = split_by_group(windows_df, ["day"], ["night"])
    fold_b_train, fold_b_val = split_by_group(windows_df, ["night"], ["day"])

    metrics_a, ckpt_a = train_fold(cfg, fold_a_train, fold_a_val, cfg.data.decoded_dir, run_dir, "fold_A")
    metrics_b, ckpt_b = train_fold(cfg, fold_b_train, fold_b_val, cfg.data.decoded_dir, run_dir, "fold_B")

    report_dir = run_dir / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / "fold_A.json").write_text(json.dumps(metrics_a, indent=2))
    (report_dir / "fold_B.json").write_text(json.dumps(metrics_b, indent=2))

    threshold_dir = run_dir / "thresholds"
    threshold_dir.mkdir(parents=True, exist_ok=True)

    device = resolve_device(cfg.train.device)

    for fold_name, ckpt_path, val_df in [
        ("fold_A", ckpt_a, fold_a_val),
        ("fold_B", ckpt_b, fold_b_val),
    ]:
        model = load_checkpoint(ckpt_path, cfg, device)
        val_dataset = __build_eval_dataset(cfg, val_df)
        val_loader = __build_eval_loader(cfg, val_dataset)
        y_true, y_prob = collect_predictions(model, val_loader, device)
        threshold = find_best_threshold(y_true, y_prob)
        _write_threshold(threshold_dir / f"{fold_name}_threshold.json", threshold, cfg)

    print("Training complete")


def __build_eval_dataset(cfg: Config, windows_df: pd.DataFrame):
    from .dataset import WindowedFridgeDataset

    return WindowedFridgeDataset(
        windows_df,
        decoded_dir=cfg.data.decoded_dir,
        sample_rate=cfg.audio.sample_rate,
        window_sec=cfg.window.train_window_sec,
        value_range=(cfg.audio.value_range_min, cfg.audio.value_range_max),
        augment=None,
    )


def __build_eval_loader(cfg: Config, dataset):
    from torch.utils.data import DataLoader

    return DataLoader(
        dataset,
        batch_size=cfg.train.batch_size,
        shuffle=False,
        num_workers=cfg.train.num_workers,
        drop_last=False,
    )


def cmd_eval(args: argparse.Namespace) -> None:
    cfg = load_config(args.config)
    run_dir = _make_run_dir(cfg.data.artifacts_root, "eval", args.run_dir)
    save_config(cfg, run_dir / "config.yaml")

    windows_df = load_windows_manifest(cfg.data.windows_manifest)
    if args.group:
        windows_df = windows_df[windows_df["group"] == args.group]
        if windows_df.empty:
            raise ValueError(f"No windows for group {args.group}")

    dataset = __build_eval_dataset(cfg, windows_df)
    loader = __build_eval_loader(cfg, dataset)
    device = resolve_device(cfg.train.device)
    model = load_checkpoint(Path(args.checkpoint), cfg, device)
    y_true, y_prob = collect_predictions(model, loader, device)
    threshold = find_best_threshold(y_true, y_prob)
    metrics = compute_metrics(y_true, y_prob, threshold)
    report = {
        "threshold": threshold,
        "auc": metrics.auc,
        "f1": metrics.f1,
        "accuracy": metrics.accuracy,
        "confusion": metrics.confusion,
    }
    report_path = run_dir / "reports" / "eval.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


def cmd_final_train(args: argparse.Namespace) -> None:
    cfg = load_config(args.config)
    run_dir = _make_run_dir(cfg.data.artifacts_root, "final_train", args.run_dir)
    save_config(cfg, run_dir / "config.yaml")

    windows_df = load_windows_manifest(cfg.data.windows_manifest)
    checkpoint_path = train_final(cfg, windows_df, cfg.data.decoded_dir, run_dir)

    recordings_df = load_recordings_manifest(cfg.data.recordings_manifest)
    holdout_df = _build_holdout_windows(
        recordings_df,
        decoded_dir=Path(cfg.data.decoded_dir),
        sample_rate=cfg.audio.sample_rate,
        window_sec=cfg.window.train_window_sec,
        hop_sec=cfg.window.train_hop_sec,
        holdout_seconds=args.holdout_seconds,
        value_range=(cfg.audio.value_range_min, cfg.audio.value_range_max),
    )
    dataset = __build_eval_dataset(cfg, holdout_df)
    loader = __build_eval_loader(cfg, dataset)
    device = resolve_device(cfg.train.device)
    model = load_checkpoint(checkpoint_path, cfg, device)
    y_true, y_prob = collect_predictions(model, loader, device)
    threshold = find_best_threshold(y_true, y_prob)

    threshold_path = run_dir / "thresholds" / "threshold.json"
    _write_threshold(threshold_path, threshold, cfg)

    print(f"Final model saved to {checkpoint_path}")
    print(f"Threshold saved to {threshold_path}")


def cmd_stream(args: argparse.Namespace) -> None:
    cfg = load_config(args.config)
    checkpoint_path = args.checkpoint or cfg.inference.checkpoint_path
    threshold_path = args.threshold or cfg.inference.threshold_path
    if not checkpoint_path or not threshold_path:
        raise ValueError("Checkpoint and threshold paths are required for streaming")
    stream_from_mic(cfg, checkpoint_path, threshold_path)


def _bandpower_800_3000(waveform: np.ndarray, sample_rate: int) -> float:
    freqs = np.fft.rfftfreq(waveform.shape[0], d=1.0 / sample_rate)
    spectrum = np.fft.rfft(waveform)
    power = np.abs(spectrum) ** 2
    mask = (freqs >= 800.0) & (freqs <= 3000.0)
    return float(np.log10(np.mean(power[mask]) + 1e-12))


def cmd_diag(args: argparse.Namespace) -> None:
    cfg = load_config(args.config)
    if args.path:
        path = Path(args.path)
        if not path.exists():
            raise FileNotFoundError(f"Audio file not found: {path}")
        waveform = load_audio(
            path,
            expected_sample_rate=cfg.audio.sample_rate,
            expected_mono=True,
            expected_dtype="float32",
            value_range=(cfg.audio.value_range_min, cfg.audio.value_range_max),
        )
        value = _bandpower_800_3000(waveform, cfg.audio.sample_rate)
        print(json.dumps({"path": str(path), "bandpower": value}, indent=2))
        return

    recordings_df = load_recordings_manifest(cfg.data.recordings_manifest)
    results = []
    for _, rec in recordings_df.iterrows():
        recording_id = rec["recording_id"]
        decoded_path = Path(cfg.data.decoded_dir) / f"{recording_id}.wav"
        waveform = load_audio(
            decoded_path,
            expected_sample_rate=cfg.audio.sample_rate,
            expected_mono=True,
            expected_dtype="float32",
            value_range=(cfg.audio.value_range_min, cfg.audio.value_range_max),
        )
        value = _bandpower_800_3000(waveform, cfg.audio.sample_rate)
        results.append({"recording_id": recording_id, "bandpower": value})
    print(json.dumps(results, indent=2))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Fridge compressor detection pipeline")
    subparsers = parser.add_subparsers(dest="command", required=True)

    prep = subparsers.add_parser("prep", help="Decode audio and generate manifests")
    prep.add_argument("--config", required=True)
    prep.add_argument("--run-dir", required=False)
    prep.add_argument("--force-decode", action="store_true")
    prep.set_defaults(func=cmd_prep)

    train = subparsers.add_parser("train", help="Run grouped cross-validation training")
    train.add_argument("--config", required=True)
    train.add_argument("--run-dir", required=False)
    train.set_defaults(func=cmd_train)

    eval_cmd = subparsers.add_parser("eval", help="Evaluate a checkpoint")
    eval_cmd.add_argument("--config", required=True)
    eval_cmd.add_argument("--checkpoint", required=True)
    eval_cmd.add_argument("--group", required=False, choices=["day", "night"])
    eval_cmd.add_argument("--run-dir", required=False)
    eval_cmd.set_defaults(func=cmd_eval)

    final_train = subparsers.add_parser("final-train", help="Train on all data and set threshold")
    final_train.add_argument("--config", required=True)
    final_train.add_argument("--run-dir", required=False)
    final_train.add_argument("--holdout-seconds", type=float, default=15.0)
    final_train.set_defaults(func=cmd_final_train)

    stream = subparsers.add_parser("stream", help="Run live streaming inference")
    stream.add_argument("--config", required=True)
    stream.add_argument("--checkpoint", required=False)
    stream.add_argument("--threshold", required=False)
    stream.set_defaults(func=cmd_stream)

    diag = subparsers.add_parser("diag", help="Diagnostic bandpower check")
    diag.add_argument("--config", required=True)
    diag.add_argument("--path", required=False)
    diag.set_defaults(func=cmd_diag)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
