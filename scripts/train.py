from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fridge.config import load_config
from fridge.data.augment import AudioAugmenter, AugmentSpec, MixSpec, SpecAugmentSpec
from fridge.data.dataset import WindowDataset, collate_batch
from fridge.data.noise_pool import build_noise_pool
from fridge.data.splits import make_split
from fridge.data.windowing import build_windows
from fridge.io.manifest import read_jsonl
from fridge.models.beats_backbone import freeze_all, unfreeze_last_blocks
from fridge.models.factory import build_model
from fridge.train.loops import EarlyStopper, evaluate, train_one_epoch
from fridge.train.optim import build_optimizer
from fridge.train.schedulers import build_warmup_cosine_scheduler
from fridge.utils.logging import setup_logging
from fridge.utils.paths import ensure_dir, find_project_root
from fridge.utils.seed import set_seed


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


def _save_checkpoint(path: Path, model: torch.nn.Module, metadata: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "metadata": metadata}, str(path))


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row))
            handle.write("\n")


def main() -> None:
    setup_logging()
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()

    config = load_config(args.config)
    root = find_project_root(Path(args.config))
    set_seed(config.run.seed, deterministic=config.run.deterministic)

    device = _device_from_config(config.run.device)

    manifest_path = config.paths.manifests_dir / "recordings.jsonl"
    records = read_jsonl(manifest_path)
    for rec in records:
        path = Path(rec["path"])
        if not path.is_absolute():
            rec["path"] = str((root / path).resolve())
    recordings = {rec["id"]: rec for rec in records}

    split = make_split(records, config.split.strategy, config.split.fold)
    train_records = [recordings[rid] for rid in split.train_ids]
    val_records = [recordings[rid] for rid in split.val_ids]

    train_windows = build_windows(
        train_records,
        window_s=config.windows.train_win_s,
        hop_s=config.windows.train_hop_s,
        trim_start_s=config.audio.trim_start_s,
        trim_end_s=config.audio.trim_end_s,
    )
    val_windows = build_windows(
        val_records,
        window_s=config.windows.eval_win_s,
        hop_s=config.windows.eval_hop_s,
        trim_start_s=config.audio.trim_start_s,
        trim_end_s=config.audio.trim_end_s,
    )

    max_noise_window = max(config.windows.train_win_s, config.windows.train_random_crop_s[1])
    noise_pool = build_noise_pool(
        recordings=records,
        train_ids=set(split.train_ids),
        sample_rate=config.audio.sample_rate,
        mono=config.audio.mono,
        trim_start_s=config.audio.trim_start_s,
        trim_end_s=config.audio.trim_end_s,
        window_s=max_noise_window,
        hop_s=config.windows.train_hop_s,
    )

    augment_spec = AugmentSpec(
        gain_db=config.augment.gain_db,
        mix_noise=MixSpec(p=config.augment.mix_noise.p, snr_db=config.augment.mix_noise.snr_db),
        hard_negative_mix=MixSpec(p=config.augment.hard_negative_mix.p, snr_db=config.augment.hard_negative_mix.snr_db),
        specaugment=SpecAugmentSpec(
            enabled=config.augment.specaugment.enabled,
            time_masks=config.augment.specaugment.time_masks,
            time_mask_frames=config.augment.specaugment.time_mask_frames,
            freq_masks=config.augment.specaugment.freq_masks,
            freq_mask_bins=config.augment.specaugment.freq_mask_bins,
        ),
    )
    augmenter = AudioAugmenter(augment_spec, noise_pool)

    train_ds = WindowDataset(
        windows=train_windows,
        recordings=recordings,
        sample_rate=config.audio.sample_rate,
        mono=config.audio.mono,
        trim_start_s=config.audio.trim_start_s,
        trim_end_s=config.audio.trim_end_s,
        train=True,
        random_crop_s=config.windows.train_random_crop_s,
        time_shift_s=config.augment.time_shift_s,
        augmenter=augmenter,
    )
    val_ds = WindowDataset(
        windows=val_windows,
        recordings=recordings,
        sample_rate=config.audio.sample_rate,
        mono=config.audio.mono,
        trim_start_s=config.audio.trim_start_s,
        trim_end_s=config.audio.trim_end_s,
        train=False,
        random_crop_s=config.windows.train_random_crop_s,
        time_shift_s=0.0,
        augmenter=None,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=config.train.batch_size,
        shuffle=True,
        num_workers=config.run.num_workers,
        pin_memory=config.run.pin_memory,
        collate_fn=collate_batch,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=config.train.batch_size,
        shuffle=False,
        num_workers=config.run.num_workers,
        pin_memory=config.run.pin_memory,
        collate_fn=collate_batch,
    )

    beats_root = root / "reference" / "unilm" / "beats"
    model = build_model(config, beats_root, device)
    model.to(device)

    ensure_dir(config.paths.checkpoints_dir)
    ensure_dir(config.paths.metrics_dir)

    train_log: list[dict] = []
    val_log: list[dict] = []

    # Stage 1
    freeze_all(model.beats)
    optimizer = build_optimizer(
        model,
        lr_head=config.train.stage1.lr_head,
        lr_backbone=None,
        weight_decay=config.train.weight_decay,
    )
    steps_per_epoch = math.ceil(len(train_loader) / config.train.grad_accum_steps)
    total_steps = steps_per_epoch * config.train.stage1.epochs
    warmup_steps = int(config.train.warmup_frac * total_steps)
    scheduler = build_warmup_cosine_scheduler(optimizer, warmup_steps, total_steps)

    best_score = -1.0
    best_checkpoint = config.paths.checkpoints_dir / "best.pt"

    for epoch in range(config.train.stage1.epochs):
        train_metrics = train_one_epoch(
            model,
            train_loader,
            optimizer,
            scheduler,
            device,
            config.train.grad_accum_steps,
            config.train.grad_clip,
            config.model.multitask.aux_ac_weight,
        )
        val_metrics = evaluate(
            model,
            val_loader,
            device,
            recordings,
            config.streaming.ema_alpha,
            config.streaming.hysteresis_on,
            config.streaming.hysteresis_off,
        )
        train_metrics["epoch"] = epoch + 1
        train_metrics["stage"] = "stage1"
        val_metrics["epoch"] = epoch + 1
        val_metrics["stage"] = "stage1"
        train_log.append(train_metrics)
        val_log.append(val_metrics)

        if val_metrics["smoothed_f1"] > best_score:
            best_score = val_metrics["smoothed_f1"]
            _save_checkpoint(best_checkpoint, model, {"stage": "stage1", "epoch": epoch + 1, "score": best_score})

    # Stage 2
    freeze_all(model.beats)
    unfreeze_last_blocks(model.beats, config.train.stage2.unfreeze_last_blocks)
    optimizer = build_optimizer(
        model,
        lr_head=config.train.stage2.lr_head,
        lr_backbone=config.train.stage2.lr_backbone,
        weight_decay=config.train.weight_decay,
    )
    steps_per_epoch = math.ceil(len(train_loader) / config.train.grad_accum_steps)
    total_steps = steps_per_epoch * config.train.stage2.epochs
    warmup_steps = int(config.train.warmup_frac * total_steps)
    scheduler = build_warmup_cosine_scheduler(optimizer, warmup_steps, total_steps)
    stopper = EarlyStopper(config.train.stage2.early_stopping_patience)

    for epoch in range(config.train.stage2.epochs):
        train_metrics = train_one_epoch(
            model,
            train_loader,
            optimizer,
            scheduler,
            device,
            config.train.grad_accum_steps,
            config.train.grad_clip,
            config.model.multitask.aux_ac_weight,
        )
        val_metrics = evaluate(
            model,
            val_loader,
            device,
            recordings,
            config.streaming.ema_alpha,
            config.streaming.hysteresis_on,
            config.streaming.hysteresis_off,
        )
        train_metrics["epoch"] = epoch + 1
        train_metrics["stage"] = "stage2"
        val_metrics["epoch"] = epoch + 1
        val_metrics["stage"] = "stage2"
        train_log.append(train_metrics)
        val_log.append(val_metrics)

        if val_metrics["smoothed_f1"] > best_score:
            best_score = val_metrics["smoothed_f1"]
            _save_checkpoint(best_checkpoint, model, {"stage": "stage2", "epoch": epoch + 1, "score": best_score})

        if stopper.update(val_metrics["smoothed_f1"]):
            break

    _write_jsonl(config.paths.metrics_dir / "train.jsonl", train_log)
    _write_jsonl(config.paths.metrics_dir / "val.jsonl", val_log)

    run_info = {
        "seed": config.run.seed,
        "best_smoothed_f1": best_score,
        "config_path": str(Path(args.config).resolve()),
        "checkpoint": str(best_checkpoint.resolve()),
    }
    config.paths.artifacts_dir.mkdir(parents=True, exist_ok=True)
    (config.paths.artifacts_dir / "run_info.json").write_text(json.dumps(run_info, indent=2))
    (config.paths.artifacts_dir / "inference_config.yaml").write_text(Path(args.config).read_text())


if __name__ == "__main__":
    main()
