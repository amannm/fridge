from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fridge.config import load_config
from fridge.data.dataset import WindowDataset, collate_batch
from fridge.data.splits import make_split
from fridge.data.windowing import build_windows
from fridge.io.manifest import read_jsonl
from fridge.models.factory import build_model, load_checkpoint
from fridge.train.loops import evaluate
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


def main() -> None:
    setup_logging()
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--checkpoint", default="checkpoints/best.pt")
    args = parser.parse_args()

    config = load_config(args.config)
    root = find_project_root(Path(args.config))
    device = _device_from_config(config.run.device)

    manifest_path = config.paths.manifests_dir / "recordings.jsonl"
    records = read_jsonl(manifest_path)
    for rec in records:
        path = Path(rec["path"])
        if not path.is_absolute():
            rec["path"] = str((root / path).resolve())
    recordings = {rec["id"]: rec for rec in records}

    split = make_split(records, config.split.strategy, config.split.fold)
    val_records = [recordings[rid] for rid in split.val_ids]

    val_windows = build_windows(
        val_records,
        window_s=config.windows.eval_win_s,
        hop_s=config.windows.eval_hop_s,
        trim_start_s=config.audio.trim_start_s,
        trim_end_s=config.audio.trim_end_s,
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
    load_checkpoint(model, Path(args.checkpoint))

    metrics = evaluate(
        model,
        val_loader,
        device,
        recordings,
        config.streaming.ema_alpha,
        config.streaming.hysteresis_on,
        config.streaming.hysteresis_off,
    )

    output_path = config.paths.metrics_dir / "eval.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
