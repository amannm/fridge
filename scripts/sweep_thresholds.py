from __future__ import annotations

import argparse
import yaml
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
from fridge.eval.stream_eval import smoothed_f1
from fridge.io.manifest import read_jsonl
from fridge.models.factory import build_model, load_checkpoint
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


def _collect_scores(model, loader, device):
    model.eval()
    scores: list[float] = []
    labels: list[int] = []
    recording_ids: list[str] = []
    start_times: list[float] = []
    with torch.no_grad():
        for batch in loader:
            waveforms = batch.waveforms.to(device)
            padding_mask = batch.padding_mask.to(device)
            fridge_labels = batch.fridge_labels.to(device)
            fridge_logit, _ = model(waveforms, padding_mask=padding_mask)
            scores.extend(torch.sigmoid(fridge_logit).cpu().tolist())
            labels.extend(fridge_labels.cpu().int().tolist())
            recording_ids.extend(batch.recording_ids)
            start_times.extend(batch.start_times)
    return scores, labels, recording_ids, start_times


def main() -> None:
    setup_logging()
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--checkpoint", default="checkpoints/best.pt")
    args = parser.parse_args()

    config = load_config(args.config)
    root = find_project_root(Path(args.config))
    device = _device_from_config(config.run.device)

    records = read_jsonl(config.paths.manifests_dir / "recordings.jsonl")
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

    scores, labels, recording_ids, start_times = _collect_scores(model, val_loader, device)

    best = {"f1": -1.0, "on": None, "off": None}
    for on_thr in config.eval.threshold_sweep_on:
        for off_thr in config.eval.threshold_sweep_off:
            if on_thr <= off_thr:
                continue
            f1 = smoothed_f1(
                recordings=recordings,
                y_scores=scores,
                recording_ids=recording_ids,
                start_times=start_times,
                ema_alpha=config.streaming.ema_alpha,
                hysteresis_on=on_thr,
                hysteresis_off=off_thr,
            )
            if f1 > best["f1"]:
                best = {"f1": f1, "on": on_thr, "off": off_thr}

    if best["on"] is None:
        raise RuntimeError("No valid threshold pairs found")

    output = {
        "hysteresis_on": best["on"],
        "hysteresis_off": best["off"],
        "smoothed_f1": best["f1"],
    }
    config.paths.thresholds_path.parent.mkdir(parents=True, exist_ok=True)
    config.paths.thresholds_path.write_text(yaml.safe_dump(output, sort_keys=False))


if __name__ == "__main__":
    main()
