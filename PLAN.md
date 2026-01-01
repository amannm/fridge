# PLAN — Fridge Compressor Noise Detection System

This plan implements the full pipeline described in `BRAINSTORM.md` and `SPEC.md` using BEATs with a small multitask head, deterministic training, grouped validation, and streaming inference with EMA + hysteresis. The plan is written to be executed top‑down; each phase produces concrete artifacts and validates critical assumptions early. No fallbacks are used — missing requirements should fail fast with explicit errors.

---

## 0) Guiding Principles (Non‑Negotiable)

- **Use `uv` for all Python work** (env creation, installs, running scripts).
- **No legacy code**: clean new modules; no backward compatibility.
- **No fallbacks**: hard‑fail on missing audio backend, missing checkpoint, or invalid config.
- **Single source of truth**: one YAML config for all tunables.
- **Determinism**: fixed seeds where possible, saved configs, and reproducible data splits.

---

## 1) Project Layout (Target Structure)

```
configs/
  default.yaml
  inference.yaml
data/
  raw/                 # optional mirror of samples/ if user wants
  processed/
    wav/               # 16 kHz mono wavs
    windows/           # optional cached windows (if enabled)
  manifests/
    recordings.jsonl   # one record per clip
    windows.jsonl      # one record per window (optional cache)
checkpoints/
metrics/
artifacts/
reference/             # BEATs source (already present)
samples/               # raw m4a clips (already present)
src/fridge/
  config.py
  io/audio.py
  io/manifest.py
  data/windowing.py
  data/splits.py
  data/augment.py
  data/noise_pool.py
  features/beats_frontend.py
  models/beats_backbone.py
  models/heads.py
  models/wrapper.py
  train/loops.py
  train/optim.py
  train/schedulers.py
  eval/metrics.py
  eval/stream_eval.py
  streaming/buffer.py
  streaming/hysteresis.py
  streaming/runner.py
  utils/seed.py
  utils/logging.py
scripts/
  prepare_data.py
  train.py
  evaluate.py
  sweep_thresholds.py
  stream.py
  replay_stream.py
```

Notes:
- `samples/` remains the canonical raw input directory unless the user explicitly relocates data.
- BEATs code remains in `reference/unilm/beats`; we wrap it without modification unless necessary.

---

## 2) Environment & Dependencies (uv‑first)

### Tasks
- Add dependencies to `pyproject.toml`:
  - Core: `torch`, `torchaudio`, `numpy`, `scipy`, `pyyaml`, `tqdm`
  - Metrics: `scikit-learn` (AUROC/F1), or implement minimal AUROC if avoiding extra deps
  - Optional logging: `tensorboard` (if desired)
  - Audio input (streaming): `sounddevice` (requires system audio device)
- Provide a `scripts/check_env.py` that validates:
  - Python version
  - `ffmpeg` availability (for m4a → wav conversion)
  - `torch` + `torchaudio` import success
  - GPU availability if required (not mandatory)

### Artifacts
- `pyproject.toml` updated
- `scripts/check_env.py`

---

## 3) Configuration (Single YAML Source of Truth)

### Tasks
- Create `configs/default.yaml` matching the specification exactly.
- Create `configs/inference.yaml` that references the same defaults but allows threshold overrides and audio device selection for live streaming.
- Implement `src/fridge/config.py`:
  - Pydantic or dataclass loader with schema validation.
  - Strict type checks (no silent coercion).
  - Explicit validation: trim times < clip length, window sizes > 0, thresholds in [0,1], etc.

### Artifacts
- `configs/default.yaml`
- `configs/inference.yaml`
- `src/fridge/config.py`

---

## 4) Data Ingest & Canonicalization

### Tasks
1. **Manifest building**
   - Parse filenames in `samples/`:
     - `ac-on` / `ac-off`
     - `fridge-on` / `fridge-off`
     - `day` / `night`
   - Emit `data/manifests/recordings.jsonl` with fields:
     - `path`, `fridge_on`, `ac_on`, `environment`, `duration_s`
   - Fail if any filename does not match the schema.

2. **Decode + resample**
   - Use `ffmpeg` to convert each `.m4a` to 16 kHz mono `.wav` in `data/processed/wav/`.
   - No per‑file normalization; preserve raw amplitude.
   - Verify output sample rate + channels via `torchaudio.info`.

3. **Trim**
   - Apply `trim_start_s` and `trim_end_s` during window extraction (no physical trim needed unless caching windows).

### Artifacts
- `scripts/prepare_data.py`
- `data/processed/wav/*.wav`
- `data/manifests/recordings.jsonl`

---

## 5) Windowing & Splits (Leak‑Free)

### Tasks
- Implement `src/fridge/data/windowing.py`:
  - Sliding windows of length `train_win_s` with hop `train_hop_s`.
  - Each window inherits clip labels.
  - Optional random crop per `train_random_crop_s` (sample length then crop).
- Implement `src/fridge/data/splits.py`:
  - Grouped split by recording.
  - Default validation pair = `(ac-on, night, fridge-off)` + `(ac-on, night, fridge-on)`.
  - Optional 4‑fold “environment holdout” support.
- Add a deterministic seed to ensure stable splits.
- Optional caching of window metadata to `data/manifests/windows.jsonl` to speed training.

### Artifacts
- `src/fridge/data/windowing.py`
- `src/fridge/data/splits.py`
- Optional `data/manifests/windows.jsonl`

---

## 6) Augmentation Pipeline

### Tasks
- Implement `src/fridge/data/augment.py` with:
  - Gain jitter `[-6, +6] dB`
  - Time shift ±0.25 s (roll + crop, not padding)
  - Noise mixing from fridge‑off pool at SNR `[0, 20] dB`
  - Hard negatives: mix fridge‑on snippets into fridge‑off at 15–25 dB SNR
- Implement `src/fridge/data/noise_pool.py`:
  - Build pool of fridge‑off windows from training set only.
  - Ensure no leakage from validation recordings.
- Add optional SpecAugment on fbank features (time + freq masks).

### Artifacts
- `src/fridge/data/augment.py`
- `src/fridge/data/noise_pool.py`

---

## 7) BEATs Feature Frontend (Native)

### Tasks
- Implement a wrapper `src/fridge/features/beats_frontend.py`:
  - Use `reference/unilm/beats/BEATs.py` preprocessing (Kaldi fbank).
  - Confirm mean/std constants (15.41663 / 6.55582).
  - Ensure input is `torch.float32` waveform in range [-1, 1].
- Add unit tests to validate fbank shapes for a 2 s window.

### Artifacts
- `src/fridge/features/beats_frontend.py`
- Tests (see Section 12)

---

## 8) Model Architecture (Backbone + Heads)

### Tasks
- Implement `src/fridge/models/beats_backbone.py`:
  - Load BEATs config + weights from `models/BEATs_iter3_plus_AS2M.pt`.
  - Expose an `extract_features()` method returning framewise embeddings.
  - Provide selective unfreezing of last N transformer blocks.
- Implement `src/fridge/models/heads.py`:
  - MLP head: LN → Linear → GELU → Dropout → Linear(→1).
  - Separate head for `fridge_on` and `ac_on`.
- Implement `src/fridge/models/wrapper.py`:
  - Mean pooling across time.
  - Return fridge logit + ac logit.

### Artifacts
- `src/fridge/models/*.py`

---

## 9) Training System (Two‑Stage Fine‑Tune)

### Tasks
- Implement `src/fridge/train/loops.py`:
  - Stage 1: freeze backbone, train heads.
  - Stage 2: unfreeze last 2 blocks + final norm.
  - Early stopping on validation smoothed F1.
- Implement `src/fridge/train/optim.py` and `schedulers.py`:
  - AdamW with weight decay 0.01.
  - Warmup + cosine schedule (warmup_frac = 0.05).
  - Gradient clipping at 1.0.
- Implement metrics logging per epoch:
  - Window AUROC / F1
  - Smoothed F1 (streaming simulation)
- Save best checkpoint and config snapshot to `checkpoints/`.

### Artifacts
- `scripts/train.py`
- `checkpoints/`
- `metrics/`

---

## 10) Evaluation & Threshold Sweep

### Tasks
- Implement `src/fridge/eval/metrics.py`:
  - Window AUROC, window F1.
  - Smoothed F1 by running the same streaming logic offline.
- Implement `src/fridge/eval/stream_eval.py`:
  - Slide windows in chronological order.
  - Apply EMA + hysteresis.
  - Compute smoothed F1 vs true clip labels.
- Implement `scripts/sweep_thresholds.py`:
  - Grid search `hysteresis_on` / `hysteresis_off`.
  - Save `thresholds.yaml` with best pair.

### Artifacts
- `scripts/evaluate.py`
- `scripts/sweep_thresholds.py`
- `thresholds.yaml`

---

## 11) Streaming Inference (Live)

### Tasks
- Implement `src/fridge/streaming/buffer.py`:
  - Rolling buffer of 2.0 s of audio updated every 0.5 s.
- Implement `src/fridge/streaming/hysteresis.py`:
  - EMA probability smoothing.
  - Hysteresis thresholds with state retention.
- Implement `src/fridge/streaming/runner.py`:
  - Audio input via `sounddevice` (fail if unavailable).
  - Run BEATs inference, update state, emit `fridge_on` boolean.
- Provide `scripts/stream.py` (live) and `scripts/replay_stream.py` (offline file replay for testing).

### Artifacts
- `src/fridge/streaming/*.py`
- `scripts/stream.py`
- `scripts/replay_stream.py`

---

## 12) Tests & Validation

### Minimum tests (unit + integration)
- Windowing correctness:
  - Number of windows for known clip duration.
  - Label inheritance.
- Split correctness:
  - Validation pair contains the exact required files.
  - No leakage (train/val disjoint by recording).
- Augmentation invariants:
  - Output shape same as input.
  - SNR mixing bounds respected.
- Streaming logic:
  - Hysteresis state transitions at threshold boundaries.
  - EMA update correctness.
- BEATs frontend:
  - Fbank shape and dtype for 2 s audio.

### Artifacts
- `tests/` with `pytest` (if added)
- A `scripts/run_tests.py` using `uv run` for reproducible execution.

---

## 13) Outputs & Reproducibility

### Outputs
- `checkpoints/best.pt`
- `metrics/train.jsonl` + `metrics/val.jsonl`
- `thresholds.yaml`
- `artifacts/inference_config.yaml` (frozen for deployment)
- `artifacts/run_info.json` (seed, git hash, date)

### Reproducibility
- Record seeds, config snapshots, and split manifest.
- Freeze train/val lists in `data/manifests/`.

---

## 14) Execution Order (Concrete Steps)

1. `uv venv && uv pip install -e .` (after `pyproject.toml` is updated)
2. `uv run scripts/check_env.py`
3. `uv run scripts/prepare_data.py` → builds manifests + wavs
4. `uv run scripts/train.py --config configs/default.yaml`
5. `uv run scripts/evaluate.py --config configs/default.yaml`
6. `uv run scripts/sweep_thresholds.py --config configs/default.yaml`
7. `uv run scripts/stream.py --config configs/inference.yaml`

---

## 15) Acceptance Criteria Mapping

- **Trained model**: `checkpoints/best.pt` exists with saved config.
- **Grouped holdout**: Validation uses AC‑on night pair.
- **Metrics**: Window AUROC + Smoothed F1 reported.
- **Thresholds**: `thresholds.yaml` generated from sweep.
- **Streaming**: live inference returns stable boolean using EMA + hysteresis.
- **Reproducible**: full pipeline can re‑run from raw clips using one config.

---

## 16) Implementation Risks & Mitigations

- **Audio decoding failures (m4a)**: enforce `ffmpeg` and fail fast; document requirement.
- **Tiny dataset overfitting**: strict grouped split, early stopping, minimal unfreezing.
- **Streaming flicker**: enforce EMA + hysteresis and select thresholds by smoothed F1.
- **Backend mismatch**: ensure BEATs preprocessing matches reference implementation (Kaldi fbank).

---

If you want, I can proceed to implement this plan step‑by‑step, starting with config + data ingest, then model + training, then evaluation and streaming.
