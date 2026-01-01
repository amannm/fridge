# PLAN: Fridge Compressor On/Off Detection System

This plan translates `THEORY.md` + `SPEC.md` into an implementable, end-to-end system: data prep, training, evaluation, and live streaming inference. It is explicit, deterministic, and avoids any fallback or legacy behavior.

---

## 0) Outcomes and acceptance criteria

### Primary outcomes
- **Live binary state** (ON/OFF) from apartment mic with stable streaming behavior.
- **Leak-free training pipeline** with day↔night grouped evaluation.
- **Reproducible artifacts**: trained weights, metrics, thresholds, and config snapshot.

### Acceptance criteria
- Training pipeline runs from raw clips to model artifacts with **no silent fallbacks**.
- Cross-validation uses **recording-grouped splits** exactly as in SPEC.
- Streaming inference uses **2.0s window, 0.25s hop, EMA + hysteresis**.
- Final deployment model trained on **all four clips**.

---

## 1) Repository layout (new files/dirs)

```
src/fridge/
  __init__.py
  config.py              # Typed config objects + defaults from SPEC
  io_audio.py            # Decode/resample, WAV/PCM utilities
  windowing.py           # Window/hop logic and ring buffer
  beats_frontend.py      # BEATs fbank + normalization wrapper (on-the-fly)
  model.py               # BEATs backbone + pooling + head
  augment.py             # Gain/time jitter/noise augmentation
  dataset.py             # Windowed dataset + manifest handling
  train.py               # Training loop + early stopping
  eval.py                # Metrics, threshold selection
  streaming.py           # Live inference loop + post-processing
  postprocess.py         # EMA + hysteresis state machine
  cli.py                 # Entry points for data prep/train/eval/stream
configs/
  base.yaml              # Canonical parameters from SPEC
  train.yaml
  stream.yaml
  data.yaml
manifests/
  recordings.csv         # Raw clip inventory + labels
  windows.csv            # Window-level manifest (generated)
artifacts/
  checkpoints/
  reports/
  thresholds/
```

Notes:
- `reference/unilm/beats` is used as the **only** BEATs source. No alternative feature pipelines.
- `artifacts/` is the default output root; it is cleanly versioned via config snapshots.

---

## 2) Environment & dependencies (uv-only)

### Python
- Pin a single Python version in `pyproject.toml` (e.g., 3.11) for determinism.

### Core deps (non-exhaustive)
- `torch`, `torchaudio` (or `soundfile` if torchaudio lacks m4a support on macOS)
- `numpy`, `scipy`
- `pandas`
- `scikit-learn` (AUC/F1/confusion)
- `tqdm`
- `pyyaml` (or `omegaconf`) for config
- `sounddevice` (live mic capture)

### Hard requirement: decoder
- Standardize on **ffmpeg** for `.m4a` decode.
- If ffmpeg is unavailable, **fail fast** with a clear error (no fallback).

### uv usage
- All environment setup, installs, and execution use `uv` (e.g., `uv run python -m fridge.cli ...`).

---

## 3) Configuration system

### Goals
- Single source of truth for all SPEC parameters.
- Deterministic run config stored alongside outputs.

### Implementation
- `config.py` defines typed dataclasses:
  - AudioConfig, FbankConfig, WindowConfig, TrainConfig, AugmentConfig, StreamConfig, PostProcessConfig
- YAML configs in `configs/` override defaults via explicit merge.
- Every run writes a full resolved config to `artifacts/.../config.yaml`.

---

## 4) Data preparation pipeline

### 4.1 Inventory manifest
- `manifests/recordings.csv` with columns:
  - `recording_id`, `path`, `label` (0/1), `group` (day/night), `duration_sec`
- This is human-editable, explicit, and versionable.

### 4.2 Decode + resample
- Decode `.m4a` with ffmpeg to **16k mono float32**.
- Verify length, sample rate, and range `[-1, 1]`.
- Cache as `.wav` or `.npy` in `artifacts/decoded/` to avoid repeated decode.

### 4.3 Windowing
- Window length = 2.0s (32000 samples at 16k).
- Hop = 0.5s (training) and 0.25s (streaming).
- For training windows, generate `windows.csv` with:
  - `window_id`, `recording_id`, `start_sample`, `end_sample`, `label`, `group`
- **No per-window loudness normalization**.

### 4.4 Feature extraction
- Use BEATs fbank with SPEC params:
  - `num_mel_bins=128`, `frame_length=25ms`, `frame_shift=10ms`
  - Normalize: `(fbank - 15.41663) / (2 * 6.55582)`
- Wrap BEATs frontend in `beats_frontend.py` so train/infer are identical.
- **On-the-fly only**: no fbank caching (eliminates cache/feature drift risk).

---

## 5) Dataset & augmentation

### Dataset class
- `WindowedFridgeDataset` loads from `windows.csv` and decoded audio cache.
- Returns `(waveform_window, label)`; features computed on-the-fly (or cached).

### Augmentation (training only)
- Random gain ±6 dB
- Random time shift within window
- Optional: mild EQ tilt
- Optional: additive external noise at 10–30 dB SNR (same distribution for ON and OFF)

### Guardrails
- No mixup between ON/OFF.
- No normalization beyond BEATs fbank normalization.

---

## 6) Split strategy (leak-free CV)

### Two folds (recording-grouped)
- Fold A train: day; eval: night
- Fold B train: night; eval: day

### Implementation
- Split logic in `dataset.py` or `cli.py` uses recording `group` to build datasets.
- Reject any split that mixes day/night within train/eval.

---

## 7) Model architecture

### Backbone
- BEATs `BEATs_iter3_plus_AS2M.pt` loaded from `models/`.
- Output last-layer features `(B, T, C)`.

### Pooling + head
- Mean pooling over time.
- Head:
  - `Linear(C, 256) -> GELU -> Dropout(0.2) -> Linear(256, 1)`
- Loss: `BCEWithLogitsLoss`

### Freezing policy
- Phase 1: freeze full backbone.
- Phase 2 (optional): unfreeze last 2 transformer blocks.

---

## 8) Training loop

### Phase 1 (default)
- Optimizer: AdamW
- LR head: 1e-3, WD: 1e-2
- Batch size: 64
- Epochs: up to 20
- Early stopping patience: 5 on validation AUC/F1

### Phase 2 (only if fold performance is not already perfect)
- Unfreeze last 2 blocks
- Discriminative LR:
  - Backbone: 1e-5
  - Head: 3e-4
- Warmup 5% steps, cosine decay
- Epochs: 10–20 with early stopping

### Logging & outputs
- Per-epoch metrics: AUC, F1, accuracy, confusion matrix
- Best checkpoint per fold stored in `artifacts/checkpoints/`
- Fold report saved to `artifacts/reports/`

---

## 9) Evaluation & threshold selection

### Metrics
- AUC, F1, accuracy, confusion matrix

### Threshold selection
- Use validation predictions to choose threshold maximizing F1.
- Save threshold to `artifacts/thresholds/threshold.json`.

### Final training
- Train on **all four recordings**.
- Use held-out slices (e.g., last 15s each) or fold validation results to set final threshold.

---

## 10) Streaming inference pipeline

### Audio capture
- Live mic stream via `sounddevice` at 16k mono.
- Hard fail if sample rate mismatches or device not found (no fallback).

### Ring buffer + inference
- Ring buffer holds 2.0s (32000 samples).
- Every 0.25s, extract window and compute:
  - BEATs fbank -> BEATs backbone -> head logit -> sigmoid
- EMA smoothing: `p_hat = 0.8 * p_hat_prev + 0.2 * p`.
- Hysteresis:
  - ON if `p_hat > 0.7` for 3 frames
  - OFF if `p_hat < 0.3` for 6 frames

### Outputs (JSON stream)
- Emit one JSON object per hop with timestamp, probability, smoothed probability, and state.
- Boolean state is the authoritative output; probabilities are for debugging/monitoring.

---

## 11) Baseline diagnostic (non-deploy)

- Implement log bandpower 800–3000 Hz check as a **debug-only** tool.
- Use it to validate live audio path matches training (no fallback).

---

## 12) CLI commands

Expose single entry via `python -m fridge.cli` with subcommands:
- `prep` → decode, resample, generate manifests
- `train` → run CV folds + optional fine-tune
- `eval` → metrics + threshold selection
- `final-train` → train on all data
- `stream` → live inference loop
- `diag` → baseline diagnostic tool

Each command:
- Requires an explicit config path.
- Writes outputs to a run-stamped directory under `artifacts/`.

---

## 13) Testing & validation

### Unit tests
- Windowing logic (start/end indices, count expectations)
- Fbank normalization (known inputs)
- EMA + hysteresis state transitions

### Integration tests
- Small end-to-end on a single short clip (smoke test)
- Verify deterministic output with fixed seed

### Manual validation
- Run `diag` on both ON and OFF recordings; verify bandpower separation.
- Run `stream` with known clip injected (if possible) to validate state changes.

---

## 14) Implementation sequence (recommended order)

1. **Config + project scaffolding** (`config.py`, YAMLs, `cli.py`)
2. **Audio decode + windowing** (`io_audio.py`, `windowing.py`)
3. **BEATs frontend wrapper** (`beats_frontend.py`)
4. **Dataset + augmentation** (`dataset.py`, `augment.py`)
5. **Model + training loop** (`model.py`, `train.py`)
6. **Evaluation + threshold selection** (`eval.py`)
7. **Streaming inference + postprocessing** (`streaming.py`, `postprocess.py`)
8. **Baseline diagnostic tool** (`diag` command)
9. **Tests + smoke runs**

---

## 15) Milestones

- **M1: Data prep complete** (decoded cache + windows manifest)
- **M2: Fold training complete** (metrics + best checkpoints)
- **M3: Final model + thresholds** (all data)
- **M4: Live streaming validated** (stable ON/OFF)

---

## 16) Non-negotiables (from SPEC)

- Use BEATs fbank with exact normalization constants.
- No denoise/AGC; raw mic path only.
- Leak-free day↔night split.
- No fallback behaviors.
- Always use `uv` for Python workflows.

---

## 17) Resolved decisions

- BEATs checkpoint location: `models/`
- Live inference output: JSON stream
- Feature extraction: on-the-fly (no fbank caching)
