Here’s a concrete “best-shot” recipe for your exact setup (8×3-min steady-state clips, fixed mic/location, fridge compressor + AC blower as dominant sources) using **BEATs_iter3_plus_AS2M.pt + a small head**, *without collecting more data*.

I also did a quick spectral check on your 8 recordings after converting to 16 kHz mono: **“fridge on” adds consistent extra energy in mid-low bands (roughly 200–1000 Hz) and some narrow tonal peaks**, while **AC-on raises the low-frequency floor a lot**, so you want (1) short-window classification + smoothing, and (2) training that forces invariance to AC state.

---

## 1) Data prep (do this exactly)

### 1. Decode + resample (match BEATs expectations)

BEATs uses **16 kHz** audio and internally operates on **128-bin log Mel filterbanks** with **25 ms window / 10 ms hop** (Kaldi fbank style). ([Proceedings of Machine Learning Research][1])
So:

* Convert each file → **mono 16 kHz** WAV (or float waveform in memory)
* Keep raw amplitude (don’t per-file normalize to 0 dBFS); BEATs’ fbank normalization handles scale statistically.

### 2. Trim edges (avoid start/stop artifacts)

Even “steady-state” recordings often have subtle handling/AGC settling at the beginning.

* **Drop first 8–10 s and last 3–5 s** of every recording.

### 3. Windowing (this is your real dataset)

Treat each 3-min recording as thousands of supervised examples:

**Training windows (defaults)**

* `win_len = 2.0 s`
* `hop = 1.0 s` (50% overlap)
* Label every window with the clip label (fridge on/off).

**Why 2 s?**

* Long enough to capture compressor tonality + low-frequency modulation
* Short enough for responsive streaming inference

### 4. Split correctly (avoid leakage optimism)

Your windows within a recording are extremely correlated. Don’t random-split windows.

**Group split by recording**:

* 6 recordings train, 2 recordings val (or 5/3 if you prefer stability).
* Make sure val includes **AC-on** *and* **AC-off** conditions across runs.

A strong default:

* **Validation = both “night” recordings with AC-on (fridge off + fridge on)** once, then swap to AC-off night on another run.

---

## 2) Feature path: use BEATs’ native fbank preprocessing

BEATs commonly preprocesses by extracting **Kaldi fbank** (128 mel bins, 16k, 25ms/10ms) and normalizing with constants (often mean=15.41663, std=6.55582, with division by `2*std`). ([Hugging Face][2])

**Default: do NOT invent your own spectrogram frontend.** Feed waveform → BEATs preprocessing → BEATs encoder.

---

## 3) Model head (simple, strong defaults)

You want a head that’s:

* small (tiny data)
* regularized
* tolerant of nuisance (AC)

**Pooling**

* Take BEATs frame sequence → **mean-pool over time** (default)
* (Optional upgrade) learnable attention pooling, but mean-pool is hard to beat on tiny datasets.

**Head architecture (default)**

* `LayerNorm`
* `Linear(d_model → 128)`
* `GELU`
* `Dropout(p=0.2)`
* `Linear(128 → 1)` (logit for fridge_on)

---

## 4) Training strategy (this matters most for accuracy)

### Key idea: make AC a nuisance explicitly (multi-task for free)

Because you *already have* AC on/off labels (from filenames), you can add an auxiliary output and improve separation:

* Output A: **fridge_on** (main)
* Output B: **ac_on** (aux)

Total loss:

* `L = L_fridge + 0.3 * L_ac`

This pushes the representation to encode blower noise separately instead of confusing it with compressor.

### Optimization (defaults)

* Optimizer: **AdamW**
* Weight decay: `0.01`
* Gradient clip: `1.0`
* LR schedule: **5% warmup + cosine decay** (or fixed LR if you want simplicity)

### Two-stage fine-tune (best default with tiny data)

**Stage 1 (stabilize head)**

* Freeze all BEATs weights
* Train head (and aux head) for `5–10` epochs
* LR head: `1e-3`

**Stage 2 (adapt a little)**

* Unfreeze **only the last 2 transformer blocks + final norm** in BEATs
* LR backbone: `1e-5`
* LR head: `3e-4`
* Train `10–30` epochs with **early stopping on group-val**

Why partial unfreeze? Full fine-tune often overfits fast with your effective data size.

### Batch sizing

Use what fits, but don’t chase giant batches. Defaults:

* Batch: `32` windows (2.0 s each)
* If memory tight: batch `16` and gradient-accumulate ×2.

### Loss

* **BCEWithLogitsLoss**
* No class weighting needed if you generate similar #windows for on/off (you will).

---

## 5) Augmentations (the “no new recordings” lever)

These are safe + high-impact for your scenario:

### A) Gain jitter (mandatory)

* Random gain in `[-6, +6] dB`

### B) Time shift (mandatory)

* Randomly shift crop start within the window by up to `±0.25 s`

### C) Noise mixing from your own clips (high impact)

Build a noise pool from **fridge-off** windows and mix into both classes:

* With prob 0.5:

  * `x = x + α * noise`
  * Choose SNR uniformly in `[0, 20] dB`

Also do the reverse *sometimes* (hard negatives):

* Mix a low-level fridge-on snippet into fridge-off at high SNR (e.g., 15–25 dB) so the model learns the boundary between “barely on” and “off”.

### D) SpecAugment-style masking (optional but good)

If you can insert it on fbank features before the encoder:

* Time mask: 1–2 masks, up to ~10–20 frames
* Freq mask: 1 mask, up to ~8–16 mel bins

(Keep it light; too strong can erase compressor tonality.)

---

## 6) What to optimize/evaluate for *streaming boolean* (not just window accuracy)

Your real metric is “does the boolean flicker / miss long runs”.

During validation, compute:

* **Window AUROC**
* **Smoothed F1** after applying the same smoothing/hysteresis you’ll use live

That prevents you from selecting a model that looks good per-window but is unusable as a stable signal.

---

## 7) Streaming inference defaults (turn logits into a stable boolean)

Run a rolling buffer and smooth aggressively:

**Runtime windowing**

* Buffer: `2.0 s`
* Update every: `0.5 s` (or 1.0 s if you want low CPU)

**Probability smoothing**

* Convert logit → `p = sigmoid(logit)`
* EMA: `p_ema = 0.2 * p + 0.8 * p_ema_prev`

**Hysteresis thresholds (default)**

* Turn **ON** if `p_ema >= 0.65`
* Turn **OFF** if `p_ema <= 0.35`
* Otherwise keep previous state

These numbers are intentionally “sticky”. Then, *tune them using your group-held-out validation recordings* by sweeping thresholds to maximize **smoothed F1**.

---

## 8) Concrete default config (copy/paste as your starting point)

```yaml
audio:
  sample_rate: 16000
  mono: true
  trim_start_s: 10
  trim_end_s: 5

windows:
  train_win_s: 2.0
  train_hop_s: 1.0
  train_random_crop_s: [1.5, 3.0]   # pick length uniformly, then crop
  eval_win_s: 2.0
  eval_hop_s: 1.0

model:
  backbone: BEATs_iter3_plus_AS2M.pt
  pooling: mean
  head:
    mlp_hidden: 128
    dropout: 0.2
  multitask:
    enabled: true
    aux_ac_weight: 0.3

train:
  optimizer: adamw
  weight_decay: 0.01
  grad_clip: 1.0
  schedule: warmup_cosine
  warmup_frac: 0.05

  stage1:
    freeze_backbone: true
    epochs: 8
    lr_head: 1e-3

  stage2:
    unfreeze_last_blocks: 2
    epochs: 25
    lr_backbone: 1e-5
    lr_head: 3e-4
    early_stopping_patience: 5

augment:
  gain_db: [-6, 6]
  time_shift_s: 0.25
  mix_noise:
    p: 0.5
    snr_db: [0, 20]
  specaugment:
    enabled: true
    time_masks: 2
    time_mask_frames: 20
    freq_masks: 1
    freq_mask_bins: 12

streaming:
  buffer_s: 2.0
  update_every_s: 0.5
  ema_alpha: 0.2
  hysteresis_on: 0.65
  hysteresis_off: 0.35
```

---

If you want, I can also:

* sanity-check that your **“fridge on”** clips are actually stationary (no brief off cycles inside), and
* suggest the *best* train/val split for your 8 recordings based on which pair is hardest to separate (usually the best proxy for deployment performance).

[1]: https://proceedings.mlr.press/v202/chen23ag/chen23ag.pdf?utm_source=chatgpt.com "BEATS : Audio Pre-Training with Acoustic Tokenizers"
[2]: https://huggingface.co/spaces/fffiloni/SALMONN-7B-gradio/blob/b0c18cadf430eafa34de23d7e29453306b8bca86/beats/BEATs.py?utm_source=chatgpt.com "beats/BEATs.py · fffiloni/SALMONN-7B-gradio at ..."

---

Use a **grouped holdout by “environment”** (same AC + same day/night), and hold out the *hardest* environment.

I decoded your 8 clips to 16 kHz mono, trimmed 10 s head / 5 s tail, computed a mean log-PSD (Welch). The **smallest on/off separation is when AC is ON at night**, so that’s the best stress-test split.

### Recommended 6/2 split (primary)

**Validation (2 files):**

* `/mnt/data/ac-on-fridge-off-night.m4a`
* `/mnt/data/ac-on-fridge-on-night.m4a`

**Train (6 files):**

* `/mnt/data/ac-off-fridge-off-day.m4a`
* `/mnt/data/ac-off-fridge-on-day.m4a`
* `/mnt/data/ac-off-fridge-off-night.m4a`
* `/mnt/data/ac-off-fridge-on-night.m4a`
* `/mnt/data/ac-on-fridge-off-day.m4a`
* `/mnt/data/ac-on-fridge-on-day.m4a`

Why this is best: **AC-on increases the noise floor and makes fridge-on/off spectra look most similar** (night was the closest pair), so if you do well here, you’ll almost certainly do well on the easier conditions.

### If you can afford a bit more rigor (still tiny)

Do a 4-fold “environment holdout” where each fold validates on one pair:

* (ac-on, night)  ✅ hardest
* (ac-on, day)
* (ac-off, night)
* (ac-off, day)

Pick hyperparams on the average, then **train the final model on all 8 recordings** and keep the chosen thresholds/hysteresis for streaming.
