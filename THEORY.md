Based on your 4 “steady-state” clips (on/off × day/night), this is an *easy* acoustic separation in your apartment: the “compressor on” recordings have **~10× higher high-frequency band energy (≈800–3000 Hz)** than “off”, and it’s stable across the full 3 minutes. That’s great news: you should be able to get near-perfect clip accuracy with a small head, and then focus your effort on **leak-free splitting** + **streaming post-processing** so the live boolean is stable.

Below are concrete, “best-accuracy” defaults for **data prep**, **splits**, **training**, and **streaming inference** with `BEATs_iter3_plus_AS2M.pt`.

---

## 1) Data prep (match BEATs exactly)

### 1.1 Decode + resample

* **Mono, 16 kHz**, float32 waveform in `[-1, 1]`
* Don’t apply denoise/AGC. For streaming, use the same “raw mic” path you used when recording (Mac audio processing mismatches are a common silent failure mode).

### 1.2 Use BEATs’ expected frontend (fbank)

BEATs’ common implementation computes **Kaldi fbank** with:

* `num_mel_bins=128`
* `sample_frequency=16000`
* `frame_length=25 ms`
* `frame_shift=10 ms`
* then normalizes with the **official preset mean/std**: `mean=15.41663`, `std=6.55582`, and (importantly) divides by `2*std` (so the resulting features land around ~0.5 std). ([SpeechBrain][1])

**Recommendation:** don’t reinvent this—call the BEATs `preprocess()` / feature path (or copy these exact params) so training/inference are identical. ([SpeechBrain][1])

---

## 2) Turn 4 long recordings into a dataset (without “fake” leakage)

You’ll generate lots of training examples by chunking, but you must avoid “neighbor window leakage” in validation/testing.

### 2.1 Windowing defaults (optimized for streaming latency + accuracy)

Use fixed windows with overlap:

**Training windows**

* `window_len = 2.0 s`
* `hop = 0.5 s` (75% overlap is fine too, but 0.5s is a good default)
* This yields ~**357 windows per 3-minute file** (so ~1400 total).

**Streaming windows**

* same `window_len = 2.0 s`
* smaller hop: `hop = 0.25 s` (4 predictions/sec) for faster detection
* (You’ll smooth/hysteresis after; see §5)

Why 2s? Your compressor signature is steady and separable even with short context; 2s is a good tradeoff between **latency** and **robustness**.

### 2.2 Labels

Every window inherits the recording label:

* `on = 1` for `fridge-on-*.m4a`
* `off = 0` for `fridge-off-*.m4a`

Keep all “incidental noises” inside those recordings; that’s your only built-in robustness to random apartment sounds.

### 2.3 Normalization / loudness

Don’t per-window normalize (it can erase the very signal you want).
Instead:

* keep waveform scale as captured
* add **random gain augmentation** during training (below) to make the classifier invariant to small level drift.

---

## 3) Splits that actually measure generalization

If you randomly split windows, you’ll get a meaningless 99–100% because adjacent windows are nearly identical.

### 3.1 Use *recording-grouped* evaluation: “day ↔ night” 2-fold CV

Do **two folds**:

**Fold A**

* Train: `fridge-on-day` + `fridge-off-day`
* Validate/Test: `fridge-on-night` + `fridge-off-night`

**Fold B**

* Train: `fridge-on-night` + `fridge-off-night`
* Validate/Test: `fridge-on-day` + `fridge-off-day`

This directly answers: “does my model still work when the ambient apartment sound changes?”

### 3.2 Final training for deployment

After you pick hyperparams via the two folds, train one final model on **all four recordings** (all windows), then set your **decision threshold + hysteresis** using a small held-out slice (e.g., last 15 seconds of each file) *or* using one fold’s validation predictions.

---

## 4) Training recipe (small data → freeze-first, then tiny fine-tune)

### 4.1 Head architecture (strong default)

Input: BEATs last-layer features `(B, T, C)`.

**Pooling**

* start with **mean pooling over time** (simple + works well for steady-state)
* if you want to squeeze extra accuracy: attention pooling (but mean is usually enough here)

**Classifier head**

* `Linear(C → 256) + GELU + Dropout(0.2) + Linear(256 → 1)`
* Loss: `BCEWithLogitsLoss`

### 4.2 Optimization defaults (best chance of not overfitting)

**Phase 1: train head only (recommended starting point)**

* Freeze BEATs backbone (common strategy for small labeled data) ([GitHub][2])
* Optimizer: `AdamW`
* LR: `1e-3` (head)
* Weight decay: `1e-2`
* Batch size: `64` windows
* Epochs: `20` max with early stopping (patience 5) on **fold validation AUC/F1**

**Phase 2: if (and only if) fold performance isn’t perfect**

* Unfreeze last **2 transformer blocks** (keep earlier blocks frozen)
* Discriminative LRs:

  * backbone: `1e-5`
  * head: `3e-4`
* Warmup: 5% steps, cosine decay
* Epochs: 10–20 with early stopping

### 4.3 Augmentation (the most valuable thing you can do without new recordings)

Keep it label-preserving and targeted:

**Always-on augmentations**

* Random gain: ±6 dB
* Random time shift within the window (circular shift or crop jitter)
* Optional: small EQ tilt (very mild) to reduce mic-response brittleness

**If you care about false positives from “other apartment noises”**

* Add external noise at random SNR (e.g., 10–30 dB) to BOTH classes:

  * for OFF: teaches “random noise ≠ compressor”
  * for ON: teaches “compressor persists under noise”
* Use any permissively licensed noise set (speech, footsteps, dishes). This isn’t “collecting more recordings” from your apartment, but it *does* help robustness.

Avoid mixup between on/off (it creates physically ambiguous labels).

---

## 5) Turning per-window probabilities into a rock-solid boolean stream

Even a perfect classifier will “flutter” on a live mic if you emit raw logits.

### 5.1 Smoothing

Compute `p_t = sigmoid(logit_t)` each hop, then:

* EMA: `p̂_t = 0.8 * p̂_{t-1} + 0.2 * p_t`

### 5.2 Hysteresis + debounce (concrete defaults)

* Turn **ON** if `p̂_t > 0.7` for **3 consecutive frames**
* Turn **OFF** if `p̂_t < 0.3` for **6 consecutive frames**

This gives you:

* fast-ish “on” detection (~0.75s if hop=0.25s)
* more conservative “off” detection (reduces chatter)

If you’d rather miss short cycles than ever false-alarm, raise the ON threshold to 0.8 and/or require 4–5 frames.

---

## 6) A quick “sanity baseline” (useful even if you go full BEATs)

In your current steady-state clips, a simple feature like **log bandpower in ~800–3000 Hz** separates ON vs OFF extremely well (order-of-magnitude gap). You don’t have to deploy this baseline, but it’s excellent for:

* verifying your live audio path matches training (if the baseline suddenly collapses, you have an input mismatch)
* diagnosing false positives (e.g., a fan that mimics the compressor’s spectrum)

---

## “Best default” checklist (copy/paste)

* **Audio:** mono 16 kHz, no denoise/AGC
* **Frontend:** Kaldi fbank 128 mel, 25ms/10ms, normalize `(fbank-15.41663)/(2*6.55582)` ([SpeechBrain][1])
* **Windows:** 2.0s; train hop 0.5s; stream hop 0.25s
* **Splits:** grouped by recording; 2-fold day↔night CV
* **Model:** BEATs frozen + MLP head (256 hidden, dropout 0.2)
* **Opt:** AdamW, head LR 1e-3, wd 1e-2, bs 64, early stopping
* **Aug:** gain ±6 dB, time jitter; optional additive external noise 10–30 dB SNR
* **Postproc:** EMA(0.8) + hysteresis (ON>0.7×3, OFF<0.3×6)

If you want, I can also give you a minimal PyTorch training script layout + a tiny streaming loop structure (windowing + smoothing + hysteresis) that matches these defaults exactly.

[1]: https://speechbrain.readthedocs.io/en/develop/_modules/speechbrain/lobes/models/beats.html "speechbrain.lobes.models.beats — SpeechBrain 0.5.0 documentation"
[2]: https://github.com/fschmid56/PretrainedSED?utm_source=chatgpt.com "fschmid56/PretrainedSED"
