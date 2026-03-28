# Resonator Synth — Physics-Informed Grand Piano Synthesizer

A physics-informed additive synthesizer for grand piano sample bank generation.
Partial parameters (frequency, inharmonicity, bi-exponential decay, beating, stereo width)
are extracted from real recordings and used to drive a bank of sinusoidal oscillators.
A FastAPI + vanilla JS GUI provides session management, parameter editing, and generation.

---

## Quick Start

```bash
# 1. Install dependencies
.venv312/Scripts/pip install -r requirements.txt

# 2a. Classic pipeline: extract parameters from source sample bank
python analysis/extract_params.py
python analysis/compute_spectral_eq.py
python analysis/smooth_tau_cross_velocity.py --inplace --color-blend 0.7

# 2b. DDSP pipeline: learn parameters directly from WAV files (recommended)
python analysis/train_ddsp.py \
    --wav-dir C:/SoundBanks/IthacaPlayer/ks-grand \
    --out analysis/params_profile.json

# 3. (Optional) Smooth a sparse bank with spline interpolation
python analysis/interpolate_missing_notes.py

# 4. (Optional) Further smooth with NN instrument profile
python analysis/train_instrument_profile.py \
    --in analysis/params.json \
    --out analysis/params_profile.json

# 5. Launch GUI
python -m uvicorn gui.server:app --port 8989 --reload
# Open http://localhost:8989
```

---

## Architecture

### Synthesis model (`analysis/physics_synth.py`)

Each piano note is synthesised as a sum of sinusoidal partials with bi-exponential envelopes:

```
partial_k(t) = A0_k * [a * exp(-t/tau1_k) + (1-a) * exp(-t/tau2_k)] * cos(2π*f_k*t + φ_k)

f_k = k * f0 * sqrt(1 + B * k²)   # inharmonicity (stiffness coefficient B)
```

For notes with multiple strings, each string is rendered as a pair of slightly
detuned oscillators (`df` Hz apart), producing the characteristic amplitude beating.
Stereo imaging is derived from per-string M/S width factors extracted from the source.

### Analysis pipeline

| Script | Purpose |
|--------|---------|
| `analysis/extract_params.py` | Extracts f0, B (inharmonicity), partial A0/tau/df from WAV files → `params.json` |
| `analysis/compute_spectral_eq.py` | Computes LTASE-based spectral EQ + stereo width factor → stored in `params.json` |
| `analysis/smooth_tau_cross_velocity.py` | SNR-based cross-velocity tau smoothing + spectral colour blending |
| `analysis/interpolate_missing_notes.py` | Spline interpolation of missing notes for sparse banks |
| `analysis/train_instrument_profile.py` | NN smoothing of extracted parameters (classic pipeline) |
| `analysis/train_ddsp.py` | **DDSP end-to-end trainer**: learns parameters directly from WAV audio |

### DDSP Training (`analysis/train_ddsp.py`)

The DDSP (Differentiable DSP) trainer bypasses the extraction step entirely:

```
WAV files (88×8) → InstrumentProfile NN(midi, vel) → differentiable synthesis
                 → multi-scale STFT loss vs original WAV → backprop
```

Advantages over the classic extract → smooth pipeline:
- No extraction artefacts (noisy tau estimates, wrong partial tracking at low SNR)
- Trains on all available WAV files including those that failed extraction
- The NN is directly optimised to reproduce the audio spectrogram
- Single command replaces the three-step classic pipeline

Architecture details:
- Batched synthesis: 8 notes simultaneously via `[N, K, T]` tensor operations
- Segment length: 0.5 s (sufficient for spectral shape + early decay)
- Loss: log-magnitude STFT at two scales (n_fft 256 and 2048)
- ~11 s/epoch, 300 epochs ≈ 55 min on CPU

### InstrumentProfile NN (`analysis/train_instrument_profile.py`)

Factorised MLP shared by both training pipelines:

```
B_net        : MLP(midi)               → log(B)          inharmonicity
tau1_k1_net  : MLP(midi, vel)          → log(tau1) k=1   fundamental sustain
tau_ratio_net: MLP(midi, k)            → log(tau_k/tau1) decay ratio k>1
A0_net       : MLP(midi, k, vel)       → log(A0_ratio)   spectral shape
df_net       : MLP(midi, k)            → log(df)         beating frequency
dur_net      : MLP(midi)               → log(duration)
eq_net       : MLP(midi, freq)         → gain_db         body EQ
wf_net       : MLP(midi)              → log(width_factor)
```

Sinusoidal MIDI embedding captures register transitions:
`[m, sin(πm), sin(2πm), sin(4πm), cos(πm), cos(2πm)]`

The `tau1_k1_net` is dedicated to the fundamental decay (k=1 sustain).
The `tau_ratio_net` predicts `log(tau_k/tau_k1)` — physically constrained to ≤ 0
so no partial decays slower than the fundamental.

~72k parameters, trains in ~30 s (classic) or ~55 min (DDSP).

### GUI (`gui/`)

FastAPI backend + vanilla JS SPA, served on port **8989**.

```
gui/
  server.py          # FastAPI app, mounts routers, serves static files
  config_schema.py   # Parameter metadata (ranges, defaults, docs, groups)
  logger.py          # Rotating file handler → gui/logs/server.log
  routers/
    sessions.py      # Session CRUD, config get/put, per-note overrides, velocity profile
    generate.py      # Background generation job, status polling, file listing
    audio.py         # WAV serving, Welch spectrum endpoint
  static/
    index.html       # Single-page app
    app.js           # All UI logic (session management, sliders, spectrum, player)
    style.css        # RetroLCD LED-green theme
```

---

## GUI Workflow

1. **New Session** — name, source `params.json` path, instrument metadata
2. **Parameters panel** (left) — adjust global synthesis parameters, save
3. **Per-note overrides** — load a specific MIDI note, apply delta offsets
4. **Generate panel** (center) — select MIDI range and velocity layers, click Generate
5. **Instrument Velocity Profile** — compute RMS ratios from original WAV files
6. **Player + Spectrum** (right) — click any generated file to preview with spectrum

Sessions are stored in `gui/sessions/<name>/`:
- `config.json` — all parameters
- `params.json` — copy of source params (partial data)
- `generated/` — output WAV files + `instrument-definition.json`

---

## Synthesis Parameters

### RENDER

| Parameter | Default | Range | Description |
|-----------|---------|-------|-------------|
| `sr` | 44100 | 22050–48000 Hz | Sample rate |
| `duration` | auto | 0.5–10 s | Note duration; `null` = use duration from params.json |
| `fade_out` | 0.5 | 0–5 s | Fade-out at note end |
| `target_rms` | 0.06 | 0.01–0.25 | Output RMS for velocity 7 (≈ −24 dBFS) |
| `velocity_curve_gamma` | 0.7 | 0.2–2.0 | Velocity power curve: `rms = target_rms × ((vel+1)/8)^γ` |

### TIMBRE

| Parameter | Default | Range | Description |
|-----------|---------|-------|-------------|
| `harmonic_brightness` | 1.0 | 0–3 | Upper harmonic boost: `gain(k) = 1 + brightness × log₂(k)` |
| `beat_scale` | 1.0 | 0–3 × | Inter-string beating frequency multiplier (0 = mono/static) |
| `eq_strength` | 0.5 | 0–1 | Spectral EQ strength (LTASE correction); 0 = bypass |
| `eq_freq_min` | 400 | 50–2000 Hz | EQ lower cutoff; fades to flat below this |
| `soundboard_strength` | 0.0 | 0–1 | Modal IR convolution (parked; current IR causes band-pass artefacts) |
| `vel_color_blend` | 0.7 | 0–1 | Blend spectral colour toward reference velocity; preserves total energy |
| `vel_color_ref` | 4 | 0–7 | Reference velocity for colour blending (vel4 = SNR sweet spot) |

### STEREO

| Parameter | Default | Range | Description |
|-----------|---------|-------|-------------|
| `pan_spread` | 0.55 | 0–1.5 rad | Per-string pan angle spread; 0 = mono |
| `stereo_boost` | 1.0 | 0.5–4 × | M/S side-channel multiplier on top of extracted width_factor |

### PER-NOTE OVERRIDES (additive deltas)

| Parameter | Default | Range | Description |
|-----------|---------|-------|-------------|
| `harmonic_brightness_delta` | 0.0 | ±2 | Added to global harmonic_brightness for this note |
| `beat_scale_delta` | 0.0 | ±2 | Added to global beat_scale |
| `pan_spread_delta` | 0.0 | ±1 | Added to global pan_spread |
| `tau1_k1_scale` | 1.0 | 0.1–5 × | Multiplier for k=1 (fundamental) decay time tau1 |

---

## Tau Cross-Velocity Smoothing (`smooth_tau_cross_velocity.py`)

String decay times (tau) are material properties and should be velocity-independent.
Low-velocity extractions have poor SNR for weaker partials, producing unreliable tau values.

- Per-velocity reliability = `A0 / A0_max`
- If `reliability ≥ snr_threshold` (default 0.25): keep original tau
- If `reliability < snr_threshold`: blend toward amplitude-weighted-median reference tau

```bash
# Smooth tau + apply colour blend 0.7, overwrite params.json
python analysis/smooth_tau_cross_velocity.py --inplace --color-blend 0.7

# Preview before/after for MIDI note 64
python analysis/smooth_tau_cross_velocity.py --check 64
```

---

## Velocity Colour Blend

Each velocity layer is recorded from a different hammer strike energy, causing slightly
different spectral balances per velocity (higher velocity → more upper harmonics).
Combined with SNR-related extraction noise at lower velocities, this makes each layer
sound like a different instrument.

`vel_color_blend` transfers the spectral character of the reference velocity (default: vel4)
to all other layers, while preserving each layer's total energy:

```
shape_blended = (1 - blend) * shape_vel + blend * shape_ref
A0_new = shape_blended / ||shape_blended|| * ||A0_vec_vel||
```

`vel4` is the SNR sweet spot: strong enough signal for reliable extraction,
not so loud that hard hammer contact shifts spectral balance.

---

## Missing Note Interpolation (`interpolate_missing_notes.py`)

For sparse sample banks (not all 88 notes × 8 velocities recorded):

- Fits smoothing splines per parameter over available MIDI notes
- Handles: B, tau1/tau2 per k, A0 ratios, df, duration_s, spectral_eq, stereo_width_factor
- IQR outlier filtering before spline fitting
- Marks interpolated entries with `_interpolated: True`

The DDSP trainer (`train_ddsp.py`) supersedes this for banks where original WAV files
exist but extraction failed — it can train directly on any available WAV regardless.

---

## NN Instrument Profile (`train_instrument_profile.py`)

Classic pipeline: smooth noisy extracted params with a small NN.

```bash
python analysis/train_instrument_profile.py \
    --in  analysis/params.json \
    --out analysis/params_profile.json \
    [--epochs 800] [--plot]
```

Generates `profile.pt` (model weights) and `params_profile.json`.
By default preserves measured originals and only fills missing notes with NN output.
Use `--no-preserve-orig` to replace all samples with NN predictions.

---

## DDSP Training (`train_ddsp.py`)

End-to-end training directly from WAV files.

```bash
python analysis/train_ddsp.py \
    --wav-dir C:/SoundBanks/IthacaPlayer/ks-grand \
    --out     analysis/params_profile.json \
    --model   analysis/profile_ddsp.pt \
    [--epochs 300] [--seg 0.5] [--kmax 16] [--batch 8]

# Warm-start from extracted profile for faster convergence
python analysis/train_ddsp.py \
    --init analysis/profile.pt \
    --epochs 150
```

| Flag | Default | Description |
|------|---------|-------------|
| `--wav-dir` | `C:/SoundBanks/...` | Source WAV directory |
| `--epochs` | 300 | Training epochs (~11 s each) |
| `--seg` | 0.5 | Segment length in seconds |
| `--kmax` | 16 | Max partials per note |
| `--batch` | 8 | Notes per gradient step |
| `--init` | — | Warm-start from existing profile.pt |
| `--no-preserve-orig` | — | Replace all samples with NN (default: preserve originals) |

---

## Output Format

Generated files: `mXXX-velY-fZZ.wav`
- `XXX` — MIDI note number (021–108)
- `Y` — velocity layer (0–7)
- `ZZ` — sample rate code (44 = 44100 Hz, 48 = 48000 Hz)

`instrument-definition.json` is written to the `generated/` directory after each job:

```json
{
  "instrumentName": "KS Grand Piano",
  "velocityMaps": "8",
  "instrumentVersion": "1",
  "author": "n/a",
  "description": "n/a",
  "category": "Piano",
  "sampleCount": 704
}
```

---

## Data

Source: `C:/SoundBanks/IthacaPlayer/ks-grand/` — 704 WAV files, 88 MIDI notes × 8 velocity layers @ 44.1 kHz.

Papers: `C:/Users/jindr/OneDrive/Osobni/LordAudio/IhtacaPapers/` — 16 documents on physics-based piano modelling (Chabassier/Inria group, Simionato 2024 DDSP, Bank/Chabassier 2019 review).

---

## Project Structure

```
analysis/
  extract_params.py            # Partial parameter extraction from WAV
  compute_spectral_eq.py       # LTASE spectral EQ + stereo width
  smooth_tau_cross_velocity.py # Cross-velocity tau + colour correction
  interpolate_missing_notes.py # Spline interpolation for sparse banks
  train_instrument_profile.py  # NN smoothing of extracted parameters
  train_ddsp.py                # DDSP end-to-end trainer (WAV → params)
  physics_synth.py             # Core synthesis engine
  params.json                  # Extracted parameters (88 notes × 8 vel)
  params_profile.json          # NN-smoothed / DDSP-trained parameters
  profile.pt                   # Saved NN weights (train_instrument_profile)
  profile_ddsp.pt              # Saved NN weights (train_ddsp)

gui/
  server.py                    # FastAPI entry point (port 8989)
  config_schema.py             # Parameter metadata and defaults
  routers/                     # API endpoints
  static/                      # Frontend (index.html, app.js, style.css)
  sessions/                    # Per-session configs and generated files (gitignored)
  logs/                        # Server logs (gitignored)
```
