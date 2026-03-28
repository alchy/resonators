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

# 2. Extract partial parameters from source sample bank
python analysis/extract_params.py

# 3. Compute spectral EQ + stereo width factor for all samples
python analysis/compute_spectral_eq.py

# 4. (Optional) Smooth tau cross-velocity inconsistencies
python analysis/smooth_tau_cross_velocity.py --inplace --color-blend 0.7

# 5. Launch GUI
python -m uvicorn gui.server:app --port 8989 --reload
# Open http://localhost:8989
```

---

## Architecture

### Synthesis model (`analysis/physics_synth.py`)

Each piano note is synthesized as a sum of sinusoidal partials with bi-exponential envelopes:

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
| `analysis/smooth_tau_cross_velocity.py` | Post-processes `params.json` to correct per-velocity tau inconsistencies |

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
5. **Player + Spectrum** (right) — click any generated file to preview with spectrum

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
| `velocity_curve_gamma` | 0.7 | 0.2–2.0 | Velocity power curve exponent: `rms = target_rms × ((vel+1)/8)^γ` |

### TIMBRE

| Parameter | Default | Range | Description |
|-----------|---------|-------|-------------|
| `harmonic_brightness` | 1.0 | 0–3 | Upper harmonic boost: `gain(k) = 1 + brightness × log₂(k)` |
| `beat_scale` | 1.0 | 0–3 × | Inter-string beating frequency multiplier (0 = mono/static) |
| `eq_strength` | 0.5 | 0–1 | Spectral EQ strength (LTASE correction); 0 = bypass |
| `eq_freq_min` | 400 | 50–2000 Hz | EQ lower cutoff; fades to flat below this to protect the fundamental |
| `soundboard_strength` | 0.0 | 0–1 | Modal IR convolution (parked; current IR causes band-pass artifacts) |
| `vel_color_blend` | 0.7 | 0–1 | Blend spectral color (A0 ratios) toward reference velocity; preserves total energy |
| `vel_color_ref` | 4 | 0–7 | Reference velocity for color blending (vel4 = SNR sweet spot) |

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

## Velocity Color Blend

Each velocity layer is recorded from a different hammer strike energy, causing the
partial analysis to extract slightly different spectral balances per velocity
(higher velocity → more upper harmonics, different hammer contact time). Combined with
SNR-related extraction noise at lower velocities, this can make each layer sound like a
different instrument.

`vel_color_blend` transfers the spectral character of the reference velocity (default: vel4)
to all other layers, while preserving each layer's total energy:

```
shape_blended = (1 - blend) * shape_vel + blend * shape_ref
A0_new = shape_blended / ||shape_blended|| * ||A0_vec_vel||
```

`vel4` is the SNR sweet spot: strong enough signal for reliable partial extraction,
but not so loud that hard hammer contact shifts spectral balance.

---

## Tau Cross-Velocity Smoothing (`smooth_tau_cross_velocity.py`)

String decay times (tau) are material properties and should be velocity-independent.
However, low-velocity extractions have poor SNR for weaker partials, producing
unreliable tau values. The smoothing script corrects this:

- For each (MIDI, partial k): compute amplitude-weighted median tau across all 8 velocities
- Per-velocity reliability = `A0 / A0_max`
- If `reliability ≥ snr_threshold` (default 0.25): keep original tau (trustworthy extraction)
- If `reliability < snr_threshold`: blend toward reference tau
  `tau_final = (1 - blend_weight) * tau_orig + blend_weight * tau_ref`

```bash
# Smooth tau + apply color blend 0.7, overwrite params.json
python analysis/smooth_tau_cross_velocity.py --inplace --color-blend 0.7

# Preview before/after for MIDI note 64
python analysis/smooth_tau_cross_velocity.py --check 64

# Custom thresholds
python analysis/smooth_tau_cross_velocity.py --snr-threshold 0.3 --color-blend 0.5 --color-ref 5
```

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
  extract_params.py          # Partial parameter extraction from WAV
  compute_spectral_eq.py     # LTASE spectral EQ + stereo width
  smooth_tau_cross_velocity.py  # Cross-velocity tau + color correction
  physics_synth.py           # Core synthesis engine
  params.json                # Extracted parameters (88 notes × 8 vel)

gui/
  server.py                  # FastAPI entry point (port 8989)
  config_schema.py           # Parameter metadata and defaults
  routers/                   # API endpoints
  static/                    # Frontend (index.html, app.js, style.css)
  sessions/                  # Per-session configs and generated files (gitignored)
  logs/                      # Server logs (gitignored)
```
