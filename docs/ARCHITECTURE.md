# Architecture

## Overview

Three-stage pipeline: analytical extraction → neural smoothing → additive synthesis.

```
┌─────────────────┐    ┌─────────────────────┐    ┌──────────────────────┐
│  WAV sample bank│───▶│  extract-params.py  │───▶│  params-{bank}.json  │
│  88 × 8 files   │    │  signal analysis     │    │  measured, sparse,   │
│  44.1 kHz       │    │  no NN              │    │  noisy at low vel    │
└─────────────────┘    └─────────────────────┘    └──────────┬───────────┘
                                                              │
                        ┌─────────────────────┐              │
                        │compute-spectral-eq.py│◀─────────────┤
                        │LTASE H(f) per note  │              │
                        │modifies in-place    │──────────────▶│ +spectral_eq
                        └─────────────────────┘              │
                                                              ▼
                        ┌────────────────────────────────────────────────┐
                        │  train-instrument-profile.py                   │
                        │  factorised NN: MLP per physical parameter     │
                        │  ~90 000 params, ~30 s CPU, 800 epochs         │
                        └────────────────────────────┬───────────────────┘
                                                     ▼
                        params-nn-profile-{bank}.json  (88 × 8 = 704 notes, complete)
                                                     │
                        ┌────────────────────────────▼───────────────────┐
                        │  physics_synth.py :: synthesize_note()         │
                        │  additive synthesis: inharmonicity + bi-exp    │
                        │  decay + multi-string beating + spectral EQ    │
                        └────────────────────────────────────────────────┘
```

---

## Synthesis model (`analysis/physics_synth.py`)

Each note = sum of sinusoidal partials with bi-exponential envelope and per-string beating.

```
partial_k(t) = A0_k · [a1·exp(-t/τ1_k) + (1-a1)·exp(-t/τ2_k)] · Σ_i cos(2π·f_k_i·t + φ_i)

f_k     = k · f0 · √(1 + B·k²)          inharmonicity (string stiffness)
f_k_i   = f_k + δ_i                      per-string frequency offset
Σ δ_i   = 0,  |δ_i| ~ beat_hz/2         beating emerges from superposition
```

Beating arises naturally from independent string oscillators — not from amplitude modulation. This gives full-depth modulation (0 to N_strings × A0) as opposed to the limited `(1 + depth·cos)` approximation.

**Strings per MIDI range:**

| Range | Strings | Note |
|-------|---------|------|
| MIDI 21–27 (A0–Eb1) | 1 | Bass monochord |
| MIDI 28–48 (E1–C3)  | 2 | Bichord (wound) |
| MIDI 49–108 (C#3+)  | 3 | Trichord |

Each string has an independent pan angle. The stereo field is derived from per-string panning + M/S width scaling + Schroeder all-pass decorrelation.

**Reference:** Chabassier et al. — string inharmonicity and multi-string coupling; Bank & Sujbert (2005) — partial tracking and beating model.

---

## Parameter extraction (`analysis/extract-params.py`)

Pure signal analysis, no neural network.

### Inharmonicity B

FFT peak detection → fit `f_k = k·f0·√(1+B·k²)` by least squares.

### Bi-exponential decay τ1, τ2, a1

Frequency-adaptive STFT window: target 20 bins/harmonic.

| MIDI | f₀ (Hz) | Window | Resolution |
|------|---------|--------|------------|
| A0 (21) | 27.5 | 32768 samples | 1.35 Hz/bin |
| A2 (45) | 110  | 8192 samples  | 5.38 Hz/bin |
| A4 (69) | 440  | 2048 samples  | 21.5 Hz/bin |

Per-partial amplitude envelope → bi-exponential LSQ fit → τ1 (fast attack decay), τ2 (slow sustain decay), a1 (mixing ratio).

### Beating Δf

Envelope modulation frequency extracted from Hilbert transform of per-partial amplitude.

### Noise model

Attack burst: exponential fit of the noise floor rise/fall → `attack_tau_s`, `centroid_hz`, `A_noise`.

### Spectral EQ (`analysis/compute-spectral-eq.py`)

LTASE method (Long-Term Average Spectral Envelope):

```
H(f) = LTASE_orig(f) / LTASE_synth(f)
```

1. LTASE = mean of amplitude spectra over all STFT frames (N_FFT=8192)
2. Ratio H(f) with regularisation (ε = 0.1% of peak)
3. 1/6-octave Gaussian smoothing
4. Normalised to 0 dB mean above 100 Hz
5. Resampled to 64 log-spaced points (20 Hz – 20 kHz)
6. `stereo_width_factor` = M/S ratio orig/synth (skip first 100 ms)

`eq_freq_min = 400 Hz` default: EQ fades to flat below 400 Hz, protecting fundamentals from low-frequency extraction noise.

**Reference:** Simionato (2024, ISMIR) — LTASE-based spectral correction for DDSP piano.

---

## NN instrument profile (`analysis/train-instrument-profile.py`)

Factorised network: each physical parameter has its own sub-network, trained simultaneously. All positive outputs in log-space (geometric MSE).

```
B_net         MLP(midi)           → log(B)                inharmonicity
dur_net       MLP(midi)           → log(duration)
tau1_k1_net   MLP(midi, vel)      → log(τ1) for k=1       fundamental sustain
tau_ratio_net MLP(midi, k)        → log(τk/τk1) ≤ 0       decay ratio (physical constraint)
A0_net        MLP(midi, k, vel)   → log(A0_ratio)          spectral shape
df_net        MLP(midi, k)        → log(beat_hz)           beating
eq_net        MLP(midi, freq)     → gain_db               body EQ
wf_net        MLP(midi)           → log(width_factor)      stereo width
noise_net     MLP(midi, vel)      → [log(τ_noise),
                                     log(centroid_hz),
                                     log(A_noise)]
biexp_net     MLP(midi, k, vel)   → [logit(a1), log(τ2/τ1)]
```

**MIDI embedding** (sinusoidal, captures register transitions):
`[m, sin(πm), sin(2πm), sin(4πm), cos(πm), cos(2πm)]` where `m = (midi−21)/87`

**Physical constraints:** `tau_ratio ≤ 0` (no partial decays slower than fundamental); `τ2 > τ1` always (bi-exp ordering).

~90 000 parameters. Training ~30 s CPU (800 epochs, Adam, lr=0.003).

**Reference:** Simionato (2024) — factorised NN for piano parameter prediction; Chabassier/Inria group — physical constraints in piano modelling.

---

## File naming convention

```
analysis/params-{bank}.json               raw extracted  (extract-params.py output)
analysis/params-nn-profile-{bank}.json    NN-smoothed    (train-instrument-profile.py output)
analysis/profile.pt                       NN weights     (reusable for fine-tuning)

gui/sessions/{bank}/config.json           synth settings (render/timbre/stereo/vel_profile)
gui/sessions/{bank}/generated/            WAV output
  m{midi:03d}-vel{v}-f{sr_code}.wav
  instrument-definition.json

snapshots/{bank}-{YYYYMMDD-HHMM}/        timestamped archive (Snapshot NN button)
  params-nn-profile-{bank}.json
  config.json
```

---

## Papers referenced

Located in `C:/Users/jindr/OneDrive/Osobni/LordAudio/IhtacaPapers/` (16 documents):

- **Chabassier / Inria group** — string inharmonicity, multi-string coupling, soundboard interaction
- **Bank & Sujbert (2005)** — partial tracking, beating model, efficient piano synthesis
- **Bank & Chabassier (2019)** — review of piano synthesis methods
- **Simionato (2024, ISMIR)** — DDSP piano synthesis, LTASE spectral correction, NN parameter estimation
