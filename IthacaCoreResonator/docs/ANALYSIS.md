# IthacaCoreResonator — Cross-Paper Analysis

Synthesis of 18 acoustic piano papers vs. current C++ implementation.
Each paper has a detailed report in `paper_compare/`.

---

## Implemented Principles

The following physics are correctly implemented and confirmed across multiple papers:

| Principle | Implementation | Papers |
|-----------|---------------|--------|
| Inharmonicity `f_k = k·f₀·√(1+B·k²)` | `note_lut.cpp`, B from `params.json` | Bensa 2003, Teng 2012, Simionato 2024, all Chabassier papers |
| Bi-exponential decay `a1·e^(-t/τ₁) + (1-a1)·e^(-t/τ₂)` | `resonator_voice.cpp` | Bank 2019, Chabassier 2013, Piano-model-revis, all Inria reports |
| Inter-string beating `f_k ± beat_hz/2` per string | `resonator_voice.cpp`, `STRING_SIGNS` | Bensa 2003, Chabassier 2012, Teng 2012, Piano-Chaigne-SMAC13 |
| Frequency-dependent decay (τ per partial) | `PartialParams.tau1/tau2` | Bensa 2003, Bank 2019, all Inria RR papers |
| Multi-string choir (n_strings = 1/2/3) | `n_strings_` in voice | Chabassier 2012, Piano-model-revis |
| Per-note spectral EQ from recorded data | `BiquadEQ`, `eq_gains_db[64]` | MAESSTRO, Bank 2019 |
| Velocity layer interpolation (8 layers) | `interpolateNoteLayers()` | Teng 2012, Simionato 2024 |
| Noise at attack (LP-filtered, decaying) | `resonator_voice.cpp` noise block | Piano-Chaigne-SMAC13, Teng 2012 |
| Stereo: per-string pan + M/S width | `pan_l/r`, `width_factor` | — (beyond most papers) |
| Schroeder all-pass decorrelation | `biquad_eq.cpp` | — (beyond most papers) |

---

## Missing Principles — Priority Order

### 1. Phantom Partials (longitudinal waves)

**Cited by:** Chabassier/Chaigne/Joly 2013, RR-8181, RR-9516, SubM2AN, Simionato 2024 (×3 papers), Bank/Chabassier 2019, Piano-model-revis — **8 of 18 papers**

**Physics:** Geometric nonlinearity of the string couples transverse and longitudinal vibrations.
Longitudinal modes appear at `f_k^L ≈ 2·f_k` with `τ^L ≈ τ/2`.
Also cross-product combinations `f_i ± f_j` (stronger in bass/forte).

**Perceptual impact:** Characteristic "zing" of the attack, especially in bass notes (MIDI < 48) and at forte dynamics. Clearly audible in recordings, absent in current synthesis.

**Implementation path** (additive synth — no PDE needed):
1. In `extract_params.py`: detect longitudinal peaks at `2·f_k` in the spectrum; fit `A0_L`, `tau_L`.
2. Add to `PartialParams`: `float A0_long; float tau_long;` (or store as extra partial entries with a `is_longitudinal` flag).
3. In `resonator_voice.cpp`: render longitudinal partials as single-string (mono), no beating, `f = 2·f_k`.

---

### 2. Velocity-Dependent Spectral Shape

**Cited by:** HammerJSVV3 (Chabassier/Duruflé 2014), Piano-Chaigne-SMAC13 (Chaigne 2013), Teng 2012, Bank 2019 — **5 of 18 papers**

**Physics:** Hammer shank flexibility causes spectral differences of 10–25 dB around 600–1100 Hz between legato and staccato at equal loudness. Forte has spectral width ~7 kHz vs. ~5 kHz for piano.

**Current state:** `vel_gamma` scales amplitude only — forte is louder piano, not spectrally different.

**Implementation path:**
- `eq_gains_db[EQ_BANDS]` is already per-velocity-layer.
  Currently only `A0` is interpolated; interpolate `eq_gains_db` across velocity layers too.
- `extract_params.py` already computes EQ per layer — wire interpolation at `noteOn`.
- Cost: 8 floats interpolated at note-on, zero audio-path cost.

---

### 3. Soundboard Modal Transients

**Cited by:** Breve (Chabassier 2013), MAESSTRO, RR-8181, SubM2AN, Piano-model-revis — **5 of 18 papers**

**Physics:** The soundboard has ~2400 modal frequencies. At note-on, the bridge impulse excites these modes; they ring briefly (~50–200 ms), especially below 1 kHz. Current `BiquadEQ` approximates the steady-state spectrum but not the time-varying modal texture at attack.

**Perceptual impact:** Missing "body" at attack onset; low notes sound thin in the first 50–100 ms.

**Implementation path (approximate, RT-compatible):**
- Add 6–12 IIR bandpass resonators per voice, tuned to dominant soundboard modes (shared across notes, constant per instrument).
- Excited by a brief impulse at `noteOn`, decay independently with τ ≈ 50–300 ms.
- Parameters could be globally fixed (not per-note) or extracted from low-frequency EQ residuals.

---

### 4. Physical Hammer Model

**Cited by:** HammerJSVV3, ChabassierChaigneJoly, RR-8181, SubM2AN — **4 of 18 papers**

**Physics:** Hertz nonlinear contact force `F = K·(x_H - x_S)^p` (p ≈ 2.5–3.5). Hammer shank flexibility (Timoshenko beam) determines spectral brightness. Contact duration ≈ 2–5 ms and its dependence on velocity is the primary source of timbre change with dynamics.

**Current state:** 3 ms linear onset ramp — click prevention only, not physical.

**Note:** Full physical hammer model requires PDE solver (not RT). The practical benefit is captured by velocity-dependent spectral shape (item 2 above), which is implementable.

---

### 5. Pitch Glide at Attack

**Cited by:** RR-9516, RR-8181, Simionato 2024 — **3 of 18 papers**

**Physics:** At large amplitudes (forte), geometric nonlinearity causes `f₀` to start slightly high and fall within the first 50–200 ms as amplitude decreases.

**Implementation path:**
- Add a brief frequency modulation: `f₀ *= (1 + pitch_glide · env_total)` where `env_total` is the overall amplitude envelope at time `t`.
- `pitch_glide` would be a new `SynthConfig` parameter (small positive value, ~0.001–0.005).
- Applied only when velocity > threshold (forte only).

---

### 6. Weinreich Coupled-String Double Decay

**Cited by:** Piano-model-revis, Bensa 2003 — **2 of 18 papers**

**Physics:** Strings coupled via the bridge produce asymmetric double decay: fast initial decay (energy transferred to soundboard) followed by very slow decay (string re-absorbs energy). The ratio of fast/slow τ depends on bridge admittance.

**Current state:** `tau1/tau2/a1` are fit from recorded data and correctly capture the *result*, but the two τ values are symmetric per string. A Weinreich model would fit two decay rates from per-polarization measurements.

**Implementation path:** Refine `extract_params.py` to fit `tau1/tau2` from the actual double-decay visible in the EDC (Schroeder integral), rather than from power spectrum fitting.

---

### 7. Longitudinal Precursor at Bridge

**Cited by:** RR_9530 (Chabassier 2023) — **1 paper, specific to bass**

**Physics:** Longitudinal waves travel ~6× faster than transverse. For D1 (f₀ = 36 Hz), the longitudinal wave reaches the bridge ~10 ms before the transverse wave — a distinct precursor click audible in low notes.

**Implementation path:** Add a short noise burst at `noteOn` for MIDI < 50, duration ~2–10 ms scaled by string length (1/f₀).

---

## Summary Table

| Priority | Missing Principle | Papers | Effort | Perceptual Impact |
|----------|-------------------|--------|--------|-------------------|
| 1 | Phantom partials (f = 2·f_k, τ/2) | 8/18 | Medium | High — bass/forte |
| 2 | Velocity-dependent spectral EQ | 5/18 | Low | High — all dynamics |
| 3 | Soundboard modal transients | 5/18 | Medium | Medium — attack body |
| 4 | Physical hammer → spectral shape | 4/18 | (covered by #2) | — |
| 5 | Pitch glide at forte | 3/18 | Low | Medium — forte bass |
| 6 | Weinreich double decay (better τ fit) | 2/18 | Low | Low-Medium |
| 7 | Longitudinal precursor (bass only) | 1/18 | Low | Low — bass notes |

---

## Recommended Next Steps

**Phase A — Low effort, high impact:**
1. Wire `eq_gains_db` velocity interpolation at `noteOn` (velocity-dependent timbre).
2. Add pitch glide parameter to `SynthConfig` and apply at forte.

**Phase B — Medium effort, high impact:**
3. Add phantom partial extraction to `extract_params.py` and render in `resonator_voice.cpp`.

**Phase C — Architecture consideration:**
4. Reparametrize `tau1/tau2` as physical `b1/b3` coefficients for Phase 2 DDSP training and Phase 3 latent space interpolation (Simionato 2024 model).
5. For latent space (Phase 3): model instrument interpolation via soundboard mobility proxy, not raw param vectors (MAESSTRO recommendation).
