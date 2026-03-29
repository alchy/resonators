# IthacaCoreResonator — Implementation Plan

Physics-based real-time piano synthesizer engine.
Replaces WAV sample playback (IthacaCore) with additive synthesis from
a pre-computed parameter table (`params-nn-profile-{bank}.json`).

---

## Architecture Overview

```
MIDI note-on (midi, vel)
        │
        ▼
   NoteLUT[midi-21][vel]          ← loaded from params.json at startup
        │  PartialParams[K]
        │  SpectralEQCurve
        ▼
  ResonatorVoice::noteOn()
        │
        ▼
  ResonatorVoice::processBlock()  ← called each audio buffer (256 samples)
   ├─ envelope: bi-exp decay coefficients (pre-computed at note-on)
   ├─ oscillators: K partials × N_strings (SIMD AVX2)
   │   inharmonic freq: f_k = k·f0·√(1 + B·k²)
   │   beating: f_ki = f_k + beat_hz[k] × string_detune[str]
   ├─ spectral EQ: biquad cascade (8 bands, designed from EQCurve at note-on)
   └─ stereo: Mid = mono sum, Side = mono × width_factor → L=M+S, R=M-S
        │
        ▼
  VoiceManager::processBlockUninterleaved(L, R, n)
        │  sum active voices, apply limiter + BBE
        ▼
  audio output (float32 stereo)
```

---

## Component Map

### New (physics synth)

| File | Responsibility |
|------|----------------|
| `synth/note_params.h` | `PartialParams`, `NoteParams`, `NoteParamLUT` structs |
| `synth/note_lut.h/.cpp` | Load `params.json` → `note_lut[88][8]` via nlohmann/json |
| `synth/resonator_voice.h/.cpp` | Single voice: oscillators + envelope + EQ + stereo |
| `synth/resonator_voice_simd.cpp` | AVX2-vectorized inner loop (optional, platform-specific) |
| `synth/voice_manager.h/.cpp` | Polyphony pool (up to 88 voices), note-on/off, sustain pedal |
| `synth/biquad_eq.h/.cpp` | Design + apply biquad cascade from log-spaced EQ gains |

### Reused from IthacaCore

| File | Notes |
|------|-------|
| `sampler/core_logger.h/.cpp` | RT-safe ring-buffer logger |
| `dsp/dsp_effect.h` | Base DSP effect interface |
| `dsp/dsp_chain.h/.cpp` | Serial DSP chain |
| `dsp/bbe/*` | BBE harmonic enhancer |
| `dsp/limiter/*` | Soft limiter |

### Removed vs. IthacaCore

- `InstrumentLoader` — no WAV files
- `SampleRateConverter` — synthesis at native sample rate
- `EnvelopeStaticData` / ADSR — replaced by bi-exponential decay
- `libsndfile` submodule — not needed (keep for WAV export tests only)
- `speexdsp` submodule — not needed

---

## Key Data Structures (`synth/note_params.h`)

```cpp
static constexpr int MAX_PARTIALS = 64;
static constexpr int MAX_STRINGS  = 3;
static constexpr int EQ_BANDS     = 8;

struct PartialParams {
    float f_hz;                         // inharmonic frequency (Hz)
    float A0;                           // initial amplitude
    float tau1, tau2;                   // bi-exp decay time constants (s)
    float a1;                           // bi-exp mixing weight (fast component)
    float beat_hz;                      // inter-string beating (Hz)
    float beat_depth;                   // beating depth (0..1, currently unused)
    bool  mono;                         // single-string (no stereo spread)
};

struct NoteParams {
    float    f0_hz;                     // fundamental frequency
    float    B;                         // inharmonicity coefficient
    float    f0_offset_cents;           // fine tuning
    float    width_factor;              // M/S stereo width
    float    noise_floor;               // noise level relative to A0_sum
    float    centroid_hz;               // noise LP filter cutoff
    float    attack_tau_s;              // noise envelope decay
    int      n_partials;                // valid entries in partials[]
    int      n_strings;                 // 1, 2, or 3
    PartialParams partials[MAX_PARTIALS];
    float    eq_gains_db[EQ_BANDS];     // log-spaced 80..16000 Hz gains
    bool     valid;
};

// Pre-loaded at startup — zero allocation in audio path
using NoteParamLUT = std::array<std::array<NoteParams, VEL_LAYERS>, MIDI_COUNT>;
```

---

## Voice Rendering

### Envelope (no `expf()` in audio loop)

```cpp
// At noteOn: pre-compute per-sample decay multipliers
for (int k = 0; k < n_partials; k++) {
    decay1[k] = expf(-1.f / (p.partials[k].tau1 * sample_rate));
    decay2[k] = expf(-1.f / (p.partials[k].tau2 * sample_rate));
    env1[k]   = a1[k]       * amplitude[k];
    env2[k]   = (1-a1[k])   * amplitude[k];
}

// processBlock inner loop (per sample):
//   env1[k] *= decay1[k];
//   env2[k] *= decay2[k];
//   float amp = env1[k] + env2[k];
```

### SIMD Strategy (AVX2)

Inner loop processes 8 partials simultaneously:
- Phase accumulation: 8× `_mm256_add_ps`
- `cosf` approximation: degree-5 polynomial (error < 1e-5, no libm)
- Envelope multiply: 8× `_mm256_mul_ps`

Estimated: ~1 µs for 64 partials × 3 strings per 256-sample buffer.

### A0 Normalization

At `noteOn`, amplitudes are normalized before rendering:
```cpp
float A0_ref = partials[0].A0;          // first partial as reference
float sum_sq  = Σ (A0_k / A0_ref)²;    // expected instantaneous power
float level_scale = target_rms * sqrt(2) / sqrt(sum_sq) * vel_gain;
```

This matches the Python `physics_synth.py` normalization (E[cos²(φ)] = 0.5 for random phases).

---

## Spectral EQ (`synth/biquad_eq.h`)

8-band biquad cascade designed at note-on from `eq_gains_db[EQ_BANDS]`:

| Band | Center Hz |
|------|-----------|
| 0    | 80        |
| 1    | 160       |
| 2    | 320       |
| 3    | 640       |
| 4    | 1250      |
| 5    | 2500      |
| 6    | 5000      |
| 7    | 12000     |

Each band: peaking EQ biquad, Q=1.4. Cost: 8 × (5 muls + 4 adds) per sample.

---

## Build System

```cmake
cmake_minimum_required(VERSION 3.16)
project(IthacaCoreResonator VERSION 1.0.0 LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)

add_executable(IthacaCoreResonatorGUI
    main.cpp
    gui/resonator_gui.cpp
    synth/note_lut.cpp
    synth/resonator_voice.cpp
    synth/voice_manager.cpp
    synth/resonator_engine.cpp
    synth/biquad_eq.cpp
    sampler/core_logger.cpp
    dsp/dsp_chain.cpp
    dsp/bbe/...
    dsp/limiter/...
)

# SIMD — AVX2 on x86_64
if (MSVC)
    target_compile_options(... PRIVATE /arch:AVX2)
else()
    target_compile_options(... PRIVATE -mavx2 -mfma)
endif()
```

Dependencies (all header-only or submodule, no package manager):
- `nlohmann/json` — `third_party/json.hpp` (MIT)
- `miniaudio` — audio I/O (single-header, MIT)
- `imgui` — GUI (submodule)
- `glfw` — window/input (submodule)

---

## Implementation Phases

### Phase 0 — Parameter extraction (complete)
- [x] `analysis/extract_params.py` — extract physics params from WAV bank
- [x] `analysis/params.json` — 88 × 8 velocity layers, per-partial PartialParams
- [x] LTASE spectral EQ method, window resolution analysis

### Phase 1 — Core synthesis (complete)
- [x] `synth/note_params.h` — structs
- [x] `synth/note_lut.cpp` — parse params.json → NoteLUT, interpolate missing notes
- [x] `synth/resonator_voice.cpp` — scalar oscillator + bi-exp envelope + EQ + stereo
- [x] `synth/biquad_eq.cpp` — 8-band peaking EQ designer
- [x] A0 normalization + target_rms level calibration
- [x] Onset ramp (post-EQ), noise model

### Phase 2 — Polyphony + MIDI (complete)
- [x] `synth/voice_manager.cpp` — pool, note-on/off, sustain pedal
- [x] `synth/resonator_engine.cpp` — audio callback, miniaudio integration
- [x] DSP chain: BBE + limiter on master bus

### Phase 3 — GUI (complete)
- [x] `gui/resonator_gui.cpp` — ImGui frontend
- [x] Piano keyboard display, voice matrix
- [x] Per-note parameter display (NoteParams from LUT)
- [x] Synthesis config panel with live edit
- [x] Peak metering, seed display

### Phase 4 — Physics refinement (planned)
- [ ] Velocity-dependent spectral EQ interpolation (see `docs/ANALYSIS.md` §2)
- [ ] Phantom partials: longitudinal waves at 2·f_k (see `docs/ANALYSIS.md` §1)
- [ ] Pitch glide at forte (see `docs/ANALYSIS.md` §5)

### Phase 5 — Differentiable training (future)
- [ ] Reparametrize τ as physical b1/b3 coefficients
- [ ] DDSP training loop (Python/PyTorch)
- [ ] Multi-instrument latent space (Steinway ↔ Bösendorfer)

### Phase 6 — Plugin wrapper (future)
- [ ] JUCE AudioProcessor wrapping ResonatorVoiceManager
- [ ] VST3 / CLAP export

---

## Comparison: IthacaCore vs. IthacaCoreResonator

| Aspect | IthacaCore | IthacaCoreResonator |
|--------|-----------|---------------------|
| Sound source | WAV samples (PCM playback) | Additive synthesis from params |
| Per-note data | `float* pcm[8]` (MB of audio) | `NoteParams[8]` (~4 KB total) |
| Memory footprint | Hundreds of MB (WAV bank) | < 1 MB (param table) |
| Envelope | ADSR | Bi-exponential decay (physics) |
| Tuning | Fixed (recorded pitch) | Exact physics (inharmonicity) |
| Velocity layers | 8 discrete samples | 8 discrete param sets + interp |
| Beating / chorus | None | Explicit per-partial beat_hz |
| Stereo | Fixed (recorded) | Parametric M/S + per-string pan |
