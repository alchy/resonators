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

### Reused from IthacaCore (copy verbatim)

| File | Notes |
|------|-------|
| `sampler/core_logger.h/.cpp` | RT-safe ring-buffer logger — unchanged |
| `dsp/dsp_effect.h` | Base DSP effect interface — unchanged |
| `dsp/dsp_chain.h/.cpp` | Serial DSP chain — unchanged |
| `dsp/bbe/*` | BBE harmonic enhancer — unchanged |
| `dsp/limiter/*` | Soft limiter — unchanged |

### Not needed (removed vs IthacaCore)

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
static constexpr int EQ_BANDS     = 8;   // biquad cascade bands

struct PartialParams {
    float A0;                          // initial amplitude
    float tau1, tau2;                  // bi-exp decay time constants (s)
    float a1;                          // bi-exp mixing weight
    float beat_hz;                     // inter-string beating frequency (Hz)
    float beat_depth;                  // beating depth (0..1)
};

struct NoteParams {
    float    f0_hz;                    // fundamental frequency
    float    B;                        // inharmonicity coefficient
    float    f0_offset_cents;          // fine tuning
    float    width_factor;             // stereo width
    float    noise_floor;              // noise level relative to A0_sum
    int      n_partials;               // valid entries in partials[]
    int      n_strings;                // 1, 2, or 3
    PartialParams partials[MAX_PARTIALS];
    float    eq_gains_db[EQ_BANDS];    // log-spaced 80..16000 Hz gains
};

// Pre-loaded at startup — zero allocation in audio path
using NoteLUT = std::array<std::array<NoteParams, 8>, 88>;  // [midi-21][vel]
```

---

## Voice Rendering (`synth/resonator_voice.h`)

```cpp
class ResonatorVoice {
public:
    void noteOn(int midi, int vel, const NoteParams& p, float sample_rate);
    void noteOff();
    bool isActive() const;
    void processBlock(float* out_l, float* out_r, int n_samples);

private:
    // Per-partial state (allocated at construction, reused across notes)
    float phase[MAX_PARTIALS][MAX_STRINGS];       // oscillator phases
    float env1[MAX_PARTIALS], env2[MAX_PARTIALS]; // bi-exp state
    float decay1[MAX_PARTIALS], decay2[MAX_PARTIALS]; // per-sample multipliers

    // Computed at note-on (avoid division in audio loop)
    float freq[MAX_PARTIALS][MAX_STRINGS];        // inharmonic freqs (rad/sample)
    float pan_l[MAX_STRINGS], pan_r[MAX_STRINGS]; // string pan gains
    float a1[MAX_PARTIALS], a2[MAX_PARTIALS];     // envelope mix weights

    // Release fade (click prevention)
    float release_gain = 1.0f;
    float release_step = 0.0f;
    bool  releasing = false;

    // EQ (designed from NoteParams::eq_gains_db at note-on)
    BiquadEQ eq;

    int64_t sample_idx = 0;
    float   sample_rate = 44100.f;
    bool    active = false;
};
```

### Envelope computation at note-on (no expf() in audio loop)

```cpp
// At noteOn: pre-compute per-sample decay multipliers
for (int k = 0; k < n_partials; k++) {
    decay1[k] = expf(-1.f / (p.partials[k].tau1 * sample_rate));
    decay2[k] = expf(-1.f / (p.partials[k].tau2 * sample_rate));
    env1[k]   = p.partials[k].A0 * p.partials[k].a1;
    env2[k]   = p.partials[k].A0 * (1.f - p.partials[k].a1);
}

// In processBlock (inner loop, per sample):
//   env1[k] *= decay1[k];   // one multiply per partial per sample
//   env2[k] *= decay2[k];
//   float env = env1[k] + env2[k];
```

### SIMD strategy (AVX2)

Inner loop over K partials processes 8 partials simultaneously:
- `phase` accumulation: 8× `_mm256_add_ps`
- `cosf` approximation: polynomial (degree 5, error < 1e-5) — no libm call
- `env` multiply: 8× `_mm256_mul_ps`

Estimated: ~1 µs for 64 partials × 3 strings per 256-sample buffer (REALTIME.md estimate confirmed).

---

## Spectral EQ (`synth/biquad_eq.h`)

Design an 8-band biquad cascade at note-on from `eq_gains_db[EQ_BANDS]`:
- Band center frequencies: 80, 160, 320, 640, 1250, 2500, 5000, 12000 Hz
- Each band: peaking EQ biquad (Q=1.4)
- Total cost: 8 × 5 muls + 4 adds per sample per voice (negligible)

Alternative (future): 256-tap FIR via FFT overlap-add (REALTIME.md suggestion).
Start with biquad cascade — simpler, sufficient quality.

---

## JSON Loading (`synth/note_lut.cpp`)

Library: `nlohmann/json` (single-header, `third_party/json.hpp`)

```cpp
NoteLUT loadNoteLUT(const std::string& params_json_path) {
    // Parse params.json → samples["m060_vel3"] → PartialParams
    // note_lut[60-21][3] = parseNoteParams(sample_json)
    // Fill missing notes via linear interpolation of neighbours
}
```

Startup time: ~50–200 ms for 704 notes (acceptable, not in audio path).

---

## VoiceManager (`synth/voice_manager.h`)

Mirrors IthacaCore `VoiceManager` API — drop-in replacement:

```cpp
class ResonatorVoiceManager {
public:
    void initialize(const std::string& params_json, float sample_rate, Logger& logger);

    // MIDI interface (identical to IthacaCore)
    void setNoteStateMIDI(uint8_t midi, bool note_on, uint8_t velocity);
    void setSustainPedalMIDI(uint8_t val);
    void processBlockUninterleaved(float* L, float* R, int n_samples);

    // MIDI control (subset of IthacaCore)
    void setLimiterThresholdMIDI(uint8_t val);
    void setLimiterReleaseMIDI(uint8_t val);
    void setBBEDefinitionMIDI(uint8_t val);
    void setBBEBassBoostMIDI(uint8_t val);

private:
    NoteLUT           lut_;
    ResonatorVoice    voices_[88];          // one per MIDI note (88 notes max)
    DspChain          master_dsp_;          // limiter + BBE on master bus
    std::atomic<bool> sustain_pedal_{false};
    std::vector<uint8_t> held_notes_;       // notes waiting for pedal release
    float             sample_rate_;
    Logger*           logger_;
};
```

Polyphony: up to 88 simultaneous voices (all piano keys). At ~1 µs/voice, 88 voices ≈ 88 µs — well within 5.8 ms budget.

---

## Build System (CMakeLists.txt)

```cmake
cmake_minimum_required(VERSION 3.16)
project(IthacaCoreResonator VERSION 1.0.0 LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)

# Sources
add_executable(IthacaCoreResonator
    main.cpp
    synth/note_lut.cpp
    synth/resonator_voice.cpp
    synth/voice_manager.cpp
    synth/biquad_eq.cpp
    sampler/core_logger.cpp
    dsp/dsp_chain.cpp
    dsp/bbe/bbe_processor.cpp
    dsp/bbe/biquad_filter.cpp
    dsp/bbe/harmonic_enhancer.cpp
    dsp/limiter/limiter.cpp
)

target_include_directories(IthacaCoreResonator PRIVATE . third_party)

# SIMD — AVX2 on x86_64
if (CMAKE_SYSTEM_PROCESSOR MATCHES "AMD64|x86_64")
    if (MSVC)
        target_compile_options(IthacaCoreResonator PRIVATE /arch:AVX2)
    else()
        target_compile_options(IthacaCoreResonator PRIVATE -mavx2 -mfma)
    endif()
endif()
```

Dependencies (all header-only or submodule, no package manager):
- `nlohmann/json` — `third_party/json.hpp` (single header, MIT)
- No other external deps (no FFTW, no libsndfile, no speexdsp)

---

## Implementation Phases

### Phase 1 — Skeleton + JSON loading (1–2 days)
- [ ] `synth/note_params.h` — structs
- [ ] `synth/note_lut.cpp` — parse params.json → NoteLUT
- [ ] `main.cpp` — load LUT, print summary, exit
- [ ] CMakeLists.txt — build green on MSVC

### Phase 2 — Single voice synthesis (2–3 days)
- [ ] `synth/resonator_voice.cpp` — scalar oscillator + bi-exp envelope
- [ ] `synth/biquad_eq.cpp` — 8-band peaking EQ designer
- [ ] Offline render test: synthesize m060_vel3, export WAV, compare to original

### Phase 3 — Polyphony + MIDI interface (1–2 days)
- [ ] `synth/voice_manager.cpp` — pool, note-on/off, sustain pedal
- [ ] Copy DSP chain (BBE + limiter) from IthacaCore verbatim
- [ ] Interactive test: keyboard input → note trigger → WAV export

### Phase 4 — SIMD optimisation (1–2 days)
- [ ] `synth/resonator_voice_simd.cpp` — AVX2 inner loop
- [ ] Benchmark: confirm < 10 µs / voice / buffer

### Phase 5 — Plugin wrapper (future)
- [ ] JUCE AudioProcessor wrapping ResonatorVoiceManager
- [ ] VST3 / CLAP export

---

## Differences from IthacaCore

| Aspect | IthacaCore | IthacaCoreResonator |
|--------|-----------|---------------------|
| Sound source | WAV samples (PCM playback) | Additive synthesis from params |
| Per-note data | `float* pcm[8]` (MB of audio) | `NoteParams[8]` (~4 KB total) |
| Note-on latency | Instant (seek to PCM start) | Instant (init decay coefficients) |
| Memory footprint | Hundreds of MB (WAV bank) | < 1 MB (param table) |
| Envelope | ADSR (IthacaCore) | Bi-exponential decay (physics) |
| Sample rate conv | speexdsp offline resampler | Not needed |
| File I/O | libsndfile (WAV loading) | nlohmann/json (params loading) |
| Tuning | Fixed (recorded pitch) | Exact physics (inharmonicity) |
| Velocity layers | 8 discrete samples | 8 discrete param sets + interp |
| Beating / chorus | None | Explicit per-partial beat_hz |
