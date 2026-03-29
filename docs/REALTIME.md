# Real-Time Synthesis in C++

Concept for converting the Python additive synthesizer into a real-time plugin (VST3/CLAP/AU).

---

## Computation per note

For one active note at 44100 Hz, buffer size 256 samples (~5.8 ms):

| Component | Operations / buffer | Notes |
|-----------|--------------------|----|
| Oscillator phase advance | N_partials × N_strings × 256 adds | ~60–90 per note |
| sin/cos | same | SIMD: 8× float32 per cycle with AVX2 |
| Envelope evaluation | N_partials × 4 muls+adds | exp() pre-baked to lookup table |
| Stereo panning | N_strings × 2 muls | constant per note |
| Spectral EQ | 64-point FIR or 1 FFT per buffer | FFT approach: N_FFT=256, overlap-add |

Rough estimate for one note (30 partials × 3 strings = 90 oscillators):
- Phase + sin/cos: ~23 000 float ops
- With AVX2 (8-wide SIMD): ~2 900 cycles ≈ **1 µs** at 3 GHz

For 10 simultaneous notes: ~10 µs computation — negligible at 5.8 ms buffer budget.

---

## Latency budget

| Source | Latency |
|--------|---------|
| Audio buffer (256 samples @ 44100) | 5.8 ms |
| Audio buffer (128 samples @ 44100) | 2.9 ms |
| Audio buffer (64 samples @ 44100)  | 1.5 ms |
| MIDI stack (USB HID)               | ~1 ms |
| DAW overhead                       | ~1–2 ms |
| **Total typical (128 samples)**    | **~5 ms** |

5 ms is below the perceptual threshold for piano key-to-sound latency (~10 ms is acceptable; <20 ms imperceptible in practice).

---

## Implementation sketch

### Parameter loading

At note-on: look up `params-nn-profile-{bank}.json` for the MIDI note + velocity.
All parameters are pre-loaded into a flat array at startup — no file I/O on note trigger.

```cpp
struct PartialParams {
    float f0_hz;
    float B;               // inharmonicity
    float tau1, tau2, a1;  // bi-exp decay
    float beat_hz;          // beating
    float A0;
    float pan_angle;        // per-string
};

struct NoteParams {
    PartialParams partials[MAX_PARTIALS];  // ~30
    int n_strings;                         // 1, 2, or 3
    float duration_s;
    EQCurve spectral_eq;                   // 64-point log-spaced gains
    float width_factor;
};

// Pre-loaded lookup: [midi 21..108][vel 0..7]
NoteParams note_lut[88][8];
```

### Voice rendering

```cpp
void Voice::process(float* out_l, float* out_r, int n_samples) {
    for (int s = 0; s < n_samples; s++) {
        float t = (sample_idx + s) / sample_rate;
        float l = 0, r = 0;

        for (int k = 0; k < n_partials; k++) {
            float fk = k * f0 * sqrtf(1.f + B * k * k);
            float env = A0[k] * (a1[k] * expf(-t / tau1[k])
                                + (1-a1[k]) * expf(-t / tau2[k]));
            for (int str = 0; str < n_strings; str++) {
                float fki = fk + delta[str][k];  // beating offset
                phase[str][k] += 2 * M_PI * fki / sample_rate;
                float sig = env * cosf(phase[str][k]);
                l += sig * pan_l[str];
                r += sig * pan_r[str];
            }
        }
        out_l[s] += l;
        out_r[s] += r;
    }
    sample_idx += n_samples;
}
```

### Envelope optimisation

Pre-compute envelope sample-by-sample for the note duration and store in a buffer.
Avoid per-sample `expf()` — expensive and unnecessary.

```cpp
// At note-on, pre-compute envelope for each partial:
for (int k = 0; k < n_partials; k++) {
    float decay1 = expf(-1.f / (tau1[k] * sample_rate));
    float decay2 = expf(-1.f / (tau2[k] * sample_rate));
    env1_state[k] = A0[k] * a1[k];
    env2_state[k] = A0[k] * (1 - a1[k]);
    // In process loop: env1_state[k] *= decay1; (one mul per sample)
}
```

### Spectral EQ

Apply as a linear-phase FIR or FFT overlap-add per voice.
At 64 EQ points log-spaced 20 Hz–20 kHz: design a 256-tap FIR via frequency sampling.
Cost: one FFT per buffer per voice (negligible vs. oscillators).

---

## Polyphony

| Voices | CPU (estimated) | Buffer size |
|--------|-----------------|-------------|
| 10 | < 1% modern CPU | 256 samples |
| 32 | ~3% modern CPU  | 256 samples |
| 88 (all notes) | ~8% modern CPU | 256 samples |

Piano rarely exceeds 10 simultaneous notes with normal pedal use.

---

## Recommended C++ stack

| Component | Library |
|-----------|---------|
| Plugin format | JUCE (VST3 + AU + CLAP) or iPlug2 |
| SIMD | `xsimd` or hand-written AVX2 intrinsics |
| JSON loading | `nlohmann/json` (params at startup) |
| Audio | JUCE AudioProcessorGraph or raw callback |

---

## Migration from Python

The Python synthesis in `analysis/physics_synth.py` maps directly:

| Python | C++ |
|--------|-----|
| `synthesize_note(sample, **kwargs)` | `Voice::noteOn(midi, vel, params)` |
| `params_data["samples"]["m060_vel3"]` | `note_lut[midi-21][vel]` |
| `resolve_note_params(cfg, midi)` | compile-time struct with defaults + per-note delta table |
| Spectral EQ application | FFT overlap-add per voice |
| Schroeder all-pass decorrelation | two all-pass filters per voice (L/R) |

The NN inference does **not** run in real-time — the `params-nn-profile-{bank}.json` is
pre-computed offline and loaded at plugin startup. No torch dependency in the C++ plugin.
