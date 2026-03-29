#pragma once
#include <array>
#include <string>

// Maximum dimensions — compile-time constants
static constexpr int MAX_PARTIALS = 96;   // bass notes can have up to ~90 partials
static constexpr int MAX_STRINGS  = 3;
static constexpr int EQ_POINTS    = 64;   // log-spaced EQ curve points from params.json
static constexpr int MIDI_MIN     = 21;
static constexpr int MIDI_MAX     = 108;
static constexpr int MIDI_COUNT   = MIDI_MAX - MIDI_MIN + 1; // 88
static constexpr int VEL_LAYERS   = 8;

// ── Per-partial physics parameters ────────────────────────────────────────────
// Matches params.json partial fields exactly.

struct PartialParams {
    int   k           = 0;     // partial number (1-based, harmonic index)
    float f_hz        = 0.f;   // actual inharmonic frequency (Hz)
    float A0          = 0.f;   // initial amplitude
    float tau1        = 3.f;   // fast bi-exp decay time constant (s)
    float tau2        = 10.f;  // slow bi-exp decay time constant (s); null → tau1*3
    float a1          = 0.6f;  // bi-exp mixing weight (tau1 fraction)
    float beat_hz     = 0.f;   // inter-string detuning (Hz)
    float beat_depth  = 0.f;   // beating depth (0..1), 0 = pure tone
    bool  mono        = false; // true → no inter-string beating for this partial
};

// ── Noise parameters ──────────────────────────────────────────────────────────

struct NoiseParams {
    float attack_tau_s        = 0.05f;  // noise envelope decay (s)
    float floor_rms           = 0.f;    // noise RMS level
    float centroid_hz         = 2000.f; // spectral centroid (FIR filter design)
    float spectral_slope_db_oct = -12.f; // roll-off slope
};

// ── Per-note parameters (one entry per midi+velocity) ─────────────────────────

struct NoteParams {
    int   midi         = 60;
    int   vel          = 3;
    float f0_hz        = 440.f;   // fitted fundamental (f0_fitted_hz)
    float B            = 1e-4f;   // inharmonicity coefficient
    float width_factor = 0.5f;    // stereo width (spectral_eq.stereo_width_factor)
    float duration_s   = 8.f;     // full note duration from recording
    int   sr           = 48000;   // source sample rate
    int   n_partials   = 0;       // valid entries in partials[]
    int   n_strings    = 2;       // 1 (low bass), 2, or 3 strings

    PartialParams partials[MAX_PARTIALS];
    NoiseParams   noise;

    // Raw spectral EQ curve from params.json (64 log-spaced points, 20–20000 Hz)
    float eq_freqs_hz[EQ_POINTS] = {};
    float eq_gains_db[EQ_POINTS] = {};

    bool  valid = false;  // false → note absent from params.json
};

// ── Pre-loaded lookup table: [midi-21][velocity_layer] ────────────────────────
// Populated at startup from params.json. Zero-allocation in audio path.

using NoteLUT = std::array<std::array<NoteParams, VEL_LAYERS>, MIDI_COUNT>;
