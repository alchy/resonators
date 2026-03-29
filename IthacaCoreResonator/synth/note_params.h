#pragma once
#include <array>
#include <string>

// Maximum dimensions — compile-time constants
static constexpr int MAX_PARTIALS = 64;
static constexpr int MAX_STRINGS  = 3;
static constexpr int EQ_BANDS     = 8;   // biquad cascade band count
static constexpr int MIDI_MIN     = 21;
static constexpr int MIDI_MAX     = 108;
static constexpr int MIDI_COUNT   = MIDI_MAX - MIDI_MIN + 1; // 88
static constexpr int VEL_LAYERS   = 8;

// ── Per-partial physics parameters ────────────────────────────────────────────

struct PartialParams {
    float A0          = 0.f;   // initial amplitude
    float tau1        = 3.f;   // fast bi-exp decay constant (s)
    float tau2        = 10.f;  // slow bi-exp decay constant (s)
    float a1          = 0.6f;  // bi-exp mixing weight (tau1 fraction)
    float beat_hz     = 0.f;   // inter-string detuning (Hz)
    float beat_depth  = 0.f;   // beating depth relative to A0 (0..1)
};

// ── Per-note parameters (one entry per midi+velocity) ─────────────────────────

struct NoteParams {
    float f0_hz            = 440.f;  // fundamental frequency (Hz)
    float B                = 1e-4f;  // inharmonicity coefficient
    float f0_offset_cents  = 0.f;    // fine tuning in cents
    float width_factor     = 0.5f;   // stereo width (0 = mono, 1 = full)
    float noise_rms        = 0.f;    // noise floor (relative to signal RMS)

    int   n_partials = 0;            // valid entries in partials[]
    int   n_strings  = 2;            // 1, 2, or 3 strings per note

    PartialParams partials[MAX_PARTIALS];

    // Spectral EQ: gains in dB at EQ_BANDS log-spaced centre frequencies.
    // Band centres: 80, 160, 320, 640, 1250, 2500, 5000, 12000 Hz
    float eq_gains_db[EQ_BANDS] = {};

    bool valid = false;              // false → note not in params.json
};

// ── Pre-loaded lookup table: [midi-21][velocity_layer] ────────────────────────
// Populated at startup from params.json. Zero-allocation in audio path.

using NoteLUT = std::array<std::array<NoteParams, VEL_LAYERS>, MIDI_COUNT>;
