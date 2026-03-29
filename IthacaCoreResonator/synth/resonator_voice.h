#pragma once
#include "note_params.h"
#include "biquad_eq.h"
#include "synth_config.h"

// Single synthesizer voice.
//
// Physics model:
//   f_k   = k · f0_adj · sqrt(1 + B·k²)              inharmonic partial freq
//   env_k = A0[k] · (a1·exp(-t/τ1) + (1-a1)·exp(-t/τ2))  bi-exp decay
//           (or A0·exp(-t/τ1) when partial is mono)
//   osc_k = env_k · Σ_str cos(2π·(f_k + beat_offset[str][k])·t + φ)
//   out   = Σ_k osc_k / n_strings, then EQ → decorr → M/S
//
// Amplitude normalisation: sum over n_strings is divided by n_strings so that
// multi-string notes have the same level as single-string (matches Python).

class ResonatorVoice {
public:
    ResonatorVoice() = default;

    // Initialise with synthesis parameters. Resets all state.
    void noteOn(int midi, int vel, const NoteParams& p,
                float sample_rate, const SynthConfig& cfg);

    // Begin release fade (click-free, ~10 ms ramp to silence).
    void noteOff();

    bool isActive()    const { return active_; }
    bool isReleasing() const { return releasing_; }
    int  midiNote()    const { return midi_; }

    // Add this voice into out_l / out_r (additive mix, n_samples frames).
    void processBlock(float* out_l, float* out_r, int n_samples);

private:
    // ── Oscillator state ──────────────────────────────────────────────────────
    float phase_[MAX_PARTIALS][MAX_STRINGS] = {};  // current phase (radians)
    float omega_[MAX_PARTIALS][MAX_STRINGS] = {};  // angular freq per sample

    // ── Envelope state (bi-exponential, no expf in loop) ─────────────────────
    float env1_[MAX_PARTIALS] = {};    // fast-decay amplitude state
    float env2_[MAX_PARTIALS] = {};    // slow-decay amplitude state
    float d1_[MAX_PARTIALS]   = {};    // fast decay per-sample coeff
    float d2_[MAX_PARTIALS]   = {};    // slow decay per-sample coeff

    // ── Stereo panning per string ─────────────────────────────────────────────
    float pan_l_[MAX_STRINGS] = {};
    float pan_r_[MAX_STRINGS] = {};
    float str_norm_            = 1.f;  // 1/n_strings (amplitude normalisation)

    // ── Spectral EQ ───────────────────────────────────────────────────────────
    BiquadEQ eq_l_, eq_r_;

    // ── Envelope initial amplitudes ───────────────────────────────────────────
    float a1_[MAX_PARTIALS] = {};    // fast component initial amp  (A0 * a1 or A0 if mono)
    float a2_[MAX_PARTIALS] = {};    // slow component initial amp  (A0 * (1-a1), 0 if mono)

    // ── Noise state (independent L and R channels, like Python) ──────────────
    float noise_env_    = 0.f;   // current noise envelope amplitude
    float noise_decay_  = 0.f;   // per-sample decay multiplier
    float noise_alpha_  = 0.5f;  // 1-pole LP coefficient  (matches Python formula)
    float noise_state_l_= 0.f;   // 1-pole LP delay — left channel
    float noise_state_r_= 0.f;   // 1-pole LP delay — right channel

    // ── Stereo post-processing ────────────────────────────────────────────────
    float width_factor_  = 0.5f; // M/S side gain (from params.json)
    float stereo_boost_  = 1.0f; // extra M/S side boost (from SynthConfig)

    // ── Stereo decorrelation (Schroeder first-order all-pass) ─────────────────
    // lfilter([-g, 1.0], [1.0, g], x) — different g sign per channel
    float ap_x_l_      = 0.f;
    float ap_y_l_      = 0.f;
    float ap_x_r_      = 0.f;
    float ap_y_r_      = 0.f;
    float ap_g_l_      = 0.f;
    float ap_g_r_      = 0.f;
    float ap_strength_ = 0.f;

    // ── Onset ramp (prevents click at note start) ─────────────────────────────
    float onset_gain_  = 1.f;   // current ramp gain (0→1 over onset_ms)
    float onset_step_  = 0.f;   // per-sample increment
    bool  in_onset_    = false;

    // ── Release ramp ─────────────────────────────────────────────────────────
    float release_gain_ = 1.f;
    float release_step_ = 0.f;
    bool  releasing_    = false;

    static constexpr float RELEASE_MS = 10.f;

    // ── Misc ──────────────────────────────────────────────────────────────────
    int      n_partials_  = 0;
    int      n_strings_   = 2;
    float    sample_rate_ = 44100.f;
    int64_t  sample_idx_  = 0;
    int      midi_        = -1;
    bool     active_      = false;

    // First raw rand() value consumed at noteOn — proxy for RNG state at that moment.
    // Identical seed = identical phase pattern = identical sound (for a given note).
    uint32_t last_seed_   = 0;

public:
    uint32_t getLastSeed() const noexcept { return last_seed_; }
};
