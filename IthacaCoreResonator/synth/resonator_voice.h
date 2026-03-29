#pragma once
#include "note_params.h"
#include "biquad_eq.h"

// Single synthesizer voice.
//
// Physics model:
//   f_k   = k · f0_adj · sqrt(1 + B·k²)              inharmonic partial freq
//   env_k = A0[k] · (a1·exp(-t/τ1) + (1-a1)·exp(-t/τ2))  bi-exp decay
//   osc_k = env_k · Σ_str cos(2π·(f_k + beat_offset[str][k])·t + φ)
//   out   = Σ_k osc_k,  EQ → stereo M/S
//
// Envelope implemented as per-sample multiply (no expf() in audio loop):
//   decay_coeff = exp(-1 / (tau * sr))   computed once at note-on
//   env_state  *= decay_coeff            per sample — one multiply

class ResonatorVoice {
public:
    ResonatorVoice() = default;

    // Initialise with synthesis parameters. Resets all state.
    void noteOn(int midi, int vel, const NoteParams& p, float sample_rate);

    // Begin release fade (click-free, ~10 ms ramp to silence).
    void noteOff();

    bool isActive()  const { return active_; }
    bool isReleasing() const { return releasing_; }
    int  midiNote()  const { return midi_; }

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

    // ── Spectral EQ ───────────────────────────────────────────────────────────
    BiquadEQ eq_l_, eq_r_;

    // ── Envelope initial amplitudes ───────────────────────────────────────────
    float a1_[MAX_PARTIALS] = {};    // A0 * a1  (fast component initial amp)
    float a2_[MAX_PARTIALS] = {};    // A0 * (1-a1)

    // ── Noise state ───────────────────────────────────────────────────────────
    float noise_env_   = 0.f;   // current noise envelope amplitude
    float noise_decay_ = 0.f;   // per-sample decay multiplier
    float noise_alpha_ = 0.5f;  // 1-pole LP coefficient
    float noise_state_ = 0.f;   // 1-pole LP delay

    float width_factor_ = 0.5f; // M/S stereo width

    // ── Release ramp ─────────────────────────────────────────────────────────
    float release_gain_ = 1.f;
    float release_step_ = 0.f;   // negative, computed from RELEASE_MS
    bool  releasing_    = false;

    static constexpr float RELEASE_MS = 10.f;  // click-prevention fade

    // ── Misc ──────────────────────────────────────────────────────────────────
    int     n_partials_  = 0;
    int     n_strings_   = 2;
    float   sample_rate_ = 44100.f;
    int64_t sample_idx_  = 0;
    int     midi_        = -1;
    bool    active_      = false;
};
