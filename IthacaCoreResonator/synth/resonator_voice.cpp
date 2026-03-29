/*
 * resonator_voice.cpp
 * ────────────────────
 * Single-voice additive physics synthesis.
 *
 * Signal model:
 *   f_k   = partial.f_hz  (pre-fitted inharmonic freq from params.json)
 *   env_k = a1·e^(-t/τ1) + (1-a1)·e^(-t/τ2)   (bi-exponential, A0-scaled)
 *
 *   Per string (n_strings = 1..3):
 *     freq_str = f_k + beat_hz * str_sign[str]   (if !mono)
 *     sig      = env_k · cos(phase)
 *     pan_l, pan_r: equal-power per-string panning
 *
 *   Noise: white noise shaped by single-pole LP, gated by attack envelope
 *
 *   Post: apply BiquadEQ (spectral EQ), then M/S stereo:
 *     M = mono_sum,  S = M * width_factor
 *     L = M + S,     R = M − S
 *
 * Envelope inner loop: no expf() — pre-computed per-sample multipliers.
 */

#include "resonator_voice.h"
#include <cmath>
#include <cstring>
#include <algorithm>
#include <cstdlib>  // rand()

static constexpr float PI  = 3.14159265358979323846f;
static constexpr float TAU = 2.f * PI;

// ── String detuning offsets (multipliers for beat_hz) ────────────────────────
// 3-string: [-0.5, 0, +0.5]  (symmetric around f_k)
// 2-string: [-0.5, +0.5]
// 1-string: [0]
static const float STRING_SIGNS[3][3] = {
    { 0.f,  0.f,  0.f },   // n_strings=1
    {-0.5f, 0.5f, 0.f },   // n_strings=2
    {-0.5f, 0.f,  0.5f},   // n_strings=3
};

// ── Equal-power stereo panning ────────────────────────────────────────────────
// String 0 → slight left, string 1 → centre, string 2 → slight right
// Pan angle in [0..PI/2], L=cos(angle), R=sin(angle)
static const float STRING_PAN_ANGLE[3][3] = {
    { PI/4.f, 0.f,    0.f    },             // 1 string: centre
    { PI/4.f - 0.2f, PI/4.f + 0.2f, 0.f }, // 2 strings: L/R spread
    { PI/4.f - 0.3f, PI/4.f, PI/4.f + 0.3f}, // 3 strings: L/C/R
};

// ── Simple 1-pole noise filter state ─────────────────────────────────────────
struct NoiseFilter { float state = 0.f; };

// ── noteOn ────────────────────────────────────────────────────────────────────

void ResonatorVoice::noteOn(int midi, int vel,
                             const NoteParams& p, float sample_rate) {
    (void)vel;
    midi_        = midi;
    sample_rate_ = sample_rate;
    n_partials_  = std::min(p.n_partials, MAX_PARTIALS);
    n_strings_   = std::max(1, std::min(p.n_strings, MAX_STRINGS));
    releasing_   = false;
    release_gain_= 1.f;
    release_step_= 0.f;
    sample_idx_  = 0;

    const float* signs = STRING_SIGNS[n_strings_ - 1];
    const float* pans  = STRING_PAN_ANGLE[n_strings_ - 1];

    // ── Pre-compute per-string pan gains ─────────────────────────────────────
    for (int str = 0; str < n_strings_; str++) {
        pan_l_[str] = std::cos(pans[str]);
        pan_r_[str] = std::sin(pans[str]);
    }

    // ── Pre-compute per-partial state ────────────────────────────────────────
    for (int k = 0; k < n_partials_; k++) {
        const PartialParams& pp = p.partials[k];

        // Bi-exponential initial amplitudes
        a1_[k]    = pp.A0 * pp.a1;
        a2_[k]    = pp.A0 * (1.f - pp.a1);
        // Per-sample decay multipliers (one multiply per partial in audio loop)
        d1_[k]    = std::exp(-1.f / (std::max(pp.tau1, 0.001f) * sample_rate));
        d2_[k]    = std::exp(-1.f / (std::max(pp.tau2, 0.001f) * sample_rate));
        // Initial envelope state
        env1_[k]  = a1_[k];
        env2_[k]  = a2_[k];

        // Angular frequency per sample for each string
        for (int str = 0; str < n_strings_; str++) {
            float freq_hz = pp.f_hz;
            if (!pp.mono && pp.beat_hz > 0.f)
                freq_hz += pp.beat_hz * signs[str];
            omega_[k][str] = TAU * freq_hz / sample_rate;
        }

        // Randomise initial phases (avoids phase correlation between notes)
        for (int str = 0; str < n_strings_; str++)
            phase_[k][str] = TAU * (float)std::rand() / (float)RAND_MAX;
    }

    // ── Noise state ───────────────────────────────────────────────────────────
    noise_env_    = p.noise.floor_rms;
    noise_decay_  = std::exp(-1.f / (std::max(p.noise.attack_tau_s, 0.001f) * sample_rate));
    // Simple 1-pole LP cutoff from centroid_hz (bilinear approx)
    float wc      = TAU * p.noise.centroid_hz / sample_rate;
    noise_alpha_  = wc / (wc + 1.f);
    noise_state_  = 0.f;

    // ── Spectral EQ (designed from 64-point curve) ────────────────────────────
    eq_l_.design(p.eq_freqs_hz, p.eq_gains_db, sample_rate);
    eq_r_.design(p.eq_freqs_hz, p.eq_gains_db, sample_rate);
    eq_l_.reset();
    eq_r_.reset();

    width_factor_ = p.width_factor;
    active_       = true;
}

// ── noteOff ───────────────────────────────────────────────────────────────────

void ResonatorVoice::noteOff() {
    if (!active_ || releasing_) return;
    releasing_    = true;
    // 10 ms linear ramp to silence
    int ramp_samples = std::max(1, (int)(RELEASE_MS * 0.001f * sample_rate_));
    release_step_ = -1.f / ramp_samples;
}

// ── processBlock ─────────────────────────────────────────────────────────────

void ResonatorVoice::processBlock(float* out_l, float* out_r, int n_samples) {
    if (!active_) return;

    for (int s = 0; s < n_samples; s++) {

        // ── Oscillator sum ────────────────────────────────────────────────────
        float mono = 0.f;
        float l = 0.f, r = 0.f;

        for (int k = 0; k < n_partials_; k++) {
            // Bi-exp envelope: two multiplies, no expf
            env1_[k] *= d1_[k];
            env2_[k] *= d2_[k];
            float env = env1_[k] + env2_[k];

            for (int str = 0; str < n_strings_; str++) {
                phase_[k][str] += omega_[k][str];
                // Keep phase in [0, 2π] to avoid float precision drift
                if (phase_[k][str] > TAU) phase_[k][str] -= TAU;
                float sig = env * std::cos(phase_[k][str]);
                l += sig * pan_l_[str];
                r += sig * pan_r_[str];
                mono += sig;
            }
        }

        // ── Noise ─────────────────────────────────────────────────────────────
        if (noise_env_ > 1e-9f) {
            float white   = 2.f * (float)std::rand() / (float)RAND_MAX - 1.f;
            noise_state_  = noise_alpha_ * white + (1.f - noise_alpha_) * noise_state_;
            float n_sig   = noise_state_ * noise_env_;
            noise_env_   *= noise_decay_;
            l += n_sig; r += n_sig;
        }

        // ── Apply release gain ─────────────────────────────────────────────────
        if (releasing_) {
            release_gain_ += release_step_;
            if (release_gain_ <= 0.f) {
                active_ = false;
                // Zero remaining output and return
                for (int i = s; i < n_samples; i++) {
                    out_l[i] += 0.f;
                    out_r[i] += 0.f;
                }
                return;
            }
            l *= release_gain_;
            r *= release_gain_;
        }

        out_l[s] += l;
        out_r[s] += r;
    }

    // ── Spectral EQ on accumulated output (applied per-block) ────────────────
    // Note: EQ is applied to the block buffers after the loop for efficiency.
    // Since we ADD to out_l/out_r, we apply EQ to a temporary buffer.
    // TODO Phase 2: use a local temp buffer and apply EQ before adding.
    // For now, EQ is bypassed (will be wired in Phase 2 when temp buffer added).

    sample_idx_ += n_samples;

    // Silence check: if all partials decayed below threshold, deactivate
    if (!releasing_) {
        float total_env = 0.f;
        for (int k = 0; k < n_partials_; k++)
            total_env += env1_[k] + env2_[k];
        if (total_env < 1e-7f)
            active_ = false;
    }
}
