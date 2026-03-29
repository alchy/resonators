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

// ── Dynamic string pan angles (mirrors Python _string_angles) ────────────────
// center = PI/4 + (midi - 64.5) / 87 * 0.20   (bass=left, treble=right)
// half   = pan_spread / 2  (pan_spread=0.55 → half=0.275)
// 1 str: [center]
// 2 str: [center-half, center+half]
// 3 str: [center-half, center, center+half]
static void computeStringAngles(int midi, int n_strings, float* angles) {
    float center = PI/4.f + ((float)midi - 64.5f) / 87.f * 0.20f;
    float half   = 0.275f;   // pan_spread = 0.55
    if (n_strings == 1) {
        angles[0] = center;
    } else if (n_strings == 2) {
        angles[0] = center - half;
        angles[1] = center + half;
    } else {
        angles[0] = center - half;
        angles[1] = center;
        angles[2] = center + half;
    }
}

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

    // ── Per-string pan: MIDI-dependent center tilt + wide spread ─────────────
    // Mirrors Python: center = PI/4 + (midi-64.5)/87*0.20, half=0.275
    {
        float angles[MAX_STRINGS] = {};
        computeStringAngles(midi, n_strings_, angles);
        for (int str = 0; str < n_strings_; str++) {
            pan_l_[str] = std::cos(angles[str]);
            pan_r_[str] = std::sin(angles[str]);
        }
    }

    // ── Schroeder all-pass decorrelation coefficients ─────────────────────────
    // Mirrors Python: decor_strength = min(1,(midi-40)/60)*0.45
    //   g_L =  0.35 + decor_strength*0.25  (positive)
    //   g_R = -(0.35 + decor_strength*0.20) (negative → opposite phase shift)
    {
        float ds  = std::min(1.f, std::max(0.f, ((float)midi - 40.f) / 60.f)) * 0.45f;
        ap_g_l_      =  0.35f + ds * 0.25f;
        ap_g_r_      = -(0.35f + ds * 0.20f);
        ap_strength_ =  ds;
        ap_x_l_ = ap_y_l_ = ap_x_r_ = ap_y_r_ = 0.f;
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
// Signal chain per block:
//   1. Oscillator sum → stereo via MIDI-dependent string panning
//   2. Noise injection
//   3. Release ramp
//   4. BiquadEQ (spectral correction, per-channel)
//   5. Schroeder all-pass decorrelation (L/R get opposite-sign coefficients)
//   6. M/S stereo width (width_factor from params.json)
//   7. Add into output buffers

void ResonatorVoice::processBlock(float* out_l, float* out_r, int n_samples) {
    if (!active_) return;

    // Local voice output buffers — EQ and decorrelation applied before mixing
    static constexpr int VOICE_BUF_MAX = 512;
    int n = n_samples < VOICE_BUF_MAX ? n_samples : VOICE_BUF_MAX;

    float vl[VOICE_BUF_MAX];
    float vr[VOICE_BUF_MAX];

    int stop = n;  // samples written (may shorten if release completes early)

    for (int s = 0; s < n; s++) {
        float l = 0.f, r = 0.f;

        // ── Oscillator sum ────────────────────────────────────────────────────
        for (int k = 0; k < n_partials_; k++) {
            env1_[k] *= d1_[k];
            env2_[k] *= d2_[k];
            float env = env1_[k] + env2_[k];

            for (int str = 0; str < n_strings_; str++) {
                phase_[k][str] += omega_[k][str];
                if (phase_[k][str] > TAU) phase_[k][str] -= TAU;
                float sig = env * std::cos(phase_[k][str]);
                l += sig * pan_l_[str];
                r += sig * pan_r_[str];
            }
        }

        // ── Noise ─────────────────────────────────────────────────────────────
        if (noise_env_ > 1e-9f) {
            float white  = 2.f * (float)std::rand() / (float)RAND_MAX - 1.f;
            noise_state_ = noise_alpha_ * white + (1.f - noise_alpha_) * noise_state_;
            float n_sig  = noise_state_ * noise_env_;
            noise_env_  *= noise_decay_;
            l += n_sig;
            r += n_sig;
        }

        // ── Release ramp ──────────────────────────────────────────────────────
        if (releasing_) {
            release_gain_ += release_step_;
            if (release_gain_ <= 0.f) {
                active_ = false;
                vl[s] = 0.f;  vr[s] = 0.f;
                stop = s + 1;
                // Zero remaining frames
                for (int i = stop; i < n; i++) { vl[i] = 0.f; vr[i] = 0.f; }
                break;
            }
            l *= release_gain_;
            r *= release_gain_;
        }

        vl[s] = l;
        vr[s] = r;
    }

    // ── Spectral EQ (per-channel biquad cascade) ──────────────────────────────
    eq_l_.processBlock(vl, n);
    eq_r_.processBlock(vr, n);

    // ── Decorrelation + M/S width → mix into output ───────────────────────────
    // Schroeder first-order all-pass: y[n] = -g*x[n] + x[n-1] - g*y[n-1]
    // L uses g_L > 0, R uses g_R < 0 → opposite phase shifts = decorrelation
    const bool do_ap  = (ap_strength_ > 1e-4f);
    const float wl    = 0.5f * (1.f + width_factor_);  // M/S premix coefficients
    const float wr    = 0.5f * (1.f - width_factor_);

    for (int s = 0; s < n; s++) {
        float l = vl[s], r = vr[s];

        if (do_ap) {
            float y_l  = -ap_g_l_ * l + ap_x_l_ - ap_g_l_ * ap_y_l_;
            float y_r  = -ap_g_r_ * r + ap_x_r_ - ap_g_r_ * ap_y_r_;
            ap_x_l_ = l;  ap_y_l_ = y_l;
            ap_x_r_ = r;  ap_y_r_ = y_r;
            l = l * (1.f - ap_strength_) + y_l * ap_strength_;
            r = r * (1.f - ap_strength_) + y_r * ap_strength_;
        }

        // M/S width: expand or narrow the stereo image
        out_l[s] += l * wl + r * wr;
        out_r[s] += r * wl + l * wr;
    }

    sample_idx_ += n;

    // Silence check
    if (!releasing_) {
        float total_env = 0.f;
        for (int k = 0; k < n_partials_; k++)
            total_env += env1_[k] + env2_[k];
        if (total_env < 1e-7f)
            active_ = false;
    }
}
