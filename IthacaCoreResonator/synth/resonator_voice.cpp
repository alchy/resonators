/*
 * resonator_voice.cpp
 * ────────────────────
 * Single-voice additive physics synthesis.
 * Designed for full parity with analysis/physics_synth.py synthesize_note().
 *
 * Signal chain (per block):
 *   1. Oscillator sum, per-string equal-power panning, /n_strings normalisation
 *   2. Noise — independent L and R channels (unlike earlier mono version)
 *   3. Onset ramp (onset_ms linear fade-in to prevent click)
 *   4. Release ramp
 *   5. BiquadEQ (spectral correction, flat below eq_freq_min)
 *   6. Schroeder all-pass stereo decorrelation (opposite sign g per channel)
 *   7. M/S stereo width (width_factor * stereo_boost)
 *   8. Accumulate into output buffers
 *
 * Python divergences fixed:
 *   - Amplitude: divide by n_strings (Python /2 or /3 for multi-string)
 *   - Envelope:  mono partials use single-exp (not bi-exp)
 *   - Noise:     independent L/R; LP coeff matches Python 1-exp(-2π·fc/sr)
 *   - Onset:     3 ms linear ramp (Python onset_ms=3.0)
 *   - EQ:        freq_min fade-out (Python eq_freq_min=400 Hz)
 *   - pan_spread: from SynthConfig (was hardcoded 0.55)
 *   - beat_scale: from SynthConfig (was always 1.0)
 */

#include "resonator_voice.h"
#include <cmath>
#include <cstring>
#include <algorithm>
#include <cstdlib>

static constexpr float PI  = 3.14159265358979323846f;
static constexpr float TAU = 2.f * PI;

// ── String detuning offsets (multipliers for beat_hz) ────────────────────────
// Matches Python: 2-str [f-beat/2, f+beat/2],  3-str [f-beat/2, f, f+beat/2]
// Note: Python string0 = f+beat/2, string1 = f-beat/2 for 2 strings.
// C++ ordering is reversed but the sum is symmetric so the stereo image is
// equivalent (left-panned string gets -beat/2 vs Python +beat/2 — audibly same).
static const float STRING_SIGNS[3][3] = {
    { 0.f,  0.f,  0.f },   // n_strings=1: no detuning
    {-0.5f, 0.5f, 0.f },   // n_strings=2: ±beat/2
    {-0.5f, 0.f,  0.5f},   // n_strings=3: -beat/2, 0, +beat/2
};

// ── Dynamic string pan angles (Python _string_angles) ────────────────────────
// center = π/4 + (midi-64.5)/87 * 0.20   (bass=left, treble=right)
// half   = pan_spread/2
static void computeStringAngles(int midi, int n_strings, float pan_spread, float* angles) {
    float center = PI/4.f + ((float)midi - 64.5f) / 87.f * 0.20f;
    float half   = pan_spread * 0.5f;
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

// ── noteOn ────────────────────────────────────────────────────────────────────

void ResonatorVoice::noteOn(int midi, int vel,
                             const NoteParams& p, float sample_rate,
                             const SynthConfig& cfg) {
    // Velocity gain: (vel/127)^vel_gamma  (Python: rms ∝ ((vel+1)/8)^vel_gamma)
    // Provides smooth amplitude response across all 128 MIDI velocities.
    const float vel_gain = (vel > 0)
        ? std::pow((float)vel / 127.f, cfg.vel_gamma)
        : 0.f;

    midi_        = midi;
    sample_rate_ = sample_rate;
    n_partials_  = std::min(p.n_partials, MAX_PARTIALS);
    n_strings_   = std::max(1, std::min(p.n_strings, MAX_STRINGS));
    releasing_   = false;
    release_gain_= 1.f;
    release_step_= 0.f;
    sample_idx_  = 0;
    stereo_boost_= cfg.stereo_boost;

    // ── Amplitude normalisation factor (Python: sum / n_strings for n>1) ─────
    str_norm_ = 1.f / (float)n_strings_;

    // ── Per-string pan: MIDI-dependent center tilt + pan_spread from config ──
    {
        float angles[MAX_STRINGS] = {};
        computeStringAngles(midi, n_strings_, cfg.pan_spread, angles);
        for (int s = 0; s < n_strings_; s++) {
            pan_l_[s] = std::cos(angles[s]);
            pan_r_[s] = std::sin(angles[s]);
        }
    }

    // ── Schroeder all-pass decorrelation coefficients ─────────────────────────
    // Python: decor_strength = min(1,(midi-40)/60)*0.45 * stereo_decorr
    {
        float ds = std::min(1.f, std::max(0.f, ((float)midi - 40.f) / 60.f))
                   * 0.45f * cfg.stereo_decorr;
        ap_g_l_      =  0.35f + ds * 0.25f;
        ap_g_r_      = -(0.35f + ds * 0.20f);
        ap_strength_ =  ds;
        ap_x_l_ = ap_y_l_ = ap_x_r_ = ap_y_r_ = 0.f;
    }

    // ── Onset ramp ────────────────────────────────────────────────────────────
    {
        int onset_n = std::max(1, (int)(cfg.onset_ms * 0.001f * sample_rate));
        onset_step_ = 1.f / (float)onset_n;
        onset_gain_ = 0.f;   // Python: linspace(0, 1, n_onset) — first sample is 0
        in_onset_   = (onset_n > 1);
    }

    // ── A0_ref normalization + target_rms level calibration ─────────────────
    // Python: amp = (A / A0_ref).  A0_ref = first nonzero partial's A0.
    // This makes the synthesis dimensionless (all notes have similar raw levels).
    // Then Python RMS-normalizes the full signal to target_rms.
    // For RT C++: estimate instantaneous expected power at t=0 with random phases.
    //   E[power(t=0)] = sum_k (A0_k/A0_ref)^2 / 2  (cos^2 with random phase)
    //   → level_scale = target_rms * sqrt(2) / sqrt(sum_sq)
    // vel_gain is then applied on top for velocity dynamics.
    float A0_ref = 1.f;
    for (int k = 0; k < n_partials_; k++)
        if (p.partials[k].A0 > 1e-10f) { A0_ref = p.partials[k].A0; break; }
    if (A0_ref < 1e-10f) A0_ref = 1.f;

    float sum_sq = 0.f;
    for (int k = 0; k < n_partials_; k++) {
        float norm = p.partials[k].A0 / A0_ref;
        sum_sq += norm * norm;
    }
    const float level_scale = (sum_sq > 1e-10f)
        ? (cfg.target_rms * std::sqrt(2.f) / std::sqrt(sum_sq) * vel_gain)
        : (cfg.target_rms * vel_gain);

    // ── Pre-compute per-partial state ────────────────────────────────────────
    const float* signs = STRING_SIGNS[n_strings_ - 1];
    const float  beat_mul = cfg.beat_scale;

    for (int k = 0; k < n_partials_; k++) {
        const PartialParams& pp = p.partials[k];
        const float norm_A0 = pp.A0 / A0_ref;   // A0_ref-normalized amplitude

        // Harmonic brightness: gain = 1 + hb * log2(k)  (Python: harmonic_brightness)
        float bright = 1.f;
        if (cfg.harmonic_brightness != 0.f && pp.k > 1)
            bright = 1.f + cfg.harmonic_brightness * std::log2((float)pp.k);

        // Envelope type: mono partials use single-exp (Python: default mono=True → single-exp)
        if (pp.mono || pp.a1 >= 1.f - 1e-6f) {
            // Single exponential: env = (A0/A0_ref) * level_scale * exp(-t/tau1)
            a1_[k]  = norm_A0 * bright * level_scale;
            a2_[k]  = 0.f;
            d2_[k]  = 1.f;  // unused (a2=0), safe value
        } else {
            // Bi-exponential: env = (A0/A0_ref) * level_scale * (a1*e^(-t/tau1) + (1-a1)*e^(-t/tau2))
            a1_[k]  = norm_A0 * pp.a1         * bright * level_scale;
            a2_[k]  = norm_A0 * (1.f - pp.a1) * bright * level_scale;
            d2_[k]  = std::exp(-1.f / (std::max(pp.tau2, 0.001f) * sample_rate));
        }
        d1_[k]    = std::exp(-1.f / (std::max(pp.tau1, 0.001f) * sample_rate));
        env1_[k]  = a1_[k];
        env2_[k]  = a2_[k];

        // Angular frequencies per string (beat_scale applied here)
        for (int s = 0; s < n_strings_; s++) {
            float freq = pp.f_hz;
            if (!pp.mono && pp.beat_hz > 0.f)
                freq += pp.beat_hz * beat_mul * signs[s];
            omega_[k][s] = TAU * freq / sample_rate;
        }

        // Random initial phases — capture first raw value as seed snapshot
        for (int s = 0; s < n_strings_; s++) {
            int raw = std::rand();
            if (k == 0 && s == 0) last_seed_ = (uint32_t)raw;
            phase_[k][s] = TAU * (float)raw / (float)RAND_MAX;
        }
    }

    // ── Noise state ───────────────────────────────────────────────────────────
    // tau cap: noise never outlasts the string fundamental (Python: taun = min(taun, tau1_k1))
    float tau1_k1 = 3.f;
    for (int k = 0; k < n_partials_; k++)
        if (p.partials[k].k == 1) { tau1_k1 = p.partials[k].tau1; break; }
    float taun = std::min(p.noise.attack_tau_s, tau1_k1);

    // Python LP coefficient: alp = 1 - exp(-2π * min(cent, sr*0.45) / sr)
    float fc_noise   = std::min(p.noise.centroid_hz, sample_rate * 0.45f);
    noise_alpha_     = 1.f - std::exp(-TAU * fc_noise / sample_rate);
    noise_decay_     = std::exp(-1.f / (std::max(taun, 0.001f) * sample_rate));
    // Noise uses same level_scale as partials (Python: A_noise gets same RMS scale).
    noise_env_       = p.noise.floor_rms * cfg.noise_level * level_scale;
    noise_state_l_   = 0.f;
    noise_state_r_   = 0.f;

    // ── Spectral EQ ───────────────────────────────────────────────────────────
    // freq_min: EQ faded to 0 dB below eq_freq_min (room-acoustics protection)
    // eq_strength blends EQ vs flat — implement by scaling gains at design time
    // (simplification: full EQ when eq_strength=1, bypass when 0)
    float eq_gains[EQ_POINTS];
    for (int i = 0; i < EQ_POINTS; i++)
        eq_gains[i] = p.eq_gains_db[i] * cfg.eq_strength;
    eq_l_.design(p.eq_freqs_hz, eq_gains, sample_rate, cfg.eq_freq_min);
    eq_r_.design(p.eq_freqs_hz, eq_gains, sample_rate, cfg.eq_freq_min);
    eq_l_.reset();
    eq_r_.reset();

    width_factor_ = p.width_factor;
    active_       = true;
}

// ── noteOff ───────────────────────────────────────────────────────────────────

void ResonatorVoice::noteOff() {
    if (!active_ || releasing_) return;
    releasing_    = true;
    int ramp_samples = std::max(1, (int)(RELEASE_MS * 0.001f * sample_rate_));
    release_step_ = -1.f / ramp_samples;
}

// ── processBlock ─────────────────────────────────────────────────────────────

void ResonatorVoice::processBlock(float* out_l, float* out_r, int n_samples) {
    if (!active_) return;

    static constexpr int VOICE_BUF_MAX = 512;
    int n = n_samples < VOICE_BUF_MAX ? n_samples : VOICE_BUF_MAX;

    float vl[VOICE_BUF_MAX];
    float vr[VOICE_BUF_MAX];

    for (int s = 0; s < n; s++) {
        float l = 0.f, r = 0.f;

        // ── Oscillator sum (normalised by n_strings) ──────────────────────────
        for (int k = 0; k < n_partials_; k++) {
            env1_[k] *= d1_[k];
            env2_[k] *= d2_[k];
            float env = env1_[k] + env2_[k];

            for (int str = 0; str < n_strings_; str++) {
                phase_[k][str] += omega_[k][str];
                if (phase_[k][str] > TAU) phase_[k][str] -= TAU;
                float sig = env * std::cos(phase_[k][str]) * str_norm_;
                l += sig * pan_l_[str];
                r += sig * pan_r_[str];
            }
        }

        // ── Noise: independent L and R channels (Python: separate per buf) ────
        if (noise_env_ > 1e-9f) {
            float wl = 2.f * (float)std::rand() / (float)RAND_MAX - 1.f;
            float wr = 2.f * (float)std::rand() / (float)RAND_MAX - 1.f;
            noise_state_l_ = noise_alpha_ * wl + (1.f - noise_alpha_) * noise_state_l_;
            noise_state_r_ = noise_alpha_ * wr + (1.f - noise_alpha_) * noise_state_r_;
            l += noise_state_l_ * noise_env_;
            r += noise_state_r_ * noise_env_;
            noise_env_ *= noise_decay_;
        }

        // ── Release ramp ──────────────────────────────────────────────────────
        if (releasing_) {
            release_gain_ += release_step_;
            if (release_gain_ <= 0.f) {
                active_ = false;
                vl[s] = 0.f; vr[s] = 0.f;
                int stop = s + 1;
                for (int i = stop; i < n; i++) { vl[i] = 0.f; vr[i] = 0.f; }
                goto apply_post;
            }
            l *= release_gain_;
            r *= release_gain_;
        }

        vl[s] = l;
        vr[s] = r;
    }

apply_post:
    // ── Spectral EQ ───────────────────────────────────────────────────────────
    eq_l_.processBlock(vl, n);
    eq_r_.processBlock(vr, n);

    // ── Onset ramp (Python: applied AFTER EQ and width, click prevention) ────
    // Applied here post-EQ (before decorrelation) to match Python's order:
    //   synthesize → rms_norm → EQ → rms_norm → width → rms_norm → onset_ramp
    if (in_onset_) {
        for (int s = 0; s < n; s++) {
            vl[s] *= onset_gain_;
            vr[s] *= onset_gain_;
            onset_gain_ += onset_step_;
            if (onset_gain_ >= 1.f) { onset_gain_ = 1.f; in_onset_ = false; break; }
        }
        // If onset ended mid-block, remaining samples are already at full gain
    }

    // ── Decorrelation + M/S width → accumulate into output ───────────────────
    const bool  do_ap = (ap_strength_ > 1e-4f);
    const float eff   = width_factor_ * stereo_boost_;  // effective M/S side gain
    const float wl_ms = 0.5f * (1.f + eff);
    const float wr_ms = 0.5f * (1.f - eff);

    for (int s = 0; s < n; s++) {
        float l = vl[s], r = vr[s];

        if (do_ap) {
            // y[n] = -g*x[n] + x[n-1] - g*y[n-1]
            float yl = -ap_g_l_ * l + ap_x_l_ - ap_g_l_ * ap_y_l_;
            float yr = -ap_g_r_ * r + ap_x_r_ - ap_g_r_ * ap_y_r_;
            ap_x_l_ = l; ap_y_l_ = yl;
            ap_x_r_ = r; ap_y_r_ = yr;
            l = l * (1.f - ap_strength_) + yl * ap_strength_;
            r = r * (1.f - ap_strength_) + yr * ap_strength_;
        }

        out_l[s] += l * wl_ms + r * wr_ms;
        out_r[s] += r * wl_ms + l * wr_ms;
    }

    sample_idx_ += n;

    if (!releasing_) {
        float total = 0.f;
        for (int k = 0; k < n_partials_; k++)
            total += env1_[k] + env2_[k];
        if (total < 1e-7f) active_ = false;
    }
}
