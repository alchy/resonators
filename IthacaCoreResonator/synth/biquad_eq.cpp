/*
 * biquad_eq.cpp
 * ─────────────
 * 8-band peaking biquad EQ designed from the 64-point spectral_eq curve.
 *
 * At note-on, for each of our 8 band centres we interpolate the gain from
 * the 64-point log-spaced curve in params.json, then design a peaking
 * biquad (RBJ Audio EQ Cookbook).
 */

#include "biquad_eq.h"
#include <cmath>
#include <algorithm>

static constexpr float PI = 3.14159265358979323846f;

// ── RBJ peaking EQ biquad ─────────────────────────────────────────────────────
// gain_db: ±dB, fc_hz: centre frequency, Q: bandwidth
BiquadCoeffs BiquadEQ::designPeaking(float fc_hz, float gain_db,
                                      float Q, float sample_rate) {
    float A  = std::pow(10.f, gain_db / 40.f);   // sqrt(10^(dB/20))
    float w0 = 2.f * PI * fc_hz / sample_rate;
    float cs = std::cos(w0);
    float sn = std::sin(w0);
    float al = sn / (2.f * Q);

    float b0 =  1.f + al * A;
    float b1 = -2.f * cs;
    float b2 =  1.f - al * A;
    float a0 =  1.f + al / A;
    float a1_c = -2.f * cs;
    float a2 =  1.f - al / A;

    float inv_a0 = 1.f / a0;
    BiquadCoeffs c;
    c.b0 = b0 * inv_a0;
    c.b1 = b1 * inv_a0;
    c.b2 = b2 * inv_a0;
    c.a1 = a1_c * inv_a0;
    c.a2 = a2 * inv_a0;
    return c;
}

// ── Interpolate gain at fc_hz from the 64-point log-spaced curve ──────────────
static float interpGain(const float* freqs, const float* gains, int n, float fc) {
    if (n <= 0) return 0.f;
    if (fc <= freqs[0])   return gains[0];
    if (fc >= freqs[n-1]) return gains[n-1];

    // Binary search for surrounding bin
    int lo = 0, hi = n - 1;
    while (hi - lo > 1) {
        int mid = (lo + hi) / 2;
        if (freqs[mid] <= fc) lo = mid; else hi = mid;
    }

    // Log-frequency interpolation
    float t = (std::log(fc) - std::log(freqs[lo]))
            / (std::log(freqs[hi]) - std::log(freqs[lo]));
    return gains[lo] + t * (gains[hi] - gains[lo]);
}

// ── Public: design ────────────────────────────────────────────────────────────

void BiquadEQ::design(const float eq_freqs_hz[EQ_POINTS],
                      const float eq_gains_db[EQ_POINTS],
                      float sample_rate, float freq_min) {
    for (int b = 0; b < EQ_BANDS; b++) {
        float fc   = EQ_BAND_FREQS_HZ[b];
        float gain = interpGain(eq_freqs_hz, eq_gains_db, EQ_POINTS, fc);

        // Fade EQ to 0 dB below freq_min to avoid room-acoustics contamination.
        // Transition: flat below freq_min/2, linear ramp to full gain at freq_min.
        if (freq_min > 0.f) {
            float fade_low = freq_min * 0.5f;
            if (fc < fade_low) {
                gain = 0.f;
            } else if (fc < freq_min) {
                gain *= (fc - fade_low) / (freq_min - fade_low);
            }
        }

        gain = std::max(-24.f, std::min(24.f, gain));
        coeffs_[b] = designPeaking(fc, gain, /*Q=*/1.4f, sample_rate);
    }
}

void BiquadEQ::reset() {
    for (auto& s : state_) s = BiquadState{};
}

// ── Per-sample processing ─────────────────────────────────────────────────────

inline float BiquadEQ::processSample(float x) {
    for (int b = 0; b < EQ_BANDS; b++) {
        const BiquadCoeffs& c = coeffs_[b];
        BiquadState&        s = state_[b];
        float y = c.b0 * x + c.b1 * s.x1 + c.b2 * s.x2
                            - c.a1 * s.y1 - c.a2 * s.y2;
        s.x2 = s.x1; s.x1 = x;
        s.y2 = s.y1; s.y1 = y;
        x = y;
    }
    return x;
}

void BiquadEQ::processBlock(float* buf, int n_samples) {
    for (int i = 0; i < n_samples; i++)
        buf[i] = processSample(buf[i]);
}
