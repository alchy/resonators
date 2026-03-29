#include "limiter.h"
#include <cmath>
#include <algorithm>

static constexpr float ATTACK_MS = 1.f;   // fixed fast attack

void Limiter::prepare(float sample_rate, int /*max_block_size*/) {
    sample_rate_   = sample_rate;
    attack_coeff_  = std::exp(-1.f / (ATTACK_MS * 0.001f * sample_rate));
    release_coeff_ = std::exp(-1.f / (200.f    * 0.001f * sample_rate)); // default 200 ms
    gain_          = 1.f;
}

void Limiter::setThresholdDb(float db) {
    threshold_lin_ = std::pow(10.f, db / 20.f);
}

void Limiter::setReleaseMs(float ms) {
    ms = std::max(10.f, std::min(ms, 2000.f));
    release_coeff_ = std::exp(-1.f / (ms * 0.001f * sample_rate_));
}

void Limiter::process(float* L, float* R, int n_samples) {
    if (!enabled_) {
        gain_red_db_ = 0.f;
        return;
    }

    for (int i = 0; i < n_samples; i++) {
        float peak = std::max(std::abs(L[i]), std::abs(R[i]));

        // Desired gain: reduce if peak > threshold
        float target = (peak > threshold_lin_ && peak > 1e-9f)
                     ? threshold_lin_ / peak
                     : 1.f;

        // Smooth envelope: attack if reducing, release if recovering
        if (target < gain_)
            gain_ = attack_coeff_  * gain_ + (1.f - attack_coeff_)  * target;
        else
            gain_ = release_coeff_ * gain_ + (1.f - release_coeff_) * target;

        gain_ = std::min(gain_, 1.f);

        L[i] *= gain_;
        R[i] *= gain_;
    }

    // Meter: average gain reduction over the block
    gain_red_db_ = 20.f * std::log10(std::max(gain_, 1e-9f));
}

float Limiter::gainReductionDb() const {
    return gain_red_db_;
}
