#pragma once
/*
 * limiter.h — Simple stereo peak limiter.
 *
 * Algorithm:
 *   - Per-block peak detection
 *   - If peak > threshold: gain = threshold / peak  (instantaneous, no lookahead)
 *   - Gain envelope: attack fixed 1 ms, release variable (midi-controlled)
 *   - Output gain reduction available for metering
 */

class Limiter {
public:
    void prepare(float sample_rate, int max_block_size);

    // threshold_db: 0..-40 dB mapped from MIDI 127..0
    void setThresholdDb(float db);

    // release_ms: 10..2000 ms mapped from MIDI 0..127
    void setReleaseMs(float ms);

    void setEnabled(bool on) { enabled_ = on; }
    bool isEnabled() const   { return enabled_; }

    // Process stereo interleaved block in-place.
    void process(float* L, float* R, int n_samples);

    // Current gain reduction in dB (0 = no limiting, negative = reducing)
    float gainReductionDb() const;

private:
    float sample_rate_  = 48000.f;
    float threshold_lin_= 1.f;    // linear threshold
    float gain_         = 1.f;    // current envelope gain
    float attack_coeff_ = 0.f;    // per-sample attack multiplier toward target
    float release_coeff_= 0.f;    // per-sample release multiplier toward 1.0
    float gain_red_db_  = 0.f;    // metered gain reduction
    bool  enabled_      = false;
};
