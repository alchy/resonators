#pragma once
#include "note_params.h"

static constexpr int EQ_BANDS = 8;   // number of biquad bands in cascade

// 8-band peaking-EQ biquad cascade.
//
// Designed at note-on from NoteParams::eq_gains_db[EQ_BANDS].
// Band centre frequencies (Hz): 80, 160, 320, 640, 1250, 2500, 5000, 12000
// Q factor: 1.4 (moderate bandwidth, smooth response)
//
// Processing cost: 8 × (5 muls + 4 adds) per sample — negligible.

struct BiquadCoeffs {
    float b0, b1, b2;   // feedforward
    float a1, a2;        // feedback (a0 normalised to 1)
};

struct BiquadState {
    float x1 = 0.f, x2 = 0.f;
    float y1 = 0.f, y2 = 0.f;
};

class BiquadEQ {
public:
    // Design all bands from the 64-point EQ curve (params.json spectral_eq).
    // Interpolates gain at each band centre, designs RBJ peaking biquads.
    // Call once at note-on.
    // freq_min: EQ is faded to 0 dB below this frequency (mirrors Python eq_freq_min=400 Hz).
    // Prevents room-acoustics contamination from LTASE ratio at low frequencies.
    void design(const float eq_freqs_hz[EQ_POINTS],
                const float eq_gains_db[EQ_POINTS],
                float sample_rate,
                float freq_min = 400.f);

    // Reset delay-line state (call at note-on after design).
    void reset();

    // Process one sample in-place.
    float processSample(float x);

    // Process a block in-place.
    void processBlock(float* buf, int n_samples);

private:
    BiquadCoeffs coeffs_[EQ_BANDS];
    BiquadState  state_[EQ_BANDS];

    static BiquadCoeffs designPeaking(float fc_hz, float gain_db,
                                       float Q, float sample_rate);
};

// Band centre frequencies used by BiquadEQ::design()
static constexpr float EQ_BAND_FREQS_HZ[EQ_BANDS] = {
    80.f, 160.f, 320.f, 640.f, 1250.f, 2500.f, 5000.f, 12000.f
};
