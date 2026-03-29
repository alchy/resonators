#pragma once
/*
 * bbe.h — Simplified BBE Sonic Maximizer.
 *
 * Approximation of the BBE process using two RBJ biquad shelving filters:
 *   Definition : high shelf boost at 5 kHz   (0..12 dB, MIDI-controlled)
 *   Bass Boost : low  shelf boost at 180 Hz  (0..10 dB, MIDI-controlled)
 *
 * Applied to stereo L/R independently.
 */

struct BiquadShelveState { float x1=0,x2=0,y1=0,y2=0; };
struct BiquadShelveCoeff { float b0,b1,b2,a1,a2; };

class BBE {
public:
    void prepare(float sample_rate);
    void setDefinition(float gain_db);   // 0..12 dB
    void setBassBoost (float gain_db);   // 0..10 dB
    void setEnabled(bool on) { enabled_ = on; }
    bool isEnabled()   const { return enabled_; }

    void process(float* L, float* R, int n_samples);
    void reset();

private:
    static BiquadShelveCoeff highShelf(float fc, float gain_db, float sr);
    static BiquadShelveCoeff lowShelf (float fc, float gain_db, float sr);
    static float processBiquad(float x, BiquadShelveCoeff& c, BiquadShelveState& s);

    float sample_rate_ = 48000.f;

    BiquadShelveCoeff def_coeff_{};
    BiquadShelveState def_state_l_{}, def_state_r_{};

    BiquadShelveCoeff bass_coeff_{};
    BiquadShelveState bass_state_l_{}, bass_state_r_{};

    float def_gain_db_  = 0.f;
    float bass_gain_db_ = 0.f;
    bool  enabled_      = false;
};
