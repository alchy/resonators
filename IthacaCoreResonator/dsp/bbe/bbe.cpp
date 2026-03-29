#include "bbe.h"
#include <cmath>

static constexpr float PI = 3.14159265358979f;

// ── RBJ high shelf ─────────────────────────────────────────────────────────────
BiquadShelveCoeff BBE::highShelf(float fc, float gain_db, float sr) {
    float A  = std::pow(10.f, gain_db / 40.f);
    float w0 = 2.f * PI * fc / sr;
    float cosw = std::cos(w0);
    float sinw = std::sin(w0);
    float S  = 1.f;   // shelf slope = 1
    float al = sinw / 2.f * std::sqrt((A + 1.f/A) * (1.f/S - 1.f) + 2.f);

    float b0 =  A * ((A+1.f) + (A-1.f)*cosw + 2.f*std::sqrt(A)*al);
    float b1 = -2.f*A * ((A-1.f) + (A+1.f)*cosw);
    float b2 =  A * ((A+1.f) + (A-1.f)*cosw - 2.f*std::sqrt(A)*al);
    float a0 =       (A+1.f) - (A-1.f)*cosw + 2.f*std::sqrt(A)*al;
    float a1 =  2.f * ((A-1.f) - (A+1.f)*cosw);
    float a2 =        (A+1.f) - (A-1.f)*cosw - 2.f*std::sqrt(A)*al;

    return {b0/a0, b1/a0, b2/a0, a1/a0, a2/a0};
}

// ── RBJ low shelf ──────────────────────────────────────────────────────────────
BiquadShelveCoeff BBE::lowShelf(float fc, float gain_db, float sr) {
    float A  = std::pow(10.f, gain_db / 40.f);
    float w0 = 2.f * PI * fc / sr;
    float cosw = std::cos(w0);
    float sinw = std::sin(w0);
    float S  = 1.f;
    float al = sinw / 2.f * std::sqrt((A + 1.f/A) * (1.f/S - 1.f) + 2.f);

    float b0 =  A * ((A+1.f) - (A-1.f)*cosw + 2.f*std::sqrt(A)*al);
    float b1 =  2.f*A * ((A-1.f) - (A+1.f)*cosw);
    float b2 =  A * ((A+1.f) - (A-1.f)*cosw - 2.f*std::sqrt(A)*al);
    float a0 =       (A+1.f) + (A-1.f)*cosw + 2.f*std::sqrt(A)*al;
    float a1 = -2.f * ((A-1.f) + (A+1.f)*cosw);
    float a2 =        (A+1.f) + (A-1.f)*cosw - 2.f*std::sqrt(A)*al;

    return {b0/a0, b1/a0, b2/a0, a1/a0, a2/a0};
}

// ── Biquad DF-II transposed ────────────────────────────────────────────────────
float BBE::processBiquad(float x, BiquadShelveCoeff& c, BiquadShelveState& s) {
    float y = c.b0*x + c.b1*s.x1 + c.b2*s.x2 - c.a1*s.y1 - c.a2*s.y2;
    s.x2 = s.x1; s.x1 = x;
    s.y2 = s.y1; s.y1 = y;
    return y;
}

// ── Public ─────────────────────────────────────────────────────────────────────

void BBE::prepare(float sample_rate) {
    sample_rate_ = sample_rate;
    def_coeff_   = highShelf(5000.f, def_gain_db_,  sample_rate);
    bass_coeff_  = lowShelf (180.f,  bass_gain_db_, sample_rate);
    reset();
}

void BBE::setDefinition(float gain_db) {
    def_gain_db_ = gain_db;
    def_coeff_   = highShelf(5000.f, gain_db, sample_rate_);
}

void BBE::setBassBoost(float gain_db) {
    bass_gain_db_ = gain_db;
    bass_coeff_   = lowShelf(180.f, gain_db, sample_rate_);
}

void BBE::reset() {
    def_state_l_  = {};
    def_state_r_  = {};
    bass_state_l_ = {};
    bass_state_r_ = {};
}

void BBE::process(float* L, float* R, int n_samples) {
    if (!enabled_) return;

    for (int i = 0; i < n_samples; i++) {
        L[i] = processBiquad(L[i], def_coeff_,  def_state_l_);
        R[i] = processBiquad(R[i], def_coeff_,  def_state_r_);
        L[i] = processBiquad(L[i], bass_coeff_, bass_state_l_);
        R[i] = processBiquad(R[i], bass_coeff_, bass_state_r_);
    }
}
