#include "dsp_chain.h"
#include <cstdint>
#include <cmath>
#include <algorithm>

static float midiToDb(uint8_t midi, float db_min, float db_max) {
    return db_min + (db_max - db_min) * (midi / 127.f);
}

static float midiToMs(uint8_t midi, float ms_min, float ms_max) {
    return ms_min + (ms_max - ms_min) * (midi / 127.f);
}

void DspChain::prepare(float sample_rate, int max_block_size) {
    limiter_.prepare(sample_rate, max_block_size);
    bbe_.prepare(sample_rate);
    // Apply cached MIDI values
    setLimiterThreshold(lim_thr_midi_);
    setLimiterRelease  (lim_rel_midi_);
    setLimiterEnabled  (lim_ena_midi_);
    setBBEDefinition   (bbe_def_midi_);
    setBBEBassBoost    (bbe_bas_midi_);
}

void DspChain::reset() {
    bbe_.reset();
}

void DspChain::process(float* L, float* R, int n_samples) {
    bbe_.process    (L, R, n_samples);
    limiter_.process(L, R, n_samples);
}

// ── Limiter ───────────────────────────────────────────────────────────────────

void DspChain::setLimiterThreshold(uint8_t midi) {
    lim_thr_midi_ = midi;
    // MIDI 127 = 0 dB, MIDI 0 = -40 dB
    float db = midiToDb(midi, -40.f, 0.f);
    limiter_.setThresholdDb(db);
}

void DspChain::setLimiterRelease(uint8_t midi) {
    lim_rel_midi_ = midi;
    float ms = midiToMs(midi, 10.f, 2000.f);
    limiter_.setReleaseMs(ms);
}

void DspChain::setLimiterEnabled(uint8_t midi) {
    lim_ena_midi_ = midi;
    limiter_.setEnabled(midi >= 64);
}

uint8_t DspChain::getLimiterGainReduction() const {
    float db = limiter_.gainReductionDb();   // 0..-40 dB
    // Map 0 dB=0, -40 dB=127
    float v = std::max(0.f, std::min(-db / 40.f, 1.f)) * 127.f;
    return (uint8_t)v;
}

// ── BBE ───────────────────────────────────────────────────────────────────────

void DspChain::setBBEDefinition(uint8_t midi) {
    bbe_def_midi_ = midi;
    float db = 12.f * (midi / 127.f);
    bbe_.setDefinition(db);
    bbe_.setEnabled(bbe_def_midi_ > 0 || bbe_bas_midi_ > 0);
}

void DspChain::setBBEBassBoost(uint8_t midi) {
    bbe_bas_midi_ = midi;
    float db = 10.f * (midi / 127.f);
    bbe_.setBassBoost(db);
    bbe_.setEnabled(bbe_def_midi_ > 0 || bbe_bas_midi_ > 0);
}
