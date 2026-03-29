#pragma once
#include <cstdint>
/*
 * dsp_chain.h — Master bus DSP chain: Limiter → BBE.
 *
 * Call order (per block):
 *   prepare()  — call once at init with sample_rate + max_block_size
 *   process()  — called from audio thread (RT-safe after prepare)
 *   reset()    — clear filter states
 *
 * MIDI mappings:
 *   Limiter threshold: MIDI 0..127 → -40..0 dB
 *   Limiter release:   MIDI 0..127 → 10..2000 ms
 *   BBE definition:    MIDI 0..127 → 0..12 dB  (5 kHz high shelf)
 *   BBE bass boost:    MIDI 0..127 → 0..10 dB  (180 Hz low shelf)
 */

#include "limiter/limiter.h"
#include "bbe/bbe.h"

class DspChain {
public:
    void prepare(float sample_rate, int max_block_size);
    void reset();

    // Process stereo block in-place (L/R non-interleaved).
    void process(float* L, float* R, int n_samples);

    // ── Limiter controls ──────────────────────────────────────────────────────
    void setLimiterThreshold(uint8_t midi);   // 127=0 dB, 0=-40 dB
    void setLimiterRelease  (uint8_t midi);   // 0=10 ms, 127=2000 ms
    void setLimiterEnabled  (uint8_t midi);   // >= 64 = on

    uint8_t getLimiterThreshold()     const { return lim_thr_midi_;   }
    uint8_t getLimiterRelease()       const { return lim_rel_midi_;   }
    uint8_t getLimiterEnabled()       const { return lim_ena_midi_;   }
    uint8_t getLimiterGainReduction() const;  // 0=no reduction, 127=full (-40 dB)

    // ── BBE controls ──────────────────────────────────────────────────────────
    void setBBEDefinition(uint8_t midi);   // 0..127 → 0..12 dB
    void setBBEBassBoost (uint8_t midi);   // 0..127 → 0..10 dB

    uint8_t getBBEDefinition() const { return bbe_def_midi_; }
    uint8_t getBBEBassBoost()  const { return bbe_bas_midi_; }

    Limiter& limiter() { return limiter_; }
    BBE&     bbe()     { return bbe_;     }

    int getEffectCount() const { return 2; }

private:
    Limiter limiter_;
    BBE     bbe_;

    uint8_t lim_thr_midi_ = 127;
    uint8_t lim_rel_midi_ = 64;
    uint8_t lim_ena_midi_ = 0;
    uint8_t bbe_def_midi_ = 0;
    uint8_t bbe_bas_midi_ = 0;
};
