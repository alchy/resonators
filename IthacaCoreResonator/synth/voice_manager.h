#pragma once
#include "note_params.h"
#include "resonator_voice.h"
#include <string>
#include <vector>
#include <atomic>

// Forward declarations (from IthacaCore DSP layer)
class Logger;
class DspChain;

// Polyphonic voice manager — drop-in API replacement for IthacaCore VoiceManager.
//
// Manages up to 88 simultaneous ResonatorVoice instances (one per MIDI note).
// Master DSP bus: limiter + optional BBE enhancement (copied from IthacaCore).

class ResonatorVoiceManager {
public:
    ResonatorVoiceManager();
    ~ResonatorVoiceManager();

    // Load params.json and prepare all voices for playback.
    // Must be called before any audio processing.
    void initialize(const std::string& params_json_path,
                    float sample_rate,
                    Logger& logger);

    // ── MIDI interface (identical to IthacaCore VoiceManager) ────────────────

    void setNoteStateMIDI(uint8_t midi_note, bool note_on, uint8_t velocity);
    void setSustainPedalMIDI(uint8_t val);  // >= 64 = pedal down

    // ── Audio rendering ───────────────────────────────────────────────────────

    // Mix all active voices into out_l / out_r (non-interleaved float32).
    // Applies master DSP (limiter, BBE) in-place after summing voices.
    void processBlockUninterleaved(float* out_l, float* out_r, int n_samples);

    // ── Master DSP controls (MIDI 0–127) ─────────────────────────────────────

    void setLimiterThresholdMIDI(uint8_t val);   // 0 = -20 dB, 127 = 0 dB (off)
    void setLimiterReleaseMIDI(uint8_t val);      // 0 = 1 ms, 127 = 1000 ms
    void setLimiterEnabledMIDI(uint8_t val);      // >= 64 = enabled
    void setBBEDefinitionMIDI(uint8_t val);        // harmonic enhancement 0–127
    void setBBEBassBoostMIDI(uint8_t val);         // bass boost 0–127

    // ── Diagnostics ───────────────────────────────────────────────────────────

    int  activeVoiceCount() const;
    bool isInitialized()    const { return initialized_; }

private:
    void handleNoteOn (uint8_t midi, uint8_t vel);
    void handleNoteOff(uint8_t midi);

    NoteLUT              lut_;
    ResonatorVoice       voices_[MIDI_COUNT];      // one slot per MIDI note 21..108
    float                sample_rate_ = 44100.f;
    bool                 initialized_ = false;
    Logger*              logger_      = nullptr;

    std::atomic<bool>    sustain_pedal_{false};
    std::vector<uint8_t> held_notes_;             // waiting for pedal release

    // Master bus DSP (from IthacaCore — limiter + BBE)
    // DspChain             master_dsp_;           // uncomment when DSP files copied
};
