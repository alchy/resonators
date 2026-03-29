#pragma once
/*
 * voice_manager.h
 * ────────────────
 * Polyphonic voice pool — API-compatible with IthacaCore VoiceManager.
 *
 * Differences from IthacaCore:
 *   - No WAV sample loading (physics synthesis from params.json)
 *   - No ADSR envelope (bi-exp decay per partial in ResonatorVoice)
 *   - initialize() replaces initializeSystem/loadForSampleRate/loadSampleBank
 *   - Voice slots indexed 0..87 (MIDI 21..108) instead of 0..127
 */

#include "note_params.h"
#include "note_lut.h"
#include "resonator_voice.h"
#include "synth_config.h"
#include "../sampler/core_logger.h"
#include "../dsp/dsp_chain.h"

#include <string>
#include <vector>
#include <array>
#include <atomic>
#include <cstdint>

static constexpr uint8_t ITHACA_DEFAULT_VELOCITY = 80;

class ResonatorVoiceManager {
public:
    ResonatorVoiceManager();
    ~ResonatorVoiceManager();

    // ── Initialization ────────────────────────────────────────────────────────
    // All-in-one: load params.json, prepare voices, set sample rate.
    // Equivalent to IthacaCore: initializeSystem → loadForSampleRate → prepareToPlay
    void initialize(const std::string& params_json_path,
                    float sample_rate,
                    Logger& logger);

    // Change sample rate after initialization (recomputes voice decay coeffs on next noteOn)
    void changeSampleRate(float new_sample_rate, Logger& logger);
    float getCurrentSampleRate() const noexcept { return sample_rate_; }

    // Prepare audio buffers for given max block size (mirrors JUCE prepareToPlay)
    void prepareToPlay(int max_block_size);

    // ── MIDI note control ─────────────────────────────────────────────────────

    void setNoteStateMIDI(uint8_t midi_note, bool note_on, uint8_t velocity) noexcept;
    void setNoteStateMIDI(uint8_t midi_note, bool note_on) noexcept;  // default velocity

    void setSustainPedalMIDI(uint8_t val)  noexcept;  // >= 64 = down
    void setSustainPedalMIDI(bool pedal_down) noexcept;
    bool getSustainPedalActive() const noexcept { return sustain_pedal_.load(); }

    // ── Audio rendering ───────────────────────────────────────────────────────

    // Full block: zero buffers, sum voices, apply LFO pan + DSP. Returns true if any active.
    bool processBlockUninterleaved(float* out_l, float* out_r, int n_samples) noexcept;

    // Segment variant for sample-accurate MIDI: accumulates WITHOUT zeroing or DSP.
    // Call finalizeBlock() after all segments to apply LFO + DSP.
    bool processBlockSegment(float* out_l, float* out_r, int n_samples) noexcept;
    void finalizeBlock      (float* out_l, float* out_r, int n_samples) noexcept;

    // Interleaved stereo output (L0 R0 L1 R1 ...) — uses internal temp buffers
    bool processBlockInterleaved(float* out_interleaved, int n_samples) noexcept;

    // Apply LFO pan to an already-rendered stereo buffer
    void applyLfoPanToFinalMix(float* out_l, float* out_r, int n_samples) noexcept;

    // ── Voice control ─────────────────────────────────────────────────────────

    void stopAllVoices()              noexcept;
    void resetAllVoices(Logger& logger);

    // ── Global voice parameters ───────────────────────────────────────────────

    void setAllVoicesMasterGainMIDI      (uint8_t val, Logger& logger);
    void setAllVoicesPanMIDI             (uint8_t val) noexcept;  // 64 = centre
    void setAllVoicesStereoFieldAmountMIDI(uint8_t val) noexcept; // 0=mono,127=full

    // LFO panning (electric-piano style)
    void setAllVoicesPanSpeedMIDI (uint8_t val) noexcept;  // 0..127 → 0..2 Hz
    void setAllVoicesPanDepthMIDI (uint8_t val) noexcept;  // 0..127 → 0..1
    bool isLfoPanningActive()      const noexcept;

    // ── Statistics ────────────────────────────────────────────────────────────

    int  getActiveVoicesCount()    const noexcept;
    int  activeVoiceCount()        const noexcept { return getActiveVoicesCount(); }
    int  getSustainingVoicesCount() const noexcept;
    int  getReleasingVoicesCount()  const noexcept;

    void setRealTimeMode(bool enabled) noexcept { rt_mode_.store(enabled); }
    bool isRealTimeMode()          const noexcept { return rt_mode_.load(); }

    bool isInitialized() const noexcept { return initialized_; }

    void logSystemStatistics(Logger& logger);

    // ── DSP effects (master bus) ──────────────────────────────────────────────

    void setLimiterThresholdMIDI  (uint8_t val) noexcept;
    void setLimiterReleaseMIDI    (uint8_t val) noexcept;
    void setLimiterEnabledMIDI    (uint8_t val) noexcept;
    uint8_t getLimiterThresholdMIDI()   const noexcept;
    uint8_t getLimiterReleaseMIDI()     const noexcept;
    uint8_t getLimiterEnabledMIDI()     const noexcept;
    uint8_t getLimiterGainReductionMIDI() const noexcept;

    void setBBEDefinitionMIDI (uint8_t val) noexcept;
    void setBBEBassBoostMIDI  (uint8_t val) noexcept;

    DspChain* getDspChain() noexcept { return &dsp_chain_; }

    // Output peak level (linear, after full DSP chain). -20 dB/s decay.
    // Thread-safe: written by audio thread, read by GUI thread.
    float    getOutputPeakLin()  const noexcept { return output_peak_lin_.load(std::memory_order_relaxed); }
    uint32_t getLastNoteSeed()   const noexcept { return last_note_seed_ .load(std::memory_order_relaxed); }

    // ── Synthesis rendering config (mirrors physics_synth.py kwargs) ──────────
    // Changes take effect on next noteOn.
    void setSynthPanSpread         (float v) noexcept { synth_cfg_.pan_spread          = v; }
    void setSynthBeatScale         (float v) noexcept { synth_cfg_.beat_scale          = v; }
    void setSynthStereoDecorr      (float v) noexcept { synth_cfg_.stereo_decorr       = v; }
    void setSynthStereoBoost       (float v) noexcept { synth_cfg_.stereo_boost        = v; }
    void setSynthEqStrength        (float v) noexcept { synth_cfg_.eq_strength         = v; }
    void setSynthEqFreqMin         (float v) noexcept { synth_cfg_.eq_freq_min         = v; }
    void setSynthNoiseLevel        (float v) noexcept { synth_cfg_.noise_level         = v; }
    void setSynthOnsetMs           (float v) noexcept { synth_cfg_.onset_ms            = v; }
    void setSynthHarmonicBrightness(float v) noexcept { synth_cfg_.harmonic_brightness = v; }
    void setSynthTargetRms         (float v) noexcept { synth_cfg_.target_rms          = v; }
    void setSynthVelGamma          (float v) noexcept { synth_cfg_.vel_gamma           = v; }

    float getSynthPanSpread()          const noexcept { return synth_cfg_.pan_spread;          }
    float getSynthBeatScale()          const noexcept { return synth_cfg_.beat_scale;          }
    float getSynthStereoDecorr()       const noexcept { return synth_cfg_.stereo_decorr;       }
    float getSynthStereoBoost()        const noexcept { return synth_cfg_.stereo_boost;        }
    float getSynthEqStrength()         const noexcept { return synth_cfg_.eq_strength;         }
    float getSynthEqFreqMin()          const noexcept { return synth_cfg_.eq_freq_min;         }
    float getSynthNoiseLevel()         const noexcept { return synth_cfg_.noise_level;         }
    float getSynthOnsetMs()            const noexcept { return synth_cfg_.onset_ms;            }
    float getSynthHarmonicBrightness() const noexcept { return synth_cfg_.harmonic_brightness; }
    float getSynthTargetRms()          const noexcept { return synth_cfg_.target_rms;          }
    float getSynthVelGamma()           const noexcept { return synth_cfg_.vel_gamma;           }

    const SynthConfig& getSynthConfig() const noexcept { return synth_cfg_; }

    // Look up interpolated NoteParams for a MIDI note + velocity (GUI read — LUT is read-only).
    NoteParams lookupNote(int midi, int vel) const noexcept {
        if (!initialized_) return {};
        float vel_pos = (float)vel * (VEL_LAYERS - 1.f) / 127.f;
        return interpolateNoteLayers(lut_, midi, vel_pos);
    }

private:
    void handleNoteOn (uint8_t midi, uint8_t vel) noexcept;
    void handleNoteOff(uint8_t midi)               noexcept;
    void processDelayedNoteOffs()                  noexcept;

    // ── Data ──────────────────────────────────────────────────────────────────

    NoteLUT        lut_;
    ResonatorVoice voices_[MIDI_COUNT];     // indexed [midi - MIDI_MIN]

    float   sample_rate_    = 44100.f;
    int     max_block_size_ = 512;
    bool    initialized_    = false;
    Logger* logger_         = nullptr;

    // Master gain / pan
    float   master_gain_    = 1.f;
    float   pan_l_          = 1.f;   // computed from setAllVoicesPanMIDI
    float   pan_r_          = 1.f;
    float   stereo_field_   = 1.f;   // 0=mono, 1=full width

    // Sustain pedal
    std::atomic<bool>   sustain_pedal_{false};
    std::array<bool,128> delayed_note_offs_{};

    // LFO panning
    float lfo_speed_  = 0.f;   // Hz
    float lfo_depth_  = 0.f;   // 0..1
    float lfo_phase_  = 0.f;   // radians

    // RT mode
    std::atomic<bool> rt_mode_{false};

    // DSP chain (limiter + BBE — populated when dsp/ files are present)
    DspChain dsp_chain_;

    // Synthesis rendering config (forwarded to each voice at noteOn)
    SynthConfig synth_cfg_;

    // Limiter state cache (for getters, since DspChain stubs don't store values yet)
    uint8_t limiter_threshold_midi_ = 127;
    uint8_t limiter_release_midi_   = 64;
    uint8_t limiter_enabled_midi_   = 0;

    // BBE state cache
    uint8_t bbe_definition_midi_    = 0;
    uint8_t bbe_bass_boost_midi_    = 0;

    // Temp buffers for processBlockInterleaved (allocated in prepareToPlay)
    std::vector<float> tmp_l_;
    std::vector<float> tmp_r_;

    // Peak metering (written by audio thread, read by GUI)
    std::atomic<float>    output_peak_lin_{0.f};
    float                 peak_decay_coeff_ = 0.9878f;  // recomputed in prepareToPlay

    // Seed snapshot of last triggered note (first rand() value from phase init)
    std::atomic<uint32_t> last_note_seed_{0};
};
