#pragma once
/*
 * resonator_engine.h
 * ───────────────────
 * Top-level real-time engine — mirrors IthacaCore's Sampler/runSampler pattern.
 *
 * Responsibilities:
 *   - Initialize ResonatorVoiceManager (load params.json)
 *   - Open audio device via miniaudio
 *   - Run real-time audio callback (processBlock per buffer)
 *   - Thread-safe MIDI note trigger from any thread
 *
 * Usage:
 *   ResonatorEngine engine;
 *   engine.initialize("params.json", logger);
 *   engine.start();                  // opens audio device, starts callback
 *   engine.noteOn(60, 3);
 *   engine.noteOff(60);
 *   engine.stop();
 */

#include "voice_manager.h"
#include "note_params.h"
#include "../sampler/core_logger.h"
#include "../dsp/dsp_chain.h"
#include <string>
#include <atomic>
#include <cstdint>

// Forward-declare miniaudio device (avoid pulling the header into every TU)
struct ma_device;

// ── Constants (mirrors IthacaCore IthacaConfig.h) ─────────────────────────────
static constexpr int   RESONATOR_DEFAULT_SAMPLE_RATE = 48000;
static constexpr int   RESONATOR_DEFAULT_BLOCK_SIZE  = 256;
static constexpr int   RESONATOR_DEFAULT_CHANNELS    = 2;

// ── ResonatorEngine ───────────────────────────────────────────────────────────

class ResonatorEngine {
public:
    ResonatorEngine();
    ~ResonatorEngine();

    // Phase 1: load params, prepare voice pool
    bool initialize(const std::string& params_json_path, Logger& logger);

    // Phase 2: open audio device and start real-time callback
    bool start();

    // Phase 3: stop audio device (blocks until callback thread exits)
    void stop();

    bool isRunning() const { return running_.load(); }

    // ── Thread-safe MIDI interface ────────────────────────────────────────────
    // Safe to call from any thread (main, MIDI, GUI).
    void noteOn (uint8_t midi, uint8_t velocity);
    void noteOff(uint8_t midi);
    void sustainPedal(uint8_t val);     // val >= 64 = down

    // ── Master DSP + mix controls (forwarded to VoiceManager) ────────────────
    void setLimiterThreshold   (uint8_t v);
    void setLimiterRelease     (uint8_t v);
    void setBBEDefinition      (uint8_t v);
    void setBBEBassBoost       (uint8_t v);
    void setAllVoicesMasterGain(uint8_t v);
    void setAllVoicesPan       (uint8_t v);
    void setAllVoicesPanSpeed  (uint8_t v);
    void setAllVoicesPanDepth  (uint8_t v);

    // ── Stats ─────────────────────────────────────────────────────────────────
    int  activeVoices() const;
    int  sampleRate()   const { return sample_rate_; }
    int  blockSize()    const { return block_size_; }

    // ── DSP chain access (for GUI metering / state readback) ──────────────────
    DspChain*          getDspChain()      { return vm_.getDspChain(); }
    float              getOutputPeakLin() const { return vm_.getOutputPeakLin(); }
    const SynthConfig& getSynthConfig()   const { return vm_.getSynthConfig(); }

    // Last note-on (updated on every noteOn call, thread-safe)
    uint8_t  getLastNoteMidi() const { return last_note_midi_.load(std::memory_order_relaxed); }
    uint8_t  getLastNoteVel()  const { return last_note_vel_ .load(std::memory_order_relaxed); }
    uint32_t getLastNoteSeed() const { return vm_.getLastNoteSeed(); }

    // Look up interpolated NoteParams for a (midi, vel) — read-only LUT, safe from GUI thread
    NoteParams lookupNote(int midi, int vel) const { return vm_.lookupNote(midi, vel); }

private:
    // Called by miniaudio audio thread — must be RT-safe (no alloc, no lock)
    static void audioCallback(ma_device* device,
                               void*       output,
                               const void* input,
                               uint32_t    frame_count);

    void processBlock(float* out_interleaved, uint32_t frame_count);

    ResonatorVoiceManager  vm_;
    Logger*                logger_      = nullptr;
    std::atomic<uint8_t>   last_note_midi_{60};
    std::atomic<uint8_t>   last_note_vel_ {80};
    ma_device*             device_      = nullptr;   // heap-allocated (opaque type)
    std::atomic<bool>      running_     {false};
    int                    sample_rate_ = RESONATOR_DEFAULT_SAMPLE_RATE;
    int                    block_size_  = RESONATOR_DEFAULT_BLOCK_SIZE;

    // Temporary non-interleaved buffers (allocated once in initialize)
    float* buf_l_ = nullptr;
    float* buf_r_ = nullptr;
};

// ── Convenience: full startup + interactive loop (like runSampler) ────────────
// midi_port: index into MidiInput::listPorts() (-1 = auto/first)
int runResonator(Logger& logger, const std::string& params_json_path,
                 int midi_port = 0);
