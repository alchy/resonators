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
#include <string>
#include <atomic>
#include <cstdint>

// Forward-declare miniaudio device (avoid pulling the header into every TU)
struct ma_device;

// Placeholder logger interface (replace with core_logger when available)
struct Logger {
    void log(const char* tag, int /*severity*/, const std::string& msg);
};

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

    // ── Master DSP controls ───────────────────────────────────────────────────
    void setLimiterThreshold(uint8_t midi_val);
    void setLimiterRelease  (uint8_t midi_val);
    void setBBEDefinition   (uint8_t midi_val);
    void setBBEBassBoost    (uint8_t midi_val);

    // ── Stats ─────────────────────────────────────────────────────────────────
    int  activeVoices() const;
    int  sampleRate()   const { return sample_rate_; }
    int  blockSize()    const { return block_size_; }

private:
    // Called by miniaudio audio thread — must be RT-safe (no alloc, no lock)
    static void audioCallback(ma_device* device,
                               void*       output,
                               const void* input,
                               uint32_t    frame_count);

    void processBlock(float* out_interleaved, uint32_t frame_count);

    ResonatorVoiceManager  vm_;
    Logger*                logger_      = nullptr;
    ma_device*             device_      = nullptr;   // heap-allocated (opaque type)
    std::atomic<bool>      running_     {false};
    int                    sample_rate_ = RESONATOR_DEFAULT_SAMPLE_RATE;
    int                    block_size_  = RESONATOR_DEFAULT_BLOCK_SIZE;

    // Temporary non-interleaved buffers (allocated once in initialize)
    float* buf_l_ = nullptr;
    float* buf_r_ = nullptr;
};

// ── Convenience: full startup + interactive loop (like runSampler) ────────────
int runResonator(Logger& logger, const std::string& params_json_path);
