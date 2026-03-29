/*
 * resonator_engine.cpp
 * ─────────────────────
 * Real-time engine using miniaudio for audio output.
 *
 * Audio thread flow (called by miniaudio every BLOCK_SIZE frames):
 *   audioCallback()
 *     → processBlock()
 *         → vm_.processBlockUninterleaved(buf_l_, buf_r_, frames)
 *         → interleave buf_l_ + buf_r_ → output (float32 stereo)
 *
 * All VoiceManager methods are called exclusively from the audio thread
 * EXCEPT noteOn/noteOff/sustainPedal which use atomic note queue.
 */

// miniaudio implementation — compiled once here
#define MINIAUDIO_IMPLEMENTATION
#include "miniaudio.h"

#include "resonator_engine.h"
#include "midi_input.h"
#include "note_lut.h"
#include <cstring>
#include <cstdio>
#include <stdexcept>
#include <algorithm>
#include <memory>

#ifdef _WIN32
  #include <conio.h>
#else
  #include <termios.h>
  #include <unistd.h>
  #include <fcntl.h>
#endif

// ── MIDI event queue (lock-free single-producer / single-consumer) ─────────────
// Simple ring buffer — main thread writes, audio thread reads.
struct MidiEvent {
    enum Type : uint8_t { NOTE_ON, NOTE_OFF, SUSTAIN } type;
    uint8_t midi;
    uint8_t value;
};

static constexpr int MIDI_QUEUE_SIZE = 256;
static MidiEvent  midi_queue[MIDI_QUEUE_SIZE];
static std::atomic<int> midi_write_idx{0};
static std::atomic<int> midi_read_idx{0};

static void pushMidi(MidiEvent::Type type, uint8_t midi, uint8_t val) {
    int w = midi_write_idx.load(std::memory_order_relaxed);
    int next = (w + 1) % MIDI_QUEUE_SIZE;
    if (next == midi_read_idx.load(std::memory_order_acquire)) return; // full, drop
    midi_queue[w] = {type, midi, val};
    midi_write_idx.store(next, std::memory_order_release);
}

// ── Constructor / Destructor ──────────────────────────────────────────────────

ResonatorEngine::ResonatorEngine()
    : device_(new ma_device{}) {}

ResonatorEngine::~ResonatorEngine() {
    stop();
    delete[] buf_l_;
    delete[] buf_r_;
    delete device_;
}

// ── initialize ────────────────────────────────────────────────────────────────

bool ResonatorEngine::initialize(const std::string& params_json_path,
                                  Logger& logger) {
    logger_ = &logger;
    logger.log("ResonatorEngine", LogSeverity::Info, "Initializing...");

    vm_.initialize(params_json_path, sample_rate_, logger);
    if (!vm_.isInitialized()) {
        logger.log("ResonatorEngine", LogSeverity::Error, "VoiceManager initialization failed");
        return false;
    }

    // Pre-allocate non-interleaved buffers (reused every callback)
    buf_l_ = new float[block_size_];
    buf_r_ = new float[block_size_];

    logger.log("ResonatorEngine", LogSeverity::Info,
        "Ready. SR=" + std::to_string(sample_rate_) +
        " block=" + std::to_string(block_size_));
    return true;
}

// ── Audio callback ────────────────────────────────────────────────────────────

void ResonatorEngine::audioCallback(ma_device*  device,
                                     void*       output,
                                     const void* /*input*/,
                                     uint32_t    frame_count) {
    auto* engine = reinterpret_cast<ResonatorEngine*>(device->pUserData);
    engine->processBlock(reinterpret_cast<float*>(output), frame_count);
}

void ResonatorEngine::processBlock(float* out_interleaved, uint32_t frame_count) {
    // Drain MIDI event queue (no alloc, no lock)
    int r = midi_read_idx.load(std::memory_order_acquire);
    int w = midi_write_idx.load(std::memory_order_relaxed);
    while (r != w) {
        const MidiEvent& ev = midi_queue[r];
        switch (ev.type) {
            case MidiEvent::NOTE_ON:  vm_.setNoteStateMIDI(ev.midi, true,  ev.value); break;
            case MidiEvent::NOTE_OFF: vm_.setNoteStateMIDI(ev.midi, false, 0);        break;
            case MidiEvent::SUSTAIN:  vm_.setSustainPedalMIDI(ev.value);              break;
        }
        r = (r + 1) % MIDI_QUEUE_SIZE;
    }
    midi_read_idx.store(r, std::memory_order_release);

    // Synthesize (handles blocks larger than our pre-allocated buffer in chunks)
    uint32_t remaining = frame_count;
    uint32_t offset    = 0;
    while (remaining > 0) {
        uint32_t chunk = remaining < (uint32_t)block_size_ ? remaining : (uint32_t)block_size_;
        vm_.processBlockUninterleaved(buf_l_, buf_r_, (int)chunk);

        // Interleave into float32 stereo output
        float* dst = out_interleaved + offset * 2;
        for (uint32_t i = 0; i < chunk; i++) {
            dst[i * 2]     = buf_l_[i];
            dst[i * 2 + 1] = buf_r_[i];
        }
        offset    += chunk;
        remaining -= chunk;
    }
}

// ── start / stop ─────────────────────────────────────────────────────────────

bool ResonatorEngine::start() {
    if (!logger_ || !vm_.isInitialized()) return false;

    ma_device_config cfg = ma_device_config_init(ma_device_type_playback);
    cfg.playback.format   = ma_format_f32;
    cfg.playback.channels = RESONATOR_DEFAULT_CHANNELS;
    cfg.sampleRate        = sample_rate_;
    cfg.dataCallback      = audioCallback;
    cfg.pUserData         = this;
    cfg.periodSizeInFrames= block_size_;

    if (ma_device_init(nullptr, &cfg, device_) != MA_SUCCESS) {
        logger_->log("ResonatorEngine", LogSeverity::Error, "Failed to open audio device");
        return false;
    }

    if (ma_device_start(device_) != MA_SUCCESS) {
        logger_->log("ResonatorEngine", LogSeverity::Error, "Failed to start audio device");
        ma_device_uninit(device_);
        return false;
    }

    running_.store(true);
    logger_->log("ResonatorEngine", LogSeverity::Info,
        "Audio device started: " + std::string(device_->playback.name));
    return true;
}

void ResonatorEngine::stop() {
    if (!running_.load()) return;
    ma_device_stop(device_);
    ma_device_uninit(device_);
    running_.store(false);
    if (logger_) logger_->log("ResonatorEngine", LogSeverity::Info, "Audio device stopped");
}

// ── Thread-safe MIDI interface ────────────────────────────────────────────────

void ResonatorEngine::noteOn (uint8_t midi, uint8_t velocity) {
    last_note_midi_.store(midi,     std::memory_order_relaxed);
    last_note_vel_ .store(velocity, std::memory_order_relaxed);
    pushMidi(MidiEvent::NOTE_ON, midi, velocity);
}
void ResonatorEngine::noteOff(uint8_t midi) {
    pushMidi(MidiEvent::NOTE_OFF, midi, 0);
}
void ResonatorEngine::sustainPedal(uint8_t val) {
    pushMidi(MidiEvent::SUSTAIN, 0, val);
}

// ── Master DSP ────────────────────────────────────────────────────────────────

void ResonatorEngine::setLimiterThreshold   (uint8_t v) { vm_.setLimiterThresholdMIDI(v);    }
void ResonatorEngine::setLimiterRelease     (uint8_t v) { vm_.setLimiterReleaseMIDI(v);      }
void ResonatorEngine::setBBEDefinition      (uint8_t v) { vm_.setBBEDefinitionMIDI(v);       }
void ResonatorEngine::setBBEBassBoost       (uint8_t v) { vm_.setBBEBassBoostMIDI(v);        }
void ResonatorEngine::setAllVoicesMasterGain(uint8_t v) { vm_.setAllVoicesMasterGainMIDI(v, *logger_); }
void ResonatorEngine::setAllVoicesPan       (uint8_t v) { vm_.setAllVoicesPanMIDI(v);        }
void ResonatorEngine::setAllVoicesPanSpeed  (uint8_t v) { vm_.setAllVoicesPanSpeedMIDI(v);   }
void ResonatorEngine::setAllVoicesPanDepth  (uint8_t v) { vm_.setAllVoicesPanDepthMIDI(v);   }

int ResonatorEngine::activeVoices() const { return vm_.activeVoiceCount(); }

// ── runResonator — main loop (mirrors runSampler) ─────────────────────────────

int runResonator(Logger& logger, const std::string& params_json_path,
                 int midi_port) {
    logger.log("runResonator", LogSeverity::Info,
               "=== IthacaCoreResonator STARTING ===");

    // Heap-allocate engine: ResonatorVoiceManager contains ~4 MB of data
    // (NoteLUT 2.8 MB + 88 ResonatorVoice objects) — too large for stack.
    auto engine = std::make_unique<ResonatorEngine>();
    if (!engine->initialize(params_json_path, logger)) {
        logger.log("runResonator", LogSeverity::Error, "Initialization failed");
        return 1;
    }
    if (!engine->start()) {
        logger.log("runResonator", LogSeverity::Error, "Audio start failed");
        return 1;
    }

    // ── MIDI input ────────────────────────────────────────────────────────────
    MidiInput midi;
    auto ports = MidiInput::listPorts();
    if (!ports.empty()) {
        std::printf("[MIDI] Available ports:\n");
        for (int i = 0; i < (int)ports.size(); i++)
            std::printf("  [%d] %s\n", i, ports[i].c_str());
        midi.open(*engine, midi_port);
    }
#ifndef _WIN32
    if (!midi.isOpen())
        midi.openVirtual(*engine);   // macOS/Linux virtual port as fallback
#endif

    // ── Keyboard fallback (a-k = C4..C5, z = sustain, q = quit) ─────────────
    const char  keys[] = "asdfghjk";
    const int  midis[] = { 60, 62, 64, 65, 67, 69, 71, 72 };
    bool       sustain = false;

    std::printf("\nKeyboard fallback: a-k = C4-C5  |  z = sustain  |  q = quit\n\n");

#ifdef _WIN32
    while (true) {
        if (_kbhit()) {
            int ch = _getch();
            if (ch == 'q' || ch == 'Q') break;
            if (ch == 'z') {
                sustain = !sustain;
                engine->sustainPedal(sustain ? 127 : 0);
                std::printf("Sustain: %s\n", sustain ? "ON" : "OFF");
                continue;
            }
            for (int i = 0; i < 8; i++) {
                if (ch == keys[i]) {
                    engine->noteOn((uint8_t)midis[i], 80);
                    ma_sleep(300);
                    engine->noteOff((uint8_t)midis[i]);
                }
            }
        }
        ma_sleep(1);
    }
#else
    struct termios oldt, newt;
    tcgetattr(STDIN_FILENO, &oldt);
    newt = oldt;
    newt.c_lflag &= ~(ICANON | ECHO);
    tcsetattr(STDIN_FILENO, TCSANOW, &newt);
    fcntl(STDIN_FILENO, F_SETFL, O_NONBLOCK);
    while (true) {
        char ch;
        if (read(STDIN_FILENO, &ch, 1) == 1) {
            if (ch == 'q' || ch == 'Q') break;
            if (ch == 'z') {
                sustain = !sustain;
                engine->sustainPedal(sustain ? 127 : 0);
                std::printf("Sustain: %s\n", sustain ? "ON" : "OFF");
            }
            for (int i = 0; i < 8; i++) {
                if (ch == keys[i]) {
                    engine->noteOn((uint8_t)midis[i], 80);
                    ma_sleep(300);
                    engine->noteOff((uint8_t)midis[i]);
                }
            }
        }
        ma_sleep(1);
    }
    tcsetattr(STDIN_FILENO, TCSANOW, &oldt);
#endif

    midi.close();
    engine->stop();
    logger.log("runResonator", LogSeverity::Info,
               "=== IthacaCoreResonator STOPPED ===");
    return 0;
}
