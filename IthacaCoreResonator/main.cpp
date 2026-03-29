/*
 * main.cpp — IthacaCoreResonator
 * ─────────────────────────────────
 * Usage:
 *   IthacaCoreResonator [params.json] [midi] [vel] [output.wav]
 *
 * Default: renders m060_vel3 (middle C, velocity layer 3) to exports/m060_vel3.wav
 */

#include "synth/voice_manager.h"
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <cstdint>
#include <algorithm>
#include <cstdio>

// ── Minimal placeholder logger ────────────────────────────────────────────────
// Replace with IthacaCore core_logger when dsp/ files are copied.
struct Logger {
    void log(const char* tag, int /*severity*/, const std::string& msg) {
        std::printf("[%s] %s\n", tag, msg.c_str());
    }
};

// ── Minimal WAV writer (16-bit stereo, no libsndfile dependency) ──────────────
static bool writeWav(const std::string& path,
                     const float* L, const float* R,
                     int n_samples, int sample_rate) {
    std::ofstream f(path, std::ios::binary);
    if (!f) return false;

    uint32_t data_size = n_samples * 2 * sizeof(int16_t);
    uint32_t riff_size = 36 + data_size;

    // RIFF header
    f.write("RIFF", 4);
    f.write(reinterpret_cast<const char*>(&riff_size), 4);
    f.write("WAVE", 4);
    // fmt chunk
    f.write("fmt ", 4);
    uint32_t fmt_size = 16; f.write(reinterpret_cast<const char*>(&fmt_size), 4);
    uint16_t pcm = 1;       f.write(reinterpret_cast<const char*>(&pcm), 2);
    uint16_t ch  = 2;       f.write(reinterpret_cast<const char*>(&ch),  2);
    uint32_t sr  = sample_rate; f.write(reinterpret_cast<const char*>(&sr), 4);
    uint32_t byte_rate = sr * 2 * 2; f.write(reinterpret_cast<const char*>(&byte_rate), 4);
    uint16_t block_align = 4; f.write(reinterpret_cast<const char*>(&block_align), 2);
    uint16_t bits = 16;    f.write(reinterpret_cast<const char*>(&bits), 2);
    // data chunk
    f.write("data", 4);
    f.write(reinterpret_cast<const char*>(&data_size), 4);

    // Interleave and convert to int16
    for (int i = 0; i < n_samples; i++) {
        auto to16 = [](float x) -> int16_t {
            x = std::max(-1.f, std::min(1.f, x));
            return (int16_t)(x * 32767.f);
        };
        int16_t l = to16(L[i]);
        int16_t r = to16(R[i]);
        f.write(reinterpret_cast<const char*>(&l), 2);
        f.write(reinterpret_cast<const char*>(&r), 2);
    }
    return true;
}

// ── main ──────────────────────────────────────────────────────────────────────

int main(int argc, char* argv[]) {
    std::printf("IthacaCoreResonator — Physics Piano Synthesizer\n");

    // Defaults
    std::string params_json = "../analysis/params.json";
    int  midi_note  = 60;   // middle C
    int  vel_layer  = 3;
    float duration  = 4.f;  // seconds to render
    std::string out_wav = "exports/m060_vel3.wav";

    if (argc > 1) params_json = argv[1];
    if (argc > 2) midi_note   = std::atoi(argv[2]);
    if (argc > 3) vel_layer   = std::atoi(argv[3]);
    if (argc > 4) out_wav     = argv[4];

    // Ensure output directory exists
    std::system("mkdir -p exports 2>/dev/null || mkdir exports 2>nul");

    Logger logger;
    ResonatorVoiceManager vm;
    vm.initialize(params_json, 48000.f, logger);

    if (!vm.isInitialized()) {
        std::fprintf(stderr, "ERROR: could not initialize synthesizer\n");
        return 1;
    }

    const int SR         = 48000;
    const int BLOCK_SIZE = 256;
    const int n_blocks   = (int)(duration * SR / BLOCK_SIZE) + 1;
    const int n_total    = n_blocks * BLOCK_SIZE;

    std::vector<float> out_l(n_total, 0.f);
    std::vector<float> out_r(n_total, 0.f);

    // Trigger note-on
    vm.setNoteStateMIDI((uint8_t)midi_note, true, (uint8_t)(vel_layer * 16 + 8));

    // Render blocks
    for (int b = 0; b < n_blocks; b++) {
        int offset = b * BLOCK_SIZE;
        // Release at 75% of duration for a natural decay
        if (b == (int)(n_blocks * 0.75f))
            vm.setNoteStateMIDI((uint8_t)midi_note, false, 0);

        vm.processBlockUninterleaved(out_l.data() + offset,
                                      out_r.data() + offset,
                                      BLOCK_SIZE);
    }

    // Export WAV
    if (!writeWav(out_wav, out_l.data(), out_r.data(), n_total, SR)) {
        std::fprintf(stderr, "ERROR: could not write %s\n", out_wav.c_str());
        return 1;
    }

    std::printf("Written: %s  (%d samples, %.1f s)\n",
                out_wav.c_str(), n_total, (float)n_total / SR);
    std::printf("Active voices at end: %d\n", vm.activeVoiceCount());

    return 0;
}
