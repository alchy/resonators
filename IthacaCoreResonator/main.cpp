/*
 * main.cpp — IthacaCoreResonator
 * ─────────────────────────────────
 * Usage:
 *   IthacaCoreResonator [params.json] [midi_port]
 *
 *   params.json  — physics parameter table (default: ../analysis/params.json)
 *   midi_port    — MIDI input port index (default: 0, first available)
 *
 * At startup, lists all available MIDI ports.
 * If no MIDI hardware is present, keyboard fallback is active (a-k = C4-C5).
 * On macOS/Linux, also opens a virtual MIDI port for DAW routing.
 */

#include "synth/resonator_engine.h"
#include <iostream>
#include <string>
#include <cstdlib>
#include <cstdio>

int main(int argc, char* argv[]) {
    // Disable stdout/stderr buffering so logs are visible even when piped.
    setvbuf(stdout, nullptr, _IONBF, 0);
    setvbuf(stderr, nullptr, _IONBF, 0);
    std::cout << "IthacaCoreResonator v1.0 — Physics Piano Synthesizer\n\n";

    const std::string params_json = (argc > 1)
        ? argv[1]
        : "../analysis/params.json";

    int midi_port = (argc > 2) ? std::atoi(argv[2]) : 0;

    try {
        Logger logger;
        return runResonator(logger, params_json, midi_port);

    } catch (const std::exception& e) {
        std::cerr << "CRITICAL ERROR: " << e.what() << "\n";
        return 1;
    } catch (...) {
        std::cerr << "UNKNOWN CRITICAL ERROR\n";
        return 1;
    }
}
