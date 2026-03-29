/*
 * main.cpp — IthacaCoreResonator
 * ─────────────────────────────────
 * Usage:
 *   IthacaCoreResonator [params.json]
 *
 * Opens the default audio device, loads the physics param table,
 * and enters an interactive keyboard → MIDI loop.
 *
 * Mirrors IthacaCore main.cpp pattern:
 *   Logger → runResonator() → ResonatorEngine → audio callback
 */

#include "synth/resonator_engine.h"
#include <iostream>
#include <string>

int main(int argc, char* argv[]) {
    std::cout << "IthacaCoreResonator v1.0 — Physics Piano Synthesizer\n";

    const std::string params_json = (argc > 1)
        ? argv[1]
        : "../analysis/params.json";

    try {
        Logger logger;
        return runResonator(logger, params_json);

    } catch (const std::exception& e) {
        std::cerr << "CRITICAL ERROR: " << e.what() << "\n";
        return 1;
    } catch (...) {
        std::cerr << "UNKNOWN CRITICAL ERROR\n";
        return 1;
    }
}
