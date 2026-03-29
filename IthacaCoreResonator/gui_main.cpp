/*
 * gui_main.cpp — IthacaCoreResonatorGUI
 * ────────────────────────────────────
 * Launches the engine + Dear ImGui window.
 * Usage:
 *   IthacaCoreResonatorGUI [params.json] [midi_port]
 */
#include "synth/resonator_engine.h"
#include "gui/resonator_gui.h"
#include <iostream>
#include <string>
#include <cstdlib>
#include <cstdio>
#include <memory>

int main(int argc, char* argv[]) {
    setvbuf(stdout, nullptr, _IONBF, 0);
    setvbuf(stderr, nullptr, _IONBF, 0);

    const std::string params_json = (argc > 1)
        ? argv[1]
        : "../analysis/params-salamander.json";

    try {
        Logger logger;
        logger.log("main", LogSeverity::Info,
                   "=== IthacaCoreResonatorGUI STARTING ===");

        auto engine = std::make_unique<ResonatorEngine>();
        if (!engine->initialize(params_json, logger)) {
            logger.log("main", LogSeverity::Error, "Engine init failed");
            return 1;
        }
        if (!engine->start()) {
            logger.log("main", LogSeverity::Error, "Audio start failed");
            return 1;
        }

        int ret = runResonatorGui(*engine, logger, params_json);

        engine->stop();
        logger.log("main", LogSeverity::Info,
                   "=== IthacaCoreResonatorGUI STOPPED ===");
        return ret;

    } catch (const std::exception& e) {
        std::cerr << "CRITICAL ERROR: " << e.what() << "\n";
        return 1;
    }
}
