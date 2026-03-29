#include "synth/voice_manager.h"
// #include "sampler/core_logger.h"   // uncomment when copied from IthacaCore
#include <iostream>
#include <string>

// Placeholder logger until core_logger is copied from IthacaCore
struct Logger {
    void log(const char* tag, int, const std::string& msg) {
        std::cout << "[" << tag << "] " << msg << "\n";
    }
};

int main(int argc, char* argv[]) {
    std::cout << "IthacaCoreResonator — Physics Piano Synthesizer\n";

    const std::string params_json = (argc > 1)
        ? argv[1]
        : "../analysis/params.json";

    try {
        Logger logger;
        ResonatorVoiceManager vm;
        vm.initialize(params_json, 44100.f, logger);

        if (!vm.isInitialized()) {
            std::cerr << "Failed to initialize synthesizer.\n";
            return 1;
        }

        std::cout << "Active voices: " << vm.activeVoiceCount() << "\n";
        std::cout << "Synthesizer ready.\n";

        // TODO: add interactive MIDI loop or offline render test

    } catch (const std::exception& e) {
        std::cerr << "ERROR: " << e.what() << "\n";
        return 1;
    }

    return 0;
}
