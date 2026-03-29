#pragma once
#include "../synth/resonator_engine.h"
#include "../sampler/core_logger.h"
#include <string>

// Run the GUI event loop (blocks until window closed).
// engine must already be initialized and started.
int runResonatorGui(ResonatorEngine& engine, Logger& logger,
                    const std::string& params_json_path);
