#pragma once
/*
 * midi_input.h
 * ─────────────
 * Cross-platform MIDI input via RtMidi.
 * Receives note-on/off, sustain pedal, and passes them to ResonatorEngine.
 */

#include "../third_party/RtMidi.h"
#include "resonator_engine.h"
#include <string>
#include <vector>

class MidiInput {
public:
    MidiInput() = default;
    ~MidiInput() { close(); }

    // List available MIDI input ports (for user selection)
    static std::vector<std::string> listPorts();

    // Open port by index (0 = first available). Returns false if none found.
    bool open(ResonatorEngine& engine, int port_index = 0);

    // Open virtual port (macOS/Linux — allows DAW to send MIDI)
    bool openVirtual(ResonatorEngine& engine, const std::string& name = "IthacaCoreResonator");

    void close();
    bool isOpen() const { return midi_ && midi_->isPortOpen(); }
    std::string portName() const { return port_name_; }

private:
    static void callback(double /*timestamp*/,
                         std::vector<unsigned char>* msg,
                         void* user_data);

    RtMidiIn*       midi_      = nullptr;
    ResonatorEngine* engine_   = nullptr;
    std::string     port_name_;
};
