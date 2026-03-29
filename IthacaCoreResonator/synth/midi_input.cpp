/*
 * midi_input.cpp
 * ───────────────
 * MIDI message parsing and routing to ResonatorEngine.
 *
 * Handled messages:
 *   0x80 / 0x90  Note Off / Note On
 *   0xB0 CC 64   Sustain pedal
 *   0xB0 CC 7    Channel volume → setAllVoicesMasterGainMIDI
 *   0xB0 CC 10   Pan → setAllVoicesPanMIDI
 *   0xB0 CC 91   Reverb (LFO depth)
 *   0xB0 CC 93   Chorus (LFO speed)
 */

#include "midi_input.h"
#include <iostream>
#include <stdexcept>
#include <chrono>

static uint64_t nowMs() {
    return (uint64_t)std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
}

// ── listPorts ─────────────────────────────────────────────────────────────────

std::vector<std::string> MidiInput::listPorts() {
    std::vector<std::string> names;
    try {
        RtMidiIn midi;
        unsigned int n = midi.getPortCount();
        for (unsigned int i = 0; i < n; i++)
            names.push_back(midi.getPortName(i));
    } catch (...) {}
    return names;
}

// ── open ──────────────────────────────────────────────────────────────────────

bool MidiInput::open(ResonatorEngine& engine, int port_index) {
    close();
    try {
        midi_   = new RtMidiIn();
        engine_ = &engine;

        unsigned int n = midi_->getPortCount();
        if (n == 0) {
            std::cout << "[MIDI] No input ports found — keyboard fallback active\n";
            delete midi_; midi_ = nullptr;
            return false;
        }

        int idx = (port_index < 0 || port_index >= (int)n) ? 0 : port_index;
        port_name_ = midi_->getPortName(idx);
        midi_->openPort(idx);
        midi_->ignoreTypes(true, true, true);  // ignore sysex, timing, active sensing
        midi_->setCallback(&MidiInput::callback, this);
        std::cout << "[MIDI] Opened: " << port_name_ << "\n";
        return true;
    } catch (RtMidiError& e) {
        std::cerr << "[MIDI] Error: " << e.getMessage() << "\n";
        delete midi_; midi_ = nullptr;
        return false;
    }
}

bool MidiInput::openVirtual(ResonatorEngine& engine, const std::string& name) {
    close();
    try {
        midi_   = new RtMidiIn();
        engine_ = &engine;
        midi_->openVirtualPort(name);
        midi_->ignoreTypes(true, true, true);
        midi_->setCallback(&MidiInput::callback, this);
        port_name_ = name + " (virtual)";
        std::cout << "[MIDI] Virtual port: " << name << "\n";
        return true;
    } catch (RtMidiError& e) {
        std::cerr << "[MIDI] Virtual port error: " << e.getMessage() << "\n";
        delete midi_; midi_ = nullptr;
        return false;
    }
}

void MidiInput::close() {
    if (midi_) {
        if (midi_->isPortOpen()) midi_->closePort();
        delete midi_;
        midi_ = nullptr;
    }
}

// ── MIDI callback (called from RtMidi's background thread) ───────────────────

void MidiInput::callback(double /*ts*/,
                          std::vector<unsigned char>* msg,
                          void* user_data) {
    if (!msg || msg->size() < 2) return;
    auto* self = reinterpret_cast<MidiInput*>(user_data);
    if (!self->engine_) return;

    uint8_t status  = (*msg)[0];
    uint8_t data1   = (*msg)[1];
    uint8_t data2   = (msg->size() > 2) ? (*msg)[2] : 0;
    uint8_t type    = status & 0xF0;

    uint64_t t = nowMs();
    self->activity_.any_ms.store(t, std::memory_order_relaxed);

    switch (type) {
        case 0x90:  // Note On
            if (data2 > 0) {
                self->activity_.note_on_ms.store(t, std::memory_order_relaxed);
                self->engine_->noteOn(data1, data2);
            } else {
                self->activity_.note_off_ms.store(t, std::memory_order_relaxed);
                self->engine_->noteOff(data1);  // velocity 0 = note off
            }
            break;

        case 0x80:  // Note Off
            self->activity_.note_off_ms.store(t, std::memory_order_relaxed);
            self->engine_->noteOff(data1);
            break;

        case 0xB0:  // Control Change
            switch (data1) {
                case 64:  self->activity_.pedal_ms.store(t, std::memory_order_relaxed);
                          self->engine_->sustainPedal(data2);               break;
                case 7:   self->engine_->setAllVoicesMasterGain(data2);     break;
                case 10:  self->engine_->setAllVoicesPan(data2);            break;
                case 93:  self->engine_->setAllVoicesPanSpeed(data2);       break;
                case 91:  self->engine_->setAllVoicesPanDepth(data2);       break;
                case 74:  self->engine_->setLimiterThreshold(data2);        break;
                default:  break;
            }
            break;

        default: break;
    }
}
