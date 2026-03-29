/*
 * voice_manager.cpp
 * ─────────────────
 * Polyphonic voice pool. One ResonatorVoice slot per MIDI note (21..108).
 * MIDI API mirrors IthacaCore VoiceManager for drop-in compatibility.
 */

#include "voice_manager.h"
#include "note_lut.h"
#include <iostream>
#include <algorithm>
#include <cstring>

// ── Constructor / Destructor ──────────────────────────────────────────────────

ResonatorVoiceManager::ResonatorVoiceManager()  = default;
ResonatorVoiceManager::~ResonatorVoiceManager() = default;

// ── initialize ────────────────────────────────────────────────────────────────

void ResonatorVoiceManager::initialize(const std::string& params_json_path,
                                        float sample_rate,
                                        Logger& logger) {
    sample_rate_ = sample_rate;
    logger_      = &logger;

    try {
        lut_ = loadNoteLUT(params_json_path);
    } catch (const std::exception& e) {
        logger.log("VoiceManager", 0, std::string("Failed to load params: ") + e.what());
        return;
    }

    // Count loaded notes
    int valid = 0;
    for (int m = 0; m < MIDI_COUNT; m++)
        for (int v = 0; v < VEL_LAYERS; v++)
            if (lut_[m][v].valid) valid++;

    logger.log("VoiceManager", 0,
        "Loaded " + std::to_string(valid) + " note entries from " + params_json_path);

    initialized_ = true;
}

// ── MIDI note-on / note-off ───────────────────────────────────────────────────

void ResonatorVoiceManager::setNoteStateMIDI(uint8_t midi_note,
                                              bool note_on,
                                              uint8_t velocity) {
    if (!initialized_) return;
    if (midi_note < MIDI_MIN || midi_note > MIDI_MAX) return;

    if (note_on) {
        handleNoteOn(midi_note, velocity);
    } else {
        if (sustain_pedal_.load(std::memory_order_relaxed)) {
            // Defer note-off until pedal release
            held_notes_.push_back(midi_note);
        } else {
            handleNoteOff(midi_note);
        }
    }
}

void ResonatorVoiceManager::setSustainPedalMIDI(uint8_t val) {
    bool down = (val >= 64);
    bool was_down = sustain_pedal_.exchange(down, std::memory_order_relaxed);

    if (was_down && !down) {
        // Pedal released — send deferred note-offs
        for (uint8_t midi : held_notes_)
            handleNoteOff(midi);
        held_notes_.clear();
    }
}

void ResonatorVoiceManager::handleNoteOn(uint8_t midi, uint8_t velocity) {
    int vel_layer = std::min(7, (int)velocity * VEL_LAYERS / 128);
    const NoteParams& p = lookupNote(lut_, midi, vel_layer);
    if (!p.valid) return;

    int idx = midi - MIDI_MIN;
    voices_[idx].noteOn(midi, vel_layer, p, sample_rate_);
}

void ResonatorVoiceManager::handleNoteOff(uint8_t midi) {
    int idx = midi - MIDI_MIN;
    if (idx >= 0 && idx < MIDI_COUNT)
        voices_[idx].noteOff();
}

// ── processBlockUninterleaved ─────────────────────────────────────────────────

void ResonatorVoiceManager::processBlockUninterleaved(float* out_l,
                                                       float* out_r,
                                                       int n_samples) {
    // Zero output buffers
    std::memset(out_l, 0, sizeof(float) * n_samples);
    std::memset(out_r, 0, sizeof(float) * n_samples);

    // Sum all active voices
    for (int i = 0; i < MIDI_COUNT; i++) {
        if (voices_[i].isActive())
            voices_[i].processBlock(out_l, out_r, n_samples);
    }

    // TODO: apply master DSP (limiter, BBE) once dsp/ files are copied
}

// ── Diagnostics ───────────────────────────────────────────────────────────────

int ResonatorVoiceManager::activeVoiceCount() const {
    int count = 0;
    for (int i = 0; i < MIDI_COUNT; i++)
        if (voices_[i].isActive()) count++;
    return count;
}

// ── Master DSP controls (stubs — wired when dsp/ files are present) ──────────

void ResonatorVoiceManager::setLimiterThresholdMIDI(uint8_t) {}
void ResonatorVoiceManager::setLimiterReleaseMIDI(uint8_t)   {}
void ResonatorVoiceManager::setLimiterEnabledMIDI(uint8_t)   {}
void ResonatorVoiceManager::setBBEDefinitionMIDI(uint8_t)    {}
void ResonatorVoiceManager::setBBEBassBoostMIDI(uint8_t)     {}
