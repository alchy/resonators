#pragma once
#include "note_params.h"
#include <string>

// Load params.json (from extract_params / SetterNN pipeline) into a NoteLUT.
//
// Key format in params.json:  "m060_vel3"  (midi=60, vel=3)
// Velocity layer:             vel 0..7  → VEL_LAYERS index
// MIDI note:                  21..108   → MIDI_COUNT index
//
// Missing notes (no WAV source) are left with valid=false.
// Callers should fall back to the nearest valid neighbour.

// Fill lut in-place (avoids a 2.7 MB stack-local temporary).
void loadNoteLUT(const std::string& params_json_path, NoteLUT& lut);

// Return the nearest valid NoteParams for (midi, vel).
// Searches outward in midi, then vel until a valid entry is found.
const NoteParams& lookupNote(const NoteLUT& lut, int midi, int vel);

// Interpolate between adjacent velocity layers for a given float position.
// vel_pos: 0.0 = layer 0 (softest) .. 7.0 = layer 7 (loudest).
// Linearly blends A0, tau1/tau2/a1/beat_hz/f_hz, noise, EQ gains.
// Returns a blended copy — valid for immediate use within the calling frame.
NoteParams interpolateNoteLayers(const NoteLUT& lut, int midi, float vel_pos);
