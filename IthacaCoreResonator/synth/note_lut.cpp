/*
 * note_lut.cpp
 * ─────────────
 * Load params.json → NoteLUT[88][8]
 *
 * JSON key format: "m060_vel3"  (zero-padded midi, vel 0..7)
 * Missing keys are left with valid=false; lookupNote() finds nearest neighbour.
 *
 * Null-safe: tau2 can be JSON null → falls back to tau1 * 3.0
 * Requires: third_party/json.hpp (nlohmann/json single-header)
 */

#include "note_lut.h"
#include <fstream>
#include <stdexcept>
#include <cmath>
#include <cstdio>

// nlohmann/json single-header — place at third_party/json.hpp
#include "json.hpp"
using json = nlohmann::json;

// ── Helpers ───────────────────────────────────────────────────────────────────

static float get_f(const json& j, const char* key, float fallback) {
    if (!j.contains(key) || j[key].is_null()) return fallback;
    return j[key].get<float>();
}

static int get_i(const json& j, const char* key, int fallback) {
    if (!j.contains(key) || j[key].is_null()) return fallback;
    return j[key].get<int>();
}

// ── Parse one sample entry ────────────────────────────────────────────────────

static NoteParams parseSample(const json& s) {
    NoteParams p;
    p.valid       = true;
    p.midi        = get_i(s, "midi",        60);
    p.vel         = get_i(s, "vel",         3);
    p.f0_hz       = get_f(s, "f0_fitted_hz", get_f(s, "f0_nominal_hz", 440.f));
    p.B           = get_f(s, "B",           1e-4f);
    p.duration_s  = get_f(s, "duration_s",  8.f);
    p.sr          = get_i(s, "sr",          48000);

    // Determine n_strings from midi (standard piano stringing):
    //   21–27  → 1 string (sub-bass)
    //   28–48  → 2 strings
    //   49–108 → 3 strings
    if (p.midi <= 27)      p.n_strings = 1;
    else if (p.midi <= 48) p.n_strings = 2;
    else                   p.n_strings = 3;

    // ── Partials ─────────────────────────────────────────────────────────────
    if (s.contains("partials") && s["partials"].is_array()) {
        int idx = 0;
        for (const auto& part : s["partials"]) {
            if (idx >= MAX_PARTIALS) break;
            PartialParams& pp = p.partials[idx];
            pp.k          = get_i(part, "k",          idx + 1);
            pp.f_hz       = get_f(part, "f_hz",       p.f0_hz * pp.k);
            pp.A0         = get_f(part, "A0",         0.f);
            pp.tau1       = get_f(part, "tau1",       3.f);
            // tau2 may be null for high partials — fall back to 3×tau1
            float tau2_fb = pp.tau1 * 3.f;
            pp.tau2       = get_f(part, "tau2",       tau2_fb);
            if (pp.tau2 <= 0.f) pp.tau2 = tau2_fb;
            pp.a1         = get_f(part, "a1",         0.6f);
            pp.beat_hz    = get_f(part, "beat_hz",    0.f);
            pp.beat_depth = get_f(part, "beat_depth", 0.f);
            pp.mono       = part.contains("mono") && !part["mono"].is_null()
                            && part["mono"].get<bool>();
            idx++;
        }
        p.n_partials = idx;
    }

    // ── Noise ─────────────────────────────────────────────────────────────────
    if (s.contains("noise") && s["noise"].is_object()) {
        const auto& n = s["noise"];
        p.noise.attack_tau_s         = get_f(n, "attack_tau_s",         0.05f);
        p.noise.floor_rms            = get_f(n, "floor_rms",            0.f);
        p.noise.centroid_hz          = get_f(n, "centroid_hz",          2000.f);
        p.noise.spectral_slope_db_oct= get_f(n, "spectral_slope_db_oct",-12.f);
    }

    // ── Spectral EQ ───────────────────────────────────────────────────────────
    if (s.contains("spectral_eq") && s["spectral_eq"].is_object()) {
        const auto& eq = s["spectral_eq"];
        p.width_factor = get_f(eq, "stereo_width_factor", 0.5f);

        if (eq.contains("freqs_hz") && eq["freqs_hz"].is_array()) {
            int i = 0;
            for (const auto& v : eq["freqs_hz"]) {
                if (i >= EQ_POINTS) break;
                p.eq_freqs_hz[i++] = v.is_null() ? 0.f : v.get<float>();
            }
        }
        if (eq.contains("gains_db") && eq["gains_db"].is_array()) {
            int i = 0;
            for (const auto& v : eq["gains_db"]) {
                if (i >= EQ_POINTS) break;
                p.eq_gains_db[i++] = v.is_null() ? 0.f : v.get<float>();
            }
        }
    }

    return p;
}

// ── Public: loadNoteLUT ───────────────────────────────────────────────────────

NoteLUT loadNoteLUT(const std::string& params_json_path) {
    std::ifstream f(params_json_path);
    if (!f) throw std::runtime_error("Cannot open: " + params_json_path);

    json root;
    f >> root;

    NoteLUT lut{};  // all valid=false by default

    if (!root.contains("samples") || !root["samples"].is_object())
        throw std::runtime_error("params.json: missing 'samples' object");

    int loaded = 0;
    for (const auto& [key, val] : root["samples"].items()) {
        // key format: "m060_vel3"
        int midi = 0, vel = 0;
        if (std::sscanf(key.c_str(), "m%d_vel%d", &midi, &vel) != 2) continue;
        if (midi < MIDI_MIN || midi > MIDI_MAX) continue;
        if (vel < 0 || vel >= VEL_LAYERS) continue;

        lut[midi - MIDI_MIN][vel] = parseSample(val);
        loaded++;
    }

    // Optional: warn about missing notes
    // (caller can use lookupNote() for nearest-neighbour fallback)
    return lut;
}

// ── Public: lookupNote ────────────────────────────────────────────────────────
// Find the nearest valid NoteParams for (midi, vel).
// Search strategy: same vel, expand midi outward; then try adjacent vel layers.

static const NoteParams kEmptyNote{};  // returned only if entire LUT is empty

const NoteParams& lookupNote(const NoteLUT& lut, int midi, int vel) {
    midi = std::max(MIDI_MIN, std::min(MIDI_MAX, midi));
    vel  = std::max(0, std::min(VEL_LAYERS - 1, vel));

    // 1. Exact match
    const NoteParams& exact = lut[midi - MIDI_MIN][vel];
    if (exact.valid) return exact;

    // 2. Expand midi radius, same vel
    for (int r = 1; r < MIDI_COUNT; r++) {
        for (int sign : {-1, +1}) {
            int m = midi + sign * r;
            if (m < MIDI_MIN || m > MIDI_MAX) continue;
            if (lut[m - MIDI_MIN][vel].valid) return lut[m - MIDI_MIN][vel];
        }
    }

    // 3. Try other velocity layers
    for (int dv = 1; dv < VEL_LAYERS; dv++) {
        for (int sign : {-1, +1}) {
            int v = vel + sign * dv;
            if (v < 0 || v >= VEL_LAYERS) continue;
            if (lut[midi - MIDI_MIN][v].valid) return lut[midi - MIDI_MIN][v];
        }
    }

    return kEmptyNote;
}
