#pragma once
/*
 * synth_config.h
 * ────────────────
 * Per-session synthesis rendering parameters.
 * Mirrors physics_synth.py  synthesize_note()  keyword arguments exactly.
 *
 * Stored in ResonatorVoiceManager and forwarded to each ResonatorVoice at
 * noteOn time.  Changing a field affects notes started after the change.
 */

struct SynthConfig {
    // ── Stereo geometry ───────────────────────────────────────────────────────
    float pan_spread          = 0.55f;   // string spread in rad  (half = pan_spread/2)
    float stereo_decorr       = 1.0f;    // Schroeder all-pass blend multiplier
    float stereo_boost        = 1.0f;    // M/S side-channel boost on top of width_factor

    // ── Timbre ────────────────────────────────────────────────────────────────
    float beat_scale          = 1.0f;    // beat_hz multiplier (1.0=extracted, 1.5-2.5=vivid)
    float harmonic_brightness = 0.0f;    // upper-partial boost: gain = 1 + hb*log2(k)

    // ── Spectral EQ ───────────────────────────────────────────────────────────
    float eq_strength         = 1.0f;    // EQ blend (0=bypass, 1=full)
    float eq_freq_min         = 400.0f;  // EQ flat below this Hz (room-acoustics guard)

    // ── Noise ─────────────────────────────────────────────────────────────────
    float noise_level         = 1.0f;    // noise amplitude multiplier

    // ── Attack ────────────────────────────────────────────────────────────────
    float onset_ms            = 3.0f;    // linear ramp length to prevent click

    // ── Level ─────────────────────────────────────────────────────────────────
    // Target RMS for each synthesized note (matches Python target_rms=0.06).
    // Implemented via A0_ref normalization + instantaneous RMS estimate at t=0.
    float target_rms  = 0.06f;   // −24.4 dBFS RMS; Python default

    // ── Velocity ──────────────────────────────────────────────────────────────
    // vel_gain = (midi_vel / 127)^vel_gamma  applied to all partial amplitudes.
    // 0.7 matches Python synthesize_preview_set(vel_gamma=0.7).
    float vel_gamma           = 0.7f;
};
