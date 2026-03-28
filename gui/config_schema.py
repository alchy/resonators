"""
gui/config_schema.py
────────────────────
Default session config structure with parameter metadata (ranges, docs).
This drives both the API schema and the frontend slider/input rendering.
"""

from typing import Any

# Parameter metadata: {key: {default, min, max, step, unit, doc}}
PARAM_META = {
    # ── Render ────────────────────────────────────────────────────────────────
    "sr": {
        "default": 44100, "min": 22050, "max": 48000, "step": 1,
        "unit": "Hz", "group": "render",
        "doc": "Sample rate. 44100 Hz matches the source sample bank."
    },
    "duration": {
        "default": None, "min": 0.5, "max": 10.0, "step": 0.1,
        "unit": "s", "group": "render",
        "doc": "Note duration. null = use duration from params.json (up to 8s). Override for shorter/longer renders."
    },
    "fade_out": {
        "default": 0.5, "min": 0.0, "max": 5.0, "step": 0.05,
        "unit": "s", "group": "render",
        "doc": "Fade-out duration at note end. Prevents hard cut-off click."
    },
    "target_rms": {
        "default": 0.06, "min": 0.01, "max": 0.25, "step": 0.005,
        "unit": "", "group": "render",
        "doc": "Output RMS level for velocity 7 (loudest layer). Lower velocities are scaled down by the velocity curve. 0.06 ≈ -24 dBFS."
    },
    "velocity_curve_gamma": {
        "default": 0.7, "min": 0.2, "max": 2.0, "step": 0.05,
        "unit": "", "group": "render",
        "doc": "Velocity curve exponent: rms = target_rms × ((vel+1)/8)^gamma. "
               "gamma=1.0: linear (vel 0 = 12.5% of vel 7). "
               "gamma=0.5: square-root (vel 0 = 35% of vel 7, more natural). "
               "gamma=0.7: default, good piano feel."
    },

    # ── Timbre ────────────────────────────────────────────────────────────────
    "harmonic_brightness": {
        "default": 1.0, "min": 0.0, "max": 3.0, "step": 0.05,
        "unit": "", "group": "timbre",
        "doc": "Boosts upper harmonics: gain(k) = 1 + brightness × log₂(k). "
               "0 = flat extraction, 1 = moderate, 2 = strong. "
               "Compensates for extraction underestimating high-k attack energy. "
               "Best value found empirically: 1.0."
    },
    "beat_scale": {
        "default": 1.0, "min": 0.0, "max": 3.0, "step": 0.1,
        "unit": "×", "group": "timbre",
        "doc": "Scales inter-string beating frequency. 0 = no beating (static). "
               "1 = extracted value. 2 = twice as fast beating. "
               "Affects 'liveliness' and amplitude modulation depth."
    },
    "eq_strength": {
        "default": 0.5, "min": 0.0, "max": 1.0, "step": 0.05,
        "unit": "", "group": "timbre",
        "doc": "Spectral EQ strength (LTASE_orig / LTASE_synth ratio). "
               "Captures body resonance: cuts harsh highs, shapes spectral envelope. "
               "Works together with eq_freq_min — below that frequency EQ is flat. "
               "0 = bypass. 0.5 = recommended starting point. 1.0 = full correction."
    },
    "eq_freq_min": {
        "default": 400.0, "min": 50.0, "max": 2000.0, "step": 50.0,
        "unit": "Hz", "group": "timbre",
        "doc": "EQ lower cutoff frequency. Below this the EQ fades to flat (0 dB). "
               "Prevents room acoustics contamination from distorting the fundamental. "
               "400 Hz = safe default (protects k=1 and k=2 of most notes). "
               "Lower → more EQ correction, but risks cutting the fundamental."
    },
    "soundboard_strength": {
        "default": 0.0, "min": 0.0, "max": 1.0, "step": 0.05,
        "unit": "", "group": "timbre",
        "doc": "Modal IR convolution for soundboard body resonance. "
               "PARKED at 0: current 40-mode IR causes band-pass distortion "
               "('croaking'). Will be redesigned from measured IR."
    },

    # ── Velocity color ────────────────────────────────────────────────────────
    "vel_color_blend": {
        "default": 0.7, "min": 0.0, "max": 1.0, "step": 0.05,
        "unit": "", "group": "timbre",
        "doc": "Blend spectral color (A0 amplitude ratios) toward a reference velocity. "
               "0 = per-velocity shapes unchanged. 1 = all velocities get ref velocity's spectral shape. "
               "Preserves total energy per velocity. Recommended: 0.3–0.5."
    },
    "vel_color_ref": {
        "default": 4, "min": 0, "max": 7, "step": 1,
        "unit": "vel", "group": "timbre",
        "doc": "Reference velocity for color blending (0–7). vel4 is the sweet spot: "
               "strong enough signal for reliable extraction, not so loud that the hammer "
               "contact time shifts spectral balance."
    },

    # ── Stereo ────────────────────────────────────────────────────────────────
    "pan_spread": {
        "default": 0.55, "min": 0.0, "max": 1.5, "step": 0.05,
        "unit": "rad", "group": "stereo",
        "doc": "Per-string pan angle spread. 0 = mono center. "
               "0.55 = natural spread. Bass notes pan left, treble right (±0.20 rad global offset). "
               "Multi-string notes get one pan position per string → natural stereo width."
    },
    "stereo_boost": {
        "default": 1.0, "min": 0.5, "max": 4.0, "step": 0.1,
        "unit": "×", "group": "stereo",
        "doc": "M/S stereo width multiplier applied ON TOP of the per-note "
               "sample-derived width_factor. 1.0 = use derived factor only. "
               "2.0 = double the side channel. Effective side gain = width_factor × stereo_boost."
    },
}

# Per-note delta parameters (additive offsets on top of global)
PER_NOTE_DELTA_META = {
    "harmonic_brightness_delta": {
        "min": -2.0, "max": 2.0, "step": 0.05, "default": 0.0,
        "doc": "Per-note offset added to global harmonic_brightness. "
               "E.g. +0.5 makes this note brighter than the global setting."
    },
    "beat_scale_delta": {
        "min": -2.0, "max": 2.0, "step": 0.1, "default": 0.0,
        "doc": "Per-note offset added to global beat_scale."
    },
    "pan_spread_delta": {
        "min": -1.0, "max": 1.0, "step": 0.05, "default": 0.0,
        "doc": "Per-note offset added to global pan_spread."
    },
    "tau1_k1_scale": {
        "min": 0.1, "max": 5.0, "step": 0.1, "default": 1.0,
        "doc": "Multiplier for the k=1 (fundamental) decay time tau1. "
               "A2 k=1 has extracted tau=0.29s (too short). Try 2.0 → 0.58s for a fuller fundamental."
    },
}

NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]


def midi_to_name(midi: int) -> str:
    octave = (midi // 12) - 1
    name = NOTE_NAMES[midi % 12]
    return f"{name}{octave}"


def default_velocity_profile(gamma: float = 0.7) -> dict:
    """Default velocity RMS ratios derived from gamma power curve."""
    return {str(v): round(((v + 1) / 8.0) ** gamma, 4) for v in range(8)}


def default_config(source_params: str = "analysis/params.json") -> dict:
    """Return a fresh session config with all defaults."""
    cfg = {
        "source_params": source_params,
        "render": {},
        "timbre": {},
        "stereo": {},
        "per_note": {},
        "velocity_rms_profile": default_velocity_profile(),
    }
    for key, meta in PARAM_META.items():
        cfg[meta["group"]][key] = meta["default"]
    return cfg


def resolve_note_params(config: dict, midi: int) -> dict:
    """
    Merge global config with per-note deltas for a specific MIDI note.
    Returns a flat dict of synthesize_note() kwargs.
    """
    # Flatten global params
    flat = {}
    for section in ("render", "timbre", "stereo"):
        flat.update(config.get(section, {}))

    # Apply per-note deltas
    note_cfg = config.get("per_note", {}).get(str(midi), {})
    for key, val in note_cfg.items():
        if key.endswith("_delta"):
            base = key[:-6]
            if base in flat and flat[base] is not None:
                flat[base] = flat[base] + val
        elif key == "tau1_k1_scale":
            flat["_tau1_k1_scale"] = val  # handled in synthesis wrapper
        # future: other per-note transforms here

    return flat
