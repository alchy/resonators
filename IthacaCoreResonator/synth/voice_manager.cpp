/*
 * voice_manager.cpp
 * ─────────────────
 * API-compatible implementation of IthacaCore VoiceManager for physics synth.
 */

#include "voice_manager.h"
#include "note_lut.h"
#include <cstring>
#include <cmath>
#include <algorithm>
#include <string>

static constexpr float PI  = 3.14159265358979f;
static constexpr float TAU = 2.f * PI;

// ── Constructor / Destructor ──────────────────────────────────────────────────

ResonatorVoiceManager::ResonatorVoiceManager()  = default;
ResonatorVoiceManager::~ResonatorVoiceManager() = default;

// ── Initialization ────────────────────────────────────────────────────────────

void ResonatorVoiceManager::initialize(const std::string& params_json_path,
                                        float sample_rate,
                                        Logger& logger) {
    sample_rate_ = sample_rate;
    logger_      = &logger;

    logger.log("VoiceManager", LogSeverity::Info, "Loading: " + params_json_path);
    try {
        loadNoteLUT(params_json_path, lut_);
    } catch (const std::exception& e) {
        logger.log("VoiceManager", LogSeverity::Error,
            std::string("Failed to load params: ") + e.what());
        return;
    }

    int valid = 0;
    for (int m = 0; m < MIDI_COUNT; m++)
        for (int v = 0; v < VEL_LAYERS; v++)
            if (lut_[m][v].valid) valid++;

    logger.log("VoiceManager", LogSeverity::Info,
        std::to_string(valid) + " note entries loaded. SR=" +
        std::to_string((int)sample_rate_));

    prepareToPlay(512);
    initialized_ = true;
}

void ResonatorVoiceManager::changeSampleRate(float new_sr, Logger& logger) {
    sample_rate_ = new_sr;
    logger.log("VoiceManager", LogSeverity::Info,
        "Sample rate changed to " + std::to_string((int)new_sr));
    // Decay coefficients are recomputed at next noteOn; active voices continue
    // at old rate until their natural end (safe trade-off for live SR change).
}

void ResonatorVoiceManager::prepareToPlay(int max_block_size) {
    max_block_size_ = max_block_size;
    tmp_l_.assign(max_block_size, 0.f);
    tmp_r_.assign(max_block_size, 0.f);
    dsp_chain_.prepare(sample_rate_, max_block_size);

    // Peak meter decay: -20 dB/s.  coeff = 10^(-1 / blocks_per_sec)
    float blocks_per_sec = sample_rate_ / (float)max_block_size;
    peak_decay_coeff_ = std::pow(10.f, -1.f / blocks_per_sec);
}

// ── MIDI note control ─────────────────────────────────────────────────────────

void ResonatorVoiceManager::setNoteStateMIDI(uint8_t midi, bool on,
                                              uint8_t vel) noexcept {
    if (midi < MIDI_MIN || midi > MIDI_MAX) return;
    if (on) {
        handleNoteOn(midi, vel);
    } else {
        if (sustain_pedal_.load(std::memory_order_relaxed))
            delayed_note_offs_[midi] = true;
        else
            handleNoteOff(midi);
    }
}

void ResonatorVoiceManager::setNoteStateMIDI(uint8_t midi, bool on) noexcept {
    setNoteStateMIDI(midi, on, ITHACA_DEFAULT_VELOCITY);
}

void ResonatorVoiceManager::setSustainPedalMIDI(uint8_t val) noexcept {
    setSustainPedalMIDI(val >= 64);
}

void ResonatorVoiceManager::setSustainPedalMIDI(bool down) noexcept {
    bool was = sustain_pedal_.exchange(down, std::memory_order_acq_rel);
    if (was && !down) processDelayedNoteOffs();
}

void ResonatorVoiceManager::handleNoteOn(uint8_t midi, uint8_t vel) noexcept {
    // Map MIDI 0–127 to float position 0.0–7.0 in velocity layer space.
    // interpolateNoteLayers blends adjacent layers for smooth velocity response.
    float vel_pos = (float)vel * (VEL_LAYERS - 1.f) / 127.f;
    NoteParams p  = interpolateNoteLayers(lut_, (int)midi, vel_pos);
    if (!p.valid) return;

    // Pass raw MIDI velocity so voice applies vel_gamma amplitude curve.
    voices_[midi - MIDI_MIN].noteOn((int)midi, (int)vel, p, sample_rate_, synth_cfg_);
    last_note_seed_.store(voices_[midi - MIDI_MIN].getLastSeed(), std::memory_order_relaxed);
}

void ResonatorVoiceManager::handleNoteOff(uint8_t midi) noexcept {
    int idx = (int)midi - MIDI_MIN;
    if (idx >= 0 && idx < MIDI_COUNT)
        voices_[idx].noteOff();
}

void ResonatorVoiceManager::processDelayedNoteOffs() noexcept {
    for (int m = MIDI_MIN; m <= MIDI_MAX; m++) {
        if (delayed_note_offs_[m]) {
            delayed_note_offs_[m] = false;
            handleNoteOff((uint8_t)m);
        }
    }
}

// ── Audio rendering ───────────────────────────────────────────────────────────

bool ResonatorVoiceManager::processBlockUninterleaved(float* out_l,
                                                       float* out_r,
                                                       int n) noexcept {
    std::memset(out_l, 0, sizeof(float) * n);
    std::memset(out_r, 0, sizeof(float) * n);
    bool any = processBlockSegment(out_l, out_r, n);
    finalizeBlock(out_l, out_r, n);
    return any;
}

bool ResonatorVoiceManager::processBlockSegment(float* out_l, float* out_r,
                                                  int n) noexcept {
    bool any = false;
    for (int i = 0; i < MIDI_COUNT; i++) {
        if (voices_[i].isActive()) {
            voices_[i].processBlock(out_l, out_r, n);
            any = true;
        }
    }
    // Apply master gain and pan
    if (master_gain_ != 1.f || pan_l_ != 1.f || pan_r_ != 1.f) {
        for (int s = 0; s < n; s++) {
            out_l[s] *= master_gain_ * pan_l_;
            out_r[s] *= master_gain_ * pan_r_;
        }
    }
    return any;
}

void ResonatorVoiceManager::finalizeBlock(float* out_l, float* out_r,
                                           int n) noexcept {
    applyLfoPanToFinalMix(out_l, out_r, n);
    dsp_chain_.process(out_l, out_r, n);

    // Peak metering — after full DSP chain, immediate attack, -20 dB/s decay
    float block_peak = 0.f;
    for (int i = 0; i < n; i++) {
        float s = std::abs(out_l[i]);
        if (s > block_peak) block_peak = s;
        s = std::abs(out_r[i]);
        if (s > block_peak) block_peak = s;
    }
    float prev   = output_peak_lin_.load(std::memory_order_relaxed);
    float decayed = prev * peak_decay_coeff_;
    output_peak_lin_.store(block_peak > decayed ? block_peak : decayed,
                            std::memory_order_relaxed);
}

bool ResonatorVoiceManager::processBlockInterleaved(float* out, int n) noexcept {
    // Ensure temp buffers are large enough
    if ((int)tmp_l_.size() < n) {
        tmp_l_.resize(n, 0.f);
        tmp_r_.resize(n, 0.f);
    }
    std::memset(tmp_l_.data(), 0, sizeof(float) * n);
    std::memset(tmp_r_.data(), 0, sizeof(float) * n);

    bool any = processBlockSegment(tmp_l_.data(), tmp_r_.data(), n);
    finalizeBlock(tmp_l_.data(), tmp_r_.data(), n);

    // Interleave L R L R ...
    for (int i = 0; i < n; i++) {
        out[i * 2]     = tmp_l_[i];
        out[i * 2 + 1] = tmp_r_[i];
    }
    return any;
}

void ResonatorVoiceManager::applyLfoPanToFinalMix(float* out_l, float* out_r,
                                                    int n) noexcept {
    if (lfo_depth_ < 1e-4f || lfo_speed_ < 1e-4f) return;

    float phase_inc = TAU * lfo_speed_ / sample_rate_;
    for (int s = 0; s < n; s++) {
        float lfo      = lfo_depth_ * std::sin(lfo_phase_);
        lfo_phase_    += phase_inc;
        if (lfo_phase_ > TAU) lfo_phase_ -= TAU;

        // Constant-power pan: centre ±lfo
        float angle    = (PI / 4.f) * (1.f + lfo);
        float gl       = std::cos(angle);
        float gr       = std::sin(angle);
        out_l[s]      *= gl;
        out_r[s]      *= gr;
    }
}

// ── Voice control ─────────────────────────────────────────────────────────────

void ResonatorVoiceManager::stopAllVoices() noexcept {
    for (int i = 0; i < MIDI_COUNT; i++)
        if (voices_[i].isActive())
            voices_[i].noteOff();
}

void ResonatorVoiceManager::resetAllVoices(Logger& logger) {
    stopAllVoices();
    delayed_note_offs_.fill(false);
    sustain_pedal_.store(false);
    lfo_phase_ = 0.f;
    logger.log("VoiceManager", LogSeverity::Info, "All voices reset");
}

// ── Global voice parameters ───────────────────────────────────────────────────

void ResonatorVoiceManager::setAllVoicesMasterGainMIDI(uint8_t val,
                                                         Logger& logger) {
    master_gain_ = (float)val / 127.f;
    logger.log("VoiceManager", LogSeverity::Debug,
        "Master gain: " + std::to_string(master_gain_));
}

void ResonatorVoiceManager::setAllVoicesPanMIDI(uint8_t val) noexcept {
    // 64 = centre, 0 = hard left, 127 = hard right
    float pan    = ((float)val - 64.f) / 64.f;   // -1..+1
    float angle  = (PI / 4.f) * (1.f + pan);
    pan_l_       = std::cos(angle);
    pan_r_       = std::sin(angle);
}

void ResonatorVoiceManager::setAllVoicesStereoFieldAmountMIDI(uint8_t val) noexcept {
    stereo_field_ = (float)val / 127.f;
    // Applied to width_factor at next noteOn via lookupNote result
    // (stored as a scale factor; voices started after this call use new width)
}

void ResonatorVoiceManager::setAllVoicesPanSpeedMIDI(uint8_t val) noexcept {
    lfo_speed_ = (float)val / 127.f * 2.f;   // 0..2 Hz
}

void ResonatorVoiceManager::setAllVoicesPanDepthMIDI(uint8_t val) noexcept {
    lfo_depth_ = (float)val / 127.f;
}

bool ResonatorVoiceManager::isLfoPanningActive() const noexcept {
    return lfo_speed_ > 1e-4f && lfo_depth_ > 1e-4f;
}

// ── Statistics ────────────────────────────────────────────────────────────────

int ResonatorVoiceManager::getActiveVoicesCount() const noexcept {
    int n = 0;
    for (int i = 0; i < MIDI_COUNT; i++)
        if (voices_[i].isActive()) n++;
    return n;
}

int ResonatorVoiceManager::getSustainingVoicesCount() const noexcept {
    int n = 0;
    for (int i = 0; i < MIDI_COUNT; i++)
        if (voices_[i].isActive() && !voices_[i].isReleasing()) n++;
    return n;
}

int ResonatorVoiceManager::getReleasingVoicesCount() const noexcept {
    int n = 0;
    for (int i = 0; i < MIDI_COUNT; i++)
        if (voices_[i].isReleasing()) n++;
    return n;
}

void ResonatorVoiceManager::logSystemStatistics(Logger& logger) {
    logger.log("VoiceManager", LogSeverity::Info,
        "SR=" + std::to_string((int)sample_rate_) +
        " active=" + std::to_string(getActiveVoicesCount()) +
        " sustaining=" + std::to_string(getSustainingVoicesCount()) +
        " releasing=" + std::to_string(getReleasingVoicesCount()) +
        " lfo=" + (isLfoPanningActive() ? "ON" : "off") +
        " pedal=" + (sustain_pedal_.load() ? "DOWN" : "up") +
        " gain=" + std::to_string(master_gain_));
}

// ── DSP effects ───────────────────────────────────────────────────────────────

void ResonatorVoiceManager::setLimiterThresholdMIDI(uint8_t v) noexcept {
    limiter_threshold_midi_ = v;
    dsp_chain_.setLimiterThreshold(v);
}
void ResonatorVoiceManager::setLimiterReleaseMIDI(uint8_t v) noexcept {
    limiter_release_midi_ = v;
    dsp_chain_.setLimiterRelease(v);
}
void ResonatorVoiceManager::setLimiterEnabledMIDI(uint8_t v) noexcept {
    limiter_enabled_midi_ = v;
    dsp_chain_.setLimiterEnabled(v);
}
uint8_t ResonatorVoiceManager::getLimiterThresholdMIDI()     const noexcept { return limiter_threshold_midi_; }
uint8_t ResonatorVoiceManager::getLimiterReleaseMIDI()       const noexcept { return limiter_release_midi_;   }
uint8_t ResonatorVoiceManager::getLimiterEnabledMIDI()       const noexcept { return limiter_enabled_midi_;   }
uint8_t ResonatorVoiceManager::getLimiterGainReductionMIDI() const noexcept {
    return dsp_chain_.getLimiterGainReduction();
}

void ResonatorVoiceManager::setBBEDefinitionMIDI(uint8_t v) noexcept {
    bbe_definition_midi_ = v;
    dsp_chain_.setBBEDefinition(v);
}
void ResonatorVoiceManager::setBBEBassBoostMIDI(uint8_t v) noexcept {
    bbe_bass_boost_midi_ = v;
    dsp_chain_.setBBEBassBoost(v);
}
