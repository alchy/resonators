/*
 * resonator_gui.cpp
 * ──────────────────
 * Dear ImGui + GLFW + OpenGL3 GUI for IthacaCoreResonator.
 *
 * Layout:
 *   Top bar   — MIDI port selector, connect button, active-voice + sustain badge
 *   Center    — Clickable piano keyboard (C2..C7, 5 octaves), active notes lit
 *   Bottom    — Master gain, pan, LFO speed/depth sliders
 */

#include "resonator_gui.h"
#include "../synth/midi_input.h"
#include "../synth/note_params.h"
#include "../dsp/dsp_chain.h"

#include "imgui.h"
#include "imgui_impl_glfw.h"
#include "imgui_impl_opengl3.h"

#include <GLFW/glfw3.h>

#include <string>
#include <vector>
#include <cmath>
#include <cstring>
#include <algorithm>
#include <chrono>

static uint64_t guiNowMs() {
    return (uint64_t)std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
}

// ── Piano key layout constants ────────────────────────────────────────────────
static constexpr int   PIANO_MIDI_LOW  = 36;   // C2
static constexpr int   PIANO_MIDI_HIGH = 96;   // C7
static constexpr float WHITE_W  = 22.f;
static constexpr float WHITE_H  = 90.f;
static constexpr float BLACK_W  = 14.f;
static constexpr float BLACK_H  = 56.f;

// Is this MIDI note a black key?
static bool isBlack(int midi) {
    int n = midi % 12;
    return n == 1 || n == 3 || n == 6 || n == 8 || n == 10;
}

// Count white keys below midi (from PIANO_MIDI_LOW)
static int whitesBefore(int midi) {
    int count = 0;
    for (int m = PIANO_MIDI_LOW; m < midi; m++)
        if (!isBlack(m)) count++;
    return count;
}

// ── Note name helper ──────────────────────────────────────────────────────────
static const char* noteName(int midi) {
    static const char* names[] = {"C","C#","D","D#","E","F","F#","G","G#","A","A#","B"};
    return names[midi % 12];
}

// ── GUI state ─────────────────────────────────────────────────────────────────
struct GuiState {
    // MIDI port management
    std::vector<std::string> ports;
    int  selected_port   = 0;
    bool midi_connected  = false;

    // Active notes (for keyboard highlight)
    bool active_notes[128] = {};

    // Piano mouse interaction
    int  mouse_held_note = -1;   // currently held via mouse click

    // Controller values (0..127)
    uint8_t master_gain  = 100;
    uint8_t pan          = 64;   // 64 = centre
    uint8_t lfo_speed    = 0;
    uint8_t lfo_depth    = 0;

    // Limiter
    uint8_t limiter_thr     = 100;  // MIDI 100 ≈ -8 dB
    uint8_t limiter_rel     = 50;
    bool    limiter_enabled = false;

    // BBE
    uint8_t bbe_def     = 0;
    uint8_t bbe_bass    = 0;
    bool    bbe_enabled = false;

    // Stats (polled each frame)
    int     active_voices = 0;
    bool    sustain_on    = false;

    // UI style
    bool    dark_mode     = true;
};

// ── GLFW error callback ───────────────────────────────────────────────────────
static void glfwErrorCb(int err, const char* desc) {
    (void)err;
    fprintf(stderr, "[GLFW] %s\n", desc);
}

// ── Draw piano keyboard, return MIDI note under mouse (-1 if none) ────────────
// Also takes active_notes[] to colour pressed keys.
static int drawPiano(GuiState& gs, ResonatorEngine& engine) {
    ImDrawList* dl = ImGui::GetWindowDrawList();
    ImVec2 origin  = ImGui::GetCursorScreenPos();

    // Total width = number of white keys × WHITE_W
    int total_white = 0;
    for (int m = PIANO_MIDI_LOW; m <= PIANO_MIDI_HIGH; m++)
        if (!isBlack(m)) total_white++;
    float total_w = total_white * WHITE_W;
    float total_h = WHITE_H + 4.f;

    // Reserve space in ImGui layout
    ImGui::Dummy(ImVec2(total_w, total_h));

    bool  lmb_down  = ImGui::IsMouseDown(ImGuiMouseButton_Left);
    ImVec2 mp       = ImGui::GetMousePos();

    int hit_note = -1;  // note under mouse this frame

    // ── Pass 1: white keys ────────────────────────────────────────────────────
    for (int midi = PIANO_MIDI_LOW; midi <= PIANO_MIDI_HIGH; midi++) {
        if (isBlack(midi)) continue;
        int    wi  = whitesBefore(midi);
        float  x   = origin.x + wi * WHITE_W;
        float  y   = origin.y;
        ImVec2 tl  = {x + 1.f, y};
        ImVec2 br  = {x + WHITE_W - 1.f, y + WHITE_H};

        // Check mouse hit (white keys only if not covered by black)
        bool hit = lmb_down
            && mp.x >= tl.x && mp.x <= br.x
            && mp.y >= tl.y && mp.y <= br.y;
        if (hit) hit_note = midi;

        ImU32 col;
        if (gs.active_notes[midi])         col = IM_COL32(120, 160, 255, 255);
        else if (hit)                      col = IM_COL32(200, 220, 255, 255);
        else                               col = IM_COL32(240, 240, 240, 255);

        dl->AddRectFilled(tl, br, col, 2.f);
        dl->AddRect(tl, br, IM_COL32(80, 80, 80, 200), 2.f);

        // Label C notes
        if (midi % 12 == 0) {
            char buf[8];
            snprintf(buf, sizeof(buf), "C%d", midi / 12 - 1);
            dl->AddText({tl.x + 2.f, br.y - 14.f}, IM_COL32(60,60,60,200), buf);
        }
    }

    // ── Pass 2: black keys (drawn on top) ─────────────────────────────────────
    for (int midi = PIANO_MIDI_LOW; midi <= PIANO_MIDI_HIGH; midi++) {
        if (!isBlack(midi)) continue;
        // Position: to the right of the previous white key
        int prev_white = midi - 1;
        while (isBlack(prev_white)) prev_white--;
        int    wi  = whitesBefore(prev_white);
        float  x   = origin.x + wi * WHITE_W + WHITE_W - BLACK_W * 0.5f;
        float  y   = origin.y;
        ImVec2 tl  = {x, y};
        ImVec2 br  = {x + BLACK_W, y + BLACK_H};

        bool hit = lmb_down
            && mp.x >= tl.x && mp.x <= br.x
            && mp.y >= tl.y && mp.y <= br.y;
        if (hit) hit_note = midi;  // black key takes priority

        ImU32 col;
        if (gs.active_notes[midi])     col = IM_COL32(80, 120, 255, 255);
        else if (hit)                  col = IM_COL32(60, 80, 140, 255);
        else                           col = IM_COL32(30, 30, 30, 255);

        dl->AddRectFilled(tl, br, col, 2.f);
        dl->AddRect(tl, br, IM_COL32(0,0,0,255), 2.f);
    }

    // ── Mouse note-on / note-off logic ────────────────────────────────────────
    if (lmb_down && hit_note >= 0) {
        if (gs.mouse_held_note != hit_note) {
            if (gs.mouse_held_note >= 0)
                engine.noteOff((uint8_t)gs.mouse_held_note);
            engine.noteOn((uint8_t)hit_note, gs.master_gain);
            gs.mouse_held_note = hit_note;
        }
    } else {
        if (gs.mouse_held_note >= 0) {
            engine.noteOff((uint8_t)gs.mouse_held_note);
            gs.mouse_held_note = -1;
        }
    }

    return hit_note;
}

// ── Main GUI loop ─────────────────────────────────────────────────────────────

int runResonatorGui(ResonatorEngine& engine, Logger& logger,
                    const std::string& /*params_json_path*/) {
    logger.log("GUI", LogSeverity::Info, "Starting GLFW + ImGui");

    glfwSetErrorCallback(glfwErrorCb);
    if (!glfwInit()) {
        logger.log("GUI", LogSeverity::Error, "glfwInit failed");
        return 1;
    }

    // OpenGL 3.3 core
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
#ifdef __APPLE__
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
#endif

    GLFWwindow* win = glfwCreateWindow(1350, 680,
        "IthacaCoreResonator — Physics Piano", nullptr, nullptr);
    if (!win) {
        logger.log("GUI", LogSeverity::Error, "glfwCreateWindow failed");
        glfwTerminate();
        return 1;
    }
    glfwMakeContextCurrent(win);
    glfwSwapInterval(1);  // vsync

    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO& io = ImGui::GetIO();
    io.IniFilename = nullptr;  // don't save layout to file

    ImGui::StyleColorsDark();
    ImGui::GetStyle().WindowRounding   = 4.f;
    ImGui::GetStyle().FrameRounding    = 3.f;
    ImGui::GetStyle().GrabRounding     = 3.f;

    ImGui_ImplGlfw_InitForOpenGL(win, true);
    ImGui_ImplOpenGL3_Init("#version 330");

    GuiState gs;
    gs.ports = MidiInput::listPorts();
    MidiInput midi_in;

    // Auto-connect to first port
    if (!gs.ports.empty()) {
        midi_in.open(engine, 0);
        gs.midi_connected = midi_in.isOpen();
        logger.log("GUI", LogSeverity::Info,
            gs.midi_connected ? "Auto-connected: " + gs.ports[0] : "MIDI open failed");
    }

    logger.log("GUI", LogSeverity::Info, "GUI loop started");

    while (!glfwWindowShouldClose(win)) {
        glfwPollEvents();

        // ── Update stats ──────────────────────────────────────────────────────
        gs.active_voices = engine.activeVoices();
        gs.sustain_on    = false;  // TODO: expose from engine

        // ── ImGui frame ───────────────────────────────────────────────────────
        ImGui_ImplOpenGL3_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();

        int fb_w, fb_h;
        glfwGetFramebufferSize(win, &fb_w, &fb_h);
        ImGui::SetNextWindowPos({0, 0});
        ImGui::SetNextWindowSize({(float)fb_w, (float)fb_h});
        ImGui::Begin("##main", nullptr,
            ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoResize |
            ImGuiWindowFlags_NoMove     | ImGuiWindowFlags_NoScrollbar);

        // ── Top bar: MIDI port selector ───────────────────────────────────────
        ImGui::Text("MIDI Port:");
        ImGui::SameLine();
        ImGui::SetNextItemWidth(280.f);
        const char* preview = gs.ports.empty() ? "(none)"
                            : gs.ports[gs.selected_port].c_str();
        if (ImGui::BeginCombo("##port", preview)) {
            for (int i = 0; i < (int)gs.ports.size(); i++) {
                bool sel = (i == gs.selected_port);
                if (ImGui::Selectable(gs.ports[i].c_str(), sel))
                    gs.selected_port = i;
                if (sel) ImGui::SetItemDefaultFocus();
            }
            ImGui::EndCombo();
        }
        ImGui::SameLine();
        if (gs.midi_connected) {
            ImGui::PushStyleColor(ImGuiCol_Button, IM_COL32(60, 150, 60, 255));
            if (ImGui::Button("Disconnect")) {
                midi_in.close();
                gs.midi_connected = false;
            }
            ImGui::PopStyleColor();
        } else {
            if (ImGui::Button("Connect") && !gs.ports.empty()) {
                midi_in.open(engine, gs.selected_port);
                gs.midi_connected = midi_in.isOpen();
            }
        }
        ImGui::SameLine(0, 20.f);
        ImGui::TextColored(
            gs.midi_connected ? ImVec4(0.4f,1.f,0.4f,1.f) : ImVec4(1.f,0.4f,0.4f,1.f),
            gs.midi_connected ? "MIDI: connected" : "MIDI: not connected");
        ImGui::SameLine(0, 20.f);
        ImGui::Text("Voices: %d  |  Stereo 48kHz", gs.active_voices);
        // LFO activity badge
        bool lfo_active = gs.lfo_speed > 0 && gs.lfo_depth > 0;
        ImGui::SameLine(0, 14.f);
        if (lfo_active) {
            // Animate a simple phase indicator (changes color over time)
            float t = (float)ImGui::GetTime();
            float pulse = 0.5f + 0.5f * std::sin(t * 2.f * 3.14159f *
                          (2.f * gs.lfo_speed / 127.f));
            ImGui::TextColored({0.3f + 0.7f*pulse, 0.8f, 1.f, 1.f}, "LFO");
        } else {
            ImGui::TextDisabled("LFO off");
        }
        if (gs.sustain_on) {
            ImGui::SameLine(0, 10.f);
            ImGui::TextColored({1.f,0.9f,0.2f,1.f}, "[SUSTAIN]");
        }

        // ── Refresh MIDI port list button ─────────────────────────────────────
        ImGui::SameLine(0, 20.f);
        if (ImGui::SmallButton("Refresh")) {
            gs.ports = MidiInput::listPorts();
            gs.selected_port = 0;
        }

        ImGui::Separator();

        // ── MIDI activity indicators ──────────────────────────────────────────
        // LED dot + label, lights green for 80 ms after each event
        {
            const uint64_t now   = guiNowMs();
            const uint64_t flash = 80;
            const auto& act      = midi_in.activity();

            auto midiLed = [&](const char* label, uint64_t last_ms) {
                float        r   = 5.f;
                float        th  = ImGui::GetTextLineHeight();
                ImVec2       p   = ImGui::GetCursorScreenPos();
                bool         lit = (last_ms > 0) && ((now - last_ms) < flash);
                ImU32        col = lit ? IM_COL32(50, 230, 80, 255)
                                       : IM_COL32(35, 65, 35, 220);
                ImU32        rim = lit ? IM_COL32(120, 255, 140, 180)
                                       : IM_COL32(60, 90, 60, 160);
                ImGui::GetWindowDrawList()->AddCircleFilled(
                    {p.x + r, p.y + th * 0.5f}, r, col);
                ImGui::GetWindowDrawList()->AddCircle(
                    {p.x + r, p.y + th * 0.5f}, r, rim, 12, 1.f);
                ImGui::Dummy({r * 2.f + 2.f, th});
                ImGui::SameLine(0, 3.f);
                if (lit)
                    ImGui::TextColored({0.4f, 1.f, 0.5f, 1.f}, "%s", label);
                else
                    ImGui::TextDisabled("%s", label);
            };

            auto ledSep = [&]() {
                ImGui::SameLine(0, 10.f);
                ImVec2 pos = ImGui::GetCursorScreenPos();
                float  cy  = pos.y + ImGui::GetTextLineHeight() * 0.5f;
                ImGui::GetWindowDrawList()->AddLine(
                    {pos.x, cy}, {pos.x + 14.f, cy},
                    ImGui::GetColorU32(ImGuiCol_Separator), 1.f);
                ImGui::Dummy({14.f, ImGui::GetTextLineHeight()});
                ImGui::SameLine(0, 10.f);
            };

            midiLed("MIDI DATA",   act.any_ms.load(std::memory_order_relaxed));
            ledSep();
            midiLed("Note ON",     act.note_on_ms.load(std::memory_order_relaxed));
            ledSep();
            midiLed("Note OFF",    act.note_off_ms.load(std::memory_order_relaxed));
            ledSep();
            midiLed("Pedal Event", act.pedal_ms.load(std::memory_order_relaxed));

            // ── Output level indicator: red LED when peak > -9 dB ─────────────
            static constexpr float CLIP_THRESH = 0.3548f;  // 10^(-9/20)
            float peak_lin = engine.getOutputPeakLin();
            bool  over     = (peak_lin > CLIP_THRESH);
            {
                ledSep();
                float        r   = 5.f;
                float        th  = ImGui::GetTextLineHeight();
                ImVec2       p   = ImGui::GetCursorScreenPos();
                ImU32        col = over ? IM_COL32(230, 40, 40, 255)
                                        : IM_COL32(65, 30, 30, 220);
                ImU32        rim = over ? IM_COL32(255, 120, 120, 200)
                                        : IM_COL32(90, 50, 50, 160);
                ImGui::GetWindowDrawList()->AddCircleFilled(
                    {p.x + r, p.y + th * 0.5f}, r, col);
                ImGui::GetWindowDrawList()->AddCircle(
                    {p.x + r, p.y + th * 0.5f}, r, rim, 12, 1.f);
                ImGui::Dummy({r * 2.f + 2.f, th});
                ImGui::SameLine(0, 3.f);
                // Show dB value next to the LED
                float peak_db = (peak_lin > 1e-9f)
                    ? 20.f * std::log10(peak_lin) : -99.f;
                if (over)
                    ImGui::TextColored({1.f, 0.3f, 0.3f, 1.f},
                        "LEVEL %.1f dB", peak_db);
                else
                    ImGui::TextDisabled("LEVEL %.1f dB", peak_db);
            }
        }

        ImGui::Separator();

        // ── Two-column layout: [piano + matrix] | [params panel] ─────────────
        // Piano width = white_key_count * WHITE_W (+window padding on both sides)
        {
            int nw = 0;
            for (int m = PIANO_MIDI_LOW; m <= PIANO_MIDI_HIGH; m++)
                if (!isBlack(m)) nw++;
            float piano_px  = nw * WHITE_W;
            float pad       = ImGui::GetStyle().WindowPadding.x;
            float left_w    = piano_px + pad * 2.f;

            // ── Left child: piano + 2×2 matrix ───────────────────────────────
            ImGui::BeginChild("##left_panel", {left_w, 0.f}, false,
                              ImGuiWindowFlags_NoScrollbar);

        // ── Piano keyboard ────────────────────────────────────────────────────
        ImGui::PushStyleVar(ImGuiStyleVar_ItemSpacing, {0.f, 4.f});
        drawPiano(gs, engine);
        ImGui::PopStyleVar();

        ImGui::Separator();

        // ── 2×2 controller matrix ─────────────────────────────────────────────
        DspChain* dsp = engine.getDspChain();

        // Helper: draw one labeled slider with description line below
        // Returns true if value changed.
        auto labeledSlider = [](const char* id, const char* label,
                                const char* desc, int* val, int lo, int hi) -> bool {
            ImGui::Text("%s", label);
            ImGui::SameLine();
            ImGui::SetNextItemWidth(-1);
            bool changed = ImGui::SliderInt(id, val, lo, hi);
            ImGui::PushStyleColor(ImGuiCol_Text, IM_COL32(140,140,140,255));
            ImGui::TextUnformatted(desc);
            ImGui::PopStyleColor();
            ImGui::Spacing();
            return changed;
        };

        constexpr ImGuiTableFlags tflags =
            ImGuiTableFlags_BordersOuter |
            ImGuiTableFlags_BordersInnerV |
            ImGuiTableFlags_SizingStretchSame |
            ImGuiTableFlags_PadOuterX;

        if (ImGui::BeginTable("ctrl_matrix", 2, tflags)) {
            ImGui::TableNextRow();

            // ╔══════════════════╗  ╔══════════════════╗
            // ║      MIX         ║  ║     LFO PAN      ║
            // ╚══════════════════╝  ╚══════════════════╝

            ImGui::TableSetColumnIndex(0);
            ImGui::SeparatorText("MIX");
            {
                int v = gs.master_gain;
                char desc[48];
                snprintf(desc, sizeof(desc), "Output level: %d / 127", gs.master_gain);
                if (labeledSlider("##gain", "Gain", desc, &v, 0, 127)) {
                    gs.master_gain = (uint8_t)v;
                    engine.setAllVoicesMasterGain(gs.master_gain);
                }
            }
            {
                int v = gs.pan;
                char desc[48];
                float pan_pct = (gs.pan - 64) / 64.f;
                if (std::abs(pan_pct) < 0.02f)
                    snprintf(desc, sizeof(desc), "Stereo balance: center");
                else
                    snprintf(desc, sizeof(desc), "Stereo balance: %.0f%% %s",
                        std::abs(pan_pct)*100.f, pan_pct < 0 ? "L" : "R");
                if (labeledSlider("##pan", "Pan ", desc, &v, 0, 127)) {
                    gs.pan = (uint8_t)v;
                    engine.setAllVoicesPan(gs.pan);
                }
            }

            ImGui::TableSetColumnIndex(1);
            ImGui::SeparatorText("LFO PAN");
            {
                int v = gs.lfo_speed;
                char desc[48];
                snprintf(desc, sizeof(desc), "Rotation rate: %.2f Hz",
                    2.f * (gs.lfo_speed / 127.f));
                if (labeledSlider("##lfospd", "Speed", desc, &v, 0, 127)) {
                    gs.lfo_speed = (uint8_t)v;
                    engine.setAllVoicesPanSpeed(gs.lfo_speed);
                }
            }
            {
                int v = gs.lfo_depth;
                char desc[48];
                snprintf(desc, sizeof(desc), "Sweep width: %.0f%%",
                    100.f * (gs.lfo_depth / 127.f));
                if (labeledSlider("##lfodep", "Depth", desc, &v, 0, 127)) {
                    gs.lfo_depth = (uint8_t)v;
                    engine.setAllVoicesPanDepth(gs.lfo_depth);
                }
            }
            // Pan position indicator: animated dot sweeping L↔R
            {
                bool active = gs.lfo_speed > 0 && gs.lfo_depth > 0;
                ImDrawList* dl2 = ImGui::GetWindowDrawList();
                float bar_w = ImGui::GetContentRegionAvail().x;
                ImVec2 bar_pos = ImGui::GetCursorScreenPos();
                float bar_h = 10.f;
                // Background track
                dl2->AddRectFilled(bar_pos,
                    {bar_pos.x + bar_w, bar_pos.y + bar_h},
                    IM_COL32(40,40,40,200), 3.f);
                // Moving dot
                float pos_x = bar_pos.x + bar_w * 0.5f;
                if (active) {
                    float t = (float)ImGui::GetTime();
                    float hz = 2.f * (gs.lfo_speed / 127.f);
                    float depth = gs.lfo_depth / 127.f;
                    float lfo_val = depth * std::sin(t * 2.f * 3.14159f * hz);
                    pos_x = bar_pos.x + bar_w * 0.5f * (1.f + lfo_val);
                }
                ImU32 dot_col = active ? IM_COL32(80,200,255,255)
                                       : IM_COL32(80,80,80,180);
                dl2->AddCircleFilled({pos_x, bar_pos.y + bar_h*0.5f}, 5.f, dot_col);
                // L / R labels
                dl2->AddText({bar_pos.x, bar_pos.y}, IM_COL32(120,120,120,200), "L");
                dl2->AddText({bar_pos.x + bar_w - 8.f, bar_pos.y},
                    IM_COL32(120,120,120,200), "R");
                ImGui::Dummy({bar_w, bar_h + 2.f});
                if (!active) {
                    ImGui::PushStyleColor(ImGuiCol_Text, IM_COL32(140,140,140,255));
                    ImGui::TextUnformatted("Set Speed AND Depth > 0 to enable");
                    ImGui::PopStyleColor();
                }
            }

            // ╔══════════════════╗  ╔══════════════════╗
            // ║    LIMITER       ║  ║      BBE         ║
            // ╚══════════════════╝  ╚══════════════════╝

            ImGui::TableNextRow();
            ImGui::TableSetColumnIndex(0);
            {
                bool ena = gs.limiter_enabled;
                if (ImGui::Checkbox("##limon", &ena)) {
                    gs.limiter_enabled = ena;
                    if (dsp) dsp->setLimiterEnabled(ena ? 127 : 0);
                }
                ImGui::SameLine();
                ImGui::AlignTextToFramePadding();
                ImGui::TextUnformatted("LIMITER");
                ImGui::SameLine();
                {
                    ImVec2 pos = ImGui::GetCursorScreenPos();
                    float  w   = ImGui::GetContentRegionAvail().x;
                    float  mid = pos.y + ImGui::GetFrameHeight() * 0.5f;
                    ImGui::GetWindowDrawList()->AddLine(
                        {pos.x + 4.f, mid}, {pos.x + w, mid},
                        ImGui::GetColorU32(ImGuiCol_Separator));
                    ImGui::Dummy({w, 0.f});
                }
            }
            {
                int v = gs.limiter_thr;
                float db = -40.f + 40.f * (v / 127.f);
                char desc[48];
                snprintf(desc, sizeof(desc), "Ceiling: %.1f dB", db);
                if (labeledSlider("##limthr", "Threshold", desc, &v, 0, 127)) {
                    gs.limiter_thr = (uint8_t)v;
                    engine.setLimiterThreshold(gs.limiter_thr);
                }
            }
            {
                int v = gs.limiter_rel;
                float ms = 10.f + 1990.f * (v / 127.f);
                char desc[48];
                snprintf(desc, sizeof(desc), "Gain recovery: %.0f ms", ms);
                if (labeledSlider("##limrel", "Release  ", desc, &v, 0, 127)) {
                    gs.limiter_rel = (uint8_t)v;
                    engine.setLimiterRelease(gs.limiter_rel);
                }
            }
            // GR meter
            if (dsp) {
                float gr = gs.limiter_enabled
                    ? std::max(0.f, std::min(-dsp->limiter().gainReductionDb() / 40.f, 1.f))
                    : 0.f;
                char ovl[24];
                if (gs.limiter_enabled)
                    snprintf(ovl, sizeof(ovl), "GR  %.1f dB",
                             dsp->limiter().gainReductionDb());
                else
                    snprintf(ovl, sizeof(ovl), "GR  (disabled)");
                ImGui::PushStyleColor(ImGuiCol_PlotHistogram,
                    gs.limiter_enabled ? IM_COL32(220,60,60,200)
                                       : IM_COL32(70,70,70,150));
                ImGui::ProgressBar(gr, {-1.f, 14.f}, ovl);
                ImGui::PopStyleColor();
            }

            ImGui::TableSetColumnIndex(1);
            {
                bool ena = gs.bbe_enabled;
                if (ImGui::Checkbox("##bbeon", &ena)) {
                    gs.bbe_enabled = ena;
                    if (dsp) {
                        dsp->setBBEDefinition(ena ? gs.bbe_def : 0);
                        dsp->setBBEBassBoost (ena ? gs.bbe_bass : 0);
                    }
                }
                ImGui::SameLine();
                ImGui::AlignTextToFramePadding();
                ImGui::TextUnformatted("BBE  Sonic Maximizer");
                ImGui::SameLine();
                {
                    ImVec2 pos = ImGui::GetCursorScreenPos();
                    float  w   = ImGui::GetContentRegionAvail().x;
                    float  mid = pos.y + ImGui::GetFrameHeight() * 0.5f;
                    ImGui::GetWindowDrawList()->AddLine(
                        {pos.x + 4.f, mid}, {pos.x + w, mid},
                        ImGui::GetColorU32(ImGuiCol_Separator));
                    ImGui::Dummy({w, 0.f});
                }
            }
            {
                int v = gs.bbe_def;
                char desc[48];
                snprintf(desc, sizeof(desc), "5 kHz presence: +%.1f dB",
                    12.f * (v / 127.f));
                if (labeledSlider("##bbedef", "Definition", desc, &v, 0, 127)) {
                    gs.bbe_def = (uint8_t)v;
                    gs.bbe_enabled = (v > 0 || gs.bbe_bass > 0);
                    engine.setBBEDefinition(gs.bbe_def);
                }
            }
            {
                int v = gs.bbe_bass;
                char desc[48];
                snprintf(desc, sizeof(desc), "180 Hz warmth: +%.1f dB",
                    10.f * (v / 127.f));
                if (labeledSlider("##bbebas", "Bass Boost", desc, &v, 0, 127)) {
                    gs.bbe_bass = (uint8_t)v;
                    gs.bbe_enabled = (v > 0 || gs.bbe_def > 0);
                    engine.setBBEBassBoost(gs.bbe_bass);
                }
            }

            ImGui::EndTable();
        }

            ImGui::EndChild();  // ##left_panel

            // ── Vertical separator ────────────────────────────────────────────
            ImGui::SameLine(0, 0);
            {
                ImVec2 p = ImGui::GetCursorScreenPos();
                float  h = ImGui::GetContentRegionAvail().y;
                ImGui::GetWindowDrawList()->AddLine(
                    p, {p.x, p.y + h},
                    ImGui::GetColorU32(ImGuiCol_Separator), 1.f);
                ImGui::SetCursorScreenPos({p.x + 1.f, p.y});
            }
            ImGui::SameLine(0, 8.f);

            // ── Right child: synthesis params + live note data ────────────────
            ImGui::BeginChild("##right_panel", {0.f, 0.f}, false,
                              ImGuiWindowFlags_NoScrollbar);
            {
                // ── Static synthesis constants (hardcoded, not yet in SynthConfig) ─
                ImGui::SeparatorText("FIXED CONSTANTS");
                constexpr ImGuiTableFlags fcf =
                    ImGuiTableFlags_BordersInnerV | ImGuiTableFlags_SizingStretchSame;
                if (ImGui::BeginTable("##fixedconst", 3, fcf)) {
                    ImGui::TableSetupColumn("ENVELOPE");
                    ImGui::TableSetupColumn("DECORRELATION");
                    ImGui::TableSetupColumn("EQ / FILTER");
                    ImGui::TableHeadersRow();
                    ImGui::TableNextRow();

                    auto frow = [](const char* name, const char* val, const char* unit) {
                        ImGui::PushStyleColor(ImGuiCol_Text, IM_COL32(210,200,140,220));
                        ImGui::Text("%-14s", name);   // pad to 14 → values align
                        ImGui::PopStyleColor();
                        ImGui::SameLine(0, 0);
                        ImGui::Text("%s %s", val, unit);
                    };

                    ImGui::TableSetColumnIndex(0);
                    frow("release_ms",   "10.0",  "ms");

                    ImGui::TableSetColumnIndex(1);
                    frow("ap_base_gain", "0.35",  "");
                    frow("ap_scale_l",   "0.25",  "L");
                    frow("ap_scale_r",   "0.20",  "R");

                    ImGui::TableSetColumnIndex(2);
                    frow("eq_q",         "1.4",   "Q");
                    frow("eq_gain_clamp","+-24",  "dB");

                    ImGui::EndTable();
                }
                ImGui::Spacing();
                ImGui::Separator();

                // ── SynthConfig — 3 columns by nature ────────────────────────
                ImGui::SeparatorText("SYNTHESIS PARAMS");
                constexpr ImGuiTableFlags scf =
                    ImGuiTableFlags_BordersInnerV | ImGuiTableFlags_SizingStretchSame;
                if (ImGui::BeginTable("##synthcfg", 3, scf)) {
                    const SynthConfig& sc = engine.getSynthConfig();

                    // Column headers
                    ImGui::TableSetupColumn("STEREO");
                    ImGui::TableSetupColumn("TIMBRE");
                    ImGui::TableSetupColumn("LEVEL / ENV");
                    ImGui::TableHeadersRow();
                    ImGui::TableNextRow();

                    auto cv = [](const char* name, const char* fmt, float val, const char* unit) {
                        ImGui::PushStyleColor(ImGuiCol_Text, IM_COL32(160,200,255,200));
                        ImGui::Text("%-14s", name);   // pad to 14 → values align
                        ImGui::PopStyleColor();
                        char buf[32]; snprintf(buf, sizeof(buf), fmt, val);
                        ImGui::SameLine(0, 0); ImGui::Text("%s %s", buf, unit);
                    };

                    // STEREO col
                    ImGui::TableSetColumnIndex(0);
                    cv("pan_spread",   "%.3f", sc.pan_spread,   "rad");
                    cv("stereo_decorr","%.3f", sc.stereo_decorr,"");
                    cv("stereo_boost", "%.3f", sc.stereo_boost, "");

                    // TIMBRE col
                    ImGui::TableSetColumnIndex(1);
                    cv("beat_scale",    "%.3f", sc.beat_scale,         "");
                    cv("hb_brightness", "%.3f", sc.harmonic_brightness,"");
                    cv("eq_strength",   "%.3f", sc.eq_strength,        "");
                    cv("eq_freq_min",   "%.0f", sc.eq_freq_min,        "Hz");

                    // LEVEL/ENV col
                    ImGui::TableSetColumnIndex(2);
                    cv("target_rms",   "%.4f", sc.target_rms,  "");
                    cv("vel_gamma",    "%.3f", sc.vel_gamma,   "");
                    cv("noise_level",  "%.3f", sc.noise_level, "");
                    cv("onset_ms",     "%.1f", sc.onset_ms,    "ms");

                    ImGui::EndTable();
                }

                ImGui::Spacing();
                ImGui::Separator();

                // ── Last note header ──────────────────────────────────────────
                int   ln_midi = engine.getLastNoteMidi();
                int   ln_vel  = engine.getLastNoteVel();
                static const char* nnames[] = {
                    "C","C#","D","D#","E","F","F#","G","G#","A","A#","B"};
                uint32_t ln_seed = engine.getLastNoteSeed();
                ImGui::PushStyleColor(ImGuiCol_Text, IM_COL32(255,220,100,255));
                ImGui::Text("LAST NOTE  %s%d  (MIDI %d)  vel %d",
                    nnames[ln_midi % 12], ln_midi / 12 - 1, ln_midi, ln_vel);
                ImGui::PopStyleColor();
                ImGui::SameLine(0, 16.f);
                ImGui::PushStyleColor(ImGuiCol_Text, IM_COL32(180,180,180,200));
                ImGui::Text("seed  0x%08X  (%u)", ln_seed, ln_seed);
                ImGui::PopStyleColor();

                NoteParams np = engine.lookupNote(ln_midi, ln_vel);
                if (!np.valid) {
                    ImGui::TextDisabled("(no data for this note)");
                } else {
                    // ── Note meta — 3 columns: structure | noise | EQ ─────────
                    constexpr ImGuiTableFlags mf =
                        ImGuiTableFlags_BordersInnerV | ImGuiTableFlags_SizingStretchSame;
                    if (ImGui::BeginTable("##notemeta", 3, mf)) {
                        ImGui::TableSetupColumn("STRUCTURE");
                        ImGui::TableSetupColumn("NOISE");
                        ImGui::TableSetupColumn("SPECTRAL EQ");
                        ImGui::TableHeadersRow();
                        ImGui::TableNextRow();

                        ImGui::TableSetColumnIndex(0);
                        ImGui::Text("strings  %d", np.n_strings);
                        ImGui::Text("partials %d", np.n_partials);
                        ImGui::Text("width    %.3f", np.width_factor);

                        ImGui::TableSetColumnIndex(1);
                        ImGui::Text("centroid  %.0f Hz", np.noise.centroid_hz);
                        ImGui::Text("floor_rms %.4f",   np.noise.floor_rms);
                        ImGui::Text("tau       %.3f s", np.noise.attack_tau_s);

                        // EQ stats
                        float eq_min = np.eq_gains_db[0], eq_max = np.eq_gains_db[0], eq_sum = 0.f;
                        for (int i = 0; i < EQ_POINTS; i++) {
                            if (np.eq_gains_db[i] < eq_min) eq_min = np.eq_gains_db[i];
                            if (np.eq_gains_db[i] > eq_max) eq_max = np.eq_gains_db[i];
                            eq_sum += np.eq_gains_db[i];
                        }
                        ImGui::TableSetColumnIndex(2);
                        ImGui::Text("points   %d", EQ_POINTS);
                        ImGui::Text("min      %.1f dB", eq_min);
                        ImGui::Text("max      %.1f dB", eq_max);
                        ImGui::Text("mean     %.1f dB", eq_sum / EQ_POINTS);

                        ImGui::EndTable();
                    }

                    ImGui::Spacing();

                    // ── Partials table — scrollable ───────────────────────────
                    ImGui::SeparatorText("PARTIALS");
                    constexpr ImGuiTableFlags ptf =
                        ImGuiTableFlags_BordersInnerV | ImGuiTableFlags_RowBg |
                        ImGuiTableFlags_SizingFixedFit | ImGuiTableFlags_ScrollY;
                    float row_h = ImGui::GetTextLineHeightWithSpacing();
                    float tbl_h = 12.5f * row_h;  // ~12 rows visible, scroll for rest

                    if (ImGui::BeginTable("##partials", 8, ptf, {0.f, tbl_h})) {
                        ImGui::TableSetupScrollFreeze(0, 1);
                        ImGui::TableSetupColumn("k",       ImGuiTableColumnFlags_WidthFixed, 26.f);
                        ImGui::TableSetupColumn("f_hz",    ImGuiTableColumnFlags_WidthFixed, 66.f);
                        ImGui::TableSetupColumn("A0",      ImGuiTableColumnFlags_WidthFixed, 66.f);
                        ImGui::TableSetupColumn("tau1",    ImGuiTableColumnFlags_WidthFixed, 48.f);
                        ImGui::TableSetupColumn("tau2",    ImGuiTableColumnFlags_WidthFixed, 48.f);
                        ImGui::TableSetupColumn("a1",      ImGuiTableColumnFlags_WidthFixed, 44.f);
                        ImGui::TableSetupColumn("beat_hz", ImGuiTableColumnFlags_WidthFixed, 58.f);
                        ImGui::TableSetupColumn("mo",      ImGuiTableColumnFlags_WidthFixed, 22.f);
                        ImGui::TableHeadersRow();

                        for (int k = 0; k < np.n_partials; k++) {
                            const PartialParams& pp = np.partials[k];
                            ImGui::TableNextRow();
                            ImGui::TableSetColumnIndex(0); ImGui::Text("%d", pp.k);
                            ImGui::TableSetColumnIndex(1); ImGui::Text("%.2f", pp.f_hz);
                            ImGui::TableSetColumnIndex(2); ImGui::Text("%.5f", pp.A0);
                            ImGui::TableSetColumnIndex(3); ImGui::Text("%.2f", pp.tau1);
                            ImGui::TableSetColumnIndex(4);
                            if (pp.a1 < 1.f - 1e-5f) ImGui::Text("%.2f", pp.tau2);
                            else                       ImGui::TextDisabled("-");
                            ImGui::TableSetColumnIndex(5); ImGui::Text("%.3f", pp.a1);
                            ImGui::TableSetColumnIndex(6);
                            if (pp.beat_hz > 1e-6f) ImGui::Text("%.4f", pp.beat_hz);
                            else                     ImGui::TextDisabled("0");
                            ImGui::TableSetColumnIndex(7);
                            ImGui::TextDisabled(pp.mono ? "y" : "n");
                        }
                        ImGui::EndTable();
                    }
                }
            }
            ImGui::EndChild();  // ##right_panel
        }  // two-column scope

        // ── Sustain button (mouse/keyboard fallback) ──────────────────────────
        ImGui::Separator();
        ImGui::Text("Spacebar = sustain   |   Click keys to play");
        ImGui::SameLine(0, 20.f);
        if (ImGui::IsKeyPressed(ImGuiKey_Space)) {
            static bool sus = false;
            sus = !sus;
            engine.sustainPedal(sus ? 127 : 0);
            gs.sustain_on = sus;
        }

        // Keyboard shortcuts (qwerty → notes C4..B4)
        const ImGuiKey qkeys[] = {
            ImGuiKey_A, ImGuiKey_W, ImGuiKey_S, ImGuiKey_E,
            ImGuiKey_D, ImGuiKey_F, ImGuiKey_T, ImGuiKey_G,
            ImGuiKey_Y, ImGuiKey_H, ImGuiKey_U, ImGuiKey_J
        };
        const int qmidis[] = { 60,61,62,63,64,65,66,67,68,69,70,71 };
        for (int i = 0; i < 12; i++) {
            if (ImGui::IsKeyPressed(qkeys[i], false)) {
                engine.noteOn((uint8_t)qmidis[i], gs.master_gain);
                gs.active_notes[qmidis[i]] = true;
            }
            if (ImGui::IsKeyReleased(qkeys[i])) {
                engine.noteOff((uint8_t)qmidis[i]);
                gs.active_notes[qmidis[i]] = false;
            }
        }

        ImGui::End();

        // ── Render ────────────────────────────────────────────────────────────
        ImGui::Render();
        glViewport(0, 0, fb_w, fb_h);
        glClearColor(0.12f, 0.12f, 0.14f, 1.f);
        glClear(GL_COLOR_BUFFER_BIT);
        ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
        glfwSwapBuffers(win);
    }

    // ── Cleanup ───────────────────────────────────────────────────────────────
    midi_in.close();
    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();
    glfwDestroyWindow(win);
    glfwTerminate();
    logger.log("GUI", LogSeverity::Info, "GUI shutdown");
    return 0;
}
