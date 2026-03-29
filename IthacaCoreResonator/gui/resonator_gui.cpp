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

    GLFWwindow* win = glfwCreateWindow(1000, 420,
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
        ImGui::Text("Voices: %d", gs.active_voices);
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

        // ── Piano keyboard ────────────────────────────────────────────────────
        ImGui::PushStyleVar(ImGuiStyleVar_ItemSpacing, {0.f, 4.f});
        drawPiano(gs, engine);
        ImGui::PopStyleVar();

        ImGui::Separator();

        // ── Controls ──────────────────────────────────────────────────────────
        float col_w = (float)(fb_w - 40) / 4.f;

        // Master gain
        {
            int v = gs.master_gain;
            ImGui::SetNextItemWidth(col_w);
            if (ImGui::SliderInt("Master Gain", &v, 0, 127)) {
                gs.master_gain = (uint8_t)v;
                engine.setAllVoicesMasterGain(gs.master_gain);
            }
        }
        ImGui::SameLine(0, 16.f);
        // Pan
        {
            int v = gs.pan;
            ImGui::SetNextItemWidth(col_w);
            if (ImGui::SliderInt("Pan", &v, 0, 127)) {
                gs.pan = (uint8_t)v;
                engine.setAllVoicesPan(gs.pan);
            }
        }
        ImGui::SameLine(0, 16.f);
        // LFO speed
        {
            int v = gs.lfo_speed;
            ImGui::SetNextItemWidth(col_w);
            if (ImGui::SliderInt("LFO Speed", &v, 0, 127)) {
                gs.lfo_speed = (uint8_t)v;
                engine.setAllVoicesPanSpeed(gs.lfo_speed);
            }
        }
        ImGui::SameLine(0, 16.f);
        // LFO depth
        {
            int v = gs.lfo_depth;
            ImGui::SetNextItemWidth(col_w);
            if (ImGui::SliderInt("LFO Depth", &v, 0, 127)) {
                gs.lfo_depth = (uint8_t)v;
                engine.setAllVoicesPanDepth(gs.lfo_depth);
            }
        }

        // ── DSP chain ─────────────────────────────────────────────────────────
        ImGui::Separator();
        DspChain* dsp = engine.getDspChain();

        // Limiter row
        ImGui::TextColored({0.9f,0.7f,0.2f,1.f}, "LIMITER");
        ImGui::SameLine(0, 6.f);
        {
            bool ena = gs.limiter_enabled;
            if (ImGui::Checkbox("##limon", &ena)) {
                gs.limiter_enabled = ena;
                if (dsp) dsp->setLimiterEnabled(ena ? 127 : 0);
            }
        }
        ImGui::SameLine(0, 12.f);
        {
            int v = gs.limiter_thr;
            ImGui::SetNextItemWidth(col_w - 20.f);
            if (ImGui::SliderInt("Threshold##lim", &v, 0, 127)) {
                gs.limiter_thr = (uint8_t)v;
                engine.setLimiterThreshold(gs.limiter_thr);
            }
            if (ImGui::IsItemHovered()) {
                float db = -40.f + 40.f * (gs.limiter_thr / 127.f);
                ImGui::SetTooltip("%.1f dB", db);
            }
        }
        ImGui::SameLine(0, 12.f);
        {
            int v = gs.limiter_rel;
            ImGui::SetNextItemWidth(col_w - 20.f);
            if (ImGui::SliderInt("Release##lim", &v, 0, 127)) {
                gs.limiter_rel = (uint8_t)v;
                engine.setLimiterRelease(gs.limiter_rel);
            }
            if (ImGui::IsItemHovered()) {
                float ms = 10.f + 1990.f * (gs.limiter_rel / 127.f);
                ImGui::SetTooltip("%.0f ms", ms);
            }
        }
        // GR meter
        if (dsp && gs.limiter_enabled) {
            ImGui::SameLine(0, 12.f);
            float gr = -dsp->limiter().gainReductionDb() / 40.f;
            gr = std::max(0.f, std::min(gr, 1.f));
            char ovl[16]; snprintf(ovl, sizeof(ovl), "%.1f dB",
                dsp->limiter().gainReductionDb());
            ImGui::PushStyleColor(ImGuiCol_PlotHistogram, IM_COL32(220,60,60,200));
            ImGui::ProgressBar(gr, {80.f, 14.f}, ovl);
            ImGui::PopStyleColor();
        }

        // BBE row
        ImGui::SameLine(0, 24.f);
        ImGui::TextColored({0.4f,0.85f,1.f,1.f}, "BBE");
        ImGui::SameLine(0, 6.f);
        {
            bool ena = gs.bbe_enabled;
            if (ImGui::Checkbox("##bbeon", &ena)) {
                gs.bbe_enabled = ena;
                if (dsp) {
                    dsp->setBBEDefinition(ena ? gs.bbe_def : 0);
                    dsp->setBBEBassBoost (ena ? gs.bbe_bass : 0);
                }
            }
        }
        ImGui::SameLine(0, 12.f);
        {
            int v = gs.bbe_def;
            ImGui::SetNextItemWidth(col_w - 20.f);
            if (ImGui::SliderInt("Definition##bbe", &v, 0, 127)) {
                gs.bbe_def = (uint8_t)v;
                gs.bbe_enabled = (v > 0 || gs.bbe_bass > 0);
                engine.setBBEDefinition(gs.bbe_def);
            }
            if (ImGui::IsItemHovered())
                ImGui::SetTooltip("5 kHz shelf: +%.1f dB", 12.f*(gs.bbe_def/127.f));
        }
        ImGui::SameLine(0, 12.f);
        {
            int v = gs.bbe_bass;
            ImGui::SetNextItemWidth(col_w - 20.f);
            if (ImGui::SliderInt("Bass Boost##bbe", &v, 0, 127)) {
                gs.bbe_bass = (uint8_t)v;
                gs.bbe_enabled = (v > 0 || gs.bbe_def > 0);
                engine.setBBEBassBoost(gs.bbe_bass);
            }
            if (ImGui::IsItemHovered())
                ImGui::SetTooltip("180 Hz shelf: +%.1f dB", 10.f*(gs.bbe_bass/127.f));
        }

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
