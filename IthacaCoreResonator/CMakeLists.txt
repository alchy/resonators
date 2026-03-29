cmake_minimum_required(VERSION 3.16)
project(IthacaCoreResonator VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

# ── Sources ────────────────────────────────────────────────────────────────────

set(SYNTH_SOURCES
    synth/note_lut.cpp
    synth/resonator_voice.cpp
    synth/voice_manager.cpp
    synth/biquad_eq.cpp
    synth/resonator_engine.cpp
    synth/midi_input.cpp
    third_party/RtMidi.cpp
)

# core_logger is header-only stub (no .cpp needed until IthacaCore DSP is copied)
set(LOGGER_SOURCES)

# DSP chain is a header-only stub until IthacaCore dsp/ files are copied
set(DSP_SOURCES)

add_executable(IthacaCoreResonator
    main.cpp
    ${SYNTH_SOURCES}
    ${LOGGER_SOURCES}
    ${DSP_SOURCES}
)

# ── Include paths ─────────────────────────────────────────────────────────────

target_include_directories(IthacaCoreResonator PRIVATE
    ${CMAKE_SOURCE_DIR}
    ${CMAKE_SOURCE_DIR}/synth         # resonator_engine.h includes miniaudio.h
    ${CMAKE_SOURCE_DIR}/third_party   # nlohmann/json.hpp, miniaudio.h
)

# ── Platform dependencies (miniaudio + RtMidi) ────────────────────────────────
if (WIN32)
    target_compile_definitions(IthacaCoreResonator PRIVATE __WINDOWS_MM__)
    target_link_libraries(IthacaCoreResonator PRIVATE ole32 winmm)

elseif (APPLE)
    target_compile_definitions(IthacaCoreResonator PRIVATE __MACOSX_CORE__)
    target_link_libraries(IthacaCoreResonator PRIVATE
        "-framework CoreAudio"
        "-framework AudioToolbox"
        "-framework CoreFoundation"
        "-framework CoreMIDI")

else()  # Linux
    # MIDI backend: prefer JACK, fall back to ALSA
    find_package(PkgConfig QUIET)
    if (PkgConfig_FOUND)
        pkg_check_modules(JACK QUIET jack)
    endif()

    if (JACK_FOUND)
        target_compile_definitions(IthacaCoreResonator PRIVATE __UNIX_JACK__)
        target_link_libraries(IthacaCoreResonator PRIVATE ${JACK_LIBRARIES})
    else()
        target_compile_definitions(IthacaCoreResonator PRIVATE __LINUX_ALSA__)
        target_link_libraries(IthacaCoreResonator PRIVATE asound)
    endif()

    target_link_libraries(IthacaCoreResonator PRIVATE pthread m dl)
endif()

# ── SIMD ─────────────────────────────────────────────────────────────────────

if (CMAKE_SYSTEM_PROCESSOR MATCHES "AMD64|x86_64|X86_64")
    if (MSVC)
        target_compile_options(IthacaCoreResonator PRIVATE /arch:AVX2)
    else()
        target_compile_options(IthacaCoreResonator PRIVATE -mavx2 -mfma)
    endif()
endif()

# ── Compiler flags ────────────────────────────────────────────────────────────

if (MSVC)
    target_compile_options(IthacaCoreResonator PRIVATE
        $<$<CONFIG:Debug>:   /Od /Zi>
        $<$<CONFIG:Release>: /O2 /DNDEBUG>
    )
    target_compile_definitions(IthacaCoreResonator PRIVATE
        _CRT_SECURE_NO_WARNINGS
        WIN32_LEAN_AND_MEAN
    )
else()
    target_compile_options(IthacaCoreResonator PRIVATE
        $<$<CONFIG:Debug>:   -g -O0>
        $<$<CONFIG:Release>: -O3 -DNDEBUG>
    )
endif()

# ── Custom targets ────────────────────────────────────────────────────────────

add_custom_target(run
    COMMAND IthacaCoreResonator
    DEPENDS IthacaCoreResonator
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
)

add_custom_target(clean-exports
    COMMAND ${CMAKE_COMMAND} -E remove_directory ${CMAKE_SOURCE_DIR}/exports
    COMMENT "Removing exported WAV files"
)
