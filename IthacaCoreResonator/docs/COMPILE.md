# IthacaCoreResonator — Kompilace a spuštění

---

## Požadavky

### Windows (primární platforma)

| Nástroj | Verze | Poznámka |
|---------|-------|----------|
| Visual Studio 2022 | 17.x (Community / Pro) | MSVC toolchain — nutné |
| CMake | ≥ 3.16 | `winget install Kitware.CMake` nebo https://cmake.org |
| Git | libovolná | pro FetchContent (GLFW, ImGui) |
| OpenGL | systémový | součást Windows, není třeba instalovat |

Kompilátor: **MSVC 14.44+** (`cl.exe`), `x64` architektura.
AVX2 je zapnuté automaticky (`/arch:AVX2`).

> Alternativně lze použít **MinGW-w64 + GCC** (viz sekce níže), ale primárně se builduje přes MSVC.

---

## Adresářová struktura

```
resonators/                         ← kořen repozitáře
├── IthacaCoreResonator/            ← C++ projekt
│   ├── CMakeLists.txt
│   ├── main.cpp                    ← CLI vstupní bod
│   ├── gui_main.cpp                ← GUI vstupní bod
│   ├── synth/                      ← syntézní jádro
│   ├── dsp/                        ← BBE + limiter
│   ├── gui/                        ← Dear ImGui frontend
│   ├── sampler/                    ← logger (z IthacaCore)
│   ├── third_party/                ← vendorované deps (součást repozitáře)
│   │   ├── json.hpp                ← nlohmann/json (MIT)
│   │   ├── miniaudio.h             ← audio I/O (MIT)
│   │   ├── RtMidi.h / RtMidi.cpp  ← MIDI I/O (MIT)
│   ├── soundbanks/                 ← parametrické banky (*.json, nejsou v gitu)
│   │   ├── salamander.json         ← Salamander Grand Piano (zkopírovat ručně)
│   │   └── .gitignore
│   ├── build/                      ← CMake build adresář (není v gitu)
│   └── docs/                       ← dokumentace
```

GLFW a Dear ImGui se stáhnou automaticky při prvním CMake configure přes FetchContent.

---

## Build — Windows (MSVC, doporučený postup)

Všechny příkazy se spouštějí z adresáře `IthacaCoreResonator/`.

### 1. Configure

```bat
cd IthacaCoreResonator

cmake -B build -G "Visual Studio 17 2022" -A x64
```

CMake při configure automaticky stáhne:
- GLFW 3.4 z `https://github.com/glfw/glfw.git`
- Dear ImGui v1.91.9 z `https://github.com/ocornut/imgui.git`

Vyžaduje připojení k internetu při prvním spuštění. Následná buildení jsou offline.

### 2. Build (Release)

```bat
cmake --build build --config Release
```

Výstupní binárky:

| Binary | Cesta |
|--------|-------|
| GUI (primární) | `build/bin/Release/IthacaCoreResonatorGUI.exe` |
| CLI (headless) | `build/bin/Release/IthacaCoreResonator.exe` |

### 3. Build (Debug)

```bat
cmake --build build --config Debug
```

Výstup: `build/bin/Debug/IthacaCoreResonatorGUI.exe`

---

## Build — alternativa přes Developer Command Prompt

Pokud `cmake` není v PATH nebo je třeba ručně nastavit toolchain:

```bat
:: Otevři "Developer Command Prompt for VS 2022" (x64)
"C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat"

cd IthacaCoreResonator
cmake -B build -G "Visual Studio 17 2022" -A x64
cmake --build build --config Release
```

---

## Build — MinGW-w64 / GCC (alternativní)

Pokud není k dispozici Visual Studio:

```bash
# Požadavky: mingw-w64 s g++ ≥ 11, cmake ≥ 3.16
cmake -B build-mingw -G "MinGW Makefiles" \
      -DCMAKE_BUILD_TYPE=Release
cmake --build build-mingw

# Výstup: build-mingw/bin/IthacaCoreResonatorGUI.exe
```

---

## Soundbanka — umístění a spuštění

### Kde banka musí být

Binárky se spouštějí z `IthacaCoreResonator/` jako working directory.
Výchozí cesta k bance je relativní:

```
IthacaCoreResonator/soundbanks/salamander.json
```

Soubor `soundbanks/*.json` **není v gitu** (15 MB, viz `soundbanks/.gitignore`).
Musí být zkopírován ručně.

### Zkopírování banky

Banka vzniká extrakcí z WAV sample banku příkazem (viz Python část projektu):

```bat
python analysis/extract-params.py --bank C:/SoundBanks/IthacaPlayer/ks-grand --out analysis/params-salamander.json
```

Po extrakci zkopírovat do `soundbanks/`:

```bat
copy analysis\params-salamander.json IthacaCoreResonator\soundbanks\salamander.json
```

Na Linuxu/macOS:
```bash
cp analysis/params-salamander.json IthacaCoreResonator/soundbanks/salamander.json
```

### Spuštění GUI

```bat
:: Z adresáře IthacaCoreResonator/
build\bin\Release\IthacaCoreResonatorGUI.exe

:: Explicitní cesta k bance (volitelné, pokud se liší od výchozí):
build\bin\Release\IthacaCoreResonatorGUI.exe soundbanks\salamander.json
```

### Spuštění CLI (headless, bez GUI)

```bat
build\bin\Release\IthacaCoreResonator.exe soundbanks\salamander.json [midi_port]
```

`midi_port` — index MIDI vstupu (výchozí: 0, první dostupný). Dostupné porty jsou vypsány při startu.

---

## Rebuild po změně kódu

```bat
:: Rychlý rebuild (jen změněné soubory):
cmake --build build --config Release

:: Pokud se změnil CMakeLists.txt nebo se přidal soubor:
cmake -B build -G "Visual Studio 17 2022" -A x64
cmake --build build --config Release
```

Pokud je GUI spuštěné, je třeba ho zavřít před rebuildem (linker zamkne .exe):

```bat
:: Git Bash / PowerShell:
cmd /c "taskkill /F /IM IthacaCoreResonatorGUI.exe" 2>/dev/null; cmake --build build --config Release
```

---

## Čistý build od nuly

```bat
rmdir /S /Q build
cmake -B build -G "Visual Studio 17 2022" -A x64
cmake --build build --config Release
```

---

## Závislosti — přehled

| Závislost | Zdroj | Licence | Způsob |
|-----------|-------|---------|--------|
| nlohmann/json | `third_party/json.hpp` | MIT | vendored |
| miniaudio | `third_party/miniaudio.h` | MIT/Unlicense | vendored |
| RtMidi | `third_party/RtMidi.h/.cpp` | MIT | vendored |
| GLFW 3.4 | GitHub (FetchContent) | MIT | automaticky |
| Dear ImGui v1.91.9 | GitHub (FetchContent) | MIT | automaticky |
| OpenGL | systém (Windows) | — | `find_package` |
| MSVC (Windows) | Windows SDK | — | systém |
| WinMM (MIDI) | Windows SDK | — | `-lwinmm` |
