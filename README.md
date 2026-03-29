# Ithaca Resonator Synth — Physics-Informed Grand Piano Synthesizer

Fyzikálně věrný aditivní syntezátor grand piána. Parametry jsou extrahovány z reálných
sample banků a použity k řízení banky sinusoidálních oscilátorů. Výsledkem je plná
88 × 8 sample banka ve WAV formátu kompatibilní s Ithaca Player.

---

## Pipeline — dvě nezávislé fáze

```
WAV soubory (88 not × 8 velocity vrstev)
         │
         ▼
┌─────────────────────────────────────────────────────────────┐
│  FÁZE 1 — Fyzikální extrakce  (extract_params.py)          │
│                                                             │
│  Čistá analýza signálu, žádná NN.                          │
│  Pro každý WAV soubor:                                      │
│    • FFT → harmonické píky → f0, B (inharmonicita)         │
│    • STFT (frekvence-adaptivní okno) → obálka každého      │
│      parciálu → bi-exponenciální fit → tau1, tau2, a1      │
│    • Analýza modulace obálky → beat_hz (tlukot strun)      │
│    • Šumový model útoku → attack_tau, centroid, A_noise    │
│    • LTASE porovnání orig/synth → spektrální EQ            │
│    • M/S analýza → stereo_width_factor                     │
│                                                             │
│  Výsledek: analysis/params.json  (701 měřených not)        │
└─────────────────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────────┐
│  FÁZE 2 — NN vyhlazení / interpolace  (train_instrument_   │
│           profile.py)                                       │
│                                                             │
│  Malá faktorizovaná NN se učí z params.json.               │
│  Účel: vyhladit šum extrakce, interpolovat chybějící noty, │
│  zajistit fyzikální konzistenci přes celý rozsah MIDI.     │
│                                                             │
│  Výsledek: analysis/params_profile.json  (88 × 8 = 704)   │
└─────────────────────────────────────────────────────────────┘
         │
         ▼
    GUI generátor → WAV sample banka
```

**Proč dvě fáze?** Extrakce je přesná ale šumová (nízké velocity → nízký SNR → nespolehlivý tau).
NN je hladká funkce přes MIDI prostor — regularizuje a doplňuje mezery bez ztráty fyzikální
interpretovatelnosti.

---

## Quick Start

```bash
# 1. Prostředí
.venv312/Scripts/pip install -r requirements.txt

# 2. Extrakce parametrů z WAV souborů (Fáze 1)
python analysis/extract_params.py \
    --bank C:/SoundBanks/IthacaPlayer/ks-grand \
    --out  analysis/params.json

# 3. Spektrální EQ (porovnání s originály)
python analysis/compute_spectral_eq.py

# 4. NN vyhlazení parametrů (Fáze 2)
python analysis/train_instrument_profile.py \
    --in    analysis/params.json \
    --out   analysis/params_profile.json \
    --model analysis/profile.pt \
    --epochs 800

# 5. Spuštění GUI
python -m uvicorn gui.server:app --port 8989 --reload
# Otevřít http://localhost:8989
```

---

## Syntézní model (`analysis/physics_synth.py`)

Každá nota je součet sinusoidálních parciálů s bi-exponenciální obálkou:

```
partial_k(t) = A0_k · [a1·exp(-t/τ1_k) + (1-a1)·exp(-t/τ2_k)] · cos(2π·f_k·t + φ_k)

f_k = k · f0 · √(1 + B·k²)     # inharmonicita (tuhost struny)
```

Pro noty s více strunami (2 nebo 3 dle MIDI rozsahu) má každá struna vlastní pan úhel
a frekvenci `f_k ± beat_hz/2` — tlukot vzniká přirozeně jako superpozice, nikoli
jako amplitudová modulace. Výsledek je stereofonní signál s různými L/R obálkami.

### Počet strun dle MIDI

| Rozsah | Struny | Poznámka |
|--------|--------|----------|
| MIDI 21–27 (A0–Eb1) | 1 | Basové jednochordální struny |
| MIDI 28–48 (E1–C3)  | 2 | Bichord (wound) |
| MIDI 49–108 (C#3+)  | 3 | Trichord |

---

## NN Instrument Profile (`train_instrument_profile.py`)

Faktorizovaná síť — každý fyzikální parametr má vlastní sub-síť:

```
B_net         MLP(midi)           → log(B)               inharmonicita (vel-nezávislá)
dur_net       MLP(midi)           → log(duration)         délka noty
tau1_k1_net   MLP(midi, vel)      → log(τ1) pro k=1       sustain fundamentálu
tau_ratio_net MLP(midi, k)        → log(τk/τk1)           decay ratio k>1  [≤ 0]
A0_net        MLP(midi, k, vel)   → log(A0_ratio)         spektrální tvar
df_net        MLP(midi, k)        → log(beat_hz)          tlukot strun
eq_net        MLP(midi, freq)     → gain_db               tělesová EQ
wf_net        MLP(midi)           → log(width_factor)     stereo šířka
noise_net     MLP(midi, vel)      → [log(τ_noise),        šumový model útoku:
                                     log(centroid_hz),     • decay tiempo útoku
                                     log(A_noise)]         • barva (centroid)
                                                           • amplituda
biexp_net     MLP(midi, k, vel)   → [logit(a1),           bi-exponenciální obálka:
                                     log(τ2/τ1)]           • podíl rychlého/pomalého
                                                           • poměr τ2/τ1
```

**Sinusoidální MIDI embedding** zachycuje přechody mezi rejstříky:
`[m, sin(πm), sin(2πm), sin(4πm), cos(πm), cos(2πm)]`  kde `m = (midi-21)/87`

**Fyzikální omezení:** `tau_ratio ≤ 0` → žádný parciál se nerozpadá pomaleji než fundamentál.
`biexp`: `τ2 > τ1` vždy (pomalejší druhý exponent).

~90 000 parametrů, trénink ~30 s na CPU (800 epoch).

---

## Frekvence-adaptivní STFT okno (extract_params.py)

Klíčová volba při extrakci obálky parciálů. Fixní okno způsobuje problém u basových not:

| MIDI | f0 | frame (fixní) | binů/harmoniku |
|------|----|---------------|----------------|
| A0 (21) | 27.5 Hz | 8192 | **5** ← overlap k=1/k=2! |
| A2 (45) | 110 Hz  | 8192 | 20 ← OK |
| A5 (81) | 880 Hz  | 8192 | 163 ← výborné |

S adaptivním oknem (`TARGET = 20 binů/harmoniku`):

| MIDI | f0 | frame | res/bin | hop |
|------|----|-------|---------|-----|
| A0 (21) | 27.5 Hz | 32768 | 1.35 Hz | 185 ms |
| A1 (33) | 55 Hz   | 16384 | 2.69 Hz | 93 ms  |
| A2 (45) | 110 Hz  |  8192 | 5.38 Hz | 46 ms  |
| A4 (69) | 440 Hz  |  2048 | 21.5 Hz | 12 ms  |

Výšky dostávají lepší časové rozlišení (kratší obálky). Basy dostávají lepší frekvenční
rozlišení (zabraňuje přetékání energie mezi harmonickými při extrakci obálky).

---

## Synthesis Parameters

### RENDER

| Parametr | Default | Rozsah | Popis |
|----------|---------|--------|-------|
| `sr` | 44100 | 22050–48000 Hz | Vzorkovací kmitočet |
| `duration` | auto | 0.5–10 s | Délka noty; `null` = z params.json |
| `fade_out` | 0.5 | 0–5 s | Fade-out na konci noty |
| `target_rms` | 0.06 | 0.01–0.25 | RMS pro velocity 7 (≈ −24 dBFS) |
| `velocity_curve_gamma` | 0.7 | 0.2–2.0 | Mocninná křivka velocity: `rms = target_rms × ((vel+1)/8)^γ` |
| `onset_ms` | 3.0 | 0–20 ms | Délka náběhové rampy (eliminace kliknutí z náhodné fáze oscilátoru) |

### TIMBRE

| Parametr | Default | Rozsah | Popis |
|----------|---------|--------|-------|
| `harmonic_brightness` | 1.0 | 0–3 | Boost výšin: `gain(k) = 1 + brightness × log₂(k)` |
| `beat_scale` | 1.0 | 0–3 × | Multiplikátor tlukotu strun (0 = bez tlukotu) |
| `eq_strength` | 0.5 | 0–1 | Síla spektrální EQ (LTASE korekce); 0 = bypass |
| `eq_freq_min` | 400 | 50–2000 Hz | Dolní mez EQ; pod touto frekvencí EQ klesá na 0 dB |
| `soundboard_strength` | 0.0 | 0–1 | Konvoluce s IR resonanční desky (PARKED — IR způsobuje band-pass artefakty) |
| `noise_level` | 1.0 | 0–5 × | Globální multiplikátor amplitudy útokového šumu (NN-predicted A_noise × noise_level) |
| `vel_color_blend` | 0.7 | 0–1 | Mísení spektrální barvy směrem k referenční velocity |
| `vel_color_ref` | 4 | 0–7 | Referenční velocity pro color blend (vel4 = SNR sweet spot) |

### STEREO

| Parametr | Default | Rozsah | Popis |
|----------|---------|--------|-------|
| `pan_spread` | 0.55 | 0–1.5 rad | Rozptyl pan úhlů per struna; 0 = mono |
| `stereo_boost` | 1.0 | 0.5–4 × | M/S side-channel multiplikátor na vrchu extrahovaného width_factor |
| `stereo_decorr` | 1.0 | 0–3 × | Multiplikátor síly Schroederova all-pass dekorelátor (0 = L/R identické) |

### PER-NOTE OVERRIDES (aditivní delty)

| Parametr | Default | Rozsah | Popis |
|----------|---------|--------|-------|
| `harmonic_brightness_delta` | 0.0 | ±2 | Delta k globálnímu harmonic_brightness |
| `beat_scale_delta` | 0.0 | ±2 | Delta k globálnímu beat_scale |
| `pan_spread_delta` | 0.0 | ±1 | Delta k globálnímu pan_spread |
| `tau1_k1_scale` | 1.0 | 0.1–5 × | Multiplikátor τ1 pro k=1 (sustain fundamentálu) |

---

## GUI Workflow

GUI je FastAPI backend + vanilla JS SPA na portu **8989**.

```
1. New Session    — název, zdrojový params.json, metadata nástroje
2. RENDER panel   — sr, duration, target_rms, onset_ms
3. TIMBRE panel   — brightness, beat_scale, EQ, noise_level
4. STEREO panel   — pan_spread, stereo_boost, stereo_decorr
5. PER-NOTE       — MIDI vstup, delta overrides pro konkrétní notu
6. GENERATE       — MIDI rozsah, velocity vrstvy → WAV banka
7. VELOCITY PROFILE — výpočet RMS poměrů z originálních WAV
8. PLAYER + SPECTRUM — poslechový náhled + Welch spektrum
```

LCD header zobrazuje:
- Řádek 1: `ITHACA RESONATOR SYNTH`
- Řádek 2: `PATCH NAME: <jméno session>`
- Řádek 3: `▶ <aktuálně přehrávaný soubor>`

Sessions jsou uloženy v `gui/sessions/<name>/`:
- `config.json` — všechny parametry
- `params.json` — kopie zdrojových dat
- `generated/` — WAV soubory + `instrument-definition.json`

---

## Výstupní formát

Soubory: `mXXX-velY-fZZ.wav`
- `XXX` — MIDI nota (021–108)
- `Y` — velocity vrstva (0–7)
- `ZZ` — vzorkovací kmitočet (44 = 44100 Hz)

`instrument-definition.json` v `generated/` po každé generaci.

---

## Struktura projektu

```
analysis/
  extract_params.py            # Fáze 1: fyzikální extrakce z WAV → params.json
  compute_spectral_eq.py       # LTASE spektrální EQ + stereo width
  smooth_tau_cross_velocity.py # Cross-velocity tau + colour korekce
  interpolate_missing_notes.py # Spline interpolace pro řídké banky
  train_instrument_profile.py  # Fáze 2: NN vyhlazení → params_profile.json
  train_ddsp.py                # Alternativa: DDSP end-to-end trénink z WAV
  physics_synth.py             # Jádro syntézy
  params.json                  # Extrahované parametry (Fáze 1 výstup)
  params_profile.json          # NN-vyhlazené parametry (Fáze 2 výstup)
  profile.pt                   # Uložené váhy NN (train_instrument_profile)

gui/
  server.py                    # FastAPI (port 8989)
  config_schema.py             # Metadata parametrů, defaults, rozsahy
  routers/                     # API endpointy
  static/                      # Frontend (index.html, app.js, style.css)
  sessions/                    # Session configs a generované soubory (gitignored)
```

---

---

## Onboarding nové WAV banky — kompletní workflow

### 1. Formát vstupních souborů

Každý WAV soubor musí splňovat:

| Požadavek | Hodnota |
|-----------|---------|
| Formát | WAV (PCM 16 nebo 24 bit, nebo float32) |
| Vzorkovací kmitočet | 44100 Hz (doporučeno) nebo 48000 Hz |
| Kanály | stereo (2ch) — mono je automaticky duplexováno |
| Délka | alespoň 1 s; ideálně přirozené doznění (10–30 s pro bas) |
| Název souboru | `mXXX-velY-fZZ.wav` viz níže |

**Konvence pojmenování:**
```
m060-vel3-f44.wav
│    │    └─ fZZ: vzorkovací kmitočet (44 = 44100 Hz, 48 = 48000 Hz)
│    └────── velY: velocity vrstva 0–7  (0 = nejslabší, 7 = nejsilnější)
└─────────── mXXX: MIDI nota 021–108   (060 = C4/Middle C)
```

Příklad úplné banky: `m021-vel0-f44.wav` až `m108-vel7-f44.wav` = 88 × 8 = 704 souborů.

> Nejsou potřeba všechny noty — NN interpoluje chybějící pozice. Minimálně pokrýt
> representativní vzorek (každou oktávu, all velocity vrstvy).

---

### 2. Nahrání a příprava

```
C:/SoundBanks/MujNastroj/
  m021-vel0-f44.wav
  m021-vel1-f44.wav
  ...
  m108-vel7-f44.wav
```

Zkontrolovat délky a výstřely:
```bash
python -c "
import soundfile as sf, pathlib, collections
d = pathlib.Path('C:/SoundBanks/MujNastroj')
for f in sorted(d.glob('*.wav')):
    info = sf.info(str(f))
    print(f'{f.name:30s}  {info.duration:.1f}s  {info.samplerate}Hz  {info.channels}ch')
"
```

---

### 3. Fáze 1 — Fyzikální extrakce parametrů

```bash
python analysis/extract_params.py \
    --bank C:/SoundBanks/MujNastroj \
    --out  analysis/params.json
```

Trvání: ~5–15 minut (7 workers, 704 souborů).

Výstup `params.json` obsahuje pro každý soubor:
- `B` — koeficient inharmonicity (stiffness)
- `partials[]` — pro každý parciál k: `f_hz`, `A0`, `tau1`, `tau2`, `a1`, `beat_hz`
- `noise{}` — model útoku: `attack_tau_s`, `centroid_hz`
- `spectral_eq{}` — EQ korekce (přidá se v kroku 4)
- `duration_s`, `midi`, `vel`

---

### 4. Spektrální EQ (volitelné, doporučeno)

Porovná LTASE (long-term average spectral envelope) originálu vs. syntézy a uloží
korekční filtr per nota → přidá přirozenou barvu těla nástroje.

```bash
python analysis/compute_spectral_eq.py \
    --params analysis/params.json \
    --bank   C:/SoundBanks/MujNastroj
```

---

### 5. Cross-velocity tau smoothing (volitelné)

Tau (doba dozvuku) je fyzikální vlastnost struny — neměla by záviset na velocity.
Nízké velocity mají nízký SNR → nespolehlivý tau. Tento krok vyhlazuje tau přes
velocity vrstvy podle SNR spolehlivosti.

```bash
python analysis/smooth_tau_cross_velocity.py \
    --inplace \
    --color-blend 0.7
```

`--color-blend 0.7`: přenese 70 % spektrální barvy z referenční velocity (vel4)
na ostatní vrstvy — kompenzuje SNR-způsobené rozdíly ve spektrálním tvaru.

---

### 6. Fáze 2 — NN vyhlazení a interpolace

```bash
python analysis/train_instrument_profile.py \
    --in    analysis/params.json \
    --out   analysis/params_profile.json \
    --model analysis/profile.pt \
    --epochs 800
```

Trvání: ~30 sekund na CPU.

NN se naučí fyzikálně konzistentní parametry přes celý MIDI rozsah.
Výstup `params_profile.json` obsahuje kompletních 88 × 8 = 704 not
(i ty, které chyběly v originální bance — NN interpoluje).

Sledovat dataset coverage ve výpisu:
```
B=657  tau=5212  tau1_k1=686  A0=34779  df=30194  noise=701  biexp=8196
```
Nízký počet `df` nebo `biexp` = málo detekovatelného tlukotu / bi-exp. rozpadu.

---

### 7. Vytvoření GUI session

1. Spustit GUI: `python -m uvicorn gui.server:app --port 8989 --reload`
2. Otevřít `http://localhost:8989`
3. Kliknout **+ New Session**:
   - Název session (např. `bechstein_d282`)
   - Source params: `analysis/params_profile.json`  ← NN-vyhlazená verze
   - Metadata nástroje (jméno, autor, kategorie)
4. **Save Parameters** po nastavení globálních parametrů

---

### 8. Generování WAV sample banky

V GUI:
1. Nastavit **MIDI from** / **MIDI to** (typicky 21–108)
2. Zaškrtnout požadované **velocity vrstvy** (0–7)
3. Kliknout **▶ Generate**

Nebo z příkazové řádky (pro batch generaci):
```bash
# Viz gui/routers/generate.py pro API endpointy
curl -X POST http://localhost:8989/api/sessions/bechstein_d282/generate \
     -H "Content-Type: application/json" \
     -d '{"midi_from": 21, "midi_to": 108, "vel_layers": [0,1,2,3,4,5,6,7]}'
```

Výstupní soubory: `gui/sessions/bechstein_d282/generated/mXXX-velY-f44.wav`

---

### 9. Poslechová verifikace

Použít **PLAYER** panel v GUI — kliknout na soubor pro přehrání se spektrem.

Věci ke kontrole:
- **Bass noty (MIDI 21–45)**: přirozený tlukot strun, správná délka dozvuku
- **Střední rejstřík (MIDI 46–72)**: konzistentní barva přes velocity vrstvy
- **Výšky (MIDI 73–108)**: správná délka, bez aliasingu
- **Velocity přechody**: plynulý RMS přechod vel0 → vel7 (gamma křivka ~0.7)

---

### Přehled příkazů — celý workflow

```bash
# 1. Extrakce
python analysis/extract_params.py --bank C:/SoundBanks/MujNastroj --out analysis/params.json

# 2. Spektrální EQ
python analysis/compute_spectral_eq.py --params analysis/params.json --bank C:/SoundBanks/MujNastroj

# 3. Cross-velocity smoothing
python analysis/smooth_tau_cross_velocity.py --inplace --color-blend 0.7

# 4. NN profil
python analysis/train_instrument_profile.py --in analysis/params.json --out analysis/params_profile.json --model analysis/profile.pt --epochs 800

# 5. GUI → New Session → Generate
python -m uvicorn gui.server:app --port 8989 --reload
```

---

## Data

Zdroj: `C:/SoundBanks/IthacaPlayer/ks-grand/` — 704 WAV souborů, 88 MIDI not × 8 velocity vrstev @ 44.1 kHz.

Papers: `C:/Users/jindr/OneDrive/Osobni/LordAudio/IhtacaPapers/` — 16 dokumentů o fyzikálním modelování piána (skupina Chabassier/Inria, Simionato 2024 DDSP, Bank/Chabassier 2019 review).

---

## TODO / Known Limitations

### Soundboard (rezonanční deska) — PARKED

`soundboard_strength` parametr existuje v GUI i syntezátoru, ale je defaultně `0.0` a nedoporučuje se používat.

**Proč je zaparkován:** Aktuální syntetický modální IR (40 módů) způsobuje:
- Band-pass distorzi — zužuje frekvenční odezvu místo rozšíření
- Amplitudovou modulaci ("croaking") z modálních rezonancí
- Celkové snížení vnímaného těla místo jeho přidání

**Co skutečný soundboard dělá:** přidává difuzní reverb ocas, lehce zesvětluje transient,
NEZUŽUJE band. Tento efekt nelze věrně replikovat syntetickým IR bez měřeného impulzního
záznamu z konkrétního nástroje.

**Aktuální náhrada:** `eq_strength` + `compute_spectral_eq.py` pokrývá spektrální tvarování
těla (rezonanční boosty, rolloff výšin) bez IR artefaktů.

**Plánované řešení:** Naměřit skutečný IR z fyzického nástroje (pistolový výstřel nebo
sweep) a použít ho místo syntetického. Alternativně: konvoluční reverb z piano recording
odečtený od přímého signálu.
