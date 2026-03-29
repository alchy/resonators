# IthacaCoreResonator — MIDI SysEx Parameter Protocol

Každý zobrazovaný parametr syntezátoru má přiřazené **16bitové ID** a lze jej nastavit
nebo číst přes standardní MIDI System Exclusive zprávu.
Protokol pokrývá statické konstanty (FIXED CONSTANTS), dynamické parametry
(SynthConfig) i read-only data z LUT (per-note params.json data).

---

## Schéma parametrického ID (16 bit)

```
 15  14  13  12 | 11  10   9   8   7   6   5   4   3   2   1   0
[  CATEGORY  ]  [            PARAMETER INDEX                    ]
   4 bity            12 bitů
   (0–15)             (0–4095)

ID = (category << 12) | param_index

Příklady:
  0x3002  →  kategorie 3 (STEREO), parametr 2 (stereo_boost)
  0x4001  →  kategorie 4 (TIMBRE), parametr 1 (harmonic_brightness)
```

### Tabulka kategorií

| Cat | Hex prefix  | Název         | R/W    |
|-----|-------------|---------------|--------|
|  0  | `0x0___`    | ENVELOPE      | R/W    |
|  1  | `0x1___`    | DECORRELATION | R/W    |
|  2  | `0x2___`    | EQ / FILTER   | R/W    |
|  3  | `0x3___`    | STEREO        | R/W    |
|  4  | `0x4___`    | TIMBRE        | R/W    |
|  5  | `0x5___`    | LEVEL / ENV   | R/W    |
|  6  | `0x6___`    | STRUCTURE     | R only |
|  7  | `0x7___`    | NOISE         | R only |
|  8  | `0x8___`    | SPECTRAL EQ   | R only |
|  9  | `0x9___`    | PARTIALS      | R only |
| 10–15 | `0xA___`–`0xF___` | Reserved | —  |

---

## Formát SysEx zprávy

Všechny bajty v SysEx musí být ≤ 0x7F (7bitová data — MIDI standard).

### Identifikátor zařízení

```
Manufacturer ID:  0x7D              (non-commercial / educational)
Device signature: 0x49 0x43 0x52   ("ICR" — IthacaCoreResonator)
```

### Typy zpráv

| Code | Název           | Směr       |
|------|-----------------|------------|
| 0x01 | SET_PARAM       | Host → ICR |
| 0x02 | GET_PARAM       | Host → ICR |
| 0x03 | PARAM_RESPONSE  | ICR → Host |
| 0x04 | SET_ALL         | Host → ICR |
| 0x05 | REQUEST_ALL     | Host → ICR |
| 0x06 | ALL_PARAMS_DUMP | ICR → Host |

---

### SET_PARAM — nastavení jednoho parametru

```
F0  7D  49 43 52  01  [id0] [id1] [id2]  [v0] [v1] [v2] [v3] [v4]  F7
```

Celková délka: **15 bajtů**.

**Kódování ID** (16 bit → 3 × 7bitové bajty):
```
id0 = (param_id >> 14) & 0x03   // bity 15–14
id1 = (param_id >>  7) & 0x7F   // bity 13– 7
id2 =  param_id        & 0x7F   // bity  6– 0
```

**Kódování hodnoty** (IEEE 754 float → 5 × 7bitové bajty):
```c
uint32_t raw;
memcpy(&raw, &float_value, 4);
v0 =  raw        & 0x7F   // bity  6– 0
v1 = (raw >>  7) & 0x7F   // bity 13– 7
v2 = (raw >> 14) & 0x7F   // bity 20–14
v3 = (raw >> 21) & 0x7F   // bity 27–21
v4 = (raw >> 28) & 0x0F   // bity 31–28
```

---

### GET_PARAM — dotaz na hodnotu

```
F0  7D  49 43 52  02  [id0] [id1] [id2]  F7
```

Délka: 9 bajtů. ICR odpoví zprávou PARAM_RESPONSE.

---

### PARAM_RESPONSE — odpověď na GET_PARAM

```
F0  7D  49 43 52  03  [id0] [id1] [id2]  [v0] [v1] [v2] [v3] [v4]  F7
```

Stejné kódování ID a hodnoty jako SET_PARAM.

---

### SET_ALL — hromadné nastavení

Opakované SET_PARAM bloky (pouze R/W parametry, kategorie 0–5):
```
F0  7D  49 43 52  04
    [id0 id1 id2 v0 v1 v2 v3 v4]  ...  (N parametrů)
F7
```

---

### REQUEST_ALL / ALL_PARAMS_DUMP

```
F0  7D  49 43 52  05  F7              ← host žádá dump
F0  7D  49 43 52  06  [data...]  F7   ← ICR odpovídá (stejný formát jako SET_ALL)
```

---

## Kompletní tabulka parametrů

### 0x0 — ENVELOPE (R/W)

| ID     | Klíč       | Popis                         | Default | Min  | Max   | Typ      |
|--------|------------|-------------------------------|---------|------|-------|----------|
| 0x0000 | release_ms | Délka note-off release rampy  | 10.0    | 1.0  | 500.0 | float ms |

### 0x1 — DECORRELATION (R/W)

| ID     | Klíč         | Popis                                          | Default | Min  | Max  | Typ   |
|--------|--------------|------------------------------------------------|---------|------|------|-------|
| 0x1000 | ap_base_gain | Základní koef. Schroeder all-pass              | 0.35    | 0.00 | 0.95 | float |
| 0x1001 | ap_scale_l   | Škálování dekorelace L kanálu dle MIDI polohy  | 0.25    | 0.00 | 1.00 | float |
| 0x1002 | ap_scale_r   | Škálování dekorelace R kanálu dle MIDI polohy  | 0.20    | 0.00 | 1.00 | float |

### 0x2 — EQ / FILTER (R/W)

| ID     | Klíč          | Popis                               | Default | Min  | Max  | Typ      |
|--------|---------------|-------------------------------------|---------|------|------|----------|
| 0x2000 | eq_q          | Q faktor peaking biquad EQ          | 1.4     | 0.1  | 10.0 | float    |
| 0x2001 | eq_gain_clamp | Max. korekce spektrálního EQ        | 24.0    | 6.0  | 48.0 | float dB |

### 0x3 — STEREO (R/W)

| ID     | Klíč          | Popis                                  | Default | Min  | Max  | Typ       |
|--------|---------------|----------------------------------------|---------|------|------|-----------|
| 0x3000 | pan_spread    | Rozestup strun ve stereo obraze        | 0.55    | 0.00 | 1.57 | float rad |
| 0x3001 | stereo_decorr | Intenzita Schroeder dekorelace         | 1.00    | 0.00 | 1.00 | float     |
| 0x3002 | stereo_boost  | Boost M/S side kanálu nad width_factor | 1.00    | 0.00 | 4.00 | float     |

### 0x4 — TIMBRE (R/W)

| ID     | Klíč                | Popis                                       | Default | Min   | Max    | Typ      |
|--------|---------------------|---------------------------------------------|---------|-------|--------|----------|
| 0x4000 | beat_scale          | Multiplikátor beat_hz                       | 1.00    | 0.00  | 5.00   | float    |
| 0x4001 | harmonic_brightness | Zesílení vyšších harmonických: 1+hb·log₂(k) | 0.00   | -2.00 | 4.00   | float    |
| 0x4002 | eq_strength         | Blend spektrálního EQ (0=bypass, 1=plný)   | 1.00    | 0.00  | 1.00   | float    |
| 0x4003 | eq_freq_min         | EQ je flat pod touto frekvencí             | 400.0   | 20.0  | 2000.0 | float Hz |

### 0x5 — LEVEL / ENV (R/W)

| ID     | Klíč        | Popis                                   | Default | Min   | Max   | Typ      |
|--------|-------------|-----------------------------------------|---------|-------|-------|----------|
| 0x5000 | target_rms  | Cílová RMS při normalizaci úrovně       | 0.0600  | 0.001 | 0.500 | float    |
| 0x5001 | vel_gamma   | Exponent velocity křivky: (vel/127)^γ   | 0.700   | 0.10  | 3.00  | float    |
| 0x5002 | noise_level | Multiplikátor amplitudy šumu            | 1.000   | 0.00  | 4.00  | float    |
| 0x5003 | onset_ms    | Délka click-prevention onset rampy      | 3.00    | 0.00  | 50.0  | float ms |

### 0x6 — STRUCTURE (read-only, per-note z LUT)

| ID     | Klíč         | Popis                              | Typ   |
|--------|--------------|------------------------------------|-------|
| 0x6000 | n_strings    | Počet strun pro danou notu         | uint8 |
| 0x6001 | n_partials   | Počet parciálů                     | uint8 |
| 0x6002 | width_factor | M/S width faktor ze sample analýzy | float |

### 0x7 — NOISE (read-only, per-note z LUT)

| ID     | Klíč        | Popis                         | Typ      |
|--------|-------------|-------------------------------|----------|
| 0x7000 | centroid_hz | Centroid šumu (LP filtr)      | float Hz |
| 0x7001 | floor_rms   | Amplituda šumu útoku          | float    |
| 0x7002 | attack_tau_s| Časová konstanta rozpadu šumu | float s  |

### 0x8 — SPECTRAL EQ (read-only, per-note z LUT)

| ID     | Klíč       | Popis                    | Typ      |
|--------|------------|--------------------------|----------|
| 0x8000 | eq_points  | Počet EQ bodů (vždy 64) | uint8    |
| 0x8001 | eq_min_db  | Minimální EQ korekce     | float dB |
| 0x8002 | eq_max_db  | Maximální EQ korekce     | float dB |
| 0x8003 | eq_mean_db | Průměrná EQ korekce      | float dB |

### 0x9 — PARTIALS (read-only, per-note z LUT)

| ID     | Klíč             | Popis                                | Typ      |
|--------|------------------|--------------------------------------|----------|
| 0x9000 | partial_k        | Index parciálu k                     | uint8    |
| 0x9001 | partial_f_hz     | Frekvence parciálu                   | float Hz |
| 0x9002 | partial_A0       | Amplituda parciálu (normalizovaná)   | float    |
| 0x9003 | partial_tau1     | Rychlá časová konstanta              | float s  |
| 0x9004 | partial_tau2     | Pomalá časová konstanta              | float s  |
| 0x9005 | partial_a1       | Podíl rychlé složky bi-exp           | float    |
| 0x9006 | partial_beat_hz  | Beating frekvence strun              | float Hz |
| 0x9007 | partial_mono     | Mono příznak (single-exp)            | uint8    |

---

## Příklady zpráv

### Nastavit pan_spread = 0.8 rad (ID 0x3000)

```
ID:    0x3000  →  id0=0x00  id1=0x18  id2=0x00
float 0.8f → IEEE 754: 0x3F4CCCCD
  v0=0x4D  v1=0x19  v2=0x53  v3=0x7A  v4=0x03

Zpráva:  F0 7D 49 43 52  01  00 18 00  4D 19 53 7A 03  F7
```

### Dotaz na stereo_decorr (ID 0x3001)

```
F0 7D 49 43 52  02  00 18 01  F7
```

### Nastavit beat_scale = 2.0 (ID 0x4000)

```
ID:    0x4000  →  id0=0x00  id1=0x20  id2=0x00
float 2.0f → IEEE 754: 0x40000000
  v0=0x00  v1=0x00  v2=0x00  v3=0x00  v4=0x04

Zpráva:  F0 7D 49 43 52  01  00 20 00  00 00 00 00 04  F7
```

---

## Setter pattern — C++ implementace

```cpp
// Decode incoming SysEx message
bool parseSysEx(const uint8_t* data, int len, ResonatorEngine& engine) {
    if (len < 9) return false;
    if (data[0] != 0xF0) return false;
    if (data[1] != 0x7D || data[2] != 0x49 || data[3] != 0x43 || data[4] != 0x52)
        return false;

    uint8_t msg_type = data[5];

    if (msg_type == 0x01 && len >= 14) {  // SET_PARAM
        uint16_t param_id = ((uint16_t)(data[6] & 0x03) << 14)
                          | ((uint16_t)(data[7] & 0x7F) <<  7)
                          |  (uint16_t)(data[8] & 0x7F);

        uint32_t raw = (uint32_t)(data[9]  & 0x7F)
                    | ((uint32_t)(data[10] & 0x7F) <<  7)
                    | ((uint32_t)(data[11] & 0x7F) << 14)
                    | ((uint32_t)(data[12] & 0x7F) << 21)
                    | ((uint32_t)(data[13] & 0x0F) << 28);
        float value;
        memcpy(&value, &raw, 4);

        return applyParam(param_id, value, engine);
    }
    return false;
}

// Dispatch by category and index
bool applyParam(uint16_t param_id, float value, ResonatorEngine& engine) {
    uint8_t  cat = (param_id >> 12) & 0x0F;
    uint16_t idx =  param_id        & 0x0FFF;

    switch (cat) {
        case 0x0:  // ENVELOPE
            if (idx == 0x000) { /* engine.setReleaseMsGlobal(value); */ return true; }
            break;
        case 0x1:  // DECORRELATION
            // engine.setApBaseGain(value) etc. — not yet wired
            break;
        case 0x2:  // EQ/FILTER
            // engine.setEqQ(value) etc. — not yet wired
            break;
        case 0x3:  // STEREO
            if (idx == 0x000) { engine.vm().setSynthPanSpread(value);    return true; }
            if (idx == 0x001) { engine.vm().setSynthStereoDecorr(value); return true; }
            if (idx == 0x002) { engine.vm().setSynthStereoBoost(value);  return true; }
            break;
        case 0x4:  // TIMBRE
            if (idx == 0x000) { engine.vm().setSynthBeatScale(value);          return true; }
            if (idx == 0x001) { engine.vm().setSynthHarmonicBrightness(value); return true; }
            if (idx == 0x002) { engine.vm().setSynthEqStrength(value);         return true; }
            if (idx == 0x003) { engine.vm().setSynthEqFreqMin(value);          return true; }
            break;
        case 0x5:  // LEVEL/ENV
            if (idx == 0x000) { engine.vm().setSynthTargetRms(value);   return true; }
            if (idx == 0x001) { engine.vm().setSynthVelGamma(value);    return true; }
            if (idx == 0x002) { engine.vm().setSynthNoiseLevel(value);  return true; }
            if (idx == 0x003) { engine.vm().setSynthOnsetMs(value);     return true; }
            break;
        // 0x6–0x9: read-only, SET ignored
    }
    return false;
}
```

---

## Počty parametrů

| Kategorie     | Settable | Read-only | Celkem |
|---------------|----------|-----------|--------|
| ENVELOPE      | 1        | 0         | 1      |
| DECORRELATION | 3        | 0         | 3      |
| EQ/FILTER     | 2        | 0         | 2      |
| STEREO        | 3        | 0         | 3      |
| TIMBRE        | 4        | 0         | 4      |
| LEVEL/ENV     | 4        | 0         | 4      |
| STRUCTURE     | 0        | 3         | 3      |
| NOISE         | 0        | 3         | 3      |
| SPECTRAL EQ   | 0        | 4         | 4      |
| PARTIALS      | 0        | 8×N       | 8×N    |
| **Celkem R/W**| **17**   |           |        |

SET_ALL dump (17 R/W parametrů): `17 × 8 + 7 header + 1 F7` = **144 bajtů**.
