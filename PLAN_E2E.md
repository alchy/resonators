# End-to-End Differentiable Piano Synthesizer — Refactoring Plan
## Branch: dev-idea

---

## Motivace

Současná architektura (EGRB / PhysicsResonatorBank) trénuje **globálně sdílené parametry** pro
celý keyboard — jeden vektor tau1/tau2/A0/beat_hz pro všechny noty. Toto je fundamentální
omezení: každá nota klavíru má unikátní fyzikální charakteristiku.

Nová architektura: **Setter NN** mapuje `(midi, vel) → kompletní sadu fyzikálních parametrů`
per nota, trénovaná end-to-end přes audio loss. Warm-start z analytické extrakce zajišťuje
rozumný počáteční bod a předchází cold-start problému.

Oporu tvoří Simionato et al. (2024) "Physics-informed differentiable method for piano modeling"
— stejný přístup, ale náš model je rozšířen o: bi-exponenciální útlum, per-parciální beating,
noise model, spektrální EQ, stereo width, a plný keyboard (88 × 8 vs 24 × 7).

---

## Klíčové designové rozhodnutí (poučení z Simionato 2024)

1. **Dvě tréninkové fáze** — nejdřív B/f0 (zakotvení harmonické struktury), pak amplitudy.
   U nás tři fáze: warm-start supervised → spektrální audio loss → plná loss.
2. **Menší výstupní frame = lepší STFT loss** — syntéza po 24-240 vzorcích umožňuje
   přesnější kontrolu časového vývoje parciálů. Náš syntetizér bude frame-based (240 vzorků
   @ 48kHz = 5ms).
3. **logMSE F0 loss** jako anchor — zakotvuje první parciál před trénováním amplitud.
4. **Interpolace** mezi neviděnými notami funguje dobře, zejména pro B (quasi-lineární
   v log prostoru). Velocity interpolace je těžší (nelineární energie parciálů).

---

## Co se ODSTRANÍ (stará EGRB architektura)

```
models/physics_bank.py      → NAHRAZENO models/diff_synth.py
models/resonator_bank.py    → ODSTRANĚNO (EGRB s gate mechanismem)
models/controller.py        → ODSTRANĚNO (frame controller pro EGRB)
models/envelope_net.py      → ODSTRANĚNO (EnvelopeNet pro EGRB)
losses/losses.py            → NAHRAZENO losses/piano_loss.py
train.py                    → NAHRAZENO train_e2e.py
data/dataset.py             → NAHRAZENO data/piano_dataset.py
```

Co se ZACHOVÁVÁ:
```
analysis/extract-params.py        → beze změny (zdroj warm-start dat)
analysis/compute-spectral-eq.py   → beze změny
analysis/physics_synth.py         → beze změny (numpy inference, ne tréninková cesta)
analysis/train-instrument-profile.py → zachováno jako reference
gui/                              → beze změny (pipeline GUI)
infer.py                          → aktualizovat pro nový model
```

---

## Nová struktura souborů

```
models/
  setter_nn.py        # NEW: (midi_norm, vel_norm) → param dict
  diff_synth.py       # NEW: param dict → audio (PyTorch, differenciabilní)

losses/
  piano_loss.py       # NEW: MRSTFT + L1 + attack-weight + F0_logMSE + physics_reg

data/
  piano_dataset.py    # NEW: WAV + params.json dataset, segmentace, oversampling útoku

train_e2e.py          # NEW: 3-fázový trénink s warm-startem
config_e2e.json       # NEW: konfigurace pro novou architekturu
```

---

## Komponenty — specifikace

---

### `models/setter_nn.py` — SetterNN

**Vstup:** `(f0_norm, vel_norm)` — 2 skaláry (f0 normalizován log-lineárně A0→C8)

**Výstup:** param dict pro jeden tón:
```python
{
  'B':           (1,)      # inharmonicita, ReLU výstup (≥ 0)
  'f0_offset':   (1,)      # ladění v cents, tanh*50
  'A0':          (K,)      # amplitudy parciálů, Softplus
  'tau1':        (K,)      # rychlý útlum [s], Softplus + min_clamp
  'tau2':        (K,)      # pomalý útlum [s], Softplus + min_clamp
  'a1':          (K,)      # mixing ratio, Sigmoid
  'beat_hz':     (K,)      # detuning Hz, Softplus + min_clamp
  'beat_depth':  (K,)      # hloubka beatingu, Sigmoid
  'noise':       (4,)      # [attack_tau, floor_rms, centroid_norm, slope]
}
```

**Architektura** (inspirováno Simionato, rozšířeno):
```
B_net:         Linear(1, 64) → ReLU → Linear(64, 1) → Softplus
               vstup: pouze f0_norm (B nezávisí na velocity)

beat_net:      Linear(2, 64) → ReLU → Linear(64, K*2) → [Softplus, Sigmoid]
               výstup: [beat_hz, beat_depth] pro K parciálů

decay_net:     Linear(2, 256) → ReLU → Linear(256, 256) → ReLU → Linear(256, K*3)
               výstup: [tau1, tau2, a1] pro K parciálů

amp_net:       Linear(2, 256) → ReLU → Linear(256, 256) → ReLU → Linear(256, K)
               výstup: log_A0 → Softmax → K amplitud

noise_net:     Linear(2, 64) → ReLU → Linear(64, 4) → Softplus/Sigmoid
```

**Warm-start**: metoda `init_from_params(params_json)` — trénink supervised
na extrahovaných parametrech pomocí MSE loss. Toto je fáze 0 trénování.

**K = 64 parciálů** (kompromis: bass noty mají ~80, výšky ~20; průměr pokrývá majority).

---

### `models/diff_synth.py` — DifferentiablePianoSynth

**Fyzikální model** (diferencovatelný PyTorch, bez smyček — plně vektorizovaný):

```
f_k = k * f0 * sqrt(1 + B*k²)                       # inharmonicita (Simionato Eq.5)
env_k(t) = A0_k * (a1_k*exp(-t/τ1_k) + (1-a1_k)*exp(-t/τ2_k))   # bi-exp. útlum
beat_k(t) = beat_depth_k * cos(2π*beat_hz_k*t)       # beating detuned string
osc_k(t) = cos(2π*f_k*t + φ_k) + beat_k(t)*cos(2π*(f_k+beat_hz_k)*t + φ_k')
harmonic(t) = Σ_k env_k(t) * osc_k(t)
noise(t) = shaped_noise(centroid, slope) * exp(-t/attack_tau) * floor_rms_env
audio(t) = harmonic(t) + noise(t)
```

**Frame-based synthesis** (Simionato insight — lepší gradient flow):
- Každý "frame" = 240 vzorků @ 48kHz = 5ms
- Envelope interpolována per-frame (lineárně), fáze akumulována pro kontinuitu
- Umožňuje budoucí rozšíření na LSTM decay (jako Simionato)

**Stereo**: na základě `stereo_width_factor` z spectral_eq — Mid/Side syntéza.

**Spektrální EQ**: konvoluce výstupu s EQ křivkou (z `spectral_eq.freqs_hz/gains_db`).
Implementace: zero-phase FIR z interpolované EQ křivky, F.conv1d.

**Memory**: pro 88 not × 8 vel × 2s @ 48kHz = batch synth v segmentech (ne celý tón).

---

### `losses/piano_loss.py` — PianoLoss

```python
L_total = 1.0  * L_mrstft(synth, orig)            # [256, 1024, 4096, 16384]
        + 0.2  * L_l1_time(synth, orig)            # time-domain stabilita
        + 0.3  * L_mrstft_attack(synth, orig)      # attack-weighted MRSTFT
        + 0.05 * L_f0_logmse(params['B'], f0)      # F0 anchor (Simionato Eq.6)
        + 0.05 * L_rms_envelope(synth, orig)       # RMS per-frame (Simionato)
        + 0.02 * L_physics_reg(params)             # fyzikální konzistence
```

**Attack weight** (z ddsp-base, adaptováno):
```python
w(t) = 1.0 + alpha * max(0, d/dt[RMS(t)])   # 5× emphasis na rising loudness
alpha = 4.0, Gaussian smooth σ=2 frames
```

**F0 logMSE** (Simionato Eq.6):
```python
f1_pred = f0 * sqrt(1 + B * 1²)
L_f0 = MSE(log2(f1_pred + 1), log2(f1_target + 1))
```

**Physics regularization** (soft constraints):
```python
L_phys = relu(-B).mean()                          # B ≥ 0
       + relu(B - 0.4).mean()                     # B ≤ 0.4 (fyzikální max)
       + relu(tau1 - tau2).mean()                 # τ1 < τ2
       + relu(tau2[:, 1:] - tau2[:, :-1]).mean()  # τ2 klesá s k
       + relu(beat_hz - 8.0).mean()               # beat ≤ 8 Hz
       + relu(0.05 - beat_hz).mean()              # beat ≥ 0.05 Hz
```

**MRSTFT** — stejná implementace jako v stávajícím `losses.py`, ale
pro 4 škály (256, 1024, 4096, 16384) místo 3 — lepší pokrytí od transientů po bas.

---

### `data/piano_dataset.py` — PianoDataset

```python
class PianoDataset(Dataset):
    """
    Každý vzorek: (midi, vel, audio_segment, params_extracted)

    audio_segment: 2s náhodné okno z WAV souboru
    params_extracted: z params.json (pro warm-start loss v fázi 0)

    Oversampling: útok (prvních 500ms) 3× pravděpodobnější
    → zajišťuje dostatek gradientů pro tvarování envelopy
    """

    def __init__(self, bank_dir, params_path, segment_duration=2.0, sr=48000):
        ...

    def __getitem__(self, idx):
        # Načti WAV, náhodný offset (s bias k útoku)
        # Normalize do [-1, 1]
        # Vrať: (midi, vel_norm, audio_seg, params_dict)
```

---

### `train_e2e.py` — End-to-End Training

```
━━━ Fáze 0: Warm-start (supervised parametrická) ━━━━━━━━━━━
  Model: SetterNN (standalone, bez Synth)
  Loss:  L_warm = MSE(params_pred, params_extracted)
         Separátní MSE pro B, tau, A0, beat (různé škály!)
  Optimalizér: Adam lr=1e-3
  Epochy: 100
  Trvání: ~5 minut

  Simionato insight: nejdřív jen B_net (F0 loss), pak zbytek.
  Implementace: první 20 epoch trénujeme pouze B_net, pak unlock.

━━━ Fáze 1: Spektrální audio loss ━━━━━━━━━━━━━━━━━━━━━━━━━
  Model: SetterNN → DiffSynth (end-to-end)
  Loss:  MRSTFT + L1 + RMS_envelope
  Optimalizér: Adam lr=1e-4, gradient clip 1.0
  Epochy: 300
  Batch: 8 segmentů (2s každý, náhodné noty)
  Scheduler: CosineAnnealing 1e-4 → 1e-5

  Tréninková smyčka:
    params = setter_nn(midi_norm, vel_norm)
    audio  = diff_synth(params, f0, vel, segment_duration)
    loss   = piano_loss_phase1(audio_synth, audio_orig)

━━━ Fáze 2: Plná loss ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Loss:  + attack-weighted MRSTFT + F0_logMSE + physics_reg
  Optimalizér: Adam lr=3e-5 → 1e-6 cosine
  Epochy: 500
  Batch: 16 segmentů

Checkpointy: každých 50 epoch → checkpoints/e2e_epoch_{N}.pt
Best model: checkpoints/e2e_best.pt (podle val loss)
```

---

### `config_e2e.json` — konfigurace

```json
{
  "sample_rate": 48000,
  "n_partials": 64,
  "frame_size": 240,
  "setter_nn": {
    "hidden_dim": 256,
    "depth": 3
  },
  "training": {
    "phase0_epochs": 100,
    "phase1_epochs": 300,
    "phase2_epochs": 500,
    "batch_size": 8,
    "segment_duration": 2.0,
    "lr_phase0": 1e-3,
    "lr_phase1": 1e-4,
    "lr_phase2": 3e-5,
    "grad_clip": 1.0
  },
  "loss": {
    "w_mrstft": 1.0,
    "w_l1": 0.2,
    "w_attack": 0.3,
    "w_f0": 0.05,
    "w_rms": 0.05,
    "w_physics": 0.02,
    "mrstft_ffts": [256, 1024, 4096, 16384],
    "attack_alpha": 4.0
  }
}
```

---

## Implementační plán (pořadí kroků)

```
Krok 1: config_e2e.json
Krok 2: models/setter_nn.py  (SetterNN + warm_start_from_params)
Krok 3: models/diff_synth.py (DifferentiablePianoSynth, frame-based)
Krok 4: losses/piano_loss.py (PianoLoss)
Krok 5: data/piano_dataset.py (PianoDataset)
Krok 6: train_e2e.py          (3-fázový trénink)
Krok 7: Integrace do GUI pipeline (nový krok "e2e_train")
Krok 8: Smoke test na 10 notách (rychlá validace)
Krok 9: Plný trénink na Salamander / ks-grand
Krok 10: Aktualizace infer.py pro nový model
```

---

## Očekávané výhody vs EGRB

| Aspekt | EGRB | E2E Setter NN |
|---|---|---|
| Per-nota specializace | Ne (globální params) | Ano (NN per midi/vel) |
| Velocity modeling | Lineární brightness | Nelineární (naučená) |
| Optimalizační cíl | Shoda s parametry | Shoda zvuku |
| Beating přesnost | Průměr přes noty | Per-nota, audio-optimalizovaná |
| Noise model | Fixed šablona | Audio-optimalizovaný |
| Cold-start | Problém | Vyřešen warm-startem |
| Interpolace | Mimo rozsah | Přirozená (NN smooth manifold) |

---

## Reference

- Simionato, Fasciani, Holm (2024): "Physics-informed differentiable method for piano modeling"
  → dvě-fázový trénink, logMSE F0 loss, RMS envelope loss, frame-based synthesis
- ddsp-base (alchy/ddsp-base): attack-weighted MRSTFT, 4-scale STFT
- Bank & Chabassier (2019): bi-exponenciální útlum, inharmonicita B, beating fyzika
