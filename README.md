# Ithaca Resonator Synth

Physics-informed grand piano synthesizer. Extracts physical parameters (inharmonicity, decay, beating, noise) from a real WAV sample bank, smooths them with a small neural network, and renders a full 88 × 8 WAV sample bank via additive synthesis.

Output is compatible with Ithaca Player.

---

## What it does

```
WAV bank (88 notes × 8 velocities)
    │
    ▼  extract-params.py     — inharmonicity B, decay τ1/τ2, beating Δf, noise model
    ▼  compute-spectral-eq.py — LTASE body EQ H(f) per note
    ▼  train-instrument-profile.py — NN smoothing across MIDI range
    │
    ▼
params-nn-profile-{bank}.json
    │
    ▼  GUI / generate-samples.py
    │
    ▼
m021-vel0-f44.wav … m108-vel7-f44.wav  +  instrument-definition.json
```

## Install

```bash
git clone ...
cd resonators
python -m venv .venv312
.venv312/Scripts/pip install -r requirements.txt
```

## Quick start

```bash
# Run GUI (port 8989)
.venv312/Scripts/python gui/server.py
# open http://localhost:8989

# Or run pipeline from CLI:
.venv312/Scripts/python -u analysis/extract-params.py \
    --bank C:/SoundBanks/IthacaPlayer/ks-grand \
    --out  analysis/params-ks-grand.json

.venv312/Scripts/python -u analysis/compute-spectral-eq.py \
    --params analysis/params-ks-grand.json \
    --bank   C:/SoundBanks/IthacaPlayer/ks-grand

.venv312/Scripts/python -u analysis/train-instrument-profile.py \
    --in  analysis/params-ks-grand.json \
    --out analysis/params-nn-profile-ks-grand.json \
    --epochs 800

.venv312/Scripts/python -u analysis/generate-samples.py \
    --params  analysis/params-nn-profile-ks-grand.json \
    --session gui/sessions/ks-grand/config.json \
    --out-dir gui/sessions/ks-grand/generated
```

## Source data

WAV bank: `C:/SoundBanks/IthacaPlayer/ks-grand/` — 704 files, 88 MIDI × 8 velocity @ 44.1 kHz.
File naming: `m{midi:03d}-vel{v}-f{sr_code}.wav` (e.g. `m060-vel3-f44.wav`).

## Documentation

| Doc | Contents |
|-----|----------|
| [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) | Physics model, NN design, extraction algorithms, paper references |
| [docs/GUI.md](docs/GUI.md) | FastAPI endpoints, frontend panels, session/config structure |
| [docs/ROADMAP.md](docs/ROADMAP.md) | Completed work, known limitations, future directions |
| [docs/REALTIME.md](docs/REALTIME.md) | C++ real-time synthesis concept, latency estimates |
