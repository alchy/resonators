# EGRB — Envelope-Gated Resonator Bank

Neural synthesiser combining a GRU controller with a bank of 64 differentiable sinusoidal
resonators. The control density is coupled to the envelope phase: dense during attack,
sparse during sustain/decay.

## Requirements

Python **3.12** (3.13 has known Keras compatibility issues).

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

## Workflow

```bash
# 1. Preprocess Salamander soundbank → NPZ cache + manifest.json
python data/prepare.py

# 2. Train (200 epochs, 3-phase curriculum)
python train.py

# 3. Generate full piano bank (m021–m108 × vel0–7)
python generate.py --checkpoint checkpoints/best.pt --output generated/
```

Resume training: `python train.py --resume checkpoints/last.pt`

Custom config: `python train.py --config my_config.json`

### Training output

```
Device: cpu
Parameters: 776,641  (controller=776,512  bank=129)
Epoch    0/200  [phase1]  train: mrstft=82.2053  l1=12.1844  kin=21.1057  eng=14.7685  sparse=0.4968  total=102.6621  ||  val: mrstft=64.7015  l1=11.9473  kin=3.5376  eng=14.0042  sparse=0.5178  total=77.7074  (1346s)
  ** new best  val_total=77.70742
```

| Field | Meaning |
|-------|---------|
| `Device` | Compute backend — `cpu`, `cuda` (NVIDIA GPU), or `mps` (Apple Silicon) |
| `Parameters` | Total trainable weights, broken down by submodule |
| `Epoch N/200` | Current epoch / total epochs |
| `[phase1/2/3]` | Active curriculum phase (see below) |
| `train:` / `val:` | Averaged losses over training / validation set for this epoch |
| `mrstft` | Multi-Resolution STFT loss — spectral fidelity across 256/1024/4096-point windows |
| `l1` | L1 waveform loss — sample-level accuracy |
| `kin` | Kinetics loss — penalises abrupt changes in control signals (1st + 2nd derivative) |
| `eng` | Energy loss — match target RMS envelope |
| `sparse` | Gate sparsity loss — encourages binary gates outside the attack phase |
| `total` | Weighted sum of all active losses |
| `(Ns)` | Wall-clock time for the epoch in seconds |
| `** new best` | Validation total improved; checkpoint saved to `checkpoints/best.pt` |

**Curriculum phases** — losses are enabled gradually to stabilise early training:

| Phase | Epochs (default) | Active losses |
|-------|-----------------|---------------|
| `phase1` | 0 – 49 | `mrstft`, `l1` |
| `phase2` | 50 – 99 | + `kin`, `eng` |
| `phase3` | 100 – 199 | + `sparse` (full composite loss) |

Two checkpoints are maintained throughout training:
- `checkpoints/last.pt` — overwritten every epoch (safe resume point)
- `checkpoints/best.pt` — only updated when `val_total` improves

## Architecture

```
GRUController  (f0, vel, RMS, phase_label) → (Δf, ΔA, gate) per resonator per frame
ResonatorBank  64 resonators:
  0–47   harmonic  f_i = i·f0·√(1 + inh·i²)   slow decay
  48–55  noise     log-spaced 200–10 kHz        random phase jitter
  56–63  transient f_i = f0·[2..8]              fast decay, attack-only
```

## Loss functions

| Loss | Weight | Purpose |
|------|--------|---------|
| MRSTFT (256/1024/4096) | 1.0 | Spectral fidelity |
| L1 waveform | 0.5 | Sample accuracy |
| Kinetics (1st+2nd deriv) | 0.4 | Smooth transitions |
| Energy RMS | 0.4 | Envelope dynamics |
| Gate sparsity | 0.03 | Sparse control outside attack |

Attack frames are weighted **5×** in all losses.

## Data

Source: `C:/SoundBanks/ddsp/salamander/`  — 240 WAV files, 30 MIDI notes × 8 velocities @ 48 kHz.
