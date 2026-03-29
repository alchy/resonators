# Roadmap

## Done

### Physics extraction
- [x] Inharmonicity coefficient B via FFT peak fitting
- [x] Bi-exponential decay (τ1, τ2, a1) via frequency-adaptive STFT window
- [x] Multi-string beating Δf via Hilbert envelope modulation
- [x] Attack noise model (τ_noise, centroid_hz, A_noise)
- [x] M/S stereo width factor extraction
- [x] LTASE spectral EQ per note (H(f) = orig/synth)
- [x] Cross-velocity tau smoothing (`smooth_tau_cross_velocity.py`)
- [x] Spline interpolation for sparse banks (`interpolate_missing_notes.py`)

### NN profile
- [x] Factorised MLP: B, τ, A0, Δf, noise, biexp, EQ, width — separate sub-nets
- [x] Sinusoidal MIDI embedding
- [x] Physical constraints: tau_ratio ≤ 0, τ2 > τ1
- [x] Log-space MSE (geometric error)
- [x] Preserve original measured values by default (`--no-preserve-orig` flag to override)
- [x] Velocity profile derived from A0 energies (replaces gamma curve)
- [x] Missing velocity layer interpolation

### Synthesis
- [x] Additive synthesis with inharmonicity (f_k = k·f0·√(1+B·k²))
- [x] Bi-exponential envelope per partial
- [x] Multi-string beating via independent oscillators (not AM approximation)
- [x] Per-string panning + M/S stereo width
- [x] Schroeder all-pass decorrelation (register-dependent)
- [x] Spectral EQ applied in frequency domain
- [x] Velocity color blending across layers
- [x] Per-note parameter overrides (delta system)
- [x] Attack noise injection
- [x] Onset ramp (eliminates phase-click)

### GUI
- [x] FastAPI backend + vanilla JS SPA
- [x] Full pipeline panel: extract → EQ → train with live SSE log streaming
- [x] CMD preview for all pipeline steps and generate
- [x] Session = bank (1:1), auto-created from bank name
- [x] RENDER / TIMBRE / STEREO / PER-NOTE / VEL PROFILE parameter columns
- [x] Moog-style rotary knobs
- [x] Bank-suffix params naming: `params-{bank}.json`, `params-nn-profile-{bank}.json`
- [x] Generate params file selectable/overridable in GUI
- [x] Snapshot NN: timestamped archive of profile + config
- [x] LCD header: patch name, pipeline step progress, currently playing
- [x] PLAYER + Welch spectrum
- [x] EGRB status polling from `checkpoints/train.log`

---

## Known limitations

### Soundboard (parked)
`soundboard_strength` exists but defaults to 0 and should not be used.
Current synthetic modal IR (40 modes) causes band-pass distortion and amplitude modulation
("croaking") instead of adding body/warmth. The spectral EQ (`eq_strength`) covers
the body's spectral shaping without IR artefacts.

**Fix:** Measure a real IR (pistol shot or sweep) from the physical instrument.
Alternatively: deconvolve piano recording from direct-injected signal.

### Bass EQ resolution
LTASE EQ below ~E2 (MIDI < 40) has less than 2 bins per 1/6-octave smoothing window —
effectively no smoothing. The `eq_freq_min = 400 Hz` default mitigates this by fading
EQ to flat below that frequency.

### Workers per step in full pipeline
"Full Pipeline" sends a single `workers` value for all three steps. Individual per-step
worker fields only apply when running steps individually.

---

## Possible future work

### Sound quality
- [ ] Real soundboard IR convolution (measured impulse response)
- [ ] Sympathetic string resonance model (damper pedal simulation)
- [ ] Una corda (soft pedal): reduce string count, shift spectral centroid
- [ ] Release sample layer (key-up noise)
- [ ] Hammer hardness variation model across velocity layers

### Multi-instrument
- [ ] Latent space interpolation between two banks (Steinway ↔ Bösendorfer)
- [ ] Instrument embedding vector added to all sub-networks
- [ ] Training on multiple banks simultaneously

### NN improvements
- [ ] EGRB (end-to-end differentiable training from WAV)
  — currently tracked via `checkpoints/train.log`
- [ ] Adaptive N_FFT in compute-spectral-eq.py (matches extract-params.py)
- [ ] Uncertainty-weighted loss (low SNR notes contribute less)

### Workflow
- [ ] Multi-bank session support (for comparison / AB testing)
- [ ] Export directly to SFZ or Kontakt format
- [ ] Web-based spectrum comparison: original WAV vs. synthesized overlay
- [ ] Automatic quality scoring per note (spectral distance, beating fidelity)

### Real-time
- See [REALTIME.md](REALTIME.md) for C++ concept and latency estimates.
