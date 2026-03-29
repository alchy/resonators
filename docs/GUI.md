# GUI Documentation

FastAPI backend + vanilla JS SPA. Run on port **8989**.

```bash
.venv312/Scripts/python gui/server.py
# or: uvicorn gui.server:app --reload --port 8989
```

---

## Frontend panels

The interface is a horizontally scrollable rack of 12 columns.

| Column | Panel | Description |
|--------|-------|-------------|
| 1 | WAV SOURCE | Bank directory, pipeline controls, Snapshot NN |
| 2 | EXTRACT PARAMS | Pipeline step 1: run extract-params.py |
| 3 | SPECTRAL EQ | Pipeline step 2: run compute-spectral-eq.py |
| 4 | TRAIN PROFILE | Pipeline step 3: run train-instrument-profile.py |
| 5 | RENDER | sr, duration, fade_out, target_rms, onset_ms |
| 6 | TIMBRE | brightness, beat_scale, EQ, noise, color blend |
| 7 | STEREO | pan_spread, stereo_boost, stereo_decorr |
| 8 | PER-NOTE | Per-MIDI delta overrides |
| 9 | VEL PROFILE | Velocity RMS ratios (derived from A0 energies) |
| 10 | GENERATE | MIDI range, velocity layers, params file, CMD preview |
| 11 | PLAYER | Audio playback + Welch spectrum |
| 12 | GENERATED FILES | List of generated WAV files |

### LCD header (always visible, viewport-fixed)

- Left column: patch name, currently playing file
- Right column: pipeline step progress bars (EXTRACT / EQ / TRAIN / EGRB)

### Session workflow

1. Set **Bank dir** in WAV SOURCE (e.g. `C:/SoundBanks/IthacaPlayer/ks-grand`)
2. Click **▶ Use Bank** — creates session `ks-grand` if not exists, selects it
3. Click **▶▶ Full Pipeline** — runs extract → EQ → train in sequence
4. Click **→ Apply to Session** — sets `source_params` to trained profile
5. Click **▶ Generate** — renders WAV bank into `gui/sessions/ks-grand/generated/`
6. Click **💾 Snapshot NN** — archives current profile + config with timestamp

### CMD preview

Each pipeline step (EXTRACT, EQ, TRAIN) and GENERATE show the exact CLI command
that will be / was executed. Updates live as parameters change.

---

## Session config structure

`gui/sessions/{bank}/config.json`

```json
{
  "source_params": "analysis/params-nn-profile-ks-grand.json",
  "render": {
    "sr": 44100,
    "duration": null,
    "fade_out": 0.5,
    "target_rms": 0.06,
    "velocity_curve_gamma": 0.7,
    "onset_ms": 3.0
  },
  "timbre": {
    "harmonic_brightness": 1.0,
    "beat_scale": 1.0,
    "eq_strength": 0.5,
    "eq_freq_min": 400.0,
    "soundboard_strength": 0.0,
    "noise_level": 1.0,
    "vel_color_blend": 0.7,
    "vel_color_ref": 4
  },
  "stereo": {
    "pan_spread": 0.55,
    "stereo_boost": 1.0,
    "stereo_decorr": 1.0
  },
  "per_note": {},
  "velocity_rms_profile": {
    "0": 0.2333, "1": 0.3789, "2": 0.5033, "3": 0.6156,
    "4": 0.7196, "5": 0.8176, "6": 0.9108, "7": 1.0
  },
  "instrument_meta": {
    "instrumentName": "ks-grand",
    "author": "n/a",
    "category": "Piano",
    "instrumentVersion": "1",
    "description": "n/a",
    "velocityMaps": "8",
    "sampleCount": 0
  }
}
```

`source_params` — path to the params JSON used for synthesis. Set automatically by
**Apply to Session** to `analysis/params-nn-profile-{bank}.json`. Can be overridden
manually in the GENERATE panel params field.

`velocity_rms_profile` — relative RMS per velocity layer, normalised so `"7": 1.0`.
Derived from extracted A0 energies after pipeline. Overrides `velocity_curve_gamma`
when present.

`per_note` — map of `{midi: {param_delta: value, ...}}`. Applied additively on top
of global timbre/stereo values for that specific MIDI note.

---

## FastAPI endpoints

### Sessions  `prefix: /api/sessions`

| Method | Path | Description |
|--------|------|-------------|
| GET | `/` | List sessions (name, source_params, n_generated) |
| POST | `/` | Create session `{name}` — sets source_params to `analysis/params-nn-profile-{name}.json` |
| DELETE | `/{name}` | Delete session + all generated files |
| GET | `/{name}/config` | Get config + param metadata |
| PUT | `/{name}/config` | Update render/timbre/stereo/per_note/vel_profile |
| GET | `/{name}/note/{midi}` | Per-note overrides + resolved params |
| PUT | `/{name}/note/{midi}` | Set per-note overrides `{overrides: {...}}` |
| DELETE | `/{name}/note/{midi}` | Clear per-note overrides |
| GET | `/{name}/params` | List all notes in session params.json |

### Generate  `prefix: /api/sessions`

| Method | Path | Description |
|--------|------|-------------|
| POST | `/{name}/generate` | Start generation job `{midi_from, midi_to, vel_layers, params_file}` |
| GET | `/{name}/generate/status` | Poll progress (done, total, progress_pct, last_file) |
| POST | `/{name}/generate/cancel` | Cancel running job |
| GET | `/{name}/files` | List generated WAV files |

`params_file` in GenerateRequest — optional override for `source_params`. Empty string = use session default.

### Pipeline  `prefix: /api/pipeline`

| Method | Path | Description |
|--------|------|-------------|
| POST | `/run` | Start pipeline `{wav_dir, params_out, out, epochs, lr, hidden, workers, no_preserve, from_step}` |
| GET | `/status` | Poll pipeline state (step, step progress, log tail) |
| POST | `/cancel` | Interrupt current step |
| GET | `/log/{step}` | Last N lines from `runtime-logs/{step}-log.txt` |
| GET | `/log-stream/{step}` | SSE: live tail of log file (EventSource) |
| GET | `/egrb_status` | EGRB training state from `checkpoints/train.log` |
| GET | `/vel_profile` | Compute velocity profile from `analysis/params-{bank}.json` |
| POST | `/apply/{session}` | Point session `source_params` at trained profile, update vel_profile |
| POST | `/snapshot/{session}` | Save timestamped copy to `snapshots/{session}-{YYYYMMDD-HHMM}/` |

**Pipeline PipelineRequest fields:**

| Field | Default | Description |
|-------|---------|-------------|
| `wav_dir` | `C:/SoundBanks/IthacaPlayer/ks-grand` | WAV bank directory |
| `params_out` | `analysis/params.json` | Derived by frontend: `analysis/params-{bank}.json` |
| `out` | `analysis/params-nn-profile.json` | Derived: `analysis/params-nn-profile-{bank}.json` |
| `epochs` | 800 | Training epochs |
| `lr` | 0.003 | Learning rate |
| `hidden` | 64 | MLP hidden size |
| `workers` | 4 | Parallel workers for extract/EQ |
| `no_preserve` | false | If true: replace all samples with NN output |
| `from_step` | `extract` | Start from: `extract` \| `eq` \| `train` |

### Profile  `prefix: /api`

| Method | Path | Description |
|--------|------|-------------|
| GET | `/profile/list` | List available params JSON files in `analysis/` |
| GET | `/profile/models` | List `.pt` model files |

### Audio  `prefix: /api/sessions`

| Method | Path | Description |
|--------|------|-------------|
| GET | `/{name}/audio/{filename}` | Stream WAV file |
| GET | `/{name}/spectrum/{filename}` | Welch PSD for spectrum display |

### Static

| Mount | Source | Description |
|-------|--------|-------------|
| `/audio/{session}/generated/*` | `gui/sessions/` | Generated WAV files |
| `/` | `gui/static/` | Frontend (index.html, app.js, style.css) |

---

## Logging

| File | Contents |
|------|----------|
| `gui/logs/server.log` | FastAPI + uvicorn internal logs (plain FileHandler — RotatingFileHandler causes WinError 32 on Windows with reload mode) |
| `runtime-logs/extract-params-log.txt` | stdout tee from extract-params.py (auto-created on first run) |
| `runtime-logs/spectral-eq-log.txt` | stdout tee from compute-spectral-eq.py |
| `runtime-logs/train-profile-log.txt` | stdout tee from train-instrument-profile.py |

SSE endpoint `/api/pipeline/log-stream/{step}` tails `runtime-logs/` files and pushes
new lines as JSON-encoded arrays: `{"replace": bool, "lines": [...]}`.
