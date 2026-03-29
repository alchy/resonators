"""
data/piano_dataset.py
──────────────────────
PianoDataset: loads WAV files + params.json for end-to-end piano training.

Each item returns:
    midi        int     MIDI note number
    vel_norm    float   velocity normalised [0, 1]
    f0          float   fundamental frequency Hz
    audio       (2, n_seg_samples)  stereo segment, float32, normalised [-1, 1]
    params      dict of tensors     extracted params for warm-start loss
    width_factor float  stereo width factor from spectral_eq (or 1.0 if absent)

Segment sampling strategy:
    - Segment duration: 2 s (configurable)
    - Attack bias: first 500 ms is 3× more likely to be the segment start.
      Ensures the model receives adequate gradient signal on transients.
    - Segments that are mostly silence (RMS < 0.001) are resampled.

Sample-rate handling:
    - WAV files may be 44.1 kHz or 48 kHz (Salamander = 48 kHz).
    - Resampled to cfg['sample_rate'] on load if needed (torchaudio).
"""

import json
import math
import random
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

try:
    import soundfile as sf
    _HAS_SF = True
except ImportError:
    _HAS_SF = False

try:
    import torchaudio
    _HAS_TA = True
except ImportError:
    _HAS_TA = False

from models.setter_nn import (
    B_MIN, B_MAX, TAU1_MIN, TAU1_MAX, TAU2_MIN, TAU2_MAX, BEAT_MIN, BEAT_MAX
)


# ── WAV loader ────────────────────────────────────────────────────────────────

def _load_wav(path: str, target_sr: int) -> Optional[torch.Tensor]:
    """
    Load WAV, convert to stereo float32 tensor (2, n_samples), resample if needed.
    Returns None on error.
    """
    try:
        if _HAS_SF:
            import soundfile as sf
            data, sr = sf.read(str(path), dtype='float32', always_2d=True)
            audio = torch.from_numpy(data.T)   # (C, N)
        elif _HAS_TA:
            audio, sr = torchaudio.load(str(path))
        else:
            raise RuntimeError("Neither soundfile nor torchaudio is available")

        # Mono → stereo
        if audio.shape[0] == 1:
            audio = audio.repeat(2, 1)
        elif audio.shape[0] > 2:
            audio = audio[:2]

        # Resample if needed
        if sr != target_sr and _HAS_TA:
            audio = torchaudio.functional.resample(audio, sr, target_sr)

        return audio.float()

    except Exception as e:
        print(f"[PianoDataset] WARNING: could not load {path}: {e}")
        return None


# ── Per-sample params builder ─────────────────────────────────────────────────

def _build_params_tensor(sample: dict, K: int) -> dict:
    """
    Convert one params.json sample entry to tensors of shape (K,).
    Missing partials filled with physically sensible defaults.
    """
    partials = sample.get('partials', [])

    A0_vec   = [1e-6]  * K
    tau1_vec = [0.10]  * K
    tau2_vec = [3.00]  * K
    a1_vec   = [0.25]  * K
    bHz_vec  = [0.30]  * K
    bDep_vec = [0.05]  * K

    for p in partials:
        k_idx = int(p['k']) - 1
        if k_idx >= K:
            continue
        A0_vec[k_idx]   = max(float(p.get('A0',          1e-6)), 1e-6)
        tau1_vec[k_idx] = max(float(p.get('tau1',        0.10)), TAU1_MIN)
        tau2_vec[k_idx] = max(float(p.get('tau2',        3.00)), TAU2_MIN)
        a1_vec[k_idx]   = max(0.01, min(0.99, float(p.get('a1', 0.25))))
        bHz_vec[k_idx]  = max(float(p.get('beat_hz',     0.30)), BEAT_MIN)
        bDep_vec[k_idx] = max(0.001, min(0.499, float(p.get('beat_depth', 0.05))))

    # Normalise A0 (mean = 1.0)
    A0_t = torch.tensor(A0_vec, dtype=torch.float32)
    if A0_t.sum() > 0:
        A0_t = A0_t / (A0_t.mean() + 1e-8)

    B_val = max(float(sample.get('B', 1e-4)), B_MIN)

    return {
        'B':          torch.tensor([B_val],    dtype=torch.float32),
        'A0':         A0_t,
        'tau1':       torch.tensor(tau1_vec,   dtype=torch.float32),
        'tau2':       torch.tensor(tau2_vec,   dtype=torch.float32),
        'a1':         torch.tensor(a1_vec,     dtype=torch.float32),
        'beat_hz':    torch.tensor(bHz_vec,    dtype=torch.float32),
        'beat_depth': torch.tensor(bDep_vec,   dtype=torch.float32),
    }


# ── Dataset ───────────────────────────────────────────────────────────────────

class PianoDataset(Dataset):
    """
    Args:
        bank_dir         : path to WAV bank directory
        params_json      : path to params.json from extract-params.py
        segment_duration : segment length in seconds (default 2.0)
        target_sr        : output sample rate (default 48000)
        K                : number of partials (default 64)
        attack_bias      : oversample first `attack_window_s` seconds
                           (probability multiplier, default 3.0)
        attack_window_s  : duration of attack window for bias (default 0.5)
        min_rms          : reject silent segments below this RMS (default 0.001)
        max_retries      : max resampling attempts for silent segments (default 5)
    """

    def __init__(
        self,
        bank_dir:         str,
        params_json:      str,
        segment_duration: float = 2.0,
        target_sr:        int   = 48000,
        K:                int   = 64,
        attack_bias:      float = 3.0,
        attack_window_s:  float = 0.5,
        min_rms:          float = 0.001,
        max_retries:      int   = 5,
    ):
        self.bank_dir        = Path(bank_dir)
        self.segment_dur     = segment_duration
        self.sr              = target_sr
        self.K               = K
        self.attack_bias     = attack_bias
        self.attack_samps    = int(attack_window_s * target_sr)
        self.seg_samps       = int(segment_duration * target_sr)
        self.min_rms         = min_rms
        self.max_retries     = max_retries

        with open(params_json, 'r') as f:
            data = json.load(f)
        samples = data['samples']

        # Build index of valid samples (those with matching WAV)
        self.items: list[dict] = []
        for key, s in samples.items():
            midi = int(s['midi'])
            vel  = int(s['vel'])
            # Match glob pattern (e.g., m060-vel4-f48.wav)
            matches = sorted(self.bank_dir.glob(f'm{midi:03d}-vel{vel}-f*.wav'))
            if not matches:
                continue
            wav_path = matches[0]

            f0   = float(s.get('f0_fitted_hz') or s.get('f0_nominal_hz', 440.0))
            seq  = s.get('spectral_eq', {})
            wf   = float(seq.get('stereo_width_factor', 1.0))

            self.items.append({
                'key':          key,
                'midi':         midi,
                'vel':          vel,
                'vel_norm':     vel / 7.0,
                'f0':           f0,
                'wav_path':     str(wav_path),
                'params':       _build_params_tensor(s, K),
                'width_factor': wf,
            })

        print(f"[PianoDataset] {len(self.items)} valid samples in {bank_dir}")

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> dict:
        item = self.items[idx]

        audio = _load_wav(item['wav_path'], self.sr)
        if audio is None or audio.shape[1] < self.seg_samps:
            # Fallback: zero segment (will be filtered by min_rms)
            audio = torch.zeros(2, self.seg_samps)

        # ── Segment sampling with attack bias ─────────────────────────
        n_total = audio.shape[1]
        seg_audio = self._sample_segment(audio, n_total)

        # ── Normalise to [-1, 1] (peak normalise per segment) ─────────
        peak = seg_audio.abs().max().clamp(min=1e-6)
        seg_audio = seg_audio / peak

        return {
            'midi':         item['midi'],
            'vel_norm':     torch.tensor(item['vel_norm'], dtype=torch.float32),
            'f0':           torch.tensor(item['f0'],       dtype=torch.float32),
            'audio':        seg_audio,                       # (2, seg_samps)
            'params':       item['params'],                  # dict of (K,) tensors
            'width_factor': torch.tensor(item['width_factor'], dtype=torch.float32),
        }

    def _sample_segment(self, audio: torch.Tensor, n_total: int) -> torch.Tensor:
        """Sample a segment with attack-region bias, retry if silent."""
        seg = self.seg_samps
        if n_total <= seg:
            # Pad with zeros
            pad = seg - n_total
            return F.pad(audio, (0, pad))

        atk = min(self.attack_samps, n_total - seg)

        for _ in range(self.max_retries):
            # Attack bias: attack_bias times more likely to start in [0, atk]
            total_weight = atk * self.attack_bias + (n_total - seg - atk)
            if total_weight <= 0:
                start = 0
            elif random.random() < (atk * self.attack_bias / total_weight):
                start = random.randint(0, atk)
            else:
                start = random.randint(atk + 1, n_total - seg)

            seg_audio = audio[:, start: start + seg]

            # Reject silent segments
            rms = seg_audio.pow(2).mean().sqrt().item()
            if rms >= self.min_rms:
                return seg_audio

        # All retries failed — return the attack region regardless
        return audio[:, :seg]


# ── Collate function ──────────────────────────────────────────────────────────

def collate_fn(batch: list) -> dict:
    """
    Custom collate: stacks tensors, collects params sub-dicts.
    """
    keys_to_stack = ['vel_norm', 'f0', 'audio', 'width_factor']
    out = {k: torch.stack([item[k] for item in batch]) for k in keys_to_stack}
    out['midi'] = [item['midi'] for item in batch]

    # Params: stack each sub-tensor
    param_keys = batch[0]['params'].keys()
    out['params'] = {
        pk: torch.stack([item['params'][pk] for item in batch])
        for pk in param_keys
    }

    return out
