"""
data/dataset.py
PyTorch Dataset for EGRB training.

Each sample is a clip of clip_frames consecutive frames drawn from one NPZ file.
The sampler prefers clips that start near the attack transient (70% of draws),
so the model sees plenty of the hardest part of the envelope.
"""

import os
import json
import random
import numpy as np
import torch
from torch.utils.data import Dataset


class EGRBDataset(Dataset):
    PHASE_ATTACK = 0

    def __init__(self, manifest_path: str, cfg: dict, split: str = 'train'):
        with open(manifest_path) as f:
            manifest = json.load(f)

        val_split = float(cfg['training'].get('val_split', 0.1))
        n_val     = max(1, int(len(manifest) * val_split))

        if split == 'train':
            self.entries = manifest[n_val:]
        else:
            self.entries = manifest[:n_val]

        self.clip_frames = int(cfg['training']['clip_frames'])
        self.frame_size  = int(cfg['frame_size'])
        self._cache: dict = {}

    # ------------------------------------------------------------------
    def _load(self, file_idx: int) -> dict:
        if file_idx not in self._cache:
            entry = self.entries[file_idx]
            d = np.load(entry['npz'])
            self._cache[file_idx] = {
                'audio':    d['audio'],               # (2, T)
                'rms':      d['rms'].astype(np.float32),
                'phases':   d['phases'].astype(np.int64),
                'f0':       float(d['f0']),
                'vel_norm': float(d['vel_norm']),
            }
        return self._cache[file_idx]

    # ------------------------------------------------------------------
    def __len__(self) -> int:
        # multiple random crops per file so that small dataset feels larger
        return len(self.entries) * 20

    def __getitem__(self, idx: int) -> dict:
        file_idx = idx % len(self.entries)
        d        = self._load(file_idx)

        n_frames    = d['rms'].shape[0]
        clip_frames = min(self.clip_frames, n_frames)

        # Prefer attack-anchored clips
        attack_frames = np.where(d['phases'] == self.PHASE_ATTACK)[0]
        if len(attack_frames) > 0 and random.random() < 0.7:
            atk_first = int(attack_frames[0])
            start = max(0, atk_first - random.randint(0, 3))
        else:
            start = random.randint(0, max(0, n_frames - clip_frames))

        end           = min(start + clip_frames, n_frames)
        actual_frames = end - start

        audio  = d['audio'][:, start * self.frame_size : end * self.frame_size]
        rms    = d['rms'][start:end]
        phases = d['phases'][start:end]

        # Zero-pad on the right if clip shorter than clip_frames
        if actual_frames < clip_frames:
            pad = clip_frames - actual_frames
            audio  = np.pad(audio,  ((0, 0), (0, pad * self.frame_size)))
            rms    = np.pad(rms,    (0, pad), constant_values=0.0)
            phases = np.pad(phases, (0, pad), constant_values=3)  # release

        return {
            'audio':    torch.from_numpy(audio).float(),            # (2, T_clip)
            'rms':      torch.from_numpy(rms).float(),              # (clip_frames,)
            'phases':   torch.from_numpy(phases),                   # (clip_frames,) int64
            'f0':       torch.tensor(d['f0'],       dtype=torch.float32),
            'vel_norm': torch.tensor(d['vel_norm'], dtype=torch.float32),
        }
