"""Training and dev data from the audited manifest."""
from __future__ import annotations

from collections import defaultdict
import csv
import json

import numpy as np
import torch
from torch.utils.data import Dataset, Sampler

from detector import paths
from detector.audio import SR, decode, prepare
from detector.augment import apply_codec, augment
from detector.data.sources import read_bytes

WINDOW_S = 4.0


def load_manifest(roles):
    report = json.loads((paths.AUDIT / "report.json").read_text())
    if not report["passed"]:
        raise SystemExit("The audit gate failed; fix the data before training.")
    with (paths.AUDIT / "manifest.csv").open() as f:
        rows = [r for r in csv.DictReader(f) if not r["excluded"] and r["role"] in roles]
    for r in rows:
        r["label"] = int(r["label"])
    return rows


def load_prepared(row):
    wave, _ = decode(read_bytes(row["locator"]))
    return prepare(wave)


def crop(wave, rng, size=int(WINDOW_S * SR)):
    """Random window; short recordings are repeated, never zero padded."""
    if len(wave) == 0:
        return np.zeros(size, np.float32)
    if len(wave) < size:
        wave = np.tile(wave, int(np.ceil(size / len(wave))))
    start = rng.integers(0, len(wave) - size + 1)
    return wave[start:start + size]


class TrainWindows(Dataset):
    """Item = (index, epoch-specific seed). Returns one augmented 4 s window."""
    def __init__(self, rows, augment_p=1.0):
        self.rows = rows
        self.augment_p = augment_p

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, item):
        index, seed = item
        rng = np.random.default_rng(seed)
        row = self.rows[index]
        try:
            window = crop(load_prepared(row), rng)
        except Exception:
            window = np.zeros(int(WINDOW_S * SR), np.float32)  # counted as a load failure below
            return torch.from_numpy(window), torch.tensor(float(row["label"])), -1
        if rng.random() < self.augment_p:
            window, _ = augment(window, rng)
        return torch.from_numpy(np.ascontiguousarray(window)), torch.tensor(float(row["label"])), index


class BalancedCellSampler(Sampler):
    """Each draw picks a (source, label) cell uniformly, then a recording in it.

    Sources have very different fake fractions (ASVspoof 5 is ~90% spoof), so
    uniform sampling over recordings would let "which corpus" stand in for
    "is it fake". Equal cell weights remove that prior."""
    def __init__(self, rows, steps, batch_size, seed):
        self.cells = defaultdict(list)
        for i, r in enumerate(rows):
            self.cells[(r["source"], r["label"])].append(i)
        self.keys = sorted(self.cells)
        self.n = steps * batch_size
        self.seed = seed
        self.epoch = 0

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __len__(self):
        return self.n

    def __iter__(self):
        rng = np.random.default_rng([self.seed, self.epoch])
        cells = rng.integers(0, len(self.keys), self.n)
        for c, item_seed in zip(cells, rng.integers(0, 2**63 - 1, self.n)):
            members = self.cells[self.keys[c]]
            yield members[rng.integers(len(members))], int(item_seed)


def dev_subset(rows, per_cell, seed=7):
    """Deterministic subsample: at most per_cell recordings per (source, label, attack)."""
    cells = defaultdict(list)
    for r in sorted(rows, key=lambda r: r["uid"]):
        cells[(r["source"], r["label"], r["attack"])].append(r)
    rng = np.random.default_rng(seed)
    out = []
    for key in sorted(cells):
        items = cells[key]
        idx = rng.permutation(len(items))[:per_cell]
        out.extend(items[i] for i in sorted(idx))
    return out


class DevWindows(Dataset):
    """Deterministic windows for evaluation: up to max_windows non-overlapping
    4 s windows per recording, under a fixed channel ("clean" or a codec name)."""
    def __init__(self, rows, channel="clean", max_windows=4):
        self.rows, self.channel, self.max_windows = rows, channel, max_windows

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        size = int(WINDOW_S * SR)
        try:
            wave, _ = decode(read_bytes(self.rows[index]["locator"]))
            if self.channel != "clean":
                # An upload arrives already encoded, so the codec is applied to the
                # decoded file before silence trimming and loudness normalisation.
                wave = apply_codec(wave, self.channel)
            wave = prepare(wave)
        except Exception:
            return torch.zeros(0, size), index
        if len(wave) < size:
            wave = np.tile(wave, int(np.ceil(size / max(len(wave), 1))))[:size]
        # Enough windows to cover the whole recording (the last one ends at the end),
        # capped at max_windows, so medium-length clips are not scored on their start only.
        n = min(self.max_windows, int(np.ceil(len(wave) / size)))
        starts = np.linspace(0, len(wave) - size, n).astype(int) if n > 1 else [0]
        return torch.from_numpy(np.stack([wave[s:s + size] for s in starts])), index


def collate_dev(batch):
    windows = [w for w, _ in batch]
    owners = [torch.full((len(w),), i, dtype=torch.long) for w, i in batch]
    return torch.cat(windows), torch.cat(owners)
