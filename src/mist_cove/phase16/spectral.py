"""Fixed, stage-blind, duration-matched spectral interval controls."""

import numpy as np


def atom_scores(waveform, sample_rate: int, *, atom_seconds: int = 2) -> dict[str, np.ndarray]:
    x = np.asarray(waveform, dtype=np.float64)
    if x.ndim != 1 or not x.size or not np.isfinite(x).all():
        raise ValueError("finite one-channel waveform required")
    if not 50 <= sample_rate <= 4096 or atom_seconds <= 0:
        raise ValueError("unsupported rate or atom duration")
    width = sample_rate * atom_seconds
    if x.size % width:
        raise ValueError("continuous signal must contain complete physical atoms")
    atoms = x.reshape(-1, width)
    centered = atoms - atoms.mean(axis=1, keepdims=True)
    taper = np.hanning(width)
    power = np.abs(np.fft.rfft(centered * taper, axis=1)) ** 2
    freqs = np.fft.rfftfreq(width, 1 / sample_rate)
    sigma = power[:, (freqs >= 12) & (freqs <= 15)].sum(axis=1)
    broad = power[:, (freqs >= 0.5) & (freqs <= min(45, sample_rate / 2))].sum(axis=1)
    return {"sigma_power": sigma, "relative_sigma": sigma / np.maximum(broad, 1e-12),
            "rms": np.sqrt(np.mean(centered ** 2, axis=1)),
            "peak_to_peak": np.ptp(atoms, axis=1)}


def top_atoms(scores: np.ndarray, count: int) -> list[int]:
    values = np.asarray(scores, dtype=float)
    if values.ndim != 1 or not np.isfinite(values).all() or not 0 <= count <= values.size:
        raise ValueError("invalid ranking or duration budget")
    return sorted(int(i) for i in np.argsort(-values, kind="stable")[:count])


def selected_intervals(indices: list[int], *, atom_seconds: int = 2) -> list[tuple[int, int]]:
    if len(set(indices)) != len(indices) or any(i < 0 for i in indices):
        raise ValueError("invalid atom selection")
    return [(i * atom_seconds, (i + 1) * atom_seconds) for i in sorted(indices)]
