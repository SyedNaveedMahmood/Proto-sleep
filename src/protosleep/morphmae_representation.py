from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from .attnsleep import MRCNN
from .config import EXPECTED_FS, EXPECTED_SAMPLES_PER_EPOCH, NUM_WORKERS, PIN_MEMORY
from .data import _canonicalize_epoch_shape
from .mist import extract_mrcnn_state_dict, load_mrcnn_checkpoint, sha256_file
from .morphmae_bridge import discover_npz_subject_files
from .utils import DEVICE, cpu_state_dict, seed_everything


BANDS: Tuple[Tuple[str, float, float], ...] = (
    ("delta", 0.5, 4.0),
    ("theta", 4.0, 8.0),
    ("alpha", 8.0, 12.0),
    ("sigma", 12.0, 16.0),
    ("beta", 16.0, 30.0),
)
TARGET_NAMES: Tuple[str, ...] = tuple(
    [f"full_logrel_{name}" for name, _, _ in BANDS]
    + [f"segment_logrel_std_{name}" for name, _, _ in BANDS]
    + [f"segment_logrel_max_{name}" for name, _, _ in BANDS]
    + ["normalized_line_length"]
)


@dataclass(frozen=True)
class RepresentationRefineRecipe:
    """Frozen exploratory recipe for improving stage-relevant MorphMAE representations."""

    name: str = "morphspec_r1"
    epochs: int = 30
    batch_size: int = 128
    patch_size: int = 25
    mask_ratio: float = 0.30
    encoder_lr: float = 2e-5
    head_lr: float = 1e-3
    encoder_weight_decay: float = 1e-4
    head_weight_decay: float = 1e-4
    target_segment_seconds: int = 5
    seed: int = 4242


MORPHSPEC_R1 = RepresentationRefineRecipe()


@dataclass
class UnlabeledRecording:
    path: str
    recording_id: str
    subject_id: int
    x: np.ndarray  # [T,1,3000]

    @property
    def n_epochs(self) -> int:
        return int(self.x.shape[0])


class UnlabeledEpochDataset(Dataset):
    def __init__(self, recordings: Sequence[UnlabeledRecording]):
        self.recordings = list(recordings)
        self.cum = np.cumsum([r.n_epochs for r in self.recordings], dtype=np.int64)
        self.total = int(self.cum[-1]) if len(self.cum) else 0

    def __len__(self) -> int:
        return self.total

    def __getitem__(self, idx: int) -> torch.Tensor:
        rec_i = int(np.searchsorted(self.cum, idx, side="right"))
        prev = 0 if rec_i == 0 else int(self.cum[rec_i - 1])
        ep_i = int(idx - prev)
        return torch.from_numpy(self.recordings[rec_i].x[ep_i])


def load_unlabeled_recordings(
    data_dir: Path | str,
    subject_ids: Sequence[int],
) -> Tuple[List[UnlabeledRecording], Dict[str, Any]]:
    """Open x (and optional fs) only for explicitly allowed subjects; y is never read."""

    requested = sorted({int(x) for x in subject_ids})
    if not requested:
        raise ValueError("subject_ids must not be empty")
    by_subject = discover_npz_subject_files(data_dir)
    missing = sorted(set(requested) - set(by_subject))
    if missing:
        raise RuntimeError(f"Missing requested subjects: {missing}")

    recordings: List[UnlabeledRecording] = []
    opened_files: List[str] = []
    for sid in requested:
        for path in by_subject[sid]:
            with np.load(path, allow_pickle=False) as d:
                if "x" not in d.files:
                    raise KeyError(f"{path.name}: expected x array")
                x = _canonicalize_epoch_shape(d["x"], path.name)
                if "fs" in d.files:
                    fs = int(round(float(np.asarray(d["fs"]).reshape(-1)[0])))
                else:
                    fs = int(x.shape[-1] // 30)
            if x.shape[-1] != EXPECTED_SAMPLES_PER_EPOCH:
                raise ValueError(
                    f"{path.name}: expected {EXPECTED_SAMPLES_PER_EPOCH} samples, got {x.shape[-1]}"
                )
            if fs != EXPECTED_FS:
                raise ValueError(f"{path.name}: expected {EXPECTED_FS} Hz, got {fs}")
            if not np.isfinite(x).all():
                raise ValueError(f"{path.name}: non-finite EEG")
            recordings.append(
                UnlabeledRecording(
                    path=str(path), recording_id=path.stem, subject_id=sid, x=x
                )
            )
            opened_files.append(str(path))

    actual = sorted({r.subject_id for r in recordings})
    if actual != requested:
        raise RuntimeError(f"Loaded subject mismatch: expected {requested}, got {actual}")
    return recordings, {
        "subjects": actual,
        "opened_files": opened_files,
        "n_recordings": len(recordings),
        "n_epochs": int(sum(r.n_epochs for r in recordings)),
        "stage_label_arrays_read": False,
    }


def make_unlabeled_loader(
    recordings: Sequence[UnlabeledRecording],
    batch_size: int,
    shuffle: bool,
    seed: int,
) -> DataLoader:
    generator = torch.Generator()
    generator.manual_seed(int(seed))
    return DataLoader(
        UnlabeledEpochDataset(recordings),
        batch_size=int(batch_size),
        shuffle=bool(shuffle),
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        drop_last=False,
        generator=generator,
    )


def _relative_band_power(signal: torch.Tensor, fs: int) -> torch.Tensor:
    """Return relative power in BANDS for signal [..., samples]."""

    n = int(signal.shape[-1])
    spec = torch.fft.rfft(signal.float(), dim=-1)
    power = spec.real.square() + spec.imag.square()
    freqs = torch.fft.rfftfreq(n, d=1.0 / float(fs), device=signal.device)
    total_mask = (freqs >= 0.5) & (freqs < 30.0)
    total = power[..., total_mask].sum(dim=-1, keepdim=True).clamp_min(1e-12)
    values = []
    for _, lo, hi in BANDS:
        mask = (freqs >= lo) & (freqs < hi)
        values.append(power[..., mask].sum(dim=-1, keepdim=True) / total)
    return torch.cat(values, dim=-1)


def morphology_targets(x: torch.Tensor, fs: int = EXPECTED_FS, segment_seconds: int = 5) -> torch.Tensor:
    """Label-free, amplitude-robust spectral/morphology targets for a 30-s epoch."""

    if x.ndim != 3 or x.shape[1] != 1:
        raise ValueError(f"Expected [B,1,L], got {tuple(x.shape)}")
    signal = x[:, 0].float()
    if signal.shape[-1] != EXPECTED_SAMPLES_PER_EPOCH:
        raise ValueError(f"Expected L={EXPECTED_SAMPLES_PER_EPOCH}, got {signal.shape[-1]}")

    full_rel = _relative_band_power(signal, fs).clamp_min(1e-6)
    full_log = torch.log(full_rel)

    seg_len = int(fs * segment_seconds)
    if seg_len <= 0 or signal.shape[-1] % seg_len != 0:
        raise ValueError("segment length must exactly divide the 30-s epoch")
    segments = signal.reshape(signal.shape[0], -1, seg_len)
    seg_rel = _relative_band_power(segments, fs).clamp_min(1e-6)
    seg_log = torch.log(seg_rel)
    seg_std = seg_log.std(dim=1, unbiased=False)
    seg_max = seg_log.max(dim=1).values

    centered = signal - signal.mean(dim=-1, keepdim=True)
    scale = centered.std(dim=-1, unbiased=False).clamp_min(1e-6)
    line = torch.mean(torch.abs(torch.diff(centered, dim=-1)), dim=-1) / scale
    line = line.unsqueeze(-1)

    target = torch.cat([full_log, seg_std, seg_max, line], dim=-1)
    if target.shape[-1] != len(TARGET_NAMES):
        raise RuntimeError(f"Target size mismatch: {target.shape[-1]} vs {len(TARGET_NAMES)}")
    if not torch.isfinite(target).all():
        raise RuntimeError("Non-finite morphology target")
    return target


def patch_mask_view(x: torch.Tensor, patch_size: int, mask_ratio: float) -> Tuple[torch.Tensor, torch.Tensor]:
    """Zero whole sample patches independently for each epoch."""

    if not 0.0 < float(mask_ratio) < 1.0:
        raise ValueError("mask_ratio must be in (0,1)")
    if x.shape[-1] % int(patch_size) != 0:
        raise ValueError("patch_size must divide the epoch length")
    b = int(x.shape[0])
    n_patches = int(x.shape[-1] // patch_size)
    n_mask = max(1, min(n_patches - 1, int(round(n_patches * float(mask_ratio)))))
    scores = torch.rand(b, n_patches, device=x.device)
    idx = scores.argsort(dim=1)[:, :n_mask]
    mask = torch.zeros(b, n_patches, dtype=torch.bool, device=x.device)
    mask.scatter_(1, idx, True)
    view = x.clone().reshape(b, x.shape[1], n_patches, patch_size)
    view = view.masked_fill(mask[:, None, :, None], 0.0)
    return view.reshape_as(x), mask


def summarize_mrcnn(afr: torch.Tensor) -> torch.Tensor:
    if afr.ndim != 3 or afr.shape[1] != 30:
        raise ValueError(f"Expected MRCNN [B,30,N], got {tuple(afr.shape)}")
    mean = afr.mean(dim=-1)
    std = afr.std(dim=-1, unbiased=False)
    return torch.cat([mean, std], dim=-1)


class MorphologyPredictionHead(nn.Module):
    def __init__(self, out_dim: int = len(TARGET_NAMES)):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(60),
            nn.Linear(60, 128),
            nn.GELU(),
            nn.Linear(128, out_dim),
        )

    def forward(self, summary: torch.Tensor) -> torch.Tensor:
        return self.net(summary)


@torch.no_grad()
def estimate_target_stats(
    loader: DataLoader,
    device: torch.device | str = DEVICE,
    segment_seconds: int = 5,
) -> Tuple[torch.Tensor, torch.Tensor]:
    vals = []
    device = torch.device(device)
    for x in loader:
        x = x.to(device=device, dtype=torch.float32, non_blocking=True)
        vals.append(morphology_targets(x, EXPECTED_FS, segment_seconds).cpu())
    all_targets = torch.cat(vals, dim=0)
    mean = all_targets.mean(dim=0)
    std = all_targets.std(dim=0, unbiased=False).clamp_min(1e-4)
    return mean, std


def refine_morphmae_encoder(
    source_checkpoint: Path | str,
    train_recordings: Sequence[UnlabeledRecording],
    output_checkpoint: Path | str,
    fold: int,
    train_subjects: Sequence[int],
    val_subjects: Sequence[int],
    test_subjects: Sequence[int],
    recipe: RepresentationRefineRecipe = MORPHSPEC_R1,
    seed: int | None = None,
) -> Dict[str, Any]:
    """Refine MorphMAE-v2 with label-free stage-relevant morphology targets."""

    source_checkpoint = Path(source_checkpoint).expanduser().resolve()
    output_checkpoint = Path(output_checkpoint).expanduser().resolve()
    run_seed = int(recipe.seed if seed is None else seed)
    seed_everything(run_seed)

    _, source_meta = extract_mrcnn_state_dict(source_checkpoint)
    declared = None
    for key in ("train_subjects", "pretrain_subjects", "subjects"):
        if key in source_meta:
            declared = sorted(int(v) for v in source_meta[key])
            break
    expected = sorted(int(v) for v in train_subjects)
    if declared != expected:
        raise RuntimeError(
            f"Source MorphMAE split mismatch: checkpoint={declared}, expected fold-{fold} train={expected}"
        )

    encoder = MRCNN(30)
    load_mrcnn_checkpoint(encoder, source_checkpoint)
    head = MorphologyPredictionHead()
    encoder = encoder.to(DEVICE)
    head = head.to(DEVICE)

    stats_loader = make_unlabeled_loader(
        train_recordings, recipe.batch_size, shuffle=False, seed=run_seed
    )
    target_mean, target_std = estimate_target_stats(
        stats_loader, DEVICE, recipe.target_segment_seconds
    )
    target_mean = target_mean.to(DEVICE)
    target_std = target_std.to(DEVICE)

    optimizer = torch.optim.AdamW(
        [
            {
                "params": encoder.parameters(),
                "lr": float(recipe.encoder_lr),
                "weight_decay": float(recipe.encoder_weight_decay),
            },
            {
                "params": head.parameters(),
                "lr": float(recipe.head_lr),
                "weight_decay": float(recipe.head_weight_decay),
            },
        ]
    )

    history: List[Dict[str, float]] = []
    for epoch in range(1, int(recipe.epochs) + 1):
        encoder.train()
        head.train()
        loader = make_unlabeled_loader(
            train_recordings,
            recipe.batch_size,
            shuffle=True,
            seed=run_seed + epoch,
        )
        total = 0.0
        n_seen = 0
        for x in loader:
            x = x.to(device=DEVICE, dtype=torch.float32, non_blocking=True)
            with torch.no_grad():
                target = morphology_targets(
                    x, EXPECTED_FS, recipe.target_segment_seconds
                )
                target = (target - target_mean) / target_std
            masked, _ = patch_mask_view(x, recipe.patch_size, recipe.mask_ratio)

            optimizer.zero_grad(set_to_none=True)
            # FFT target construction stays FP32; MRCNN/head training also stays FP32 here so
            # the refinement is independent of mixed-precision FFT/runtime behavior.
            pred = head(summarize_mrcnn(encoder(masked)))
            loss = F.smooth_l1_loss(pred, target)
            if not torch.isfinite(loss):
                raise RuntimeError("Non-finite MorphSpec refinement loss")
            loss.backward()
            nn.utils.clip_grad_norm_(list(encoder.parameters()) + list(head.parameters()), 5.0)
            optimizer.step()

            bs = int(x.shape[0])
            total += float(loss.detach().cpu()) * bs
            n_seen += bs

        row = {"epoch": float(epoch), "loss": total / max(1, n_seen)}
        history.append(row)
        print(f"morphspec e{epoch:03d} loss={row['loss']:.6f}")

    state = cpu_state_dict(encoder)
    output_checkpoint.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "state_dict": state,
        "train_subjects": expected,
        "val_subjects": sorted(int(v) for v in val_subjects),
        "test_subjects": sorted(int(v) for v in test_subjects),
        "fold": int(fold),
        "seed": int(run_seed),
        "project_version": "morphmae-morphspec-r1",
        "recipe": asdict(recipe),
        "target_names": list(TARGET_NAMES),
        "target_mean": target_mean.detach().cpu(),
        "target_std": target_std.detach().cpu(),
        "uses_sleep_stage_labels": False,
        "stage_label_arrays_read": False,
        "source_checkpoint": str(source_checkpoint),
        "source_sha256": sha256_file(source_checkpoint),
        "source_metadata": source_meta,
        "history": history,
    }
    torch.save(payload, output_checkpoint)

    # Re-open through the same strict bridge used downstream.
    _, exported_meta = extract_mrcnn_state_dict(output_checkpoint)
    if sorted(int(v) for v in exported_meta.get("train_subjects", [])) != expected:
        raise RuntimeError("Exported MorphSpec checkpoint lost split metadata")

    return {
        "checkpoint": str(output_checkpoint),
        "source_checkpoint": str(source_checkpoint),
        "fold": int(fold),
        "seed": int(run_seed),
        "train_subjects": expected,
        "val_subjects": sorted(int(v) for v in val_subjects),
        "test_subjects": sorted(int(v) for v in test_subjects),
        "recipe": asdict(recipe),
        "target_names": list(TARGET_NAMES),
        "final_loss": float(history[-1]["loss"]),
        "stage_labels_used": False,
        "stage_label_arrays_read": False,
    }
