"""
Train Method 5: Hybrid DSP-DL U-Net on FUSS.

Set the FUSS_ROOT environment variable before running.
Expected FUSS layout follows the existing project scripts:

ROOT/
  example00000.wav
  example00000_sources/
    foreground*.wav
    background*.wav
"""

from __future__ import annotations


# --- Project import bootstrap ---
import pathlib as _project_pathlib
import sys as _project_sys

_PROJECT_ROOT = _project_pathlib.Path(__file__).resolve().parents[2]

if str(_PROJECT_ROOT) not in _project_sys.path:
    _project_sys.path.insert(0, str(_PROJECT_ROOT))

for _rel in [
    "methods/baseline",
    "methods/method_1_dsp_spp",
    "methods/method_2_ml_lgbm_crf",
    "methods/method_3_dl_unet",
    "methods/method_4_hybrid_dsp_ml",
    "methods/method_5_hybrid_dsp_dl",
]:
    _p = _PROJECT_ROOT / _rel
    if _p.exists() and str(_p) not in _project_sys.path:
        _project_sys.path.insert(0, str(_p))
# --- End project import bootstrap ---

import csv
import os
import pathlib
import random
import time
from typing import Dict, List, Tuple

import numpy as np
import scipy.signal
import soundfile as sf
import torch
from torch.utils.data import DataLoader, Dataset

import hybrid_dsp_dl_method as hdl


# -----------------------------------------------------------------------------
# Paths and experiment settings
# -----------------------------------------------------------------------------

 


from utils.paths import load_config, get_config_path, get_fuss_root, ensure_dir

config = load_config()
ROOT = get_fuss_root(config)

MODELS_DIR = ensure_dir(get_config_path(config, "models_folder"))
FUSS_OUTPUTS_DIR = ensure_dir(get_config_path(config, "fuss_outputs_folder"))

CHECKPOINT_PATH = MODELS_DIR / "fuss_hybrid_dsp_dl_best.pt"
TRAIN_LOG_CSV = FUSS_OUTPUTS_DIR / "fuss_hybrid_dsp_dl_training_log.csv"

SEED = 42
NUM_EXAMPLES = None       # set to e.g. 200 for quick debugging; None means all valid examples
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15          # remaining files are left for external test/evaluate script

RESAMPLED_SR = 16000
NPERSEG = 256
NOVERLAP = 128

# Model
BASE_CHANNELS = 16
DEPTH = 3
DROPOUT = 0.05

# Training
EPOCHS = 10
BATCH_SIZE = 1             # keep 1 because clips may have different time lengths
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-5
BCE_WEIGHT = 1.0
DICE_WEIGHT = 1.0
POS_WEIGHT = None          # set e.g. 2.0 if recall is too low
GRAD_CLIP_NORM = 5.0
PATIENCE = 8

# Optional random time crop for memory/speed. None uses the full spectrogram.
MAX_TIME_FRAMES = None     # e.g. 512

THRESHOLD_GRID = [0.40, 0.45, 0.50, 0.55]
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# -----------------------------------------------------------------------------
# FUSS utilities
# -----------------------------------------------------------------------------


def load_audio_resampled(wav_path: pathlib.Path, target_sr: int | None) -> Tuple[np.ndarray, int]:
    data, sr = sf.read(str(wav_path))
    if data.ndim > 1:
        data = np.mean(data, axis=1)

    if target_sr is not None and sr != target_sr:
        n_samples = round(len(data) * float(target_sr) / sr)
        data = scipy.signal.resample(data, n_samples)
        sr = int(target_sr)

    return data.astype(np.float32), int(sr)


def spectrogram_power(data: np.ndarray, sr: int) -> np.ndarray:
    _, _, Sxx = scipy.signal.spectrogram(
        data,
        fs=sr,
        nperseg=NPERSEG,
        noverlap=NOVERLAP,
        mode="psd",
    )
    return np.maximum(Sxx, np.finfo(np.float32).eps).astype(np.float32)


def compute_truth_mask(source_dir: pathlib.Path) -> np.ndarray:
    """
    Ground truth TF mask from FUSS separated sources:
        foreground power > background power => 1
        else                                => 0
    """
    fg_sum = None
    bg_sum = None

    wavs = sorted(source_dir.glob("*.wav"))
    if not wavs:
        raise FileNotFoundError(f"No source wavs found in {source_dir}")

    for wav in wavs:
        data, sr = load_audio_resampled(wav, RESAMPLED_SR)
        Sxx = spectrogram_power(data, sr)

        if wav.name.startswith("background"):
            bg_sum = Sxx if bg_sum is None else (bg_sum + Sxx)
        elif wav.name.startswith("foreground"):
            fg_sum = Sxx if fg_sum is None else (fg_sum + Sxx)

    if fg_sum is None:
        raise ValueError(f"No foreground wavs found in {source_dir}")
    if bg_sum is None:
        bg_sum = np.zeros_like(fg_sum)

    return (fg_sum > bg_sum).astype(np.float32)


def has_foreground(source_dir: pathlib.Path) -> bool:
    return any(p.name.startswith("foreground") and p.suffix.lower() == ".wav" for p in source_dir.glob("*.wav"))


def list_valid_mixture_files(root: pathlib.Path) -> List[pathlib.Path]:
    files = []
    for p in root.iterdir():
        if p.is_file() and p.suffix.lower() == ".wav" and p.stem.startswith("example"):
            source_dir = root / f"{p.stem}_sources"
            if source_dir.exists() and has_foreground(source_dir):
                files.append(p)
    return sorted(files)


def split_files(files: List[pathlib.Path]) -> Tuple[List[pathlib.Path], List[pathlib.Path], List[pathlib.Path]]:
    n_total = len(files)
    if n_total < 3:
        raise ValueError("Need at least 3 valid FUSS examples.")

    n_train = max(1, int(TRAIN_RATIO * n_total))
    n_val = max(1, int(VAL_RATIO * n_total))
    n_test = n_total - n_train - n_val
    if n_test < 1:
        n_test = 1
        if n_train >= n_val and n_train > 1:
            n_train -= 1
        elif n_val > 1:
            n_val -= 1

    return files[:n_train], files[n_train:n_train + n_val], files[n_train + n_val:]


def compute_metrics(pred: np.ndarray, gt: np.ndarray) -> Dict[str, float]:
    nf = min(pred.shape[0], gt.shape[0])
    nt = min(pred.shape[1], gt.shape[1])
    pred = pred[:nf, :nt].astype(bool)
    gt = gt[:nf, :nt].astype(bool)

    tp = np.logical_and(pred, gt).sum()
    fp = np.logical_and(pred, ~gt).sum()
    fn = np.logical_and(~pred, gt).sum()
    tn = np.logical_and(~pred, ~gt).sum()

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    iou = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0.0
    dice = (2 * tp) / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0.0

    return {
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "tn": int(tn),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "iou": float(iou),
        "dice": float(dice),
    }


# -----------------------------------------------------------------------------
# Dataset
# -----------------------------------------------------------------------------


class FussMaskDataset(Dataset):
    def __init__(self, files: List[pathlib.Path], root: pathlib.Path, train: bool, max_time_frames: int | None = None):
        self.files = list(files)
        self.root = pathlib.Path(root)
        self.train = bool(train)
        self.max_time_frames = max_time_frames

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int):
        mix_path = self.files[idx]
        source_dir = self.root / f"{mix_path.stem}_sources"

        x, _, _, _, _ = hdl.make_hybrid_input_file(
            mix_path,
            nperseg=NPERSEG,
            noverlap=NOVERLAP,
            resampled_sr=RESAMPLED_SR,
            dsp_noise_mode="quantile",
            dsp_noise_quantile=0.2,
            dsp_final_threshold=0.5,
            dsp_postprocess=True,
        )
        y = compute_truth_mask(source_dir)

        nf = min(x.shape[1], y.shape[0])
        nt = min(x.shape[2], y.shape[1])
        x = x[:, :nf, :nt]
        y = y[:nf, :nt]

        if self.train and self.max_time_frames is not None and nt > self.max_time_frames:
            start = random.randint(0, nt - self.max_time_frames)
            end = start + self.max_time_frames
            x = x[:, :, start:end]
            y = y[:, start:end]

        x_tensor = torch.from_numpy(x).float()
        y_tensor = torch.from_numpy(y[None, :, :]).float()
        return x_tensor, y_tensor, str(mix_path)


def collate_single(batch):
    # Batch size is intentionally 1; this avoids padding variable-length audio in DataLoader.
    if len(batch) != 1:
        raise ValueError("Use BATCH_SIZE=1 or implement a padded collate function.")
    return batch[0]


# -----------------------------------------------------------------------------
# Training and validation
# -----------------------------------------------------------------------------


def run_one_epoch(model, loader, optimizer=None) -> float:
    training = optimizer is not None
    model.train(training)
    losses = []

    for x, y, _ in loader:
        x = x[None, :, :, :].to(DEVICE)  # [1, 2, F, T]
        y = y[None, :, :, :].to(DEVICE)

        x_padded, original_shape = hdl.pad_to_multiple_tensor(x, multiple=2 ** DEPTH)
        logits = model(x_padded)
        logits = hdl.crop_tensor_to_shape(logits, original_shape)

        loss = hdl.bce_dice_loss(
            logits,
            y,
            bce_weight=BCE_WEIGHT,
            dice_weight=DICE_WEIGHT,
            pos_weight=POS_WEIGHT,
        )

        if training:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if GRAD_CLIP_NORM is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(GRAD_CLIP_NORM))
            optimizer.step()

        losses.append(float(loss.detach().cpu().item()))

    return float(np.mean(losses)) if losses else float("nan")


@torch.no_grad()
def collect_validation_probabilities(model, files: List[pathlib.Path]) -> List[Tuple[np.ndarray, np.ndarray, str]]:
    model.eval()
    outputs = []
    for mix_path in files:
        source_dir = ROOT / f"{mix_path.stem}_sources"
        x, _, _, _, _ = hdl.make_hybrid_input_file(
            mix_path,
            nperseg=NPERSEG,
            noverlap=NOVERLAP,
            resampled_sr=RESAMPLED_SR,
            dsp_noise_mode="quantile",
            dsp_noise_quantile=0.2,
            dsp_final_threshold=0.5,
            dsp_postprocess=True,
        )
        prob = hdl.predict_probability_map_from_input(x, model=model, device=DEVICE, pad_multiple=2 ** DEPTH)
        gt = compute_truth_mask(source_dir)
        outputs.append((prob, gt, mix_path.name))
    return outputs


def find_best_threshold(validation_outputs: List[Tuple[np.ndarray, np.ndarray, str]]) -> Tuple[float, Dict[str, float]]:
    best_threshold = THRESHOLD_GRID[0]
    best_metrics = None

    for threshold in THRESHOLD_GRID:
        metric_rows = []
        for prob, gt, _ in validation_outputs:
            pred = prob >= float(threshold)
            metric_rows.append(compute_metrics(pred, gt))

        mean_metrics = {
            key: float(np.mean([row[key] for row in metric_rows]))
            for key in ["precision", "recall", "f1", "iou", "dice"]
        }

        if best_metrics is None or mean_metrics["iou"] > best_metrics["iou"]:
            best_threshold = float(threshold)
            best_metrics = mean_metrics

    return best_threshold, best_metrics if best_metrics is not None else {}


def main():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    files = list_valid_mixture_files(ROOT)
    random.Random(SEED).shuffle(files)
    if NUM_EXAMPLES is not None:
        files = files[: int(NUM_EXAMPLES)]

    train_files, val_files, test_files = split_files(files)
    print(f"Device: {DEVICE}")
    print(f"Valid files: {len(files)} | train={len(train_files)} val={len(val_files)} heldout_test={len(test_files)}")
    print(f"Checkpoint: {CHECKPOINT_PATH}")

    train_dataset = FussMaskDataset(train_files, ROOT, train=True, max_time_frames=MAX_TIME_FRAMES)
    val_dataset = FussMaskDataset(val_files, ROOT, train=False, max_time_frames=None)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_single)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_single)

    metadata = {
        "method": "hybrid_dsp_dl_unet",
        "in_channels": 2,
        "base_channels": BASE_CHANNELS,
        "depth": DEPTH,
        "dropout": DROPOUT,
        "resampled_sr": RESAMPLED_SR,
        "nperseg": NPERSEG,
        "noverlap": NOVERLAP,
        "normalize_mode": "per_clip",
        "threshold": 0.5,
        "seed": SEED,
        "dsp_noise_mode": "quantile",
        "dsp_noise_quantile": 0.2,
        "dsp_final_threshold": 0.5,
        "dsp_postprocess": True,
    }

    model = hdl.create_model({
        "in_channels": 2,
        "base_channels": BASE_CHANNELS,
        "depth": DEPTH,
        "dropout": DROPOUT,
    }).to(DEVICE)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    best_val_loss = float("inf")
    bad_epochs = 0
    log_rows = []

    for epoch in range(1, EPOCHS + 1):
        start = time.time()
        train_loss = run_one_epoch(model, train_loader, optimizer=optimizer)
        val_loss = run_one_epoch(model, val_loader, optimizer=None)
        elapsed = time.time() - start

        improved = val_loss < best_val_loss
        if improved:
            best_val_loss = val_loss
            bad_epochs = 0
            metadata.update({"best_epoch": epoch, "best_val_loss": float(best_val_loss)})
            hdl.save_checkpoint(CHECKPOINT_PATH, model, metadata=metadata)
        else:
            bad_epochs += 1

        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "best_val_loss": best_val_loss,
            "improved": int(improved),
            "elapsed_sec": elapsed,
        }
        log_rows.append(row)
        print(
            f"Epoch {epoch:03d}/{EPOCHS} | train_loss={train_loss:.4f} "
            f"val_loss={val_loss:.4f} best={best_val_loss:.4f} improved={improved}"
        )

        if bad_epochs >= PATIENCE:
            print(f"Early stopping after {PATIENCE} epochs without improvement.")
            break

    # Save training log
    with TRAIN_LOG_CSV.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(log_rows[0].keys()))
        writer.writeheader()
        writer.writerows(log_rows)

    # Reload best model and select threshold on validation set.
    best_model, best_metadata = hdl.load_checkpoint(CHECKPOINT_PATH, device=DEVICE)
    validation_outputs = collect_validation_probabilities(best_model, val_files)
    best_threshold, best_metrics = find_best_threshold(validation_outputs)
    best_metadata.update({"threshold": best_threshold, "val_threshold_metrics": best_metrics})
    hdl.save_checkpoint(CHECKPOINT_PATH, best_model, metadata=best_metadata)

    print(f"Best validation threshold: {best_threshold:.2f}")
    print("Validation metrics:", best_metrics)
    print("Training complete.")


if __name__ == "__main__":
    main()
