"""
Evaluate Method 3: Pure DL Spectrogram U-Net on FUSS.

This script loads a trained checkpoint and computes Precision, Recall, F1, IoU,
and Dice against FUSS source-derived ground-truth masks.
"""

from __future__ import annotations


# --- Project import bootstrap ---
import pathlib as _project_pathlib
import sys as _project_sys

_PROJECT_ROOT = _project_pathlib.Path(__file__).resolve().parents[2]
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
from typing import Dict, List, Tuple

import numpy as np
import scipy.signal
import soundfile as sf
import torch

import dl_unet_method as dl


def get_fuss_root() -> pathlib.Path:
    """Return the local FUSS ssdata/train folder from an environment variable."""
    fuss_root = os.environ.get("FUSS_ROOT")
    if not fuss_root:
        raise RuntimeError(
            "FUSS_ROOT is not set. Set it to your local FUSS ssdata/train folder. "
            "Example: export FUSS_ROOT=/path/to/FUSS/ssdata/train"
        )

    root = pathlib.Path(fuss_root).expanduser()
    if not root.exists():
        raise FileNotFoundError(f"FUSS_ROOT does not exist: {root}")
    return root


ROOT = get_fuss_root()

CHECKPOINT_PATH = pathlib.Path(__file__).with_name("models") / "fuss_dl_unet_best.pt"
OUT_CSV = pathlib.Path(__file__).with_name("models") / "fuss_dl_unet_metrics.csv"

SEED = 42
NUM_EXAMPLES = None
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
EVALUATE_SPLIT = "test"      # "train", "val", "test", or "all"

RESAMPLED_SR = 16000
NPERSEG = 256
NOVERLAP = 128

# If None, use checkpoint metadata threshold. Otherwise set a float, e.g. 0.50.
FIXED_THRESHOLD = None
THRESHOLD_GRID = None        # e.g. [0.35, 0.40, 0.45, 0.50, 0.55] to search on this split

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


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

    return (fg_sum > bg_sum).astype(bool)


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
        raise ValueError("Need at least 3 files.")

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


def choose_files(files: List[pathlib.Path]) -> List[pathlib.Path]:
    random.Random(SEED).shuffle(files)
    if NUM_EXAMPLES is not None:
        files = files[: int(NUM_EXAMPLES)]

    train_files, val_files, test_files = split_files(files)
    if EVALUATE_SPLIT == "train":
        return train_files
    if EVALUATE_SPLIT == "val":
        return val_files
    if EVALUATE_SPLIT == "test":
        return test_files
    if EVALUATE_SPLIT == "all":
        return files
    raise ValueError("EVALUATE_SPLIT must be 'train', 'val', 'test', or 'all'")


def evaluate_at_threshold(model, files: List[pathlib.Path], threshold: float):
    rows = []
    for i, mix_path in enumerate(files, start=1):
        print(f"Evaluating {i}/{len(files)}: {mix_path.name}")
        source_dir = ROOT / f"{mix_path.stem}_sources"

        mask, prob, _ = dl.estimate_mask_file(
            mix_path,
            model=model,
            nperseg=NPERSEG,
            noverlap=NOVERLAP,
            resampled_sr=RESAMPLED_SR,
            threshold=threshold,
            device=DEVICE,
            postprocess=False,
        )
        gt = compute_truth_mask(source_dir)
        metrics = compute_metrics(mask, gt)
        metrics.update({"file": mix_path.name, "threshold": float(threshold)})
        rows.append(metrics)
    return rows


def summarize(rows):
    keys = ["precision", "recall", "f1", "iou", "dice"]
    return {key: float(np.mean([row[key] for row in rows])) for key in keys}


def main():
    model, metadata = dl.load_checkpoint(CHECKPOINT_PATH, device=DEVICE)

    files = list_valid_mixture_files(ROOT)
    eval_files = choose_files(files)
    print(f"Device: {DEVICE}")
    print(f"Evaluating split={EVALUATE_SPLIT}, files={len(eval_files)}")

    if THRESHOLD_GRID is not None:
        best_threshold = None
        best_summary = None
        best_rows = None
        for threshold in THRESHOLD_GRID:
            rows = evaluate_at_threshold(model, eval_files, threshold=float(threshold))
            summary = summarize(rows)
            print(f"threshold={threshold}: {summary}")
            if best_summary is None or summary["f1"] > best_summary["f1"]:
                best_threshold = float(threshold)
                best_summary = summary
                best_rows = rows
        rows = best_rows
        threshold = best_threshold
        summary = best_summary
    else:
        if FIXED_THRESHOLD is not None:
            threshold = float(FIXED_THRESHOLD)
        else:
            threshold = float(metadata.get("threshold", 0.5))
        rows = evaluate_at_threshold(model, eval_files, threshold=threshold)
        summary = summarize(rows)

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with OUT_CSV.open("w", newline="", encoding="utf-8") as f:
        fieldnames = ["file", "threshold", "tp", "fp", "fn", "tn", "precision", "recall", "f1", "iou", "dice"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Threshold used: {threshold}")
    print("Mean metrics:", summary)
    print(f"Saved per-file metrics to: {OUT_CSV}")


if __name__ == "__main__":
    main()
