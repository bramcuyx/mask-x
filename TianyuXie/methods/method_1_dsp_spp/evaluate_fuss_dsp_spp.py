
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

import numpy as np
import scipy.signal
import soundfile as sf

import dsp_spp_method as dsp


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

SEED = 42
NUM_EXAMPLES = 3000   # 改成 None 就跑全部有 foreground 的样本

RESAMPLED_SR = 16000
NPERSEG = 256
NOVERLAP = 128

# FUSS 上不要用 marine 的边缘噪声假设
PADDING = 0.0
NOISE_MODE = "edge"
NOISE_QUANTILE = 0.2

SPP_ALPHA = 1.5
SPP_BETA = 2.0
SMOOTH_KERNEL = (3, 3)
FINAL_THRESHOLD = 0.4

if NUM_EXAMPLES is None:
    OUT_CSV = ROOT / "fuss_dsp_spp_metrics_all_foreground.csv"
else:
    OUT_CSV = ROOT / f"fuss_dsp_spp_metrics_seed{SEED}_{NUM_EXAMPLES}.csv"


def load_audio_resampled(wav_path: pathlib.Path, target_sr: int | None):
    data, sr = sf.read(str(wav_path))
    if data.ndim > 1:
        data = np.mean(data, axis=1)

    if target_sr is not None and sr != target_sr:
        n_samples = round(len(data) * float(target_sr) / sr)
        data = scipy.signal.resample(data, n_samples)
        sr = target_sr

    return data.astype(float), sr


def spectrogram_power(data: np.ndarray, sr: int):
    _, _, Sxx = scipy.signal.spectrogram(
        data,
        fs=sr,
        nperseg=NPERSEG,
        noverlap=NOVERLAP,
    )
    return np.maximum(Sxx, np.finfo(float).eps)


def compute_truth_mask(source_dir: pathlib.Path) -> np.ndarray:
    """
    Ground-truth mask for FUSS:
      foreground power > background power  => 1
      else                                 => 0
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

    gt_mask = fg_sum > bg_sum
    return gt_mask.astype(bool)


def compute_metrics(pred: np.ndarray, gt: np.ndarray):
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


def has_foreground(source_dir: pathlib.Path) -> bool:
    return any(
        p.name.startswith("foreground") and p.suffix.lower() == ".wav"
        for p in source_dir.glob("*.wav")
    )


def list_valid_mixture_files(root: pathlib.Path):
    valid_files = []
    for p in root.iterdir():
        if p.is_file() and p.suffix.lower() == ".wav" and p.stem.startswith("example"):
            source_dir = root / f"{p.stem}_sources"
            if source_dir.exists() and has_foreground(source_dir):
                valid_files.append(p)
    return sorted(valid_files)


def main():
    if not ROOT.exists():
        raise FileNotFoundError(f"ROOT not found: {ROOT}")

    mixture_files = list_valid_mixture_files(ROOT)
    if not mixture_files:
        raise FileNotFoundError(f"No valid example*.wav found in {ROOT}")

    if NUM_EXAMPLES is None:
        selected_files = mixture_files
    else:
        if len(mixture_files) < NUM_EXAMPLES:
            raise ValueError(
                f"Requested {NUM_EXAMPLES} examples, but only found {len(mixture_files)} valid examples"
            )
        rng = random.Random(SEED)
        selected_files = rng.sample(mixture_files, NUM_EXAMPLES)

    rows = []

    for i, mix_path in enumerate(selected_files, start=1):
        source_dir = ROOT / f"{mix_path.stem}_sources"

        try:
            pred_mask, soft_spp, noise_median_db, Sxx = dsp.estimate_mask_file(
                mix_path,
                nperseg=NPERSEG,
                noverlap=NOVERLAP,
                padding=PADDING,
                plot=False,
                resampled_sr=RESAMPLED_SR,
                return_axes=False,
                noise_mode=NOISE_MODE,
                noise_quantile=NOISE_QUANTILE,
                spp_alpha=SPP_ALPHA,
                spp_beta=SPP_BETA,
                smooth_kernel=SMOOTH_KERNEL,
                final_threshold=FINAL_THRESHOLD,
            )

            gt_mask = compute_truth_mask(source_dir)
            metrics = compute_metrics(pred_mask, gt_mask)

            rows.append({
                "file": mix_path.stem,
                **metrics,
            })

        except Exception as e:
            print(f"[WARN] Failed on {mix_path.name}: {e}")

        if i % 20 == 0 or i == len(selected_files):
            print(f"Processed {i}/{len(selected_files)}")

    if not rows:
        raise RuntimeError("No evaluation results were produced.")

    fieldnames = ["file", "tp", "fp", "fn", "tn", "precision", "recall", "f1", "iou", "dice"]
    with OUT_CSV.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    precision_vals = np.array([r["precision"] for r in rows], dtype=float)
    recall_vals = np.array([r["recall"] for r in rows], dtype=float)
    f1_vals = np.array([r["f1"] for r in rows], dtype=float)
    iou_vals = np.array([r["iou"] for r in rows], dtype=float)
    dice_vals = np.array([r["dice"] for r in rows], dtype=float)

    print("\n=== DSP-SPP on FUSS ===")
    print(f"Seed:           {SEED if NUM_EXAMPLES is not None else 'ALL'}")
    print(f"Evaluated:      {len(rows)}")
    print(f"Mean Precision: {precision_vals.mean():.4f}")
    print(f"Mean Recall:    {recall_vals.mean():.4f}")
    print(f"Mean F1:        {f1_vals.mean():.4f}")
    print(f"Mean IoU:       {iou_vals.mean():.4f}")
    print(f"Mean Dice:      {dice_vals.mean():.4f}")
    print(f"Saved CSV:      {OUT_CSV}")


if __name__ == "__main__":
    main()