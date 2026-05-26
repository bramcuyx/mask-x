"""
Evaluate Method 4: Hybrid DSP-ML on FUSS.

This script leaves the existing DSP, ML, and DL methods unchanged. It trains a
new LightGBM classifier using the original ML feature set plus DSP-SPP guidance
features, then evaluates on the FUSS held-out test split.
"""


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

import hybrid_dsp_ml_method as hml

import warnings
warnings.filterwarnings(
    "ignore",
    message="X does not have valid feature names, but LGBMClassifier was fitted with feature names"
)

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
NUM_EXAMPLES = None   # 改成 None 就跑全部有 foreground 的样本

# Train / Val / Test split
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
# 剩下的自动给 test

RESAMPLED_SR = 16000
NPERSEG = 256
NOVERLAP = 128

# 训练时每个文件最多抽多少正/负 bin
# 如果机器内存比较紧，可以继续减小
MAX_POS_BINS_PER_FILE = 300
MAX_NEG_BINS_PER_FILE = 300

# LightGBM params
LGBM_N_ESTIMATORS = 400
LGBM_LEARNING_RATE = 0.05
LGBM_NUM_LEAVES = 63
LGBM_MIN_CHILD_SAMPLES = 20
LGBM_SUBSAMPLE = 0.8
LGBM_COLSAMPLE_BYTREE = 0.8
LGBM_REG_LAMBDA = 1.0

# CRF params
APPLY_CRF = True
CRF_SPATIAL_SIGMA = (1.0, 1.0)
CRF_PAIRWISE_WEIGHT = 1.2
CRF_UNARY_WEIGHT = 1.0
CRF_ITERATIONS = 5

# 后处理
POSTPROCESS = True
MIN_REGION_SIZE = 8

# 阈值搜索
THRESHOLD_GRID = [0.40, 0.45, 0.50, 0.55]

# 是否重用已训练模型
REUSE_SAVED_MODEL = False

if NUM_EXAMPLES is None:
    MODEL_PKL = ROOT / "fuss_hybrid_dsp_ml_model_all.pkl"
    OUT_CSV = ROOT / "fuss_hybrid_dsp_ml_metrics_all.csv"
else:
    MODEL_PKL = ROOT / f"fuss_hybrid_dsp_ml_model_seed{SEED}_{NUM_EXAMPLES}.pkl"
    OUT_CSV = ROOT / f"fuss_hybrid_dsp_ml_metrics_seed{SEED}_{NUM_EXAMPLES}.csv"


def load_audio_resampled(wav_path: pathlib.Path, target_sr: int | None):
    data, sr = sf.read(str(wav_path))
    if data.ndim > 1:
        data = np.mean(data, axis=1)

    if target_sr is not None and sr != target_sr:
        n_samples = round(len(data) * float(target_sr) / sr)
        data = scipy.signal.resample(data, n_samples)
        sr = target_sr

    return data.astype(np.float32), sr



def spectrogram_power(data: np.ndarray, sr: int):
    _, _, Sxx = scipy.signal.spectrogram(
        data,
        fs=sr,
        nperseg=NPERSEG,
        noverlap=NOVERLAP,
        mode="psd",
    )
    return np.maximum(Sxx, np.finfo(float).eps).astype(np.float32)


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


def split_files(files):
    n_total = len(files)
    if n_total < 3:
        raise ValueError("Need at least 3 files to create train/val/test split.")

    n_train = int(TRAIN_RATIO * n_total)
    n_val = int(VAL_RATIO * n_total)
    n_test = n_total - n_train - n_val

    # 保证每个 split 至少 1 个
    if n_train < 1:
        n_train = 1
    if n_val < 1:
        n_val = 1
    n_test = n_total - n_train - n_val
    if n_test < 1:
        n_test = 1
        if n_train > n_val:
            n_train -= 1
        else:
            n_val -= 1

    train_files = files[:n_train]
    val_files = files[n_train:n_train + n_val]
    test_files = files[n_train + n_val:]

    return train_files, val_files, test_files


def sample_bin_indices(y_flat, rng, max_pos_bins, max_neg_bins):
    pos_idx = np.flatnonzero(y_flat == 1)
    neg_idx = np.flatnonzero(y_flat == 0)

    chosen = []

    if pos_idx.size > 0:
        n_pos = min(max_pos_bins, pos_idx.size)
        if n_pos < pos_idx.size:
            pos_pick = rng.choice(pos_idx, size=n_pos, replace=False)
        else:
            pos_pick = pos_idx
        chosen.append(pos_pick)

    if neg_idx.size > 0:
        n_neg = min(max_neg_bins, neg_idx.size)
        if n_neg < neg_idx.size:
            neg_pick = rng.choice(neg_idx, size=n_neg, replace=False)
        else:
            neg_pick = neg_idx
        chosen.append(neg_pick)

    if not chosen:
        raise ValueError("No bins available for sampling.")

    idx = np.concatenate(chosen)
    rng.shuffle(idx)
    return idx


def build_sampled_dataset(file_list, rng):
    xs = []
    ys = []

    for i, mix_path in enumerate(file_list, start=1):
        source_dir = ROOT / f"{mix_path.stem}_sources"

        try:
            X_full, Sxx, _, _ = hml.extract_feature_matrix_file(
                mix_path,
                nperseg=NPERSEG,
                noverlap=NOVERLAP,
                resampled_sr=RESAMPLED_SR,
            )
            gt_mask = compute_truth_mask(source_dir)

            n_freq_mix, n_time_mix = Sxx.shape
            n_features = X_full.shape[1]

            X_cube = X_full.reshape(n_freq_mix, n_time_mix, n_features)

            nf = min(n_freq_mix, gt_mask.shape[0])
            nt = min(n_time_mix, gt_mask.shape[1])

            X_crop = X_cube[:nf, :nt, :].reshape(-1, n_features)
            y_crop = gt_mask[:nf, :nt].reshape(-1).astype(np.uint8)

            idx = sample_bin_indices(
                y_crop,
                rng=rng,
                max_pos_bins=MAX_POS_BINS_PER_FILE,
                max_neg_bins=MAX_NEG_BINS_PER_FILE,
            )

            xs.append(X_crop[idx].astype(np.float32))
            ys.append(y_crop[idx].astype(np.uint8))

        except Exception as e:
            print(f"[WARN] Failed while building dataset from {mix_path.name}: {e}")

        if i % 20 == 0 or i == len(file_list):
            print(f"Sampled training rows from {i}/{len(file_list)} files")

    if not xs:
        raise RuntimeError("No sampled training data were produced.")

    X = np.concatenate(xs, axis=0).astype(np.float32)
    y = np.concatenate(ys, axis=0).astype(np.uint8)
    return X, y


def summarize_rows(rows, title):
    precision_vals = np.array([r["precision"] for r in rows], dtype=float)
    recall_vals = np.array([r["recall"] for r in rows], dtype=float)
    f1_vals = np.array([r["f1"] for r in rows], dtype=float)
    iou_vals = np.array([r["iou"] for r in rows], dtype=float)
    dice_vals = np.array([r["dice"] for r in rows], dtype=float)

    print(f"\n=== {title} ===")
    print(f"Evaluated:      {len(rows)}")
    print(f"Mean Precision: {precision_vals.mean():.4f}")
    print(f"Mean Recall:    {recall_vals.mean():.4f}")
    print(f"Mean F1:        {f1_vals.mean():.4f}")
    print(f"Mean IoU:       {iou_vals.mean():.4f}")
    print(f"Mean Dice:      {dice_vals.mean():.4f}")

    return {
        "precision": float(precision_vals.mean()),
        "recall": float(recall_vals.mean()),
        "f1": float(f1_vals.mean()),
        "iou": float(iou_vals.mean()),
        "dice": float(dice_vals.mean()),
    }


def evaluate_files(file_list, model, threshold):
    rows = []

    for i, mix_path in enumerate(file_list, start=1):
        source_dir = ROOT / f"{mix_path.stem}_sources"

        try:
            pred_mask, raw_prob, decoded_prob, dsp_prob, Sxx = hml.estimate_mask_file(
                mix_path,
                model=model,
                nperseg=NPERSEG,
                noverlap=NOVERLAP,
                plot=False,
                resampled_sr=RESAMPLED_SR,
                return_axes=False,
                threshold=threshold,
                apply_crf=APPLY_CRF,
                crf_spatial_sigma=CRF_SPATIAL_SIGMA,
                crf_pairwise_weight=CRF_PAIRWISE_WEIGHT,
                crf_unary_weight=CRF_UNARY_WEIGHT,
                crf_iterations=CRF_ITERATIONS,
                postprocess=POSTPROCESS,
                min_region_size=MIN_REGION_SIZE,
            )

            gt_mask = compute_truth_mask(source_dir)
            metrics = compute_metrics(pred_mask, gt_mask)

            rows.append({
                "file": mix_path.stem,
                **metrics,
            })

        except Exception as e:
            print(f"[WARN] Failed on {mix_path.name}: {e}")

        if i % 20 == 0 or i == len(file_list):
            print(f"Evaluated {i}/{len(file_list)}")

    if not rows:
        raise RuntimeError("No evaluation results were produced.")

    return rows


def choose_best_threshold(val_files, model):
    best_threshold = None
    best_score = -1.0

    for threshold in THRESHOLD_GRID:
        print(f"\n[VAL] Testing threshold = {threshold:.2f}")
        rows = evaluate_files(val_files, model=model, threshold=threshold)
        summary = summarize_rows(rows, title=f"Validation @ threshold={threshold:.2f}")
        score = summary["f1"]

        if score > best_score:
            best_score = score
            best_threshold = threshold

    return best_threshold, best_score


def main():
    if not ROOT.exists():
        raise FileNotFoundError(f"ROOT not found: {ROOT}")

    mixture_files = list_valid_mixture_files(ROOT)
    if not mixture_files:
        raise FileNotFoundError(f"No valid example*.wav found in {ROOT}")

    TEST_LIST = ROOT / f"fuss_hybrid_dsp_ml_test_files.txt"

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
        rng_py = random.Random(SEED)
        selected_files = rng_py.sample(mixture_files, NUM_EXAMPLES)

    train_files, val_files, test_files = split_files(selected_files)

    TEST_LIST = ROOT / "fuss_hybrid_dsp_ml_test_files.txt"
    with TEST_LIST.open("w", encoding="utf-8") as f:
        for p in test_files:
            f.write(p.name + "\n")

    print(f"Saved test file list: {TEST_LIST}")
    print(f"Total selected files: {len(selected_files)}")
    print(f"Train files:         {len(train_files)}")
    print(f"Val files:           {len(val_files)}")
    print(f"Test files:          {len(test_files)}")

    rng_np = np.random.default_rng(SEED)

    if REUSE_SAVED_MODEL and MODEL_PKL.exists():
        payload = hml.load_model_bundle(MODEL_PKL)
        model = payload["model"]
        saved_threshold = payload["metadata"].get("best_threshold", None)
        print(f"Loaded saved model: {MODEL_PKL}")

        if saved_threshold is None:
            best_threshold, best_val_f1 = choose_best_threshold(val_files, model=model)
        else:
            best_threshold = float(saved_threshold)
            best_val_f1 = None
    else:
        print("\nBuilding sampled training set with DSP-guided features...")
        X_train, y_train = build_sampled_dataset(train_files, rng=rng_np)
        print(f"X_train shape: {X_train.shape}")
        print(f"y_train shape: {y_train.shape}")
        print(f"Train positive ratio: {y_train.mean():.4f}")

        print("\nBuilding sampled validation set with DSP-guided features...")
        X_val, y_val = build_sampled_dataset(val_files, rng=rng_np)
        print(f"X_val shape: {X_val.shape}")
        print(f"y_val shape: {y_val.shape}")
        print(f"Val positive ratio: {y_val.mean():.4f}")

        print("\nTraining Hybrid DSP-ML LightGBM...")
        model = hml.train_lgbm_classifier(
            X_train,
            y_train,
            X_val=X_val,
            y_val=y_val,
            random_state=SEED,
            n_estimators=LGBM_N_ESTIMATORS,
            learning_rate=LGBM_LEARNING_RATE,
            num_leaves=LGBM_NUM_LEAVES,
            min_child_samples=LGBM_MIN_CHILD_SAMPLES,
            subsample=LGBM_SUBSAMPLE,
            colsample_bytree=LGBM_COLSAMPLE_BYTREE,
            reg_lambda=LGBM_REG_LAMBDA,
        )

        print("\nChoosing best threshold on validation split...")
        best_threshold, best_val_f1 = choose_best_threshold(val_files, model=model)
        print(f"\nBest threshold on validation split: {best_threshold:.2f}")
        print(f"Best validation mean F1:            {best_val_f1:.4f}")

        hml.save_model_bundle(
            MODEL_PKL,
            model=model,
            metadata={
                "seed": SEED,
                "num_examples": NUM_EXAMPLES,
                "best_threshold": best_threshold,
                "nperseg": NPERSEG,
                "noverlap": NOVERLAP,
                "resampled_sr": RESAMPLED_SR,
                "apply_crf": APPLY_CRF,
                "crf_spatial_sigma": CRF_SPATIAL_SIGMA,
                "crf_pairwise_weight": CRF_PAIRWISE_WEIGHT,
                "crf_unary_weight": CRF_UNARY_WEIGHT,
                "crf_iterations": CRF_ITERATIONS,
                "postprocess": POSTPROCESS,
                "min_region_size": MIN_REGION_SIZE,
                "dsp_noise_mode": "quantile",
                "dsp_noise_quantile": 0.2,
                "dsp_final_threshold": 0.5,
            },
        )
        print(f"Saved model bundle: {MODEL_PKL}")

    print("\nEvaluating on test split...")
    test_rows = evaluate_files(test_files, model=model, threshold=best_threshold)

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["file", "tp", "fp", "fn", "tn", "precision", "recall", "f1", "iou", "dice"]
    with OUT_CSV.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(test_rows)

    summary = summarize_rows(test_rows, title="Hybrid DSP-ML LightGBM + 2D CRF on FUSS (test split)")
    print(f"Seed:           {SEED if NUM_EXAMPLES is not None else 'ALL'}")
    print(f"Threshold:      {best_threshold:.2f}")
    print(f"Saved CSV:      {OUT_CSV}")


if __name__ == "__main__":
    main()