
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
import pathlib
import numpy as np
import yaml
import soundfile as sf

CONFIG_PATH = pathlib.Path(__file__).with_name("mask_config.yaml")


def load_config(config_path: pathlib.Path) -> dict:
    with config_path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def compute_mask_metrics(mask: np.ndarray, wav_path: pathlib.Path, padding: float) -> dict:
    """
    只做“自动体检”，不是真值准确率。
    利用已知先验：
    - 前 padding 秒是背景
    - 后 padding 秒是背景
    - 中间才可能是事件
    """
    info = sf.info(str(wav_path))
    duration = float(info.duration)

    if mask.ndim != 2:
        raise ValueError(f"Mask shape should be 2D, got {mask.shape} for {wav_path.name}")

    n_freq, n_frames = mask.shape
    eps = np.finfo(float).eps

    if duration <= 0:
        raise ValueError(f"Invalid duration for {wav_path.name}: {duration}")

    frames_per_second = n_frames / duration
    pad_frames = int(round(padding * frames_per_second))

    event_start = min(n_frames, max(0, pad_frames))
    event_end = max(event_start, n_frames - pad_frames)

    background_region = np.zeros_like(mask, dtype=bool)
    background_region[:, :event_start] = True
    background_region[:, event_end:] = True

    event_region = np.zeros_like(mask, dtype=bool)
    event_region[:, event_start:event_end] = True

    background_total = int(background_region.sum())
    event_total = int(event_region.sum())

    background_positive = int(np.logical_and(mask, background_region).sum())
    event_positive = int(np.logical_and(mask, event_region).sum())

    background_positive_ratio = background_positive / max(background_total, 1)
    event_positive_ratio = event_positive / max(event_total, 1)
    contrast_ratio = event_positive_ratio / max(background_positive_ratio, eps)

    middle_mask = mask[:, event_start:event_end]
    if middle_mask.size == 0:
        time_frame_activity_ratio = 0.0
        frequency_bin_activity_ratio = 0.0
    else:
        time_frame_activity_ratio = float(np.mean(np.any(middle_mask, axis=0)))
        frequency_bin_activity_ratio = float(np.mean(np.any(middle_mask, axis=1)))

    # 自动标签：只是帮你筛可疑样本
    flags = []
    if background_positive_ratio > 0.01:
        flags.append("high_background_leakage")
    if event_positive_ratio < 0.005:
        flags.append("too_sparse")
    if event_positive_ratio > 0.60:
        flags.append("too_dense")
    if contrast_ratio < 5.0:
        flags.append("weak_event_background_separation")

    auto_flag = ";".join(flags) if flags else "ok"

    return {
        "background_positive_ratio": background_positive_ratio,
        "event_positive_ratio": event_positive_ratio,
        "contrast_ratio": contrast_ratio,
        "time_frame_activity_ratio": time_frame_activity_ratio,
        "frequency_bin_activity_ratio": frequency_bin_activity_ratio,
        "auto_flag": auto_flag,
    }


def main():
    config = load_config(CONFIG_PATH)

    events_folder = pathlib.Path(config["paths"]["events_folder"])
    masks_folder = pathlib.Path(config["paths"]["masks_folder"])
    plots_folder = pathlib.Path(config["paths"]["plots_folder"])
    padding = float(config["processing"]["padding"])
    threshold_db = float(config["processing"]["initial_threshold_db"])

    if not masks_folder.exists():
        raise FileNotFoundError(f"Masks folder not found: {masks_folder}")
    if not events_folder.exists():
        raise FileNotFoundError(f"Events folder not found: {events_folder}")

    rows = []
    mask_files = sorted(masks_folder.glob("*.npy"))

    if not mask_files:
        raise FileNotFoundError(f"No .npy masks found in {masks_folder}")

    for idx, mask_file in enumerate(mask_files, start=1):
        wav_path = events_folder / f"{mask_file.stem}.wav"
        if not wav_path.exists():
            print(f"[WARN] Missing wav for mask: {mask_file.name}")
            continue

        try:
            mask = np.load(mask_file)
            metrics = compute_mask_metrics(mask, wav_path, padding)
            rows.append({
                "file": mask_file.stem,
                "threshold_db": threshold_db,
                **metrics,
            })
        except Exception as e:
            print(f"[WARN] Failed on {mask_file.name}: {e}")

        if idx % 20 == 0 or idx == len(mask_files):
            print(f"Processed {idx}/{len(mask_files)} masks")

    if not rows:
        raise RuntimeError("No metrics were computed.")

    output_csv = plots_folder / "metrics_existing_masks.csv"
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "file",
        "threshold_db",
        "background_positive_ratio",
        "event_positive_ratio",
        "contrast_ratio",
        "time_frame_activity_ratio",
        "frequency_bin_activity_ratio",
        "auto_flag",
    ]

    with output_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    bg = np.array([r["background_positive_ratio"] for r in rows], dtype=float)
    ev = np.array([r["event_positive_ratio"] for r in rows], dtype=float)
    cr = np.array([r["contrast_ratio"] for r in rows], dtype=float)
    ok_count = sum(r["auto_flag"] == "ok" for r in rows)

    print("\n=== Summary ===")
    print(f"Total masks analysed: {len(rows)}")
    print(f"Median background leakage: {np.median(bg):.4f}   (越小越好)")
    print(f"Median event fill ratio:  {np.median(ev):.4f}   (太小/太大都不好)")
    print(f"Median contrast ratio:    {np.median(cr):.2f}   (越大越好)")
    print(f"Auto-flagged ok:          {ok_count}/{len(rows)}")
    print(f"Saved CSV: {output_csv}")


if __name__ == "__main__":
    main()