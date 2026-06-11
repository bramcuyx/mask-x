
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

from utils.paths import load_config, get_config_path, ensure_dir, resolve_path

import pathlib
import numpy as np
import matplotlib.pyplot as plt
 

import dsp_spp_method as dsp
import mask.mask as baseline_mask   # 只借用 plot_masked_spect


config = load_config()

events_folder = get_config_path(config, "marine_events_folder")
marine_outputs_dir = ensure_dir(get_config_path(config, "marine_outputs_folder"))

MASKS_FOLDER = marine_outputs_dir / "masks" / "dsp_spp"
PLOTS_FOLDER = marine_outputs_dir / "plots" / "dsp_spp"

RESAMPLED_SR = 48000
PADDING = 1.0

# 你当前 DSP 最好版本的参数
NPERSEG = 256
NOVERLAP = 128
NOISE_MODE = "edge"
NOISE_QUANTILE = 0.2

SPP_ALPHA = 1.5
SPP_BETA = 2.0
SMOOTH_KERNEL = (3, 3)
FINAL_THRESHOLD = 0.4

POSTPROCESS = True
MIN_REGION_SIZE = 8
MAX_FULLBAND_RATIO = 0.9
MAX_IMPULSE_WIDTH_FRAMES = 2


def get_event_files(events_folder: pathlib.Path):
    files = sorted(events_folder.glob("*.wav"))
    if not files:
        raise FileNotFoundError(f"No .wav files found in {events_folder}")
    return files


def main():
    MASKS_FOLDER.mkdir(parents=True, exist_ok=True)
    PLOTS_FOLDER.mkdir(parents=True, exist_ok=True)

    event_files = get_event_files(events_folder)

    for i, file in enumerate(event_files, start=1):
        print(f"Processing {i}/{len(event_files)}: {file.name}")

        mask, soft_spp, noise_median_db, Sxx, f, t = dsp.estimate_mask_file(
            file,
            nperseg=NPERSEG,
            noverlap=NOVERLAP,
            padding=PADDING,
            plot=False,
            resampled_sr=RESAMPLED_SR,
            return_axes=True,
            noise_mode=NOISE_MODE,
            noise_quantile=NOISE_QUANTILE,
            spp_alpha=SPP_ALPHA,
            spp_beta=SPP_BETA,
            smooth_kernel=SMOOTH_KERNEL,
            final_threshold=FINAL_THRESHOLD,
            postprocess=POSTPROCESS,
            min_region_size=MIN_REGION_SIZE,
            max_fullband_ratio=MAX_FULLBAND_RATIO,
            max_impulse_width_frames=MAX_IMPULSE_WIDTH_FRAMES,
        )

        np.save(MASKS_FOLDER / f"{file.stem}.npy", mask)

        fig, ax = plt.subplots(1, 3, figsize=(20, 6))
        plot_masked_spect(
            Sxx,
            mask,
            f,
            t,
            ax_mask=ax[1],
            ax_spect=ax[0],
            ax_mask_only=ax[2],
            show=False,
        )
        fig.savefig(PLOTS_FOLDER / f"{file.stem}.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

    print("Done.")


if __name__ == "__main__":
    main()