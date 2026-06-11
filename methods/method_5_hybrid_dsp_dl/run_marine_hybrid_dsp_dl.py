"""
Run Method 5: Hybrid DSP-DL U-Net on marine event wav files.

This script generates:
  - binary masks:   masks_folder/hybrid_dsp_dl/*.npy
  - probabilities:  masks_folder/hybrid_dsp_dl_prob/*.npy
  - plots:          plots_folder/hybrid_dsp_dl/*.png

Edit CHECKPOINT_PATH after training on FUSS.
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

import pathlib
from utils.paths import load_config, get_config_path, ensure_dir, resolve_path
import matplotlib.pyplot as plt
import numpy as np
import scipy.signal
import soundfile as sf
 
import torch

import hybrid_dsp_dl_method as hdl


RESAMPLED_SR = 48000
NPERSEG = 256
NOVERLAP = 128
MIN_DB = None

# Visualization only: use the same wide dB range as the DSP plots.
PLOT_DB_MIN = -150.0
PLOT_DB_MAX = -60.0

# Use the threshold saved in the checkpoint by default. Set a float to override.
THRESHOLD_OVERRIDE = None

# Keep disabled for the strict pure-DL setting. Enable only if you explicitly report it.
POSTPROCESS = False
MIN_REGION_SIZE = 8

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


  
    
def to_plot_db(Sxx, eps=1e-12):
    """
    Convert spectrogram power to raw dB for visualization only.
    Do not use this for model input normalization.
    """
    return 10.0 * np.log10(np.maximum(Sxx, eps))


def compute_plot_spectrogram(file, resampled_sr, nperseg, noverlap):
    """
    Recompute a raw PSD spectrogram from the waveform for visualization only.

    This avoids plotting any clipped/normalized/intermediate spectrogram that may
    be returned by the DL inference pipeline. It is intended to make the
    spectrogram panel visually comparable with the DSP runner.
    """
    data, sample_rate = sf.read(str(file))
    if data.size == 0:
        raise ValueError(f"Audio file is empty: {file}")
    if data.ndim > 1:
        data = np.mean(data, axis=1)

    if resampled_sr is not None and sample_rate != resampled_sr:
        n_samples = round(len(data) * float(resampled_sr) / sample_rate)
        data = scipy.signal.resample(data, n_samples)
        sample_rate = int(resampled_sr)

    f_plot, t_plot, Sxx_plot = scipy.signal.spectrogram(
        data.astype(np.float32),
        fs=sample_rate,
        nperseg=nperseg,
        noverlap=noverlap,
        mode="psd",
    )
    Sxx_plot = np.maximum(Sxx_plot, np.finfo(np.float32).eps).astype(np.float32)
    return f_plot, t_plot, Sxx_plot


def crop_to_common_grid(Sxx, f, t, *arrays):
    """
    Crop spectrogram and mask/probability arrays to a common [freq, time] grid.
    This is only for robust plotting.
    """
    nf = min([Sxx.shape[0], len(f)] + [a.shape[0] for a in arrays if a is not None])
    nt = min([Sxx.shape[1], len(t)] + [a.shape[1] for a in arrays if a is not None])

    cropped = [None if a is None else a[:nf, :nt] for a in arrays]
    return Sxx[:nf, :nt], f[:nf], t[:nt], cropped

def get_event_files(events_folder: pathlib.Path):
    files = sorted(events_folder.glob("*.wav"))
    if not files:
        raise FileNotFoundError(f"No .wav files found in {events_folder}")
    return files

def enforce_marine_padding(mask, t, padding=1.0, probability_maps=None):
    """
    Enforce the marine prior that the first and last `padding` seconds
    are background-only regions.

    This is used only during marine inference/visualization.
    It should not be used for FUSS benchmark evaluation.
    """
    if probability_maps is None:
        probability_maps = []

    if padding is None or padding <= 0:
        return mask, probability_maps

    if t is None or len(t) == 0:
        return mask, probability_maps

    start_time = float(t[0])
    end_time = float(t[-1])

    edge_cols = (t <= start_time + padding) | (t >= end_time - padding)

    mask = mask.copy()
    mask[:, edge_cols] = False

    cleaned_maps = []
    for p in probability_maps:
        if p is None:
            cleaned_maps.append(None)
        else:
            p = p.copy()
            p[:, edge_cols] = 0.0
            cleaned_maps.append(p)

    return mask, cleaned_maps

def save_prediction_figure(path, Sxx, mask, prob, dsp_prob, f, t):
    spectrogram_db = to_plot_db(Sxx)

    fig, ax = plt.subplots(1, 5, figsize=(30, 6))

    mesh0 = ax[0].pcolormesh(t, f, spectrogram_db, shading="gouraud")
    ax[0].set_title("Spectrogram")
    ax[0].set_xlabel("Time [sec]")
    ax[0].set_ylabel("Frequency [Hz]")
    fig.colorbar(mesh0, ax=ax[0], label="Intensity [dB]")

    mesh1 = ax[1].imshow(
        dsp_prob,
        aspect="auto",
        origin="lower",
        cmap="viridis",
        extent=[t[0], t[-1], f[0], f[-1]],
        vmin=0.0,
        vmax=1.0,
    )
    ax[1].set_title("DSP-SPP Probability")
    ax[1].set_xlabel("Time [sec]")
    ax[1].set_ylabel("Frequency [Hz]")
    fig.colorbar(mesh1, ax=ax[1], label="Probability")

    mesh2 = ax[2].imshow(
        prob,
        aspect="auto",
        origin="lower",
        cmap="viridis",
        extent=[t[0], t[-1], f[0], f[-1]],
        vmin=0.0,
        vmax=1.0,
    )
    ax[2].set_title("Hybrid U-Net Probability")
    ax[2].set_xlabel("Time [sec]")
    ax[2].set_ylabel("Frequency [Hz]")
    fig.colorbar(mesh2, ax=ax[2], label="Probability")

    ax[3].imshow(
        mask,
        aspect="auto",
        origin="lower",
        cmap="Reds",
        extent=[t[0], t[-1], f[0], f[-1]],
    )
    ax[3].set_title("Binary Mask")
    ax[3].set_xlabel("Time [sec]")
    ax[3].set_ylabel("Frequency [Hz]")

    mesh4 = ax[4].pcolormesh(t, f, spectrogram_db, shading="gouraud")
    ax[4].imshow(
        mask,
        aspect="auto",
        origin="lower",
        cmap="Reds",
        alpha=0.35,
        extent=[t[0], t[-1], f[0], f[-1]],
        zorder=3,
    )
    ax[4].set_title("Masked Spectrogram")
    ax[4].set_xlabel("Time [sec]")
    ax[4].set_ylabel("Frequency [Hz]")
    fig.colorbar(mesh4, ax=ax[4], label="Intensity [dB]")

    for a in ax:
        a.set_xlim(t[0], t[-1])
        a.set_ylim(f[0], f[-1])

    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)

def main():
    config = load_config()

    checkpoint_path = get_config_path(config, "hybrid_dsp_dl_checkpoint")

    events_folder = get_config_path(config, "marine_events_folder")
    marine_outputs_dir = ensure_dir(get_config_path(config, "marine_outputs_folder"))

    masks_folder = marine_outputs_dir / "masks" / "hybrid_dsp_dl"
    prob_folder = marine_outputs_dir / "probabilities" / "hybrid_dsp_dl"
    plots_folder = marine_outputs_dir / "plots" / "hybrid_dsp_dl"

    masks_folder.mkdir(parents=True, exist_ok=True)
    prob_folder.mkdir(parents=True, exist_ok=True)
    plots_folder.mkdir(parents=True, exist_ok=True)

    model, metadata = hdl.load_checkpoint(checkpoint_path, device=DEVICE)

    resampled_sr = int(metadata.get("resampled_sr", RESAMPLED_SR))
    nperseg = int(metadata.get("nperseg", NPERSEG))
    noverlap = int(metadata.get("noverlap", NOVERLAP))
    normalize_mode = metadata.get("normalize_mode", "per_clip")
    threshold = float(THRESHOLD_OVERRIDE if THRESHOLD_OVERRIDE is not None else metadata.get("threshold", 0.5))

    dsp_noise_mode = metadata.get("dsp_noise_mode", "quantile")
    dsp_noise_quantile = float(metadata.get("dsp_noise_quantile", 0.2))
    dsp_final_threshold = float(metadata.get("dsp_final_threshold", 0.5))
    dsp_postprocess = bool(metadata.get("dsp_postprocess", True))

    event_files = get_event_files(events_folder)
    print(f"Device: {DEVICE}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Threshold: {threshold}")

    for i, file in enumerate(event_files, start=1):
        print(f"Processing {i}/{len(event_files)}: {file.name}")
        mask, prob, dsp_prob, Sxx, f, t = hdl.estimate_mask_file(
            file,
            model=model,
            nperseg=nperseg,
            noverlap=noverlap,
            resampled_sr=resampled_sr,
            threshold=threshold,
            normalize_mode=normalize_mode,
            device=DEVICE,
            postprocess=POSTPROCESS,
            min_region_size=MIN_REGION_SIZE,
            min_db=MIN_DB,
            return_axes=True,
            dsp_noise_mode=dsp_noise_mode,
            dsp_noise_quantile=dsp_noise_quantile,
            dsp_padding=float(config.get("processing", {}).get("padding", 1.0)),
            dsp_final_threshold=dsp_final_threshold,
            dsp_postprocess=dsp_postprocess,
        )

        padding = float(config.get("processing", {}).get("padding", 1.0))

        mask, cleaned = enforce_marine_padding(
            mask,
            t,
            padding=padding,
            probability_maps=[prob, dsp_prob],
        )

        prob, dsp_prob = cleaned

        f_plot, t_plot, Sxx_plot = compute_plot_spectrogram(file, resampled_sr, nperseg, noverlap)
        Sxx_plot, f_plot, t_plot, cropped = crop_to_common_grid(Sxx_plot, f_plot, t_plot, mask, prob, dsp_prob)
        mask_plot, prob_plot, dsp_prob_plot = cropped

        np.save(masks_folder / f"{file.stem}.npy", mask.astype(bool))
        np.save(prob_folder / f"{file.stem}.npy", prob.astype(np.float32))
        save_prediction_figure(plots_folder / f"{file.stem}.png", Sxx_plot, mask_plot, prob_plot, dsp_prob_plot, f_plot, t_plot)

    print("Done.")


if __name__ == "__main__":
    main()
