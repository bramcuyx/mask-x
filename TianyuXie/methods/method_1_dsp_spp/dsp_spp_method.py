
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

import pathlib

import matplotlib.pyplot as plt
import numpy as np
import scipy.ndimage
import scipy.signal
import soundfile as sf
import scipy.ndimage


def _postprocess_mask(
    mask,
    min_region_size=8,
    max_fullband_ratio=0.9,
    max_impulse_width_frames=2,
):
    """
    Post-processing for binary masks:
    1) remove tiny connected components
    2) suppress very short nearly full-band vertical impulses
    """
    mask = mask.astype(bool).copy()

    # ---- 1) remove tiny connected components ----
    structure = np.ones((3, 3), dtype=int)
    labeled, num = scipy.ndimage.label(mask, structure=structure)

    if num > 0:
        component_sizes = np.bincount(labeled.ravel())
        for label_id in range(1, len(component_sizes)):
            if component_sizes[label_id] < min_region_size:
                mask[labeled == label_id] = False

    # ---- 2) suppress very short full-band impulses ----
    if mask.shape[1] > 0:
        frame_band_ratio = mask.sum(axis=0) / max(mask.shape[0], 1)
        fullband_frames = frame_band_ratio >= max_fullband_ratio

        # 连续 fullband 段
        labeled_1d, num_1d = scipy.ndimage.label(fullband_frames.astype(int))
        if num_1d > 0:
            counts = np.bincount(labeled_1d.ravel())
            for seg_id in range(1, len(counts)):
                if counts[seg_id] <= max_impulse_width_frames:
                    cols = labeled_1d == seg_id
                    mask[:, cols] = False

    return mask

def _load_audio(file, resampled_sr=None):
    data, sample_rate = sf.read(file)
    if data.size == 0:
        raise ValueError("Audio file is empty.")
    if data.ndim > 1:
        data = np.mean(data, axis=1)

    if resampled_sr is not None and sample_rate != resampled_sr:
        number_of_samples = round(len(data) * float(resampled_sr) / sample_rate)
        data = scipy.signal.resample(data, number_of_samples)
        sample_rate = resampled_sr

    return data.astype(float), sample_rate


def _compute_spectrogram(data, sample_rate, nperseg=256, noverlap=128):
    f, t, Sxx = scipy.signal.spectrogram(
        data,
        fs=sample_rate,
        nperseg=nperseg,
        noverlap=noverlap,
    )
    Sxx = np.maximum(Sxx, np.finfo(float).eps)
    return f, t, Sxx


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -50.0, 50.0)))


def _estimate_noise_statistics(
    log_sxx,
    t,
    padding=1.0,
    noise_mode="edge",
    noise_quantile=0.2,
):
    """
    Estimate per-frequency background statistics.

    noise_mode="edge":
        Use left/right padding regions as background. Best for marine data.
    noise_mode="quantile":
        Use low-energy frames across the whole clip. Best for FUSS benchmark.
    """
    n_freq, n_time = log_sxx.shape

    if noise_mode == "edge" and padding > 0:
        start_time = float(t[0])
        end_time = float(t[-1])

        edge_mask = (t <= start_time + padding) | (t >= end_time - padding)

        # If edge region is too small, fall back to quantile mode
        if int(edge_mask.sum()) >= max(4, int(0.05 * n_time)):
            noise_region = log_sxx[:, edge_mask]
            noise_median = np.median(noise_region, axis=1, keepdims=True)
            noise_mad = np.median(np.abs(noise_region - noise_median), axis=1, keepdims=True)
            noise_sigma = 1.4826 * noise_mad + 1e-3
            return noise_median, noise_sigma

    # Fallback / benchmark mode: low-energy frames
    q = np.quantile(log_sxx, noise_quantile, axis=1, keepdims=True)
    low_energy_mask = log_sxx <= q

    noise_median = np.empty((n_freq, 1), dtype=float)
    noise_sigma = np.empty((n_freq, 1), dtype=float)

    for k in range(n_freq):
        vals = log_sxx[k, low_energy_mask[k]]
        if vals.size == 0:
            vals = log_sxx[k, :]
        med = np.median(vals)
        mad = np.median(np.abs(vals - med))
        noise_median[k, 0] = med
        noise_sigma[k, 0] = 1.4826 * mad + 1e-3

    return noise_median, noise_sigma


def _plot_masked_spect(Sxx, mask, soft_spp, f, t, min_db=None):
    if min_db is None:
        spectrogram_db = 10 * np.log10(np.maximum(Sxx, np.finfo(float).eps))
    else:
        power_floor = 10 ** (min_db / 10.0)
        spectrogram_db = 10 * np.log10(np.maximum(Sxx, power_floor))

    fig, ax = plt.subplots(1, 4, figsize=(24, 6))

    mesh0 = ax[0].pcolormesh(t, f, spectrogram_db, shading="gouraud")
    ax[0].set_title("Spectrogram")
    ax[0].set_xlabel("Time [sec]")
    ax[0].set_ylabel("Frequency [Hz]")
    fig.colorbar(mesh0, ax=ax[0], label="Intensity [dB]")

    mesh1 = ax[1].pcolormesh(t, f, spectrogram_db, shading="gouraud")
    ax[1].imshow(
        mask,
        aspect="auto",
        origin="lower",
        cmap="Reds",
        alpha=0.3,
        extent=[t[0], t[-1], f[0], f[-1]],
        zorder=3,
    )
    ax[1].set_title("Masked Spectrogram")
    ax[1].set_xlabel("Time [sec]")
    ax[1].set_ylabel("Frequency [Hz]")
    fig.colorbar(mesh1, ax=ax[1], label="Intensity [dB]")

    ax[2].imshow(
        mask,
        aspect="auto",
        origin="lower",
        cmap="Reds",
        extent=[t[0], t[-1], f[0], f[-1]],
    )
    ax[2].set_title("Binary Mask")
    ax[2].set_xlabel("Time [sec]")
    ax[2].set_ylabel("Frequency [Hz]")

    mesh3 = ax[3].imshow(
        soft_spp,
        aspect="auto",
        origin="lower",
        cmap="viridis",
        extent=[t[0], t[-1], f[0], f[-1]],
        vmin=0.0,
        vmax=1.0,
    )
    ax[3].set_title("Soft SPP")
    ax[3].set_xlabel("Time [sec]")
    ax[3].set_ylabel("Frequency [Hz]")
    fig.colorbar(mesh3, ax=ax[3], label="Probability")

    for a in ax:
        a.set_xlim(t[0], t[-1])
        a.set_ylim(f[0], f[-1])

    plt.tight_layout()
    plt.show()


def estimate_mask_file(
    file,
    nperseg=256,
    noverlap=128,
    padding=1.0,
    plot=False,
    resampled_sr=None,
    min_db=None,
    return_axes=False,
    noise_mode="edge",
    noise_quantile=0.2,
    spp_alpha=1.5,
    spp_beta=2.0,
    smooth_kernel=(3, 3),
    final_threshold=0.5,
    postprocess=True,
    min_region_size=8,
    max_fullband_ratio=0.9,
    max_impulse_width_frames=2,
):
    """
    Basic DSP SPP estimator:
      1) STFT / spectrogram
      2) estimate background statistics per frequency
      3) compute z-score saliency
      4) map to soft SPP with sigmoid
      5) smooth in TF
      6) threshold to binary mask
    """
    file = pathlib.Path(file)
    data, sample_rate = _load_audio(file, resampled_sr=resampled_sr)
    f, t, Sxx = _compute_spectrogram(
        data,
        sample_rate,
        nperseg=nperseg,
        noverlap=noverlap,
    )

    log_sxx = 10.0 * np.log10(np.maximum(Sxx, np.finfo(float).eps))

    noise_median_db, noise_sigma_db = _estimate_noise_statistics(
        log_sxx,
        t,
        padding=padding,
        noise_mode=noise_mode,
        noise_quantile=noise_quantile,
    )
    z_raw = (log_sxx - noise_median_db) / noise_sigma_db
    z_local = scipy.ndimage.uniform_filter(z_raw, size=(3, 3))
    z_map = 0.8 * z_raw + 0.2 * z_local

    soft_spp = _sigmoid(spp_alpha * (z_map - spp_beta))

    if smooth_kernel is not None:
        soft_spp = scipy.ndimage.median_filter(soft_spp, size=smooth_kernel)

    mask = soft_spp >= final_threshold

    if postprocess:
        mask = _postprocess_mask(
            mask,
            min_region_size=min_region_size,
            max_fullband_ratio=max_fullband_ratio,
            max_impulse_width_frames=max_impulse_width_frames,
        )

    # If we truly know edge regions are background (marine case), enforce it
    if noise_mode == "edge" and padding > 0:
        start_time = float(t[0])
        end_time = float(t[-1])
        edge_mask = (t <= start_time + padding) | (t >= end_time - padding)
        mask[:, edge_mask] = 0
        soft_spp[:, edge_mask] = 0.0

    if plot:
        _plot_masked_spect(Sxx, mask, soft_spp, f, t, min_db=min_db)

    if return_axes:
        return mask, soft_spp, noise_median_db, Sxx, f, t

    return mask, soft_spp, noise_median_db, Sxx