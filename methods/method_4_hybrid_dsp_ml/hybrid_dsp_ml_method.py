"""
Method 4: Hybrid DSP-ML mask estimation.

This method keeps the existing LightGBM + optional 2D CRF pipeline, but augments
its tabular spectrogram features with DSP-SPP guidance features:

    [pure ML handcrafted features] + [DSP soft probability, DSP binary mask, DSP local density]

The existing Pure DSP, Pure ML, and Pure DL files are not modified by this method.
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
from typing import Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import scipy.ndimage

import dsp_spp_method as dsp
import ml_lgbm_crf_method as ml


# -----------------------------------------------------------------------------
# DSP-guided feature extraction
# -----------------------------------------------------------------------------


def _safe_binary_density(mask: np.ndarray, size: Tuple[int, int] = (5, 5)) -> np.ndarray:
    mask_f = mask.astype(np.float32)
    return scipy.ndimage.uniform_filter(mask_f, size=size, mode="nearest").astype(np.float32)


def _extract_hybrid_feature_cube(log_sxx: np.ndarray, dsp_prob: np.ndarray, dsp_mask: np.ndarray) -> np.ndarray:
    """
    Return feature tensor [freq, time, n_features].

    Base features come from the pure ML method. DSP features are appended as
    additional channels so the model can learn when to trust or reject DSP-SPP
    candidate regions.
    """
    base_cube = ml._extract_feature_cube(log_sxx)

    nf = min(base_cube.shape[0], dsp_prob.shape[0], dsp_mask.shape[0])
    nt = min(base_cube.shape[1], dsp_prob.shape[1], dsp_mask.shape[1])

    base_cube = base_cube[:nf, :nt, :]
    dsp_prob = np.asarray(dsp_prob[:nf, :nt], dtype=np.float32)
    dsp_mask = np.asarray(dsp_mask[:nf, :nt], dtype=np.float32)
    dsp_density = _safe_binary_density(dsp_mask, size=(5, 5))

    # Local statistics of the DSP probability map.
    dsp_prob_mean3 = scipy.ndimage.uniform_filter(dsp_prob, size=(3, 3), mode="nearest").astype(np.float32)
    dsp_prob_contrast3 = (dsp_prob - dsp_prob_mean3).astype(np.float32)

    dsp_cube = np.stack(
        [
            dsp_prob,
            dsp_mask,
            dsp_density,
            dsp_prob_mean3,
            dsp_prob_contrast3,
        ],
        axis=-1,
    ).astype(np.float32)

    return np.concatenate([base_cube, dsp_cube], axis=-1).astype(np.float32)


def compute_dsp_guidance_file(
    file: str | pathlib.Path,
    nperseg: int = 256,
    noverlap: int = 128,
    resampled_sr: Optional[int] = None,
    dsp_noise_mode: str = "quantile",
    dsp_noise_quantile: float = 0.2,
    dsp_padding: float = 1.0,
    dsp_final_threshold: float = 0.5,
    dsp_postprocess: bool = True,
):
    """
    Run the existing DSP-SPP estimator and return mask/probability/spectrogram.
    """
    mask, soft_prob, _, Sxx, f, t = dsp.estimate_mask_file(
        file,
        nperseg=nperseg,
        noverlap=noverlap,
        padding=dsp_padding,
        plot=False,
        resampled_sr=resampled_sr,
        return_axes=True,
        noise_mode=dsp_noise_mode,
        noise_quantile=dsp_noise_quantile,
        final_threshold=dsp_final_threshold,
        postprocess=dsp_postprocess,
    )
    return mask.astype(bool), soft_prob.astype(np.float32), Sxx.astype(np.float32), f, t


def extract_feature_matrix_file(
    file: str | pathlib.Path,
    nperseg: int = 256,
    noverlap: int = 128,
    resampled_sr: Optional[int] = None,
    dsp_noise_mode: str = "quantile",
    dsp_noise_quantile: float = 0.2,
    dsp_padding: float = 1.0,
    dsp_final_threshold: float = 0.5,
    dsp_postprocess: bool = True,
):
    """
    Returns:
        X   : [n_bins, n_features]
        Sxx : [freq, time]
        f,t : axes
    """
    dsp_mask, dsp_prob, Sxx, f, t = compute_dsp_guidance_file(
        file,
        nperseg=nperseg,
        noverlap=noverlap,
        resampled_sr=resampled_sr,
        dsp_noise_mode=dsp_noise_mode,
        dsp_noise_quantile=dsp_noise_quantile,
        dsp_padding=dsp_padding,
        dsp_final_threshold=dsp_final_threshold,
        dsp_postprocess=dsp_postprocess,
    )
    log_sxx = ml._to_log_spectrogram(Sxx)
    feature_cube = _extract_hybrid_feature_cube(log_sxx, dsp_prob=dsp_prob, dsp_mask=dsp_mask)
    X = feature_cube.reshape(-1, feature_cube.shape[-1]).astype(np.float32)
    return X, Sxx[: feature_cube.shape[0], : feature_cube.shape[1]], f, t


def _predict_probability_map_from_sxx_and_dsp(Sxx: np.ndarray, dsp_prob: np.ndarray, dsp_mask: np.ndarray, model) -> np.ndarray:
    log_sxx = ml._to_log_spectrogram(Sxx)
    feature_cube = _extract_hybrid_feature_cube(log_sxx, dsp_prob=dsp_prob, dsp_mask=dsp_mask)
    n_freq, n_time, n_features = feature_cube.shape
    X = feature_cube.reshape(-1, n_features).astype(np.float32)
    prob = model.predict_proba(X)[:, 1].reshape(n_freq, n_time)
    return prob.astype(np.float32)


# -----------------------------------------------------------------------------
# Model I/O wrappers
# -----------------------------------------------------------------------------


def train_lgbm_classifier(*args, **kwargs):
    return ml.train_lgbm_classifier(*args, **kwargs)


def save_model_bundle(path, model, metadata=None):
    return ml.save_model_bundle(path, model, metadata=metadata)


def load_model_bundle(path):
    return ml.load_model_bundle(path)


# -----------------------------------------------------------------------------
# Inference and plotting
# -----------------------------------------------------------------------------


def estimate_mask_file(
    file: str | pathlib.Path,
    model,
    nperseg: int = 256,
    noverlap: int = 128,
    plot: bool = False,
    resampled_sr: Optional[int] = None,
    min_db: Optional[float] = None,
    return_axes: bool = False,
    threshold: float = 0.5,
    apply_crf: bool = True,
    crf_spatial_sigma: Tuple[float, float] = (1.0, 1.0),
    crf_pairwise_weight: float = 1.2,
    crf_unary_weight: float = 1.0,
    crf_iterations: int = 5,
    postprocess: bool = True,
    min_region_size: int = 8,
    dsp_noise_mode: str = "quantile",
    dsp_noise_quantile: float = 0.2,
    dsp_padding: float = 1.0,
    dsp_final_threshold: float = 0.5,
    dsp_postprocess: bool = True,
):
    """
    Predict a binary TF mask using DSP-guided LightGBM + optional CRF.

    Returns:
        mask         : final binary mask
        raw_prob     : LightGBM probability map
        decoded_prob : CRF-refined probability map
        dsp_prob     : DSP-SPP guidance probability map
        Sxx          : spectrogram
        f, t         : axes if return_axes=True
    """
    dsp_mask, dsp_prob, Sxx, f, t = compute_dsp_guidance_file(
        file,
        nperseg=nperseg,
        noverlap=noverlap,
        resampled_sr=resampled_sr,
        dsp_noise_mode=dsp_noise_mode,
        dsp_noise_quantile=dsp_noise_quantile,
        dsp_padding=dsp_padding,
        dsp_final_threshold=dsp_final_threshold,
        dsp_postprocess=dsp_postprocess,
    )

    raw_prob = _predict_probability_map_from_sxx_and_dsp(Sxx, dsp_prob=dsp_prob, dsp_mask=dsp_mask, model=model)

    if apply_crf:
        decoded_prob = ml._crf_mean_field_binary(
            raw_prob,
            spatial_sigma=crf_spatial_sigma,
            pairwise_weight=crf_pairwise_weight,
            unary_weight=crf_unary_weight,
            n_iters=crf_iterations,
        )
    else:
        decoded_prob = raw_prob

    mask = decoded_prob >= float(threshold)

    if postprocess:
        mask = ml._postprocess_mask(mask, min_region_size=min_region_size)

    if plot:
        _plot_masked_spect(Sxx, mask, raw_prob, decoded_prob, dsp_prob, f, t, min_db=min_db)

    if return_axes:
        return mask, raw_prob, decoded_prob, dsp_prob, Sxx, f, t

    return mask, raw_prob, decoded_prob, dsp_prob, Sxx


def _plot_masked_spect(Sxx, mask, raw_prob, decoded_prob, dsp_prob, f, t, min_db=None):
    if min_db is None:
        spectrogram_db = 10.0 * np.log10(np.maximum(Sxx, np.finfo(float).eps))
    else:
        power_floor = 10 ** (float(min_db) / 10.0)
        spectrogram_db = 10.0 * np.log10(np.maximum(Sxx, power_floor))

    fig, ax = plt.subplots(1, 5, figsize=(30, 6))

    mesh0 = ax[0].pcolormesh(t, f, spectrogram_db, shading="gouraud")
    ax[0].set_title("Spectrogram")
    ax[0].set_xlabel("Time [sec]")
    ax[0].set_ylabel("Frequency [Hz]")
    fig.colorbar(mesh0, ax=ax[0], label="Intensity [dB]")

    for idx, (arr, title) in enumerate(
        [(dsp_prob, "DSP-SPP Probability"), (raw_prob, "Hybrid LightGBM Probability"), (decoded_prob, "CRF Probability")],
        start=1,
    ):
        mesh = ax[idx].imshow(
            arr,
            aspect="auto",
            origin="lower",
            cmap="viridis",
            extent=[t[0], t[-1], f[0], f[-1]],
            vmin=0.0,
            vmax=1.0,
        )
        ax[idx].set_title(title)
        ax[idx].set_xlabel("Time [sec]")
        ax[idx].set_ylabel("Frequency [Hz]")
        fig.colorbar(mesh, ax=ax[idx], label="Probability")

    mesh4 = ax[4].pcolormesh(t, f, spectrogram_db, shading="gouraud")
    ax[4].imshow(mask, aspect="auto", origin="lower", cmap="Reds", alpha=0.35, extent=[t[0], t[-1], f[0], f[-1]], zorder=3)
    ax[4].set_title("Final Binary Mask")
    ax[4].set_xlabel("Time [sec]")
    ax[4].set_ylabel("Frequency [Hz]")
    fig.colorbar(mesh4, ax=ax[4], label="Intensity [dB]")

    for a in ax:
        a.set_xlim(t[0], t[-1])
        a.set_ylim(f[0], f[-1])

    plt.tight_layout()
    plt.show()
