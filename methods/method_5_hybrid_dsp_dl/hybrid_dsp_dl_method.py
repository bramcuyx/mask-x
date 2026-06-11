"""
Method 5: Hybrid DSP-DL mask estimation.

This method uses the existing U-Net architecture but changes its input from one
channel to two channels:

    channel 1: normalized log-power spectrogram
    channel 2: DSP-SPP soft probability map

The pure DSP, pure ML, and pure DL files are not modified by this method.
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
from typing import Any, Dict, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

import dl_unet_method as dl
import dsp_spp_method as dsp


# Reuse pure-DL utilities where appropriate.
load_audio = dl.load_audio
compute_spectrogram = dl.compute_spectrogram
to_log_spectrogram = dl.to_log_spectrogram
normalize_log_spectrogram = dl.normalize_log_spectrogram
pad_to_multiple_tensor = dl.pad_to_multiple_tensor
crop_tensor_to_shape = dl.crop_tensor_to_shape
bce_dice_loss = dl.bce_dice_loss
postprocess_mask = dl.postprocess_mask


# -----------------------------------------------------------------------------
# Model and checkpoints
# -----------------------------------------------------------------------------


def create_model(metadata: Optional[Dict[str, Any]] = None) -> dl.SpectrogramUNet:
    metadata = {} if metadata is None else dict(metadata)
    metadata["in_channels"] = int(metadata.get("in_channels", 2))
    return dl.SpectrogramUNet(
        in_channels=int(metadata.get("in_channels", 2)),
        base_channels=int(metadata.get("base_channels", 16)),
        depth=int(metadata.get("depth", 3)),
        dropout=float(metadata.get("dropout", 0.0)),
    )


def save_checkpoint(path: str | pathlib.Path, model: nn.Module, metadata: Optional[Dict[str, Any]] = None) -> None:
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    metadata = {} if metadata is None else dict(metadata)
    metadata["in_channels"] = 2
    torch.save({"state_dict": model.state_dict(), "metadata": metadata}, str(path))


def load_checkpoint(path: str | pathlib.Path, device: str | torch.device = "cpu"):
    path = pathlib.Path(path)
    payload = torch.load(str(path), map_location=device)
    metadata = dict(payload.get("metadata", {}))
    metadata["in_channels"] = int(metadata.get("in_channels", 2))
    model = create_model(metadata)
    model.load_state_dict(payload["state_dict"])
    model.to(device)
    model.eval()
    return model, metadata


# -----------------------------------------------------------------------------
# DSP-guided input construction
# -----------------------------------------------------------------------------


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


def make_hybrid_input_from_sxx(
    Sxx: np.ndarray,
    dsp_prob: np.ndarray,
    normalize_mode: str = "per_clip",
    global_mean: Optional[float] = None,
    global_std: Optional[float] = None,
) -> np.ndarray:
    log_sxx = to_log_spectrogram(Sxx)

    if normalize_mode == "global":
        x_log, _, _ = normalize_log_spectrogram(log_sxx, mean=global_mean, std=global_std)
    elif normalize_mode == "per_clip":
        x_log, _, _ = normalize_log_spectrogram(log_sxx)
    else:
        raise ValueError("normalize_mode must be 'per_clip' or 'global'")

    nf = min(x_log.shape[0], dsp_prob.shape[0])
    nt = min(x_log.shape[1], dsp_prob.shape[1])

    x_log = x_log[:nf, :nt].astype(np.float32)
    dsp_prob = np.asarray(dsp_prob[:nf, :nt], dtype=np.float32)
    dsp_prob = np.clip(dsp_prob, 0.0, 1.0)

    return np.stack([x_log, dsp_prob], axis=0).astype(np.float32)


def make_hybrid_input_file(
    file: str | pathlib.Path,
    nperseg: int = 256,
    noverlap: int = 128,
    resampled_sr: Optional[int] = None,
    normalize_mode: str = "per_clip",
    global_mean: Optional[float] = None,
    global_std: Optional[float] = None,
    dsp_noise_mode: str = "quantile",
    dsp_noise_quantile: float = 0.2,
    dsp_padding: float = 1.0,
    dsp_final_threshold: float = 0.5,
    dsp_postprocess: bool = True,
):
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
    x = make_hybrid_input_from_sxx(
        Sxx,
        dsp_prob=dsp_prob,
        normalize_mode=normalize_mode,
        global_mean=global_mean,
        global_std=global_std,
    )
    return x, dsp_prob[: x.shape[1], : x.shape[2]], Sxx[: x.shape[1], : x.shape[2]], f, t


# -----------------------------------------------------------------------------
# Inference
# -----------------------------------------------------------------------------


@torch.no_grad()
def predict_probability_map_from_input(
    x: np.ndarray,
    model: nn.Module,
    device: str | torch.device = "cpu",
    pad_multiple: int = 16,
) -> np.ndarray:
    tensor = torch.from_numpy(x[None, :, :, :]).float().to(device)
    tensor, original_shape = pad_to_multiple_tensor(tensor, multiple=pad_multiple)
    logits = model(tensor)
    logits = crop_tensor_to_shape(logits, original_shape)
    prob = torch.sigmoid(logits)[0, 0].detach().cpu().numpy()
    return prob.astype(np.float32)


def estimate_mask_file(
    file: str | pathlib.Path,
    model: nn.Module,
    nperseg: int = 256,
    noverlap: int = 128,
    resampled_sr: Optional[int] = None,
    threshold: float = 0.5,
    device: str | torch.device = "cpu",
    normalize_mode: str = "per_clip",
    global_mean: Optional[float] = None,
    global_std: Optional[float] = None,
    postprocess: bool = False,
    min_region_size: int = 8,
    plot: bool = False,
    min_db: Optional[float] = None,
    return_axes: bool = False,
    dsp_noise_mode: str = "quantile",
    dsp_noise_quantile: float = 0.2,
    dsp_padding: float = 1.0,
    dsp_final_threshold: float = 0.5,
    dsp_postprocess: bool = True,
):
    x, dsp_prob, Sxx, f, t = make_hybrid_input_file(
        file,
        nperseg=nperseg,
        noverlap=noverlap,
        resampled_sr=resampled_sr,
        normalize_mode=normalize_mode,
        global_mean=global_mean,
        global_std=global_std,
        dsp_noise_mode=dsp_noise_mode,
        dsp_noise_quantile=dsp_noise_quantile,
        dsp_padding=dsp_padding,
        dsp_final_threshold=dsp_final_threshold,
        dsp_postprocess=dsp_postprocess,
    )
    prob = predict_probability_map_from_input(x, model=model, device=device, pad_multiple=16)
    mask = prob >= float(threshold)

    if postprocess:
        mask = postprocess_mask(mask, min_region_size=min_region_size)

    if plot:
        plot_prediction(Sxx, mask, prob, dsp_prob, f, t, min_db=min_db)

    if return_axes:
        return mask, prob, dsp_prob, Sxx, f, t
    return mask, prob, dsp_prob, Sxx


def plot_prediction(Sxx, mask, prob, dsp_prob, f, t, min_db=None):
    if min_db is None:
        spectrogram_db = 10.0 * np.log10(np.maximum(Sxx, np.finfo(np.float32).eps))
    else:
        power_floor = 10 ** (float(min_db) / 10.0)
        spectrogram_db = 10.0 * np.log10(np.maximum(Sxx, power_floor))

    fig, ax = plt.subplots(1, 5, figsize=(30, 6))

    mesh0 = ax[0].pcolormesh(t, f, spectrogram_db, shading="gouraud")
    ax[0].set_title("Spectrogram")
    ax[0].set_xlabel("Time [sec]")
    ax[0].set_ylabel("Frequency [Hz]")
    fig.colorbar(mesh0, ax=ax[0], label="Intensity [dB]")

    mesh1 = ax[1].imshow(dsp_prob, aspect="auto", origin="lower", cmap="viridis", extent=[t[0], t[-1], f[0], f[-1]], vmin=0.0, vmax=1.0)
    ax[1].set_title("DSP-SPP Probability")
    ax[1].set_xlabel("Time [sec]")
    ax[1].set_ylabel("Frequency [Hz]")
    fig.colorbar(mesh1, ax=ax[1], label="Probability")

    mesh2 = ax[2].imshow(prob, aspect="auto", origin="lower", cmap="viridis", extent=[t[0], t[-1], f[0], f[-1]], vmin=0.0, vmax=1.0)
    ax[2].set_title("Hybrid U-Net Probability")
    ax[2].set_xlabel("Time [sec]")
    ax[2].set_ylabel("Frequency [Hz]")
    fig.colorbar(mesh2, ax=ax[2], label="Probability")

    ax[3].imshow(mask, aspect="auto", origin="lower", cmap="Reds", extent=[t[0], t[-1], f[0], f[-1]])
    ax[3].set_title("Binary Mask")
    ax[3].set_xlabel("Time [sec]")
    ax[3].set_ylabel("Frequency [Hz]")

    mesh4 = ax[4].pcolormesh(t, f, spectrogram_db, shading="gouraud")
    ax[4].imshow(mask, aspect="auto", origin="lower", cmap="Reds", alpha=0.35, extent=[t[0], t[-1], f[0], f[-1]], zorder=3)
    ax[4].set_title("Masked Spectrogram")
    ax[4].set_xlabel("Time [sec]")
    ax[4].set_ylabel("Frequency [Hz]")
    fig.colorbar(mesh4, ax=ax[4], label="Intensity [dB]")

    for a in ax:
        a.set_xlim(t[0], t[-1])
        a.set_ylim(f[0], f[-1])

    plt.tight_layout()
    plt.show()
