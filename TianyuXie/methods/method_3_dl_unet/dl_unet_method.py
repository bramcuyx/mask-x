"""
Pure DL Method: Spectrogram U-Net for time-frequency mask estimation.

This file intentionally keeps the method pure DL:
    audio -> log-power spectrogram -> U-Net -> probability mask -> threshold

No DSP saliency map, LightGBM, CRF, or NMF is used by default.
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

import pathlib
from typing import Any, Dict, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import scipy.ndimage
import scipy.signal
import soundfile as sf
import torch
import torch.nn as nn
import torch.nn.functional as F


# -----------------------------------------------------------------------------
# Audio and spectrogram utilities
# -----------------------------------------------------------------------------


def load_audio(file: str | pathlib.Path, resampled_sr: Optional[int] = None) -> Tuple[np.ndarray, int]:
    """Load mono audio. If resampled_sr is provided, resample with scipy.signal.resample."""
    file = pathlib.Path(file)
    data, sample_rate = sf.read(str(file))

    if data.size == 0:
        raise ValueError(f"Audio file is empty: {file}")

    if data.ndim > 1:
        data = np.mean(data, axis=1)

    if resampled_sr is not None and sample_rate != resampled_sr:
        n_samples = round(len(data) * float(resampled_sr) / sample_rate)
        data = scipy.signal.resample(data, n_samples)
        sample_rate = int(resampled_sr)

    return data.astype(np.float32), sample_rate


def compute_spectrogram(
    data: np.ndarray,
    sample_rate: int,
    nperseg: int = 256,
    noverlap: int = 128,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return frequency axis, time axis, and power spectrogram [freq, time]."""
    f, t, Sxx = scipy.signal.spectrogram(
        data,
        fs=sample_rate,
        nperseg=nperseg,
        noverlap=noverlap,
        mode="psd",
    )
    Sxx = np.maximum(Sxx, np.finfo(np.float32).eps).astype(np.float32)
    return f, t, Sxx


def to_log_spectrogram(Sxx: np.ndarray) -> np.ndarray:
    """Convert power spectrogram to log-power dB scale."""
    return (10.0 * np.log10(np.maximum(Sxx, np.finfo(np.float32).eps))).astype(np.float32)


def normalize_log_spectrogram(
    log_sxx: np.ndarray,
    mean: Optional[float] = None,
    std: Optional[float] = None,
) -> Tuple[np.ndarray, float, float]:
    """
    Normalize one log spectrogram.

    If mean/std are None, per-clip normalization is used. This keeps inference on
    marine data simple and avoids depending on hidden training-set statistics.
    """
    x = np.asarray(log_sxx, dtype=np.float32)
    if mean is None:
        mean = float(x.mean())
    if std is None:
        std = float(x.std())
    std = max(float(std), 1e-6)
    return ((x - mean) / std).astype(np.float32), float(mean), float(std)


# -----------------------------------------------------------------------------
# U-Net model
# -----------------------------------------------------------------------------


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, dropout: float = 0.0):
        super().__init__()
        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        ]
        if dropout > 0:
            layers.append(nn.Dropout2d(p=dropout))
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class SpectrogramUNet(nn.Module):
    """
    Small U-Net for binary TF mask estimation.

    Input shape : [batch, 1, freq, time]
    Output shape: [batch, 1, freq, time] logits, not sigmoid probabilities
    """

    def __init__(
        self,
        in_channels: int = 1,
        base_channels: int = 16,
        depth: int = 3,
        dropout: float = 0.0,
    ):
        super().__init__()
        if depth < 2:
            raise ValueError("depth must be >= 2")

        self.in_channels = int(in_channels)
        self.base_channels = int(base_channels)
        self.depth = int(depth)
        self.dropout = float(dropout)

        channels = [base_channels * (2 ** i) for i in range(depth)]

        self.encoders = nn.ModuleList()
        prev_ch = in_channels
        for ch in channels:
            self.encoders.append(ConvBlock(prev_ch, ch, dropout=dropout))
            prev_ch = ch

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.bottleneck = ConvBlock(channels[-1], channels[-1] * 2, dropout=dropout)

        self.upconvs = nn.ModuleList()
        self.decoders = nn.ModuleList()
        decoder_in = channels[-1] * 2
        for skip_ch in reversed(channels):
            self.upconvs.append(nn.ConvTranspose2d(decoder_in, skip_ch, kernel_size=2, stride=2))
            self.decoders.append(ConvBlock(skip_ch * 2, skip_ch, dropout=dropout))
            decoder_in = skip_ch

        self.out_conv = nn.Conv2d(channels[0], 1, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skips = []
        for encoder in self.encoders:
            x = encoder(x)
            skips.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)

        for upconv, decoder, skip in zip(self.upconvs, self.decoders, reversed(skips)):
            x = upconv(x)

            # Handle odd input dimensions by padding/cropping the upsampled tensor.
            diff_freq = skip.shape[-2] - x.shape[-2]
            diff_time = skip.shape[-1] - x.shape[-1]
            if diff_freq != 0 or diff_time != 0:
                x = F.pad(
                    x,
                    [diff_time // 2, diff_time - diff_time // 2, diff_freq // 2, diff_freq - diff_freq // 2],
                )

            x = torch.cat([skip, x], dim=1)
            x = decoder(x)

        return self.out_conv(x)


def create_model(metadata: Optional[Dict[str, Any]] = None) -> SpectrogramUNet:
    """Create a U-Net from optional checkpoint metadata."""
    metadata = {} if metadata is None else dict(metadata)
    return SpectrogramUNet(
        in_channels=int(metadata.get("in_channels", 1)),
        base_channels=int(metadata.get("base_channels", 16)),
        depth=int(metadata.get("depth", 3)),
        dropout=float(metadata.get("dropout", 0.0)),
    )


# -----------------------------------------------------------------------------
# Tensor helpers, loss, and checkpoints
# -----------------------------------------------------------------------------


def pad_to_multiple_tensor(x: torch.Tensor, multiple: int = 16) -> Tuple[torch.Tensor, Tuple[int, int]]:
    """Pad tensor [..., freq, time] to multiples of `multiple`. Returns padded tensor and original shape."""
    original_shape = (int(x.shape[-2]), int(x.shape[-1]))
    pad_freq = (multiple - original_shape[0] % multiple) % multiple
    pad_time = (multiple - original_shape[1] % multiple) % multiple
    if pad_freq > 0 or pad_time > 0:
        x = F.pad(x, [0, pad_time, 0, pad_freq])
    return x, original_shape


def crop_tensor_to_shape(x: torch.Tensor, shape: Tuple[int, int]) -> torch.Tensor:
    """Crop tensor [..., freq, time] to shape."""
    return x[..., : shape[0], : shape[1]]


def dice_loss_from_logits(logits: torch.Tensor, targets: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    probs = torch.sigmoid(logits)
    targets = targets.float()
    dims = tuple(range(1, probs.ndim))
    intersection = torch.sum(probs * targets, dim=dims)
    union = torch.sum(probs, dim=dims) + torch.sum(targets, dim=dims)
    dice = (2.0 * intersection + eps) / (union + eps)
    return 1.0 - dice.mean()


def bce_dice_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    bce_weight: float = 1.0,
    dice_weight: float = 1.0,
    pos_weight: Optional[float] = None,
) -> torch.Tensor:
    if pos_weight is not None:
        pw = torch.tensor(float(pos_weight), device=logits.device, dtype=logits.dtype)
        bce = F.binary_cross_entropy_with_logits(logits, targets.float(), pos_weight=pw)
    else:
        bce = F.binary_cross_entropy_with_logits(logits, targets.float())
    dsc = dice_loss_from_logits(logits, targets)
    return float(bce_weight) * bce + float(dice_weight) * dsc


def save_checkpoint(
    path: str | pathlib.Path,
    model: nn.Module,
    metadata: Optional[Dict[str, Any]] = None,
) -> None:
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "state_dict": model.state_dict(),
        "metadata": {} if metadata is None else dict(metadata),
    }
    torch.save(payload, str(path))


def load_checkpoint(
    path: str | pathlib.Path,
    device: str | torch.device = "cpu",
) -> Tuple[SpectrogramUNet, Dict[str, Any]]:
    path = pathlib.Path(path)
    payload = torch.load(str(path), map_location=device)
    metadata = dict(payload.get("metadata", {}))
    model = create_model(metadata)
    model.load_state_dict(payload["state_dict"])
    model.to(device)
    model.eval()
    return model, metadata


# -----------------------------------------------------------------------------
# Prediction and mask export
# -----------------------------------------------------------------------------


@torch.no_grad()
def predict_probability_map_from_sxx(
    Sxx: np.ndarray,
    model: nn.Module,
    device: str | torch.device = "cpu",
    normalize_mode: str = "per_clip",
    global_mean: Optional[float] = None,
    global_std: Optional[float] = None,
    pad_multiple: int = 16,
) -> np.ndarray:
    """Predict probability map [freq, time] from a power spectrogram."""
    log_sxx = to_log_spectrogram(Sxx)

    if normalize_mode == "global":
        x, _, _ = normalize_log_spectrogram(log_sxx, mean=global_mean, std=global_std)
    elif normalize_mode == "per_clip":
        x, _, _ = normalize_log_spectrogram(log_sxx)
    else:
        raise ValueError("normalize_mode must be 'per_clip' or 'global'")

    tensor = torch.from_numpy(x[None, None, :, :]).float().to(device)
    tensor, original_shape = pad_to_multiple_tensor(tensor, multiple=pad_multiple)

    logits = model(tensor)
    logits = crop_tensor_to_shape(logits, original_shape)
    prob = torch.sigmoid(logits)[0, 0].detach().cpu().numpy()
    return prob.astype(np.float32)


def postprocess_mask(mask: np.ndarray, min_region_size: Optional[int] = None) -> np.ndarray:
    """
    Optional tiny-component removal.

    Keep disabled in quantitative pure-DL experiments unless explicitly stated.
    """
    mask = mask.astype(bool).copy()
    if min_region_size is None or int(min_region_size) <= 1:
        return mask

    structure = np.ones((3, 3), dtype=int)
    labeled, num = scipy.ndimage.label(mask, structure=structure)
    if num > 0:
        sizes = np.bincount(labeled.ravel())
        for label_id in range(1, len(sizes)):
            if sizes[label_id] < int(min_region_size):
                mask[labeled == label_id] = False
    return mask


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
):
    """
    Predict a binary TF mask for one wav file.

    Returns:
        mask : [freq, time] bool
        prob : [freq, time] float32 probability map
        Sxx  : [freq, time] power spectrogram
        f, t : axes if return_axes=True
    """
    data, sample_rate = load_audio(file, resampled_sr=resampled_sr)
    f, t, Sxx = compute_spectrogram(data, sample_rate, nperseg=nperseg, noverlap=noverlap)

    prob = predict_probability_map_from_sxx(
        Sxx,
        model=model,
        device=device,
        normalize_mode=normalize_mode,
        global_mean=global_mean,
        global_std=global_std,
    )
    mask = prob >= float(threshold)

    if postprocess:
        mask = postprocess_mask(mask, min_region_size=min_region_size)

    if plot:
        plot_prediction(Sxx, mask, prob, f, t, min_db=min_db)

    if return_axes:
        return mask, prob, Sxx, f, t
    return mask, prob, Sxx


def plot_prediction(
    Sxx: np.ndarray,
    mask: np.ndarray,
    prob: np.ndarray,
    f: np.ndarray,
    t: np.ndarray,
    min_db: Optional[float] = None,
):
    if min_db is None:
        spectrogram_db = 10.0 * np.log10(np.maximum(Sxx, np.finfo(np.float32).eps))
    else:
        power_floor = 10 ** (float(min_db) / 10.0)
        spectrogram_db = 10.0 * np.log10(np.maximum(Sxx, power_floor))

    fig, ax = plt.subplots(1, 4, figsize=(24, 6))

    mesh0 = ax[0].pcolormesh(t, f, spectrogram_db, shading="gouraud")
    ax[0].set_title("Spectrogram")
    ax[0].set_xlabel("Time [sec]")
    ax[0].set_ylabel("Frequency [Hz]")
    fig.colorbar(mesh0, ax=ax[0], label="Intensity [dB]")

    mesh1 = ax[1].imshow(
        prob,
        aspect="auto",
        origin="lower",
        cmap="viridis",
        extent=[t[0], t[-1], f[0], f[-1]],
        vmin=0.0,
        vmax=1.0,
    )
    ax[1].set_title("U-Net Probability Mask")
    ax[1].set_xlabel("Time [sec]")
    ax[1].set_ylabel("Frequency [Hz]")
    fig.colorbar(mesh1, ax=ax[1], label="Probability")

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

    mesh3 = ax[3].pcolormesh(t, f, spectrogram_db, shading="gouraud")
    ax[3].imshow(
        mask,
        aspect="auto",
        origin="lower",
        cmap="Reds",
        alpha=0.35,
        extent=[t[0], t[-1], f[0], f[-1]],
        zorder=3,
    )
    ax[3].set_title("Masked Spectrogram")
    ax[3].set_xlabel("Time [sec]")
    ax[3].set_ylabel("Frequency [Hz]")
    fig.colorbar(mesh3, ax=ax[3], label="Intensity [dB]")

    for a in ax:
        a.set_xlim(t[0], t[-1])
        a.set_ylim(f[0], f[-1])

    plt.tight_layout()
    plt.show()
