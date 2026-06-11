
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
import pickle

import matplotlib.pyplot as plt
import numpy as np
import scipy.ndimage
import scipy.signal
import soundfile as sf
from lightgbm import LGBMClassifier, early_stopping, log_evaluation


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

    return data.astype(np.float32), sample_rate


def _compute_spectrogram(data, sample_rate, nperseg=256, noverlap=128):
    f, t, Sxx = scipy.signal.spectrogram(
        data,
        fs=sample_rate,
        nperseg=nperseg,
        noverlap=noverlap,
        mode="psd",
    )
    Sxx = np.maximum(Sxx, np.finfo(float).eps).astype(np.float32)
    return f, t, Sxx


def _to_log_spectrogram(Sxx):
    return (10.0 * np.log10(np.maximum(Sxx, np.finfo(float).eps))).astype(np.float32)


def _local_mean_std(arr, size):
    mean = scipy.ndimage.uniform_filter(arr, size=size, mode="nearest")
    mean_sq = scipy.ndimage.uniform_filter(arr * arr, size=size, mode="nearest")
    var = np.maximum(mean_sq - mean * mean, 0.0)
    std = np.sqrt(var)
    return mean.astype(np.float32), std.astype(np.float32)


def _extract_feature_cube(log_sxx):
    """
    Build a feature tensor of shape [freq, time, n_features].
    All features are classical hand-crafted tabular features.
    """
    eps = 1e-6
    x = log_sxx.astype(np.float32)

    # Per-frequency normalization
    freq_mean = x.mean(axis=1, keepdims=True)
    freq_std = x.std(axis=1, keepdims=True) + eps
    z_freq = (x - freq_mean) / freq_std

    # Per-time normalization
    time_mean = x.mean(axis=0, keepdims=True)
    time_std = x.std(axis=0, keepdims=True) + eps
    z_time = (x - time_mean) / time_std

    # Local statistics
    mean3, std3 = _local_mean_std(x, size=(3, 3))
    mean5, std5 = _local_mean_std(x, size=(5, 5))

    contrast3 = x - mean3
    contrast5 = x - mean5

    # Gradients
    grad_f, grad_t = np.gradient(x)
    abs_grad_f = np.abs(grad_f)
    abs_grad_t = np.abs(grad_t)

    # Position features
    n_freq, n_time = x.shape
    f_pos = np.linspace(0.0, 1.0, n_freq, dtype=np.float32)[:, None]
    t_pos = np.linspace(0.0, 1.0, n_time, dtype=np.float32)[None, :]
    f_pos = np.broadcast_to(f_pos, x.shape)
    t_pos = np.broadcast_to(t_pos, x.shape)

    features = np.stack(
        [
            x,
            z_freq,
            z_time,
            mean3,
            std3,
            mean5,
            std5,
            contrast3,
            contrast5,
            grad_f,
            grad_t,
            abs_grad_f,
            abs_grad_t,
            f_pos,
            t_pos,
        ],
        axis=-1,
    )

    return features.astype(np.float32)


def extract_feature_matrix_file(
    file,
    nperseg=256,
    noverlap=128,
    resampled_sr=None,
):
    """
    Returns:
        X      : [n_bins, n_features]
        Sxx    : [n_freq, n_time]
        f, t   : axes
    """
    file = pathlib.Path(file)
    data, sample_rate = _load_audio(file, resampled_sr=resampled_sr)
    f, t, Sxx = _compute_spectrogram(
        data,
        sample_rate,
        nperseg=nperseg,
        noverlap=noverlap,
    )
    log_sxx = _to_log_spectrogram(Sxx)
    feature_cube = _extract_feature_cube(log_sxx)
    X = feature_cube.reshape(-1, feature_cube.shape[-1]).astype(np.float32)
    return X, Sxx, f, t


def train_lgbm_classifier(
    X_train,
    y_train,
    X_val=None,
    y_val=None,
    random_state=42,
    n_estimators=400,
    learning_rate=0.05,
    num_leaves=63,
    min_child_samples=20,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_lambda=1.0,
):
    X_train = np.asarray(X_train, dtype=np.float32)
    y_train = np.asarray(y_train, dtype=np.uint8)

    if X_val is not None:
        X_val = np.asarray(X_val, dtype=np.float32)
    if y_val is not None:
        y_val = np.asarray(y_val, dtype=np.uint8)
    model = LGBMClassifier(
        objective="binary",
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        num_leaves=num_leaves,
        min_child_samples=min_child_samples,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        reg_lambda=reg_lambda,
        class_weight="balanced",
        random_state=random_state,
        n_jobs=-1,
        verbose=-1,
    )

    if X_val is not None and y_val is not None and len(X_val) > 0:
        model.fit(
            X_train,
            y_train,
            eval_set=[(X_val, y_val)],
            eval_metric="binary_logloss",
            callbacks=[
                early_stopping(stopping_rounds=30, verbose=True),
                log_evaluation(period=20),
            ],
        )
    else:
        model.fit(X_train, y_train)

    return model


def save_model_bundle(path, model, metadata=None):
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model": model,
        "metadata": {} if metadata is None else metadata,
    }
    with path.open("wb") as f:
        pickle.dump(payload, f)


def load_model_bundle(path):
    path = pathlib.Path(path)
    with path.open("rb") as f:
        payload = pickle.load(f)

    if isinstance(payload, dict) and "model" in payload:
        return payload

    return {
        "model": payload,
        "metadata": {},
    }


def _predict_probability_map_from_sxx(Sxx, model):
    log_sxx = _to_log_spectrogram(Sxx)
    feature_cube = _extract_feature_cube(log_sxx)
    n_freq, n_time, n_features = feature_cube.shape
    X = np.asarray(feature_cube.reshape(-1, n_features), dtype=np.float32)
    prob = model.predict_proba(X)[:, 1].reshape(n_freq, n_time)
    return prob.astype(np.float32)


def _crf_mean_field_binary(
    prob_map,
    spatial_sigma=(1.0, 1.0),
    pairwise_weight=1.2,
    unary_weight=1.0,
    n_iters=5,
):
    """
    Lightweight binary 2D CRF mean-field decoding on a grid.

    Unary:
        from LightGBM probabilities

    Pairwise:
        Gaussian spatial smoothing with Potts compatibility
    """
    p = np.clip(prob_map.astype(np.float32), 1e-5, 1.0 - 1e-5)

    unary_0 = -np.log(1.0 - p)
    unary_1 = -np.log(p)

    q1 = p.copy()

    for _ in range(max(int(n_iters), 0)):
        q0 = 1.0 - q1

        msg_1 = scipy.ndimage.gaussian_filter(q1, sigma=spatial_sigma, mode="nearest")
        msg_0 = scipy.ndimage.gaussian_filter(q0, sigma=spatial_sigma, mode="nearest")

        energy_0 = unary_weight * unary_0 + pairwise_weight * msg_1
        energy_1 = unary_weight * unary_1 + pairwise_weight * msg_0

        logits = np.stack([-energy_0, -energy_1], axis=0)
        logits -= logits.max(axis=0, keepdims=True)

        probs = np.exp(logits)
        probs /= probs.sum(axis=0, keepdims=True)

        q1 = probs[1].astype(np.float32)

    return q1


def _postprocess_mask(mask, min_region_size=8):
    mask = mask.astype(bool).copy()

    if min_region_size is None or min_region_size <= 1:
        return mask

    structure = np.ones((3, 3), dtype=int)
    labeled, num = scipy.ndimage.label(mask, structure=structure)

    if num > 0:
        sizes = np.bincount(labeled.ravel())
        for label_id in range(1, len(sizes)):
            if sizes[label_id] < min_region_size:
                mask[labeled == label_id] = False

    return mask


def _plot_masked_spect(Sxx, mask, raw_prob, crf_prob, f, t, min_db=None):
    if min_db is None:
        spectrogram_db = 10.0 * np.log10(np.maximum(Sxx, np.finfo(float).eps))
    else:
        power_floor = 10 ** (min_db / 10.0)
        spectrogram_db = 10.0 * np.log10(np.maximum(Sxx, power_floor))

    fig, ax = plt.subplots(1, 4, figsize=(24, 6))

    mesh0 = ax[0].pcolormesh(t, f, spectrogram_db, shading="gouraud")
    ax[0].set_title("Spectrogram")
    ax[0].set_xlabel("Time [sec]")
    ax[0].set_ylabel("Frequency [Hz]")
    fig.colorbar(mesh0, ax=ax[0], label="Intensity [dB]")

    mesh1 = ax[1].imshow(
        raw_prob,
        aspect="auto",
        origin="lower",
        cmap="viridis",
        extent=[t[0], t[-1], f[0], f[-1]],
        vmin=0.0,
        vmax=1.0,
    )
    ax[1].set_title("LightGBM Probability")
    ax[1].set_xlabel("Time [sec]")
    ax[1].set_ylabel("Frequency [Hz]")
    fig.colorbar(mesh1, ax=ax[1], label="Probability")

    mesh2 = ax[2].imshow(
        crf_prob,
        aspect="auto",
        origin="lower",
        cmap="viridis",
        extent=[t[0], t[-1], f[0], f[-1]],
        vmin=0.0,
        vmax=1.0,
    )
    ax[2].set_title("CRF Probability")
    ax[2].set_xlabel("Time [sec]")
    ax[2].set_ylabel("Frequency [Hz]")
    fig.colorbar(mesh2, ax=ax[2], label="Probability")

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
    ax[3].set_title("Final Binary Mask")
    ax[3].set_xlabel("Time [sec]")
    ax[3].set_ylabel("Frequency [Hz]")
    fig.colorbar(mesh3, ax=ax[3], label="Intensity [dB]")

    for a in ax:
        a.set_xlim(t[0], t[-1])
        a.set_ylim(f[0], f[-1])

    plt.tight_layout()
    plt.show()


def estimate_mask_file(
    file,
    model,
    nperseg=256,
    noverlap=128,
    plot=False,
    resampled_sr=None,
    min_db=None,
    return_axes=False,
    threshold=0.5,
    apply_crf=True,
    crf_spatial_sigma=(1.0, 1.0),
    crf_pairwise_weight=1.2,
    crf_unary_weight=1.0,
    crf_iterations=5,
    postprocess=True,
    min_region_size=8,
):
    """
    Predict a binary TF mask for one wav file.

    Returns:
        mask         : final binary mask
        raw_prob     : LightGBM probability map
        decoded_prob : CRF-refined probability map
        Sxx          : spectrogram
        f, t         : axes (if return_axes=True)
    """
    file = pathlib.Path(file)
    data, sample_rate = _load_audio(file, resampled_sr=resampled_sr)
    f, t, Sxx = _compute_spectrogram(
        data,
        sample_rate,
        nperseg=nperseg,
        noverlap=noverlap,
    )

    raw_prob = _predict_probability_map_from_sxx(Sxx, model)

    if apply_crf:
        decoded_prob = _crf_mean_field_binary(
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
        mask = _postprocess_mask(mask, min_region_size=min_region_size)

    if plot:
        _plot_masked_spect(Sxx, mask, raw_prob, decoded_prob, f, t, min_db=min_db)

    if return_axes:
        return mask, raw_prob, decoded_prob, Sxx, f, t

    return mask, raw_prob, decoded_prob, Sxx