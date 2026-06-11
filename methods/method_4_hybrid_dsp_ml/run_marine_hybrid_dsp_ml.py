"""
Run Method 4: Hybrid DSP-Hybrid DSP-ML LightGBM + optional lightweight 2D CRF on marine wav files.

Usage examples:
    python run_marine_hybrid_dsp_ml.py --model "D:/Fuss_For_ICT/.../ssdata/train/fuss_hybrid_dsp_ml_model_all.pkl"

Optional:
    Add this to mask_config.yaml:

    paths:
      hybrid_dsp_ml_model_bundle: "D:/Fuss_For_ICT/.../ssdata/train/fuss_hybrid_dsp_ml_model_all.pkl"

Then you can run:
    python run_marine_hybrid_dsp_ml.py
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

from utils.paths import load_config, get_config_path, ensure_dir, resolve_path
import argparse
import pathlib

import matplotlib.pyplot as plt
import numpy as np
 
import hybrid_dsp_ml_method as hml


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

def plot_hybrid_dsp_ml_result(Sxx, mask, raw_prob, decoded_prob, dsp_prob, f, t, out_png: pathlib.Path):
    spectrogram_db = 10.0 * np.log10(np.maximum(Sxx, np.finfo(float).eps))

    fig, ax = plt.subplots(1, 5, figsize=(30, 6))

    mesh0 = ax[0].pcolormesh(t, f, spectrogram_db, shading="gouraud")
    ax[0].set_title("Spectrogram")
    ax[0].set_xlabel("Time [sec]")
    ax[0].set_ylabel("Frequency [Hz]")
    fig.colorbar(mesh0, ax=ax[0], label="Intensity [dB]")

    for idx, (arr, title) in enumerate(
        [
            (dsp_prob, "DSP-SPP Probability"),
            (raw_prob, "Hybrid LightGBM Probability"),
            (decoded_prob, "Decoded Probability"),
        ],
        start=1,
    ):
        mesh = ax[idx].imshow(
            arr,
            aspect="auto",
            origin="lower",
            extent=[t[0], t[-1], f[0], f[-1]],
            vmin=0.0,
            vmax=1.0,
        )
        ax[idx].set_title(title)
        ax[idx].set_xlabel("Time [sec]")
        ax[idx].set_ylabel("Frequency [Hz]")
        fig.colorbar(mesh, ax=ax[idx], label="Probability")

    mesh4 = ax[4].pcolormesh(t, f, spectrogram_db, shading="gouraud")
    ax[4].imshow(
        mask,
        aspect="auto",
        origin="lower",
        alpha=0.35,
        extent=[t[0], t[-1], f[0], f[-1]],
        zorder=3,
    )
    ax[4].set_title("Final Binary Mask")
    ax[4].set_xlabel("Time [sec]")
    ax[4].set_ylabel("Frequency [Hz]")
    fig.colorbar(mesh4, ax=ax[4], label="Intensity [dB]")

    for a in ax:
        a.set_xlim(t[0], t[-1])
        a.set_ylim(f[0], f[-1])

    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Path to the .pkl model bundle saved by evaluate_fuss_hybrid_dsp_ml.py.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    config = load_config()

    events_folder = get_config_path(config, "marine_events_folder")
    marine_outputs_dir = ensure_dir(get_config_path(config, "marine_outputs_folder"))

    masks_folder = marine_outputs_dir / "masks" / "hybrid_dsp_ml"
    prob_folder = marine_outputs_dir / "probabilities" / "hybrid_dsp_ml"
    raw_prob_folder = marine_outputs_dir / "probabilities" / "hybrid_dsp_ml_raw"
    plots_folder = marine_outputs_dir / "plots" / "hybrid_dsp_ml"

    if args.model is not None:
        model_path = resolve_path(args.model)
    else:
        model_path = get_config_path(config, "hybrid_dsp_ml_model_bundle")

    if not model_path.exists():
        raise FileNotFoundError(f"Model bundle not found: {model_path}")

    payload = hml.load_model_bundle(model_path)
    model = payload["model"]
    meta = payload.get("metadata", {})

    nperseg = int(meta.get("nperseg", 256))
    noverlap = int(meta.get("noverlap", 128))
    resampled_sr = meta.get("resampled_sr", 16000)
    threshold = float(meta.get("best_threshold", 0.5))

    apply_crf = bool(meta.get("apply_crf", True))
    crf_spatial_sigma = tuple(meta.get("crf_spatial_sigma", (1.0, 1.0)))
    crf_pairwise_weight = float(meta.get("crf_pairwise_weight", 1.2))
    crf_unary_weight = float(meta.get("crf_unary_weight", 1.0))
    crf_iterations = int(meta.get("crf_iterations", 5))

    postprocess = bool(meta.get("postprocess", True))
    min_region_size = int(meta.get("min_region_size", 8))

    masks_folder.mkdir(parents=True, exist_ok=True)
    prob_folder.mkdir(parents=True, exist_ok=True)
    raw_prob_folder.mkdir(parents=True, exist_ok=True)
    plots_folder.mkdir(parents=True, exist_ok=True)

    event_files = get_event_files(events_folder)

    print(f"Loaded model:       {model_path}")
    print(f"Events folder:      {events_folder}")
    print(f"Number of files:    {len(event_files)}")
    print(f"nperseg/noverlap:   {nperseg}/{noverlap}")
    print(f"resampled_sr:       {resampled_sr}")
    print(f"threshold:          {threshold}")
    print(f"apply_crf:          {apply_crf}")

    for i, wav_path in enumerate(event_files, start=1):
        print(f"Processing {i}/{len(event_files)}: {wav_path.name}")

        mask, raw_prob, decoded_prob, dsp_prob, Sxx, f, t = hml.estimate_mask_file(
            wav_path,
            model=model,
            nperseg=nperseg,
            noverlap=noverlap,
            plot=False,
            resampled_sr=resampled_sr,
            return_axes=True,
            threshold=threshold,
            apply_crf=apply_crf,
            crf_spatial_sigma=crf_spatial_sigma,
            crf_pairwise_weight=crf_pairwise_weight,
            crf_unary_weight=crf_unary_weight,
            crf_iterations=crf_iterations,
            postprocess=postprocess,
            min_region_size=min_region_size,
        )

        padding = float(config.get("processing", {}).get("padding", 1.0))

        mask, cleaned = enforce_marine_padding(
            mask,
            t,
            padding=padding,
            probability_maps=[raw_prob, decoded_prob, dsp_prob],
        )

        raw_prob, decoded_prob, dsp_prob = cleaned

        np.save(masks_folder / f"{wav_path.stem}.npy", mask)
        np.save(prob_folder / f"{wav_path.stem}.npy", decoded_prob)
        np.save(raw_prob_folder / f"{wav_path.stem}.npy", raw_prob)
        plot_hybrid_dsp_ml_result(Sxx, mask, raw_prob, decoded_prob, dsp_prob, f, t, plots_folder / f"{wav_path.stem}.png")

    print("Done.")
    print(f"Saved masks:        {masks_folder}")
    print(f"Saved probabilities:{prob_folder}")
    print(f"Saved plots:        {plots_folder}")


if __name__ == "__main__":
    main()
