# Method 4: Hybrid DSP-ML — DSP-guided LightGBM + 2D CRF

## Goal

This method combines the Pure DSP-SPP method and the Pure ML LightGBM method without modifying either one.

The goal is to use DSP-SPP as a weak proposal generator, then let LightGBM learn which DSP-highlighted TF regions should be kept or rejected.

## Pipeline

```text
mixture wav
→ spectrogram
→ Pure ML handcrafted features
→ DSP-SPP soft probability / binary mask / local DSP density
→ concatenate features
→ LightGBM classifier
→ optional lightweight 2D CRF
→ binary TF mask
```

## Difference from Pure ML

Pure ML uses only handcrafted spectrogram features.

Hybrid DSP-ML appends DSP guidance features:

```text
DSP soft probability
DSP binary mask
DSP local mask density
local DSP probability mean
local DSP probability contrast
```

## Scripts

```text
hybrid_dsp_ml_method.py
    Core method implementation.

evaluate_fuss_hybrid_dsp_ml.py
    Train and evaluate on FUSS.

run_marine_hybrid_dsp_ml.py
    Apply trained model to marine wav files.
```

## FUSS usage

```powershell
python evaluate_fuss_hybrid_dsp_ml.py
```
During FUSS evaluation, the probability threshold is selected on the validation split. The candidate thresholds are `0.40`, `0.45`, `0.50`, and `0.55`. The selected threshold is the one with the highest mean IoU on the validation split. F1 is still reported as an evaluation metric, but IoU is used as the primary metric for threshold selection.

This produces:

```text
outputs/models/fuss_hybrid_dsp_ml_model_all.pkl
results/fuss/fuss_hybrid_dsp_ml_metrics_all.csv
outputs/fuss/fuss_hybrid_dsp_ml_test_files.txt
```

The exact folders are controlled by `models_folder`, `fuss_results_folder`, and `fuss_outputs_folder` in `config/mask_config.yaml`.

## Marine usage

After FUSS training has produced the `.pkl` model:

```powershell
python run_marine_hybrid_dsp_ml.py --model "outputs/models/fuss_hybrid_dsp_ml_model_all.pkl"
```

Outputs:

```text
outputs/marine/masks/hybrid_dsp_ml/*.npy
outputs/marine/probabilities/hybrid_dsp_ml/*.npy
outputs/marine/probabilities/hybrid_dsp_ml_raw/*.npy
outputs/marine/plots/hybrid_dsp_ml/*.png
```
