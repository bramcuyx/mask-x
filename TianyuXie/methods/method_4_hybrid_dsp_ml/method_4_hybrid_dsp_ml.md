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

This produces:

```text
fuss_hybrid_dsp_ml_model_all.pkl
fuss_hybrid_dsp_ml_metrics_all.csv
```

inside the configured FUSS `ROOT` folder.

## Marine usage

After FUSS training has produced the `.pkl` model:

```powershell
python run_marine_hybrid_dsp_ml.py --model "D:/.../ssdata/train/fuss_hybrid_dsp_ml_model_all.pkl"
```

Outputs:

```text
masks_folder/hybrid_dsp_ml/*.npy
masks_folder/hybrid_dsp_ml_prob/*.npy
masks_folder/hybrid_dsp_ml_raw_prob/*.npy
plots_folder/hybrid_dsp_ml/*.png
```
