# Method 5: Hybrid DSP-DL — DSP-guided U-Net

## Goal

This method combines DSP-SPP with U-Net segmentation without modifying the existing Pure DSP or Pure DL files.

The goal is to give U-Net an explicit DSP prior so the network does not need to infer all candidate event regions only from the spectrogram.

## Pipeline

```text
mixture wav
→ log-power spectrogram
→ DSP-SPP soft probability map
→ stack as 2-channel input
→ U-Net
→ probability mask
→ binary TF mask
```

## Input channels

```text
Channel 1: normalized log-power spectrogram
Channel 2: DSP-SPP soft probability map
```

## Difference from Pure DL

Pure DL:

```text
[log spectrogram] → U-Net
```

Hybrid DSP-DL:

```text
[log spectrogram, DSP probability] → U-Net
```

The U-Net architecture is otherwise the same small U-Net used by the Pure DL method, except the first layer receives two input channels.

## Scripts

```text
hybrid_dsp_dl_method.py
    Core 2-channel U-Net method.

train_fuss_hybrid_dsp_dl.py
    Train on FUSS.

evaluate_fuss_hybrid_dsp_dl.py
    Evaluate trained checkpoint on FUSS.

run_marine_hybrid_dsp_dl.py
    Apply trained checkpoint to marine wav files.
```

## FUSS training

```powershell
python train_fuss_hybrid_dsp_dl.py
```

During training, the binary-mask threshold is selected on the FUSS validation split. The candidate thresholds are `0.40`, `0.45`, `0.50`, and `0.55`. The selected threshold is the one with the highest mean IoU on the validation split. F1 is still reported as an evaluation metric, but IoU is used as the primary metric for threshold selection.

This produces:

```text
outputs/models/fuss_hybrid_dsp_dl_best.pt
outputs/fuss/fuss_hybrid_dsp_dl_training_log.csv
```

## FUSS evaluation

```powershell
python evaluate_fuss_hybrid_dsp_dl.py
```

This produces:

```text
results/fuss/fuss_hybrid_dsp_dl_metrics.csv
```

## Marine usage

```powershell
python run_marine_hybrid_dsp_dl.py
```

Outputs:

```text
outputs/marine/masks/hybrid_dsp_dl/*.npy
outputs/marine/probabilities/hybrid_dsp_dl/*.npy
outputs/marine/plots/hybrid_dsp_dl/*.png
```
