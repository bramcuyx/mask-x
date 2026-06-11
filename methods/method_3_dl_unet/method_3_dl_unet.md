# Method 3: Pure DL Spectrogram U-Net

## 1. Goal

This method estimates a time-frequency binary mask directly from a mixture audio signal using a deep learning model.

The method is intentionally designed as a **pure DL** method:

```text
mixture audio -> log-power spectrogram -> U-Net -> probability mask -> binary mask
```

It does not use NMF, DSP saliency maps, LightGBM, CRF, or hand-crafted classical feature classifiers.

---

## 2. Input and output

### Input

The input is a mono mixture waveform. It is converted into a power spectrogram using STFT/spectrogram parameters consistent with the other project methods:

```text
nperseg = 256
noverlap = 128
```

The power spectrogram is converted to log scale:

```text
X = 10 log10(Sxx)
```

Then per-clip normalization is applied:

```text
X_norm = (X - mean(X)) / std(X)
```

### Output

The model outputs a probability mask:

```text
P(f, t) in [0, 1]
```

A binary mask is obtained using a fixed threshold selected on the FUSS validation split. The candidate thresholds are `0.40`, `0.45`, `0.50`, and `0.55`, and the selected threshold is the one with the highest mean IoU on the validation split:

```text
M(f, t) = 1 if P(f, t) >= threshold
M(f, t) = 0 otherwise
```

---

## 3. Network architecture

The model is a small U-Net for spectrogram segmentation.

| Stage | Description |
|---|---|
| Encoder | convolution blocks extract local TF patterns |
| Bottleneck | compact high-level representation |
| Decoder | upsampling reconstructs the mask resolution |
| Skip connections | preserve fine spectrogram details |
| Output layer | 1x1 convolution produces one logit per TF bin |

Default model configuration:

```text
in_channels = 1
base_channels = 16
depth = 3
dropout = 0.05
```

The final sigmoid is applied outside the model during inference.

---

## 4. Training data: FUSS

FUSS is used as the quantitative benchmark because it provides mixture files and separated source files.

For each FUSS example:

```text
exampleXXXX.wav
exampleXXXX_sources/foreground*.wav
exampleXXXX_sources/background*.wav
```

A ground-truth binary TF mask is generated from source spectrograms:

```text
foreground power > background power  -> 1
otherwise                            -> 0
```

This gives supervised training targets for the U-Net.

---

## 5. Loss function

The training loss is:

```text
Loss = BCEWithLogitsLoss + DiceLoss
```

BCE handles bin-level binary classification. Dice loss improves overlap quality and helps with sparse masks.

---

## 6. Scripts

### Core method

```text
dl_unet_method.py
```

Contains the U-Net, checkpoint utilities, spectrogram processing, inference, plotting, and optional tiny-component postprocessing.

### Training on FUSS

```text
train_fuss_dl_unet.py
```

Set the local FUSS folder through the `FUSS_ROOT` environment variable or `paths.fuss_root` in `config/mask_config.yaml`, then run:

```bash
python train_fuss_dl_unet.py
```

The trained model and training log are saved to:

```text
outputs/models/fuss_dl_unet_best.pt
outputs/fuss/fuss_dl_unet_training_log.csv
```

### Evaluation on FUSS

```text
evaluate_fuss_dl_unet.py
```

Run after training:

```bash
python evaluate_fuss_dl_unet.py
```

Outputs per-file metrics to:

```text
results/fuss/fuss_dl_unet_metrics.csv
```

Metrics include:

```text
Precision, Recall, F1, IoU, Dice
```

### Marine application

```text
run_marine_dl_unet.py
```

This loads the trained U-Net and applies it to the marine dataset configured in `mask_config.yaml`.

It saves:

```text
outputs/marine/masks/dl_unet/*.npy
outputs/marine/probabilities/dl_unet/*.npy
outputs/marine/plots/dl_unet/*.png
```

---

## 7. Method boundary

This method should be reported as **Pure DL** only if the default setting is used:

```text
postprocess = False
```

If connected-component cleanup or other classical postprocessing is enabled, report it explicitly as an additional postprocessing step.
