# Mask-X: Time-Frequency Mask Estimation Experiments

This repository contains several candidate methods for binary time-frequency mask estimation.  
The original target application is mask generation for marine acoustic event clips. Since the marine clips do not provide ground-truth masks, the FUSS dataset is used as an external benchmark for quantitative evaluation.

The current repository is organized by method. Each method folder contains its own implementation, FUSS evaluation script, marine inference script, and method documentation.

---

## 1. Repository Structure

```text
mask_x_github_ready/
├── config/
│   └── mask_config.example.yaml
├── docs/
│   └── experiments_summary_fuss_only.md
├── methods/
│   ├── baseline/
│   ├── method_1_dsp_spp/
│   ├── method_2_ml_lgbm_crf/
│   ├── method_3_dl_unet/
│   ├── method_4_hybrid_dsp_ml/
│   └── method_5_hybrid_dsp_dl/
├── results/
│   └── fuss/
├── PROJECT_STRUCTURE.md
├── requirements.txt
└── README.md
```

### Method folders

| Folder | Method |
|---|---|
| `methods/baseline/` | Provided baseline / NMF-related reference scripts |
| `methods/method_1_dsp_spp/` | Pure DSP-SPP method |
| `methods/method_2_ml_lgbm_crf/` | Pure ML method: LightGBM + 2D CRF |
| `methods/method_3_dl_unet/` | Pure DL method: spectrogram U-Net |
| `methods/method_4_hybrid_dsp_ml/` | Hybrid DSP-ML method |
| `methods/method_5_hybrid_dsp_dl/` | Hybrid DSP-DL method |

---

## 2. Methods

The project currently compares five non-baseline methods:

1. **Pure DSP-SPP**  
   A signal-processing method based on background estimation, SPP-like soft mapping, thresholding, and structural post-processing.

2. **Pure ML: LightGBM + 2D CRF**  
   A supervised method using handcrafted time-frequency features and a LightGBM classifier, followed by a simple 2D CRF-style smoothing step.

3. **Pure DL: U-Net**  
   A spectrogram-to-mask segmentation model trained on FUSS-derived reference masks.

4. **Hybrid DSP-ML**  
   A LightGBM-based method that augments the ML feature representation with DSP-derived prior features.

5. **Hybrid DSP-DL**  
   A U-Net-based method that uses both the log spectrogram and DSP-derived probability guidance as input channels.

Each method has a dedicated markdown file inside its own folder.

---

## 3. Data

### 3.1 FUSS benchmark

FUSS is used for quantitative evaluation because it provides mixtures and source-level references.  
The source references are used to derive foreground/background time-frequency masks.

The FUSS dataset itself is not included in this repository because of its size.  
Download it separately and set the `FUSS_ROOT` environment variable to the local `ssdata/train` folder. Do not commit machine-specific dataset paths to Git.

### 3.2 Marine event clips

The marine clips are the target application data. They are used for mask generation and qualitative inspection.  
They are not included in this repository.

---

## 4. Installation

Create and activate a Python environment, then install the dependencies:

```bash
pip install -r requirements.txt
```

For GPU training with PyTorch, install the CUDA-compatible PyTorch build appropriate for your machine.  
For example, on Windows with a compatible NVIDIA driver:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
```

If CUDA is not available, the DL scripts can still run on CPU, but training will be much slower.

---

## 5. Configuration

A template config file is provided at:

```text
config/mask_config.example.yaml
```

For local use, copy it to a method folder as:

```text
mask_config.yaml
```

and edit the paths, for example:

```yaml
paths:
  events_folder: "path/to/marine/events"
  masks_folder: "path/to/output/masks"
  plots_folder: "path/to/output/masks_plots"

processing:
  resampled_sr: 48000
  padding: 1.0
```

The real `mask_config.yaml` file is ignored by Git because it contains local machine paths.

For FUSS experiments, set `FUSS_ROOT` outside the codebase. It should point to the local FUSS `ssdata/train` folder.

Linux/macOS:

```bash
export FUSS_ROOT=/path/to/FUSS/ssdata/train
```

Windows PowerShell:

```powershell
$env:FUSS_ROOT="D:\path\to\FUSS\ssdata\train"
```

Windows Command Prompt:

```cmd
set FUSS_ROOT=D:\path\to\FUSS\ssdata\train
```

---

## 6. Running FUSS Experiments

Run scripts from the corresponding method folder.

### Method 1: Pure DSP-SPP

```bash
cd methods/method_1_dsp_spp
python evaluate_fuss_dsp_spp.py
```

### Method 2: Pure ML

```bash
cd methods/method_2_ml_lgbm_crf
python evaluate_fuss_ml_lgbm_crf.py
```

### Method 3: Pure DL

```bash
cd methods/method_3_dl_unet
python train_fuss_dl_unet.py
python evaluate_fuss_dl_unet.py
```

### Method 4: Hybrid DSP-ML

```bash
cd methods/method_4_hybrid_dsp_ml
python evaluate_fuss_hybrid_dsp_ml.py
```

### Method 5: Hybrid DSP-DL

```bash
cd methods/method_5_hybrid_dsp_dl
python train_fuss_hybrid_dsp_dl.py
python evaluate_fuss_hybrid_dsp_dl.py
```

Before running FUSS scripts, make sure `FUSS_ROOT` is set in your terminal.

---

## 7. Running Marine Inference

Marine inference uses local event clips and produces masks and plot images.

### Method 1: Pure DSP-SPP

```bash
cd methods/method_1_dsp_spp
python run_marine_dsp_spp.py
```

### Method 2: Pure ML

```bash
cd methods/method_2_ml_lgbm_crf
python run_marine_ml_lgbm_crf.py --model "path/to/fuss_ml_lgbm_crf_model_all.pkl"
```

### Method 3: Pure DL

```bash
cd methods/method_3_dl_unet
python run_marine_dl_unet.py
```

### Method 4: Hybrid DSP-ML

```bash
cd methods/method_4_hybrid_dsp_ml
python run_marine_hybrid_dsp_ml.py --model "path/to/fuss_hybrid_dsp_ml_model_all.pkl"
```

### Method 5: Hybrid DSP-DL

```bash
cd methods/method_5_hybrid_dsp_dl
python run_marine_hybrid_dsp_dl.py
```

Trained model checkpoints are not included in this repository. They should be generated locally or stored separately.

---

## 8. Results

Saved FUSS result CSV files are stored in:

```text
results/fuss/
```

The main summary document is:

```text
docs/experiments_summary_fuss_only.md
```

Current FUSS results indicate that the hybrid methods improve over their corresponding pure ML/DL versions. The strongest method in the current benchmark summary is the Hybrid DSP-DL method, based on IoU and Dice/F1.

---

## 9. GitHub Notes

The following files and folders should not be committed:

- local virtual environments such as `.venv/`,
- FUSS data,
- marine `.wav` files,
- generated masks and plots,
- local `mask_config.yaml`,
- trained model files such as `.pt`, `.pth`, and `.pkl`.

The `.gitignore` file is set up to exclude these by default.

---

## 10. Documentation

Method-specific documentation:

```text
methods/method_1_dsp_spp/method_1_dsp_spp.md
methods/method_2_ml_lgbm_crf/method_2_ml_lgbm_crf.md
methods/method_3_dl_unet/method_3_dl_unet.md
methods/method_4_hybrid_dsp_ml/method_4_hybrid_dsp_ml.md
methods/method_5_hybrid_dsp_dl/method_5_hybrid_dsp_dl.md
```

Experiment summary:

```text
docs/experiments_summary_fuss_only.md
```

Project structure notes:

```text
PROJECT_STRUCTURE.md
```

---

## 11. Current Limitations

- FUSS dataset location is read from the `FUSS_ROOT` environment variable, so local machine paths are not committed.
- Trained model checkpoints are not included in the repository.
- Marine results are qualitative because the marine clips do not contain ground-truth masks.
- The baseline/reference result may not use the exact same split as the learned methods, so strict comparison should be treated carefully unless rerun on the same split.
