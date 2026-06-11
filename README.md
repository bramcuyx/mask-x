# Mask-X: Time-Frequency Mask Estimation Experiments

This repository contains several candidate methods for binary time-frequency mask estimation.

The original target application is mask generation for marine acoustic event clips. Since the marine clips do not provide ground-truth masks, the FUSS dataset is used as an external benchmark for quantitative evaluation.

The repository is organized by method. Each method folder contains the method implementation, FUSS evaluation or training scripts, marine inference scripts, and method-specific documentation.

---

## 1. Repository Structure

```text
TianyuXie/
├── config/
│   └── mask_config.example.yaml
├── docs/
│   └── experiments_summary_fuss_only.md
├── methods/
│   ├── method_1_dsp_spp/
│   ├── method_2_ml_lgbm_crf/
│   ├── method_3_dl_unet/
│   ├── method_4_hybrid_dsp_ml/
│   └── method_5_hybrid_dsp_dl/
├── results/
│   └── fuss/
├── utils/
│   └── paths.py
├── outputs/                  # Generated locally, ignored by Git
│   ├── models/
│   ├── fuss/
│   └── marine/
├── pyproject.toml
├── poetry.lock
├── requirements.txt
└── README.md
```

### Method folders

| Folder | Method |
|---|---|
| `methods/method_1_dsp_spp/` | Pure DSP-SPP method |
| `methods/method_2_ml_lgbm_crf/` | Pure ML method: LightGBM + 2D CRF |
| `methods/method_3_dl_unet/` | Pure DL method: spectrogram U-Net |
| `methods/method_4_hybrid_dsp_ml/` | Hybrid DSP-ML method |
| `methods/method_5_hybrid_dsp_dl/` | Hybrid DSP-DL method |

### Main utility folder

| Folder | Purpose |
|---|---|
| `utils/` | Shared project utilities |
| `utils/paths.py` | Centralized path and configuration handling |

### Output folders

Generated outputs are written under:

```text
outputs/
```

This folder is ignored by Git and should not be committed.

Typical generated outputs include:

```text
outputs/models/       # Trained checkpoints and model bundles
outputs/fuss/         # FUSS logs, split files, temporary outputs
outputs/marine/       # Marine masks, probabilities, and plots
```

Small result CSV files that are useful for documentation can be stored in:

```text
results/fuss/
```

---

## 2. Methods

The project currently compares five methods:

1. **Pure DSP-SPP**  
   A signal-processing method based on background estimation, SPP-like soft mapping, thresholding, and structural post-processing.

2. **Pure ML: LightGBM + 2D CRF**  
   A supervised method using handcrafted time-frequency features and a LightGBM classifier, followed by 2D CRF-style smoothing.

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

FUSS is used for quantitative evaluation because it provides mixtures and source-level references. The source references are used to derive foreground/background time-frequency masks.

The FUSS dataset is not included in this repository because of its size. Download it separately and either set the `FUSS_ROOT` environment variable or configure the local path in `config/mask_config.yaml`.

The expected path should point to the local FUSS `ssdata/train` folder.

Example:

```bash
export FUSS_ROOT=/path/to/FUSS/ssdata/train
```

### 3.2 Marine event clips

The marine clips are the target application data. They are used for mask generation and qualitative inspection.

Marine audio clips are not included in this repository. Configure their local folder in `config/mask_config.yaml`.

---

## 4. Installation

Poetry is the recommended way to set up the environment.

### 4.1 Install with Poetry

From the repository root:

```bash
poetry install
```

Then run scripts with:

```bash
poetry run python <script_path>
```

For example:

```bash
poetry run python methods/method_3_dl_unet/train_fuss_dl_unet.py
```

To check where Poetry stores virtual environments:

```bash
poetry config --list
```

To display the active environment path for this project:

```bash
poetry env info --path
```

This path can be selected as the Python interpreter in Visual Studio Code.

### 4.2 Alternative: install with pip

If Poetry is not available, install dependencies with:

```bash
pip install -r requirements.txt
```

### 4.3 PyTorch / CUDA note

The default dependency setup can be used for CPU execution. For GPU training, install a PyTorch build that matches the CUDA version on the target machine.

Example:

```bash
poetry run pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
```

or, if using pip directly:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
```

---

## 5. Configuration

A template config file is provided at:

```text
config/mask_config.example.yaml
```

For local use, copy it to:

```text
config/mask_config.yaml
```

The local `mask_config.yaml` file is ignored by Git because it may contain machine-specific paths.

### 5.1 Example configuration

```yaml
paths:
  # Input data
  marine_events_folder: "data/marine/events"

  # FUSS data root. This can also be set through the FUSS_ROOT environment variable.
  fuss_root: ""

  # Generated outputs
  outputs_root: "outputs"
  models_folder: "outputs/models"
  fuss_outputs_folder: "outputs/fuss"
  marine_outputs_folder: "outputs/marine"

  # Commit-friendly result CSVs
  fuss_results_folder: "results/fuss"

  # Saved model paths for marine inference
  ml_model_bundle: "outputs/models/fuss_ml_lgbm_crf_model_all.pkl"
  hybrid_dsp_ml_model_bundle: "outputs/models/fuss_hybrid_dsp_ml_model_all.pkl"
  dl_unet_checkpoint: "outputs/models/fuss_dl_unet_best.pt"
  hybrid_dsp_dl_checkpoint: "outputs/models/fuss_hybrid_dsp_dl_best.pt"

processing:
  resampled_sr: 48000
  padding: 1.0
```

### 5.2 Path handling

All relative paths in `mask_config.yaml` are resolved relative to the repository root.

For example:

```yaml
models_folder: "outputs/models"
```

is interpreted as:

```text
<repository_root>/outputs/models
```

Absolute paths can also be used for local datasets.

### 5.3 FUSS root priority

The FUSS dataset location is resolved in this order:

1. `FUSS_ROOT` environment variable
2. `paths.fuss_root` in `config/mask_config.yaml`

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

Run all scripts from the repository root.

### Method 1: Pure DSP-SPP

```bash
poetry run python methods/method_1_dsp_spp/evaluate_fuss_dsp_spp.py
```

### Method 2: Pure ML

```bash
poetry run python methods/method_2_ml_lgbm_crf/evaluate_fuss_ml_lgbm_crf.py
```

### Method 3: Pure DL

Train:

```bash
poetry run python methods/method_3_dl_unet/train_fuss_dl_unet.py
```

Evaluate:

```bash
poetry run python methods/method_3_dl_unet/evaluate_fuss_dl_unet.py
```

### Method 4: Hybrid DSP-ML

```bash
poetry run python methods/method_4_hybrid_dsp_ml/evaluate_fuss_hybrid_dsp_ml.py
```

### Method 5: Hybrid DSP-DL

Train:

```bash
poetry run python methods/method_5_hybrid_dsp_dl/train_fuss_hybrid_dsp_dl.py
```

Evaluate:

```bash
poetry run python methods/method_5_hybrid_dsp_dl/evaluate_fuss_hybrid_dsp_dl.py
```

### FUSS outputs

Generated FUSS-related files are written to:

```text
outputs/fuss/
```

Trained model artifacts are written to:

```text
outputs/models/
```

Small CSV metrics intended for documentation are written to:

```text
results/fuss/
```

---

## 7. Running Marine Inference

Marine inference uses local marine event clips and produces masks, probability maps, and plots.

Before running marine inference, set the marine input folder in:

```text
config/mask_config.yaml
```

Example:

```yaml
paths:
  marine_events_folder: "/path/to/marine/events"
  marine_outputs_folder: "outputs/marine"
```

Run scripts from the repository root.

### Method 1: Pure DSP-SPP

```bash
poetry run python methods/method_1_dsp_spp/run_marine_dsp_spp.py
```

### Method 2: Pure ML

By default, the model path is read from:

```yaml
paths.ml_model_bundle
```

Run:

```bash
poetry run python methods/method_2_ml_lgbm_crf/run_marine_ml_lgbm_crf.py
```

Optionally override the model path:

```bash
poetry run python methods/method_2_ml_lgbm_crf/run_marine_ml_lgbm_crf.py --model /path/to/model.pkl
```

### Method 3: Pure DL

By default, the checkpoint path is read from:

```yaml
paths.dl_unet_checkpoint
```

Run:

```bash
poetry run python methods/method_3_dl_unet/run_marine_dl_unet.py
```

### Method 4: Hybrid DSP-ML

By default, the model path is read from:

```yaml
paths.hybrid_dsp_ml_model_bundle
```

Run:

```bash
poetry run python methods/method_4_hybrid_dsp_ml/run_marine_hybrid_dsp_ml.py
```

Optionally override the model path:

```bash
poetry run python methods/method_4_hybrid_dsp_ml/run_marine_hybrid_dsp_ml.py --model /path/to/model.pkl
```

### Method 5: Hybrid DSP-DL

By default, the checkpoint path is read from:

```yaml
paths.hybrid_dsp_dl_checkpoint
```

Run:

```bash
poetry run python methods/method_5_hybrid_dsp_dl/run_marine_hybrid_dsp_dl.py
```

### Marine outputs

Marine outputs are written under:

```text
outputs/marine/
```

Typical output layout:

```text
outputs/marine/
├── masks/
├── probabilities/
└── plots/
```

Trained model checkpoints and model bundles are not included in this repository. They should be generated locally or stored separately.

---

## 8. Results

FUSS result CSV files intended for documentation are stored in:

```text
results/fuss/
```

The main FUSS experiment summary is:

```text
docs/experiments_summary_fuss_only.md
```

Generated logs, split files, temporary files, trained models, masks, probability maps, and plots are stored under:

```text
outputs/
```

and are ignored by Git.

---

## 9. Documentation

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

---

## 10. GitHub Notes

The following files and folders should not be committed:

- local virtual environments such as `.venv/`,
- FUSS data,
- marine `.wav` files,
- local `config/mask_config.yaml`,
- generated outputs under `outputs/`,
- trained model files such as `.pt`, `.pth`, and `.pkl`,
- generated NumPy masks or probability maps such as `.npy`,
- generated plots such as `.png`.

The `.gitignore` file is set up to exclude these by default.

Recommended files to commit include:

```text
README.md
pyproject.toml
poetry.lock
requirements.txt
config/mask_config.example.yaml
utils/
methods/
docs/
results/fuss/
```

---

## 11. Current Limitations

- The FUSS dataset is not included in the repository.
- Marine audio clips are not included in the repository.
- Trained model checkpoints and model bundles are not included in the repository.
- Marine results are qualitative because the marine clips do not contain ground-truth masks.
- Quantitative evaluation is performed on FUSS-derived masks.
