# Project Structure

This repository is organized by method so that each candidate approach has its own folder.

## Folders

- `methods/baseline/`  
  Provided baseline / NMF-related reference scripts.

- `methods/method_1_dsp_spp/`  
  Pure DSP-SPP method, FUSS evaluation, and marine runner.

- `methods/method_2_ml_lgbm_crf/`  
  Pure ML LightGBM + 2D CRF method, FUSS evaluation, and marine runner.

- `methods/method_3_dl_unet/`  
  Pure DL U-Net method, FUSS training/evaluation, and marine runner.

- `methods/method_4_hybrid_dsp_ml/`  
  Hybrid DSP-ML method, FUSS evaluation, and marine runner.

- `methods/method_5_hybrid_dsp_dl/`  
  Hybrid DSP-DL method, FUSS training/evaluation, and marine runner.

- `docs/`  
  Experiment summary and general documentation.

- `results/fuss/`  
  Saved CSV metrics from FUSS benchmark runs.

- `config/`  
  Example configuration file. Copy `mask_config.example.yaml` to `mask_config.yaml` locally and edit paths.

## Notes

Large datasets, audio clips, generated masks, plots, trained PyTorch checkpoints, and LightGBM model bundles are intentionally excluded from the repository.

FUSS dataset location should be provided through the `FUSS_ROOT` environment variable. Local dataset paths should not be committed.
