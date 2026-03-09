# Tool for extracting masks as input for the underwater acoustics simulation tool

## Environment setup

Install Poetry with `pipx`:

```bash
python3 -m pip install --user pipx
python3 -m pipx ensurepath
pipx install poetry
```

Create/install the project virtual environment and dependencies:

```bash
poetry lock
poetry install
```
For more information, see the [Poetry documentation](https://python-poetry.org/docs/).

## Mask curation workflow

The reusable workflow lives in [mask/run_mask.py](mask/run_mask.py). It is designed for a manual review loop where bad masks are removed based on visual inspection of plots.

### Process overview

1) Compute masks at threshold 3.0 dB and write plots to the main plots folder.
2) Visually inspect plots and delete bad plots (manual, outside the script).
3) Recompute only the deleted ones at threshold 3.0 dB and 4.5 dB to alternative folders.
4) Visually inspect alternative plots and delete bad ones (manual, outside the script).
5) Copy curated alternative plots into the main plots folder.
6) Delete masks that do not have a plot.

### How to run each step

Edit [mask/mask_config.yaml](mask/mask_config.yaml):

- `flags.run_initial_masks: true` runs step 1.
- `flags.run_reestimate_missing: true` runs step 3 for files whose plots were deleted in step 2.
- `flags.copy_curated_plots: true` runs step 5. Set `paths.curated_plots_subfolder` to the curated alternative folder you want to copy from.
- `flags.remove_masks_without_plots: true` runs step 6.

The script prints summary counts for masks and plots and lists how many masks do not have plots.

Run the workflow script:

```bash
poetry run python mask/run_mask.py
```

### Quick start examples

Use these as a guide for each phase by toggling values in [mask/mask_config.yaml](mask/mask_config.yaml):

- Phase 1 (initial masks): set `flags.run_initial_masks: true` and keep the other flags `false`.
- Phase 2 (recompute missing): set `flags.run_reestimate_missing: true` and keep the other flags `false`.
- Phase 3 (copy curated plots into main): set `flags.copy_curated_plots: true` and set `paths.curated_plots_subfolder` to the curated alternative folder you approved.
- Phase 4 (final cleanup): set `flags.remove_masks_without_plots: true`.

### Output locations

The default paths are defined in [mask/mask_config.yaml](mask/mask_config.yaml):

- Events (input): `/mnt/fscompute_shared/simulation_dataset/events`
- Masks (main): `/mnt/fscompute_shared/simulation_dataset/masks`
- Plots (main): `/mnt/fscompute_shared/simulation_dataset/masks_plots`
- Alternative outputs: `/mnt/fscompute_shared/simulation_dataset/masks_plots_alternative`

If you need different locations, update the `paths` section in [mask/mask_config.yaml](mask/mask_config.yaml).

