# Tool for extracting masks as input for the underwater acoustics simulation tool

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

Open [mask/run_mask.py](mask/run_mask.py) and toggle the flags at the top:

- `RUN_INITIAL_MASKS = True` runs step 1.
- `RUN_REESTIMATE_MISSING = True` runs step 3 for files whose plots were deleted in step 2.
- `COPY_CURATED_PLOTS = True` runs step 5. Set `CURATED_PLOTS_FOLDER` to the curated alternative folder you want to copy from.
- `REMOVE_MASKS_WITHOUT_PLOTS = True` runs step 6.

The script prints summary counts for masks and plots and lists how many masks do not have plots.

### Quick start examples

Use these as a guide for each phase by toggling the flags in [mask/run_mask.py](mask/run_mask.py):

- Phase 1 (initial masks at 3.0 dB): set `RUN_INITIAL_MASKS = True` and leave the other flags `False`.
- Phase 2 (recompute missing at 3.0 dB and 4.5 dB): set `RUN_REESTIMATE_MISSING = True` and leave the other flags `False`.
- Phase 3 (copy curated plots into main): set `COPY_CURATED_PLOTS = True` and set `CURATED_PLOTS_FOLDER` to the curated alternative folder you approved.
- Phase 4 (final cleanup): set `REMOVE_MASKS_WITHOUT_PLOTS = True`.

### Output locations

The default paths are defined near the top of [mask/run_mask.py](mask/run_mask.py):

- Events (input): `/mnt/fscompute_shared/simulation_dataset/events`
- Masks (main): `/mnt/fscompute_shared/simulation_dataset/masks`
- Plots (main): `/mnt/fscompute_shared/simulation_dataset/masks_plots`
- Alternative outputs: `/mnt/fscompute_shared/simulation_dataset/masks_plots_alternative`

If you need different locations, update the constants in [mask/run_mask.py](mask/run_mask.py).

