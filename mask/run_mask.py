import pathlib
import shutil
import numpy as np
import matplotlib.pyplot as plt
import yaml
import mask as m

# %%
# Reusable workflow:
# 1) Compute masks at threshold 3.0 and save plots to the main plots folder.
# 2) Manually inspect main plots and delete bad plots (outside this script).
# 3) Recompute only the deleted ones at threshold 3.0 and 4.5, save to alternative folders.
# 4) Manually inspect alternative plots and delete bad ones (outside this script).
# 5) Copy curated alternative plots into the main plots folder.
# 6) Delete masks that do not have a plot.

CONFIG_PATH = pathlib.Path(__file__).with_name('mask_config.yaml')


def load_config(config_path: pathlib.Path) -> dict:
    with config_path.open('r', encoding='utf-8') as file:
        return yaml.safe_load(file)


config = load_config(CONFIG_PATH)

EVENTS_FOLDER = pathlib.Path(config['paths']['events_folder'])
MASKS_FOLDER = pathlib.Path(config['paths']['masks_folder'])
PLOTS_FOLDER = pathlib.Path(config['paths']['plots_folder'])

ALT_BASE_FOLDER = pathlib.Path(config['paths']['alt_base_folder'])
ALT_3_DB_PLOTS = ALT_BASE_FOLDER / config['paths']['alt_3_db_plots_subfolder']
ALT_3_DB_MASKS = ALT_BASE_FOLDER / config['paths']['alt_3_db_masks_subfolder']
ALT_4_5_DB_PLOTS = ALT_BASE_FOLDER / config['paths']['alt_4_5_db_plots_subfolder']
ALT_4_5_DB_MASKS = ALT_BASE_FOLDER / config['paths']['alt_4_5_db_masks_subfolder']

RESAMPLED_SR = config['processing']['resampled_sr']
PADDING = config['processing']['padding']

RUN_INITIAL_MASKS = config['flags']['run_initial_masks']
RUN_REESTIMATE_MISSING = config['flags']['run_reestimate_missing']
COPY_CURATED_PLOTS = config['flags']['copy_curated_plots']
REMOVE_MASKS_WITHOUT_PLOTS = config['flags']['remove_masks_without_plots']

INITIAL_THRESHOLD_DB = config['processing']['initial_threshold_db']
REESTIMATE_THRESHOLDS_DB = config['processing']['reestimate_thresholds_db']

curated_subfolder = config['paths']['curated_plots_subfolder']
CURATED_PLOTS_FOLDER = ALT_BASE_FOLDER / curated_subfolder


def get_event_files(events_folder: pathlib.Path) -> list[pathlib.Path]:
    event_files = sorted(events_folder.glob('*.wav'))
    if not event_files:
        raise FileNotFoundError(f"No .wav files found in {events_folder}")
    return event_files


def estimate_and_save(
    files: list[pathlib.Path],
    threshold: float,
    masks_folder: pathlib.Path,
    plots_folder: pathlib.Path,
    overwrite_masks: bool = False,
    overwrite_plots: bool = False,
) -> None:
    masks_folder.mkdir(exist_ok=True, parents=True)
    plots_folder.mkdir(exist_ok=True, parents=True)
    for index, file in enumerate(files, start=1):
        print(f"Processing file {index}/{len(files)}: {file}")
        (
            mask,
            _med_subtracted,
            _median,
            Sxx,
            f,
            t,
        ) = m.estimate_mask_file(
            file,
            threshold=threshold,
            padding=PADDING,
            plot=False,
            resampled_sr=RESAMPLED_SR,
            return_axes=True,
        )

        mask_path = masks_folder / f"{file.stem}.npy"
        if overwrite_masks or not mask_path.exists():
            np.save(mask_path, mask)

        plot_path = plots_folder / f"{file.stem}.png"
        if overwrite_plots or not plot_path.exists():
            fig, ax = plt.subplots(1, 3, figsize=(20, 6))
            m.plot_masked_spect(
                Sxx,
                mask,
                f,
                t,
                ax_mask=ax[1],
                ax_spect=ax[0],
                ax_mask_only=ax[2],
                show=False,
            )
            fig.savefig(plot_path, dpi=150, bbox_inches='tight')
            plt.close(fig)


def count_files(folder: pathlib.Path, pattern: str) -> int:
    return len(list(folder.glob(pattern)))


def mask_stems_without_plots(masks_folder: pathlib.Path, plots_folder: pathlib.Path) -> list[str]:
    missing = []
    for mask_file in masks_folder.glob('*.npy'):
        if not (plots_folder / f"{mask_file.stem}.png").exists():
            missing.append(mask_file.stem)
    return missing


def stems_to_event_files(stems: list[str], events_folder: pathlib.Path) -> list[pathlib.Path]:
    return [events_folder / f"{stem}.wav" for stem in stems]


def copy_plots(source_folder: pathlib.Path, target_folder: pathlib.Path) -> None:
    target_folder.mkdir(exist_ok=True, parents=True)
    for plot in source_folder.glob('*.png'):
        shutil.copy2(plot, target_folder / plot.name)


def delete_masks_without_plots(masks_folder: pathlib.Path, plots_folder: pathlib.Path) -> int:
    deleted = 0
    for mask_file in masks_folder.glob('*.npy'):
        if not (plots_folder / f"{mask_file.stem}.png").exists():
            mask_file.unlink()
            deleted += 1
    return deleted


# %%
event_files = get_event_files(EVENTS_FOLDER)

if RUN_INITIAL_MASKS:
    # Step 1: initial mask estimation at threshold 3.0
    estimate_and_save(
        event_files,
        threshold=INITIAL_THRESHOLD_DB,
        masks_folder=MASKS_FOLDER,
        plots_folder=PLOTS_FOLDER,
        overwrite_masks=False,
        overwrite_plots=False,
    )

# Step 2 (manual): inspect plots in PLOTS_FOLDER and delete bad plots.

missing_stems = mask_stems_without_plots(MASKS_FOLDER, PLOTS_FOLDER)
print(f"Total mask plots saved: {count_files(PLOTS_FOLDER, '*.png')}")
print(f"Total masks saved: {count_files(MASKS_FOLDER, '*.npy')}")
print(f"Files with masks but no plots: {len(missing_stems)}")

if RUN_REESTIMATE_MISSING:
    # Step 3: recompute only the deleted ones at threshold 3.0 and 4.5
    missing_files = stems_to_event_files(missing_stems, EVENTS_FOLDER)
    threshold_3_db, threshold_4_5_db = REESTIMATE_THRESHOLDS_DB
    estimate_and_save(
        missing_files,
        threshold=threshold_3_db,
        masks_folder=ALT_3_DB_MASKS,
        plots_folder=ALT_3_DB_PLOTS,
        overwrite_masks=True,
        overwrite_plots=True,
    )
    estimate_and_save(
        missing_files,
        threshold=threshold_4_5_db,
        masks_folder=ALT_4_5_DB_MASKS,
        plots_folder=ALT_4_5_DB_PLOTS,
        overwrite_masks=True,
        overwrite_plots=True,
    )

# Step 4 (manual): inspect alternative plots and delete bad ones.

if COPY_CURATED_PLOTS:
    # Step 5: copy curated alternative plots into the main plots folder
    copy_plots(CURATED_PLOTS_FOLDER, PLOTS_FOLDER)

if REMOVE_MASKS_WITHOUT_PLOTS:
    # Step 6: delete masks that still do not have plots
    deleted = delete_masks_without_plots(MASKS_FOLDER, PLOTS_FOLDER)
    print(f"Masks deleted because plot was missing: {deleted}")
    print(f"Total mask files after cleanup: {count_files(MASKS_FOLDER, '*.npy')}")

