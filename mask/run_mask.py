# %%
import pathlib
import numpy as np
import random
import mask as m

events_folder = pathlib.Path('/mnt/fscompute_shared/simulation_dataset/events/')
event_files = list(events_folder.glob('*.wav'))

selection = random.sample(event_files, 5)
print("Selected files for testing:")
for file in selection:
    print(file, "\n")

for file in selection:
    print(f"Processing file: {file}")
    mask, med_subtracted, median, Sxx = m.estimate_mask_file(file, threshold=4.5, padding=1.0, plot=True, resampled_sr=48000)


# %%
