# Experiments Summary

## 1. Purpose

This document summarizes the current FUSS benchmark experiments for the time-frequency mask estimation project.

The goal of the experiments is to compare several candidate methods for binary time-frequency mask estimation using a dataset with source-level reference information. The original application target is marine acoustic event masking, but this summary focuses only on the FUSS benchmark results and does not include marine qualitative tuning or marine visual inspection.

---

## 2. FUSS Benchmark Setup

The FUSS dataset is used as the quantitative evaluation benchmark because it provides mixture waveforms together with source-level references.

For each example:

1. the mixture waveform is used as the input signal;
2. the source-level references are used to derive a reference time-frequency mask;
3. the predicted binary mask is compared against this reference mask.

The reference mask is constructed as a foreground-dominance mask in the time-frequency domain:

- foreground-dominant bins are assigned value 1;
- background-dominant bins are assigned value 0.

This turns the task into a segmentation-style mask evaluation problem.

---

## 3. Evaluation Metrics

The following metrics are used:

- **Precision**
- **Recall**
- **F1**
- **Intersection over Union (IoU)**
- **Dice**

The numbers reported below are macro-averages over files. Each CSV stores per-file metrics, and the final reported value is the mean of each metric column.

### 3.1 Metric Priority

For this task, false negatives are slightly more harmful than false positives.

A false negative means that a true event-related time-frequency bin is missing from the predicted mask. This removes part of the target event structure and may make later analysis or reconstruction incomplete.

A false positive means that additional background is included in the mask. This reduces mask cleanliness, but it does not remove the event itself.

Therefore, recall is important because it reflects how much event structure is preserved. However, recall alone is not sufficient: a method can achieve high recall by producing overly broad masks. For this reason, the main comparison should focus on:

1. **IoU**, because it measures direct overlap between predicted and reference masks;
2. **F1 / Dice**, because they balance precision and recall;
3. **Recall**, as a secondary metric for event preservation;
4. **Precision**, as a diagnostic metric for over-detection.

Accuracy is not emphasized because time-frequency masks can be highly imbalanced and dominated by true-negative background bins.

---

## 4. Compared Methods

The current experiments are organized around five methods:

| ID | Method | Short Description |
|---|---|---|
| Method 1 | Pure DSP-SPP | Signal-processing method based on SPP-like soft mapping, fixed thresholding, and structural post-processing |
| Method 2 | Pure ML | Handcrafted spectrogram features + LightGBM + 2D CRF |
| Method 3 | Pure DL | U-Net spectrogram segmentation |
| Method 4 | Hybrid DSP-ML | DSP-guided features + LightGBM + 2D CRF |
| Method 5 | Hybrid DSP-DL | DSP-guided U-Net |

The former baseline/reference result is not treated as one of the five final methods. If it is reported, it should be clearly marked as an external reference rather than as Method 1.

---

## 5. Quantitative Results

### 5.1 Threshold Protocol

The threshold protocol is method-dependent:

| Method | Threshold protocol |
|---|---|
| Method 1: Pure DSP-SPP | Fixed threshold `0.40` |
| Method 2: Pure ML | Validation search over `0.40`, `0.45`, `0.50`, `0.55`; selected by highest validation mean IoU |
| Method 3: Pure DL | Validation search over `0.40`, `0.45`, `0.50`, `0.55`; selected by highest validation mean IoU and stored in the checkpoint |
| Method 4: Hybrid DSP-ML | Validation search over `0.40`, `0.45`, `0.50`, `0.55`; selected by highest validation mean IoU |
| Method 5: Hybrid DSP-DL | Validation search over `0.40`, `0.45`, `0.50`, `0.55`; selected by highest validation mean IoU and stored in the checkpoint |

The held-out test split is used only for final evaluation, not for threshold selection.

### 5.2 Recorded Quantitative Results

All methods are evaluated under the same FUSS benchmark protocol. IoU is used as the primary metric, while F1/Dice, precision, and recall are reported as supporting metrics.

| Method | IoU | F1/Dice | Precision | Recall |
|---|---:|---:|---:|---:|
| Baseline | 0.2496 | 0.3239 | 0.5357 | 0.4786 |
| M1 | 0.5525 | 0.5953 | 0.6440 | 0.8511 |
| M2 | 0.4479 | 0.5721 | 0.5874 | 0.6946 |
| M3 | 0.4942 | 0.5020 | 0.6455 | 0.7903 |
| M4 | 0.4692 | 0.5915 | 0.5942 | 0.7154 |
| M5 | 0.6438 | 0.6450 | 0.6438 | 0.9430 |

---

## 6. Main Findings

### 6.1 Best Overall Method

Using IoU as the primary metric, the best current method is:

- **Hybrid DSP-DL: DSP-guided U-Net**
- IoU = **0.6438**
- F1 / Dice = **0.6450**
- Recall = **0.9430**
- Precision = **0.6438**

This method also achieves the best F1 / Dice score in the current results.

### 6.2 Best Precision

The highest precision is obtained by:

- **Pure DL: U-Net**
- Precision = **0.6455**

Precision measures how clean the predicted mask is. A high value means that fewer background bins are incorrectly included as foreground.

### 6.3 Best Recall

The highest recall is obtained by:

- **Hybrid DSP-DL: DSP-guided U-Net**
- Recall = **0.9430**

Recall is important for this task because it indicates how much of the true event-related mask is preserved.

---

## 7. Method-Level Interpretation

### 7.1 Pure DSP-SPP

M1 performs strongly despite being non-learned:

IoU = 0.5525
Recall = 0.8511

This indicates that the DSP-SPP design captures useful time-frequency structure for foreground mask estimation.

---

### 7.2 Pure ML: LightGBM + 2D CRF

The pure ML method improves over the baseline/reference result in the main metrics.

This suggests that handcrafted time-frequency features combined with a supervised classifier provide a stronger decision rule than the baseline/reference approach.

The method is relatively balanced: it does not simply maximize recall, and it obtains a reasonable trade-off between precision and recall.

---

### 7.3 Pure DL: U-Net

The pure DL method obtains strong precision and a relatively strong IoU. This indicates that the U-Net can learn useful segmentation structure from the FUSS benchmark.

However, its F1 / Dice score is lower than the best hybrid method. This suggests that the plain spectrogram-to-mask U-Net benefits from additional guidance.

---

### 7.4 Hybrid DSP-ML

The hybrid DSP-ML method improves over the pure ML method.

This supports the idea that DSP-derived information can be useful as an additional prior feature representation. The DSP features provide event-presence cues, while the ML classifier learns how to use these cues together with the original spectrogram features.

---

### 7.5 Hybrid DSP-DL

The hybrid DSP-DL method is currently the strongest method overall in the available FUSS results.

It combines the raw spectrogram representation with a DSP-derived guidance map. The result suggests that the U-Net can use the DSP prior to preserve more foreground event structure while maintaining high precision.

This is consistent with the hypothesis that DSP information is useful not only as a standalone method, but also as a guidance signal for supervised models.

---

## 8. Ranking

### 8.1 Ranking by IoU

| Rank | Method | IoU |
|---:|---|---:|
| 1 | Hybrid DSP-DL: DSP-guided U-Net | 0.6438 |
| 2 | Pure DL: U-Net | 0.4942 |
| 3 | Hybrid DSP-ML: DSP-guided LightGBM + 2D CRF | 0.4692 |
| 4 | Pure ML: LightGBM + 2D CRF | 0.4479 |
| 5 | Baseline / provided reference | 0.3317 |

### 8.2 Ranking by F1 / Dice

| Rank | Method | F1 / Dice |
|---:|---|---:|
| 1 | Hybrid DSP-DL: DSP-guided U-Net | 0.6450 |
| 2 | Hybrid DSP-ML: DSP-guided LightGBM + 2D CRF | 0.5915 |
| 3 | Pure ML: LightGBM + 2D CRF | 0.5721 |
| 4 | Pure DL: U-Net | 0.5020 |
| 5 | Baseline / provided reference | 0.4328 |

---

## 9. Current Conclusion

The FUSS benchmark results show that supervised and hybrid approaches outperform the baseline/reference result.

The hybrid methods are especially important:

- Hybrid DSP-ML improves over pure ML.
- Hybrid DSP-DL performs best overall under IoU and F1 / Dice.

The current best candidate is therefore **Hybrid DSP-DL**, because it achieves the best overlap metrics while preserving a high amount of foreground structure.

The main experimental conclusion is that DSP information is useful not only as an independent method but also as a prior or guidance signal for learned mask estimation models.

---

## 10. Notes and Next Steps

1. Keep the FUSS benchmark results separate from marine qualitative examples.
2. Use IoU as the primary metric for method ranking.
3. Report F1/Dice as second metric, precision, and recall as supporting metrics.
4. Keep the baseline clearly marked as an external reference, not as one of M1-M5.
5. Keep local dataset paths and generated outputs outside Git.
