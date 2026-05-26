# Method 2: Pure ML LightGBM + CRF Mask Estimation

## 1. Goal

The goal of this method is to estimate binary time-frequency masks for acoustic events using a purely machine-learning-based pipeline.

More specifically, the method takes an input waveform, computes a spectrogram, extracts hand-crafted time-frequency features from the mixture spectrogram, and trains a classifier to predict whether each time-frequency bin belongs to foreground/event activity or background.

At this stage, this method is intended to serve as the main pure ML candidate before moving to DL and hybrid alternatives.

---

## 2. Baseline Summary

The provided baseline uses an NMF-based approach on the spectrogram, followed by median subtraction and thresholding, together with prior knowledge about background-only regions near the clip edges.

The baseline therefore relies on:
- spectrogram computation,
- partial NMF reconstruction,
- background estimation,
- hard thresholding to produce a binary mask.

This project method explores a different route: instead of manually defining the final mask decision rule, it trains a supervised ML classifier on FUSS-derived ground-truth masks.

---

## 3. Proposed ML Method

### 3.1 Overview

This method formulates time-frequency mask estimation as a per-bin supervised classification problem.

The method works in the spectrogram domain:
1. compute the mixture spectrogram,
2. extract hand-crafted features for each time-frequency bin,
3. construct a ground-truth binary mask from FUSS source references,
4. train a LightGBM classifier,
5. convert classifier probabilities into a soft mask,
6. apply a simplified 2D CRF-style smoothing step,
7. threshold the refined probability map into a binary mask.

The main idea is to let a discriminative classifier learn the mapping from spectrogram-derived features to foreground/background mask labels.

---

### 3.2 Method Steps

Given an input waveform \(x[n]\), the method proceeds as follows:

1. **Audio loading and optional resampling**  
   The input waveform is loaded and converted to mono if necessary.

2. **Spectrogram computation**  
   A spectrogram \(S(t,f)\) is computed using STFT.

3. **Log-power representation**  
   The spectrogram is converted to a log-power representation.

4. **Feature extraction**  
   For each time-frequency bin, a feature vector is computed using local, frequency-wise, time-wise, and positional information.

5. **Ground-truth mask construction on FUSS**  
   Foreground and background source spectrograms are compared to construct a reference binary mask.

6. **Classifier training**  
   A LightGBM classifier is trained to predict foreground/background labels for individual time-frequency bins.

7. **Probability map prediction**  
   The trained classifier outputs a foreground probability for each time-frequency bin.

8. **2D CRF-style smoothing**  
   A lightweight 2D spatial refinement is applied to encourage local mask consistency.

9. **Binary thresholding**  
   The refined probability map is thresholded to produce a binary mask.

---

### 3.3 Pseudocode

```text
    Input: waveform x[n]

    Training:
    1. Load FUSS mixture waveform
    2. Compute mixture spectrogram
    3. Extract per-bin hand-crafted features
    4. Load foreground and background source waveforms
    5. Construct ground-truth binary TF mask
    6. Sample time-frequency bins for training
    7. Train LightGBM classifier

    Inference:
    1. Load waveform
    2. Compute spectrogram
    3. Extract the same per-bin features
    4. Predict foreground probability for each TF bin
    5. Apply simplified 2D CRF smoothing
    6. Threshold probability map
    7. Return binary time-frequency mask
```

---

### 3.4 Main Equations

Let \(L(t,f)\) denote the log-power spectrogram.  
For each time-frequency bin, a feature vector is constructed:

\[
\mathbf{x}_{t,f} =
[
L(t,f),
z_f(t,f),
z_t(t,f),
\mu_{\text{local}}(t,f),
\sigma_{\text{local}}(t,f),
c_{\text{local}}(t,f),
g(t,f),
r_f,
r_t
]
\]

where:
- \(z_f(t,f)\) is a frequency-wise normalized score,
- \(z_t(t,f)\) is a time-wise normalized score,
- \(\mu_{\text{local}}\) and \(\sigma_{\text{local}}\) describe local neighborhood statistics,
- \(c_{\text{local}}\) is a local contrast feature,
- \(g(t,f)\) is a gradient-related feature,
- \(r_f\) and \(r_t\) are normalized frequency and time positions.

The classifier estimates:

\[
p(t,f) = P(M(t,f)=1 \mid \mathbf{x}_{t,f})
\]

A binary mask is then obtained by thresholding:

\[
M(t,f) =
\begin{cases}
1, & p_{\text{refined}}(t,f) \ge \tau \\
0, & \text{otherwise}
\end{cases}
\]

where \(p_{\text{refined}}(t,f)\) denotes the probability map after 2D CRF-style smoothing.

---

## 4. Current Implementation Settings

The current implementation uses the following general settings:

- classifier: `LightGBM`
- post-processing: simplified `2D CRF` smoothing
- input representation: log-power spectrogram
- training target: FUSS-derived foreground/background binary mask
- threshold selection: validation-set search
- selected threshold in the latest run: `0.55`
- evaluation split: held-out FUSS test split
- evaluated examples in the latest test run: `2266`

The exact implementation-level parameters are defined in:
- `ml_lgbm_crf_method.py`
- `evaluate_fuss_ml_lgbm_crf.py`

---

## 5. Parameter Rationale

### Hand-crafted feature representation
The method uses local, frequency-wise, time-wise, and positional features because acoustic events may appear as localized structures in the time-frequency plane. A single log-power value alone is not sufficient to distinguish foreground event energy from background variation.

### LightGBM classifier
LightGBM was selected because it is efficient for tabular features, handles nonlinear feature interactions, and can train faster than many deep models while still providing strong supervised performance.

### CRF-style refinement
The classifier makes per-bin decisions, which may be noisy or fragmented. The simplified 2D CRF-style smoothing step is used to improve local consistency in the output probability map.

### Threshold tuning
A fixed probability threshold is selected on the validation set. This avoids tuning the threshold directly on the test set and makes the final test evaluation more meaningful.

---

## 6. Literature Motivation

This ML method was motivated by several ideas from the literature and related fields:

### Supervised mask estimation
If a dataset with source-level references is available, mask estimation can be treated as supervised binary classification over time-frequency bins.

### Hand-crafted acoustic features
Classical ML methods often benefit from manually designed features that encode energy, contrast, local statistics, and positional information.

### Segmentation refinement
Mask estimation is structurally similar to image segmentation. Post-processing methods that encourage local consistency can improve the spatial structure of predicted masks.

---

## 7. FUSS Benchmark Setup

To obtain quantitative evaluation metrics, this method is tested on the FUSS dataset.

### Why FUSS is used
The original marine dataset does not provide ground-truth masks, so direct computation of segmentation-style metrics is not possible there.

FUSS provides:
- a mixture waveform,
- source waveforms for background and foreground components.

This makes it possible to construct a reference binary mask.

### Input / target definition
- **Input to the method:** the mixture waveform `exampleXXXXX.wav`
- **Reference signal:** the corresponding `foreground*.wav` and `background*.wav` files

The reference binary mask is defined as:
- foreground-dominant bins → mask value 1
- background-dominant bins → mask value 0

### Evaluation metrics
The following segmentation-style metrics are used:
- Precision
- Recall
- F1
- IoU
- Dice

### Current benchmark protocol
The current FUSS benchmark run uses:
- training / validation / test split
- validation-based threshold selection
- held-out test evaluation
- evaluated test examples: `2266`

---

## 8. Current ML Benchmark Result

With the current implementation, the pure ML method produced the following FUSS held-out test result:

- **Precision:** 0.5874
- **Recall:** 0.6946
- **F1:** 0.5721
- **IoU:** 0.4479
- **Dice:** 0.5721
- **Selected threshold:** 0.55

### Interpretation
Compared with the pure DSP method, this ML method is less recall-oriented but substantially more precise and more balanced.

The result suggests that supervised learning from FUSS-derived masks helps reduce false positives compared with the purely DSP-based method.

The method provides a strong non-deep-learning reference point for the project.

---

## 9. Marine Data Application

The ML method can also be applied to the original marine clips after the LightGBM model has been trained on FUSS.

Since no ground-truth masks are available for the marine dataset, this part is used for qualitative analysis only.

The marine inference pipeline is:

```text
    marine waveform
    → spectrogram
    → hand-crafted feature extraction
    → LightGBM probability prediction
    → 2D CRF-style smoothing
    → binary mask
```

The purpose is to inspect whether the ML-trained classifier produces visually plausible masks on the target marine recordings.

Representative examples and failure cases should be added after selecting marine output plots from the generated results.

---

## 10. What Was Tried During Development

Several aspects were explored during development.

### Per-bin supervised classifier
The first version treated each time-frequency bin as an independent sample for supervised classification.

### Local statistical features
Local mean, local standard deviation, contrast, and gradient-like features were included to provide contextual information beyond raw spectrogram energy.

### Frequency-wise and time-wise normalization
Both frequency-wise and time-wise normalized features were used to capture deviations from typical background structure.

### CRF-style smoothing
A simplified 2D smoothing/refinement step was added to make the output mask less fragmented and more spatially coherent.

### Threshold search
Several probability thresholds were tested on the validation set, and the best threshold was selected before final test evaluation.

---

## 11. Current Status

At the current stage, this ML method is the main pure machine-learning candidate.

It has the following advantages:
- supervised and data-driven,
- more balanced than the pure DSP method,
- interpretable hand-crafted feature input,
- faster and simpler than deep learning,
- produces a useful comparison point for hybrid methods.

It also has limitations:
- performance depends heavily on the quality of hand-crafted features,
- it does not learn hierarchical features automatically,
- per-bin classification still requires smoothing or refinement to improve structure,
- FUSS-to-marine generalization still needs qualitative inspection.

---

## 12. Next Steps

The next steps for this method are:

1. inspect ML masks on marine clips,
2. compare pure ML with the pure DSP and pure DL methods,
3. use this method as the reference for the Hybrid DSP-ML method,
4. document how the selected threshold affects precision-recall trade-offs.
