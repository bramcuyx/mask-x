# Method 1: DSP-Based SPP Mask Estimation for Marine Event Masks

## 1. Goal

The goal of this method is to estimate binary time-frequency masks for marine acoustic events using a purely DSP-based pipeline.

More specifically, the method takes an input waveform and produces a binary mask that indicates which time-frequency bins are likely to contain an event rather than background noise.

At this stage, this method is intended to serve as the main DSP candidate before exploring ML / DL alternatives.

---

## 2. Baseline Summary

The provided baseline uses an NMF-based approach on the spectrogram, followed by median subtraction and thresholding, together with prior knowledge about background-only regions near the clip edges.

The baseline therefore relies on:
- spectrogram computation,
- partial NMF reconstruction,
- background estimation,
- hard thresholding to produce a binary mask.

This project method explores an alternative DSP route based more directly on noise statistics and soft event-presence estimation.

---

## 3. Proposed DSP Method

### 3.1 Overview

This method estimates a binary mask using a DSP-based SPP-style pipeline.

Instead of relying on NMF reconstruction, it works directly in the spectrogram domain:
1. compute the spectrogram,
2. estimate background noise statistics,
3. compute a per-bin saliency / score map,
4. convert this into a soft SPP-like map,
5. smooth and threshold the result,
6. apply structural post-processing.

The main idea is to identify time-frequency bins that deviate sufficiently from the background noise while preserving local event structure.

---

### 3.2 Method Steps

Given an input waveform \(x[n]\), the method proceeds as follows:

1. **Audio loading and optional resampling**  
   The input waveform is loaded and converted to mono if necessary.

2. **Spectrogram computation**  
   A spectrogram \(S(t,f)\) is computed using STFT.

3. **Log-power representation**  
   The spectrogram is converted to a log-power representation.

4. **Background estimation**  
   Background statistics are estimated per frequency bin.
   In the marine setting, edge regions are used for this purpose.

5. **Per-bin saliency estimation**  
   A z-score-like map is computed to measure how strongly each time-frequency bin deviates from the estimated background.

6. **Local time-frequency fusion**  
   A light neighborhood fusion step is applied to incorporate local time-frequency context without excessively blurring event structure.

7. **Soft SPP-like mapping**  
   The score map is converted into a soft probability-like map.

8. **Binary thresholding**  
   The soft map is thresholded to produce a binary mask.

9. **Structural post-processing**  
   Small isolated components are removed, and very short nearly full-band artifacts are suppressed.

---

### 3.3 Pseudocode

```text
    Input: waveform x[n]

    1. Load waveform
    2. Convert to mono if needed
    3. Resample if required
    4. Compute spectrogram S(t,f)
    5. Convert S(t,f) to log-power representation
    6. Estimate per-frequency background statistics
    7. Compute raw saliency / z-score map
    8. Fuse raw saliency with local TF neighborhood information
    9. Convert fused score map into soft SPP-like probabilities
    10. Threshold the soft map to obtain a binary mask
    11. Apply structural post-processing
    12. Return final binary time-frequency mask
```

### 3.4 Main Equations

Let \(L(t,f)\) denote the log-power spectrogram.

A per-frequency normalized saliency map is computed as:

\[
z(t,f) = \frac{L(t,f) - \mu_f}{\sigma_f + \epsilon}
\]

where:
- \(\mu_f\) is the estimated background center for frequency bin \(f\),
- \(\sigma_f\) is the estimated background scale,
- \(\epsilon\) is a small constant for numerical stability.

A fused local score map is then obtained by combining the raw score with a local time-frequency average:

\[
z_{\text{fused}}(t,f) = \lambda z_{\text{raw}}(t,f) + (1-\lambda) z_{\text{local}}(t,f)
\]

The fused score map is converted into a soft SPP-like map using a sigmoid-type transformation:

\[
p(t,f) = \sigma\left(\alpha \left(z_{\text{fused}}(t,f) - \beta\right)\right)
\]

Finally, the binary mask is obtained by thresholding:

\[
M(t,f) =
\begin{cases}
1, & p(t,f) \ge \tau \\
0, & \text{otherwise}
\end{cases}
\]

---

## 4. Current Implementation Settings

The current implementation uses the following settings:

- `NPERSEG = 256`
- `NOVERLAP = 128`
- `NOISE_MODE = "edge"`
- `NOISE_QUANTILE = 0.2`

- `SPP_ALPHA = 1.5`
- `SPP_BETA = 2.0`
- `SMOOTH_KERNEL = (3, 3)`
- `FINAL_THRESHOLD = 0.4`

- `POSTPROCESS = True`
- `MIN_REGION_SIZE = 8`
- `MAX_FULLBAND_RATIO = 0.9`
- `MAX_IMPULSE_WIDTH_FRAMES = 2`

---

## 5. Parameter Rationale

### STFT parameters
`NPERSEG = 256` and `NOVERLAP = 128` provide a practical balance between time resolution and frequency resolution while keeping the representation stable.

### Noise estimation
`NOISE_MODE = "edge"` is used in the marine setting because the clip edges contain background-only regions that can be exploited for background estimation.

`NOISE_QUANTILE = 0.2` is retained as a conservative low-energy reference where relevant.

### Soft SPP-like mapping
`SPP_ALPHA = 1.5` controls the steepness of the probability mapping.

`SPP_BETA = 2.0` sets the effective offset in the score-to-probability conversion. Compared with more conservative settings, this lower beta reduces the risk of producing nearly empty masks on marine clips.

### Thresholding
`FINAL_THRESHOLD = 0.4` is used to convert the soft map into a binary mask. This relatively moderate threshold was selected to avoid over-suppressing weak event regions.

### Smoothing
`SMOOTH_KERNEL = (3, 3)` applies mild local smoothing in the time-frequency domain and helps reduce isolated noisy activations without strongly blurring narrow structures.

### Post-processing
Structural post-processing is enabled:
- `MIN_REGION_SIZE = 8` removes very small connected components,
- `MAX_FULLBAND_RATIO = 0.9` identifies nearly full-band activations,
- `MAX_IMPULSE_WIDTH_FRAMES = 2` suppresses very short vertical full-band artifacts.

---

## 6. Literature Motivation

This DSP method was motivated by several ideas from the literature:

### Soft SPP instead of hard binary decisions
SPP-related work suggests that soft presence probabilities are often more stable than hard VAD-like decisions. This motivated the use of a soft SPP-like map before the final thresholding step.

### Time-frequency neighborhood information
Prior work on SPP estimation indicates that neighboring time-frequency bins are not independent and that local time-frequency context can improve estimation. This motivated the local TF fusion step used in the current method.

### Structural completeness
Work on weakly labeled sound separation suggests that a good mask should not only pick the most salient isolated points, but should also preserve meaningful event structure. This motivated the structural post-processing step.

---

## 7. FUSS Benchmark Setup

To obtain quantitative evaluation metrics, this method is also tested on the FUSS dataset.

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

These metrics are borrowed from image segmentation evaluation, which is appropriate here because the task is also a mask estimation problem.

### Current benchmark protocol
The current FUSS benchmark run uses:
- random seed: `42`
- evaluated examples: `3000`

---

## 8. Current DSP Benchmark Result

With the current parameter setting, the DSP method produced the following FUSS benchmark result:

- **Precision:** 0.3767
- **Recall:** 0.8726
- **F1:** 0.4540
- **IoU:** 0.3408
- **Dice:** 0.4540

### Interpretation
This result indicates that the current DSP method is highly recall-oriented:
- it captures many foreground bins,
- but it also introduces a substantial number of false positives.

In other words, the method is relatively sensitive, but still tends to over-detect foreground structure.

At the current stage, this quantitative result is mainly used as an external benchmark reference for the DSP method.

### Optional note on baseline comparison
A direct quantitative comparison with the provided baseline should only be reported when both methods are evaluated on the same FUSS subset, using the same seed and the same number of examples.  
If this condition is not satisfied yet, the baseline comparison should be reported separately after rerunning the baseline under the same benchmark protocol.

---

## 9. Marine Data Application

The DSP-SPP method was also applied to the original marine clips in order to inspect its behaviour on the target task data.

Since no ground-truth masks are available for the marine dataset, this part is used only for qualitative analysis.  
The purpose is not to make a strict quantitative claim, but to observe whether the produced masks are visually plausible, whether they capture structured event regions, and whether they remain overly sparse, overly dense, or dominated by artifacts.

### 9.1 Representative Examples

#### Example 1
![Representative example 1](dsp_images/200504111910_816_mono1_snippet_1040_1045.png)

Observation: In this example, the method produces a non-empty and reasonably structured mask concentrated around the central event region. The strongest vertical structure around the middle of the clip is captured, and the mask also retains some additional lower-frequency activity nearby. Although the mask is still sparse and does not fully cover all visible structure, it remains visually plausible and substantially more informative than an empty output.


#### Example 2
![Representative example 2](dsp_images/200504111910_816_mono1_snippet_3726_3728.png)

Observation: This clip shows another acceptable case where the method captures a compact cluster of activations near the central event area. Compared with the previous example, the detected region is weaker and more fragmented, but the output still follows a visible concentration of activity rather than degenerating into isolated random noise. This suggests that the method can preserve some meaningful event structure even when the signal is less prominent.

### 9.2 Failure Cases

#### Example 3
![Failure case 1](dsp_images/channelA_2021-03-17_17-21-13_snippet_256_256.png)

Observation: This is a failure case. The spectrogram contains a weak and very localized structure near the center of the clip, but the method produces an empty mask. This indicates that the current DSP configuration still misses some weaker events, especially when the event is not sufficiently separated from the estimated background.

#### Example 4
![Failure case 2](dsp_images/channelA_2021-03-17_18-53-01_snippet_270_270.png)

Observation: This is another failure case with a nearly empty output. As in the previous example, only a very weak local structure is visible in the spectrogram, and the current method does not retain it after thresholding and post-processing. This suggests that the method remains sensitive to low-energy or weakly contrasted events.


### Summary of Marine Qualitative Behaviour

Overall, the marine examples show that the DSP-SPP method is capable of producing visually meaningful masks when the event structure is sufficiently pronounced. In such cases, the output is localized around the central region and captures at least part of the visible time-frequency structure.

However, the failure cases also show that the current method is still not consistently robust for weaker events. In particular, low-energy or weakly contrasted structures may still be removed completely, leading to empty masks. Therefore, while the current DSP version is clearly more usable than earlier overly conservative versions, it still has limited sensitivity in difficult marine examples.

---

## 10. What Was Tried During Development

Several variants were explored during development.

### Initial DSP-SPP version
A basic version using:
- spectrogram computation,
- background estimation,
- z-score-like saliency,
- sigmoid-based soft mapping,
- thresholding.

This provided the first workable DSP baseline.

### Strong smoothing before saliency estimation
A variant that smoothed the log-spectrogram before score estimation was tested.  
This led to much lower recall and worse overall F1 / IoU, so it was not retained.

### Local TF fusion variant
A lighter local time-frequency fusion strategy was then tested.  
This preserved local detail better and was therefore retained as a better structural modification.

### More statistical SPP-like formulation
A more explicitly statistical SPP-inspired version was also tested.  
In the current implementation, this did not outperform the simpler main DSP version, so it was not kept as the main candidate.

### Structural post-processing
A post-processing stage was added to suppress tiny isolated regions and very short full-band artifacts.  
This improved the structural cleanliness of the masks, although the quantitative gains were limited.

---

## 11. Current Status

At the current stage, this DSP method is the main pure DSP candidate.

It has the following advantages:
- fully DSP-based,
- interpretable processing steps,
- soft probability stage before final thresholding,
- structurally cleaner masks than early raw versions.

It also still has clear limitations:
- the current version remains relatively recall-heavy,
- false positives are still substantial,
- quantitative performance leaves room for improvement.

Nevertheless, it is already a reasonable and complete DSP method candidate for this project and provides a meaningful point of comparison for future ML / DL / hybrid methods.

---

## 12. Next Steps

The next steps for this DSP method are:

1. document marine qualitative observations more systematically,
2. compare it directly with the provided baseline,
3. use it as the main DSP reference method when developing later ML / DL approaches.