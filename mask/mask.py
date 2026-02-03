import nmf
import pathlib
import soundfile as sf
import scipy.signal
import matplotlib.pyplot as plt
import numpy as np

def estimate_mask(file, rank = 25, reduced_rank=15, nperseg = 256, noverlap = 128, padding=1, maxiter=1000, plot = False):
    """Estimate a binary mask for the input audio file using NMF.

    Args:
        file (pathlib.Path): Path to the input audio file.
        rank (int): Rank for NMF decomposition.
        nperseg (int): Length of each segment for STFT.
        noverlap (int): Number of overlapping samples between segments.
        padding (int): Padding around the event in seconds.

    intermediates:
        data (np.ndarray): Audio time series data.
        sampleRate (int): Sampling rate of the audio file.
        f (np.ndarray): Frequencies of the STFT.
        t (np.ndarray): Times of the STFT.
        Sxx (np.ndarray): Spectrogram of the audio signal.
        B (np.ndarray): Basis matrix from NMF.
        G (np.ndarray): Activation matrix from NMF.

    Returns:
        mask (np.ndarray): Estimated binary mask.
        residual (np.ndarray): Residual signal subtraction of the low rank reconstruction.
    """
    data, sampleRate = sf.read(file)

    f,t,Sxx = scipy.signal.spectrogram(data, fs=sampleRate, nperseg=nperseg, noverlap=noverlap)
    
    B, G = nmf.nmf_approximation(Sxx, rank=rank, maxiter=1000, alpha=0.0, beta=0.0)

    Sxx_reconstructed = B @ G

    if plot:
        plt.figure(figsize=(12, 6))
        plt.subplot(2, 1, 1)
        plt.title('Original Spectrogram')
        plt.pcolormesh(t, f, 10 * np.log10(Sxx + 1e-10), shading='gouraud')
        plt.ylabel('Frequency [Hz]')
        plt.xlabel('Time [sec]')
        plt.colorbar(label='Intensity [dB]')

        plt.subplot(2, 1, 2)
        plt.title('NMF Reconstruction')
        plt.pcolormesh(t, f, 10 * np.log10(Sxx_reconstructed + 1e-10), shading='gouraud')
        plt.ylabel('Frequency [Hz]')
        plt.xlabel('Time [sec]')
        plt.colorbar(label='Intensity [dB]')

        plt.tight_layout()
        plt.show()

    # reconstruct the noise keeping only the components with highest energy
    beginning_rank = rank - reduced_rank
    ordering = np.argsort(np.sum(B**2, axis= 0)*np.sum(G, axis= 1))
    B_ordered = B[:, ordering]
    G_ordered = G[ordering, :]
    B_reduced = B_ordered[:, beginning_rank:]
    G_reduced = G_ordered[beginning_rank:, :]
    Sxx_reduced = B_reduced @ G_reduced
    
    residual = Sxx - Sxx_reduced

    if plot:
        plt.figure(figsize=(12, 6))
        plt.subplot(2, 1, 1)
        plt.title('Reduced NMF Reconstruction')
        plt.pcolormesh(t, f, 10 * np.log10(Sxx_reduced + 1e-10), shading='gouraud')
        plt.ylabel('Frequency [Hz]')
        plt.xlabel('Time [sec]')
        plt.colorbar(label='Intensity [dB]')

        plt.subplot(2, 1, 2)
        plt.title('Residual Spectrogram')
        plt.pcolormesh(t, f, 10 * np.log10(np.maximum(residual, 1e-10)), shading='gouraud')
        plt.ylabel('Frequency [Hz]')
        plt.xlabel('Time [sec]')
        plt.colorbar(label='Intensity [dB]')

        plt.tight_layout()
        plt.show()
    
    nframes = Sxx.shape[1]
    frames_per_sec = nframes / (t[-1]-t[0])
    start_frame = int(padding*frames_per_sec)
    end_frame = int((t[-1]-padding)*frames_per_sec)

    mask = np.zeros_like(Sxx, dtype=bool)
    mask[:, start_frame:end_frame] = 0
    medi = np.quantile(residual,0.75, axis=1, keepdims=True)
    var = (residual-medi)/medi  


def plot_masked_spect(Sxx, mask, f, t):
    """Plot the spectrogram with the mask overlaid.

    Args:
        Sxx (np.ndarray): Spectrogram of the audio signal.
        mask (np.ndarray): Binary mask.
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    plt.title('Masked Spectrogram')
    plt.pcolormesh(t, f, 10 * np.log10(Sxx + 1e-10), shading='gouraud', vmin=-130, vmax=-70)
    plt.colorbar(label='Intensity [dB]')
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [sec]')
    plt.tight_layout()
    ax.imshow(
        mask,
        aspect='auto',
        origin='lower',
        cmap='Reds',
        alpha=0.3,
        extent=[t[0], t[-1], f[0], f[-1]],
        zorder=3
    )
    plt.show()

def estimate_mask_median_subtraction(B, G, f, t, threshold=3, padding=1.0):
    """Estimate a binary mask using median subtraction on NMF reconstruction.
    
    This method calculates the median of the NMF reconstruction in the first second
    (where no event is expected), then subtracts it from the entire reconstruction
    in dB scale. Values above the threshold are marked as mask pixels.
    
    Args:
        B (np.ndarray): Basis matrix from NMF (frequency x components).
        G (np.ndarray): Activation matrix from NMF (components x time).
        f (np.ndarray): Frequency array from spectrogram.
        t (np.ndarray): Time array from spectrogram.
        threshold (float): Threshold in dB for mask detection. Default is 3.
        padding (float): Padding in seconds at the start and end where no event is expected. Default is 1.0.
    
    Returns:
        mask (np.ndarray): Binary mask where True indicates detected event.
        med_substracted (np.ndarray): Median-subtracted spectrogram in dB.
        median (np.ndarray): Estimated median spectrum.
    """
    # Calculate NMF reconstruction
    nmf_reconstruction = B @ G
    
    # Calculate frames per second
    nframes = nmf_reconstruction.shape[1]
    frames_per_second = nframes / (t[-1] - t[0])
    
    # Calculate event boundaries in frames
    event_start_frame = int(padding * frames_per_second)
    event_end_frame = nframes - int(padding * frames_per_second)
    
    # Estimate median from the first second (where no event is present)
    median_nmf = np.quantile(np.abs(nmf_reconstruction)[:, :int(frames_per_second)], 0.5, axis=1)
    
    # Subtract median in dB scale
    med_substracted_nmf = 10 * np.log10(nmf_reconstruction) - 10 * np.log10(median_nmf[:, np.newaxis])
    
    # Create mask based on threshold
    mask_med_subs_nmf = med_substracted_nmf > threshold
    
    # Zero out mask before event start and after event end
    mask_med_subs_nmf[:, :event_start_frame] = 0
    mask_med_subs_nmf[:, event_end_frame:] = 0
    
    return mask_med_subs_nmf, med_substracted_nmf, median_nmf