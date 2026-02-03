import nmf

def reconstruct_spectrogram(spectrogram, rank=25, maxiter=500, alpha = 1e-2, beta = 1e-8):
    B, G = nmf.nmf_approximation(spectrogram, rank=rank, maxiter=maxiter, alpha=alpha, beta=beta)
    
    reconstructed_spectrogram
