# fft_filters.py - FFT processing filters for real-time analyzer
import numpy as np
import scipy.ndimage

def peak_picking(magnitude, threshold_ratio=0.3):
    """Keep only peaks above a threshold and higher than neighbors."""
    threshold = np.max(magnitude) * threshold_ratio
    peaks = (magnitude > threshold) & \
            (magnitude > np.roll(magnitude, 1)) & \
            (magnitude > np.roll(magnitude, -1))
    return magnitude * peaks

def harmonic_masking(magnitude, sample_rate, fft_size):
    """Suppress harmonics of dominant frequencies."""
    result = magnitude.copy()
    freq = np.fft.rfftfreq(fft_size, 1 / sample_rate)
    peak_indices = np.argpartition(magnitude, -5)[-5:]

    for idx in peak_indices:
        base_freq = freq[idx]
        if base_freq < 20: continue
        harmonics = [2, 3, 4, 5]
        for h in harmonics:
            target_freq = base_freq * h
            target_idx = np.argmin(np.abs(freq - target_freq))
            if 0 <= target_idx < len(result):
                result[target_idx] *= 0.1  # suppress overtone
    return result

def smoothing(magnitude, sigma=1.0):
    """Apply Gaussian smoothing."""
    return scipy.ndimage.gaussian_filter1d(magnitude, sigma)
