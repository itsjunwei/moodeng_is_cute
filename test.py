import numpy as np
import librosa
 
def spectrogram(audio_input):
    """
    Computes the Short-Time Fourier Transform (STFT) spectrogram for a single-channel audio signal.
 
    Parameters:
    audio_input (np.ndarray): 1D array of audio samples with shape (n_samples,).
 
    Returns:
    np.ndarray: Spectrogram with shape (time_frames, frequency_bins).
    """
    # Define parameters
    fs = 22050          # Sampling rate (Hz)
    nfft = 1024         # Number of FFT points
    hop_length = 512    # Hop length (samples)
    win_length = 1024   # Window length (samples)
    window = 'hann'     # Window type
 
    # Ensure the audio is in the correct format
    audio = np.asfortranarray(audio_input)
 
    # Compute STFT
    stft_result = librosa.stft(audio, n_fft=nfft, hop_length=hop_length,
                               win_length=win_length, window=window)
 
    # Transpose to get shape (time_frames, frequency_bins)
    spectrogram = stft_result.T
 
    return spectrogram
 
def get_mel_spectrogram(linear_spectra):
    """
    Converts a linear spectrogram to a Mel-scaled spectrogram and applies logarithmic scaling.
 
    Parameters:
    linear_spectra (np.ndarray): 2D array of the linear spectrogram with shape (time_frames, frequency_bins).
 
    Returns:
    np.ndarray: Log-Mel spectrogram with shape (time_frames, mel_bins).
    """
    # Define parameters
    fs = 22050          # Sampling rate (Hz)
    nfft = 1024         # Number of FFT points
    nb_mel_bins = 128   # Number of Mel bins
    fmin = 50           # Minimum frequency (Hz)
    fmax = None         # Maximum frequency (Hz), None defaults to Nyquist
 
    # Create Mel filter bank
    mel_wts = librosa.filters.mel(sr=fs, n_fft=nfft, n_mels=nb_mel_bins,
                                  fmin=fmin, fmax=fmax).T  # Shape: (frequency_bins, mel_bins)
 
    # Compute magnitude squared of the spectrogram
    mag_spectra = np.abs(linear_spectra) ** 2  # Shape: (time_frames, frequency_bins)
 
    # Apply Mel filter bank
    mel_spectra = np.dot(mag_spectra, mel_wts)  # Shape: (time_frames, mel_bins)
 
    # Convert to decibels
    log_mel_spectra = librosa.power_to_db(mel_spectra, ref=np.max)
 
    return log_mel_spectra
 
# Example usage:
if __name__ == "__main__":
 
    # Compute the spectrogram
    spec = spectrogram(audio) # resample accordingly if needed
    print(f"Spectrogram shape: {spec.shape}")  # (time_frames, frequency_bins)
 
    # Compute the Mel spectrogram
    mel_spec = get_mel_spectrogram(spec)
    print(f"Mel Spectrogram shape: {mel_spec.shape}")  # (time_frames, mel_bins)