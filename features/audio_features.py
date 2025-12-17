import librosa
import numpy as np

def get_audio_features(audio_path):
    # Load in the audio file to extract features from
    # .load returns the audio time series and the sampling rate
    y, sr = librosa.load(audio_path, sr=None)

    # Extract MFCC, Mel-frequency cepstral coefficients, spectral features from the audio
    # Take the mean across time to get a fixed-size vector of 13 coefficients
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13).mean(axis=1)

    # Obtain energy calculations from the audio signal (average squared amplitude)
    energy = np.array([np.mean(y**2)])

    # Get pitch (fundamental frequency) using librosa's yin function
    # Between 50Hz and 300Hz for typical human voice range with buffer
    pitch = np.array([librosa.yin(y, fmin=50, fmax=300).mean()])

    # Spectral centroid, the "center of mass" of the spectrum
    spectral_centroid = np.array([librosa.feature.spectral_centroid(y=y, sr=sr).mean()])

    # Combine all features into a single feature vector
    # All were kept as 1D arrays for concatenation
    features = np.concatenate([mfcc, energy, pitch, spectral_centroid])
    # Size 16 vector: 13 MFCC + 1 energy + 1 pitch + 1 spectral centroid
    # May have to scale/normalize features before using in model
    
    return features
