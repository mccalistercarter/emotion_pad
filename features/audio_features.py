import librosa
import numpy as np
import soundfile as sf

def get_audio_features(audio_path, target_sr=16000, max_seconds=5):
    # Load in the audio file to extract features from
    # Load was taking 2-2.5 seconds initially so changed to soundifle and downsampling first

    y, sr = sf.read(audio_path, dtype='float32')

    # Convert stereo to mono
    if y.ndim > 1:
        y = np.mean(y, axis=1)

    # Trim to max seconds if provided
    y = y[:int(max_seconds * sr)]
    
    if sr != target_sr:
        y = librosa.resample(y.T, orig_sr=sr, target_sr=target_sr).T
        sr = target_sr

    # Feature extraction with reduced frames
    hop_length = 2048
    # Extract MFCC, Mel-frequency cepstral coefficients, spectral features from the audio
    # Take the mean across time to get a fixed-size vector while using fewer frames
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=8, hop_length=hop_length).mean(axis=1)

    # Obtain energy calculations from the audio signal
    energy = np.array([librosa.feature.rms(y=y, hop_length=hop_length).mean()])

    # Get pitch (fundamental frequency) using librosa's yin function
    # Between 50Hz and 300Hz for typical human voice range with buffer
    pitch = np.array([librosa.yin(y=y, fmin=50, fmax=300, frame_length=hop_length).mean()])

    # Spectral centroid, the "center of mass" of the spectrum, also used fewer frames
    spectral_centroid = np.array([librosa.feature.spectral_centroid(y=y, sr=sr, hop_length=hop_length).mean()])

    # Combine all features into a single feature vector
    # All were kept as 1D arrays for concatenation
    features = np.concatenate([mfcc, energy, pitch, spectral_centroid])
    
    return features