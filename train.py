import os
import pandas as pd
from sklearn.model_selection import train_test_split
from models.text_regressor import TextRegressor
from models.audio_regressor import AudioRegressor
from models.video_regressor import VideoRegressor
import numpy as np
import joblib # for saving/loading models

# PAD Mapping Dictionary using approximate values from studies for Audio and Video modalities
emotion_to_pad = {
    "joy":       [0.76, 0.48, 0.35],
    "fear":      [-0.64, 0.60, -0.43],
    "neutral":   [0.0, 0.0, 0.0],
    "calm":      [0.4, -0.2, 0.3],
    "happy":     [0.76, 0.48, 0.35],
    "sad":       [-0.63, -0.27, -0.33],
    "sadness":   [-0.63, -0.27, -0.33],
    "anger":     [-0.51, 0.59, 0.25],
    "angry":     [-0.51, 0.59, 0.25],
    "fearful":   [-0.8, 0.8, -0.7],
    "disgust":   [-0.6, 0.35, 0.11],
    "surprised": [0.4, 0.9, 0.3],
    "shame":     [-0.45, 0.30, -0.20],
    "guilt":     [-0.45, 0.30, -0.15],
}

# RAVDESS emotion code mapping, for audio and video files

ravdess_emotion_map = {
    "01": "neutral",
    "02": "calm",
    "03": "happy", 
    "04": "sad",
    "05": "angry",
    "06": "fearful",
    "07": "disgust",
    "08": "surprised"
}

# Helper function to load in data from RAVDESS dataset files for training on audio and visual

def load_ravdess_data(file_list, extensions):
    # Paths will hold file paths and targets will hold corresponding PAD values
    paths, targets = [], []
    for file in os.listdir(file_list):
        if any(file.endswith(ext) for ext in extensions):
            # Split filename by '-' to extract emotion code
            parts = file.split("-")
            # 3rd element is the emotion code
            emotion_code = parts[2]
            emotion = ravdess_emotion_map.get(emotion_code)
            if emotion:
                pad_values = emotion_to_pad[emotion]
                paths.append(os.path.join(file_list, file))
                targets.append(pad_values)
    return paths, np.array(targets)

# TEXT TRAINING

# Load text dataset for testing, ISEAR CSV
def load_isear(sample_size=None):
    # Only pulling the text and emotion rows and skipping malformed rows
    df = pd.read_csv("data/text_data/isear.csv", sep="|", usecols = ["Field1", "SIT"], engine="python", on_bad_lines = "skip")

    df = df.rename(columns={"Field1": "emotion", "SIT": "sentence"})

    # Map emotions to PAD values using the function above
    df["PAD"] = df["emotion"].map(emotion_to_pad)

    if sample_size:
        df = df.sample(n=sample_size, random_state=42)
    
    return df

# Load in a sample of 500 for initial training for sake of time
df = load_isear(sample_size=500)

# Now have just three columns (emotion, sentence, and PAD)
texts = df["sentence"].tolist()
pad_targets_text = df["PAD"].tolist()

# Create a train test split for text entries
train_texts, test_texts, train_targets_text, test_targets_text = train_test_split(
    texts, pad_targets_text, test_size = 0.2, random_state = 42)

# Initialize and train the text regressor model
regressor = TextRegressor()
regressor.train(train_texts, train_targets_text)

# Save the trained text model for future use
joblib.dump(regressor, "text_regressor_model.joblib")

# AUDIO TRAINING

# Pull in the data using the function defined above
audio_paths, pad_targets_audio = load_ravdess_data("data/audio_data", [".wav"])

# Split into train/test sets
train_audios, test_audios, train_targets_audio, test_targets_audio = train_test_split(
    audio_paths, pad_targets_audio, test_size=0.2, random_state=42)

# Initialize and train the audio regressor model
audio_regressor = AudioRegressor()
audio_regressor.train(train_audios, train_targets_audio)

# Save the trained audio model for future use
joblib.dump(audio_regressor, "audio_regressor_model.joblib")

# VIDEO TRAINING

# Pull in the video data using the defined function as done above for audio
video_paths, pad_targets_video = load_ravdess_data("data/video_data", [".mp4"])

# Split into train/test sets, in this 80-20 split for train-test
train_videos, test_videos, train_targets_video, test_targets_video = train_test_split(
    video_paths, pad_targets_video, test_size=0.2, random_state=42)

# Initialize and train the video regressor model
video_regressor = VideoRegressor()
video_regressor.train(train_videos, train_targets_video)

# Save the trained video model for future use
joblib.dump(video_regressor, "video_regressor_model.joblib")

print("Training complete. Models saved for text, audio, and video modalities.")