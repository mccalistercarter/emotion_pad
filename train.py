import os
from sklearn.model_selection import train_test_split
from models.text_regressor import TextRegressor
from models.audio_regressor import AudioRegressor
from models.video_regressor import VideoRegressor
import numpy as np
import joblib # for saving/loading models

# PAD Mapping Dictionary using approximate values from studies for Audio and Video modalities
emotion_to_pad = {
    "neutral":   [0.0, 0.0, 0.0],
    "calm":      [0.4, -0.2, 0.3],
    "happy":     [0.8, 0.6, 0.6],
    "sad":       [-0.6, -0.4, -0.5],
    "angry":     [-0.7, 0.9, 0.8],
    "fearful":   [-0.8, 0.8, -0.7],
    "disgust":   [-0.6, 0.3, -0.6],
    "surprised": [0.4, 0.9, 0.3]
}

# RAVDESS emotion code mapping

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

# Training data for linear regression model for text PAD prediction
texts = ["I am happy", "I am sad", "I am angry"]
pad_targets_text = [
    [0.8, 0.5, 0.6],
    [-0.6, -0.4, -0.5],
    [-0.7, 0.9, 0.8]
]

# Initialize and train the text regressor model
regressor = TextRegressor()
regressor.train(texts, pad_targets_text)

# Save the trained text model for future use
joblib.dump(regressor, "text_regressor_model.joblib")

print("Prediction:", regressor.predict("I am excited"))

# AUDIO TRAINING

# Pull in the data using the function defined above
audio_paths, pad_targets_audio = load_ravdess_data("ravdess_audio", [".wav"])

# Split into train/test sets
train_audios, test_audios, train_targets_audio, test_targets_audio = train_test_split(
    audio_paths, pad_targets_audio, test_size=0.2, random_state=42)

# Initialize and train the audio regressor model
audio_regressor = AudioRegressor()
audio_regressor.train(train_audios, train_targets_audio)

# Save the trained audio model for future use
joblib.dump(audio_regressor, "audio_regressor_model.joblib")

print("Audio Prediction:", audio_regressor.predict(test_audios[0]))

# VIDEO TRAINING

# Pull in the video data using the defined function as done above for audio
video_paths, pad_targets_video = load_ravdess_data("ravdess_video", [".mp4"])

# Split into train/test sets, in this 80-20 split for train-test
train_videos, test_videos, train_targets_video, test_targets_video = train_test_split(
    video_paths, pad_targets_video, test_size=0.2, random_state=42)

# Initialize and train the video regressor model
video_regressor = VideoRegressor()
video_regressor.train(train_videos, train_targets_video)

# Save the trained video model for future use
joblib.dump(video_regressor, "video_regressor_model.joblib")

print("Video Prediction:", video_regressor.predict(test_videos[0]))