import os
import joblib
from utils.helpers import load_isear, load_ravdess_data
from sklearn.model_selection import train_test_split
from models.text_regressor import TextRegressor
from models.audio_regressor import AudioRegressor
from models.video_regressor import VideoRegressor

# Define where to save the models
SAVE_DIR = "saved_models"
os.makedirs(SAVE_DIR, exist_ok=True)

def main():
    # TEXT TRAINING

    # Load in a sample of 500 for initial training for sake of time
    df = load_isear(sample_size=500)

    # Now have just three columns (emotion, sentence, and PAD)
    texts = df["sentence"].tolist()
    pad_targets_text = df["PAD"].tolist()

    # Create a train test split for text entries
    train_texts, test_texts, train_targets_text, test_targets_text = train_test_split(
        texts, pad_targets_text, test_size = 0.2, random_state = 42)

    # Initialize and train the text regressor model
    text_regressor = TextRegressor()
    text_regressor.train(train_texts, train_targets_text)

    # Save the trained text model for future use
    joblib.dump(text_regressor, f"{SAVE_DIR}/text_regressor_model.joblib")

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
    joblib.dump(audio_regressor, f"{SAVE_DIR}/audio_regressor_model.joblib")

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
    joblib.dump(video_regressor, f"{SAVE_DIR}/video_regressor_model.joblib")

    print("Training complete. Models saved for text, audio, and video modalities.")

if __name__ == "__main__":
    main()