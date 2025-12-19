import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from train import load_ravdess_data
from train import load_isear
from train import emotion_to_pad

def main():

    # TEXT INFERENCE

    # Reloading the data so that the test set from text can be used
    df = load_isear(sample_size=500)

    texts = df["sentence"].tolist()
    pad_targets_text = df["PAD"].tolist()

    # Create a train test split for text entries
    train_texts, test_texts, train_targets_text, test_targets_text = train_test_split(
        texts, pad_targets_text, test_size = 0.2, random_state = 42)

    try:
        text_model = joblib.load("text_regressor_model.joblib")
        example_text = test_texts[0]
        actual_pad_text = test_targets_text[0]
        pad_prediction_text = text_model.predict(example_text)

        print("Text: ", example_text)
        print("Predicted PAD values (Text):", pad_prediction_text)
        print("Actual PAD values (Text):", actual_pad_text)
    except FileNotFoundError:
        print("Could not find 'text_regressor_model.joblib'. Run training or place the file in the project root.")
    except Exception as e:
        print("Error during text inference:", e)

    # AUDIO INFERENCE

    try:
        audio_model = joblib.load("audio_regressor_model.joblib")
        # Pull in the data using the same function as in training
        audio_paths, pad_targets_audio = load_ravdess_data("data/audio_data", [".wav"])

        # Split into train/test sets with same seed as previously used in training
        _, test_audios, _, test_targets_audio = train_test_split(
            audio_paths, pad_targets_audio, test_size=0.2, random_state=42)
        
        # Run the prediction on the first test sample
        example_audio_path = test_audios[0]
        pad_prediction_audio = audio_model.predict(example_audio_path)

        print("Audio File: ", example_audio_path)
        print("Predicted PAD values (Audio):", pad_prediction_audio)
        print("Actual PAD values (Audio):", test_targets_audio[0])
    except FileNotFoundError:
        print("Could not find 'audio_regressor_model.joblib'. Run training or place the file in the project root.")
    except Exception as e:
        print("Error during audio inference:", e)

    # VIDEO INFERENCE

    try:
        video_model = joblib.load("video_regressor_model.joblib")
        
        # Pull in the video data using the defined function as done above for audio
        video_paths, pad_targets_video = load_ravdess_data("data/video_data", [".mp4"])

        # Split into train/test sets using same as training for consistency
        _, test_videos, _, test_targets_video = train_test_split(
            video_paths, pad_targets_video, test_size=0.2, random_state=42)

        # Run the prediction on the first test sample
        example_video_path = test_videos[0]
        pad_prediction_video = video_model.predict(example_video_path)

        print("Video File: ", example_video_path)
        print("Predicted PAD values (Video):", pad_prediction_video)
        print("Actual PAD values (Video):", test_targets_video[0])
    except FileNotFoundError:
        print("Could not find 'video_regressor_model.joblib'. Run training or place the file in the project root.")
    except Exception as e:
        print("Error during video inference:", e)

if __name__ == "__main__":
    main()
