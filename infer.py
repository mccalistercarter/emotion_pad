from models.text_regressor import TextRegressor
from features.text_features import prep_features
from models.audio_regressor import AudioRegressor
from features.audio_features import get_audio_features
import joblib

def main():

    # TEXT INFERENCE

    try:
        text_model = joblib.load("text_regressor_model.joblib")
        example_text = "I am thrilled to be part of this amazing journey!"
        pad_prediction_text = text_model.predict(example_text)

        print("Text: ", example_text)
        print("Predicted PAD values (Text):", pad_prediction_text)
    except FileNotFoundError:
        print("Could not find 'text_regressor_model.joblib'. Run training or place the file in the project root.")
    except Exception as e:
        print("Error during text inference:", e)

    # AUDIO INFERENCE

    try:
        audio_model = joblib.load("audio_regressor_model.joblib")
        example_audio_path = "audio_excited.wav"
        pad_prediction_audio = audio_model.predict(example_audio_path)

        print("Audio File: ", example_audio_path)
        print("Predicted PAD values (Audio):", pad_prediction_audio)
    except FileNotFoundError:
        print("Could not find 'audio_regressor_model.joblib'. Run training or place the file in the project root.")
    except Exception as e:
        print("Error during audio inference:", e)

    # VIDEO INFERENCE

    try:
        video_model = joblib.load("video_regressor_model.joblib")
        example_video_path = "video_excited.mp4"
        pad_prediction_video = video_model.predict(example_video_path)

        print("Video File: ", example_video_path)
        print("Predicted PAD values (Video):", pad_prediction_video)
    except FileNotFoundError:
        print("Could not find 'video_regressor_model.joblib'. Run training or place the file in the project root.")
    except Exception as e:
        print("Error during video inference:", e)

if __name__ == "__main__":
    main()
