import time
import joblib
import numpy as np
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from train import load_ravdess_data, load_isear

def main():
    num_eval_samples = 5
    text_preds = []
    audio_preds = []
    video_preds = []

    # Initailize containers for latency and error measurements
    latencies = {"text": [], "audio": [], "video": []}
    errors = {"text": [], "audio": [], "video": []}
    fused_errors = []

    # TEXT INFERENCE

    # Reloading the data so that the test set from text can be used
    df = load_isear(sample_size=2500)

    texts = df["sentence"].tolist()
    pad_targets_text = df["PAD"].tolist()

    # Create a train test split for text entries
    _, test_texts, _, test_targets_text = train_test_split(
        texts, pad_targets_text, test_size = 0.2, random_state = 42)

    try:
        text_model = joblib.load("saved_models/text_regressor_model.joblib")

        for i in range(num_eval_samples):
            example_text = test_texts[i]
            actual_pad_text = test_targets_text[i]

            start = time.perf_counter()
            pad_prediction_text = text_model.predict(example_text)
            text_preds.append(pad_prediction_text)
            elapsed = (time.perf_counter() - start) * 1000

            latencies["text"].append(elapsed)
            errors["text"].append(mean_absolute_error([actual_pad_text], pad_prediction_text))

            # Print only the first example
            if i == 0:
                print("Text: ", example_text)
                print("Predicted PAD values (Text):", pad_prediction_text)
                print("Actual PAD values (Text):", actual_pad_text)
                print(f"Text Inference Latency: (elapsed:.2f) ms")

    except FileNotFoundError:
        print("Could not find 'text_regressor_model.joblib'. Run training or place the file in the project root.")
    except Exception as e:
        print("Error during text inference:", e)

    # AUDIO INFERENCE

    try:
        audio_model = joblib.load("saved_models/audio_regressor_model.joblib")
        # Pull in the data using the same function as in training
        audio_paths, pad_targets_audio = load_ravdess_data("data/audio_data", [".wav"])

        # Split into train/test sets with same seed as previously used in training
        _, test_audios, _, test_targets_audio = train_test_split(
            audio_paths, pad_targets_audio, test_size=0.2, random_state=42)
        
        for i in range(num_eval_samples):
            example_audio = test_audios[i]
            actual_pad_audio = test_targets_audio[i]

            start = time.perf_counter()
            pad_prediction_audio = audio_model.predict(example_audio)
            audio_preds.append(pad_prediction_audio)
            elapsed = (time.perf_counter() - start) * 1000

            latencies["audio"].append(elapsed)
            errors["audio"].append(mean_absolute_error([actual_pad_audio], pad_prediction_audio))

            # Print only the first example
            if i == 0:
                print("Audio: ", example_audio)
                print("Predicted PAD values (Audio):", pad_prediction_audio)
                print("Actual PAD values (Audio):", actual_pad_audio)
                print(f"Audio Inference Latency: (elapsed:.2f) ms")

    except FileNotFoundError:
        print("Could not find 'audio_regressor_model.joblib'. Run training or place the file in the project root.")
    except Exception as e:
        print("Error during audio inference:", e)

    # VIDEO INFERENCE

    try:
        video_model = joblib.load("saved_models/video_regressor_model.joblib")
        # Pull in the video data using the defined function as done above for audio
        video_paths, pad_targets_video = load_ravdess_data("data/video_data", [".mp4"])

        # Split into train/test sets using same as training for consistency
        _, test_videos, _, test_targets_video = train_test_split(
            video_paths, pad_targets_video, test_size=0.2, random_state=42)

        for i in range(num_eval_samples):
            example_video_path = test_videos[i]
            actual_pad_video = test_targets_video[i]

            start = time.perf_counter()
            pad_prediction_video = video_model.predict(example_video_path)
            video_preds.append(pad_prediction_video)
            elapsed = (time.perf_counter() - start) * 1000

            latencies["video"].append(elapsed)
            errors["video"].append(mean_absolute_error([actual_pad_video], pad_prediction_video))

            # Print only the first example
            if i == 0:
                print("Video: ", example_text)
                print("Predicted PAD values (Video):", pad_prediction_video)
                print("Actual PAD values (Video):", actual_pad_video)
                print(f"Video Inference Latency: (elapsed:.2f) ms")

    except FileNotFoundError:
        print("Could not find 'video_regressor_model.joblib'. Run training or place the file in the project root.")
    except Exception as e:
        print("Error during video inference:", e)

    # Doing Fused MAE calucations, given that all three examples don't have the same expected PAD we will average those and
    # average the predictions and then calculate the error based upon that
    for i in range(num_eval_samples):
        # Add all stored predictions into a single list
        preds = []
        if i < len(text_preds): preds.append(text_preds[i][0])
        if i < len(audio_preds): preds.append(audio_preds[i][0])
        if i < len(video_preds): preds.append(video_preds[i][0])

        if preds:
            # Take the average of the predictions list
            fused_pred = np.mean(preds, axis=0)
            fused_errors.append(mean_absolute_error([test_targets_text[i]], [fused_pred]))

    # Print out a summary of the data collected, latencies and errors
    print("Inference Summary")

    for modality in latencies:
        if latencies[modality]:
            print(
                f"{modality.capitalize()} latency - "
                f"avg: {np.mean(latencies[modality]):.2f} ms, "
                f"max: {np.max(latencies[modality]):.2f} ms"
            )

    
    for modality in errors:
        if errors[modality]:
            print(
                f"{modality.capitalize()} MAE: "
                f"{np.mean(errors[modality]):.4f}"
            )
    
    if fused_errors:
        print(f"Fused MAE: {np.mean(fused_errors):.4f}")

if __name__ == "__main__":
    main()
