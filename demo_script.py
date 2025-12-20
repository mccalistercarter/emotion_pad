import time
import joblib
from models.fusion_model import fuse_pad

def main():
    print("Multimodal Emotion Recognition Demo")

    # Load in the trained models

    try:
        text_model = joblib.load("saved_models/text_regressor_model.joblib")
        audio_model = joblib.load("saved_models/audio_regressor_model.joblib")
        video_model = joblib.load("saved_models/video_regressor_model.joblib")
    except Exception as e:
        print("Error loading models: ", e)
        return
    
    # Example inputs

    example_text = "data/demo_data/demo.txt"
    example_audio_path = "data/demo_data/demo_ex1_happy.wav"
    example_video_path = "data/demo_data/demo_ex1_happy.mp4"

    print("Text Input: ", example_text)
    print("Audio Input: ", example_audio_path)
    print("Video Input: ", example_video_path)

    # Run the predictions using the models
    total_start = time.time()

    try:
        t0 = time.time()
        text_pad = text_model.predict([example_text])[0]
        text_time = (time.time() - t0) * 1000
        t0 = time.time()
        audio_pad = audio_model.predict(example_audio_path)[0]
        audio_time = (time.time() - t0) * 1000
        t0 = time.time()
        video_pad = video_model.predict(example_video_path)[0]
        video_time = (time.time() - t0) * 1000
    except Exception as e:
        print("Error during prediction: ", e)
        return
    
    # Print the individual predictions
    print(f"Text Pad: {text_pad}    ({text_time:.2f} ms)")
    print(f"Audio Pad: {audio_pad}   ({audio_time:.2f} ms)")
    print(f"Video Pad: {video_pad}  ({video_time:.2f} ms)")

    # Fusion using the simple function defined
    fused_pad = fuse_pad([text_pad, audio_pad, video_pad])

    total_time = (time.time() - total_start) * 1000

    print("Fused PAD (Average of all Modalities): ", fused_pad)
    print(f"Total Runtime: {total_time:.2f} ms")
    print("Demo complete.")

if __name__ == "__main__":
    main()