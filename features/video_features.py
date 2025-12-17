import cv2
import numpy as np
from deepface import DeepFace

def get_video_features(video_path):
    # Need to implement video feature extraction
    # Read the video file and break into frames
    cap = cv2.VideoCapture(video_path)
    frame_features = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        try:
            # Run analysis on a frame to extract emotion features
            analysis = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)
            emotions = analysis[0]['emotion']
            # Convert to a fixed order vector
            emotion_vector = np.array([list(emotions.values())])
            # Append the emotion vector results to a frame features list
            frame_features.append(emotion_vector.flatten())
        except:
            # Skip frames where analysis fails
            continue 

    # Closes the video file
    cap.release()

    if len(frame_features) == 0:
        # If no frames were processed, return a zero vector
        return np.zeros((7,))  # Assuming 7 emotion categories
        
    # Take the mean emotion probabilities across all frames giving one feature vector for the video clip
    # Average should reflect the most prominent emotions across the video
    processed_frames = np.mean(frame_features, axis=0)

    return processed_frames