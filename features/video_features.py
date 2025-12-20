import cv2
import numpy as np
from fer import FER

detector = FER()

def get_video_features(video_path, sample_fps=1, max_seconds=None, resize_dim=(160,160)):
    # Need to implement video feature extraction
    # Read the video file and break into frames
    cap = cv2.VideoCapture(video_path)
    frame_features = []

    # Only sampling 1fps for sake of efficiency, would need imporvement for accuracy
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30 # Default if unknown

    frame_interval = max(1, int(fps / sample_fps))
    frame_idx = 0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    max_frames = int(min(total_frames, (max_seconds * fps) if max_seconds else total_frames))

    while frame_idx < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        
        if frame_idx % frame_interval == 0:
            try: # Replaced deepface with FER to try and make it faster
                frame_resized = cv2.resize(frame, resize_dim)
                results = detector.detect_emotions(frame_resized)
                if results:
                    emotions = results[0]["emotions"] # Dict of 7 emotion probabilities
                    emotion_vector = np.array(list(emotions.values()))
                    frame_features.append(emotion_vector)
            except:
                # Skip frames where analysis fails
                pass 
            
        frame_idx += 1

    # Closes the video file
    cap.release()

    if len(frame_features) == 0:
        # If no frames were processed, return a zero vector
        return np.zeros((7,))  # Assuming 7 emotion categories
        
    # Take the mean emotion probabilities across all frames giving one feature vector for the video clip
    # Average should reflect the most prominent emotions across the video
    processed_frames = np.mean(frame_features, axis=0)

    return processed_frames