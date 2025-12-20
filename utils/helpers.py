import os
import numpy as np
import pandas as pd
from config.emotion_mapping import emotion_to_pad
from config.ravdess_mapping import ravdess_emotion_map

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