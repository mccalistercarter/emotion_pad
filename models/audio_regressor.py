# import libraries
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import numpy as np
import joblib
from features.audio_features import get_audio_features

class AudioRegressor:
    def __init__(self):
        # initialize a linear regression model and a scaler for feature normalization
        self.model = LinearRegression()
        self.scaler = StandardScaler()

    def train(self, audio_paths, pad_values):
        # train the model with features X and target y
        # convert the audio files to feature models
        X = np.array([get_audio_features(path) for path in audio_paths])
        # scale features for better regression performance
        X = self.scaler.fit_transform(X)
        # targeted PAD values for the training sets
        y = np.array(pad_values)
        # learn weights with linear regression
        self.model.fit(X, y)

    def predict(self, audio_path):
        # predict using the trained model
        # extract features from the audio file
        X = get_audio_features(audio_path).reshape(1, -1)
        # scale features using the previously fitted scaler
        X = self.scaler.transform(X)
        return self.model.predict(X)