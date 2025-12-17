import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from features.video_features import get_video_features

class VideoRegressor:
    def __init__(self):
        # initialize a linear regression model and a scaler for feature normalization
        self.model = LinearRegression()
        self.scaler = StandardScaler()

    def train(self, video_paths, pad_values):
        # train the model with features X and target y
        # convert the video files to feature models
        X = np.array([get_video_features(path) for path in video_paths])

        # scale features for better regression performance
        X = self.scaler.fit_transform(X)

        # targeted PAD values for the training sets
        y = np.array(pad_values)

        # learn weights with linear regression
        self.model.fit(X, y)

    def predict(self, video_path):
        # Predict using the trained model

        # First extract features from the video file
        features = get_video_features(video_path).reshape(1, -1)

        # scale features using the previously fitted scaler
        features = self.scaler.transform(features)

        return self.model.predict(features)