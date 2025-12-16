# import regression models and libraries
from sklearn.linear_model import LinearRegression
import numpy as np
from features.text_features import prep_features

# define a simple text regressor model class
class TextRegressor:
    def __init__(self):
        # initialize a linear regression model
        self.model = LinearRegression()

    def train(self, texts, pad_values):
        # train the model with features X and target y
        # convert the text to feature models
        X = np.array([prep_features(t) for t in texts])
        # targeted PAD values fro the test sets
        y = np.array(pad_values)
        # learn weights with linear regression
        self.model.fit(X, y)

    def predict(self, text):
        # predict using the trained model
        # reshape because sklearn expects 2D input
        X = prep_features(text).reshape(1, -1)
        return self.model.predict(X)