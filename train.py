from models.text_regressor import TextRegressor
from features.text_features import prep_features
import numpy as np
import joblib # for saving/loading models

# Training data for linear regression model for text PAD prediction
texts = ["I am happy", "I am sad", "I am angry"]
pad_targets = [
    [0.8, 0.5, 0.6],
    [-0.6, -0.4, -0.5],
    [-0.7, 0.9, 0.8]
]

# Initialize and train the text regressor model
regressor = TextRegressor()
regressor.train(texts, pad_targets)

# Save the trained model for future use
joblib.dump(regressor, "text_regressor_model.joblib")

print("Prediction:", regressor.predict("I am excited"))
