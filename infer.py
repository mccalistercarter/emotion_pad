from models.text_regressor import TextRegressor
from features.text_features import prep_features
import joblib

def main():
    # Load the pre-trained text regressor model
    model = joblib.load("text_regressor_model.joblib")

    # Example text input for inference
    example_text = "I am thrilled to be part of this amazing journey!"
    # Pass the raw text string, not precomputed features, works because .predict calls prep_features internally

    # Predict PAD values using the loaded model
    pad_prediction = model.predict(example_text)

    print("Text: ", example_text)
    print("Predicted PAD values:", pad_prediction)

# Added from chatbot request
if __name__ == "__main__":
    try:
        main()
    except FileNotFoundError:
        print("Could not find 'text_regressor_model.joblib'. Run training or place the file in the project root.")
    except Exception as e:
        print("Error during inference:", e)
