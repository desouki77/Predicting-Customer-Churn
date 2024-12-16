import joblib
import os

# Load the trained model
model_path = os.path.join('app', 'models', 'rf_model.pkl')
model = joblib.load(model_path)

def predict_churn(features):
    # Process features and make predictions
    return model.predict([features])
