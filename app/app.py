from flask import Flask, request, jsonify, render_template
import pandas as pd
import joblib

app = Flask(__name__)

# Load the pre-trained model and columns used during training
try:
    with open('./scripts/app/models/rf_model.pkl', 'rb') as model_file:
        model = joblib.load(model_file)
    with open('./scripts/app/models/model_columns.pkl', 'rb') as columns_file:
        model_input_columns = joblib.load(columns_file)

    if model and model_input_columns:
        print("Model and columns loaded successfully.")
except FileNotFoundError:
    print("Error: Model or columns file not found.")
    model = None
    model_input_columns = None
except Exception as e:
    print(f"Error loading model or columns: {str(e)}")
    model = None
    model_input_columns = None


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if model is None or model_input_columns is None:
        return jsonify({'error': 'Model or columns not loaded properly'}), 500
    
    try:
        # Read the CSV file with the correct delimiter
        input_data = pd.read_csv(file, delimiter=';')  # Use semicolon as delimiter

        # Ensure columns match model's expected features
        required_columns = ['customerID', 'gender', 'SeniorCitizen', 'Partner', 'Dependents',
                            'tenure', 'PhoneService', 'MultipleLines', 'InternetService', 
                            'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport', 
                            'StreamingTV', 'StreamingMovies', 'Contract', 'PaperlessBilling', 
                            'PaymentMethod', 'MonthlyCharges', 'TotalCharges', 'Churn']
        
        # Check if all required columns are in the uploaded file
        missing_columns = [col for col in required_columns if col not in input_data.columns]
        
        if missing_columns:
            return jsonify({'error': f'Missing columns: {", ".join(missing_columns)}'}), 400
        
        # Preprocess the input data (apply one-hot encoding to categorical columns)
        input_data_encoded = pd.get_dummies(input_data, drop_first=True)

        # Ensure the model input columns are present in the encoded data
        missing_features = [col for col in model_input_columns if col not in input_data_encoded.columns]
        
        if missing_features:
            # Add missing features with default value of 0
            for feature in missing_features:
                input_data_encoded[feature] = 0
        
        # Ensure the encoded data has the correct columns, matching the model's expectations
        input_data_encoded = input_data_encoded[model_input_columns]

        # Predict using the model
        X = input_data_encoded
        predictions = model.predict(X)  # Ensure the model is an object with .predict() method
        
        # Add predictions to the results DataFrame
        results = input_data.copy()
        results['Churn Prediction'] = predictions
        
        # Return results as JSON
        return results.to_json(orient='records')
    
    except Exception as e:
        print(f"Error occurred: {str(e)}")
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True)
