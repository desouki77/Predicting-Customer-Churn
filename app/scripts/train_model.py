import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib

# Load the dataset
data = pd.read_csv('../uploads/TelcoCustomerChurn.csv', delimiter=';', on_bad_lines='skip')

# Strip extra spaces from column names
data.columns = data.columns.str.strip()

# Convert categorical variables to numerical using one-hot encoding
data_encoded = pd.get_dummies(data, drop_first=True)

# Define features and target
X = data_encoded.drop('Churn_Yes', axis=1)  # Drop the 'Churn' column (target)
y = data_encoded['Churn_Yes']  # Target variable (encoded as 'Churn_Yes')

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = RandomForestClassifier(n_estimators=100, random_state=42)

# Fit the model
model.fit(X_train, y_train)

# Check if the model has been fitted
if hasattr(model, 'estimators_'):
    print("Model is successfully trained and fitted.")
    print(f"Number of estimators: {len(model.estimators_)}")  # Print number of trees in the model
else:
    print("Model fitting failed.")

# Create the models directory if it doesn't exist
model_dir = os.path.join('app', 'models')
os.makedirs(model_dir, exist_ok=True)

# Save the trained model
model_path = os.path.join(model_dir, 'rf_model.pkl')
joblib.dump(model, model_path)

# Save the columns used for training the model
columns_path = os.path.join(model_dir, 'model_columns.pkl')
joblib.dump(X_train.columns.tolist(), columns_path)

print(f"Model saved at {model_path}")
print(f"Model columns saved at {columns_path}")
