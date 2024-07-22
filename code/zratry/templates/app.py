import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import os
from flask import Flask, request, jsonify, render_template

app = Flask(__name__)

# Path to the dataset on the desktop (replace 'YourUsername' with your actual username)
desktop_path = os.path.join(os.path.expanduser("~"), "Desktop", "stress_dataset.csv")

# Load the dataset
dataset = pd.read_csv(desktop_path)

# Select input features (X) and target variable (y)
X = dataset[['Age', 'Gender', 'Body Temperature (°F)', 'Oxygen Level (%)', 'Heart Rate (bpm)']]
y = dataset['Stress Level (1-10)']

# Convert categorical variable 'Gender' to numerical using one-hot encoding
X = pd.get_dummies(X, columns=['Gender'], drop_first=True)

# Get the columns after get_dummies transformation
column_names = X.columns

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Initialize the Linear Regression model
model = LinearRegression()

# Train the model
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error:", mse)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        age = data['age']
        gender = data['gender']
        body_temp = data['body_temp']
        oxygen_level = data['oxygen_level']
        heart_rate = data['heart_rate']

        # Create a DataFrame for the new data
        new_data = pd.DataFrame({'Age': [age], 'Body Temperature (°F)': [body_temp], 
                                 'Oxygen Level (%)': [oxygen_level], 'Heart Rate (bpm)': [heart_rate], 
                                 'Gender_Male': [1 if gender.lower() == 'male' else 0]})
        
        # Ensure the order of columns matches the training data
        new_data = new_data[column_names]
        
        # Make prediction
        predicted_stress = model.predict(new_data)
        return jsonify({'predicted_stress': predicted_stress[0]})
    except Exception as e:
        print("Error:", e)
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
