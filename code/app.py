from flask import Flask, request, jsonify, render_template, redirect, url_for
from flask_cors import CORS
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import os

app = Flask(__name__, static_folder='static', template_folder='templates')
CORS(app)

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
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the Linear Regression model
model = LinearRegression()

# Train the model
model.fit(X_train, y_train)

@app.route('/')
def index():
    return render_template('index1.html')

@app.route('/result')
def result():
    return render_template('result.html')

@app.route('/api/predict', methods=['POST'])
def predict_stress():
    data = request.json
    age = data['age']
    gender = data['gender']
    body_temp = data['bodyTemp']
    oxygen_level = data['oxygenLevel']
    heart_rate = data['heartRate']

    # Create a DataFrame for the new data
    new_data = pd.DataFrame({
        'Age': [age], 
        'Body Temperature (°F)': [body_temp], 
        'Oxygen Level (%)': [oxygen_level], 
        'Heart Rate (bpm)': [heart_rate], 
        'Gender_Male': [1 if gender.lower() == 'male' else 0]
    })
    
    # Ensure the order of columns matches the training data
    new_data = new_data.reindex(columns=column_names, fill_value=0)
    
    # Make prediction
    predicted_stress = model.predict(new_data)
    return jsonify({'stress_level': predicted_stress[0]})

if __name__ == '__main__':
    app.run(debug=True)
