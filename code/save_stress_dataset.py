import pandas as pd
import random
import os

# Generate random data
names = ["John Doe", "Jane Smith", "Bob Brown", "Alice Green", "Charlie Black", 
         "Emily White", "Frank Red", "Grace Blue", "Henry Yellow", "Irene Pink"]
genders = ["Male", "Female"]

data = []
for _ in range(1000):
    name = random.choice(names)
    age = random.randint(17, 60)
    gender = random.choice(genders)
    body_temp = round(random.uniform(97.0, 99.5), 1)
    oxygen_level = random.randint(94, 100)
    heart_rate = random.randint(60, 100)
    stress_level = random.randint(1, 10)
    data.append([name, age, gender, body_temp, oxygen_level, heart_rate, stress_level])

# Create DataFrame
df = pd.DataFrame(data, columns=["Name", "Age", "Gender", "Body Temperature (°F)", "Oxygen Level (%)", "Heart Rate (bpm)", "Stress Level (1-10)"])

# Display the DataFrame
print(df)

# Path to save the file
desktop_path = os.path.join(os.path.expanduser("~"), "Desktop", "stress_dataset.csv")

# Ensure the directory exists
os.makedirs(os.path.dirname(desktop_path), exist_ok=True)

# Save to CSV file
df.to_csv(desktop_path, index=False)

print(f"Dataset saved to {desktop_path}")
