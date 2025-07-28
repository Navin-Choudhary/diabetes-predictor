
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import joblib

# Step 1: Load dataset
# Dataset link: https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv
# Column names
columns = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness',
           'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age', 'Outcome']

# Load data directly from URL
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
data = pd.read_csv(url, names=columns)

# Step 2: Prepare data
X = data.drop('Outcome', axis=1)
y = data['Outcome']

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Step 3: Train Logistic Regression Model
model = LogisticRegression(max_iter=200)
model.fit(X_train, y_train)

# Step 4: Test model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy * 100:.2f}%")

# Step 5: Save the model
joblib.dump(model, "diabetes_model.pkl")
print("Model saved as diabetes_model.pkl")

# Step 6: Predict new data (user input)
print("\nEnter patient details:")
pregnancies = int(input("Number of Pregnancies: "))
glucose = int(input("Glucose Level: "))
blood_pressure = int(input("Blood Pressure: "))
skin_thickness = int(input("Skin Thickness: "))
insulin = int(input("Insulin Level: "))
bmi = float(input("BMI: "))
dpf = float(input("Diabetes Pedigree Function: "))
age = int(input("Age: "))

# Create a DataFrame for prediction
user_data = pd.DataFrame([[pregnancies, glucose, blood_pressure, skin_thickness,
                           insulin, bmi, dpf, age]],
                          columns=['Pregnancies', 'Glucose', 'BloodPressure',
                                   'SkinThickness', 'Insulin', 'BMI',
                                   'DiabetesPedigreeFunction', 'Age'])

prediction = model.predict(user_data)
print("\nDiabetes Prediction:", "Yes" if prediction[0] == 1 else "No")

