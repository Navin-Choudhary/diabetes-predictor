import sys
import joblib
import pandas as pd

model = joblib.load("diabetes_model.pkl")

inputs = [float(x) for x in sys.argv[1:]]

columns = ['Pregnancies','Glucose','BloodPressure','SkinThickness','Insulin',
           'BMI','DiabetesPedigreeFunction','Age']

data = pd.DataFrame([inputs],columns = columns)

prediction = model.predict(data)

print("Yes" if prediction[0] == 1 else "No", flush=True)