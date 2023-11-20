# 1. Library imports
import uvicorn
from fastapi import FastAPI
from Diabetes import diseases
import numpy as np
import joblib
import pandas as pd
# 2. Create the app object
app = FastAPI()
pickle_in = open("diabetes.pkl","rb")
classifier=joblib.load(pickle_in)

# 3. Index route, opens automatically on http://127.0.0.1:8000
@app.get('/')
def index():
    return {'Welcome To BeeOwn'}


# 3. Expose the prediction functionality, make a prediction from the passed
#    JSON data and return the predicted Diseases with the confidence
@app.post('/predict')
def predict_Diseases(data: diseases):
    data = data.dict()
    Pregnancies = data['Pregnancies']
    Glucose = data['Glucose']
    BloodPressure = data['BloodPressure']
    SkinThickness = data['SkinThickness']
    Insulin = data['Insulin']
    BMI = data['BMI']
    DiabetesPedigreeFunction = data['DiabetesPedigreeFunction']
    Age = data['Age']

    prediction = classifier.predict([[Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age]])
    if (prediction[0] == 1):
        prediction = "The person is diabetic"
    else:
        prediction = "The person is not diabetic"
    return {
        'prediction': prediction
    }


# 5. Run the API with uvicorn
#    Will run on http://127.0.0.1:8000
if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8000)

# uvicorn app:app --reload