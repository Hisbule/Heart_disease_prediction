from fastapi import FastAPI
from Fast_api.schemas import HeartDiseaseInput
import joblib
import numpy as np

# Load the trained model
model = joblib.load(r"D:\Heart_disease_prediction\model\heart_model.joblib")

app = FastAPI()

@app.get("/health")
def health_check():
    return {"status": "API is running"}

@app.get("/info")
def get_info():
    return {
        "model_type": "RandomForestClassifier",
        "features": [
            "age", "sex", "cp", "trestbps", "chol", "fbs",
            "restecg", "thalach", "exang", "oldpeak",
            "slope", "ca", "thal"
        ]
    }

@app.post("/predict")
def predict(input_data: HeartDiseaseInput):
    # Prepare features as 2D array
    features = np.array([[ 
        input_data.age, input_data.sex, input_data.cp,
        input_data.trestbps, input_data.chol, input_data.fbs,
        input_data.restecg, input_data.thalach, input_data.exang,
        input_data.oldpeak, input_data.slope, input_data.ca, input_data.thal
    ]])

    # Predict (0 = no disease, 1 = disease)
    prediction = model.predict(features)[0]

    return {"heart_disease": bool(prediction)}
