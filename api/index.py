import os
import json
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional
import numpy as np

app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model data
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(CURRENT_DIR, 'model.json')

model_data = None
feature_cols = None
cat_features = None
num_features = None
medians = None
modes = None
category_mappings = None
means = None
stds = None
X_train = None
y_train = None

def load_artifacts():
    global model_data, feature_cols, cat_features, num_features, medians, modes, category_mappings, means, stds, X_train, y_train
    if model_data is not None:
        return
        
    try:
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"model.json not found at {model_path}")

        with open(model_path, 'r') as f:
            model_data = json.load(f)

        feature_cols = model_data["feature_cols"]
        cat_features = model_data["cat_features"]
        num_features = model_data["num_features"]
        medians = model_data["medians"]
        modes = model_data["modes"]
        category_mappings = model_data["category_mappings"]
        means = np.array(model_data["means"])
        stds = np.array(model_data["stds"])
        X_train = np.array(model_data["X_train"])
        y_train = np.array(model_data["y_train"])
    except Exception as e:
        print(f"Error loading model artifacts: {str(e)}")

# Define API input schema
class LoanApplicantDetails(BaseModel):
    Gender: Optional[str] = Field(default=None)
    Married: Optional[str] = Field(default=None)
    Dependents: Optional[str] = Field(default=None)
    Education: Optional[str] = Field(default=None)
    Self_Employed: Optional[str] = Field(default=None)
    ApplicantIncome: Optional[float] = Field(default=None)
    CoapplicantIncome: Optional[float] = Field(default=None)
    LoanAmount: Optional[float] = Field(default=None)
    Loan_Amount_Term: Optional[float] = Field(default=None)
    Credit_History: Optional[float] = Field(default=None)
    Property_Area: Optional[str] = Field(default=None)

@app.get("/api")
def read_root():
    return {"status": "healthy", "message": "Vercel Serverless Loan Prediction API is active."}

@app.post("/api/predict")
def predict_loan(applicant: LoanApplicantDetails):
    global model_data, feature_cols, cat_features, num_features, medians, modes, category_mappings, means, stds, X_train, y_train
    
    load_artifacts()
    if model_data is None:
        raise HTTPException(status_code=503, detail="Model is not loaded.")

    try:
        input_data = applicant.dict()

        # 1. Impute missing values
        for col in num_features:
            if input_data[col] is None:
                input_data[col] = medians[col]

        for col in cat_features:
            if input_data[col] is None:
                input_data[col] = modes[col]

        # 2. Encode categorical features
        encoded_data = {}
        for col in cat_features:
            mapping = category_mappings[col]
            val = input_data[col]
            if val not in mapping:
                val = modes[col]
            encoded_data[col] = mapping[val]

        # Copy numerical values
        for col in num_features:
            encoded_data[col] = input_data[col]

        # 3. Align features
        x_raw = []
        for col in feature_cols:
            x_raw.append(encoded_data[col])
        x_raw = np.array(x_raw, dtype=float)

        # 4. Standard scale features
        x_scaled = (x_raw - means) / stds

        # 5. Run KNN classification (K = 13 nearest neighbors)
        distances = np.sqrt(np.sum((X_train - x_scaled) ** 2, axis=1))
        nearest_indices = np.argsort(distances)[:13]
        nearest_labels = y_train[nearest_indices]
        
        approval_prob = float(np.mean(nearest_labels))
        prediction = "Approved" if approval_prob >= 0.5 else "Rejected"
        confidence = approval_prob if prediction == "Approved" else (1.0 - approval_prob)

        return {
            "prediction": prediction,
            "probability": approval_prob,
            "confidence": confidence,
            "features_received": input_data
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")
