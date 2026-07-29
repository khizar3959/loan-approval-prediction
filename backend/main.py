import os
import json
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional
import numpy as np

app = FastAPI(
    title="Loan Approval Prediction API",
    description="A REST API serving a trained KNN model in pure Python/Numpy.",
    version="1.0.0"
)

# Enable CORS for Next.js frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify the exact domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Paths to saved model components
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(CURRENT_DIR, 'model')

# Global variables for loaded artifacts
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

@app.on_event("startup")
def load_artifacts():
    global model_data, feature_cols, cat_features, num_features, medians, modes, category_mappings, means, stds, X_train, y_train
    try:
        model_path = os.path.join(MODEL_DIR, 'model.json')
        if not os.path.exists(model_path):
            print("WARNING: model.json not found. Please run the training script first!")
            return

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

        print("All machine learning artifacts loaded successfully in pure Numpy!")
    except Exception as e:
        print(f"Error loading model artifacts: {str(e)}")

# Define API input schema
class LoanApplicantDetails(BaseModel):
    Gender: Optional[str] = Field(default=None, examples=["Male", "Female"])
    Married: Optional[str] = Field(default=None, examples=["Yes", "No"])
    Dependents: Optional[str] = Field(default=None, examples=["0", "1", "2", "3+"])
    Education: Optional[str] = Field(default=None, examples=["Graduate", "Not Graduate"])
    Self_Employed: Optional[str] = Field(default=None, examples=["Yes", "No"])
    ApplicantIncome: Optional[float] = Field(default=None, examples=[5000.0])
    CoapplicantIncome: Optional[float] = Field(default=None, examples=[0.0])
    LoanAmount: Optional[float] = Field(default=None, examples=[150.0])
    Loan_Amount_Term: Optional[float] = Field(default=None, examples=[360.0])
    Credit_History: Optional[float] = Field(default=None, examples=[1.0])
    Property_Area: Optional[str] = Field(default=None, examples=["Urban", "Semiurban", "Rural"])

@app.get("/")
def read_root():
    return {"status": "healthy", "message": "Loan Prediction API is up and running."}

@app.post("/api/predict")
def predict_loan(applicant: LoanApplicantDetails):
    global model_data, feature_cols, cat_features, num_features, medians, modes, category_mappings, means, stds, X_train, y_train
    
    # Check if model is loaded
    if model_data is None:
        load_artifacts()
        if model_data is None:
            raise HTTPException(status_code=503, detail="Model is not trained or loaded. Please run train_model.py first.")

    try:
        # Convert input Pydantic model to dict
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
            
            # Safe encoding check: If input is unexpected, map it to the mode class
            if val not in mapping:
                val = modes[col]
                
            encoded_data[col] = mapping[val]

        # Copy numerical values
        for col in num_features:
            encoded_data[col] = input_data[col]

        # 3. Align features in the exact column order expected by model
        x_raw = []
        for col in feature_cols:
            x_raw.append(encoded_data[col])
        x_raw = np.array(x_raw, dtype=float)

        # 4. Standard scale features
        x_scaled = (x_raw - means) / stds

        # 5. Run KNN classification (K = 13 nearest neighbors)
        # Compute Euclidean distance from x_scaled to all points in X_train
        distances = np.sqrt(np.sum((X_train - x_scaled) ** 2, axis=1))
        
        # Get indices of the K=13 smallest distances
        k = 13
        nearest_indices = np.argsort(distances)[:k]
        
        # Get labels of these 13 nearest neighbors
        nearest_labels = y_train[nearest_indices]
        
        # Calculate approval probability (average class of neighbors)
        approval_prob = float(np.mean(nearest_labels))
        
        # Final decision threshold at 0.5
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
