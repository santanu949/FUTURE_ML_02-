from fastapi import FastAPI
import joblib
import pandas as pd
from pydantic import BaseModel
import uvicorn
import os

app = FastAPI(title="Telecom Churn Prediction API")

# Load model
MODEL_PATH = "models/churn_model_v2.pkl"
if os.path.exists(MODEL_PATH):
    model = joblib.load(MODEL_PATH)
else:
    model = None

class CustomerData(BaseModel):
    gender: str
    age: int
    state: str
    city: str
    estimated_salary: float
    plan_type: str
    tenure_months: int
    total_complaints: int
    avg_calls_3m: float
    avg_data_3m: float
    usage_drop_ratio: float
    recharge_frequency: float
    last_month_calls: int
    last_month_data: int

@app.get("/")
def read_root():
    return {"message": "Telecom Churn API is running"}

@app.post("/predict")
def predict(data: CustomerData):
    if model is None:
        return {"error": "Model not loaded"}
    
    # Convert input to DataFrame
    input_df = pd.DataFrame([data.dict()])
    
    # Predict
    prob = model.predict_proba(input_df)[0, 1]
    churn = int(model.predict(input_df)[0])
    
    # Suggest strategy
    strategy = "None"
    if prob > 0.8: strategy = "High Discount"
    elif prob > 0.5: strategy = "Standard Offer"
    elif prob > 0.3: strategy = "Support Call"
    
    return {
        "churn_probability": round(float(prob), 4),
        "prediction": "Churn" if churn == 1 else "Loyal",
        "recommended_action": strategy
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
