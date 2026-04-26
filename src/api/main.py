from fastapi import FastAPI
import joblib
import pandas as pd
from pydantic import BaseModel
import uvicorn
import os
import logging

# Configure Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Telecom Churn AI API", version="3.0.0")

# Load model v3
MODEL_PATH = "models/churn_ensemble_v3.pkl"
if os.path.exists(MODEL_PATH):
    model = joblib.load(MODEL_PATH)
    logger.info("Ensemble Model v3 loaded.")
else:
    model = None
    logger.warning("No production model found.")

class PredictionInput(BaseModel):
    gender: str
    age: int
    state: str
    city: str
    estimated_salary: float
    plan_type: str
    device_type: str
    payment_method: str
    tenure_days: int
    total_complaints: int
    last_month_calls: int
    last_month_data: int
    last_month_sms: int
    avg_calls_6m: float
    avg_data_6m: float
    usage_drop_ratio: float
    recharge_consistency: float

@app.get("/")
def health_check():
    return {"status": "online", "model_loaded": model is not None}

@app.post("/predict")
def predict(data: PredictionInput):
    if model is None:
        return {"error": "Production model not found on server."}
    
    # 1. Convert input to DataFrame
    input_df = pd.DataFrame([data.dict()])
    
    # 2. Pipeline handles engineering and preprocessing automatically
    prob = model.predict_proba(input_df)[0, 1]
    churn = int(model.predict(input_df)[0])
    
    # 3. Decision Logic
    strategy = "Standard"
    if prob > 0.8: strategy = "Immediate Retention: 30% Discount + Concierge Support"
    elif prob > 0.5: strategy = "Proactive Retention: 15% Discount + Service Quality Check"
    elif prob > 0.3: strategy = "Monitoring: Survey Call + Personalised SMS"
    
    return {
        "churn_probability": round(float(prob), 4),
        "prediction": "CHURN_RISK" if churn == 1 else "LOYAL",
        "recommended_action": strategy,
        "clv_impact_estimate": "$1,200" # Static for demo
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
