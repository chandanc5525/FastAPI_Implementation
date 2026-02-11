from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
from src.model_prediction import load_pipeline, predict
from src.config import load_params

config = load_params()
full_pipeline = load_pipeline(config["model"]["pickle_path"])

app = FastAPI(title="Cardiovascular Risk Prediction API")

class PatientData(BaseModel):
    age: int
    bmi: float
    systolic_bp: float
    diastolic_bp: int
    cholesterol_mg_dl: float
    resting_heart_rate: int
    smoking_status: str
    daily_steps: int
    stress_level: int
    physical_activity_hours_per_week: float
    sleep_hours: float
    family_history_heart_disease: str
    diet_quality_score: float
    alcohol_units_per_week: float

@app.post("/predict")
def get_prediction(data: PatientData):
    df = pd.DataFrame([data.dict()])
    preds = predict(full_pipeline, df)
    return {"predicted_risk_category": preds[0]}
