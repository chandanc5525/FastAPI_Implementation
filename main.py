from src.config import load_params
from src.data_ingestion import run_data_ingestion
from src.model_train import run_training
from src.model_prediction import load_pipeline,predict 
import pandas as pd 

def main():
    
    config = load_params()
    df = run_data_ingestion(config)
    full_pipeline = run_training(config)
    
    # Sample to test prediction
    pipeline = load_pipeline("models/cardio_pipeline.pkl")
    sample_input = pd.DataFrame([{
        "age": 50,
        "bmi": 28,
        "systolic_bp": 130,
        "diastolic_bp": 85,
        "cholesterol_mg_dl": 200,
        "resting_heart_rate": 75,
        "smoking_status": "Yes",
        "daily_steps": 5000,
        "stress_level": 3,
        "physical_activity_hours_per_week": 2,
        "sleep_hours": 6,
        "family_history_heart_disease": "Yes",
        "diet_quality_score": 7,
        "alcohol_units_per_week": 3
    }])
    
    preds = predict(pipeline, sample_input)
    print("Predicted risk_category:", preds[0])

if __name__ == "__main__":
    main()
