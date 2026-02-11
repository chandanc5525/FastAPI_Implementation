import pickle
import pandas as pd

def load_pipeline(path):
    return pickle.load(open(path, "rb"))

def predict(full_pipeline, input_df):
    
    X_transformed = full_pipeline["pipeline"].named_steps["preprocessor"].transform(input_df)
    preds = full_pipeline["model"].predict(X_transformed)
    
    return preds