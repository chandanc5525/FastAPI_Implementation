import pandas as pd
import logging
from src.config import load_params
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def run_data_ingestion(config):
  
    df = pd.read_csv(config["data"]["raw_data_path"])
    
    # Ensure the processed folder exists
    processed_dir = os.path.dirname(config["data"]["processed_data_path"])
    os.makedirs(processed_dir, exist_ok=True)
    
    df.to_csv(config["data"]["processed_data_path"], index=False)
    return df
