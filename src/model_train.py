import pandas as pd
import pickle
import logging
from src.config import load_params
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from flaml import AutoML


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def run_training(config):
   
    df = pd.read_csv(config["data"]["processed_data_path"])
    target = config["train"]["target_column"]

    # Numeric & categorical columns
    numeric_cols = [c for c in df.select_dtypes(exclude = 'object').columns if c != target]
    categorical_cols = [c for c in df.select_dtypes(include="object").columns if c != target]

    X = df.drop(columns=[target])
    y = df[target]

    # ColumnTransformer + SMOTE ---> preprocessor pipeline
    preprocessor = ColumnTransformer([
        ("num", StandardScaler(), numeric_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols)
    ])

    pipeline = ImbPipeline([
        ("preprocessor", preprocessor),
        ("smote", SMOTE(random_state=config["train"]["random_state"]))
    ])

    # Fit and resample
    X_res, y_res = pipeline.fit_resample(X, y)
    logger.info(f"After SMOTE, dataset shape: {X_res.shape}")

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_res, y_res, test_size=config["train"]["test_size"],
        random_state=config["train"]["random_state"]
    )

    
    # FLAML AutoML
    automl = AutoML()
    automl.fit(X_train, y_train, task="classification",
               max_iter=config["train"]["max_iter"], n_concurrent_trials=4)
    logger.info(f"Test accuracy: {automl.score(X_test, y_test)}")

    # Save full pipeline
    full_pipeline = {"pipeline": pipeline, "model": automl}
    pickle.dump(full_pipeline, open(config["model"]["pickle_path"], "wb"))
    logger.info(f"Pipeline + model saved at {config['model']['pickle_path']}")

    return full_pipeline
