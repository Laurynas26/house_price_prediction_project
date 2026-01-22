"""
Train baseline models for house price prediction.

This script:
- Loads and preprocesses data
- Prepares features according to model configuration
- Trains baseline models (Linear Regression, Random Forest, XGBoost)
- Evaluates models using a unified evaluator
- Logs metrics and models to MLflow

Usage:
    python scripts/train_baselines.py

Optional extensions (future):
    --use-extended-features
    --use-log-transform
    --model [linear|rf|xgb]
"""

# ------------------------------------------------------------------
# Paths and Imports
# ------------------------------------------------------------------
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import yaml
import numpy as np
import pandas as pd

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb

from src.data_loading.data_loading.data_loader import load_data_from_json
from src.data_loading.preprocessing.preprocessing import preprocess_df
from src.data_loading.preprocessing.imputation import impute_missing_values
from src.features.data_prep_for_modelling.data_preparation import prepare_data
from src.model.evaluate import ModelEvaluator
from src.model.mlflow_logger import MLFlowLogger
from src.model.utils import load_model_config_and_search_space

# ------------------------------------------------------------------
# Paths & config
# ------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]

PREPROCESSING_CONFIG_PATH = ROOT / "config" / "preprocessing_config.yaml"
MODEL_CONFIG_PATH = ROOT / "config" / "model_config.yaml"


DATA_PATH = ROOT / "data" / "parsed_json" / "*.json"

# ------------------------------------------------------------------
# Load config
# ------------------------------------------------------------------
with open(PREPROCESSING_CONFIG_PATH) as f:
    PREPROCESSING_CONFIG = yaml.safe_load(f)

# ------------------------------------------------------------------
# Load & preprocess data
# ------------------------------------------------------------------
df_raw = load_data_from_json(str(DATA_PATH))

df_clean = preprocess_df(
    df_raw,
    drop_raw=PREPROCESSING_CONFIG["preprocessing"]["drop_raw"],
    numeric_cols=PREPROCESSING_CONFIG["preprocessing"]["numeric_cols"],
)

df_clean = impute_missing_values(
    df_clean,
    PREPROCESSING_CONFIG["preprocessing"]["imputation"],
)

# Drop rows without target
df_clean = df_clean[df_clean["price_num"].notna()]

# Optional manual drops from notebook decisions
df_clean = df_clean.drop(columns=["living_area"], errors="ignore")

# ------------------------------------------------------------------
# MLflow setup
# ------------------------------------------------------------------
tracking_uri = ROOT / "logs" / "mlruns"
os.makedirs(tracking_uri / ".trash", exist_ok=True)

logger = MLFlowLogger(tracking_uri=str(tracking_uri))
evaluator = ModelEvaluator()

# ------------------------------------------------------------------
# 1. Linear Regression (scaled features)
# ------------------------------------------------------------------
X_train, X_test, y_train, y_test, X_val, y_val, scaler, _ = prepare_data(
    df=df_clean,
    config_path=MODEL_CONFIG_PATH,
    model_name="linear_regression",
    use_extended_features=False,
    cv=False,
)

lr_model = LinearRegression()
trained_lr, *_, lr_results = evaluator.evaluate(
    model=lr_model,
    X_train=X_train,
    y_train=y_train,
    X_test=X_test,
    y_test=y_test,
    X_val=X_val,
    y_val=y_val,
)

logger.log_model(trained_lr, "LinearRegression_Baseline", lr_results)

# ------------------------------------------------------------------
# 2. Random Forest baseline
# ------------------------------------------------------------------
X_train, X_test, y_train, y_test, X_val, y_val, scaler, _ = prepare_data(
    df=df_clean,
    config_path=MODEL_CONFIG_PATH,
    model_name="random_forest",
    use_extended_features=False,
    cv=False,
)

rf_model = RandomForestRegressor(random_state=42)
trained_rf, *_, rf_results = evaluator.evaluate(
    model=rf_model,
    X_train=X_train,
    y_train=y_train,
    X_test=X_test,
    y_test=y_test,
    X_val=X_val,
    y_val=y_val,
)

logger.log_model(trained_rf, "RandomForest_Baseline", rf_results)

# ------------------------------------------------------------------
# 3. XGBoost baseline with early stopping
# ------------------------------------------------------------------
X_train, X_test, y_train, y_test, X_val, y_val, scaler, _ = prepare_data(
    df=df_clean,
    config_path=MODEL_CONFIG_PATH,
    model_name="xgboost_early_stopping",
    use_extended_features=False,
    cv=False,
)

xgb_model_params, xgb_fit_params, _ = load_model_config_and_search_space(
    MODEL_CONFIG_PATH,
    model_name="xgboost_early_stopping",
)

trained_xgb, *_, xgb_results = evaluator.evaluate(
    model=None,
    X_train=X_train,
    y_train=y_train,
    X_test=X_test,
    y_test=y_test,
    X_val=X_val,
    y_val=y_val,
    model_params=xgb_model_params,
    fit_params=xgb_fit_params,
    use_xgb_train=True,
)

logger.log_model(
    trained_xgb,
    "XGBoost_Baseline_EarlyStopping",
    xgb_results,
    use_xgb_train=True,
)

print("Baseline training completed successfully.")
