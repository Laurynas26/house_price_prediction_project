"""
Select and retrain the best model from MLflow runs.

This script:
- Loads all MLflow runs from the experiment
- Filters to Optuna + feature-engineered models
- Selects the best model based on test RMSE
- Re-trains the model on full train + validation data
- Logs the final model as the production model in MLflow

Usage:
    python scripts/select_best_model.py
"""

# ------------------------------------------------------------------
# Paths & imports
# ------------------------------------------------------------------
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import yaml
import mlflow
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor

from src.data_loading.data_loading.data_loader import load_data_from_json
from src.data_loading.preprocessing.preprocessing import preprocess_df
from src.data_loading.preprocessing.imputation import impute_missing_values
from src.features.data_prep_for_modelling.data_preparation import prepare_data
from src.model.evaluate import ModelEvaluator

from src.model.mlflow_logger import MLFlowLogger

logger = MLFlowLogger(
    experiment_name="house_price_prediction",
    tracking_uri=str(ROOT / "logs" / "mlruns"),
)


# ------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------
EXPERIMENT_NAME = "house_price_prediction"
RANKING_METRIC = "metrics.test_rmse"

PREPROCESSING_CONFIG_PATH = ROOT / "config" / "preprocessing_config.yaml"
MODEL_CONFIG_PATH = ROOT / "config" / "model_config.yaml"
DATA_PATH = ROOT / "data" / "parsed_json" / "*.json"

# ------------------------------------------------------------------
# Load configs
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

df_clean = df_clean[df_clean["price_num"].notna()]
df_clean = df_clean.drop(columns=["living_area"], errors="ignore")

# ------------------------------------------------------------------
# Load MLflow runs
# ------------------------------------------------------------------
experiment = mlflow.get_experiment_by_name(EXPERIMENT_NAME)
if experiment is None:
    raise RuntimeError(f"MLflow experiment '{EXPERIMENT_NAME}' not found")

runs_df = mlflow.search_runs(experiment_ids=[experiment.experiment_id])

# Keep only Optuna + feature-engineered runs
candidates = runs_df[
    runs_df["tags.mlflow.runName"].str.contains("Optuna", case=False, na=False)
]

if candidates.empty:
    raise RuntimeError("No Optuna runs found in MLflow")

best_run = candidates.sort_values(RANKING_METRIC).iloc[0]

print("Selected best run:")
print("Run name:", best_run["tags.mlflow.runName"])
print("Test MAE:", best_run[RANKING_METRIC])

# ------------------------------------------------------------------
# Extract model metadata
# ------------------------------------------------------------------
run_id = best_run["run_id"]
run_name = best_run["tags.mlflow.runName"]

use_extended_features = "feature_eng" in run_name.lower()
use_log = "log" in run_name.lower()

# Identify model family
if "xgb" in run_name.lower():
    model_family = "xgboost"
elif "rf" in run_name.lower():
    model_family = "random_forest"
else:
    raise ValueError(f"Unknown model family in run name: {run_name}")

# Extract hyperparameters
params = {
    k.replace("params.", ""): v
    for k, v in best_run.items()
    if k.startswith("params.")
}

# Convert numeric params
for k, v in params.items():
    try:
        params[k] = float(v)
        if params[k].is_integer():
            params[k] = int(params[k])
    except Exception:
        pass

# ------------------------------------------------------------------
# Prepare data for final training
# ------------------------------------------------------------------
X_train, X_test, y_train, y_test, X_val, y_val, scaler, _ = prepare_data(
    df=df_clean,
    config_path=MODEL_CONFIG_PATH,
    model_name=(
        "xgboost_early_stopping"
        if model_family == "xgboost"
        else "random_forest_optuna"
    ),
    use_extended_features=use_extended_features,
    cv=False,
)

# Combine train + validation
X_train_full = (
    pd.concat([X_train, X_val], axis=0) if X_val is not None else X_train
)
y_train_full = (
    pd.concat([y_train, y_val], axis=0) if y_val is not None else y_train
)

# ------------------------------------------------------------------
# Train final model
# ------------------------------------------------------------------
evaluator = ModelEvaluator(
    metrics=None,
    target_transform=np.log1p if use_log else None,
    inverse_transform=np.expm1 if use_log else None,
)

if model_family == "random_forest":
    model = RandomForestRegressor(**params)
    trained_model, *_, results = evaluator.evaluate(
        model=model,
        X_train=X_train_full,
        y_train=y_train_full,
        X_test=X_test,
        y_test=y_test,
        use_xgb_train=False,
    )

else:
    trained_model, *_, results = evaluator.evaluate(
        model=None,
        X_train=X_train_full,
        y_train=y_train_full,
        X_test=X_test,
        y_test=y_test,
        X_val=X_val,
        y_val=y_val,
        model_params=params,
        fit_params={"num_boost_round": 1000, "early_stopping_rounds": 50},
        use_xgb_train=True,
    )

# ------------------------------------------------------------------
# Log final production model
# ------------------------------------------------------------------
logger.log_model(
    trained_model,
    model_name="production_model",
    results=results,
    use_xgb_train=(model_family == "xgboost"),
    params=params,
)


print("Production model trained and logged successfully.")
