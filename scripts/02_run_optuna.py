"""
Run Optuna hyperparameter tuning for house price models.

This script:
- Loads and preprocesses data
- Runs Optuna studies for:
    - XGBoost with early stopping
    - Random Forest
- Supports log-transform and extended features
- Uses unified Optuna objective from src.model.objectives_optuna
- Stores results via MLflow (handled inside objective)

Usage:
    python scripts/run_optuna.py
"""

# ------------------------------------------------------------------
# Paths and Imports
# ------------------------------------------------------------------
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import yaml
import optuna
from functools import partial

from src.data_loading.data_loading.data_loader import load_data_from_json
from src.data_loading.preprocessing.preprocessing import preprocess_df
from src.data_loading.preprocessing.imputation import impute_missing_values
from src.model.objectives_optuna import unified_objective


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

df_clean = df_clean[df_clean["price_num"].notna()]
df_clean = df_clean.drop(columns=["living_area"], errors="ignore")

# ------------------------------------------------------------------
# Optuna configuration
# ------------------------------------------------------------------
N_SPLITS = 5
USE_LOG = True
USE_EXTENDED_FEATURES = True

sampler = optuna.samplers.TPESampler(seed=42)
pruner = optuna.pruners.MedianPruner(n_warmup_steps=10)

# ------------------------------------------------------------------
# XGBoost Optuna study
# ------------------------------------------------------------------
study_xgb = optuna.create_study(
    direction="minimize",
    sampler=sampler,
    pruner=pruner,
)

objective_xgb = partial(
    unified_objective,
    model_name="xgboost_early_stopping",
    df=df_clean,
    features_config=MODEL_CONFIG_PATH,
    model_config=MODEL_CONFIG_PATH,
    use_log=USE_LOG,
    n_splits=N_SPLITS,
    use_extended_features=USE_EXTENDED_FEATURES,
)

study_xgb.optimize(objective_xgb, n_trials=30)

print("Best XGBoost params:", study_xgb.best_params)
print("Best XGBoost RMSE:", study_xgb.best_value)

# ------------------------------------------------------------------
# Random Forest Optuna study
# ------------------------------------------------------------------
study_rf = optuna.create_study(direction="minimize")

objective_rf = partial(
    unified_objective,
    model_name="random_forest_optuna",
    df=df_clean,
    features_config=MODEL_CONFIG_PATH,
    model_config=MODEL_CONFIG_PATH,
    use_log=USE_LOG,
    n_splits=N_SPLITS,
    use_extended_features=USE_EXTENDED_FEATURES,
)

study_rf.optimize(objective_rf, n_trials=30)

print("Best RF params:", study_rf.best_params)
print("Best RF RMSE:", study_rf.best_value)

print("Optuna tuning completed.")
