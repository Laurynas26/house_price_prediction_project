"""
Load the production house price prediction model from MLflow.

This script:
- Connects to the local MLflow tracking store
- Loads the latest model logged as `production_model`
- Returns a ready-to-use model object

Intended usage:
- Batch prediction scripts
- API inference
- Interactive testing
"""

# ------------------------------------------------------------------
# Paths & imports
# ------------------------------------------------------------------
import sys
from pathlib import Path
import mlflow

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# ------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------
TRACKING_URI = "logs/mlruns"
EXPERIMENT_NAME = "house_price_prediction"
MODEL_NAME = "production_model"


# ------------------------------------------------------------------
# Load model
# ------------------------------------------------------------------
def load_production_model():
    """
    Load the latest production model from MLflow.

    Returns
    -------
    model : object
        Trained sklearn or XGBoost model
    run_id : str
        MLflow run ID from which the model was loaded
    """
    mlflow.set_tracking_uri(TRACKING_URI)

    experiment = mlflow.get_experiment_by_name(EXPERIMENT_NAME)
    if experiment is None:
        raise RuntimeError(f"MLflow experiment '{EXPERIMENT_NAME}' not found")

    runs_df = mlflow.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string=f"tags.mlflow.runName = '{MODEL_NAME}'",
    )

    if runs_df.empty:
        raise RuntimeError(
            "No production model found in MLflow. "
            "Have you run select_and_train_best_model.py?"
        )

    # Use the most recent production run
    best_run = runs_df.sort_values("start_time", ascending=False).iloc[0]
    run_id = best_run["run_id"]

    model_uri = f"runs:/{run_id}/model"

    try:
        model = mlflow.sklearn.load_model(model_uri)
    except Exception:
        # Fallback for XGBoost models trained with xgb.train
        model = mlflow.xgboost.load_model(model_uri)

    return model, run_id


# ------------------------------------------------------------------
# CLI usage
# ------------------------------------------------------------------
if __name__ == "__main__":
    model, run_id = load_production_model()
    print(f"Loaded production model from run {run_id}")
    print(f"Model type: {type(model)}")
