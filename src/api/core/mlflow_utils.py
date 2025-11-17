import os
from pathlib import Path
import mlflow
import mlflow.xgboost
from mlflow.tracking import MlflowClient

# ----------------------------
# Environment variables
# ----------------------------
MODEL_BUCKET = os.environ.get("MODEL_BUCKET")
MODEL_FOLDER = os.environ.get("MODEL_FOLDER")


def load_latest_mlflow_model(
    model_name: str,
    experiment_name: str = "house_price_prediction"
):
    """
    Load the latest XGBoost model either from local MLflow
    or S3 if running in AWS Lambda.

    Args:
        model_name: MLflow run name
        experiment_name: MLflow experiment name
    Returns:
        Loaded XGBoost model
    """

    # --- Check if running in AWS Lambda ---
    if os.environ.get("AWS_LAMBDA_FUNCTION_NAME"):
        model_uri = f"s3://{MODEL_BUCKET}/{MODEL_FOLDER}"
        print(f"[Lambda] Loading model from S3: {model_uri}")
        model = mlflow.xgboost.load_model(model_uri)
        print("[Lambda] ✅ Model loaded successfully from S3.")
        return model

    # --- Local/dev: use MLflow local tracking ---
    print("[Local] Using local MLflow tracking.")

    # Resolve logs/mlruns path dynamically
    project_root = Path(__file__).resolve().parent
    for _ in range(10):
        candidate = project_root / "logs/mlruns"
        if candidate.exists():
            mlruns_path = candidate.resolve()
            break
        project_root = project_root.parent
    else:
        raise RuntimeError("❌ Could not find 'logs/mlruns' folder.")

    tracking_uri = f"file:///{mlruns_path.as_posix()}"
    mlflow.set_tracking_uri(tracking_uri)

    client = MlflowClient()
    experiment = client.get_experiment_by_name(experiment_name)
    if experiment is None:
        raise RuntimeError(
            f"❌ Experiment '{experiment_name}' not found in MLflow."
        )

    experiment_id = experiment.experiment_id

    # Get latest finished run for the given model_name
    all_runs = client.search_runs(
        experiment_ids=[experiment_id],
        filter_string=f"tags.mlflow.runName = '{model_name}'",
        order_by=["attributes.start_time DESC"],
    )

    valid_runs = [
        r
        for r in all_runs
        if r.info.status == "FINISHED" and r.info.artifact_uri
    ]
    if not valid_runs:
        raise RuntimeError(
            f"No valid MLflow runs found for runName '{model_name}'"
        )

    latest_run = valid_runs[0]
    run_id = latest_run.info.run_id
    model_uri = f"runs:/{run_id}/model"
    print(f"[Local] Loading model from MLflow run: {model_uri}")
    model = mlflow.xgboost.load_model(model_uri)
    print("[Local] ✅ Model loaded successfully from MLflow.")
    return model
