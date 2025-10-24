from pathlib import Path
import mlflow
import mlflow.xgboost
from mlflow.tracking import MlflowClient


def load_latest_mlflow_model(
    model_name: str, experiment_name: str = "house_price_prediction"
):
    """
    Load the latest MLflow run of a model by name,
    ignoring .trash or incomplete runs.

    Args:
        model_name: str, MLflow run name of the model.
        experiment_name: str, MLflow experiment name.

    Returns:
        Loaded XGBoost model.
    """
    # --- Step 1: Resolve project root dynamically ---
    project_root = Path(__file__).resolve().parent
    for _ in range(10):
        candidate = project_root / "logs/mlruns"
        if candidate.exists():
            mlruns_path = candidate.resolve()
            break
        project_root = project_root.parent
    else:
        raise RuntimeError(
            "❌ Could not find 'logs/mlruns' folder in project hierarchy."
        )

    print(f"[MLflow] Using MLruns folder: {mlruns_path}")

    # --- Step 2: Set tracking URI ---
    tracking_uri = f"file:///{mlruns_path.as_posix()}"
    mlflow.set_tracking_uri(tracking_uri)
    print(f"[MLflow] Tracking URI set to: {tracking_uri}")

    # --- Step 3: Get experiment ID ---
    client = MlflowClient()
    experiment = client.get_experiment_by_name(experiment_name)
    if experiment is None:
        raise RuntimeError(
            f"❌ Experiment '{experiment_name}' not found in MLflow."
        )
    experiment_id = experiment.experiment_id
    print(
        f"[MLflow] Using experiment '{experiment_name}' "
        f"with ID {experiment_id}"
    )

    # --- Step 4: Get all finished runs with the correct runName ---
    all_runs = client.search_runs(
        experiment_ids=[experiment_id],
        filter_string=f"tags.mlflow.runName = '{model_name}'",
        order_by=["attributes.start_time DESC"],
    )

    # Filter out runs with empty artifact_uri or not FINISHED
    valid_runs = [
        r
        for r in all_runs
        if r.info.status == "FINISHED" and r.info.artifact_uri
    ]
    if not valid_runs:
        raise RuntimeError(
            f"❌ No valid MLflow runs found for runName '{model_name}'"
            f"in experiment '{experiment_name}'"
        )

    latest_run = valid_runs[0]
    run_id = latest_run.info.run_id
    artifact_uri = latest_run.info.artifact_uri

    print(f"[MLflow] Found latest valid run ID: {run_id}")
    print(f"[MLflow] Artifact URI: {artifact_uri}")

    # --- Step 5: Load the XGBoost model ---
    model_uri = f"runs:/{run_id}/model"
    print(f"[MLflow] Loading model from: {model_uri}")
    model = mlflow.xgboost.load_model(model_uri)
    print(f"[MLflow] ✅ Loaded model '{model_name}' successfully.")

    return model
