import os
from pathlib import Path
import logging
import time
import random
import mlflow
import mlflow.xgboost
from mlflow.tracking import MlflowClient

logger = logging.getLogger(__name__)

MODEL_BUCKET = os.environ.get("MODEL_BUCKET")
MODEL_FOLDER = os.environ.get("MODEL_FOLDER")


# --------------------------
# Retry helper
# --------------------------
def _retry(func, retries=3, delay=1, backoff=2, jitter=True):
    """
    Retry helper for transient errors.

    Args:
        func: callable to execute
        retries: number of attempts
        delay: initial delay in seconds
        backoff: multiplier for delay each retry
        jitter: add random fraction to delay
    Returns:
        func() result if successful
    Raises:
        last exception if all retries fail
    """
    attempt = 0
    while attempt < retries:
        try:
            return func()
        except Exception as e:
            attempt += 1
            if attempt >= retries:
                logger.error("All %d attempts failed: %s", retries, e)
                raise
            sleep_time = delay * (backoff ** (attempt - 1))
            if jitter:
                sleep_time += random.uniform(0, 0.5)
            logger.warning(
                "Attempt %d/%d failed with %s. Retrying in %.1f sec...",
                attempt,
                retries,
                type(e).__name__,
                sleep_time,
            )
            time.sleep(sleep_time)


# --------------------------
# Internal helpers
# --------------------------
def _resolve_mlruns_path(max_depth: int = 10) -> Path:
    """Find 'logs/mlruns' folder for local MLflow."""
    project_root = Path(__file__).resolve().parent
    for _ in range(max_depth):
        candidate = project_root / "logs/mlruns"
        if candidate.exists():
            return candidate.resolve()
        project_root = project_root.parent
    raise RuntimeError("Could not find 'logs/mlruns' folder.")


def _load_from_s3() -> any:
    """Load model from S3 (used in AWS Lambda)."""
    model_uri = f"s3://{MODEL_BUCKET}/{MODEL_FOLDER}"
    logger.info("Loading model from S3: %s", model_uri)
    model = mlflow.xgboost.load_model(model_uri)
    logger.info("✅ Model loaded successfully from S3")
    return model


def _load_from_local(experiment_name: str, model_name: str) -> any:
    """Load model from local MLflow."""
    logger.info("Using local MLflow tracking")

    mlruns_path = _resolve_mlruns_path()
    tracking_uri = f"file:///{mlruns_path.as_posix()}"
    mlflow.set_tracking_uri(tracking_uri)

    client = MlflowClient()
    experiment = client.get_experiment_by_name(experiment_name)
    if experiment is None:
        raise RuntimeError(
            f"Experiment '{experiment_name}' not found in MLflow"
        )

    experiment_id = experiment.experiment_id

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

    logger.info("Loading model from MLflow run: %s", model_uri)
    model = mlflow.xgboost.load_model(model_uri)
    logger.info("✅ Model loaded successfully from MLflow")
    return model


# --------------------------
# Public functions
# --------------------------
def load_latest_mlflow_model(
    model_name: str, experiment_name: str = "house_price_prediction"
) -> any:
    """Load latest model from S3 (Lambda) or local MLflow with retry."""
    if os.environ.get("AWS_LAMBDA_FUNCTION_NAME"):
        return _retry(_load_from_s3)
    else:
        return _retry(
            lambda: _load_from_local(
                experiment_name=experiment_name, model_name=model_name
            )
        )


def load_production_model(model_cfg: dict, experiment_name: str = None) -> any:
    """
    Load production ML model using config dictionary.

    Args:
        model_cfg: dictionary containing 'production_model_name'
        experiment_name: optional override
    """
    production_model_name = model_cfg.get("production_model_name")
    if not production_model_name:
        raise RuntimeError(
            "production_model_name missing in model_config.yaml"
        )

    experiment = experiment_name or model_cfg.get(
        "experiment_name", "house_price_prediction"
    )
    logger.info(
        "Loading production model '%s' from MLflow experiment '%s'",
        production_model_name,
        experiment,
    )

    model = load_latest_mlflow_model(
        production_model_name, experiment_name=experiment
    )
    logger.info("✅ Production model loaded successfully")
    return model
