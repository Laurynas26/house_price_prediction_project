import mlflow
import mlflow.sklearn
import mlflow.xgboost
import time
import os
from pathlib import Path


class MLFlowLogger:
    """
    MLflow logger for sklearn and XGBoost models.

    Supports logging of models, metrics, and parameters to MLflow.
    Can optionally log transformed metrics (with `_trans` suffix).

    Attributes
    ----------
    log_transformed_metrics : bool
        Whether to log metrics with the `_trans` suffix.
    """

    def __init__(
        self,
        log_transformed_metrics: bool = False,
        experiment_name: str = "house_price_prediction",
        tracking_uri: str | None = None,
    ):
        """
        Initialize MLFlowLogger.

        Parameters
        ----------
        log_transformed_metrics : bool, default=False
            If True, logs transformed metrics (with `_trans` suffix).
        """
        self.log_transformed_metrics = log_transformed_metrics
        if tracking_uri is None:
            tracking_uri = "logs/mlruns"

        os.makedirs(Path(tracking_uri) / ".trash", exist_ok=True)

        tracking_uri = Path(tracking_uri).resolve().as_uri()
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(experiment_name)

    def log_model(
        self,
        model,
        model_name: str = None,
        results: dict = None,
        use_xgb_train: bool = False,
        params: dict = None,
        folder_name: str = "model",
    ) -> None:
        """
        Log model, metrics, and parameters to MLflow.

        Parameters
        ----------
        model : object
            Trained model object (sklearn estimator or XGBoost model).
        model_name : str, optional
            Run name in MLflow. If None, a name with timestamp is generated.
        results : dict, optional
            Dictionary of metrics (keys are metric names, values are floats).
        use_xgb_train : bool, default=False
            If True, indicates the model was trained with `xgboost.train`
            (logged via `mlflow.xgboost.log_model`).
        params : dict, optional
            Hyperparameters to log. For sklearn, parameters are taken from
            `model.get_params()` if available.
        folder_name : str, default="model"
            Artifact folder path where the model and metrics will be stored.

        Notes
        -----
        - Only metrics with prefixes `train_`, `val_`, or `test_` are logged.
        - If `log_transformed_metrics=False`, metrics containing `_trans`
          are skipped.
        """
        results = results or {}
        model_name = model_name or f"model_run_{int(time.time())}"

        # Filter metrics
        metrics_to_log = {}
        for k, v in results.items():
            if any(prefix in k for prefix in ["train_", "val_", "test_"]):
                if self.log_transformed_metrics or "_trans" not in k:
                    metrics_to_log[k] = float(v)

        with mlflow.start_run(run_name=model_name):
            if use_xgb_train:
                mlflow.xgboost.log_model(model, artifact_path=folder_name)
            else:
                mlflow.sklearn.log_model(model, artifact_path=folder_name)

            for metric_name, value in metrics_to_log.items():
                mlflow.log_metric(metric_name, value)

            if use_xgb_train:
                if params:
                    mlflow.log_params(params)
            else:
                if hasattr(model, "get_params"):
                    mlflow.log_params(model.get_params())

            # Log metrics as artifact
            mlflow.log_text(
                str(metrics_to_log), artifact_file=f"{folder_name}/metrics.txt"
            )

            print(f"{model_name} -> {metrics_to_log}")
