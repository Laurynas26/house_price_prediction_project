import mlflow
import mlflow.sklearn
import mlflow.xgboost
import time


class MLFlowLogger:
    def __init__(self, log_transformed_metrics=False):
        """
        MLflow logger for sklearn and XGBoost models.

        Args:
            log_transformed_metrics: If True, logs metrics with "_trans" suffix.
        """
        self.log_transformed_metrics = log_transformed_metrics

    def log_model(self, model, model_name=None, results=None, use_xgb_train=False, params=None, folder_name="model"):
        """
        Logs model, metrics, and parameters to MLflow.

        Args:
            model: Trained model object (sklearn or XGBoost)
            model_name: Name of the run (optional, will add timestamp if None)
            results: dict of metrics
            use_xgb_train: True if model was trained via xgb.train
            params: dict of hyperparameters to log
            folder_name: Artifact path folder name
        """
        results = results or {}
        model_name = model_name or f"model_run_{int(time.time())}"

        # Filter metrics: log only train/val/test metrics; optionally include "_trans"
        metrics_to_log = {}
        for k, v in results.items():
            if any(prefix in k for prefix in ["train_", "val_", "test_"]):
                if self.log_transformed_metrics or "_trans" not in k:
                    metrics_to_log[k] = float(v)

        with mlflow.start_run(run_name=model_name):
            # Log model
            if use_xgb_train:
                mlflow.xgboost.log_model(model, artifact_path=folder_name)
            else:
                mlflow.sklearn.log_model(model, artifact_path=folder_name)

            # Log metrics
            for metric_name, value in metrics_to_log.items():
                mlflow.log_metric(metric_name, value)

            # Log parameters
            if use_xgb_train:
                if params:
                    mlflow.log_params(params)
            else:
                if hasattr(model, "get_params"):
                    mlflow.log_params(model.get_params())

            # Log metrics as artifact
            mlflow.log_text(str(metrics_to_log), artifact_file=f"{folder_name}/metrics.txt")

            print(f"{model_name} -> {metrics_to_log}")
