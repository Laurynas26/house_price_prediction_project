import mlflow
import mlflow.sklearn
import mlflow.xgboost

class MLFlowLogger:
    def __init__(self):
        pass

    def log_model(
        self,
        model,
        model_name,
        results=None,
        use_xgb_train=False,
        folder_name=None
    ):
        results = results or {}
        folder_name = folder_name or "model"
        model_name = model_name or "model_run"

        with mlflow.start_run(run_name=model_name):
            if use_xgb_train:
                mlflow.xgboost.log_model(model, artifact_path=folder_name)
            else:
                mlflow.sklearn.log_model(model, artifact_path=folder_name)

            mlflow.log_metrics(results)

            if not use_xgb_train and hasattr(model, "get_params"):
                mlflow.log_params(model.get_params())

            print(f"{model_name} -> {results}")
