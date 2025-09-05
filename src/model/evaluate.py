import mlflow
import mlflow.sklearn
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def evaluate_model(
    model, X_train, y_train, X_test, y_test, metrics=None, fit_params=None
):
    """
    Fit model, predict, and return evaluation metrics.
    """
    if fit_params is None:
        fit_params = {}
    model.fit(X_train, y_train, **fit_params)

    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    results = {}
    if metrics is None:
        metrics = {
            "rmse": lambda y, y_pred: np.sqrt(mean_squared_error(y, y_pred)),
            "mea": lambda y, y_pred: mean_absolute_error(y, y_pred),
            "r2": r2_score,
        }

    for name, func in metrics.items():
        results[f"train_{name}"] = func(y_train, y_train_pred)
        results[f"test_{name}"] = func(y_test, y_test_pred)

    return model, results


def log_to_mlflow(model, model_name, results):
    """
    Log model and metrics to MLflow.
    """
    with mlflow.start_run(run_name=model_name):
        mlflow.sklearn.log_model(model, f"{model_name}_model")
        mlflow.log_metrics(results)
        if hasattr(model, "get_params"):
            mlflow.log_params(model.get_params())
        print(f"{model_name} -> {results}")
