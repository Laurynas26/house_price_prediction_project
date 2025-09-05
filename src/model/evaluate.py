import mlflow
import mlflow.sklearn
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import xgboost as xgb


class ModelEvaluator:
    def __init__(self, metrics=None, default_fit_params=None):
        self.metrics = metrics or {
            "rmse": lambda y, y_pred: np.sqrt(mean_squared_error(y, y_pred)),
            "mae": lambda y, y_pred: mean_absolute_error(y, y_pred),
            "r2": r2_score,
        }
        self.results = {}
        self.default_fit_params = default_fit_params or {}

    def evaluate(
        self,
        model,
        X_train,
        y_train,
        X_test,
        y_test,
        fit_params=None,
        use_xgb_train=False,
        model_name=None,
    ):
        """
        Train model, make predictions, and compute metrics.

        Args:
            model: sklearn-style model OR xgboost params dict (if use_xgb_train=True)
            use_xgb_train: if True, use xgb.train instead of model.fit
        """
        model_name = model_name or "model_run"
        self.results = {}
        model_defaults = self.default_fit_params.get(model_name, {})
        params = {**model_defaults, **(fit_params or {})}
        X_val = params.pop("X_val", None)
        y_val = params.pop("y_val", None)

        if use_xgb_train:
            dtrain = xgb.DMatrix(X_train, label=y_train)

            if X_val is not None and y_val is not None:
                dval = xgb.DMatrix(X_val, label=y_val)
                evals = params.pop("evals", [(dval, "validation")])
            else:
                dval = xgb.DMatrix(X_test, label=y_test)
                evals = params.pop("evals", [(dval, "eval")])

            trained_model = xgb.train(
                model,
                dtrain,
                num_boost_round=params.pop("num_boost_round", 500),
                evals=evals,
                early_stopping_rounds=params.pop("early_stopping_rounds", 50),
                verbose_eval=params.pop("verbose_eval", False),
            )

            y_train_pred = trained_model.predict(dtrain)
            y_test_pred = trained_model.predict(
                xgb.DMatrix(X_test, label=y_test)
            )

        else:
            trained_model = model
            trained_model.fit(X_train, y_train, **params)
            y_train_pred = trained_model.predict(X_train)
            y_test_pred = trained_model.predict(X_test)

        for name, func in self.metrics.items():
            self.results[f"train_{name}"] = func(y_train, y_train_pred)
            self.results[f"test_{name}"] = func(y_test, y_test_pred)

        return trained_model, self.results

    def log_to_mlflow(
        self,
        model,
        model_name,
        results=None,
        use_xgb_train=False,
        folder_name=None,
    ):
        """
        Log model and metrics to MLflow.
        """
        model_name = model_name or "model_run"
        results = results or self.results
        folder_name = folder_name or "model"  # folder inside the run

        with mlflow.start_run(run_name=model_name):
            if use_xgb_train:
                mlflow.xgboost.log_model(model, artifact_path=folder_name)
            else:
                mlflow.sklearn.log_model(model, artifact_path=folder_name)

            mlflow.log_metrics(results)

            if not use_xgb_train and hasattr(model, "get_params"):
                mlflow.log_params(model.get_params())

            print(f"{model_name} -> {results}")
