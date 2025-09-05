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
        self.default_fit_params = default_fit_params or {}
        self.results = {}

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
            y_test_pred = trained_model.predict(xgb.DMatrix(X_test))

        else:
            trained_model = model
            trained_model.fit(X_train, y_train, **params)
            y_train_pred = trained_model.predict(X_train)
            y_test_pred = trained_model.predict(X_test)

        for name, func in self.metrics.items():
            self.results[f"train_{name}"] = func(y_train, y_train_pred)
            self.results[f"test_{name}"] = func(y_test, y_test_pred)

        return trained_model, self.results
