import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import xgboost as xgb


class ModelEvaluator:
    def __init__(
        self,
        metrics=None,
        default_fit_params=None,
        target_transform=None,
        inverse_transform=None,
    ):
        """
        Args:
            metrics: dict of metric_name -> function(y_true, y_pred)
            default_fit_params: dict of model_name -> default fit params
            target_transform: function to apply to y before fitting (e.g., np.log1p)
            inverse_transform: function to inverse-transform predictions (e.g., np.expm1)
        """
        self.metrics = metrics or {
            "rmse": lambda y, y_pred: np.sqrt(mean_squared_error(y, y_pred)),
            "mae": lambda y, y_pred: mean_absolute_error(y, y_pred),
            "r2": r2_score,
        }
        self.default_fit_params = default_fit_params or {}
        self.target_transform = target_transform
        self.inverse_transform = inverse_transform
        self.results = {}

    def evaluate(
        self,
        model,
        X_train,
        y_train,
        X_test,
        y_test,
        X_val=None,
        y_val=None,
        fit_params=None,
        use_xgb_train=False,
        model_name=None,
    ):
        """
        Evaluate a model (scikit-learn or XGBoost).
        For XGBoost, supports early stopping
        with optional validation set, and optional target transform.
        """
        model_name = model_name or "model_run"
        self.results = {}
        model_defaults = self.default_fit_params.get(model_name, {})
        params = {**model_defaults, **(fit_params or {})}

        y_train_trans = (
            self.target_transform(y_train)
            if self.target_transform
            else y_train
        )
        y_val_trans = (
            self.target_transform(y_val)
            if (y_val is not None and self.target_transform)
            else y_val
        )

        if use_xgb_train:
            dtrain = xgb.DMatrix(X_train, label=y_train_trans)

            if X_val is not None and y_val_trans is not None:
                dval = xgb.DMatrix(X_val, label=y_val_trans)
                evals = [(dval, "validation")]
            else:
                dval = xgb.DMatrix(X_test, label=y_test)
                evals = [(dval, "eval")]

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
            trained_model.fit(X_train, y_train_trans, **params)
            y_train_pred = trained_model.predict(X_train)
            y_test_pred = trained_model.predict(X_test)

        if self.inverse_transform:
            y_train_pred = self.inverse_transform(y_train_pred)
            y_test_pred = self.inverse_transform(y_test_pred)

        for name, func in self.metrics.items():
            self.results[f"train_{name}"] = func(y_train, y_train_pred)
            self.results[f"test_{name}"] = func(y_test, y_test_pred)

        return trained_model, y_train_pred, y_test_pred, self.results
