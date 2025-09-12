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
        Evaluate models with optional target transformation and early stopping.

        Args:
            metrics: dict of metric_name -> function(y_true, y_pred)
            default_fit_params: dict of model_name -> default fit parameters
            target_transform: function applied to y before fitting (e.g., np.log1p)
            inverse_transform: function applied to predictions to revert transform (e.g., np.expm1)
        """
        self.metrics = metrics or {
            "rmse": lambda y, y_pred: np.sqrt(mean_squared_error(y, y_pred)),
            "mae": lambda y, y_pred: mean_absolute_error(y, y_pred),
            "r2": r2_score,
            "mape": lambda y, y_pred: np.mean(np.abs((y - y_pred) / y)) * 100,
        }
        self.default_fit_params = default_fit_params or {}
        self.target_transform = target_transform
        self.inverse_transform = inverse_transform
        self.results = {}

    def _apply_transform(self, y, inverse=False):
        if inverse and self.inverse_transform:
            return self.inverse_transform(y)
        elif not inverse and self.target_transform:
            return self.target_transform(y)
        return y

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
        Evaluate a model with training and validation sets.

        Args:
            model: sklearn model or XGBoost params dict
            X_train, y_train: training data
            X_val, y_val: validation data (required for XGBoost early stopping)
            X_test, y_test: test data (metrics only)
            fit_params: additional fit parameters
            use_xgb_train: if True, use xgboost.train with early stopping
            model_name: optional string for default fit parameters
        """
        model_name = model_name or "model_run"
        self.results = {}
        model_defaults = self.default_fit_params.get(model_name, {})
        params = {**model_defaults, **(fit_params or {})}

        # Transform targets
        y_train_trans = self._apply_transform(y_train)
        y_val_trans = (
            self._apply_transform(y_val) if y_val is not None else None
        )
        y_test_trans = self._apply_transform(y_test)

        if use_xgb_train:
            if X_val is None or y_val is None:
                raise ValueError(
                    "Validation set must be provided for XGBoost to avoid leakage."
                )

            dtrain = xgb.DMatrix(X_train, label=y_train_trans)
            dval = xgb.DMatrix(X_val, label=y_val_trans)
            evals = [(dval, "validation")]

            trained_model = xgb.train(
                params,
                dtrain,
                num_boost_round=params.pop("num_boost_round", 500),
                evals=evals,
                early_stopping_rounds=params.pop("early_stopping_rounds", 50),
                verbose_eval=params.pop("verbose_eval", False),
            )

            y_train_pred = trained_model.predict(dtrain)
            y_val_pred = trained_model.predict(dval)
            y_test_pred = trained_model.predict(xgb.DMatrix(X_test))

        else:
            # Standard sklearn model
            trained_model = model
            trained_model.fit(X_train, y_train_trans, **params)
            y_train_pred = trained_model.predict(X_train)
            y_val_pred = (
                trained_model.predict(X_val) if X_val is not None else None
            )
            y_test_pred = trained_model.predict(X_test)

        # Inverse transform predictions
        y_train_pred = self._apply_transform(y_train_pred, inverse=True)
        y_val_pred = (
            self._apply_transform(y_val_pred, inverse=True)
            if y_val_pred is not None
            else None
        )
        y_test_pred = self._apply_transform(y_test_pred, inverse=True)

        # Compute metrics
        for name, func in self.metrics.items():
            self.results[f"train_{name}"] = func(y_train, y_train_pred)
            if y_val_pred is not None:
                self.results[f"val_{name}"] = func(y_val, y_val_pred)
            self.results[f"test_{name}"] = func(y_test, y_test_pred)

        return (
            trained_model,
            y_train_pred,
            y_val_pred,
            y_test_pred,
            self.results,
        )
