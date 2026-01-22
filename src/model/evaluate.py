import numpy as np
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score,
)
from src.model.utils import huber_loss
import xgboost as xgb


class ModelEvaluator:
    def __init__(
        self,
        metrics: dict[str, callable] | None = None,
        target_transform=None,
        inverse_transform=None,
        huber_delta=1.0,
    ):
        """
        Unified model training and evaluation utility supporting:
        - sklearn-style estimators
        - raw XGBoost training with early stopping
        - optional target transformation (e.g. log-scale training)

        Computes consistent train / validation / test metrics and
        returns both predictions and trained model artifacts.
        """
        self.huber_delta = huber_delta
        self.metrics = metrics or {
            "rmse": lambda y, y_pred: np.sqrt(mean_squared_error(y, y_pred)),
            "mae": lambda y, y_pred: mean_absolute_error(y, y_pred),
            "r2": r2_score,
            "mape": lambda y, y_pred: np.mean(
                np.abs((y - y_pred) / np.maximum(np.abs(y), 1e-8))
            )
            * 100,
            "huber": lambda y, y_pred: huber_loss(
                y, y_pred, delta=self.huber_delta
            ),
        }
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
        model_params=None,
        fit_params=None,
        use_xgb_train=False,
    ):
        """
        Train (optionally) and evaluate a model using train/val/test splits.

        Supports:
        - sklearn estimators via `.fit()` / `.predict()`
        - XGBoost via `xgb.train()` with early stopping

        Target transformations are applied consistently during training
        and inverted for metric computation.

        Returns:
            trained_model
            y_train_pred
            y_val_pred
            y_test_pred
            results (dict of metric_name -> value)
        """
        self.results = {}

        # Apply target transformation
        y_train_trans = self._apply_transform(y_train)
        y_val_trans = (
            self._apply_transform(y_val) if y_val is not None else None
        )
        if use_xgb_train:
            if X_val is None or y_val is None:
                raise ValueError(
                    "Validation set must be provided for XGBoost early"
                    " stopping."
                )
            dtrain = xgb.DMatrix(X_train, label=y_train_trans)
            dval = xgb.DMatrix(X_val, label=y_val_trans)
            evals = [(dval, "validation")]

            trained_model = xgb.train(
                params=model_params or {},
                dtrain=dtrain,
                num_boost_round=(fit_params or {}).get("num_boost_round", 500),
                evals=evals,
                early_stopping_rounds=(fit_params or {}).get(
                    "early_stopping_rounds", 50
                ),
                verbose_eval=(fit_params or {}).get("verbose_eval", False),
            )

            y_train_pred = trained_model.predict(dtrain)
            y_val_pred = trained_model.predict(dval)
            y_test_pred = trained_model.predict(xgb.DMatrix(X_test))

        else:
            # Standard sklearn model
            trained_model = model
            combined_fit_params = {
                **(model_params or {}),
                **(fit_params or {}),
            }
            trained_model.fit(X_train, y_train_trans, **combined_fit_params)

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
