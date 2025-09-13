import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression

from src.model.utils import (
    load_model_config_and_search_space,
    suggest_params_from_space,
)
from src.model.evaluate import ModelEvaluator
from src.features.data_prep_for_modelling.data_preparation import prepare_data
from src.features.feature_engineering.encoding import (
    encode_energy_labels_train_test_val,
)


def unified_objective(
    trial,
    model_name: str,
    df: pd.DataFrame,
    features_config: str,
    model_config: str,
    use_log: bool = False,
    use_extended_features: bool = False,
    X_train=None,
    y_train=None,
    X_val=None,
    y_val=None,
) -> float:
    """
    Unified Optuna objective for any model defined in YAML.
    Supports XGBoost, Random Forest, Linear Regression, with optional log-transform.

    Returns:
        float: Validation RMSE (inverse-transformed if log applied)
    """
    # Load model config + optional search space
    model_params, fit_params, search_space = (
        load_model_config_and_search_space(model_config, model_name)
    )
    model_params, fit_params = suggest_params_from_space(
        trial, model_params, fit_params, search_space
    )

    # Prepare data if not provided
    if X_train is None or X_val is None or y_train is None or y_val is None:
        X_train, _, y_train, _, _, X_val, y_val = prepare_data(
            df,
            config_path=features_config,
            model_name=model_name,
            use_extended_features=use_extended_features,
        )
        X_train, _, X_val, _ = encode_energy_labels_train_test_val(
            X_train, None, X_val
        )

    # Target transform
    target_transform = np.log1p if use_log else None
    inverse_transform = np.expm1 if use_log else None

    evaluator = ModelEvaluator(
        target_transform=target_transform, inverse_transform=inverse_transform
    )

    # Determine model type
    if "xgb" in model_name.lower():
        # XGBoost with early stopping
        _, _, _, _, results = evaluator.evaluate(
            model=None,
            X_train=X_train,
            y_train=y_train,
            X_test=X_val,
            y_test=y_val,
            X_val=X_val,
            y_val=y_val,
            model_params=model_params,
            fit_params=fit_params,
            use_xgb_train=True,
        )

    elif "rf" in model_name.lower():
        model = RandomForestRegressor(**model_params)
        _, _, _, _, results = evaluator.evaluate(
            model=model,
            X_train=X_train,
            y_train=y_train,
            X_test=X_val,
            y_test=y_val,
            X_val=X_val,
            y_val=y_val,
            fit_params=fit_params,
        )

    elif "linear" in model_name.lower():
        model = LinearRegression(**model_params)
        _, _, _, _, results = evaluator.evaluate(
            model=model,
            X_train=X_train,
            y_train=y_train,
            X_test=X_val,
            y_test=y_val,
            X_val=X_val,
            y_val=y_val,
            fit_params=fit_params,
        )

    else:
        raise ValueError(f"Unsupported model: {model_name}")

    return results["test_rmse"]
