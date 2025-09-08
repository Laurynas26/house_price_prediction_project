from typing import Optional
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

from src.model.hyperparam_utils import (
    load_search_space,
    suggest_params_from_space,
)
from src.model.evaluate import ModelEvaluator
from src.features.data_prep_for_modelling.data_preparation import prepare_data
from src.features.feature_engineering.encoding import (
    encode_energy_label,
    encode_energy_labels_train_test_val,
)


def objective_xgb(
    trial,
    df: pd.DataFrame,
    features_config: str,
    hyperparam_config: str,
    model_name: str,
) -> float:
    """
    Optuna objective function for XGBoost regression.

    Args:
        trial: Optuna trial object.
        df: Cleaned DataFrame with features and target.
        features_config: Path to YAML with feature config.
        hyperparam_config: Path to YAML with hyperparameter search space.

    Returns:
        test_rmse: The RMSE on the test set (to minimize in Optuna).
    """
    # Load search space from config
    base_params, search_space = load_search_space(hyperparam_config, "xgboost")
    params = suggest_params_from_space(trial, base_params, search_space)

    # Prepare data (train/test/optional validation)
    X_train, X_test, y_train, y_test, _, X_val, y_val = prepare_data(
        df, config_path=features_config, model_name=model_name
    )

    # Encode categorical features
    X_train, X_test, X_val, enc = encode_energy_labels_train_test_val(
        X_train, X_test, X_val
    )

    # Evaluate model
    evaluator = ModelEvaluator()
    _, _, _, results = evaluator.evaluate(
        params,
        X_train,
        y_train,
        X_test,
        y_test,
        X_val=X_val,
        y_val=y_val,
        fit_params={"num_boost_round": 1000, "early_stopping_rounds": 50},
        use_xgb_train=True,
        model_name="xgb_optuna",
    )

    return results["test_rmse"]


def objective_rf(
    trial,
    df: pd.DataFrame,
    features_config: str,
    hyperparam_config: str,
    model_name: str,
) -> float:
    """
    Optuna objective function for Random Forest regression.

    Args:
        trial: Optuna trial object.
        df: Cleaned DataFrame with features and target.
        features_config: Path to YAML with feature config.
        hyperparam_config: Path to YAML with hyperparameter search space.

    Returns:
        test_rmse: The RMSE on the test set (to minimize in Optuna).
    """
    # Load search space from config
    base_params, search_space = load_search_space(
        hyperparam_config, "random_forest"
    )
    params = suggest_params_from_space(trial, base_params, search_space)

    # Prepare data
    X_train, X_test, y_train, y_test, _, X_val, y_val = prepare_data(
        df, config_path=features_config, model_name=model_name
    )

    # Encode categorical features
    X_train, X_test, X_val, enc = encode_energy_labels_train_test_val(
        X_train, X_test, X_val
    )

    # Initialize and fit Random Forest
    model = RandomForestRegressor(**params)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    # Evaluate test RMSE
    test_rmse = np.sqrt(mean_squared_error(y_test, preds))
    return test_rmse
