import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor

from src.model.hyperparam_utils import (
    load_search_space,
    suggest_params_from_space,
)
from src.model.evaluate import ModelEvaluator
from src.features.data_prep_for_modelling.data_preparation import prepare_data
from src.features.feature_engineering.encoding import (
    encode_energy_labels_train_test_val,
)


def objective_xgb(
    trial,
    df: pd.DataFrame,
    features_config: str,
    hyperparam_config: str,
    model_name: str,
    use_log: bool = False,
) -> float:
    """
    Optuna objective function for XGBoost regression.

    Args:
        trial: Optuna trial object.
        df: Cleaned DataFrame with features and target.
        features_config: Path to YAML with feature config.
        hyperparam_config: Path to YAML with hyperparameter search space.
        use_log: Whether to apply log transform to the target.

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

    target_transform = np.log1p if use_log else None
    inverse_transform = np.expm1 if use_log else None

    # Evaluate model
    evaluator = ModelEvaluator(
        target_transform=target_transform,
        inverse_transform=inverse_transform,
    )

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
        model_name="xgb_optuna_log" if use_log else "xgb_optuna",
    )

    return results["test_rmse"]


def objective_rf(
    trial,
    df: pd.DataFrame,
    features_config: str,
    hyperparam_config: str,
    model_name: str,
    use_log: bool = False,
) -> float:
    """
    Optuna objective function for Random Forest regression.

    This function prepares the data, applies optional log transformation
    to the target, samples hyperparameters from the search space, fits
    a Random Forest model, and returns the test RMSE.

    Args:
        trial: Optuna trial object.
        df: Cleaned DataFrame with features and target.
        features_config: Path to YAML with feature configuration.
        hyperparam_config: Path to YAML with hyperparameter search space.
        model_name: Name of the model (used for data prep and logging).
        use_log: If True, applies log1p transform to the target and
                 expm1 to predictions.

    Returns:
        float: Test RMSE on the target
        (after inverse transform if log applied).
    """
    # Load search space
    base_params, search_space = load_search_space(
        hyperparam_config, "random_forest"
    )
    params = suggest_params_from_space(trial, base_params, search_space)

    # Prepare data
    X_train, X_test, y_train, y_test, _, X_val, y_val = prepare_data(
        df, config_path=features_config, model_name=model_name
    )
    X_train, X_test, X_val, _ = encode_energy_labels_train_test_val(
        X_train, X_test, X_val
    )

    # Set target transforms if use_log=True
    target_transform = np.log1p if use_log else None
    inverse_transform = np.expm1 if use_log else None

    # Use evaluator to fit and evaluate
    evaluator = ModelEvaluator(
        target_transform=target_transform, inverse_transform=inverse_transform
    )

    model = RandomForestRegressor(**params)
    _, _, _, results = evaluator.evaluate(
        model,
        X_train,
        y_train,
        X_test,
        y_test,
        X_val=X_val,
        y_val=y_val,
        model_name="rf_optuna_log" if use_log else "rf_optuna",
    )

    return results["test_rmse"]
