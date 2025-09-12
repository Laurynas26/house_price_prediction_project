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
    use_extended_features: bool = False,
    X_train=None,
    y_train=None,
    X_val=None,
    y_val=None,
) -> float:
    """
    Optuna objective function for XGBoost regression with optional log-transformed target.

    This function samples hyperparameters from a given search space (defined in a YAML file),
    prepares the training and validation datasets if they are not provided, applies optional
    log transformation to the target, fits an XGBoost model with early stopping, and returns
    the RMSE on the validation set (used as the optimization objective for Optuna).

    Args:
        trial: Optuna trial object for hyperparameter suggestions.
        df: Cleaned pandas DataFrame containing features and target.
        features_config: Path to YAML file specifying feature configuration.
        hyperparam_config: Path to YAML file specifying hyperparameter search space.
        model_name: Name of the model (used for logging and data preparation).
        use_log: If True, applies `log1p` transformation to the target before fitting
                 and `expm1` to predictions.
        use_extended_features: Reserved for future feature extension logic (currently unused).
        X_train: Optional pre-prepared training features. If None, data will be prepared from `df`.
        y_train: Optional pre-prepared training target. If None, data will be prepared from `df`.
        X_val: Optional pre-prepared validation features. If None, data will be prepared from `df`.
        y_val: Optional pre-prepared validation target. If None, data will be prepared from `df`.

    Returns:
        float: Validation RMSE (after inverse transformation if `use_log=True`), which is
               minimized by Optuna during hyperparameter tuning.
    """
    # Load search space from config
    base_params, search_space = load_search_space(hyperparam_config, "xgboost")
    params = suggest_params_from_space(trial, base_params, search_space)

    # Prepare data if not already provided
    if X_train is None or X_val is None or y_train is None or y_val is None:
        X_train, _, y_train, _, _, X_val, y_val = prepare_data(
            df,
            config_path=features_config,
            model_name=model_name,
            use_extended_features=use_extended_features,
        )
        # Encode categorical features
        X_train, _, X_val, _ = encode_energy_labels_train_test_val(
            X_train, None, X_val
        )

    target_transform = np.log1p if use_log else None
    inverse_transform = np.expm1 if use_log else None

    # Evaluate model
    evaluator = ModelEvaluator(
        target_transform=target_transform,
        inverse_transform=inverse_transform,
    )

    _, _, _, _, results = evaluator.evaluate(
        params,
        X_train,
        y_train,
        X_test=X_val,
        y_test=y_val,
        X_val=X_val,     # same validation set for early stopping
        y_val=y_val,
        fit_params={"num_boost_round": 1000, "early_stopping_rounds": 50},
        use_xgb_train=True,
        model_name="xgb_optuna_log" if use_log else "xgb_optuna",
    )

    return results["test_rmse"]


def objective_rf(
    trial,
    df=None,
    features_config=None,
    hyperparam_config=None,
    model_name=None,
    use_log: bool = False,
    use_extended_features: bool = False,
    X_train=None,
    y_train=None,
    X_val=None,
    y_val=None,
) -> float:
    """
    Optuna objective function for Random Forest regression.

    Args:
        trial: Optuna trial object.
        df: Cleaned DataFrame with features and target.
        features_config: Path to YAML with feature configuration.
        hyperparam_config: Path to YAML with hyperparameter search space.
        model_name: Name of the model (used for data prep and logging).
        use_log: If True, applies log1p transform to the target and expm1 to predictions.
        X_train, y_train, X_val, y_val: Optionally provide pre-split data.

    Returns:
        float: Validation RMSE on the target (after inverse transform if log applied).
    """
    # Load hyperparameter search space
    base_params, search_space = load_search_space(
        hyperparam_config, "random_forest"
    )
    params = suggest_params_from_space(trial, base_params, search_space)

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

    # Target transformation
    target_transform = np.log1p if use_log else None
    inverse_transform = np.expm1 if use_log else None

    # Initialize evaluator
    evaluator = ModelEvaluator(
        target_transform=target_transform, inverse_transform=inverse_transform
    )

    # Fit Random Forest and evaluate on validation set
    model = RandomForestRegressor(**params)
    _, _, _, _, results = evaluator.evaluate(
        model,
        X_train,
        y_train,
        X_test=X_val,
        y_test=y_val,
        X_val=X_val,     
        y_val=y_val,
        model_name="rf_optuna_log" if use_log else "rf_optuna",
    )

    # Return validation RMSE
    return results["test_rmse"]
