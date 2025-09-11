from src.features.feature_engineering.feature_engineering import (
    prepare_features,
)
from src.features.data_prep_for_modelling.data_preparation import prepare_data


def prepare_full_data(df, config_path, model_name):
    """
    Full data preparation pipeline:
    1. Select features & split according to YAML.
    2. Fit feature engineering on train set.
    3. Apply same transformations to test/validation sets.

    Args:
        df: Raw dataframe.
        config_path: Path to YAML defining features, target, splits, scaling.
        model_name: Name of model/config in YAML.

    Returns:
        X_train_fe, X_test_fe, y_train_fe, y_test_fe, scaler,
        X_val_fe, y_val_fe, log_cols, fe_encoders
    """
    # ------------------------
    # Step 1: Split & scale raw features using YAML
    # ------------------------
    X_train, X_test, y_train, y_test, scaler, X_val, y_val = prepare_data(
        df, config_path, model_name
    )

    # ------------------------
    # Step 2: Fit feature engineering on train
    # ------------------------
    X_train_fe, y_train_fe, log_cols, fe_encoders = prepare_features(
        X_train, y_train, return_encoders=True
    )

    # ------------------------
    # Step 3: Apply same feature engineering to test/val
    # ------------------------
    X_test_fe, y_test_fe, _ = prepare_features(
        X_test, y_test, fitted_encoders=fe_encoders
    )
    if X_val is not None:
        X_val_fe, y_val_fe, _ = prepare_features(
            X_val, y_val, fitted_encoders=fe_encoders
        )
    else:
        X_val_fe, y_val_fe = None, None

    return (
        X_train_fe,
        X_test_fe,
        y_train_fe,
        y_test_fe,
        scaler,
        X_val_fe,
        y_val_fe,
        log_cols,
        fe_encoders,
    )
