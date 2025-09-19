import pandas as pd
from src.features.feature_engineering import feature_engineering_cv as fe_cv
from src.features.data_prep_for_modelling.data_preparation import (
    load_features_config,
)


def prepare_base_data(df: pd.DataFrame, features_config: str, model_name: str):
    """
    Prepare base feature set for CV folds without fold-specific feature engineering.

    Selects features from the YAML config for the model, plus the target,
    and replaces "N/A" with np.nan for proper feature engineering.

    Args:
        df (pd.DataFrame): Raw cleaned dataframe.
        features_config (str): Path to YAML feature configuration.
        model_name (str): Model name to fetch features from the config.

    Returns:
        X_full (pd.DataFrame): Base feature dataframe (pre-FE, no encoding yet)
        y_full (pd.Series): Target variable
    """
    features, target, _, _ = load_features_config(features_config, model_name)

    if target not in df.columns:
        raise KeyError(f"Target column '{target}' not found in dataframe.")

    y_full = df[target]

    missing_cols = [c for c in features if c not in df.columns]
    if missing_cols:
        raise KeyError(
            f"The following features are missing in the dataframe: {missing_cols}"
        )

    X_full = df[features].copy()
    X_full.replace("N/A", pd.NA, inplace=True)

    return X_full, y_full


def prepare_fold_features(
    X_train: pd.DataFrame,
    X_val: pd.DataFrame = None,
    use_extended_features: bool = True,
):
    """
    CV-safe fold-wise feature engineering wrapper.

    Delegates to feature_engineering_cv.prepare_fold_features
    to avoid recursion and double encoding.

    Returns:
        X_train_fe, X_val_fe, meta, fold_encoders
    """
    return fe_cv.prepare_fold_features(X_train, X_val, use_extended_features)
