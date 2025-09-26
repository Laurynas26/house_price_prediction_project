import pandas as pd
import numpy as np
from src.features.feature_engineering import feature_engineering_cv as fe_cv
from src.features.data_prep_for_modelling.data_preparation import (
    load_features_config,
)


def prepare_base_data(
    df: pd.DataFrame,
    features_config: str,
    model_name: str,
    extended_fe: bool = True,
):
    """
    Prepare base feature set for CV folds with optional extra columns for extended feature engineering.

    Args:
        df: Input dataframe
        features_config: Path to YAML config
        model_name: Model name in YAML
        extended_fe: Whether to include extra columns needed for fold-wise feature engineering

    Returns:
        X_full (pd.DataFrame): Feature dataframe
        y_full (pd.Series): Target
    """
    features, target, _, _ = load_features_config(features_config, model_name)

    df_copy = df.copy()

    extra_cols = (
        [
            "facilities",
            "facilities_list",
            "garden_location",
            "num_parcels",
            "parcels_concat",
            "has_zwembad",
            "has_verwarming",
            "has_vliering",
            "has_windmolen",
        ]
        if extended_fe
        else []
    )

    cols_to_keep = [c for c in features + extra_cols if c in df_copy.columns]
    missing_cols = [c for c in features if c not in df_copy.columns]
    if missing_cols:
        raise KeyError(
            f"The following required features are missing: {missing_cols}"
        )

    X_full = df_copy[cols_to_keep].replace("N/A", np.nan).fillna(0)
    y_full = df_copy[target]

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
