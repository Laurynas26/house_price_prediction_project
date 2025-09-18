import pandas as pd
import numpy as np
from src.features.feature_engineering.feature_engineering import (
    prepare_features_train_val,
)
from src.features.feature_engineering.encoding import encode_train_val_only

from src.features.data_prep_for_modelling.data_preparation import (
    load_features_config,
)


def prepare_base_data(df: pd.DataFrame, features_config: str, model_name: str):
    """
    Prepare base feature set for CV folds without applying fold-specific feature engineering.

    Selects the raw features specified in the YAML config for the model, plus the target,
    and replaces "N/A" with np.nan.

    Args:
        df (pd.DataFrame): Raw cleaned dataframe with precomputed numeric columns (e.g., size_num).
        features_config (str): Path to YAML feature configuration.
        model_name (str): Name of the model to fetch features from the config.

    Returns:
        X_full (pd.DataFrame): Base feature dataframe (pre-FE, no encoding applied yet)
        y_full (pd.Series): Target variable
    """
    # Load features and target from YAML
    features, target, _, _ = load_features_config(features_config, model_name)

    # Ensure target is present
    if target not in df.columns:
        raise KeyError(f"Target column '{target}' not found in dataframe.")

    y_full = df[target]

    # Ensure all selected features exist in the dataframe
    missing_cols = [c for c in features if c not in df.columns]
    if missing_cols:
        raise KeyError(
            f"The following features are missing in the dataframe: {missing_cols}"
        )

    X_full = df[features].copy()

    # Replace "N/A" with np.nan for proper FE handling
    X_full.replace("N/A", np.nan, inplace=True)

    return X_full, y_full


def prepare_fold_features(
    X_train: pd.DataFrame,
    X_val: pd.DataFrame,
    use_extended_features: bool = True,
):
    """
    Apply fold-wise feature engineering safely, avoiding test leakage.

    Handles:
      - Extra numeric features: floor_level, lease_years_remaining, backyard_num, balcony_flag
      - Log-transforms for skewed numeric features
      - Binary flags
      - One-hot encoding for categorical columns
      - Fold-safe energy_label encoding

    Args:
        X_train (pd.DataFrame): Training features for the fold
        X_val (pd.DataFrame): Validation features for the fold
        use_extended_features (bool): Whether to apply full extended feature engineering

    Returns:
        X_train_fe (pd.DataFrame): Transformed training features
        X_val_fe (pd.DataFrame): Transformed validation features
        meta (dict): Metadata from feature engineering (log_cols, encoders, etc.)
        fold_encoders (dict): Encoders created for this fold
    """
    if not use_extended_features:
        return X_train, X_val, {}, {}

    X_train_fe, X_val_fe, meta = prepare_features_train_val(X_train, X_val)

    # Fold-safe encoding for energy_label
    X_train_fe, X_val_fe, energy_encoder = encode_train_val_only(
        X_train_fe, X_val_fe
    )
    meta["energy_label_encoder"] = energy_encoder

    fold_encoders = {"energy_label": energy_encoder}

    return X_train_fe, X_val_fe, meta, fold_encoders
