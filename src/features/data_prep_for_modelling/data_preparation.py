import yaml
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from src.features.feature_engineering.feature_engineering import (
    prepare_features_train_val,
    prepare_features_test,
)
from src.features.feature_engineering.encoding import (
    encode_energy_labels_train_test_val,
)

SCALERS = {
    "StandardScaler": StandardScaler,
    "MinMaxScaler": MinMaxScaler,
    "RobustScaler": RobustScaler,
}


def load_features_config(config_path, model_name):
    """Load features, target, split, and scaling config for a specific model from YAML."""
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    model_cfg = cfg[model_name]
    features = model_cfg.get("features", [])
    target = model_cfg["target"]
    split_cfg = model_cfg.get(
        "train_test_split", {"test_size": 0.2, "random_state": 42}
    )
    scaling_cfg = model_cfg.get(
        "scaling", {"method": "StandardScaler", "scale": True}
    )
    return features, target, split_cfg, scaling_cfg


def select_and_clean(df, features, target):
    """Select columns and handle missing values."""
    X = df[features].replace("N/A", np.nan).fillna(0)
    y = df[target]
    return X, y


def split_train_val_test_data(
    X, y, test_size=0.2, val_size=0.1, random_state=42, val_required=False
):
    """Split features and target into train/val/test sets."""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    if val_required:
        val_relative_size = val_size / (1 - test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_train,
            y_train,
            test_size=val_relative_size,
            random_state=random_state,
        )
        return X_train, X_val, X_test, y_train, y_val, y_test
    return X_train, X_test, y_train, y_test


def scale_data(X_train, X_test, X_val=None, scaler_cls=StandardScaler):
    """Scale numeric features using provided scaler class."""
    scaler = scaler_cls()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    X_val_scaled = scaler.transform(X_val) if X_val is not None else None
    return X_train_scaled, X_test_scaled, X_val_scaled, scaler


def prepare_data(
    df: pd.DataFrame,
    config_path: str,
    model_name: str,
    use_extended_features: bool = False,
    cv: bool = False,
):
    """
    Main entry point for preparing data splits and feature engineering.

    Modes:
        - Normal training/testing:
            Splits into train/test (+ optional val if config says so).
            Applies extended feature engineering to all splits (train/val/test).
        - Cross-validation (cv=True):
            Splits into train/val only (per fold).
            Applies extended feature engineering only to train/val (no test prep).

    Args:
        df (pd.DataFrame): Raw dataset.
        config_path (str): Path to YAML config controlling features and splits.
        model_name (str): Model name (used to select config options).
        use_extended_features (bool): Whether to apply extended feature engineering.
        cv (bool): If True, skip test transformations (CV handles val folds only).

    Returns:
        X_train (pd.DataFrame): Training features.
        X_test (pd.DataFrame | None): Test features, or None if cv=True.
        y_train (pd.Series): Training target.
        y_test (pd.Series | None): Test target, or None if cv=True.
        X_val (pd.DataFrame | None): Validation features (if config includes val split).
        y_val (pd.Series | None): Validation target.
        scaler (sklearn transformer | None): Fitted scaler, if scaling applied.
        meta (dict): Metadata/encoders for feature engineering.
    """
    # --- Load config ---
    features, target, split_cfg, scaling_cfg = load_features_config(
        config_path, model_name
    )
    X, y = select_and_clean(df, features, target)

    # --- Train/test/(val) split ---
    val_required = split_cfg.get("val_required", False)
    val_size = split_cfg.get("val_size", 0.1)

    split_result = split_train_val_test_data(
        X,
        y,
        test_size=split_cfg.get("test_size", 0.2),
        random_state=split_cfg.get("random_state", 42),
        val_required=val_required,
        val_size=val_size,
    )

    if val_required:
        X_train, X_val, X_test, y_train, y_val, y_test = split_result
    else:
        X_train, X_test, y_train, y_test = split_result
        X_val, y_val = None, None

    meta = {}
    fe_encoders = {}

    # --- Extended FE ---
    if use_extended_features:
        if cv:
            # Train/val only for CV
            X_train, X_val, y_train, y_val, meta = prepare_features_train_val(
                pd.concat([X_train, y_train], axis=1),
                (
                    pd.concat([X_val, y_val], axis=1)
                    if X_val is not None
                    else None
                ),
            )
            X_test, y_test = None, None
        else:
            # Train/val
            X_train, X_val, y_train, y_val, meta = prepare_features_train_val(
                pd.concat([X_train, y_train], axis=1),
                (
                    pd.concat([X_val, y_val], axis=1)
                    if X_val is not None
                    else None
                ),
            )
            # Test
            X_test = (
                prepare_features_test(
                    pd.concat([X_test, y_test], axis=1), meta
                )
                if X_test is not None
                else None
            )

    # --- Energy label encoding globally if not CV and not extended FE ---
    if not cv and not use_extended_features:
        X_train, X_test, X_val, energy_enc = (
            encode_energy_labels_train_test_val(X_train, X_test, X_val)
        )
        fe_encoders["energy_label"] = energy_enc

    # --- Scaling ---
    if scaling_cfg.get("scale", True):
        scaler_cls = SCALERS.get(
            scaling_cfg.get("method", "StandardScaler"), StandardScaler
        )
        X_train, X_test, X_val, scaler = scale_data(
            X_train, X_test, X_val, scaler_cls
        )
    else:
        scaler = None

    return (
        X_train,
        X_test,
        y_train,
        y_test,
        X_val,
        y_val,
        scaler,
        {**meta, **fe_encoders},
    )
