import yaml
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from src.features.feature_engineering.feature_engineering import (
    prepare_features_train_val,
    prepare_features_test,
)
from src.features.feature_engineering.feature_expansion import (
    feature_expansion,
)
from src.features.feature_engineering.encoding import (
    encode_energy_labels_train_test_val,
)

SCALERS = {
    "StandardScaler": StandardScaler,
    "MinMaxScaler": MinMaxScaler,
    "RobustScaler": RobustScaler,
}


def load_features_config(config_path: str, model_name: str):
    """
    Load features, target, split, and scaling configuration for a model
    from YAML.

    Returns:
        features (list): Feature column names.
        target (str): Target column name.
        split_cfg (dict): Train/test split configuration.
        scaling_cfg (dict): Scaling configuration.
    """
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


def select_and_clean(
    df: pd.DataFrame, features: list, target: str, extended_fe: bool = False
):
    """
    Select relevant columns and clean missing values.

    Args:
        df: Input dataframe.
        features: List of features.
        target: Target column name.
        extended_fe: Whether to include extra columns for extended feature
        engineering.

    Returns:
        X (pd.DataFrame): Feature DataFrame
        y (pd.Series): Target series
    """
    df_copy = df.copy()
    extra_cols = (
        [
            # Location / property metadata
            "located_on",
            "ownership_type",
            "postal_code_clean",
            "status",
            "roof_type",
            "location",
            "garden",
            # Property amenities / features
            "backyard",
            "balcony",
            "facilities",
            "facilities_list",
            "garden_location",
            "num_parcels",
            "parcels_concat",
            # Extra binary / luxury features
            "has_zwembad",
            "has_verwarming",
            "has_vliering",
            "has_windmolen",
        ]
        if extended_fe
        else []
    )

    cols_to_keep = list(set(features + extra_cols))
    X = df_copy[cols_to_keep].replace("N/A", np.nan).fillna(0)
    y = df_copy[target]
    return X, y


def split_train_val_test_data(
    X, y, test_size=0.2, val_size=0.1, random_state=42, val_required=False
):
    """
    Split features and target into train/val/test sets.

    Returns:
        X_train, X_val, X_test, y_train, y_val, y_test
    """
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
    """
    Scale numeric + OHE features using specified scaler.
    """
    scaler = scaler_cls()
    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train),
        columns=X_train.columns,
        index=X_train.index,
    )
    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test), columns=X_test.columns, index=X_test.index
    )
    X_val_scaled = (
        pd.DataFrame(
            scaler.transform(X_val), columns=X_val.columns, index=X_val.index
        )
        if X_val is not None
        else None
    )
    return X_train_scaled, X_test_scaled, X_val_scaled, scaler


def prepare_data(
    df: pd.DataFrame,
    config_path: str,
    model_name: str,
    use_extended_features: bool = False,
    cv: bool = False,
):
    """
    Full data preparation pipeline: split, feature engineering, energy encoding,
    and scaling.

    Args:
        df: Raw dataset
        config_path: Path to YAML config
        model_name: Model name
        use_extended_features: Apply extended feature engineering if True
        cv: Skip test prep if True (cross-validation mode)

    Returns:
        X_train, X_test, y_train, y_test, X_val, y_val, scaler, meta
    """
    # --- Load config ---
    features, target, split_cfg, scaling_cfg = load_features_config(
        config_path, model_name
    )

    # --- Select and clean columns ---
    X, y = select_and_clean(
        df, features, target, extended_fe=use_extended_features
    )

    # --- Split data ---
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

    # --- Extended feature engineering ---
    if use_extended_features:
        # Train/val features only; target not included in FE
        X_train, X_val, meta = prepare_features_train_val(X_train, X_val)

        X_train = feature_expansion(X_train)
        if X_val is not None:
            X_val = feature_expansion(X_val)

        # Test features
        if not cv and X_test is not None:
            X_test = prepare_features_test(X_test, meta)
            X_test = feature_expansion(X_test)

    # --- Energy label encoding for non-extended pipeline ---
    if not cv and not use_extended_features:
        X_train, X_test, X_val, energy_enc = (
            encode_energy_labels_train_test_val(X_train, X_test, X_val)
        )
        fe_encoders["energy_label"] = energy_enc

    # --- Ensure numeric columns are float for XGBoost ---
    numeric_cols = meta.get("numeric_features", [])
    log_cols = meta.get("log_cols", [])
    for col in numeric_cols + log_cols:
        if col in X_train.columns:
            X_train[col] = X_train[col].astype(float)
        if X_val is not None and col in X_val.columns:
            X_val[col] = X_val[col].astype(float)
        if X_test is not None and col in X_test.columns:
            X_test[col] = X_test[col].astype(float)

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
