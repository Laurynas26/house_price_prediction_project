import yaml
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from src.features.feature_engineering.feature_engineering import (
    prepare_features_train_val,
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
    df, config_path, model_name, use_extended_features=False, cv=False
):
    """
    Prepare data for modeling with optional extended feature engineering and energy_label encoding.

    Args:
        df (pd.DataFrame): Input dataset.
        config_path (str/Path): Path to YAML config for features and model.
        model_name (str): Key in YAML specifying which model's config to load.
        use_extended_features (bool): Whether to apply extended feature engineering.
        cv (bool): If True, skip global energy_label encoding (will be handled fold-wise in CV).
                   If False, encode energy_label for all splits.

    Returns:
        X_train (pd.DataFrame): Training features.
        X_test (pd.DataFrame): Test features.
        y_train (pd.Series): Training target.
        y_test (pd.Series): Test target.
        scaler (sklearn Scaler or None): Fitted scaler object.
        X_val (pd.DataFrame or None): Validation features (if configured).
        y_val (pd.Series or None): Validation target (if configured).
        fe_encoders (dict): Dictionary of fitted feature encoders (includes energy_label if cv=False).
    """
    # Load features, target, and config
    features, target, split_cfg, scaling_cfg = load_features_config(
        config_path, model_name
    )
    X, y = select_and_clean(df, features, target)

    # Split train/test (+ optional val)
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

    # Optional extended feature engineering
    if use_extended_features:
        if X_val is not None:
            X_train, X_val, X_test, fe_encoders = prepare_features_train_val(
                X_train, X_val, X_test
            )
        else:
            X_train, _, X_test, fe_encoders = prepare_features_train_val(
                X_train, None, X_test
            )
    else:
        fe_encoders = {}

    # Encode energy_label globally only if not doing CV
    if not cv and not use_extended_features:
        X_train, X_test, X_val, energy_enc = (
            encode_energy_labels_train_test_val(X_train, X_test, X_val)
        )
        fe_encoders["energy_label"] = energy_enc

    # Scale features if required
    if scaling_cfg.get("scale", True):
        scaler_cls = SCALERS.get(
            scaling_cfg.get("method", "StandardScaler"), StandardScaler
        )
        X_train, X_test, X_val, scaler = scale_data(
            X_train, X_test, X_val, scaler_cls
        )
    else:
        scaler = None

    return X_train, X_test, y_train, y_test, scaler, X_val, y_val, fe_encoders
