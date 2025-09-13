import yaml
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler

SCALERS = {
    "StandardScaler": StandardScaler,
    "MinMaxScaler": MinMaxScaler,
    "RobustScaler": RobustScaler,
}


def load_features_config(config_path, model_name, use_extended_features=False):
    """Load features, target, split, and scaling config for a specific model from YAML."""
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    model_cfg = cfg[model_name]
    features = model_cfg.get("features", [])
    target = model_cfg["target"]

    # Include extended features if flag is True
    if use_extended_features:
        extended = model_cfg.get("extended_features", [])
        features += extended

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
    df, config_path, model_name, use_extended_features=False
):
    """
    Wrapper function to prepare data for a given model:
    - Select features
    - Split into train/test and optional validation
    - Scale numeric features if required
    """
    # Load feature-related config
    features, target, split_cfg, scaling_cfg = load_features_config(
        config_path,
        model_name,
        use_extended_features=use_extended_features,
    )

    # Select columns and handle missing values
    X, y = select_and_clean(df, features, target)

    # Determine validation set
    val_required = split_cfg.get("val_required", False)
    val_size = split_cfg.get("val_size", 0.1)

    # Split data
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

    return X_train, X_test, y_train, y_test, scaler, X_val, y_val
