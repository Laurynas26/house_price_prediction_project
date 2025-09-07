import yaml
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler

SCALERS = {
    "StandardScaler": StandardScaler,
    "MinMaxScaler": MinMaxScaler,
    "RobustScaler": RobustScaler,
}


def load_features_config(config_path, model_name):
    """Load features, target, split, and scaling config for a
    specific model from YAML."""
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    model_cfg = cfg[model_name]
    features = model_cfg["features"]
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


def split_data(X, y, test_size=0.2, random_state=42):
    """Split features and target into train/test sets."""
    return train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )


def scale_data(X_train, X_test, scaler_cls=StandardScaler):
    """Scale numeric features using provided scaler class."""
    scaler = scaler_cls()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled, scaler


def prepare_data(df, config_path, model_name):
    """
    Wrapper function: select features, split, and optionally scale data
    according to YAML config.

    Args:
        df: DataFrame
        config_path: path to YAML config
        model_name: name of the model section in YAML

    Returns:
        X_train, X_test, y_train, y_test, scaler (None if scale=False)
    """
    features, target, split_cfg, scaling_cfg = load_features_config(
        config_path, model_name
    )

    X, y = select_and_clean(df, features, target)

    X_train, X_test, y_train, y_test = split_data(
        X,
        y,
        test_size=split_cfg.get("test_size", 0.2),
        random_state=split_cfg.get("random_state", 42),
    )

    if scaling_cfg.get("scale", True):
        scaler_cls = SCALERS.get(
            scaling_cfg.get("method", "StandardScaler"), StandardScaler
        )
        X_train, X_test, scaler = scale_data(X_train, X_test, scaler_cls)
    else:
        scaler = None

    return X_train, X_test, y_train, y_test, scaler
