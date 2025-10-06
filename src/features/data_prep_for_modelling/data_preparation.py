import yaml
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional
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


def load_geo_config(config_path: Path):
    config_path = Path(config_path).resolve()
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    cfg = yaml.safe_load(open(config_path))
    geo_cfg = cfg.get("geo_feature_exp", {})

    geo_cache_file_name = geo_cfg.get("geo_cache_file")
    if geo_cache_file_name is None:
        raise ValueError("geo_cache_file not defined in YAML")
    geo_cache_file_path = (config_path.parent / geo_cache_file_name).resolve()
    if not geo_cache_file_path.exists():
        raise FileNotFoundError(
            f"Geo cache file not found: {geo_cache_file_path}"
        )

    amenities_file_name = geo_cfg.get("amenities_file")
    amenities_file_path = (
        (config_path.parent / amenities_file_name).resolve()
        if amenities_file_name
        else None
    )
    amenities_df = (
        pd.read_csv(amenities_file_path)
        if amenities_file_path and amenities_file_path.exists()
        else None
    )

    amenity_radius_map = geo_cfg.get("amenity_radius_map", None)

    return str(geo_cache_file_path), amenities_df, amenity_radius_map


def prepare_data_from_config(
    df, config_path, model_name, geo_cache_file=None, enable_cache_save=False
):
    geo_cache_file_from_config, amenities_df, amenity_radius_map = (
        load_geo_config(config_path)
    )
    if geo_cache_file is None:
        geo_cache_file = geo_cache_file_from_config

    if geo_cache_file is None:
        raise FileNotFoundError(
            "Geo cache file not found! " \
            "Check your YAML config or provide geo_cache_file explicitly."
        )

    return prepare_data(
        df=df,
        config_path=str(config_path),
        model_name=model_name,
        use_extended_features=True,
        include_distance=True,
        include_amenities=(
            amenities_df is not None and amenity_radius_map is not None
        ),
        amenities_df=amenities_df,
        amenity_radius_map=amenity_radius_map,
        geo_cache_file=str(geo_cache_file),
        enable_cache_save=enable_cache_save,
    )


def load_features_config(config_path: str, model_name: str):
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
    df_copy = df.copy()
    extra_cols = (
        [
            "located_on",
            "ownership_type",
            "postal_code_clean",
            "status",
            "roof_type",
            "location",
            "garden",
            "backyard",
            "balcony",
            "facilities",
            "facilities_list",
            "garden_location",
            "num_parcels",
            "parcels_concat",
            "has_zwembad",
            "has_verwarming",
            "has_vliering",
            "has_windmolen",
            "address",
        ]
        if extended_fe
        else []
    )

    cols_to_keep = list(set(features + extra_cols))
    X = df_copy[cols_to_keep].copy()

    numeric_cols = X.select_dtypes(include=[np.number]).columns
    X[numeric_cols] = X[numeric_cols].replace("N/A", np.nan).fillna(0)

    y = df_copy[target]
    return X, y


def split_train_val_test_data(
    X, y, test_size=0.2, val_size=0.1, random_state=42, val_required=False
):
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
    include_distance: bool = False,
    include_amenities: bool = False,
    amenities_df: Optional[pd.DataFrame] = None,
    amenity_radius_map: Optional[dict] = None,
    geo_cache_file: Optional[str] = None,
    cv: bool = False,
    enable_cache_save: bool = False,
):
    features, target, split_cfg, scaling_cfg = load_features_config(
        config_path, model_name
    )
    X, y = select_and_clean(
        df, features, target, extended_fe=use_extended_features
    )

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

    meta, fe_encoders = {}, {}

    if use_extended_features:
        # Train/val feature engineering
        X_train, X_val, meta = prepare_features_train_val(
            X_train,
            X_val,
            use_geolocation=include_distance,
            geo_cache_file=geo_cache_file,
            use_amenities=include_amenities,
            amenities_df=amenities_df,
            amenity_radius_map=amenity_radius_map,
        )

        # Feature expansion
        pre_exp_cols = set(X_train.columns)
        X_train, geo_meta_out, amenity_meta_out = feature_expansion(
            X_train,
            use_geolocation=include_distance,
            geo_meta=meta.get("geo_meta"),
            use_amenities=include_amenities,
            amenities_df=amenities_df,
            amenity_radius_map=amenity_radius_map,
            amenity_meta=meta.get("amenity_meta"),
            fit=True,
            geo_cache_file=geo_cache_file,
            enable_cache_save=enable_cache_save,
        )

        if X_val is not None:
            X_val, _, _ = feature_expansion(
                X_val,
                use_geolocation=include_distance,
                geo_meta=geo_meta_out,
                use_amenities=include_amenities,
                amenities_df=amenities_df,
                amenity_radius_map=amenity_radius_map,
                amenity_meta=amenity_meta_out,
                fit=False,
                geo_cache_file=geo_cache_file,
                enable_cache_save=enable_cache_save,
            )

        meta["expanded_features"] = list(set(X_train.columns) - pre_exp_cols)
        meta["geo_meta"] = geo_meta_out
        meta["amenity_meta"] = amenity_meta_out

        if not cv and X_test is not None:
            test_geo_cache_file = geo_cache_file
            if (
                test_geo_cache_file is None
                and meta.get("geo_cache_file") is not None
            ):
                test_geo_cache_file = meta["geo_cache_file"]

            X_test = prepare_features_test(
                X_test,
                meta,
                use_geolocation=include_distance,
                use_amenities=include_amenities,
                amenities_df=amenities_df,
                amenity_radius_map=amenity_radius_map,
                geo_cache_file=test_geo_cache_file,
            )

        # --- DROP 'address' ONLY AFTER ALL FEATURE ENGINEERING ---
        for df_ in [X_train, X_val, X_test]:
            if df_ is not None and "address" in df_.columns:
                df_.drop(columns="address", inplace=True)

    if not cv and not use_extended_features:
        X_train, X_test, X_val, energy_enc = (
            encode_energy_labels_train_test_val(X_train, X_test, X_val)
        )
        fe_encoders["energy_label"] = energy_enc

    # Ensure numeric/log columns are float
    numeric_cols = meta.get("numeric_features", [])
    log_cols = meta.get("log_cols", [])
    for col in numeric_cols + log_cols:
        for df_ in [X_train, X_val, X_test]:
            if df_ is not None and col in df_.columns:
                df_[col] = df_[col].astype(float)

    # Scaling
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
