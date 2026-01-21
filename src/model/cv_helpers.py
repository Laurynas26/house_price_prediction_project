import pandas as pd
import numpy as np
from src.features.feature_engineering import feature_engineering_cv as fe_cv
from src.features.data_prep_for_modelling.data_preparation import (
    load_features_config,
    load_geo_config,
)


def prepare_base_data(
    df: pd.DataFrame,
    features_config: str,
    model_name: str,
    extended_fe: bool = True,
):
    """
    Prepare base features and target for CV or training.

    Parameters
    ----------
    df : pd.DataFrame
        Input raw dataset.
    features_config : str
        YAML path with features and target definition.
    model_name : str
        Model key used in YAML.
    extended_fe : bool
        If True, include extra columns for extended feature engineering.

    Returns
    -------
    X_full : pd.DataFrame
        Feature dataframe ready for modeling.
    y_full : pd.Series
        Target column.
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
            "address",
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
    features_config: str = None,
    use_extended_features: bool = True,
    enable_cache_save: bool = False,
):
    """
    CV-safe fold-wise feature engineering wrapper.

    Loads geo/amenities config from YAML if extended features are enabled.
    Drops 'address' column after geolocation enrichment to avoid leaking
    location info into the model.

    Returns:
        X_train_fe, X_val_fe, meta, fold_encoders
    """
    if use_extended_features and features_config is not None:
        geo_cache_file, amenities_df, amenity_radius_map = load_geo_config(
            features_config
        )
    else:
        geo_cache_file, amenities_df, amenity_radius_map = None, None, None

    X_train_fe, X_val_fe, meta, fold_encoders = fe_cv.prepare_fold_features(
        X_train,
        X_val,
        use_extended_features=use_extended_features,
        include_distance=True,
        include_amenities=(
            amenities_df is not None and amenity_radius_map is not None
        ),
        amenities_df=amenities_df,
        amenity_radius_map=amenity_radius_map,
        geo_cache_file=geo_cache_file,
        enable_cache_save=enable_cache_save,
    )

    # Drop 'address' if present
    for df in [X_train_fe, X_val_fe]:
        if df is not None and "address" in df.columns:
            df.drop(columns=["address"], inplace=True)

    return X_train_fe, X_val_fe, meta, fold_encoders
