import pandas as pd
from .utils import (
    to_float,
    extract_floor,
    extract_lease_years,
    drop_low_variance_dummies,
    auto_log_transform_train,
    apply_log_transform,
    simplify_roof,
    simplify_ownership,
    simplify_location,
)
from .encoding import encode_energy_label
from src.features.feature_engineering.feature_expansion import (
    feature_expansion,
)


# ------------------- Luxury Amenities -------------------
LUXURY_AMENITIES = [
    "has_lift",
    "has_sauna",
    "has_domotica",
    "has_airconditioning",
    "has_zwembad",
]

LUXURY_AMENITIES_WEIGHTS = {
    "has_lift": 1,
    "has_sauna": 2,
    "has_domotica": 1.5,
    "has_airconditioning": 1,
    "has_zwembad": 3,
}


def add_luxury_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["luxury_score"] = sum(
        df[col] * w for col, w in LUXURY_AMENITIES_WEIGHTS.items()
    )
    df["num_luxury_facilities"] = df[LUXURY_AMENITIES].sum(axis=1)
    df["has_high_end_facilities"] = (df["num_luxury_facilities"] >= 3).astype(
        int
    )
    df["luxury_density"] = df["luxury_score"] / df["nr_rooms"].replace(0, 1)
    df["size_per_luxury"] = df["size_num"] / (df["luxury_score"] + 1)
    return df


def add_luxury_interactions(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "luxury_score" not in df.columns:
        raise ValueError(
            "Run add_luxury_features before add_luxury_interactions"
        )

    df["luxury_x_price_m2"] = (
        df["luxury_score"] * df["price_per_m2_neighborhood"]
    )
    df["luxury_x_size"] = df["luxury_score"] * df["size_num"]
    df["luxury_x_inhabitants"] = (
        df["luxury_score"] * df["inhabitants_in_neighborhood"]
    )

    return df


# ------------------- Training / Validation -------------------
def prepare_features_train_val(
    df_train: pd.DataFrame,
    df_val: pd.DataFrame = None,
    numeric_features=None,
    binary_flags=None,
    threshold_skew=0.5,
    encode_energy=True,
    use_geolocation=False,
    geo_meta=None,
    geo_cache_file=None,
    use_amenities=False,
    amenities_df=None,
    amenity_radius_map=None,
    amenity_meta=None,
    fit=True,
    enable_cache_save=False,
):
    df_train = df_train.copy()
    df_val = df_val.copy() if df_val is not None else None

    # ------------------- Numeric -------------------
    if numeric_features is None:
        numeric_features = [
            "size_num",
            "contribution_vve_num",
            "external_storage_num",
            "nr_rooms",
            "bathrooms",
            "toilets",
            "num_facilities",
            "inhabitants_in_neighborhood",
            "families_with_children_pct",
            "price_per_m2_neighborhood",
        ]

    train_medians = {}
    for col in numeric_features:
        df_train[col] = df_train[col].apply(to_float)
        median_val = df_train[col].median()
        train_medians[col] = median_val
        df_train[col] = df_train[col].fillna(median_val)
        if df_val is not None:
            df_val[col] = df_val[col].apply(to_float).fillna(median_val)

    # ------------------- Log-transform -------------------
    df_train, log_cols = auto_log_transform_train(
        df_train, numeric_features, threshold_skew
    )
    if df_val is not None:
        df_val = apply_log_transform(df_val, log_cols)

    # ------------------- Binary flags -------------------
    if binary_flags is None:
        binary_flags = [
            "has_mechanische_ventilatie",
            "has_tv_kabel",
            "has_lift",
            "has_natuurlijke_ventilatie",
            "has_schuifpui",
            "has_glasvezelkabel",
            "has_frans_balkon",
            "has_buitenzonwering",
            "has_zonnepanelen",
            "has_airconditioning",
            "has_domotica",
            "has_sauna",
            "has_zwembad",
        ]

    for col in binary_flags:
        df_train[col] = df_train[col].fillna(0).astype(int)
        if df_val is not None:
            df_val[col] = df_val[col].fillna(0).astype(int)

    # ------------------- Extra numeric -------------------
    for df_ in [df_train] + ([df_val] if df_val is not None else []):
        df_["floor_level"] = df_["located_on"].apply(extract_floor)
        df_["lease_years_remaining"] = (
            df_["ownership_type"].apply(extract_lease_years).fillna(0)
        )
        df_["backyard_num"] = df_["backyard"].apply(to_float).fillna(0)
        df_["balcony_flag"] = df_["balcony"].apply(
            lambda x: 0 if pd.isna(x) or x == "N/A" else 1
        )

    # ------------------- Luxury + Interactions -------------------
    df_train = add_luxury_features(df_train)
    df_train = add_luxury_interactions(df_train)
    if df_val is not None:
        df_val = add_luxury_features(df_val)
        df_val = add_luxury_interactions(df_val)

    # ------------------- Energy label -------------------
    if encode_energy:
        df_train, encoder_energy = encode_energy_label(
            df_train, column="energy_label", fit=True
        )
        if df_val is not None:
            df_val, _ = encode_energy_label(
                df_val,
                column="energy_label",
                encoder=encoder_energy,
                fit=False,
            )
    else:
        encoder_energy = None

    # ------------------- Categoricals -------------------
    if "postal_code_clean" in df_train.columns:
        df_train["postal_district"] = (
            df_train["postal_code_clean"].astype(str).str[:3]
        )
        if df_val is not None:
            df_val["postal_district"] = (
                df_val["postal_code_clean"].astype(str).str[:3]
            )

    cat_cols = {
        "postal_district": df_train["postal_district"],
        "status": df_train["status"].fillna("N/A"),
        "roof_type": df_train["roof_type"].apply(simplify_roof),
        "ownership_type": df_train["ownership_type"].apply(simplify_ownership),
        "location": df_train["location"].apply(simplify_location),
        "garden": df_train["garden"].fillna("None"),
    }

    ohe_train_list, ohe_val_list = [], []
    for col_name, series in cat_cols.items():
        ohe = pd.get_dummies(series, prefix=col_name, drop_first=True)
        ohe_train_list.append(ohe)
        if df_val is not None:
            val_series = df_val[col_name]
            val_ohe = pd.get_dummies(
                val_series, prefix=col_name, drop_first=True
            )
            for c in ohe.columns:
                if c not in val_ohe:
                    val_ohe[c] = 0
            val_ohe = val_ohe[ohe.columns]
            ohe_val_list.append(val_ohe)

    ohe_train_concat = pd.concat(ohe_train_list, axis=1)
    ohe_train_reduced, dropped_cols = drop_low_variance_dummies(
        ohe_train_concat
    )
    ohe_val_reduced = (
        pd.concat(ohe_val_list, axis=1).drop(
            columns=dropped_cols, errors="ignore"
        )
        if df_val is not None
        else None
    )

    # ------------------- Combine features -------------------
    preserve_cols = ["address", "postal_code_clean"] if use_geolocation else []
    model_features = (
        numeric_features
        + log_cols
        + binary_flags
        + [
            "floor_level",
            "lease_years_remaining",
            "backyard_num",
            "balcony_flag",
            "energy_label_encoded",
            # Luxury
            "luxury_score",
            "num_luxury_facilities",
            "has_high_end_facilities",
            "luxury_density",
            "size_per_luxury",
            # Interactions
            "luxury_x_price_m2",
            "luxury_x_size",
            "luxury_x_inhabitants",
        ]
    )

    X_train = pd.concat(
        [df_train[model_features + preserve_cols], ohe_train_reduced], axis=1
    )
    X_val = (
        pd.concat(
            [df_val[model_features + preserve_cols], ohe_val_reduced], axis=1
        )
        if df_val is not None
        else None
    )

    # ------------------- Feature expansion -------------------
    pre_exp_cols = set(X_train.columns)
    X_train, geo_meta_out, amenity_meta_out = feature_expansion(
        X_train,
        use_geolocation=use_geolocation,
        geo_meta=geo_meta,
        geo_cache_file=geo_cache_file,
        use_amenities=use_amenities,
        amenities_df=amenities_df,
        amenity_radius_map=amenity_radius_map,
        amenity_meta=amenity_meta,
        fit=fit,
        enable_cache_save=enable_cache_save,
    )
    if X_val is not None:
        X_val, _, _ = feature_expansion(
            X_val,
            use_geolocation=use_geolocation,
            geo_meta=geo_meta_out,
            geo_cache_file=geo_cache_file,
            use_amenities=use_amenities,
            amenities_df=amenities_df,
            amenity_radius_map=amenity_radius_map,
            amenity_meta=amenity_meta_out,
            fit=False,
            enable_cache_save=enable_cache_save,
        )
    expanded_features = list(set(X_train.columns) - pre_exp_cols)

    # Ensure numeric/log columns are float
    for col in numeric_features + log_cols:
        X_train[col] = X_train[col].astype(float)
        if X_val is not None:
            X_val[col] = X_val[col].astype(float)

    meta = {
        "log_cols": log_cols,
        "encoder_energy": encoder_energy,
        "ohe_columns": ohe_train_reduced.columns.tolist(),
        "dropped_ohe_columns": dropped_cols,
        "numeric_features": numeric_features,
        "binary_flags": binary_flags,
        "train_medians": train_medians,
        "expanded_features": expanded_features,
        "geo_meta": geo_meta_out,
        "amenity_meta": amenity_meta_out,
        "geo_cache_file": geo_cache_file,
    }

    return X_train, X_val, meta


# ------------------- Test -------------------
def prepare_features_test(
    df_test: pd.DataFrame,
    meta: dict,
    use_geolocation: bool = False,
    use_amenities: bool = False,
    amenities_df: pd.DataFrame = None,
    amenity_radius_map: dict = None,
    geo_cache_file: str = None,
    enable_cache_save: bool = False,
):
    """
    Prepare test features for modeling, including numeric/log transforms, 
    binary flags, luxury features, energy label encoding, categorical OHE, 
    and optional feature expansion with geolocation or amenities.

    Args:
        df_test (pd.DataFrame): Raw test dataframe.
        meta (dict): Metadata from training/validation pipeline including 
                    medians, log columns, OHE columns, 
                    numeric/binary features, and geolocation/amenity metadata.
        use_geolocation (bool): If True, compute distance-based features.
        use_amenities (bool): If True, compute proximity to amenities features.
        amenities_df (pd.DataFrame): Optional amenities dataset for test set.
        amenity_radius_map (dict): Optional radius mapping 
        for amenity features.

    Returns:
        pd.DataFrame: Transformed test dataframe ready for modeling.
    """
    df_test = df_test.copy()

    # ------------------- Numeric -------------------
    for col in meta["numeric_features"]:
        df_test[col] = (
            df_test[col]
            .apply(to_float)
            .fillna(meta["train_medians"].get(col, 0))
        )

    # ------------------- Log-transform -------------------
    df_test = apply_log_transform(df_test, meta["log_cols"])

    # ------------------- Binary flags -------------------
    for col in meta["binary_flags"]:
        df_test[col] = df_test[col].fillna(0).astype(int)

    # ------------------- Extra numeric -------------------
    df_test["floor_level"] = df_test["located_on"].apply(extract_floor)
    df_test["lease_years_remaining"] = (
        df_test["ownership_type"].apply(extract_lease_years).fillna(0)
    )
    df_test["backyard_num"] = df_test["backyard"].apply(to_float).fillna(0)
    df_test["balcony_flag"] = df_test["balcony"].apply(
        lambda x: 0 if pd.isna(x) or x == "N/A" else 1
    )

    # ------------------- Luxury + Interactions -------------------
    df_test = add_luxury_features(df_test)
    df_test = add_luxury_interactions(df_test)

    # ------------------- Energy label -------------------
    if meta.get("encoder_energy") is not None:
        df_test, _ = encode_energy_label(
            df_test,
            column="energy_label",
            encoder=meta["encoder_energy"],
            fit=False,
        )

    # ------------------- Categoricals -------------------
    cat_cols = {
        "postal_district": df_test["postal_code_clean"].astype(str).str[:3],
        "status": df_test["status"].fillna("N/A"),
        "roof_type": df_test["roof_type"].apply(simplify_roof),
        "ownership_type": df_test["ownership_type"].apply(simplify_ownership),
        "location": df_test["location"].apply(simplify_location),
        "garden": df_test["garden"].fillna("None"),
    }

    ohe_list = []
    for col_name, series in cat_cols.items():
        ohe = pd.get_dummies(series, prefix=col_name, drop_first=True)
        train_cols = [c for c in meta["ohe_columns"] if c.startswith(col_name)]
        for c in train_cols:
            if c not in ohe:
                ohe[c] = 0
        ohe = ohe[train_cols]
        ohe_list.append(ohe)

    ohe_concat = pd.concat(ohe_list, axis=1)

    # ------------------- Preserve columns for geolocation -------------------
    preserve_cols = ["address", "postal_code_clean"] if use_geolocation else []
    for col in preserve_cols:
        if col not in df_test.columns:
            raise ValueError(
                f"Column '{col}' is required for geolocation but "
                "missing in test dataframe."
            )

    # ------------------- Combine features -------------------
    model_features = (
        meta["numeric_features"]
        + meta["log_cols"]
        + meta["binary_flags"]
        + [
            "floor_level",
            "lease_years_remaining",
            "backyard_num",
            "balcony_flag",
            "energy_label_encoded",
            # Luxury
            "luxury_score",
            "num_luxury_facilities",
            "has_high_end_facilities",
            "luxury_density",
            "size_per_luxury",
            # Interactions
            "luxury_x_price_m2",
            "luxury_x_size",
            "luxury_x_inhabitants",
        ]
    )

    X_test_transformed = pd.concat(
        [df_test[model_features + preserve_cols], ohe_concat], axis=1
    )

    # ------------------- Feature expansion -------------------
    X_test_transformed, _, _ = feature_expansion(
        X_test_transformed,
        use_geolocation=use_geolocation,
        geo_meta=meta.get("geo_meta"),
        geo_cache_file=geo_cache_file,
        use_amenities=use_amenities,
        amenities_df=amenities_df,
        amenity_radius_map=amenity_radius_map,
        amenity_meta=meta.get("amenity_meta"),
        fit=False,
        enable_cache_save=enable_cache_save,
    )

    # Ensure numeric/log columns are float
    for col in meta["numeric_features"] + meta["log_cols"]:
        X_test_transformed[col] = X_test_transformed[col].astype(float)

    return X_test_transformed
