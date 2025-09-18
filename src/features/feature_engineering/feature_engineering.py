import pandas as pd
import numpy as np
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


def prepare_features_train_val(
    df_train: pd.DataFrame,
    df_val: pd.DataFrame = None,
    numeric_features=None,
    binary_flags=None,
    threshold_skew=0.5,
    encode_energy=True,
):
    """
    Prepare train and optional validation features for modeling.

    Handles:
      - Numeric features (fillna, float conversion)
      - Log-transforms
      - Binary flags
      - Extra numeric features (floor_level, lease_years_remaining, backyard, balcony)
      - Energy label encoding
      - Categorical OHE (aligned train/val)

    Args:
        df_train: Training dataframe (features only, target excluded)
        df_val: Optional validation dataframe
        numeric_features: List of numeric columns
        binary_flags: List of binary flag columns
        threshold_skew: Threshold for log-transform
        encode_energy: Whether to encode energy label

    Returns:
        X_train: Transformed training features
        X_val: Transformed validation features (or None)
        meta: Metadata for transforming test set (medians, encoders, OHE columns)
    """
    df_train = df_train.copy()
    df_val = df_val.copy() if df_val is not None else None

    # -------------------
    # Numeric features
    # -------------------
    if numeric_features is None:
        numeric_features = [
            "size_num",
            "contribution_vve_num",
            "external_storage_num",
            # "living_area",
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

    # -------------------
    # Log-transform
    # -------------------
    df_train, log_cols = auto_log_transform_train(
        df_train, numeric_features, threshold_skew
    )
    if df_val is not None:
        df_val = apply_log_transform(df_val, log_cols)

    # -------------------
    # Binary flags
    # -------------------
    if binary_flags is None:
        binary_flags = [
            "has_mechanische_ventilatie",
            "has_tv_kabel",
            "has_lift",
            "has_natuurlijke_ventilatie",
            "has_n/a",
            "has_schuifpui",
            "has_glasvezelkabel",
            "has_frans_balkon",
            "has_buitenzonwering",
            "has_zonnepanelen",
        ]

    for col in binary_flags:
        df_train[col] = df_train[col].fillna(0).astype(int)
        if df_val is not None:
            df_val[col] = df_val[col].fillna(0).astype(int)

    # -------------------
    # Extra numeric features
    # -------------------
    for df in [df_train] + ([df_val] if df_val is not None else []):
        df["floor_level"] = df["located_on"].apply(extract_floor)
        df["lease_years_remaining"] = (
            df["ownership_type"].apply(extract_lease_years).fillna(0)
        )
        df["backyard_num"] = df["backyard"].apply(to_float).fillna(0)
        df["balcony_flag"] = df["balcony"].apply(
            lambda x: 0 if pd.isna(x) or x == "N/A" else 1
        )

    # -------------------
    # Energy label
    # -------------------
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

    # -------------------
    # Categorical features
    # -------------------
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

    # -------------------
    # Combine features
    # -------------------
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
        ]
    )

    X_train = pd.concat([df_train[model_features], ohe_train_reduced], axis=1)
    X_val = (
        pd.concat([df_val[model_features], ohe_val_reduced], axis=1)
        if df_val is not None
        else None
    )

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
    }

    return X_train, X_val, meta


def prepare_features_test(df_test: pd.DataFrame, meta: dict):
    """
    Apply the same transformations to test set as train/val.

    Aligns numeric, log-transformed, binary, extra numeric, energy label, and OHE features.

    Args:
        df_test: Raw test features (target excluded)
        meta: Metadata from prepare_features_train_val

    Returns:
        X_test_transformed: Fully transformed test features
    """
    df_test = df_test.copy()

    # -------------------
    # Numeric
    # -------------------
    for col in meta["numeric_features"]:
        df_test[col] = (
            df_test[col]
            .apply(to_float)
            .fillna(meta["train_medians"].get(col, 0))
        )

    # Log-transform
    df_test = apply_log_transform(df_test, meta["log_cols"])

    # Binary flags
    for col in meta["binary_flags"]:
        df_test[col] = df_test[col].fillna(0).astype(int)

    # Extra numeric
    df_test["floor_level"] = df_test["located_on"].apply(extract_floor)
    df_test["lease_years_remaining"] = (
        df_test["ownership_type"].apply(extract_lease_years).fillna(0)
    )
    df_test["backyard_num"] = df_test["backyard"].apply(to_float).fillna(0)
    df_test["balcony_flag"] = df_test["balcony"].apply(
        lambda x: 0 if pd.isna(x) or x == "N/A" else 1
    )

    # Energy label
    if meta["encoder_energy"] is not None:
        df_test, _ = encode_energy_label(
            df_test, encoder=meta["encoder_energy"], fit=False
        )

    # Categorical OHE
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
        ]
    )

    X_test_transformed = pd.concat(
        [df_test[model_features], ohe_concat], axis=1
    )

    # Ensure numeric/log columns are float
    for col in meta["numeric_features"] + meta["log_cols"]:
        X_test_transformed[col] = X_test_transformed[col].astype(float)

    return X_test_transformed
