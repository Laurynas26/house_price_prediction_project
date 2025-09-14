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
    Prepare train/val features with safe transformations for CV or Optuna.

    Returns:
        X_train, X_val, y_train, y_val, meta
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
            "living_area",
            "nr_rooms",
            "bathrooms",
            "toilets",
            "num_facilities",
            "inhabitants_in_neighborhood",
            "families_with_children_pct",
            "price_per_m2_neighborhood",
        ]

    for col in numeric_features:
        df_train[col] = df_train[col].apply(to_float).fillna(df_train[col].median())
        if df_val is not None:
            df_val[col] = df_val[col].apply(to_float).fillna(df_train[col].median())

    # Log-transform on TRAIN only
    df_train, log_cols = auto_log_transform_train(df_train, numeric_features, threshold_skew)
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
        df["lease_years_remaining"] = df["ownership_type"].apply(extract_lease_years).fillna(0)
        df["backyard_num"] = df["backyard"].apply(to_float).fillna(0)
        df["balcony_flag"] = df["balcony"].apply(lambda x: 0 if pd.isna(x) or x == "N/A" else 1)

    # -------------------
    # Energy label
    # -------------------
    if encode_energy:
        df_train, encoder_energy = encode_energy_label(df_train, column="energy_label", fit=True)
        if df_val is not None:
            df_val, _ = encode_energy_label(df_val, column="energy_label", encoder=encoder_energy, fit=False)

    # -------------------
    # Categorical OHE
    # -------------------
    cat_cols = {
        "postal_district": df_train["postal_code_clean"].str[:3],
        "status": df_train["status"].fillna("N/A"),
        "roof_type": df_train["roof_type"].apply(simplify_roof),
        "ownership_type": df_train["ownership_type"].apply(simplify_ownership),
        "location": df_train["location"].apply(simplify_location),
        "garden": df_train["garden"].fillna("None"),
    }

    ohe_train_list = []
    ohe_val_list = []
    ohe_columns_list = []

    for col_name, series in cat_cols.items():
        ohe = pd.get_dummies(series, prefix=col_name, drop_first=True)
        ohe_train_list.append(ohe)
        ohe_columns_list.extend(ohe.columns.tolist())

        if df_val is not None:
            val_series = df_val[col_name]
            val_ohe = pd.get_dummies(val_series, prefix=col_name, drop_first=True)
            # Align columns
            for c in ohe.columns:
                if c not in val_ohe:
                    val_ohe[c] = 0
            val_ohe = val_ohe[ohe.columns]
            ohe_val_list.append(val_ohe)

    # Concatenate OHE and drop low-variance based on TRAIN only
    ohe_train_concat = pd.concat(ohe_train_list, axis=1)
    ohe_train_reduced, dropped_cols = drop_low_variance_dummies(ohe_train_concat)

    if df_val is not None:
        ohe_val_concat = pd.concat(ohe_val_list, axis=1)
        ohe_val_reduced = ohe_val_concat.drop(columns=dropped_cols, errors="ignore")
    else:
        ohe_val_reduced = None

    # -------------------
    # Combine all features
    # -------------------
    model_features = numeric_features + log_cols + binary_flags + [
        "floor_level",
        "lease_years_remaining",
        "backyard_num",
        "balcony_flag",
        "energy_label_encoded",
    ]

    X_train = pd.concat([df_train[model_features], ohe_train_reduced], axis=1)
    X_val = pd.concat([df_val[model_features], ohe_val_reduced], axis=1) if df_val is not None else None

    y_train = df_train["price_num"]
    y_val = df_val["price_num"] if df_val is not None else None

    meta = {
        "log_cols": log_cols,
        "encoder_energy": encoder_energy,
        "ohe_columns": ohe_train_reduced.columns.tolist(),
        "dropped_ohe_columns": dropped_cols,
    }

    return X_train, X_val, y_train, y_val, meta
