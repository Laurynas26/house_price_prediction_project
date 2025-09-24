import pandas as pd
import numpy as np
from sklearn.preprocessing import OrdinalEncoder
from .encoding import encode_train_val_only
from .utils import extract_floor, extract_lease_years, to_float
from src.features.feature_engineering.feature_expansion import (
    feature_expansion,
)


# --------------------------
# Energy Label Encoding
# --------------------------
def encode_energy_label(
    X, column: str = "energy_label", encoder=None, fit: bool = True
):
    """
    Encode the `energy_label` column into an ordinal numeric scale.

    Values are mapped to a fixed domain-specific order:
    ["G", "F", "E", "D", "C", "B", "A", "A+", "A++", "A+++", "A++++"].

    Missing values, "N/A", or 0 are replaced with "G" before encoding.

    Parameters
    ----------
    X : pandas.DataFrame
        Input dataframe containing the energy label column.
    column : str, default="energy_label"
        Column to encode.
    encoder : sklearn.preprocessing.OrdinalEncoder, optional
        Pre-fitted encoder. If None, a new encoder is created.
    fit : bool, default=True
        If True, fit the encoder on `X[column]`.
        If False, only transform with an existing encoder.

    Returns
    -------
    X : pandas.DataFrame
        DataFrame with `energy_label_encoded` column added and the original
        `energy_label` column dropped.
    encoder : sklearn.preprocessing.OrdinalEncoder
        The fitted encoder.
    """
    X = X.copy()

    # Replace missing or invalid values with "G"
    X[column] = X[column].replace({0: "G"}).replace("N/A", "G")
    X[column] = X[column].fillna("G")

    energy_order = [
        "G",
        "F",
        "E",
        "D",
        "C",
        "B",
        "A",
        "A+",
        "A++",
        "A+++",
        "A++++",
    ]

    if encoder is None:
        encoder = OrdinalEncoder(categories=[energy_order])

    if fit:
        X["energy_label_encoded"] = encoder.fit_transform(X[[column]])
    else:
        X["energy_label_encoded"] = encoder.transform(X[[column]])

    X = X.drop(columns=[column])
    return X, encoder


def encode_train_val_only(X_train: pd.DataFrame, X_val: pd.DataFrame):
    """
    Encode the `energy_label` feature for training and validation sets.

    The encoder is fitted on the training data and then applied to the
    validation set to ensure leakage-safe encoding.

    Parameters
    ----------
    X_train : pandas.DataFrame
        Training dataframe containing `energy_label`.
    X_val : pandas.DataFrame
        Validation dataframe containing `energy_label`.

    Returns
    -------
    X_train_enc : pandas.DataFrame
        Training set with encoded energy labels.
    X_val_enc : pandas.DataFrame
        Validation set with encoded energy labels.
    encoder : sklearn.preprocessing.OrdinalEncoder
        Encoder fitted on the training data.
    """
    X_train_enc, encoder = encode_energy_label(X_train, fit=True)
    X_val_enc, _ = encode_energy_label(X_val, encoder=encoder, fit=False)
    return X_train_enc, X_val_enc, encoder


# --------------------------
# Fold-wise Feature Engineering
# --------------------------
import pandas as pd
import numpy as np
from sklearn.preprocessing import OrdinalEncoder
from .encoding import encode_train_val_only
from .utils import extract_floor, extract_lease_years, to_float
from src.features.feature_engineering.feature_expansion import (
    feature_expansion,
)
from .utils import (
    drop_low_variance_dummies,
    auto_log_transform_train,
    apply_log_transform,
    simplify_roof,
    simplify_ownership,
    simplify_location,
    to_float,
    extract_floor,
    extract_lease_years,
)
from .encoding import encode_train_val_only
from src.features.feature_engineering.feature_expansion import (
    feature_expansion,
)


def prepare_fold_features(
    X_train: pd.DataFrame,
    X_val: pd.DataFrame = None,
    numeric_features=None,
    binary_flags=None,
    threshold_skew=0.5,
    encode_energy=True,
    use_extended_features=True,
):
    """
    Fold-wise feature prep (CV) mimicking the non-CV pipeline.
    - Numeric: fillna, optional log-transform
    - Binary flags
    - Extra numeric features
    - Energy label encoding
    - Categorical simplification + OHE
    - Low-variance dummy drop
    - Optional safe feature expansion

    Parameters
    ----------
    X_train : pd.DataFrame
        Training dataframe for the current fold.
    X_val : pd.DataFrame, optional
        Validation dataframe for the current fold. Default is None.
    use_extended_features : bool, default=True
        If False, skip generating extended numeric and categorical features.

    Returns
    -------
    X_train_final : pd.DataFrame
        Transformed training set with numeric, binary, extended, and encoded features.
    X_val_final : pd.DataFrame or None
        Transformed validation set (if provided), otherwise None.
    meta : dict
        Metadata including numeric columns, training medians, binary flags,
        extra numeric columns, OHE columns, energy label encoder, and expanded features.
    fold_encoders : dict
        Dictionary of fitted encoders used in this fold (e.g., for energy label).
    """
    X_train = X_train.copy()
    X_val = X_val.copy() if X_val is not None else None
    meta = {}
    fold_encoders = {}

    # ---------------- Numeric ----------------
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
        X_train[col] = X_train[col].apply(to_float)
        median_val = X_train[col].median()
        train_medians[col] = median_val
        X_train[col] = X_train[col].fillna(median_val)
        if X_val is not None:
            X_val[col] = X_val[col].apply(to_float).fillna(median_val)

    # Log-transform
    X_train, log_cols = auto_log_transform_train(
        X_train, numeric_features, threshold_skew
    )
    if X_val is not None:
        X_val = apply_log_transform(X_val, log_cols)

    meta["numeric_features"] = numeric_features
    meta["train_medians"] = train_medians
    meta["log_cols"] = log_cols

    # ---------------- Binary Flags ----------------
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
        X_train[col] = X_train[col].fillna(0).astype(int)
        if X_val is not None:
            X_val[col] = X_val[col].fillna(0).astype(int)
    meta["binary_flags"] = binary_flags

    # ---------------- Extra Numeric Features ----------------
    for df in [X_train] + ([X_val] if X_val is not None else []):
        df["floor_level"] = df["located_on"].apply(extract_floor)
        df["lease_years_remaining"] = (
            df["ownership_type"].apply(extract_lease_years).fillna(0)
        )
        df["backyard_num"] = df["backyard"].apply(to_float).fillna(0)
        df["balcony_flag"] = df["balcony"].apply(
            lambda x: 0 if pd.isna(x) or x == "N/A" else 1
        )

    extra_numeric_cols = [
        "floor_level",
        "lease_years_remaining",
        "backyard_num",
        "balcony_flag",
    ]

    # ---------------- Energy Label ----------------
    if encode_energy:
        X_train, X_val, encoder_energy = encode_train_val_only(X_train, X_val)
        fold_encoders["energy_label"] = encoder_energy
    else:
        encoder_energy = None

    # ---------------- Categorical Features ----------------
    # Simplify high-cardinality columns
    X_train["postal_district"] = (
        X_train["postal_code_clean"].astype(str).str[:3]
    )
    if X_val is not None:
        X_val["postal_district"] = (
            X_val["postal_code_clean"].astype(str).str[:3]
        )

    cat_cols = {
        "postal_district": X_train["postal_district"],
        "status": X_train["status"].fillna("N/A"),
        "roof_type": X_train["roof_type"].apply(simplify_roof),
        "ownership_type": X_train["ownership_type"].apply(simplify_ownership),
        "location": X_train["location"].apply(simplify_location),
        "garden": X_train["garden"].fillna("None"),
    }

    ohe_train_list, ohe_val_list = [], []
    for col_name, series in cat_cols.items():
        ohe = pd.get_dummies(series, prefix=col_name, drop_first=True)
        ohe_train_list.append(ohe)
        if X_val is not None:
            val_series = X_val[col_name]
            val_ohe = pd.get_dummies(
                val_series, prefix=col_name, drop_first=True
            )
            # Align with training
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
        if X_val is not None
        else None
    )
    meta["ohe_columns"] = ohe_train_reduced.columns.tolist()
    meta["dropped_ohe_columns"] = dropped_cols

    # ---------------- Combine Features ----------------
    model_features = (
        numeric_features
        + log_cols
        + binary_flags
        + extra_numeric_cols
        + ["energy_label_encoded"]
    )
    X_train_final = pd.concat(
        [X_train[model_features], ohe_train_reduced], axis=1
    )
    X_val_final = (
        pd.concat([X_val[model_features], ohe_val_reduced], axis=1)
        if X_val is not None
        else None
    )

    # ---------------- Feature Expansion (optional) ----------------
    if use_extended_features:
        pre_exp_cols = set(X_train_final.columns)
        X_train_final = feature_expansion(X_train_final)
        if X_val_final is not None:
            X_val_final = feature_expansion(X_val_final)
        meta["expanded_features"] = list(
            set(X_train_final.columns) - pre_exp_cols
        )
    else:
        meta["expanded_features"] = []

    meta["extra_numeric_cols"] = extra_numeric_cols
    meta["encoder_energy"] = encoder_energy

    return X_train_final, X_val_final, meta, fold_encoders
