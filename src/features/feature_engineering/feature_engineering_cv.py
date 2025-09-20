import pandas as pd
import numpy as np
from sklearn.preprocessing import OrdinalEncoder
from .encoding import encode_train_val_only
from .utils import extract_floor, extract_lease_years, to_float


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
def prepare_fold_features(
    X_train: pd.DataFrame,
    X_val: pd.DataFrame = None,
    use_extended_features: bool = True,
):
    """
    Perform fold-specific, leakage-safe feature engineering for CV.

    Operations include:
    - Filling numeric NaNs with training medians.
    - Converting binary flags to integers and filling missing with 0.
    - Optionally generating extended features (floor level, lease years,
      backyard size, balcony flag).
    - One-hot encoding selected categorical variables.
    - Ordinal-encoding the `energy_label` column.

    Parameters
    ----------
    X_train : pandas.DataFrame
        Training dataframe for the current fold.
    X_val : pandas.DataFrame, optional
        Validation dataframe for the current fold. Default is None.
    use_extended_features : bool, default=True
        If False, skip generating extended numeric and categorical features.

    Returns
    -------
    X_train_final : pandas.DataFrame
        Transformed training set with numeric, binary, extended, and encoded features.
    X_val_final : pandas.DataFrame or None
        Transformed validation set (if provided), otherwise None.
    meta : dict
        Metadata including:
        - "numeric_cols": List of numeric columns filled with medians.
        - "train_medians": Dict of training medians.
        - "binary_flags": List of binary flag columns.
        - "extra_numeric_cols": List of engineered numeric columns.
        - "ohe_columns": One-hot encoded feature names.
        - "encoder_energy": The fitted energy label encoder.
    fold_encoders : dict
        Dictionary of fitted encoders used in this fold (e.g., for energy label).
    """
    X_train = X_train.copy()
    X_val = X_val.copy() if X_val is not None else None

    meta = {}
    fold_encoders = {}

    # ---------------- Numeric columns ----------------
    numeric_cols = [
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
    train_medians = X_train[numeric_cols].median()
    X_train[numeric_cols] = X_train[numeric_cols].fillna(train_medians)
    if X_val is not None:
        X_val[numeric_cols] = X_val[numeric_cols].fillna(train_medians)

    meta["numeric_cols"] = numeric_cols
    meta["train_medians"] = train_medians.to_dict()

    # ---------------- Binary flags ----------------
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
    X_train[binary_flags] = X_train[binary_flags].fillna(0).astype(int)
    if X_val is not None:
        X_val[binary_flags] = X_val[binary_flags].fillna(0).astype(int)

    meta["binary_flags"] = binary_flags

    # ---------------- Extended features ----------------
    if use_extended_features:
        for df in [X_train] + ([X_val] if X_val is not None else []):
            df["floor_level"] = df["located_on"].apply(extract_floor)
            df["lease_years_remaining"] = (
                df["ownership_type"].apply(extract_lease_years).fillna(0)
            )
            df["backyard_num"] = df["backyard"].apply(to_float).fillna(0)
            df["balcony_flag"] = df["balcony"].apply(
                lambda x: 0 if pd.isna(x) or x == "N/A" else 1
            )

        cat_cols = [
            "postal_code_clean",
            "status",
            "roof_type",
            "ownership_type",
            "location",
            "garden",
        ]
        ohe_train_list, ohe_val_list = [], []
        for col in cat_cols:
            ohe_train = pd.get_dummies(
                X_train[col].fillna("N/A"), prefix=col, drop_first=True
            )
            ohe_train_list.append(ohe_train)
            if X_val is not None:
                ohe_val = pd.get_dummies(
                    X_val[col].fillna("N/A"), prefix=col, drop_first=True
                )
                for c in ohe_train.columns:
                    if c not in ohe_val:
                        ohe_val[c] = 0
                ohe_val = ohe_val[ohe_train.columns]
                ohe_val_list.append(ohe_val)

        ohe_train_concat = pd.concat(ohe_train_list, axis=1)
        ohe_val_concat = (
            pd.concat(ohe_val_list, axis=1) if X_val is not None else None
        )
        extra_numeric_cols = [
            "floor_level",
            "lease_years_remaining",
            "backyard_num",
            "balcony_flag",
        ]
    else:
        ohe_train_concat = pd.DataFrame(index=X_train.index)
        ohe_val_concat = (
            pd.DataFrame(index=X_val.index) if X_val is not None else None
        )
        extra_numeric_cols = []

    # ---------------- Energy label ----------------
    X_train, X_val, encoder_energy = encode_train_val_only(X_train, X_val)
    fold_encoders["energy_label"] = encoder_energy

    # ---------------- Final dataset ----------------
    X_train_final = pd.concat(
        [
            X_train[numeric_cols + binary_flags + extra_numeric_cols],
            ohe_train_concat,
        ],
        axis=1,
    )
    X_val_final = (
        pd.concat(
            [
                X_val[numeric_cols + binary_flags + extra_numeric_cols],
                ohe_val_concat,
            ],
            axis=1,
        )
        if X_val is not None
        else None
    )

    meta["extra_numeric_cols"] = extra_numeric_cols
    meta["ohe_columns"] = ohe_train_concat.columns.tolist()
    meta["encoder_energy"] = encoder_energy

    return X_train_final, X_val_final, meta, fold_encoders
