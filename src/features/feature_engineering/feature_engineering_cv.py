import pandas as pd
import numpy as np
from sklearn.preprocessing import OrdinalEncoder
from .encoding import encode_train_val_only
from .utils import extract_floor, extract_lease_years


# --------------------------
# Energy Label Encoding
# --------------------------
def encode_energy_label(X, column="energy_label", encoder=None, fit=True):
    """
    Encode the energy label column with a fixed order using OrdinalEncoder.
    Handles NaNs safely by replacing with "G".
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


def encode_train_val_only(X_train, X_val):
    """
    Fit energy label encoder on X_train and safely transform X_val.
    """
    X_train_enc, encoder = encode_energy_label(X_train, fit=True)
    X_val_enc, _ = encode_energy_label(X_val, encoder=encoder, fit=False)
    return X_train_enc, X_val_enc, encoder


# --------------------------
# Fold-wise Feature Engineering
# --------------------------
import pandas as pd
import numpy as np
from .utils import to_float, extract_floor, extract_lease_years
from .encoding import encode_energy_label


def prepare_fold_features(
    X_train: pd.DataFrame,
    X_val: pd.DataFrame = None,
    use_extended_features: bool = True,
):
    """
    CV-safe, fold-wise feature engineering.

    Args:
        X_train: Training features for the fold.
        X_val: Validation features for the fold (optional).
        use_extended_features: If False, skip all extended features.

    Returns:
        X_train_final, X_val_final, meta, fold_encoders
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
        # Extra numeric
        for df in [X_train] + ([X_val] if X_val is not None else []):
            df["floor_level"] = df["located_on"].apply(extract_floor)
            df["lease_years_remaining"] = (
                df["ownership_type"].apply(extract_lease_years).fillna(0)
            )
            df["backyard_num"] = df["backyard"].apply(to_float).fillna(0)
            df["balcony_flag"] = df["balcony"].apply(
                lambda x: 0 if pd.isna(x) or x == "N/A" else 1
            )

        # Categorical
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
