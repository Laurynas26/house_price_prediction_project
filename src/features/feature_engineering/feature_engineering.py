import numpy as np
import pandas as pd
from .utils import (
    to_float,
    extract_floor,
    extract_lease_years,
    drop_low_variance_dummies,
)
from .encoding import encode_energy_label


def auto_log_transform(df, numeric_cols, threshold_skew=0.5):
    """Automatically log-transform skewed positive numeric features."""
    log_cols = []
    for col in numeric_cols:
        if (df[col] > 0).all():
            skewness = df[col].skew()
            if abs(skewness) > threshold_skew:
                df[f"log_{col}"] = np.log1p(df[col])
                log_cols.append(f"log_{col}")
    return log_cols


def simplify_roof(roof):
    if pd.isna(roof) or roof == "N/A":
        return "Unknown"
    if "Plat dak" in roof:
        return "Flat"
    if "Zadeldak" in roof:
        return "Saddle"
    if "Samengesteld dak" in roof:
        return "Composite"
    if "Mansarde" in roof:
        return "Mansard"
    return "Other"


def simplify_ownership(x):
    if pd.isna(x) or x.strip() == "":
        return "Unknown"
    if "Volle eigendom" in x:
        return "Full"
    if "Erfpacht" in x and "Gemeentelijk" in x:
        return "Municipal"
    if "Erfpacht" in x:
        return "Leasehold"
    return "Other"


def simplify_location(x):
    if pd.isna(x):
        return "Unknown"
    if "centrum" in x:
        return "Central"
    if "woonwijk" in x:
        return "Residential"
    if "vrij uitzicht" in x:
        return "OpenView"
    if "park" in x:
        return "Park"
    return "Other"


def prepare_features(df: pd.DataFrame):
    """Full feature engineering pipeline for modeling."""
    df = df.copy()

    # 1. Numeric features
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
        df[col] = df[col].apply(to_float)
        df[col] = df[col].fillna(df[col].median())

    log_cols = auto_log_transform(df, numeric_features)

    # 2. Binary flags
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
        df[col] = df[col].fillna(0).astype(int)

    # 3. Extra numeric features
    df["floor_level"] = df["located_on"].apply(extract_floor)
    df["lease_years_remaining"] = (
        df["ownership_type"].apply(extract_lease_years).fillna(0)
    )
    df["backyard_num"] = df["backyard"].apply(to_float).fillna(0)
    df["balcony_flag"] = df["balcony"].apply(
        lambda x: 0 if pd.isna(x) or x == "N/A" else 1
    )

    # 4. Energy label
    df, encoder_energy = encode_energy_label(
        df, column="energy_label", fit=True
    )

    # 5. Categorical OHE
    postal_ohe = pd.get_dummies(
        df["postal_code_clean"].str[:3], prefix="district", drop_first=True
    )
    status_ohe = pd.get_dummies(
        df["status"].fillna("N/A"), prefix="status", drop_first=True
    )
    roof_ohe = pd.get_dummies(
        df["roof_type"].apply(simplify_roof), prefix="roof", drop_first=True
    )
    ownership_ohe = pd.get_dummies(
        df["ownership_type"].apply(simplify_ownership),
        prefix="ownership",
        drop_first=True,
    )
    location_ohe = pd.get_dummies(
        df["location"].apply(simplify_location),
        prefix="location",
        drop_first=True,
    )
    garden_ohe = pd.get_dummies(
        df["garden"].fillna("None"), prefix="garden", drop_first=True
    )

    ohe_all = pd.concat(
        [
            postal_ohe,
            status_ohe,
            roof_ohe,
            ownership_ohe,
            location_ohe,
            garden_ohe,
        ],
        axis=1,
    )
    ohe_reduced, dropped_cols = drop_low_variance_dummies(ohe_all)

    # 6. Combine features
    model_features = (
        log_cols
        + binary_flags
        + numeric_features
        + [
            "floor_level",
            "lease_years_remaining",
            "backyard_num",
            "balcony_flag",
            "energy_label_encoded",
        ]
    )
    X = pd.concat([df[model_features], ohe_reduced], axis=1)
    y = df["price_num"]

    return (
        X,
        y,
        log_cols,
        {
            "energy_label": encoder_energy,
            "ohe_columns": ohe_reduced.columns.tolist(),
        },
    )
