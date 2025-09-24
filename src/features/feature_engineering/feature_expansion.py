import numpy as np
import pandas as pd


def feature_expansion(df: pd.DataFrame) -> pd.DataFrame:
    """
    Expand features with safe row-wise ratios and derived metrics.
    Ensures consistent engineered columns across folds for CV stability.

    Args:
        df: DataFrame after base preprocessing & encoding.

    Returns:
        df: Expanded DataFrame with engineered features added.
    """
    df = df.copy()
    current_year = 2025

    # -------------------
    # Size and Rooms
    # -------------------
    df["size_per_room"] = df["size_num"] / df["nr_rooms"].replace(0, np.nan)
    df["bathroom_per_room"] = df["bathrooms"] / df["nr_rooms"].replace(
        0, np.nan
    )
    df["toilet_per_room"] = df["toilets"] / df["nr_rooms"].replace(0, np.nan)

    # -------------------
    # Space Utilization
    # -------------------
    if "floor_level" in df.columns:
        df["floor_to_size"] = df["floor_level"] / df["size_num"].replace(
            0, np.nan
        )
    else:
        df["floor_to_size"] = 0

    df["external_storage_ratio"] = df["external_storage_num"] / df[
        "size_num"
    ].replace(0, np.nan)

    # -------------------
    # Neighborhood Features
    # -------------------
    if "inhabitants_in_neighborhood" in df.columns:
        df["inhabitants_per_room"] = df["inhabitants_in_neighborhood"] / df[
            "nr_rooms"
        ].replace(0, np.nan)

        df["neighborhood_facility_ratio"] = df["num_facilities"] / df[
            "inhabitants_in_neighborhood"
        ].replace(0, np.nan)
    else:
        df["inhabitants_per_room"] = 0
        df["neighborhood_facility_ratio"] = 0

    # -------------------
    # Building Age / Lease
    # -------------------
    if "year_of_construction" in df.columns:
        df["building_age"] = current_year - df["year_of_construction"].fillna(
            current_year
        )
        df["old_house_flag"] = (df["building_age"] > 100).astype(int)
    else:
        df["building_age"] = 0
        df["old_house_flag"] = 0

    if "lease_years_remaining" in df.columns:
        df["lease_ratio"] = df["lease_years_remaining"] / df[
            "building_age"
        ].replace(0, np.nan)
    else:
        df["lease_ratio"] = 0

    # -------------------
    # Balcony/Backyard Interactions
    # -------------------
    if "balcony_flag" in df.columns and "backyard_num" in df.columns:
        df["balcony_or_backyard_flag"] = (
            (df["balcony_flag"] > 0) | (df["backyard_num"] > 0)
        ).astype(int)

        df["balcony_to_size"] = df["balcony_flag"] / df["size_num"].replace(
            0, np.nan
        )
        df["backyard_to_size"] = df["backyard_num"] / df["size_num"].replace(
            0, np.nan
        )
    else:
        df["balcony_or_backyard_flag"] = 0
        df["balcony_to_size"] = 0
        df["backyard_to_size"] = 0

    # -------------------
    # Cleanup
    # -------------------
    df = df.replace([np.inf, -np.inf], np.nan).fillna(0)

    # -------------------
    # Column Consistency Guarantee
    # -------------------
    expected_features = [
        "size_per_room",
        "bathroom_per_room",
        "toilet_per_room",
        "floor_to_size",
        "external_storage_ratio",
        "inhabitants_per_room",
        "neighborhood_facility_ratio",
        "building_age",
        "old_house_flag",
        "lease_ratio",
        "balcony_or_backyard_flag",
        "balcony_to_size",
        "backyard_to_size",
    ]

    for col in expected_features:
        if col not in df.columns:
            df[col] = 0  # fill missing engineered features with 0

    return df
