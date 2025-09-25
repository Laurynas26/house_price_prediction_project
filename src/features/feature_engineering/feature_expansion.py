import numpy as np
import pandas as pd


def feature_expansion(df: pd.DataFrame) -> pd.DataFrame:
    """
    Refactored feature expansion pipeline with grouped features:
    - Row-wise ratios grouped into 'room_utilization_ratio'
    - Balcony/backyard grouped into 'has_outdoor_space' and 'outdoor_area_ratio'
    - Neighborhood-level ratios reduced to key signals
    - Building age / lease features preserved
    """

    df = df.copy()
    current_year = 2025

    # ------------------- Row-wise Ratios -------------------
    # Combine size, bathrooms, toilets per room into single metric
    df["room_utilization_ratio"] = (
        df["size_num"] + df["bathrooms"] + df["toilets"]
    ) / df["nr_rooms"].replace(0, np.nan)

    # Keep floor and external storage ratios
    df["floor_to_size"] = df.get("floor_level", 0) / df["size_num"].replace(
        0, np.nan
    )
    df["external_storage_ratio"] = df["external_storage_num"] / df[
        "size_num"
    ].replace(0, np.nan)

    # ------------------- Neighborhood Features -------------------
    if "inhabitants_in_neighborhood" in df.columns:
        df["inhabitants_per_room"] = df["inhabitants_in_neighborhood"] / df[
            "nr_rooms"
        ].replace(0, np.nan)
        df["neighborhood_facility_density"] = df["num_facilities"] / df[
            "inhabitants_in_neighborhood"
        ].replace(0, np.nan)
    else:
        df["inhabitants_per_room"] = 0
        df["neighborhood_facility_density"] = 0

    # ------------------- Building Age / Lease -------------------
    df["building_age"] = current_year - df.get(
        "year_of_construction", current_year
    )
    df["old_house_flag"] = (df["building_age"] > 100).astype(int)
    df["lease_ratio"] = df.get("lease_years_remaining", 0) / df[
        "building_age"
    ].replace(0, np.nan)

    # ------------------- Balcony / Backyard -------------------
    df["has_outdoor_space"] = (
        (df.get("balcony_flag", 0) > 0) | (df.get("backyard_num", 0) > 0)
    ).astype(int)
    df["outdoor_area_ratio"] = (
        df.get("balcony_flag", 0) + df.get("backyard_num", 0)
    ) / df["size_num"].replace(0, np.nan)

    # ------------------- Cleanup -------------------
    df = df.replace([np.inf, -np.inf], np.nan).fillna(0)

    # ------------------- Ensure Column Consistency -------------------
    expected_features = [
        "room_utilization_ratio",
        "floor_to_size",
        "external_storage_ratio",
        "inhabitants_per_room",
        "neighborhood_facility_density",
        "building_age",
        "lease_ratio",
        "has_outdoor_space",
        "outdoor_area_ratio",
    ]
    for col in expected_features:
        if col not in df.columns:
            df[col] = 0

    return df
