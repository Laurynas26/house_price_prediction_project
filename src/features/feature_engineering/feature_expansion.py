import numpy as np
import pandas as pd
from typing import Optional, Dict, Tuple

from src.features.feature_engineering.location_feature_enrichment import (
    enrich_with_geolocation,
    enrich_with_amenities,
)


def feature_expansion(
    df: pd.DataFrame,
    current_year: int = 2025,
    use_geolocation: bool = False,
    geo_meta: Optional[Dict] = None,
    use_amenities: bool = False,
    amenities_df: Optional[pd.DataFrame] = None,
    amenity_radius_map: Optional[Dict[str, list]] = None,
    amenity_meta: Optional[Dict] = None,
    fit: bool = True,
    geo_cache_file: Optional[str] = None,
    enable_cache_save: bool = False,
) -> Tuple[pd.DataFrame, Optional[Dict], Optional[Dict]]:
    """
    Expand features with row-wise, neighborhood, building, outdoor,
    and optional geo/amenities metrics.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe.
    current_year : int
        Year for calculating building age.
    use_geolocation : bool
        Whether to enrich with lat/lon + distance-to-center features.
    geo_meta : dict or None
        Geo metadata from training stage (used if fit=False).
    use_amenities : bool
        Whether to enrich with amenities counts/features.
    amenities_df : pd.DataFrame
        Pre-fetched amenities data (required if use_amenities=True).
    amenity_radius_map : dict
        E.g. {"school": [0.5, 1.0], "park": [0.5,1.0], ...} (km).
    amenity_meta : dict
        Meta info from training stage (used if fit=False).
    fit : bool
        Whether this is training mode (fit=True) or test/inference (fit=False).

    Returns
    -------
    df_expanded : pd.DataFrame
        Expanded dataframe with new features.
    geo_meta_out : dict or None
        Geo metadata (only returned if use_geolocation=True).
    amenity_meta_out : dict or None
        Amenity metadata (only returned if use_amenities=True).
    """

    df = df.copy()

    # ------------------- Row-wise Ratios -------------------
    df["room_utilization_ratio"] = (
        df.get("size_num", 0) + df.get("bathrooms", 0) + df.get("toilets", 0)
    ) / df.get("nr_rooms", 1).replace(0, np.nan)
    df["floor_to_size"] = df.get("floor_level", 0) / df.get("size_num", 1).replace(
        0, np.nan
    )
    df["external_storage_ratio"] = df.get("external_storage_num", 0) / df.get(
        "size_num", 1
    ).replace(0, np.nan)

    # ------------------- Neighborhood Features -------------------
    if "inhabitants_in_neighborhood" in df.columns:
        df["inhabitants_per_room"] = df["inhabitants_in_neighborhood"] / df.get(
            "nr_rooms", 1
        ).replace(0, np.nan)
        df["neighborhood_facility_density"] = df.get("num_facilities", 0) / df[
            "inhabitants_in_neighborhood"
        ].replace(0, np.nan)
    else:
        df["inhabitants_per_room"] = 0
        df["neighborhood_facility_density"] = 0

    # ------------------- Building Age / Lease -------------------
    df["building_age"] = current_year - df.get("year_of_construction", current_year)
    df["old_house_flag"] = (df["building_age"] > 100).astype(int)
    df["lease_ratio"] = df.get("lease_years_remaining", 0) / df["building_age"].replace(
        0, np.nan
    )

    # ------------------- Balcony / Backyard -------------------
    df["has_outdoor_space"] = (
        (df.get("balcony_flag", 0) > 0) | (df.get("backyard_num", 0) > 0)
    ).astype(int)
    df["outdoor_area_ratio"] = (
        df.get("balcony_flag", 0) + df.get("backyard_num", 0)
    ) / df.get("size_num", 1).replace(0, np.nan)

    # ------------------- Geo Enrichment -------------------
    geo_meta_out = None

    if use_geolocation:
        df, geo_meta_out = enrich_with_geolocation(
            df,
            use_geopy=False,
            cache_file=geo_cache_file,
            fit=fit,
            geo_meta=geo_meta,
            enable_cache_save=enable_cache_save,
        )

    # ------------------- Amenities Enrichment -------------------
    amenity_meta_out = None
    if use_amenities:
        if amenities_df is None or amenity_radius_map is None:
            raise ValueError(
                "amenities_df and amenity_radius_map must be provided"
                " when use_amenities=True"
            )
        df, amenity_meta_out = enrich_with_amenities(
            df,
            amenities_df=amenities_df,
            amenity_radius_map=amenity_radius_map,
            fit=fit,
            amenity_meta=amenity_meta,
        )

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

    return df, geo_meta_out, amenity_meta_out
