import numpy as np
import pandas as pd

def feature_expansion(df: pd.DataFrame) -> pd.DataFrame:
    """
    Expand features with safe interactions, ratios, and derived metrics,
    including all amenities, room/space ratios, neighborhood features,
    age/lease info, and categorical interactions.
    
    Args:
        df: DataFrame after basic preprocessing & feature encoding.
        
    Returns:
        df: DataFrame with added features.
    """
    df = df.copy()
    current_year = 2025

    # -------------------
    # Size and Rooms
    # -------------------
    df["size_per_room"] = df["size_num"] / df["nr_rooms"].replace(0, np.nan)
    df["bathroom_per_room"] = df["bathrooms"] / df["nr_rooms"].replace(0, np.nan)
    df["toilet_per_room"] = df["toilets"] / df["nr_rooms"].replace(0, np.nan)

    # -------------------
    # Space Utilization
    # -------------------
    df["floor_to_size"] = df["floor_level"] / (df["size_num"] + 1e-6)
    df["external_storage_ratio"] = df["external_storage_num"] / (df["size_num"] + 1e-6)

    # -------------------
    # Amenities / Luxury
    # -------------------
    amenities = [
        "has_mechanische_ventilatie", "has_tv_kabel", "has_lift",
        "has_natuurlijke_ventilatie", "has_n/a", "has_schuifpui",
        "has_glasvezelkabel", "has_frans_balkon", "has_buitenzonwering",
        "has_zonnepanelen", "has_airconditioning", "has_balansventilatie",
        "has_dakraam", "has_alarminstallatie", "has_domotica",
        "has_rookkanaal", "has_elektra", "has_sauna",
        "has_zonnecollectoren", "has_cctv", "has_rolluiken",
        "has_stromend_water", "has_satellietschotel"
    ]
    available_amenities = [col for col in amenities if col in df.columns]

    # sum for total amenities
    df["amenity_count"] = df[available_amenities].sum(axis=1)

    # amenities per room/floor
    df["amenities_per_room"] = df["amenity_count"] / (df["nr_rooms"] + 1e-6)
    df["amenities_per_floor"] = df["amenity_count"] / (df["floor_level"] + 1e-6)

    # optional interaction with age for luxury index
    if "year_of_construction" in df.columns:
        df["building_age"] = current_year - df["year_of_construction"].fillna(current_year)
        df["old_house_flag"] = (df["building_age"] > 100).astype(int)
        df["amenity_age_index"] = df["amenity_count"] * df["building_age"]

    df["facilities_per_room"] = df["num_facilities"] / (df["nr_rooms"] + 1e-6)

    # -------------------
    # Neighborhood Features
    # -------------------
    if "inhabitants_in_neighborhood" in df.columns:
        df["inhabitants_per_room"] = df["inhabitants_in_neighborhood"] / (
            df["nr_rooms"] + 1e-6
        )
        df["neighborhood_facility_ratio"] = df["num_facilities"] / (
            df["inhabitants_in_neighborhood"] + 1e-6
        )

    # -------------------
    # Lease
    # -------------------
    if "lease_years_remaining" in df.columns and "building_age" in df.columns:
        df["lease_ratio"] = df["lease_years_remaining"] / (df["building_age"] + 1e-6)

    # -------------------
    # Energy & Roof Interactions
    # -------------------
    if "energy_label_encoded" in df.columns and "roof_type" in df.columns:
        df["roof_simple"] = df["roof_type"].fillna("Other").apply(lambda x: x.split()[0])
        roof_dummies = pd.get_dummies(df["roof_simple"], prefix="roof")
        df = pd.concat([df, roof_dummies], axis=1)
        df["energy_roof_index"] = df["energy_label_encoded"] * roof_dummies.sum(axis=1)

    # -------------------
    # Balcony/Backyard Interactions
    # -------------------
    if "balcony_flag" in df.columns and "backyard_num" in df.columns:
        df["balcony_or_backyard_flag"] = ((df["balcony_flag"] > 0) | (df["backyard_num"] > 0)).astype(int)
        df["balcony_to_size"] = df["balcony_flag"] / (df["size_num"] + 1e-6)
        df["backyard_to_size"] = df["backyard_num"] / (df["size_num"] + 1e-6)

    return df
