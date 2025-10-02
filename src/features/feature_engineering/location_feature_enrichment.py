import os
import pandas as pd
import numpy as np
from geopy.geocoders import Nominatim
from sklearn.neighbors import BallTree
import time
from pathlib import Path


# ------------------- Config -------------------
CITY_CENTER = (52.3730, 4.8923)  # Dam Square, Amsterdam
CACHE_FILE = "../data/df_with_lat_lon_encoded.csv"


# ------------------- Utilities -------------------
def haversine(lat1, lon1, lat2, lon2):
    """Great-circle distance in meters between two points."""
    R = 6371000  # meters
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlambda = np.radians(lon2 - lon1)
    a = (
        np.sin(dphi / 2) ** 2
        + np.cos(phi1) * np.cos(phi2) * np.sin(dlambda / 2) ** 2
    )
    return R * 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))


def load_cache(cache_file):
    """
    Load address -> (lat, lon) mapping from a CSV, ensuring unique addresses.
    """

    if cache_file is None:
        raise ValueError("cache_file cannot be None")

    cache_file = Path(cache_file).resolve()
    if not cache_file.exists():
        raise FileNotFoundError(f"Cache file not found: {cache_file}")

    df_cache = pd.read_csv(cache_file)

    # Keep only relevant columns if they exist
    possible_cols = ["address", "lat", "lon", "price", "neighborhood", "size"]
    keep_cols = [c for c in possible_cols if c in df_cache.columns]
    df_cache = df_cache[keep_cols]

    # Drop duplicates
    if all(c in df_cache.columns for c in ["price", "neighborhood", "size"]):
        df_cache = df_cache.drop_duplicates(
            subset=["address", "price", "neighborhood", "size"]
        )
    else:
        df_cache = df_cache.drop_duplicates(subset=["address"])

    # Deduplicate on address keeping first valid lat/lon
    if "lat" in df_cache.columns and "lon" in df_cache.columns:
        df_cache = df_cache.sort_values(["lat", "lon"], na_position="last")
        df_cache = df_cache.drop_duplicates(subset=["address"], keep="first")

    # Build dict
    return {
        row["address"]: (row["lat"], row["lon"])
        for _, row in df_cache.iterrows()
        if "lat" in row
        and "lon" in row
        and pd.notna(row["lat"])
        and pd.notna(row["lon"])
    }


def save_cache(lat_lon_cache, cache_file=CACHE_FILE):
    pd.DataFrame(
        [(addr, lat, lon) for addr, (lat, lon) in lat_lon_cache.items()],
        columns=["address", "lat", "lon"],
    ).to_csv(cache_file, index=False)


def enrich_with_geolocation(
    df: pd.DataFrame,
    use_geopy=False,
    geopy_pause=1,
    cache_file=CACHE_FILE,
    fit=True,
    geo_meta=None,
) -> tuple[pd.DataFrame, dict]:
    """
    Adds lat/lon + distance-to-center features using a cache + fallback system.
    If fit=True, learns dummy columns and saves them into geo_meta.
    If fit=False, aligns to geo_meta["distbin_columns"].
    """

    df = df.copy()

    # --- Load cache with deduplication ---
    lat_lon_cache = load_cache(cache_file)
    # --- Optionally fetch missing via geopy ---
    if use_geopy:
        from geopy.geocoders import Nominatim
        import time

        geolocator = Nominatim(user_agent="house_price_project")
        for addr in df["address"].unique():
            if addr not in lat_lon_cache:
                try:
                    location = geolocator.geocode(addr)
                    if location:
                        lat_lon_cache[addr] = (
                            location.latitude,
                            location.longitude,
                        )
                    else:
                        lat_lon_cache[addr] = (None, None)
                except Exception as e:
                    print(f"Error geocoding {addr}: {e}")
                    lat_lon_cache[addr] = (None, None)
                time.sleep(geopy_pause)

    # --- Attach cached lat/lon directly to df ---
    df["lat"] = df["address"].map(
        lambda a: lat_lon_cache.get(a, (None, None))[0]
    )
    df["lon"] = df["address"].map(
        lambda a: lat_lon_cache.get(a, (None, None))[1]
    )
    # --- Postal & neighborhood centroids ---
    df_valid = df[
        df["address"].isin(lat_lon_cache)
        & df["address"].map(lambda a: lat_lon_cache[a][0] is not None)
    ]

    postal_code_coords = (
        df_valid.groupby("postal_code_clean")[["lat", "lon"]]
        .mean()
        .to_dict(orient="index")
        if "postal_code_clean" in df.columns
        else {}
    )

    neighborhood_coords = (
        df_valid.groupby("neighborhood")[["lat", "lon"]]
        .mean()
        .to_dict(orient="index")
        if "neighborhood" in df.columns
        else {}
    )

    # --- Map addresses ---
    lat_list, lon_list = [], []
    for _, row in df.iterrows():
        addr = row["address"]
        if addr in lat_lon_cache and lat_lon_cache[addr][0] is not None:
            lat, lon = lat_lon_cache[addr]
        else:
            postal = row.get("postal_code_clean")
            neigh = row.get("neighborhood")
            if postal in postal_code_coords:
                lat, lon = (
                    postal_code_coords[postal]["lat"],
                    postal_code_coords[postal]["lon"],
                )
            elif neigh in neighborhood_coords:
                lat, lon = (
                    neighborhood_coords[neigh]["lat"],
                    neighborhood_coords[neigh]["lon"],
                )
            else:
                lat, lon = None, None
            lat_lon_cache[addr] = (lat, lon)
        lat_list.append(lat)
        lon_list.append(lon)

    df["lat"] = lat_list
    df["lon"] = lon_list

    # --- Save updated cache ---
    save_cache(lat_lon_cache, cache_file)

    # --- Missing flag ---
    df["lat_lon_missing"] = df["lat"].isna().astype(int)

    # --- Distance to city center ---
    CITY_CENTER = (52.3730, 4.8923)  # Dam Square, Amsterdam

    def haversine(lat1, lon1, lat2, lon2):
        R = 6371000  # meters
        phi1, phi2 = np.radians(lat1), np.radians(lat2)
        dphi = np.radians(lat2 - lat1)
        dlambda = np.radians(lon2 - lon1)
        a = (
            np.sin(dphi / 2) ** 2
            + np.cos(phi1) * np.cos(phi2) * np.sin(dlambda / 2) ** 2
        )
        return R * 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

    df["dist_to_center_m"] = df.apply(
        lambda r: (
            haversine(r["lat"], r["lon"], CITY_CENTER[0], CITY_CENTER[1])
            if pd.notna(r["lat"]) and r["lat"] != -1
            else -1
        ),
        axis=1,
    )

    # --- Distance bins ---
    bins = [-1, 0, 2000, 5000, 10000, 20000, np.inf]
    labels = ["missing", "0–2km", "2–5km", "5–10km", "10–20km", "20km+"]
    df["dist_to_center_bin"] = pd.cut(
        df["dist_to_center_m"], bins=bins, labels=labels, include_lowest=True
    )

    # --- One-hot encode bins ---
    dist_dummies = pd.get_dummies(df["dist_to_center_bin"], prefix="distbin")

    if fit:
        distbin_columns = dist_dummies.columns.tolist()
        geo_meta = {
            "distbin_columns": distbin_columns,
            "cache_file": cache_file,
        }
    else:
        distbin_columns = geo_meta["distbin_columns"]
        for col in distbin_columns:
            if col not in dist_dummies:
                dist_dummies[col] = 0
        dist_dummies = dist_dummies[distbin_columns]

    df = pd.concat([df, dist_dummies], axis=1).drop(
        columns=["dist_to_center_bin"]
    )

    return df, geo_meta


def enrich_with_amenities(
    df, amenities_df=None, amenity_radius_map=None, fit=True, amenity_meta=None
):
    """
    Add amenity-based features (counts of schools, parks, etc. within distance).

    Parameters
    ----------
    df : pd.DataFrame
        Listings with lat/lon columns (already enriched with geolocation).
    amenities_df : pd.DataFrame
        Pre-fetched amenities (columns: ['amenity_type','lat','lon']).
    amenity_radius_map : dict
        E.g. {"school": [0.5, 1.0], "park": [0.5,1.0], ...} (radii in km).
    fit : bool
        If True, compute binning thresholds and return meta. If False, apply meta.
    amenity_meta : dict
        Meta info from training stage, containing bin labels & dummy column names.

    Returns
    -------
    df_out : pd.DataFrame
        Original df with new amenity features added.
    amenity_meta : dict
        Stores bin labels, dummy columns, etc. for reuse in test set.
    """

    df = df.copy()
    earth_radius_km = 6371.0

    if fit:
        amenity_meta = {
            "amenity_radius_map": amenity_radius_map,
            "bin_edges": [-1, 0, 2, 5, 10, np.inf],
            "bin_labels": ["0", "1-2", "3-5", "6-10", "10+"],
            "bin_mapping": {
                label: i
                for i, label in enumerate(["0", "1-2", "3-5", "6-10", "10+"])
            },
        }

    # Convert listing coords
    listing_coords_rad = np.radians(df[["lat", "lon"]].values)

    for amenity, radii in amenity_meta["amenity_radius_map"].items():
        subset = amenities_df[amenities_df["amenity_type"] == amenity]
        if subset.empty:
            continue
        tree = BallTree(
            np.radians(subset[["lat", "lon"]].values), metric="haversine"
        )

        for r_km in radii:
            r_rad = r_km / earth_radius_km
            counts = tree.query_radius(
                listing_coords_rad, r=r_rad, count_only=True
            )
            col_name = f"count_{amenity}_within_{int(r_km*1000)}m"
            df[col_name] = counts

            bin_col = f"{col_name}_bin"
            ord_col = f"{col_name}_bin_encoded"
            df[bin_col] = pd.cut(
                df[col_name],
                bins=amenity_meta["bin_edges"],
                labels=amenity_meta["bin_labels"],
                include_lowest=True,
            )
            df[ord_col] = df[bin_col].map(amenity_meta["bin_mapping"])

    # Drop raw + bin cols, keep encoded
    raw_cols = [
        c
        for c in df.columns
        if c.startswith("count_")
        and not (c.endswith("_bin") or c.endswith("_bin_encoded"))
    ]
    bin_cols = [
        c for c in df.columns if c.startswith("count_") and c.endswith("_bin")
    ]
    df.drop(columns=raw_cols + bin_cols, inplace=True)

    if fit:
        amenity_meta["encoded_columns"] = [
            c for c in df.columns if c.endswith("_bin_encoded")
        ]
    else:
        # Ensure alignment with train columns
        for col in amenity_meta["encoded_columns"]:
            if col not in df.columns:
                df[col] = 0
        df = df.reindex(
            columns=list(df.columns)
            + [
                c
                for c in amenity_meta["encoded_columns"]
                if c not in df.columns
            ]
        )

    return df, amenity_meta
