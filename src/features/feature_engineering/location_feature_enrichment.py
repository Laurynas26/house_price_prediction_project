import pandas as pd
import numpy as np
from geopy.geocoders import Nominatim
from sklearn.neighbors import BallTree
import time
from pathlib import Path
import os
import boto3

# ------------------- Config -------------------
CITY_CENTER = (52.3730, 4.8923)  # Dam Square, Amsterdam
CACHE_FILE = "../data/df_with_lat_lon_encoded.csv"
S3_BUCKET = os.environ.get("S3_BUCKET")


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


def load_cache(cache_file=None):
    """
    Load address -> (lat, lon) mapping from a CSV,
    ensuring unique addresses.
    Supports local file or Lambda S3 deployment.
    """

    # Check if running inside Lambda
    if os.environ.get("AWS_LAMBDA_FUNCTION_NAME"):
        # Lambda: download from S3
        s3_bucket = S3_BUCKET
        s3_key = "data/df_with_lat_lon_encoded.csv"
        local_path = "/tmp/df_with_lat_lon_encoded.csv"

        s3 = boto3.client("s3")
        s3.download_file(s3_bucket, s3_key, local_path)
        cache_file = local_path
    else:
        # Local dev: use provided path
        if cache_file is None:
            raise ValueError("cache_file cannot be None")
        cache_file = Path(cache_file).resolve()

    if not Path(cache_file).exists():
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


def save_cache(lat_lon_cache, cache_file="data/df_with_lat_lon_encoded.csv"):
    # ----------- FIX: choose correct output dir -----------
    if os.environ.get("AWS_LAMBDA_FUNCTION_NAME"):
        # Lambda is read-only except /tmp
        cache_file = "/tmp/df_with_lat_lon_encoded.csv"
    else:
        cache_file = Path(cache_file)

    # Ensure folder exists
    folder = Path(cache_file).parent
    folder.mkdir(parents=True, exist_ok=True)

    # Save CSV
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
    enable_cache_save=False,
) -> tuple[pd.DataFrame, dict]:
    """
    Adds lat/lon + distance-to-center features using a cache + fallback system.

    Training mode (fit=True): missing addresses can be fetched via geopy and
    postal/neighborhood centroids are computed for later fallback.

    Inference mode (fit=False): missing addresses are filled using
    cached lat/lon and precomputed centroids from geo_meta.
    """
    df = df.copy()

    # --- Load cache ---
    lat_lon_cache = load_cache(cache_file)

    # --- Optional geopy lookup for training only ---
    if fit and use_geopy:
        geolocator = Nominatim(user_agent="house_price_project")
        for addr in df["address"].unique():
            if addr not in lat_lon_cache:
                try:
                    location = geolocator.geocode(addr)
                    lat_lon_cache[addr] = (
                        (location.latitude, location.longitude)
                        if location
                        else (None, None)
                    )
                except Exception as e:
                    print(f"Error geocoding {addr}: {e}")
                    lat_lon_cache[addr] = (None, None)
                time.sleep(geopy_pause)

    # --- Attach cached lat/lon ---
    df["lat"] = df["address"].map(
        lambda a: lat_lon_cache.get(a, (None, None))[0]
    )
    df["lon"] = df["address"].map(
        lambda a: lat_lon_cache.get(a, (None, None))[1]
    )

    # --- Compute postal/neighborhood centroids ---
    if fit:
        df_valid = df[df["lat"].notna() & df["lon"].notna()]
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
        # Save centroids in geo_meta for inference
        geo_meta = {
            "distbin_column": "dist_to_center_bin_encoded",
            "cache_file": cache_file,
            "postal_code_centroids": postal_code_coords,
            "neighborhood_centroids": neighborhood_coords,
        }
    else:
        if geo_meta is None:
            geo_meta = {}
        postal_code_coords = geo_meta.get("postal_code_centroids", {})
        neighborhood_coords = geo_meta.get("neighborhood_centroids", {})

    # --- Fill missing lat/lon ---
    lat_list, lon_list = [], []
    CITY_CENTER = (52.3730, 4.8923)
    for _, row in df.iterrows():
        addr = row["address"]
        lat, lon = lat_lon_cache.get(addr, (None, None))

        if pd.isna(lat) or pd.isna(lon):
            # Use fallback
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
                # Final fallback to city center
                lat, lon = CITY_CENTER
            lat_lon_cache[addr] = (lat, lon)

        lat_list.append(lat)
        lon_list.append(lon)

    df["lat"] = lat_list
    df["lon"] = lon_list

    # --- Save cache in training if requested ---
    if fit and enable_cache_save:
        save_cache(lat_lon_cache, cache_file)

    # --- Missing flag ---
    df["lat_lon_missing"] = df[["lat", "lon"]].isna().any(axis=1).astype(int)

    # --- Distance to city center ---
    df["dist_to_center_m"] = df.apply(
        lambda r: haversine(
            r["lat"], r["lon"], CITY_CENTER[0], CITY_CENTER[1]
        ),
        axis=1,
    )

    # --- Distance bins ---
    bins = [-1, 0, 2000, 5000, 10000, 20000, np.inf]
    labels = ["missing", "0–2km", "2–5km", "5–10km", "10–20km", "20km+"]
    distbin_mapping = {label: i for i, label in enumerate(labels)}
    df["dist_to_center_bin_encoded"] = (
        pd.cut(df["dist_to_center_m"], bins=bins, labels=labels)
        .map(distbin_mapping)
        .astype(int)
    )
    df.drop(columns=["dist_to_center_m"], inplace=True)

    return df, geo_meta


def enrich_with_amenities(
    df, amenities_df=None, amenity_radius_map=None, fit=True, amenity_meta=None
):
    """
    Add amenity-based features (counts of schools, parks, etc.)
    using integer-encoded bins.
    Keeps only the `_bin_encoded` columns. Handles numpy arrays correctly.
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
            # Wrap counts in pd.Series to allow map()
            df[col_name + "_bin_encoded"] = (
                pd.cut(
                    pd.Series(counts),  # <-- wrap counts
                    bins=amenity_meta["bin_edges"],
                    labels=amenity_meta["bin_labels"],
                )
                .map(
                    amenity_meta["bin_mapping"], na_action=None
                )  # explicit na_action
                .astype(int)
            )

    # Drop raw columns and intermediate bin columns
    keep_cols = [
        c
        for c in df.columns
        if not (c.startswith("count_") and not c.endswith("_bin_encoded"))
    ]
    df = df[keep_cols]

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
