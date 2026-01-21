import glob
import json
import pandas as pd
from typing import Union, List

import boto3
from io import BytesIO


# ----------------------------
# S3 Utility Functions
# ----------------------------
def list_s3_files(bucket_name: str, prefix: str) -> List[str]:
    """
    List all files in an S3 bucket under a given prefix.
    """
    s3 = boto3.client("s3")
    resp = s3.list_objects_v2(Bucket=bucket_name, Prefix=prefix)
    if "Contents" not in resp:
        return []
    return [obj["Key"] for obj in resp["Contents"] if not obj["Key"].endswith("/")]


def load_json_from_s3(bucket_name: str, key: str) -> dict:
    """
    Load a single JSON file from S3 into a Python dict.
    """
    s3 = boto3.client("s3")
    obj = s3.get_object(Bucket=bucket_name, Key=key)
    return json.load(BytesIO(obj["Body"].read()))


def load_data_from_s3_json(bucket_name: str, prefix: str) -> pd.DataFrame:
    """
    Load all JSON files under a given S3 prefix into a DataFrame.
    """
    keys = list_s3_files(bucket_name, prefix)
    if not keys:
        return pd.DataFrame()  # empty bucket/prefix

    data_list = [load_json_from_s3(bucket_name, key) for key in keys]

    # Feed into your strict loader
    return pd.DataFrame(data_list)


# ----------------------------
# Local JSON Loading
# ----------------------------


def load_data_from_json(path_pattern: str) -> pd.DataFrame:
    """
    Load multiple JSON files matching a path pattern into a single DataFrame.

    Parameters
    ----------
    path_pattern : str
        Glob-style file path pattern to match JSON files
        (e.g., "data/raw/*.json").

    Returns
    -------
    pd.DataFrame
        DataFrame where each row corresponds to one JSON file's content.

    Notes
    -----
    - Assumes each JSON file contains a dictionary-like structure.
    - If JSON files contain nested structures, further normalization may be
    required.
    - Files are read using UTF-8 encoding.
    """
    files = glob.glob(path_pattern)
    data_list = []
    for file in files:
        with open(file, "r", encoding="utf-8") as f:
            data_list.append(json.load(f))
    return pd.DataFrame(data_list)


# ----------------------------
# Schema-Safe Loader
# ----------------------------
def json_to_df_raw_strict(
    source: Union[str, dict, List[dict]], verbose: bool = False
) -> pd.DataFrame:
    """
    Load one or many raw property JSONs into a normalized,
    schema-safe DataFrame.

    Designed for perfect compatibility with `preprocess_df()`, including:
    - Type normalization for nested structures
    - Default fallbacks for missing keys
    - Consistent schema (no missing columns)
    """

    # ------------------------- Load phase -------------------------
    if isinstance(source, dict):
        data_list = [source]
    elif isinstance(source, list):
        data_list = source
    elif isinstance(source, str):
        files = glob.glob(source)
        if not files and source.endswith(".json"):
            # single JSON file
            with open(source, "r", encoding="utf-8") as f:
                data_list = [json.load(f)]
        else:
            data_list = []
            for file in files:
                with open(file, "r", encoding="utf-8") as f:
                    data_list.append(json.load(f))
    else:
        raise ValueError("Unsupported input type for json_to_df_raw_strict")

    # ------------------------- Expected schema -------------------------
    expected_schema = {
        # Core numeric/parsed
        "price": (None, (int, float, str, type(None))),
        "contribution_vve": (None, (int, float, str, type(None))),
        "size": (None, (int, float, str, type(None))),
        "external_storage": (None, (int, float, str, type(None))),
        "year_of_construction": (None, (int, float, str, type(None))),
        "nr_rooms": (None, (int, float, str, type(None))),
        "bathrooms": (None, (int, float, str, type(None))),
        "toilets": (None, (int, float, str, type(None))),
        "bedrooms": (None, (int, float, str, type(None))),
        # Nested / complex
        "facilities": ("", (str, list, type(None))),
        "outdoor_features": ({}, (dict, type(None))),
        "cadastral_parcels": ([], (list, type(None))),
        "ownership_situations": ([], (list, type(None))),
        "charges": ([], (list, type(None))),
        "postal_code": (None, (str, type(None))),
        "neighborhood_details": ({}, (dict, type(None))),
        # Meta / optional
        "address": (None, (str, type(None))),
        "roof_type": (None, (str, type(None))),
        "status": (None, (str, type(None))),
        "ownership_type": (None, (str, type(None))),
        "location": (None, (str, type(None))),
        "energy_label": (None, (str, type(None))),
        "located_on": (None, (str, type(None))),
        "backyard": (None, (str, type(None))),
        "balcony": (None, (str, type(None))),
    }

    rows = []
    for i, item in enumerate(data_list):
        row = {}
        for key, (default, allowed_types) in expected_schema.items():
            val = item.get(key, default)

            # --- Type corrections ---
            if not isinstance(val, allowed_types):
                if verbose:
                    print(
                        f"[json_to_df_raw_strict] Warning: '{key}' "
                        f"has invalid type {type(val)} in record {i}"
                    )
                val = default

            # Handle common misformats
            if key in ["facilities"] and isinstance(val, list):
                val = ", ".join(map(str, val))
            elif key in [
                "charges",
                "cadastral_parcels",
                "ownership_situations",
            ]:
                if isinstance(val, str):
                    val = [val]
                elif not isinstance(val, list):
                    val = default
            elif key in ["outdoor_features", "neighborhood_details"]:
                if not isinstance(val, dict):
                    val = default

            row[key] = val

        rows.append(row)

    df = pd.DataFrame(rows)
    df = df[list(expected_schema.keys())]  # ensure consistent column order

    return df
