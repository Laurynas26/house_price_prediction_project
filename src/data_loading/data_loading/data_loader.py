import glob
import json
import pandas as pd
from typing import Union, List, Dict


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


def json_to_df_raw_strict(
    source: Union[str, dict, List[dict]], verbose: bool = False
) -> pd.DataFrame:
    """
    Load one or many raw property JSONs into a normalized, schema-safe DataFrame.

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
                        f"[json_to_df_raw_strict] Warning: '{key}' has invalid type {type(val)} in record {i}"
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
