import pandas as pd


def preprocess_facilities(facilities_str):
    """
    Parse the facilities string into a cleaned list of facility keywords.
    """
    if not facilities_str or not isinstance(facilities_str, str):
        return []

    # Replace " en " with comma to unify separators
    cleaned_str = facilities_str.replace(" en ", ", ")

    # Split by comma and strip whitespace, lowercase everything
    items = [item.strip().lower() for item in cleaned_str.split(",")]

    # Remove empty strings if any
    return [item for item in items if item]


def create_facilities_features(df, key_facilities=None):
    """
    Given a DataFrame with a 'facilities_list' column,
    create binary indicator columns for each facility.

    If key_facilities is None, use the most common facilities in the dataset.
    """
    if key_facilities is None:
        # Get list of unique facilities sorted by frequency
        key_facilities = (
            df["facilities_list"].explode().value_counts().index.tolist()
        )

    for facility in key_facilities:
        col_name = f"has_{facility.replace(' ', '_')}"
        df[col_name] = df["facilities_list"].apply(lambda x: facility in x)
    df["num_facilities"] = df["facilities_list"].apply(len)
    return df


def preprocess_facilities_column(df):
    """
    Add facilities_list column and indicator features for key facilities.
    """
    df["facilities_list"] = df["facilities"].apply(preprocess_facilities)
    df = create_facilities_features(df)
    return df


def preprocess_outdoor_features(outdoor_dict):
    """
    Clean and standardize the outdoor features dictionary.

    Args:
        outdoor_dict (dict): e.g. {'Ligging': 'In woonwijk', 'Tuin': None,
        'Achtertuin': None, 'Ligging tuin': None}

    Returns:
        dict: cleaned features with standardized keys and values
        or pd.NA for missing.
    """
    import pandas as pd

    # Default structure with pd.NA for missing values
    default_features = {
        "location": pd.NA,
        "garden": pd.NA,
        "backyard": pd.NA,
        "garden_location": pd.NA,
    }

    if not isinstance(outdoor_dict, dict):
        return default_features

    # Map original keys to new standardized keys
    key_map = {
        "Ligging": "location",
        "Tuin": "garden",
        "Achtertuin": "backyard",
        "Ligging tuin": "garden_location",
    }

    cleaned = {}

    for orig_key, new_key in key_map.items():
        val = outdoor_dict.get(orig_key)
        # If None or empty string, convert to pd.NA
        if val in (None, "", "N/A"):
            cleaned[new_key] = pd.NA
        else:
            cleaned[new_key] = val.strip() if isinstance(val, str) else val

    return cleaned


def preprocess_outdoor_features_column(df):
    """
    Apply outdoor features preprocessing and add as new columns.
    """
    df_outdoor = (
        df["outdoor_features"]
        .apply(preprocess_outdoor_features)
        .apply(pd.Series)
    )
    df = pd.concat([df.drop(columns=["outdoor_features"]), df_outdoor], axis=1)
    return df
