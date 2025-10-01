from .facilities_details import (
    preprocess_facilities_column,
    preprocess_outdoor_features_column,
)
from .cadastral_info import (
    preprocess_cadastral_column,
    preprocess_ownership_column,
    preprocess_charges_column,
)
from .neighborhood_statistics import expand_neighborhood_details_column
from .postal_city_details import preprocess_postal_city_column
from .utils import (
    parse_price,
    parse_size,
    parse_year,
    coerce_numeric,
    apply_parsers,
)


def preprocess_df(df, drop_raw: bool = False, numeric_cols: list = None):
    """
    Apply full preprocessing pipeline to a raw scraped dataframe.

    Steps
    -----
    1. Standardize column names.
    2. Parse raw strings into numeric or datetime formats (e.g., price, size).
    3. Optionally coerce specified columns to numeric dtype.
    4. Expand and normalize complex/raw columns into structured features:
       - Outdoor features
       - Facilities
       - Cadastral info
       - Ownership
       - Charges
       - Postal/city details
       - Neighborhood statistics
    5. Optionally drop raw string columns.

    Parameters
    ----------
    df : pd.DataFrame
        Raw dataframe (scraped or ingested) containing real estate listing
        data.
    drop_raw : bool, default=False
        If True, drop original raw columns after preprocessing 
        (e.g., `price`, `size`).
    numeric_cols : list of str, optional
        List of columns to coerce to numeric dtype using `coerce_numeric`.

    Returns
    -------
    pd.DataFrame
        Preprocessed dataframe with parsed, expanded, and normalized features.

    Notes
    -----
    - Raw columns dropped if `drop_raw=True`:
      `["price", "contribution_vve", "size", "external_storage"]`.
    - Additional preprocessing steps can be added by extending
      the `preprocessing_steps` list.
    """
    # Standardize column naming
    df = df.rename(
        columns={
            "contribution": "contribution_vve",
        }
    )

    # Apply parsing functions for key raw fields
    df = apply_parsers(
        df,
        {
            "price": parse_price,
            "contribution_vve": parse_price,
            "size": parse_size,
            "external_storage": parse_size,
            "year_of_construction": parse_year,
        },
    )

    # Optional numeric coercion
    if numeric_cols:
        df[numeric_cols] = coerce_numeric(df, numeric_cols)

    # Run preprocessing modules in sequence
    preprocessing_steps = [
        preprocess_outdoor_features_column,
        preprocess_facilities_column,
        preprocess_cadastral_column,
        preprocess_ownership_column,
        preprocess_charges_column,
        preprocess_postal_city_column,
        expand_neighborhood_details_column,
    ]
    for step in preprocessing_steps:
        df = step(df)

    # Drop raw unstructured columns if requested
    if drop_raw:
        columns_to_drop = [
            "price",
            "contribution_vve",
            "size",
            "external_storage",
        ]
        df = df.drop(columns=columns_to_drop)

    df = df.drop(columns="has_n/a")

    return df
