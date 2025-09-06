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


def preprocess_df(df, drop_raw=False, numeric_cols=None):
    df = df.rename(
        columns={
            "contribution": "contribution_vve",
        }
    )
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

    if numeric_cols:
        df[numeric_cols] = coerce_numeric(df, numeric_cols)

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

    if drop_raw:
        columns_to_drop = [
            "price",
            "contribution_vve",
            "size",
            "external_storage",
        ]
        df = df.drop(columns=columns_to_drop)

    return df
