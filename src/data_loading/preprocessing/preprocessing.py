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
from .utils import parse_price, parse_size


def preprocess_df(df):
    df["price_num"] = df["price"].apply(parse_price)
    df["contribution_vve_num"] = df["contribution"].apply(parse_price)
    df["size_num"] = df["size"].apply(parse_size)
    df["external_storage_num"] = df["external_storage"].apply(parse_size)

    df = preprocess_outdoor_features_column(df)

    df = preprocess_facilities_column(df)

    df = preprocess_cadastral_column(df)
    df = preprocess_ownership_column(df)
    df = preprocess_charges_column(df)

    df = preprocess_postal_city_column(df)

    df = expand_neighborhood_details_column(df)

    return df
