import pandas as pd
from .neighborhood_statistics import expand_neighborhood_details_column
from .postal_city_details import preprocess_postal_city_column
from .utils import parse_price, parse_size, split_postal_city

def preprocess_df(df):
    df["price_num"] = df["price"].apply(parse_price)
    df["size_num"] = df["size"].apply(parse_size)

    df = preprocess_postal_city_column(df)

    df = expand_neighborhood_details_column(df)


    return df
