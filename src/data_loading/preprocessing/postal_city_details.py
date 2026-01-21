import pandas as pd
from .utils import split_postal_city


def preprocess_postal_city_column(df):
    postal_city_df = df["postal_code"].apply(split_postal_city).apply(pd.Series)
    postal_city_df.columns = ["postal_code_clean", "city"]
    return pd.concat([df.drop(columns=["postal_code"]), postal_city_df], axis=1)
