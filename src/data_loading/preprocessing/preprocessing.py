import pandas as pd
from .neighborhood_statistics import preprocess_neighborhood_details

def preprocess_df(df):
    # Apply the neighborhood details preprocessing and expand the dict into columns
    df_neigh = df["neighborhood_details"].apply(preprocess_neighborhood_details).apply(pd.Series)

    # Drop the original nested column and concat the new cleaned columns
    df = pd.concat([df.drop(columns=["neighborhood_details"]), df_neigh], axis=1)

    return df
