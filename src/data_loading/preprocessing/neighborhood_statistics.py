from .utils import to_int, to_float_pct
import pandas as pd


def preprocess_neighborhood_details(neighborhood_details):
    """
    Convert neighborhood details dict into clean numeric values.

    Args:
        details (dict): e.g.
            {
              'Inhabitants in neighborhood': '1.830',
              'Families with children': '38%',
              'Price per m² in neighborhood': '€ 5.990'
            }
    Returns:
        dict: cleaned values
    """
    if not isinstance(neighborhood_details, dict):
        return {
            "inhabitants_in_neighborhood": pd.NA,
            "families_with_children": pd.NA,
            "price_per_m2": pd.NA,
        }

    return {
        "inhabitants_in_neighborhood": to_int(neighborhood_details.get("Inhabitants in neighborhood")),
        "families_with_children_pct": to_float_pct(
            neighborhood_details.get("Families with children")
        ),
        "price_per_m2": to_int(
            neighborhood_details.get("Price per m² in neighborhood")
        ),
    }

def expand_neighborhood_details_column(df):
    """
    Takes a DataFrame with a 'neighborhood_details' column of dicts,
    applies preprocessing, expands the dict into separate columns,
    drops the original nested column, and returns the modified df.
    """
    df_neigh = df["neighborhood_details"].apply(preprocess_neighborhood_details).apply(pd.Series)
    df = pd.concat([df.drop(columns=["neighborhood_details"]), df_neigh], axis=1).reset_index(drop=True)
    return df
