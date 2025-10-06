import pandas as pd
from src.data_loading.preprocessing.preprocessing import preprocess_df
from src.data_loading.preprocessing.neighborhood_statistics import (
    preprocess_neighborhood_details,
)


def test_preprocess_neighborhood_details_valid():
    input_data = {
        "Inhabitants in neighborhood": "1.830",
        "Families with children": "38%",
        "Price per m² in neighborhood": "€ 5.990",
    }
    result = preprocess_neighborhood_details(input_data)
    assert result["inhabitants_in_neighborhood"] == 1830
    assert abs(result["families_with_children_pct"] - 0.38) < 1e-6
    assert result["price_per_m2_neighborhood"] == 5990


def test_preprocess_neighborhood_details_invalid():
    # Passing None or invalid type returns pd.NA values
    result = preprocess_neighborhood_details(None)
    assert pd.isna(result["inhabitants_in_neighborhood"])
    assert pd.isna(result["families_with_children_pct"])
    assert pd.isna(result["price_per_m2_neighborhood"])


def test_preprocess_df_basic():
    # Prepare a minimal DataFrame with nested neighborhood_details
    data = [
        {
            "price": "€ 500.000",
            "size": "100 m²",
            "contribution": "€ 200",
            "external_storage": "10 m²",
            "year_of_construction": "2000",
            "outdoor_features": {
                "Ligging": "In woonwijk",
                "Tuin": None,
                "Achtertuin": None,
                "Ligging tuin": None,
            },
            "cadastral_parcels": [
                {"parcel": "Parcel 1"},
                {"parcel": "Parcel 2"},
            ],
            "ownership_situations": [{"ownership_type": "Freehold"}],
            "charges": [
                "Servicekosten € 50",
                "Gemeentelijke belastingen € 100",
            ],
            "postal_code": "1234 AB Amsterdam",
            "neighborhood_details": {
                "Inhabitants in neighborhood": "1.000",
                "Families with children": "20%",
                "Price per m² in neighborhood": "€ 5.000",
            },
            "facilities": "Lift, schuifpui en TV kabel",
        }
    ]
    df = pd.DataFrame(data)
    df = df.drop(columns="has_n/a", errors="ignore")
    cleaned_df = preprocess_df(df)

    # Check if price_num and size_num are parsed correctly
    assert cleaned_df.loc[0, "price_num"] == 500000
    assert cleaned_df.loc[0, "size_num"] == 100

    # Check neighborhood details expanded
    assert cleaned_df.loc[0, "inhabitants_in_neighborhood"] == 1000
    assert abs(cleaned_df.loc[0, "families_with_children_pct"] - 0.20) < 1e-6
    assert cleaned_df.loc[0, "price_per_m2_neighborhood"] == 5000

    # Check facilities_list column created
    assert isinstance(cleaned_df.loc[0, "facilities_list"], list)
    assert "lift" in cleaned_df.loc[0, "facilities_list"]
